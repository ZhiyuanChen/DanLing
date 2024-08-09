# DanLing
# Copyright (C) 2022-Present  DanLing
#
# This file is part of DanLing.
#
# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0
#
# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from typing import Any
from warnings import warn

import torch
from lazy_imports import try_import
from torch import nn

from danling.optim import OPTIMIZERS

from .base_runner import BaseRunner
from .checkpoints import TorchDistributedCheckpointManager
from .compile import maybe_compile_model
from .config import FsdpConfig
from .torch_runner import TorchRunner, get_precision

with try_import() as fsdp:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

with try_import() as device_mesh:
    from torch.distributed.device_mesh import init_device_mesh

with try_import() as dcp:
    from torch.distributed.checkpoint.state_dict import StateDictOptions


class FsdpRunner(TorchRunner):
    """Torch runner with FSDP backend owned by the subclass.

    Checkpoint invariants:
    - Use `checkpoint.backend="dcp"` for distributed FSDP.
    - Save/load model and optimizer through torch.distributed.checkpoint state-dict APIs.
    - Restore order is model first, then optimizer, then scheduler.

    FSDP config:
    - `fsdp.mode="full_shard"` (default FSDP).
    - `fsdp.mode="hybrid_shard"` (HSDP), requires:
      - `fsdp.replicate_degree`
      - `fsdp.shard_degree`
    """

    checkpoint_manager: TorchDistributedCheckpointManager

    def __init__(self, config: Mapping[str, Any]) -> None:
        fsdp.check()
        dcp.check()
        super().__init__(config)

    def init_distributed(self) -> None:
        super().init_distributed()
        if self.world_size <= 1:
            raise RuntimeError("FsdpRunner requires distributed mode (WORLD_SIZE > 1)")
        if self.config.checkpoint.backend.lower() != "dcp":
            warn(
                "FsdpRunner overrides checkpoint.backend to 'dcp'",
                RuntimeWarning,
                stacklevel=2,
            )
            self.config.checkpoint.backend = "dcp"
            self.checkpoint_manager = TorchDistributedCheckpointManager(self)

    def materialize_model(self) -> None:
        if self.model is None:
            raise ValueError("cannot materialize FSDP model: model is not initialized")
        if FSDP is None:
            raise RuntimeError("cannot initialize FSDP runner: torch.distributed.fsdp is required")
        if not torch.cuda.is_available():
            raise RuntimeError("FsdpRunner requires CUDA when WORLD_SIZE > 1")

        fsdp_kwargs = self._fsdp_kwargs(self.config.fsdp)
        model = self.apply_activation_checkpointing(self.model)
        model = maybe_compile_model(model, self.config)
        self.model = FSDP(model, **fsdp_kwargs)

        if self.ema is not None:
            self.ema = self.ema.to(self.device)

    def _fsdp_kwargs(self, fsdp_config: FsdpConfig) -> dict[str, Any]:
        fsdp_mode = str(fsdp_config.get("mode", "full_shard")).lower()
        fsdp_kwargs = {key: value for key, value in fsdp_config.items() if key != "mode"}
        replicate_degree = fsdp_kwargs.pop("replicate_degree", None)
        shard_degree = fsdp_kwargs.pop("shard_degree", None)

        if fsdp_mode in {"hybrid_shard", "hsdp"}:
            if ShardingStrategy is None:
                raise RuntimeError("cannot initialize HSDP: torch.distributed.fsdp.ShardingStrategy is required")
            if "device_mesh" not in fsdp_kwargs:
                device_mesh.check()
                if replicate_degree is None or shard_degree is None:
                    raise ValueError(
                        "cannot initialize HSDP: `fsdp.replicate_degree` and `fsdp.shard_degree` must be set"
                    )
                if replicate_degree * shard_degree != self.world_size:
                    raise ValueError(
                        "cannot initialize HSDP: world size mismatch "
                        f"(replicate_degree={replicate_degree}, "
                        f"shard_degree={shard_degree}, world_size={self.world_size})"
                    )
                fsdp_kwargs["device_mesh"] = init_device_mesh(
                    "cuda",
                    (replicate_degree, shard_degree),
                    mesh_dim_names=("replicate", "shard"),
                )
            fsdp_kwargs.setdefault("sharding_strategy", ShardingStrategy.HYBRID_SHARD)
        elif fsdp_mode not in {"full_shard", "fsdp"}:
            raise ValueError(f"invalid fsdp mode: {fsdp_mode!r}; expected 'full_shard' or 'hybrid_shard'")

        fsdp_kwargs.setdefault("device_id", self.local_rank)
        return fsdp_kwargs

    def build_optimizer(self) -> None:
        if self.optimizer is not None or self.model is None:
            return
        optim_cfg = self.config.get("optim")
        if optim_cfg is None:
            optim_cfg = self.config.get("optimizer")
        if not optim_cfg:
            return
        self.optimizer = OPTIMIZERS.build(params=self.model.parameters(), **optim_cfg)

    def unwrap(self, model: nn.Module) -> nn.Module:
        if FSDP is not None and isinstance(model, FSDP):
            return model.module
        return super().unwrap(model)

    @contextmanager
    def train_context(self):
        micro_steps = self.train_state.micro_step + 1
        if (
            self.accum_steps > 1
            and micro_steps % self.accum_steps != 0
            and FSDP is not None
            and isinstance(self.model, FSDP)
        ):
            if self.fp8_enabled:
                autocast_context = self.fp8_autocast()
            else:
                precision = self.precision
                if precision is None:
                    autocast_context = nullcontext()
                else:
                    autocast_context = torch.autocast(self.device.type, dtype=get_precision(precision))

            with autocast_context, self.model.no_sync():
                yield
            return

        with super().train_context():
            yield

    def load_model(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
        if self.model is None:
            raise ValueError("cannot load model weights: model is not initialized")

        self.checkpoint_manager.load_model_state(
            model=self.model,
            model_state_dict=state_dict,
            options_cls=StateDictOptions,
            strict=True,
        )

    def load_optimizer(self, state_dict: Mapping[str, Any] | None, *args, **kwargs) -> None:
        if self.optimizer is None:
            return

        if state_dict is None:
            raise ValueError(
                "cannot restore optimizer: checkpoint has no optimizer state\n"
                "Use `load_pretrained` for model-only checkpoints instead of `load_checkpoint`"
            )

        if self.model is None:
            raise ValueError("cannot load checkpoint: model is not initialized")

        self.checkpoint_manager.load_optimizer_state(
            model=self.model,
            optimizer=self.optimizer,
            optimizer_state_dict=state_dict,
            options_cls=StateDictOptions,
            strict=True,
        )

    def state_dict(self, cls: type = dict) -> Mapping:
        if self.model is None:
            raise ValueError("cannot build checkpoint state: model is not initialized")

        state = cls(BaseRunner.state_dict(self, cls))
        state["ema"] = self.ema.state_dict() if self.ema else None
        state["scheduler"] = self.scheduler.state_dict() if self.scheduler else None

        model_state_dict, optim_state_dict = self.checkpoint_manager.export_model_optimizer_state(
            model=self.model,
            optimizer=self.optimizer,
            options_cls=StateDictOptions,
            strict=True,
        )
        state["model"] = model_state_dict
        if self.optimizer is not None:
            state["optimizer"] = optim_state_dict
        return state

    def apply_activation_checkpointing(self, model: nn.Module) -> nn.Module:
        """Hook for activation-checkpoint wrapping before compile/FSDP wrapping."""
        return model
