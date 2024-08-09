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
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from lazy_imports import try_import
from torch import distributed as dist
from torch import nn

from .compile import maybe_compile_model
from .torch_runner import get_precision
from .tppp_runner import TpppRunner, TpppTopology

with try_import() as dtpp_fsdp:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

with try_import() as dtpp_mesh:
    from torch.distributed.device_mesh import init_device_mesh


@dataclass(init=False)
class DtppTopology(TpppTopology):
    tp_degree: int
    pp_degree: int
    dp_degree: int
    shard_degree: int
    replicate_degree: int
    tp_rank: int
    pp_rank: int
    dp_rank: int
    shard_rank: int
    replicate_rank: int

    def __init__(
        self,
        *,
        world_size: int,
        rank: int,
        tp_degree: int,
        pp_degree: int,
        mode: str,
        shard_degree: int | None,
        replicate_degree: int | None,
    ) -> None:
        if tp_degree < 1 or pp_degree < 1:
            raise ValueError(
                "invalid DTPP topology: tp_degree and pp_degree must be positive integers, "
                f"got tp_degree={tp_degree}, pp_degree={pp_degree}"
            )

        model_parallel_degree = tp_degree * pp_degree
        if world_size % model_parallel_degree != 0:
            raise ValueError(
                "invalid DTPP topology: "
                f"WORLD_SIZE({world_size}) is not divisible by "
                f"tp_degree({tp_degree}) * pp_degree({pp_degree}) = {model_parallel_degree}"
            )

        dp_degree = world_size // model_parallel_degree
        mode = mode.lower()
        if mode in {"full_shard", "fsdp"}:
            resolved_shard_degree = int(shard_degree or dp_degree)
            resolved_replicate_degree = int(replicate_degree or 1)
            if resolved_replicate_degree != 1:
                raise ValueError("invalid DTPP topology: full_shard mode requires replicate_degree=1")
            if resolved_shard_degree != dp_degree:
                raise ValueError(
                    "invalid DTPP topology: "
                    f"full_shard requires shard_degree={dp_degree}, got {resolved_shard_degree}"
                )
        elif mode in {"hybrid_shard", "hsdp"}:
            if shard_degree is None or replicate_degree is None:
                raise ValueError(
                    "invalid DTPP topology: hybrid_shard mode requires dtpp.shard_degree and dtpp.replicate_degree"
                )
            resolved_shard_degree = int(shard_degree)
            resolved_replicate_degree = int(replicate_degree)
            if resolved_shard_degree * resolved_replicate_degree != dp_degree:
                raise ValueError(
                    "invalid DTPP topology: "
                    f"replicate_degree({resolved_replicate_degree}) * shard_degree({resolved_shard_degree}) "
                    f"!= dp_degree({dp_degree})"
                )
        else:
            raise ValueError(f"invalid dtpp mode: {mode!r}; expected 'full_shard' or 'hybrid_shard'")

        tp_rank = rank % tp_degree
        pp_rank = (rank // tp_degree) % pp_degree
        dp_rank = rank // model_parallel_degree
        shard_rank = dp_rank % resolved_shard_degree
        replicate_rank = dp_rank // resolved_shard_degree

        self.tp_degree = tp_degree
        self.pp_degree = pp_degree
        self.dp_degree = dp_degree
        self.shard_degree = resolved_shard_degree
        self.replicate_degree = resolved_replicate_degree
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.shard_rank = shard_rank
        self.replicate_rank = replicate_rank


class DtppRunner(TpppRunner):
    """Torch runner for DTPP stacks (shard/replicate + TP + PP)."""

    topology: DtppTopology
    shard_group = None
    replicate_group = None

    def __init__(self, config: Mapping[str, Any]) -> None:
        dtpp_fsdp.check()
        dtpp_mesh.check()
        super().__init__(config)

    def build_topology(self) -> DtppTopology:
        return DtppTopology(
            world_size=self.world_size,
            rank=self.rank,
            tp_degree=self.config.tppp.tp_degree,
            pp_degree=self.config.tppp.pp_degree,
            mode=self.config.dtpp.mode,
            shard_degree=self.config.dtpp.shard_degree,
            replicate_degree=self.config.dtpp.replicate_degree,
        )

    def reset_model_parallel_groups(self) -> None:
        super().reset_model_parallel_groups()
        self.shard_group = None
        self.replicate_group = None

    def init_model_parallel_groups(self) -> None:
        if not self.config.dtpp.use_device_mesh:
            raise RuntimeError("cannot initialize DTPP process groups: set `dtpp.use_device_mesh=True`.")

        mesh_device_type = self.config.dtpp.mesh_device_type
        if mesh_device_type is None:
            mesh_device_type = "cuda" if torch.cuda.is_available() else "cpu"

        if self.replicate_degree > 1:
            self.device_mesh = init_device_mesh(
                mesh_device_type,
                mesh_shape=(self.replicate_degree, self.shard_degree, self.pp_degree, self.tp_degree),
                mesh_dim_names=("replicate", "shard", "pp", "tp"),
            )
            self.replicate_group = self.device_mesh.get_group("replicate")
        else:
            self.device_mesh = init_device_mesh(
                mesh_device_type,
                mesh_shape=(self.shard_degree, self.pp_degree, self.tp_degree),
                mesh_dim_names=("shard", "pp", "tp"),
            )
            self.replicate_group = None

        self.shard_group = self.device_mesh.get_group("shard")
        self.pp_group = self.device_mesh.get_group("pp")
        self.tp_group = self.device_mesh.get_group("tp")

    def _fsdp_kwargs(self) -> dict[str, Any]:
        mode = self.config.dtpp.mode.lower()
        fsdp_kwargs = {
            key: value
            for key, value in self.config.dtpp.items()
            if key not in {"mode", "replicate_degree", "shard_degree", "use_device_mesh", "mesh_device_type"}
        }

        if mode in {"full_shard", "fsdp"}:
            fsdp_kwargs.setdefault("process_group", self.shard_group)
        elif mode in {"hybrid_shard", "hsdp"}:
            fsdp_kwargs.setdefault("process_group", (self.shard_group, self.replicate_group))
            fsdp_kwargs.setdefault("sharding_strategy", ShardingStrategy.HYBRID_SHARD)
        else:
            raise ValueError(f"invalid dtpp mode: {mode!r}; expected 'full_shard' or 'hybrid_shard'")

        fsdp_kwargs.setdefault("device_id", self.local_rank)
        return fsdp_kwargs

    def materialize_model(self) -> None:
        if FSDP is None:
            raise RuntimeError("cannot initialize DTPP runner: torch.distributed.fsdp is required")
        if not torch.cuda.is_available():
            raise RuntimeError("DtppRunner requires CUDA when WORLD_SIZE > 1")

        if self.pipeline_schedule is None:
            if self.model is None:
                if self.model_parts:
                    self.model = self.model_parts[0]
                else:
                    raise ValueError("cannot materialize DTPP model: model is not initialized")
            parts = [self.model]
        else:
            if not self.model_parts:
                if self.model is None:
                    raise ValueError("cannot materialize DTPP pipeline: model_parts are not initialized")
                self.model_parts = [self.model]
            parts = list(self.model_parts)

        fsdp_kwargs = self._fsdp_kwargs()
        wrapped_parts: list[nn.Module] = []
        for part in parts:
            module = self.apply_activation_checkpointing(part)
            module = maybe_compile_model(module.to(self.device), self.config)
            wrapped_parts.append(FSDP(module, **fsdp_kwargs))

        self.model_parts = wrapped_parts
        self.model = wrapped_parts[0]
        self.bind_pipeline_modules(self.model_parts)

        if self.ema is not None:
            self.ema = self.ema.to(self.device)

    def unwrap(self, model: nn.Module) -> nn.Module:
        if FSDP is not None and isinstance(model, FSDP):
            return model.module
        return super().unwrap(model)

    @contextmanager
    def train_context(self):
        micro_steps = self.train_state.micro_step + 1
        fsdp_parts = [module for module in (self.model_parts or []) if isinstance(module, FSDP)]
        if self.model is not None and not fsdp_parts and isinstance(self.model, FSDP):
            fsdp_parts = [self.model]

        if self.accum_steps > 1 and micro_steps % self.accum_steps != 0 and fsdp_parts:
            if self.fp8_enabled:
                autocast_context = self.fp8_autocast()
            else:
                precision = self.precision
                if precision is None:
                    autocast_context = nullcontext()
                else:
                    autocast_context = torch.autocast(self.device.type, dtype=get_precision(precision))

            with ExitStack() as stack:
                stack.enter_context(autocast_context)
                for module in fsdp_parts:
                    stack.enter_context(module.no_sync())
                yield
            return

        with super().train_context():
            yield

    def _all_reduce_data_parallel(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.shard_degree > 1:
            dist.all_reduce(tensor, group=self.shard_group)
        if self.replicate_degree > 1:
            dist.all_reduce(tensor, group=self.replicate_group)
        return tensor

    def reduce(self, tensor):
        self._all_reduce_data_parallel(tensor)
        tensor = tensor / max(self.dp_degree, 1)
        return tensor

    def reduce_loss_for_logging(self, loss: torch.Tensor | None) -> torch.Tensor | None:
        if self.pipeline_schedule is None:
            return super().reduce_loss_for_logging(loss)

        payload = torch.zeros((2,), dtype=torch.float32, device=self.device)
        is_reporter = self.pp_has_last_stage and self.tp_rank == 0
        if is_reporter:
            if loss is not None:
                loss_value = loss.detach().to(dtype=torch.float32)
                if loss_value.ndim > 0:
                    loss_value = loss_value.mean()
                payload[0] = loss_value
                payload[1] = 1.0
            self._all_reduce_data_parallel(payload)

        source_rank = (self.pp_degree - 1) * self.tp_degree
        dist.broadcast(payload, src=source_rank)
        if payload[1].item() <= 0:
            return None
        return payload[0] / payload[1]

    def apply_activation_checkpointing(self, model: nn.Module) -> nn.Module:
        """Hook for activation-checkpoint wrapping before compile/FSDP wrapping."""
        return model

    def close(self, timeout: float | None = None) -> bool:
        self.shard_group = None
        self.replicate_group = None
        return super().close(timeout=timeout)

    @property
    def shard_degree(self) -> int:
        return self.topology.shard_degree

    @property
    def replicate_degree(self) -> int:
        return self.topology.replicate_degree

    @property
    def shard_rank(self) -> int:
        return self.topology.shard_rank

    @property
    def replicate_rank(self) -> int:
        return self.topology.replicate_rank
