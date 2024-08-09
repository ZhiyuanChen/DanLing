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

import os
from collections.abc import Mapping
from typing import Any, Dict
from warnings import warn

import torch
from lazy_imports import try_import

from danling.utils import catch

from .base_runner import BaseRunner
from .checkpoints import FileCheckpointManager
from .compile import maybe_compile_loss, maybe_compile_model
from .torch_runner import TorchRunner

with try_import() as ds:
    import deepspeed


class DeepSpeedRunner(TorchRunner):
    """DeepSpeed-backed runner focused on ZeRO-1/2 training flows."""

    model: deepspeed.DeepSpeedEngine

    def __init__(self, config) -> None:
        ds.check()
        super().__init__(config)

    def init_distributed(self) -> None:
        super().init_distributed()
        backend = self.config.checkpoint.backend.lower()
        if backend != "file":
            warn(
                "DeepSpeedRunner overrides checkpoint.backend to 'file'",
                RuntimeWarning,
                stacklevel=2,
            )
            self.config.checkpoint.backend = "file"
            self.checkpoint_manager = FileCheckpointManager(self)

    def __post_init__(self) -> None:
        if self.datasets and not self.dataloaders:
            self.build_dataloaders()

        if self.model is None:
            raise ValueError("cannot initialize DeepSpeedRunner: model is not initialized")

        self.materialize_model()
        if self.criterion is not None:
            self.criterion = maybe_compile_loss(self.criterion, self.config)
        self.build_optimizer()
        self.build_scheduler()

        ds_config = self.get_deepspeed_config()
        self.config.deepspeed = ds_config
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
            config=ds_config,
        )
        self.model = model_engine
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimizer_container = None

    def _bind_optimizer_container(self) -> None:
        self.optimizer_container = None

    def materialize_model(self) -> None:
        if self.model is None:
            raise ValueError("cannot materialize DeepSpeed model: model is not initialized")

        model = self.model.to(self.device)
        model = maybe_compile_model(model, self.config)
        self.model = model

        if self.ema is not None:
            self.ema = self.ema.to(self.device)

    def get_deepspeed_config(self) -> dict[str, Any]:
        cfg = self.config.get("deepspeed")
        if cfg is None:
            ds_config: dict[str, Any] = {}
        elif isinstance(cfg, Mapping):
            ds_config = dict(cfg)
        else:
            raise ValueError(f"invalid deepspeed config: expected mapping, got {type(cfg).__name__}")

        grad_accum = ds_config.get("gradient_accumulation_steps")
        if grad_accum is not None and grad_accum != 1:
            warn(
                "DeepSpeedRunner manages accumulation via config.accum_steps; overriding "
                "deepspeed.gradient_accumulation_steps to 1",
                RuntimeWarning,
                stacklevel=2,
            )
        ds_config["gradient_accumulation_steps"] = 1

        if "train_micro_batch_size_per_gpu" not in ds_config:
            batch_size = self.config.get("dataloader.batch_size")
            if batch_size is not None:
                ds_config["train_micro_batch_size_per_gpu"] = batch_size

        precision = self.precision
        if precision is not None:
            normalized_precision = str(precision).lower().replace("-", "_")
            if normalized_precision in {"fp16", "float16", "half"} and "fp16" not in ds_config:
                ds_config["fp16"] = {"enabled": True}
            if normalized_precision in {"bf16", "bfloat16"} and "bf16" not in ds_config:
                ds_config["bf16"] = {"enabled": True}

        return ds_config

    def backward(self, loss: torch.Tensor) -> None:
        self.model.backward(loss / self.accum_steps)

    def _optimizer_step(self) -> bool:
        self.model.step()
        global_steps = getattr(self.model, "global_steps", None)
        if global_steps is None:
            self.train_state.global_step += 1
        else:
            self.train_state.global_step = int(global_steps)
        return True

    @catch
    def save_checkpoint(
        self,
        name: str = "latest",
        epochs: int | None = None,
        save_best: bool = True,
        last_step: bool = False,
    ) -> None:
        epochs = self.train_state.epoch if epochs is None else epochs
        if not self.checkpoint_manager.should_persist_checkpoint(epochs=epochs, last_step=last_step):
            return

        client_state: Dict = BaseRunner.state_dict(self, dict)  # type: ignore[assignment]
        client_state["ema"] = self.ema.state_dict() if self.ema else None

        tags: list[str] = [name]
        archive_name = self.checkpoint_manager.resolve_archive_name(epochs)
        if archive_name is not None and archive_name not in tags:
            tags.append(archive_name)
        if save_best and self.is_best and "best" not in tags:
            tags.append("best")

        for tag in tags:
            self.model.save_checkpoint(self.checkpoint_dir, tag=tag, client_state=client_state, save_latest=False)
            tag_dir = os.path.join(self.checkpoint_dir, tag)
            if os.path.isdir(tag_dir):
                self.config.yaml(os.path.join(tag_dir, "runner.yaml"))

        latest_file = os.path.join(self.checkpoint_dir, "latest")
        with open(latest_file, "w", encoding="utf-8") as fp:
            fp.write(name)

    def _resolve_deepspeed_checkpoint(self, checkpoint: bytes | str | os.PathLike) -> tuple[str, str]:
        checkpoint_path = os.fsdecode(checkpoint)

        if os.path.isfile(checkpoint_path):
            if os.path.basename(checkpoint_path) != "latest":
                raise ValueError(
                    f"invalid DeepSpeed checkpoint path: {checkpoint_path!r}. "
                    "expected a checkpoint directory or '<checkpoint_dir>/latest' file"
                )
            with open(checkpoint_path, encoding="utf-8") as fp:
                tag = fp.read().strip()
            if not tag:
                raise ValueError(f"invalid DeepSpeed latest pointer: {checkpoint_path!r} is empty")
            return os.path.dirname(checkpoint_path), tag

        if os.path.isdir(checkpoint_path):
            latest_file = os.path.join(checkpoint_path, "latest")
            if os.path.isfile(latest_file):
                with open(latest_file, encoding="utf-8") as fp:
                    tag = fp.read().strip()
                if tag:
                    return checkpoint_path, tag
            return os.path.dirname(checkpoint_path), os.path.basename(checkpoint_path)

        raise FileNotFoundError(f"checkpoint path does not exist: {checkpoint_path!r}")

    def load_checkpoint(self, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> None:
        if isinstance(checkpoint, Mapping):
            super().load_checkpoint(checkpoint, *args, **kwargs)
            return

        checkpoint_dir, checkpoint_tag = self._resolve_deepspeed_checkpoint(checkpoint)
        _, client_state = self.model.load_checkpoint(checkpoint_dir, tag=checkpoint_tag)

        if client_state is not None:
            BaseRunner.load_state_dict(self, client_state)
            if self.ema is not None and client_state.get("ema") is not None:
                self.load_ema(client_state["ema"], *args, **kwargs)

        self.config.checkpoint = checkpoint  # type: ignore[assignment]
        self.optimizer_container = None

    def load_pretrained(self, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> None:
        if isinstance(checkpoint, Mapping):
            return super().load_pretrained(checkpoint, *args, **kwargs)

        checkpoint_dir, checkpoint_tag = self._resolve_deepspeed_checkpoint(checkpoint)
        _, client_state = self.model.load_checkpoint(
            checkpoint_dir,
            tag=checkpoint_tag,
            load_module_only=True,
        )

        if client_state is not None and client_state.get("ema") is not None:
            self.load_model(client_state["ema"], *args, **kwargs)

        self.config.pretrained = checkpoint

    @classmethod
    def read_config(
        cls,
        checkpoint: Mapping | bytes | str | os.PathLike,
        *args,
        **kwargs,
    ) -> Mapping[str, Any]:
        if isinstance(checkpoint, Mapping):
            return super().read_config(checkpoint, *args, **kwargs)

        if isinstance(checkpoint, (bytes, str, os.PathLike)):
            checkpoint_path = os.fsdecode(checkpoint)

            if os.path.isdir(checkpoint_path):
                runner_yaml = os.path.join(checkpoint_path, "runner.yaml")
                if os.path.isfile(runner_yaml):
                    return cls.load(runner_yaml, *args, **kwargs)

                latest_file = os.path.join(checkpoint_path, "latest")
                if os.path.isfile(latest_file):
                    with open(latest_file, encoding="utf-8") as fp:
                        tag = fp.read().strip()
                    if tag:
                        tagged_runner_yaml = os.path.join(checkpoint_path, tag, "runner.yaml")
                        if os.path.isfile(tagged_runner_yaml):
                            return cls.load(tagged_runner_yaml, *args, **kwargs)

            if os.path.isfile(checkpoint_path) and os.path.basename(checkpoint_path) == "latest":
                with open(checkpoint_path, encoding="utf-8") as fp:
                    tag = fp.read().strip()
                if tag:
                    tagged_runner_yaml = os.path.join(os.path.dirname(checkpoint_path), tag, "runner.yaml")
                    if os.path.isfile(tagged_runner_yaml):
                        return cls.load(tagged_runner_yaml, *args, **kwargs)

        return super().read_config(checkpoint, *args, **kwargs)
