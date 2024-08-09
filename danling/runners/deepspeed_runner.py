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

from danling.optim.optimizer import normalize_scheduler_interval
from danling.utils import load

from .base_runner import BaseRunner
from .config import RunnerConfig
from .torch_runner import TorchRunner

with try_import() as ds:
    import deepspeed as deepspeed


class DeepSpeedRunner(TorchRunner):
    """DeepSpeed-backed runner focused on ZeRO-1/2 training flows."""

    model: deepspeed.DeepSpeedEngine
    deepspeed_config: dict[str, Any]
    _supports_torchft_runtime: bool = False

    def __init__(self, config) -> None:
        ds.check()
        if not isinstance(config, RunnerConfig):
            config = RunnerConfig(config)
        config.stack = "deepspeed"
        backend = str(config.checkpoint.backend).strip().lower()
        if backend == "auto":
            backend = "file"
        elif backend == "dcp":
            warn(
                "DeepSpeedRunner overrides checkpoint.backend to 'file'",
                RuntimeWarning,
                stacklevel=2,
            )
            backend = "file"
        elif backend != "file":
            raise ValueError(f"invalid checkpoint backend: {backend!r}. Expected one of: 'auto', 'file', 'dcp'.")
        config.checkpoint.backend = backend
        super().__init__(config)

    def init_distributed(self) -> None:
        backend = str(self.config.get("checkpoint.backend", "auto")).strip().lower()
        if backend != "file":
            raise ValueError(f"invalid checkpoint backend: {backend!r}. Expected one of: 'auto', 'file', 'dcp'.")
        super().init_distributed()

    def materialize_model(self) -> None:
        if self.model is None:
            raise ValueError("cannot materialize DeepSpeed model: model is not initialized")

        model = self.model.to(self.device)
        self.model = model
        if self.fp8_enabled:
            self.apply_fp8_module_policy_to_model_parts()
            model = self.model
        model = self.compiler.compile(model)
        self.model = model

        if self.ema is not None:
            self.ema = self.ema.to(self.device)

    def get_deepspeed_config(self) -> dict[str, Any]:
        runtime_config = getattr(self, "deepspeed_config", None)
        if runtime_config is not None:
            return dict(runtime_config)

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

    def _resolve_deepspeed_scheduler(self, scheduler: object | None) -> object | None:
        if scheduler is None:
            return None
        sched_cfg = self._get_scheduler_config()
        interval = sched_cfg.get("interval") if sched_cfg is not None else None
        if normalize_scheduler_interval(interval, scheduler) != "step":
            return None
        return scheduler

    def _finalize_runtime_components(self) -> None:
        ds_config = self.get_deepspeed_config()
        self.deepspeed_config = ds_config
        runner_scheduler = self.scheduler
        # DeepSpeed should own only per-step schedulers. Epoch and metric schedulers
        # still need the runner's explicit step boundary and metric resolution path.
        deepspeed_scheduler = self._resolve_deepspeed_scheduler(runner_scheduler)
        self._runner_owns_scheduler = runner_scheduler is not None and deepspeed_scheduler is None
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=deepspeed_scheduler,
            config=ds_config,
        )
        self.model = model_engine
        self.optimizer = optimizer
        self.scheduler = (
            scheduler
            if deepspeed_scheduler is not None and scheduler is not None
            else (deepspeed_scheduler if deepspeed_scheduler is not None else runner_scheduler)
        )

    def _bind_optimizer_container(self) -> None:
        self.optimizer_container = None

    def backward(self, loss: torch.Tensor) -> None:
        self.model.backward(self._scaled_loss_for_backward(loss))

    def optimizer_step(self) -> bool:
        """
        Perform one DeepSpeed engine optimizer update.

        DeepSpeed owns the concrete optimizer step; DanLing keeps accumulation
        normalization, runner state, profiler, timeout, and supervisor state in sync.
        """
        self.checkpoint_manager.maybe_wait_for_staging()
        grad_scale = self._gradient_scale_for_step()
        if grad_scale is not None:
            self._scale_optimizer_gradients(grad_scale)
        self.model.step()
        self._reset_accumulation_normalization()
        global_steps = getattr(self.model, "global_steps", None)
        if global_steps is None:
            self.train_state.global_step += 1
        else:
            self.train_state.global_step = int(global_steps)
        self._step_profiler()
        self._maybe_reduce_train_process_group_timeout()
        self.supervisor.maybe_collect_garbage(self.train_state.global_step, scope="train")
        return True

    def _auto_resume_source(self) -> str:
        return self.workspace.checkpoint_dir

    def _checkpoint_pointer_path(self, name: str) -> str:
        return os.path.join(self.workspace.checkpoint_dir, f"{name}.pointer")

    def _write_checkpoint_pointer(self, name: str, target_tag: str) -> None:
        pointer_path = self._checkpoint_pointer_path(name)
        pointer_tmp_path = f"{pointer_path}.tmp-{self.id}"
        with open(pointer_tmp_path, "w", encoding="utf-8") as fp:
            fp.write(target_tag)
        os.replace(pointer_tmp_path, pointer_path)

    @staticmethod
    def _read_checkpoint_pointer(checkpoint_path: bytes | str | os.PathLike) -> str:
        pointer_path = os.fsdecode(checkpoint_path)
        with open(pointer_path, encoding="utf-8") as fp:
            tag = fp.read().strip()
        if not tag:
            raise ValueError(f"invalid DeepSpeed checkpoint pointer: {pointer_path!r} is empty")
        return tag

    def _resolve_physical_checkpoint_tag(self, *, name: str, epochs: int, should_update_best: bool) -> str:
        history_name = self.checkpoint_manager.resolve_history_name(epochs)
        if history_name is not None:
            return history_name
        if should_update_best:
            return f"ckpt-g{self.train_state.global_step:012d}"
        return name

    def save_checkpoint(
        self,
        name: str = "latest",
        epochs: int | None = None,
        save_best: bool = True,
        last_step: bool = False,
        force: bool = False,
    ) -> None:
        epochs = self.train_state.epoch if epochs is None else epochs
        if not self.checkpoint_manager.should_persist_checkpoint(epochs=epochs, last_step=last_step, force=force):
            return

        client_state: Dict = BaseRunner.state_dict(self, dict)  # type: ignore[assignment]
        client_state["ema"] = self.ema.state_dict() if self.ema else None
        client_state["scheduler"] = (
            self.scheduler.state_dict() if getattr(self, "_runner_owns_scheduler", False) and self.scheduler else None
        )
        should_update_best = bool(save_best and self.is_best)
        physical_tag = self._resolve_physical_checkpoint_tag(
            name=name, epochs=epochs, should_update_best=should_update_best
        )
        self.model.save_checkpoint(
            self.workspace.checkpoint_dir,
            tag=physical_tag,
            client_state=client_state,
            save_latest=False,
        )
        if self.distributed and not self.is_main_process:
            return

        tag_dir = os.path.join(self.workspace.checkpoint_dir, physical_tag)
        if os.path.isdir(tag_dir):
            self.config.yaml(os.path.join(tag_dir, "runner.yaml"))

        self._write_checkpoint_pointer("latest", physical_tag)
        if name not in {"latest", physical_tag}:
            self._write_checkpoint_pointer(name, physical_tag)
        if should_update_best:
            self._write_checkpoint_pointer("best", physical_tag)

    @staticmethod
    def _resolve_deepspeed_checkpoint(checkpoint: bytes | str | os.PathLike) -> tuple[str, str]:
        checkpoint_path = os.fsdecode(checkpoint)

        if os.path.isfile(checkpoint_path):
            return os.path.dirname(checkpoint_path), DeepSpeedRunner._read_checkpoint_pointer(checkpoint_path)

        if os.path.isdir(checkpoint_path):
            latest_pointer = os.path.join(checkpoint_path, "latest.pointer")
            if os.path.isfile(latest_pointer):
                return checkpoint_path, DeepSpeedRunner._read_checkpoint_pointer(latest_pointer)
            latest_file = os.path.join(checkpoint_path, "latest")
            if os.path.isfile(latest_file):
                return checkpoint_path, DeepSpeedRunner._read_checkpoint_pointer(latest_file)
            if os.path.isdir(latest_file):
                return checkpoint_path, "latest"
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
            if getattr(self, "_runner_owns_scheduler", False) and client_state.get("scheduler") is not None:
                self.load_scheduler(client_state["scheduler"], *args, **kwargs)
            if self.dataloaders or "dataloaders" in client_state:
                self.load_dataloaders(client_state.get("dataloaders"))

        self.config.resume = os.fsdecode(checkpoint)
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

        self.config.pretrained = os.fsdecode(checkpoint)

    @classmethod
    def read_config(
        cls,
        checkpoint: Mapping | bytes | str | os.PathLike,
        *args,
        **kwargs,
    ) -> RunnerConfig:
        if isinstance(checkpoint, Mapping):
            return super().read_config(checkpoint, *args, **kwargs)

        if isinstance(checkpoint, (bytes, str, os.PathLike)):
            checkpoint_path = os.fsdecode(checkpoint)

            if os.path.isdir(checkpoint_path):
                runner_yaml = os.path.join(checkpoint_path, "runner.yaml")
                if os.path.isfile(runner_yaml):
                    return RunnerConfig.from_yaml(runner_yaml, *args, **kwargs)

                latest_pointer = os.path.join(checkpoint_path, "latest.pointer")
                if os.path.isfile(latest_pointer):
                    tag = cls._read_checkpoint_pointer(latest_pointer)
                    tagged_runner_yaml = os.path.join(checkpoint_path, tag, "runner.yaml")
                    if os.path.isfile(tagged_runner_yaml):
                        return load(tagged_runner_yaml, *args, **kwargs)

                latest_file = os.path.join(checkpoint_path, "latest")
                if os.path.isfile(latest_file):
                    tag = cls._read_checkpoint_pointer(latest_file)
                    if tag:
                        tagged_runner_yaml = os.path.join(checkpoint_path, tag, "runner.yaml")
                        if os.path.isfile(tagged_runner_yaml):
                            return load(tagged_runner_yaml, *args, **kwargs)
                elif os.path.isdir(latest_file):
                    tagged_runner_yaml = os.path.join(latest_file, "runner.yaml")
                    if os.path.isfile(tagged_runner_yaml):
                        return load(tagged_runner_yaml, *args, **kwargs)

            if os.path.isfile(checkpoint_path):
                tag = cls._read_checkpoint_pointer(checkpoint_path)
                if tag:
                    tagged_runner_yaml = os.path.join(os.path.dirname(checkpoint_path), tag, "runner.yaml")
                    if os.path.isfile(tagged_runner_yaml):
                        return load(tagged_runner_yaml, *args, **kwargs)

        return super().read_config(checkpoint, *args, **kwargs)
