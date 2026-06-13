# DanLing
# Copyright (C) 2022-Present  DanLing

# This file is part of DanLing.

# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from __future__ import annotations

import os
import random
import re
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from datetime import timedelta
from time import perf_counter
from typing import Any, Callable, Iterator
from warnings import warn

import torch
import torch.distributed
import torch.distributed.distributed_c10d as dist_c10d
from chanfig import NestedDict
from torch import distributed as dist
from torch import nn, optim, utils
from torch.backends import cudnn
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

try:
    from numpy import random as np_random
except ImportError:
    np_random = None  # type: ignore[assignment]

from danling.data import to_device
from danling.optim import OPTIMIZERS, SCHEDULERS
from danling.optim.optimizer import (
    SCHEDULER_METRIC_UNSET,
    OptimizerContainer,
    OptimizerParameterCache,
    normalize_scheduler_interval,
    scheduler_requires_metric,
    step_scheduler,
)
from danling.utils import RoundDict, catch

from .base_runner import BaseRunner
from .checkpoints import TorchDistributedCheckpointManager, TorchFTCheckpointManager
from .compile import Compiler
from .config import RunnerConfig, normalize_stack_name
from .mixins import Fp8Mixin
from .telemetry import LoopTelemetry
from .utils import RunnerMode, get_precision, on_main_process


class TorchRunner(Fp8Mixin, BaseRunner):
    r"""
    PyTorch-native runner for training, evaluation, and inference.

    Use this runner for single-model PyTorch training with optional DDP,
    autocast/FP8, `torch.compile`, stateful dataloaders, metric logging, and
    file or torch.distributed.checkpoint persistence.

    Users must provide `self.model` before construction completes. Most
    training tasks also provide `self.criterion`, and either `self.optimizer`
    or `config.optim`. Datasets may be supplied through `self.datasets` and
    will be materialized into `StatefulDataLoader` instances during
    `__post_init__`.

    The default batch contract is intentionally simple:
    mappings use `input`/`target`, sequences use index 0/1, and any other value
    is treated as model input with no target. Override `train_step`,
    `evaluate_step`, or `infer_step` when a task needs a different contract.

    Attributes:
        model: Local model module after materialization (possibly DDP-wrapped).
        ema: Optional EMA/evaluation model.
        criterion: Loss callable used by default train/evaluate steps.
        optimizer: Optimizer used by the runner or backend engine.
        scheduler: Optional LR scheduler.
        optimizer_container: Helper that owns optimizer step, clipping,
            non-finite checks, and step-scheduler dispatch.
        compiler: `torch.compile` policy object.
        scheduler_interval: Effective scheduler interval (`"step"` or
            epoch/metric-style interval).
        scheduler_monitor: Optional metric path used for metric schedulers.
    """

    model: nn.Module
    ema: nn.Module | None = None
    criterion: Callable | None = None
    optimizer: optim.Optimizer | None = None
    scheduler: Any | None = None
    optimizer_container: OptimizerContainer | None = None
    compiler: Compiler
    scheduler_interval: str = "step"
    scheduler_monitor: str | None = None
    _train_pg_timeout_reduced: bool = False
    _profiler_context: Any | None = None
    _profiler: Any | None = None
    _pending_loss_normalizer: int | None = None
    _accumulation_divisor_local: float = 0.0
    _accumulation_mode: str | None = None
    _train_window_will_flush: bool = False
    _optimizer_parameter_cache: OptimizerParameterCache | None = None
    _supports_torchft_runtime: bool = True

    _VALID_CHECKPOINT_BACKENDS = frozenset({"file", "dcp"})

    @classmethod
    def _validate_checkpoint_backend(cls, backend: str) -> str:
        """Normalize and validate a resolved checkpoint backend value."""
        backend = str(backend).strip().lower()
        if backend not in cls._VALID_CHECKPOINT_BACKENDS:
            raise ValueError(f"invalid checkpoint backend: {backend!r}. Expected one of: 'auto', 'file', 'dcp'.")
        return backend

    def __init__(self, config) -> None:
        if not isinstance(config, RunnerConfig):
            config = RunnerConfig(config)
        config.stack = normalize_stack_name(config.get("stack", "ddp"))
        checkpoint_backend = str(config.get("ckpt.backend")).strip().lower()
        if checkpoint_backend == "auto":
            checkpoint_backend = "dcp" if self.world_size > 1 else "file"
        config["ckpt"]["backend"] = self._validate_checkpoint_backend(checkpoint_backend)
        super().__init__(config)

    def __post_init__(self):
        self._pending_loss_normalizer = None
        self._accumulation_divisor_local = 0.0
        self._accumulation_mode = None
        self._train_window_will_flush = False
        self._optimizer_parameter_cache = None
        if self.model is None:
            raise ValueError("cannot initialize TorchRunner: model is not initialized")
        if self.datasets:
            self.build_dataloaders()
        if self.fault_tolerance is not None and self.fault_tolerance.enabled and not self._supports_torchft_runtime:
            raise NotImplementedError(
                "TorchFT integration is currently supported by TorchRunner/DDP and ParallelRunner FSDP only"
            )
        self.compiler = Compiler(self.config.compile)
        self.setup_fp8()
        self.materialize_model()
        self.build_optimizer()
        self.build_scheduler()
        self._finalize_runtime_components()
        sched_cfg = self._get_scheduler_config()
        interval = sched_cfg.get("interval") if sched_cfg is not None else None
        monitor = sched_cfg.get("monitor") if sched_cfg is not None else None
        self.scheduler_interval = normalize_scheduler_interval(interval, self.scheduler)
        self.scheduler_monitor = None if monitor is None else str(monitor)
        self._bind_optimizer_container()
        self.auto_restore()
        self._init_profiler()
        super().__post_init__()

    def _finalize_runtime_components(self) -> None:
        """Hook for backend-specific engine/materialization after optimizer and scheduler build."""

    def init_distributed(self) -> None:
        r"""
        Initialize the distributed environment.

        The default implementation initializes the default torch.distributed
        process group from `WORLD_SIZE`/`RANK`/`LOCAL_RANK` environment
        variables when `WORLD_SIZE > 1`, sets the active CUDA device,
        broadcasts `self.timestamp` from rank 0, and seeds
        `elastic_state.restart_count` from `TORCHELASTIC_RESTART_COUNT`.

        **Called when:** once during `BaseRunner.__init__`, before
        `init_checkpoint_manager`, `init_fault_tolerance`, and
        `init_garbage_collection`. The runner is partially constructed at
        this point — `self.config`, `self.workspace`, `self.timestamp`, the
        dataloader container, and the default `FileCheckpointManager` are
        bound, but the model is not materialized and optimizers/dataloaders
        are not built.

        **Precondition:** environment variables `WORLD_SIZE`, `RANK`,
        `LOCAL_RANK` are set when running distributed. The default
        torch.distributed process group is **not** already initialized when
        `WORLD_SIZE > 1` — the runner owns process-group lifecycle.

        Raises:
            RuntimeError: the default process group is already initialized
                when `WORLD_SIZE > 1`.
            ValueError: `dist.init_timeout_seconds` is non-positive.

        **Side effects:** when `WORLD_SIZE > 1`, calls
        `dist.init_process_group(...)`, sets the active CUDA device when
        CUDA is available, and broadcasts `self.timestamp` from rank 0.
        Reads `TORCHELASTIC_RESTART_COUNT` into `elastic_state.restart_count`.

        !!! danger "Do not"
            - Initialize a process group via `dist.init_process_group(...)`
              outside the runner; the runner owns its lifecycle.
            - Build the model or dataloaders here; those happen in
              `__post_init__`.
            - Bind the checkpoint manager here; `init_checkpoint_manager`
              runs next.

        **Backend notes:**

        - `ParallelRunner` extends this hook: after calling `super()`, it
          builds the parallel topology (`build_topology`) and initializes
          per-axis process groups via `init_device_mesh`.
        - `DeepSpeedRunner` inherits the default; DeepSpeed reuses the
          default process group initialized here.
        """

        backend = self.config.get("dist.backend") or os.getenv("BACKEND")
        init_method = self.config.get("dist.init_method") or os.getenv("INIT_METHOD")
        init_timeout = self._dist_timeout("dist.init_timeout_seconds")
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK", "0"))
        runtime_device = self.device
        use_cuda_runtime = torch.cuda.is_available() and runtime_device.type == "cuda"
        runtime_device_index = runtime_device.index if runtime_device.index is not None else self.local_rank
        dist_ready = dist.is_available() and dist.is_initialized()
        if world_size > 1 and dist_ready:
            raise RuntimeError(
                "default process group is already initialized; Runner requires owning process-group lifecycle"
            )
        if world_size > 1:
            if use_cuda_runtime:
                torch.cuda.set_device(runtime_device_index)
            init_kwargs: dict[str, Any] = {
                "backend": backend,
                "init_method": init_method,
                "world_size": world_size,
                "rank": rank,
            }
            if init_timeout is not None:
                init_kwargs["timeout"] = init_timeout
            dist.init_process_group(**init_kwargs)
            dist_ready = bool(dist.is_available() and dist.is_initialized())

        if dist_ready and use_cuda_runtime:
            torch.cuda.set_device(runtime_device_index)

        if dist_ready and self.world_size > 1:
            object_list = [self.timestamp]
            dist.broadcast_object_list(object_list)
            self.timestamp = str(object_list[0])

        restart_count = os.getenv("TORCHELASTIC_RESTART_COUNT")
        if restart_count is not None:
            self.elastic_state.restart_count = int(restart_count)

        self._train_pg_timeout_reduced = False

    def init_checkpoint_manager(self) -> None:
        """
        Bind the checkpoint manager corresponding to `config.ckpt.backend`.

        The default dispatches by backend: when the backend is `"dcp"`, it
        binds a `TorchDistributedCheckpointManager` (or
        `TorchFTCheckpointManager` when dataloader checkpoints are
        enabled). For `"file"` it leaves the `FileCheckpointManager` already
        bound by `BaseRunner.__init__` in place.

        **Called when:** once during `BaseRunner.__init__`, after
        `init_distributed` and before `init_fault_tolerance`. The default
        `FileCheckpointManager` is already bound at this point — overrides
        should swap it via `set_checkpoint_manager(...)`, not by direct
        attribute assignment.

        **Precondition:** `config.ckpt.backend` is normalized to one
        of `{"file", "dcp"}` (TorchRunner does this in `__init__`). When
        the backend is `"dcp"`, the default process group is initialized
        for distributed runs.

        **Side effects:** swaps `self.checkpoint_manager` via
        `set_checkpoint_manager(...)` when the backend differs from
        `"file"`. The prior manager is closed with a zero timeout.

        !!! danger "Do not"
            - Set `self.checkpoint_manager` directly; use
              `set_checkpoint_manager` so the prior manager is closed
              cleanly.
            - Initialize fault tolerance here; `init_fault_tolerance` runs
              next.
            - Bind the model or dataloaders here.

        **Backend notes:**

        - `DeepSpeedRunner` coerces `config.ckpt.backend` to `"file"`
          in `__init__`, so this hook is a no-op for that backend.
        - `ParallelRunner` coerces the backend to `"dcp"`, so this hook
          always binds `TorchDistributedCheckpointManager` or
          `TorchFTCheckpointManager`.
        """
        checkpoint_backend = str(self.config.get("ckpt.backend")).lower()
        if checkpoint_backend == "dcp":
            ft_checkpoint_enabled = bool(
                self.config.get("ft.enabled", False) or self.config.get("ckpt.dataloader_checkpoint.enabled", False)
            )
            manager_cls = TorchFTCheckpointManager if ft_checkpoint_enabled else TorchDistributedCheckpointManager
            self.set_checkpoint_manager(manager_cls(self))
            return
        # Backend is normalized to {"file", "dcp"} in `__init__`; "file" is the
        # remaining case and reuses the default `FileCheckpointManager` that
        # `BaseRunner.__init__` already bound.

    def _dist_timeout(self, key: str) -> timedelta | None:
        value = self.config.get(key)
        if value is None:
            return None
        seconds = int(value)
        if seconds <= 0:
            raise ValueError(f"{key} must be a positive integer, got {seconds}")
        return timedelta(seconds=seconds)

    def _timeout_process_groups(self) -> tuple[Any | None, ...]:
        groups: list[Any | None] = [None]
        if self.fault_tolerance is not None and self.fault_tolerance.replicate_process_group is not None:
            groups.append(self.fault_tolerance.replicate_process_group)
        return tuple(groups)

    def _set_process_group_timeout(self, timeout: timedelta) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        set_pg_timeout = getattr(dist_c10d, "_set_pg_timeout", None)
        if not callable(set_pg_timeout):
            warn(
                "torch.distributed does not expose process-group timeout mutation; "
                "skipping dist.train_timeout_seconds update",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        for group in self._timeout_process_groups():
            backend = str(dist.get_backend() if group is None else dist.get_backend(group)).lower()
            if backend != "nccl":
                continue

            barrier_kwargs = {} if group is None else {"group": group}
            if torch.cuda.is_available():
                dist.barrier(device_ids=[torch.cuda.current_device()], **barrier_kwargs)
                torch.cuda.synchronize()
            else:
                dist.barrier(**barrier_kwargs)

            try:
                set_pg_timeout(timeout, group)
            except TypeError:
                if group is not None:
                    warn(
                        "torch.distributed does not support subgroup timeout mutation; "
                        "skipping dist.train_timeout_seconds update for a non-default process group",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                set_pg_timeout(timeout)
            except Exception as exc:
                group_name = "default" if group is None else "subgroup"
                warn(f"failed to update {group_name} process-group timeout: {exc}", RuntimeWarning, stacklevel=2)

    def _maybe_reduce_train_process_group_timeout(self) -> None:
        if self._train_pg_timeout_reduced:
            return
        if self.train_state.global_step != 1:
            return
        timeout = self._dist_timeout("dist.train_timeout_seconds")
        if timeout is None:
            return
        self._set_process_group_timeout(timeout)
        self._train_pg_timeout_reduced = True

    def destroy_process_group(self) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        try:
            dist.destroy_process_group()
        except Exception as exc:
            warn(f"failed to destroy default process group: {exc}", RuntimeWarning, stacklevel=2)

    def _profiler_activities(self, configured: object | None) -> list[torch.profiler.ProfilerActivity]:
        if configured is None:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available() and self.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            return activities

        if isinstance(configured, str):
            configured = (configured,)
        if not isinstance(configured, Sequence):
            raise ValueError("profiling.activities must be a string or sequence of strings")

        aliases = {"cpu": "CPU", "cuda": "CUDA", "gpu": "CUDA", "xpu": "XPU"}
        activities = []
        for activity in configured:
            if isinstance(activity, torch.profiler.ProfilerActivity):
                activities.append(activity)
                continue
            name = aliases.get(str(activity).strip().lower(), str(activity).strip().upper())
            if not hasattr(torch.profiler.ProfilerActivity, name):
                raise ValueError(f"unsupported profiling activity: {activity!r}")
            activities.append(getattr(torch.profiler.ProfilerActivity, name))
        return activities

    def _init_profiler(self) -> None:
        profiling = self.config.get("profiling")
        if not isinstance(profiling, Mapping) or not bool(profiling.get("enabled", False)):
            return

        wait = int(profiling.get("wait", 1))
        warmup = int(profiling.get("warmup", 1))
        active = int(profiling.get("active", 3))
        repeat = profiling.get("repeat")
        if wait < 0:
            raise ValueError(f"profiling.wait must be a non-negative integer, got {wait}")
        if warmup < 0:
            raise ValueError(f"profiling.warmup must be a non-negative integer, got {warmup}")
        if active <= 0:
            raise ValueError(f"profiling.active must be a positive integer, got {active}")
        if repeat is not None:
            repeat = int(repeat)
            if repeat <= 0:
                raise ValueError(f"profiling.repeat must be a positive integer, got {repeat}")

        activities = self._profiler_activities(profiling.get("activities"))

        schedule_kwargs: dict[str, Any] = {"wait": wait, "warmup": warmup, "active": active}
        if repeat is not None:
            schedule_kwargs["repeat"] = repeat

        trace_dir = os.fsdecode(str(profiling.get("trace_dir", "profiles")))
        if not os.path.isabs(trace_dir):
            trace_dir = os.path.join(self.workspace.dir, trace_dir)
        trace_dir = os.path.join(trace_dir, self.timestamp, f"rank-{self.rank:05d}")
        os.makedirs(trace_dir, exist_ok=True)
        profile_kwargs: dict[str, Any] = {
            "activities": activities,
            "schedule": torch.profiler.schedule(**schedule_kwargs),
            "on_trace_ready": torch.profiler.tensorboard_trace_handler(trace_dir),
            "record_shapes": bool(profiling.get("record_shapes", False)),
            "profile_memory": bool(profiling.get("profile_memory", False)),
            "with_stack": bool(profiling.get("with_stack", False)),
            "with_flops": bool(profiling.get("with_flops", False)),
            "with_modules": bool(profiling.get("with_modules", False)),
            "acc_events": bool(profiling.get("acc_events", False)),
        }
        if profiling.get("use_cuda") is not None:
            profile_kwargs["use_cuda"] = bool(profiling.get("use_cuda"))
        post_processing_timeout_seconds = profiling.get("post_processing_timeout_seconds")
        if post_processing_timeout_seconds is not None:
            profile_kwargs["post_processing_timeout_s"] = float(post_processing_timeout_seconds)

        profiler_context = torch.profiler.profile(
            **profile_kwargs,
        )
        profiler = profiler_context.__enter__()
        if hasattr(profiler, "step_num"):
            profiler.step_num = self.train_state.global_step
        self._profiler_context = profiler_context
        self._profiler = profiler

    def _step_profiler(self) -> None:
        if self._profiler is None:
            return
        self._profiler.step()

    def _close_profiler(self) -> None:
        profiler_context = self._profiler_context
        self._profiler_context = None
        self._profiler = None
        if profiler_context is None:
            return
        profiler_context.__exit__(None, None, None)

    @on_main_process
    def init_tensorboard(self, *args, **kwargs) -> None:
        r"""
        Set up TensorBoard SummaryWriter.
        """

        from torch.utils.tensorboard.writer import SummaryWriter  # pylint: disable=C0415

        tensorboard_config = self.config.tensorboard
        for key in ("log_dir", "comment", "purge_step", "max_queue", "flush_secs", "filename_suffix"):
            if key not in kwargs and tensorboard_config.get(key) is not None:
                kwargs[key] = tensorboard_config[key]
        if "log_dir" not in kwargs:
            kwargs["log_dir"] = os.path.join(self.workspace.dir, "tensorboard", self.timestamp)

        self.writer = SummaryWriter(*args, **kwargs)
        self.writer.add_scalar = catch(OSError, verbose=False)(self.writer.add_scalar)  # type: ignore[method-assign]

    def set_seed(self, seed: int | None = None, bias: int | bool | None = None) -> int:
        r"""
        Set up random seed.

        Args:
            seed: Random seed to set.
                Defaults to `self.config.seed` (`config.seed`).

            bias: Make the seed different for each processes.
                This is used to ensure the data augmentation are applied differently on every processes.
                Defaults to `self.rank`.
                Set to `False` to disable this feature.
        Returns:
            Random seed set.
        """

        base_seed = seed if seed is not None else self.config.seed  # type: ignore[assignment]
        if base_seed is None:
            base_seed = random.randint(0, 2**32 - 1)
            if self.distributed and dist.is_initialized():
                object_list = [base_seed]
                dist.broadcast_object_list(object_list)
                base_seed = object_list[0]
        base_seed = int(base_seed)
        # Keep `config.seed` as the global/base seed (before per-rank bias).
        self.config.seed = base_seed

        process_seed = base_seed
        if bias is None:
            if self.fault_tolerance is not None:
                _, bias = self.fault_tolerance.data_parallel_info(self.world_size, self.rank)
            else:
                bias = self.rank
        if bias:
            process_seed += int(bias)

        torch.manual_seed(process_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(process_seed)
        if np_random is not None:
            np_random.seed(process_seed)
        random.seed(process_seed)
        self.rng_state.python = random.getstate()
        self.rng_state.numpy = np_random.get_state() if np_random is not None else None
        self.rng_state.torch_cpu = torch.get_rng_state()
        if torch.cuda.is_available():
            self.rng_state.torch_cuda = torch.cuda.get_rng_state_all()
        else:
            self.rng_state.torch_cuda = None
        return process_seed

    def set_deterministic(self) -> None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    def materialize_model(self) -> None:
        """
        Move the model to the runtime device, optionally compile, and wrap
        with DDP when distributed.

        The default is a single-module DDP-style materialization: it moves
        `self.model` to `self.device`, applies any FP8 module policy when
        FP8 is enabled, runs `torch.compile` via `self.compiler` (under the
        DDP-optimizer context when wrapping is needed), and wraps the result
        with `nn.parallel.DistributedDataParallel` when world size > 1.

        **Called when:** once during `__post_init__`, after `setup_fp8()`
        and before `build_optimizer()`. The order matters — the optimizer
        must see post-wrap parameters.

        **Precondition:** `self.model` is set (typically by the user before
        constructing the runner). `self.device` resolves to the runtime
        device.

        Raises:
            ValueError: `self.model` is not initialized.

        **Side effects:** moves `self.model` to `self.device`; applies FP8
        module policy when `self.fp8_enabled`; compiles via
        `self.compiler.compile(...)` under the DDP-optimizer context when
        wrapping is needed; wraps with `DistributedDataParallel` for world
        size > 1. Moves `self.ema` to device when EMA is bound.

        !!! danger "Do not"
            - Build the optimizer or scheduler here; they run after this
              hook.
            - Skip the device move when overriding (tensors must live on
              `self.device` before the forward pass).
            - Re-wrap an already-wrapped model (e.g., DDP-wrap a DDP module).

        **Backend notes:**

        - `DeepSpeedRunner` overrides this hook to move the model to device
          and compile only; the DeepSpeed engine wraps the model later in
          `_finalize_runtime_components`.
        - `ParallelRunner` overrides this hook for FSDP2, pipeline-parallel
          schedules, and tensor/expert/context parallelism (via the
          `parallelize_model` and `apply_activation_checkpointing` hooks).
        """
        if self.model is None:
            raise ValueError("cannot materialize model: model is not initialized")

        model = self.model.to(self.device)
        self.model = model
        if self.fp8_enabled:
            self.apply_fp8_module_policy_to_model_parts()
            model = self.model
        should_wrap_ddp = self.distributed and not isinstance(
            model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)
        )
        with self.compiler.ddp_optimizer() if should_wrap_ddp else nullcontext():
            model = self.compiler.compile(model)
        if should_wrap_ddp:
            model = nn.parallel.DistributedDataParallel(
                model,
                find_unused_parameters=bool(self.config.ddp.find_unused_parameters),
            )
        self.model = model

        if self.ema is not None:
            self.ema = self.ema.to(self.device)

    def unwrap(self, model: nn.Module) -> nn.Module:
        if isinstance(model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)):
            return model.module
        return model

    def _iter_unique_parameters(self, modules: Sequence[nn.Module]) -> Iterator[nn.Parameter]:
        seen: set[int] = set()
        for module in modules:
            for parameter in module.parameters():
                parameter_id = id(parameter)
                if parameter_id in seen:
                    continue
                seen.add(parameter_id)
                yield parameter

    def _iter_unique_named_parameters(
        self, modules: Sequence[nn.Module], prefixes: Sequence[str] | None = None
    ) -> Iterator[tuple[str, nn.Parameter]]:
        seen: set[int] = set()
        if prefixes is None:
            prefixes = ("",) * len(modules)
        if len(prefixes) != len(modules):
            raise ValueError("prefix count must match module count")
        for module, prefix in zip(modules, prefixes):
            for name, parameter in module.named_parameters():
                parameter_id = id(parameter)
                if parameter_id in seen:
                    continue
                seen.add(parameter_id)
                yield f"{prefix}{name}", parameter

    def iter_optimizer_parameters(self) -> Iterator[nn.Parameter]:
        if self.model is None:
            return
        yield from self._iter_unique_parameters((self.unwrap(self.model),))

    def iter_optimizer_named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        if self.model is None:
            return
        yield from self._iter_unique_named_parameters((self.unwrap(self.model),))

    def _optimizer_param_group_options(
        self,
        group_cfg: Mapping[str, Any],
        optimizer_cfg: Mapping[str, Any],
        *,
        index: int,
    ) -> dict[str, Any]:
        options = {
            str(key): value
            for key, value in group_cfg.items()
            if key not in {"pattern", "params", "lr_multiplier", "weight_decay_multiplier", "beta1", "beta2"}
        }
        if "lr_multiplier" in group_cfg:
            if optimizer_cfg.get("lr") is None:
                raise ValueError(f"optim.param_groups[{index}].lr_multiplier requires optim.lr")
            options["lr"] = float(optimizer_cfg["lr"]) * float(group_cfg["lr_multiplier"])
        if "weight_decay_multiplier" in group_cfg:
            if optimizer_cfg.get("weight_decay") is None:
                raise ValueError(f"optim.param_groups[{index}].weight_decay_multiplier requires optim.weight_decay")
            options["weight_decay"] = float(optimizer_cfg["weight_decay"]) * float(group_cfg["weight_decay_multiplier"])

        beta1 = group_cfg.get("beta1")
        beta2 = group_cfg.get("beta2")
        if beta1 is not None or beta2 is not None:
            if optimizer_cfg.get("betas") is None:
                raise ValueError(f"optim.param_groups[{index}].beta1/beta2 requires optim.betas")
            beta1_default, beta2_default = optimizer_cfg["betas"]
            options["betas"] = (
                float(beta1_default if beta1 is None else beta1),
                float(beta2_default if beta2 is None else beta2),
            )
        return options

    def _build_optimizer_param_groups(
        self, optimizer_cfg: Mapping[str, Any]
    ) -> list[nn.Parameter] | list[dict[str, Any]]:
        group_configs = optimizer_cfg.get("param_groups")
        if group_configs is None:
            return list(self.iter_optimizer_parameters())
        if isinstance(group_configs, (str, bytes, Mapping)) or not isinstance(group_configs, Sequence):
            raise ValueError("optim.param_groups must be a sequence of mappings")

        named_parameters = list(self.iter_optimizer_named_parameters())
        if not named_parameters:
            return []

        assigned: set[int] = set()
        param_groups: list[dict[str, Any]] = []
        for index, group_cfg in enumerate(group_configs):
            if not isinstance(group_cfg, Mapping):
                raise ValueError(f"optim.param_groups[{index}] must be a mapping")
            pattern = group_cfg.get("pattern")
            if pattern is None:
                raise ValueError(f"optim.param_groups[{index}] requires `pattern`")
            regex = re.compile(str(pattern))
            parameters = [
                parameter
                for name, parameter in named_parameters
                if id(parameter) not in assigned and regex.search(name) is not None
            ]
            if not parameters:
                warn(
                    f"optim.param_groups[{index}] pattern {pattern!r} matched no parameters",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            assigned.update(id(parameter) for parameter in parameters)
            param_groups.append(
                {
                    "params": parameters,
                    **self._optimizer_param_group_options(group_cfg, optimizer_cfg, index=index),
                }
            )

        unmatched = [parameter for _name, parameter in named_parameters if id(parameter) not in assigned]
        if unmatched:
            param_groups.append({"params": unmatched})
        return param_groups

    def build_optimizer(self) -> None:
        """
        Auto-build the optimizer from `config.optim` when `self.optimizer`
        is absent.

        The default iterates parameters via `iter_optimizer_parameters` and
        dispatches to the `OPTIMIZERS` registry with the merged config. If
        `optim.param_groups` is configured, entries are matched by regex
        `search` against `iter_optimizer_named_parameters`; unmatched
        parameters keep the optimizer-level defaults.

        **Called when:** once during `TorchRunner.__post_init__`, after
        `materialize_model` (so parameters reflect DDP/FSDP wrapping) and
        before `build_scheduler`.

        **Precondition:** `self.model` is materialized and on `self.device`.
        `self.optimizer` is `None` (the auto-build is skipped when the user
        has already bound an optimizer).

        **Side effects:** sets `self.optimizer` to the registry-built
        instance.

        !!! danger "Do not"
            - Run before `materialize_model`; parameters won't reflect
              DDP/FSDP wrapping.
            - Build a scheduler here.
            - Override parameter enumeration here; override
              `iter_optimizer_parameters` / `iter_optimizer_named_parameters`
              instead so subclass topology (e.g., `ParallelRunner.model_parts`)
              is preserved.

        **Backend notes:**

        - `DeepSpeedRunner` inherits this hook; DeepSpeed may replace the
          optimizer with a DeepSpeed-managed instance during
          `_finalize_runtime_components`.
        - `ParallelRunner` inherits this hook but overrides
          `iter_optimizer_parameters` to enumerate `self.model_parts`.
        """
        if self.optimizer is not None or self.model is None:
            return
        optimizer_cfg = self.config.get("optim")
        if not isinstance(optimizer_cfg, Mapping) or not optimizer_cfg:
            return
        optimizer_kwargs = {str(key): value for key, value in optimizer_cfg.items() if value is not None}
        optimizer_kwargs.pop("param_groups", None)
        if "type" not in optimizer_kwargs:
            return
        parameters = self._build_optimizer_param_groups(optimizer_cfg)
        if not parameters:
            return
        self.optimizer = OPTIMIZERS.build(params=parameters, **optimizer_kwargs)

    def _get_scheduler_config(self) -> Mapping[str, Any] | None:
        scheduler_cfg = self.config.get("sched")
        if not isinstance(scheduler_cfg, Mapping):
            return None
        return scheduler_cfg

    def build_scheduler(self) -> None:
        """
        Auto-build the LR scheduler from `config.sched` when
        `self.scheduler` is absent.

        The default pops `interval` and `monitor` from the config (those
        drive runner-level dispatch, not scheduler construction), defaults
        `total_steps` to `self.steps` when computable, and dispatches to
        the `SCHEDULERS` registry with `self.optimizer` and the merged
        config.

        **Called when:** once during `TorchRunner.__post_init__`, after
        `build_optimizer`.

        **Precondition:** `self.optimizer` is bound. `self.scheduler` is
        `None` (the auto-build is skipped when the user has already bound a
        scheduler).

        **Side effects:** sets `self.scheduler` to the registry-built
        instance.

        !!! danger "Do not"
            - Run before `build_optimizer`; the scheduler must wrap an
              optimizer.
            - Set scheduler interval or monitor here; configure them via
              `config.sched.interval` / `config.sched.monitor`.

        **Backend notes:**

        - `DeepSpeedRunner` inherits this hook; the scheduler may be handed
          to the DeepSpeed engine in `_finalize_runtime_components` when
          its effective interval is `"step"`. Otherwise the runner retains
          it.
        """
        if self.scheduler is not None or self.optimizer is None:
            return
        sched_cfg = self._get_scheduler_config()
        if not isinstance(sched_cfg, Mapping) or not sched_cfg:
            return
        scheduler_kwargs = {str(key): value for key, value in sched_cfg.items() if value is not None}
        scheduler_kwargs.pop("interval", None)
        scheduler_kwargs.pop("monitor", None)
        if "type" not in scheduler_kwargs:
            return
        if "total_steps" not in scheduler_kwargs:
            steps = self.steps
            if steps is not None:
                scheduler_kwargs["total_steps"] = steps
        self.scheduler = SCHEDULERS.build(self.optimizer, **scheduler_kwargs)

    def _bind_optimizer_container(self) -> None:
        if self.optimizer is None:
            self.optimizer_container = None
            return
        self.optimizer_container = OptimizerContainer(
            self.optimizer,
            scheduler=self.scheduler,
            scheduler_interval=self.scheduler_interval,
        )

    def _resolve_scheduler_metric(self, result: Mapping[str, Any]) -> Any:
        def scalarize(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(
                        "scheduler monitor must resolve to a scalar metric, "
                        f"but got tensor with shape {tuple(value.shape)}"
                    )
                return value.item()
            return value

        monitor = self.scheduler_monitor or self.config.score.metric

        if "." in monitor:
            value: Any = result
            for key in monitor.split("."):
                if not isinstance(value, Mapping) or key not in value:
                    raise ValueError(
                        f"could not resolve sched.monitor={monitor!r} from aggregated result {dict(result)!r}"
                    )
                value = value[key]
            return scalarize(value)

        score_split = self.score_split
        if score_split is not None:
            split_result = result.get(score_split)
            if isinstance(split_result, Mapping) and monitor in split_result:
                return scalarize(split_result[monitor])

        if monitor in result and not isinstance(result[monitor], Mapping):
            return scalarize(result[monitor])

        matches: list[tuple[str, Any]] = []
        for split_name, split_result in result.items():
            if isinstance(split_result, Mapping) and monitor in split_result:
                matches.append((split_name, split_result[monitor]))

        if len(matches) == 1:
            return scalarize(matches[0][1])
        if len(matches) > 1:
            splits = ", ".join(split_name for split_name, _ in matches)
            raise ValueError(
                f"ambiguous sched.monitor={monitor!r}: matched multiple splits ({splits}). "
                "Use '<split>.<metric>' to disambiguate."
            )

        raise ValueError(f"could not resolve sched.monitor={monitor!r} from aggregated result {dict(result)!r}")

    def _step_epoch_scheduler(self, result: Mapping[str, Any]) -> bool:
        if self.scheduler is None or self.scheduler_interval != "epoch":
            return False

        scheduler_metric = SCHEDULER_METRIC_UNSET
        if scheduler_requires_metric(self.scheduler):
            scheduler_metric = self._resolve_scheduler_metric(result)

        if self.optimizer_container is not None:
            return self.optimizer_container.step_scheduler(scheduler_metric=scheduler_metric)
        return step_scheduler(self.scheduler, scheduler_metric=scheduler_metric)

    def build_dataloaders(self):
        """
        Build dataloaders for dataset splits not already materialized.

        The default iterates `self.datasets`, merges `config.dataloader`
        defaults with split-specific overrides (`config.dataloader.<split>`),
        constructs a sampler via `build_datasampler`, and wraps each dataset
        in a `StatefulDataLoader` using `self.collate_fn`. Train splits
        default to `shuffle=True` and `drop_last=True`; non-train splits
        default to the opposite.

        **Called when:** once during `TorchRunner.__post_init__` when
        `self.datasets` is non-empty.

        **Precondition:** `self.datasets` is populated (typically by the
        user before constructing the runner). `self.dataloaders` is bound
        to a default-constructed `DataLoaderDict`.

        **Side effects:** populates `self.dataloaders[split]` for each
        split in `self.datasets` not already materialized. Existing entries
        in `self.dataloaders` are left untouched.

        !!! danger "Do not"
            - Override sampler logic here; override `build_datasampler`
              instead.
            - Override collation; set `self.collate_fn` or override
              `collate_fn` (classmethod) instead.
            - Bind the optimizer or scheduler here.

        **Backend notes:**

        - `ParallelRunner` substitutes `self.dataloaders` with a proxying
          dict in `__init__` so non-first/last pipeline stages receive a
          `StepProxyLoader` view. The build logic itself is inherited.
        """
        datasets = {k: d for k, d in self.datasets.items() if k not in self.dataloaders}
        dataloader_config = self.config.get("dataloader", NestedDict())
        default_kwargs = NestedDict({k: v for k, v in dataloader_config.items() if k not in self.datasets})
        split_kwargs = NestedDict({k: v for k, v in dataloader_config.items() if k in self.datasets})
        for k, dataset in datasets.items():
            kwargs = NestedDict(default_kwargs)
            if k in split_kwargs:
                kwargs.merge(split_kwargs[k], overwrite=True)
            is_train_split = k in self.train_splits
            shuffle = kwargs.pop("shuffle", is_train_split)
            collate_fn = kwargs.pop("collate_fn", self.collate_fn)
            batch_sampler = kwargs.pop("batch_sampler", None)
            if batch_sampler is not None:
                kwargs.pop("batch_size", None)
                kwargs.pop("drop_last", None)
                kwargs.pop("sampler", None)
                self.dataloaders[k] = StatefulDataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=collate_fn,
                    **kwargs,
                )
                continue
            kwargs.setdefault("drop_last", is_train_split)
            sampler = kwargs.pop("sampler", None)
            if sampler is None:
                sampler = self.build_datasampler(dataset, split=k, shuffle=shuffle)
            self.dataloaders[k] = StatefulDataLoader(dataset, sampler=sampler, collate_fn=collate_fn, **kwargs)

    def build_datasampler(self, dataset: Any, *, split: str, shuffle: bool) -> Any:
        """
        Build the sampler for one dataset split.

        **Called when:** `build_dataloaders` materializes a split from
        `self.datasets`.

        Args:
            dataset: Dataset object for the split.
            split: Split name being materialized.
            shuffle: Whether this split should be sampled in shuffled order.

        Returns:
            A local random/sequential sampler in single-process mode, or a
            `DistributedSampler` in distributed mode.

        **Backend notes:**

        - `ParallelRunner` overrides replica/rank selection so data-parallel
          sampling follows its topology instead of raw global rank.
        """
        if self.distributed:
            num_replicas = self.world_size
            rank = self.rank
            if self.fault_tolerance is not None:
                num_replicas, rank = self.fault_tolerance.data_parallel_info(num_replicas, rank)
            return utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
            )
        return utils.data.RandomSampler(dataset) if shuffle else utils.data.SequentialSampler(dataset)

    @staticmethod
    def collate_fn(batch):
        return utils.data.dataloader.default_collate(batch)

    def to_device(self, data: Any):
        """Move one batch to runtime device; override in subclasses for custom fast paths."""
        return to_device(data, self.device)

    def _step_mode_split_budget(
        self,
        *,
        remaining_steps: int,
        remaining_splits: int,
        loader: Any,
    ) -> int:
        if remaining_steps <= 0:
            return 0
        if remaining_splits <= 0:
            return remaining_steps

        fair_share = max((remaining_steps + remaining_splits - 1) // remaining_splits, 1)
        loader_length = self._loader_length(loader)
        if loader_length is None:
            return fair_share

        loader_step_budget = max((loader_length + self.accum_steps - 1) // self.accum_steps, 1)
        return min(fair_share, loader_step_budget, remaining_steps)

    @staticmethod
    def _set_loader_epoch(loader: Any, epoch: int) -> None:
        batch_sampler = getattr(loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)  # type: ignore[union-attr]

    def loop_time(self, *, sync: bool = False) -> float:
        if sync and torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return perf_counter()

    @property
    def reports_batch_telemetry(self) -> bool:
        return True

    @staticmethod
    def _as_int_or_none(value: Any) -> int | None:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return int(value)
        if torch.is_tensor(value) and value.numel() == 1:
            return int(value.detach().item())
        return None

    def _mapping_loss_normalizer(self, mapping: Mapping[str, Any] | None) -> int | None:
        if mapping is None:
            return None
        for key in ("loss_normalizer", "num_valid_tokens", "valid_tokens", "num_tokens", "token_count"):
            if key in mapping:
                normalizer = self._as_int_or_none(mapping[key])
                if normalizer is not None:
                    return normalizer
        return None

    def _tensor_loss_normalizer(self, target: Any) -> int | None:
        if not torch.is_tensor(target):
            return None
        ignore_index = getattr(self.criterion, "ignore_index", None)
        if ignore_index is not None:
            return int((target != int(ignore_index)).sum().item())
        if getattr(self.criterion, "reduction", None) == "mean":
            return int(target.numel())
        return None

    def _get_loss_normalizer(self, data: Any) -> int | None:
        if isinstance(data, Mapping):
            explicit = self._mapping_loss_normalizer(data)
            if explicit is not None:
                return explicit

            target = data.get("target")
            if isinstance(target, Mapping):
                explicit = self._mapping_loss_normalizer(target)
                if explicit is not None:
                    return explicit
            if target is not None:
                normalizer = self._tensor_loss_normalizer(target)
                if normalizer is not None:
                    return normalizer

            inputs = data.get("input")
            if isinstance(inputs, Mapping):
                attention_mask = inputs.get("attention_mask")
                if isinstance(attention_mask, torch.Tensor):
                    return int(attention_mask.detach().sum().item())
            return None

        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            target = data[1] if len(data) > 1 else None
            if isinstance(target, Mapping):
                explicit = self._mapping_loss_normalizer(target)
                if explicit is not None:
                    return explicit
            if target is not None:
                normalizer = self._tensor_loss_normalizer(target)
                if normalizer is not None:
                    return normalizer
        return None

    def _loss_normalizer_sync_divisor(self) -> int:
        if self.fault_tolerance is not None and self.fault_tolerance.replicate_process_group is not None:
            return max(int(dist.get_world_size(group=self.fault_tolerance.replicate_process_group)), 1)
        if dist.is_available() and dist.is_initialized():
            return max(self.world_size, 1)
        return 1

    def _reduce_loss_normalizer_total(self, local_total: float) -> float:
        if local_total <= 0:
            return local_total
        if self._loss_normalizer_sync_divisor() <= 1:
            return local_total
        if not (dist.is_available() and dist.is_initialized()):
            return local_total

        device = self.all_reduce_device()
        total_tensor = torch.tensor(local_total, dtype=torch.float64, device=device)
        self.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        return float(total_tensor.item())

    def all_reduce_device(self) -> torch.device:
        if self.distributed and dist.is_available() and dist.is_initialized():
            group = self.all_reduce_group()
            if group is not None:
                try:
                    backend = str(dist.get_backend(group=group)).lower()
                except TypeError:
                    backend = str(dist.get_backend(group)).lower()
                except (RuntimeError, ValueError):
                    return torch.device("cpu")
            else:
                backend = str(dist.get_backend()).lower()
            if "gloo" in backend or "mpi" in backend:
                return torch.device("cpu")
            if torch.cuda.is_available() and self.device.type == "cuda":
                return self.device
        return torch.device("cpu")

    def all_reduce_group(self):
        if self.fault_tolerance is not None and self.fault_tolerance.replicate_process_group is not None:
            return self.fault_tolerance.replicate_process_group
        return None

    def all_reduce(self, tensor: torch.Tensor, *, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """Reduce tensor over the runner's replica/data-parallel collective domain."""
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=op, group=self.all_reduce_group())
        return tensor

    def _sync_optimizer_skip_decision(self, should_skip: bool) -> bool:
        if not (self.distributed and dist.is_available() and dist.is_initialized()):
            return should_skip
        payload = torch.tensor(float(should_skip), device=self.all_reduce_device())
        self.all_reduce(payload, op=dist.ReduceOp.MAX)
        return payload.item() > 0

    def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Average-reduce tensor over the runner's collective domain."""
        if not (dist.is_available() and dist.is_initialized()):
            return tensor
        group = self.all_reduce_group()
        group_size = max(self.world_size if group is None else dist.get_world_size(group=group), 1)
        if group_size <= 1:
            return tensor

        original_device = tensor.device
        payload_device = self.all_reduce_device()
        payload = tensor if original_device == payload_device else tensor.to(payload_device)
        self.all_reduce(payload, op=dist.ReduceOp.SUM)
        payload = payload / group_size
        if payload.device != original_device:
            payload = payload.to(original_device)
        return payload

    def reduce_loss_for_logging(self, loss: torch.Tensor | None, loss_n: int | None) -> torch.Tensor | None:
        """Detach and all-reduce weighted loss tensor for logging."""
        if loss is None:
            return None
        loss_value = loss.detach().to(dtype=torch.float64)
        if loss_value.ndim > 0:
            loss_value = loss_value.mean()
        normalizer = float(max(int(loss_n or 1), 1))
        payload_device = self.all_reduce_device()
        payload = torch.stack(
            (
                loss_value.to(device=payload_device) * normalizer,
                torch.tensor(normalizer, dtype=torch.float64, device=payload_device),
            )
        )
        self.all_reduce(payload, op=dist.ReduceOp.SUM)
        if payload[1].item() <= 0:
            return None
        return payload[0] / payload[1]

    def _reset_accumulation_normalization(self) -> None:
        """Clear the per-window accumulation state.

        Called at loop start and on every optimizer flush. After reset, the
        next call to `_scaled_loss_for_backward` re-classifies the window mode
        from its first batch's normalizer.
        """
        self._pending_loss_normalizer = None
        self._accumulation_divisor_local = 0.0
        self._accumulation_mode = None

    def _loss_scale_for_backward(self) -> float:
        """Consume the pending loss-normalizer signal and return this micro-step's loss scale."""
        loss_normalizer = self._pending_loss_normalizer
        if self._accumulation_mode is None:
            self._accumulation_mode = (
                "weighted" if loss_normalizer is not None and int(loss_normalizer) > 0 else "uniform"
            )

        if self._accumulation_mode == "weighted":
            if loss_normalizer is None or int(loss_normalizer) <= 0:
                raise ValueError(
                    "loss normalizer became unavailable within the current accumulation window. "
                    "Override `train_step()` or provide consistent batch metadata for weighted normalization."
                )
            normalizer = float(int(loss_normalizer))
            self._accumulation_divisor_local += normalizer
            self._pending_loss_normalizer = None
            return normalizer

        self._accumulation_divisor_local += 1.0
        self._pending_loss_normalizer = None
        return 1.0

    def _scaled_loss_for_backward(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale and accumulate loss for one micro-step inside an accumulation window.

        Accumulation contract (window-local; reset on optimizer flush):

        1. **Mode detection.** First micro-step decides the window mode from
           ``self._pending_loss_normalizer``:
              - non-empty positive normalizer → ``"weighted"``
              - ``None`` or non-positive → ``"uniform"``
        2. **Mode is sticky.** Once the window picks ``"weighted"``, every
           subsequent micro-step in that window MUST also publish a positive
           normalizer; missing one raises with guidance to override
           `train_step()` or homogenize batch metadata. A ``"uniform"`` window
           ignores per-batch normalizers entirely.
        3. **Producer/consumer.** ``train_epoch`` / ``train_steps`` set
           ``self._pending_loss_normalizer`` from ``_get_loss_normalizer(data)``
           before each ``train_step`` call; this method consumes and clears it.
        4. **Override safety.** Subclasses overriding ``train_step`` are
           responsible for keeping the normalizer signal consistent across the
           window — either always present (weighted mode) or always absent
           (uniform mode). Mixing within one window is a programmer error.

        See `_reset_accumulation_normalization` for window boundaries and
        `_gradient_scale_for_step` for the optimizer-side rescale.
        """
        return loss * self._loss_scale_for_backward()

    def _gradient_scale_for_step(self) -> float | None:
        if self._accumulation_divisor_local <= 0:
            return None
        total = self._reduce_loss_normalizer_total(self._accumulation_divisor_local)
        if total <= 0:
            return None
        return float(max(self._loss_normalizer_sync_divisor(), 1)) / total

    def _optimizer_parameters_for_scaling(self) -> list[nn.Parameter]:
        if self.optimizer is None:
            return []
        if self.optimizer_container is not None:
            return self.optimizer_container.parameter_cache.get_parameters_for_clipping(self.optimizer)

        parameter_cache = self._optimizer_parameter_cache
        if parameter_cache is None:
            parameter_cache = OptimizerParameterCache(self.optimizer)
            self._optimizer_parameter_cache = parameter_cache
        else:
            parameter_cache.bind(self.optimizer)
        return parameter_cache.get_parameters_for_clipping()

    def _scale_optimizer_gradients(self, scale: float) -> None:
        if scale == 1.0 or self.optimizer is None:
            return
        parameters = self._optimizer_parameters_for_scaling()
        for parameter in parameters:
            grad = parameter.grad
            if grad is None:
                continue
            grad.mul_(float(scale))

    @contextmanager
    def train_context(self):
        """Context for one training micro-step (autocast + optional DDP no_sync)."""
        with self._train_step_context(no_sync_targets=self._train_no_sync_targets()):
            yield

    def _should_train_no_sync(self) -> bool:
        if self._train_window_will_flush:
            return False
        micro_steps = self.train_state.micro_step + 1
        return self.accum_steps > 1 and micro_steps % self.accum_steps != 0

    def infer_context(self):
        """Precision context used by train/eval/infer forward passes."""

        if self.fp8_enabled:
            return self.fp8_autocast()

        precision = self.precision
        if precision is None:
            return nullcontext()
        return torch.autocast(self.device.type, dtype=get_precision(precision))

    def _train_no_sync_targets(self) -> tuple[nn.Module, ...]:
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            return (self.model,)
        return ()

    @contextmanager
    def _train_step_context(self, *, no_sync_targets: tuple[nn.Module, ...] | list[nn.Module] = ()):
        autocast_context = self.infer_context()
        if self._should_train_no_sync() and no_sync_targets:
            with ExitStack() as stack:
                stack.enter_context(autocast_context)
                for module in no_sync_targets:
                    no_sync = getattr(module, "no_sync", None)
                    if callable(no_sync):
                        stack.enter_context(no_sync())
                        continue

                    set_requires_gradient_sync = getattr(module, "set_requires_gradient_sync", None)
                    if callable(set_requires_gradient_sync):
                        set_requires_gradient_sync(False)
                        stack.callback(set_requires_gradient_sync, True)
                        continue

                    raise TypeError(
                        "cannot disable gradient synchronization for "
                        f"{type(module).__name__}: expected `no_sync()` or `set_requires_gradient_sync(...)`"
                    )
                yield
            return

        with autocast_context:
            yield

    def train_step(self, data: Any) -> tuple[Any, torch.Tensor | None]:
        """
        Run one training micro-step.

        The default implementation runs forward → loss → metric update → backward
        → step for one micro-batch.

        **Called when:** once per micro-batch by `train_epoch`/`train_steps`. The
        caller seeds the loop's accumulation state before each invocation; this
        method consumes that state through `backward()` and `step()`.

        **Precondition:** `self.model`, `self.optimizer`, and `self.criterion`
        are bound; `self.mode == RunnerMode.train`.

        Args:
            data: One micro-batch. The default unpacks `data["input"]` /
                `data.get("target")` for mappings, `(data[0], data[1])` for
                non-string sequences, and `(data, None)` otherwise. Override
                `train_step` if your batch shape differs.

        Returns:
            `(pred, loss)`. `pred` is the model output (used by `metrics.update`).
            `loss` is the scalar loss returned to the caller for reduced logging.
            The default raises when `criterion` is missing or returns `None`;
            overrides may return `(pred, None)` to signal no loss available, in
            which case the caller skips loss bookkeeping.

        Raises:
            ValueError: `self.model` is not initialized, or `criterion` is missing
                or returned `None`.

        **Side effects:** moves `data` to `self.device`, runs forward under
        `train_context()` (autocast + optional DDP no-sync), updates
        `self.metrics` when bound, then calls `self.backward(loss)` and
        `self.step()` to scale gradients, advance accumulation state, and flush
        the optimizer on accumulation boundaries.

        !!! danger "Do not"
            - Zero gradients (`optimizer_step` does this on flush).
            - Call `self.optimizer.step()` directly (use `self.step()`).
            - Mutate `train_state.global_step` or `train_state.micro_step`.
            - Implement gradient scaling here (override `backward()` instead).
            - Call `save_checkpoint()` (cadence is owned by the loop method).

        **Backend notes:**

        - `DeepSpeedRunner` inherits the default; `backward`/`step` route
          through the DeepSpeed engine.
        - `ParallelRunner` overrides this method when a pipeline schedule is
          set; the schedule owns micro-batching and loss reduction.
        """
        data = self.to_device(data)
        with self.train_context():
            if isinstance(data, Mapping):
                inputs = data["input"]
                target = data.get("target")
            elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
                inputs = data[0]
                target = data[1] if len(data) > 1 else None
            else:
                inputs = data
                target = None

            if self.model is None:
                raise ValueError("cannot run train_step: model is not initialized")
            pred = self.model(**inputs) if isinstance(inputs, Mapping) else self.model(inputs)
            loss = self.criterion(pred, target) if self.criterion is not None else None
            if loss is None:
                raise ValueError("cannot run train_step: criterion did not produce a loss")
            if self.metrics is not None and pred is not None and target is not None:
                self.metrics.update(pred, target)
            self.backward(loss)
            self.step()
        return pred, loss

    def backward(self, loss: torch.Tensor) -> None:
        """
        Run backward pass on one micro-step loss.

        **Called when:** the default `train_step` has produced a loss tensor.
        The method receives the raw micro-step loss; accumulation scaling and
        loss-normalizer weighting are applied before `Tensor.backward()`.

        Args:
            loss: The loss tensor for this micro-step.

        **Side effects:** accumulates gradients on model parameters.

        !!! danger "Do not"
            - Advance the optimizer here; optimizer stepping belongs to
              `step()`/`optimizer_step()`.
            - Mutate `train_state` counters.

        **Backend notes:**

        - `DeepSpeedRunner` overrides this hook to call the DeepSpeed engine's
          backward method.
        """

        self._scaled_loss_for_backward(loss).backward()

    def step(self) -> None:
        """
        Advance the accumulation state machine after one training micro-step.

        **Called when:** `train_step` finishes backward for a micro-batch.

        **Side effects:** increments `train_state.micro_step` and calls
        `optimizer_step()` only when the accumulation boundary is reached or
        the surrounding loop marks the current batch as the final flush in a
        partial window.

        !!! danger "Do not"
            - Call this from evaluation/inference paths.
            - Call `optimizer_step()` in addition to this method from the same
              micro-step.
            - Adjust `train_state.micro_step` in `train_step` overrides.
        """
        micro_steps = self.train_state.micro_step + 1
        self.train_state.micro_step = micro_steps
        if self._train_window_will_flush:
            self.optimizer_step()
            remainder = micro_steps % self.accum_steps
            if self.accum_steps > 1 and remainder != 0:
                self.train_state.micro_step += self.accum_steps - remainder
            return
        if self.accum_steps <= 1 or micro_steps % self.accum_steps == 0:
            self.optimizer_step()

    def optimizer_step(self) -> bool:
        """
        Perform one backend optimizer update.

        The default Torch implementation waits for checkpoint staging, applies
        accumulated-loss gradient scaling, optional grad clipping, non-finite
        grad skip logic, optimizer/scheduler stepping through
        `OptimizerContainer`, gradient zeroing, profiler advancement, and
        garbage-collection cadence.

        **Called when:** `step()` reaches an accumulation boundary, or
        `_flush_pending_optimizer_step()` flushes a partial boundary before
        shutdown.

        Returns:
            `True` when an optimizer update is applied, otherwise `False`.

        **Side effects:** may update optimizer/scheduler state; increments
        `train_state.global_step` only when an update is actually applied.

        !!! danger "Do not"
            - Increment `global_step` on skipped updates.
            - Forget to zero gradients after a successful update or skipped
              non-finite update.
            - Bypass `checkpoint_manager.maybe_wait_for_staging()`.

        **Backend notes:**

        - `DeepSpeedRunner` overrides this hook because the DeepSpeed engine
          owns the concrete optimizer update.
        """
        if self.optimizer_container is None and self.optimizer is None:
            raise ValueError(
                "cannot perform optimizer step: no optimizer is configured; "
                "set `self.optimizer`, implement `build_optimizer()`, or override `optimizer_step()`"
            )

        self.checkpoint_manager.maybe_wait_for_staging()
        grad_scale = self._gradient_scale_for_step()
        if grad_scale is not None:
            self._scale_optimizer_gradients(grad_scale)
        max_grad_value = self.max_grad_value
        max_grad_norm = self.max_grad_norm
        skip_nonfinite_grad = self.skip_nonfinite_grad
        if self.optimizer_container is not None:
            if skip_nonfinite_grad:
                has_nonfinite_grad = self.optimizer_container.has_nan_inf_grad()
                has_nonfinite_grad = self._sync_optimizer_skip_decision(has_nonfinite_grad)
                if has_nonfinite_grad:
                    self.optimizer_container.zero_grad()
                    self._reset_accumulation_normalization()
                    return False

            stepped = self.optimizer_container.step(
                max_grad_value=max_grad_value,
                max_grad_norm=max_grad_norm,
                zero_grad=True,
                skip_nonfinite_grad=False,
            )
            if not stepped:
                self._reset_accumulation_normalization()
                return False
        elif self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self._reset_accumulation_normalization()
        self.train_state.global_step += 1
        self._step_profiler()
        self._maybe_reduce_train_process_group_timeout()
        self.supervisor.maybe_collect_garbage(self.train_state.global_step, scope="train")
        return True

    def _flush_pending_optimizer_step(self) -> bool:
        """
        Flush a partial accumulation window at loop boundaries.

        Returns:
            `True` when a boundary flush produced an optimizer update.
        """
        if self.accum_steps <= 1:
            return False
        remainder = self.train_state.micro_step % self.accum_steps
        if remainder == 0:
            return False
        if self.distributed and self._train_no_sync_targets():
            self._discard_pending_optimizer_step(remainder)
            return False
        stepped = self.optimizer_step()
        # Boundary flush clears current accumulation window; realign to the next
        # accumulation boundary so the next loop starts with a fresh full window.
        self.train_state.micro_step += self.accum_steps - remainder
        return stepped

    def _discard_pending_optimizer_step(self, remainder: int | None = None) -> None:
        if self.accum_steps <= 1:
            return
        if remainder is None:
            remainder = self.train_state.micro_step % self.accum_steps
        if remainder == 0:
            return
        if self.optimizer_container is not None:
            self.optimizer_container.zero_grad()
        elif self.optimizer is not None:
            self.optimizer.zero_grad()
        self._reset_accumulation_normalization()
        self.train_state.micro_step -= remainder

    def prepare_for_shutdown_checkpoint(self) -> None:
        self._flush_pending_optimizer_step()

    def _iter_train_batches(self, loader: Any) -> Iterator[tuple[int, Any, bool]]:
        iterator = iter(enumerate(loader))
        try:
            current = next(iterator)
        except StopIteration:
            return

        while True:
            try:
                next_item = next(iterator)
            except StopIteration:
                next_item = None

            iteration, data = current
            next_micro_step = self.train_state.micro_step + 1
            reaches_accum_boundary = self.accum_steps <= 1 or next_micro_step % self.accum_steps == 0
            will_flush = reaches_accum_boundary or next_item is None
            yield iteration, data, will_flush

            if next_item is None:
                break
            current = next_item

    def _resolve_requested_splits(
        self,
        requested_splits: list[str] | None,
        available_splits: list[str],
        *,
        kind: str,
    ) -> list[str]:
        if requested_splits is None:
            return available_splits

        splits = self._sorted_unique(requested_splits)
        unknown_splits = sorted(set(splits).difference(available_splits))
        if unknown_splits:
            raise ValueError(
                f"unknown {kind} split(s): {unknown_splits}; " f"available {kind} split(s): {available_splits}"
            )
        return splits

    def train(
        self,
        train_splits: list[str] | None = None,
        evaluate_splits: list[str] | None = None,
    ) -> RoundDict:
        """
        Run the full training workflow.

        Selects epoch mode or step mode from `self.is_step_mode`, validates
        explicit split lists against the runner's configured/inferred splits,
        and delegates to `train_epochs` or `train_steps`.

        **Called when:** user code starts training.

        Args:
            train_splits: Optional training splits. When `None`, use `self.train_splits`.
            evaluate_splits: Optional evaluation splits. When `None`, use `self.evaluate_splits`.

        Returns:
            Aggregated runner results (`self.results`).

        Raises:
            ValueError: no valid training split can be resolved.

        **Side effects:** prints selected splits and runs the selected training
        loop. Checkpointing, result writing, scheduler stepping, and early stop
        are owned by the delegated loop method.
        """

        train_splits = self._resolve_requested_splits(train_splits, self.train_splits, kind="training")
        if not train_splits:
            raise ValueError("cannot start training: no valid training split was resolved")

        evaluate_splits = self._resolve_requested_splits(evaluate_splits, self.evaluate_splits, kind="evaluation")

        print(f"train: splits={train_splits}")
        print(f"evaluate: splits={evaluate_splits}")
        if self.is_step_mode:
            return self.train_steps(train_splits=train_splits, evaluate_splits=evaluate_splits)
        return self.train_epochs(train_splits=train_splits, evaluate_splits=evaluate_splits)

    def train_epochs(
        self,
        train_splits: list[str] | None = None,
        evaluate_splits: list[str] | None = None,
    ) -> RoundDict:
        """
        Run epoch-mode training until `self.epochs` is reached.

        Each epoch runs all train splits, then all evaluation splits, advances
        epoch/metric schedulers, appends and writes results, saves periodic
        checkpoints on `ckpt.interval`, and refreshes the best checkpoint
        whenever the score improves.

        **Called when:** `train` dispatches while `config.epochs` is set, or
        user code explicitly wants epoch-mode semantics.

        Args:
            train_splits: Training splits for each epoch.
            evaluate_splits: Evaluation splits after each epoch.

        Returns:
            Aggregated runner results (`self.results`).

        Raises:
            ValueError: `config.epochs` is not set.
        """
        if train_splits is None:
            train_splits = self.train_splits
        if evaluate_splits is None:
            evaluate_splits = self.evaluate_splits

        total_epochs = self.epochs
        if total_epochs is None:
            raise ValueError("cannot run epoch-mode training: config.epochs is not set")
        print(f"train: epoch mode start epoch={self.train_state.epoch} total_epochs={total_epochs}")
        early_stop_counter = 0
        patience = self.patience
        for epoch in range(self.train_state.epoch, total_epochs):
            self.supervisor.maybe_handle_termination_signal()
            self.train_state.epoch = epoch
            result = RoundDict()
            for split in train_splits:
                result[split] = self.train_epoch(split)
                self.supervisor.maybe_handle_termination_signal()
            for split in evaluate_splits:
                result[split] = self.evaluate_epoch(split)
                self.supervisor.maybe_handle_termination_signal()
            self._step_epoch_scheduler(result)
            self.append_result(result, index=epoch)
            print(self.format_epoch_result(result, epochs=epoch, total_epochs=total_epochs))
            self.save_result()
            self.train_state.epoch = epoch + 1
            # Call every epoch: the manager persists periodic history on cadence and
            # publishes `best.pth` whenever the score improved this epoch.
            self.save_checkpoint(epochs=epoch)
            early_stop_counter = 0 if self.is_best else early_stop_counter + 1
            if early_stop_counter > patience:
                print("train: early-stop triggered")
                break
        self.save_checkpoint(last_step=True)
        self.save_model_checkpoint()
        return self.results

    def train_epoch(self, split: str = "train") -> RoundDict:
        """
        Run one full dataloader pass for a training split.

        This is the per-split epoch loop. It sets train mode, resets meters and
        train metrics, manages accumulation-window normalization, invokes
        `train_step` for each micro-batch, emits step logs, and records
        interval/epoch telemetry.

        **Called when:** `train_epochs` processes one train split.

        Args:
            split: Training split name.

        Returns:
            Epoch-level metric mapping for this split.

        **Side effects:** updates optimizer state through `train_step`,
        advances `train_state.global_step` on optimizer flushes, and writes
        step logs.

        !!! danger "Do not"
            - Call this for evaluation data; use `evaluate_epoch`.
            - Override this just to change one batch's forward/loss logic;
              override `train_step`.
            - Manually manage gradient zeroing inside `train_step`; this loop
              and `optimizer_step` own accumulation boundaries.
            - Increment `train_state.epoch`; the surrounding `train_epochs`
              loop owns epoch progress.
            - Save result or checkpoint aliases here; `train_epochs` owns
              epoch-level persistence.

        See Also:
            [`train_steps`][danling.runners.TorchRunner.train_steps]:
                Step-mode counterpart that consumes splits against a global
                step budget instead of one epoch per split.
        """
        loader = self.dataloaders[split]
        loader_length = self._loader_length(loader)
        length = loader_length - 1 if loader_length is not None else None
        last_loss: torch.Tensor | None = None
        last_loss_n: int | None = None
        self._set_loader_epoch(loader, self.train_state.epoch)
        self.mode = RunnerMode.train
        self.split = split
        self.meters.reset()
        self.metrics = self.train_metrics
        if self.metrics is not None:
            self.metrics.reset()
        telemetry = LoopTelemetry(self, start_time=self.loop_time())
        self._reset_accumulation_normalization()
        if self.optimizer_container is not None:
            self.optimizer_container.zero_grad()
        elif self.optimizer is not None:
            self.optimizer.zero_grad()

        for iteration, data, will_flush in self._iter_train_batches(loader):
            self.supervisor.maybe_handle_termination_signal()
            # Positive int = weighted-loss signal; None = no signal (uniform window).
            # 0 or missing collapses to None so the accumulation state machine
            # picks "uniform" cleanly instead of being silently coerced to 1.
            loss_n = self._get_loss_normalizer(data)
            if loss_n is not None and loss_n <= 0:
                loss_n = None
            self._pending_loss_normalizer = loss_n
            self._train_window_will_flush = will_flush
            try:
                _, loss = self.train_step(data)
            finally:
                self._train_window_will_flush = False
                self._pending_loss_normalizer = None

            self.supervisor.mark_heartbeat_progress()
            self.supervisor.maybe_handle_termination_signal()
            current_time = self.loop_time()
            if self.scheduler is not None and hasattr(self.scheduler, "get_last_lr"):
                self.meters.lr.update(self.scheduler.get_last_lr()[0])
            if loss is not None:
                # `loss_n or 1` weights a missing normalizer as a single-sample meter update;
                # criteria that emit a real loss for zero-valid-token batches are not supported here.
                self.meters.loss.update(loss.detach(), n=loss_n or 1)
            telemetry.observe(iteration=iteration, data=data, current_time=current_time)

            if self.log_interval > 0 and (
                (iteration > 0 and iteration % self.log_interval == 0) or iteration == length
            ):
                telemetry.emit_log(split=split, iteration=iteration, length=length, loss=loss, loss_n=loss_n)
            last_loss = loss
            last_loss_n = loss_n

        if (
            length is None
            and self.log_interval > 0
            and telemetry.last_iteration is not None
            and telemetry.last_iteration != telemetry.last_print_iteration
        ):
            assert telemetry.last_iteration is not None
            telemetry.emit_log(
                split=split,
                iteration=telemetry.last_iteration,
                length=length,
                loss=last_loss,
                loss_n=last_loss_n,
                reset_peak_stats=False,
            )
        result = self.get_epoch_result()
        telemetry.finalize_result(result, elapsed_seconds=self.loop_time(sync=True) - telemetry.start_time)
        return result

    def train_steps(
        self,
        train_splits: list[str] | None = None,
        evaluate_splits: list[str] | None = None,
    ) -> RoundDict:
        """
        Run step-mode training for the configured global step budget.

        Step mode consumes train splits in sorted split order until
        `train_state.global_step >= self.steps`, then optionally evaluates
        configured evaluation splits with `evaluate_steps`.

        **Called when:** `train` dispatches while `config.epochs` is unset, or
        user code explicitly wants a global-step budget.

        Args:
            train_splits: Training splits to consume in order.
            evaluate_splits: Evaluation splits to run after training steps finish.

        Returns:
            Aggregated runner results (`self.results`).

        Raises:
            ValueError: total step budget cannot be resolved.

        **Side effects:** updates epoch as an outer split-round counter,
        appends one result row indexed by `global_step`, writes result files,
        and saves the final checkpoint.

        !!! danger "Do not"
            - Assume a split is consumed exactly once; step mode can resume a
              split iterator across outer rounds.
            - Mutate `train_state.global_step` outside optimizer stepping.

        See Also:
            [`train_epoch`][danling.runners.TorchRunner.train_epoch]:
                Per-split epoch loop used by epoch-mode training.
        """
        if train_splits is None:
            train_splits = self.train_splits
        if evaluate_splits is None:
            evaluate_splits = self.evaluate_splits

        total_steps = self.steps
        if total_steps is None:
            raise ValueError("cannot run step-mode training: config.steps could not be resolved")
        print(f"train: step mode start global_step={self.train_state.global_step} steps={total_steps}")
        result = RoundDict()
        step_mode_iterators: dict[str, Iterator[tuple[int, Any, bool]] | None] = dict.fromkeys(train_splits)
        step_mode_sampler_epochs = {split: self.train_state.epoch for split in train_splits}
        while self.train_state.global_step < total_steps:
            self.supervisor.maybe_handle_termination_signal()
            round_start_step = self.train_state.global_step
            round_result = RoundDict()
            total_train_splits = len(train_splits)
            for split_index, split in enumerate(train_splits):
                self.supervisor.maybe_handle_termination_signal()
                self.mode = RunnerMode.train
                self.split = split
                remaining = total_steps - self.train_state.global_step
                if remaining <= 0:
                    break
                loader = self.dataloaders[split]
                remaining_splits = total_train_splits - split_index
                split_steps = self._step_mode_split_budget(
                    remaining_steps=remaining,
                    remaining_splits=remaining_splits,
                    loader=loader,
                )
                if split_steps <= 0:
                    break
                start_global_step = self.train_state.global_step
                target_global_step = start_global_step + split_steps
                self.meters.reset()
                self.metrics = self.train_metrics
                if self.metrics is not None:
                    self.metrics.reset()
                telemetry = LoopTelemetry(self, start_time=self.loop_time())
                self._reset_accumulation_normalization()
                if self.optimizer_container is not None:
                    self.optimizer_container.zero_grad()
                elif self.optimizer is not None:
                    self.optimizer.zero_grad()
                checkpoint_cadence = self.checkpoint_interval
                batch_iteration = -1

                while self.train_state.global_step < target_global_step:
                    batch: tuple[int, Any, bool] | None = None
                    iterator = step_mode_iterators[split]
                    recreated = False
                    while True:
                        if iterator is None:
                            if recreated:
                                break
                            self._set_loader_epoch(loader, step_mode_sampler_epochs[split])
                            iterator = self._iter_train_batches(loader)
                            step_mode_iterators[split] = iterator
                            recreated = True
                        try:
                            batch = next(iterator)
                            break
                        except StopIteration:
                            iterator = None
                            step_mode_iterators[split] = None
                            step_mode_sampler_epochs[split] += 1
                    if batch is None:
                        break
                    _, data, will_flush = batch
                    batch_iteration += 1
                    self.supervisor.maybe_handle_termination_signal()
                    step_before = self.train_state.global_step
                    # See `train_epoch` for normalizer semantics.
                    loss_n = self._get_loss_normalizer(data)
                    if loss_n is not None and loss_n <= 0:
                        loss_n = None
                    self._pending_loss_normalizer = loss_n
                    self._train_window_will_flush = will_flush
                    try:
                        _, loss = self.train_step(data)
                    finally:
                        self._train_window_will_flush = False
                        self._pending_loss_normalizer = None

                    self.supervisor.mark_heartbeat_progress()
                    self.supervisor.maybe_handle_termination_signal()
                    current_time = self.loop_time()
                    if self.scheduler is not None and hasattr(self.scheduler, "get_last_lr"):
                        self.meters.lr.update(self.scheduler.get_last_lr()[0])
                    if loss is not None:
                        self.meters.loss.update(loss.detach(), n=loss_n or 1)
                    telemetry.observe(iteration=batch_iteration, data=data, current_time=current_time)

                    step_after = self.train_state.global_step
                    if checkpoint_cadence > 0 and step_after != step_before and step_after % checkpoint_cadence == 0:
                        self.save_checkpoint()

                    global_step = step_after if step_after != step_before else None
                    is_log_step = (
                        self.log_interval > 0
                        and global_step is not None
                        and global_step > 0
                        and global_step % self.log_interval == 0
                    )
                    is_boundary_step = global_step in (target_global_step, total_steps)
                    if (
                        self.log_interval > 0
                        and global_step is not None
                        and (is_log_step or is_boundary_step)
                    ):
                        telemetry.emit_log(
                            split=split,
                            iteration=batch_iteration,
                            length=total_steps,
                            loss=loss,
                            loss_n=loss_n,
                            display_iteration=global_step,
                        )

                round_result[split] = self.get_epoch_result()
                telemetry.finalize_result(
                    round_result[split], elapsed_seconds=self.loop_time(sync=True) - telemetry.start_time
                )
                self.supervisor.maybe_handle_termination_signal()

            if self.train_state.global_step == round_start_step:
                remaining_steps = total_steps - self.train_state.global_step
                warn(
                    f"step-mode training made no progress after one full split pass "
                    f"(target={total_steps}, reached={self.train_state.global_step}, remaining={remaining_steps})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
            self._step_epoch_scheduler(round_result)
            result = round_result
            self.train_state.epoch += 1
        remaining_steps = total_steps - self.train_state.global_step
        if remaining_steps > 0:
            warn(
                f"step-mode training finished with {remaining_steps} step(s) remaining "
                f"(target={total_steps}, reached={self.train_state.global_step})",
                RuntimeWarning,
                stacklevel=2,
            )
        for split in evaluate_splits:
            result[split] = self.evaluate_steps(split=split)
        self.append_result(result, index=self.train_state.global_step)
        self.save_result()
        self.save_checkpoint(last_step=True)
        self.save_model_checkpoint()
        return self.results

    def evaluate_step(self, data: Any) -> tuple[Any, torch.Tensor | None]:
        """
        Run one evaluation micro-step.

        The default implementation runs forward → optional loss → optional
        metric update under `infer_context()`. No backward pass and no optimizer step.

        **Called when:** once per micro-batch by `evaluate_epoch`/`evaluate_steps`,
        which run under `torch.inference_mode()`.

        **Precondition:** at least one of `self.model` or `self.ema` is bound.
        `self.mode == RunnerMode.evaluate`. The default prefers `self.ema` over
        `self.model` when both are available.

        Args:
            data: One micro-batch. The default unpacks `data["input"]` /
                `data.get("target")` for mappings, `(data[0], data[1])` for
                non-string sequences, and `(data, None)` otherwise. Override
                `evaluate_step` if your batch shape differs.

        Returns:
            `(pred, loss)`. `pred` is the model output (used by `metrics.update`).
            `loss` is the scalar loss returned to the caller for reduced
            logging, or `None` when no `criterion` is set.

        Raises:
            ValueError: neither `self.model` nor `self.ema` is initialized.

        **Side effects:** moves `data` to `self.device`, runs forward through
        `self.ema or self.model` under `infer_context()`, computes loss when
        `criterion` is set, and updates `self.metrics` when bound.

        !!! danger "Do not"
            - Call `self.backward(...)` or `self.step()` (no optimizer here).
            - Mutate `train_state.global_step` or `train_state.micro_step`.
            - Switch the runner mode (the loop owns `self.mode`).
            - Call `save_checkpoint()` (cadence is owned by training loops only).

        **Backend notes:**

        - `ParallelRunner` overrides this method when a pipeline schedule is
          set; the schedule owns micro-batching and pipeline-stage loss
          reduction.
        """
        data = self.to_device(data)
        if isinstance(data, Mapping):
            inputs = data["input"]
            target = data.get("target")
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            inputs = data[0]
            target = data[1] if len(data) > 1 else None
        else:
            inputs = data
            target = None

        if self.model is None and self.ema is None:
            raise ValueError("cannot run evaluate_step: model is not initialized")
        model = self.ema or self.model
        with self.infer_context():
            pred = model(**inputs) if isinstance(inputs, Mapping) else model(inputs)
            loss = self.criterion(pred, target) if self.criterion is not None else None

        if self.metrics is not None and pred is not None and target is not None:
            self.metrics.update(pred, target)

        return pred, loss

    def evaluate(self, evaluate_splits: list[str] | None = None) -> RoundDict:
        """
        Run evaluation across splits with epoch-mode semantics.

        **Called when:** user code explicitly evaluates a runner, or training
        code delegates to evaluation helpers.

        Args:
            evaluate_splits: Optional evaluation splits. When `None`, use `self.evaluate_splits`.

        Returns:
            Mapping of split -> evaluation result for this call.

        Raises:
            ValueError: no valid evaluation split can be resolved.

        **Side effects:** sets evaluation mode per split, prints a formatted
        aggregate result, and writes scalar outputs through `evaluate_epoch`.
        """

        evaluate_splits = self._resolve_requested_splits(evaluate_splits, self.evaluate_splits, kind="evaluation")
        if not evaluate_splits:
            raise ValueError("cannot start evaluation: no valid evaluation split was resolved")
        print("evaluate: start")
        print(f"evaluate: splits={evaluate_splits}")
        result = RoundDict()
        for split in evaluate_splits:
            result[split] = self.evaluate_epoch(split=split)
        display_epoch = self.train_state.epoch
        if self.epochs is not None and display_epoch > 0:
            display_epoch -= 1
        print(self.format_epoch_result(result, epochs=display_epoch))
        return result

    @torch.inference_mode()
    def evaluate_epoch(self, split: str = "val") -> RoundDict:
        """
        Run one full dataloader pass for an evaluation split.

        Sets evaluation mode, resets meters/evaluation metrics, runs
        `evaluate_step` for every batch under inference mode, emits step logs,
        and writes the split result at the current epoch index.

        **Called when:** `evaluate` or `train_epochs` evaluates a split.

        Args:
            split: Evaluation split name.

        Returns:
            Epoch-level metric mapping for this split.

        **Side effects:** updates evaluation meters/metrics, emits logs, writes
        scalar results, and records telemetry. It does not update optimizer or
        training progress counters.
        """
        loader = self.dataloaders[split]
        loader_length = self._loader_length(loader)
        length = loader_length - 1 if loader_length is not None else None

        last_loss: torch.Tensor | None = None
        last_loss_n: int | None = None
        self.mode = RunnerMode.evaluate
        self.split = split
        self.meters.reset()
        self.metrics = self.evaluate_metrics
        if self.metrics is not None:
            self.metrics.reset()
        telemetry = LoopTelemetry(self, start_time=self.loop_time())
        consumed = 0
        for iteration, data in enumerate(loader):
            consumed = iteration + 1
            self.supervisor.maybe_handle_termination_signal()
            loss_n = self._get_loss_normalizer(data)
            if loss_n is not None and loss_n <= 0:
                loss_n = None
            _, loss = self.evaluate_step(data)
            self.supervisor.mark_heartbeat_progress()
            self.supervisor.maybe_handle_termination_signal()
            current_time = self.loop_time()
            if loss is not None:
                self.meters.loss.update(loss.detach(), n=loss_n or 1)
            telemetry.observe(
                iteration=iteration,
                data=data,
                current_time=current_time,
            )
            self.supervisor.maybe_collect_garbage(iteration + 1, scope=f"evaluate:{split}")

            if self.log_interval > 0 and (
                (iteration > 0 and iteration % self.log_interval == 0) or iteration == length
            ):
                telemetry.emit_log(split=split, iteration=iteration, length=length, loss=loss, loss_n=loss_n)
            last_loss = loss
            last_loss_n = loss_n

        if (
            length is None
            and self.log_interval > 0
            and telemetry.last_iteration is not None
            and telemetry.last_iteration != telemetry.last_print_iteration
        ):
            assert telemetry.last_iteration is not None
            telemetry.emit_log(
                split=split,
                iteration=telemetry.last_iteration,
                length=length,
                loss=last_loss,
                loss_n=last_loss_n,
                reset_peak_stats=False,
            )
        result = self.get_epoch_result()
        telemetry.finalize_result(
            result, elapsed_seconds=self.loop_time(sync=True) - telemetry.start_time, steps=consumed
        )
        self.write_result(result, split, self.train_state.epoch)
        return result

    @torch.inference_mode()
    def evaluate_steps(self, split: str = "val", steps: int | None = None) -> RoundDict:
        """
        Run bounded evaluation steps on one split.

        Used by step-mode training to evaluate a small fixed number of batches
        without requiring a full evaluation pass.

        **Called when:** `train_steps` evaluates configured splits after the
        step budget finishes, or user code requests bounded evaluation.

        Args:
            split: Evaluation split name.
            steps: Number of batches to evaluate. When `None`, defaults to `max(self.steps // 20, 1)`.

        Returns:
            Step-bounded evaluation metrics.

        Raises:
            ValueError: step budget cannot be inferred, `steps` is negative, or
                the dataloader exhausts before the requested number of steps.

        **Side effects:** writes scalar results at `train_state.global_step`.
        """
        if steps is None:
            total_steps = self.steps
            if total_steps is None:
                raise ValueError("cannot infer evaluation steps: step budget is unavailable; pass `steps`")
            steps = max(total_steps // 20, 1)
        if steps < 0:
            raise ValueError(f"invalid steps: expected a non-negative value, got {steps}")
        loader = self.dataloaders[split]
        length = steps - 1

        self.mode = RunnerMode.evaluate
        self.split = split
        if steps == 0:
            self.meters.reset()
            self.metrics = self.evaluate_metrics
            if self.metrics is not None:
                self.metrics.reset()
            result = self.get_epoch_result()
            self.write_result(result, split, self.train_state.global_step)
            return result

        self.meters.reset()
        self.metrics = self.evaluate_metrics
        if self.metrics is not None:
            self.metrics.reset()
        telemetry = LoopTelemetry(self, start_time=self.loop_time())
        consumed = 0
        for iteration, data in enumerate(loader):
            if steps is not None and iteration >= steps:
                break
            consumed = iteration + 1
            self.supervisor.maybe_handle_termination_signal()
            loss_n = self._get_loss_normalizer(data)
            if loss_n is not None and loss_n <= 0:
                loss_n = None
            _, loss = self.evaluate_step(data)
            self.supervisor.mark_heartbeat_progress()
            self.supervisor.maybe_handle_termination_signal()
            current_time = self.loop_time()
            if loss is not None:
                self.meters.loss.update(loss.detach(), n=loss_n or 1)
            telemetry.observe(iteration=iteration, data=data, current_time=current_time)
            self.supervisor.maybe_collect_garbage(iteration + 1, scope=f"evaluate:{split}")

            if self.log_interval > 0 and (
                (iteration > 0 and iteration % self.log_interval == 0) or iteration == length
            ):
                telemetry.emit_log(split=split, iteration=iteration, length=length, loss=loss, loss_n=loss_n)

        if steps is not None and consumed < steps:
            raise ValueError(
                f"evaluate steps exhausted early on split '{split}': requested {steps} step(s), got {consumed}"
            )
        result = self.get_epoch_result()
        telemetry.finalize_result(
            result, elapsed_seconds=self.loop_time(sync=True) - telemetry.start_time, steps=consumed
        )
        self.write_result(result, split, self.train_state.global_step)
        return result

    @torch.inference_mode()
    def infer_step(self, data: Any) -> list[float]:
        """
        Run one inference micro-step.

        The default implementation runs forward through `self.ema or self.model`,
        detaches scalar-per-example predictions, squeezes the trailing
        dimension, moves them to CPU, and returns them as a Python list.

        **Called when:** once per micro-batch by `infer`/`_iter_infer_batches`.
        The method is decorated with `torch.inference_mode()`.

        **Precondition:** at least one of `self.model` or `self.ema` is bound.
        `self.mode == RunnerMode.infer`.

        Args:
            data: One micro-batch. The default unpacks `data["input"]` for
                mappings, `data[0]` for non-string sequences, and `data`
                itself otherwise. Override `infer_step` if your batch shape
                differs or you need to pass auxiliary tensors to the model.

        Returns:
            List of CPU floats for scalar-per-example predictions. The
            default converts with `pred.squeeze(-1).detach().cpu().tolist()`.
            Override if your model emits multi-dim tensors, mappings, or
            non-numeric outputs.

        Raises:
            ValueError: neither `self.model` nor `self.ema` is initialized.

        **Side effects:** moves `data` to `self.device`, runs forward through
        `self.ema or self.model` under `infer_context()`, then converts the
        output to a CPU list.

        !!! danger "Do not"
            - Compute or accumulate metrics (inference is metric-free).
            - Mutate runner state counters.
            - Return a `torch.Tensor` (callers expect `list[float]` for
              batched aggregation and streaming).
            - Call `self.backward(...)` or `self.step()`.

        **Backend notes:**

        - `ParallelRunner` overrides this method when a pipeline schedule is
          set; non-first-stage ranks pass `data=None` and the schedule routes
          activations through pipeline communication.
        """
        data = self.to_device(data)
        if isinstance(data, Mapping):
            inputs = data["input"]
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            inputs = data[0]
        else:
            inputs = data

        if self.model is None and self.ema is None:
            raise ValueError("cannot run infer_step: model is not initialized")
        model = self.ema or self.model
        with self.infer_context():
            pred = model(**inputs) if isinstance(inputs, Mapping) else model(inputs)
        values = pred.squeeze(-1).detach().cpu().tolist()
        if isinstance(values, list):
            return values
        return [float(values)]

    def infer(
        self,
        split: str = "infer",
        *,
        steps: int | None = None,
        stream: bool | None = None,
    ) -> list[float] | Iterator[list[float]]:
        """
        Run inference on one split.

        In non-stream mode this consumes all requested batches and returns a
        flattened Python list. In stream mode it returns an iterator of
        per-batch outputs and leaves consumption to the caller.

        **Called when:** user code requests prediction-only execution.

        Args:
            split: Inference split name.
            steps: Optional max number of batches to consume.
            stream: `True` returns a generator of per-batch outputs, `False` returns a flattened list.
                When `None`, stream only for unsized loaders without explicit `steps`.

        Returns:
            Flattened predictions or a streaming iterator of batch predictions.

        Raises:
            ValueError: `steps` is negative, or non-stream inference is
                requested for an unsized loader without an explicit step count.

        **Side effects:** sets inference mode/split. It does not update metrics
        or optimizer state.
        """

        self.mode = RunnerMode.infer
        self.split = split
        loader = self.dataloaders[split]
        if steps is not None and steps < 0:
            raise ValueError(f"invalid steps: expected a non-negative value, got {steps}")

        loader_length = self._loader_length(loader)
        if stream is None:
            stream = steps is None and loader_length is None

        if not stream and loader_length is None and steps is None:
            raise ValueError("infer with stream=False requires `steps` for unsized loaders")

        iterator = self._iter_infer_batches(loader, steps=steps, split=split)
        if stream:
            return iterator

        total = steps if steps is not None else loader_length
        output: list[float] = []
        for values in tqdm(iterator, total=total, disable=self.distributed and not self.is_main_process):
            output.extend(values)
        return output

    def _iter_infer_batches(self, loader: Any, *, steps: int | None, split: str) -> Iterator[list[float]]:
        for iteration, data in enumerate(loader):
            if steps is not None and iteration >= steps:
                break
            values = self.infer_step(data)
            self.supervisor.mark_heartbeat_progress()
            yield values
            self.supervisor.maybe_collect_garbage(iteration + 1, scope=f"infer:{split}")

    def _export_checkpoint_metadata(self, cls: type = dict) -> Mapping[str, Any]:
        return cls()

    def _export_checkpoint_components(self, cls: type = dict) -> Mapping[str, Any]:
        if self.model is None:
            raise ValueError("cannot build checkpoint state: model is not initialized")
        state = cls()
        state["ema"] = self.ema.state_dict() if self.ema else None
        state["optimizer"] = self.optimizer.state_dict() if self.optimizer else None
        state["scheduler"] = self.scheduler.state_dict() if self.scheduler else None
        state["model"] = self.unwrap(self.model).state_dict()
        return state

    def state_dict(self, cls: type = dict) -> Mapping:
        """
        Return the TorchRunner checkpoint payload.

        Extends `BaseRunner.state_dict` with backend metadata plus EMA,
        optimizer, scheduler, and unwrapped model state.

        **Called when:** checkpoint managers persist a TorchRunner checkpoint.

        Args:
            cls: Mapping factory used for nested payloads.

        Returns:
            Mapping containing base runner state and torch component state.

        **Side effects:** snapshots Python/NumPy/Torch RNG state before
        exporting.
        """
        state = cls(super().state_dict(cls))
        state.update(self._export_checkpoint_metadata(cls))
        state.update(self._export_checkpoint_components(cls))
        return state

    def _restore_model_checkpoint(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
        if self.model is None:
            raise ValueError("cannot load model weights: model is not initialized")
        self.unwrap(self.model).load_state_dict(state_dict, *args, **kwargs)

    def load_model(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
        self._restore_model_checkpoint(state_dict, *args, **kwargs)

    def _restore_optimizer_checkpoint(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
        if self.optimizer is None:
            return
        self.optimizer.load_state_dict(dict(state_dict), *args, **kwargs)

    def load_optimizer(self, state_dict: Mapping[str, Any] | None, *args, **kwargs) -> None:
        if self.optimizer is None:
            return
        optimizer_state = self._require_checkpoint_component_state("optimizer", state_dict)
        self._restore_optimizer_checkpoint(optimizer_state, *args, **kwargs)

    def load_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        """
        Restore base runner state plus Torch RNG state.

        Model, optimizer, scheduler, and dataloader components are restored by
        `load_checkpoint`; this method owns only runner/RNG state.
        """
        super().load_state_dict(checkpoint)
        state_dict = checkpoint.get("state") or {}
        rng_state = state_dict.get("rng")
        if isinstance(rng_state, Mapping) and "torch_cpu" in rng_state and self.rng_state.torch_cpu is not None:
            torch.set_rng_state(self.rng_state.torch_cpu)
        if (
            torch.cuda.is_available()
            and isinstance(rng_state, Mapping)
            and "torch_cuda" in rng_state
            and self.rng_state.torch_cuda is not None
        ):
            torch.cuda.set_rng_state_all(self.rng_state.torch_cuda)

    def load_checkpoint(self, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> None:
        """
        Load a full checkpoint and rebind optimizer/scheduler helpers.

        This delegates component restore to `BaseRunner.load_checkpoint`, then
        rebuilds the `OptimizerContainer` so scheduler and optimizer state stay
        bound after restore.
        """
        super().load_checkpoint(checkpoint, *args, **kwargs)
        self._bind_optimizer_container()

    # `save_checkpoint` is inherited from `BaseRunner`; collective vs main-only
    # dispatch is owned by `checkpoint_manager.is_collective`.

    def load_pretrained(self, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> None:
        if not isinstance(checkpoint, Mapping) and str(self.config.get("ckpt.backend")).lower() == "dcp":
            checkpoint_path = os.fsdecode(checkpoint)
            if checkpoint_path.endswith(".pointer") or not os.path.isfile(checkpoint_path):
                ckpt = self.checkpoint_manager.load_model_checkpoint(checkpoint)
                if ckpt.get("ema") is not None:
                    self.load_model(ckpt["ema"], *args, **kwargs)
                elif "model" in ckpt:
                    self.load_model(ckpt["model"], *args, **kwargs)
                elif "model_parts" in ckpt:
                    self.load_model(ckpt["model_parts"], *args, **kwargs)
                else:
                    raise ValueError(
                        "cannot load pretrained weights: checkpoint has no EMA or model state\n"
                        "Use `load_checkpoint` for full checkpoint restore instead of `load_pretrained`"
                    )
                self.config.pretrained = os.fsdecode(checkpoint)
                return
        super().load_pretrained(checkpoint, *args, **kwargs)

    def read_checkpoint(self, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> Mapping[str, Any]:
        """Read checkpoint payload from mapping/file/DCP directory input."""
        if isinstance(checkpoint, Mapping):
            return checkpoint

        checkpoint_path = os.fsdecode(checkpoint)
        if str(self.config.get("ckpt.backend")).lower() == "dcp" and (
            checkpoint_path.endswith(".pointer") or not os.path.isfile(checkpoint_path)
        ):
            return self.checkpoint_manager.load_checkpoint(checkpoint)
        return super().read_checkpoint(checkpoint, *args, **kwargs)

    @classmethod
    def read_config(
        cls,
        checkpoint: Mapping | bytes | str | os.PathLike,
        *args,
        **kwargs,
    ) -> RunnerConfig:
        """Read runner config from checkpoint payload, including DCP directory inputs."""
        if isinstance(checkpoint, Mapping):
            return super().read_config(checkpoint, *args, **kwargs)

        if TorchDistributedCheckpointManager.is_checkpoint_path(checkpoint):
            return TorchDistributedCheckpointManager.read_config(checkpoint)

        return super().read_config(checkpoint, *args, **kwargs)

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")

    @property
    def mode(self) -> RunnerMode:
        return self._mode

    @mode.setter
    def mode(self, mode: str | RunnerMode) -> None:
        if isinstance(mode, str):
            mode = RunnerMode(mode)
        if getattr(self, "_mode", None) == mode:
            return
        self._mode = mode

        is_train = mode == RunnerMode.train
        model_parts = getattr(self, "model_parts", None)
        if isinstance(model_parts, Sequence) and model_parts:
            for model_part in model_parts:
                if not isinstance(model_part, nn.Module):
                    continue
                model_part.train(is_train)
        elif self.model is not None:
            self.model.train(is_train)
        if self.ema is not None:
            self.ema.train(is_train)

    @property
    def rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return int(os.getenv("RANK", "0"))

    @property
    def world_size(self) -> int:
        r"""
        Number of Processes.
        """
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return int(os.getenv("WORLD_SIZE", "1"))

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    def close(self, timeout: float | None = None) -> bool:
        """Close runner resources."""
        try:
            drained = super().close(timeout=timeout)
        except Exception:
            self._close_profiler()
            self.destroy_process_group()
            raise
        if not drained:
            return False
        self._close_profiler()
        self.destroy_process_group()
        return drained
