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
    np_random = None

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
from .checkpoints import FileCheckpointManager, TorchDistributedCheckpointManager
from .compile import Compiler
from .config import RunnerConfig, normalize_stack_name
from .mixins import Fp8Mixin
from .telemetry import LoopTelemetry
from .utils import RunnerMode, get_precision, on_main_process


class TorchRunner(Fp8Mixin, BaseRunner):
    r"""
    PyTorch-native runner for training, evaluation, and inference.

    The runner is designed for basic DDP-style training/evaluation/inference with
    a single local model module.
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
    _accumulation_normalizer_local: float = 0.0
    _accumulation_normalizer_mode: str | None = None
    _train_window_will_flush: bool = False
    _optimizer_parameter_cache: OptimizerParameterCache | None = None
    _supports_torchft_runtime: bool = True

    def __init__(self, config) -> None:
        if not isinstance(config, RunnerConfig):
            config = RunnerConfig(config)
        config.stack = normalize_stack_name(config.get("stack", "ddp"))
        checkpoint_backend = str(config.checkpoint.backend).strip().lower()
        if checkpoint_backend == "auto":
            checkpoint_backend = "dcp" if self.world_size > 1 else "file"
        if checkpoint_backend not in {"file", "dcp"}:
            raise ValueError(
                f"invalid checkpoint backend: {checkpoint_backend!r}. Expected one of: 'auto', 'file', 'dcp'."
            )
        config.checkpoint.backend = checkpoint_backend
        super().__init__(config)

    def __post_init__(self):
        self._accumulation_normalizer_local = 0.0
        self._accumulation_normalizer_mode = None
        self._train_window_will_flush = False
        self._optimizer_parameter_cache = None
        if self.model is None:
            raise ValueError("cannot initialize TorchRunner: model is not initialized")
        if self.datasets:
            self.build_dataloaders()
        if self.ft is not None and self.ft.enabled and not self._supports_torchft_runtime:
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
        Set up distributed training.

        Initialise process group and set up DDP variables.
        """

        backend = self.config.get("backend", os.getenv("BACKEND"))
        init_method = self.config.get("init_method", os.getenv("INIT_METHOD"))
        init_timeout = self._comm_timeout("comm.init_timeout_seconds")
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

        checkpoint_backend = self.config.checkpoint.backend.lower()

        if checkpoint_backend == "dcp":
            self.set_checkpoint_manager(TorchDistributedCheckpointManager(self))
        elif checkpoint_backend == "file":
            self.set_checkpoint_manager(FileCheckpointManager(self))
        else:
            raise ValueError(
                f"invalid checkpoint backend: {checkpoint_backend!r}. Expected one of: 'auto', 'file', 'dcp'."
            )

        self._train_pg_timeout_reduced = False

    def _comm_timeout(self, key: str) -> timedelta | None:
        value = self.config.get(key)
        if value is None:
            return None
        seconds = int(value)
        if seconds <= 0:
            raise ValueError(f"{key} must be a positive integer, got {seconds}")
        return timedelta(seconds=seconds)

    def _timeout_process_groups(self) -> tuple[Any | None, ...]:
        groups: list[Any | None] = [None]
        if self.ft is not None and self.ft.replicate_process_group is not None:
            groups.append(self.ft.replicate_process_group)
        return tuple(groups)

    def _set_process_group_timeout(self, timeout: timedelta) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        set_pg_timeout = getattr(dist_c10d, "_set_pg_timeout", None)
        if not callable(set_pg_timeout):
            warn(
                "torch.distributed does not expose process-group timeout mutation; "
                "skipping comm.train_timeout_seconds update",
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
                        "skipping comm.train_timeout_seconds update for a non-default process group",
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
        timeout = self._comm_timeout("comm.train_timeout_seconds")
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

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available() and self.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        schedule_kwargs: dict[str, Any] = {"wait": wait, "warmup": warmup, "active": active}
        if repeat is not None:
            schedule_kwargs["repeat"] = repeat

        trace_dir = os.fsdecode(str(profiling.get("trace_dir", "profiles")))
        if not os.path.isabs(trace_dir):
            trace_dir = os.path.join(self.workspace.dir, trace_dir)
        trace_dir = os.path.join(trace_dir, self.timestamp, f"rank-{self.rank:05d}")
        os.makedirs(trace_dir, exist_ok=True)
        profiler_context = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(**schedule_kwargs),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
            record_shapes=bool(profiling.get("record_shapes", False)),
            profile_memory=bool(profiling.get("profile_memory", False)),
            with_stack=bool(profiling.get("with_stack", False)),
            with_flops=bool(profiling.get("with_flops", False)),
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

        if "log_dir" not in kwargs:
            kwargs["log_dir"] = os.path.join(self.workspace.dir, "tensorboard", self.timestamp)

        self.writer = SummaryWriter(*args, **kwargs)
        self.writer.add_scalar = catch(OSError, verbose=False)(self.writer.add_scalar)

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
            if self.ft is not None:
                _, bias = self.ft.data_parallel_info(self.world_size, self.rank)
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
        self._refresh_torch_rng_state()
        return process_seed

    def _refresh_torch_rng_state(self) -> None:
        self.rng_state.torch_cpu = torch.get_rng_state()
        if torch.cuda.is_available():
            self.rng_state.torch_cuda = torch.cuda.get_rng_state_all()
        else:
            self.rng_state.torch_cuda = None

    def set_deterministic(self) -> None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    def materialize_model(self) -> None:
        """Move model to runtime device, optionally compile, and wrap with DDP when distributed."""
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
            model = nn.parallel.DistributedDataParallel(model)
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

    def iter_optimizer_parameters(self) -> Iterator[nn.Parameter]:
        if self.model is None:
            return
        yield from self._iter_unique_parameters((self.unwrap(self.model),))

    def build_optimizer(self) -> None:
        """Auto-build optimizer from config when `self.optimizer` is absent."""
        if self.optimizer is not None or self.model is None:
            return
        optim_cfg = self.config.get("optim")
        if optim_cfg is None:
            optim_cfg = self.config.get("optimizer")
        if not isinstance(optim_cfg, Mapping) or not optim_cfg:
            return
        parameters = list(self.iter_optimizer_parameters())
        if not parameters:
            return
        self.optimizer = OPTIMIZERS.build(params=parameters, **dict(optim_cfg))

    def _get_scheduler_config(self) -> Mapping[str, Any] | None:
        sched_cfg = self.config.get("sched")
        if sched_cfg is None:
            sched_cfg = self.config.get("scheduler")
        if not isinstance(sched_cfg, Mapping):
            return None
        return sched_cfg

    def build_scheduler(self) -> None:
        """Auto-build scheduler from config when `self.scheduler` is absent."""
        if self.scheduler is not None or self.optimizer is None:
            return
        sched_cfg = self._get_scheduler_config()
        if not isinstance(sched_cfg, Mapping) or not sched_cfg:
            return
        scheduler_kwargs = dict(sched_cfg)
        scheduler_kwargs.pop("interval", None)
        scheduler_kwargs.pop("monitor", None)
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

        monitor = self.scheduler_monitor or self.config.score_name

        if "." in monitor:
            value: Any = result
            for key in monitor.split("."):
                if not isinstance(value, Mapping) or key not in value:
                    raise ValueError(
                        f"could not resolve scheduler.monitor={monitor!r} from aggregated result {dict(result)!r}"
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
                f"ambiguous scheduler.monitor={monitor!r}: matched multiple splits ({splits}). "
                "Use '<split>.<metric>' to disambiguate."
            )

        raise ValueError(f"could not resolve scheduler.monitor={monitor!r} from aggregated result {dict(result)!r}")

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
        """Build dataloaders for dataset splits not already materialized."""
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
            kwargs.setdefault("drop_last", is_train_split)
            sampler = self.build_datasampler(dataset, split=k, shuffle=shuffle)
            self.dataloaders[k] = StatefulDataLoader(dataset, sampler=sampler, collate_fn=self.collate_fn, **kwargs)

    def build_datasampler(self, dataset, *, split: str, shuffle: bool):
        """Build split sampler (distributed or local)."""
        if self.distributed:
            num_replicas = self.world_size
            rank = self.rank
            if self.ft is not None:
                num_replicas, rank = self.ft.data_parallel_info(num_replicas, rank)
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
    def _loss_normalizer_value(value: Any) -> int:
        if isinstance(value, bool):
            raise TypeError("loss_normalizer must be an integer, got bool")
        if isinstance(value, int):
            normalizer = int(value)
        elif torch.is_tensor(value) and value.numel() == 1 and not torch.is_floating_point(value):
            normalizer = int(value.detach().item())
        else:
            raise TypeError(
                f"loss_normalizer must be an integer or single-element integer tensor, got {type(value).__name__}"
            )
        if normalizer <= 0:
            raise ValueError(f"loss_normalizer must be positive, got {normalizer}")
        return normalizer

    @staticmethod
    def _find_nested_tensor(data: Any):
        return LoopTelemetry.find_nested_tensor(data)

    def loss_normalizer(self, data: Any) -> int | None:
        """Infer the loss denominator from a target NestedTensor, when present."""
        target = None
        if isinstance(data, Mapping):
            target = data.get("target")
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)) and len(data) > 1:
            target = data[1]

        nested = self._find_nested_tensor(target)
        if nested is None:
            return None
        return int(nested.numel())

    def _effective_loss_normalizer(self, loss_normalizer: int | None) -> int:
        return self._loss_normalizer_value(loss_normalizer) if loss_normalizer is not None else 1

    def _loss_normalizer_sync_divisor(self) -> int:
        if self.ft is not None and self.ft.replicate_process_group is not None:
            return max(int(dist.get_world_size(group=self.ft.replicate_process_group)), 1)
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
            if "nccl" in backend and torch.cuda.is_available():
                return self.device
        return torch.device("cpu")

    def all_reduce_group(self):
        if self.ft is not None and self.ft.replicate_process_group is not None:
            return self.ft.replicate_process_group
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
        self._accumulation_normalizer_local = 0.0
        self._accumulation_normalizer_mode = None

    def backward(self, loss: torch.Tensor, *, loss_normalizer: int | None = None) -> None:
        """
        Run backward pass on one micro-step loss.

        Args:
            loss: The loss tensor for this micro-step.
            loss_normalizer: Explicit denominator for weighted gradient accumulation.
        """

        mode = "weighted" if loss_normalizer is not None else "uniform"
        if self._accumulation_normalizer_mode is None:
            self._accumulation_normalizer_mode = mode
        elif self._accumulation_normalizer_mode != mode:
            raise ValueError(
                "cannot mix weighted and uniform loss normalization within the same accumulation window. "
                "Pass `loss_normalizer` to every `backward()` call in the window, or to none of them."
            )

        if mode == "weighted":
            if loss_normalizer is None:
                raise ValueError(
                    "loss normalizer became unavailable within the current accumulation window. "
                    "Pass `loss_normalizer` to `backward()` or provide consistent batch metadata."
                )
            normalizer = float(self._loss_normalizer_value(loss_normalizer))
            self._accumulation_normalizer_local += normalizer
            self._backward_loss(loss * normalizer)
            return

        self._accumulation_normalizer_local += 1.0
        self._backward_loss(loss)

    def _backward_loss(self, loss: torch.Tensor) -> None:
        loss.backward()

    def _gradient_scale_for_step(self) -> float | None:
        if self._accumulation_normalizer_local <= 0:
            return None
        total = self._reduce_loss_normalizer_total(self._accumulation_normalizer_local)
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

    def forward_context(self):
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
        autocast_context = self.forward_context()
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

    def train_step(self, data) -> tuple[Any, torch.Tensor | None]:
        """Execute one micro-step: forward, loss, metric update, backward, and step logic."""
        loss_normalizer = self.loss_normalizer(data)
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
            self.backward(loss, loss_normalizer=loss_normalizer)
            self.step()
        return pred, loss

    def step(self) -> None:
        """Advance micro-step state and trigger optimizer update on accumulation boundary."""
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

        Backend runners override this hook when the engine owns optimizer stepping.
        The default Torch implementation applies optional grad clipping and
        non-finite grad skipping before advancing runner state.

        Returns:
            `True` when an optimizer update is applied, otherwise `False`.
        """
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

    def train(
        self,
        train_splits: list[str] | None = None,
        evaluate_splits: list[str] | None = None,
    ) -> RoundDict:
        """
        Run the full training workflow.

        Args:
            train_splits: Optional training splits. When `None`, use `self.train_splits`.
            evaluate_splits: Optional evaluation splits. When `None`, use `self.evaluate_splits`.

        Returns:
            Aggregated runner results (`self.results`).

        Notes:
            Dispatches to `train_steps` when `self.is_step_mode` is true; otherwise to `train_epochs`.
        """

        if train_splits is not None:
            train_splits = sorted(set(train_splits).intersection(self.train_splits))
        else:
            train_splits = self.train_splits
        if not train_splits:
            raise ValueError("cannot start training: no valid training split was resolved")

        if evaluate_splits is not None:
            evaluate_splits = sorted(set(evaluate_splits).intersection(self.evaluate_splits))
        else:
            evaluate_splits = self.evaluate_splits

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

        Args:
            train_splits: Training splits for each epoch.
            evaluate_splits: Evaluation splits after each epoch.

        Returns:
            Aggregated runner results (`self.results`).
        """
        if train_splits is None:
            train_splits = self.train_splits
        if evaluate_splits is None:
            evaluate_splits = self.evaluate_splits

        total_epochs = self.epochs
        if total_epochs is None:
            raise ValueError("cannot run epoch-mode training: config.epochs is not set")
        print(f"train: epoch mode start epoch={self.train_state.epoch} total_epochs={total_epochs}")
        checkpoint_cadence = self.checkpoint_interval
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
            if checkpoint_cadence > 0 and self.train_state.epoch % checkpoint_cadence == 0:
                self.save_checkpoint(epochs=epoch)
            early_stop_counter = 0 if self.is_best else early_stop_counter + 1
            if early_stop_counter > patience:
                print("train: early-stop triggered")
                break
        self.save_checkpoint(last_step=True)
        return self.results

    def train_epoch(self, split: str = "train") -> RoundDict:
        """
        Run one full dataloader pass for a training split.

        Args:
            split: Training split name.

        Returns:
            Epoch-level metric mapping for this split.
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
            loss_normalizer = self.loss_normalizer(data)
            loss_n = self._effective_loss_normalizer(loss_normalizer)
            self._train_window_will_flush = will_flush
            try:
                _, loss = self.train_step(data)
            finally:
                self._train_window_will_flush = False

            self.supervisor.mark_heartbeat_progress()
            self.supervisor.maybe_handle_termination_signal()
            current_time = self.loop_time()
            if self.scheduler is not None and hasattr(self.scheduler, "get_last_lr"):
                self.meters.lr.update(self.scheduler.get_last_lr()[0])
            if loss is not None:
                self.meters.loss.update(loss.detach(), n=loss_n)
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

        Args:
            train_splits: Training splits to consume in order.
            evaluate_splits: Evaluation splits to run after training steps finish.

        Returns:
            Aggregated runner results (`self.results`).
        """
        if train_splits is None:
            train_splits = self.train_splits
        if evaluate_splits is None:
            evaluate_splits = self.evaluate_splits

        total_steps = self.steps
        if total_steps is None:
            raise ValueError("cannot run step-mode training: config.steps could not be resolved")
        print("train: step mode start " f"global_step={self.train_state.global_step} steps={total_steps}")
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
                length = max(target_global_step - self.train_state.global_step - 1, 0)
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
                    loss_normalizer = self.loss_normalizer(data)
                    loss_n = self._effective_loss_normalizer(loss_normalizer)
                    self._train_window_will_flush = will_flush
                    try:
                        _, loss = self.train_step(data)
                    finally:
                        self._train_window_will_flush = False

                    self.supervisor.mark_heartbeat_progress()
                    self.supervisor.maybe_handle_termination_signal()
                    current_time = self.loop_time()
                    if self.scheduler is not None and hasattr(self.scheduler, "get_last_lr"):
                        self.meters.lr.update(self.scheduler.get_last_lr()[0])
                    if loss is not None:
                        self.meters.loss.update(loss.detach(), n=loss_n)
                    telemetry.observe(iteration=batch_iteration, data=data, current_time=current_time)

                    step_after = self.train_state.global_step
                    if checkpoint_cadence > 0 and step_after != step_before and step_after % checkpoint_cadence == 0:
                        self.save_checkpoint()

                    step_iteration = step_after - start_global_step - 1 if step_after != step_before else None
                    if (
                        self.log_interval > 0
                        and step_iteration is not None
                        and (
                            (step_iteration > 0 and step_iteration % self.log_interval == 0) or step_iteration == length
                        )
                    ):
                        telemetry.emit_log(
                            split=split,
                            iteration=batch_iteration,
                            length=length,
                            loss=loss,
                            loss_n=loss_n,
                            display_iteration=step_iteration,
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
        print(f"train: step mode result={result}")
        self.save_result()
        self.save_checkpoint(last_step=True)
        return self.results

    def evaluate_step(self, data) -> tuple[Any, torch.Tensor | None]:
        """Execute one evaluation step (forward + optional loss + metric update)."""
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
        with self.forward_context():
            pred = model(**inputs) if isinstance(inputs, Mapping) else model(inputs)
            loss = self.criterion(pred, target) if self.criterion is not None else None

        if self.metrics is not None and pred is not None and target is not None:
            self.metrics.update(pred, target)

        return pred, loss

    def evaluate(self, evaluate_splits: list[str] | None = None) -> RoundDict:
        """
        Run evaluation across splits with epoch-mode semantics.

        Args:
            evaluate_splits: Optional evaluation splits. When `None`, use `self.evaluate_splits`.

        Returns:
            Mapping of split -> evaluation result for this call.
        """

        evaluate_splits = sorted(evaluate_splits) if evaluate_splits is not None else self.evaluate_splits
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

        Args:
            split: Evaluation split name.

        Returns:
            Epoch-level metric mapping for this split.
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
            loss_n = self._effective_loss_normalizer(self.loss_normalizer(data))
            _, loss = self.evaluate_step(data)
            self.supervisor.mark_heartbeat_progress()
            self.supervisor.maybe_handle_termination_signal()
            current_time = self.loop_time()
            if loss is not None:
                self.meters.loss.update(loss.detach(), n=loss_n)
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

        Args:
            split: Evaluation split name.
            steps: Number of batches to evaluate. When `None`, defaults to `max(self.steps // 20, 1)`.

        Returns:
            Step-bounded evaluation metrics.
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
            loss_n = self._effective_loss_normalizer(self.loss_normalizer(data))
            _, loss = self.evaluate_step(data)
            self.supervisor.mark_heartbeat_progress()
            self.supervisor.maybe_handle_termination_signal()
            current_time = self.loop_time()
            if loss is not None:
                self.meters.loss.update(loss.detach(), n=loss_n)
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
        """Execute one inference step and return CPU scalar/list predictions."""
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
        with self.forward_context():
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

        Args:
            split: Inference split name.
            steps: Optional max number of batches to consume.
            stream: `True` returns a generator of per-batch outputs, `False` returns a flattened list.
                When `None`, stream only for unsized loaders without explicit `steps`.

        Returns:
            Flattened predictions or a streaming iterator of batch predictions.
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
        """Return TorchRunner checkpoint payload (runner + model + optimizer + scheduler + ema)."""
        self._refresh_torch_rng_state()
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
        self.optimizer.load_state_dict(state_dict, *args, **kwargs)

    def load_optimizer(self, state_dict: Mapping[str, Any] | None, *args, **kwargs) -> None:
        if self.optimizer is None:
            return
        state_dict = self._require_checkpoint_component_state("optimizer", state_dict)
        self._restore_optimizer_checkpoint(state_dict, *args, **kwargs)

    def load_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore runner state and RNG state from checkpoint payload."""
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
        """Load a full checkpoint and rebind optimizer container afterwards."""
        super().load_checkpoint(checkpoint, *args, **kwargs)
        self._bind_optimizer_container()

    def save_checkpoint(
        self,
        name: str = "latest",
        epochs: int | None = None,
        save_best: bool = True,
        last_step: bool = False,
        force: bool = False,
    ) -> None:
        """Save checkpoint through the active backend manager."""
        if self.config.checkpoint.backend.lower() != "dcp":
            return super().save_checkpoint(
                name=name,
                epochs=epochs,
                save_best=save_best,
                last_step=last_step,
                force=force,
            )

        epochs = self.train_state.epoch if epochs is None else epochs
        self.checkpoint_manager.save_checkpoint(
            name=name,
            epochs=epochs,
            save_best=save_best,
            last_step=last_step,
            force=force,
        )

    def read_checkpoint(self, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> Mapping[str, Any]:
        """Read checkpoint payload from mapping/file/DCP directory input."""
        if isinstance(checkpoint, Mapping):
            return checkpoint

        if self.config.checkpoint.backend.lower() == "dcp":
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
        self._close_profiler()
        drained = super().close(timeout=timeout)
        self.destroy_process_group()
        return drained
