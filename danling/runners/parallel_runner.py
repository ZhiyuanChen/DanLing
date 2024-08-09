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
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Iterator, Tuple
from warnings import warn

import torch
from lazy_imports import try_import
from torch import distributed as dist
from torch import nn, utils
from tqdm import tqdm

from danling.data import DataLoaderDict, StepProxyLoader

from .checkpoints import TorchDistributedCheckpointManager
from .config import RunnerConfig
from .fsdp import build_fsdp2_kwargs, build_mixed_precision_policy, build_offload_policy
from .topology import ParallelContext, ParallelTopology
from .torch_runner import TorchRunner
from .utils import RunnerMode

with try_import() as dcp:
    from torch.distributed.checkpoint.state_dict import StateDictOptions
    from torch.distributed.device_mesh import init_device_mesh

with try_import() as pipeline:
    from torch.distributed.pipelining import PipelineStage
    from torch.distributed.pipelining.schedules import PipelineScheduleMulti, get_schedule_class

with try_import() as parallel_fsdp:
    from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard


class _ParallelDataLoaderDict(DataLoaderDict):
    def __init__(self, runner: ParallelRunner) -> None:
        super().__init__()
        object.__setattr__(self, "_runner", runner)

    def __getitem__(self, key):
        loader = super().__getitem__(key)
        if self._runner._use_step_only_loader() and not isinstance(loader, StepProxyLoader):
            return StepProxyLoader(loader)
        return loader


class ParallelRunner(TorchRunner):
    """Torch runner for data/pipeline/tensor parallel stacks.

    Checkpoint invariants:
    - Distributed parallel runs use `checkpoint.backend="dcp"` only.
    - Model/optimizer state uses torch.distributed.checkpoint state-dict APIs when available.
    - Restore order is model first, then optimizer, then scheduler.
    """

    topology: ParallelTopology
    parallel: ParallelContext
    pipeline_schedule: Any | None = None
    pipeline_has_first_stage: bool = True
    pipeline_has_last_stage: bool = True

    tensor_group = None
    pipeline_group = None
    replicate_group = None
    shard_group = None
    context_group = None
    expert_group = None
    expert_tensor_group = None
    device_mesh = None
    _parallel_groups_initialized: bool = False
    _supports_torchft_runtime: bool = True
    _ft_reduced_domains = frozenset({"data", "batch", "loss", "optimizer", "fsdp"})
    _pipeline_loss_divisor_local: float = 0.0
    _pipeline_loss_weighting: str | None = None

    model_parts: list[nn.Module]

    checkpoint_manager: TorchDistributedCheckpointManager

    def __init__(self, config: Mapping[str, Any]) -> None:
        dcp.check()
        if not isinstance(config, RunnerConfig):
            config = RunnerConfig(config)
        requested_backend = str(config.checkpoint.backend).strip().lower()
        config.stack = "parallel"
        if requested_backend != "dcp":
            if requested_backend != "auto":
                warn(
                    f"{self.__class__.__name__} overrides checkpoint.backend to 'dcp'",
                    RuntimeWarning,
                    stacklevel=2,
                )
            config.checkpoint.backend = "dcp"
        super().__init__(config)
        self.dataloaders = _ParallelDataLoaderDict(self)

    @property
    def fsdp_enabled(self) -> bool:
        return bool(self.config.fsdp.get("enabled", False))

    def init_distributed(self) -> None:
        super().init_distributed()
        if self.world_size <= 1:
            raise RuntimeError("ParallelRunner requires distributed mode (WORLD_SIZE > 1)")
        self.topology = self.build_topology()
        if not self._parallel_groups_initialized:
            self._reset_model_parallel_groups()
            self._init_model_parallel_groups()
            self._parallel_groups_initialized = True

    def build_topology(self) -> ParallelTopology:
        axes = {
            "replicate": int(self.config.parallel.axes.replicate),
            "shard": int(self.config.parallel.axes.shard),
            "context": int(self.config.parallel.axes.context),
            "pipeline": int(self.config.parallel.axes.pipeline),
            "tensor": int(self.config.parallel.axes.tensor),
            "expert": int(self.config.parallel.axes.expert),
            "expert_tensor": int(self.config.parallel.axes.expert_tensor),
        }
        invalid_axes = {axis: degree for axis, degree in axes.items() if degree < 1}
        if invalid_axes:
            invalid = ", ".join(f"{axis}={degree}" for axis, degree in invalid_axes.items())
            raise ValueError("invalid parallel topology: axis degrees must be positive integers, " f"got {invalid}")

        topology_size = 1
        for degree in axes.values():
            topology_size *= degree
        if self.world_size != topology_size:
            axis_product = " * ".join(f"{axis}({degree})" for axis, degree in axes.items())
            raise ValueError(
                "invalid parallel topology: "
                f"WORLD_SIZE({self.world_size}) must equal {axis_product} = {topology_size}"
            )

        return ParallelTopology(
            world_size=self.world_size,
            rank=self.rank,
            axes=axes,
            domains={
                "data": ("replicate", "shard"),
                "batch": ("replicate", "shard"),
                "loss": ("replicate", "shard", "context"),
                "optimizer": tuple(axes),
                "fsdp": ("replicate", "shard", "context"),
                "context": ("context",),
                "pipeline": ("pipeline",),
                "tensor": ("tensor",),
                "expert": ("expert",),
                "expert_tensor": ("expert_tensor",),
            },
            label="parallel topology",
        )

    def _reset_model_parallel_groups(self) -> None:
        self.tensor_group = None
        self.pipeline_group = None
        self.replicate_group = None
        self.shard_group = None
        self.context_group = None
        self.expert_group = None
        self.expert_tensor_group = None
        self.device_mesh = None
        if hasattr(self, "topology"):
            self.parallel = ParallelContext(self.topology)

    def _init_model_parallel_groups(self) -> None:
        use_device_mesh = self.config.parallel.use_device_mesh
        if not use_device_mesh:
            raise RuntimeError("cannot initialize parallel process groups: set `parallel.use_device_mesh=True`.")

        mesh_device_type = self.config.parallel.mesh_device_type
        if mesh_device_type is None:
            mesh_device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_mesh = init_device_mesh(
            mesh_device_type,
            mesh_shape=self.topology.mesh_shape,
            mesh_dim_names=self.topology.axis_names,
        )
        self.parallel = ParallelContext(
            self.topology,
            device_mesh=self.device_mesh,
            groups={axis: self.device_mesh.get_group(axis) for axis in self.topology.axis_names},
        )
        self.shard_group = self.parallel.group("shard")
        self.replicate_group = self.parallel.group("replicate")
        self.context_group = self.parallel.group("context")
        self.pipeline_group = self.parallel.group("pipeline")
        self.tensor_group = self.parallel.group("tensor")
        self.expert_group = self.parallel.group("expert")
        self.expert_tensor_group = self.parallel.group("expert_tensor")

    def _timeout_process_groups(self) -> tuple[Any | None, ...]:
        groups = list(super()._timeout_process_groups())
        if hasattr(self, "parallel"):
            groups.extend(group for group in self.parallel.groups.values() if group is not None)
        return tuple(groups)

    def __post_init__(self):
        self._pipeline_loss_divisor_local = 0.0
        self._pipeline_loss_weighting = None
        if self.fsdp_enabled:
            parallel_fsdp.check()
        torchft_config_supported = (
            self.fsdp_enabled
            and int(self.config.parallel.axes.pipeline) == 1
            and int(self.config.parallel.axes.tensor) == 1
            and int(self.config.parallel.axes.context) == 1
            and int(self.config.parallel.axes.expert) == 1
            and int(self.config.parallel.axes.expert_tensor) == 1
        )
        if self.ft is not None and self.ft.enabled and not torchft_config_supported:
            raise NotImplementedError(
                "ParallelRunner TorchFT integration currently requires FSDP with "
                "pipeline/tensor/context/expert axes set to 1"
            )
        if not self.model_parts:
            if self.model is None:
                raise ValueError("cannot initialize model_parts: model is not initialized")
            self.model_parts = [self.model]
        super().__post_init__()

    def materialize_model(self) -> None:
        if self.fsdp_enabled:
            if fully_shard is None or FSDPModule is None:
                raise RuntimeError(
                    "cannot initialize ParallelRunner FSDP: torch.distributed.fsdp.fully_shard is required"
                )
            if not torch.cuda.is_available():
                raise RuntimeError("ParallelRunner FSDP requires CUDA when WORLD_SIZE > 1")

            if self.pipeline_schedule is None and self.pipeline_degree > 1 and self.pipeline_group is not None:
                if self.model_parts and len(self.model_parts) != 1:
                    raise ValueError(
                        "cannot auto-materialize FSDP pipeline from multiple local model_parts; "
                        "provide `pipeline_schedule` explicitly when pre-partitioning local stages"
                    )
                stage_model = self.model_parts[0] if self.model_parts else self.model
                if stage_model is None:
                    raise ValueError("cannot materialize FSDP pipeline: model is not initialized")
                self.pipeline_schedule = self.build_pipeline_schedule(stage_model)
                self.model_parts = [stage_model]
                self.model = stage_model
                self.pipeline_has_first_stage = self.pipeline_rank == 0
                self.pipeline_has_last_stage = self.pipeline_rank == self.pipeline_degree - 1

            if self.pipeline_schedule is None:
                if self.model is None:
                    if self.model_parts:
                        self.model = self.model_parts[0]
                    else:
                        raise ValueError("cannot materialize FSDP model: model is not initialized")
                parts = [self.model]
            else:
                if not self.model_parts:
                    if self.model is None:
                        raise ValueError("cannot materialize FSDP pipeline: model_parts are not initialized")
                    self.model_parts = [self.model]
                parts = list(self.model_parts)

            parts = [part.to(self.device) for part in parts]
            parts = [self.parallelize_model(part) for part in parts]
            self.model_parts = parts
            self.model = parts[0]
            if self.fp8_enabled:
                self.apply_fp8_module_policy_to_model_parts()
                parts = list(self.model_parts)

            fsdp_kwargs = self.fsdp_kwargs()
            wrapped_parts: list[nn.Module] = []
            for part in parts:
                module = self.apply_activation_checkpointing(part)
                module = self.compiler.compile(module)
                wrapped_parts.append(fully_shard(module, **fsdp_kwargs))

            self.model_parts = wrapped_parts
            self.model = wrapped_parts[0]
            self.bind_pipeline_modules(self.model_parts)
            if self.ft is not None and self.ft.replicate_process_group is not None:

                def all_reduce_hook(output):
                    dist.all_reduce(output, group=self.ft.replicate_process_group, op=dist.ReduceOp.AVG)

                def apply_hook(module: nn.Module) -> None:
                    set_all_reduce_hook = getattr(module, "set_all_reduce_hook", None)
                    if callable(set_all_reduce_hook):
                        set_all_reduce_hook(all_reduce_hook)

                for model in self.model_parts:
                    model.apply(apply_hook)

            if self.ema is not None:
                self.ema = self.ema.to(self.device)
            return

        if self.pipeline_schedule is None and self.pipeline_degree > 1 and self.pipeline_group is not None:
            if self.model_parts and len(self.model_parts) != 1:
                raise ValueError(
                    "cannot auto-materialize pipeline from multiple local model_parts; "
                    "provide `pipeline_schedule` explicitly when pre-partitioning local stages"
                )
            stage_model = self.model_parts[0] if self.model_parts else self.model
            if stage_model is None:
                raise ValueError("cannot materialize pipeline: model is not initialized")
            self.pipeline_schedule = self.build_pipeline_schedule(stage_model)
            self.model_parts = [stage_model]
            self.model = stage_model
            self.pipeline_has_first_stage = self.pipeline_rank == 0
            self.pipeline_has_last_stage = self.pipeline_rank == self.pipeline_degree - 1

        if self.pipeline_schedule is None:
            if self.model is None:
                if self.model_parts:
                    self.model = self.model_parts[0]
                else:
                    raise ValueError("cannot materialize parallel model: model is not initialized")
            model = self.model.to(self.device)
            model = self.parallelize_model(model)
            self.model = model
            self.model_parts = [model]
            if self.fp8_enabled:
                self.apply_fp8_module_policy_to_model_parts()
                model = self.model
            model = self.compiler.compile(model)
            self.model = model
            self.model_parts = [model]
        else:
            if not self.model_parts:
                if self.model is None:
                    raise ValueError("cannot materialize pipeline: model_parts are not initialized")
                self.model_parts = [self.model]
            self.model_parts = [model.to(self.device) for model in self.model_parts]
            self.model_parts = [self.parallelize_model(model) for model in self.model_parts]
            self.model = self.model_parts[0]
            if self.fp8_enabled:
                self.apply_fp8_module_policy_to_model_parts()
            self.model_parts = [self.compiler.compile(model) for model in self.model_parts]
            self.model = self.model_parts[0]
            self.bind_pipeline_modules(self.model_parts)

        if self.ema is not None:
            self.ema = self.ema.to(self.device)

    def parallelize_model(self, model: nn.Module) -> nn.Module:
        """Hook for model-specific TP/CP/EP transforms before compile/FSDP wrapping."""
        parallelize = getattr(model, "parallelize", None)
        if callable(parallelize):
            parallelized = parallelize(self.parallel)
            if parallelized is None:
                return model
            if not isinstance(parallelized, nn.Module):
                raise TypeError(
                    "model.parallelize(parallel) must return an nn.Module or None, "
                    f"got {type(parallelized).__name__}"
                )
            return parallelized

        if self.model_parallel_degree > 1:
            axes = ", ".join(self.model_parallel_axes)
            raise NotImplementedError(
                f"parallel axes {axes} require model-specific parallelization. "
                "Implement `model.parallelize(parallel)` or override "
                "`ParallelRunner.parallelize_model`."
            )
        return model

    def fsdp_mesh(self):
        mesh = self.config.fsdp.get("mesh")
        if mesh is not None:
            return mesh
        if self.device_mesh is None:
            raise RuntimeError("cannot initialize ParallelRunner FSDP: device mesh is not initialized")

        if self.context_degree > 1:
            raise NotImplementedError(
                "ParallelRunner FSDP with context parallelism requires a flattened FSDP mesh; "
                "set fsdp.mesh explicitly or keep parallel.axes.context=1."
            )
        if self.replicate_degree > 1:
            return self.device_mesh["replicate", "shard"]
        return self.device_mesh["shard"]

    def build_mixed_precision_policy(self) -> object | None:
        return build_mixed_precision_policy(
            policy=self.config.fsdp.get("mp_policy"),
            mixed_precision_policy_cls=MixedPrecisionPolicy,
            label="fsdp.mp_policy",
        )

    def build_offload_policy(self) -> object | None:
        return build_offload_policy(
            policy=self.config.fsdp.get("offload_policy"),
            cpu_offload_policy_cls=CPUOffloadPolicy,
            label="fsdp.offload_policy",
        )

    def fsdp_kwargs(self) -> dict[str, Any]:
        return build_fsdp2_kwargs(
            config=self.config.fsdp,
            mesh=self.fsdp_mesh(),
            mixed_precision_policy=self.build_mixed_precision_policy(),
            offload_policy=self.build_offload_policy(),
            config_name="fsdp",
            supported_keys={
                "enabled",
                "mesh",
                "reshard_after_forward",
                "shard_placement_fn",
                "mp_policy",
                "offload_policy",
                "ignored_params",
            },
            support_hint="mesh/reshard_after_forward/mp_policy/offload_policy",
        )

    def apply_activation_checkpointing(self, model: nn.Module) -> nn.Module:
        """Hook for activation-checkpoint wrapping before compile/FSDP wrapping."""
        return model

    def bind_pipeline_modules(self, modules: Sequence[nn.Module]) -> None:
        if self.pipeline_schedule is None:
            return

        stages = getattr(self.pipeline_schedule, "stages", None)
        if stages is None:
            stage = getattr(self.pipeline_schedule, "stage", None)
            if stage is not None and modules:
                stage.module = modules[0]
                return
            if hasattr(self.pipeline_schedule, "module") and modules:
                self.pipeline_schedule.module = modules[0]
            return

        for stage, module in zip(stages, modules):
            if hasattr(stage, "module"):
                stage.module = module

    def iter_optimizer_parameters(self) -> Iterator[nn.Parameter]:
        parts: list[nn.Module] = list(self.model_parts or [])
        if not parts and self.model is not None:
            parts = [self.model]
        if not parts:
            return
        yield from self._iter_unique_parameters(parts)

    def unwrap(self, model: nn.Module) -> nn.Module:
        if FSDPModule is not None and isinstance(model, FSDPModule):
            return getattr(model, "module", model)
        return super().unwrap(model)

    def _train_no_sync_targets(self) -> tuple[nn.Module, ...]:
        fsdp_parts = [
            module for module in (self.model_parts or []) if FSDPModule is not None and isinstance(module, FSDPModule)
        ]
        if self.model is not None and not fsdp_parts and FSDPModule is not None and isinstance(self.model, FSDPModule):
            fsdp_parts = [self.model]
        if fsdp_parts:
            return tuple(fsdp_parts)
        return super()._train_no_sync_targets()

    def _resolve_pipeline_n_microbatches(self) -> int:
        configured = self.config.parallel.get("pipeline_n_microbatches")
        if configured is not None:
            n_microbatches = int(configured)
            if n_microbatches <= 0:
                raise ValueError(
                    f"invalid parallel.pipeline_n_microbatches: expected a positive integer, got {configured}"
                )
            return n_microbatches

        microbatch_size = int(self.config.parallel.get("pipeline_microbatch_size", 1))
        if microbatch_size <= 0:
            raise ValueError(
                f"invalid parallel.pipeline_microbatch_size: expected a positive integer, got {microbatch_size}"
            )

        try:
            batch_size = int(self.batch_size)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "cannot infer pipeline microbatch count: set `parallel.pipeline_n_microbatches` "
                "or provide `dataloader.batch_size`."
            ) from exc

        if batch_size <= 0:
            raise ValueError(f"invalid batch size: expected a positive integer, got {batch_size}")
        if batch_size % microbatch_size != 0:
            raise ValueError(
                f"batch size ({batch_size}) must be divisible by parallel.pipeline_microbatch_size ({microbatch_size})"
            )

        n_microbatches = batch_size // microbatch_size
        if n_microbatches < self.pipeline_degree:
            warn(
                f"n_microbatches ({n_microbatches}) is less than pipeline_degree ({self.pipeline_degree}); "
                "pipeline utilization may be suboptimal.",
                RuntimeWarning,
                stacklevel=2,
            )
        return n_microbatches

    def _pipeline_loss(self, pred: Any, target: Any) -> torch.Tensor:
        if self.criterion is None:
            raise ValueError("cannot compute pipeline loss: criterion is not initialized")

        loss = self.criterion(pred, target)
        if loss is None:
            raise ValueError("cannot compute pipeline loss: criterion did not produce a loss")
        if loss.ndim > 0:
            loss = loss.mean()

        normalizer = None
        if isinstance(target, Mapping):
            normalizer = self._mapping_loss_normalizer(target)
        if normalizer is None:
            normalizer = self._tensor_loss_normalizer(target)
        divisor = float(max(int(normalizer), 1)) if normalizer is not None else 1.0
        if self._pipeline_loss_weighting is not None:
            self._pipeline_loss_divisor_local += divisor
            if self._pipeline_loss_weighting == "train":
                self._accumulation_divisor_local += divisor
            return loss * divisor
        return loss

    def build_pipeline_schedule(self, stage_model: nn.Module) -> Any:
        pipeline.check()
        schedule_name = str(self.config.parallel.get("pipeline_schedule", "1F1B")).strip() or "1F1B"
        n_microbatches = self._resolve_pipeline_n_microbatches()
        schedule_class = get_schedule_class(schedule_name)
        loss_fn = self._pipeline_loss if self.criterion is not None else None
        stage = PipelineStage(
            stage_model,
            stage_index=self.pipeline_rank,
            num_stages=self.pipeline_degree,
            device=self.device,
            group=self.pipeline_group,
        )

        # Default to non-interleaved 1F1B for pipeline schedules until
        # pytorch/pytorch#164756 is addressed upstream, then we can migrate the
        # default to Interleaved1F1B.
        if issubclass(schedule_class, PipelineScheduleMulti):
            return schedule_class(
                [stage],
                n_microbatches=n_microbatches,
                loss_fn=loss_fn,
                scale_grads=False,
            )
        return schedule_class(
            stage,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=False,
        )

    def build_datasampler(self, dataset, *, split: str, shuffle: bool):
        num_replicas = self.data_degree
        rank = self.data_rank
        if self.ft is not None:
            num_replicas, rank = self.ft.data_parallel_info(num_replicas, rank)
        return utils.data.distributed.DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

    def set_seed(self, seed: int | None = None, bias: int | bool | None = None) -> int:
        if bias is None:
            if self.ft is not None:
                _, bias = self.ft.data_parallel_info(self.data_degree, self.data_rank)
            else:
                bias = self.data_rank
        return super().set_seed(seed=seed, bias=bias)

    def _reduce_degree(self, domain: str = "data") -> int:
        degree = max(self.topology.domain_degree(domain), 1)
        if domain in self._ft_reduced_domains and self.ft is not None:
            group = self.ft.replicate_process_group
            if group is not None and dist.is_available() and dist.is_initialized():
                degree *= max(int(dist.get_world_size(group=group)), 1)
        return degree

    def all_reduce(self, tensor: torch.Tensor, *, domain: str = "data", op=dist.ReduceOp.SUM) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return tensor
        if self.topology.domain_degree(domain) > 1:
            self.parallel.all_reduce(tensor, domain=domain, op=op)
        group = self.ft.replicate_process_group if domain in self._ft_reduced_domains and self.ft is not None else None
        if group is not None:
            dist.all_reduce(tensor, op=op, group=group)
        return tensor

    def _sync_optimizer_skip_decision(self, should_skip: bool) -> bool:
        if not (self.distributed and dist.is_available() and dist.is_initialized()):
            return should_skip
        payload = torch.tensor(float(should_skip), device=self.all_reduce_device())
        self.all_reduce(payload, domain="optimizer", op=dist.ReduceOp.MAX)
        return payload.item() > 0

    def reduce(self, tensor):
        degree = self._reduce_degree("data")
        if degree <= 1 or not (dist.is_available() and dist.is_initialized()):
            return tensor
        original_device = tensor.device
        payload_device = self.all_reduce_device()
        payload = tensor if original_device == payload_device else tensor.to(payload_device)
        self.all_reduce(payload)
        payload = payload / degree
        if payload.device != original_device:
            payload = payload.to(original_device)
        return payload

    def reduce_loss_for_logging(self, loss: torch.Tensor | None, loss_n: int | None) -> torch.Tensor | None:
        if self.pipeline_schedule is None:
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
            self.all_reduce(payload, domain="loss", op=dist.ReduceOp.SUM)
            if payload[1].item() <= 0:
                return None
            return payload[0] / payload[1]
        if not (dist.is_available() and dist.is_initialized()):
            return super().reduce_loss_for_logging(loss, loss_n)
        payload = torch.zeros((3,), dtype=torch.float64, device=self.device)
        is_reporter = self.pipeline_has_last_stage and self.tensor_rank == 0
        if is_reporter:
            if loss is not None:
                normalizer = float(max(int(loss_n or 1), 1))
                loss_value = loss.detach().to(dtype=torch.float64)
                if loss_value.ndim > 0:
                    loss_value = loss_value.mean()
                payload[0] = loss_value * normalizer
                payload[1] = normalizer
                payload[2] = 1.0
            self.all_reduce(payload, domain="loss")

        source_rank = self.topology.rank_from_coordinates({"pipeline": self.pipeline_degree - 1, "tensor": 0})
        dist.broadcast(payload, src=source_rank)
        if payload[2].item() <= 0 or payload[1].item() <= 0:
            return None
        return payload[0] / payload[1]

    @property
    def reports_batch_telemetry(self) -> bool:
        return self.pipeline_has_first_stage and self.tensor_rank == 0

    def _loss_normalizer_sync_divisor(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return max(self._reduce_degree("loss"), 1)
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
        self.all_reduce(total_tensor, domain="loss", op=dist.ReduceOp.SUM)
        return float(total_tensor.item())

    def _use_step_only_loader(self) -> bool:
        return (
            self.pipeline_schedule is not None
            and not self.pipeline_has_first_stage
            and not self.pipeline_has_last_stage
        )

    def _prepare_pipeline_batch(self, data: Any) -> tuple[Any | None, Any | None]:
        if self.pipeline_has_first_stage:
            if data is None:
                raise ValueError("cannot run pipeline stage: first stage requires dataloader inputs")
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
            if not self.pipeline_has_last_stage:
                target = None
            return inputs, target

        if not self.pipeline_has_last_stage or data is None:
            return None, None
        data = self.to_device(data)
        if isinstance(data, Mapping):
            if "target" not in data:
                return None, None
            return None, data["target"]
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)) and len(data) > 1:
            return None, data[1]
        target = None
        return None, target

    def _pipeline_loss_value(self, losses: list[torch.Tensor]) -> torch.Tensor | None:
        if not (self.pipeline_has_last_stage and losses):
            return None
        loss = torch.stack(losses).sum()
        if self._pipeline_loss_divisor_local > 0:
            loss = loss / self._pipeline_loss_divisor_local
        else:
            loss = loss / len(losses)
        return loss

    def _sync_pipeline_accumulation_divisor(self) -> None:
        if self.pipeline_degree <= 1:
            return
        if self.pipeline_group is None or not (dist.is_available() and dist.is_initialized()):
            return

        value = self._pipeline_loss_divisor_local if self.pipeline_has_last_stage else 0.0
        device = self.all_reduce_device()
        payload = torch.tensor(value, dtype=torch.float64, device=device)
        coordinates = dict(self.topology.ranks)
        coordinates["pipeline"] = self.pipeline_degree - 1
        source_rank = self.topology.rank_from_coordinates(coordinates)
        dist.broadcast(payload, src=source_rank, group=self.pipeline_group)
        if not self.pipeline_has_last_stage:
            self._accumulation_divisor_local += float(payload.item())

    @contextmanager
    def train_context(self):
        if self.pipeline_schedule is None:
            with super().train_context():
                yield
            return

        with self._train_step_context(no_sync_targets=self._train_no_sync_targets()):
            yield

    def train_step(self, data) -> Tuple[Any, torch.Tensor | None]:
        if self.pipeline_schedule is None:
            return super().train_step(data)

        with self.train_context():
            self._pipeline_loss_divisor_local = 0.0
            self._pipeline_loss_weighting = "train"
            inputs, target = self._prepare_pipeline_batch(data)
            losses: list[torch.Tensor] = []
            targets = target if self.pipeline_has_last_stage else None

            try:
                if self.pipeline_has_first_stage:
                    self.pipeline_schedule.step(
                        inputs,
                        target=targets,
                        losses=losses,
                    )
                else:
                    self.pipeline_schedule.step(
                        target=targets,
                        losses=losses,
                    )
            finally:
                self._pipeline_loss_weighting = None

            loss = self._pipeline_loss_value(losses)

            pred = None
            self._sync_pipeline_accumulation_divisor()
            self.step()
        return pred, loss

    def evaluate_step(self, data) -> Tuple[Any, torch.Tensor | None]:
        if self.pipeline_schedule is None:
            return super().evaluate_step(data)

        with self.forward_context():
            self._pipeline_loss_divisor_local = 0.0
            self._pipeline_loss_weighting = "eval"
            inputs, target = self._prepare_pipeline_batch(data)
            losses: list[torch.Tensor] = []
            targets = target if self.pipeline_has_last_stage else None

            try:
                if self.pipeline_has_first_stage:
                    self.pipeline_schedule.eval(
                        inputs,
                        target=targets,
                        losses=losses,
                    )
                else:
                    self.pipeline_schedule.eval(
                        target=targets,
                        losses=losses,
                    )
            finally:
                self._pipeline_loss_weighting = None

            loss = self._pipeline_loss_value(losses)

        return None, loss

    @staticmethod
    def _normalize_infer_output(pred: Any) -> list[float]:
        if pred is None:
            return []
        if torch.is_tensor(pred):
            values = pred.detach().reshape(-1).cpu().tolist()
            if isinstance(values, list):
                return [float(value) for value in values]
            return [float(values)]
        if isinstance(pred, Mapping):
            mapped_values: list[float] = []
            for value in pred.values():
                mapped_values.extend(ParallelRunner._normalize_infer_output(value))
            return mapped_values
        if isinstance(pred, Sequence) and not isinstance(pred, (str, bytes)):
            seq_values: list[float] = []
            for value in pred:
                seq_values.extend(ParallelRunner._normalize_infer_output(value))
            return seq_values
        if isinstance(pred, (bool, int, float)):
            return [float(pred)]
        raise ValueError(
            "cannot normalize pipeline infer output: unsupported type "
            f"{type(pred).__name__}; override ParallelRunner.infer_step for custom formats"
        )

    @torch.inference_mode()
    def infer_step(self, data: Any) -> list[float]:
        if self.pipeline_schedule is None:
            return super().infer_step(data)

        with self.forward_context():
            inputs, _ = self._prepare_pipeline_batch(data)
            if self.pipeline_has_first_stage:
                pred = self.pipeline_schedule.eval(inputs)
            else:
                pred = self.pipeline_schedule.eval()
        return self._normalize_infer_output(pred)

    def infer(
        self,
        split: str = "infer",
        *,
        steps: int | None = None,
        stream: bool | None = None,
    ) -> list[float] | Iterator[list[float]]:
        if self.pipeline_schedule is None:
            return super().infer(split=split, steps=steps, stream=stream)

        self.mode = RunnerMode.infer
        self.split = split
        loader = self.dataloaders[split]

        if steps is not None and steps < 0:
            raise ValueError(f"invalid steps: expected a non-negative value, got {steps}")

        loader_length = self._loader_length(loader)
        if stream is None:
            stream = steps is None and loader_length is None

        if self.pipeline_has_first_stage:
            if not stream and loader_length is None and steps is None:
                raise ValueError("infer with stream=False requires `steps` for unsized loaders")
            if steps is not None:
                iterator = (self.infer_step(data) for iteration, data in enumerate(loader) if iteration < steps)
            else:
                iterator = (self.infer_step(data) for data in loader)
            total = steps if steps is not None else loader_length
        else:
            if steps is None:
                if loader_length is None:
                    raise ValueError("infer for non-first pipeline stages requires `steps` for unsized loaders")
                steps = loader_length
            iterator = (self.infer_step(None) for _ in range(steps))
            total = steps

        if stream:
            return iterator

        output: list[float] = []
        for values in tqdm(iterator, total=total, disable=self.distributed and not self.is_main_process):
            output.extend(values)
        return output

    def _export_checkpoint_metadata(self, cls: type = dict) -> Mapping[str, Any]:
        state = cls({"parallel": cls({"axes": cls(self.parallel_axes_state(dict))})})
        if self.fsdp_enabled:
            state["fsdp"] = cls(
                {
                    "mode": self.fsdp_mode,
                    "data_degree": self.data_degree,
                    "shard_degree": self.shard_degree,
                    "replicate_degree": self.replicate_degree,
                    "context_degree": self.context_degree,
                }
            )
        return state

    def _export_checkpoint_components(self, cls: type = dict) -> Mapping[str, Any]:
        state = cls()
        state["ema"] = self.ema.state_dict() if self.ema else None
        state["scheduler"] = self.scheduler.state_dict() if self.scheduler else None
        if len(self.model_parts) != 1:
            state["optimizer"] = self.optimizer.state_dict() if self.optimizer else None
            state["model_parts"] = [self.unwrap(model).state_dict() for model in self.model_parts]
            return state

        model_state_dict, optim_state_dict = self.checkpoint_manager.export_model_optimizer_state(
            model=self.model_parts[0],
            optimizer=self.optimizer,
            options_cls=StateDictOptions,
            strict=True,
        )
        state["model"] = model_state_dict
        state["optimizer"] = optim_state_dict if self.optimizer is not None else None
        return state

    def _restore_model_checkpoint(
        self, state_dict: Mapping[str, Any] | list[Mapping[str, Any]], *args, **kwargs
    ) -> None:
        if isinstance(state_dict, list):
            state_dicts = state_dict
            if len(state_dicts) != len(self.model_parts):
                raise ValueError(
                    "cannot load parallel checkpoint: model_parts count mismatch: "
                    f"expected {len(self.model_parts)}, got {len(state_dicts)}"
                )
            for model, model_state_dict in zip(self.model_parts, state_dicts):
                self.unwrap(model).load_state_dict(model_state_dict, *args, **kwargs)
            return

        if len(self.model_parts) == 1:
            self.checkpoint_manager.load_model_state(
                model=self.model_parts[0],
                model_state_dict=state_dict,
                options_cls=StateDictOptions,
                strict=True,
            )
            return

        super()._restore_model_checkpoint(state_dict, *args, **kwargs)

    def _restore_optimizer_checkpoint(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
        if len(self.model_parts) != 1:
            super()._restore_optimizer_checkpoint(state_dict, *args, **kwargs)
            return

        self.checkpoint_manager.load_optimizer_state(
            model=self.model_parts[0],
            optimizer=self.optimizer,
            optimizer_state_dict=state_dict,
            options_cls=StateDictOptions,
            strict=True,
        )

    def load_checkpoint(self, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> None:
        ckpt = self.read_checkpoint(checkpoint, *args, **kwargs)
        saved_topology = self._validate_checkpoint_topology(ckpt)
        if self.fsdp_enabled:
            self._validate_fsdp_checkpoint_topology(ckpt)
        current_topology = self.parallel_axes_state(dict)
        if saved_topology != current_topology:
            if len(self.model_parts) != 1:
                raise ValueError(
                    "cannot restore parallel degree change: degree change restore requires DCP state-dict API "
                    "with a single local model part. "
                    "Either keep parallel axes unchanged, or restore with a single local model part."
                )

            ckpt = dict(ckpt)
            ckpt["parallel"] = {"axes": current_topology}
            runner_config = ckpt.get("runner")
            if isinstance(runner_config, Mapping):
                runner_payload = dict(runner_config)
                parallel_config = runner_payload.get("parallel")
                if isinstance(parallel_config, Mapping):
                    updated_parallel_config = dict(parallel_config)
                    axes = dict(updated_parallel_config.get("axes", {}))
                    axes.update(current_topology)
                    updated_parallel_config["axes"] = axes
                    runner_payload["parallel"] = updated_parallel_config
                    ckpt["runner"] = runner_payload

        super().load_checkpoint(ckpt, *args, **kwargs)
        if isinstance(checkpoint, (str, bytes, os.PathLike)):
            self.config.resume = os.fsdecode(checkpoint)

    def _validate_checkpoint_topology(self, checkpoint: Mapping[str, Any]) -> dict[str, int]:
        ckpt_topology = checkpoint.get("parallel")
        current = self.parallel_axes_state(dict)
        if not isinstance(ckpt_topology, Mapping):
            return dict(current)

        axes = ckpt_topology.get("axes", {})
        saved = dict(current)
        if isinstance(axes, Mapping):
            for axis in current:
                if axis in axes:
                    saved[axis] = int(axes[axis])
        if saved == current:
            return saved

        allow_degree_change = self.config.parallel.allow_degree_change
        if allow_degree_change:
            warn(
                "parallel degree changed across restart "
                f"(saved axes={saved}, current axes={current}). "
                "Attempting to restore with current runtime mapping.",
                RuntimeWarning,
                stacklevel=2,
            )
            return saved

        raise ValueError(
            "cannot restore checkpoint: parallel degree changed across restart "
            f"(saved axes={saved}, current axes={current}). "
            "Set `config.parallel.allow_degree_change=True` to proceed explicitly."
        )

    def _validate_fsdp_checkpoint_topology(self, checkpoint: Mapping[str, Any]) -> tuple[str, int, int, int]:
        ckpt_topology = checkpoint.get("fsdp")
        current = (
            self.fsdp_mode,
            self.replicate_degree,
            self.shard_degree,
            self.context_degree,
        )
        if not isinstance(ckpt_topology, Mapping):
            raise ValueError(
                "cannot restore parallel FSDP checkpoint: checkpoint is missing 'fsdp' topology metadata. "
                "Start a new run or use a checkpoint written by the current parallel FSDP runner."
            )

        saved = (
            str(ckpt_topology.get("mode", current[0])),
            int(ckpt_topology.get("replicate_degree", current[1])),
            int(ckpt_topology.get("shard_degree", current[2])),
            int(ckpt_topology.get("context_degree", current[3])),
        )
        if saved != current:
            raise ValueError(
                "cannot restore checkpoint: parallel FSDP topology changed across restart "
                f"(saved mode/replicate/shard/context={saved}, current mode/replicate/shard/context={current})."
            )
        return saved

    def close(self, timeout: float | None = None) -> bool:
        self._reset_model_parallel_groups()
        self._parallel_groups_initialized = False
        return super().close(timeout=timeout)

    @property
    def tensor_degree(self) -> int:
        return self.topology.axis_degree("tensor")

    @property
    def pipeline_degree(self) -> int:
        return self.topology.axis_degree("pipeline")

    @property
    def data_degree(self) -> int:
        return self.topology.domain_degree("data")

    @property
    def tensor_rank(self) -> int:
        return self.topology.axis_rank("tensor")

    @property
    def pipeline_rank(self) -> int:
        return self.topology.axis_rank("pipeline")

    @property
    def data_rank(self) -> int:
        return self.topology.domain_rank("data")

    @property
    def model_parallel_axes(self) -> tuple[str, ...]:
        return tuple(
            axis for axis in ("tensor", "context", "expert", "expert_tensor") if self.topology.axis_degree(axis) > 1
        )

    @property
    def model_parallel_degree(self) -> int:
        degree = 1
        for axis in self.model_parallel_axes:
            degree *= self.topology.axis_degree(axis)
        return degree

    def parallel_axes_state(self, cls: type = dict) -> Mapping[str, int]:
        return cls({axis: self.topology.axis_degree(axis) for axis in self.topology.axis_names})

    @property
    def fsdp_mode(self) -> str:
        if self.replicate_degree > 1:
            return "hybrid_shard"
        return "full_shard"

    @property
    def shard_degree(self) -> int:
        return self.topology.axis_degree("shard")

    @property
    def replicate_degree(self) -> int:
        return self.topology.axis_degree("replicate", default=1)

    @property
    def shard_rank(self) -> int:
        return self.topology.axis_rank("shard")

    @property
    def replicate_rank(self) -> int:
        return self.topology.axis_rank("replicate", default=0)

    @property
    def context_degree(self) -> int:
        return self.topology.axis_degree("context")

    @property
    def context_rank(self) -> int:
        return self.topology.axis_rank("context")

    @property
    def expert_degree(self) -> int:
        return self.topology.axis_degree("expert")

    @property
    def expert_rank(self) -> int:
        return self.topology.axis_rank("expert")

    @property
    def expert_tensor_degree(self) -> int:
        return self.topology.axis_degree("expert_tensor")

    @property
    def expert_tensor_rank(self) -> int:
        return self.topology.axis_rank("expert_tensor")
