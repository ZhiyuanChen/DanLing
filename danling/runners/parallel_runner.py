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
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from typing import Any, Iterator
from warnings import warn

import torch
from lazy_imports import try_import
from torch import distributed as dist
from torch import nn, utils
from tqdm import tqdm

from danling.data import DataLoaderDict, StepProxyLoader

from .checkpoints import TorchDistributedCheckpointManager
from .config import RunnerConfig
from .fsdp import (
    build_fsdp2_kwargs,
    build_mixed_precision_policy,
    build_offload_policy,
    normalize_reshard_after_forward,
)
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
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

with try_import() as activation_checkpoint:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing as apply_torch_activation_checkpointing,
        checkpoint_wrapper,
    )


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
    """
    Torch runner for data, FSDP, pipeline, and model-parallel stacks.

    Use this runner when training spans explicit parallel axes (`replicate`,
    `shard`, `pipeline`, `tensor`, `context`, `expert`, `expert_tensor`) rather
    than plain DDP. It keeps the TorchRunner outer lifecycle and replaces the
    distributed topology, sampler, model materialization, collective reduction,
    pipeline step, and checkpoint semantics.

    Checkpoint invariants:
        - Distributed parallel runs use `ckpt.backend="dcp"` only.
        - Single-local-part checkpoints use torch.distributed.checkpoint
          state-dict APIs when available.
        - Restore order is model first, then optimizer, then scheduler.

    Attributes:
        topology: Rank/axis layout for the current world.
        parallel: Process-group/device-mesh context built from `topology`.
        model_parts: Local pipeline/FSDP model parts. `self.model` is the
            first local part for compatibility with TorchRunner helpers.
        pipeline_schedule: Optional PyTorch pipeline schedule.
        pipeline_has_first_stage: Whether this rank owns pipeline input.
        pipeline_has_last_stage: Whether this rank owns pipeline target/loss.
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
    _fault_tolerance_reduced_domains = frozenset({"data", "batch", "loss", "optimizer", "fsdp"})
    _pipeline_loss_divisor_local: float = 0.0
    _pipeline_loss_weighting: str | None = None

    model_parts: list[nn.Module]

    checkpoint_manager: TorchDistributedCheckpointManager

    def __init__(self, config: Mapping[str, Any]) -> None:
        dcp.check()
        if not isinstance(config, RunnerConfig):
            config = RunnerConfig(config)
        requested_backend = str(config.get("ckpt.backend")).strip().lower()
        config.stack = "parallel"
        if requested_backend != "dcp":
            if requested_backend != "auto":
                warn(
                    f"{self.__class__.__name__} overrides ckpt.backend to 'dcp'",
                    RuntimeWarning,
                    stacklevel=2,
                )
            config["ckpt"]["backend"] = "dcp"
        super().__init__(config)
        self.dataloaders = _ParallelDataLoaderDict(self)

    @property
    def fsdp_enabled(self) -> bool:
        return bool(self.config.fsdp.get("enabled", False))

    def init_distributed(self) -> None:
        """
        Initialize default distributed state and parallel process groups.

        **Called when:** `BaseRunner.__init__` invokes `init_distributed`,
        before checkpoint manager/fault-tolerance setup and before model
        materialization.

        **Precondition:** `WORLD_SIZE > 1` and the configured parallel axis
        product equals `WORLD_SIZE`.

        Raises:
            RuntimeError: distributed mode is not active, or device-mesh process
                groups cannot be initialized.
            ValueError: `build_topology` rejects the configured axis product.

        **Side effects:** calls `TorchRunner.init_distributed`, builds
        `self.topology`, initializes the device mesh, binds per-axis process
        groups, and stores `self.parallel`.

        !!! danger "Do not"
            - Initialize model/pipeline/FSDP objects here; materialization
              happens in `materialize_model`.
            - Override this just to change axis degrees; set
              `config.parallel.axes` or override `build_topology`.
        """
        super().init_distributed()
        if self.world_size <= 1:
            raise RuntimeError("ParallelRunner requires distributed mode (WORLD_SIZE > 1)")
        self.topology = self.build_topology()
        if not self._parallel_groups_initialized:
            self._reset_model_parallel_groups()
            self._init_model_parallel_groups()
            self._parallel_groups_initialized = True

    def build_topology(self) -> ParallelTopology:
        """
        Build the rank-to-axis topology for this parallel run.

        **Called when:** `init_distributed` has initialized the default process
        group and needs per-axis domains.

        Returns:
            `ParallelTopology` with axis degrees, current-rank coordinates, and
            named reduction domains.

        Raises:
            ValueError: any axis degree is less than one, or the product of axis
                degrees does not equal `WORLD_SIZE`.

        **Side effects:** none. Override this only for non-standard axis/domain
        layouts; normal users should configure `config.parallel.axes`.
        """
        axes = {
            "replicate": int(self.config.parallel.axes.replicate),
            "shard": int(self.config.parallel.axes.shard),
            "context": int(self.config.parallel.axes.context),
            "pipeline": int(self.config.parallel.axes.pipeline),
            "tensor": int(self.config.parallel.axes.tensor),
            "expert": int(self.config.parallel.axes.expert),
            "expert_tensor": int(self.config.parallel.axes.expert_tensor),
        }
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
        if self.config.activation_checkpoint.enabled:
            activation_checkpoint.check()
        torchft_config_supported = (
            self.fsdp_enabled
            and int(self.config.parallel.axes.pipeline) == 1
            and int(self.config.parallel.axes.tensor) == 1
            and int(self.config.parallel.axes.context) == 1
            and int(self.config.parallel.axes.expert) == 1
            and int(self.config.parallel.axes.expert_tensor) == 1
        )
        if self.fault_tolerance is not None and self.fault_tolerance.enabled and not torchft_config_supported:
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
        """
        Materialize local model parts for FSDP/pipeline/model-parallel training.

        **Called when:** `TorchRunner.__post_init__` reaches
        `materialize_model`, after FP8 setup and before optimizer build.

        **Precondition:** either `self.model` or `self.model_parts` is bound.
        Pipeline runs may also provide `self.pipeline_schedule`; otherwise a
        single local model is converted to a pipeline stage when
        `pipeline_degree > 1`.

        Raises:
            RuntimeError: FSDP prerequisites are unavailable.
            ValueError: model/model_parts are missing or an unsupported
                auto-pipeline shape is requested.

        **Side effects:** moves local parts to `self.device`, calls
        `parallelize_model`, applies FP8 policy and optional activation
        checkpointing, compiles each part, optionally wraps parts with FSDP2,
        binds pipeline schedule modules, installs TorchFT all-reduce hooks for
        FSDP, and moves EMA to device.

        !!! danger "Do not"
            - Build the optimizer before this hook; optimizer parameters must
              come from materialized/wrapped parts.
            - FSDP-wrap before `apply_activation_checkpointing`.
            - Replace `self.model_parts` without keeping `self.model` aligned
              to the first local part.
        """
        if self.fsdp_enabled:
            self._check_fsdp_prerequisites()
        self._maybe_init_pipeline_schedule_from_single_part()
        parts = self._prepare_local_model_parts()
        if self.fp8_enabled:
            self.apply_fp8_module_policy_to_model_parts()
            parts = list(self.model_parts)
        parts = [self.apply_activation_checkpointing(part) for part in parts]

        compiled = [self.compiler.compile(part) for part in parts]
        if self.fsdp_enabled:
            fsdp_kwargs = self.fsdp_kwargs()
            wrapped = [self.apply_fsdp(part, fsdp_kwargs) for part in compiled]
        else:
            wrapped = compiled

        self.model_parts = wrapped
        self.model = wrapped[0]
        self.bind_pipeline_modules(self.model_parts)

        if self.fsdp_enabled:
            self._apply_fault_tolerance_all_reduce_hook()
        if self.ema is not None:
            self.ema = self.ema.to(self.device)

    def _check_fsdp_prerequisites(self) -> None:
        if fully_shard is None or FSDPModule is None:
            raise RuntimeError("cannot initialize ParallelRunner FSDP: torch.distributed.fsdp.fully_shard is required")
        if not torch.cuda.is_available():
            raise RuntimeError("ParallelRunner FSDP requires CUDA when WORLD_SIZE > 1")

    def setup_grad_scaler(self) -> None:
        self.grad_scaler = None
        precision = self._normalized_precision_name(self.precision)
        if precision not in {"fp16", "float16", "half"}:
            return
        if not self.runner_owns_grad_scaling():
            return
        if self.fp8_enabled:
            raise ValueError("precision='fp16' cannot be combined with FP8 autocast")
        if self.device.type != "cuda":
            raise ValueError("runner-owned fp16 precision requires a CUDA device; use bf16 or a backend-owned scaler")
        if self.fsdp_enabled:
            parallel_fsdp.check()
            if ShardedGradScaler is None:
                raise RuntimeError("ParallelRunner FSDP fp16 requires torch.distributed.fsdp.ShardedGradScaler")
            self.grad_scaler = ShardedGradScaler()
            return
        self.grad_scaler = torch.amp.GradScaler(device=self.device.type)

    def _maybe_init_pipeline_schedule_from_single_part(self) -> None:
        if self.pipeline_schedule is not None or self.pipeline_degree <= 1 or self.pipeline_group is None:
            return
        if self.model_parts and len(self.model_parts) != 1:
            raise ValueError(
                "cannot auto-materialize pipeline from multiple local model_parts; "
                "provide `pipeline_schedule` explicitly when pre-partitioning local stages"
            )
        stage_model = self.model_parts[0] if self.model_parts else self.model
        if stage_model is None:
            raise ValueError("cannot materialize pipeline: model is not initialized")
        stage_models = self.build_pipeline_model_parts(stage_model)
        schedule_input: nn.Module | Sequence[nn.Module] = stage_models[0] if len(stage_models) == 1 else stage_models
        self.pipeline_schedule = self.build_pipeline_schedule(schedule_input)
        self.model_parts = stage_models
        self.model = stage_models[0]
        stage_indices = self.pipeline_stage_indices()
        self.pipeline_has_first_stage = 0 in stage_indices
        self.pipeline_has_last_stage = self._pipeline_num_stages() - 1 in stage_indices

    def _pipeline_num_stages(self) -> int:
        pipeline_partitions = self.config.parallel.get("pipeline_partitions")
        if pipeline_partitions is not None:
            return len(pipeline_partitions)
        return self.pipeline_degree

    def pipeline_stage_indices(self, num_stages: int | None = None) -> tuple[int, ...]:
        """
        Return the pipeline stage indices owned by this rank.

        The default supports the common looped virtual-stage mapping used by
        interleaved schedules: rank `r` owns `r`, `r + pp_degree`, ...
        Override this method for mirrored, zero-bubble, or other custom local
        stage placement.
        """
        if num_stages is None:
            num_stages = self._pipeline_num_stages()
        if num_stages < self.pipeline_degree:
            raise ValueError(
                "pipeline num_stages must be at least pipeline_degree " f"({self.pipeline_degree}), got {num_stages}"
            )
        if num_stages % self.pipeline_degree != 0:
            raise ValueError(
                "pipeline num_stages must be divisible by pipeline_degree "
                f"({self.pipeline_degree}), got {num_stages}"
            )

        stages_per_rank = num_stages // self.pipeline_degree
        if stages_per_rank == 1:
            return (self.pipeline_rank,)

        return tuple(self.pipeline_rank + offset * self.pipeline_degree for offset in range(stages_per_rank))

    def build_pipeline_model_part(self, model: nn.Module) -> nn.Module:
        """
        Return the local pipeline model part for this pipeline rank.

        The default supports two user-facing contracts:

        - If the model defines `build_pipeline_model_part(...)`, delegate to it.
        - If `parallel.pipeline_partitions` is configured, extract those
          named modules for the current pipeline rank. Multiple FQNs become a
          simple `nn.Sequential` in the provided order.

        Complex graph partitioning should be implemented in the model hook or
        by overriding this method.
        """
        stage_index = self.pipeline_stage_indices()[0]
        module_fqns = self._pipeline_module_fqns_for_stage(stage_index)
        return self._build_pipeline_model_part(model, stage_index, self._pipeline_num_stages(), module_fqns)

    def build_pipeline_model_parts(self, model: nn.Module) -> list[nn.Module]:
        """
        Return all local pipeline model parts for this pipeline rank.

        Override this when a schedule maps multiple stages to each local rank
        and the default FQN/model-owned partitioning is not expressive enough.
        """
        stage_indices = self.pipeline_stage_indices()
        if len(stage_indices) == 1:
            return [self.build_pipeline_model_part(model)]

        build_part = getattr(model, "build_pipeline_model_part", None)
        has_fqn_partitions = self.config.parallel.get("pipeline_partitions") is not None
        if not callable(build_part) and not has_fqn_partitions:
            raise ValueError(
                "multiple local pipeline stages require `parallel.pipeline_partitions`, "
                "`model.build_pipeline_model_part(...)`, or an override of "
                "`ParallelRunner.build_pipeline_model_parts`"
            )

        num_stages = self._pipeline_num_stages()
        return [
            self._build_pipeline_model_part(
                model,
                stage_index,
                num_stages,
                self._pipeline_module_fqns_for_stage(stage_index),
            )
            for stage_index in stage_indices
        ]

    def _build_pipeline_model_part(
        self,
        model: nn.Module,
        stage_index: int,
        num_stages: int,
        module_fqns: tuple[str, ...] | None,
    ) -> nn.Module:
        build_part = getattr(model, "build_pipeline_model_part", None)
        if callable(build_part):
            part = build_part(
                stage_index=stage_index,
                num_stages=num_stages,
                module_fqns=module_fqns,
                parallel=self.parallel,
            )
            if part is None:
                return model
            if not isinstance(part, nn.Module):
                raise TypeError(
                    "model.build_pipeline_model_part(...) must return an nn.Module or None, "
                    f"got {type(part).__name__}"
                )
            return part

        if module_fqns is None:
            return model
        return self._build_pipeline_model_part_from_fqns(model, module_fqns)

    def _pipeline_module_fqns_for_stage(self, stage_index: int) -> tuple[str, ...] | None:
        pipeline_partitions = self.config.parallel.get("pipeline_partitions")
        if pipeline_partitions is None:
            return None
        if stage_index < 0 or stage_index >= len(pipeline_partitions):
            raise ValueError(
                "pipeline stage index is outside parallel.pipeline_partitions: "
                f"stage_index={stage_index}, num_stages={len(pipeline_partitions)}"
            )
        module_fqns = pipeline_partitions[stage_index]
        if isinstance(module_fqns, str):
            module_fqns = (module_fqns,)
        else:
            module_fqns = tuple(str(module_fqn) for module_fqn in module_fqns)
        if not module_fqns:
            raise ValueError(
                "parallel.pipeline_partitions entries must not be empty; "
                f"pipeline stage {stage_index} has no modules"
            )
        return module_fqns

    def _pipeline_module_fqns_for_rank(self) -> tuple[str, ...] | None:
        return self._pipeline_module_fqns_for_stage(self.pipeline_stage_indices()[0])

    def _build_pipeline_model_part_from_fqns(self, model: nn.Module, module_fqns: Sequence[str]) -> nn.Module:
        modules = dict(model.named_modules())
        if "" in module_fqns:
            raise ValueError("parallel.pipeline_partitions may not select the root module")
        missing = [module_fqn for module_fqn in module_fqns if module_fqn not in modules]
        if missing:
            raise ValueError(f"unknown pipeline module FQN(s): {missing}")
        if len(set(module_fqns)) != len(module_fqns):
            raise ValueError(f"duplicate pipeline module FQN(s): {list(module_fqns)}")
        if len(module_fqns) == 1:
            return modules[module_fqns[0]]
        return nn.Sequential(
            OrderedDict((module_fqn.replace(".", "_"), modules[module_fqn]) for module_fqn in module_fqns)
        )

    def _prepare_local_model_parts(self) -> list[nn.Module]:
        if self.pipeline_schedule is None:
            if self.model is None:
                if self.model_parts:
                    self.model = self.model_parts[0]
                else:
                    raise ValueError("cannot materialize parallel model: model is not initialized")
            parts: list[nn.Module] = [self.model]
        else:
            if not self.model_parts:
                if self.model is None:
                    raise ValueError("cannot materialize pipeline: model_parts are not initialized")
                self.model_parts = [self.model]
            parts = list(self.model_parts)
        parts = [part.to(self.device) for part in parts]
        parts = [self.parallelize_model(part) for part in parts]
        self.model_parts = parts
        self.model = parts[0]
        return parts

    def _apply_fault_tolerance_all_reduce_hook(self) -> None:
        if self.fault_tolerance is None:
            return
        group = self.fault_tolerance.replicate_process_group
        if group is None:
            return

        def all_reduce_hook(output):
            dist.all_reduce(output, group=group, op=dist.ReduceOp.AVG)

        def apply_hook(module: nn.Module) -> None:
            set_all_reduce_hook = getattr(module, "set_all_reduce_hook", None)
            if callable(set_all_reduce_hook):
                set_all_reduce_hook(all_reduce_hook)

        for model in self.model_parts:
            model.apply(apply_hook)

    def parallelize_model(self, model: nn.Module) -> nn.Module:
        """
        Apply model-specific tensor/context/expert parallel transforms.

        **Called when:** `_prepare_local_model_parts` materializes each local
        part, before compile and FSDP wrapping.

        Args:
            model: Local model part to transform.

        Returns:
            The transformed model. If the model defines
            `model.parallelize(parallel)`, that method may mutate in place and
            return `None`.

        Raises:
            TypeError: `model.parallelize` returns a non-module value.
            NotImplementedError: model-parallel axes are enabled but no
                transform hook is available.

        !!! danger "Do not"
            - Move the model to device here; the surrounding `materialize_model`
              flow handles device placement before this hook runs.
            - Compile or FSDP-wrap here; those happen after this hook.
        """
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
            policy=self.config.fsdp.get("mixed_precision_policy"),
            mixed_precision_policy_cls=MixedPrecisionPolicy,
            label="fsdp.mixed_precision_policy",
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
                "module_classes",
                "mesh",
                "reshard_after_forward",
                "root_reshard_after_forward",
                "shard_placement_fn",
                "mixed_precision_policy",
                "offload_policy",
                "ignored_params",
            },
            support_hint=(
                "mesh/module_classes/reshard_after_forward/"
                "root_reshard_after_forward/shard_placement_fn/mixed_precision_policy/offload_policy"
            ),
            pipeline_enabled=self.pipeline_degree > 1,
        )

    def apply_fsdp(self, model: nn.Module, fsdp_kwargs: Mapping[str, Any]) -> nn.Module:
        """Apply configured FSDP2 wrapping to one local model part."""

        self.apply_fsdp_to_modules(model, fsdp_kwargs)
        root_kwargs = dict(fsdp_kwargs)
        root_reshard_after_forward = self.config.fsdp.get("root_reshard_after_forward")
        if root_reshard_after_forward is not None:
            root_kwargs["reshard_after_forward"] = normalize_reshard_after_forward(
                root_reshard_after_forward,
                pipeline_enabled=self.pipeline_degree > 1,
            )
        root = fully_shard(model, **root_kwargs)
        return root

    def apply_fsdp_to_modules(self, model: nn.Module, fsdp_kwargs: Mapping[str, Any]) -> tuple[nn.Module, ...]:
        """Shard explicitly configured submodules before sharding the root."""

        module_classes = self.config.fsdp.get("module_classes")
        if not module_classes:
            return ()

        matches = self.fsdp_modules(model, module_classes)
        if not matches:
            classes = ", ".join(str(module_class) for module_class in module_classes)
            raise ValueError(f"fsdp.module_classes matched no modules in {type(model).__qualname__}: {classes}")

        kwargs = dict(fsdp_kwargs)
        wrapped: list[nn.Module] = []
        for module in matches:
            wrapped.append(fully_shard(module, **kwargs))
        return tuple(wrapped)

    @staticmethod
    def fsdp_modules(model: nn.Module, module_classes: Sequence[str]) -> tuple[nn.Module, ...]:
        """Return matching child modules in child-before-parent FSDP order."""

        class_names = {str(module_class) for module_class in module_classes}
        matches: list[tuple[int, int, nn.Module]] = []
        for index, (name, module) in enumerate(model.named_modules()):
            if not name:
                continue
            module_type = type(module)
            qualified_name = f"{module_type.__module__}.{module_type.__qualname__}"
            if module_type.__name__ in class_names or qualified_name in class_names:
                matches.append((name.count("."), index, module))
        matches.sort(key=lambda item: (-item[0], item[1]))
        return tuple(module for _depth, _index, module in matches)

    def apply_activation_checkpointing(self, model: nn.Module) -> nn.Module:
        """
        Apply activation checkpointing to one local model part.

        **Called when:** `materialize_model` prepares each local part before
        compile/FSDP wrapping.

        Args:
            model: Local model part.

        Returns:
            Model part with activation checkpointing wrappers applied.

        **Side effects:** default wraps modules matching
        `config.activation_checkpoint.module_classes` when activation
        checkpointing is enabled. Overrides may mutate the module in place or
        return a wrapped module.

        !!! danger "Do not"
            - Change parameter ownership or shard layout here; FSDP has not
              wrapped the model yet.
            - Return a non-module value.
        """
        if not self.config.activation_checkpoint.enabled:
            return model

        module_classes = self.config.activation_checkpoint.module_classes
        if not module_classes:
            raise ValueError(
                "activation_checkpoint.enabled=True requires "
                "activation_checkpoint.module_classes or an overridden apply_activation_checkpointing()."
            )
        class_names = {str(name) for name in module_classes}

        def check_fn(module: nn.Module) -> bool:
            module_type = type(module)
            qualified_name = f"{module_type.__module__}.{module_type.__qualname__}"
            return module_type.__name__ in class_names or qualified_name in class_names

        wrapper = partial(checkpoint_wrapper, checkpoint_impl=self._activation_checkpoint_impl())
        apply_torch_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn)
        return model

    def _activation_checkpoint_impl(self):
        impl = str(self.config.activation_checkpoint.checkpoint_impl).strip().lower().replace("-", "_")
        aliases = {
            "no_reentrant": CheckpointImpl.NO_REENTRANT,
            "non_reentrant": CheckpointImpl.NO_REENTRANT,
            "reentrant": CheckpointImpl.REENTRANT,
        }
        if impl not in aliases:
            raise ValueError(
                "invalid activation_checkpoint.checkpoint_impl: "
                f"{self.config.activation_checkpoint.checkpoint_impl!r}. Expected 'no_reentrant' or 'reentrant'."
            )
        return aliases[impl]

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

    def iter_optimizer_named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        parts: list[nn.Module] = list(self.model_parts or [])
        if not parts and self.model is not None:
            parts = [self.model]
        if not parts:
            return
        prefixes = ("",) if len(parts) == 1 else tuple(f"part{index}." for index in range(len(parts)))
        yield from self._iter_unique_named_parameters(parts, prefixes)

    def unwrap(self, model: nn.Module) -> nn.Module:
        if FSDPModule is not None and isinstance(model, FSDPModule):
            return getattr(model, "module", model)
        return super().unwrap(model)

    def _train_no_sync_targets(self) -> tuple[nn.Module, ...]:
        fsdp_parts: list[nn.Module] = [
            module for module in (self.model_parts or []) if FSDPModule is not None and isinstance(module, FSDPModule)
        ]
        if self.model is not None and not fsdp_parts and FSDPModule is not None and isinstance(self.model, FSDPModule):
            fsdp_parts = [self.model]
        if fsdp_parts:
            return tuple(fsdp_parts)
        return super()._train_no_sync_targets()

    def _resolve_pipeline_microbatches(self) -> int:
        configured = self.config.parallel.get("pipeline_microbatches")
        if configured is not None:
            microbatches = int(configured)
            if microbatches <= 0:
                raise ValueError(
                    f"invalid parallel.pipeline_microbatches: expected a positive integer, got {configured}"
                )
            return microbatches

        microbatch_size = int(self.config.parallel.get("pipeline_microbatch_size", 1))
        if microbatch_size <= 0:
            raise ValueError(
                f"invalid parallel.pipeline_microbatch_size: expected a positive integer, got {microbatch_size}"
            )

        try:
            batch_size = int(self.batch_size)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "cannot infer pipeline microbatch count: set `parallel.pipeline_microbatches` "
                "or provide `dataloader.batch_size`."
            ) from exc

        if batch_size <= 0:
            raise ValueError(f"invalid batch size: expected a positive integer, got {batch_size}")
        if batch_size % microbatch_size != 0:
            raise ValueError(
                f"batch size ({batch_size}) must be divisible by parallel.pipeline_microbatch_size ({microbatch_size})"
            )

        microbatches = batch_size // microbatch_size
        if microbatches < self.pipeline_degree:
            warn(
                f"pipeline_microbatches ({microbatches}) is less than pipeline_degree ({self.pipeline_degree}); "
                "pipeline utilization may be suboptimal.",
                RuntimeWarning,
                stacklevel=2,
            )
        return microbatches

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

    def build_pipeline_schedule(self, stage_model: nn.Module | Sequence[nn.Module]) -> Any:
        """
        Build the PyTorch pipeline schedule for this rank.

        **Called when:** `materialize_model` sees `pipeline_degree > 1` and no
        explicit `pipeline_schedule` is already bound.

        Args:
            stage_model: Local stage module for this pipeline rank, or all
                local stage modules for an interleaved/multi-stage schedule.

        Returns:
            A PyTorch pipeline schedule instance.

        Raises:
            ValueError: pipeline microbatch count cannot be inferred or is
                inconsistent with batch size.

        **Side effects:** none beyond schedule construction. The caller binds
        the schedule modules after compile/FSDP wrapping.

        !!! danger "Do not"
            - Set `scale_grads=True`; DanLing owns gradient/loss scaling.
            - Build the optimizer here.
        """
        pipeline.check()
        schedule_name = str(self.config.parallel.get("pipeline_schedule", "1F1B")).strip() or "1F1B"
        n_microbatches = self._resolve_pipeline_microbatches()
        schedule_class = get_schedule_class(schedule_name)
        loss_fn = self._pipeline_loss if self.criterion is not None else None
        stage_models = [stage_model] if isinstance(stage_model, nn.Module) else list(stage_model)
        num_stages = self._pipeline_num_stages()
        stage_indices = self.pipeline_stage_indices(num_stages)
        if len(stage_models) != len(stage_indices):
            raise ValueError(
                "pipeline stage model count must match local pipeline stage indices: "
                f"{len(stage_models)} != {len(stage_indices)}"
            )
        stages = [
            PipelineStage(
                module,
                stage_index=stage_index,
                num_stages=num_stages,
                device=self.device,
                group=self.pipeline_group,
            )
            for module, stage_index in zip(stage_models, stage_indices)
        ]

        # Default to non-interleaved 1F1B for pipeline schedules until
        # pytorch/pytorch#164756 is addressed upstream, then we can migrate the
        # default to Interleaved1F1B.
        if issubclass(schedule_class, PipelineScheduleMulti):
            return schedule_class(
                stages,
                n_microbatches=n_microbatches,
                loss_fn=loss_fn,
                scale_grads=False,
            )
        if len(stages) != 1:
            raise ValueError(
                f"pipeline schedule {schedule_name!r} accepts one local stage, got {len(stages)}; "
                "choose an interleaved/multi-stage schedule or override `build_pipeline_schedule`."
            )
        return schedule_class(
            stages[0],
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=False,
        )

    def build_datasampler(self, dataset: Any, *, split: str, shuffle: bool) -> Any:
        """
        Build a data-parallel sampler for one split.

        **Called when:** inherited `build_dataloaders` materializes a dataset
        split.

        Args:
            dataset: Dataset object for the split.
            split: Split name being materialized.
            shuffle: Whether to shuffle the split.

        Returns:
            `DistributedSampler` using topology data-parallel degree/rank,
            adjusted by TorchFT when active.
        """
        num_replicas = self.data_degree
        rank = self.data_rank
        if self.fault_tolerance is not None:
            num_replicas, rank = self.fault_tolerance.data_parallel_info(num_replicas, rank)
        return utils.data.distributed.DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

    def set_seed(self, seed: int | None = None, bias: int | bool | None = None) -> int:
        if bias is None:
            if self.fault_tolerance is not None:
                _, bias = self.fault_tolerance.data_parallel_info(self.data_degree, self.data_rank)
            else:
                bias = self.data_rank
        return super().set_seed(seed=seed, bias=bias)

    def _reduce_degree(self, domain: str = "data") -> int:
        degree = max(self.topology.domain_degree(domain), 1)
        if domain in self._fault_tolerance_reduced_domains and self.fault_tolerance is not None:
            group = self.fault_tolerance.replicate_process_group
            if group is not None and dist.is_available() and dist.is_initialized():
                degree *= max(int(dist.get_world_size(group=group)), 1)
        return degree

    def all_reduce(self, tensor: torch.Tensor, *, domain: str = "data", op=dist.ReduceOp.SUM) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return tensor
        if self.topology.domain_degree(domain) > 1:
            self.parallel.all_reduce(tensor, domain=domain, op=op)
        group = (
            self.fault_tolerance.replicate_process_group
            if domain in self._fault_tolerance_reduced_domains and self.fault_tolerance is not None
            else None
        )
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

    def train_step(self, data: Any) -> tuple[Any, torch.Tensor | None]:
        """
        Run one training micro-step for plain or pipeline-parallel execution.

        Non-pipeline configurations delegate to `TorchRunner.train_step`.
        Pipeline configurations call the schedule, compute loss only on last
        stages, synchronize accumulation normalization across the pipeline, and
        then delegate optimizer-boundary handling to `step()`.

        **Called when:** `train_epoch`/`train_steps` consume one micro-batch.

        Args:
            data: Micro-batch from the local loader. Non-first/non-last
                pipeline stages may receive `None` through `StepProxyLoader`.

        Returns:
            `(None, loss)` for pipeline mode, where `loss` is present only on
            ranks that can report last-stage loss. Non-pipeline mode returns
            the TorchRunner result.

        !!! danger "Do not"
            - Call the optimizer directly; use `step()`.
            - Update metrics from pipeline mode here; pipeline schedule outputs
              are not a normal full-batch prediction.
            - Manually divide gradients by pipeline microbatch count.
        """
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

    def evaluate_step(self, data: Any) -> tuple[Any, torch.Tensor | None]:
        """
        Run one evaluation micro-step for plain or pipeline execution.

        Non-pipeline configurations delegate to `TorchRunner.evaluate_step`.
        Pipeline configurations call the schedule in eval mode and report
        normalized loss from last-stage ranks.

        **Called when:** `evaluate_epoch`/`evaluate_steps` consume one
        micro-batch under inference mode.

        Args:
            data: Micro-batch from the local loader. Non-first/non-last
                pipeline stages may receive `None`.

        Returns:
            `(None, loss)` for pipeline mode. Non-pipeline mode returns the
            TorchRunner result.

        !!! danger "Do not"
            - Call backward or step.
            - Assume every rank has targets; only last-stage ranks need them.
        """
        if self.pipeline_schedule is None:
            return super().evaluate_step(data)

        with self.infer_context():
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
        """
        Run one inference micro-step for plain or pipeline execution.

        Non-pipeline configurations delegate to `TorchRunner.infer_step`.
        Pipeline configurations call the schedule in eval mode and normalize
        whatever the schedule returns into a flat list of floats.

        Args:
            data: Micro-batch on first-stage ranks; `None` on non-first stages
                that only participate in pipeline communication.

        Returns:
            Flat list of numeric predictions. Non-output ranks may return an
            empty list.

        Raises:
            ValueError: pipeline output cannot be normalized into floats.
        """
        if self.pipeline_schedule is None:
            return super().infer_step(data)

        with self.infer_context():
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
        """
        Run inference across a pipeline-aware loader.

        Non-pipeline configurations delegate to `TorchRunner.infer`. Pipeline
        configurations consume real dataloader batches only on first-stage
        ranks; other stages run `infer_step(None)` for the same number of
        steps.

        Args:
            split: Inference split name.
            steps: Optional maximum number of batches/stage ticks.
            stream: Whether to return a per-batch iterator instead of a
                flattened list.

        Returns:
            Flattened predictions or a streaming iterator.

        Raises:
            ValueError: `steps` is negative, or a non-first pipeline stage has
                an unsized loader and no explicit step count.
        """
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

    def load_checkpoint(
        self,
        checkpoint: Mapping | bytes | str | os.PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Restore a parallel checkpoint with topology validation.

        The checkpoint is read through the active DCP manager, validated against
        current parallel axes, optionally remapped for allowed non-FSDP degree
        changes, and then restored through the TorchRunner component loaders.

        Args:
            checkpoint: In-memory checkpoint mapping or DCP checkpoint path.
            *args: Forwarded to checkpoint reading and component loaders.
            **kwargs: Forwarded to checkpoint reading and component loaders.

        Raises:
            ValueError: saved topology is incompatible with the current run, or
                FSDP topology metadata is missing/changed.

        **Side effects:** restores model/optimizer/scheduler/runner state and
        updates `config.checkpoint` for path inputs.

        !!! danger "Do not"
            - Suppress topology validation for FSDP restores; shard metadata is
              part of the checkpoint contract.
            - Attempt degree-change restore with multiple local model parts.
        """
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

        super().load_checkpoint(ckpt, *args, _restore_source=checkpoint, **kwargs)
        if isinstance(checkpoint, (str, bytes, os.PathLike)):
            self.config.checkpoint = os.fsdecode(checkpoint)

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
        try:
            drained = super().close(timeout=timeout)
        except Exception:
            self._reset_model_parallel_groups()
            self._parallel_groups_initialized = False
            raise
        if not drained:
            return False
        self._reset_model_parallel_groups()
        self._parallel_groups_initialized = False
        return True

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
