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

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

import chanfig


class CompileConfig(chanfig.Config):
    r"""
    `torch.compile` and graph-compiler options.

    Attributes:
        enabled: Enable runner-selected model compilation points.
        backend, mode, fullgraph, dynamic, options: Forwarded to `torch.compile`.
        optimize_ddp: Optional `torch._dynamo.config.optimize_ddp` value. Defaults
            to `"ddp_optimizer"` when model compile is enabled.
        precompile_artifact_dir: Optional GraphRunner compiler-cache directory.
        memory_policy: Optional graph-memory policy label for experimental graph paths.
    """

    enabled: bool = False
    backend: Optional[str] = None
    mode: Optional[str] = None
    fullgraph: Optional[bool] = None
    dynamic: Optional[bool] = None
    options: Optional[Mapping[str, Any]] = None
    optimize_ddp: Optional[str] = "ddp_optimizer"
    precompile_artifact_dir: Optional[str] = None
    memory_policy: Optional[str] = None


class FsdpConfig(chanfig.Config):
    r"""
    FSDP2 wrapping options consumed by `ParallelRunner`.

    Attributes:
        enabled: Enable FSDP2 wrapping.
        mesh: Optional explicit FSDP mesh. Usually derived from `parallel.axes`.
        module_classes: Optional class-name or FQN list to shard before the root module.
        reshard_after_forward: Optional FSDP2 reshard policy for matched modules and root.
            Accepts booleans plus `"always"`, `"never"`, and `"default"`.
        root_reshard_after_forward: Optional root-only override.
        shard_placement_fn: Optional FSDP2 shard placement callable.
        mixed_precision_policy: Optional FSDP2 mixed precision policy.
        offload_policy: Optional FSDP2 CPU offload policy.
        ignored_params: Parameters excluded from FSDP2 wrapping.
    """

    enabled: bool = False
    mesh: Optional[Any] = None
    module_classes: Optional[Sequence[str]] = None
    reshard_after_forward: Union[bool, int, str, None] = None
    root_reshard_after_forward: Union[bool, int, str, None] = None
    shard_placement_fn: Optional[Any] = None
    mixed_precision_policy: Union[Any, Mapping[str, Any], None] = None
    offload_policy: Union[Any, Mapping[str, Any], None] = None
    ignored_params: Optional[Sequence[Any]] = None


class ActivationCheckpointConfig(chanfig.Config):
    r"""
    Activation checkpoint wrapping policy for `ParallelRunner`.

    `module_classes` entries match either the class name (for example
    `"TransformerBlock"`) or fully qualified class name. DanLing requires an
    explicit boundary list so users do not accidentally checkpoint arbitrary
    containers.
    """

    enabled: bool = False
    module_classes: Optional[Sequence[str]] = None
    checkpoint_impl: str = "no_reentrant"


class ParallelAxesConfig(chanfig.Config):
    r"""
    Parallelism degrees used to build the device mesh.

    Set one axis, commonly `shard`, to `-1` to auto-fill it from `WORLD_SIZE`
    and the other configured axes.
    """

    replicate: int = 1
    shard: int = 1
    context: int = 1
    pipeline: int = 1
    tensor: int = 1
    expert: int = 1
    expert_tensor: int = 1


class ParallelConfig(chanfig.Config):
    r"""
    Pipeline, tensor, context, expert, and FSDP mesh configuration.

    Attributes:
        axes: Parallelism degrees for replication, sharding, context, pipeline,
            tensor, expert, and expert-tensor dimensions.
        pipeline_schedule: Schedule class name resolved by
            `torch.distributed.pipelining.schedules.get_schedule_class`.
        pipeline_microbatch_size: Local microbatch size used to infer schedule
            microbatch count as `dataloader.batch_size // pipeline_microbatch_size`.
        pipeline_microbatches: Explicit schedule microbatch count.
        pipeline_partitions: Optional module FQNs for simple pipeline extraction.
        loss_parallel: Optional loss-parallel override. `None` enables loss
            parallel automatically when tensor parallelism is active.
        use_device_mesh: Use PyTorch `DeviceMesh` APIs when available.
        mesh_device_type: Optional mesh device type override.
        allow_degree_change: Allow loading checkpoints with a different topology.
    """

    axes: ParallelAxesConfig
    pipeline_schedule: str = "1F1B"
    pipeline_microbatch_size: int = 1
    pipeline_microbatches: Optional[int] = None
    pipeline_partitions: Optional[Sequence[Sequence[str]]] = None
    loss_parallel: Optional[bool] = None
    use_device_mesh: bool = True
    mesh_device_type: Optional[str] = None
    allow_degree_change: bool = False

    def __post_init__(self, *args, **kwargs) -> None:
        super().__post_init__(*args, **kwargs)
        if "axes" not in self:
            self.axes = ParallelAxesConfig()
        elif not isinstance(self.axes, ParallelAxesConfig):
            self.axes = ParallelAxesConfig(self.axes)


class DataloaderCheckpointConfig(chanfig.Config):
    r"""
    Per-replica dataloader checkpoint settings.
    """

    enabled: bool = False
    replica_id: Optional[str] = None
    prefix: str = "dataloader-replica"


class CheckpointConfig(chanfig.Config):
    r"""
    Checkpoint write policy and backend settings.

    Attributes:
        enabled: Persist checkpoints when `True`; loading remains available when disabled.
        dir: Checkpoint directory, relative to `workspace.dir` unless absolute.
        backend: `"auto"`, `"file"`, or `"dcp"`. Runners resolve `"auto"` by world size.
        wait_timeout_seconds: Timeout while draining async checkpoint writes.
        interval: Save cadence for latest, best, and history checkpoints.
        keep_latest_k: Number of framework-generated history checkpoints to retain.
        async_mode: `"disabled"`, `"async"`, or `"async_with_pinned_mem"`.
        dedicated_async_process_group: Use a dedicated process group for async DCP I/O.
        async_process_group_backend: Backend for that process group.
        dataloader_checkpoint: Per-replica dataloader checkpoint policy.
        fail_on_error: Raise deferred async checkpoint errors when `True`.
        export_dtype: Optional dtype cast for model-only checkpoint export.
    """

    enabled: bool = True
    dir: str = "checkpoints"
    backend: str = "auto"
    wait_timeout_seconds: Optional[float] = None
    interval: Optional[int] = None
    keep_latest_k: int = 0
    async_mode: str = "async"
    dedicated_async_process_group: bool = True
    async_process_group_backend: str = "gloo"
    dataloader_checkpoint: DataloaderCheckpointConfig
    fail_on_error: bool = True
    export_dtype: Optional[str] = None

    def __post_init__(self, *args, **kwargs) -> None:
        super().__post_init__(*args, **kwargs)
        if "dataloader_checkpoint" not in self:
            self.dataloader_checkpoint = DataloaderCheckpointConfig()
        elif not isinstance(self.dataloader_checkpoint, DataloaderCheckpointConfig):
            self.dataloader_checkpoint = DataloaderCheckpointConfig(self.dataloader_checkpoint)


class ScoreConfig(chanfig.Config):
    r"""
    Model-selection and early-stopping settings.

    Attributes:
        split: Dataset split to use for model selection. If unset, the runner
            infers once (`val` -> `validate` -> first available) and reuses it
            unless that split disappears from results.
        metric: Metric key to use for model selection. Defaults to `"loss"`.
        patience: Early-stop patience in epoch mode. Defaults to infinity.
    """

    split: Optional[str] = None
    metric: str = "loss"
    patience: Union[int, float] = float("inf")


class WorkspaceConfig(chanfig.Config):
    r"""
    Experiment workspace layout settings.
    """

    root: str = "experiments"
    lineage: str = "lin"
    experiment: str = "exp"
    dir: Optional[str] = None


class LoggingConfig(chanfig.Config):
    r"""
    File logging settings.
    """

    enabled: bool = True
    interval: Optional[int] = None
    file: Optional[str] = None


class TensorboardConfig(chanfig.Config):
    r"""
    TensorBoard `SummaryWriter` settings.
    """

    enabled: bool = False
    log_dir: Optional[str]
    comment: Optional[str]
    purge_step: Optional[int]
    max_queue: Optional[int]
    flush_secs: Optional[int]
    filename_suffix: Optional[str]


class DistributedConfig(chanfig.Config):
    r"""
    Torch distributed process-group settings.
    """

    backend: Optional[str] = None
    init_method: Optional[str] = None
    init_timeout_seconds: Optional[int] = None
    train_timeout_seconds: Optional[int] = None


class DdpConfig(chanfig.Config):
    r"""
    ``torch.nn.parallel.DistributedDataParallel`` wrapping settings.
    """

    find_unused_parameters: bool = False


class GcConfig(chanfig.Config):
    r"""
    Runner-managed Python garbage-collection settings.
    """

    interval: Optional[int] = None
    generation: int = 1
    disable_automatic: bool = True


class ProfilingConfig(chanfig.Config):
    r"""
    Bounded-step `torch.profiler` tracing settings.
    """

    enabled: bool = False
    activities: Union[Sequence[str], str, None] = None
    wait: int = 1
    warmup: int = 1
    active: int = 3
    repeat: Optional[int] = None
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_flops: bool = False
    with_modules: bool = False
    acc_events: bool = False
    use_cuda: Optional[bool] = None
    post_processing_timeout_seconds: Optional[float] = None
    trace_dir: str = "profiles"
    operator_table_rows: int = 100


class HeartbeatConfig(chanfig.Config):
    r"""
    Machine-readable per-rank heartbeat/progress-file settings.
    """

    enabled: bool = False
    interval_seconds: float = 60.0
    dir: str = "heartbeats"


class WandbConfig(chanfig.Config):
    r"""
    Weights & Biases run settings forwarded to `wandb.init`.
    """

    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None
    id: Optional[str] = None
    group: Optional[str] = None
    name: Optional[str] = None
    notes: Optional[str] = None
    job_type: Optional[str] = None
    tags: Union[Sequence[str], str, None] = None
    dir: Optional[str] = None
    mode: Optional[str] = None
    resume: Any = None
    save_code: Optional[bool] = None
    sync_tensorboard: Optional[bool] = None


class OptimizerConfig(chanfig.Config):
    r"""
    Optimizer registry options.

    Attributes:
        type: Optimizer registry key, for example `"adamw"` or `"sgd"`.
            When unset, the runner does not auto-build an optimizer.
        lr, weight_decay, betas, eps, momentum: Common optimizer kwargs.
        param_groups: Optional regex-based parameter groups. Each entry requires
            `pattern`, matched against `TorchRunner.iter_optimizer_named_parameters()`
            with `re.search` semantics, and may provide optimizer group options.
            `lr_multiplier`, `weight_decay_multiplier`, `beta1`, and `beta2`
            derive group values from top-level optimizer settings.
    """

    type: Optional[str]
    lr: Optional[float]
    weight_decay: Optional[float]
    betas: Optional[Sequence[float]]
    eps: Optional[float]
    momentum: Optional[float]
    dampening: Optional[float]
    nesterov: Optional[bool]
    amsgrad: Optional[bool]
    foreach: Optional[bool]
    maximize: Optional[bool]
    capturable: Optional[bool]
    differentiable: Optional[bool]
    fused: Optional[bool]
    param_groups: Optional[Sequence[Mapping[str, Any]]] = None


class SchedulerConfig(chanfig.Config):
    r"""
    Learning-rate scheduler registry options.

    Attributes:
        type: Scheduler registry key, for example `"cosine"`, `"linear"`,
            `"step"`, or `"reduce_on_plateau"`. When unset, the runner does not
            auto-build a scheduler.
        interval: Scheduler advancement policy. Supported values are `"step"`
            and `"epoch"`/`"validation"`. Non-metric schedulers default to
            `"step"`. Metric schedulers such as `ReduceLROnPlateau` default to
            `"epoch"` and advance after the aggregated round result is available.
        monitor: Optional dotted metric selector for metric schedulers, such as
            `"val.loss"`. When unset, the runner prefers `score.split`/`score.metric`
            and otherwise resolves `score.metric` from the aggregated result.
        total_steps, warmup_steps, cooldown_steps, final_lr_ratio, final_lr:
            Common DanLing `LRScheduler` kwargs.
        step_size, milestones, gamma, T_max, eta_min, patience, factor:
            Common PyTorch scheduler kwargs.
    """

    type: Optional[str]
    interval: Optional[str] = None
    monitor: Optional[str] = None
    total_steps: Optional[int]
    warmup_steps: Optional[int]
    cooldown_steps: Optional[int]
    final_lr_ratio: Optional[float]
    final_lr: Optional[float]
    min_lr: Optional[float]
    step_size: Optional[int]
    milestones: Optional[Sequence[int]]
    gamma: Optional[float]
    T_max: Optional[int]
    T_0: Optional[int]
    T_mult: Optional[int]
    eta_min: Optional[float]
    max_lr: Optional[Union[float, Sequence[float]]]
    base_lr: Optional[Union[float, Sequence[float]]]
    patience: Optional[int]
    factor: Optional[float]
    mode: Optional[str]


class Fp8Config(chanfig.Config):
    r"""
    FP8 autocast recipe settings.
    """

    enabled: Optional[bool] = None
    recipe: Optional[Any] = None
    recipe_cls: Optional[Union[str, Any]] = None
    recipe_kwargs: Optional[Mapping[str, Any]] = None
    group: Optional[Any] = None


class FaultToleranceConfig(chanfig.Config):
    r"""
    TorchFT-managed fault-tolerance settings.
    """

    enabled: bool = False
    process_group: str = "gloo"
    process_group_timeout_seconds: float = 10.0
    replica_id: int = 0
    group_size: int = 1
    min_replica_size: int = 1


class DataloaderConfig(chanfig.Config):
    r"""
    Default and split-specific `StatefulDataLoader` options.

    Attributes:
        batch_size: Local dataloader batch size.
        shuffle: Optional shuffle override. When unset, train splits shuffle
            and non-train splits do not.
        sampler, batch_sampler, collate_fn: Optional dataloader construction hooks.
        drop_last: Optional drop-last override. When unset, train splits drop
            incomplete batches and non-train splits keep them.
        num_workers, persistent_workers, prefetch_factor, pin_memory, pin_memory_device:
            Standard PyTorch dataloader kwargs.
        in_order: PyTorch dataloader ordering flag.
        snapshot_every_n_steps: StatefulDataLoader snapshot cadence.
        <split>: Split-specific overrides merged on top of default dataloader
            kwargs, for example `dataloader.train.shuffle=False`.
    """

    batch_size: Optional[int]
    shuffle: Optional[bool]
    sampler: Optional[Any]
    batch_sampler: Optional[Any]
    collate_fn: Optional[Any]
    drop_last: Optional[bool]
    num_workers: int
    pin_memory: bool
    pin_memory_device: str
    timeout: float
    worker_init_fn: Optional[Any]
    multiprocessing_context: Optional[Any]
    generator: Optional[Any]
    prefetch_factor: Optional[int]
    persistent_workers: bool
    in_order: bool
    snapshot_every_n_steps: Optional[int]


class PerformanceConfig(chanfig.Config):
    r"""
    Optional model-throughput accounting.

    FLOP fields are intentionally opt-in: generic runners cannot infer a
    meaningful model-specific FLOP count. When supplied, loop telemetry emits
    TFLOP/s and utilization fields alongside throughput.
    """

    model_flops_per_sample: Optional[float] = None
    model_flops_per_token: Optional[float] = None
    hardware_flops_per_sample: Optional[float] = None
    hardware_flops_per_token: Optional[float] = None
    peak_flops: Optional[float] = None


def normalize_stack_name(stack: object) -> str:
    normalized = str(stack or "auto").strip().lower().replace("-", "_")
    aliases = {
        "auto": "ddp",
        "ddp": "ddp",
        "torch": "ddp",
        "graph": "graph",
        "graph_runner": "graph",
        "deepspeed": "deepspeed",
        "ds": "deepspeed",
        "parallel": "parallel",
    }
    return aliases.get(normalized, normalized)


NON_SEMANTIC_CONFIG_KEYS: tuple[str, ...] = (
    "score",
    "name",
    "logging",
    "tensorboard",
    "wandb",
    "ft",
    "workspace",
    "checkpoint",
    "resume",
    "pretrained",
    "dist",
    "gc",
    "profiling",
    "heartbeat",
)

DEFAULT_FILTERED_CONFIG_SECTIONS: tuple[tuple[str, chanfig.Config], ...] = (
    ("optim", OptimizerConfig()),
    ("sched", SchedulerConfig()),
    ("fp8", Fp8Config()),
    ("dataloader", DataloaderConfig()),
    ("compile", CompileConfig()),
    ("performance", PerformanceConfig()),
    ("activation_checkpoint", ActivationCheckpointConfig()),
)


class RunnerConfig(chanfig.Config):  # pylint: disable=too-many-instance-attributes
    r"""
    Top-level configuration for DanLing runners.

    `RunnerConfig` owns runner lifecycle settings, restore sources, and typed
    subsystem sections. Detailed subsystem field semantics live on the matching
    subconfig class, for example `OptimizerConfig`, `SchedulerConfig`,
    `ScoreConfig`, `CheckpointConfig`, `WorkspaceConfig`, `DataloaderConfig`,
    `FsdpConfig`, and `ParallelConfig`.

    `RunnerConfig` inherits from [`Config`][chanfig.Config] and provides
    attribute-style access to nested values:

    ```python
    config = RunnerConfig()
    config.workspace.experiment = "resnet50"
    config.dataloader.batch_size = 32
    config["optim"] = {"type": "adamw", "lr": 1e-3}
    ```

    Command-line integration is built in:

    ```python
    config = MyConfig()
    config.parse()  # Parse CLI args, e.g., --epochs 20 --optim.lr 0.01
    ```

    Core attributes:
        stack: Runner stack selector used by `danling.runners.Runner`. Supported
            values include `"auto"`, `"ddp"`/`"torch"`, `"graph"`,
            `"deepspeed"`/`"ds"`, and `"parallel"`.
        seed, deterministic: Reproducibility controls.
        steps, epochs: Mutually exclusive training boundaries.
        accum_steps: Number of micro-batches per optimizer step.
        train_splits, evaluate_splits: Optional split selection overrides.
        precision: Optional autocast precision.
        max_grad_value, max_grad_norm, skip_nonfinite_grad: Gradient safety controls.
        checkpoint, resume, pretrained: Restore sources. Source priority is
            `checkpoint` > `resume` > `pretrained`.
        deepspeed: Optional raw DeepSpeed config mapping.

    Nested sections:
        `optim`, `sched`, `score`, `workspace`, `logging`, `tensorboard`,
        `wandb`, `ft`, `compile`, `dist`, `gc`, `profiling`, `heartbeat`,
        `ckpt`, `dataloader`, `performance`, `activation_checkpoint`,
        `fsdp`, and `parallel`.

    Examples:
        Basic usage:
        ```python
        # Create a config
        config = RunnerConfig()
        config.workspace.experiment = "resnet18"
        config.dataloader.batch_size = 32
        config["optim"] = {"type": "adamw", "lr": 1e-3}
        config.epochs = 10

        # Use in a runner
        runner = Runner(config)
        ```

        Custom config class with typed attributes:
        ```python
        class TrainingConfig(RunnerConfig):
            # Type annotations provide auto-completion and validation
            model: str = "resnet18"
            epochs: int = 100
            precision: str = "bf16"

            def __init__(self):
                super().__init__()
                self.dataloader.batch_size = 32
                self["optim"] = {"type": "adamw", "lr": 1e-3}

            def post(self):
                # Called after parsing CLI args
                super().post()
                # Create derived settings
                lr = self.get("optim.lr")
                self.workspace.experiment = f"{self.model}_bs{self.dataloader.batch_size}_lr{lr}"
        ```

        Command-line integration:
        ```bash
        # Override config settings via CLI
        python train.py --epochs 50 --dataloader.batch_size 64 --optim.lr 0.0005
        ```

    Note:
        Always store all parameters needed to reproduce a run in the RunnerConfig.
        The RunnerConfig is automatically saved with checkpoints, enabling exact resumption.

    See Also:
        - [`Runner`][danling.runners.Runner]: Main runner class that uses this config.
        - [`chanfig.Config`](https://github.com/ultmaster/chanfig): Base config implementation.
    """

    stack: str = "auto"
    name: Optional[str] = None

    seed: Optional[int] = 1016
    deterministic: bool = False

    steps: Optional[int] = None
    epochs: Optional[int] = None
    accum_steps: int = 1
    train_splits: Union[Sequence[str], str, None] = None
    evaluate_splits: Union[Sequence[str], str, None] = None
    precision: Optional[str] = None
    max_grad_value: Optional[float] = None
    max_grad_norm: Optional[float] = None
    skip_nonfinite_grad: bool = False

    checkpoint: Optional[str] = None
    resume: bool = False
    pretrained: Optional[str] = None

    optim: Optional[OptimizerConfig]
    sched: Optional[SchedulerConfig]
    fp8: Fp8Config = Fp8Config()
    deepspeed: Optional[Mapping[str, Any]] = None

    score: ScoreConfig = ScoreConfig()
    workspace: WorkspaceConfig = WorkspaceConfig()
    logging: LoggingConfig = LoggingConfig()
    tensorboard: TensorboardConfig = TensorboardConfig()
    wandb: WandbConfig = WandbConfig()
    ft: FaultToleranceConfig = FaultToleranceConfig()

    compile: CompileConfig = CompileConfig()
    dist: DistributedConfig = DistributedConfig()
    ddp: DdpConfig = DdpConfig()
    gc: GcConfig = GcConfig()
    profiling: ProfilingConfig = ProfilingConfig()
    heartbeat: HeartbeatConfig = HeartbeatConfig()
    ckpt: CheckpointConfig = CheckpointConfig()
    dataloader: DataloaderConfig = DataloaderConfig()
    performance: PerformanceConfig = PerformanceConfig()
    activation_checkpoint: ActivationCheckpointConfig = ActivationCheckpointConfig()
    fsdp: FsdpConfig = FsdpConfig()
    parallel: ParallelConfig = ParallelConfig()

    def __post_init__(self, *args, **kwargs) -> None:
        super().__post_init__(*args, **kwargs)
        self.validate()

    def post(self) -> None:
        super().post()
        self.validate()

    def validate(self) -> None:
        if self.steps is not None and self.epochs is not None:
            raise ValueError("`steps` and `epochs` are mutually exclusive; set only one training boundary")

    @staticmethod
    def _semantic_section(section: Any, defaults: chanfig.Config) -> chanfig.NestedDict:
        if not isinstance(section, Mapping):
            return chanfig.NestedDict()
        return defaults.difference(section)

    def canonical(self) -> chanfig.NestedDict:
        canonical = chanfig.NestedDict(self.dict())
        stack = normalize_stack_name(canonical.get("stack", "auto"))
        canonical["stack"] = stack
        for key in NON_SEMANTIC_CONFIG_KEYS:
            canonical.pop(key, None)

        ckpt_backend = canonical.get("ckpt.backend")
        canonical.pop("ckpt", None)
        if ckpt_backend is not None:
            ckpt_backend = str(ckpt_backend).strip().lower()
            if ckpt_backend != "auto":
                canonical["ckpt"] = chanfig.NestedDict({"backend": ckpt_backend})

        for key, defaults in DEFAULT_FILTERED_CONFIG_SECTIONS:
            semantic_section = self._semantic_section(canonical.get(key), defaults)
            if semantic_section:
                canonical[key] = semantic_section
            else:
                canonical.pop(key, None)

        if stack != "parallel":
            canonical.pop("fsdp", None)
            canonical.pop("parallel", None)
        return canonical

    def __hash__(self) -> int:
        digest = hashlib.sha1(self.canonical().yamls().encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)
