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
    enabled: bool = False
    mesh: Optional[Any] = None
    reshard_after_forward: Union[bool, int, None] = None
    shard_placement_fn: Optional[Any] = None
    mixed_precision_policy: Union[Any, Mapping[str, Any], None] = None
    offload_policy: Union[Any, Mapping[str, Any], None] = None
    ignored_params: Optional[Sequence[Any]] = None


class ParallelAxesConfig(chanfig.Config):
    replicate: int = 1
    shard: int = 1
    context: int = 1
    pipeline: int = 1
    tensor: int = 1
    expert: int = 1
    expert_tensor: int = 1


class ParallelConfig(chanfig.Config):
    axes: ParallelAxesConfig
    pipeline_schedule: str = "1F1B"
    pipeline_microbatch_size: int = 1
    pipeline_microbatches: Optional[int] = None
    pipeline_partitions: Optional[Sequence[Sequence[str]]] = None
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
    enabled: bool = False
    replica_id: Optional[str] = None
    prefix: str = "dataloader-replica"


class CheckpointConfig(chanfig.Config):
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
    fail_on_error: bool = False
    export_dtype: Optional[str] = None

    def __post_init__(self, *args, **kwargs) -> None:
        super().__post_init__(*args, **kwargs)
        if "dataloader_checkpoint" not in self:
            self.dataloader_checkpoint = DataloaderCheckpointConfig()
        elif not isinstance(self.dataloader_checkpoint, DataloaderCheckpointConfig):
            self.dataloader_checkpoint = DataloaderCheckpointConfig(self.dataloader_checkpoint)


class ScoreConfig(chanfig.Config):
    split: Optional[str] = None
    metric: str = "loss"
    patience: Union[int, float] = float("inf")


class WorkspaceConfig(chanfig.Config):
    root: str = "experiments"
    lineage: str = "lin"
    experiment: str = "exp"
    dir: Optional[str] = None


class LoggingConfig(chanfig.Config):
    enabled: bool = True
    interval: Optional[int] = None
    file: Optional[str] = None


class TensorboardConfig(chanfig.Config):
    enabled: bool = False
    log_dir: Optional[str]
    comment: Optional[str]
    purge_step: Optional[int]
    max_queue: Optional[int]
    flush_secs: Optional[int]
    filename_suffix: Optional[str]


class DistributedConfig(chanfig.Config):
    backend: Optional[str] = None
    init_method: Optional[str] = None
    init_timeout_seconds: Optional[int] = None
    train_timeout_seconds: Optional[int] = None


class GcConfig(chanfig.Config):
    interval: Optional[int] = None
    generation: int = 1
    disable_automatic: bool = True


class ProfilingConfig(chanfig.Config):
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


class HeartbeatConfig(chanfig.Config):
    enabled: bool = False
    interval_seconds: float = 60.0
    dir: str = "heartbeats"


class WandbConfig(chanfig.Config):
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
    enabled: Optional[bool] = None
    recipe: Optional[Any] = None
    recipe_cls: Optional[Union[str, Any]] = None
    recipe_kwargs: Optional[Mapping[str, Any]] = None
    group: Optional[Any] = None


class FaultToleranceConfig(chanfig.Config):
    enabled: bool = False
    process_group: str = "gloo"
    process_group_timeout_seconds: float = 10.0
    replica_id: int = 0
    group_size: int = 1
    min_replica_size: int = 1


class DataloaderConfig(chanfig.Config):
    """Typed surface for dataloader kwargs consumed by `TorchRunner.build_dataloaders`.

    Fields intentionally do not define defaults. Missing keys must remain absent so
    runner-owned defaults such as train-only shuffling/drop-last keep working.
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


NON_SEMANTIC_CKPT_KEYS: tuple[str, ...] = (
    "enabled",
    "dir",
    "wait_timeout_seconds",
    "interval",
    "keep_latest_k",
    "async_mode",
    "dedicated_async_process_group",
    "async_process_group_backend",
    "dataloader_checkpoint",
    "fail_on_error",
    "export_dtype",
)


class RunnerConfig(chanfig.Config):  # pylint: disable=too-many-instance-attributes
    r"""
    Configuration class for managing and persisting all states of a DanLing Runner.

    The RunnerConfig class provides a hierarchical configuration system that handles:

    1. **Parameter management**: Hyperparameters, model settings, training options
    2. **Experiment tracking**: IDs, names, and other metadata for runs and experiments
    3. **Serialization**: Save/load configurations from files or command line
    4. **Reproducibility**: Tracking seeds and settings for reproducible runs

    RunnerConfig inherits from [`Config`][chanfig.Config] and provides attribute-style access to nested values:

    ```python
    config = RunnerConfig()

    # Attribute-style access (recommended)
    config.optim.lr = 1e-3
    config.network.type = "resnet50"

    # Dictionary-style access (alternative)
    config["optim"]["lr"] = 1e-3
    config["network"]["type"] = "resnet50"
    ```

    RunnerConfig objects support three types of hierarchical attribute access patterns:

    1. **Direct assignment** for simple values:
       ```python
       config.epochs = 10
       ```

    2. **Auto-created nested objects** for hierarchical settings:
       ```python
       # Auto-creates the nested objects
       config.optim.lr = 0.01
       config.optim.weight_decay = 1e-4
       ```

    3. **Class-level annotations** for typed properties with defaults:
       ```python
       class MyConfig(RunnerConfig):
           epochs: int = 10
           learning_rate: float = 0.001
       ```

    Command-line integration is built-in. You can define a configuration and
    then override values via command line arguments:

    ```python
    config = MyConfig()
    config.parse()  # Parse CLI args, e.g., --epochs 20 --optim.lr 0.01
    ```

    Attributes: General:
        stack (str): Runner stack selector used by `danling.runners.Runner`.
            Supported values: `"auto"`, `"ddp"`/`"torch"`, `"graph"`,
            `"deepspeed"`/`"ds"`, `"parallel"`.
            Defaults to `"auto"` (resolved to `"ddp"` at runtime).

    Attributes: Reproducibility:
        seed (int): Random seed for reproducibility. If not set, a random value is generated.
        deterministic (bool): Whether to enforce deterministic operations in PyTorch.
            Defaults to `False` for better performance. Set to `True` for exact reproducibility.

    Attributes: Progress:
        steps (int | None): Final global step target for training.
            In step mode, training stops when `global_step >= steps`.
        epochs (int | None): Final epoch index boundary for training.
            In epoch mode, training iterates epochs until `epoch == epochs`.
        accum_steps (int): Number of micro-batches per optimizer step.
            Defaults to `1`.

    Attributes: Model Evaluation:
        score.split (str): Dataset split to use for model selection. Defaults to None.
            If unset, runner infers once (`val` -> `validate` -> first available) and reuses it
            unless that split disappears from results.
        score.metric (str): Metric key to use for model selection. Defaults to "loss".
        score.patience (int | float): Early-stop patience in epoch mode.
            Defaults to infinity.
        sched.interval (str): Scheduler advancement policy.
            Supported values: `"step"` and `"epoch"`/`"validation"`.
            Non-metric schedulers default to `"step"`. Metric schedulers such as
            `ReduceLROnPlateau` default to `"epoch"` and advance after the aggregated
            round result is available.
        sched.monitor (str): Optional metric selector for metric schedulers.
            Supports dotted paths such as `"val.loss"`.
            When unset, the runner prefers `score.split`/`score.metric` when available and
            otherwise resolves `score.metric` from the aggregated result.

    Attributes: Optimization:
        optim.type (str | None): Optimizer registry key, for example `"adamw"` or `"sgd"`.
            When unset, the runner does not auto-build an optimizer.
        optim.lr / weight_decay / betas / eps / momentum: Common optimizer
            kwargs forwarded to the optimizer registry when present.
        optim.param_groups (list[dict] | None): Optional regex-based optimizer
            parameter groups. Each entry requires `pattern`, matched against
            `TorchRunner.iter_optimizer_named_parameters()` with `re.search`
            semantics, and may provide optimizer group options directly. Anchor
            patterns with `^`/`$` when a full FQN position matters.
            `lr_multiplier`,
            `weight_decay_multiplier`, `beta1`, and `beta2` derive group values
            from top-level `optim.lr`, `optim.weight_decay`, and `optim.betas`.
            Unmatched parameters keep the optimizer-level defaults.
        sched.type (str | None): Scheduler registry key, for example `"cosine"`,
            `"linear"`, `"step"`, or `"reduce_on_plateau"`. When unset, the runner
            does not auto-build a scheduler.
        sched.total_steps / warmup_steps / cooldown_steps / final_lr_ratio / final_lr:
            Common DanLing `LRScheduler` kwargs forwarded when present.
        sched.step_size / milestones / gamma / T_max / eta_min / patience / factor:
            Common PyTorch scheduler kwargs forwarded when present.

    Attributes: I/O:
        workspace.root (str): Root directory for experiments. Defaults to `"experiments"`.
        checkpoint (str | None): Optional full-state checkpoint source for resume workflows.
            This is a path-like identifier consumed by runner `load_checkpoint(...)`.
        resume (bool): Auto-resume from the backend-native latest checkpoint source when `True`.
        pretrained (str | None): Optional model-only checkpoint source for finetune workflows.
            This is a path-like identifier consumed by runner `load_pretrained(...)`.
            Source priority is `checkpoint` > `resume` > `pretrained`.
        workspace.lineage (str): Top-level lineage namespace.
            Defaults to `"lin"` when unset.
            `RunnerWorkspace.dir` appends code identity (`-<git_hash>`) when available.
        workspace.experiment (str): Experiment namespace. Defaults to `"exp"`.
        ckpt.dir (str): Checkpoint directory. Relative paths are resolved under `workspace.dir`.
            Defaults to `"checkpoints"`.
        ckpt.async_mode (str): Checkpoint async behavior. Defaults to `"async"`.
            Supported values: `"disabled"`, `"async"`, `"async_with_pinned_mem"`.
        ckpt.dedicated_async_process_group (bool): Use a dedicated process group for async DCP
            checkpoint I/O to reduce interference with training collectives. Defaults to `True`.
        ckpt.async_process_group_backend (str): Backend for the dedicated async checkpoint process
            group. Defaults to `"gloo"`.
        ckpt.backend (str): Checkpoint backend selected at runtime by the runner
            (`"dcp"` for distributed runs, `"file"` otherwise when set to `"auto"`).
        ckpt.wait_timeout_seconds (float): Timeout in seconds when draining async checkpoint writes
            during runner shutdown (`None` waits indefinitely).
        parallel.axes.replicate (int): Data-replication degree for DDP/HSDP-style replication.
            Defaults to `1`.
        parallel.axes.shard (int): Data-sharding degree for FSDP-style sharding.
            Defaults to `1`. Set one parallel axis, commonly `shard`, to `-1`
            to auto-fill it from `WORLD_SIZE` and the other configured axes.
        parallel.axes.context (int): Context/sequence parallel degree. Defaults to `1`.
        parallel.axes.pipeline (int): Pipeline-parallel degree. Defaults to `1`.
        parallel.axes.tensor (int): Tensor-parallel degree. Defaults to `1`.
        parallel.axes.expert (int): Expert-parallel degree for MoE models. Defaults to `1`.
        parallel.axes.expert_tensor (int): Expert tensor-parallel degree for MoE models. Defaults to `1`.
        parallel.pipeline_schedule (str): Pipeline schedule class name resolved by
            `torch.distributed.pipelining.schedules.get_schedule_class`.
            Defaults to `"1F1B"`.
        parallel.pipeline_microbatch_size (int): Local microbatch size used to infer
            schedule microbatch count as `dataloader.batch_size // pipeline_microbatch_size`.
            Defaults to `1`.
        parallel.pipeline_microbatches (int): Explicit schedule microbatch count.
            When set, overrides `pipeline_microbatch_size`-based inference.
        parallel.pipeline_partitions (list[list[str]] | None): Optional
            module FQNs for simple pipeline stage extraction. The outer list
            length is the total pipeline stage count and must be divisible by
            `parallel.axes.pipeline`; complex partitioning should use
            `model.build_pipeline_model_part(...)` or override
            `ParallelRunner.build_pipeline_model_part` /
            `ParallelRunner.build_pipeline_model_parts`.
        logging.enabled (bool): Whether to enable file logging. Defaults to `True`.
            Logging is initialized on the main process only.
        logging.interval (int): Iterations between log outputs. If None, auto-calculated.
        logging.file (str | None): Optional log file path.
            Defaults to `workspace.dir/logs/{timestamp}.log`.
        tensorboard.enabled (bool): Whether to use TensorBoard for visualization. Defaults to `False`.
        tensorboard.log_dir (str | None): Optional TensorBoard log directory.
            Defaults to `workspace.dir/tensorboard/{timestamp}`.
        tensorboard.comment / purge_step / max_queue / flush_secs / filename_suffix:
            Optional `torch.utils.tensorboard.SummaryWriter` kwargs.
        wandb.enabled (bool): Whether to enable Weights & Biases scalar logging. Defaults to `False`.
        wandb.project (str | None): Optional W&B project name. Defaults to `lineage`.
        wandb.entity (str | None): Optional W&B entity/team override.
        wandb.id (str | None): Optional stable W&B run id.
        wandb.group (str | None): Optional W&B group name. Defaults to `experiment`.
        wandb.name (str | None): Optional W&B display name. Defaults to stable runner `id`.
        wandb.notes (str | None): Optional W&B run notes.
        wandb.job_type (str | None): Optional W&B job type.
        wandb.tags (list[str] | str | None): Optional W&B run tags.
        wandb.dir (str | None): Optional local W&B run directory. Defaults to run directory.
        wandb.mode (str | None): Optional W&B mode such as `"online"` or `"offline"`.
        wandb.resume / save_code / sync_tensorboard: Optional common W&B init kwargs.
        ft.enabled (bool): Enable TorchFT-managed fault tolerance. Defaults to `False`.
        ft.process_group (str): TorchFT coordination backend. Supported values: `"gloo"` and `"nccl"`.
            Defaults to `"gloo"`.
        ft.process_group_timeout_seconds (float): TorchFT process-group timeout in seconds.
            Defaults to `10.0`.
        ft.replica_id (int): Replica-group identifier for this run. Defaults to `0`.
        ft.group_size (int): Number of replica groups participating in TorchFT. Defaults to `1`.
        ft.min_replica_size (int): Minimum healthy replicas required by TorchFT per step.
            Defaults to `1`.
        ckpt.interval (int): Interval between checkpoint save attempts for `latest`/`best`.
            The same cadence is used for history checkpoints.
            Uses epochs in epoch mode and global steps in step mode.
            If unset, runner defaults are used by mode.
        ckpt.keep_latest_k (int): Number of framework-generated history checkpoints to retain.
            `0` disables retention pruning.
        ckpt.enabled (bool): Whether to persist checkpoints. Set `False` to allow loading while disabling writes.
        ckpt.dataloader_checkpoint.enabled (bool): Enable per-replica dataloader checkpoints.
            Uses DCP and stores checkpoints under
            `ckpt.dataloader_checkpoint.prefix-{ckpt.dataloader_checkpoint.replica_id}`.
        ckpt.dataloader_checkpoint.replica_id (str | None): Replica identifier used for dataloader checkpoint directory.
            Defaults to `FT_REPLICA_ID` environment variable, then process rank.
        ckpt.dataloader_checkpoint.prefix (str): Prefix used for per-replica dataloader checkpoint directories.
            Defaults to `"dataloader-replica"`.
        ckpt.export_dtype (str): Optional dtype cast for model-only checkpoint export
            (`fp32`/`fp16`/`bf16`/`fp64` aliases supported).
        dataloader.batch_size (int | None): Local dataloader batch size passed to
            `StatefulDataLoader`.
        dataloader.shuffle (bool | None): Optional shuffle override. When unset, train
            splits shuffle and non-train splits do not.
        dataloader.sampler / batch_sampler / collate_fn: Optional DataLoader
            construction hooks forwarded to `StatefulDataLoader`.
        dataloader.drop_last (bool | None): Optional drop-last override. When unset,
            train splits drop incomplete batches and non-train splits keep them.
        dataloader.num_workers / persistent_workers / prefetch_factor / pin_memory:
            Standard PyTorch DataLoader kwargs forwarded to `StatefulDataLoader`.
        dataloader.in_order (bool): PyTorch DataLoader ordering flag.
        dataloader.snapshot_every_n_steps (int | None): StatefulDataLoader snapshot cadence.
        dataloader.<split> (dict): Split-specific overrides merged on top of default
            dataloader kwargs, for example `dataloader.train.shuffle=False`.
        fsdp.enabled (bool): Enable FSDP2 wrapping in `ParallelRunner`.
            The FSDP mesh is derived from `parallel.axes.replicate`,
            `parallel.axes.shard`, and later `parallel.axes.context`.
        fsdp.reshard_after_forward (bool | int | None): Optional FSDP2 reshard policy.
        fsdp.shard_placement_fn: Optional FSDP2 shard placement callable.
        fsdp.mixed_precision_policy: Optional FSDP2 mixed precision policy.
        fsdp.offload_policy: Optional FSDP2 CPU offload policy.
        fsdp.ignored_params: Optional parameters excluded from FSDP2 wrapping.
        compile.enabled (bool): Whether to enable `torch.compile` for runner-selected model compilation points.
        compile.backend (str): Optional backend passed to `torch.compile`.
        compile.fullgraph (bool): Optional `fullgraph` flag for `torch.compile`.
        compile.dynamic (bool): Optional `dynamic` flag for `torch.compile`.
        compile.mode (str): Optional mode passed to `torch.compile`.
        compile.options (dict): Optional options passed to `torch.compile`.
        compile.optimize_ddp (str | None): Optional `torch._dynamo.config.optimize_ddp` value.
            Defaults to `"ddp_optimizer"` when model compile is enabled.
        compile.precompile_artifact_dir (str | None): Optional directory for GraphRunner torch compiler
            cache artifacts. Current eager runners ignore this setting.
        compile.memory_policy (str | None): Optional graph-memory policy label for experimental graph paths.
            GraphRunner currently accepts `None`/`"default"`; activation remat/offload policies require a
            dedicated graph pass pipeline.
        dist.init_timeout_seconds (int | None): Optional distributed process-group timeout used during
            initialization and early startup.
        dist.train_timeout_seconds (int | None): Optional tighter distributed process-group timeout applied
            once after the first successful optimizer step.
        gc.interval (int | None): Optional periodic Python GC cadence.
            When unset, runner-managed GC pacing is disabled.
        gc.generation (int): Python GC generation passed to `gc.collect(...)` when pacing is enabled.
            Defaults to `1`.
        gc.disable_automatic (bool): Disable CPython automatic GC while runner-managed pacing is enabled.
            Defaults to `True`.
        profiling.enabled (bool): Enable bounded-step `torch.profiler` tracing. Defaults to `False`.
        profiling.activities (str | list[str] | None): Explicit profiler activities such as
            `"cpu"` or `["cpu", "cuda"]`. When unset, CPU is used and CUDA is added
            for CUDA runners.
        profiling.wait (int): Profiler schedule wait steps before warmup. Defaults to `1`.
        profiling.warmup (int): Profiler schedule warmup steps. Defaults to `1`.
        profiling.active (int): Profiler schedule active trace steps. Defaults to `3`.
        profiling.repeat (int | None): Optional profiler schedule repeat count.
        profiling.record_shapes (bool): Enable shape recording in traces. Defaults to `False`.
        profiling.profile_memory (bool): Enable profiler-side memory recording. Defaults to `False`.
        profiling.with_stack (bool): Include Python stack traces in profiler output. Defaults to `False`.
        profiling.with_flops (bool): Enable profiler FLOPs estimation when available. Defaults to `False`.
        profiling.with_modules / acc_events / use_cuda: Optional profiler kwargs.
        profiling.post_processing_timeout_seconds (float | None): Optional profiler
            post-processing timeout in seconds.
        profiling.trace_dir (str): Relative or absolute trace output directory. Defaults to `"profiles"`.
        heartbeat.enabled (bool): Enable a machine-readable per-rank heartbeat/progress file. Defaults to `False`.
        heartbeat.interval_seconds (float): Heartbeat write interval in seconds. Defaults to `60.0`.
        heartbeat.dir (str): Heartbeat directory. Relative paths are resolved under `workspace.dir`.
            Defaults to `"heartbeats"`.
    Examples:
        Basic usage:
        ```python
        # Create a config
        config = RunnerConfig()
        config.network.type = "resnet18"
        config.optim.lr = 0.001
        config.epochs = 10

        # Use in a runner
        runner = Runner(config)
        ```

        Custom config class with typed attributes:
        ```python
        class TrainingConfig(RunnerConfig):
            # Type annotations provide auto-completion and validation
            epochs: int = 100
            batch_size: int = 32
            precision: str = "fp16"

            def __init__(self):
                super().__init__()
                # Initialize nested settings
                self.optim.type = "adamw"
                self.optim.lr = 1e-3

            def post(self):
                # Called after parsing CLI args
                super().post()
                # Create derived settings
                self.workspace.experiment = f"{self.network.type}_{self.optim.lr}"
        ```

        Command-line integration:
        ```bash
        # Override config settings via CLI
        python train.py --epochs 50 --optim.lr 0.0005 --network.type resnet50
        ```

    Note:
        Always store all parameters needed to reproduce a run in the RunnerConfig.
        The RunnerConfig is automatically saved with checkpoints, enabling exact resumption.

    See Also:
        - [`Runner`][danling.runners.Runner]: Main runner class that uses this config.
        - [`chanfig.Config`](https://github.com/ultmaster/chanfig): Base config implementation.
    """

    # DO NOT set default value in class, as they won't be stored in `__dict__`.

    stack: str = "auto"
    name: Optional[str] = None

    seed: Optional[int] = None
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

    optim: OptimizerConfig
    sched: SchedulerConfig
    fp8: Fp8Config
    deepspeed: Optional[Mapping[str, Any]] = None

    score: ScoreConfig
    workspace: WorkspaceConfig
    logging: LoggingConfig
    tensorboard: TensorboardConfig
    wandb: WandbConfig
    ft: FaultToleranceConfig

    compile: CompileConfig
    dist: DistributedConfig
    gc: GcConfig
    profiling: ProfilingConfig
    heartbeat: HeartbeatConfig
    ckpt: CheckpointConfig
    dataloader: DataloaderConfig
    fsdp: FsdpConfig
    parallel: ParallelConfig

    def __post_init__(self, *args, **kwargs) -> None:
        super().__post_init__(*args, **kwargs)
        if not isinstance(self.tensorboard, TensorboardConfig):
            self.tensorboard = TensorboardConfig()
        self.validate()

    def post(self) -> None:
        super().post()
        self.validate()

    def validate(self) -> None:
        if self.steps is not None and self.epochs is not None:
            raise ValueError("`steps` and `epochs` are mutually exclusive; set only one training boundary")

    @staticmethod
    def _semantic_section(section: Mapping[str, Any]) -> chanfig.NestedDict:
        return chanfig.NestedDict({key: value for key, value in section.items() if value is not None})

    def canonical(self) -> chanfig.NestedDict:
        canonical = chanfig.NestedDict(self.dict())
        stack = normalize_stack_name(canonical.get("stack", "auto"))
        canonical["stack"] = stack
        for key in NON_SEMANTIC_CONFIG_KEYS:
            canonical.pop(key, None)

        ckpt = canonical.get("ckpt")
        if isinstance(ckpt, Mapping):
            semantic_ckpt = chanfig.NestedDict(ckpt)
            backend = semantic_ckpt.get("backend")
            if backend is not None:
                backend = str(backend).strip().lower()
                if backend == "auto":
                    semantic_ckpt.pop("backend", None)
                else:
                    semantic_ckpt["backend"] = backend
            for key in NON_SEMANTIC_CKPT_KEYS:
                semantic_ckpt.pop(key, None)
            if semantic_ckpt:
                canonical["ckpt"] = semantic_ckpt
            else:
                canonical.pop("ckpt", None)

        for key in ("optim", "sched"):
            section = canonical.get(key)
            if isinstance(section, Mapping):
                semantic_section = self._semantic_section(section)
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
