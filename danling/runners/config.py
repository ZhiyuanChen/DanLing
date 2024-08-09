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
    enable: bool = False
    backend: Optional[str] = None
    mode: Optional[str] = None
    fullgraph: Optional[bool] = None
    dynamic: Optional[bool] = None
    options: Optional[Mapping[str, Any]] = None
    optimize_ddp: Optional[str] = "ddp_optimizer"


class FsdpConfig(chanfig.Config):
    enabled: bool = False
    mesh: Optional[Any] = None
    reshard_after_forward: Union[bool, int, None] = None
    shard_placement_fn: Optional[Any] = None
    mp_policy: Union[Any, Mapping[str, Any], None] = None
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
    pipeline_n_microbatches: Optional[int] = None
    use_device_mesh: bool = True
    mesh_device_type: Optional[str] = None
    allow_degree_change: bool = False

    def __post_init__(self, *args, **kwargs) -> None:
        super().__post_init__(*args, **kwargs)
        if "axes" not in self:
            self.axes = ParallelAxesConfig()
        elif not isinstance(self.axes, ParallelAxesConfig):
            self.axes = ParallelAxesConfig(self.axes)


class CheckpointConfig(chanfig.Config):
    dir_name: str = "checkpoints"
    backend: str = "auto"
    wait_timeout: Optional[float] = None
    interval: Optional[int] = None
    keep_latest_k: int = 0
    load_only: bool = False
    async_enabled: bool = True
    async_mode: Optional[str] = None
    dedicated_async_process_group: bool = True
    async_process_group_backend: str = "gloo"
    enable_ft_dataloader_checkpoints: bool = False
    ft_replica_id: Optional[str] = None
    ft_dataloader_checkpoint_prefix: str = "ft-replica"
    last_save_model_only: bool = False
    export_dtype: Optional[str] = None


class CommConfig(chanfig.Config):
    init_timeout_seconds: Optional[int] = None
    train_timeout_seconds: Optional[int] = None


class GcConfig(chanfig.Config):
    interval: Optional[int] = None
    generation: int = 1
    disable_automatic: bool = True


class ProfilingConfig(chanfig.Config):
    enabled: bool = False
    wait: int = 1
    warmup: int = 1
    active: int = 3
    repeat: Optional[int] = None
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_flops: bool = False
    trace_dir: str = "profiles"


class HeartbeatConfig(chanfig.Config):
    enabled: bool = False
    interval_seconds: float = 60.0
    dir_name: str = "heartbeats"


class WandbConfig(chanfig.Config):
    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None
    group: Optional[str] = None
    name: Optional[str] = None
    job_type: Optional[str] = None
    tags: Union[Sequence[str], str, None] = None
    dir: Optional[str] = None
    mode: Optional[str] = None


class FaultToleranceConfig(chanfig.Config):
    enabled: bool = False
    process_group: str = "gloo"
    process_group_timeout_ms: int = 10000
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
        "deepspeed": "deepspeed",
        "ds": "deepspeed",
        "parallel": "parallel",
    }
    return aliases.get(normalized, normalized)


NON_SEMANTIC_CONFIG_KEYS: tuple[str, ...] = (
    "score_split",
    "score_name",
    "score",
    "log",
    "log_interval",
    "tensorboard",
    "wandb",
    "ft",
    "workspace_root",
    "comm",
    "gc",
    "profiling",
    "heartbeat",
    "lineage",
    "experiment",
    "resume",
    "pretrained",
    "auto_resume",
)


NON_SEMANTIC_CHECKPOINT_KEYS: tuple[str, ...] = (
    "dir_name",
    "wait_timeout",
    "interval",
    "keep_latest_k",
    "load_only",
    "async_enabled",
    "async_mode",
    "dedicated_async_process_group",
    "async_process_group_backend",
    "enable_ft_dataloader_checkpoints",
    "ft_replica_id",
    "ft_dataloader_checkpoint_prefix",
    "last_save_model_only",
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
            Supported values: `"auto"`, `"ddp"`/`"torch"`, `"deepspeed"`/`"ds"`, `"parallel"`.
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

    Attributes: Model Evaluation:
        score_split (str): Dataset split to use for model selection. Defaults to None.
            If unset, runner infers once (`val` -> `validate` -> first available) and reuses it
            unless that split disappears from results.
        score_name (str): Metric name to use for model selection. Defaults to "loss".
        scheduler.interval (str): Scheduler advancement policy.
            Supported values: `"step"` and `"epoch"`/`"validation"`.
            Non-metric schedulers default to `"step"`. Metric schedulers such as
            `ReduceLROnPlateau` default to `"epoch"` and advance after the aggregated
            round result is available.
        scheduler.monitor (str): Optional metric selector for metric schedulers.
            Supports dotted paths such as `"val.loss"`.
            When unset, the runner prefers `score_split/score_name` when available and
            otherwise resolves `score_name` from the aggregated result.

    Attributes: I/O:
        workspace_root (str): Root directory for experiments. Defaults to `"experiments"`.
        auto_resume (bool): Auto-resume from backend latest checkpoint alias/path.
            When `True`, runner resolves the backend-native latest checkpoint source.
            Priority is `resume` > `auto_resume` > `pretrained`.
        resume (str | None): Optional full-state checkpoint source for resume workflows.
            This is a path-like identifier consumed by runner `load_checkpoint(...)`.
        pretrained (str | None): Optional model-only checkpoint source for finetune workflows.
            This is a path-like identifier consumed by runner `load_pretrained(...)`.
        lineage (str): Top-level lineage namespace.
            Defaults to `"lin"` when unset.
            `RunnerWorkspace.dir` appends code identity (`-<git_hash>`) when available.
        experiment (str): Experiment namespace. Defaults to `"exp"`.
        checkpoint.dir_name (str): Subdirectory name for checkpoints. Defaults to `"checkpoints"`.
        checkpoint.async_enabled (bool): Whether to persist checkpoints asynchronously.
            Defaults to `True`.
        checkpoint.async_mode (str | None): Checkpoint async behavior.
            Supported values: `"disabled"`, `"async"`, `"async_with_pinned_mem"`.
            When unset (`None`), the runner derives the mode from `checkpoint.async_enabled`.
        checkpoint.dedicated_async_process_group (bool): Use a dedicated process group for async DCP
            checkpoint I/O to reduce interference with training collectives. Defaults to `True`.
        checkpoint.async_process_group_backend (str): Backend for the dedicated async checkpoint process
            group. Defaults to `"gloo"`.
        checkpoint.backend (str): Checkpoint backend selected at runtime by the runner
            (`"dcp"` for distributed runs, `"file"` otherwise when set to `"auto"`).
        checkpoint.wait_timeout (float): Timeout in seconds when draining async checkpoint writes
            during runner shutdown (`None` waits indefinitely).
        parallel.axes.replicate (int): Data-replication degree for DDP/HSDP-style replication.
            Defaults to `1`.
        parallel.axes.shard (int): Data-sharding degree for FSDP-style sharding.
            Defaults to `1`.
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
        parallel.pipeline_n_microbatches (int): Explicit schedule microbatch count.
            When set, overrides `pipeline_microbatch_size`-based inference.
        log (bool): Whether to enable file logging. Defaults to `True`.
            Logging is initialized on the main process only.
        tensorboard (bool): Whether to use TensorBoard for visualization. Defaults to `False`.
        wandb.enabled (bool): Whether to enable Weights & Biases scalar logging. Defaults to `False`.
        wandb.project (str | None): Optional W&B project name. Defaults to `lineage`.
        wandb.entity (str | None): Optional W&B entity/team override.
        wandb.group (str | None): Optional W&B group name. Defaults to `experiment`.
        wandb.name (str | None): Optional W&B display name. Defaults to stable runner `id`.
        wandb.job_type (str | None): Optional W&B job type.
        wandb.tags (list[str] | str | None): Optional W&B run tags.
        wandb.dir (str | None): Optional local W&B run directory. Defaults to run dir.
        wandb.mode (str | None): Optional W&B mode such as `"online"` or `"offline"`.
        ft.enabled (bool): Enable TorchFT-managed fault tolerance. Defaults to `False`.
        ft.process_group (str): TorchFT coordination backend. Supported values: `"gloo"` and `"nccl"`.
            Defaults to `"gloo"`.
        ft.process_group_timeout_ms (int): TorchFT process-group timeout in milliseconds.
            Defaults to `10000`.
        ft.replica_id (int): Replica-group identifier for this run. Defaults to `0`.
        ft.group_size (int): Number of replica groups participating in TorchFT. Defaults to `1`.
        ft.min_replica_size (int): Minimum healthy replicas required by TorchFT per step.
            Defaults to `1`.
        log_interval (int): Iterations between log outputs. If None, auto-calculated.
        checkpoint.interval (int): Interval between checkpoint save attempts for `latest`/`best`.
            The same cadence is used for history checkpoints.
            Uses epochs in epoch mode and global steps in step mode.
            If unset, runner defaults are used by mode.
        checkpoint.keep_latest_k (int): Number of framework-generated history checkpoints to retain.
            `0` disables retention pruning.
        checkpoint.load_only (bool): Disable checkpoint persistence entirely while still allowing checkpoint loading.
        checkpoint.enable_ft_dataloader_checkpoints (bool): Enable per-replica dataloader checkpoints for FT recovery.
            Uses DCP and stores checkpoints under
            `checkpoint.ft_dataloader_checkpoint_prefix-{checkpoint.ft_replica_id}`.
        checkpoint.ft_replica_id (str | None): Replica identifier used for FT dataloader checkpoint directory naming.
            Defaults to `FT_REPLICA_ID` environment variable, then process rank.
        checkpoint.ft_dataloader_checkpoint_prefix (str): Prefix used for FT per-replica checkpoint directories.
            Defaults to `"ft-replica"`.
        checkpoint.last_save_model_only (bool): Save model-only payload on final `last_step` checkpoint.
        checkpoint.export_dtype (str): Optional dtype cast for final model-only export
            (`fp32`/`fp16`/`bf16`/`fp64` aliases supported).
        dataloader.batch_size (int | None): Local dataloader batch size passed to
            `StatefulDataLoader`.
        dataloader.shuffle (bool | None): Optional shuffle override. When unset, train
            splits shuffle and non-train splits do not.
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
        fsdp.mp_policy: Optional FSDP2 mixed precision policy.
        fsdp.offload_policy: Optional FSDP2 CPU offload policy.
        compile.enable (bool): Whether to enable `torch.compile` for runner-selected model compilation points.
        compile.backend (str): Optional backend passed to `torch.compile`.
        compile.fullgraph (bool): Optional `fullgraph` flag for `torch.compile`.
        compile.dynamic (bool): Optional `dynamic` flag for `torch.compile`.
        compile.mode (str): Optional mode passed to `torch.compile`.
        compile.options (dict): Optional options passed to `torch.compile`.
        compile.optimize_ddp (str | None): Optional `torch._dynamo.config.optimize_ddp` value.
            Defaults to `"ddp_optimizer"` when model compile is enabled.
        comm.init_timeout_seconds (int | None): Optional distributed process-group timeout used during
            initialization and early startup.
        comm.train_timeout_seconds (int | None): Optional tighter distributed process-group timeout applied
            once after the first successful optimizer step.
        gc.interval (int | None): Optional periodic Python GC cadence.
            When unset, runner-managed GC pacing is disabled.
        gc.generation (int): Python GC generation passed to `gc.collect(...)` when pacing is enabled.
            Defaults to `1`.
        gc.disable_automatic (bool): Disable CPython automatic GC while runner-managed pacing is enabled.
            Defaults to `True`.
        profiling.enabled (bool): Enable bounded-step `torch.profiler` tracing. Defaults to `False`.
        profiling.wait (int): Profiler schedule wait steps before warmup. Defaults to `1`.
        profiling.warmup (int): Profiler schedule warmup steps. Defaults to `1`.
        profiling.active (int): Profiler schedule active trace steps. Defaults to `3`.
        profiling.repeat (int | None): Optional profiler schedule repeat count.
        profiling.record_shapes (bool): Enable shape recording in traces. Defaults to `False`.
        profiling.profile_memory (bool): Enable profiler-side memory recording. Defaults to `False`.
        profiling.with_stack (bool): Include Python stack traces in profiler output. Defaults to `False`.
        profiling.with_flops (bool): Enable profiler FLOPs estimation when available. Defaults to `False`.
        profiling.trace_dir (str): Relative or absolute trace output directory. Defaults to `"profiles"`.
        heartbeat.enabled (bool): Enable a machine-readable per-rank heartbeat/progress file. Defaults to `False`.
        heartbeat.interval_seconds (float): Heartbeat write interval in seconds. Defaults to `60.0`.
        heartbeat.dir_name (str): Subdirectory under the run dir for heartbeat files. Defaults to `"heartbeats"`.
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
                self.experiment = f"{self.network.type}_{self.optim.lr}"
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

    seed: Optional[int] = None
    deterministic: bool = False

    steps: Optional[int] = None
    epochs: Optional[int] = None
    accum_steps: int = 1

    score_split: Optional[str] = None
    score_name: str = "loss"

    workspace_root: str = "experiments"
    auto_resume: bool = False
    resume: Optional[str] = None
    pretrained: Optional[str] = None
    log: bool = True
    tensorboard: bool = False
    wandb: WandbConfig
    ft: FaultToleranceConfig
    log_interval: Optional[int] = None

    compile: CompileConfig
    comm: CommConfig
    gc: GcConfig
    profiling: ProfilingConfig
    heartbeat: HeartbeatConfig
    checkpoint: CheckpointConfig
    dataloader: DataloaderConfig
    fsdp: FsdpConfig
    parallel: ParallelConfig

    def __post_init__(self, *args, **kwargs) -> None:
        super().__post_init__(*args, **kwargs)
        self.validate()
        if "compile" not in self:
            self.compile = CompileConfig()
        if "comm" not in self:
            self.comm = CommConfig()
        if "gc" not in self:
            self.gc = GcConfig()
        if "profiling" not in self:
            self.profiling = ProfilingConfig()
        if "heartbeat" not in self:
            self.heartbeat = HeartbeatConfig()
        if "wandb" not in self:
            self.wandb = WandbConfig()
        if "ft" not in self:
            self.ft = FaultToleranceConfig()
        if "checkpoint" not in self:
            self.checkpoint = CheckpointConfig()
        if "parallel" not in self:
            self.parallel = ParallelConfig()
        elif not isinstance(self.parallel, ParallelConfig):
            self.parallel = ParallelConfig(self.parallel)

    def post(self) -> None:
        super().post()
        self.validate()

    def validate(self) -> None:
        if self.steps is not None and self.epochs is not None:
            raise ValueError("`steps` and `epochs` are mutually exclusive; set only one training boundary")

    def canonical(self) -> chanfig.NestedDict:
        canonical = chanfig.NestedDict(self.dict())
        stack = normalize_stack_name(canonical.get("stack", "auto"))
        canonical["stack"] = stack
        for key in NON_SEMANTIC_CONFIG_KEYS:
            canonical.pop(key, None)

        checkpoint = canonical.get("checkpoint")
        if isinstance(checkpoint, Mapping):
            semantic_checkpoint = chanfig.NestedDict(checkpoint)
            backend = semantic_checkpoint.get("backend")
            if backend is not None:
                semantic_checkpoint["backend"] = str(backend).strip().lower()
            for key in NON_SEMANTIC_CHECKPOINT_KEYS:
                semantic_checkpoint.pop(key, None)
            if semantic_checkpoint:
                canonical["checkpoint"] = semantic_checkpoint
            else:
                canonical.pop("checkpoint", None)

        if stack != "parallel":
            canonical.pop("fsdp", None)
        if stack != "parallel":
            canonical.pop("parallel", None)
        return canonical

    def __hash__(self) -> int:
        digest = hashlib.sha1(self.canonical().yamls().encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)
