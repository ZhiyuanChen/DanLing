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
from typing import Optional, Union

import chanfig


class CompileConfig(chanfig.Config):
    enable: bool = False
    backend: Optional[str] = None
    fullgraph: Optional[bool] = None
    dynamic: Optional[bool] = None
    components: Optional[Union[Sequence[str], str]] = None
    optimize_ddp: Optional[str] = "ddp_optimizer"


class FsdpConfig(chanfig.Config):
    mode: str = "full_shard"
    replicate_degree: Optional[int] = None
    shard_degree: Optional[int] = None


class TpppConfig(chanfig.Config):
    tp_degree: int = 1
    pp_degree: int = 1
    pipeline_schedule: str = "1F1B"
    pipeline_microbatch_size: int = 1
    pipeline_n_microbatches: Optional[int] = None
    use_device_mesh: bool = True
    mesh_device_type: Optional[str] = None
    allow_degree_change: bool = False


class DtppConfig(chanfig.Config):
    mode: str = "full_shard"
    replicate_degree: Optional[int] = None
    shard_degree: Optional[int] = None
    use_device_mesh: bool = True
    mesh_device_type: Optional[str] = None


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
            Supported values: `"auto"`, `"ddp"`/`"torch"`, `"deepspeed"`/`"ds"`, `"fsdp"`, `"tppp"`, `"dtpp"`.
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
            `BaseRunner.dir` appends code identity (`-<git_hash>`) when available.
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
        tppp.tp_degree (int): Tensor-parallel degree for TPPP topology validation. Defaults to `1`.
        tppp.pp_degree (int): Pipeline-parallel degree for TPPP topology validation. Defaults to `1`.
        tppp.pipeline_schedule (str): Pipeline schedule class name resolved by
            `torch.distributed.pipelining.schedules.get_schedule_class`.
            Defaults to `"1F1B"`.
        tppp.pipeline_microbatch_size (int): Local microbatch size used to infer
            schedule microbatch count as `dataloader.batch_size // pipeline_microbatch_size`.
            Defaults to `1`.
        tppp.pipeline_n_microbatches (int): Explicit schedule microbatch count.
            When set, overrides `pipeline_microbatch_size`-based inference.
        dtpp.mode (str): DTPP sharding mode. Supported values: `"full_shard"` and `"hybrid_shard"`/`"hsdp"`.
        dtpp.replicate_degree (int): Replication degree when `dtpp.mode="hybrid_shard"`.
        dtpp.shard_degree (int): Shard degree for `dtpp.mode`.
        log (bool): Whether to enable file logging. Defaults to `True`.
            Logging is initialized on the main process only.
        tensorboard (bool): Whether to use TensorBoard for visualization. Defaults to `False`.
        log_interval (int): Iterations between log outputs. If None, auto-calculated.
        checkpoint.interval (int): Interval between checkpoint save attempts for `latest`/`best`.
            The same cadence is used for archived checkpoints.
            Uses epochs in epoch mode and global steps in step mode.
            If unset, runner defaults are used by mode.
        checkpoint.keep_latest_k (int): Number of archived checkpoints to retain. `0` disables retention pruning.
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
        fsdp.mode (str): FSDP sharding mode for `FsdpRunner`.
            Supported values: `"full_shard"` (default) and `"hybrid_shard"`/`"hsdp"`.
        fsdp.replicate_degree (int): HSDP replication degree when `fsdp.mode="hybrid_shard"`.
        fsdp.shard_degree (int): HSDP sharding degree when `fsdp.mode="hybrid_shard"`.
        compile.enable (bool): Whether to enable `torch.compile` for selected components.
        compile.components (list[str] | str): Components to compile, subset of `["model", "loss"]`.
            Defaults to `"model"` when compile is enabled and components are unset.
        compile.backend (str): Optional backend passed to `torch.compile`.
        compile.fullgraph (bool): Optional `fullgraph` flag for `torch.compile`.
        compile.dynamic (bool): Optional `dynamic` flag for `torch.compile`.
        compile.mode (str): Optional mode passed to `torch.compile`.
        compile.options (dict): Optional options passed to `torch.compile`.
        compile.optimize_ddp (str | None): Optional `torch._dynamo.config.optimize_ddp` value.
            Defaults to `"ddp_optimizer"` when model compile is enabled.

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
    log_interval: Optional[int] = None

    compile: CompileConfig
    checkpoint: CheckpointConfig
    fsdp: FsdpConfig
    tppp: TpppConfig
    dtpp: DtppConfig

    def canonical(self) -> chanfig.NestedDict:
        canonical = chanfig.NestedDict(self.dict())
        for key in NON_SEMANTIC_CONFIG_KEYS:
            canonical.pop(key, None)
        return canonical

    def __post_init__(self, *args, **kwargs) -> None:
        super().__post_init__(*args, **kwargs)
        if "compile" not in self:
            self.compile = CompileConfig()
        if "checkpoint" not in self:
            self.checkpoint = CheckpointConfig()

    def __hash__(self) -> int:
        digest = hashlib.sha1(self.canonical().yamls().encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)


NON_SEMANTIC_CONFIG_KEYS: tuple[str, ...] = (
    "score_split",
    "score_name",
    "score",
    "log",
    "log_interval",
    "tensorboard",
    "workspace_root",
    "lineage",
    "experiment",
    "fsdp",
    "tppp",
    "dtpp",
    "checkpoint",
    "auto_resume",
)
