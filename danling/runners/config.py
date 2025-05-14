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

from typing import Optional
from uuid import UUID, uuid5

import chanfig

from danling import defaults

from .utils import get_git_hash


class Config(chanfig.Config):  # pylint: disable=too-many-instance-attributes
    r"""
    Configuration class for managing and persisting all states of a DanLing Runner.

    The Config class provides a hierarchical configuration system that handles:

    1. **Parameter management**: Hyperparameters, model settings, training options
    2. **Experiment tracking**: IDs, names, and other metadata for runs and experiments
    3. **Serialization**: Save/load configurations from files or command line
    4. **Reproducibility**: Tracking seeds and settings for reproducible runs

    Config inherits from [`Config`][chanfig.Config] and provides attribute-style access to nested values:

    ```python
    config = Config()

    # Attribute-style access (recommended)
    config.optim.lr = 1e-3
    config.network.type = "resnet50"

    # Dictionary-style access (alternative)
    config["optim"]["lr"] = 1e-3
    config["network"]["type"] = "resnet50"
    ```

    Config objects support three types of hierarchical attribute access patterns:

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
       class MyConfig(Config):
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
        run_name (str): Human-readable name for this run. Defaults to `"DanLing"`.
        run_id (str): Unique identifier (hex string) for this run, derived from `run_uuid`.
        run_uuid (UUID, property): Unique UUID generated from experiment_uuid and config hash.
        experiment_name (str): Human-readable name for the experiment. Defaults to `"DanLing"`.
        experiment_id (str): Unique identifier for experiment, typically the git commit hash.
            Defaults to `"xxxxxxxxxxxxxxxx"` if not in a git repo or git/gitpython not installed.
        experiment_uuid (UUID, property): UUID derived from `experiment_id`.
            Defaults to `UUID('78787878-7878-7878-7878-787878787878')` if not in a git repo.

    Attributes: Reproducibility:
        seed (int): Random seed for reproducibility. If not set, a random value is generated.
        deterministic (bool): Whether to enforce deterministic operations in PyTorch.
            Defaults to `False` for better performance. Set to `True` for exact reproducibility.

    Attributes: Progress:
        steps (int): Current training step count.
        epochs (int): Current epoch count.
        step_begin (int): First step to run (for resuming). Defaults to 0.
        epoch_begin (int): First epoch to run (for resuming). Defaults to 0.
        step_end (int): Last step to run (optional). Use either this or `epoch_end`.
        epoch_end (int): Last epoch to run (optional). Use either this or `step_end`.

    Attributes: Model Evaluation:
        score_split (str): Dataset split to use for model selection. Defaults to None.
        score_name (str): Metric name to use for model selection. Defaults to "loss".

    Attributes: I/O:
        project_root (str): Root directory for experiments. Defaults to `"experiments"`.
        checkpoint_dir_name (str): Subdirectory name for checkpoints. Defaults to `"checkpoints"`.
        log (bool): Whether to enable file logging. Defaults to `True`.
        tensorboard (bool): Whether to use TensorBoard for visualization. Defaults to `False`.
        log_interval (int): Iterations between log outputs. If None, auto-calculated.
        save_interval (int): Epochs between checkpoint saves. If None, only save best/latest.

    Examples:
        Basic usage:
        ```python
        # Create a config
        config = Config()
        config.network.type = "resnet18"
        config.optim.lr = 0.001
        config.epochs = 10

        # Use in a runner
        runner = Runner(config)
        ```

        Custom config class with typed attributes:
        ```python
        class TrainingConfig(Config):
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
                self.experiment_name = f"{self.network.type}_{self.optim.lr}"
        ```

        Command-line integration:
        ```bash
        # Override config settings via CLI
        python train.py --epochs 50 --optim.lr 0.0005 --network.type resnet50
        ```

    Note:
        Always store all parameters needed to reproduce a run in the Config.
        The Config is automatically saved with checkpoints, enabling exact resumption.

    See Also:
        - [`Runner`][danling.runners.Runner]: Main runner class that uses this config.
        - [`chanfig.Config`](https://github.com/ultmaster/chanfig): Base config implementation.
    """

    # DO NOT set default value in class, as they won't be stored in `__dict__`.

    run_name: str = defaults.RUN_NAME
    run_id: str
    experiment_name: str = defaults.EXPERIMENT_NAME
    experiment_id: str

    seed: Optional[int] = None
    deterministic: bool = False

    steps: int = 0
    epochs: int = 0
    step_begin: int = 0
    epoch_begin: int = 0
    step_end: Optional[int] = None
    epoch_end: Optional[int] = None

    score_split: Optional[str] = None
    score_name: str = "loss"

    project_root: str = "experiments"
    checkpoint_dir_name: str = "checkpoints"
    log: bool = True
    tensorboard: bool = False
    log_interval: Optional[int] = None
    save_interval: Optional[int] = None

    def __post_init__(self):
        if "experiment_id" not in self:
            self.experiment_id = get_git_hash() or defaults.EXPERIMENT_ID
        if "run_id" not in self:
            self.run_id = self.run_uuid.hex
        self.setattr("ignored_keys_in_hash", defaults.IGNORED_NAMES_IN_HASH)

    @property
    def experiment_uuid(self) -> UUID:
        r"""
        UUID of the experiment.
        """

        return UUID(bytes=bytes(self.experiment_id.ljust(16, "x")[:16], encoding="ascii"))

    @property
    def run_uuid(self) -> UUID:
        r"""
        UUID of the run.
        """

        ignored_keys_in_hash = self.getattr("ignored_keys_in_hash", defaults.IGNORED_NAMES_IN_HASH)
        state: chanfig.Config = chanfig.Config({k: v for k, v in self.dict().items() if k not in ignored_keys_in_hash})
        return uuid5(self.experiment_uuid, state.yamls())

    def __hash__(self) -> int:
        return int(self.run_uuid.hex, 16)
