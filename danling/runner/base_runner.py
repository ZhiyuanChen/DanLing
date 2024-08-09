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

# pylint: disable=redefined-builtin, keyword-arg-before-vararg
from __future__ import annotations

import logging
import logging.config
import os
import random
import shutil
from collections.abc import Callable, Mapping, Sequence
from math import ceil
from sys import version_info
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid5
from warnings import warn

from chanfig import FlatDict, NestedDict, Variable

from danling import defaults
from danling.metrics import AverageMeter, AverageMeters, MetricMeters
from danling.typing import File, PathStr
from danling.utils import cached_ensure_dir, cached_ensure_parent_dir, cached_property, catch, load, save

try:
    from numpy import random as np_random
except ImportError:
    np_random = None

from .config import Config
from .utils import RunnerMeta, RunnerMode, format_result, get_time_str, on_main_process

if TYPE_CHECKING:
    from danling.metrics import Metrics

PY38_PLUS = version_info >= (3, 8)
__APPEND_RESULT_COUNTER__ = 0


class BaseRunner(metaclass=RunnerMeta):  # pylint: disable=too-many-public-methods
    r"""
    Base class for DanLing runners.

    The `BaseRunner` class provides a complete infrastructure for managing:

    1. **Experiment Environment**: Sets up reproducible training environment with:
        - Random seed management
        - Deterministic execution options
        - Configurable logging systems

    2. **Experiment Lifecycle**: Handles the complete experiment workflow:
        - Model, optimizer, and scheduler setup
        - Dataset and dataloader management
        - Training, evaluation, and inference loops
        - Checkpointing and restoration

    3. **Experiment Tracking**: Provides comprehensive experiment monitoring:
        - Progress tracking
        - Metrics computation and logging
        - Results collection and comparison
        - Best model selection

    4. **Distributed Training**: Supports various parallel training configurations:
        - Multi-process distributed training
        - Gradient accumulation
        - Mixed precision training

    This class defines standard attributes and methods that can be extended by more
    specialized runners. It is designed to be modular and configurable to support
    various experiment requirements.

    Attributes: ID:
        timestamp: Time string of when the runner was created.
            Format: `WDHMSUUU-` (Week, Weekday, Hour, Minute, Second, Microsecond)
        name: Human-readable name combining experiment and run names (`{experiment_name}-{run_name}`).
        id: Unique identifier generated from experiment and run IDs.
        uuid: UUID5 generated from the run ID and combined ID.

    See Also:
        [`Config`][danling.runner.Config]: Configuration for runners.

        [`RunnerMode`][danling.runner.utils.RunnerMode]: Enumeration of runner modes.

    Attributes: Model:
        model: The model being trained/evaluated.
        criterion: Loss function used for optimization.
        optimizer: Optimizer used for parameter updates.
        scheduler: Learning rate scheduler for adaptive learning rates.

    Attributes: Data:
        datasets: Dictionary mapping split names to dataset objects.
        datasamplers: Dictionary mapping split names to data samplers.
        dataloaders: Dictionary mapping split names to dataloader objects.
        split: Current active data split.
        batch_size: Number of samples per batch for current split.
        batch_size_equivalent: Total effective batch size accounting for distribution.

    Attributes: Progress:
        progress: Running progress as a value between 0 and 1.
        steps: Current step count.
        epochs: Current epoch count.
        total_steps: Total steps to be performed.
        total_epochs: Total epochs to be performed.

    Attributes: Results:
        results: Hierarchical results organized by epoch, split, and metric.
        latest_result: Most recent evaluation results.
        best_result: Best evaluation results achieved.
        scores: Main metric values used for model comparison.
        latest_score: Most recent score.
        best_score: Best score achieved.
        is_best: Whether the latest result is the best so far.

    Attributes: I/O:
        dir: Root directory for all experiment outputs.
        checkpoint_dir: Directory for saving model checkpoints.
        log_file: Path to the log file.

    Attributes: Distributed Training:
        world_size: Number of processes in distributed training.
        rank: Global process rank.
        local_rank: Local process rank on current node.
        distributed: Whether running in distributed mode.
        is_main_process: Whether current process is the main process.

    Attributes: Logging and Visualization:
        meters: Meters for tracking running averages of metrics.
        metrics: Metric computation objects.
        logger: Logger for capturing runtime information.
        writer: Writer for visualizing metrics (e.g., TensorBoard).
    """

    # DO NOT set default value in class, as they won't be stored in `__dict__`.

    timestamp: str

    _mode: RunnerMode
    _config: Config
    inited: bool = False

    model: Callable | None = None
    ema: Callable | None = None
    criterion: Callable | None = None
    optimizer: Any | None = None
    scheduler: Any | None = None

    datasets: FlatDict
    datasamplers: FlatDict
    dataloaders: FlatDict
    split: str | None = None
    _train_splits: list[str] | None = None
    _evaluate_splits: list[str] | None = None

    results: NestedDict
    meters: AverageMeters
    metrics: Metrics | MetricMeters | None = None
    train_metrics: Metrics | MetricMeters | None = None
    evaluate_metrics: Metrics | MetricMeters | None = None
    logger: logging.Logger | None = None
    writer: Any | None = None

    def __init__(self, config: Config) -> None:
        self.timestamp = get_time_str()
        if "datasets" not in self.__dict__:
            self.datasets = FlatDict()
        if "datasamplers" not in self.__dict__:
            self.datasamplers = FlatDict()
        if "dataloaders" not in self.__dict__:
            self.dataloaders = FlatDict()
        if "results" not in self.__dict__:
            self.results = NestedDict()
        self.meters = AverageMeters()
        self._mode = RunnerMode.train
        # must init config at last to avoid name conflicts
        if not isinstance(config, Config):
            config = Config(config)
        self._config = config
        self.init_distributed()
        self.inited = True
        if "checkpoint" in config:
            self.load_config(config["checkpoint"])

    def __post_init__(self):
        if self.config.seed is not None:
            self.set_seed()
        if self.config.deterministic:
            self.set_deterministic()
        if self.config.log:
            self.init_logging()
        self.init_print()
        if self.config.tensorboard:
            self.init_tensorboard()

    def init_distributed(self) -> None:
        r"""
        Initialise distributed running environment.
        """

    @on_main_process
    def init_logging(self) -> None:
        r"""
        Set up logging.
        """

        # Why is setting up proper logging so !@?#! ugly?
        logging.config.dictConfig(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
                },
                "handlers": {
                    "stdout": {
                        "level": "INFO",
                        "formatter": "standard",
                        "class": "logging.StreamHandler",
                        "stream": "ext://sys.stdout",
                    },
                    "logfile": {
                        "level": "DEBUG",
                        "formatter": "standard",
                        "class": "logging.FileHandler",
                        "filename": self.log_file,
                        "mode": "a",
                    },
                },
                "loggers": {
                    "": {
                        "handlers": ["stdout", "logfile"],
                        "level": "DEBUG",
                        "propagate": True,
                    },
                },
            }
        )
        logging.captureWarnings(True)
        self.logger = logging.getLogger("runner")
        self.logger.flush = lambda: [h.flush() for h in self.logger.handlers]  # type: ignore

    def init_print(self, process: int = 0) -> None:
        r"""
        Set up `print`.

        Only print on a specific `process` or when `force = True`.

        Args:
            process: The process to `print` on.

        Note:
            If `self.config.log = True`, the default `print` function will be override by `logging.info`.
        """

        logger = logging.getLogger("print")
        logger.flush = lambda: [h.flush for h in logger.handlers]  # type: ignore
        import builtins as __builtin__  # pylint: disable=C0415

        builtin_print = __builtin__.print

        @catch
        def print(*args, force=False, end="\n", file=None, flush=False, **kwargs):  # pylint: disable=redefined-builtin
            if self.rank == process or force:
                if self.config.log:
                    if not args:
                        args = [""]
                    logger.info(*args, **kwargs)
                else:
                    builtin_print(*args, end=end, file=file, flush=flush, **kwargs)

        __builtin__.print = print

    @on_main_process
    def init_tensorboard(self, *args, **kwargs) -> None:
        r"""
        Set up Tensoraoard SummaryWriter.
        """
        raise NotImplementedError

    def set_seed(self, seed: int = None, bias: int = None) -> int:  # type: ignore[assignment]
        r"""
        Set random seeds for reproducibility across the application.

        This method configures the random number generators for various libraries
        to ensure reproducible behavior in experiments. In the base implementation,
        it sets seeds for Python's built-in `random` module and NumPy's random
        generators if available.

        Derived runners typically extend this method to set seeds for additional
        libraries (e.g., PyTorch, TensorFlow).

        Args:
            seed: The base random seed to use.
                If None, uses the value from `self.config.seed`.

            bias: An offset to add to the seed for process-specific randomness.
                - If specified, adds this value to the seed
                - If None, uses `self.rank` (recommended for distributed training)
                - If False, disables bias (all processes use identical seed)

                Using a per-process bias is important in distributed training to avoid
                all processes applying identical data augmentation.

        Returns:
            int: The actual seed that was set (after applying bias)

        Examples:
            # Set seed from config
            runner.set_seed()

            # Set specific seed
            runner.set_seed(42)

            # Set seed with explicit bias
            runner.set_seed(42, bias=10)

            # Set same seed for all processes (not recommended for data augmentation)
            runner.set_seed(42, bias=False)

        Note:
            - Setting seeds is essential for reproducible experiments
            - Different biases across processes ensure diverse data augmentation
            - Subclasses should call `super().set_seed()` if overriding
        """

        seed = seed or self.config.seed  # type: ignore[assignment]
        bias = bias or self.rank
        if bias:
            seed += bias
        if np_random is not None:
            np_random.seed(seed)
        random.seed(seed)
        return seed

    def set_deterministic(self) -> None:
        r"""
        Set up deterministic.
        """

        raise NotImplementedError

    def scale_lr(
        self,
        lr: float,
        lr_scale_factor: float | None = None,
        batch_size_base: int | None = None,
    ) -> float:
        r"""
        Scale learning rate according to [linear scaling rule](https://arxiv.org/abs/1706.02677).
        """

        if lr_scale_factor in self.config:
            lr_scale_factor = self.config.lr_scale_factor

        if lr_scale_factor is None:
            if batch_size_base is None:
                batch_size_base = getattr(self, "batch_size_base", None)
                if batch_size_base is None:
                    raise ValueError("batch_size_base must be specified to auto scale lr")
            lr_scale_factor = self.batch_size_equivalent / batch_size_base
        elif batch_size_base is not None:
            warn(
                "batch_size_base will be ignored if lr_scale_factor is specified", category=RuntimeWarning, stacklevel=2
            )
        lr = lr * lr_scale_factor
        self.config.lr_scale_factor = lr_scale_factor
        return lr

    def advance(self, loss, *args, **kwargs) -> None:
        r"""
        Backward loss and step optimizer & scheduler.

        Args:
            loss: loss.
        """

        raise NotImplementedError

    def state_dict(self, cls: Callable = dict) -> Mapping:
        r"""
        Return dict of all attributes for checkpoint.
        """

        return cls(self.config)

    def dict(self, cls: Callable = dict) -> Mapping:
        r"""
        Convert config to Mapping.

        Args:
            cls: Target `clc to convert to.
        """

        return self.config.dict(cls)

    @catch
    def save(self, obj: Any, file: PathStr, main_process_only: bool = True, *args, **kwargs) -> File:
        r"""
        Save any file with supported extensions.

        `Runner.save` internally calls `dl.save`,
        but with additional arguments to allow it save only on the main process.
        Moreover, any error raised by `Runner.save` will be caught and logged.
        """

        if main_process_only and self.is_main_process or not main_process_only:
            return save(obj, file, *args, **kwargs)
        return file

    @staticmethod
    def load(file: PathStr, *args, **kwargs) -> Any:
        r"""
        Load any file with supported extensions.

        `Runner.load` is identical to `dl.load`.
        """

        return load(file, *args, **kwargs)

    @catch
    def json(self, file: File, main_process_only: bool = True, *args, **kwargs) -> None:  # pylint: disable=R1710
        r"""
        Dump Runner config to json file.
        """

        if main_process_only and self.is_main_process or not main_process_only:
            return self.config.json(file, *args, **kwargs)

    @classmethod
    def from_json(cls, file: File, *args, **kwargs) -> BaseRunner:
        r"""
        Construct Runner from json file.

        This function calls `self.from_jsons()` to construct object from json string.
        You may overwrite `from_jsons` in case something is not json serializable.
        """

        with FlatDict.open(file) as fp:
            return cls.from_jsons(fp.read(), *args, **kwargs)

    def jsons(self, *args, **kwargs) -> str:
        r"""
        Dump Runner config to json string.
        """

        return self.config.jsons(*args, **kwargs)

    @classmethod
    def from_jsons(cls, string: str, *args, **kwargs) -> BaseRunner:
        r"""
        Construct Runner from json string.
        """

        return cls(Config.from_jsons(string, *args, **kwargs))

    @catch
    def yaml(self, file: File, main_process_only: bool = True, *args, **kwargs) -> None:  # pylint: disable=R1710
        r"""
        Dump Runner config to yaml file.
        """

        if main_process_only and self.is_main_process or not main_process_only:
            return self.config.yaml(file, *args, **kwargs)

    @classmethod
    def from_yaml(cls, file: File, *args, **kwargs) -> BaseRunner:
        r"""
        Construct Runner from yaml file.

        This function calls `self.from_yamls()` to construct object from yaml string.
        You may overwrite `from_yamls` in case something is not yaml serializable.
        """

        with FlatDict.open(file) as fp:
            return cls.from_yamls(fp.read(), *args, **kwargs)

    def yamls(self, *args, **kwargs) -> str:
        r"""
        Dump Runner config to yaml string.
        """

        return self.config.yamls(*args, **kwargs)

    @classmethod
    def from_yamls(cls, string: str, *args, **kwargs) -> BaseRunner:
        r"""
        Construct Runner from yaml string.
        """

        return cls(Config.from_yamls(string, *args, **kwargs))

    def check_dir(self, action: str = "warn") -> bool:
        r"""
        Check if `self.dir` is not empty.

        Args:
            action (str): The action to perform if `self.dir` is not empty.
            Can be one of ("warn", "raise", "ignore"), default is "warn".
        """

        if action and action not in ("warn", "raise", "ignore"):
            raise ValueError(f"action should be one of warn, raise or ignore, but got {action}")
        if os.listdir(self.dir):
            if action == "warn":
                warn(
                    f"Directory `{self.dir}` is not empty",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
            if action == "raise":
                raise RuntimeError(f"Directory `{self.dir}` is not empty")
            return False
        return True

    @catch
    @on_main_process
    def save_checkpoint(self, name: str = "latest", epochs: int | None = None, save_best: bool = True) -> None:
        r"""
        Save the current model state and runner configuration to a checkpoint file.

        This method saves the complete state of the runner, including model weights,
        optimizer state, scheduler state, and configuration to the checkpoint directory.
        The method handles various checkpoint naming strategies based on parameters.

        Args:
            name: Base name of the checkpoint file (without extension).
                Defaults to `"latest"`, which creates a checkpoint at `{checkpoint_dir}/latest.pth`.
            epochs: Epoch number to record in the checkpoint.
                If None, uses `self.epochs` (current epoch counter).
            save_best: Whether to also save a copy of the checkpoint as `best.pth`
                when the current model performance is the best so far (`self.is_best` is True).
                Defaults to True.

        Note:
            - Checkpoints are saved only on the main process in distributed training.
            - The format is determined by the extension, defaulting to `.pth`.
            - The checkpoint includes the complete runner state from `self.state_dict()`.
            - Periodic checkpoints are saved based on `save_interval` configuration:
              If `self.config.save_interval > 0` and `(epochs + 1) % save_interval == 0`,
              the checkpoint will be copied to `{checkpoint_dir}/epoch-{epochs}.pth`.

        See Also:
            [`load_checkpoint`][danling.BaseRunner.load_checkpoint]: Load a saved checkpoint.
            [`state_dict`][danling.BaseRunner.state_dict]: Get the state dictionary to save.
        """

        epochs = epochs or self.epochs
        save_interval = self.config.get("save_interval", -1)
        latest_path = os.path.join(self.checkpoint_dir, f"{name}.pth")
        self.save(self.state_dict(), latest_path)
        if save_interval > 0 and (epochs + 1) % save_interval == 0:
            save_path = os.path.join(self.checkpoint_dir, f"epoch-{epochs}.pth")
            shutil.copy(latest_path, save_path)
        if save_best and self.is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pth")
            shutil.copy(latest_path, best_path)

    def load_config(
        self, checkpoint: Mapping | bytes | str | os.PathLike, overwrite: bool = False, *args, **kwargs
    ) -> None:
        r"""
        Load config from checkpoint.

        Args:
            checkpoint: Checkpoint (or its path) to load.
            overwrite: If `True`, overwrite the current config with the loaded config.
                Defaults to `False`.
            *args: Additional arguments to pass to `self.load`.
            **kwargs: Additional keyword arguments to pass to `self.load`.

        Raises:
            FileNotFoundError: If `checkpoint` does not exists.
        """

        if isinstance(checkpoint, (bytes, str, os.PathLike)):
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"checkpoint is set to {checkpoint!r} but does not exist")
            config = self.load(checkpoint, *args, **kwargs)
        elif isinstance(checkpoint, Mapping):
            config = checkpoint
        else:
            raise ValueError(f"checkpoint is set to {checkpoint!r} but is not a valid checkpoint")

        config = config.get("runner", config)
        self.config.merge(config, overwrite=overwrite)
        self.step_begin = config["steps"] + 1
        self.epoch_begin = config["epochs"] + 1

    def load_checkpoint(self, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> None:
        """
        Restore model, optimizer, and scheduler states from a saved checkpoint.

        This method loads the complete training state from a checkpoint, including:
        - Model weights and parameters
        - Optimizer state (learning rates, momentum buffers, etc.)
        - Learning rate scheduler state

        The method supports loading from either:
        - A file path (str, bytes, PathLike)
        - An already loaded checkpoint dictionary (Mapping)

        Args:
            checkpoint: The checkpoint to load from, which can be:
                - A path to the checkpoint file
                - A pre-loaded checkpoint dictionary
            *args: Additional arguments passed to the underlying `self.load` method.
            **kwargs: Additional keyword arguments passed to the underlying `self.load` method.
                      Common options include `map_location` for controlling device mapping.

        Raises:
            ValueError: If `self.model` is not defined (must initialize model before loading).
            ValueError: If `checkpoint` is not a recognized checkpoint format.
            FileNotFoundError: If the checkpoint file specified does not exist.

        Note:
            - This method attempts to unwrap the model before loading (useful for distributed training wrappers)
            - Unlike `load_config`, this method loads both model parameters and optimizer/scheduler states
            - Missing optimizer or scheduler states in the checkpoint will produce warnings but not errors
            - The checkpoint is stored in `self.config.checkpoint` for reference

        Example:
            ```python
            # Load from a file
            runner.load_checkpoint('checkpoints/best.pth')

            # Load with device mapping
            runner.load_checkpoint('checkpoints/latest.pth', map_location='cuda:0')
            ```

        See Also:
            [`from_checkpoint`][danling.BaseRunner.from_checkpoint]: Build a new runner instance from checkpoint.
            [`load_pretrained`][danling.BaseRunner.load_pretrained]: Load only model parameters from a checkpoint.
            [`save_checkpoint`][danling.BaseRunner.save_checkpoint]: Save current state to a checkpoint.
        """

        if self.model is None:
            raise ValueError("model is not defined")
        if isinstance(checkpoint, (bytes, str, os.PathLike)):
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"checkpoint is set to {checkpoint!r} but does not exist")
            ckpt = self.load(checkpoint, *args, **kwargs)
        elif isinstance(checkpoint, Mapping):
            ckpt = checkpoint
        else:
            raise ValueError(f"checkpoint is set to {checkpoint!r} but is not a valid checkpoint")

        state_dict = ckpt
        while "model" in state_dict or "module" in state_dict:
            state_dict = state_dict.get("model", state_dict)
            state_dict = state_dict.get("module", state_dict)
        self.unwrap(self.model).load_state_dict(state_dict)
        if self.optimizer is not None:
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            else:
                warn("optimizer is not in checkpoint", category=RuntimeWarning, stacklevel=2)
        if self.scheduler is not None:
            if "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            else:
                warn("scheduler is not in checkpoint", category=RuntimeWarning, stacklevel=2)
        self.config.checkpoint = checkpoint

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> BaseRunner:
        r"""
        Build BaseRunner from checkpoint.

        Args:
            checkpoint: Checkpoint (or its path) to load.
            *args: Additional arguments to pass to `cls.load`.
            **kwargs: Additional keyword arguments to pass to `cls.load`.

        Returns:
            (BaseRunner):
        """

        if isinstance(checkpoint, (bytes, str, os.PathLike)):
            ckpt = cls.load(checkpoint, *args, **kwargs)
        elif isinstance(checkpoint, Mapping):
            ckpt = checkpoint
        else:
            raise ValueError(f"checkpoint is set to {checkpoint} but is not a valid checkpoint")
        runner = cls(ckpt["runner"])
        runner.load_checkpoint(ckpt, override_config=False)
        return runner

    def load_pretrained(self, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> None:
        """
        Load model from pretrained checkpoint.

        This method only loads the model weights.

        Args:
            checkpoint: Pretrained checkpoint (or its path) to load.
            *args: Additional arguments to pass to `self.load`.
            **kwargs: Additional keyword arguments to pass to `self.load`.

        Raises:
            ValueError: If `model` is not defined.
            ValueError: If `checkpoint` is not a valid checkpoint.
            FileNotFoundError: If `checkpoint` does not exists.

        See Also:
            [`load_checkpoint`][danling.BaseRunner.load_checkpoint]: Load model, optimizer, and scheduler from
                checkpoint.
        """

        if self.model is None:
            raise ValueError("model is not defined")
        if isinstance(checkpoint, (bytes, str, os.PathLike)):
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"pretrained is set to {checkpoint!r} but does not exist")
            ckpt = self.load(checkpoint, *args, **kwargs)
        elif isinstance(checkpoint, Mapping):
            ckpt = checkpoint
        else:
            raise ValueError(f"pretrained is set to {checkpoint!r} but is not a valid checkpoint")

        state_dict = ckpt
        while "model" in state_dict or "module" in state_dict:
            state_dict = state_dict.get("model", state_dict)
            state_dict = state_dict.get("module", state_dict)
        self.unwrap(self.model).load_state_dict(state_dict)

    def get_step_result(self) -> NestedDict:
        result = self.meters.value()
        if self.metrics is not None:
            return self._merge_result(result, self.metrics.value())
        return result

    def get_epoch_result(self) -> NestedDict:
        result = self.meters.average()
        if self.metrics is not None:
            return self._merge_result(result, self.metrics.average())
        return result

    def _merge_result(self, meter_result, metric_result) -> NestedDict:
        for key, value in metric_result.items():
            if isinstance(value, (Mapping)) and len(value) == 1:
                value = next(iter(value.values()))
            metric_result[key] = value
        meter_result.update(metric_result)
        return meter_result

    def append_result(self, result: NestedDict, index: int | None = None) -> None:
        r"""
        Append result to `self.results`.

        Warnings:
            `self.results` is heavily relied upon for computing metrics.

            Failed to use this method may lead to unexpected behavior.
        """

        if index is None:
            index = self.epochs
            global __APPEND_RESULT_COUNTER__  # pylint: disable=global-statement
            __APPEND_RESULT_COUNTER__ += 1
            if index == 0 and __APPEND_RESULT_COUNTER__ > 1:
                warn(
                    """
                    Automatically set index to `self.epochs`.
                    Please ensure `self.epochs` updates before calling `append_result`
                    """,
                    category=RuntimeWarning,
                    stacklevel=2,
                )
        if index in self.results:
            self.results[index].merge(result)
        else:
            self.results[index] = result

    def print_result(self) -> None:
        r"""
        Print latest and best result.
        """

        print(f"latest result: {self.latest_result}")
        print(f"best result: {self.best_result}")

    def step_log(self, split: str, iteration: int, length: int | None = None):
        if length is None:
            length = len(self.dataloaders[split]) - 1
        result = self.get_step_result()
        print(self.format_step_result(result, split, iteration, length))
        if self.mode == "train":
            self.write_result(result, split)
        return result

    def format_step_result(
        self, result: NestedDict, split: str, steps: int, length: int, format_spec: str = ".4f"
    ) -> str:
        repr_str = ""
        if split is not None:
            if self.mode == "train":
                repr_str = f"training on {split} "
            elif self.mode == "evaluate":
                repr_str = f"evaluating on {split} "
            elif self.mode == "infer":
                repr_str = f"inferring on {split} "
            else:
                repr_str = f"running in {self.mode} mode on {split} "
        repr_str += f"[{steps}/{length}]\t"
        return repr_str + self.format_result(result, format_spec=format_spec)

    def format_epoch_result(
        self, result: NestedDict, epochs: int | None = None, epoch_end: int | None = None, format_spec: str = ".4f"
    ) -> str:
        epochs = epochs or self.epochs
        epoch_end = epoch_end or self.epoch_end
        repr_str = f"epoch [{epochs}/{epoch_end - 1}]" if epochs is not None and epoch_end else ""
        return repr_str + self.format_result(result, format_spec=format_spec)

    def format_result(self, result: Mapping, format_spec: str = ".4f") -> str:
        return format_result(result, format_spec=format_spec)

    def write_result(self, result: NestedDict, split: str, steps: int | None = None):
        if steps is None:
            steps = self.steps
        for name, score in result.all_items():
            name = name.replace(".", "/")
            if name == "loss" and isinstance(score, AverageMeter):
                score = score.avg
            if isinstance(score, Sequence):
                for i, s in enumerate(score):
                    self.write_score(f"{name}/{i}", s, split, steps)
            elif isinstance(score, Mapping):
                for k, s in score.items():
                    self.write_score(f"{name}/{k}", s, split, steps)
            else:
                self.write_score(name, score, split, steps)

    def write_score(self, name: str, score: float, split: str, steps: int):
        if self.writer:
            self.writer.add_scalar(f"{split}/{name}", score, steps)

    @catch
    @on_main_process
    def save_result(self) -> None:
        r"""
        Save result to `self.dir`.

        This method will save latest and best result to
        `self.dir/latest.json` and `self.dir/best.json` respectively.
        """

        results_path = os.path.join(self.dir, "results.json")
        self.save(
            {
                "name": self.name,
                "id": self.id,
                "timestamp": self.timestamp,
                "results": self.results,
            },
            results_path,
            indent=4,
        )
        ret = {"name": self.name, "id": self.id, "timestamp": self.timestamp}
        result = self.latest_result
        if isinstance(result, FlatDict):
            result = result.dict()
        # This is slower but ensure id is the first key
        if result is not None:
            ret.update(result)
        latest_path = os.path.join(self.dir, "latest.json")
        self.save(ret, latest_path, indent=4)
        if self.is_best:
            best_path = os.path.join(self.dir, "best.json")
            shutil.copy(latest_path, best_path)

    def unwrap(self, model: Any) -> Any:
        return model

    @cached_property
    def name(self) -> str:
        if "name" in self.config:
            return self.config["name"]
        return f"{self.config.experiment_name}-{self.config.run_name}"

    @cached_property
    def id(self) -> str:
        return f"{self.config.experiment_id:.8}{self.config.run_id:.8}"

    @cached_property
    def uuid(self) -> UUID:
        r"""
        UUID of the config.
        """

        return uuid5(self.run_uuid, self.id)

    @property
    def mode(self) -> RunnerMode:
        return self._mode

    @mode.setter
    def mode(self, mode: str | RunnerMode) -> None:
        if isinstance(mode, str):
            mode = RunnerMode(mode)
        self._mode = mode

    @property
    def config(self) -> Config:
        return self._config

    @property
    def batch_size(self) -> int:
        r"""
        Batch size.

        Note:
            If `self.dataloaders` is specified:
                - If `self.split` is specified, `batch_size` is the batch size of `self.split`.
                - If `self.train_splits` is specified, `batch_size` is the batch size of the first split.
                - If `self.evaluate_splits` is specified, `batch_size` is the batch size of the first split.
                - Otherwise, `batch_size` is the batch size of the first dataloader.
            Otherwise, `batch_size` is the batch size specified in `self.config.dataloader.batch_size`.
        """

        batch_size = None
        if self.dataloaders:
            if self.split:
                batch_size = self.dataloaders[self.split].batch_size
                if batch_size:
                    return batch_size
            train_splits = self.train_splits or []
            evaluate_splits = self.evaluate_splits or []
            splits = train_splits + evaluate_splits
            for split in splits:
                if split in self.dataloaders:
                    batch_size = self.dataloaders[split].batch_size
                    if batch_size:
                        return batch_size
            for dataloader in self.dataloaders.values():
                batch_size = dataloader.batch_size
                if batch_size:
                    return batch_size
        if not batch_size:
            batch_size = self.config.get("dataloader.batch_size")
        if batch_size:
            return batch_size
        raise AttributeError("batch_size could not be inferred and is not in config")

    @property
    def batch_size_equivalent(self) -> int:
        r"""
        The effective total batch size across all processes and gradient accumulation steps.

        This property calculates the total number of examples processed in a single
        optimization step by accounting for:
        - The base batch size per device
        - The number of processes (world_size for distributed training)
        - The number of gradient accumulation steps

        This is particularly useful for:
        - Learning rate scaling using the linear scaling rule
        - Calculating the total steps in an epoch
        - Estimating memory requirements across all devices

        Returns:
            (int): Effective batch size calculated as `batch_size * world_size * accum_steps`
        """

        return self.batch_size * self.world_size * self.accum_steps

    @property
    def epochs(self) -> int:
        r"""
        Current epoch.
        """
        return self.config.epochs

    @epochs.setter
    def epochs(self, epochs: int) -> None:
        self.config.epochs = epochs

    @property
    def epoch_begin(self) -> int:
        return self.config.epoch_begin

    @epoch_begin.setter
    def epoch_begin(self, epoch_begin: int) -> None:
        self.config.epoch_begin = epoch_begin

    @property
    def epoch_end(self) -> int | None:
        return self.config.epoch_end

    @epoch_end.setter
    def epoch_end(self, epoch_end: int) -> None:
        self.config.epoch_end = epoch_end

    @property
    def total_epochs(self) -> int:
        r"""
        Number of training epochs.
        """
        if self.epoch_end:
            return self.epoch_end - self.epoch_begin
        raise ValueError("epoch_end is not specified")

    @property
    def steps(self) -> int:
        r"""
        Current step.
        """
        return self.config.steps

    @steps.setter
    def steps(self, steps: int) -> None:
        self.config.steps = steps

    @property
    def step_begin(self) -> int:
        return self.config.step_begin

    @step_begin.setter
    def step_begin(self, step_begin: int) -> None:
        self.config.step_begin = step_begin

    @property
    def step_end(self) -> int | None:
        return self.config.step_end

    @step_end.setter
    def step_end(self, step_end: int) -> None:
        self.config.step_end = step_end

    @property
    def total_steps(self) -> int:
        r"""
        Number of training steps.
        """
        if self.step_end:
            return self.step_end - self.step_begin
        train_splits = self.train_splits
        if self.dataloaders and train_splits:
            return self.total_epochs * sum(len(self.dataloaders[split]) for split in train_splits)
        if self.datasets and train_splits:
            datasets_length = sum(len(self.datasets[split]) for split in self.train_splits)
            return self.total_epochs * ceil(datasets_length / self.batch_size / self.world_size)
        raise ValueError("total_steps could not be inferred and is not in config")

    @property
    def accum_steps(self) -> int:
        r"""
        Number of steps to accumulate gradients.
        """

        return self.config.get("accum_steps", 1)

    @property
    def progress(self) -> float:
        r"""
        Training Progress.

        Raises:
            RuntimeError: If no terminal is defined.
        """

        return self.steps / self.total_steps

    @property
    def train_splits(self) -> list[str]:
        if self._train_splits is not None:
            return self._train_splits
        if "train_splits" in self.config:
            self._train_splits = self.config["train_splits"]
            return self._train_splits
        if self.datasets:
            train_splits = [k for k, d in self.datasets.items() if getattr(d, "train", False)]
            if train_splits:
                self._train_splits = train_splits
                return train_splits
            dataset_splits = {k: getattr(d, "split", None) for k, d in self.datasets.items()}
            train_splits = [k for k, s in dataset_splits.items() if s == "train"]
            if train_splits:
                self._train_splits = train_splits
                return train_splits
        raise ValueError("train_splits could not be inferred and is not in config")

    @train_splits.setter
    def train_splits(self, train_splits: list[str]) -> None:
        self._train_splits = train_splits

    @property
    def evaluate_splits(self) -> list[str]:
        if self._evaluate_splits is not None:
            return self._evaluate_splits
        if "evaluate_splits" in self.config:
            self._evaluate_splits = self.config["evaluate_splits"]
            return self._evaluate_splits
        if self.datasets:
            evaluate_splits = [k for k in self.datasets if k not in self.train_splits]
            self._evaluate_splits = evaluate_splits
            return evaluate_splits
        raise ValueError("evaluate_splits could not be inferred and is not in config")

    @evaluate_splits.setter
    def evaluate_splits(self, evaluate_splits: list[str]) -> None:
        self._evaluate_splits = evaluate_splits

    @property
    def world_size(self) -> int:
        r"""
        Number of processes.
        """

        return 1

    @property
    def rank(self) -> int:
        r"""
        Process index of all processes.
        """

        return 0

    @property
    def local_rank(self) -> int:
        r"""
        Process index of local processes.
        """

        return 0

    @property
    def distributed(self) -> bool:
        r"""
        If runner is running in distributed mode.
        """

        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        r"""
        If current process is the main process of all processes.
        """

        return self.rank == 0

    @property
    def is_local_main_process(self) -> bool:
        r"""
        If current process is the main process of local processes.
        """

        return self.local_rank == 0

    @property
    def best_fn(self) -> Callable:
        r"""
        Function to determine the best score from a list of scores.

        By default, the `best_fn` returns `min` if `self.config.score_name` is `loss`,
        otherwise, returns `max`.

        Subclass can override this method to accommodate needs, such as `min`.
        """

        return max if self.config.score_name != "loss" else min

    @property
    def best_index(self) -> int:
        r"""
        Find the best index from all scores.
        """

        if not self.scores:
            return 0
        return self.best_fn(reversed(self.scores), key=self.scores.get)

    @property
    def latest_result(self) -> NestedDict | None:
        r"""
        Latest result.
        """

        if not self.results:
            return None
        latest_index = next(reversed(self.results if PY38_PLUS else list(self.results)))  # type: ignore
        ret = self.results[latest_index].clone()
        ret["index"] = latest_index
        return ret

    @property
    def best_result(self) -> NestedDict | None:
        r"""
        Best result.
        """

        if not self.results:
            return None
        best_index = self.best_index
        ret = self.results[best_index].clone()
        ret["index"] = best_index
        return ret

    @property
    def scores(self) -> FlatDict | None:
        r"""
        All scores.

        Scores are extracted from results by `score_split` and `runner.config.score_name`,
        following `[r[score_split][self.config.score_name] for r in self.results]`.

        Scores are considered as the index of the performance of the model.
        It is useful to determine the best model and the best hyper-parameters.

        `score_split` is defined in `self.config.score_split`.
        If it is not set, `DanLing` will use `val` or `validate` if they appear in the `latest_result`.
        If `DanLing` still could not find, it will fall back to the second key in the `latest_result`
        if it contains more that one element, or the first key.

        Note that certain keys are ignored when falling back, they are defined in {defaults.IGNORED_NAMES_IN_METRICS}.
        """

        if not self.results:
            return None
        subsets = [i for i in self.latest_result.keys() if i not in defaults.IGNORED_NAMES_IN_METRICS]  # type: ignore
        score_split = self.config.get("score_split")
        if score_split is None and "val" in subsets:
            score_split = "val"
        if score_split is None and "validate" in subsets:
            score_split = "validate"
        if score_split is None:
            score_split = subsets[1] if len(subsets) > 1 else subsets[0]
        return FlatDict({k: v[score_split][self.config.score_name] for k, v in self.results.items()})

    @property
    def latest_score(self) -> float | None:
        r"""
        Latest score.
        """

        if not self.results:
            return None
        if not PY38_PLUS:
            return next(reversed(list(self.scores.values())))  # type: ignore
        return next(reversed(self.scores.values()))  # type: ignore

    @property
    def best_score(self) -> float | None:
        r"""
        Best score.
        """

        if not self.results:
            return None
        return self.scores[self.best_index]  # type: ignore

    @property
    def is_best(self) -> bool:
        r"""
        Determines whether the latest model checkpoint is the best performing one.

        This property compares the `latest_score` with the `best_score` to determine
        if the most recent model evaluation represents the best performance achieved
        so far. The comparison uses a small epsilon (1e-7) to handle floating point
        comparison issues.

        The property is used by:
        - `save_checkpoint` method to determine whether to save a "best.pth" checkpoint
        - Result logging and reporting mechanisms
        - Early stopping implementations

        Returns:
            (bool): True if the latest evaluation results represent the best performance
                 or if no results exist yet, False otherwise.

        Note:
            - Returns True if no results exist (first evaluation)
            - Uses numerical comparison with tolerance to avoid floating point issues
            - The definition of "best" depends on the metric (maximization for accuracy,
              minimization for loss, etc.) and is handled by the `best_fn` property

        See Also:
            [`best_score`][danling.BaseRunner.best_score]: The best score achieved so far.
            [`latest_score`][danling.BaseRunner.latest_score]: The most recent score.
            [`best_fn`][danling.BaseRunner.best_fn]: Function defining "best" (min/max).
        """

        if not self.results:
            return True
        try:
            return abs(self.latest_score - self.best_score) < 1e-7  # type: ignore
        except TypeError:
            return True

    @cached_ensure_dir
    def dir(self) -> str:
        r"""
        Directory of the run.
        """

        if "dir" in self.config:
            return self.config.dir
        return os.path.join(self.project_root, f"{self.name}-{self.id}", self.timestamp)

    @cached_ensure_parent_dir
    def log_file(self) -> str:
        r"""
        Path of log file.
        """

        if "log_file" in self.config:
            return self.config.log_file
        return os.path.join(self.dir, "run.log")

    @cached_ensure_dir
    def checkpoint_dir(self) -> str:
        r"""
        Directory of checkpoints.
        """

        if "checkpoint_dir" in self.config:
            return self.config.checkpoint_dir
        return os.path.join(self.dir, self.config.checkpoint_dir_name)

    # def __getattribute__(self, name) -> Any:
    #     if name in ("__class__", "__dict__"):
    #         return super().__getattribute__(name)
    #     if name in self.__dict__:
    #         return self.__dict__[name]
    #     if name in dir(self):
    #         return super().__getattribute__(name)
    #     if "config" in self and name in self.config:
    #         return self.config[name]
    #     return super().__getattribute__(name)

    def __getattr__(self, name) -> Any:
        if self.inited:
            if name in self.config:
                return self.config[name]
            if name in dir(self.config):
                return getattr(self.config, name)
        return super().__getattribute__(name)

    def __setattr__(self, name, value) -> None:
        if name in self.__dict__:
            if isinstance(self.__dict__[name], Variable):
                self.__dict__[name].set(value)
            else:
                self.__dict__[name] = value
            return
        if name in dir(self):
            if isinstance(super().__getattribute__(name), Variable):
                super().__getattribute__(name).set(value)
            else:
                object.__setattr__(self, name, value)
            return
        if self.inited:
            if name in self.config:
                if isinstance(self.config[name], Variable):
                    self.config[name].set(value)
                else:
                    self.config[name] = value
                return
            if name in dir(self.config):
                setattr(self.config, name, value)
                return
        object.__setattr__(self, name, value)

    def __contains__(self, name) -> bool:
        return name in dir(self) or ("config" in self.__dict__ and name in dir(self.config))

    def __repr__(self):
        lines = []
        for key, value in self.__dict__.items():
            if key.startswith("_") or key.endswith("_") or key == "inited":
                continue
            value_str = repr(value)
            value_str = self._add_indent(value_str)
            lines.append("(" + key + "): " + value_str)

        main_str = self.__class__.__name__ + "("
        if lines:
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def _add_indent(self, text):
        lines = text.split("\n")
        # don't do anything for single-line stuff
        if len(lines) == 1:
            return text
        first = lines.pop(0)
        # add 2 spaces to each line but the first
        lines = [(2 * " ") + line for line in lines]
        lines = "\n".join(lines)
        lines = first + "\n" + lines
        return lines
