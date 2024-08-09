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

import builtins
import logging
import os
import random
from collections.abc import Callable, Mapping
from functools import cached_property
from math import ceil
from typing import Any
from warnings import warn

from chanfig import FlatDict, NestedDict

from danling.data import DataLoaderDict
from danling.metrics import AverageMeters
from danling.typing import File, PathStr
from danling.utils import RoundDict, catch, load, save

try:
    from numpy import random as np_random
except ImportError:
    np_random = None

from .checkpoints import CheckpointManager, FileCheckpointManager
from .config import RunnerConfig
from .mixins import ResultMixin, RuntimeMixin
from .state import RunnerElasticState, RunnerRNGState, RunnerState, RunnerTrainState
from .utils import (
    MetaRunner,
    RunnerMode,
    get_time_str,
    on_main_process,
)


class BaseRunner(RuntimeMixin, ResultMixin, metaclass=MetaRunner):
    """
    Backend-agnostic runner state and orchestration utilities.

    `BaseRunner` intentionally keeps only the shared runtime contract used by
    concrete runners such as `TorchRunner`:

    - configuration and process lifecycle bootstrap
    - datasets/dataloaders/result containers
    - checkpoint/result persistence helpers
    - progress and score bookkeeping

    Lifecycle:
    - `initialize_state`: construct and bind checkpointed state objects
    - `initialize_runtime`: initialize runtime metadata and mutable containers
    - `initialize_services`: initialize shared services (distributed/log/seed/etc.)
    """

    state: RunnerState
    config: RunnerConfig

    train_state: RunnerTrainState
    elastic_state: RunnerElasticState
    rng_state: RunnerRNGState

    model: Any
    ema: Any | None = None
    criterion: Callable | None = None
    optimizer: Any | None = None
    scheduler: Any | None = None

    datasets: FlatDict
    dataloaders: FlatDict
    split: str | None = None

    results: RoundDict
    meters: AverageMeters
    metrics: Any | None = None
    train_metrics: Any | None = None
    evaluate_metrics: Any | None = None

    logger: logging.Logger | None = None
    writer: Any | None = None

    checkpoint_manager: CheckpointManager

    timestamp: str

    def __init__(self, config: RunnerConfig | Mapping[str, Any]) -> None:
        self.initialize_state(config)
        self.initialize_runtime()
        self.initialize_services()

    def initialize_state(self, config: RunnerConfig | Mapping[str, Any]) -> None:
        """Construct and bind checkpointed state objects."""

        state = RunnerState(config=config)
        self.state = state
        self.config = state.config
        self.train_state = state.train
        self.elastic_state = state.elastic
        self.rng_state = state.rng

    def initialize_runtime(self) -> None:
        """Initialize non-checkpointed runtime metadata and mutable containers."""

        self.timestamp = get_time_str()
        self.name = str(self.config.get("name", f"{self.lineage}-{self.experiment}"))
        self.datasets = FlatDict()
        self.dataloaders = DataLoaderDict()
        self.results = RoundDict()
        self.meters = AverageMeters()
        self.mode = RunnerMode.train
        self.checkpoint_manager = FileCheckpointManager(self)

    def initialize_services(self) -> None:
        """Initialize shared runtime services."""

        self.init_distributed()

        if self.config.seed is not None:
            self.set_seed()

        if self.config.deterministic:
            self.set_deterministic()

        if self.config.log:
            self.init_logging()

        if self.config.tensorboard:
            self.init_tensorboard()

        self.init_print()
        self.save_metadata()

    def __post_init__(self) -> None:
        """Hook called after `__init__` by `MetaRunner`."""

    def auto_restore(self) -> None:
        """Auto-load resume/pretrained sources declared in config.

        Precedence:
            `config.resume` > `config.auto_resume` > `config.pretrained`.
        """

        resume_source = self.config.get("resume")
        auto_resume = bool(self.config.get("auto_resume", False))
        pretrained_source = self.config.get("pretrained")

        specified_count = int(bool(resume_source)) + int(auto_resume) + int(bool(pretrained_source))
        if specified_count > 2:
            warn(
                "`config.resume`, `config.auto_resume`, and `config.pretrained` are all set; "
                "precedence is `resume` > `auto_resume` > `pretrained`",
                RuntimeWarning,
                stacklevel=2,
            )

        if resume_source:
            self.load_checkpoint(resume_source)
            return

        if auto_resume:
            backend = str(self.config.get("checkpoint.backend", "auto")).strip().lower()
            if backend == "dcp":
                self.load_checkpoint(os.path.join(self.checkpoint_dir, "latest"))
            else:
                self.load_checkpoint(os.path.join(self.checkpoint_dir, "latest.pth"))
            return

        if pretrained_source:
            self.load_pretrained(pretrained_source)

    def init_distributed(self) -> None:
        """Initialize distributed environment.

        Subclasses should override this if they support distributed execution.
        """

    @on_main_process
    def init_tensorboard(self, *args, **kwargs) -> None:
        """Initialize tensorboard writer."""

        warn(
            "tensorboard is enabled, but this runner does not initialize a tensorboard writer",
            RuntimeWarning,
            stacklevel=2,
        )

    def set_seed(self, seed: int | None = None, bias: int | bool | None = None) -> int:
        """Set python/numpy RNG seeds and snapshot RNG state.

        Args:
            seed: Base seed. Defaults to `self.config.seed`.
            bias: Optional per-process bias. `None` uses `self.rank`.

        Returns:
            The process-local seed after applying bias.
        """

        base_seed = self.config.seed if seed is None else seed
        if base_seed is None:
            raise ValueError("cannot set seed: no seed is configured and no seed argument was provided")
        base_seed = int(base_seed)

        self.config.seed = base_seed

        process_seed = base_seed
        if bias is None:
            bias = self.rank
        if bias:
            process_seed += int(bias)

        random.seed(process_seed)
        if np_random is not None:
            np_random.seed(process_seed)

        self.rng_state.python = random.getstate()
        self.rng_state.numpy = np_random.get_state() if np_random is not None else None
        return process_seed

    def set_deterministic(self) -> None:
        """Enable deterministic behavior in subclass-specific backends."""

    def train(self, *args, **kwargs):
        """Run top-level training workflow."""

        raise NotImplementedError

    def train_epochs(self, *args, **kwargs):
        """Run epoch-mode training workflow."""

        raise NotImplementedError

    def train_epoch(self, *args, **kwargs):
        """Run one training epoch on a split."""

        raise NotImplementedError

    def train_steps(self, *args, **kwargs):
        """Run step-mode training workflow."""

        raise NotImplementedError

    def train_step(self, *args, **kwargs):
        """Run one training micro-step."""

        raise NotImplementedError

    def backward(self, loss, *args, **kwargs) -> None:
        """Run backward pass for one micro-step loss."""

        raise NotImplementedError

    def step(self, *args, **kwargs) -> None:
        """Advance optimizer/scheduler state when accumulation is ready."""

        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        """Run top-level evaluation workflow."""

        raise NotImplementedError

    def evaluate_epoch(self, *args, **kwargs):
        """Run one full evaluation epoch on a split."""

        raise NotImplementedError

    def evaluate_steps(self, *args, **kwargs):
        """Run bounded evaluation steps on a split."""

        raise NotImplementedError

    def evaluate_step(self, *args, **kwargs):
        """Run one evaluation step."""

        raise NotImplementedError

    def infer(self, *args, **kwargs):
        """Run top-level inference workflow."""

        raise NotImplementedError

    def infer_step(self, *args, **kwargs):
        """Run one inference step."""

        raise NotImplementedError

    def unwrap(self, model: Any) -> Any:
        """Return an unwrapped model object."""

        return model

    def state_dict(self, cls: type = dict) -> Mapping:
        """
        Build checkpoint payload for runner/runtime state.

        Returns:
            Mapping with `runner` (config snapshot) and `state` (runtime state).
        """

        current_world_size = int(self.world_size)
        if (
            self.elastic_state.last_seen_world_size > 0
            and self.elastic_state.last_seen_world_size != current_world_size
        ):
            self.elastic_state.membership_version += 1
        self.elastic_state.last_seen_world_size = current_world_size

        self.rng_state.python = random.getstate()
        self.rng_state.numpy = np_random.get_state() if np_random is not None else None

        state = self.state.state_dict()
        if cls is not dict:
            state = cls(state)

        return cls(runner=self.config.dict(), state=state)

    def load_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore runner state and process RNG from checkpoint payload.

        Notes:
            Raises when semantic config diff is detected between checkpoint and
            current runner config.
        """

        runner_config = checkpoint.get("runner")
        if runner_config is not None:
            checkpoint_config = RunnerConfig(runner_config).canonical()
            current_config = self.config.canonical()
            semantic_diff = NestedDict(checkpoint_config).diff(current_config).dict()
            if semantic_diff:
                raise ValueError(
                    "cannot load checkpoint: runner config is semantically different from current config; "
                    f"start a new experiment or align config. diff={semantic_diff}"
                )

        state_dict = checkpoint.get("state") or {}
        self.state.load_state_dict(dict(state_dict))

        rng_state = state_dict.get("rng")
        if isinstance(rng_state, Mapping) and "python" in rng_state and self.rng_state.python is not None:
            random.setstate(self.rng_state.python)

        if (
            np_random is not None
            and isinstance(rng_state, Mapping)
            and "numpy" in rng_state
            and self.rng_state.numpy is not None
        ):
            np_random.set_state(self.rng_state.numpy)

    @catch
    @on_main_process
    def save_checkpoint(
        self,
        name: str = "latest",
        epochs: int | None = None,
        save_best: bool = True,
        last_step: bool = False,
    ) -> None:
        """Persist runner state as a checkpoint.

        This path is intentionally non-fatal: save failures are logged by
        `@catch` and training must continue.
        """

        epochs = self.train_state.epoch if epochs is None else epochs
        self.checkpoint_manager.save_checkpoint(name=name, epochs=epochs, save_best=save_best, last_step=last_step)

    def load_checkpoint(self, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> None:
        """Restore model/optimizer/scheduler/runtime state from checkpoint."""

        ckpt = self.read_checkpoint(checkpoint, *args, **kwargs)
        ckpt = self.adapt_checkpoint_payload_for_load(ckpt)

        self.load_state_dict(ckpt)
        if "model" in ckpt:
            self.load_model(ckpt["model"], *args, **kwargs)
        elif "model_parts" in ckpt:
            self.load_model(ckpt["model_parts"], *args, **kwargs)
        else:
            raise ValueError(
                "cannot restore model: checkpoint has no model state\n"
                "Use `load_pretrained` only for model-only checkpoints with model/ema payloads"
            )
        self.load_ema(ckpt.get("ema"), *args, **kwargs)
        self.load_optimizer(ckpt.get("optimizer"), *args, **kwargs)
        self.load_scheduler(ckpt.get("scheduler"), *args, **kwargs)
        if isinstance(checkpoint, (str, bytes, os.PathLike)):
            self.config.resume = os.fsdecode(checkpoint)

    def adapt_checkpoint_payload_for_save(self, checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
        """Optional payload adapter hook before backend save."""

        return checkpoint

    def adapt_checkpoint_payload_for_load(self, checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
        """Optional payload adapter hook after backend read and before state restore."""

        return checkpoint

    def load_model(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
        """Load model state."""
        self.unwrap(self.model).load_state_dict(state_dict, *args, **kwargs)

    def load_ema(self, state_dict: Mapping[str, Any] | None, *args, **kwargs) -> None:
        """Load EMA state."""
        if self.ema is None:
            return
        if state_dict is None:
            raise ValueError(
                "cannot restore EMA: checkpoint has no EMA state\n"
                "Use `load_pretrained` for model-only checkpoints instead of `load_checkpoint`"
            )
        self.ema.load_state_dict(state_dict, *args, **kwargs)

    def load_optimizer(self, state_dict: Mapping[str, Any] | None, *args, **kwargs) -> None:
        """Load optimizer state."""
        if self.optimizer is None:
            return
        if state_dict is None:
            raise ValueError(
                "cannot restore optimizer: checkpoint has no optimizer state\n"
                "Use `load_pretrained` for model-only checkpoints instead of `load_checkpoint`"
            )
        self.optimizer.load_state_dict(state_dict, *args, **kwargs)

    def load_scheduler(self, state_dict: Mapping[str, Any] | None, *args, **kwargs) -> None:
        """Load scheduler state."""
        if self.scheduler is None:
            return
        if state_dict is None:
            raise ValueError(
                "cannot restore scheduler: checkpoint has no scheduler state\n"
                "Use `load_pretrained` for model-only checkpoints instead of `load_checkpoint`"
            )
        self.scheduler.load_state_dict(state_dict, *args, **kwargs)

    def load_pretrained(self, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> None:
        """Load pretrained model weights from checkpoint payload/path.

        When checkpoint payload provides EMA weights (`ema`), EMA is preferred as
        the pretrained source. Otherwise `model` is used.
        """

        if self.model is None:
            raise ValueError("cannot load pretrained weights: model is not initialized")

        ckpt = self.read_checkpoint(checkpoint, *args, **kwargs)
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

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> BaseRunner:
        """Instantiate runner from checkpoint config and restore full state."""

        config = cls.read_config(checkpoint, *args, **kwargs)
        if isinstance(checkpoint, (bytes, str, os.PathLike)):
            config.resume = os.fsdecode(checkpoint)
        runner = cls(config)
        if isinstance(checkpoint, Mapping):
            runner.load_checkpoint(checkpoint, *args, **kwargs)
        return runner

    @classmethod
    def read_config(
        cls,
        checkpoint: Mapping | bytes | str | os.PathLike,
        *args,
        **kwargs,
    ) -> RunnerConfig:
        """
        Read runner config from checkpoint mapping or file path.

        Note:
            BaseRunner only accepts file checkpoints for path input.
            Backend-specific directory checkpoints must be handled in subclasses.
        """

        if isinstance(checkpoint, Mapping):
            ckpt = checkpoint
        elif isinstance(checkpoint, (bytes, str, os.PathLike)):
            checkpoint_id = os.fspath(checkpoint)
            if os.path.isfile(checkpoint_id):
                ckpt = cls.load(checkpoint, *args, **kwargs)
            else:
                raise ValueError(
                    f"cannot read config from checkpoint path for {cls.__name__}: path must be a file; "
                    "use a backend-specific runner for directory-style checkpoints"
                )
        else:
            raise ValueError(
                "invalid checkpoint input: expected a mapping or path, "
                f"got {type(checkpoint).__name__}: {checkpoint!r}"
            )

        if "runner" not in ckpt:
            raise ValueError(
                "cannot read runner config: checkpoint is missing key 'runner'; "
                "use from_pretrained(...) for model-only checkpoints"
            )
        return RunnerConfig(ckpt["runner"])

    @classmethod
    def from_pretrained(
        cls,
        config: RunnerConfig | Mapping[str, Any],
        checkpoint: Mapping | bytes | str | os.PathLike,
        *args,
        **kwargs,
    ) -> BaseRunner:
        """Build a runner from config and load model weights only."""

        prepared = RunnerConfig(config)
        prepared.resume = None
        if isinstance(checkpoint, (bytes, str, os.PathLike)):
            prepared.pretrained = os.fsdecode(checkpoint)
        else:
            prepared.pretrained = None
        runner = cls(prepared)
        if isinstance(checkpoint, Mapping):
            runner.load_pretrained(checkpoint, *args, **kwargs)
        return runner

    def read_checkpoint(self, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> Mapping[str, Any]:
        """Normalize checkpoint input into an in-memory mapping payload."""
        if isinstance(checkpoint, (bytes, str, os.PathLike)):
            return self.load(checkpoint, *args, **kwargs)
        if isinstance(checkpoint, Mapping):
            return checkpoint
        raise ValueError(
            "invalid checkpoint input: expected a mapping or path, " f"got {type(checkpoint).__name__}: {checkpoint!r}"
        )

    @catch
    def save(self, obj: Any, file: PathStr, main_process_only: bool = True, *args, **kwargs) -> File:
        """Save an object with optional main-process guard."""

        if (main_process_only and self.is_main_process) or not main_process_only:
            return save(obj, file, *args, **kwargs)
        return file

    @staticmethod
    def load(file: PathStr, *args, **kwargs) -> Any:
        """Load an object from a file path."""

        return load(file, *args, **kwargs)

    def close(self, timeout: float | None = None) -> bool:
        """Finalize checkpoint/log/writer resources before shutdown."""

        if timeout is None:
            timeout = self.config.get("checkpoint.wait_timeout")

        drained = self.checkpoint_manager.close(timeout=timeout)

        if not drained:
            warn("runner close: timed out while draining async checkpoints", RuntimeWarning, stacklevel=2)

        writer = self.writer
        if writer is not None:
            writer.flush()
            writer.close()
            self.writer = None

        logger = self.logger
        if logger is not None:
            handlers = list(logger.handlers)
            for handler in handlers:
                handler.flush()
                handler.close()
                logger.removeHandler(handler)
            self.logger = None

        builtins.print = getattr(builtins, "_print", builtins.print)

        return drained

    @property
    def mode(self) -> RunnerMode:
        return self._mode

    @mode.setter
    def mode(self, mode: str | RunnerMode) -> None:
        if isinstance(mode, str):
            mode = RunnerMode(mode)
        self._mode = mode

    @property
    def batch_size(self) -> int:
        """Infer batch size from config or first dataloader."""
        batch_size = self.config.get("dataloader.batch_size")
        if batch_size is not None:
            return batch_size

        if self.dataloaders:
            loader = next(iter(self.dataloaders.values()))
            batch_size = getattr(loader, "batch_size", None)
            if batch_size is not None:
                return batch_size

        raise AttributeError("batch_size could not be inferred and is not in config")

    @property
    def epochs(self) -> int | None:
        """Configured epoch budget, if present."""
        epochs = self.config.get("epochs")
        if epochs is not None:
            return epochs
        return None

    @epochs.setter
    def epochs(self, epochs: int) -> None:
        self.config.epochs = epochs

    @property
    def steps(self) -> int | None:
        """Configured/derived optimizer-step budget."""
        steps = self.config.get("steps")
        if steps is not None:
            return steps
        if self.epochs is not None and self.dataloaders:
            steps_per_epoch = 0
            for split in self.train_splits:
                split_micro_steps = len(self.dataloaders[split])
                steps_per_epoch += (split_micro_steps + self.accum_steps - 1) // self.accum_steps
            return steps_per_epoch * self.epochs
        return None

    @steps.setter
    def steps(self, steps: int) -> None:
        self.config.steps = steps

    @cached_property
    def is_step_mode(self) -> bool:
        """Whether runner is in step mode (`epochs` is unset)."""
        return self.epochs is None

    @cached_property
    def accum_steps(self) -> int:
        """Gradient accumulation steps."""
        return self.config.get("accum_steps", 1)

    @cached_property
    def precision(self) -> str | None:
        """Autocast precision mode."""
        return self.config.get("precision")

    @cached_property
    def max_grad_value(self) -> float | None:
        """Gradient value clipping threshold."""
        return self.config.get("max_grad_value")

    @cached_property
    def max_grad_norm(self) -> float | None:
        """Gradient norm clipping threshold."""
        return self.config.get("max_grad_norm")

    @cached_property
    def skip_nonfinite_grad(self) -> bool:
        """Whether to skip optimizer updates when gradients are non-finite."""
        return self.config.get("skip_nonfinite_grad", False)

    @cached_property
    def patience(self) -> int | float:
        """Early-stop patience in epoch mode."""
        return self.config.get("patience", float("inf"))

    @property
    def progress(self) -> float:
        """Normalized training progress in `[0, 1]`."""
        if self.steps is not None:
            return self.train_state.global_step / self.steps
        if self.epochs is not None:
            return self.train_state.epoch / self.epochs
        raise ValueError("cannot compute progress: neither `steps` nor `epochs` is configured")

    @cached_property
    def train_splits(self) -> list[str]:
        """Configured or inferred training split names."""
        if "train_splits" in self.config:
            return sorted(set(self.config["train_splits"]))
        if self.datasets:
            inferred: list[str] = []
            for split, dataset in self.datasets.items():
                if split == "train":
                    inferred.append(split)
                if getattr(dataset, "train", False):
                    inferred.append(split)
                if getattr(dataset, "split", None) == "train":
                    inferred.append(split)
            return sorted(set(inferred))
        return []

    @cached_property
    def evaluate_splits(self) -> list[str]:
        """Configured or inferred evaluation split names."""
        if "evaluate_splits" in self.config:
            return sorted(set(self.config["evaluate_splits"]))
        if self.datasets:
            return sorted(split for split in self.datasets if split not in self.train_splits)
        return []

    @cached_property
    def checkpoint_interval(self) -> int:
        """Checkpoint cadence in optimizer steps (step mode) or epochs (epoch mode)."""
        configured = self.config.get("checkpoint.interval")
        if configured is not None:
            return configured
        if self.epochs is not None:
            return 1
        if self.steps is not None:
            return max(ceil(self.steps / 20), 1)
        return 8_192

    @cached_property
    def log_interval(self) -> int:
        """Step logging cadence."""
        configured = self.config.get("log_interval")
        if configured is not None:
            return configured
        if self.steps is not None:
            return max(ceil(self.steps / 100), 1)
        return 1_024
