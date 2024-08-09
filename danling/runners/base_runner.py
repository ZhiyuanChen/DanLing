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

import logging
import os
import random
import shutil
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from math import ceil
from typing import Any, cast
from warnings import warn

from chanfig import FlatDict, NestedDict

from danling.data import DataLoaderDict
from danling.metrics import AverageMeter, AverageMeters
from danling.typing import File, PathStr
from danling.utils import RoundDict, catch, load, save

try:
    from numpy import random as np_random
except ImportError:
    np_random = None  # type: ignore[assignment]

from .checkpoints import CheckpointManager, FileCheckpointManager
from .config import RunnerConfig
from .fault_tolerance import FaultTolerance
from .state import RunnerElasticState, RunnerRNGState, RunnerState, RunnerTrainState
from .supervisor import RunnerSupervisor
from .utils import (
    MetaRunner,
    RunnerMode,
    format_result,
    get_git_hash,
    get_time_str,
    on_main_process,
)
from .workspace import RunnerWorkspace


class BaseRunner(metaclass=MetaRunner):
    """
    Backend-agnostic runner state and orchestration utilities.

    `BaseRunner` intentionally keeps only the shared runtime contract used by
    concrete runners such as `TorchRunner`:

    - configuration and process lifecycle bootstrap
    - datasets/dataloaders/result containers
    - checkpoint/result persistence helpers
    - progress and score bookkeeping

    Concrete runners are expected to customize runtime behavior through the
    explicit training/checkpoint hooks below, not by overriding bootstrap
    internals.

    **Construction lifecycle:**

    1. Normalize config and create `RunnerState`.
    2. Bind workspace, containers, default `FileCheckpointManager`, and
       supervisor.
    3. Call early service hooks in order: `init_distributed`,
       `init_checkpoint_manager`, `init_fault_tolerance`,
       `init_garbage_collection`.
    4. Apply seed/determinism policy.
    5. Initialize logging, TensorBoard/W&B, print routing, signal handlers, and
       heartbeat.
    6. `MetaRunner` calls `__post_init__`. Concrete runners such as
       `TorchRunner` materialize models, optimizers, schedulers, and resume
       checkpoints there before delegating back to `BaseRunner.__post_init__`
       for metadata persistence.

    **Override rule:** early hooks run while the runner is only partially
    constructed; model/runtime hooks run in concrete `__post_init__`; loop
    hooks (`train_step`, `evaluate_step`, `infer_step`) run after all runtime
    components are bound.

    Attributes:
        state: Checkpointable aggregate state object.
        config: Runner configuration.
        train_state: Training progress counters.
        elastic_state: Torchelastic restart metadata.
        rng_state: Python/NumPy/Torch RNG snapshots.
        datasets: Dataset mapping keyed by split.
        dataloaders: Dataloader mapping keyed by split.
        checkpoint_manager: Active checkpoint backend manager.
        workspace: Workspace, logging, metadata, and print-routing helper.
        supervisor: Signal, heartbeat, and garbage-collection helper.
        ft: Optional fault-tolerance runtime handle.
    """

    state: RunnerState
    config: RunnerConfig
    train_state: RunnerTrainState
    elastic_state: RunnerElasticState
    rng_state: RunnerRNGState

    model: Any | None = None
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
    wandb: Any | None = None

    checkpoint_manager: CheckpointManager
    workspace: RunnerWorkspace
    supervisor: RunnerSupervisor
    ft: FaultTolerance | None

    timestamp: str
    _print_process: int

    def __init__(self, config: RunnerConfig | Mapping[str, Any]) -> None:
        if not isinstance(config, RunnerConfig):
            config = RunnerConfig(config)

        state = RunnerState(config=config)
        self.state = state
        self.config = state.config
        self.train_state = state.train
        self.elastic_state = state.elastic
        self.rng_state = state.rng

        self.timestamp = get_time_str()
        self.workspace = RunnerWorkspace(self)
        self.name = str(self.config.get("name", f"{self.workspace.lineage}-{self.workspace.experiment}"))
        self.datasets = FlatDict()
        self.dataloaders = DataLoaderDict()
        self.results = RoundDict()
        self.meters = AverageMeters()
        self.mode = RunnerMode.train
        self.checkpoint_manager = FileCheckpointManager(self)
        self.supervisor = RunnerSupervisor(self)
        self.ft = None

        self.init_distributed()
        self.init_checkpoint_manager()
        self.init_fault_tolerance()
        self.init_garbage_collection()

        if self.config.seed is not None:
            self.set_seed()

        if self.config.deterministic:
            self.set_deterministic()

        if self.config.log:
            self.workspace.init_logging()

        if self.config.tensorboard:
            self.init_tensorboard()
        if self.config.get("wandb.enabled", False):
            self.init_wandb()

        self.workspace.init_print()
        self.init_signal_handlers()
        self.init_heartbeat()

    @property
    def world_size(self) -> int:
        """Distributed world size from environment."""

        return int(os.getenv("WORLD_SIZE", "1"))

    @property
    def rank(self) -> int:
        """Global rank from environment."""

        return int(os.getenv("RANK", "0"))

    @property
    def local_rank(self) -> int:
        """Local rank from environment."""

        return int(os.getenv("LOCAL_RANK", "0"))

    @property
    def distributed(self) -> bool:
        """Whether distributed mode is active."""

        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        """Whether current rank is global main process."""

        return self.rank == 0

    @property
    def is_local_main_process(self) -> bool:
        """Whether current rank is local main process."""

        return self.local_rank == 0

    @cached_property
    def code_id(self) -> str | None:
        """Stable code identity for the current checkout."""

        return get_git_hash()

    @cached_property
    def config_id(self) -> str:
        """Stable semantic config identity for this runner."""

        return format(hash(self.config) & ((1 << 48) - 1), "012x")

    @property
    def id(self) -> str:
        """Stable run identity derived from code identity and semantic config."""

        if self.code_id is None:
            return self.config_id
        return f"{self.code_id}-{self.config_id}"

    def __post_init__(self) -> None:
        """Hook called after `__init__` by `MetaRunner`."""
        self.workspace.save_metadata()

    @cached_property
    def score_split(self) -> str | None:
        """Split used for best-score selection."""

        if "score_split" in self.config and self.config.score_split is not None:
            return self.config.score_split

        splits = self.evaluate_splits
        if not splits:
            return None
        for split in splits:
            if split.lower().startswith("val"):
                return split
        return splits[0]

    @property
    def scores(self) -> FlatDict | None:
        """Index-to-score mapping extracted from `score_split/score_name`."""

        if not self.results:
            return None

        score_split = self.score_split
        if score_split is None:
            return None

        scores = FlatDict()
        for index, result in self.results.items():
            if score_split not in result:
                continue
            split_result = result[score_split]
            if not isinstance(split_result, Mapping):
                continue
            if self.config.score_name not in split_result:
                continue
            scores[index] = split_result[self.config.score_name]

        return scores or None

    @property
    def best_index(self) -> int:
        """Best result index according to configured score metric."""

        if not self.scores:
            return 0

        scores = self.scores
        indices = list(scores.keys())
        reducer = min if self.config.score_name == "loss" else max
        return reducer(reversed(indices), key=scores.get)

    @property
    def latest_result(self) -> RoundDict | None:
        """Most recent appended result row."""

        if not self.results:
            return None

        latest_index = next(reversed(self.results))
        latest = self.results[latest_index]

        ret = RoundDict(latest)
        ret["index"] = latest_index
        return ret

    @property
    def best_result(self) -> RoundDict | None:
        """Best result row according to configured score metric."""

        if not self.results:
            return None

        best_index = self.best_index
        best = self.results[best_index]

        ret = RoundDict(best)
        ret["index"] = best_index
        return ret

    @property
    def latest_score(self) -> float | None:
        """Latest scalar score."""

        scores = self.scores
        if not scores:
            return None

        latest_index = next(reversed(scores))
        return scores[latest_index]

    @property
    def best_score(self) -> float | None:
        """Best scalar score."""

        if not self.scores:
            return None

        return self.scores[self.best_index]

    @property
    def is_best(self) -> bool:
        """Whether latest score matches current best score.

        Returns ``True`` only when comparable scalar scores are available and
        agree within tolerance. Returns ``True`` on the first iteration (no
        prior results), and ``False`` when scores cannot be resolved (e.g.,
        no `score_split`/`score_name` configured) — silently reporting best
        in that case would trigger phantom "best" checkpoint copies.
        """

        if not self.results:
            return True

        latest = self.latest_score
        best = self.best_score
        if latest is None or best is None:
            return False
        return abs(latest - best) < 1e-7

    def get_epoch_result(self) -> RoundDict:
        meter_result = self.meters.average()
        if self.metrics is None:
            return RoundDict(meter_result)
        merged = RoundDict(meter_result)
        for key, value in self.metrics.average().items():
            if isinstance(value, Mapping) and len(value) == 1:
                value = next(iter(value.values()))
            merged[key] = value
        return merged

    def get_step_result(self) -> RoundDict:
        meter_result = self.meters.value()
        if self.metrics is None:
            return RoundDict(meter_result)
        merged = RoundDict(meter_result)
        for key, value in self.metrics.value().items():
            if isinstance(value, Mapping) and len(value) == 1:
                value = next(iter(value.values()))
            merged[key] = value
        return merged

    def append_result(self, result: RoundDict | Mapping[str, Any], index: int | None = None) -> None:
        if index is None:
            index = self.train_state.epoch

        if not isinstance(result, RoundDict):
            result = RoundDict(result)

        if index in self.results:
            self.results[index].merge(result)
        else:
            self.results[index] = result

    def step_log(
        self,
        split: str,
        iteration: int,
        length: int | str | None = None,
        result: RoundDict[str, Any] | Mapping[str, Any] | None = None,
    ) -> RoundDict:
        if length is None:
            try:
                length = len(self.dataloaders[split]) - 1
            except (TypeError, NotImplementedError):
                length = "∞"

        if result is None:
            result = self.get_step_result()
        elif not isinstance(result, RoundDict):
            result = RoundDict(result)
        print(self.format_step_result(result, split, iteration, length))

        if self.mode == RunnerMode.train:
            self.write_result(result, split)

        return result

    def format_epoch_result(
        self,
        result: RoundDict[str, Any],
        epochs: int | None = None,
        total_epochs: int | None = None,
    ) -> str:
        epochs = self.train_state.epoch if epochs is None else epochs
        total_epochs = self.epochs if total_epochs is None else total_epochs

        prefix = ""
        if total_epochs is not None:
            prefix = f"epoch [{epochs + 1}/{total_epochs}]"

        return f"{prefix}{self.format_result(result)}"

    def format_step_result(self, result: RoundDict[str, Any], split: str, steps: int, length: int | str) -> str:
        if self.mode == RunnerMode.train:
            prefix = f"training on {split}"
        elif self.mode == RunnerMode.evaluate:
            prefix = f"evaluating on {split}"
        elif self.mode == RunnerMode.infer:
            prefix = f"inferring on {split}"
        else:
            prefix = f"running in {self.mode} on {split}"

        return f"{prefix} [{steps}/{length}]\t{self.format_result(result)}"

    def format_result(self, result: RoundDict[str, Any], format_spec: str = ".4f") -> str:
        return format_result(result, format_spec=format_spec)

    def flatten_result(self, result: Mapping[str, Any]) -> FlatDict[str, Any]:
        flat_result = FlatDict()

        def add_score(tag: str, score: Any) -> None:
            if isinstance(score, AverageMeter):
                score = score.avg

            if isinstance(score, Mapping):
                nested = RoundDict(score)
                nested.setattr("separator", "/")
                for nested_name, nested_score in nested.dict(flatten=True).items():
                    add_score(f"{tag}/{nested_name}", nested_score)
                return

            if isinstance(score, Sequence) and not isinstance(score, (str, bytes)):
                for idx, nested_score in enumerate(score):
                    add_score(f"{tag}/{idx}", nested_score)
                return

            flat_result[tag] = score

        flattened = RoundDict(result)
        flattened.setattr("separator", "/")
        for name, score in flattened.dict(flatten=True).items():
            add_score(str(name), score)

        return flat_result

    def write_result(self, result: RoundDict[str, Any], split: str, steps: int | None = None) -> None:
        if self.writer is None and self.wandb is None:
            return

        steps = self.train_state.global_step if steps is None else steps

        flat_result = self.flatten_result(result)

        for name, score in flat_result.items():
            self.write_score(name, score, split, steps)

        if self.wandb is not None:
            payload = {f"{split}/{name}": score for name, score in flat_result.items()}
            self.wandb.log(payload, step=steps)

    def write_score(self, name: str, score: float, split: str, steps: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(f"{split}/{name}", score, steps)

    @catch
    @on_main_process
    def save_result(self) -> None:
        if not self.latest_result:
            return
        payload = {
            "name": self.name,
            "id": self.id,
            "timestamp": self.timestamp,
            "results": round(self.results, 8),
        }
        self.save(payload, os.path.join(self.workspace.dir, "results.json"), indent=4)

        latest = round(self.latest_result, 8)
        latest_payload = {"name": self.name, "id": self.id, "timestamp": self.timestamp}
        latest_payload.update(dict(latest))

        latest_path = os.path.join(self.workspace.dir, "latest.json")
        self.save(latest_payload, latest_path, indent=4)

        if self.is_best:
            shutil.copy(latest_path, os.path.join(self.workspace.dir, "best.json"))

    def auto_restore(self) -> None:
        """Auto-load resume/pretrained sources declared in config.

        Precedence:
            `config.resume` > `config.auto_resume` > `config.pretrained`.
        """

        restore_target = self._resolve_auto_restore_target()
        if restore_target is None:
            return

        restore_kind, restore_source = restore_target
        if restore_kind == "checkpoint":
            self.load_checkpoint(restore_source)
            return
        self.load_pretrained(restore_source)

    def _resolve_auto_restore_target(self) -> tuple[str, Mapping[Any, Any] | PathStr] | None:
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
            return ("checkpoint", resume_source)

        if auto_resume:
            return ("checkpoint", self._auto_resume_source())

        if pretrained_source:
            return ("pretrained", pretrained_source)

        return None

    def _auto_resume_source(self) -> str:
        backend = str(self.config.get("checkpoint.backend", "auto")).strip().lower()
        if backend == "dcp":
            return os.path.join(self.workspace.checkpoint_dir, "latest")
        return os.path.join(self.workspace.checkpoint_dir, "latest.pth")

    def init_distributed(self) -> None:
        """
        Initialize the distributed environment.

        The default is a no-op (single-process). Concrete runners override
        this hook to initialize the torch.distributed process group; see
        [`TorchRunner.init_distributed`][danling.runners.TorchRunner.init_distributed]
        for the canonical specification.
        """

    def init_checkpoint_manager(self) -> None:
        """
        Bind the runner's checkpoint manager.

        The default is a no-op — `BaseRunner.__init__` already binds the
        `FileCheckpointManager`. Concrete runners override this hook to swap
        in the backend-appropriate manager via `set_checkpoint_manager(...)`;
        see
        [`TorchRunner.init_checkpoint_manager`][danling.runners.TorchRunner.init_checkpoint_manager]
        for the canonical specification.
        """

    def init_fault_tolerance(self) -> None:
        """Initialize optional fault-tolerance runtime support."""

        self.ft = FaultTolerance(self)

    def init_heartbeat(self) -> None:
        """Configure optional background heartbeat writer."""

        self.supervisor.init_heartbeat()

    def init_garbage_collection(self) -> None:
        """Configure optional runner-managed Python GC pacing."""

        self.supervisor.init_garbage_collection()

    def init_signal_handlers(self) -> None:
        """Install runner-owned signal handlers for graceful preemption."""

        self.supervisor.init_signal_handlers()

    def prepare_for_shutdown_checkpoint(self) -> None:
        """Finalize runner state before writing a forced shutdown checkpoint."""

    def set_checkpoint_manager(self, manager: CheckpointManager) -> None:
        current = getattr(self, "checkpoint_manager", None)
        if current is manager:
            return
        if current is not None:
            current.close(timeout=0.0)
        self.checkpoint_manager = manager

    @on_main_process
    def init_tensorboard(self, *args, **kwargs) -> None:
        """Initialize tensorboard writer."""

        warn(
            "tensorboard is enabled, but this runner does not initialize a tensorboard writer",
            RuntimeWarning,
            stacklevel=2,
        )

    @on_main_process
    def init_wandb(self, *args, **kwargs) -> None:
        """Initialize Weights & Biases run for scalar logging."""

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("wandb is enabled, but the `wandb` package is not installed") from exc

        wandb_config = self.config.wandb
        if "project" not in kwargs:
            kwargs["project"] = wandb_config.get("project") or self.workspace.lineage
        if "entity" not in kwargs and wandb_config.get("entity") is not None:
            kwargs["entity"] = wandb_config.entity
        if "group" not in kwargs:
            kwargs["group"] = wandb_config.get("group") or self.workspace.experiment
        if "name" not in kwargs:
            kwargs["name"] = wandb_config.get("name") or self.id
        if "job_type" not in kwargs and wandb_config.get("job_type") is not None:
            kwargs["job_type"] = wandb_config.job_type
        tags = wandb_config.get("tags")
        if "tags" not in kwargs and tags is not None:
            kwargs["tags"] = [tags] if isinstance(tags, str) else list(tags)
        if "dir" not in kwargs:
            kwargs["dir"] = wandb_config.get("dir") or self.workspace.dir
        if "mode" not in kwargs and wandb_config.get("mode") is not None:
            kwargs["mode"] = wandb_config.mode
        if "config" not in kwargs:
            kwargs["config"] = self.config.dict()

        self.wandb = cast(Any, wandb).init(*args, **kwargs)

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
        """
        Run one training micro-step.

        Concrete runners define the override contract; see
        [`TorchRunner.train_step`][danling.runners.TorchRunner.train_step] for
        the canonical specification.
        """

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
        """
        Run one evaluation step.

        Concrete runners define the override contract; see
        [`TorchRunner.evaluate_step`][danling.runners.TorchRunner.evaluate_step]
        for the canonical specification.
        """

        raise NotImplementedError

    def infer(self, *args, **kwargs):
        """Run top-level inference workflow."""

        raise NotImplementedError

    def infer_step(self, *args, **kwargs):
        """
        Run one inference step.

        Concrete runners define the override contract; see
        [`TorchRunner.infer_step`][danling.runners.TorchRunner.infer_step] for
        the canonical specification.
        """

        raise NotImplementedError

    def unwrap(self, model: Any) -> Any:
        """Return an unwrapped model object."""

        return model

    def state_dict(self, cls: type = dict) -> Mapping:
        """
        Build the backend-neutral runner checkpoint payload.

        The base payload contains semantic runner config, mutable runner
        state, RNG snapshots, and dataloader resume state. Backend runners
        extend this payload with model/optimizer/scheduler state.

        **Called when:** checkpoint managers build a payload for
        `save_checkpoint`, and fault-tolerance callbacks need a runner state
        snapshot.

        Args:
            cls: Mapping factory used for nested payloads. Backends may pass
                `dict`-like containers to preserve their serialization format.

        Returns:
            Mapping with `runner`, `state`, and `dataloaders` keys.

        **Side effects:** snapshots Python and NumPy RNG state into
        `self.rng_state` before exporting.

        !!! danger "Do not"
            - Mutate model or optimizer state here.
            - Drop the `runner` config payload; resume validation depends on it.
            - Override without calling `super()` unless you fully replace the
              checkpoint format.
        """

        self.rng_state.python = random.getstate()
        self.rng_state.numpy = np_random.get_state() if np_random is not None else None

        state = self.state.state_dict()
        if cls is not dict:
            state = cls(state)

        dataloader_state = self.dataloaders.state_dict()
        if cls is not dict:
            dataloader_state = cls(dataloader_state)

        return cls(runner=self.config.dict(), state=state, dataloaders=dataloader_state)

    def load_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        """
        Restore backend-neutral runner state from a checkpoint payload.

        This restores semantic runner state and Python/NumPy RNG state. Model,
        EMA, optimizer, scheduler, and dataloader component loading is owned by
        `load_checkpoint`.

        **Called when:** `load_checkpoint` restores a full checkpoint, and
        fault-tolerance callbacks receive a runner state payload.

        Args:
            checkpoint: Mapping produced by `state_dict` or a backend-specific
                superset of that payload.

        Raises:
            ValueError: checkpoint runner config differs semantically from the
                current runner config.

        **Side effects:** updates `self.state`, `self.train_state`,
        `self.elastic_state`, `self.rng_state`, and process RNG state.

        !!! danger "Do not"
            - Load model/optimizer/scheduler state here; use component loaders
              through `load_checkpoint`.
            - Suppress semantic config diffs unless you also update the resume
              policy deliberately.
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

    @staticmethod
    def _normalize_checkpoint_exclude_path(path: str) -> tuple[str, ...]:
        aliases = {
            "data_loader": "dataloaders",
            "dataloader": "dataloaders",
            "lr_scheduler": "scheduler",
        }
        parts = tuple(part for part in str(path).split(".") if part)
        if not parts:
            return ()
        return (aliases.get(parts[0], parts[0]), *parts[1:])

    def checkpoint_exclude_from_loading(self) -> tuple[tuple[str, ...], ...]:
        excluded = self.config.get("checkpoint.exclude_from_loading")
        if excluded is None:
            return ()
        if isinstance(excluded, str):
            excluded = (excluded,)
        return tuple(
            normalized for path in excluded if (normalized := self._normalize_checkpoint_exclude_path(str(path)))
        )

    @staticmethod
    def _drop_checkpoint_path(checkpoint: dict[str, Any], path: Sequence[str]) -> None:
        if not path:
            return
        key = path[0]
        if len(path) == 1:
            checkpoint.pop(key, None)
            return
        child = checkpoint.get(key)
        if isinstance(child, Mapping):
            child_copy = dict(child)
            checkpoint[key] = child_copy
            BaseRunner._drop_checkpoint_path(child_copy, path[1:])

    def _filter_checkpoint_for_loading(
        self,
        checkpoint: Mapping[str, Any],
        excluded_paths: Sequence[Sequence[str]],
    ) -> dict[str, Any]:
        filtered = dict(checkpoint)
        for path in excluded_paths:
            self._drop_checkpoint_path(filtered, path)
        return filtered

    @staticmethod
    def _is_top_level_checkpoint_excluded(excluded_paths: Sequence[Sequence[str]], *keys: str) -> bool:
        key_set = set(keys)
        return any(len(path) == 1 and path[0] in key_set for path in excluded_paths)

    def save_checkpoint(
        self,
        name: str = "latest",
        epochs: int | None = None,
        save_best: bool = True,
        last_step: bool = False,
        force: bool = False,
    ) -> None:
        """
        Persist runner state through the active checkpoint manager.

        Backend collective semantics are owned by
        `checkpoint_manager.is_collective`. File-style managers save on the
        main process only; collective managers require every rank to enter this
        method together.

        **Called when:** training loops hit checkpoint cadence, final
        `last_step` saves run, or the supervisor handles a shutdown signal.

        Args:
            name: Logical checkpoint alias, usually `"latest"` or `"best"`.
            epochs: Epoch index used for history checkpoint naming. Defaults
                to `self.train_state.epoch`.
            save_best: Whether to publish/update the best-checkpoint alias
                when `self.is_best` is true.
            last_step: Whether this save is the final save for the run.
            force: Bypass cadence checks inside the manager.

        **Side effects:** delegates to
        `self.checkpoint_manager.save_checkpoint(...)`.

        !!! danger "Do not"
            - Add a main-process guard around calls to this method; DCP-style
              managers need all ranks to participate.
            - Bypass the checkpoint manager for normal runner checkpoints.
        """

        if not (self.is_main_process or self.checkpoint_manager.is_collective):
            return
        epochs = self.train_state.epoch if epochs is None else epochs
        self.checkpoint_manager.save_checkpoint(
            name=name,
            epochs=epochs,
            save_best=save_best,
            last_step=last_step,
            force=force,
        )

    def save_seed_checkpoint(self, name: str = "seed") -> None:
        """
        Persist an initialization checkpoint for cross-topology experiments.

        Seed checkpoints are intended to be created before training advances,
        then loaded with `checkpoint.load_only=True` or `resume`/`pretrained`
        when comparing different parallel layouts from the same initial model
        state. They are saved through the final-checkpoint path, so
        `checkpoint.last_save_model_only=True` intentionally applies.
        """
        if self.train_state.global_step != 0 or self.train_state.epoch != 0:
            warn(
                "save_seed_checkpoint() is intended before training advances; "
                f"current epoch={self.train_state.epoch}, global_step={self.train_state.global_step}",
                RuntimeWarning,
                stacklevel=2,
            )
        self.save_checkpoint(name=name, epochs=0, save_best=False, last_step=True, force=True)

    def load_checkpoint(
        self,
        checkpoint: Mapping | bytes | str | os.PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Restore a full runner checkpoint.

        This is the full-state restore path: runtime state, model/EMA,
        optimizer, scheduler, and dataloader progress are restored when present
        and applicable to the current runner.

        **Called when:** users resume a run explicitly, `auto_restore` selects
        a resume source, `from_checkpoint` constructs a runner, or
        fault-tolerance callbacks restore a full runner payload.

        Args:
            checkpoint: In-memory checkpoint mapping or backend-specific path.
            *args: Forwarded to `read_checkpoint` and component loaders.
            **kwargs: Forwarded to `read_checkpoint` and component loaders.

        Raises:
            ValueError: checkpoint is missing required component state for an
                initialized component, or config validation fails.

        **Side effects:** updates runner state, model/EMA weights, optimizer,
        scheduler, dataloader progress, and `config.resume` for path inputs.

        !!! danger "Do not"
            - Use this for model-only finetuning payloads; use
              `load_pretrained` instead.
            - Override just to support a new path type; prefer overriding
              `read_checkpoint`.
        """

        ckpt = self.read_checkpoint(checkpoint, *args, **kwargs)
        excluded_paths = self.checkpoint_exclude_from_loading()
        if excluded_paths:
            if self._is_top_level_checkpoint_excluded(excluded_paths, "runner"):
                warn(
                    "`checkpoint.exclude_from_loading` contains 'runner'; "
                    "semantic runner config validation will be skipped for this load.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            ckpt = self._filter_checkpoint_for_loading(ckpt, excluded_paths)

        self.load_state_dict(ckpt)
        if not self._is_top_level_checkpoint_excluded(excluded_paths, "model", "model_parts", "module"):
            if "model" in ckpt:
                self.load_model(ckpt["model"], *args, **kwargs)
            elif "model_parts" in ckpt:
                self.load_model(ckpt["model_parts"], *args, **kwargs)
            elif self.model is not None:
                raise ValueError(
                    "cannot restore model: checkpoint has no model state\n"
                    "Use `load_pretrained` only for model-only checkpoints with model/ema payloads"
                )
        if not self._is_top_level_checkpoint_excluded(excluded_paths, "ema") and (
            self.ema is not None or "ema" in ckpt
        ):
            self.load_ema(ckpt.get("ema"), *args, **kwargs)
        if not self._is_top_level_checkpoint_excluded(excluded_paths, "optimizer") and (
            self.optimizer is not None or "optimizer" in ckpt
        ):
            self.load_optimizer(ckpt.get("optimizer"), *args, **kwargs)
        if not self._is_top_level_checkpoint_excluded(excluded_paths, "scheduler") and (
            self.scheduler is not None or "scheduler" in ckpt
        ):
            self.load_scheduler(ckpt.get("scheduler"), *args, **kwargs)
        if not self._is_top_level_checkpoint_excluded(excluded_paths, "dataloaders") and (
            self.dataloaders or "dataloaders" in ckpt
        ):
            self.load_dataloaders(ckpt.get("dataloaders"))
        if isinstance(checkpoint, (str, bytes, os.PathLike)):
            self.config.resume = os.fsdecode(checkpoint)

    @staticmethod
    def _require_checkpoint_component_state(component: str, state_dict: Any | None) -> Any:
        component_labels = {
            "ema": "EMA state",
            "optimizer": "optimizer state",
            "scheduler": "scheduler state",
        }
        if state_dict is None:
            component_label = component_labels.get(component, f"{component} state")
            raise ValueError(
                f"cannot restore {component}: checkpoint has no {component_label}\n"
                "Use `load_pretrained` for model-only checkpoints instead of `load_checkpoint`"
            )
        return state_dict

    def load_model(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
        """Load model state."""
        if self.model is None:
            raise ValueError("cannot restore model: model is not initialized")
        self.unwrap(self.model).load_state_dict(state_dict, *args, **kwargs)

    def load_ema(self, state_dict: Mapping[str, Any] | None, *args, **kwargs) -> None:
        """Load EMA state."""
        if self.ema is None:
            return
        state_dict = self._require_checkpoint_component_state("ema", state_dict)
        self.ema.load_state_dict(state_dict, *args, **kwargs)

    def load_optimizer(self, state_dict: Mapping[str, Any] | None, *args, **kwargs) -> None:
        """Load optimizer state."""
        if self.optimizer is None:
            return
        state_dict = self._require_checkpoint_component_state("optimizer", state_dict)
        self.optimizer.load_state_dict(state_dict, *args, **kwargs)

    def load_scheduler(self, state_dict: Mapping[str, Any] | None, *args, **kwargs) -> None:
        """Load scheduler state."""
        if self.scheduler is None:
            return
        state_dict = self._require_checkpoint_component_state("scheduler", state_dict)
        self.scheduler.load_state_dict(state_dict, *args, **kwargs)

    def load_dataloaders(self, state_dict: Mapping[str, Any] | None) -> None:
        """Load dataloader progress state when the current runner has matching loaders."""
        if state_dict is None:
            return
        self.dataloaders.load_state_dict(state_dict)

    def load_pretrained(
        self,
        checkpoint: Mapping | bytes | str | os.PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Load model weights only from a checkpoint payload or path.

        When checkpoint payload provides EMA weights (`ema`), EMA is preferred as
        the pretrained source. Otherwise `model` is used.

        **Called when:** users initialize from pretrained weights, or
        `auto_restore` selects `config.pretrained`.

        Args:
            checkpoint: In-memory payload or backend-specific path containing
                `ema`, `model`, or `model_parts`.
            *args: Forwarded to `read_checkpoint` and `load_model`.
            **kwargs: Forwarded to `read_checkpoint` and `load_model`.

        Raises:
            ValueError: model is not initialized, or the payload has no usable
                model/EMA state.

        **Side effects:** loads model weights and updates `config.pretrained`
        for path inputs. Optimizer, scheduler, runner state, and dataloaders
        are intentionally untouched.

        !!! danger "Do not"
            - Use this to resume training state; use `load_checkpoint` for
              full-state restore.
            - Load optimizer/scheduler state in this path.
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
        if isinstance(checkpoint, (str, bytes, os.PathLike)):
            self.config.pretrained = os.fsdecode(checkpoint)
        else:
            self.config.pretrained = None

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> BaseRunner:
        """Instantiate runner from checkpoint config and restore full state."""

        config = cls.read_config(checkpoint, *args, **kwargs)
        config.resume = None
        config.auto_resume = False
        config.pretrained = None
        runner = cls(config)
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
                kwargs = dict(kwargs)
                kwargs["map_location"] = "cpu"
                kwargs["weights_only"] = False
                ckpt = load(checkpoint, *args, **kwargs)
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
        prepared.auto_resume = False
        prepared.pretrained = None
        runner = cls(prepared)
        runner.load_pretrained(checkpoint, *args, **kwargs)
        return runner

    def read_checkpoint(self, checkpoint: Mapping | bytes | str | os.PathLike, *args, **kwargs) -> Mapping[str, Any]:
        """Normalize checkpoint input into an in-memory mapping payload."""
        if isinstance(checkpoint, (bytes, str, os.PathLike)):
            kwargs = dict(kwargs)
            kwargs["map_location"] = "cpu"
            kwargs["weights_only"] = False
            return load(checkpoint, *args, **kwargs)
        if isinstance(checkpoint, Mapping):
            return checkpoint
        raise ValueError(
            f"invalid checkpoint input: expected a mapping or path, got {type(checkpoint).__name__}: {checkpoint!r}"
        )

    def save(self, obj: Any, file: PathStr, main_process_only: bool = True, *args, **kwargs) -> File:
        """Save an object with optional main-process guard."""

        if (main_process_only and self.is_main_process) or not main_process_only:
            return save(obj, file, *args, **kwargs)
        return file

    def close(self, timeout: float | None = None) -> bool:
        """Finalize checkpoint/log/writer resources before shutdown."""

        if timeout is None:
            timeout = self.config.get("checkpoint.wait_timeout")

        drained = True
        close_error: Exception | None = None
        try:
            drained = self.checkpoint_manager.close(timeout=timeout)
        except Exception as exc:
            close_error = exc

        if close_error is None and not drained:
            warn("runner close: timed out while draining async checkpoints", RuntimeWarning, stacklevel=2)
            return False

        self.supervisor.restore_signal_handlers()
        writer = self.writer
        if writer is not None:
            writer.flush()
            writer.close()
            self.writer = None

        if self.wandb is not None:
            self.wandb.finish()

        self.workspace.close()
        self.supervisor.close()
        if self.ft is not None:
            self.ft.close()

        if close_error is not None:
            raise close_error
        return drained

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

    @staticmethod
    def _loader_length(loader: Any) -> int | None:
        try:
            return len(loader)
        except (TypeError, NotImplementedError):
            return None

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
                split_micro_steps = self._loader_length(self.dataloaders[split])
                if split_micro_steps is None:
                    return None
                steps_per_epoch += (split_micro_steps + self.accum_steps - 1) // self.accum_steps
            return steps_per_epoch * self.epochs
        return None

    @steps.setter
    def steps(self, steps: int) -> None:
        self.config.steps = steps

    @property
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

    @property
    def train_splits(self) -> list[str]:
        """Configured or inferred training split names."""
        if "train_splits" in self.config:
            return self._sorted_unique(self.config["train_splits"])
        if self.datasets:
            inferred = [
                split
                for split, dataset in self.datasets.items()
                if split == "train" or getattr(dataset, "train", False) or getattr(dataset, "split", None) == "train"
            ]
            return self._sorted_unique(inferred)
        return []

    @property
    def evaluate_splits(self) -> list[str]:
        """Configured or inferred evaluation split names."""
        if "evaluate_splits" in self.config:
            return self._sorted_unique(self.config["evaluate_splits"])
        if self.datasets:
            train_splits = set(self.train_splits)
            return sorted(split for split in self.datasets if split not in train_splits)
        return []

    @staticmethod
    def _sorted_unique(values: Sequence[str] | str) -> list[str]:
        if isinstance(values, str):
            return [values]
        return sorted(dict.fromkeys(str(value) for value in values))

    @property
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

    @property
    def log_interval(self) -> int:
        """Step logging cadence."""
        configured = self.config.get("log_interval")
        if configured is not None:
            return configured
        if self.steps is not None:
            return max(ceil(self.steps / 100), 1)
        return 1_024
