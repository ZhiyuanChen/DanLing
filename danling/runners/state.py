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

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

from danling.runners.config import RunnerConfig

try:
    from torch.distributed.checkpoint.stateful import Stateful as _StatefulBase
except ImportError:

    class _StatefulBase:  # type: ignore[no-redef]
        def state_dict(self) -> dict[str, Any]:
            raise NotImplementedError

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            raise NotImplementedError


def _load_fields(target: Any, state_dict: Mapping[str, Any]) -> None:
    """Update dataclass fields from a partial state mapping.

    Only fields present in ``state_dict`` are written; missing keys leave
    existing values untouched.
    """
    for f in fields(target):
        if f.name in state_dict:
            setattr(target, f.name, state_dict[f.name])


@dataclass
class RunnerTrainState:
    """
    Mutable training progress counters persisted in runner checkpoints.

    Attributes:
        global_step: Number of optimizer updates that have completed.
        micro_step: Number of training micro-batches consumed, including
            accumulation windows that have not yet flushed.
        epoch: Current epoch index. In epoch mode this is the loop index; in
            step mode it counts outer split-round passes.
    """

    global_step: int = 0
    micro_step: int = 0
    epoch: int = 0

    def state_dict(self) -> dict[str, int]:
        return asdict(self)

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        _load_fields(self, state_dict)


@dataclass
class RunnerElasticState:
    """
    Torchelastic restart metadata.

    Attributes:
        restart_count: Value read from `TORCHELASTIC_RESTART_COUNT` when
            present. It is informational state used for checkpoint metadata and
            heartbeat reporting; it is not a membership-change mechanism.
    """

    restart_count: int = 0

    def state_dict(self) -> dict[str, int]:
        return asdict(self)

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        _load_fields(self, state_dict)


@dataclass
class RunnerRNGState:
    """
    RNG snapshots carried by runner checkpoints.

    Attributes:
        python: State from `random.getstate()`.
        numpy: State from `numpy.random.get_state()` when NumPy is available.
        torch_cpu: State from `torch.get_rng_state()` in torch runners.
        torch_cuda: State from `torch.cuda.get_rng_state_all()` when CUDA is
            available.
    """

    python: Any = None
    numpy: Any = None
    torch_cpu: Any = None
    torch_cuda: Any = None

    def state_dict(self) -> dict[str, Any]:
        # Shallow construction (not `asdict`): RNG payloads include torch tensors,
        # numpy arrays, and tuples that we must surface by reference, not deep-copy.
        return {
            "python": self.python,
            "numpy": self.numpy,
            "torch_cpu": self.torch_cpu,
            "torch_cuda": self.torch_cuda,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        _load_fields(self, state_dict)


@dataclass
class RunnerState(_StatefulBase):
    """
    Checkpointable state container for a runner instance.

    Attributes:
        config: Runner configuration associated with this state object.
        train: Training progress counters.
        elastic: Torchelastic restart metadata.
        rng: Python/NumPy/Torch RNG snapshots.
    """

    config: RunnerConfig
    train: RunnerTrainState = field(default_factory=RunnerTrainState)
    elastic: RunnerElasticState = field(default_factory=RunnerElasticState)
    rng: RunnerRNGState = field(default_factory=RunnerRNGState)

    def __post_init__(self) -> None:
        if not isinstance(self.config, RunnerConfig):
            self.config = RunnerConfig(self.config)

    def state_dict(self) -> dict[str, Any]:
        return {
            "train": self.train.state_dict(),
            "elastic": self.elastic.state_dict(),
            "rng": self.rng.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        for name in ("train", "elastic", "rng"):
            value = state_dict.get(name)
            if isinstance(value, Mapping):
                getattr(self, name).load_state_dict(value)
