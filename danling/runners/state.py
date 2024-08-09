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


class RunnerState(_StatefulBase):
    def __init__(
        self,
        *,
        config: RunnerConfig | Mapping[str, Any],
        train: RunnerTrainState | None = None,
        elastic: RunnerElasticState | None = None,
        rng: RunnerRNGState | None = None,
    ) -> None:
        if not isinstance(config, RunnerConfig):
            config = RunnerConfig(config)
        self.config = config
        self.train = RunnerTrainState() if train is None else train
        self.elastic = RunnerElasticState() if elastic is None else elastic
        self.rng = RunnerRNGState() if rng is None else rng

    def state_dict(self) -> dict[str, Any]:
        return {
            "train": self.train.state_dict(),
            "elastic": self.elastic.state_dict(),
            "rng": self.rng.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        train_state = state_dict.get("train")
        if isinstance(train_state, dict):
            self.train.load_state_dict(train_state)

        elastic_state = state_dict.get("elastic")
        if isinstance(elastic_state, dict):
            self.elastic.load_state_dict(elastic_state)

        rng_state = state_dict.get("rng")
        if isinstance(rng_state, dict):
            self.rng.load_state_dict(rng_state)


class RunnerTrainState:
    def __init__(
        self,
        global_step: int = 0,
        micro_step: int = 0,
        epoch: int = 0,
        tokens_seen: int = 0,
        samples_seen: int = 0,
    ) -> None:
        self.global_step = int(global_step)
        self.micro_step = int(micro_step)
        self.epoch = int(epoch)
        self.tokens_seen = int(tokens_seen)
        self.samples_seen = int(samples_seen)

    def state_dict(self) -> dict[str, int]:
        return {
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "epoch": self.epoch,
            "tokens_seen": self.tokens_seen,
            "samples_seen": self.samples_seen,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "global_step" in state_dict:
            self.global_step = int(state_dict["global_step"])
        if "micro_step" in state_dict:
            self.micro_step = int(state_dict["micro_step"])
        if "epoch" in state_dict:
            self.epoch = int(state_dict["epoch"])
        if "tokens_seen" in state_dict:
            self.tokens_seen = int(state_dict["tokens_seen"])
        if "samples_seen" in state_dict:
            self.samples_seen = int(state_dict["samples_seen"])


class RunnerElasticState:
    def __init__(
        self,
        restart_count: int = 0,
        membership_version: int = 0,
        last_seen_world_size: int = 0,
    ) -> None:
        self.restart_count = int(restart_count)
        self.membership_version = int(membership_version)
        self.last_seen_world_size = int(last_seen_world_size)

    def state_dict(self) -> dict[str, int]:
        return {
            "restart_count": self.restart_count,
            "membership_version": self.membership_version,
            "last_seen_world_size": self.last_seen_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "restart_count" in state_dict:
            self.restart_count = int(state_dict["restart_count"])
        if "membership_version" in state_dict:
            self.membership_version = int(state_dict["membership_version"])
        if "last_seen_world_size" in state_dict:
            self.last_seen_world_size = int(state_dict["last_seen_world_size"])


class RunnerRNGState:
    def __init__(
        self,
        python: Any | None = None,
        numpy: Any | None = None,
        torch_cpu: Any | None = None,
        torch_cuda: Any | None = None,
    ) -> None:
        self.python = python
        self.numpy = numpy
        self.torch_cpu = torch_cpu
        self.torch_cuda = torch_cuda

    def state_dict(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "numpy": self.numpy,
            "torch_cpu": self.torch_cpu,
            "torch_cuda": self.torch_cuda,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "python" in state_dict:
            self.python = state_dict["python"]
        if "numpy" in state_dict:
            self.numpy = state_dict["numpy"]
        if "torch_cpu" in state_dict:
            self.torch_cpu = state_dict["torch_cpu"]
        if "torch_cuda" in state_dict:
            self.torch_cuda = state_dict["torch_cuda"]
