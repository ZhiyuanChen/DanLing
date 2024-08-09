# DanLing
# Copyright (C) 2022-Present  DanLing

# This program is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from __future__ import annotations

import os
from math import ceil

import torch
from chanfig import NestedDict
from lazy_imports import try_import
from torch import distributed as dist

from danling.runner.config import Config

from .torch_runner import TorchRunner

with try_import() as ds:
    import deepspeed


class DeepSpeedRunner(TorchRunner):

    def __init__(self, config: Config) -> None:
        ds.check()
        super().__init__(config)

    def init_distributed(self) -> None:
        r"""
        Set up distributed training.

        Initialise process group and set up DDP variables.
        """

        torch.cuda.set_device(self.get_local_rank())
        deepspeed.init_distributed()
        object_list = [self.id, self.timestamp]
        dist.broadcast_object_list(object_list)
        self.id, self.timestamp = object_list

    def __post_init__(self):
        super().__post_init__()
        if self.datasets:
            self.build_dataloaders()
        if self.config.get("log_interval") is None:
            self.config.log_interval = max(ceil(max(len(d) for d in self.dataloaders.values()) / 10), 1)
        self.config.deepspeed = self.get_deepspeed_config()
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
            config=self.config.deepspeed,
        )

    def backward(self, loss: torch.Tensor) -> None:
        return self.model.backward(loss)

    def advance(self, loss) -> None:
        self.backward(loss)
        self.model.step()

    def get_local_rank(self) -> int:
        local_rank = self.config.get("local_rank", os.getenv("LOCAL_RANK"))
        rank = self.config.get("rank", os.getenv("RANK"))
        world_size = self.config.get("world_size", os.getenv("WORLD_SIZE"))
        if local_rank is None:
            if world_size is None or rank is None:
                raise ValueError("Please provide either `local_rank` or `world_size` and `rank`")
            return int(world_size) % int(rank)
        return int(local_rank)

    @property
    def deepspeed(self) -> NestedDict | None:
        if isinstance(self.model, deepspeed.DeepSpeedEngine):
            return self.model.config
        return None
