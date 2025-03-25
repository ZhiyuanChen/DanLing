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

from functools import partial

from chanfig import Registry as Registry_
from torch.optim import lr_scheduler

from .lr_scheduler import LRScheduler


class Registry(Registry_):
    case_sensitive = False

    def build(self, *args, **kwargs) -> lr_scheduler._LRScheduler:
        return super().build(*args, **kwargs)


SCHEDULERS = Registry()

SCHEDULERS.register(partial(LRScheduler, strategy="linear"), "linear")
SCHEDULERS.register(partial(LRScheduler, strategy="cosine"), "cosine")
SCHEDULERS.register(partial(LRScheduler, strategy="constant"), "constant")

SCHEDULERS.register(lr_scheduler.StepLR, "step")
SCHEDULERS.register(lr_scheduler.MultiStepLR, "multistep")
SCHEDULERS.register(lr_scheduler.ExponentialLR, "exponential")
SCHEDULERS.register(lr_scheduler.CosineAnnealingLR, "cosine_annealing")
SCHEDULERS.register(lr_scheduler.ReduceLROnPlateau, "reduce_on_plateau")
SCHEDULERS.register(lr_scheduler.CyclicLR, "cyclic")
SCHEDULERS.register(lr_scheduler.OneCycleLR, "one_cycle")
SCHEDULERS.register(lr_scheduler.CosineAnnealingWarmRestarts, "cosine_warm_restarts")
