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

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from danling.optim import LRScheduler
from danling.optim.lr_scheduler.registry import SCHEDULERS


def test_standard_schedulers():
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    assert isinstance(SCHEDULERS.build(optimizer, type="step", step_size=10), lr_scheduler.StepLR)
    assert isinstance(SCHEDULERS.build(optimizer, type="multistep", milestones=[10, 20, 30]), lr_scheduler.MultiStepLR)
    assert isinstance(SCHEDULERS.build(optimizer, type="exponential", gamma=0.9), lr_scheduler.ExponentialLR)
    assert isinstance(SCHEDULERS.build(optimizer, type="cosine_annealing", T_max=10), lr_scheduler.CosineAnnealingLR)
    assert isinstance(SCHEDULERS.build(optimizer, type="STEP", step_size=10), lr_scheduler.StepLR)


def test_custom_schedulers():
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    assert isinstance(SCHEDULERS.build(optimizer, type="linear", total_steps=100), LRScheduler)
    assert isinstance(SCHEDULERS.build(optimizer, type="cosine", total_steps=100), LRScheduler)
    assert isinstance(SCHEDULERS.build(optimizer, type="constant", total_steps=100), LRScheduler)
