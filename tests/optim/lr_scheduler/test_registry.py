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

import pytest
from torch.optim import lr_scheduler

from danling.optim.lr_scheduler.lr_scheduler import LRScheduler
from danling.optim.lr_scheduler.registry import SCHEDULERS


class TestRegistry:
    @pytest.mark.parametrize(
        ("name", "options", "expected"),
        [
            ("step", {"step_size": 10}, lr_scheduler.StepLR),
            ("multistep", {"milestones": [10, 20, 30]}, lr_scheduler.MultiStepLR),
            ("exponential", {"gamma": 0.9}, lr_scheduler.ExponentialLR),
            ("cosine_annealing", {"T_max": 10}, lr_scheduler.CosineAnnealingLR),
            ("STEP", {"step_size": 10}, lr_scheduler.StepLR),
            ("linear", {"total_steps": 100}, LRScheduler),
            ("cosine", {"total_steps": 100}, LRScheduler),
            ("constant", {"total_steps": 100}, LRScheduler),
        ],
    )
    def test_build(self, optimizer, name, options, expected):
        assert isinstance(SCHEDULERS.build(optimizer, type=name, **options), expected)
