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
import torch
from torch import nn, optim

from danling.optim.optimizer import OptimizerContainer


@pytest.fixture
def parameter():
    return nn.Parameter(torch.ones(2))


@pytest.fixture
def optimizer(parameter):
    return optim.SGD([parameter], lr=1, momentum=0.9)


@pytest.fixture
def container(optimizer):
    return OptimizerContainer(optimizer)
