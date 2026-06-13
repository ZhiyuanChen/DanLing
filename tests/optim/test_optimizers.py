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
import torch.nn as nn
from packaging import version
from torch import optim

try:
    import deepspeed as ds
except ImportError:
    ds = None

from danling.optim import OptimizerContainer
from danling.optim.registry import OPTIMIZERS


def test_OPTIMIZERS():
    model = nn.Linear(10, 1)

    assert isinstance(OPTIMIZERS.build(model.parameters(), type="sgd", lr=0.01), optim.SGD)
    assert isinstance(
        OPTIMIZERS.build(model.parameters(), type="torch_adamw", lr=0.001, weight_decay=0.01), optim.AdamW
    )
    assert isinstance(OPTIMIZERS.build(model.parameters(), type="torch_adam", lr=0.001), optim.Adam)
    if version.parse(torch.__version__) >= version.parse("2.5.0"):
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="adafactor", lr=0.001), optim.Adafactor)


def test_optimizer_container_returns_step_result_with_grad_norm():
    model = nn.Linear(2, 1, bias=False)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    container = OptimizerContainer(optimizer)

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    result = container.step(max_grad_norm=1.0)

    assert result.stepped is True
    assert result.grad_norm is not None
    assert result.grad_norm > 0


def test_optimizer_container_skip_nonfinite_returns_step_result_without_update():
    model = nn.Linear(2, 1, bias=False)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    container = OptimizerContainer(optimizer)
    initial = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, float("inf"))

    result = container.step(skip_nonfinite_grad=True)

    assert result.stepped is False
    assert result.grad_norm is None
    torch.testing.assert_close(model.weight, initial)


@pytest.mark.skipif(ds is None, reason="deepspeed is not installed")
def test_deepspeed_optimizers():
    model = nn.Linear(10, 1)

    if torch.cuda.device_count() > 0:
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="ds_adam", lr=0.001), ds.ops.adam.FusedAdam)
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="ds_adamw", lr=0.001), ds.ops.adam.FusedAdam)
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="lamb", lr=0.001), ds.ops.lamb.FusedLamb)
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="lion", lr=0.001), ds.ops.lion.FusedLion)
    else:
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="ds_adam", lr=0.001), ds.ops.adam.DeepSpeedCPUAdam)
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="ds_adamw", lr=0.001), ds.ops.adam.DeepSpeedCPUAdam)
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="lamb", lr=0.001), ds.ops.lamb.DeepSpeedCPULamb)
        assert isinstance(OPTIMIZERS.build(model.parameters(), type="lion", lr=0.001), ds.ops.lion.DeepSpeedCPULion)

    assert isinstance(OPTIMIZERS.build(model.parameters(), type="adam", lr=0.001), optim.Adam)
    assert isinstance(OPTIMIZERS.build(model.parameters(), type="adamw", lr=0.001), optim.AdamW)
    assert isinstance(OPTIMIZERS.build(model.parameters(), type="lamb", lr=0.001), ds.ops.lamb.DeepSpeedCPULamb)
    assert isinstance(OPTIMIZERS.build(model.parameters(), type="lion", lr=0.001), ds.ops.lion.DeepSpeedCPULion)
