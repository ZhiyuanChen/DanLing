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
