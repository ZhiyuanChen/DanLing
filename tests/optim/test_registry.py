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
from packaging.version import Version, parse
from torch import nn, optim

from danling.optim.registry import OPTIMIZERS


class TestRegistry:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("sgd", optim.SGD),
            ("torch_adamw", optim.AdamW),
            ("torch_adam", optim.Adam),
            pytest.param(
                "adafactor",
                getattr(optim, "Adafactor", None),
                marks=pytest.mark.skipif(
                    parse(torch.__version__) < Version("2.5.0"), reason="Adafactor requires Torch 2.5"
                ),
            ),
        ],
    )
    def test_build(self, parameter, name, expected):
        assert isinstance(OPTIMIZERS.build([parameter], type=name, lr=0.01), expected)

    def test_deepspeed(self):
        ds = pytest.importorskip("deepspeed")
        if torch.cuda.is_available():
            parameter = nn.Parameter(torch.ones(1, device="cuda"))
            cases = [
                ("adam", ds.ops.adam.FusedAdam),
                ("adamw", ds.ops.adam.FusedAdam),
                ("ds_adam", ds.ops.adam.FusedAdam),
                ("ds_adamw", ds.ops.adam.FusedAdam),
                ("lamb", ds.ops.lamb.FusedLamb),
                ("lion", ds.ops.lion.FusedLion),
            ]
        else:
            parameter = nn.Parameter(torch.ones(1))
            cases = [
                ("cpu_adam", getattr(ds.ops.adam, "DeepSpeedCPUAdam", None)),
                ("cpu_adamw", getattr(ds.ops.adam, "DeepSpeedCPUAdam", None)),
                ("cpu_lamb", getattr(ds.ops.lamb, "DeepSpeedCPULamb", None)),
                ("cpu_lion", getattr(ds.ops.lion, "DeepSpeedCPULion", None)),
            ]
        for name, expected in cases:
            if expected is not None:
                assert isinstance(OPTIMIZERS.build([parameter], type=name, lr=0.001), expected)
