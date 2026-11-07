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

from danling.modules.mlp.dense import Dense


class TestDense:
    def test_forward(self):
        inputs = torch.randn(2, 8)
        module = Dense(in_features=8, out_features=8, dropout=0.0)
        assert module(inputs).shape == inputs.shape

    def test_output_features(self):
        inputs = torch.randn(2, 8)
        module = Dense(
            in_features=8, out_features=4, norm=None, activation=None, dropout=0.0, pool=None, residual=False
        )
        assert module(inputs).shape == (2, 4)
