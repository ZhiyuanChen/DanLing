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

from danling.modules.mlp import MLP, Dense


def test_dense_forward_shape_with_residual():
    module = Dense(in_features=8, out_features=8, dropout=0.0)
    inputs = torch.randn(2, 8)

    outputs = module(inputs)

    assert outputs.shape == inputs.shape


def test_dense_without_residual_uses_output_features():
    module = Dense(
        in_features=8,
        out_features=4,
        norm=None,
        activation=None,
        dropout=0.0,
        pool=None,
        residual=False,
    )
    inputs = torch.randn(2, 8)

    outputs = module(inputs)

    assert outputs.shape == (2, 4)


def test_mlp_forward_shape():
    model = MLP(8, 16, 4, dropout=0.0)
    inputs = torch.randn(3, 8)

    outputs = model(inputs)

    assert outputs.shape == (3, 4)


def test_mlp_requires_multiple_feature_sizes():
    with pytest.raises(ValueError, match="at least 2 elements"):
        MLP([8])
