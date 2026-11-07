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

from danling.modules.mlp.mlp import MLP


class TestMLP:
    @pytest.mark.parametrize("features", [(8, 16, 4), ([8, 16, 4],), ((8, 16, 4),)])
    @pytest.mark.parametrize("linear_output", [True, False])
    def test_forward(self, features, linear_output):
        model = MLP(*features, dropout=0.0, linear_output=linear_output)
        assert model(torch.randn(3, 8)).shape == (3, 4)

    @pytest.mark.parametrize("features", [(), (8,), ([8],)])
    def test_feature_count(self, features):
        with pytest.raises(ValueError, match="at least 2 elements"):
            MLP(*features)

    def test_mixed_sizes(self):
        with pytest.raises(TypeError, match="feature sizes must be integers"):
            MLP(8, [16, 4])
