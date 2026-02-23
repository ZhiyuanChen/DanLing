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

from danling.modules.transformer.ffn import FullyConnectedNetwork


def test_fully_connected_network_preserves_embed_dim():
    network = FullyConnectedNetwork(embed_dim=8, ffn_dim=16, ffn_dropout=0.0)
    inputs = torch.randn(2, 4, 8)

    outputs = network(inputs)

    assert outputs.shape == inputs.shape
