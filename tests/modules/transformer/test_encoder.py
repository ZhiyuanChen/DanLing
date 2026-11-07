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
from torch import nn

from danling.modules.transformer.encoder import TransformerEncoder, TransformerEncoderLayer


@pytest.fixture
def encoder_layer():
    return TransformerEncoderLayer(
        embed_dim=8, num_heads=2, dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0, batch_first=True
    )


class TestEncoderLayer:
    def test_forward(self, encoder_layer):
        inputs = torch.randn(2, 4, 8)
        outputs, weights = encoder_layer(inputs, need_weights=True)
        assert outputs.shape == inputs.shape
        assert weights.shape == (2, 2, 4, 4)

    def test_residual(self):
        inputs = torch.randn(2, 4, 8) + 5
        layer = TransformerEncoderLayer(8, 2, norm_first=True, dropout=0, attn_dropout=0, ffn_dropout=0)
        with torch.no_grad():
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    module.weight.zero_()
                    if module.bias is not None:
                        module.bias.zero_()
        outputs, _ = layer(inputs)
        torch.testing.assert_close(outputs, inputs)


class TestEncoder:
    def test_weights(self, encoder_layer):
        encoder = TransformerEncoder(encoder_layer, num_layers=2)
        inputs = torch.randn(2, 4, 8)
        outputs, weights = encoder(inputs, need_weights=True)
        assert outputs.shape == inputs.shape
        assert weights.shape == (2, 2, 2, 4, 4)

    def test_independent(self, encoder_layer):
        encoder = TransformerEncoder(encoder_layer, num_layers=2, share_parameters=False)
        first, second = (next(layer.parameters()) for layer in encoder.layers)
        original = second.detach().clone()
        with torch.no_grad():
            first.add_(1)
        torch.testing.assert_close(second, original)
