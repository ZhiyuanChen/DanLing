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

from danling.modules.transformer import TransformerEncoder, TransformerEncoderLayer


def _build_encoder_layer() -> TransformerEncoderLayer:
    return TransformerEncoderLayer(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        batch_first=True,
    )


def test_transformer_encoder_layer_forward_shape():
    layer = _build_encoder_layer()
    inputs = torch.randn(2, 4, 8)

    outputs, weights = layer(inputs, need_weights=True)

    assert outputs.shape == inputs.shape
    assert weights.shape == (2, 2, 4, 4)


def test_transformer_encoder_stacks_attention_weights():
    encoder = TransformerEncoder(_build_encoder_layer(), num_layers=2)
    inputs = torch.randn(2, 4, 8)

    outputs, weights = encoder(inputs, need_weights=True)

    assert outputs.shape == inputs.shape
    assert weights.shape == (2, 2, 2, 4, 4)


def test_pre_norm_zero_branches_preserve_residual():
    from torch import nn

    from danling.modules.transformer import TransformerDecoderLayer

    inputs = torch.randn(2, 4, 8) + 5
    for cls in (TransformerEncoderLayer, TransformerDecoderLayer):
        layer = cls(8, 2, norm_first=True, dropout=0, attn_dropout=0, ffn_dropout=0)
        with torch.no_grad():
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    module.weight.zero_()
                    if module.bias is not None:
                        module.bias.zero_()
        outputs, _ = layer(inputs) if cls is TransformerEncoderLayer else layer(inputs, inputs)
        torch.testing.assert_close(outputs, inputs)


def test_encoder_can_build_independent_layers():
    encoder = TransformerEncoder(_build_encoder_layer(), num_layers=2, share_parameters=False)
    first, second = (next(layer.parameters()) for layer in encoder.layers)
    original = second.detach().clone()
    with torch.no_grad():
        first.add_(1)
    torch.testing.assert_close(second, original)
