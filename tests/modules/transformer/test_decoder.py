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

from danling.modules.transformer import TransformerDecoder, TransformerDecoderLayer


def _build_decoder_layer() -> TransformerDecoderLayer:
    return TransformerDecoderLayer(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        batch_first=True,
    )


def test_transformer_decoder_layer_forward_shape():
    layer = _build_decoder_layer()
    target = torch.randn(2, 3, 8)
    memory = torch.randn(2, 5, 8)

    outputs, weights = layer(target, memory, need_weights=True)

    assert outputs.shape == target.shape
    assert weights.shape == (2, 2, 3, 5)


def test_transformer_decoder_stacks_attention_weights():
    decoder = TransformerDecoder(_build_decoder_layer(), num_layers=2)
    target = torch.randn(2, 3, 8)
    memory = torch.randn(2, 5, 8)

    outputs, weights = decoder(target, memory, need_weights=True)

    assert outputs.shape == target.shape
    assert weights.shape == (2, 2, 2, 3, 5)
