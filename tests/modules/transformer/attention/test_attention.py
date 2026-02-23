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

from danling.modules.transformer.attention import MultiHeadAttention, SimpleAttention


@pytest.mark.parametrize("attention_cls", [MultiHeadAttention, SimpleAttention])
def test_attention_forward_shape(attention_cls):
    attention = attention_cls(embed_dim=8, num_heads=2, attn_dropout=0.0, batch_first=False)
    query = torch.randn(4, 2, 8)

    outputs, weights = attention(query, query, query, need_weights=True)

    assert outputs.shape == query.shape
    assert weights.shape == (2, 2, 4, 4)


@pytest.mark.parametrize("attention_cls", [MultiHeadAttention, SimpleAttention])
def test_attention_requires_embed_dim_divisible_by_num_heads(attention_cls):
    with pytest.raises(ValueError, match="not divisible"):
        attention_cls(embed_dim=10, num_heads=3)
