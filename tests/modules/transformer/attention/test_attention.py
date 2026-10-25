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
@pytest.mark.parametrize("batch_first", [True, False])
def test_attention_forward_shape(attention_cls, batch_first):
    attention = attention_cls(embed_dim=8, num_heads=2, attn_dropout=0.0, batch_first=batch_first)
    query = torch.randn((1, 4, 8) if batch_first else (4, 1, 8), requires_grad=True)

    outputs, weights = attention(query, query, query, need_weights=True)

    assert outputs.shape == query.shape
    assert weights.shape == (1, 2, 4, 4)
    outputs.square().sum().backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()


@pytest.mark.parametrize("attention_cls", [MultiHeadAttention, SimpleAttention])
def test_attention_cross_attention(attention_cls):
    attention = attention_cls(embed_dim=8, num_heads=2, batch_first=False)
    query, memory = torch.randn(3, 1, 8, requires_grad=True), torch.randn(5, 1, 8, requires_grad=True)
    output, _ = attention(query, memory, memory)
    output.sum().backward()
    assert output.shape == query.shape
    assert query.grad is not None and memory.grad is not None


@pytest.mark.parametrize("attention_cls", [MultiHeadAttention, SimpleAttention])
def test_attention_requires_embed_dim_divisible_by_num_heads(attention_cls):
    with pytest.raises(ValueError, match="not divisible"):
        attention_cls(embed_dim=10, num_heads=3)
