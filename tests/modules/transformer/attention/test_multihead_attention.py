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

from danling.modules.transformer.attention.multihead_attention import MultiHeadAttention


class TestMultiHeadAttention:
    @pytest.mark.parametrize("batch_first", [True, False])
    def test_forward(self, batch_first):
        query = torch.randn(1, 4, 8, requires_grad=True)
        attention = MultiHeadAttention(embed_dim=8, num_heads=2, attn_dropout=0.0, batch_first=batch_first)
        value = query if batch_first else query.transpose(0, 1)
        output, weights = attention(value, value, value, need_weights=True)
        assert output.shape == value.shape
        assert weights.shape == (1, 2, 4, 4)
        output.square().sum().backward()
        assert query.grad is not None
        assert torch.isfinite(query.grad).all()

    def test_cross(self):
        query = torch.randn(1, 4, 8, requires_grad=True)
        memory = torch.randn(1, 5, 8, requires_grad=True)
        attention = MultiHeadAttention(embed_dim=8, num_heads=2, batch_first=False)
        value = query.transpose(0, 1)
        context = memory.transpose(0, 1)
        output, _ = attention(value, context, context)
        output.sum().backward()
        assert output.shape == value.shape
        assert query.grad is not None and memory.grad is not None

    def test_heads(self):
        with pytest.raises(ValueError, match="not divisible"):
            MultiHeadAttention(embed_dim=10, num_heads=3)
