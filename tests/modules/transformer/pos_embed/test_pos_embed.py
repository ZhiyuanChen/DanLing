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

from danling.modules.transformer.pos_embed.pos_embed import UnitedPositionEmbedding, relative_position_bucket


class TestRelativePositionBucket:
    def test_shape(self):
        bucket = relative_position_bucket(seq_len_max=6, num_buckets=8, max_distance=16)

        assert bucket.shape == (6, 6)
        assert not bucket.requires_grad


class TestUnitedPositionEmbedding:
    def test_forward(self):
        embedding = UnitedPositionEmbedding(
            embed_dim=8,
            num_heads=2,
            seq_len_max=6,
            pos_embed_dropout=0.0,
            has_cls_token=True,
        )
        inputs = torch.randn(2, 5, 8)

        outputs = embedding(inputs)

        assert outputs.shape == (4, 5, 5)
