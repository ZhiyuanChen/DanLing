# DanLing
# Copyright (C) 2022-Present  DanLing

# This program is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from torch import Tensor, nn


class FullyConnectedNetwork(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        activation: str = "GELU",
        ffn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = getattr(nn, activation)()
        self.dropout = nn.Dropout(ffn_dropout)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
