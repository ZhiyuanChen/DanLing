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

from functools import partial
from operator import index
from typing import Sequence, SupportsIndex

from torch import nn

from .dense import Dense


class MLP(nn.Module):
    """Construct a multilayer perceptron from integer feature widths.

    Pass widths as positional arguments or as one sequence, including the input
    and output widths. At least two widths are required. With ``linear_output``,
    the final layer is a plain linear projection; preceding layers use ``Dense``.
    """

    def __init__(
        self,
        *features: SupportsIndex | Sequence[SupportsIndex],
        norm: str = "LayerNorm",
        activation: str = "ReLU",
        dropout: float = 0.1,
        pool: str = "AdaptiveAvgPool1d",
        bias: bool = True,
        residual: bool = True,
        linear_output: bool = True,
    ) -> None:
        super().__init__()
        feature_sizes = features[0] if len(features) == 1 and isinstance(features[0], Sequence) else features
        widths: list[int] = []
        for size in feature_sizes:
            if not isinstance(size, SupportsIndex):
                raise TypeError("MLP feature sizes must be integers; pass sizes separately or as one sequence.")
            widths.append(index(size))
        if len(widths) < 2:
            raise ValueError(f"`features` of MLP should have at least 2 elements, but got {len(widths)}")
        dense = partial(
            Dense,
            norm=norm,
            activation=activation,
            dropout=dropout,
            pool=pool,
            bias=bias,
            residual=residual,
        )
        layers: list[nn.Module]
        if linear_output:
            layers = [dense(in_features, out_features) for in_features, out_features in zip(widths, widths[1:-1])]
            layers.append(nn.Linear(widths[-2], widths[-1], bias=bias))
        else:
            layers = [dense(in_features, out_features) for in_features, out_features in zip(widths, widths[1:])]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
