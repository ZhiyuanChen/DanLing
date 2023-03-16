from functools import partial
from typing import List

from torch import nn

from .dense import Dense


class MLP(nn.Module):
    def __init__(  # pylint: disable=R0913
        self,
        features: List[int],
        dropout: float = 0.1,
        norm: str = "LayerNorm",
        activation: str = "ReLU",
        bias: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if not len(features) > 1:
            raise ValueError(f"`features` of MLP should have at least 2 elements, but got {len(features)}")
        dense = partial(Dense, dropout=dropout, norm=norm, activation=activation, bias=bias, residual=residual)
        self.layers = nn.Sequential(
            *[dense(in_features, out_features) for in_features, out_features in zip(features, features[1:])]
        )

    def forward(self, x):  # pylint: disable=C0103
        out = self.layers(x)
        return out
