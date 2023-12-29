from functools import partial
from typing import Sequence

from torch import nn

from .dense import Dense


class MLP(nn.Module):
    def __init__(
        self,
        *features: Sequence[int],
        norm: str = "LayerNorm",
        activation: str = "ReLU",
        dropout: float = 0.1,
        pool: str = "AdaptiveAvgPool1d",
        bias: bool = True,
        residual: bool = True,
        linear_output: bool = True,
    ) -> None:
        super().__init__()
        if len(features) == 1 and isinstance(features, Sequence):
            features = features[0]  # type: ignore[assignment]
        if not len(features) > 1:
            raise ValueError(f"`features` of MLP should have at least 2 elements, but got {len(features)}")
        dense = partial(
            Dense,
            norm=norm,
            activation=activation,
            dropout=dropout,
            pool=pool,
            bias=bias,
            residual=residual,
        )
        if linear_output:
            layers = [dense(in_features, out_features) for in_features, out_features in zip(features, features[1:-1])]
            layers.append(nn.Linear(features[-2], features[-1], bias=bias))
        else:
            layers = [dense(in_features, out_features) for in_features, out_features in zip(features, features[1:])]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
