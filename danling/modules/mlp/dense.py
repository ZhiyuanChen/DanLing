from torch import nn


class Dense(nn.Module):
    def __init__(  # pylint: disable=R0913
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        norm: str = "LayerNorm",
        activation: str = "ReLU",
        bias: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm = getattr(nn, norm)(out_features)
        self.activation = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(out_features) if residual else None

    def forward(self, x):  # pylint: disable=C0103
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        if self.pool is not None:
            out = out + self.pool(x)
        return out
