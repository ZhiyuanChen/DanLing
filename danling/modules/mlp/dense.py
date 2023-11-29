from torch import nn


class Dense(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm: str = "LayerNorm",
        activation: str = "ReLU",
        dropout: float = 0.1,
        pool: str = "AdaptiveAvgPool1d",
        bias: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm = getattr(nn, norm)(out_features) if norm else nn.Identity()
        self.activation = getattr(nn, activation)() if activation else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.pool = getattr(nn, pool)(out_features) if pool else nn.Identity() if self.residual else None

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        if self.residual:
            out = out + self.pool(x)
        return out
