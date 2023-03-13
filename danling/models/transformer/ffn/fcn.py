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

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=C0103
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
