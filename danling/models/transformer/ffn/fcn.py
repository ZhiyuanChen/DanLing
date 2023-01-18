from typing import Any, Dict, Optional

from torch import Tensor, nn


class FullyConnectedNetwork(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        activation: str = "GELU",
        ffn_dropout: float = 0.0,
        **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        super(FullyConnectedNetwork, self).__init__()
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
