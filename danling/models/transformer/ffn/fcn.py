from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FullyConnectedNetwork(nn.Module):

    def __init__(self, embed_dim: int, feedforward_size: int,
                 activation: Optional[str] = 'GELU',
                 feedforward_dropout: Optional[float] = 0.,
                 **kwargs: Optional[Dict[str, Any]]) -> None:
        super(FullyConnectedNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_dim, feedforward_size)
        self.activation = getattr(nn, activation)()
        self.dropout = nn.Dropout(feedforward_dropout)
        self.fc2 = nn.Linear(feedforward_size, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
