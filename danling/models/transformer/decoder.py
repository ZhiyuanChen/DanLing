from functools import partial
from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadAttention
from .ffn import FullyConnectedNetwork


class TransformerDecoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(  # pylint: disable=R0913, R0914
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        activation: str = "GELU",
        layer_norm_eps: float = 1e-5,
        scale_factor: float = 1.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        batch_first: bool = True,
        norm_first: bool = False,
        Attention: Type[nn.Module] = MultiHeadAttention,
        FeedForwardNetwork: Type[nn.Module] = FullyConnectedNetwork,
        **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        super().__init__()
        if ffn_dim is None:
            ffn_dim = embed_dim * 4
        self.norm_first = norm_first
        self.self_attn = Attention(  # type: ignore
            embed_dim,
            num_heads,
            attn_dropout=attn_dropout,
            scale_factor=scale_factor,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            batch_first=batch_first,
            **kwargs
        )
        self.cross_attn = Attention(  # type: ignore
            embed_dim,
            num_heads,
            attn_dropout=attn_dropout,
            scale_factor=scale_factor,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            batch_first=batch_first,
            **kwargs
        )
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, activation, ffn_dropout, **kwargs)  # type: ignore
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(  # pylint: disable=R0913
        self,
        tgt: Tensor,
        mem: Tensor,
        tgt_bias: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        mem_bias: Optional[Tensor] = None,
        mem_mask: Optional[Tensor] = None,
        mem_key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        if self.norm_first:
            tgt = self.norm1(tgt)

        self_attn, weights = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_bias=tgt_bias,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=need_weights,
        )
        self_attn = tgt + self.dropout(self_attn)
        cross_attn, weights = self.cross_attn(
            self_attn,
            mem,
            mem,
            attn_bias=mem_bias,
            attn_mask=mem_mask,
            key_padding_mask=mem_key_padding_mask,
            need_weights=need_weights,
        )
        cross_attn = self_attn + self.dropout(cross_attn)
        attn = self.norm1(cross_attn) if not self.norm_first else self.norm2(cross_attn)

        ffn = self.ffn(attn)
        ffn = attn + self.dropout(ffn)

        if not self.norm_first:
            ffn = self.norm2(ffn)

        return ffn, weights


class TransformerDecoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, layer: TransformerDecoderLayer, num_layers: int = 6) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([])
        self.layers.extend([layer for _ in range(self.num_layers)])

    def forward(  # pylint: disable=R0913
        self,
        tgt: Tensor,
        mem: Tensor,
        tgt_bias: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        mem_bias: Optional[Tensor] = None,
        mem_mask: Optional[Tensor] = None,
        mem_key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        gradient_checkpoint: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        # pylint: disable=E1101

        output = tgt
        # attn_weights is set to torch.empty(0, requires_grad=False) to avoid errors in DDP
        attn_weights = [] if need_weights else torch.empty(0, requires_grad=False)

        for layer in self.layers:
            if gradient_checkpoint and self.training:
                layer = partial(checkpoint, layer)
                need_weights = torch.tensor(need_weights)  # type: ignore
            output, weights = layer(
                output,
                mem,
                tgt_bias,
                tgt_mask,
                tgt_key_padding_mask,
                mem_bias,
                mem_mask,
                mem_key_padding_mask,
                need_weights,
            )
            if need_weights:
                attn_weights.append(weights)  # type: ignore

        if need_weights:
            attn_weights = torch.stack(attn_weights).cpu().detach()  # type: ignore

        return output, attn_weights  # type: ignore
