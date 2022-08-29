from functools import partial
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadAttention
from .ffn import FullyConnectedNetwork


class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        embed_dim: the number of expected features in the input (required).
        num_heads: the number of heads in the multi head attention models (required).
        ffn_dim: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(embed_dim=512, num_heads=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = decoder_layer(src)
    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(embed_dim=512, num_heads=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = decoder_layer(src)
    """
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: Optional[int] = 2048,
        dropout: Optional[float] = 0.1,
        attn_dropout: Optional[float] = 0.1,
        ffn_dropout: Optional[float] = 0.1,
        activation: Optional[str] = "GELU",
        layer_norm_eps: Optional[float] = 1e-5,
        scale_factor: Optional[float] = 1.0,
        bias: Optional[bool] = True,
        add_bias_kv: Optional[bool] = False,
        add_zero_attn: Optional[bool] = False,
        batch_first: Optional[bool] = True,
        norm_first: Optional[bool] = False,
        Attention: Optional[nn.Module] = MultiHeadAttention,
        FeedForwardNetwork: Optional[nn.Module] = FullyConnectedNetwork,
        **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.norm_first = norm_first
        self.self_attn = Attention(
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
        self.cross_attn = Attention(
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
        self.ffn = FeedForwardNetwork(
            embed_dim, ffn_dim, activation, ffn_dropout, **kwargs
        )
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        mem: Tensor,
        tgt_bias: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        mem_bias: Optional[Tensor] = None,
        mem_mask: Optional[Tensor] = None,
        mem_key_padding_mask: Optional[Tensor] = None,
        need_weights: Optional[bool] = False,
    ) -> Tensor:
        r"""Pass the input through the decoder layer.
        Args:
            src: the sequence to the decoder layer (required).
            attn_mask: the mask for the src sequence (optional).
            key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

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
    r"""TransformerDecoder is a stack of N decoder layers
    Args:
        num_layers: the number of sub-decoder-layers in the decoder (required).
        layer: the sub-decoder-layer in the decoder (default=TransformerDecoderLayer).
        drop_layer: the drop layer rate (default=0.0).
    Examples::
        >>> transformer_decoder = dl.model.TransformerDecoder(num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_decoder(src)
    """
    __constants__ = ["norm"]

    def __init__(
        self,
        layer: TransformerDecoderLayer,
        num_layers: Optional[int] = 6,
        **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([])
        self.layers.extend([layer for _ in range(self.num_layers)])

    def forward(
        self,
        tgt: Tensor,
        mem: Tensor,
        tgt_bias: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        mem_bias: Optional[Tensor] = None,
        mem_mask: Optional[Tensor] = None,
        mem_key_padding_mask: Optional[Tensor] = None,
        need_weights: Optional[bool] = False,
        gradient_checkpoint: Optional[bool] = False,
    ) -> Tensor:
        r"""Pass the input through the decoder layers in turn.
        Args:
            src: the sequence to the decoder (required).
            attn_mask: the mask for the src sequence (optional).
            key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        attn_weights = [] if need_weights else torch.zeros(0, requires_grad=False)

        for layer in self.layers:
            if gradient_checkpoint and self.training:
                layer = partial(checkpoint, layer)
                need_weights = torch.tensor(need_weights)
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
                attn_weights.append(weights)

        if need_weights:
            attn_weights = torch.stack(attn_weights).cpu().detach()

        return output, attn_weights
