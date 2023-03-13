import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SimpleAttention(nn.Module):  # pylint: disable=R0902
    def __init__(  # pylint: disable=R0913
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        scale_factor: float = 1.0,
        bias: bool = True,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale_factor = scale_factor
        self.batch_first = batch_first
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = float(self.head_dim * self.scale_factor) ** -0.5
        if not self.head_dim * self.num_heads == self.embed_dim:
            raise ValueError(f"embed_dim {self.embed_dim} not divisible by num_heads {self.num_heads}")

        self.in_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=bias)
        self.dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.constant_(self.in_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(  # pylint: disable=R0912, R0913, R0914, R0915
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_bias: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # pylint: disable=C0103, E1101, R0801

        if self.batch_first:
            query, key, value = [x.transpose(0, 1) for x in (query, key, value)]

        # set up shape vars
        target_len, batch_size, embed_dim = query.shape
        source_len, _, _ = key.shape
        if not key.shape[:2] == value.shape[:2]:
            raise ValueError(f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}")

        q, k, v = self.in_projection(query, key, value)

        # prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "attn_mask is of type uint8. This type is deprecated. Please use bool or float tensors instead."
                )
                attn_mask = attn_mask.to(torch.bool)
            elif not (attn_mask.is_floating_point() or attn_mask.dtype == torch.bool):
                raise ValueError(f"attn_mask should have type float or bool, but got {attn_mask.dtype}.")
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_shape = (target_len, source_len)
                if attn_mask.shape != correct_shape:
                    raise ValueError(f"attn_mask should have shape {correct_shape}, but got {attn_mask.shape}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_shape = (batch_size * self.num_heads, target_len, source_len)  # type: ignore
                if attn_mask.shape != correct_shape:
                    raise ValueError(f"attn_mask should have shape {correct_shape}, but got {attn_mask.shape}.")
            else:
                raise RuntimeError(f"attn_mask should have dimension 2 or 3, bug got {attn_mask.dim()}.")

        # prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "key_padding_mask is of type uint8. This type is deprecated. Please use bool or float tensors instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        # reshape q, k, v for multihead attention and make em batch first
        q = q.reshape(target_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.reshape(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, source_len):
                raise ValueError(
                    f"key_padding_mask should have shape {(batch_size, source_len)}, but got {key_padding_mask.shape}"
                )
            key_padding_mask = (
                key_padding_mask.view(batch_size, 1, 1, source_len)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(batch_size * self.num_heads, 1, source_len)
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        # (deep breath) calculate attention and out projection
        attn_output, attn_output_weights = self.attention(q, k, v, attn_bias, attn_mask)
        attn_output = attn_output.transpose(0, 1).reshape(target_len, batch_size, embed_dim)
        attn_output = self.out_projection(attn_output)

        # attn_output_weights is set to torch.empty(0, requires_grad=False) to avoid errors in DDP
        attn_output_weights = (
            attn_output_weights.view(batch_size, self.num_heads, target_len, source_len)
            if need_weights
            else torch.empty(0, requires_grad=False)
        )

        if self.batch_first:
            return attn_output.transpose(0, 1), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def in_projection(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # pylint: disable=C0103, E1101

        if k is v:
            # self-attention
            if q is k:
                return self.in_proj(q).chunk(3, dim=-1)
            # encoder-decoder attention
            else:
                w_q, w_kv = self.in_proj.weight.split(self.embed_dim)
                b_q, b_kv = None, None
                if self.in_proj.bias is not None:
                    b_q, b_kv = self.in_proj.bias.split((self.embed_dim, self.embed_dim * 2))
                return (F.linear(q, w_q, b_q),) + tuple(F.linear(k, w_kv, b_kv).chunk(2, dim=-1))  # type: ignore
        else:
            w_q, w_k, w_v = self.in_proj.weight.chunk(3, -1)
            b_q, b_k, b_v = None, None, None
            if self.in_proj.bias is not None:
                b_q, b_k, b_v = self.in_proj.bias.chunk(3)
            return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    def attention(  # pylint: disable=R0913
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_bias: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # pylint: disable=C0103, E1101, R0801

        q *= self.scaling
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1))
        if attn_bias is not None:
            attn += attn_bias
        if attn_mask is not None:
            attn += attn_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)
        return output, attn

    def out_projection(self, attn_output: Tensor) -> Tensor:
        return self.out_proj(attn_output)
