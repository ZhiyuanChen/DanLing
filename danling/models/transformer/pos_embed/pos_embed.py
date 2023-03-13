import math
from typing import Optional

import torch
from torch import Tensor, nn


# this is from T5
def relative_position_bucket(
    seq_len_max: int,
    bidirectional: bool = True,
    num_buckets: int = 32,
    max_distance: int = 128,
):
    r"""Creates a bucketized version of the relative position indices.

      The relative position index is computed as:

    .. math::
          \text{rel_pos} = \text{rel_pos_0} + \ldots + \text{rel_pos_N-1},

      where :math:`\text{rel_pos_i}` is the ith relative position in the range
      :math:`[0, \text{seq\_len} - 1]`, :math:`\text{rel_pos_0}` is the first relative
      position, and :math:`\text{rel_pos_N-1}` is the last relative position.

      Args:
          seq_len_max (int): The maximum sequence length.
          bidirectional (bool, optional): Whether the model is bidirectional.
              Defaults to True.
          num_buckets (int, optional): The number of buckets. Defaults to 32.
          max_distance (int, optional): The maximum distance. Defaults to 128.
    """
    # pylint: disable=E1101

    context_position = torch.arange(seq_len_max, dtype=torch.long)[:, None]
    memory_position = torch.arange(seq_len_max, dtype=torch.long)[None, :]
    relative_position = memory_position - context_position
    ret = torch.tensor(0)
    negatives = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret = (negatives < 0).long() * num_buckets  # mtf.to_int32(mtf.less(negatives, 0)) * num_buckets
        negatives = torch.abs(negatives)
    else:
        negatives = torch.max(negatives, torch.zeros_like(negatives))
    # now negatives is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = negatives < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = torch.log(negatives.float() / max_exact) / math.log(max_distance / max_exact)
    val_if_large = max_exact + (val_if_large * (num_buckets - max_exact)).long()
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, negatives, val_if_large)
    return nn.Parameter(ret, requires_grad=False)


class UnitedPositionEmbedding(nn.Module):  # pylint: disable=R0902
    r"""United Position Embedding
    See `Rethinking Positional Encoding in Language Pre-training <https://arxiv.org/abs/2006.15595>`_
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        k_dim: total number of features in key. Default: None.
        v_dim: total number of features in value. Default: None.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
    Note that if :attr:`k_dim` and :attr:`v_dim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.

    Examples:

    ```python
    >>> query, key, value = torch.rand(8, 32, 192).chunk(3, dim=-1)
    >>> multihead_attn = nn.MultiheadAttention(64, 8)
    >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    ```
    """

    def __init__(  # pylint: disable=R0913
        self,
        embed_dim: int,
        num_heads: int,
        seq_len_max: int,
        rel_pos_embed: bool = False,
        rel_pos_embed_buckets: int = 32,
        rel_pos_embed_max: int = 128,
        pos_embed_dropout: float = 0.0,
        pos_scale_factor: int = 1,
        has_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_len_max = seq_len_max
        self.has_cls_token = has_cls_token
        self.dropout = nn.Dropout(pos_embed_dropout)
        if self.has_cls_token:
            # make room for [CLS]-to-others and others-to-[CLS]
            self.seq_len_max += 2
        self.abs_pos_embed = nn.Parameter(torch.randn(self.seq_len_max, self.embed_dim))  # pylint: disable=E1101
        self.norm = nn.LayerNorm(self.embed_dim)
        self.in_proj = nn.Linear(self.embed_dim, self.embed_dim * 2)
        self.scaling = (embed_dim / num_heads * pos_scale_factor) ** -0.5

        self.rel_pos_embed = None
        if rel_pos_embed:
            assert rel_pos_embed_buckets % 2 == 0
            self.rel_pos_embed_buckets = rel_pos_embed_buckets
            self.rel_pos_embed_max = rel_pos_embed_max
            self.rel_pos_embed = nn.Embedding(self.rel_pos_embed_buckets + 1, self.num_heads)
            self.rel_pos_embed_bucket = relative_position_bucket(
                seq_len_max=self.seq_len_max,
                num_buckets=self.rel_pos_embed_buckets,
                max_distance=self.rel_pos_embed_max,
            )

    def forward(self, src: Tensor, cls_token_index: Optional[Tensor] = None) -> Tensor:
        # pylint: disable=C0103, C0116, E1101

        B, N, C = src.shape
        # 0 is for others-to-[CLS] 1 is for [CLS]-to-others
        # Assume the input is ordered. If your input token is permuted, you may need to update this accordingly
        if self.has_cls_token:
            # only plus 1 here since because [CLS] already plused 1
            N += 1
        weight = self.norm(self.abs_pos_embed[:N, :])
        q, k = self.in_proj(weight).reshape(N, 2, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        q = q * self.scaling
        pos_embed = torch.bmm(q, k.transpose(1, 2))
        if self.has_cls_token:
            # p_0 \dot p_0 is [CLS]-to-others
            cls_2_others = pos_embed[:, 0, 0]
            # p_1 \dot p_1 is others-to-[CLS]
            others_2_cls = pos_embed[:, 1, 1]
            # offset
            pos_embed = pos_embed[:, 1:, 1:]
            # if [CLS] is not the first token
            if cls_token_index is not None:
                pos_embed = pos_embed.repeat(B, 1, 1, 1)
                pos_embed[torch.arange(B), :, cls_token_index, :] = cls_2_others.expand(B, -1).unsqueeze(-1)
                pos_embed[torch.arange(B), :, :, cls_token_index] = others_2_cls.expand(B, -1).unsqueeze(-1)
            else:
                pos_embed[:, 0, :] = cls_2_others.unsqueeze(-1)
                pos_embed[:, :, 0] = others_2_cls.unsqueeze(-1)
            N -= 1
        rel_pos_embed = torch.zeros_like(pos_embed)
        if self.rel_pos_embed:
            rel_pos_embed_bucket = self.rel_pos_embed_bucket[:N, :N]
            if self.has_cls_token:
                if cls_token_index is not None:
                    rel_pos_embed_bucket = rel_pos_embed_bucket.repeat(B, 1, 1)
                    rel_pos_embed_bucket[torch.arange(B), cls_token_index, :] = self.rel_pos_embed_buckets // 2
                    rel_pos_embed_bucket[torch.arange(B), :, cls_token_index] = self.rel_pos_embed_buckets
                    rel_pos_embed = self.rel_pos_embed(rel_pos_embed_bucket).permute(0, 3, 1, 2)
                else:
                    rel_pos_embed_bucket[0, :] = self.rel_pos_embed_buckets // 2
                    rel_pos_embed_bucket[:, 0] = self.rel_pos_embed_buckets
                    rel_pos_embed = self.rel_pos_embed(rel_pos_embed_bucket).permute(2, 0, 1)
            else:
                rel_pos_embed = self.rel_pos_embed(rel_pos_embed_bucket).permute(2, 0, 1)
            pos_embed += rel_pos_embed

        pos_embed = (
            pos_embed.view(-1, *pos_embed.shape[2:]) if cls_token_index is not None else pos_embed.repeat(B, 1, 1)
        )
        return self.dropout(pos_embed)
