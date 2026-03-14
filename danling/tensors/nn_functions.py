# DanLing
# Copyright (C) 2022-Present  DanLing

# This file is part of DanLing.

# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.
r"""
``torch.nn.functional.*`` overrides for NestedTensor via ``__torch_function__``.

This module is the **Level 3** dispatch layer, registering handlers for
``F.linear``, ``F.conv*``, ``F.max_pool*``, ``F.embedding``,
``F.layer_norm``, ``F.scaled_dot_product_attention``, and other
``torch.nn.functional`` ops.

Most ops here use **per-element dispatch** ([_apply_per_element][]) because
they require spatial-dimension awareness that the packed ``_values`` tensor
cannot provide.  A few normalisation ops (``F.layer_norm``, ``F.rms_norm``)
have a fast **packed path** ([_apply_packed][]) when the normalised shape
covers the full non-batch extent.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from .ops import (
    NestedTensorFuncRegistry,
    _concat_apply,
    _concat_apply_same_shape,
    _concat_dim_for_tensor_dim,
    _ensure_nested_input,
    _map_storage_pair,
    _map_storage_serial,
    _normalize_shape_tuple,
    _packed_layer_norm,
    _packed_rms_norm,
    _translate_non_batch_dim,
    _validate_probability,
)

if TYPE_CHECKING:
    from .nested_tensor import NestedTensor

_LOW_PRECISION_CUDA_DTYPES = {torch.float16, torch.bfloat16}

try:
    import torch.nn.attention.flex_attention as _torch_flex_attention_module
    from torch.nn.attention.flex_attention import AuxOutput as _FlexAuxOutput
    from torch.nn.attention.flex_attention import BlockMask as _TorchBlockMask
    from torch.nn.attention.flex_attention import create_block_mask as _torch_create_block_mask
    from torch.nn.attention.flex_attention import flex_attention as _torch_flex_attention
except Exception:
    _torch_flex_attention_module = None
    _torch_flex_attention = None
    _torch_create_block_mask = None
    _FlexAuxOutput = None
    _TorchBlockMask = None


# Helpers


def _extract_concat_from_padded(ref, v: Tensor) -> Tensor:
    r"""
    Extract valid elements from padded tensor *v*, matching ``ref.concat`` layout.
    """
    packed = ref._dense_to_packed_values(v)
    if packed is None:
        raise ValueError(
            "Cannot extract concatenated values from padded tensor that does not cover NestedTensor shape."
        )
    return packed


def _concat_tensors(*values: NestedTensor | Tensor) -> tuple[Tensor, ...]:
    r"""
    Return tensors for each input, expanding NestedTensor values to their concatenated representation.

    Args:
        *values: NestedTensor or Tensor inputs to expand.

    Returns:
        tuple[Tensor, ...]: A tuple of tensors with NestedTensors expanded to concatenated form.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.nn_functions import _concat_tensors
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))
        >>> a, b = _concat_tensors(nt, nt)
        >>> torch.equal(a, nt.concat) and torch.equal(b, nt.concat)
        True
    """
    from .nested_tensor import NestedTensor

    ref = None
    for v in values:
        if isinstance(v, NestedTensor):
            ref = v
            break

    if ref is None:
        return tuple(values)

    result = []
    for v in values:
        if isinstance(v, NestedTensor):
            result.append(v.concat)  # type: ignore[union-attr]
        else:
            if v.shape != ref.shape:  # type: ignore[union-attr]
                raise ValueError(
                    f"Cannot apply NestedTensor mask (shape {tuple(ref.shape)}) "  # type: ignore[union-attr]
                    f"to tensor of shape {tuple(v.shape)}"
                )
            result.append(_extract_concat_from_padded(ref, v))
    return tuple(result)


def _attention_outer_size(source: NestedTensor, num_heads: int, head_dim: int) -> torch.Size:
    max_seq = source._logical_shape[1 if source.batch_first else 0]
    if source.batch_first:
        return torch.Size((len(source), num_heads, max_seq, head_dim))
    return torch.Size((num_heads, len(source), max_seq, head_dim))


def _project_attention_values(source: NestedTensor, values: Tensor, num_heads: int, head_dim: int) -> NestedTensor:
    from .aten_functions import _packed_with_shape

    shape, packed_sizes, element_shapes = source._shape_meta_from_components(
        prefix=(num_heads,),
        keep_dims=(0,),
        suffix=(head_dim,),
    )
    return _packed_with_shape(
        source,
        values,
        shape,
        _attention_outer_size(source, num_heads, head_dim),
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
    )


def _outer_shape(source: NestedTensor) -> torch.Size:
    with torch._C.DisableTorchFunctionSubclass():
        return torch.Tensor.size(source)


def _packed_pair_offsets_match(lhs: NestedTensor, rhs: NestedTensor) -> bool:
    r"""Return whether two NestedTensors can share a packed fast path."""
    from .aten_functions import _is_fake_tensor, _structure_match

    if _is_fake_tensor(lhs._values) or _is_fake_tensor(rhs._values):
        return True
    return _structure_match(lhs, rhs)


def _nested_to_padded_tensor(source: NestedTensor) -> Tensor:
    return source.tensor


def _nested_sequence_valid_mask(source: NestedTensor) -> Tensor:
    if len(source) == 0:
        return torch.empty((0, 0), dtype=torch.bool, device=source.device)
    outer = _outer_shape(source)
    lengths = source._ragged_level_sizes(0).to(device=source.device, dtype=torch.long)
    max_len = outer[1 if source.batch_first else 0]
    positions = torch.arange(max_len, device=source.device, dtype=torch.long)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def _nested_attention_valid_mask(source: NestedTensor, key_tensor: Tensor) -> Tensor:
    rank = source._physical_shape.size(1)
    if rank <= 1:
        return _nested_sequence_valid_mask(source)

    sizes = source._physical_shape[:, :-1].to(device=source.device, dtype=torch.long)
    padded_shape = key_tensor.shape[1:-1]
    valid = torch.ones((len(source), *padded_shape), dtype=torch.bool, device=source.device)
    for dim, max_size in enumerate(padded_shape):
        coord_shape = [1] * (len(padded_shape) + 1)
        coord_shape[dim + 1] = max_size
        coord = torch.arange(max_size, device=source.device, dtype=torch.long).view(coord_shape)
        size_shape = [len(source)] + [1] * len(padded_shape)
        valid = valid & (coord < sizes[:, dim].view(size_shape))
    return valid


def _nested_from_padded_tensor(reference: NestedTensor, tensor: Tensor) -> NestedTensor:
    from .aten_functions import _packed_with_shape

    cls = type(reference)
    if len(reference) == 0:
        return cls([], **reference._meta(include_dtype=True))
    batch_dense = tensor if reference.batch_first else tensor.movedim(1, 0)
    values = batch_dense[reference._packed_dense_index(device=batch_dense.device)].contiguous()
    return _packed_with_shape(
        reference,
        values,
        reference._physical_shape_like_batch_dense(batch_dense.shape),
        torch.Size(tensor.shape),
        packed_sizes=reference._packed_sizes,
        element_shapes=reference._element_shapes_like_batch_dense(batch_dense.shape),
    )


def _apply_per_element(input: NestedTensor, op: Callable, *args, **kwargs) -> NestedTensor:
    r"""
    Applies an operator to each tensor in a NestedTensor serially.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.nn_functions import _apply_per_element
        >>> nt = NestedTensor(torch.tensor([1.0, -2.0]), torch.tensor([3.0, -4.0, 5.0]))
        >>> out = _apply_per_element(nt, torch.abs)
        >>> ref = torch.abs(nt.tensor)
        >>> torch.equal(out, ref)
        True
    """
    return _map_storage_serial(input, lambda t: op(t, *args, **kwargs))


def _apply_with_indices(input: NestedTensor, pool_fn: Callable, *args, **kwargs):
    r"""Applies a pooling op that returns (output, indices) to each element of a NestedTensor."""
    return _map_storage_pair(input, lambda t: pool_fn(t, *args, return_indices=True, **kwargs))


def _apply_pair(input: NestedTensor | Tensor, other: NestedTensor | Tensor, op: Callable, *args, **kwargs):
    r"""
    Applies a binary operator to a NestedTensor and another operand.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.nn_functions import _apply_pair
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        >>> other = torch.tensor([1.0, 1.0])
        >>> out = _apply_pair(nt, other, torch.add)
        >>> ref = torch.add(nt.tensor, other)
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    cls = type(input) if isinstance(input, NestedTensor) else type(other)
    input = _ensure_nested_input(input, other, cls)
    if len(input) == 0:
        return cls([], **input._meta(include_dtype=True))
    if isinstance(other, NestedTensor) and len(input) != len(other):
        raise ValueError(
            "NestedTensor batch length mismatch between input and other: " f"input={len(input)}, other={len(other)}"
        )
    if isinstance(other, NestedTensor):
        return cls((op(x, y, *args, **kwargs) for x, y in zip(input._unpack(), other._unpack())), **input._meta())
    return _map_storage_serial(input, lambda x: op(x, other, *args, **kwargs))


def _apply_packed(input: NestedTensor, op: Callable, *args, **kwargs) -> NestedTensor:
    r"""
    Applies an operator to a NestedTensor and return a NestedTensor result.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> from danling.tensors.nn_functions import _apply_packed
        >>> nt = NestedTensor(torch.tensor([1.0, -2.0]), torch.tensor([3.0, -4.0, 5.0]))
        >>> out = _apply_packed(nt, torch.abs)
        >>> ref = torch.abs(nt.tensor)
        >>> torch.equal(out, ref)
        True
    """
    return _concat_apply_same_shape(input, lambda t: op(t, *args, **kwargs))


def _from_batch_preserving_values(input: NestedTensor, values: Tensor) -> NestedTensor:
    r"""
    Rebuild NestedTensor metadata for ops that preserve the per-element leading (ragged) dimension.

    The packed ``_values`` layout concatenates per-element dim-0 along global dim-0.
    For ops where dim-0 is batch-like (conv/pool/interpolate), each element keeps its
    original dim-0 length and only trailing dims change uniformly.
    """
    from .aten_functions import _packed_with_shape

    out_physical_shape, outer_size, packed_sizes, element_shapes = input._leading_dim_preserving_meta(values.shape[1:])
    return _packed_with_shape(
        input,
        values,
        out_physical_shape,
        outer_size,
        permutation=input._permutation_after_replacing_trailing_dims(
            max(input._physical_shape.size(1) - 1, 0),
            len(values.shape[1:]),
        ),
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
    )


def _apply_batch_preserving_packed(input: NestedTensor, op: Callable, *args, **kwargs):
    r"""Apply *op* to packed ``_values`` for batch-preserving ops, supporting Tensor or (Tensor, Tensor) returns."""
    cls = type(input)
    if len(input) == 0:
        return cls([], **input._meta(include_dtype=True))
    result = op(input._values, *args, **kwargs)
    if isinstance(result, Tensor):
        return _from_batch_preserving_values(input, result)
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], Tensor)
        and isinstance(result[1], Tensor)
    ):
        return _from_batch_preserving_values(input, result[0]), _from_batch_preserving_values(input, result[1])
    raise TypeError(f"Unsupported return type from packed op {op}: {type(result)}")


# Activations
# Most activations (F.relu, F.gelu, F.silu, etc.) are NOT registered here.
# They fall through to torch.* registrations in torch_functions.py or aten dispatch
# in aten_functions.py via PyTorch's __torch_function__ chain.


# Attention


def _build_sdpa_mask(
    key: NestedTensor | Tensor,
    attn_mask: NestedTensor | Tensor | None,
    is_causal: bool,
    query_tensor: Tensor,
    key_tensor: Tensor,
) -> tuple:
    r"""
    Build combined attention mask for batched SDPA.

    Constructs a mask that prevents attention to padding key positions,
    combines with any user-provided attn_mask, and handles is_causal conflicts.

    Returns:
        (combined_mask, is_causal): The combined mask and updated is_causal flag.
    """
    from .nested_tensor import NestedTensor

    # 1. Build key padding mask from NestedTensor structure
    key_attn_mask = None
    if isinstance(key, NestedTensor):
        key_mask = _nested_attention_valid_mask(key, key_tensor)

        # Expand to match SDPA score dimensions (batch/head..., T_q, T_k).
        # Key mask is (..., T_k) -> (..., 1, T_k).
        target_ndim = query_tensor.dim()
        while key_mask.dim() < target_ndim - 1:
            key_mask = key_mask.unsqueeze(1)
        key_attn_mask = key_mask.unsqueeze(-2)
        while key_attn_mask.dim() < target_ndim:
            key_attn_mask = key_attn_mask.unsqueeze(1)

    # 2. Process user-provided attn_mask
    user_mask = None
    if attn_mask is not None:
        if isinstance(attn_mask, NestedTensor):
            user_mask = _nested_to_padded_tensor(attn_mask)
        else:
            user_mask = attn_mask

    # 3. Handle is_causal + mask conflict (can't pass both to SDPA)
    causal_mask = None
    if is_causal and (key_attn_mask is not None or user_mask is not None):
        q_len = query_tensor.size(-2)
        k_len = key_tensor.size(-2)
        causal_mask = torch.ones(q_len, k_len, dtype=torch.bool, device=query_tensor.device).tril()
        is_causal = False  # handled explicitly via the mask

    # 4. Combine all masks (all in True=attend convention)
    combined = None
    for m in [key_attn_mask, user_mask, causal_mask]:
        if m is None:
            continue
        if combined is None:
            combined = m
        elif combined.dtype == torch.bool and m.dtype == torch.bool:
            combined = combined & m
        else:
            # Mixed bool/float: convert bool to float (-inf for masked)
            if combined.dtype == torch.bool:
                combined = torch.where(
                    combined,
                    torch.zeros((), device=combined.device),
                    torch.tensor(float("-inf"), device=combined.device),
                )
            if m.dtype == torch.bool:
                m = torch.where(m, torch.zeros((), device=m.device), torch.tensor(float("-inf"), device=m.device))
            combined = combined + m

    return combined, is_causal


def _sdpa_pack_native(nt: NestedTensor) -> tuple[Tensor, Tensor, Tensor, int]:
    r"""
    Return the native varlen layout for ``(heads, seq_i, dim)`` elements:
    ``(sum_seq, heads, dim)`` plus cumulative sequence lengths.
    """
    if not _is_native_attention_layout(nt):
        raise ValueError("Native SDPA fast path requires elements shaped like (heads, seq, dim).")

    lengths_cpu = nt._ragged_level_sizes(0)
    cumulative = nt._ragged_level_offsets(0).to(device=nt.device, dtype=torch.int32)
    max_seqlen = int(lengths_cpu.max().item()) if lengths_cpu.numel() else 0
    return nt._values.contiguous(), lengths_cpu, cumulative, max_seqlen


def _sdpa_restore_native(attention: Tensor, query: NestedTensor) -> NestedTensor:
    r"""Restore fused-kernel output without unpacking per-element tensors."""
    from .aten_functions import _packed_with_shape

    if not _is_native_attention_layout(query):
        raise ValueError("Native SDPA restore requires elements shaped like (heads, seq, dim).")

    output_shape, packed_sizes, element_shapes = query._replace_trailing_physical_dims_meta((attention.size(-1),))
    return _packed_with_shape(
        query,
        attention.contiguous(),
        output_shape,
        query._logical_shape[:-1] + (attention.size(-1),),
        permutation=query._permutation_after_replacing_trailing_dims(1, 1),
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
    )


def _is_native_attention_layout(nt: NestedTensor) -> bool:
    r"""Return True when attention elements are stored as second-dim ragged packed values."""
    return (
        nt._physical_shape.size(1) == 3
        and nt._values.dim() == 3
        and nt._varying_dims == (1,)
        and nt._static_dims == (0, 2)
    )


def _flash_attention_forward_values(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_cumulative: Tensor,
    k_cumulative: Tensor,
    q_max: int,
    k_max: int,
    *,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
) -> Tensor:
    r"""Run varlen FlashAttention directly on packed ``(total_seq, heads, dim)`` values."""
    original_head_dim = query.size(-1)
    q_padded = _pad_last_dim_for_flash(query)
    k_padded = _pad_last_dim_for_flash(key)
    v_padded = _pad_last_dim_for_flash(value)
    softmax_scale = scale if scale is not None else original_head_dim**-0.5
    attention = torch.ops.aten._flash_attention_forward(
        q_padded,
        k_padded,
        v_padded,
        q_cumulative,
        k_cumulative,
        q_max,
        k_max,
        dropout_p,
        is_causal,
        False,
        scale=softmax_scale,
    )[0]
    if attention.size(-1) != original_head_dim:
        attention = attention[..., :original_head_dim]
    return attention


def _as_flex_packed_dense(nt: NestedTensor) -> Tensor:
    r"""Build a dense packed FlexAttention view shaped ``(1, heads, total_seq, dim)``."""
    if _torch_flex_attention is None:
        raise RuntimeError("FlexAttention is unavailable in this PyTorch build.")
    if not _is_native_attention_layout(nt):
        raise ValueError("DanLing FlexAttention expects elements shaped like (heads, seq, dim).")
    return nt._values.contiguous().movedim(1, 0).unsqueeze(0).contiguous()


def _restore_flex_dense_tensor(output: Tensor, query: NestedTensor) -> NestedTensor:
    r"""Restore dense packed FlexAttention outputs into DanLing NestedTensor storage."""
    from .aten_functions import _packed_with_shape

    if output.dim() < 3 or output.size(0) != 1:
        raise ValueError(
            "DanLing FlexAttention dense restore expects output shaped like (1, heads, total_seq, ...), "
            f"got {tuple(output.shape)}"
        )
    values = output.squeeze(0).movedim(0, 1).contiguous()
    tail_sizes = tuple(int(size) for size in values.shape[2:])
    shape_tensor, packed_sizes, element_shapes = query._drop_trailing_physical_dims_meta(1, suffix=tail_sizes)
    return _packed_with_shape(
        query,
        values,
        shape_tensor,
        query._logical_shape[:3] + tail_sizes,
        permutation=query._permutation_after_replacing_trailing_dims(1, len(tail_sizes)),
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
    )


def _restore_flex_output(output, query: NestedTensor):
    r"""Recursively convert dense packed FlexAttention outputs back into DanLing outputs."""
    if isinstance(output, Tensor) and output.dim() >= 3 and output.size(0) == 1:
        return _restore_flex_dense_tensor(output, query)
    if _FlexAuxOutput is not None and isinstance(output, _FlexAuxOutput):
        return type(output)(
            lse=_restore_flex_output(output.lse, query) if output.lse is not None else None,
            max_scores=_restore_flex_output(output.max_scores, query) if output.max_scores is not None else None,
        )
    if isinstance(output, tuple):
        return tuple(_restore_flex_output(item, query) for item in output)
    return output


def _flex_allow_all(_batch, _head, q_idx, _kv_idx):
    r"""Default FlexAttention mask that allows every position inside each sequence."""
    return torch.ones_like(q_idx, dtype=torch.bool)


def _flex_wrap_mask_mod(mask_mod: Callable, query: NestedTensor, key: NestedTensor) -> Callable:
    r"""Adapt a per-sequence DanLing Flex mask function to packed global indices."""
    q_offsets = query._ragged_level_offsets(0).to(device=query.device, dtype=torch.long)
    k_offsets = key._ragged_level_offsets(0).to(device=key.device, dtype=torch.long)
    q_ends = q_offsets[1:]
    k_ends = k_offsets[1:]

    def wrapped(batch, head, q_idx, kv_idx):
        q_batch = torch.searchsorted(q_ends, q_idx, right=True)
        kv_batch = torch.searchsorted(k_ends, kv_idx, right=True)
        same_sequence = q_batch == kv_batch
        q_local = q_idx - q_offsets[q_batch]
        kv_local = kv_idx - k_offsets[kv_batch]
        inner = mask_mod(q_batch, head, q_local, kv_local)
        return same_sequence & inner

    return wrapped


def _flex_wrap_score_mod(score_mod: Callable, query: NestedTensor, key: NestedTensor) -> Callable:
    r"""Adapt a per-sequence DanLing Flex score modifier to packed global indices."""
    q_offsets = query._ragged_level_offsets(0).to(device=query.device, dtype=torch.long)
    k_offsets = key._ragged_level_offsets(0).to(device=key.device, dtype=torch.long)
    q_ends = q_offsets[1:]
    k_ends = k_offsets[1:]

    def wrapped(score, batch, head, q_idx, kv_idx):
        q_batch = torch.searchsorted(q_ends, q_idx, right=True)
        kv_batch = torch.searchsorted(k_ends, kv_idx, right=True)
        same_sequence = q_batch == kv_batch
        q_local = q_idx - q_offsets[q_batch]
        kv_local = kv_idx - k_offsets[kv_batch]
        modified = score_mod(score, q_batch, head, q_local, kv_local)
        return torch.where(same_sequence, modified, score)

    return wrapped


@functools.lru_cache(maxsize=8)
def _cached_same_sequence_block_mask(
    q_sizes: tuple[int, ...],
    k_sizes: tuple[int, ...],
    num_heads: int | None,
    block_size: int | tuple[int, int],
    device_type: str,
    device_index: int | None,
):
    r"""Cache the default same-sequence Flex block mask for repeated packed layouts."""
    if device_index is None:
        device = torch.device(device_type)
    else:
        device = torch.device(device_type, device_index)
    if isinstance(block_size, int):
        q_block_size = kv_block_size = block_size
    else:
        q_block_size, kv_block_size = block_size

    q_total = int(sum(q_sizes))
    k_total = int(sum(k_sizes))
    q_blocks = (q_total + q_block_size - 1) // q_block_size
    k_blocks = (k_total + kv_block_size - 1) // kv_block_size
    partial_blocks = torch.zeros((q_blocks, k_blocks), dtype=torch.int32)
    full_blocks = torch.zeros((q_blocks, k_blocks), dtype=torch.int32)
    has_full_blocks = False

    q_start = 0
    k_start = 0
    for q_len, k_len in zip(q_sizes, k_sizes):
        q_end = q_start + int(q_len)
        k_end = k_start + int(k_len)

        q_row_start = q_start // q_block_size
        q_row_end = (q_end + q_block_size - 1) // q_block_size
        k_col_start = k_start // kv_block_size
        k_col_end = (k_end + kv_block_size - 1) // kv_block_size
        partial_blocks[q_row_start:q_row_end, k_col_start:k_col_end] = 1

        q_full_start = (q_start + q_block_size - 1) // q_block_size
        q_full_end = q_end // q_block_size
        k_full_start = (k_start + kv_block_size - 1) // kv_block_size
        k_full_end = k_end // kv_block_size
        if q_full_start < q_full_end and k_full_start < k_full_end:
            full_blocks[q_full_start:q_full_end, k_full_start:k_full_end] = 1
            partial_blocks[q_full_start:q_full_end, k_full_start:k_full_end] = 0
            has_full_blocks = True

        q_start = q_end
        k_start = k_end

    def _dense_to_ordered(mask: Tensor) -> tuple[Tensor, Tensor]:
        return mask.sum(dim=-1).to(torch.int32), torch.argsort(mask, dim=-1, descending=True, stable=True).to(
            torch.int32
        )

    kv_num_blocks, kv_indices = _dense_to_ordered(partial_blocks)
    head_count = 1 if num_heads is None else int(num_heads)
    kv_num_blocks = kv_num_blocks.unsqueeze(0).unsqueeze(0).expand(1, head_count, q_blocks).contiguous().to(device)
    kv_indices = kv_indices.unsqueeze(0).unsqueeze(0).expand(1, head_count, q_blocks, k_blocks).contiguous().to(device)

    if has_full_blocks:
        full_kv_num_blocks, full_kv_indices = _dense_to_ordered(full_blocks)
        full_kv_num_blocks = (
            full_kv_num_blocks.unsqueeze(0).unsqueeze(0).expand(1, head_count, q_blocks).contiguous().to(device)
        )
        full_kv_indices = (
            full_kv_indices.unsqueeze(0).unsqueeze(0).expand(1, head_count, q_blocks, k_blocks).contiguous().to(device)
        )
    else:
        full_kv_num_blocks = None
        full_kv_indices = None

    q_sizes_tensor = torch.tensor(q_sizes, device=device, dtype=torch.long)
    q_offsets = torch.empty((q_sizes_tensor.numel() + 1,), device=device, dtype=torch.long)
    q_offsets[0] = 0
    if q_sizes_tensor.numel() > 0:
        torch.cumsum(q_sizes_tensor, dim=0, out=q_offsets[1:])

    k_sizes_tensor = torch.tensor(k_sizes, device=device, dtype=torch.long)
    k_offsets = torch.empty((k_sizes_tensor.numel() + 1,), device=device, dtype=torch.long)
    k_offsets[0] = 0
    if k_sizes_tensor.numel() > 0:
        torch.cumsum(k_sizes_tensor, dim=0, out=k_offsets[1:])

    q_ends = q_offsets[1:]
    k_ends = k_offsets[1:]

    def wrapped(_batch, _head, q_idx, kv_idx):
        q_batch = torch.searchsorted(q_ends, q_idx, right=True)
        kv_batch = torch.searchsorted(k_ends, kv_idx, right=True)
        return q_batch == kv_batch

    return _TorchBlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=(q_block_size, kv_block_size),
        mask_mod=wrapped,
        seq_lengths=(q_total, k_total),
    )


def _packed_sizes_tuple(source: NestedTensor) -> tuple[int, ...]:
    r"""Return per-element packed sizes directly from canonical packed offsets."""
    if source._packed_sizes is not None:
        return source._packed_sizes

    from .aten_functions import _is_fake_tensor

    if _is_fake_tensor(source._offsets):
        raise RuntimeError(
            "Packed sizes are unavailable for FlexAttention block-mask construction under fake tensor mode."
        )
    return tuple(int(size) for size in (source._offsets[1:] - source._offsets[:-1]).tolist())


def create_flex_block_mask(
    mask_mod: Callable,
    query: NestedTensor,
    key: NestedTensor | None = None,
    *,
    num_heads: int | None = None,
    block_size: int | tuple[int, int] = 128,
    compile_mask: bool = False,
):
    r"""Create a FlexAttention block mask directly from DanLing ragged attention storage."""
    if _torch_create_block_mask is None:
        raise RuntimeError("FlexAttention is unavailable in this PyTorch build.")
    if key is None:
        key = query
    if len(query) != len(key):
        raise ValueError(
            "NestedTensor batch length mismatch between query and key: " f"query={len(query)}, key={len(key)}"
        )
    q_sizes = _packed_sizes_tuple(query)
    k_sizes = _packed_sizes_tuple(key)
    compile_requested = compile_mask or torch.compiler.is_compiling()
    if mask_mod is _flex_allow_all and not compile_requested:
        return _cached_same_sequence_block_mask(
            q_sizes,
            k_sizes,
            num_heads,
            block_size,
            query.device.type,
            query.device.index,
        )
    wrapped_mask_mod = _flex_wrap_mask_mod(mask_mod, query, key)
    return _torch_create_block_mask(
        wrapped_mask_mod,
        1,
        num_heads,
        int(sum(q_sizes)),
        int(sum(k_sizes)),
        device=query.device,
        BLOCK_SIZE=block_size,
        _compile=compile_requested,
    )


def _pad_last_dim_for_flash(tensor: Tensor, alignment_size: int = 8) -> Tensor:
    r"""Pad the last dim for Flash Attention alignment requirements."""
    last_dim = tensor.size(-1)
    if last_dim % alignment_size == 0:
        return tensor
    return F.pad(tensor, (0, alignment_size - (last_dim % alignment_size)))


def _sdpa_via_native_flash(
    query: NestedTensor,
    key: NestedTensor,
    value: NestedTensor,
    *,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
) -> NestedTensor:
    r"""Run SDPA directly on DanLing storage via varlen Flash Attention kernels."""
    q_values, _q_lengths_cpu, q_cumulative, q_max = _sdpa_pack_native(query)
    k_values, _k_lengths_cpu, k_cumulative, k_max = _sdpa_pack_native(key)
    v_values, _v_lengths_cpu, _v_cumulative, _v_max = _sdpa_pack_native(value)
    attention = _flash_attention_forward_values(
        q_values,
        k_values,
        v_values,
        q_cumulative,
        k_cumulative,
        q_max,
        k_max,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    return _sdpa_restore_native(attention, query)


@NestedTensorFuncRegistry.implement(F.multi_head_attention_forward)
def multi_head_attention_forward(
    query: NestedTensor,
    key: NestedTensor | Tensor,
    value: NestedTensor | Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor | None,
    bias_k: Tensor | None,
    bias_v: Tensor | None,
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Tensor | None = None,
    need_weights: bool = True,
    attn_mask: Tensor | None = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Tensor | None = None,
    k_proj_weight: Tensor | None = None,
    v_proj_weight: Tensor | None = None,
    static_k: Tensor | None = None,
    static_v: Tensor | None = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
):
    r"""
    Forward method for MultiHeadAttention.
    See also [torch.nn.functional.multi_head_attention_forward][].

    Args:
        query: The query NestedTensor.
        key: The key NestedTensor or Tensor.
        value: The value NestedTensor or Tensor.
        embed_dim_to_check: Total dimension of the model.
        num_heads: Number of parallel attention heads.
        in_proj_weight: Input projection weight.
        in_proj_bias: Input projection bias.
        bias_k: Bias of the key projection.
        bias_v: Bias of the value projection.
        add_zero_attn: Whether to add a batch of zeros to key and value.
        dropout_p: Dropout probability.
        out_proj_weight: Output projection weight.
        out_proj_bias: Output projection bias.
        training: Whether in training mode.
        key_padding_mask: Mask for padded key positions.
        need_weights: Whether to return attention weights.
        attn_mask: Attention mask tensor.
        use_separate_proj_weight: Whether to use separate projection weights for q, k, v.
        q_proj_weight: Query projection weight when using separate weights.
        k_proj_weight: Key projection weight when using separate weights.
        v_proj_weight: Value projection weight when using separate weights.
        static_k: Static key tensor.
        static_v: Static value tensor.
        average_attn_weights: Whether to average attention weights over heads.
        is_causal: Whether to apply causal masking.

    Returns:
        tuple[NestedTensor, Tensor]: The attention output and attention weights.

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> q1 = torch.randn(2, 4)
        >>> q2 = torch.randn(3, 4)
        >>> query = NestedTensor(q1, q2)
        >>> embed_dim = 4
        >>> num_heads = 2
        >>> in_w = torch.randn(3 * embed_dim, embed_dim)
        >>> in_b = torch.randn(3 * embed_dim)
        >>> out_w = torch.randn(embed_dim, embed_dim)
        >>> out_b = torch.randn(embed_dim)
        >>> out, attn = F.multi_head_attention_forward(
        ...     query, query, query, embed_dim, num_heads, in_w, in_b, None, None, False, 0.0, out_w, out_b,
        ...     training=False, need_weights=True,
        ... )
        >>> q_t = query.tensor.transpose(0, 1)
        >>> key_padding_mask = ~query.mask
        >>> ref_out, ref_attn = F.multi_head_attention_forward(
        ...     q_t, q_t, q_t, embed_dim, num_heads, in_w, in_b, None, None, False, 0.0, out_w, out_b,
        ...     training=False, need_weights=True, key_padding_mask=key_padding_mask,
        ... )
        >>> ref_out = ref_out.transpose(0, 1)
        >>> torch.allclose(out, ref_out)
        True
    """
    from .nested_tensor import NestedTensor

    cls = type(query)
    if not isinstance(query, NestedTensor):
        raise TypeError("query must be a NestedTensor")
    if not isinstance(key, NestedTensor) and key.shape == query.shape:
        key = query.nested_like(key, strict=False)
    if not isinstance(value, NestedTensor) and value.shape == query.shape:
        value = query.nested_like(value, strict=False)
    if isinstance(key, NestedTensor):
        if len(query) != len(key):
            raise ValueError(
                "NestedTensor batch length mismatch between query and key: " f"query={len(query)}, key={len(key)}"
            )
        if query.batch_first != key.batch_first:
            raise ValueError(
                "NestedTensor batch_first mismatch between query and key: "
                f"query.batch_first={query.batch_first}, key.batch_first={key.batch_first}. "
                "Use the same batch_first setting for query, key, and value."
            )
    if isinstance(value, NestedTensor):
        if len(query) != len(value):
            raise ValueError(
                "NestedTensor batch length mismatch between query and value: " f"query={len(query)}, value={len(value)}"
            )
        if query.batch_first != value.batch_first:
            raise ValueError(
                "NestedTensor batch_first mismatch between query and value: "
                f"query.batch_first={query.batch_first}, value.batch_first={value.batch_first}. "
                "Use the same batch_first setting for query, key, and value."
            )
    if len(query) == 0:
        empty = cls([], **query._meta(include_dtype=True))
        return empty, (torch.empty(0, dtype=query.dtype, device=query.device) if need_weights else None)

    if (
        not need_weights
        and key_padding_mask is None
        and attn_mask is None
        and bias_k is None
        and bias_v is None
        and not add_zero_attn
        and static_k is None
        and static_v is None
        and query._physical_shape.size(1) == 2
        and key._physical_shape.size(1) == 2
        and value._physical_shape.size(1) == 2
        and query._values.dim() == 2
        and key._values.dim() == 2
        and value._values.dim() == 2
    ):
        head_dim = embed_dim_to_check // num_heads
        if head_dim * num_heads != embed_dim_to_check:
            raise ValueError(
                f"embed_dim {embed_dim_to_check} must be divisible by num_heads {num_heads} for MultiheadAttention"
            )
        effective_dropout_p = dropout_p if training else 0.0

        if use_separate_proj_weight:
            if q_proj_weight is None or k_proj_weight is None or v_proj_weight is None:
                raise ValueError(
                    "use_separate_proj_weight=True requires q_proj_weight, k_proj_weight, and v_proj_weight"
                )
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
            q_values, k_values, v_values = F._in_projection(
                query._values,
                key._values,
                value._values,
                q_proj_weight,
                k_proj_weight,
                v_proj_weight,
                b_q,
                b_k,
                b_v,
            )
        else:
            q_values, k_values, v_values = F._in_projection_packed(
                query._values,
                key._values,
                value._values,
                in_proj_weight,
                in_proj_bias,
            )

        q_heads = q_values.unflatten(-1, (num_heads, head_dim)).contiguous()
        k_heads = k_values.unflatten(-1, (num_heads, head_dim)).contiguous()
        v_heads = v_values.unflatten(-1, (num_heads, head_dim)).contiguous()

        from .aten_functions import _packed_like

        q_attn = _project_attention_values(query, q_heads, num_heads, head_dim)
        k_attn = _project_attention_values(key, k_heads, num_heads, head_dim)
        v_attn = _project_attention_values(value, v_heads, num_heads, head_dim)

        attn_output = F.scaled_dot_product_attention(
            q_attn,
            k_attn,
            v_attn,
            attn_mask=None,
            dropout_p=effective_dropout_p,
            is_causal=is_causal,
        )
        out_values = F.linear(
            attn_output._values.reshape(attn_output._values.size(0), -1), out_proj_weight, out_proj_bias
        )
        return (
            _packed_like(query, out_values),
            None,
        )

    # Materialize padded tensors
    q_padded = _nested_to_padded_tensor(query)
    k_padded = _nested_to_padded_tensor(key) if isinstance(key, NestedTensor) else key
    v_padded = _nested_to_padded_tensor(value) if isinstance(value, NestedTensor) else value

    # Transpose to (T, B, E) — F.multi_head_attention_forward always expects this layout
    if query.batch_first:
        q_padded = q_padded.transpose(0, 1)
        if isinstance(k_padded, Tensor) and k_padded.dim() == 3:
            k_padded = k_padded.transpose(0, 1)
        if isinstance(v_padded, Tensor) and v_padded.dim() == 3:
            v_padded = v_padded.transpose(0, 1)

    # Build key_padding_mask from NestedTensor structure if not provided
    # PyTorch convention: True=ignore (padding)
    if key_padding_mask is None and isinstance(key, NestedTensor):
        key_padding_mask = ~_nested_sequence_valid_mask(key)

    # Single batched MHA call (replaces per-sample Python loop)
    output, weights = F.multi_head_attention_forward(  # type: ignore[call-arg]
        q_padded,
        k_padded,
        v_padded,
        embed_dim_to_check,
        num_heads,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        add_zero_attn,
        dropout_p,
        out_proj_weight,
        out_proj_bias,
        training=training,
        key_padding_mask=key_padding_mask,
        need_weights=need_weights,
        attn_mask=attn_mask,
        use_separate_proj_weight=use_separate_proj_weight,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        static_k=static_k,
        static_v=static_v,
        average_attn_weights=average_attn_weights,
        is_causal=is_causal,
    )

    # Transpose back to match query's batch_first convention
    if query.batch_first:
        output = output.transpose(0, 1)  # (T, B, E) → (B, T, E)

    # Reconstruct NestedTensor (strips padding via query's structure)
    nt_output = _nested_from_padded_tensor(query, output)

    return nt_output, weights


@NestedTensorFuncRegistry.implement(F.scaled_dot_product_attention)
def scaled_dot_product_attention(
    query: NestedTensor,
    key: NestedTensor | Tensor,
    value: NestedTensor | Tensor,
    attn_mask: NestedTensor | Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> NestedTensor:
    r"""
    Compute scaled dot-product attention for variable-length batches.
    See also [torch.nn.functional.scaled_dot_product_attention][].

    Args:
        query: The query NestedTensor.
        key: The key NestedTensor or Tensor.
        value: The value NestedTensor or Tensor.
        attn_mask: Optional attention mask.
        dropout_p: Dropout probability.
        is_causal: Whether to apply causal masking.
        scale: Optional scaling factor for attention scores.
        enable_gqa: Whether to enable grouped query attention.

    Returns:
        NestedTensor: The attention output.

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> q = NestedTensor(torch.randn(2, 3, 4), torch.randn(2, 2, 4))
        >>> out = F.scaled_dot_product_attention(q, q, q, dropout_p=0.0)
        >>> ref = NestedTensor(
        ...     F.scaled_dot_product_attention(q[0], q[0], q[0], dropout_p=0.0),
        ...     F.scaled_dot_product_attention(q[1], q[1], q[1], dropout_p=0.0),
        ... )
        >>> torch.allclose(out, ref, atol=1e-6, rtol=1e-6)
        True
    """
    from .nested_tensor import NestedTensor

    if not isinstance(query, NestedTensor):
        raise TypeError("query must be a NestedTensor")
    if not isinstance(key, NestedTensor) and key.shape == query.shape:
        key = query.nested_like(key, strict=False)
    if not isinstance(value, NestedTensor) and value.shape == query.shape:
        value = query.nested_like(value, strict=False)

    if isinstance(key, NestedTensor) and len(query) != len(key):
        raise ValueError(
            "NestedTensor batch length mismatch between query and key: " f"query={len(query)}, key={len(key)}"
        )
    if isinstance(key, NestedTensor) and query.batch_first != key.batch_first:
        raise ValueError(
            "NestedTensor batch_first mismatch between query and key: "
            f"query.batch_first={query.batch_first}, key.batch_first={key.batch_first}. "
            "Use the same batch_first setting for query, key, and value."
        )
    if isinstance(value, NestedTensor) and len(query) != len(value):
        raise ValueError(
            "NestedTensor batch length mismatch between query and value: " f"query={len(query)}, value={len(value)}"
        )
    if isinstance(value, NestedTensor) and query.batch_first != value.batch_first:
        raise ValueError(
            "NestedTensor batch_first mismatch between query and value: "
            f"query.batch_first={query.batch_first}, value.batch_first={value.batch_first}. "
            "Use the same batch_first setting for query, key, and value."
        )
    if isinstance(attn_mask, NestedTensor) and len(query) != len(attn_mask):
        raise ValueError(
            "NestedTensor batch length mismatch between query and attn_mask: "
            f"query={len(query)}, attn_mask={len(attn_mask)}"
        )
    if isinstance(attn_mask, NestedTensor) and query.batch_first != attn_mask.batch_first:
        raise ValueError(
            "NestedTensor batch_first mismatch between query and attn_mask: "
            f"query.batch_first={query.batch_first}, attn_mask.batch_first={attn_mask.batch_first}. "
            "Use the same batch_first setting for query, key, value, and attn_mask."
        )
    if len(query) == 0:
        return NestedTensor([], **query._meta(include_dtype=True))

    native_attention_inputs = (
        isinstance(key, NestedTensor)
        and isinstance(value, NestedTensor)
        and _is_native_attention_layout(query)
        and _is_native_attention_layout(key)
        and _is_native_attention_layout(value)
    )
    if native_attention_inputs and attn_mask is None and not enable_gqa and query._values.is_cuda:
        if query.dtype in _LOW_PRECISION_CUDA_DTYPES and torch.backends.cuda.flash_sdp_enabled():
            return _sdpa_via_native_flash(
                query,
                key,
                value,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
        if dropout_p == 0.0 and _torch_flex_attention is not None:
            return flex_attention(query, key, value, scale=scale, enable_gqa=enable_gqa)

    # Fallback: pad-based SDPA
    q_padded = _nested_to_padded_tensor(query)
    k_padded = _nested_to_padded_tensor(key) if isinstance(key, NestedTensor) else key
    v_padded = _nested_to_padded_tensor(value) if isinstance(value, NestedTensor) else value
    user_attn_mask: NestedTensor | Tensor | None = (
        _nested_to_padded_tensor(attn_mask) if isinstance(attn_mask, NestedTensor) else attn_mask
    )

    # Normalize to batch-leading layout for SDPA fallback and mask broadcast.
    if not query.batch_first:
        if q_padded.dim() >= 2:
            q_padded = q_padded.transpose(0, 1)
        if isinstance(k_padded, Tensor) and k_padded.dim() >= 2:
            k_padded = k_padded.transpose(0, 1)
        if isinstance(v_padded, Tensor) and v_padded.dim() >= 2:
            v_padded = v_padded.transpose(0, 1)
        if isinstance(user_attn_mask, Tensor) and user_attn_mask.dim() >= 3:
            user_attn_mask = user_attn_mask.transpose(0, 1)

    combined_mask, is_causal = _build_sdpa_mask(key, user_attn_mask, is_causal, q_padded, k_padded)
    if (
        combined_mask is not None
        and isinstance(combined_mask, Tensor)
        and combined_mask.dtype == torch.bool
        and q_padded.is_cuda
        and q_padded.dtype in _LOW_PRECISION_CUDA_DTYPES
    ):
        combined_mask = torch.where(
            combined_mask,
            torch.zeros((), device=combined_mask.device, dtype=q_padded.dtype),
            torch.full((), float("-inf"), device=combined_mask.device, dtype=q_padded.dtype),
        )

    out_padded = F.scaled_dot_product_attention(
        q_padded,
        k_padded,
        v_padded,
        attn_mask=combined_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    if not query.batch_first and out_padded.dim() >= 2:
        out_padded = out_padded.transpose(0, 1)

    return _nested_from_padded_tensor(query, out_padded)


if _torch_flex_attention is not None:

    @NestedTensorFuncRegistry.implement(_torch_flex_attention)
    def flex_attention(
        query: NestedTensor,
        key: NestedTensor | Tensor,
        value: NestedTensor | Tensor,
        score_mod: Callable | None = None,
        block_mask=None,
        scale: float | None = None,
        enable_gqa: bool = False,
        return_lse: bool = False,
        kernel_options: dict | None = None,
        *,
        return_aux=None,
    ):
        r"""Run FlexAttention on DanLing ragged attention tensors via zero-copy jagged views."""
        from .nested_tensor import NestedTensor

        if not isinstance(query, NestedTensor):
            raise TypeError("query must be a NestedTensor")
        if not isinstance(key, NestedTensor) or not isinstance(value, NestedTensor):
            raise ValueError(
                "FlexAttention does not support mixed NestedTensor / non-NestedTensor inputs for DanLing. "
                "Pass NestedTensor query, key, and value together."
            )
        if len(query) != len(key) or len(query) != len(value):
            raise ValueError(
                "NestedTensor batch length mismatch between attention inputs: "
                f"query={len(query)}, key={len(key)}, value={len(value)}"
            )
        if query.batch_first != key.batch_first or query.batch_first != value.batch_first:
            raise ValueError(
                "NestedTensor batch_first mismatch between attention inputs. "
                "Use the same batch_first setting for query, key, and value."
            )
        if len(query) == 0:
            empty = NestedTensor([], **query._meta(include_dtype=True))
            if return_lse:
                return empty, NestedTensor([], **query._meta(include_dtype=True))
            if return_aux is not None:
                return empty, _FlexAuxOutput(lse=None, max_scores=None)
            return empty

        q_view = _as_flex_packed_dense(query)
        k_view = _as_flex_packed_dense(key)
        v_view = _as_flex_packed_dense(value)
        # Nested FlexAttention does not implicitly isolate batch items inside the
        # packed ragged sequence. Build the same-sequence block mask by default
        # so DanLing semantics match per-element dense attention.
        if block_mask is None:
            if torch.compiler.is_compiling():
                from torch._subclasses.fake_tensor import unset_fake_temporarily

                with unset_fake_temporarily():
                    block_mask = create_flex_block_mask(_flex_allow_all, query, key, compile_mask=True)
            else:
                block_mask = create_flex_block_mask(_flex_allow_all, query, key)
        wrapped_score_mod = _flex_wrap_score_mod(score_mod, query, key) if score_mod is not None else None

        if torch.compiler.is_compiling():
            from torch._subclasses.fake_tensor import fake_tensor_tls

            old_allow_non_fake_inputs = fake_tensor_tls.allow_non_fake_inputs_override
            fake_tensor_tls.allow_non_fake_inputs_override = True
            try:
                output = _torch_flex_attention(
                    q_view,
                    k_view,
                    v_view,
                    score_mod=wrapped_score_mod,
                    block_mask=block_mask,
                    scale=scale,
                    enable_gqa=enable_gqa,
                    return_lse=return_lse,
                    kernel_options=kernel_options,
                    return_aux=return_aux,
                )
            finally:
                fake_tensor_tls.allow_non_fake_inputs_override = old_allow_non_fake_inputs
        else:
            output = _torch_flex_attention(
                q_view,
                k_view,
                v_view,
                score_mod=wrapped_score_mod,
                block_mask=block_mask,
                scale=scale,
                enable_gqa=enable_gqa,
                return_lse=return_lse,
                kernel_options=kernel_options,
                return_aux=return_aux,
            )
        return _restore_flex_output(output, query)

    _danling_flex_attention = flex_attention

    if _torch_flex_attention_module is not None:

        def _public_flex_attention(
            query,
            key,
            value,
            score_mod=None,
            block_mask=None,
            scale=None,
            enable_gqa: bool = False,
            return_lse: bool = False,
            kernel_options=None,
            *,
            return_aux=None,
        ):
            from .nested_tensor import NestedTensor

            if isinstance(query, NestedTensor) or isinstance(key, NestedTensor) or isinstance(value, NestedTensor):
                return _danling_flex_attention(
                    query,
                    key,
                    value,
                    score_mod=score_mod,
                    block_mask=block_mask,
                    scale=scale,
                    enable_gqa=enable_gqa,
                    return_lse=return_lse,
                    kernel_options=kernel_options,
                    return_aux=return_aux,
                )
            return _torch_flex_attention(
                query,
                key,
                value,
                score_mod=score_mod,
                block_mask=block_mask,
                scale=scale,
                enable_gqa=enable_gqa,
                return_lse=return_lse,
                kernel_options=kernel_options,
                return_aux=return_aux,
            )

        _public_flex_attention.__name__ = _torch_flex_attention.__name__
        _public_flex_attention.__qualname__ = _torch_flex_attention.__qualname__
        _public_flex_attention.__doc__ = _torch_flex_attention.__doc__
        _torch_flex_attention_module.flex_attention = _public_flex_attention


# Criterions


@NestedTensorFuncRegistry.implement(F.ctc_loss)
def ctc_loss(
    input: NestedTensor,
    target: NestedTensor | Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    blank: int = 0,
    reduction: str = "mean",
    zero_infinity: bool = False,
) -> Tensor:
    r"""
    Compute the Connectionist Temporal Classification loss.
    See also [torch.nn.functional.ctc_loss][].

    Args:
        input: The input NestedTensor of log probabilities.
        target: The target tensor or NestedTensor.
        input_lengths: Lengths of the inputs.
        target_lengths: Lengths of the targets.
        blank: Blank label index.
        reduction: Specifies the reduction: 'none', 'mean', or 'sum'.
        zero_infinity: If True, zero infinite losses and associated gradients.

    Returns:
        Tensor: The computed loss.

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> input = NestedTensor(torch.tensor([[-1.0, -2.0]]), torch.tensor([[-2.0, -1.0]]))
        >>> target = NestedTensor(torch.tensor([0]), torch.tensor([1]))
        >>> input_lengths = torch.tensor([1, 1])
        >>> target_lengths = torch.tensor([1, 1])
        >>> out = F.ctc_loss(input, target, input_lengths, target_lengths)
        >>> logits = input.tensor.transpose(0, 1)
        >>> ref = F.ctc_loss(logits, target.concat, input_lengths, target_lengths)
        >>> torch.allclose(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if isinstance(input, NestedTensor):
        logits = input.tensor
        if input.batch_first:
            logits = logits.transpose(0, 1)
    else:
        logits = input
    targets = target.concat if isinstance(target, NestedTensor) else target
    return F.ctc_loss(
        logits,
        targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )


# Table-driven registrations — loss functions
# All loss functions concat NestedTensor positional args, then call the original function.

_LOSS_OPS_2 = [
    F.binary_cross_entropy,
    F.binary_cross_entropy_with_logits,
    F.cross_entropy,
    F.gaussian_nll_loss,
    F.hinge_embedding_loss,
    F.huber_loss,
    F.kl_div,
    F.l1_loss,
    F.mse_loss,
    F.multi_margin_loss,
    F.multilabel_margin_loss,
    F.multilabel_soft_margin_loss,
    F.nll_loss,
    F.poisson_nll_loss,
    F.smooth_l1_loss,
    F.soft_margin_loss,
]
_LOSS_OPS_3 = [
    F.cosine_embedding_loss,
    F.margin_ranking_loss,
    F.triplet_margin_loss,
    F.triplet_margin_with_distance_loss,
]

for _op, _n in [*((op, 2) for op in _LOSS_OPS_2), *((op, 3) for op in _LOSS_OPS_3)]:

    @NestedTensorFuncRegistry.implement(_op)
    def _loss_impl(*args, _fn=_op, _n=_n, **kwargs):
        tensor_args = _concat_tensors(*args[:_n])
        return _fn(*tensor_args, *args[_n:], **kwargs)


# Linear & Embeddings


@NestedTensorFuncRegistry.implement(F.embedding)
def embedding(
    input: NestedTensor,
    weight: Tensor,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> NestedTensor:
    r"""
    Generate a simple lookup table that looks up embeddings in a fixed dictionary and size.
    See also [torch.nn.functional.embedding][].

    Args:
        input: The input NestedTensor of indices.
        weight: The embedding matrix with number of rows equal to the maximum index + 1.
        padding_idx: If specified, entries at this index do not contribute to the gradient.
        max_norm: If given, renormalize embeddings to have norm at most this value.
        norm_type: The p of the p-norm to compute for max_norm.
        scale_grad_by_freq: If True, scale gradients by inverse frequency of the words.
        sparse: If True, gradient w.r.t. weight will be a sparse tensor.

    Returns:
        NestedTensor: The embedded output.

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> weight = torch.arange(6.0).reshape(3, 2)
        >>> nt = NestedTensor(torch.tensor([0, 1]), torch.tensor([1, 2, 0]))
        >>> out = F.embedding(nt, weight)
        >>> ref = F.embedding(nt.tensor, weight)
        >>> torch.allclose(out, ref)
        True
    """
    return _concat_apply(
        input,
        lambda t: F.embedding(
            t,
            weight,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
        ),
        lambda shape: torch.Size([*shape, weight.shape[1]]),
    )


@NestedTensorFuncRegistry.implement(F.embedding_bag)
def embedding_bag(
    input: NestedTensor,
    weight: Tensor,
    offsets: Tensor | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    mode: str = "mean",
    sparse: bool = False,
    per_sample_weights: Tensor | None = None,
    include_last_offset: bool = False,
    padding_idx: int | None = None,
) -> NestedTensor:
    r"""
    Compute sums, means or maxes of `bags` of embeddings.
    See also [torch.nn.functional.embedding_bag][].

    Args:
        input: The input NestedTensor of indices.
        weight: The embedding matrix.
        offsets: Offsets into input for each bag. Required for 1D input.
        max_norm: If given, renormalize embeddings to have norm at most this value.
        norm_type: The p of the p-norm to compute for max_norm.
        scale_grad_by_freq: If True, scale gradients by inverse frequency of the words.
        mode: Aggregation mode: 'mean', 'sum', or 'max'.
        sparse: If True, gradient w.r.t. weight will be a sparse tensor.
        per_sample_weights: Per-sample weights for weighted sum mode.
        include_last_offset: If True, treat the last offset as the size of input.
        padding_idx: If specified, entries at this index do not contribute to the gradient.

    Returns:
        NestedTensor: The embedded bag output.

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> weight = torch.arange(6.0).reshape(3, 2)
        >>> offsets = torch.tensor([0])
        >>> a = torch.tensor([0, 1])
        >>> b = torch.tensor([1, 2, 0])
        >>> nt = NestedTensor(a, b)
        >>> out = F.embedding_bag(nt, weight, offsets=offsets)
        >>> ref = NestedTensor(
        ...     F.embedding_bag(a, weight, offsets=offsets),
        ...     F.embedding_bag(b, weight, offsets=offsets),
        ... )
        >>> torch.allclose(out, ref)
        True
    """
    return _map_storage_serial(
        input,
        lambda t: F.embedding_bag(
            t,
            weight,
            offsets=offsets if offsets is not None else torch.tensor([0], device=t.device, dtype=torch.long),
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            mode=mode,
            sparse=sparse,
            per_sample_weights=per_sample_weights,
            include_last_offset=include_last_offset,
            padding_idx=padding_idx,
        ),
    )


if hasattr(F, "grouped_mm"):

    @NestedTensorFuncRegistry.implement(F.grouped_mm)
    def grouped_mm(
        mat_a: NestedTensor,
        mat_b: Tensor | NestedTensor,
        *,
        offs: Tensor | None = None,
        bias: Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> NestedTensor:
        r"""
        Applies grouped matrix multiplication.
        See also [torch.nn.functional.grouped_mm][].

        Args:
            mat_a: The first input NestedTensor.
            mat_b: The second input Tensor or NestedTensor.
            offs: Optional offsets tensor for grouping.
            bias: Optional bias tensor.
            out_dtype: Optional output dtype.

        Returns:
            NestedTensor: The result of grouped matrix multiplication.

        Note:
            No inline doctest is provided here because ``torch.nn.functional.grouped_mm``
            is backend-sensitive and may require kernel-specific layout constraints that
            are not portable across test environments.
        """
        from .nested_tensor import NestedTensor

        cls = type(mat_a)
        if isinstance(mat_b, NestedTensor):
            if len(mat_a) != len(mat_b):
                raise ValueError(
                    "NestedTensor batch length mismatch between mat_a and mat_b: "
                    f"mat_a={len(mat_a)}, mat_b={len(mat_b)}"
                )
            if len(mat_a) == 0:
                return cls([], **mat_a._meta(include_dtype=True))
            outputs = [
                F.grouped_mm(a, b, offs=offs, bias=bias, out_dtype=out_dtype)
                for a, b in zip(mat_a._storage, mat_b._storage)
            ]
        else:
            if len(mat_a) == 0:
                return cls([], **mat_a._meta(include_dtype=True))
            outputs = [F.grouped_mm(a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype) for a in mat_a._storage]
        return cls(outputs, **mat_a._meta())


@NestedTensorFuncRegistry.implement(F.linear)
def linear(input: NestedTensor, weight: Tensor, bias: Tensor | None = None) -> NestedTensor:
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    See also [torch.nn.functional.linear][].

    Args:
        input: The input NestedTensor.
        weight: The weight matrix.
        bias: Optional bias vector.

    Returns:
        NestedTensor: The linearly transformed output.

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        >>> bias = torch.tensor([0.5, -0.5])
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        >>> out = F.linear(nt, weight, bias)
        >>> ref = F.linear(nt.tensor, weight, bias)
        >>> torch.allclose(out, ref)
        True
    """
    cls = type(input)
    if len(input) == 0:
        return cls([], **input._meta(include_dtype=True))
    if input._values.dim() >= 2:
        from .aten_functions import _packed_new_last_dim

        new_values = F.linear(input._values, weight, bias)
        return _packed_new_last_dim(input, new_values, int(weight.shape[0]))
    return _apply_per_element(input, F.linear, weight, bias)


def _scaled_mm(mat_a: NestedTensor, mat_b, fn, **kwargs) -> NestedTensor:
    r"""Shared implementation for scaled_mm and scaled_grouped_mm."""
    from .nested_tensor import NestedTensor

    cls = type(mat_a)
    if len(mat_a) == 0:
        return cls([], **mat_a._meta(include_dtype=True))
    if isinstance(mat_b, NestedTensor):
        if len(mat_a) != len(mat_b):
            raise ValueError(
                "NestedTensor batch length mismatch between mat_a and mat_b: " f"mat_a={len(mat_a)}, mat_b={len(mat_b)}"
            )
        outputs = [fn(a, b, **kwargs) for a, b in zip(mat_a._storage, mat_b._storage)]
    else:
        outputs = [fn(a, mat_b, **kwargs) for a in mat_a._storage]
    return cls(outputs, **mat_a._meta())


if hasattr(F, "scaled_grouped_mm"):

    @NestedTensorFuncRegistry.implement(F.scaled_grouped_mm)
    def scaled_grouped_mm(
        mat_a: NestedTensor,
        mat_b: Tensor | NestedTensor,
        scale_a,
        scale_recipe_a,
        scale_b,
        scale_recipe_b,
        swizzle_a=None,
        swizzle_b=None,
        bias: Tensor | None = None,
        offs: Tensor | None = None,
        output_dtype: torch.dtype | None = torch.bfloat16,
        contraction_dim=(),
        use_fast_accum: bool = False,
    ) -> NestedTensor:
        r"""
        Applies scaled grouped matrix multiplication.
        See also [torch.nn.functional.scaled_grouped_mm][].

        Args:
            mat_a: The first input NestedTensor.
            mat_b: The second input Tensor or NestedTensor.
            scale_a: Scale factor for mat_a.
            scale_recipe_a: Scale recipe for mat_a.
            scale_b: Scale factor for mat_b.
            scale_recipe_b: Scale recipe for mat_b.
            swizzle_a: Optional swizzle pattern for mat_a.
            swizzle_b: Optional swizzle pattern for mat_b.
            bias: Optional bias tensor.
            offs: Optional offsets tensor for grouping.
            output_dtype: Optional output dtype.
            contraction_dim: Contraction dimensions.
            use_fast_accum: Whether to use fast accumulation.

        Returns:
            NestedTensor: The result of scaled grouped matrix multiplication.

        Note:
            No inline doctest is provided here because
            ``torch.nn.functional.scaled_grouped_mm`` has evolving upstream argument
            requirements and backend-specific execution constraints.
        """
        return _scaled_mm(
            mat_a,
            mat_b,
            F.scaled_grouped_mm,
            scale_a=scale_a,
            scale_recipe_a=scale_recipe_a,
            scale_b=scale_b,
            scale_recipe_b=scale_recipe_b,
            swizzle_a=swizzle_a,
            swizzle_b=swizzle_b,
            bias=bias,
            offs=offs,
            output_dtype=output_dtype,
            contraction_dim=contraction_dim,
            use_fast_accum=use_fast_accum,
        )


if hasattr(F, "scaled_mm"):

    @NestedTensorFuncRegistry.implement(F.scaled_mm)
    def scaled_mm(
        mat_a: NestedTensor,
        mat_b: Tensor | NestedTensor,
        scale_a,
        scale_recipe_a,
        scale_b,
        scale_recipe_b,
        swizzle_a=None,
        swizzle_b=None,
        bias: Tensor | None = None,
        output_dtype: torch.dtype | None = torch.bfloat16,
        contraction_dim=(),
        use_fast_accum: bool = False,
    ) -> NestedTensor:
        r"""
        Applies scaled matrix multiplication.
        See also [torch.nn.functional.scaled_mm][].

        Args:
            mat_a: The first input NestedTensor.
            mat_b: The second input Tensor or NestedTensor.
            scale_a: Scale factor for mat_a.
            scale_recipe_a: Scale recipe for mat_a.
            scale_b: Scale factor for mat_b.
            scale_recipe_b: Scale recipe for mat_b.
            swizzle_a: Optional swizzle pattern for mat_a.
            swizzle_b: Optional swizzle pattern for mat_b.
            bias: Optional bias tensor.
            output_dtype: Optional output dtype.
            contraction_dim: Contraction dimensions.
            use_fast_accum: Whether to use fast accumulation.

        Returns:
            NestedTensor: The result of scaled matrix multiplication.

        Note:
            No inline doctest is provided here because ``torch.nn.functional.scaled_mm``
            has evolving upstream argument requirements and backend-specific execution
            constraints.
        """
        return _scaled_mm(
            mat_a,
            mat_b,
            F.scaled_mm,
            scale_a=scale_a,
            scale_recipe_a=scale_recipe_a,
            scale_b=scale_b,
            scale_recipe_b=scale_recipe_b,
            swizzle_a=swizzle_a,
            swizzle_b=swizzle_b,
            bias=bias,
            output_dtype=output_dtype,
            contraction_dim=contraction_dim,
            use_fast_accum=use_fast_accum,
        )


# Normalization


@NestedTensorFuncRegistry.implement(F.batch_norm)
def batch_norm(
    input: NestedTensor,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> NestedTensor:
    r"""
    Applies Batch Normalization for each channel across a batch of data.
    See also [torch.nn.functional.batch_norm][].

    Args:
        input: The input NestedTensor.
        running_mean: The running mean tensor.
        running_var: The running variance tensor.
        weight: Learnable per-channel affine weight (gamma).
        bias: Learnable per-channel affine bias (beta).
        training: Whether in training mode.
        momentum: Value used for running mean and variance computation.
        eps: Value added to the denominator for numerical stability.

    Returns:
        NestedTensor: The normalized output.

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.arange(4.0).view(1, 1, 4),
        ...     torch.arange(6.0).view(1, 1, 6),
        ... )
        >>> running_mean = torch.zeros(1)
        >>> running_var = torch.ones(1)
        >>> out = F.batch_norm(nt, running_mean, running_var, training=False)
        >>> ref = F.batch_norm(nt.tensor, running_mean, running_var, training=False)
        >>> torch.allclose(out, ref)
        True
    """
    if not training:
        # Eval mode is a per-channel affine transform — padding doesn't affect valid positions
        output = F.batch_norm(input.tensor, running_mean, running_var, weight, bias, False, momentum, eps)
        return input.nested_like(output, strict=False)

    # Training mode: concatenate to compute batch statistics without padding contamination
    from .nested_tensor import NestedTensor

    concat, shapes = input.concatenate()
    normalized = F.batch_norm(concat, running_mean, running_var, weight, bias, training, momentum, eps)
    return NestedTensor.from_concatenated(normalized, shapes, **input._meta())


@NestedTensorFuncRegistry.implement(F.group_norm)
def group_norm(
    input: NestedTensor,
    num_groups: int,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> NestedTensor:
    r"""
    Applies Group Normalization for last certain number of dimensions.
    See also [torch.nn.functional.group_norm][].

    Args:
        input: The input NestedTensor.
        num_groups: Number of groups to separate the channels into.
        weight: Learnable per-channel affine weight (gamma).
        bias: Learnable per-channel affine bias (beta).
        eps: Value added to the denominator for numerical stability.

    Returns:
        NestedTensor: The normalized output.

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(8.0).view(4, 2), torch.arange(12.0).view(6, 2))
        >>> out = F.group_norm(nt, num_groups=2)
        >>> ref = NestedTensor(
        ...     F.group_norm(nt[0].unsqueeze(0), num_groups=2).squeeze(0),
        ...     F.group_norm(nt[1].unsqueeze(0), num_groups=2).squeeze(0),
        ... )
        >>> torch.allclose(out, ref)
        True
    """
    return _map_storage_serial(input, lambda t: F.group_norm(t.unsqueeze(0), num_groups, weight, bias, eps).squeeze(0))


@NestedTensorFuncRegistry.implement(F.instance_norm)
def instance_norm(
    input: NestedTensor,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> NestedTensor:
    r"""
    Applies Instance Normalization independently for each channel in every data sample within a batch.
    See also [torch.nn.functional.instance_norm][].

    Args:
        input: The input NestedTensor.
        running_mean: The running mean tensor.
        running_var: The running variance tensor.
        weight: Learnable per-channel affine weight (gamma).
        bias: Learnable per-channel affine bias (beta).
        use_input_stats: If True, use input statistics; otherwise use running stats.
        momentum: Value used for running mean and variance computation.
        eps: Value added to the denominator for numerical stability.

    Returns:
        NestedTensor: The normalized output.

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(4.0).view(1, 4), torch.arange(5.0).view(1, 5))
        >>> out = F.instance_norm(nt, use_input_stats=True)
        >>> ref = NestedTensor(
        ...     F.instance_norm(nt[0].unsqueeze(0), use_input_stats=True).squeeze(0),
        ...     F.instance_norm(nt[1].unsqueeze(0), use_input_stats=True).squeeze(0),
        ... )
        >>> torch.allclose(out, ref)
        True
    """
    return _map_storage_serial(
        input,
        lambda t: F.instance_norm(
            t.unsqueeze(0),
            running_mean,
            running_var,
            weight,
            bias,
            use_input_stats,
            momentum,
            eps,
        ).squeeze(0),
    )


@NestedTensorFuncRegistry.implement(F.layer_norm)
def layer_norm(
    input: NestedTensor,
    normalized_shape: tuple,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> NestedTensor:
    r"""
    Applies Layer Normalization over the last certain number of dimensions.
    See also [torch.nn.functional.layer_norm][].

    Args:
        input: The input NestedTensor.
        normalized_shape: Input shape from an expected input of size.
        weight: Learnable affine weight (gamma).
        bias: Learnable affine bias (beta).
        eps: Value added to the denominator for numerical stability.

    Returns:
        NestedTensor: The normalized output.

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        >>> out = F.layer_norm(nt, (2,))
        >>> ref = F.layer_norm(nt.tensor, (2,))
        >>> torch.allclose(out, ref)
        True
    """
    normalized = _normalize_shape_tuple(normalized_shape)
    output = _packed_layer_norm(input, normalized, weight, bias, eps)
    if output is not None:
        return output
    return _apply_per_element(input, F.layer_norm, normalized, weight, bias, eps)


@NestedTensorFuncRegistry.implement(F.local_response_norm)
def local_response_norm(
    input: NestedTensor, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0
) -> NestedTensor:
    r"""
    Applies local response normalization over an input signal.
    See also [torch.nn.functional.local_response_norm][].

    Args:
        input: The input NestedTensor.
        size: Amount of neighbouring channels used for normalization.
        alpha: Multiplicative factor.
        beta: Exponent.
        k: Additive factor.

    Returns:
        NestedTensor: The normalized output.

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.arange(4.0).view(1, 4), torch.arange(5.0).view(1, 5))
        >>> out = F.local_response_norm(nt, size=2)
        >>> ref = F.local_response_norm(nt.tensor, size=2)
        >>> torch.allclose(out, ref)
        True
    """
    return _map_storage_serial(input, lambda t: F.local_response_norm(t.unsqueeze(0), size, alpha, beta, k).squeeze(0))


@NestedTensorFuncRegistry.implement(F.rms_norm)
def rms_norm(
    input: NestedTensor, normalized_shape: tuple, weight: Tensor | None = None, eps: float = 1e-5
) -> NestedTensor:
    r"""
    Applies Root Mean Square Layer Normalization. See also [torch.nn.functional.rms_norm][].

    Args:
        input: The input NestedTensor.
        normalized_shape: Input shape from an expected input of size.
        weight: Learnable affine weight.
        eps: Value added to the denominator for numerical stability.

    Returns:
        NestedTensor: The normalized output.

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        >>> out = F.rms_norm(nt, (2,))
        >>> ref = F.rms_norm(nt.tensor, (2,))
        >>> torch.allclose(out, ref)
        True
    """
    normalized = _normalize_shape_tuple(normalized_shape)
    output = _packed_rms_norm(input, normalized, weight, eps)
    if output is not None:
        return output
    return _apply_per_element(input, F.rms_norm, normalized, weight, eps)


@NestedTensorFuncRegistry.implement(F.normalize)
def _normalize_impl(input, p=2.0, dim=1, eps=1e-12, out=None):
    dim_adj = _translate_non_batch_dim(input, dim)
    if dim_adj == 0:
        if out is not None:
            raise NotImplementedError("F.normalize(..., out=...) is not supported on ragged dimensions.")
        if isinstance(p, (int, float)) and p > 0:
            from .aten_functions import _packed_like, _packed_to_padded

            padded, _, _, batch_idx, local_idx, _ = _packed_to_padded(input, fill_value=0.0)
            denom = torch.linalg.vector_norm(padded, ord=float(p), dim=1, keepdim=True)
            denom = torch.clamp(denom, min=eps)
            return _packed_like(input, (padded / denom)[batch_idx, local_idx])
        return _apply_per_element(input, F.normalize, p=p, dim=dim_adj, eps=eps, out=None)
    concat_dim = _concat_dim_for_tensor_dim(input, dim_adj)
    if concat_dim is None:
        if out is not None:
            raise NotImplementedError("F.normalize(..., out=...) is not supported on ragged dimensions.")
        return _apply_per_element(input, F.normalize, p=p, dim=dim_adj, eps=eps, out=None)
    return _apply_packed(input, F.normalize, p=p, dim=concat_dim, eps=eps, out=out)


@NestedTensorFuncRegistry.implement(F.pdist)
def _pdist_impl(input, p=2.0):
    return _apply_per_element(input, F.pdist, p=p)


# Per-element apply op: inspects input dimensionality.
@NestedTensorFuncRegistry.implement(F.affine_grid)
def _affine_grid_impl(input, *args, **kwargs):
    return _apply_per_element(input, F.affine_grid, *args, **kwargs)


@NestedTensorFuncRegistry.implement(F.pad)
def _pad_impl(input, pad, mode="constant", value=None):
    pad_tuple = tuple(pad)
    if input._values.dim() > 1 and len(pad_tuple) % 2 == 0:
        padded_dims = len(pad_tuple) // 2
        # Fast path when pad only touches trailing (static) dims, not ragged dim-0.
        if padded_dims < input._values.dim():
            return _apply_batch_preserving_packed(input, F.pad, pad_tuple, mode=mode, value=value)
    return _apply_per_element(input, F.pad, pad_tuple, mode=mode, value=value)


# Canonical batch-preserving path for ops where per-element dim-0 is interpreted
# as a batch-like axis, so applying once to packed _values is equivalent to
# applying per element.
_BATCH_PRESERVING_PACKED_OPS = [
    # Conv
    (F.conv1d, (3,)),
    (F.conv2d, (4,)),
    (F.conv3d, (5,)),
    (F.conv_transpose1d, (3,)),
    (F.conv_transpose2d, (4,)),
    (F.conv_transpose3d, (5,)),
    # Pooling
    (F.avg_pool1d, (3,)),
    (F.avg_pool2d, (4,)),
    (F.avg_pool3d, (5,)),
    (F.max_pool1d, (3,)),
    (F.max_pool2d, (4,)),
    (F.max_pool3d, (5,)),
    (F.adaptive_avg_pool1d, (3,)),
    (F.adaptive_avg_pool2d, (4,)),
    (F.adaptive_avg_pool3d, (5,)),
    (F.adaptive_max_pool1d, (3,)),
    (F.adaptive_max_pool2d, (4,)),
    (F.adaptive_max_pool3d, (5,)),
    (F.lp_pool1d, (3,)),
    (F.lp_pool2d, (4,)),
    (F.lp_pool3d, (5,)),
    # Grid / spatial
    (F.fold, (3,)),
    (F.interpolate, (3, 4, 5)),
    (F.unfold, (4,)),
    # Pixel shuffle / channel
    (F.channel_shuffle, (3, 4, 5)),
    (F.pixel_shuffle, (4, 5)),
    (F.pixel_unshuffle, (4, 5)),
]

for _op, _packed_dims in _BATCH_PRESERVING_PACKED_OPS:

    @NestedTensorFuncRegistry.implement(_op)
    def _apply_batch_preserving_impl(input, *args, _fn=_op, _dims=_packed_dims, **kwargs):
        if input._values.dim() in _dims:
            return _apply_batch_preserving_packed(input, _fn, *args, **kwargs)
        return _apply_per_element(input, _fn, *args, **kwargs)


@NestedTensorFuncRegistry.implement(F.one_hot)
def _one_hot_impl(input, num_classes: int = -1):
    if input._values.dim() > 1 or input._physical_shape.size(1) == 1:
        from .aten_functions import _packed_with_shape

        out_values = F.one_hot(input._values, num_classes=num_classes)
        out_physical_shape, packed_sizes, element_shapes = input._drop_trailing_physical_dims_meta(
            0,
            suffix=(int(out_values.shape[-1]),),
        )
        return _packed_with_shape(
            input,
            out_values,
            out_physical_shape,
            (*input._logical_shape, int(out_values.shape[-1])),
            permutation=input._permutation_after_replacing_trailing_dims(0, 1),
            packed_sizes=packed_sizes,
            element_shapes=element_shapes,
        )
    return _apply_per_element(input, F.one_hot, num_classes=num_classes)


@NestedTensorFuncRegistry.implement(F.dropout)
def _dropout_impl(input, p=0.5, training=True, inplace=False):
    from .aten_functions import _packed_like

    _validate_probability(float(p), error_type=ValueError)
    if (not training) or p == 0:
        return input
    if inplace:
        F.dropout(input._values, p=p, training=training, inplace=True)
        input._cached_storage = None
        return input
    return _packed_like(input, torch.ops.aten.dropout.default(input._values, p, training))


@NestedTensorFuncRegistry.implement(F.alpha_dropout)
def _alpha_dropout_impl(input, p=0.5, training=False, inplace=False):
    from .aten_functions import _packed_like

    _validate_probability(float(p), error_type=ValueError)
    if (not training) or p == 0:
        return input
    if inplace:
        F.alpha_dropout(input._values, p=p, training=training, inplace=True)
        input._cached_storage = None
        return input
    return _packed_like(input, torch.ops.aten.alpha_dropout.default(input._values, p, training))


# Channel-wise dropout variants.
_DROPOUT_CHANNELWISE_PACKED_OPS = [
    (F.dropout1d, (2, 3)),
    (F.dropout2d, (3, 4)),
    (F.dropout3d, (4, 5)),
    (F.feature_alpha_dropout, (2, 3, 4, 5)),
]

for _op, _packed_dims in _DROPOUT_CHANNELWISE_PACKED_OPS:

    @NestedTensorFuncRegistry.implement(_op)
    def _dropout_channelwise_impl(input, *args, _fn=_op, _dims=_packed_dims, **kwargs):
        training = args[1] if len(args) > 1 else kwargs.get("training", True)
        p = args[0] if len(args) > 0 else kwargs.get("p", 0.5)
        _validate_probability(float(p), error_type=ValueError)
        if (not training) or p == 0:
            return input
        if input._values.dim() in _dims:
            return _apply_batch_preserving_packed(input, _fn, *args, **kwargs)
        return _map_storage_serial(input, lambda t: _fn(t, *args, **kwargs))


@NestedTensorFuncRegistry.implement(F.bilinear)
def _bilinear_impl(input1, input2, weight, bias=None):
    from .aten_functions import _packed_new_last_dim
    from .nested_tensor import NestedTensor

    cls = type(input1) if isinstance(input1, NestedTensor) else type(input2)
    input1 = _ensure_nested_input(input1, input2, cls)
    if isinstance(input2, NestedTensor):
        if len(input1) != len(input2):
            raise ValueError(
                "NestedTensor batch length mismatch between input1 and input2: "
                f"input1={len(input1)}, input2={len(input2)}"
            )
        if input1._values.dim() >= 2 and input2._values.dim() >= 2 and _packed_pair_offsets_match(input1, input2):
            out_values = F.bilinear(input1._values, input2._values, weight, bias)
            return _packed_new_last_dim(input1, out_values, int(weight.size(0)))
    return _apply_pair(input1, input2, F.bilinear, weight, bias)


@NestedTensorFuncRegistry.implement(F.grid_sample)
def _grid_sample_impl(input, grid, *args, **kwargs):
    from .nested_tensor import NestedTensor

    cls = type(input) if isinstance(input, NestedTensor) else type(grid)
    input = _ensure_nested_input(input, grid, cls)
    if isinstance(grid, NestedTensor):
        if len(input) != len(grid):
            raise ValueError(
                "NestedTensor batch length mismatch between input and grid: " f"input={len(input)}, grid={len(grid)}"
            )
        if input._values.dim() in (4, 5) and grid._values.dim() in (4, 5) and _packed_pair_offsets_match(input, grid):
            return _from_batch_preserving_values(input, F.grid_sample(input._values, grid._values, *args, **kwargs))
    elif (
        isinstance(grid, Tensor)
        and input._values.dim() in (4, 5)
        and grid.dim() in (4, 5)
        and grid.size(0) == input._values.size(0)
    ):
        return _from_batch_preserving_values(input, F.grid_sample(input._values, grid, *args, **kwargs))
    return _apply_pair(input, grid, F.grid_sample, *args, **kwargs)


@NestedTensorFuncRegistry.implement(F.pairwise_distance)
def _pairwise_distance_impl(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    from .nested_tensor import NestedTensor

    cls = type(x1) if isinstance(x1, NestedTensor) else type(x2)
    x1 = _ensure_nested_input(x1, x2, cls)
    if isinstance(x2, NestedTensor):
        if len(x1) != len(x2):
            raise ValueError("NestedTensor batch length mismatch between x1 and x2: " f"x1={len(x1)}, x2={len(x2)}")
        if x1._values.dim() == 2 and x2._values.dim() == 2 and _packed_pair_offsets_match(x1, x2):
            from .aten_functions import _packed_with_shape

            out_values = F.pairwise_distance(x1._values, x2._values, p=p, eps=eps, keepdim=keepdim)
            if keepdim:
                out_physical_shape, packed_sizes, element_shapes = x1._replace_trailing_physical_dims_meta((1,))
            else:
                out_physical_shape, packed_sizes, element_shapes = x1._drop_trailing_physical_dims_meta(1)
            return _packed_with_shape(
                x1,
                out_values,
                out_physical_shape,
                ((*x1._logical_shape[:-1], 1) if keepdim else x1._logical_shape[:-1]),
                permutation=(
                    x1._permutation_after_replacing_trailing_dims(1, 1)
                    if keepdim
                    else x1._permutation_after_dropping_trailing_dims(1)
                ),
                packed_sizes=packed_sizes,
                element_shapes=element_shapes,
            )
    return _apply_pair(x1, x2, F.pairwise_distance, p=p, eps=eps, keepdim=keepdim)


# Table-driven registrations — pool ops that return (output, indices)

_POOL_WITH_INDICES_OPS = [
    F.max_pool1d_with_indices,
    F.max_pool2d_with_indices,
    F.max_pool3d_with_indices,
    F.adaptive_max_pool1d_with_indices,
    F.adaptive_max_pool2d_with_indices,
    F.adaptive_max_pool3d_with_indices,
    F.fractional_max_pool2d_with_indices,
    F.fractional_max_pool3d_with_indices,
]

for _op in _POOL_WITH_INDICES_OPS:

    @NestedTensorFuncRegistry.implement(_op)
    def _pool_indices_impl(input, *args, _fn=_op, **kwargs):
        return _map_storage_pair(input, _fn, *args, **kwargs)


# Table-driven registrations — max unpool ops (binary per-element: input + indices)

_MAX_UNPOOL_OPS = [F.max_unpool1d, F.max_unpool2d, F.max_unpool3d]

for _op in _MAX_UNPOOL_OPS:

    @NestedTensorFuncRegistry.implement(_op)
    def _unpool_impl(input, indices, *args, _fn=_op, **kwargs):
        return _apply_pair(input, indices, _fn, *args, **kwargs)


# Softmax family


@NestedTensorFuncRegistry.implement(F.gumbel_softmax)
def _gumbel_softmax_impl(logits, *args, dim=-1, **kwargs):
    from .aten_functions import _packed_like, _packed_to_padded

    dim_adj = _translate_non_batch_dim(logits, dim)
    if dim_adj == 0:
        padded, _, _, batch_idx, local_idx, _ = _packed_to_padded(logits, fill_value=float("-inf"))
        out_padded = F.gumbel_softmax(padded, *args, dim=1, **kwargs)
        return _packed_like(logits, out_padded[batch_idx, local_idx])
    concat_dim = _concat_dim_for_tensor_dim(logits, dim_adj)
    if concat_dim is not None:
        return _apply_packed(logits, F.gumbel_softmax, *args, dim=concat_dim, **kwargs)
    return _map_storage_serial(logits, lambda t: F.gumbel_softmax(t, *args, dim=dim_adj, **kwargs))


@NestedTensorFuncRegistry.implement(F.log_softmax)
def _f_log_softmax_impl(input, dim=-1, _stacklevel=3, dtype=None):
    del _stacklevel  # keep F.log_softmax-compatible signature
    return torch.log_softmax(input, dim=dim, dtype=dtype)


@NestedTensorFuncRegistry.implement(F.softmax)
def _f_softmax_impl(input, dim=-1, _stacklevel=3, dtype=None):
    del _stacklevel  # keep F.softmax-compatible signature
    return torch.softmax(input, dim=dim, dtype=dtype)


@NestedTensorFuncRegistry.implement(F.softmin)
def _f_softmin_impl(input, dim=-1, _stacklevel=3, dtype=None):
    del _stacklevel  # keep F.softmin-compatible signature
    source = input if dtype is None else input.to(dtype=dtype)
    return torch.softmax(torch.neg(source), dim=dim, dtype=None)


# Fractional max pool — boolean_dispatch wrappers that support return_indices


@NestedTensorFuncRegistry.implement(F.fractional_max_pool2d)
def _fractional_max_pool2d_impl(input, *args, return_indices=False, **kwargs):
    if return_indices:
        return _apply_with_indices(input, F.fractional_max_pool2d, *args, **kwargs)
    random_samples = kwargs.get("_random_samples")
    random_samples_ok = random_samples is None or (
        isinstance(random_samples, Tensor) and random_samples.size(0) == input._values.size(0)
    )
    if input._values.dim() == 4 and random_samples_ok:
        return _apply_batch_preserving_packed(input, F.fractional_max_pool2d, *args, **kwargs)
    return _apply_per_element(input, F.fractional_max_pool2d, *args, **kwargs)


@NestedTensorFuncRegistry.implement(F.fractional_max_pool3d)
def _fractional_max_pool3d_impl(input, *args, return_indices=False, **kwargs):
    if return_indices:
        return _apply_with_indices(input, F.fractional_max_pool3d, *args, **kwargs)
    random_samples = kwargs.get("_random_samples")
    random_samples_ok = random_samples is None or (
        isinstance(random_samples, Tensor) and random_samples.size(0) == input._values.size(0)
    )
    if input._values.dim() == 5 and random_samples_ok:
        return _apply_batch_preserving_packed(input, F.fractional_max_pool3d, *args, **kwargs)
    return _apply_per_element(input, F.fractional_max_pool3d, *args, **kwargs)
