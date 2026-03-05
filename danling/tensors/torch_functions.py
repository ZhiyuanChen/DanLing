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
``torch.*`` function overrides for NestedTensor via ``__torch_function__``.

This module is the **Level 2** dispatch layer.  When a ``torch.*`` call
(e.g. ``torch.cat``, ``torch.mean``, ``torch.einsum``) involves a
[NestedTensor][], ``__torch_function__`` checks
[NestedTensorFuncRegistry][] for a registered handler.

Handlers here use several strategies depending on the op's needs:

* **Packed fast-path** — ops that work directly on the concatenated
  ``_values`` tensor (via [_from_packed][nested_tensor.NestedTensor._from_packed]) without
  knowing element boundaries.
* **Per-element dispatch** — ops that must be applied to each element
  individually (via [_map_storage][ops._map_storage]), e.g. when dimension
  indices need translation or output shapes differ per element.
* **Concat-apply** — ops that pad to a uniform shape, apply, then extract
  valid regions (via [_concat_apply_same_shape][ops._concat_apply_same_shape]).

If no handler is registered here, the call falls through to aten
decomposition and then to ``__torch_dispatch__`` (see ``aten_functions``).
"""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .ops import (
    TORCH_BINARY_ELEMENTWISE_OPS,
    TORCH_UNARY_ELEMENTWISE_OPS,
    NestedTensorFuncRegistry,
    _binary_op_maybe_tensor,
    _concat_apply_same_shape,
    _concat_dim_for_tensor_dim,
    _get_batch_dim,
    _map_storage,
    _map_storage_pair,
    _normalize_dim,
    _reduce,
    _reduce_dim,
    _reduce_dim_pair,
    _reduce_none,
    _reduce_none_pair,
    _translate_dim,
    _translate_dims,
    _translate_non_batch_dim,
    _try_stack,
)

if TYPE_CHECKING:
    from .nested_tensor import NestedTensor


# Arithmetic


# clamp and nan_to_num are NOT registered here.
# They fall through to aten decomposition → __torch_dispatch__ where
# _unary_handler operates directly on _values (no per-element loop).


@NestedTensorFuncRegistry.implement(torch.addcdiv)
def addcdiv(input, tensor1, tensor2, *, value=1):
    r"""
    Performs the element-wise division of `tensor1` by `tensor2`, multiplies the result by the scalar
    `value` and adds it to `input`.
    See also [torch.addcdiv][].

    A torch-level handler is needed because one or more arguments may be a plain Tensor
    (e.g. the padded view) mixed with NestedTensors — the aten ternary handler only works
    when all tensor arguments are NestedTensor.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> t1 = torch.tensor([2.0, 3.0])
        >>> t2 = torch.tensor([4.0, 5.0])
        >>> torch.allclose(torch.addcdiv(nt, t1, t2), torch.addcdiv(nt.tensor, t1, t2))
        True
    """
    from .nested_tensor import NestedTensor
    from .ops import _ternary_op

    ref = next((t for t in (input, tensor1, tensor2) if isinstance(t, NestedTensor)), None)
    if ref is None:
        return torch.addcdiv(input, tensor1, tensor2, value=value)
    return _ternary_op(ref, input, tensor1, tensor2, torch.addcdiv, value=value)


@NestedTensorFuncRegistry.implement(torch.addcmul)
def addcmul(input, tensor1, tensor2, *, value=1):
    r"""
    Performs the element-wise multiplication of `tensor1` by `tensor2`, multiplies the result by the scalar
    `value` and adds it to `input`.
    See also [torch.addcmul][].

    See [addcdiv][] for why the torch-level handler is required.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> t1 = torch.tensor([2.0, 3.0])
        >>> t2 = torch.tensor([4.0, 5.0])
        >>> torch.allclose(torch.addcmul(nt, t1, t2), torch.addcmul(nt.tensor, t1, t2))
        True
    """
    from .nested_tensor import NestedTensor
    from .ops import _ternary_op

    ref = next((t for t in (input, tensor1, tensor2) if isinstance(t, NestedTensor)), None)
    if ref is None:
        return torch.addcmul(input, tensor1, tensor2, value=value)
    return _ternary_op(ref, input, tensor1, tensor2, torch.addcmul, value=value)


# Comparison


def _compare(input, other, op, **kwargs) -> bool:
    r"""Shared comparison logic for allclose/equal."""
    from .aten_functions import _offsets_match
    from .nested_tensor import NestedTensor

    if not isinstance(input, NestedTensor):
        input = other.nested_like(input)
    elif not isinstance(other, NestedTensor):
        other = input.nested_like(other)
    if len(input) != len(other):
        return False
    if _offsets_match(input._offsets, other._offsets):
        return op(input._values, other._values, **kwargs)
    return all(op(x, y, **kwargs) for x, y in zip(input._storage, other._storage))


def _validate_pairwise_batch_length(
    lhs: NestedTensor, rhs: NestedTensor, *, op_name: str, lhs_name: str, rhs_name: str
) -> None:
    r"""Validate two NestedTensor operands have matching batch lengths."""
    lhs_len = len(lhs)
    rhs_len = len(rhs)
    if lhs_len != rhs_len:
        raise ValueError(
            f"{op_name}: NestedTensor batch length mismatch between {lhs_name} and {rhs_name}: "
            f"{lhs_name}={lhs_len}, {rhs_name}={rhs_len}"
        )


@NestedTensorFuncRegistry.implement(torch.allclose)
def allclose(
    input: NestedTensor, other: NestedTensor | Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    r"""
    This function checks if `input` and `other` satisfy the condition:
    ``|input_i - other_i| <= atol + rtol * |other_i|`` elementwise, for all
    elements of `input` and `other`. The behaviour of this function is analogous to
    [numpy.allclose](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html).
    See also [torch.allclose][].

    Args:
        input: The first NestedTensor to compare.
        other: The second tensor to compare.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: If True, NaN values are considered equal.

    Returns:
        bool: True if all elements are close, False otherwise.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> torch.allclose(nt, nt.tensor)
        True
    """
    return _compare(input, other, torch.allclose, rtol=rtol, atol=atol, equal_nan=equal_nan)


@NestedTensorFuncRegistry.implement(torch.equal)
def equal(input: NestedTensor, other: NestedTensor | Tensor) -> bool:
    r"""
    ``True`` if two tensors have the same size and elements, ``False`` otherwise.
    See also [torch.equal][].

    Args:
        input: The first NestedTensor to compare.
        other: The second tensor to compare.

    Returns:
        bool: True if both tensors have the same size and elements.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1, 2]), torch.tensor([3, 4]))
        >>> torch.equal(nt, nt.tensor)
        True
    """
    return _compare(input, other, torch.equal)


# Creation
# NOTE: The aten-level handlers in aten_functions.py handle the standard decomposition
# path (preserving packing layout).  These torch-level handlers are reached via
# direct torch.*_like() calls.


# Creation ops (zeros_like, ones_like, etc.) are NOT registered here.
# They fall through to aten decomposition → __torch_dispatch__ →
# _unary_handler on _values, which is faster than per-element _map_storage.


# Concatenation & Splitting


def _validate_cat_nested_meta(tensors: tuple[Tensor | NestedTensor, ...], ref: NestedTensor) -> None:
    r"""Validate that all NestedTensor inputs to cat share structural metadata."""
    first_idx = next(i for i, t in enumerate(tensors) if isinstance(t, type(ref)))
    first_meta = (ref.batch_first, ref.padding_value, ref.mask_value)
    for idx, t in enumerate(tensors):
        if not isinstance(t, type(ref)):
            continue
        current_meta = (t.batch_first, t.padding_value, t.mask_value)
        if current_meta != first_meta:
            raise ValueError(
                "torch.cat for NestedTensor requires all NestedTensor inputs to share "
                "batch_first, padding_value, and mask_value, but got "
                f"tensors[{first_idx}]=(batch_first={ref.batch_first}, "
                f"padding_value={ref.padding_value}, mask_value={ref.mask_value}) and "
                f"tensors[{idx}]=(batch_first={t.batch_first}, "
                f"padding_value={t.padding_value}, mask_value={t.mask_value})."
            )


@NestedTensorFuncRegistry.implement(torch.cat)
def cat(tensors: tuple[Tensor | NestedTensor, ...], dim: int = 0):
    r"""
    Concatenates the given sequence of tensors in `tensors` in the given dimension.
    See also [torch.cat][].

    Args:
        tensors: Sequence of tensors or NestedTensors to concatenate.
        dim: Dimension along which to concatenate.

    Returns:
        Tensor | NestedTensor: The concatenated result.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.cat((nt, nt), dim=0)
        >>> ref = torch.cat((nt.tensor, nt.tensor), dim=0)
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    ref = next((t for t in tensors if isinstance(t, NestedTensor)), None)
    if ref is None:
        return torch.cat(tensors, dim=dim)
    _validate_cat_nested_meta(tensors, ref)

    dim = _normalize_dim(dim, ref.dim())
    batch_dim = _get_batch_dim(ref)

    if dim == batch_dim:
        # Check if all inputs are NestedTensor (common case — enables packed fast path)
        all_nt = all(isinstance(t, NestedTensor) for t in tensors)
        if all_nt:
            nt_tensors = tensors  # type: ignore[assignment]
            packed_rank = nt_tensors[0]._values.dim()
            packed_tail = nt_tensors[0]._values.shape[1:]
            can_cat_packed = True
            for t in nt_tensors[1:]:
                if t._values.dim() != packed_rank:
                    can_cat_packed = False
                    break
                if packed_rank > 1 and t._values.shape[1:] != packed_tail:
                    can_cat_packed = False
                    break

            if not can_cat_packed:
                # Incompatible packed layouts (e.g., one flattened, one N-D packed):
                # fall back to unpack→repack.
                storage: list[Tensor] = []
                state: Mapping = ref._meta()
                for tensor in nt_tensors:
                    storage.extend(tensor._storage)
                return NestedTensor(storage, **state)

            # Fast path: merge packed representations directly (no unpack/repack)
            values_list = [t._values for t in nt_tensors]
            new_values = torch.cat(values_list, dim=0)

            # Merge offsets: shift each NT's offsets by the cumulative _values length
            offset_parts = []
            cumulative = 0
            for i, t in enumerate(nt_tensors):
                # Skip the leading 0 for all but the first NT
                offsets_i = t._offsets if i == 0 else t._offsets[1:] + cumulative
                offset_parts.append(offsets_i)
                cumulative += int(t._offsets[-1].item())
            new_offsets = torch.cat(offset_parts, dim=0)

            # Pad shape_tensors to the same width before stacking
            max_cols = builtins.max(t._shape_tensor.size(1) for t in nt_tensors)
            if max_cols > 0:
                padded_shapes = []
                for t in nt_tensors:
                    st = t._shape_tensor
                    if st.size(1) < max_cols:
                        st = torch.nn.functional.pad(st, (0, max_cols - st.size(1)))
                    padded_shapes.append(st)
                new_shape_tensor = torch.cat(padded_shapes, dim=0)
            else:
                new_shape_tensor = torch.empty(len(new_offsets) - 1, 0, dtype=torch.long)

            return NestedTensor._from_packed(
                new_values,
                new_offsets,
                new_shape_tensor,
                batch_first=ref.batch_first,
                padding_value=ref.padding_value,
                mask_value=ref.mask_value,
                pin_memory=ref._pin_memory,
            )
        # Fallback: mix of NT and plain tensors
        storage: list = []
        state: Mapping = ref._meta()
        for tensor in tensors:
            if isinstance(tensor, NestedTensor):
                storage.extend(tensor._storage)
            else:
                storage.append(tensor)
        return NestedTensor(storage, **state)

    if not all(isinstance(t, NestedTensor) for t in tensors):
        raise NotImplementedError("NestedTensor cat along non-batch dim requires all inputs to be NestedTensor.")
    first: NestedTensor = tensors[0]  # type: ignore[index]
    if any(len(t) != len(first) for t in tensors):
        lengths = [len(t) for t in tensors]
        raise ValueError(
            "NestedTensor cat along non-batch dim requires the same batch length, " f"but got lengths {lengths}."
        )

    dim_adj = _translate_dim(first, dim)
    storage = [
        torch.cat([t._storage[i] for t in tensors], dim=dim_adj) for i in range(len(first))  # type: ignore[index]
    ]
    return NestedTensor(storage, **first._meta())


# Aliases for torch.cat
for _alias in (torch.concat, torch.concatenate):

    @NestedTensorFuncRegistry.implement(_alias)
    def _cat_alias(tensors, dim: int = 0, _fn=_alias):  # noqa: B023 — default binds loop var
        return torch.cat(tuple(tensors), dim=dim)


@NestedTensorFuncRegistry.implement(torch.chunk)
def chunk(input: NestedTensor, chunks: int, dim: int = 0):
    r"""
    Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor.
    See also [torch.chunk][].

    Args:
        input: The input NestedTensor.
        chunks: Number of chunks to split into.
        dim: Dimension along which to split.

    Returns:
        tuple[NestedTensor, ...]: Tuple of NestedTensor chunks.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor

        Chunk along batch dimension:
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.chunk(nt, 2, dim=0)
        >>> ref = (NestedTensor(nt[0]), NestedTensor(nt[1]))
        >>> torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
        True

        Chunk along feature dimension:
        >>> a = torch.randn(3, 6)
        >>> b = torch.randn(5, 6)
        >>> nt = NestedTensor(a, b)
        >>> parts = torch.chunk(nt, 3, dim=-1)
        >>> len(parts)
        3
        >>> parts[0][0].shape
        torch.Size([3, 2])
    """
    from .nested_tensor import NestedTensor

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)

    if chunks <= 0:
        raise ValueError("chunks must be a positive integer.")

    # ── Batch dim chunk ──
    if dim == batch_dim:
        storage = input._storage
        if not storage:
            return ()
        chunk_size = (len(storage) + chunks - 1) // chunks
        return tuple(
            NestedTensor(storage[i : i + chunk_size], **input._meta())  # noqa: E203
            for i in range(0, len(storage), chunk_size)
        )

    # ── Non-batch dim chunk ──
    storage = input._storage
    if not storage:
        return ()

    elem_dim = _translate_dim(input, dim)
    chunk_results = [torch.chunk(t, chunks, dim=elem_dim) for t in storage]
    chunk_counts = [len(parts) for parts in chunk_results]
    num_chunks = chunk_counts[0]
    if any(count != num_chunks for count in chunk_counts[1:]):
        raise ValueError(
            "torch.chunk along non-batch dim requires uniform per-element chunk counts, "
            f"but got counts {chunk_counts}."
        )
    return tuple(
        NestedTensor([chunk_results[i][k] for i in range(len(storage))], **input._meta()) for k in range(num_chunks)
    )


@NestedTensorFuncRegistry.implement(torch.split)
def split(input: NestedTensor, split_size_or_sections, dim: int = 0):
    r"""
    Splits the tensor into chunks. Each chunk is a view of the original tensor.
    See also [torch.split][].

    Supports splitting along both the batch dimension and non-batch dimensions.

    Args:
        input: The input NestedTensor.
        split_size_or_sections: Size of each chunk or list of sizes for each chunk.
        dim: Dimension along which to split.

    Returns:
        tuple[NestedTensor, ...]: Tuple of NestedTensor chunks.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor

        Split along batch dimension:
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.split(nt, 1, dim=0)
        >>> ref = (NestedTensor(nt[0]), NestedTensor(nt[1]))
        >>> torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
        True

        Split along feature dimension:
        >>> a = torch.randn(3, 6)
        >>> b = torch.randn(5, 6)
        >>> nt = NestedTensor(a, b)
        >>> parts = torch.split(nt, 2, dim=-1)
        >>> len(parts)
        3
        >>> torch.equal(parts[0][0], a[:, :2]) and torch.equal(parts[0][1], b[:, :2])
        True

        Method-style call:
        >>> parts = nt.split(3, dim=-1)
        >>> len(parts)
        2
        >>> torch.equal(parts[0][0], a[:, :3]) and torch.equal(parts[0][1], b[:, :3])
        True
    """
    from .nested_tensor import NestedTensor

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)

    # ── Batch dim split ──
    if dim == batch_dim:
        if isinstance(split_size_or_sections, int):
            split_size = split_size_or_sections
            if split_size <= 0:
                raise ValueError("split_size must be a positive integer.")
            storage = input._storage
            return tuple(
                NestedTensor(storage[i : i + split_size], **input._meta())  # noqa: E203
                for i in range(0, len(storage), split_size)
            )

        if not isinstance(split_size_or_sections, (list, tuple)):
            raise TypeError(
                f"split_size_or_sections must be int or a sequence of ints, got {type(split_size_or_sections)}"
            )

        storage = input._storage
        chunks = []
        start = 0
        for section in split_size_or_sections:
            if section < 0:
                raise ValueError("split sections must be non-negative.")
            end = start + int(section)
            chunks.append(NestedTensor(storage[start:end], **input._meta()))
            start = end
        if start != len(storage):
            raise ValueError("split sections do not sum to the NestedTensor batch size.")
        return tuple(chunks)

    # ── Non-batch dim split ──
    storage = input._storage
    if not storage:
        return (NestedTensor([], **input._meta(include_dtype=True)),)

    elem_dim = _translate_dim(input, dim)
    split_results = [torch.split(t, split_size_or_sections, dim=elem_dim) for t in storage]
    split_counts = [len(parts) for parts in split_results]
    num_chunks = split_counts[0]
    if any(count != num_chunks for count in split_counts[1:]):
        raise ValueError(
            "torch.split along non-batch dim requires uniform per-element split counts, "
            f"but got counts {split_counts}."
        )
    return tuple(
        NestedTensor([split_results[i][k] for i in range(len(storage))], **input._meta()) for k in range(num_chunks)
    )


@NestedTensorFuncRegistry.implement(torch.stack)
def stack(*args, **kwargs):
    r"""
    Concatenates a sequence of tensors along a new dimension.
    See also [torch.stack][].

    Args:
        *args: Positional arguments; first is the sequence of NestedTensors.
        **kwargs: Keyword arguments; supports `dim` (only 0 is supported).

    Returns:
        NestedTensor: The stacked result.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.stack((nt, nt), dim=0)
        >>> ref = NestedTensor(
        ...     torch.stack((nt[0], nt[0]), dim=0),
        ...     torch.stack((nt[1], nt[1]), dim=0),
        ... )
        >>> torch.equal(out, ref)
        True
    """
    tensors = args[0] if args else ()
    dim = kwargs.get("dim", 0)
    if dim != 0:
        raise NotImplementedError(f"NestedTensor only supports stack when dim=0, but got {dim}")
    if not isinstance(tensors, (tuple, list)) or not tensors:
        raise ValueError("Expected a non-empty sequence of NestedTensor objects.")
    from .nested_tensor import NestedTensor

    if not all(isinstance(t, NestedTensor) for t in tensors):
        raise NotImplementedError("torch.stack for NestedTensor requires all inputs to be NestedTensor.")
    first: NestedTensor = tensors[0]
    if any(
        (t.batch_first != first.batch_first)
        or (t.padding_value != first.padding_value)
        or (t.mask_value != first.mask_value)
        for t in tensors[1:]
    ):
        raise ValueError("All NestedTensor inputs must share batch_first, padding_value, and mask_value to stack.")
    if any(len(t) != len(first) for t in tensors[1:]):
        raise ValueError("All NestedTensor inputs must have the same batch length to stack.")
    state = {
        "batch_first": first.batch_first,
        "padding_value": first.padding_value,
        "mask_value": first.mask_value,
        "pin_memory": first._pin_memory,
        "device": first.device,
        "dtype": first.dtype,
    }
    storage = [torch.stack([t._storage[i] for t in tensors], dim=0) for i in range(len(first))]
    return NestedTensor(storage, **state)


@NestedTensorFuncRegistry.implement(torch.unbind)
def unbind(input: NestedTensor, dim: int = 0):
    r"""
    Removes a tensor dimension.
    See also [torch.unbind][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to remove (only batch dimension is supported).

    Returns:
        list[Tensor]: List of tensors from the storage.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.unbind(nt, dim=0)
        >>> torch.equal(out[0], nt[0]) and torch.equal(out[1], nt[1])
        True
    """
    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim != batch_dim:
        raise NotImplementedError("torch.unbind for NestedTensor only supports unbinding along the batch dimension.")
    return input._storage


# Cumulative


@NestedTensorFuncRegistry.implement(torch.cummax)
def cummax(input: NestedTensor, dim: int):
    return torch.ops.aten.cummax.default(input, dim)


@NestedTensorFuncRegistry.implement(torch.cummin)
def cummin(input: NestedTensor, dim: int):
    return torch.ops.aten.cummin.default(input, dim)


@NestedTensorFuncRegistry.implement(torch.cumsum)
def cumsum(input: NestedTensor, dim: int, *, dtype: torch.dtype | None = None):
    if dtype is None:
        return torch.ops.aten.cumsum.default(input, dim)
    return torch.ops.aten.cumsum.default(input, dim, dtype=dtype)


@NestedTensorFuncRegistry.implement(torch.cumprod)
def cumprod(input: NestedTensor, dim: int, *, dtype: torch.dtype | None = None):
    if dtype is None:
        return torch.ops.aten.cumprod.default(input, dim)
    return torch.ops.aten.cumprod.default(input, dim, dtype=dtype)


@NestedTensorFuncRegistry.implement(torch.logcumsumexp)
def logcumsumexp(input: NestedTensor, dim: int):
    return torch.ops.aten.logcumsumexp.default(input, dim)


# Normalization


def _can_concat_normalize(input: NestedTensor, normalized_shape: tuple[int, ...]) -> bool:
    if input._values.dim() <= 1:
        return False
    if len(normalized_shape) == 0 or len(normalized_shape) > input._values.dim() - 1:
        return False
    return tuple(input._values.shape[-len(normalized_shape) :]) == normalized_shape


@NestedTensorFuncRegistry.implement(torch.layer_norm)
def layer_norm(
    input: NestedTensor,
    normalized_shape,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
    cudnn_enable: bool = True,
):
    normalized = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    if _can_concat_normalize(input, normalized):
        from .aten_functions import _packed_like

        output, _, _ = torch.ops.aten.native_layer_norm.default(input._values, normalized, weight, bias, eps)
        return _packed_like(input, output)
    return _map_storage(input, lambda t: torch.layer_norm(t, normalized, weight, bias, eps, cudnn_enable))


if hasattr(torch, "rms_norm"):

    @NestedTensorFuncRegistry.implement(torch.rms_norm)
    def rms_norm(input: NestedTensor, normalized_shape, weight: Tensor | None = None, eps: float | None = None):
        normalized = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        if _can_concat_normalize(input, normalized):
            from .aten_functions import _packed_like

            out_values = torch.ops.aten.rms_norm.default(input._values, normalized, weight, eps)
            return _packed_like(input, out_values)
        return _map_storage(input, lambda t: torch.rms_norm(t, normalized, weight, eps))


# Dropout & Sampling


@NestedTensorFuncRegistry.implement(torch.bernoulli)
def bernoulli(input: NestedTensor, *, generator=None):
    r"""
    Draws binary random numbers (0 or 1) from a Bernoulli distribution.
    See also [torch.bernoulli][].

    Args:
        input: The input NestedTensor of probabilities.
        generator: Optional random number generator.

    Returns:
        NestedTensor: A new NestedTensor with binary samples.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([0.2, 0.8]), torch.tensor([0.6, 0.4, 0.1]))
        >>> out = torch.bernoulli(nt)
        >>> out[0].shape == nt[0].shape and out[1].shape == nt[1].shape
        True
    """
    return torch.ops.aten.bernoulli.default(input, generator=generator)


def _validate_dropout_probability(p: float) -> None:
    # Keep torch.* API parity: torch.dropout/alpha_dropout/feature_alpha_dropout
    # raise RuntimeError for out-of-range probabilities.
    if p < 0.0 or p > 1.0:
        raise RuntimeError(f"dropout probability has to be between 0 and 1, but got {p}")


@NestedTensorFuncRegistry.implement(torch.dropout)
def dropout(input: NestedTensor, p: float = 0.5, train: bool = True):
    r"""
    Applies dropout to the input.
    See also [torch.dropout][].

    Args:
        input: The input NestedTensor.
        p: Probability of an element to be zeroed.
        train: Whether the module is in training mode.

    Returns:
        NestedTensor: The result with dropout applied.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> torch.allclose(torch.dropout(nt, p=0.0, train=False), torch.dropout(nt.tensor, p=0.0, train=False))
        True
    """
    _validate_dropout_probability(float(p))
    if (not train) or p == 0:
        return input
    from .aten_functions import _packed_like

    return _packed_like(input, torch.ops.aten.dropout.default(input._values, p, train))


@NestedTensorFuncRegistry.implement(torch.alpha_dropout)
def alpha_dropout(input: NestedTensor, p: float = 0.5, train: bool = False):
    r"""Applies alpha dropout to the input NestedTensor."""
    _validate_dropout_probability(float(p))
    if (not train) or p == 0:
        return input
    from .aten_functions import _packed_like

    return _packed_like(input, torch.ops.aten.alpha_dropout.default(input._values, p, train))


@NestedTensorFuncRegistry.implement(torch.feature_alpha_dropout)
def feature_alpha_dropout(input: NestedTensor, p: float = 0.5, train: bool = False):
    r"""Applies feature alpha dropout to the input NestedTensor."""
    _validate_dropout_probability(float(p))
    if (not train) or p == 0:
        return input
    from .aten_functions import _packed_like

    return _packed_like(input, torch.ops.aten.feature_alpha_dropout.default(input._values, p, train))


# Indexing & Masking


@NestedTensorFuncRegistry.implement(torch.gather)
def gather(input: NestedTensor, dim: int, index):
    r"""
    Gathers values along an axis specified by `dim`.
    See also [torch.gather][].

    Args:
        input: The input NestedTensor.
        dim: Dimension along which to gather.
        index: Indices of elements to gather.

    Returns:
        NestedTensor: The gathered result.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> index = torch.tensor([[0, 1], [1, 0]])
        >>> out = torch.gather(nt, 1, index)
        >>> ref = torch.gather(nt.tensor, 1, index)
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    dim_adj = _translate_non_batch_dim(input, dim, name="gather")
    if isinstance(index, Tensor) and input.shape == index.shape:
        index = input.nested_like(index, strict=False)
    if isinstance(index, NestedTensor):
        if len(input) != len(index):
            raise ValueError(
                "NestedTensor batch length mismatch between input and index: " f"input={len(input)}, index={len(index)}"
            )
        return NestedTensor(torch.gather(t, dim_adj, idx) for t, idx in zip(input._storage, index._storage))
    return NestedTensor(torch.gather(t, dim_adj, index) for t in input._storage)


def _register_index_op(func, name):
    @NestedTensorFuncRegistry.implement(func)
    def impl(input: NestedTensor, dim: int, index, source, **kwargs):
        from .nested_tensor import NestedTensor

        dim_adj = _translate_non_batch_dim(input, dim, name=name)

        if isinstance(index, Tensor):
            index = index.to(dtype=torch.long)
        if isinstance(source, Tensor) and source.shape == input.shape:
            source = input.nested_like(source, strict=False)

        if isinstance(index, NestedTensor) and len(index) != len(input):
            raise ValueError(
                "NestedTensor batch length mismatch between input and index: " f"input={len(input)}, index={len(index)}"
            )
        if isinstance(source, NestedTensor) and len(source) != len(input):
            raise ValueError(
                "NestedTensor batch length mismatch between input and source: "
                f"input={len(input)}, source={len(source)}"
            )

        storage = []
        for i, t in enumerate(input._storage):
            idx = index._storage[i] if isinstance(index, NestedTensor) else index
            src = source._storage[i] if isinstance(source, NestedTensor) else source
            if isinstance(idx, Tensor) and idx.device != t.device:
                idx = idx.to(device=t.device)
            if isinstance(src, Tensor) and src.device != t.device:
                src = src.to(device=t.device)
            storage.append(func(t, dim_adj, idx, src, **kwargs))
        return NestedTensor(storage, **input._meta())


_register_index_op(torch.index_add, "index_add")
_register_index_op(torch.index_copy, "index_copy")


@NestedTensorFuncRegistry.implement(torch.index_put)
def index_put(input: NestedTensor, indices, values, accumulate: bool = False):
    r"""
    Puts values into the input tensor at the specified indices.
    See also [torch.index_put][].

    Args:
        input: The input NestedTensor.
        indices: Tuple of index tensors.
        values: Values to put into the tensor.
        accumulate: Whether to accumulate into self instead of replacing.

    Returns:
        NestedTensor: The result with values placed at indices.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> a = torch.tensor([1.0, 2.0])
        >>> b = torch.tensor([3.0, 4.0, 5.0])
        >>> nt = NestedTensor(a, b)
        >>> idx = torch.tensor([0])
        >>> values = torch.tensor([9.0])
        >>> out = torch.index_put(nt, (idx,), values)
        >>> ref = NestedTensor(torch.index_put(a, (idx,), values), torch.index_put(b, (idx,), values))
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if not isinstance(indices, (tuple, list)):
        indices = (indices,)

    if isinstance(values, Tensor) and values.shape == input.shape:
        values = input.nested_like(values, strict=False)

    for idx in indices:
        if isinstance(idx, NestedTensor) and len(idx) != len(input):
            raise ValueError(
                "NestedTensor batch length mismatch between input and index: " f"input={len(input)}, index={len(idx)}"
            )
    if isinstance(values, NestedTensor) and len(values) != len(input):
        raise ValueError(
            "NestedTensor batch length mismatch between input and values: " f"input={len(input)}, values={len(values)}"
        )

    storage = []
    for i, t in enumerate(input._storage):
        per_tensor_indices = []
        for idx in indices:
            idx_i = idx._storage[i] if isinstance(idx, NestedTensor) else idx
            if isinstance(idx_i, Tensor) and idx_i.device != t.device:
                idx_i = idx_i.to(device=t.device)
            per_tensor_indices.append(idx_i)
        value_i = values._storage[i] if isinstance(values, NestedTensor) else values
        if isinstance(value_i, Tensor) and value_i.device != t.device:
            value_i = value_i.to(device=t.device)
        storage.append(torch.index_put(t, tuple(per_tensor_indices), value_i, accumulate=accumulate))
    return NestedTensor(storage, **input._meta())


@NestedTensorFuncRegistry.implement(torch.index_select)
def index_select(input: NestedTensor, dim: int, index: Tensor):
    r"""
    Returns a new tensor which indexes the `input` tensor along dimension `dim` using the entries in
    `index` which is a `LongTensor`.
    See also [torch.index_select][].

    Args:
        input: The input NestedTensor.
        dim: Dimension along which to index.
        index: A LongTensor containing the indices to select.

    Returns:
        NestedTensor: A new NestedTensor with selected elements.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> index = torch.tensor([0])
        >>> out = torch.index_select(nt, 1, index)
        >>> ref = torch.index_select(nt.tensor, 1, index)
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        indices = index.to(dtype=torch.long, device="cpu").tolist()
        return NestedTensor([input._storage[i] for i in indices], **input._meta())
    dim_adj = _translate_dim(input, dim)
    index_device = index if index.device == input.device else index.to(device=input.device)
    return NestedTensor(torch.index_select(t, dim_adj, index_device) for t in input._storage)


@NestedTensorFuncRegistry.implement(torch.masked_fill)
def masked_fill(input: NestedTensor, mask, value):
    r"""
    Fills elements of input where mask is True with value.
    See also [torch.masked_fill][].

    Args:
        input: The input NestedTensor.
        mask: Boolean mask tensor.
        value: Value to fill where mask is True.

    Returns:
        NestedTensor: A new NestedTensor with masked positions filled.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> mask = NestedTensor(torch.tensor([True, False]), torch.tensor([False, True, False]))
        >>> out = torch.masked_fill(nt, mask, 0.0)
        >>> ref = NestedTensor(torch.masked_fill(nt[0], mask[0], 0.0), torch.masked_fill(nt[1], mask[1], 0.0))
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if isinstance(mask, Tensor) and mask.shape == input.shape:
        mask = input.nested_like(mask, strict=False)
    if isinstance(mask, NestedTensor):
        if len(input) != len(mask):
            raise ValueError(
                "NestedTensor batch length mismatch between input and mask: " f"input={len(input)}, mask={len(mask)}"
            )
        return NestedTensor(torch.masked_fill(t, m, value) for t, m in zip(input._storage, mask._storage))
    storage = []
    for t in input._storage:
        m = mask
        if not isinstance(m, Tensor):
            m = torch.as_tensor(m, dtype=torch.bool, device=t.device)
        else:
            m = m.to(device=t.device)
        storage.append(torch.masked_fill(t, m, value))
    return NestedTensor(storage, **input._meta())


@NestedTensorFuncRegistry.implement(torch.masked_scatter)
def masked_scatter(input: NestedTensor, mask, source):
    r"""
    Copies elements from source into input where mask is True.
    See also [torch.masked_scatter][].

    Args:
        input: The input NestedTensor.
        mask: Boolean mask tensor.
        source: Tensor whose elements are scattered into input.

    Returns:
        NestedTensor: A new NestedTensor with scattered values.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> a = torch.tensor([1.0, 2.0])
        >>> b = torch.tensor([3.0, 4.0, 5.0])
        >>> nt = NestedTensor(a, b)
        >>> mask = NestedTensor(torch.tensor([True, False]), torch.tensor([False, True, False]))
        >>> src = NestedTensor(torch.tensor([9.0]), torch.tensor([8.0]))
        >>> out = torch.masked_scatter(nt, mask, src)
        >>> ref = NestedTensor(
        ...     torch.masked_scatter(a, mask[0], src[0]),
        ...     torch.masked_scatter(b, mask[1], src[1]),
        ... )
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if isinstance(mask, Tensor) and mask.shape == input.shape:
        mask = input.nested_like(mask, strict=False)
    if isinstance(source, Tensor) and source.shape == input.shape:
        source = input.nested_like(source, strict=False)
    if isinstance(mask, NestedTensor) and len(input) != len(mask):
        raise ValueError(
            "NestedTensor batch length mismatch between input and mask: " f"input={len(input)}, mask={len(mask)}"
        )
    if isinstance(source, NestedTensor) and len(input) != len(source):
        raise ValueError(
            "NestedTensor batch length mismatch between input and source: " f"input={len(input)}, source={len(source)}"
        )

    storage = []
    for i, t in enumerate(input._storage):
        m = mask._storage[i] if isinstance(mask, NestedTensor) else mask
        s = source._storage[i] if isinstance(source, NestedTensor) else source
        if isinstance(m, Tensor) and m.device != t.device:
            m = m.to(device=t.device)
        if isinstance(s, Tensor) and s.device != t.device:
            s = s.to(device=t.device)
        storage.append(torch.masked_scatter(t, m, s))
    return NestedTensor(storage, **input._meta())


@NestedTensorFuncRegistry.implement(torch.masked_select)
def masked_select(input: NestedTensor, mask):
    r"""
    Returns a new 1-D tensor which indexes the `input` tensor according to the boolean mask `mask` which is
    a `BoolTensor`.
    See also [torch.masked_select][].

    Args:
        input: The input NestedTensor.
        mask: Boolean mask tensor selecting elements.

    Returns:
        NestedTensor: A new NestedTensor with selected elements.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> a = torch.tensor([1.0, 0.0])
        >>> b = torch.tensor([2.0, 3.0, 4.0])
        >>> nt = NestedTensor(a, b)
        >>> mask = NestedTensor(torch.tensor([True, False]), torch.tensor([True, False, True]))
        >>> out = torch.masked_select(nt, mask)
        >>> ref = NestedTensor(torch.masked_select(a, mask[0]), torch.masked_select(b, mask[1]))
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if isinstance(mask, Tensor) and mask.shape == input.shape:
        mask = input.nested_like(mask, strict=False)
    if isinstance(mask, NestedTensor):
        if len(input) != len(mask):
            raise ValueError(
                "NestedTensor batch length mismatch between input and mask: " f"input={len(input)}, mask={len(mask)}"
            )
        return NestedTensor(torch.masked_select(t, m) for t, m in zip(input._storage, mask._storage))
    storage = []
    for t in input._storage:
        m = mask
        if not isinstance(m, Tensor):
            m = torch.as_tensor(m, dtype=torch.bool, device=t.device)
        else:
            m = m.to(device=t.device)
        storage.append(torch.masked_select(t, m))
    return NestedTensor(storage, **input._meta())


@NestedTensorFuncRegistry.implement(torch.nonzero)
def nonzero(input: NestedTensor, *, out=None, as_tuple: bool = False):
    r"""
    Note: [torch.nonzero][] (default) returns a 2-D tensor where each row is the index for a nonzero value.
    See also [torch.nonzero][].

    Args:
        input: The input NestedTensor.
        out: Not supported for NestedTensor.
        as_tuple: If True, returns a tuple of 1-D tensors per dimension.

    Returns:
        NestedTensor | tuple[NestedTensor, ...]: Indices of nonzero elements.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> a = torch.tensor([0, 1])
        >>> b = torch.tensor([2, 0, 3])
        >>> nt = NestedTensor(a, b)
        >>> out = torch.nonzero(nt)
        >>> ref = NestedTensor(torch.nonzero(a, as_tuple=False), torch.nonzero(b, as_tuple=False))
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if out is not None:
        raise NotImplementedError("torch.nonzero(..., out=...) is not supported for NestedTensor.")

    if not input._storage:
        if as_tuple:
            return ()
        return NestedTensor([], **input._meta(include_dtype=True))

    if not as_tuple:
        return NestedTensor(torch.nonzero(t, as_tuple=False) for t in input._storage)

    ndims = {t.dim() for t in input._storage}
    if len(ndims) != 1:
        raise NotImplementedError("torch.nonzero(as_tuple=True) requires all tensors in NestedTensor to share ndim.")
    (ndim,) = ndims
    if ndim == 0:
        return ()
    per_dim: list[list[Tensor]] = [[] for _ in range(ndim)]
    for t in input._storage:
        indices = torch.nonzero(t, as_tuple=True)
        for dim, idx in enumerate(indices):
            per_dim[dim].append(idx)
    return tuple(NestedTensor(per_dim[dim], **input._meta()) for dim in range(ndim))


def _scatter_impl(input, dim, index, src, apply_fn, name="scatter"):
    from .nested_tensor import NestedTensor

    dim_adj = _translate_non_batch_dim(input, dim, name=name)
    if isinstance(index, Tensor) and input.shape == index.shape:
        index = input.nested_like(index, strict=False)
    if isinstance(src, Tensor) and input.shape == src.shape:
        src = input.nested_like(src, strict=False)
    if isinstance(index, NestedTensor):
        indices = index._storage
        if len(input) != len(index):
            raise ValueError(
                "NestedTensor batch length mismatch between input and index: " f"input={len(input)}, index={len(index)}"
            )
    else:
        indices = tuple(index for _ in input._storage)
    if isinstance(src, NestedTensor):
        srcs = src._storage
        if len(input) != len(src):
            raise ValueError(
                "NestedTensor batch length mismatch between input and src: " f"input={len(input)}, src={len(src)}"
            )
    else:
        srcs = tuple(src for _ in input._storage)
    return NestedTensor(apply_fn(t, dim_adj, idx, s) for t, idx, s in zip(input._storage, indices, srcs))


@NestedTensorFuncRegistry.implement(torch.scatter)
def scatter(input: NestedTensor, dim: int, index, src):
    r"""
    Out-of-place version of [torch.Tensor.scatter_][]
    See also [torch.scatter][].

    Args:
        input: The input NestedTensor.
        dim: Dimension along which to scatter.
        index: Indices of elements to scatter.
        src: Source values to scatter into the input.

    Returns:
        NestedTensor: A new NestedTensor with scattered values.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> index = torch.tensor([[0, 1], [1, 0]])
        >>> out = torch.scatter(nt, 1, index, 0.0)
        >>> ref = torch.scatter(nt.tensor, 1, index, 0.0)
        >>> torch.equal(out, ref)
        True
    """
    return _scatter_impl(input, dim, index, src, torch.scatter, name="scatter")


@NestedTensorFuncRegistry.implement(torch.scatter_add)
def scatter_add(input: NestedTensor, dim: int, index, src):
    r"""
    Out-of-place version of [torch.Tensor.scatter_add_][]
    See also [torch.scatter_add][].

    Args:
        input: The input NestedTensor.
        dim: Dimension along which to scatter.
        index: Indices of elements to scatter.
        src: Source values to add into the input.

    Returns:
        NestedTensor: A new NestedTensor with additively scattered values.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> index = torch.tensor([[0, 1], [1, 0]])
        >>> src = torch.ones_like(nt.tensor)
        >>> out = torch.scatter_add(nt, 1, index, src)
        >>> ref = torch.scatter_add(nt.tensor, 1, index, src)
        >>> torch.equal(out, ref)
        True
    """
    return _scatter_impl(input, dim, index, src, torch.scatter_add, name="scatter_add")


if hasattr(torch, "scatter_reduce"):

    @NestedTensorFuncRegistry.implement(torch.scatter_reduce)
    def scatter_reduce(input: NestedTensor, dim: int, index, src, reduce: str, *, include_self: bool = True):
        r"""
        Out-of-place version of [torch.Tensor.scatter_reduce_][]
        See also [torch.scatter_reduce][].

        Args:
            input: The input NestedTensor.
            dim: Dimension along which to scatter.
            index: Indices of elements to scatter.
            src: Source values to scatter into the input.
            reduce: Reduction operation (`'sum'`, `'prod'`, `'mean'`, `'amax'`, `'amin'`).
            include_self: Whether to include existing values in the reduction.

        Returns:
            NestedTensor: A new NestedTensor with scatter-reduced values.

        Examples:
            >>> import torch
            >>> from danling.tensors import NestedTensor
            >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
            >>> index = torch.tensor([[0, 1], [1, 0]])
            >>> src = torch.ones_like(nt.tensor)
            >>> out = torch.scatter_reduce(nt, 1, index, src, reduce='sum')
            >>> ref = torch.scatter_reduce(nt.tensor, 1, index, src, reduce='sum')
            >>> torch.equal(out, ref)
            True
        """
        from functools import partial

        return _scatter_impl(
            input,
            dim,
            index,
            src,
            partial(torch.scatter_reduce, reduce=reduce, include_self=include_self),
            name="scatter_reduce",
        )


@NestedTensorFuncRegistry.implement(torch.take)
def take(input: NestedTensor, index, *, out=None):
    r"""
    Returns a new tensor with the elements of `input` at the given indices.
    See also [torch.take][].

    Args:
        input: The input NestedTensor.
        index: Indices into the flattened tensor.
        out: Not supported for NestedTensor.

    Returns:
        Tensor | NestedTensor: Elements at the given indices.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1, 2]), torch.tensor([3, 4]))
        >>> index = torch.tensor([0, 3])
        >>> torch.equal(torch.take(nt, index), torch.take(nt.tensor, index))
        True
    """
    from .nested_tensor import NestedTensor

    if out is not None:
        raise NotImplementedError("torch.take(..., out=...) is not supported for NestedTensor.")

    if isinstance(index, NestedTensor):
        if len(input) != len(index):
            raise ValueError(
                "NestedTensor batch length mismatch between input and index: " f"input={len(input)}, index={len(index)}"
            )
        return NestedTensor(torch.take(t.reshape(-1), i) for t, i in zip(input._storage, index._storage))

    flat = torch.cat([t.reshape(-1) for t in input._storage]) if input._storage else input.tensor.reshape(-1)
    return torch.take(flat, index)


if hasattr(torch, "take_along_dim"):

    @NestedTensorFuncRegistry.implement(torch.take_along_dim)
    def take_along_dim(input: NestedTensor, indices, dim=None):
        r"""
        Selects values from `input` at the 1-dimensional indices from `indices` along the given `dim`.
        See also [torch.take_along_dim][].

        Args:
            input: The input NestedTensor.
            indices: Indices to take along the given dimension.
            dim: Dimension to select along. If None, operates on flattened tensors.

        Returns:
            NestedTensor: A new NestedTensor with selected values.

        Examples:
            >>> import torch
            >>> from danling.tensors import NestedTensor
            >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
            >>> indices = torch.tensor([[0, 1], [1, 0]])
            >>> out = torch.take_along_dim(nt, indices, dim=1)
            >>> ref = torch.take_along_dim(nt.tensor, indices, dim=1)
            >>> torch.equal(out, ref)
            True
        """
        from .nested_tensor import NestedTensor

        if dim is None:
            if isinstance(indices, Tensor) and indices.shape == input.shape:
                indices = input.nested_like(indices, strict=False)
            if isinstance(indices, NestedTensor):
                if len(input) != len(indices):
                    raise ValueError(
                        "NestedTensor batch length mismatch between input and indices: "
                        f"input={len(input)}, indices={len(indices)}"
                    )
                return NestedTensor(
                    torch.take_along_dim(t, i, dim=None) for t, i in zip(input._storage, indices._storage)
                )
            indices_device = indices if indices.device == input.device else indices.to(device=input.device)
            return NestedTensor(torch.take_along_dim(t, indices_device, dim=None) for t in input._storage)

        dim_adj = _translate_non_batch_dim(input, dim, name="take_along_dim")
        if isinstance(indices, Tensor) and indices.shape == input.shape:
            indices = input.nested_like(indices, strict=False)
        if isinstance(indices, NestedTensor):
            if len(input) != len(indices):
                raise ValueError(
                    "NestedTensor batch length mismatch between input and indices: "
                    f"input={len(input)}, indices={len(indices)}"
                )
            return NestedTensor(
                torch.take_along_dim(t, i, dim=dim_adj) for t, i in zip(input._storage, indices._storage)
            )
        indices_device = indices if indices.device == input.device else indices.to(device=input.device)
        return NestedTensor(torch.take_along_dim(t, indices_device, dim=dim_adj) for t in input._storage)


# Linear Algebra
# NOTE: The common aten-level cases (NT * plain Tensor, matching offsets) are
# handled by aten_functions.py (aten.mm, aten.bmm, aten.addmm) operating directly on
# _values.  These torch-level handlers cover mixed-type cases and mismatched offsets.


@NestedTensorFuncRegistry.implement(torch.bmm)
def bmm(input, mat2, *, out=None):
    r"""
    Performs a batch matrix-matrix product of matrices stored in `input` and `mat2`.
    See also [torch.bmm][].

    Args:
        input: The first batch of matrices.
        mat2: The second batch of matrices.
        out: Not supported for NestedTensor.

    Returns:
        NestedTensor: The batch matrix-matrix product.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> a = torch.arange(8.0).reshape(1, 2, 4)
        >>> b = torch.arange(8.0).reshape(1, 4, 2)
        >>> nt_a = NestedTensor(a, a)
        >>> nt_b = NestedTensor(b, b)
        >>> out = torch.bmm(nt_a, nt_b)
        >>> ref = NestedTensor(torch.bmm(a, b), torch.bmm(a, b))
        >>> torch.allclose(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if out is not None:
        raise NotImplementedError("torch.bmm(..., out=...) is not supported for NestedTensor.")

    if isinstance(input, NestedTensor) and isinstance(mat2, NestedTensor):
        if len(input) != len(mat2):
            raise ValueError(
                "NestedTensor batch length mismatch between input and mat2: " f"input={len(input)}, mat2={len(mat2)}"
            )
        return NestedTensor((torch.bmm(x, y) for x, y in zip(input, mat2)), **input._meta())

    if isinstance(input, NestedTensor):
        return NestedTensor((torch.bmm(t, mat2) for t in input), **input._meta())

    if isinstance(mat2, NestedTensor):
        return NestedTensor((torch.bmm(input, t) for t in mat2), **mat2._meta())

    return torch.bmm(input, mat2)


@NestedTensorFuncRegistry.implement(torch.matmul)
def matmul(input, other, *, out=None):
    r"""
    Matrix product of two tensors.
    See also [torch.matmul][].

    Args:
        input: The first tensor to multiply.
        other: The second tensor to multiply.
        out: Not supported for NestedTensor.

    Returns:
        NestedTensor: The matrix product.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> other = torch.eye(2)
        >>> torch.allclose(torch.matmul(nt, other), torch.matmul(nt.tensor, other))
        True
    """
    from .nested_tensor import NestedTensor

    if out is not None:
        raise NotImplementedError("torch.matmul(..., out=...) is not supported for NestedTensor.")

    if isinstance(input, NestedTensor) and isinstance(other, NestedTensor):
        if len(input) != len(other):
            raise ValueError(
                "NestedTensor batch length mismatch between input and other: " f"input={len(input)}, other={len(other)}"
            )
        return NestedTensor((torch.matmul(x, y) for x, y in zip(input, other)), **input._meta())

    if isinstance(input, NestedTensor):
        if isinstance(other, Tensor) and input.shape == other.shape:
            other = input.nested_like(other)
            if len(input) != len(other):
                raise ValueError(
                    "NestedTensor batch length mismatch between input and other: "
                    f"input={len(input)}, other={len(other)}"
                )
            return NestedTensor((torch.matmul(x, y) for x, y in zip(input, other)), **input._meta())
        return NestedTensor((torch.matmul(t, other) for t in input), **input._meta())

    if isinstance(other, NestedTensor):
        if isinstance(input, Tensor) and other.shape == input.shape:
            input = other.nested_like(input)
            if len(input) != len(other):
                raise ValueError(
                    "NestedTensor batch length mismatch between input and other: "
                    f"input={len(input)}, other={len(other)}"
                )
            return NestedTensor((torch.matmul(x, y) for x, y in zip(input, other)), **other._meta())
        return NestedTensor((torch.matmul(input, t) for t in other), **other._meta())

    return torch.matmul(input, other)


@NestedTensorFuncRegistry.implement(torch.mm)
def mm(input, mat2, *, out=None):
    r"""
    Performs a matrix multiplication of the matrices `input` and `mat2`.
    See also [torch.mm][].

    Args:
        input: The first matrix.
        mat2: The second matrix.
        out: Not supported for NestedTensor.

    Returns:
        NestedTensor: The matrix multiplication result.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        >>> nt = NestedTensor(a, b)
        >>> other = torch.eye(2)
        >>> out = torch.mm(nt, other)
        >>> ref = NestedTensor(torch.mm(a, other), torch.mm(b, other))
        >>> torch.allclose(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if out is not None:
        raise NotImplementedError("torch.mm(..., out=...) is not supported for NestedTensor.")

    if isinstance(input, NestedTensor) and isinstance(mat2, NestedTensor):
        if len(input) != len(mat2):
            raise ValueError(
                "NestedTensor batch length mismatch between input and mat2: " f"input={len(input)}, mat2={len(mat2)}"
            )
        return NestedTensor((torch.mm(x, y) for x, y in zip(input, mat2)), **input._meta())

    if isinstance(input, NestedTensor):
        return NestedTensor((torch.mm(t, mat2) for t in input), **input._meta())

    if isinstance(mat2, NestedTensor):
        return NestedTensor((torch.mm(input, t) for t in mat2), **mat2._meta())

    return torch.mm(input, mat2)


# Reductions


# Table-driven: simple reductions that delegate entirely to _reduce
_SIMPLE_REDUCE_OPS = [
    (torch.all, {"fill_value": True}),
    (torch.any, {"fill_value": False}),
    (torch.sum, {"fill_value": 0}),
    (torch.prod, {}),
]

for _op, _extra_kwargs in _SIMPLE_REDUCE_OPS:

    @NestedTensorFuncRegistry.implement(_op)
    def _reduce_impl(input, dim=None, keepdim=False, *, _fn=_op, _extra=_extra_kwargs, **kwargs):
        return _reduce(input, _fn, dim, keepdim, **kwargs, **_extra)


@NestedTensorFuncRegistry.implement(torch.sum)
def sum(input: NestedTensor, dim: int | Sequence[int] | None = None, keepdim: bool = False, *, dtype=None):
    r"""Compute sums via aten fastpaths for global and single-dim reductions."""
    if dim is None:
        return _reduce_none(input, torch.sum, dtype=dtype, keepdim=keepdim)
    if isinstance(dim, (list, tuple)):
        if len(dim) == 1:
            dim = dim[0]
        else:
            dims = tuple(_normalize_dim(d, input.dim()) for d in dim)
            # Preserve the previous masked path when reducing batch with other dims.
            if _get_batch_dim(input) in dims:
                return _reduce(input, torch.sum, dim, keepdim, dtype=dtype, fill_value=0)
            return torch.ops.aten.sum.dim_IntList(input, list(dim), keepdim, dtype=dtype)
    return torch.ops.aten.sum.dim_IntList(input, [dim], keepdim, dtype=dtype)


@NestedTensorFuncRegistry.implement(torch.amax)
def amax(input: NestedTensor, dim: int | Sequence[int] | None = None, keepdim: bool = False):
    r"""
    Returns the maximum value of each slice of the `input` tensor in the given dimension(s) `dim`.
    See also [torch.amax][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce. If None, reduces over all elements.
        keepdim: Whether to retain the reduced dimension.

    Returns:
        Tensor | NestedTensor: The maximum values.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.amax(nt, dim=1), torch.amax(nt.tensor, dim=1))
        True
    """
    if dim is None:
        return _reduce_none(input, torch.amax, keepdim=keepdim)
    if isinstance(dim, (list, tuple)):
        if len(dim) == 1:
            dim = dim[0]
        else:
            return _reduce(input, torch.amax, dim, keepdim)
    return torch.ops.aten.amax.default(input, [dim], keepdim)


@NestedTensorFuncRegistry.implement(torch.amin)
def amin(input: NestedTensor, dim: int | Sequence[int] | None = None, keepdim: bool = False):
    r"""
    Returns the minimum value of each slice of the `input` tensor in the given dimension(s) `dim`.
    See also [torch.amin][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce. If None, reduces over all elements.
        keepdim: Whether to retain the reduced dimension.

    Returns:
        Tensor | NestedTensor: The minimum values.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.amin(nt, dim=1), torch.amin(nt.tensor, dim=1))
        True
    """
    if dim is None:
        return _reduce_none(input, torch.amin, keepdim=keepdim)
    if isinstance(dim, (list, tuple)):
        if len(dim) == 1:
            dim = dim[0]
        else:
            return _reduce(input, torch.amin, dim, keepdim)
    return torch.ops.aten.amin.default(input, [dim], keepdim)


@NestedTensorFuncRegistry.implement(torch.aminmax)
def aminmax(input: NestedTensor, *, dim: int | None = None, keepdim: bool = False):
    r"""
    Computes the minimum and maximum values of the `input` tensor.
    See also [torch.aminmax][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce. If None, reduces over all elements.
        keepdim: Whether to retain the reduced dimension.

    Returns:
        tuple[Tensor, Tensor]: A tuple of (min values, max values).

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_min, out_max = torch.aminmax(nt, dim=1)
        >>> ref_min, ref_max = torch.aminmax(nt.tensor, dim=1)
        >>> torch.equal(out_min, ref_min) and torch.equal(out_max, ref_max)
        True
    """
    if dim is None:
        return _reduce_none(input, torch.amin, keepdim=keepdim), _reduce_none(input, torch.amax, keepdim=keepdim)
    return _reduce_dim_pair(input, torch.aminmax, dim, keepdim)


@NestedTensorFuncRegistry.implement(torch.count_nonzero)
def count_nonzero(input: NestedTensor, dim: int | Sequence[int] | None = None):
    r"""
    Counts the number of non-zero values in the tensor `input` along the given `dim`.
    See also [torch.count_nonzero][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce. If None, counts over all elements.

    Returns:
        Tensor | NestedTensor: The count of nonzero elements.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([0, 1]), torch.tensor([2, 0]))
        >>> torch.equal(torch.count_nonzero(nt), torch.count_nonzero(nt.tensor))
        True
    """
    if dim is None:
        return torch.count_nonzero(input._values)
    if isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)
    batch_dim = _get_batch_dim(input)
    for d in dims:
        if _normalize_dim(d, input.dim()) == batch_dim:
            raise ValueError("count_nonzero along the batch dimension is not supported for NestedTensor.")
    return torch.ops.aten.count_nonzero.dim_IntList(input, dims)


@NestedTensorFuncRegistry.implement(torch.dist)
def dist(input: NestedTensor, other: NestedTensor | Tensor, p=2):
    r"""
    Returns the p-norm of (`input` - `other`) The shapes of `input` and `other` must be
    [broadcastable][].
    See also [torch.dist][].

    Args:
        input: The first NestedTensor.
        other: The second tensor to compute distance against.
        p: The order of the norm.

    Returns:
        Tensor: The p-norm distance per batch element.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> a = torch.tensor([1.0, 2.0])
        >>> b = torch.tensor([3.0, 4.0])
        >>> nt = NestedTensor(a, b)
        >>> out = torch.dist(nt, nt)
        >>> ref = torch.stack([torch.dist(a, a), torch.dist(b, b)])
        >>> torch.allclose(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if not isinstance(input, NestedTensor):
        input = other.nested_like(input)
    elif not isinstance(other, NestedTensor):
        other = input.nested_like(other)
    if len(input) != len(other):
        raise ValueError(
            "NestedTensor batch length mismatch between input and other: " f"input={len(input)}, other={len(other)}"
        )
    if not input._storage:
        return torch.empty((0,), device=input.device)
    return torch.stack([torch.dist(x, y, p=p) for x, y in zip(input._storage, other._storage)])


@NestedTensorFuncRegistry.implement(torch.logsumexp)
def logsumexp(input: NestedTensor, dim: int | Sequence[int], keepdim: bool = False):
    r"""
    Returns the log of summed exponentials of each row of the `input` tensor in the given dimension `dim`.
    The computation is numerically stabilized.
    See also [torch.logsumexp][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce.
        keepdim: Whether to retain the reduced dimension.

    Returns:
        Tensor | NestedTensor: The log-sum-exp result.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.logsumexp(nt, dim=1), torch.logsumexp(nt.tensor, dim=1))
        True
    """
    dims = [dim] if isinstance(dim, int) else list(dim)
    return torch.ops.aten.logsumexp.default(input, dims, keepdim)


@NestedTensorFuncRegistry.implement(torch.max)
def max(input: NestedTensor, dim: int | None = None, keepdim: bool = False):
    r"""
    Returns the maximum value of all elements in the ``input`` tensor.
    See also [torch.max][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce. If None, returns the global maximum.
        keepdim: Whether to retain the reduced dimension.

    Returns:
        Tensor | namedtuple: The maximum value, or (values, indices) when dim is given.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.max(nt, dim=1)
        >>> ref = torch.max(nt.tensor, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    if dim is None:
        return _reduce_none(input, torch.max)
    values, indices = _reduce_dim_pair(input, torch.max, dim, keepdim)
    return torch.return_types.max((values, indices))


@NestedTensorFuncRegistry.implement(torch.mean)
def mean(
    input,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
):
    r"""
    Note: If the `input` tensor is empty, ``torch.mean()`` returns ``nan``.
    See also [torch.mean][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce. If None, reduces over all elements.
        keepdim: Whether to retain the reduced dimension.
        dtype: Desired data type of the result.

    Returns:
        Tensor | NestedTensor: The mean values.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.mean(nt, dim=1), torch.mean(nt.tensor, dim=1))
        True
    """
    if dim is None:
        return _reduce_none(input, torch.mean, dtype=dtype, keepdim=keepdim)
    if isinstance(dim, int):
        return torch.ops.aten.mean.dim(input, [dim], keepdim, dtype=dtype)
    if isinstance(dim, (list, tuple)):
        if len(dim) == 1:
            return torch.ops.aten.mean.dim(input, [dim[0]], keepdim, dtype=dtype)
        else:
            dims = tuple(_normalize_dim(d, input.dim()) for d in dim)
            tensor, mask = input.tensor_mask
            # `mask_value` is the padding-position fill value in `mask`.
            valid = mask if not input.mask_value else ~mask
            while valid.dim() < tensor.dim():
                valid = valid.unsqueeze(-1)
            data = tensor.to(dtype=dtype) if dtype is not None else tensor
            data = torch.where(valid, data, torch.zeros_like(data))
            summed = torch.sum(data, dim=dims, keepdim=keepdim)
            count = torch.sum(valid, dim=dims, keepdim=keepdim)
            return summed / count.to(dtype=summed.dtype)
    return _reduce_dim(input, torch.mean, dim, keepdim, dtype=dtype)


@NestedTensorFuncRegistry.implement(torch.min)
def min(input: NestedTensor, dim: int | None = None, keepdim: bool = False):
    r"""
    Returns the minimum value of all elements in the `input` tensor.
    See also [torch.min][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce. If None, returns the global minimum.
        keepdim: Whether to retain the reduced dimension.

    Returns:
        Tensor | namedtuple: The minimum value, or (values, indices) when dim is given.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.min(nt, dim=1)
        >>> ref = torch.min(nt.tensor, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    if dim is None:
        return _reduce_none(input, torch.min)
    values, indices = _reduce_dim_pair(input, torch.min, dim, keepdim)
    return torch.return_types.min((values, indices))


@NestedTensorFuncRegistry.implement(torch.nanmean)
def nanmean(
    input: NestedTensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
):
    r"""
    Computes the mean of all `non-NaN` elements along the specified dimensions.
    See also [torch.nanmean][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce. If None, reduces over all elements.
        keepdim: Whether to retain the reduced dimension.
        dtype: Desired data type of the result.

    Returns:
        Tensor | NestedTensor: The NaN-ignoring mean.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.nanmean(nt, dim=1), torch.nanmean(nt.tensor, dim=1))
        True
    """
    if dim is None:
        return _reduce_none(input, torch.nanmean, dtype=dtype, keepdim=keepdim)
    dims = [dim] if isinstance(dim, int) else list(dim)
    return torch.ops.aten.nanmean.default(input, dims, keepdim, dtype=dtype)


@NestedTensorFuncRegistry.implement(torch.nansum)
def nansum(
    input: NestedTensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
):
    r"""
    Returns the sum of all elements, treating Not a Numbers (NaNs) as zero.
    See also [torch.nansum][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce. If None, reduces over all elements.
        keepdim: Whether to retain the reduced dimension.
        dtype: Desired data type of the result.

    Returns:
        Tensor | NestedTensor: The NaN-ignoring sum.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.nansum(nt, dim=1), torch.nansum(nt.tensor, dim=1))
        True
    """
    if dim is None:
        return _reduce_none(input, torch.nansum, dtype=dtype, keepdim=keepdim)
    dims = [dim] if isinstance(dim, int) else list(dim)
    return torch.ops.aten.nansum.default(input, dims, keepdim, dtype=dtype)


@NestedTensorFuncRegistry.implement(torch.numel)
def numel(input: NestedTensor) -> int:
    r"""
    Returns the total number of elements in the `input` tensor.
    See also [torch.numel][].

    Args:
        input: The input NestedTensor.

    Returns:
        int: Total number of elements across all tensors in the storage.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1, 2]), torch.tensor([3, 4]))
        >>> torch.numel(nt) == torch.numel(nt.tensor)
        True
    """
    return input.numel()


@NestedTensorFuncRegistry.implement(torch.std)
def std(
    input: NestedTensor,
    dim: int | Sequence[int] | None = None,
    *,
    correction: int = 1,
    keepdim: bool = False,
):
    r"""
    Calculates the standard deviation over the dimensions specified by `dim`.
    See also [torch.std][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce. If None, reduces over all elements.
        correction: Difference between the sample size and sample degrees of freedom.
        keepdim: Whether to retain the reduced dimension.

    Returns:
        Tensor | NestedTensor: The standard deviation.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.std(nt, dim=1), torch.std(nt.tensor, dim=1))
        True
    """
    if dim is None:
        return _reduce_none(input, torch.std, keepdim=keepdim, correction=correction)
    dims = [dim] if isinstance(dim, int) else list(dim)
    return torch.ops.aten.std.correction(input, dims, correction=correction, keepdim=keepdim)


@NestedTensorFuncRegistry.implement(torch.var)
def var(
    input: NestedTensor,
    dim: int | Sequence[int] | None = None,
    *,
    correction: int = 1,
    keepdim: bool = False,
):
    r"""
    Calculates the variance over the dimensions specified by `dim`. `dim` can be a single dimension, list of
    dimensions, or ``None`` to reduce over all dimensions.
    See also [torch.var][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce. If None, reduces over all elements.
        correction: Difference between the sample size and sample degrees of freedom.
        keepdim: Whether to retain the reduced dimension.

    Returns:
        Tensor | NestedTensor: The variance.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.var(nt, dim=1), torch.var(nt.tensor, dim=1))
        True
    """
    if dim is None:
        return _reduce_none(input, torch.var, keepdim=keepdim, correction=correction)
    dims = [dim] if isinstance(dim, int) else list(dim)
    return torch.ops.aten.var.correction(input, dims, correction=correction, keepdim=keepdim)


@NestedTensorFuncRegistry.implement(torch.var_mean)
def var_mean(
    input: NestedTensor,
    dim: int | Sequence[int] | None = None,
    *,
    correction: int = 1,
    keepdim: bool = False,
):
    r"""
    Calculates the variance and mean over the dimensions specified by `dim`.
    See also [torch.var_mean][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce. If None, reduces over all elements.
        correction: Difference between the sample size and sample degrees of freedom.
        keepdim: Whether to retain the reduced dimension.

    Returns:
        tuple[Tensor | NestedTensor, Tensor | NestedTensor]: A tuple of (variance, mean).

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out = torch.var_mean(nt, dim=1)
        >>> ref = torch.var_mean(nt.tensor, dim=1)
        >>> torch.allclose(out[0], ref[0]) and torch.allclose(out[1], ref[1])
        True
    """
    if dim is None:
        return _reduce_none_pair(input, torch.var_mean, keepdim=keepdim, correction=correction)
    dims = [dim] if isinstance(dim, int) else list(dim)
    return torch.ops.aten.var_mean.correction(input, dims, correction=correction, keepdim=keepdim)


# Shape Manipulation


@NestedTensorFuncRegistry.implement(torch.flatten)
def flatten(input: NestedTensor, start_dim: int = 0, end_dim: int = -1):
    r"""
    Flattens `input` by reshaping it into a one-dimensional tensor. If `start_dim` or `end_dim` are
    passed, only dimensions starting with `start_dim` and ending with `end_dim` are flattened.
    See also [torch.flatten][].

    Args:
        input: The input NestedTensor.
        start_dim: First dimension to flatten.
        end_dim: Last dimension to flatten.

    Returns:
        Tensor | NestedTensor: The flattened result.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out = torch.flatten(nt, start_dim=1)
        >>> ref = torch.flatten(nt.tensor, start_dim=1)
        >>> torch.equal(out, ref)
        True
    """
    ndims = input.dim()
    start = start_dim if start_dim >= 0 else start_dim + ndims
    end = end_dim if end_dim >= 0 else end_dim + ndims
    if start < 0 or end < 0 or start >= ndims or end >= ndims:
        raise IndexError(f"start_dim and end_dim must be in range [0, {ndims}), got ({start_dim}, {end_dim})")
    if start > end:
        raise ValueError(f"start_dim must be <= end_dim, got ({start_dim}, {end_dim})")

    batch_dim = _get_batch_dim(input)
    if start <= batch_dim <= end:
        return torch.flatten(input.tensor, start_dim=start_dim, end_dim=end_dim)
    return torch.ops.aten.flatten.using_ints(input, start, end)


@NestedTensorFuncRegistry.implement(torch.flip)
def flip(input: NestedTensor, dims: Sequence[int]) -> NestedTensor:
    r"""
    Reverse the order of an n-D tensor along given axis in dims.
    See also [torch.flip][].

    Args:
        input: The input NestedTensor.
        dims: Dimensions to flip.

    Returns:
        NestedTensor: A new NestedTensor with flipped dimensions.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...     torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        ... )
        >>> out = torch.flip(nt, dims=(1,))
        >>> ref = NestedTensor(
        ...     torch.flip(nt[0], dims=(0,)),
        ...     torch.flip(nt[1], dims=(0,)),
        ... )
        >>> torch.equal(out, ref)
        True
    """
    return torch.ops.aten.flip.default(input, list(dims))


@NestedTensorFuncRegistry.implement(torch.moveaxis)
def moveaxis(input: NestedTensor, source, destination):
    r"""
    Alias for [torch.movedim][].
    See also [torch.moveaxis][].

    Args:
        input: The input NestedTensor.
        source: Original positions of the dims to move.
        destination: Destination positions for each of the original dims.

    Returns:
        NestedTensor: A new NestedTensor with moved dimensions.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...     torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        ... )
        >>> out = torch.moveaxis(nt, 1, 2)
        >>> ref = torch.moveaxis(nt.tensor, 1, 2)
        >>> torch.equal(out, ref)
        True
    """
    ndims = input.dim()
    src_axes = (source,) if isinstance(source, int) else tuple(source)
    dst_axes = (destination,) if isinstance(destination, int) else tuple(destination)
    if len(src_axes) != len(dst_axes):
        raise ValueError("moveaxis: `source` and `destination` must have the same number of dimensions")

    src_norm = [_normalize_dim(d, ndims) for d in src_axes]
    dst_norm = [_normalize_dim(d, ndims) for d in dst_axes]

    if len(set(src_norm)) != len(src_norm) or len(set(dst_norm)) != len(dst_norm):
        raise ValueError("moveaxis: duplicate dimensions are not allowed")

    order = [i for i in range(ndims) if i not in src_norm]
    for dest, src in sorted(zip(dst_norm, src_norm)):
        order.insert(dest, src)

    return input.permute(*order)


@NestedTensorFuncRegistry.implement(torch.permute)
def permute(input: NestedTensor, dims: Sequence[int]) -> NestedTensor:
    r"""
    Returns a view of the original tensor `input` with its dimensions permuted.
    See also [torch.permute][].

    Args:
        input: The input NestedTensor.
        dims: The desired ordering of dimensions.

    Returns:
        NestedTensor: A new NestedTensor with permuted dimensions.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]))  # noqa: E501
        >>> out = torch.permute(nt, (0, 2, 1))
        >>> ref = torch.permute(nt.tensor, (0, 2, 1))
        >>> torch.equal(out, ref)
        True
    """
    if len(dims) != input.dim():
        raise ValueError(f"Expected {input.dim()} dimensions, got {len(dims)}")

    dim_count = input.dim()
    normalized_dims = tuple(d if d >= 0 else d + dim_count for d in dims)
    batch_dim = _get_batch_dim(input)
    if set(normalized_dims) != set(range(dim_count)):
        raise ValueError(f"Invalid permutation dims {dims} for shape with {dim_count} dims")
    if normalized_dims[batch_dim] != batch_dim:
        raise ValueError("Permuting the batch dimension is not supported for NestedTensor.")

    return torch.ops.aten.permute.default(input, list(normalized_dims))


@NestedTensorFuncRegistry.implement(torch.reshape)
def reshape(input: NestedTensor, shape: Sequence[int]) -> NestedTensor:
    r"""
    Returns a tensor with the same data and number of elements as `input`, but with the specified shape. When
    possible, the returned tensor will be a view of `input`. Otherwise, it will be a copy. Contiguous inputs and
    inputs with compatible strides can be reshaped without copying, but you should not depend on the copying vs. viewing
    behavior.
    See also [torch.reshape][].

    Args:
        input: The input NestedTensor.
        shape: The new shape.

    Returns:
        NestedTensor: A new NestedTensor with the specified shape.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([5.0, 6.0, 7.0, 8.0]))
        >>> out = torch.reshape(nt, (2, 2, 2))
        >>> ref = torch.reshape(nt.tensor, (2, 2, 2))
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    if not input._storage:
        return NestedTensor([], **input._meta(include_dtype=True))
    view_shapes = input._view_shapes(shape)
    reshaped_tensors = [t.reshape(s) for t, s in zip(input._storage, view_shapes)]
    return NestedTensor(reshaped_tensors, **input._meta())


@NestedTensorFuncRegistry.implement(torch.roll)
def roll(input: NestedTensor, shifts, dims=None):
    r"""
    Roll the tensor `input` along the given dimension(s). Elements that are shifted beyond the last position are
    re-introduced at the first position. If `dims` is `None`, the tensor will be flattened before rolling and then
    restored to the original shape.
    See also [torch.roll][].

    Args:
        input: The input NestedTensor.
        shifts: Number of places by which elements are shifted.
        dims: Dimensions along which to roll. If None, flattens before rolling.

    Returns:
        NestedTensor: A new NestedTensor with rolled elements.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...     torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        ... )
        >>> out = torch.roll(nt, shifts=1, dims=1)
        >>> ref = NestedTensor(
        ...     torch.roll(nt[0], shifts=1, dims=0),
        ...     torch.roll(nt[1], shifts=1, dims=0),
        ... )
        >>> torch.equal(out, ref)
        True
    """
    if isinstance(shifts, int):
        shifts = [shifts]
    else:
        shifts = list(shifts)

    if dims is None:
        return torch.ops.aten.roll.default(input, shifts, [])
    if isinstance(dims, int):
        dims = [dims]
    else:
        dims = list(dims)
    return torch.ops.aten.roll.default(input, shifts, dims)


@NestedTensorFuncRegistry.implement(torch.rot90)
def rot90(input: NestedTensor, k: int = 1, dims: Sequence[int] = (0, 1)) -> NestedTensor:
    r"""
    Rotate an n-D tensor by 90 degrees in the plane specified by dims axis.
    See also [torch.rot90][].

    Args:
        input: The input NestedTensor.
        k: Number of times to rotate by 90 degrees.
        dims: Axes that define the plane of rotation (must be length 2).

    Returns:
        NestedTensor: A new NestedTensor rotated by 90*k degrees.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...     torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        ... )
        >>> out = torch.rot90(nt, k=1, dims=(1, 2))
        >>> ref = torch.rot90(nt.tensor, k=1, dims=(1, 2))
        >>> torch.equal(out, ref)
        True
    """
    if len(dims) != 2:
        raise ValueError("rot90 dims must be a sequence of two dimensions.")
    return torch.ops.aten.rot90.default(input, k, list(dims))


@NestedTensorFuncRegistry.implement(torch.squeeze)
def squeeze(input: NestedTensor, dim: int | None = None):
    r"""
    Returns a tensor with all specified dimensions of `input` of size `1` removed.
    See also [torch.squeeze][].

    Args:
        input: The input NestedTensor.
        dim: If given, only squeezes that specific dimension.

    Returns:
        NestedTensor: A new NestedTensor with size-1 dimensions removed.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0], [5.0]]))
        >>> out = torch.squeeze(nt, dim=2)
        >>> ref = torch.squeeze(nt.tensor, dim=2)
        >>> torch.equal(out, ref)
        True
    """
    if dim is None:
        return torch.ops.aten.squeeze.default(input)
    dim_norm = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim_norm <= batch_dim:
        raise ValueError("Cannot squeeze the batch dimension or dimensions before it for NestedTensor.")
    return torch.ops.aten.squeeze.dim(input, dim_norm)


@NestedTensorFuncRegistry.implement(torch.swapaxes)
def swapaxes(input: NestedTensor, axis0: int, axis1: int):
    r"""
    Alias for [torch.transpose][].
    See also [torch.swapaxes][].

    Args:
        input: The input NestedTensor.
        axis0: First axis to swap.
        axis1: Second axis to swap.

    Returns:
        NestedTensor: A new NestedTensor with swapped axes.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...     torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        ... )
        >>> out = torch.swapaxes(nt, 1, 2)
        >>> ref = torch.swapaxes(nt.tensor, 1, 2)
        >>> torch.equal(out, ref)
        True
    """
    axis0 = _normalize_dim(axis0, input.dim())
    axis1 = _normalize_dim(axis1, input.dim())
    batch_dim = _get_batch_dim(input)
    if axis0 == batch_dim or axis1 == batch_dim:
        raise ValueError("Cannot swap the batch dimension for NestedTensor.")
    return torch.ops.aten.transpose.int(input, axis0, axis1)


# swapdims is an alias for swapaxes
NestedTensorFuncRegistry[torch.swapdims] = NestedTensorFuncRegistry[torch.swapaxes]


@NestedTensorFuncRegistry.implement(torch.transpose)
def transpose(input: NestedTensor, dim0: int, dim1: int) -> NestedTensor:
    r"""
    Returns a tensor that is a transposed version of `input`.
    See also [torch.transpose][].

    Args:
        input: The input NestedTensor.
        dim0: First dimension to transpose.
        dim1: Second dimension to transpose.

    Returns:
        NestedTensor: A new NestedTensor with transposed dimensions.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(
        ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...     torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        ... )
        >>> out = torch.transpose(nt, 1, 2)
        >>> ref = torch.transpose(nt.tensor, 1, 2)
        >>> torch.equal(out, ref)
        True
    """
    ndims = input.dim()
    if dim0 < 0:
        dim0 += ndims
    if dim1 < 0:
        dim1 += ndims

    if dim0 < 0 or dim0 >= ndims or dim1 < 0 or dim1 >= ndims:
        raise IndexError(f"Dimension out of range for NestedTensor with {ndims} dims: ({dim0}, {dim1})")

    batch_dim = _get_batch_dim(input)
    if dim0 == batch_dim or dim1 == batch_dim:
        raise ValueError("Cannot transpose the batch dimension for NestedTensor.")
    return torch.ops.aten.transpose.int(input, dim0, dim1)


@NestedTensorFuncRegistry.implement(torch.unflatten)
def unflatten(input: NestedTensor, dim: int, sizes):
    r"""
    Expands a dimension of the input tensor over multiple dimensions.
    See also [torch.unflatten][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to unflatten.
        sizes: New shape for the unflattened dimension.

    Returns:
        NestedTensor: A new NestedTensor with the dimension expanded.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        >>> out = torch.unflatten(nt, 1, (1, 2))
        >>> ref = torch.unflatten(nt.tensor, 1, (1, 2))
        >>> torch.equal(out, ref)
        True
    """
    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim <= batch_dim:
        raise ValueError("unflatten at or before the batch dimension is not supported for NestedTensor.")
    return torch.ops.aten.unflatten.int(input, dim, sizes)


@NestedTensorFuncRegistry.implement(torch.unsqueeze)
def unsqueeze(input: NestedTensor, dim: int):
    r"""
    Returns a new tensor with a dimension of size one inserted at the specified position.
    See also [torch.unsqueeze][].

    Args:
        input: The input NestedTensor.
        dim: Position at which to insert the singleton dimension.

    Returns:
        NestedTensor: A new NestedTensor with an added dimension.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> out = torch.unsqueeze(nt, dim=2)
        >>> ref = torch.unsqueeze(nt.tensor, dim=2)
        >>> torch.equal(out, ref)
        True
    """
    ndims = input.dim()
    if dim < 0:
        dim += ndims + 1
    if dim < 0 or dim > ndims:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-ndims - 1}, {ndims}], but got {dim})")

    batch_dim = _get_batch_dim(input)
    if dim <= batch_dim:
        raise ValueError("Cannot unsqueeze at or before the batch dimension for NestedTensor.")
    return torch.ops.aten.unsqueeze.default(input, dim)


@NestedTensorFuncRegistry.implement(torch.where)
def where(condition, input, other):
    r"""
    Return a tensor of elements selected from either `input` or `other`, depending on `condition`.
    See also [torch.where][].

    Args:
        condition: Boolean condition tensor.
        input: Values selected where condition is True.
        other: Values selected where condition is False.

    Returns:
        NestedTensor: A new NestedTensor with elements chosen by condition.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
        >>> cond = nt > 2
        >>> out = torch.where(cond, nt, 0.0)
        >>> ref = torch.where(cond.tensor, nt.tensor, 0.0)
        >>> torch.equal(out, ref)
        True
    """
    from .nested_tensor import NestedTensor

    def to_nested(value, ref: NestedTensor, *, dtype=None) -> NestedTensor:
        if isinstance(value, NestedTensor):
            return value.to(dtype=dtype) if dtype is not None else value
        if isinstance(value, Tensor) and value.shape == ref.shape:
            return ref.nested_like(value).to(dtype=dtype)
        storage = []
        for t in ref._storage:
            vt = torch.as_tensor(value, device=t.device)
            if dtype is not None:
                vt = vt.to(dtype)
            try:
                vt = vt.expand_as(t)
            except RuntimeError:
                vt = vt.expand(t.shape)
            storage.append(vt)
        return NestedTensor(storage, **ref._meta())

    ref = next((v for v in (input, other, condition) if isinstance(v, NestedTensor)), None)
    if ref is None:
        return torch.where(condition, input, other)

    cond_nt = to_nested(condition, ref, dtype=torch.bool)
    input_nt = to_nested(input, ref)
    other_nt = to_nested(other, ref)
    for nested in (cond_nt, input_nt, other_nt):
        if len(nested) != len(ref):
            raise ValueError(
                "NestedTensor batch length mismatch between ref and nested: " f"ref={len(ref)}, nested={len(nested)}"
            )
    return NestedTensor(
        (torch.where(c, x, y) for c, x, y in zip(cond_nt._storage, input_nt._storage, other_nt._storage)),
        **ref._meta(),
    )


# Softmax


@NestedTensorFuncRegistry.implement(torch.log_softmax)
def log_softmax(input: NestedTensor, dim: int, dtype: torch.dtype | None = None) -> NestedTensor:
    r"""
    Applies log-softmax over the specified dimension. See also [torch.log_softmax][].

    Args:
        input: The input NestedTensor.
        dim: Dimension along which log-softmax is computed.
        dtype: Desired data type of the result.

    Returns:
        NestedTensor: The log-softmax result.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))
        >>> out = torch.log_softmax(nt, dim=-1)
        >>> ref = torch.log_softmax(nt.tensor, dim=-1)
        >>> torch.allclose(out, ref)
        True
    """
    source = input if dtype is None else torch.ops.aten._to_copy.default(input, dtype=dtype)
    return torch.ops.aten._log_softmax.default(source, dim, False)


@NestedTensorFuncRegistry.implement(torch.softmax)
def softmax(input: NestedTensor, dim: int, dtype: torch.dtype | None = None) -> NestedTensor:
    r"""
    Applies a softmax function over the specified dimension. See also [torch.softmax][].

    Args:
        input: The input NestedTensor.
        dim: Dimension along which softmax is computed.
        dtype: Desired data type of the result.

    Returns:
        NestedTensor: The softmax result.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))
        >>> out = torch.softmax(nt, dim=-1)
        >>> ref = torch.softmax(nt.tensor, dim=-1)
        >>> torch.allclose(out, ref)
        True
    """
    source = input if dtype is None else torch.ops.aten._to_copy.default(input, dtype=dtype)
    return torch.ops.aten._softmax.default(source, dim, False)


# Sorting & Selection


@NestedTensorFuncRegistry.implement(torch.argmax)
def argmax(input: NestedTensor, dim: int | None = None, keepdim: bool = False):
    r"""
    Returns the indices of the maximum value of all elements in the `input` tensor.
    See also [torch.argmax][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce. If None, returns indices into the flattened tensor.
        keepdim: Whether to retain the reduced dimension.

    Returns:
        Tensor | NestedTensor: Indices of the maximum values.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.equal(torch.argmax(nt, dim=1), torch.argmax(nt.tensor, dim=1))
        True
    """
    if dim is None:
        return torch.stack([t.reshape(-1).argmax() for t in input._storage])
    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        output = torch.stack([t.reshape(-1).argmax() for t in input._storage])
        if keepdim:
            return output.unsqueeze(0 if input.batch_first else 1)
        return output
    dim_adj = _translate_dim(input, dim)
    ret = [t.argmax(dim=dim_adj, keepdim=keepdim) for t in input._storage]
    return _try_stack(ret, input)


@NestedTensorFuncRegistry.implement(torch.argmin)
def argmin(input: NestedTensor, dim: int | None = None, keepdim: bool = False):
    r"""
    Returns the indices of the minimum value(s) of the flattened tensor or along a dimension This is the second value
    returned by [torch.min][]. See its documentation for the exact semantics of this method.
    See also [torch.argmin][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce. If None, returns indices into the flattened tensor.
        keepdim: Whether to retain the reduced dimension.

    Returns:
        Tensor | NestedTensor: Indices of the minimum values.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.equal(torch.argmin(nt, dim=1), torch.argmin(nt.tensor, dim=1))
        True
    """
    if dim is None:
        return torch.stack([t.reshape(-1).argmin() for t in input._storage])
    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        output = torch.stack([t.reshape(-1).argmin() for t in input._storage])
        if keepdim:
            return output.unsqueeze(0 if input.batch_first else 1)
        return output
    dim_adj = _translate_dim(input, dim)
    ret = [t.argmin(dim=dim_adj, keepdim=keepdim) for t in input._storage]
    return _try_stack(ret, input)


@NestedTensorFuncRegistry.implement(torch.argsort)
def argsort(input: NestedTensor, dim: int = -1, descending: bool = False, stable: bool | None = None):
    r"""
    Returns the indices that sort a tensor along a given dimension in ascending order by value.
    See also [torch.argsort][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to sort along.
        descending: If True, sorts in descending order.
        stable: If True, preserves the relative order of equal elements.

    Returns:
        NestedTensor: Indices that would sort the tensor.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.equal(torch.argsort(nt, dim=1), torch.argsort(nt.tensor, dim=1))
        True
    """
    if stable is None:
        return torch.ops.aten.argsort.default(input, dim, descending)
    return torch.ops.aten.argsort.stable(input, stable=stable, dim=dim, descending=descending)


@NestedTensorFuncRegistry.implement(torch.kthvalue)
def kthvalue(input: NestedTensor, k: int, dim: int = -1, keepdim: bool = False):
    r"""
    Returns a namedtuple ``(values, indices)`` where ``values`` is the `k` th smallest element of each row of the
    `input` tensor in the given dimension `dim`. And ``indices`` is the index location of each element
    found.
    See also [torch.kthvalue][].

    Args:
        input: The input NestedTensor.
        k: The k-th smallest element to find.
        dim: Dimension to reduce along.
        keepdim: Whether to retain the reduced dimension.

    Returns:
        namedtuple: A (values, indices) namedtuple.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.kthvalue(nt, k=1, dim=1)
        >>> ref = torch.kthvalue(nt.tensor, k=1, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    return _reduce_dim_pair(input, torch.kthvalue, dim, keepdim, k=k)


@NestedTensorFuncRegistry.implement(torch.median)
def median(input: NestedTensor, dim: int | None = None, keepdim: bool = False):
    r"""
    Returns the median of the values in `input`.
    See also [torch.median][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce. If None, computes the global median.
        keepdim: Whether to retain the reduced dimension.

    Returns:
        Tensor | namedtuple: The median value, or (values, indices) when dim is given.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.median(nt, dim=1)
        >>> ref = torch.median(nt.tensor, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    if dim is None:
        return _reduce_none(input, torch.median)
    return _reduce_dim_pair(input, torch.median, dim, keepdim)


@NestedTensorFuncRegistry.implement(torch.mode)
def mode(input: NestedTensor, dim: int = -1, keepdim: bool = False):
    r"""
    Returns a namedtuple ``(values, indices)`` where ``values`` is the mode value of each row of the `input`
    tensor in the given dimension `dim`, i.e. a value which appears most often in that row, and ``indices`` is the
    index location of each mode value found.
    See also [torch.mode][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce along.
        keepdim: Whether to retain the reduced dimension.

    Returns:
        namedtuple: A (values, indices) namedtuple.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.mode(nt, dim=1)
        >>> ref = torch.mode(nt.tensor, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    return _reduce_dim_pair(input, torch.mode, dim, keepdim)


@NestedTensorFuncRegistry.implement(torch.nanmedian)
def nanmedian(input: NestedTensor, dim: int | None = None, keepdim: bool = False):
    r"""
    Returns the median of the values in `input`, ignoring ``NaN`` values.
    See also [torch.nanmedian][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to reduce. If None, computes the global median.
        keepdim: Whether to retain the reduced dimension.

    Returns:
        Tensor | namedtuple: The median value, or (values, indices) when dim is given.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.nanmedian(nt, dim=1)
        >>> ref = torch.nanmedian(nt.tensor, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    if dim is None:
        return _reduce_none(input, torch.nanmedian)
    return _reduce_dim_pair(input, torch.nanmedian, dim, keepdim)


def _quantile(input: NestedTensor, op, q, dim, keepdim, interpolation):
    r"""Shared implementation for quantile and nanquantile."""
    if dim is None:
        flat = torch.cat([t.reshape(-1) for t in input._storage]) if input._storage else input.tensor.reshape(-1)
        output = op(flat, q, dim=0, keepdim=False, interpolation=interpolation)
        if keepdim:
            output = output.reshape(tuple(output.shape) + (1,) * input.dim())
        return output

    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        ret = [op(t.reshape(-1), q, dim=0, keepdim=False, interpolation=interpolation) for t in input._storage]
        output = torch.stack(ret)
        if keepdim:
            output = output.unsqueeze(0 if input.batch_first else 1)
        return output

    dim_adj = _translate_dim(input, dim)
    ret = [op(t, q, dim=dim_adj, keepdim=keepdim, interpolation=interpolation) for t in input._storage]
    return _try_stack(ret, input)


@NestedTensorFuncRegistry.implement(torch.nanquantile)
def nanquantile(input: NestedTensor, q, dim: int | None = None, keepdim: bool = False, interpolation: str = "linear"):
    r"""
    This is a variant of [torch.quantile][] that "ignores" ``NaN`` values, computing the quantiles `q` as if
    ``NaN`` values in `input` did not exist. If all values in a reduced row are ``NaN`` then the quantiles for
    that reduction will be ``NaN``. See the documentation for [torch.quantile][].
    See also [torch.nanquantile][].

    Args:
        input: The input NestedTensor.
        q: Quantile value(s) in [0, 1].
        dim: Dimension to reduce. If None, reduces over all elements.
        keepdim: Whether to retain the reduced dimension.
        interpolation: Interpolation method when quantile lies between two data points.

    Returns:
        Tensor | NestedTensor: The computed quantile(s).

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.nanquantile(nt, q=0.5, dim=1), torch.nanquantile(nt.tensor, q=0.5, dim=1))
        True
    """
    return _quantile(input, torch.nanquantile, q, dim, keepdim, interpolation)


@NestedTensorFuncRegistry.implement(torch.quantile)
def quantile(
    input: NestedTensor,
    q,
    dim: int | None = None,
    keepdim: bool = False,
    interpolation: str = "linear",
):
    r"""
    Computes the q-th quantiles of each row of the `input` tensor along the dimension `dim`.
    See also [torch.quantile][].

    Args:
        input: The input NestedTensor.
        q: Quantile value(s) in [0, 1].
        dim: Dimension to reduce. If None, reduces over all elements.
        keepdim: Whether to retain the reduced dimension.
        interpolation: Interpolation method when quantile lies between two data points.

    Returns:
        Tensor | NestedTensor: The computed quantile(s).

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> torch.allclose(torch.quantile(nt, q=0.5, dim=1), torch.quantile(nt.tensor, q=0.5, dim=1))
        True
    """
    return _quantile(input, torch.quantile, q, dim, keepdim, interpolation)


@NestedTensorFuncRegistry.implement(torch.sort)
def sort(input: NestedTensor, dim: int = -1, descending: bool = False, stable: bool | None = None):
    r"""
    Sorts the elements of the `input` tensor along a given dimension in ascending order by value.
    See also [torch.sort][].

    Args:
        input: The input NestedTensor.
        dim: Dimension to sort along.
        descending: If True, sorts in descending order.
        stable: If True, preserves the relative order of equal elements.

    Returns:
        tuple[NestedTensor, NestedTensor]: A tuple of (sorted values, indices).

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.sort(nt, dim=1)
        >>> ref = torch.sort(nt.tensor, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    if stable is None:
        return torch.ops.aten.sort.default(input, dim, descending)
    return torch.ops.aten.sort.stable(input, stable=stable, dim=dim, descending=descending)


@NestedTensorFuncRegistry.implement(torch.topk)
def topk(input: NestedTensor, k, dim: int | None = None, largest: bool = True, sorted: bool = True):
    r"""
    Returns the `k` largest elements of the given `input` tensor along a given dimension.
    See also [torch.topk][].

    Args:
        input: The input NestedTensor.
        k: Number of top elements to return.
        dim: Dimension to sort along. Defaults to the last dimension.
        largest: If True, returns the k largest; otherwise the k smallest.
        sorted: If True, the returned elements are sorted.

    Returns:
        tuple[NestedTensor, NestedTensor]: A tuple of (values, indices).

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> nt = NestedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        >>> out_vals, out_idx = torch.topk(nt, k=1, dim=1)
        >>> ref = torch.topk(nt.tensor, k=1, dim=1)
        >>> torch.equal(out_vals, ref.values) and torch.equal(out_idx, ref.indices)
        True
    """
    dim = -1 if dim is None else dim
    return torch.ops.aten.topk.default(input, k, dim, largest, sorted)


@NestedTensorFuncRegistry.implement(torch.einsum)
def einsum(equation, *operands):
    r"""
    Apply [torch.einsum][] per element of a NestedTensor.

    The equation string describes the dimensions of individual elements
    (without the batch dimension). At least one operand must be a NestedTensor;
    plain Tensor operands are broadcast to every element.
    """
    from .nested_tensor import NestedTensor

    # Find the NestedTensor operand(s) and the reference for metadata
    nt_indices = [i for i, op in enumerate(operands) if isinstance(op, NestedTensor)]
    if not nt_indices:
        raise TypeError("einsum: at least one operand must be a NestedTensor")

    ref_idx = nt_indices[0]
    ref = operands[ref_idx]
    for idx in nt_indices[1:]:
        other = operands[idx]
        _validate_pairwise_batch_length(
            ref, other, op_name="einsum", lhs_name=f"operand[{ref_idx}]", rhs_name=f"operand[{idx}]"
        )

    n = len(ref._storage)
    results = []
    for elem_idx in range(n):
        elem_operands = []
        for _op_idx, op in enumerate(operands):
            if isinstance(op, NestedTensor):
                elem_operands.append(op._storage[elem_idx])
            else:
                elem_operands.append(op)
        results.append(torch.einsum(equation, *elem_operands))
    return NestedTensor(results, **ref._meta())


@NestedTensorFuncRegistry.implement(torch.searchsorted)
def searchsorted(sorted_sequence, values, *, out_int32=False, right=False, side=None, sorter=None):
    r"""Apply [torch.searchsorted][] using aten fastpaths when available."""
    from .nested_tensor import NestedTensor

    sorted_is_nt = isinstance(sorted_sequence, NestedTensor)
    values_is_nt = isinstance(values, NestedTensor)
    sorter_is_nt = isinstance(sorter, NestedTensor)

    if sorter_is_nt and not sorted_is_nt:
        raise TypeError("searchsorted: NestedTensor sorter requires sorted_sequence to be a NestedTensor.")

    if not isinstance(values, (NestedTensor, Tensor)):
        if sorted_is_nt:
            if sorter_is_nt:
                if len(sorter) != len(sorted_sequence):
                    raise ValueError(
                        "searchsorted: NestedTensor batch length mismatch between sorted_sequence and sorter: "
                        f"sorted_sequence={len(sorted_sequence)}, sorter={len(sorter)}"
                    )
                return NestedTensor(
                    (
                        torch.searchsorted(t, values, out_int32=out_int32, right=right, side=side, sorter=s)
                        for t, s in zip(sorted_sequence._storage, sorter._storage)
                    ),
                    **sorted_sequence._meta(),
                )
            return _map_storage(
                sorted_sequence,
                lambda t: torch.searchsorted(t, values, out_int32=out_int32, right=right, side=side, sorter=sorter),
            )
        return torch.searchsorted(sorted_sequence, values, out_int32=out_int32, right=right, side=side, sorter=sorter)

    if sorted_is_nt and values_is_nt:
        _validate_pairwise_batch_length(
            sorted_sequence, values, op_name="searchsorted", lhs_name="sorted_sequence", rhs_name="values"
        )
        return torch.ops.aten.searchsorted.Tensor(
            sorted_sequence, values, out_int32=out_int32, right=right, side=side, sorter=sorter
        )

    if values_is_nt or sorted_is_nt:
        return torch.ops.aten.searchsorted.Tensor(
            sorted_sequence, values, out_int32=out_int32, right=right, side=side, sorter=sorter
        )

    return torch.searchsorted(sorted_sequence, values, out_int32=out_int32, right=right, side=side, sorter=sorter)


# Matrix / linalg ops (per-element)
# ---------------------------------------------------------------------------
# These operate on each element independently. Most are shape-preserving;
# some return scalars per element, and some return tuples.


@NestedTensorFuncRegistry.implement(torch.triu)
def triu(input, diagonal=0):
    r"""Apply [torch.triu][] via aten fastpaths when possible."""
    return torch.ops.aten.triu.default(input, diagonal)


@NestedTensorFuncRegistry.implement(torch.tril)
def tril(input, diagonal=0):
    r"""Apply [torch.tril][] via aten fastpaths when possible."""
    return torch.ops.aten.tril.default(input, diagonal)


@NestedTensorFuncRegistry.implement(torch.matrix_exp)
def matrix_exp(input):
    r"""Apply [torch.matrix_exp][] via aten fastpaths when possible."""
    return torch.ops.aten.matrix_exp.default(input)


@NestedTensorFuncRegistry.implement(torch.diag)
def diag(input, diagonal=0):
    r"""
    Apply [torch.diag][] to each element of a NestedTensor.

    For 1-D elements, returns a 2-D diagonal matrix per element.
    For 2-D elements, returns the diagonal as a 1-D tensor per element.
    """
    return _map_storage(input, lambda t: torch.diag(t, diagonal=diagonal))


@NestedTensorFuncRegistry.implement(torch.diagflat)
def diagflat(input, offset=0):
    r"""Apply [torch.diagflat][] to each element of a NestedTensor."""
    return _map_storage(input, lambda t: torch.diagflat(t, offset=offset))


@NestedTensorFuncRegistry.implement(torch.diagonal)
def diagonal(input, offset=0, dim1=0, dim2=1):
    r"""
    Apply [torch.diagonal][] to each element of a NestedTensor.

    Dimensions are translated to skip the batch dimension.
    """
    dim1_adj = _translate_dim(input, dim1)
    dim2_adj = _translate_dim(input, dim2)
    return _map_storage(input, lambda t: torch.diagonal(t, offset=offset, dim1=dim1_adj, dim2=dim2_adj))


@NestedTensorFuncRegistry.implement(torch.trace)
def trace(input):
    r"""
    Apply [torch.trace][] to each element of a NestedTensor.

    Returns a NestedTensor of scalar tensors (one per element).
    """
    return _map_storage(input, lambda t: torch.trace(t))


@NestedTensorFuncRegistry.implement(torch.det)
def det(input):
    r"""Apply [torch.det][] to each element of a NestedTensor."""
    return _map_storage(input, lambda t: torch.det(t))


@NestedTensorFuncRegistry.implement(torch.inverse)
def inverse(input):
    r"""Apply [torch.inverse][] to each element of a NestedTensor."""
    return torch.ops.aten.inverse.default(input)


@NestedTensorFuncRegistry.implement(torch.matrix_power)
def matrix_power(input, n):
    r"""Apply [torch.matrix_power][] to each element of a NestedTensor."""
    return _map_storage(input, lambda t: torch.matrix_power(t, n))


@NestedTensorFuncRegistry.implement(torch.repeat_interleave)
def repeat_interleave(input, repeats, dim=None, *, output_size=None):
    r"""
    Apply [torch.repeat_interleave][] to each element of a NestedTensor.

    ``dim=None`` flattens each element before repeating. Other dims are
    translated to skip the batch dimension.
    """
    if dim is None:
        return _map_storage(input, lambda t: torch.repeat_interleave(t, repeats, dim=None, output_size=output_size))
    dim_adj = _translate_non_batch_dim(input, dim, name="repeat_interleave")
    return _map_storage(input, lambda t: torch.repeat_interleave(t, repeats, dim=dim_adj, output_size=output_size))


# --- torch.linalg ops ---


@NestedTensorFuncRegistry.implement(torch.linalg.inv)
def linalg_inv(input):
    r"""Apply [torch.linalg.inv][] to each element of a NestedTensor."""
    return torch.ops.aten.linalg_inv.default(input)


@NestedTensorFuncRegistry.implement(torch.linalg.det)
def linalg_det(input):
    r"""Apply [torch.linalg.det][] to each element of a NestedTensor."""
    return _map_storage(input, lambda t: torch.linalg.det(t))


@NestedTensorFuncRegistry.implement(torch.linalg.norm)
def linalg_norm(input, ord=None, dim=None, keepdim=False, *, dtype=None):
    r"""
    Apply [torch.linalg.norm][] to each element of a NestedTensor.

    ``dim`` is translated to skip the batch dimension when specified.
    """
    if dim is not None:
        if isinstance(dim, int):
            dim = _translate_non_batch_dim(input, dim, name="linalg.norm")
        else:
            dim = tuple(_translate_non_batch_dim(input, d, name="linalg.norm") for d in dim)
    return _map_storage(input, lambda t: torch.linalg.norm(t, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype))


@NestedTensorFuncRegistry.implement(torch.linalg.svd)
def linalg_svd(input, full_matrices=True):
    r"""
    Apply [torch.linalg.svd][] to each element of a NestedTensor.

    Returns a tuple of three NestedTensors ``(U, S, Vh)``.
    """
    from .nested_tensor import NestedTensor

    Us, Ss, Vhs = [], [], []
    for t in input._storage:
        U, S, Vh = torch.linalg.svd(t, full_matrices=full_matrices)
        Us.append(U)
        Ss.append(S)
        Vhs.append(Vh)
    meta = input._meta()
    return NestedTensor(Us, **meta), NestedTensor(Ss, **meta), NestedTensor(Vhs, **meta)


@NestedTensorFuncRegistry.implement(torch.linalg.qr)
def linalg_qr(input, mode="reduced"):
    r"""
    Apply [torch.linalg.qr][] to each element of a NestedTensor.

    Returns a tuple of two NestedTensors ``(Q, R)``.
    """
    return _map_storage_pair(input, torch.linalg.qr, mode=mode)


@NestedTensorFuncRegistry.implement(torch.linalg.cholesky)
def linalg_cholesky(input, *, upper=False):
    r"""Apply [torch.linalg.cholesky][] to each element of a NestedTensor."""
    return torch.ops.aten.linalg_cholesky.default(input, upper=upper)


@NestedTensorFuncRegistry.implement(torch.linalg.eigh)
def linalg_eigh(input, UPLO="L"):
    r"""
    Apply [torch.linalg.eigh][] to each element of a NestedTensor.

    Returns a tuple of two NestedTensors ``(eigenvalues, eigenvectors)``.
    """
    return _map_storage_pair(input, torch.linalg.eigh, UPLO=UPLO)


@NestedTensorFuncRegistry.implement(torch.linalg.solve)
def linalg_solve(input, B):
    r"""
    Apply [torch.linalg.solve][] to each element pair of two NestedTensors.

    ``input`` is the coefficient matrix A, ``B`` is the right-hand side.
    """
    from .nested_tensor import NestedTensor

    if isinstance(B, NestedTensor):
        _validate_pairwise_batch_length(input, B, op_name="linalg.solve", lhs_name="input", rhs_name="B")
        results = [torch.linalg.solve(a, b) for a, b in zip(input._storage, B._storage)]
    else:
        results = [torch.linalg.solve(t, B) for t in input._storage]
    return NestedTensor(results, **input._meta())


# Bulk elementwise registrations
# ---------------------------------------------------------------------------
# These register elementwise ops in the NestedTensorFuncRegistry, which is
# caught early in __torch_function__ — bypassing the aten decomposition →
# __torch_dispatch__ path (~14x speedup for elementwise ops on CPU).
#
# Unary ops: apply directly to packed _values via _from_packed.
# Binary ops: use _binary_op_maybe_tensor (handles NT+NT, NT+scalar,
#   and NT+matching-shape-Tensor via nested_like conversion; already has
#   a fast path for same-layout NT+NT via _from_packed internally).

for _op in TORCH_UNARY_ELEMENTWISE_OPS:
    if _op not in NestedTensorFuncRegistry:

        @NestedTensorFuncRegistry.implement(_op)
        def _unary_impl(input, *args, _fn=_op, **kwargs):
            return type(input)._from_packed(
                _fn(input._values, *args, **kwargs),
                input._offsets,
                input._shape_tensor,
                batch_first=input.batch_first,
                padding_value=input.padding_value,
                mask_value=input.mask_value,
                pin_memory=input._pin_memory,
                outer_size=input._logical_shape,
            )


for _op in TORCH_BINARY_ELEMENTWISE_OPS:
    if _op not in NestedTensorFuncRegistry:

        @NestedTensorFuncRegistry.implement(_op)
        def _binary_impl(input, other, *args, _fn=_op, **kwargs):
            return _binary_op_maybe_tensor(input, other, _fn, *args, **kwargs)


# Register Tensor method variants so __torch_function__ finds them directly
# (avoids per-call dict lookup that _METHOD_TO_FUNC previously required)
for _method, _func in (
    (torch.Tensor.split, torch.split),
    (torch.Tensor.chunk, torch.chunk),
    (torch.Tensor.matmul, torch.matmul),
):
    NestedTensorFuncRegistry[_method] = NestedTensorFuncRegistry[_func]

# torch.movedim is a separate function object from torch.moveaxis; register it as alias.
NestedTensorFuncRegistry[torch.movedim] = NestedTensorFuncRegistry[torch.moveaxis]


# Register Tensor.method variants for binary ops so that nt + tensor goes through
# the torch-level handler (which handles nested_like conversion) rather than falling
# through to __torch_dispatch__ (which requires pre-converted NestedTensor operands).
# NOTE: nt + t dispatches __torch_function__ with func=torch.Tensor.add (the method),
# not torch.add (the module-level function). These are different objects in PyTorch.
_TENSOR_METHOD_TO_TORCH_FUNC = {
    torch.Tensor.add: torch.add,
    torch.Tensor.sub: torch.sub,
    torch.Tensor.mul: torch.mul,
    torch.Tensor.div: torch.div,
    torch.Tensor.remainder: torch.remainder,
    torch.Tensor.floor_divide: torch.floor_divide,
    torch.Tensor.pow: torch.pow,
    torch.Tensor.eq: torch.eq,
    torch.Tensor.ne: torch.ne,
    torch.Tensor.lt: torch.lt,
    torch.Tensor.le: torch.le,
    torch.Tensor.gt: torch.gt,
    torch.Tensor.ge: torch.ge,
    torch.Tensor.fmod: torch.fmod,
}

for _method, _func in _TENSOR_METHOD_TO_TORCH_FUNC.items():
    if _func in NestedTensorFuncRegistry:
        NestedTensorFuncRegistry[_method] = NestedTensorFuncRegistry[_func]


# Python function wrappers for dunder ops that delegate to torch.* internally.
# __floordiv__ is a Python wrapper around torch.floor_divide; it doesn't go through
# the C++ Tensor.floor_divide method, so it needs its own registration.
# NOTE: Do NOT register __rsub__, __rtruediv__, __rmod__, __rfloordiv__ here —
# they have reversed semantics (a.__rsub__(b) = b - a) that _binary_op_maybe_tensor
# can't distinguish from the normal order. They fall through to the underlying
# Python wrappers which call torch.sub(b, a) etc., properly ordered.
_DUNDER_WRAPPER_TO_TORCH_FUNC = {
    torch.Tensor.__floordiv__: torch.floor_divide,
}

for _method, _func in _DUNDER_WRAPPER_TO_TORCH_FUNC.items():
    if _func in NestedTensorFuncRegistry:
        NestedTensorFuncRegistry[_method] = NestedTensorFuncRegistry[_func]


# In-place Tensor.method_ variants (add_, sub_, etc.) that go through __torch_function__.
# These need a wrapper that converts the operand then delegates to the aten in-place op.
def _inplace_binary_torch_handler(self, other, *args, _fn=None, **kwargs):
    r"""Torch-level handler for in-place binary ops that converts tensor operands."""
    from .nested_tensor import NestedTensor

    if (
        isinstance(other, Tensor)
        and other.dim() > 0
        and not isinstance(other, NestedTensor)
        and self.shape == other.shape
    ):
        other = self.nested_like(other, strict=False)
    # Dispatch to the underlying in-place aten op via default path
    with torch._C.DisableTorchFunctionSubclass():
        return _fn(self, other, *args, **kwargs)


_INPLACE_METHOD_MAP = {
    torch.Tensor.add_: torch.Tensor.add_,
    torch.Tensor.sub_: torch.Tensor.sub_,
    torch.Tensor.mul_: torch.Tensor.mul_,
    torch.Tensor.div_: torch.Tensor.div_,
    torch.Tensor.remainder_: torch.Tensor.remainder_,
    torch.Tensor.floor_divide_: torch.Tensor.floor_divide_,
    torch.Tensor.pow_: torch.Tensor.pow_,
    torch.Tensor.fmod_: torch.Tensor.fmod_,
}

for _method, _underlying in _INPLACE_METHOD_MAP.items():

    @NestedTensorFuncRegistry.implement(_method)
    def _inplace_impl(self, other, *args, _fn=_underlying, **kwargs):
        return _inplace_binary_torch_handler(self, other, *args, _fn=_fn, **kwargs)
