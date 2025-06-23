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
``__torch_dispatch__`` handlers for NestedTensor aten ops.

This module implements the dispatch table that maps aten ops to optimized handlers
operating on the packed representation (_values, _offsets, _physical_shape).

Architecture:
    - Elementwise ops operate directly on ``_values`` (no unpack/repack overhead)
    - Structural ops (clone, detach, to_copy) operate on all inner tensors
    - Unregistered ops fall back to per-element application via ``_storage``
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .ops import (
    ATEN_BINARY_ELEMENTWISE_OPS,
    ATEN_UNARY_ELEMENTWISE_OPS,
    NestedTensorAtenRegistry,
    _get_batch_dim,
    _maybe_align_dense_to_nested,
    _normalize_dim,
    _translate_dim,
    _translate_dims,
    _try_stack,
)

if TYPE_CHECKING:
    from .nested_tensor import NestedTensor

aten = torch.ops.aten

_MISSING = object()

try:
    from torch._subclasses.fake_tensor import is_fake as _torch_is_fake
except ImportError:
    _torch_is_fake = None


def _is_fake_tensor(tensor: Tensor) -> bool:
    if _torch_is_fake is None:
        return False
    return bool(_torch_is_fake(tensor))


def _offsets_match(a: Tensor, b: Tensor) -> bool:
    r"""
    Check if two offset tensors represent the same packing layout.

    Uses pointer identity as a fast-path before falling back to elementwise comparison.
    """
    if _is_fake_tensor(a) or _is_fake_tensor(b):
        if a is b:
            return True
        if a.shape != b.shape:
            return False
        a_const = getattr(a, "constant", None)
        b_const = getattr(b, "constant", None)
        if isinstance(a_const, Tensor) and isinstance(b_const, Tensor):
            return bool(torch.equal(a_const, b_const))
        # Offsets are CPU metadata derived from packing layout. Under fake tensor mode
        # we prefer semantics over object identity so compile/tracing can exercise the
        # same packed paths as eager execution.
        return True
    try:
        if a.data_ptr() == b.data_ptr():
            return True
    except RuntimeError:
        # Functional tensors may not expose data_ptr.
        pass
    if a.shape != b.shape:
        return False
    return bool(torch.equal(a, b))


def _offsets_match_identity_if_fake(a: Tensor, b: Tensor) -> bool:
    r"""Check offset compatibility, but require object identity under fake tensor mode."""
    if _is_fake_tensor(a) or _is_fake_tensor(b):
        return a is b
    return _offsets_match(a, b)


def _layout_match(lhs: NestedTensor, rhs: NestedTensor) -> bool:
    r"""Return whether two NestedTensors share the same packed layout contract."""
    return lhs._has_same_layout(rhs)


def _structure_match(lhs: NestedTensor, rhs: NestedTensor) -> bool:
    r"""Return whether two NestedTensors share the same ragged packing structure."""
    return lhs._has_same_structure(rhs)


def _find_nested(*args) -> NestedTensor | None:
    r"""Find and return the first NestedTensor in args, searching recursively."""
    from .nested_tensor import NestedTensor

    for a in args:
        if isinstance(a, NestedTensor):
            return a
        if isinstance(a, (list, tuple)):
            result = _find_nested(*a)
            if result is not None:
                return result
    return None


@torch._dynamo.disable
def _per_element_fallback_serial(func, args, kwargs, source):
    r"""Slow per-element fallback body for non-empty NestedTensor inputs."""
    from .nested_tensor import NestedTensor

    batch_size = len(source)

    def replace_nested_with_element(obj, idx):
        r"""Replace each NestedTensor in obj with its idx-th element."""
        if isinstance(obj, NestedTensor):
            return obj._storage[idx]
        if isinstance(obj, (list, tuple)):
            return type(obj)(replace_nested_with_element(x, idx) for x in obj)
        return obj

    results = []
    for i in range(batch_size):
        elem_args = replace_nested_with_element(args, i)
        elem_kwargs = {k: replace_nested_with_element(v, i) for k, v in kwargs.items()} if kwargs else {}
        results.append(func(*elem_args, **elem_kwargs))

    if all(isinstance(r, Tensor) for r in results):
        return NestedTensor(
            results,
            batch_first=source.batch_first,
            padding_value=source.padding_value,
            mask_value=source.mask_value,
        )

    if isinstance(results[0], tuple):
        num_outputs = len(results[0])
        unpacked = []
        for out_idx in range(num_outputs):
            elements = [r[out_idx] for r in results]
            if all(isinstance(e, Tensor) for e in elements):
                unpacked.append(
                    NestedTensor(
                        elements,
                        batch_first=source.batch_first,
                        padding_value=source.padding_value,
                        mask_value=source.mask_value,
                    )
                )
            else:
                unpacked.append(elements)
        return tuple(unpacked)

    return results


def per_element_fallback(func, args, kwargs):
    r"""
    Fallback for unregistered ops: unpack to individual tensors, apply op, repack.

    Used both by ``__torch_dispatch__`` as a catch-all for unregistered ops and by
    registered handlers that detect a packed fast path would change dense semantics.

    Note:
        This function is decorated with ``@torch._dynamo.disable``, so any op
        that reaches this fallback will exit a compiled graph. Register aten-level
        handlers in ``NestedTensorAtenRegistry`` for ops that must be compile-friendly.
    """
    from .nested_tensor import NestedTensor

    source = _find_nested(*args)
    if source is None:
        source = _find_nested(*kwargs.values()) if kwargs else None
    if source is None:
        return func(*args, **kwargs)

    if len(source) == 0:

        def replace_nested_with_values(obj):
            if isinstance(obj, NestedTensor):
                return obj._values
            if isinstance(obj, (list, tuple)):
                return type(obj)(replace_nested_with_values(x) for x in obj)
            return obj

        packed_args = replace_nested_with_values(args)
        packed_kwargs = {k: replace_nested_with_values(v) for k, v in kwargs.items()} if kwargs else {}

        def rebuild_empty(t: Tensor):
            return _packed_like(source, t)

        try:
            empty_result = func(*packed_args, **packed_kwargs)
        except (TypeError, RuntimeError, ValueError):
            return rebuild_empty(source._values)

        if isinstance(empty_result, Tensor):
            return rebuild_empty(empty_result)
        if isinstance(empty_result, tuple):
            return tuple(rebuild_empty(x) if isinstance(x, Tensor) else x for x in empty_result)
        return empty_result
    return _per_element_fallback_serial(func, args, kwargs, source)


def _apply_per_element_nested(source: NestedTensor, op):
    r"""
    Apply ``op`` to each element and always rebuild a NestedTensor.

    Unlike ``per_element_fallback``, this helper is not ``@torch._dynamo.disable``.
    Use this only when we intentionally preserve NestedTensor output structure.
    """
    cls = type(source)
    if len(source) == 0:
        return cls([], **source._meta(include_dtype=True))
    return cls((op(t) for t in source._unpack()), **source._meta())


# ---------------------------------------------------------------------------
# Elementwise unary ops — apply directly to _values
# ---------------------------------------------------------------------------


def _unary_handler(func, args, kwargs):
    r"""Dispatch handler for elementwise unary ops applied to _values."""
    source = args[0]
    return _packed_like(source, func(source._values, *args[1:], **kwargs))


# ---------------------------------------------------------------------------
# Elementwise binary ops — apply directly to _values
# ---------------------------------------------------------------------------


def _resolve_other(source, other, func):
    r"""
    Resolve the *other* operand for a binary op where *source* is a NestedTensor.

    Returns the value to use alongside ``source._values``:
    - NestedTensor with matching packed layout → ``other._values``
    - Scalar or 0-dim tensor → ``other`` (broadcast-compatible with packed _values)
    - Dense tensor with shape matching ``source.shape`` → packed via ``nested_like(..., strict=False)``
    - Anything else → raises ``NotImplementedError``
    """
    from .nested_tensor import NestedTensor

    if isinstance(other, NestedTensor):
        if _structure_match(source, other):
            return other._values
        raise NotImplementedError(f"NestedTensor: {func} with mismatched packing layouts")
    if isinstance(other, Tensor) and other.dim() > 0:
        aligned = _maybe_align_dense_to_nested(source, other)
        if aligned is not None:
            if _structure_match(source, aligned):
                return aligned._values
        raise NotImplementedError(f"NestedTensor: {func} with non-scalar Tensor operand; convert to NestedTensor first")
    return other


def _element_shape(shape_tensor: Tensor, idx: int) -> tuple[int, ...]:
    r"""Return one full element shape from packed shape metadata."""
    shape = shape_tensor[idx].tolist()
    while shape and shape[-1] == 0:
        shape.pop()
    return tuple(shape)


def _broadcasts_per_element(source, candidate: Tensor) -> bool:
    r"""
    Return whether a dense tensor is safe for a packed fast path.

    Dense parity requires two things:
    1. the tensor broadcasts against every individual NestedTensor element
    2. any non-trivial broadcast extent lands only on packed static dims, never on
       the leading packed dim that concatenates ragged rows across elements
    """
    element_ndim = source._physical_shape.size(1)
    if element_ndim == 0:
        return candidate.dim() == 0
    if source._values.dim() != element_ndim:
        return False

    candidate_shape = tuple(candidate.shape)
    padded_shape = (1,) * max(0, element_ndim - len(candidate_shape)) + candidate_shape
    if padded_shape and padded_shape[0] != 1:
        return False
    for idx in range(len(source)):
        try:
            torch.broadcast_shapes(_element_shape(source._physical_shape, idx), candidate_shape)
        except RuntimeError:
            return False
    return True


def _resolve_ternary_other(source, other, func):
    r"""
    Resolve a ternary-op operand against ``source``.

    This accepts the same layout-aligned NestedTensor and logical-shape-aligned
    dense tensor cases as ``_resolve_other``. Plain tensors are only accepted when
    they broadcast against each element individually, which matches dense semantics.
    """
    from .nested_tensor import NestedTensor

    device = source._values.device
    if isinstance(other, NestedTensor):
        if _structure_match(source, other):
            values = other._values
            if values.device != device:
                values = values.to(device=device)
            return values
        raise NotImplementedError(f"NestedTensor: {func} requires matching packed structure across all NT operands")
    if isinstance(other, Tensor):
        if other.dim() == 0:
            return other if other.device == device else other.to(device=device)
        aligned = _maybe_align_dense_to_nested(source, other)
        if aligned is not None:
            if _structure_match(source, aligned):
                values = aligned._values
                if values.device != device:
                    values = values.to(device=device)
                return values
        candidate = other if other.device == device else other.to(device=device)
        if _broadcasts_per_element(source, candidate):
            return candidate
        raise NotImplementedError(
            f"NestedTensor: {func} with non-scalar Tensor operand that is neither shape-aligned nor "
            "broadcast-compatible with each NestedTensor element"
        )
    return other


def _binary_handler(func, args, kwargs):
    r"""Dispatch handler for elementwise binary ops on packed _values."""
    from .nested_tensor import NestedTensor

    lhs, rhs = args[0], args[1]
    extra = args[2:]
    if isinstance(lhs, NestedTensor):
        resolved = _resolve_other(lhs, rhs, func)
        return _packed_like(lhs, func(lhs._values, resolved, *extra, **kwargs))
    # lhs is scalar/tensor, rhs is NestedTensor
    resolved = _resolve_other(rhs, lhs, func)
    return _packed_like(rhs, func(resolved, rhs._values, *extra, **kwargs))


# ---------------------------------------------------------------------------
# Structured / shape-changing ops
# ---------------------------------------------------------------------------


@NestedTensorAtenRegistry.implement(aten.constant_pad_nd.default)
def _constant_pad_nd(func, args, kwargs):
    r"""Compile-safe handler for constant padding on packed values when ragged dim is untouched."""
    source = args[0]
    pad = tuple(args[1])
    if len(pad) % 2 != 0:
        return per_element_fallback(func, args, kwargs)

    padded_dims = len(pad) // 2
    # Packed fast path is valid only when padding targets trailing static dims.
    if source._values.dim() <= 1 or padded_dims >= source._values.dim():
        return per_element_fallback(func, args, kwargs)

    out_values = func(source._values, *args[1:], **kwargs)
    out_physical_shape = source._physical_shape.clone()
    for i in range(padded_dims):
        out_physical_shape[:, -(i + 1)] += pad[2 * i] + pad[2 * i + 1]
    out_logical = list(source._logical_physical_dims())
    for i in range(padded_dims):
        out_logical[-(i + 1)] += int(pad[2 * i] + pad[2 * i + 1])
    return _packed_with_shape(
        source,
        out_values,
        out_physical_shape,
        source._logical_shape_from_physical_dims(out_logical),
        permutation=source._permutation,
    )


@NestedTensorAtenRegistry.implement(aten.clone.default)
def _clone(func, args, kwargs):
    r"""Clone all internal tensors of a NestedTensor."""
    source = args[0]
    return type(source)._from_packed(
        source._values.clone(),
        source._offsets.clone(),
        source._physical_shape.clone(),
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=source._logical_shape,
    )


@NestedTensorAtenRegistry.implement(aten.detach.default)
def _detach(func, args, kwargs):
    r"""Detach all internal tensors from the computation graph."""
    source = args[0]
    return type(source)._from_packed(
        source._values.detach(),
        source._offsets.detach(),
        source._physical_shape.detach(),
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=source._logical_shape,
    )


@NestedTensorAtenRegistry.implement(aten._to_copy.default)
def _to_copy(func, args, kwargs):
    r"""
    Copy _values to a new dtype/device while preserving metadata tensors.

    Note: memory_format is applied to the packed _values buffer, not per-element.
    For non-contiguous formats like channels_last, the result may not have
    meaningful per-element layout since _values is a concatenation of
    variable-length elements.
    """
    source = args[0]
    # Offsets and shape_tensor stay on CPU — they are metadata, not compute tensors.
    return _packed_like(source, func(source._values, **kwargs))


@NestedTensorAtenRegistry.implement(aten.alias.default)
def _alias(func, args, kwargs):
    r"""Create an alias of the NestedTensor sharing the same _values storage."""
    source = args[0]
    return _packed_like(source, source._values.alias())


# ---------------------------------------------------------------------------
# Shape/view ops — operate on packed _values and update metadata
# ---------------------------------------------------------------------------


@NestedTensorAtenRegistry.implement(aten.flatten.using_ints)
def _flatten(func, args, kwargs):
    r"""Flatten static per-element dims on packed values when the batch axis is untouched."""
    source = args[0]
    kw_start = kwargs.pop("start_dim", _MISSING)
    kw_end = kwargs.pop("end_dim", _MISSING)
    if len(args) > 1:
        if kw_start is not _MISSING:
            raise TypeError("flatten() got multiple values for argument 'start_dim'")
        start_dim = args[1]
    else:
        start_dim = 0 if kw_start is _MISSING else kw_start
    if len(args) > 2:
        if kw_end is not _MISSING:
            raise TypeError("flatten() got multiple values for argument 'end_dim'")
        end_dim = args[2]
    else:
        end_dim = -1 if kw_end is _MISSING else kw_end
    ndims = source.dim()
    start = _normalize_dim(start_dim, ndims)
    end = _normalize_dim(end_dim, ndims)
    if start < 0 or end < 0 or start >= ndims or end >= ndims:
        raise IndexError(f"start_dim and end_dim must be in range [0, {ndims}), got ({start_dim}, {end_dim})")
    if start > end:
        raise ValueError(f"start_dim must be <= end_dim, got ({start_dim}, {end_dim})")

    batch_dim = _get_batch_dim(source)
    if start <= batch_dim <= end:
        # Flattening the batch axis yields a plain tensor by definition.
        return func(source.tensor, start_dim, end_dim, **kwargs)

    start_adj = _translate_dim(source, start)
    end_adj = _translate_dim(source, end)
    if start_adj == 0:
        # Flattening across ragged dim-0 can alter per-element rank/shape patterns.
        # Keep the generic fallback here because it preserves "stack when uniform"
        # behavior used by NestedTensor reductions/fallback semantics.
        return per_element_fallback(func, (source, start_adj, end_adj), kwargs)

    out_values = func(source._values, start_adj, end_adj, **kwargs)
    merged = torch.prod(source._physical_shape[:, start_adj : end_adj + 1], dim=1, keepdim=True)
    out_shape = torch.cat(
        (source._physical_shape[:, :start_adj], merged, source._physical_shape[:, end_adj + 1 :]),
        dim=1,
    )
    physical_dims = list(source._logical_physical_dims())
    physical_dims[start_adj : end_adj + 1] = [math.prod(physical_dims[start_adj : end_adj + 1])]
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_physical_dims(physical_dims),
    )


@NestedTensorAtenRegistry.implement(aten.view.default)
@NestedTensorAtenRegistry.implement(aten.view_copy.default)
@NestedTensorAtenRegistry.implement(aten.reshape.default)
def _view_like(func, args, kwargs):
    r"""Apply view-like reshapes with packed fastpath when output tails are uniform."""
    source = args[0]
    if len(args) > 1:
        target = tuple(args[1])
    elif "size" in kwargs:
        target = tuple(kwargs.pop("size"))
    elif "shape" in kwargs:
        target = tuple(kwargs.pop("shape"))
    else:
        raise ValueError(f"NestedTensor: {func} missing target shape")

    view_shapes = source._view_shapes(target)
    if not view_shapes:
        return type(source)([], **source._meta(include_dtype=True))

    def rebuild_per_element():
        outputs = [func(t, list(s), **kwargs) for t, s in zip(source._unpack(), view_shapes)]
        return type(source)(outputs, **source._meta())

    def rebuild_linearized():
        return rebuild_per_element()

    rank = len(view_shapes[0])
    if any(-1 in s for s in view_shapes):
        return rebuild_per_element()

    if not all(len(s) == rank for s in view_shapes):
        return rebuild_linearized()

    if rank > 0:
        tail = view_shapes[0][1:]
        tails_uniform = all(s[1:] == tail for s in view_shapes[1:])
    else:
        tails_uniform = True

    if not tails_uniform:
        return rebuild_linearized()

    lengths = [int(s[0]) if rank > 0 else 1 for s in view_shapes]
    total_length = int(sum(lengths))
    packed_shape = [total_length, *view_shapes[0][1:]] if rank > 0 else [len(view_shapes)]
    out_values = func(source._values, packed_shape, **kwargs)

    if rank > 0:
        out_physical_shape = torch.as_tensor(
            view_shapes,
            dtype=source._physical_shape.dtype,
            device=source._physical_shape.device,
        )
        packed_sizes = tuple(lengths)
        element_shapes = tuple(tuple(shape) for shape in view_shapes)
        max_sizes = [max(shape[dim] for shape in view_shapes) for dim in range(rank)]
    else:
        out_physical_shape = torch.empty(
            (len(view_shapes), 0),
            dtype=source._physical_shape.dtype,
            device=source._physical_shape.device,
        )
        packed_sizes = tuple(1 for _ in view_shapes)
        element_shapes = tuple(() for _ in view_shapes)
        max_sizes = []

    lengths_tensor = torch.as_tensor(lengths, dtype=source._offsets.dtype, device=source._offsets.device)
    out_offsets = torch.empty((lengths_tensor.numel() + 1,), dtype=source._offsets.dtype, device=source._offsets.device)
    out_offsets[0] = 0
    if lengths_tensor.numel() > 0:
        out_offsets[1:] = torch.cumsum(lengths_tensor, dim=0)

    if source.batch_first:
        out_logical = [len(source), *max_sizes]
    elif max_sizes:
        out_logical = [max_sizes[0], len(source), *max_sizes[1:]]
    else:
        out_logical = [len(source)]

    return _packed_with_shape(
        source,
        out_values,
        out_physical_shape,
        out_logical,
        offsets=out_offsets,
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
    )


@NestedTensorAtenRegistry.implement(aten.permute.default)
def _permute(func, args, kwargs):
    r"""Permute static per-element dims while keeping the batch axis fixed."""
    source = args[0]
    kw_dims = kwargs.pop("dims", _MISSING)
    if len(args) > 1:
        if kw_dims is not _MISSING:
            raise TypeError("permute() got multiple values for argument 'dims'")
        dims = args[1]
    else:
        if kw_dims is _MISSING:
            raise ValueError("NestedTensor: permute missing dims")
        dims = kw_dims
    dim_count = source.dim()
    if len(dims) != dim_count:
        raise ValueError(f"Expected {dim_count} dimensions, got {len(dims)}")

    normalized_dims = tuple(_normalize_dim(d, dim_count) for d in dims)
    if set(normalized_dims) != set(range(dim_count)):
        raise ValueError(f"Invalid permutation dims {dims} for shape with {dim_count} dims")

    batch_dim = _get_batch_dim(source)
    if normalized_dims[batch_dim] != batch_dim:
        raise ValueError("Permuting the batch dimension is not supported for NestedTensor.")

    tensor_dims = tuple(_translate_dim(source, d) for d in normalized_dims if d != batch_dim)
    if tensor_dims[0] != 0:
        # Ragged dim move stays per-element but should remain in the compile graph
        # when possible, so we avoid the dynamo-disabled generic fallback.
        return _apply_per_element_nested(source, lambda t: t.permute(*tensor_dims))
    out_values = func(source._values, list(tensor_dims), **kwargs)
    out_shape = source._physical_shape[:, tensor_dims]
    out_logical = source._logical_shape_from_physical_dims(
        tuple(source._logical_physical_dims()[dim] for dim in tensor_dims)
    )
    return _packed_with_shape(source, out_values, out_shape, out_logical)


@NestedTensorAtenRegistry.implement(aten.transpose.int)
def _transpose(func, args, kwargs):
    r"""Transpose two non-batch logical dims by swapping per-element dimensions."""
    source = args[0]
    dim0 = _normalize_dim(args[1], source.dim())
    dim1 = _normalize_dim(args[2], source.dim())
    batch_dim = _get_batch_dim(source)
    if dim0 == batch_dim or dim1 == batch_dim:
        raise ValueError("Cannot transpose the batch dimension for NestedTensor.")

    dim0_adj = _translate_dim(source, dim0)
    dim1_adj = _translate_dim(source, dim1)
    if dim0_adj == 0 or dim1_adj == 0:
        # Same rationale as _permute: preserve NestedTensor output without forcing
        # a dynamo-disabled generic fallback.
        return _apply_per_element_nested(source, lambda t: t.transpose(dim0_adj, dim1_adj))
    out_values = func(source._values, dim0_adj, dim1_adj, **kwargs)
    out_shape = source._physical_shape.clone()
    out_shape[:, [dim0_adj, dim1_adj]] = out_shape[:, [dim1_adj, dim0_adj]]
    physical_dims = list(source._logical_physical_dims())
    physical_dims[dim0_adj], physical_dims[dim1_adj] = physical_dims[dim1_adj], physical_dims[dim0_adj]
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_physical_dims(physical_dims),
    )


@NestedTensorAtenRegistry.implement(aten.unsqueeze.default)
def _unsqueeze(func, args, kwargs):
    r"""Insert a singleton logical dim after the batch axis and update metadata."""
    source = args[0]
    dim = args[1]
    ndims = source.dim()
    if dim < 0:
        dim += ndims + 1
    if dim < 0 or dim > ndims:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-ndims - 1}, {ndims}], but got {dim})")

    batch_dim = _get_batch_dim(source)
    if dim <= batch_dim:
        raise ValueError("Cannot unsqueeze at or before the batch dimension for NestedTensor.")

    dim_adj = dim - 1
    out_values = func(source._values, dim_adj, **kwargs)
    ones = torch.ones(
        (source._physical_shape.size(0), 1),
        dtype=source._physical_shape.dtype,
        device=source._physical_shape.device,
    )
    out_shape = torch.cat(
        (source._physical_shape[:, :dim_adj], ones, source._physical_shape[:, dim_adj:]),
        dim=1,
    )
    physical_dims = list(source._logical_physical_dims())
    physical_dims.insert(dim_adj, 1)
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_physical_dims(physical_dims),
    )


@NestedTensorAtenRegistry.implement(aten.squeeze.dim)
def _squeeze_dim(func, args, kwargs):
    r"""Squeeze one logical dim; use packed fastpath for static per-element dims."""
    source = args[0]
    dim = _normalize_dim(args[1], source.dim())
    batch_dim = _get_batch_dim(source)
    if dim <= batch_dim:
        raise ValueError("Cannot squeeze the batch dimension or dimensions before it for NestedTensor.")

    dim_adj = _translate_dim(source, dim)
    if dim_adj == 0:
        # Squeezing ragged dim-0 is per-element but shape-structure preserving.
        return _apply_per_element_nested(source, lambda t: t.squeeze(dim_adj))

    out_values = func(source._values, dim_adj, **kwargs)
    if source._values.size(dim_adj) != 1:
        return _packed_like(source, out_values)

    out_shape = torch.cat(
        (source._physical_shape[:, :dim_adj], source._physical_shape[:, dim_adj + 1 :]),
        dim=1,
    )
    physical_dims = list(source._logical_physical_dims())
    del physical_dims[dim_adj]
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_physical_dims(physical_dims),
    )


@NestedTensorAtenRegistry.implement(aten.squeeze.default)
def _squeeze_default(func, args, kwargs):
    r"""Squeeze all singleton per-element dims with a packed fastpath when ragged dim-0 is untouched."""
    source = args[0]
    rank = source._physical_shape.size(1)
    if rank == 0:
        return _packed_like(source, source._values)

    # If any sample has ragged size 1, squeezing dim-0 is per-element.
    if source._physical_shape.size(0) > 0 and bool(torch.any(source._physical_shape[:, 0] == 1)):
        return _apply_per_element_nested(source, lambda t: t.squeeze())

    out_values = func(source._values, **kwargs)
    if source._physical_shape.size(0) == 0:
        squeeze_mask = torch.zeros(
            (rank,),
            dtype=torch.bool,
            device=source._physical_shape.device,
        )
        for i in range(1, rank):
            if source._logical_shape[i] == 1:
                squeeze_mask[i] = True
    else:
        squeeze_mask = source._physical_shape.eq(1).all(dim=0)
        squeeze_mask[0] = False

    out_shape = source._physical_shape[:, ~squeeze_mask]
    physical_dims = [
        size for index, size in enumerate(source._logical_physical_dims()) if not bool(squeeze_mask[index])
    ]
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_physical_dims(physical_dims),
    )


@NestedTensorAtenRegistry.implement(aten.unflatten.int)
def _unflatten(func, args, kwargs):
    r"""Unflatten static per-element dims on packed values and expand metadata."""
    source = args[0]
    dim = _normalize_dim(args[1], source.dim())
    sizes = args[2]
    batch_dim = _get_batch_dim(source)
    if dim <= batch_dim:
        raise ValueError("unflatten at or before the batch dimension is not supported for NestedTensor.")

    dim_adj = _translate_dim(source, dim)
    if dim_adj == 0:
        # Unflattening ragged dim can produce shape patterns that may collapse to a
        # plain Tensor when uniform across batch; keep generic fallback semantics.
        return per_element_fallback(func, (source, dim_adj, sizes), kwargs)

    out_values = func(source._values, dim_adj, sizes, **kwargs)
    inserted_rank = out_values.dim() - source._values.dim() + 1
    resolved_sizes = out_values.shape[dim_adj : dim_adj + inserted_rank]
    inserted = source._physical_shape.new_tensor(resolved_sizes).unsqueeze(0).expand(source._physical_shape.size(0), -1)
    out_shape = torch.cat(
        (source._physical_shape[:, :dim_adj], inserted, source._physical_shape[:, dim_adj + 1 :]),
        dim=1,
    )
    physical_dims = list(source._logical_physical_dims())
    physical_dims[dim_adj : dim_adj + 1] = [int(size) for size in resolved_sizes]
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_physical_dims(physical_dims),
    )


# ---------------------------------------------------------------------------
# Matrix multiply ops — apply to _values, update last dim of shape_tensor
# ---------------------------------------------------------------------------


def _packed_new_last_dim(source: NestedTensor, new_values: Tensor, new_last_dim: int) -> NestedTensor:
    r"""Rebuild a NestedTensor with a changed last dimension (e.g. after matmul)."""
    new_physical_shape, packed_sizes, element_shapes = source._replace_trailing_physical_dims_meta((int(new_last_dim),))
    new_outer_size = list(source._logical_shape)
    if new_outer_size:
        new_outer_size[-1] = int(new_last_dim)
    return _packed_with_shape(
        source,
        new_values,
        new_physical_shape,
        tuple(new_outer_size),
        permutation=source._permutation_after_replacing_trailing_dims(1, 1),
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
    )


def _packed_new_dim_size(source: NestedTensor, new_values: Tensor, dim_adj: int, new_dim_size: int) -> NestedTensor:
    r"""Rebuild a NestedTensor with a changed per-element dimension size."""
    new_physical_shape, packed_sizes, element_shapes = source._shape_meta_from_components(
        replace_dims={int(dim_adj): int(new_dim_size)}
    )
    return _packed_with_shape(
        source,
        new_values,
        new_physical_shape,
        source._logical_shape_from_components(replace_dims={int(dim_adj): int(new_dim_size)}),
        permutation=source._permutation,
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
    )


def _packed_like(source: NestedTensor, new_values: Tensor) -> NestedTensor:
    r"""Rebuild a NestedTensor from source metadata and a new packed value tensor."""
    return _packed_with_shape(
        source,
        new_values,
        source._physical_shape,
        source._logical_shape,
        permutation=source._permutation,
        packed_sizes=source._packed_sizes,
        element_shapes=source._element_shapes,
    )


def _packed_with_shape(
    source: NestedTensor,
    new_values: Tensor,
    new_physical_shape: Tensor,
    new_logical_shape=None,
    *,
    offsets: Tensor | None = None,
    permutation: tuple[int, ...] | None = None,
    packed_sizes: tuple[int, ...] | None = None,
    element_shapes: tuple[tuple[int, ...], ...] | None = None,
) -> NestedTensor:
    r"""Rebuild a NestedTensor with explicit ``shape_tensor`` and logical shape."""
    if offsets is None:
        offsets = source._offsets
    if new_logical_shape is None:
        new_logical_shape = type(source)._logical_shape_from_physical_shape(
            new_physical_shape, offsets, source.batch_first
        )
    return type(source)._from_packed(
        new_values,
        offsets,
        new_physical_shape,
        permutation=permutation,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=torch.Size(new_logical_shape),
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
    )


def _packed_with_tail_from_values(source: NestedTensor, new_values: Tensor) -> NestedTensor:
    r"""
    Rebuild a NestedTensor by preserving per-element dim-0 lengths and using ``new_values`` static tail dims.

    This is used by packed fast paths whose outputs keep the ragged leading element dim
    but may change trailing per-element dimensions.
    """
    if source._physical_shape.size(1) == 0:
        return _packed_like(source, new_values)

    tail = tuple(int(x) for x in new_values.shape[1:])
    out_shape, outer_size, packed_sizes, element_shapes = source._leading_dim_preserving_meta(tail)
    return _packed_with_shape(
        source,
        new_values,
        out_shape,
        outer_size,
        permutation=source._permutation_after_replacing_trailing_dims(
            max(source._physical_shape.size(1) - 1, 0), len(tail)
        ),
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
    )


def _from_uniform_batched_output(source: NestedTensor, batched_values: Tensor) -> NestedTensor:
    r"""Wrap a batch-major tensor ``[B, *shape]`` as a NestedTensor with uniform per-element shape."""
    batch_size = len(source)
    elem_shape = tuple(int(x) for x in batched_values.shape[1:])
    if not elem_shape:
        out_values = batched_values.reshape(batch_size)
        out_offsets = torch.arange(batch_size + 1, dtype=source._offsets.dtype, device=source._offsets.device)
        out_shape = source._physical_shape.new_empty((batch_size, 0))
        packed_sizes = tuple(1 for _ in range(batch_size))
        element_shapes = tuple(() for _ in range(batch_size))
    else:
        lengths = source._offsets.new_full((batch_size,), elem_shape[0])
        out_offsets = torch.empty((batch_size + 1,), dtype=source._offsets.dtype, device=source._offsets.device)
        out_offsets[0] = 0
        if lengths.numel() > 0:
            out_offsets[1:] = torch.cumsum(lengths, dim=0)
        out_shape = source._physical_shape.new_tensor(elem_shape).reshape(1, -1).expand(batch_size, -1).clone()
        out_values = batched_values.reshape(batch_size * elem_shape[0], *elem_shape[1:])
        packed_sizes = tuple(int(elem_shape[0]) for _ in range(batch_size))
        element_shapes = tuple(tuple(elem_shape) for _ in range(batch_size))
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        batched_values.shape,
        offsets=out_offsets,
        permutation=source._permutation_after_replacing_trailing_dims(
            max(source._physical_shape.size(1) - 1, 0), len(elem_shape[1:])
        ),
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
    )


def _logical_dim_from_tensor_dim(source: NestedTensor, dim_adj: int) -> int:
    r"""Map a per-element tensor dimension index to logical NestedTensor dimension."""
    return dim_adj + 1 if source.batch_first else (dim_adj if dim_adj == 0 else dim_adj + 1)


def _reduce_non_ragged_packed(source: NestedTensor, out_values: Tensor, dim_adj: int, keepdim: bool):
    r"""Wrap non-ragged dim reductions on packed values as a NestedTensor."""
    logical_dim = _logical_dim_from_tensor_dim(source, dim_adj)
    if keepdim:
        out_shape, packed_sizes, element_shapes = source._shape_meta_from_components(replace_dims={dim_adj: 1})
    else:
        keep_dims = tuple(i for i in range(source._physical_shape.size(1)) if i != dim_adj)
        out_shape, packed_sizes, element_shapes = source._shape_meta_from_components(keep_dims=keep_dims)
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        (
            source._logical_shape_from_components(replace_dims={dim_adj: 1})
            if keepdim
            else source._logical_shape_from_components(keep_dims=keep_dims)
        ),
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
    )


def _reduce_non_ragged_packed_dims(source: NestedTensor, out_values: Tensor, dims_adj: tuple[int, ...], keepdim: bool):
    r"""Wrap non-ragged multi-dim reductions on packed values as a NestedTensor."""
    dims_adj = tuple(sorted({int(d) for d in dims_adj}))
    if keepdim:
        out_shape, packed_sizes, element_shapes = source._shape_meta_from_components(
            replace_dims={int(d): 1 for d in dims_adj}
        )
        return _packed_with_shape(
            source,
            out_values,
            out_shape,
            source._logical_shape_from_components(replace_dims={int(d): 1 for d in dims_adj}),
            packed_sizes=packed_sizes,
            element_shapes=element_shapes,
        )

    keep_dims = tuple(i for i in range(source._physical_shape.size(1)) if i not in set(dims_adj))
    out_shape, packed_sizes, element_shapes = source._shape_meta_from_components(keep_dims=keep_dims)
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_components(keep_dims=keep_dims),
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
    )


def _packed_new_ragged_size(source: NestedTensor, new_values: Tensor, new_ragged_size) -> NestedTensor:
    r"""Rebuild a NestedTensor when per-element dim-0 size changes uniformly."""
    batch_size = source._offsets.size(0) - 1
    # Keep offsets on the same device as the source metadata (CPU by design).
    new_offsets = torch.arange(batch_size + 1, dtype=torch.long, device=source._offsets.device) * new_ragged_size
    new_physical_shape = source._physical_shape.clone()
    if new_physical_shape.numel() > 0:
        new_physical_shape[:, 0] = new_ragged_size
    return _packed_with_shape(
        source,
        new_values,
        new_physical_shape,
        source._logical_shape_from_components(replace_dims={0: int(new_ragged_size)}),
        offsets=new_offsets,
        permutation=source._permutation,
    )


def _packed_to_padded(source: NestedTensor, *, fill_value) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
    r"""Convert packed values [sum(L_i), ...] into padded [B, max(L_i), ...] plus gather indices."""
    lengths = source._offsets[1:] - source._offsets[:-1]
    device = source._values.device
    lengths_dev = lengths.to(device=device, dtype=torch.long)
    padded = source._materialize_batch_leading(fill_value)
    batch_idx, local_idx = source._packed_batch_local_indices(device=device)
    max_len = int(padded.size(1)) if padded.dim() > 1 else 0
    return padded, lengths, lengths_dev, batch_idx, local_idx, max_len


def _topk_fill_value(dtype: torch.dtype, largest: bool):
    if dtype.is_floating_point or dtype.is_complex:
        return float("-inf") if largest else float("inf")
    if dtype == torch.bool:
        return False if largest else True
    info = torch.iinfo(dtype)
    return info.min if largest else info.max


def _needs_masked_topk_scores(dtype: torch.dtype) -> bool:
    return (not dtype.is_floating_point) and (not dtype.is_complex)


@NestedTensorAtenRegistry.implement(aten.triu.default)
@NestedTensorAtenRegistry.implement(aten.tril.default)
@NestedTensorAtenRegistry.implement(aten.matrix_exp.default)
@NestedTensorAtenRegistry.implement(aten.inverse.default)
@NestedTensorAtenRegistry.implement(aten.matrix_power.default)
@NestedTensorAtenRegistry.implement(aten.linalg_inv.default)
@NestedTensorAtenRegistry.implement(aten.linalg_cholesky.default)
def _matrix_last2_unary(func, args, kwargs):
    r"""Apply matrix-style unary ops on packed values when ragged dim-0 is a batch axis."""
    source = args[0]
    if source._values.dim() <= 2:
        return _apply_per_element_nested(source, lambda t: func(t, *args[1:], **kwargs))
    return _packed_like(source, func(source._values, *args[1:], **kwargs))


@NestedTensorAtenRegistry.implement(aten.det.default)
@NestedTensorAtenRegistry.implement(aten.linalg_det.default)
def _matrix_last2_to_scalar(func, args, kwargs):
    r"""Apply determinant-like ops and drop trailing matrix dims in metadata."""
    source = args[0]
    if source._values.dim() <= 2:
        if source._physical_shape.size(1) != 2:
            return _apply_per_element_nested(source, lambda t: func(t, *args[1:], **kwargs))

        device = source.device
        batch = len(source)
        rows = source._physical_shape[:, 0].to(device=device, dtype=torch.long)
        cols = source._physical_shape[:, 1].to(device=device, dtype=torch.long)
        max_rows, max_cols = source._logical_physical_dims()

        padded = source._values.new_zeros((batch, max_rows, max_cols))
        if source._values.numel() > 0:
            padded[source._packed_dense_index(device=device)] = source._values

        row_coords = torch.arange(max_rows, device=device, dtype=torch.long).view(1, max_rows, 1)
        col_coords = torch.arange(max_cols, device=device, dtype=torch.long).view(1, 1, max_cols)
        inside = (row_coords < rows.view(batch, 1, 1)) & (col_coords < cols.view(batch, 1, 1))
        eye = torch.eye(max_rows, max_cols, dtype=source._values.dtype, device=device).expand(batch, -1, -1)
        values = func(torch.where(inside, padded, eye), *args[1:], **kwargs)
        return source._from_scalar_result_values(values)
    out_values = func(source._values, *args[1:], **kwargs)
    out_shape, packed_sizes, element_shapes = source._drop_trailing_physical_dims_meta(2)
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_components(keep_dims=tuple(range(max(source._physical_shape.size(1) - 2, 0)))),
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
    )


@NestedTensorAtenRegistry.implement(aten.diagonal.default)
def _diagonal(func, args, kwargs):
    r"""Apply ``diagonal`` on packed values when both selected dims are static per-element dims."""
    source = args[0]
    offset = args[1]
    dim1 = _normalize_dim(args[2], source.dim())
    dim2 = _normalize_dim(args[3], source.dim())
    batch_dim = _get_batch_dim(source)
    if dim1 == batch_dim or dim2 == batch_dim:
        raise ValueError("diagonal along the batch dimension is not supported for NestedTensor.")

    dim1_adj = _translate_dim(source, dim1)
    dim2_adj = _translate_dim(source, dim2)
    if dim1_adj == 0 or dim2_adj == 0:
        # Once the ragged leading dim participates in the diagonal, output lengths become
        # element-dependent again, so we intentionally stay on the per-element path.
        return _apply_per_element_nested(source, lambda t: func(t, offset, dim1_adj, dim2_adj, **kwargs))

    out_values = func(source._values, offset, dim1_adj, dim2_adj, **kwargs)
    return _packed_with_tail_from_values(source, out_values)


@NestedTensorAtenRegistry.implement(aten.trace.default)
def _trace(func, args, kwargs):
    r"""Apply ``trace`` per element to preserve the underlying 2-D tensor semantics."""
    source = args[0]
    return _apply_per_element_nested(source, lambda t: func(t, **kwargs))


# See also torch_functions.py::mm for the torch-level handler (mixed-type cases).
@NestedTensorAtenRegistry.implement(aten.mm.default)
def _mm(func, args, kwargs):
    r"""Dispatch handler for matrix multiply (NT x dense) on packed _values."""
    from .nested_tensor import NestedTensor

    mat1, mat2 = args[0], args[1]
    if isinstance(mat1, NestedTensor) and not isinstance(mat2, NestedTensor) and mat1._values.dim() == 2:
        new_values = func(mat1._values, mat2, **kwargs)
        return _packed_new_last_dim(mat1, new_values, mat2.shape[1])
    raise NotImplementedError(f"NestedTensor: {func} requires (NT × dense) with 2-D _values")


@NestedTensorAtenRegistry.implement(aten.addmm.default)
def _addmm(func, args, kwargs):
    r"""Dispatch handler for bias + matrix multiply (NT x dense) on packed _values."""
    from .nested_tensor import NestedTensor

    bias, mat1, mat2 = args[0], args[1], args[2]
    if isinstance(mat1, NestedTensor) and not isinstance(mat2, NestedTensor) and mat1._values.dim() == 2:
        new_values = func(bias, mat1._values, mat2, **kwargs)
        return _packed_new_last_dim(mat1, new_values, mat2.shape[1])
    raise NotImplementedError(f"NestedTensor: {func} requires (NT × dense) with 2-D _values")


# See also torch_functions.py::bmm for the torch-level handler (mismatched offsets).
@NestedTensorAtenRegistry.implement(aten.bmm.default)
def _bmm(func, args, kwargs):
    r"""Dispatch handler for batched matrix multiply between two NestedTensors."""
    from .nested_tensor import NestedTensor

    mat1, mat2 = args[0], args[1]
    if isinstance(mat1, NestedTensor) and isinstance(mat2, NestedTensor) and _structure_match(mat1, mat2):
        new_values = func(mat1._values, mat2._values, **kwargs)
        return _packed_new_last_dim(mat1, new_values, mat2._values.shape[-1])
    raise NotImplementedError(f"NestedTensor: {func} requires two NTs with matching packed structure")


@NestedTensorAtenRegistry.implement(aten.matmul.default)
def _matmul(func, args, kwargs):
    r"""Dispatch handler for matmul with packed fast paths when ragged dim remains element-local."""
    from .nested_tensor import NestedTensor

    lhs, rhs = args[0], args[1]

    if isinstance(lhs, NestedTensor):
        if isinstance(rhs, NestedTensor):
            if _structure_match(lhs, rhs) and lhs._values.dim() > 2 and rhs._values.dim() > 2:
                return _packed_with_tail_from_values(lhs, func(lhs._values, rhs._values, **kwargs))
            return per_element_fallback(func, args, kwargs)
        if isinstance(rhs, Tensor) and lhs._values.dim() >= 2:
            return _packed_with_tail_from_values(lhs, func(lhs._values, rhs, **kwargs))
        return per_element_fallback(func, args, kwargs)

    if isinstance(rhs, NestedTensor):
        if isinstance(lhs, Tensor) and rhs._values.dim() >= 2:
            return _packed_with_tail_from_values(rhs, func(lhs, rhs._values, **kwargs))
        return per_element_fallback(func, args, kwargs)

    return func(*args, **kwargs)


@NestedTensorAtenRegistry.implement(aten.linalg_qr.default)
def _linalg_qr(func, args, kwargs):
    r"""Dispatch handler for linalg_qr with packed fast path for matrix-batched packed values."""
    source = args[0]
    cls = type(source)
    if source._values.dim() <= 2:
        if len(source) == 0:
            empty = cls([], **source._meta(include_dtype=True))
            return empty, empty
        q_list, r_list = [], []
        for t in source._storage:
            q, r = func(t, *args[1:], **kwargs)
            q_list.append(q)
            r_list.append(r)
        return cls(q_list, **source._meta()), cls(r_list, **source._meta())

    q_values, r_values = func(source._values, *args[1:], **kwargs)
    q_shape, q_packed_sizes, q_element_shapes = source._replace_trailing_physical_dims_meta(q_values.shape[-2:])
    r_shape, r_packed_sizes, r_element_shapes = source._replace_trailing_physical_dims_meta(r_values.shape[-2:])
    return (
        _packed_with_shape(
            source,
            q_values,
            q_shape,
            source._logical_shape_from_components(
                replace_dims={
                    max(source._physical_shape.size(1) - 2, 0): int(q_values.shape[-2]),
                    max(source._physical_shape.size(1) - 1, 0): int(q_values.shape[-1]),
                }
            ),
            permutation=source._permutation,
            packed_sizes=q_packed_sizes,
            element_shapes=q_element_shapes,
        ),
        _packed_with_shape(
            source,
            r_values,
            r_shape,
            source._logical_shape_from_components(
                replace_dims={
                    max(source._physical_shape.size(1) - 2, 0): int(r_values.shape[-2]),
                    max(source._physical_shape.size(1) - 1, 0): int(r_values.shape[-1]),
                }
            ),
            permutation=source._permutation,
            packed_sizes=r_packed_sizes,
            element_shapes=r_element_shapes,
        ),
    )


@NestedTensorAtenRegistry.implement(aten.linalg_eigh.default)
@NestedTensorAtenRegistry.implement(aten._linalg_eigh.default)
def _linalg_eigh(func, args, kwargs):
    r"""Dispatch handler for linalg_eigh with packed fast path for matrix-batched packed values."""
    source = args[0]
    cls = type(source)
    if source._values.dim() <= 2:
        if len(source) == 0:
            empty = cls([], **source._meta(include_dtype=True))
            return empty, empty
        eigvals_list, eigvecs_list = [], []
        for t in source._storage:
            eigvals, eigvecs = func(t, *args[1:], **kwargs)
            eigvals_list.append(eigvals)
            eigvecs_list.append(eigvecs)
        return cls(eigvals_list, **source._meta()), cls(eigvecs_list, **source._meta())

    eigvals_values, eigvecs_values = func(source._values, *args[1:], **kwargs)
    eigvals_shape, eigvals_packed_sizes, eigvals_element_shapes = source._drop_trailing_physical_dims_meta(1)
    return (
        _packed_with_shape(
            source,
            eigvals_values,
            eigvals_shape,
            source._logical_shape_from_components(keep_dims=tuple(range(max(source._physical_shape.size(1) - 1, 0)))),
            permutation=source._permutation_after_dropping_trailing_dims(1),
            packed_sizes=eigvals_packed_sizes,
            element_shapes=eigvals_element_shapes,
        ),
        _packed_like(source, eigvecs_values),
    )


@NestedTensorAtenRegistry.implement(aten.linalg_svd.default)
@NestedTensorAtenRegistry.implement(aten._linalg_svd.default)
def _linalg_svd(func, args, kwargs):
    r"""Dispatch handler for linalg_svd with packed fast path for matrix-batched packed values."""
    source = args[0]
    cls = type(source)
    if source._values.dim() <= 2:
        if len(source) == 0:
            empty = cls([], **source._meta(include_dtype=True))
            return empty, empty, empty
        u_list, s_list, vh_list = [], [], []
        for t in source._storage:
            u, s, vh = func(t, *args[1:], **kwargs)
            u_list.append(u)
            s_list.append(s)
            vh_list.append(vh)
        meta = source._meta()
        return cls(u_list, **meta), cls(s_list, **meta), cls(vh_list, **meta)

    u_values, s_values, vh_values = func(source._values, *args[1:], **kwargs)
    k = int(s_values.shape[-1])
    u_shape, u_packed_sizes, u_element_shapes = source._replace_trailing_physical_dims_meta(u_values.shape[-2:])
    vh_shape, vh_packed_sizes, vh_element_shapes = source._replace_trailing_physical_dims_meta(vh_values.shape[-2:])
    s_shape, s_packed_sizes, s_element_shapes = source._drop_trailing_physical_dims_meta(2, suffix=(k,))
    return (
        _packed_with_shape(
            source,
            u_values,
            u_shape,
            source._logical_shape_from_components(
                replace_dims={
                    max(source._physical_shape.size(1) - 2, 0): int(u_values.shape[-2]),
                    max(source._physical_shape.size(1) - 1, 0): int(u_values.shape[-1]),
                }
            ),
            permutation=source._permutation,
            packed_sizes=u_packed_sizes,
            element_shapes=u_element_shapes,
        ),
        _packed_with_shape(
            source,
            s_values,
            s_shape,
            source._logical_shape_from_components(
                keep_dims=tuple(range(max(source._physical_shape.size(1) - 2, 0))),
                suffix=(k,),
            ),
            permutation=source._permutation_after_replacing_trailing_dims(2, 1),
            packed_sizes=s_packed_sizes,
            element_shapes=s_element_shapes,
        ),
        _packed_with_shape(
            source,
            vh_values,
            vh_shape,
            source._logical_shape_from_components(
                replace_dims={
                    max(source._physical_shape.size(1) - 2, 0): int(vh_values.shape[-2]),
                    max(source._physical_shape.size(1) - 1, 0): int(vh_values.shape[-1]),
                }
            ),
            permutation=source._permutation,
            packed_sizes=vh_packed_sizes,
            element_shapes=vh_element_shapes,
        ),
    )


@NestedTensorAtenRegistry.implement(aten.linalg_solve.default)
def _linalg_solve(func, args, kwargs):
    r"""Dispatch handler for linalg_solve with packed fast paths when ragged dim remains element-local."""
    from .nested_tensor import NestedTensor

    mat_a, mat_b = args[0], args[1]
    if isinstance(mat_a, NestedTensor):
        if isinstance(mat_b, NestedTensor):
            if len(mat_a) != len(mat_b):
                raise ValueError(
                    "linalg.solve: NestedTensor batch length mismatch between input and B: "
                    f"input={len(mat_a)}, B={len(mat_b)}"
                )
            if _structure_match(mat_a, mat_b) and mat_a._values.dim() > 2 and mat_b._values.dim() > 1:
                return _packed_with_tail_from_values(mat_a, func(mat_a._values, mat_b._values, *args[2:], **kwargs))
            return per_element_fallback(func, args, kwargs)
        if isinstance(mat_b, Tensor) and mat_a._values.dim() > 2:
            return _packed_with_tail_from_values(mat_a, func(mat_a._values, mat_b, *args[2:], **kwargs))
        return per_element_fallback(func, args, kwargs)

    if isinstance(mat_b, NestedTensor):
        if isinstance(mat_a, Tensor) and mat_a.dim() == 2 and mat_b._values.dim() > 1:
            return _packed_with_tail_from_values(mat_b, func(mat_a, mat_b._values, *args[2:], **kwargs))
        return per_element_fallback(func, args, kwargs)

    return func(*args, **kwargs)


@NestedTensorAtenRegistry.implement(aten._linalg_solve_ex.default)
def _linalg_solve_ex(func, args, kwargs):
    r"""Dispatch handler for ``_linalg_solve_ex`` to support ``linalg_solve`` decomposition paths."""
    from .nested_tensor import NestedTensor

    mat_a, mat_b = args[0], args[1]
    if isinstance(mat_a, NestedTensor):
        if isinstance(mat_b, NestedTensor):
            if len(mat_a) != len(mat_b):
                raise ValueError(
                    "linalg.solve: NestedTensor batch length mismatch between input and B: "
                    f"input={len(mat_a)}, B={len(mat_b)}"
                )
            if _structure_match(mat_a, mat_b) and mat_a._values.dim() > 2 and mat_b._values.dim() > 1:
                result, lu, pivots, info = func(mat_a._values, mat_b._values, *args[2:], **kwargs)
                return (
                    _packed_with_tail_from_values(mat_a, result),
                    _packed_like(mat_a, lu),
                    _packed_with_tail_from_values(mat_a, pivots),
                    _packed_with_tail_from_values(mat_a, info),
                )
            return per_element_fallback(func, args, kwargs)
        if isinstance(mat_b, Tensor) and mat_a._values.dim() > 2:
            result, lu, pivots, info = func(mat_a._values, mat_b, *args[2:], **kwargs)
            return (
                _packed_with_tail_from_values(mat_a, result),
                _packed_like(mat_a, lu),
                _packed_with_tail_from_values(mat_a, pivots),
                _packed_with_tail_from_values(mat_a, info),
            )
        return per_element_fallback(func, args, kwargs)

    if isinstance(mat_b, NestedTensor):
        return per_element_fallback(func, args, kwargs)

    return func(*args, **kwargs)


@NestedTensorAtenRegistry.implement(aten._linalg_check_errors.default)
def _linalg_check_errors(func, args, kwargs):
    r"""Dispatch handler for ``_linalg_check_errors`` that preserves its ``None`` return contract."""
    from .nested_tensor import NestedTensor

    info = args[0]
    if isinstance(info, NestedTensor):
        func(info._values, *args[1:], **kwargs)
        return None
    return func(*args, **kwargs)


# ---------------------------------------------------------------------------
# Sorting / cumulative / reordering ops.
# Fast path: N-D packing with non-ragged target dims (operate on _values).
# Fallback: 1-D packing or ragged dim -> per-element path.
# ---------------------------------------------------------------------------


@NestedTensorAtenRegistry.implement(aten.sort.default)
@NestedTensorAtenRegistry.implement(aten.sort.stable)
def _sort(func, args, kwargs):
    r"""Sort along a non-ragged dim by operating directly on packed _values."""
    source = args[0]
    stable_overload = func is aten.sort.stable
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_descending = kwargs.pop("descending", _MISSING)
    kw_stable = kwargs.pop("stable", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError("sort() got multiple values for argument 'dim'")
        dim = args[1]
    else:
        dim = -1 if kw_dim is _MISSING else kw_dim
    if len(args) > 2:
        if kw_descending is not _MISSING:
            raise TypeError("sort() got multiple values for argument 'descending'")
        descending = args[2]
    else:
        descending = False if kw_descending is _MISSING else kw_descending
    if len(args) > 3:
        if kw_stable is not _MISSING:
            raise TypeError("sort() got multiple values for argument 'stable'")
        stable = args[3]
    else:
        if kw_stable is _MISSING:
            stable = True if stable_overload else None
        else:
            stable = kw_stable

    def _call_sort(tensor: Tensor, dim_value: int):
        if stable_overload or stable is not None:
            return torch.ops.aten.sort.stable(tensor, stable=stable, dim=dim_value, descending=descending, **kwargs)
        return func(tensor, dim_value, descending, **kwargs)

    dim_adj = _translate_dim(source, dim)
    if source._values.dim() > 1 and dim_adj > 0:
        vals, idxs = _call_sort(source._values, dim_adj)
        return _packed_like(source, vals), _packed_like(source, idxs)
    if dim_adj == 0:
        fill_value = _topk_fill_value(source._values.dtype, largest=descending)
        padded, _, _, batch_idx, local_idx, _ = _packed_to_padded(source, fill_value=fill_value)
        vals, idxs = _call_sort(padded, 1)
        return _packed_like(source, vals[batch_idx, local_idx]), _packed_like(source, idxs[batch_idx, local_idx])
    if stable_overload or stable is not None:
        return per_element_fallback(
            torch.ops.aten.sort.stable,
            (source,),
            {"stable": stable, "dim": dim_adj, "descending": descending, **kwargs},
        )
    return per_element_fallback(func, (source, dim_adj, descending), kwargs)


@NestedTensorAtenRegistry.implement(aten.argsort.default)
@NestedTensorAtenRegistry.implement(aten.argsort.stable)
def _argsort(func, args, kwargs):
    r"""Return sort indices along a non-ragged dim by operating on packed _values."""
    source = args[0]
    stable_overload = func is aten.argsort.stable
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_descending = kwargs.pop("descending", _MISSING)
    kw_stable = kwargs.pop("stable", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError("argsort() got multiple values for argument 'dim'")
        dim = args[1]
    else:
        dim = -1 if kw_dim is _MISSING else kw_dim
    if len(args) > 2:
        if kw_descending is not _MISSING:
            raise TypeError("argsort() got multiple values for argument 'descending'")
        descending = args[2]
    else:
        descending = False if kw_descending is _MISSING else kw_descending
    if len(args) > 3:
        if kw_stable is not _MISSING:
            raise TypeError("argsort() got multiple values for argument 'stable'")
        stable = args[3]
    else:
        if kw_stable is _MISSING:
            stable = True if stable_overload else None
        else:
            stable = kw_stable

    def _call_argsort(tensor: Tensor, dim_value: int):
        if stable_overload or stable is not None:
            return torch.ops.aten.argsort.stable(
                tensor, stable=bool(stable), dim=dim_value, descending=descending, **kwargs
            )
        return func(tensor, dim_value, descending, **kwargs)

    dim_adj = _translate_dim(source, dim)
    if source._values.dim() > 1 and dim_adj > 0:
        return _packed_like(source, _call_argsort(source._values, dim_adj))
    if dim_adj == 0:
        fill_value = _topk_fill_value(source._values.dtype, largest=descending)
        padded, _, _, batch_idx, local_idx, _ = _packed_to_padded(source, fill_value=fill_value)
        idxs = _call_argsort(padded, 1)
        return _packed_like(source, idxs[batch_idx, local_idx])
    if stable_overload or stable is not None:
        return per_element_fallback(
            torch.ops.aten.argsort.stable,
            (source,),
            {"stable": bool(stable), "dim": dim_adj, "descending": descending, **kwargs},
        )
    return per_element_fallback(func, (source, dim_adj, descending), kwargs)


@NestedTensorAtenRegistry.implement(aten.topk.default)
def _topk(func, args, kwargs):
    r"""Compute top-k along a non-ragged dim by operating on packed _values."""
    source = args[0]
    kw_k = kwargs.pop("k", _MISSING)
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_largest = kwargs.pop("largest", _MISSING)
    kw_sorted = kwargs.pop("sorted", _MISSING)
    if len(args) > 1:
        if kw_k is not _MISSING:
            raise TypeError("topk() got multiple values for argument 'k'")
        k = args[1]
    else:
        if kw_k is _MISSING:
            raise TypeError("topk() missing required argument 'k'")
        k = kw_k
    if len(args) > 2:
        if kw_dim is not _MISSING:
            raise TypeError("topk() got multiple values for argument 'dim'")
        dim = args[2]
    else:
        dim = -1 if kw_dim is _MISSING else kw_dim
    if len(args) > 3:
        if kw_largest is not _MISSING:
            raise TypeError("topk() got multiple values for argument 'largest'")
        largest = args[3]
    else:
        largest = True if kw_largest is _MISSING else kw_largest
    if len(args) > 4:
        if kw_sorted is not _MISSING:
            raise TypeError("topk() got multiple values for argument 'sorted'")
        sorted_output = args[4]
    else:
        sorted_output = True if kw_sorted is _MISSING else kw_sorted
    dim_adj = _translate_dim(source, dim)
    if source._values.dim() > 1 and dim_adj > 0:
        vals, idxs = func(source._values, k, dim_adj, largest, sorted_output, **kwargs)
        return (
            _packed_new_dim_size(source, vals, dim_adj, k),
            _packed_new_dim_size(source, idxs, dim_adj, k),
        )
    if dim_adj == 0:
        padded, lengths, lengths_dev, batch_idx, local_idx, max_len = _packed_to_padded(
            source, fill_value=_topk_fill_value(source._values.dtype, largest)
        )
        if lengths.numel() == 0:
            return _packed_like(source, source._values), _packed_like(source, source._values.to(dtype=torch.long))

        k_value = int(k)
        if not _is_fake_tensor(source._values):
            min_len = int(lengths.min().item())
            if k_value > min_len:
                raise ValueError(
                    f"NestedTensor topk along ragged dim requires k <= min segment length, "
                    f"but got k={k_value}, min={min_len}"
                )

        if _needs_masked_topk_scores(source._values.dtype):
            valid_mask = torch.arange(max_len, device=source._values.device, dtype=torch.long).unsqueeze(0)
            valid_mask = valid_mask < lengths_dev.unsqueeze(1)
            while valid_mask.dim() < padded.dim():
                valid_mask = valid_mask.unsqueeze(-1)
            score_dtype = torch.float64
            scores = padded.to(dtype=score_dtype)
            fill_score = torch.full(
                (),
                float("-inf") if largest else float("inf"),
                dtype=score_dtype,
                device=scores.device,
            )
            scores = torch.where(valid_mask, scores, fill_score)
            _, idxs = func(scores, k_value, 1, largest, sorted_output, **kwargs)
            vals = torch.gather(padded, 1, idxs)
        else:
            vals, idxs = func(padded, k_value, 1, largest, sorted_output, **kwargs)

        vals_packed = vals.reshape(-1, *vals.shape[2:])
        idxs_packed = idxs.reshape(-1, *idxs.shape[2:])
        return (
            _packed_new_ragged_size(source, vals_packed, k_value),
            _packed_new_ragged_size(source, idxs_packed, k_value),
        )
    return per_element_fallback(func, (source, k, dim_adj, largest, sorted_output), kwargs)


@NestedTensorAtenRegistry.implement(aten.cumsum.default)
@NestedTensorAtenRegistry.implement(aten.cumprod.default)
@NestedTensorAtenRegistry.implement(aten.logcumsumexp.default)
def _cumulative(func, args, kwargs):
    r"""Apply cumulative ops on packed _values when the target dim is static."""
    source = args[0]
    kw_dim = kwargs.pop("dim", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError(f"{func._schema.name.split('::')[-1]}() got multiple values for argument 'dim'")
        dim = args[1]
    else:
        if kw_dim is _MISSING:
            raise TypeError(f"{func._schema.name.split('::')[-1]}() missing required argument 'dim'")
        dim = kw_dim
    dim_adj = _translate_dim(source, dim)
    extra_args = args[2:] if len(args) > 2 else ()
    if source._values.dim() > 1 and dim_adj > 0:
        return _packed_like(source, func(source._values, dim_adj, *extra_args, **kwargs))
    if dim_adj == 0:
        if func is aten.cumsum.default:
            neutral = 0
        elif func is aten.cumprod.default:
            neutral = 1
        else:
            # logcumsumexp identity: log(0) == -inf
            neutral = float("-inf")
        padded, _, _, batch_idx, local_idx, _ = _packed_to_padded(source, fill_value=neutral)
        out_padded = func(padded, 1, *extra_args, **kwargs)
        return _packed_like(source, out_padded[batch_idx, local_idx])
    return per_element_fallback(func, (source, dim_adj, *extra_args), kwargs)


@NestedTensorAtenRegistry.implement(aten.cummax.default)
@NestedTensorAtenRegistry.implement(aten.cummin.default)
def _cumulative_pair(func, args, kwargs):
    r"""Apply cumulative pair ops (cummax/cummin) on packed _values."""
    source = args[0]
    kw_dim = kwargs.pop("dim", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError(f"{func._schema.name.split('::')[-1]}() got multiple values for argument 'dim'")
        dim = args[1]
    else:
        if kw_dim is _MISSING:
            raise TypeError(f"{func._schema.name.split('::')[-1]}() missing required argument 'dim'")
        dim = kw_dim
    dim_adj = _translate_dim(source, dim)
    if source._values.dim() > 1 and dim_adj > 0:
        vals, idxs = func(source._values, dim_adj, **kwargs)
        return _packed_like(source, vals), _packed_like(source, idxs)
    if dim_adj == 0:
        largest = func is aten.cummax.default
        fill_value = _topk_fill_value(source._values.dtype, largest=largest)
        padded, _, _, batch_idx, local_idx, _ = _packed_to_padded(source, fill_value=fill_value)
        vals, idxs = func(padded, 1, **kwargs)
        return _packed_like(source, vals[batch_idx, local_idx]), _packed_like(source, idxs[batch_idx, local_idx])
    return per_element_fallback(func, (source, dim_adj), kwargs)


@NestedTensorAtenRegistry.implement(aten.flip.default)
def _flip(func, args, kwargs):
    r"""Flip along non-ragged dims by operating directly on packed _values."""
    source = args[0]
    kw_dims = kwargs.pop("dims", _MISSING)
    if len(args) > 1:
        if kw_dims is not _MISSING:
            raise TypeError("flip() got multiple values for argument 'dims'")
        dims = args[1]
    else:
        dims = () if kw_dims is _MISSING else kw_dims
    if isinstance(dims, int):
        dims = (dims,)
    dims_adj = tuple(_translate_dim(source, dim) for dim in dims)
    if source._values.dim() > 1 and all(dim > 0 for dim in dims_adj):
        return _packed_like(source, func(source._values, dims_adj, **kwargs))
    if any(dim == 0 for dim in dims_adj):
        padded, _, lengths_dev, batch_idx, local_idx, max_len = _packed_to_padded(
            source, fill_value=source.padding_value
        )
        padded_dims = tuple(1 if dim == 0 else dim + 1 for dim in dims_adj)
        out_padded = func(padded, padded_dims, **kwargs)
        ragged_flips = sum(dim == 0 for dim in dims_adj)
        if ragged_flips % 2 == 1:
            row_idx = max_len - lengths_dev[batch_idx] + local_idx
        else:
            row_idx = local_idx
        return _packed_like(source, out_padded[batch_idx, row_idx])
    return per_element_fallback(func, (source, dims_adj), kwargs)


@NestedTensorAtenRegistry.implement(aten.roll.default)
def _roll(func, args, kwargs):
    r"""Roll along non-ragged dims on packed values; fallback for ragged/flatten cases."""
    source = args[0]
    kw_shifts = kwargs.pop("shifts", _MISSING)
    kw_dims = kwargs.pop("dims", _MISSING)
    if len(args) > 1:
        if kw_shifts is not _MISSING:
            raise TypeError("roll() got multiple values for argument 'shifts'")
        shifts = args[1]
    else:
        if kw_shifts is _MISSING:
            raise TypeError("roll() missing required argument 'shifts'")
        shifts = kw_shifts
    if len(args) > 2:
        if kw_dims is not _MISSING:
            raise TypeError("roll() got multiple values for argument 'dims'")
        dims = args[2]
    else:
        dims = () if kw_dims is _MISSING else kw_dims
    if isinstance(shifts, int):
        shifts = [shifts]
    else:
        shifts = list(shifts)

    if isinstance(dims, int):
        dims = (dims,)
    else:
        dims = tuple(dims)

    # dims=[] (or omitted) follows torch.roll flatten semantics per element.
    if len(dims) == 0:
        return per_element_fallback(func, (source, shifts, []), kwargs)

    dims_adj = tuple(_translate_dim(source, dim) for dim in dims)
    if source._values.dim() > 1 and all(dim > 0 for dim in dims_adj):
        return _packed_like(source, func(source._values, shifts, list(dims_adj), **kwargs))
    return per_element_fallback(func, (source, shifts, list(dims_adj)), kwargs)


@NestedTensorAtenRegistry.implement(aten.rot90.default)
def _rot90(func, args, kwargs):
    r"""Rotate over two non-ragged dims on packed values; fallback for ragged dims."""
    source = args[0]
    kw_k = kwargs.pop("k", _MISSING)
    kw_dims = kwargs.pop("dims", _MISSING)
    if len(args) > 1:
        if kw_k is not _MISSING:
            raise TypeError("rot90() got multiple values for argument 'k'")
        k = args[1]
    else:
        k = 1 if kw_k is _MISSING else kw_k
    if len(args) > 2:
        if kw_dims is not _MISSING:
            raise TypeError("rot90() got multiple values for argument 'dims'")
        dims = args[2]
    else:
        dims = (0, 1) if kw_dims is _MISSING else kw_dims

    dims = tuple(dims)
    if len(dims) != 2:
        raise ValueError("rot90 dims must be a sequence of two dimensions.")

    dim_count = source.dim()
    dims_norm = tuple(_normalize_dim(d, dim_count) for d in dims)
    dims_adj = tuple(_translate_dim(source, d) for d in dims_norm)

    if source._values.dim() > 1 and all(dim > 0 for dim in dims_adj):
        out_values = func(source._values, k, list(dims_adj), **kwargs)
        out_shape = source._physical_shape.clone()
        out_shape[:, [dims_adj[0], dims_adj[1]]] = out_shape[:, [dims_adj[1], dims_adj[0]]]
        out_logical = list(source._logical_shape)
        out_logical[dims_norm[0]], out_logical[dims_norm[1]] = out_logical[dims_norm[1]], out_logical[dims_norm[0]]
        return _packed_with_shape(source, out_values, out_shape, out_logical)

    return per_element_fallback(func, (source, k, list(dims_adj)), kwargs)


@NestedTensorAtenRegistry.implement(aten.searchsorted.Tensor)
def _searchsorted_tensor(func, args, kwargs):
    r"""Apply searchsorted with packed fastpaths for supported NestedTensor layouts."""
    from .nested_tensor import NestedTensor

    sorted_sequence, values = args[0], args[1]

    out_int32 = kwargs.pop("out_int32", False)
    right = kwargs.pop("right", False)
    side = kwargs.pop("side", None)
    sorter = kwargs.pop("sorter", None)
    sorted_is_nt = isinstance(sorted_sequence, NestedTensor)
    values_is_nt = isinstance(values, NestedTensor)
    sorter_is_nt = isinstance(sorter, NestedTensor)

    if sorter_is_nt and not sorted_is_nt:
        raise TypeError("searchsorted: NestedTensor sorter requires sorted_sequence to be a NestedTensor.")

    if sorted_is_nt and values_is_nt:
        if len(sorted_sequence) != len(values):
            raise ValueError(
                "searchsorted: NestedTensor batch length mismatch between sorted_sequence and values: "
                f"sorted_sequence={len(sorted_sequence)}, values={len(values)}"
            )
        offsets_match = (
            sorted_sequence._values.dim() >= 2
            and values._values.dim() >= 2
            and _offsets_match_identity_if_fake(sorted_sequence._offsets, values._offsets)
        )

        if offsets_match:
            sorter_ok = sorter is None
            sorter_values = None
            if sorter_is_nt:
                if len(sorter) != len(sorted_sequence):
                    raise ValueError(
                        "searchsorted: NestedTensor batch length mismatch between sorted_sequence and sorter: "
                        f"sorted_sequence={len(sorted_sequence)}, sorter={len(sorter)}"
                    )
                sorter_ok = _offsets_match_identity_if_fake(sorted_sequence._offsets, sorter._offsets)
                if sorter_ok:
                    sorter_values = sorter._values
            elif isinstance(sorter, Tensor):
                sorter_ok = True
                sorter_values = sorter
            else:
                sorter_ok = False
            if sorter_ok:
                out_values = func(
                    sorted_sequence._values,
                    values._values,
                    out_int32=out_int32,
                    right=right,
                    side=side,
                    sorter=sorter_values,
                    **kwargs,
                )
                return _packed_like(values, out_values)

        if sorter_is_nt:
            if len(sorter) != len(sorted_sequence):
                raise ValueError(
                    "searchsorted: NestedTensor batch length mismatch between sorted_sequence and sorter: "
                    f"sorted_sequence={len(sorted_sequence)}, sorter={len(sorter)}"
                )
            sorter_storage = sorter._storage
        elif sorter is None or isinstance(sorter, Tensor):
            sorter_storage = [sorter] * len(sorted_sequence)
        else:
            raise TypeError("searchsorted: sorter must be Tensor, NestedTensor, or None.")
        results = [
            torch.searchsorted(
                s,
                v,
                out_int32=out_int32,
                right=right,
                side=side,
                sorter=sort_i,
                **kwargs,
            )
            for s, v, sort_i in zip(sorted_sequence._storage, values._storage, sorter_storage)
        ]
        return type(values)(results, **values._meta())

    if values_is_nt:
        if sorter_is_nt:
            raise TypeError(
                "searchsorted: NestedTensor sorter is only supported when sorted_sequence is a NestedTensor."
            )
        if (
            isinstance(sorted_sequence, Tensor)
            and sorted_sequence.dim() <= 1
            and (sorter is None or isinstance(sorter, Tensor))
        ):
            out_values = func(
                sorted_sequence,
                values._values,
                out_int32=out_int32,
                right=right,
                side=side,
                sorter=sorter,
                **kwargs,
            )
            return _packed_like(values, out_values)

        results = [
            torch.searchsorted(
                sorted_sequence,
                v,
                out_int32=out_int32,
                right=right,
                side=side,
                sorter=sorter,
                **kwargs,
            )
            for v in values._storage
        ]
        return type(values)(results, **values._meta())

    if sorted_is_nt:
        if sorter_is_nt:
            if len(sorter) != len(sorted_sequence):
                raise ValueError(
                    "searchsorted: NestedTensor batch length mismatch between sorted_sequence and sorter: "
                    f"sorted_sequence={len(sorted_sequence)}, sorter={len(sorter)}"
                )
            sorter_storage = sorter._storage
        elif sorter is None or isinstance(sorter, Tensor):
            sorter_storage = [sorter] * len(sorted_sequence)
        else:
            raise TypeError("searchsorted: sorter must be Tensor, NestedTensor, or None.")
        results = [
            torch.searchsorted(
                s,
                values,
                out_int32=out_int32,
                right=right,
                side=side,
                sorter=sort_i,
                **kwargs,
            )
            for s, sort_i in zip(sorted_sequence._storage, sorter_storage)
        ]
        return type(sorted_sequence)(results, **sorted_sequence._meta())

    raise RuntimeError(
        "searchsorted: reached NestedTensor aten handler with neither sorted_sequence nor values as NestedTensor."
    )


# ---------------------------------------------------------------------------
# Indexing read ops — packed fast paths when index layouts align.
# ---------------------------------------------------------------------------


@NestedTensorAtenRegistry.implement(aten.gather.default)
def _gather(func, args, kwargs):
    r"""Apply ``gather`` with packed fast paths when the index layout matches the source."""
    from .nested_tensor import NestedTensor

    source = args[0]
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_index = kwargs.pop("index", _MISSING)
    kw_sparse_grad = kwargs.pop("sparse_grad", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError("gather() got multiple values for argument 'dim'")
        dim = args[1]
    else:
        if kw_dim is _MISSING:
            raise TypeError("gather() missing required argument 'dim'")
        dim = kw_dim
    if len(args) > 2:
        if kw_index is not _MISSING:
            raise TypeError("gather() got multiple values for argument 'index'")
        index = args[2]
    else:
        if kw_index is _MISSING:
            raise TypeError("gather() missing required argument 'index'")
        index = kw_index
    if len(args) > 3:
        if kw_sparse_grad is not _MISSING:
            raise TypeError("gather() got multiple values for argument 'sparse_grad'")
        sparse_grad = args[3]
    else:
        sparse_grad = False if kw_sparse_grad is _MISSING else kw_sparse_grad

    dim = _normalize_dim(dim, source.dim())
    batch_dim = _get_batch_dim(source)
    if dim == batch_dim:
        raise ValueError("gather along the batch dimension is not supported for NestedTensor.")
    dim_adj = _translate_dim(source, dim)

    aligned_index = source._maybe_exact_shape_nested_like(index)
    if aligned_index is not None:
        index = aligned_index

    if isinstance(index, NestedTensor):
        if len(source) != len(index):
            raise ValueError(
                "NestedTensor batch length mismatch between input and index: "
                f"input={len(source)}, index={len(index)}"
            )

        if dim_adj > 0 and source._values.dim() > 1 and _structure_match(source, index):
            out_values = func(source._values, dim_adj, index._values, sparse_grad=sparse_grad, **kwargs)
            return _packed_with_shape(source, out_values, index._physical_shape, index._logical_shape)

        if dim_adj == 0 and source._values.dim() > 1 and index._values.dim() > 1:
            padded, lengths, lengths_dev, _, _, _ = _packed_to_padded(source, fill_value=0)
            index_padded, _, _, batch_idx, local_idx, _ = _packed_to_padded(index, fill_value=0)

            if index._values.numel() > 0 and not (_is_fake_tensor(source._values) or _is_fake_tensor(index._values)):
                gather_lengths = lengths_dev[batch_idx]
                while gather_lengths.dim() < index._values.dim():
                    gather_lengths = gather_lengths.unsqueeze(-1)
                invalid = (index._values < 0) | (index._values >= gather_lengths)
                if bool(invalid.any().item()):
                    raise RuntimeError("index out of bounds for gather along the ragged dimension")

            out_padded = func(padded, 1, index_padded, sparse_grad=sparse_grad, **kwargs)
            return type(source)._from_packed(
                out_padded[batch_idx, local_idx],
                index._offsets,
                index._physical_shape,
                batch_first=source.batch_first,
                padding_value=source.padding_value,
                mask_value=source.mask_value,
                pin_memory=source._pin_memory,
                outer_size=torch.Size(index._logical_shape),
            )

        storage = []
        for tensor, idx in zip(source._storage, index._storage):
            if idx.device != tensor.device:
                idx = idx.to(device=tensor.device)
            storage.append(func(tensor, dim_adj, idx, sparse_grad=sparse_grad, **kwargs))
        return type(source)(storage, **source._meta())

    storage = []
    for tensor in source._storage:
        idx = index
        if isinstance(idx, Tensor) and idx.device != tensor.device:
            idx = idx.to(device=tensor.device)
        storage.append(func(tensor, dim_adj, idx, sparse_grad=sparse_grad, **kwargs))
    return type(source)(storage, **source._meta())


@NestedTensorAtenRegistry.implement(aten.take.default)
def _take(func, args, kwargs):
    r"""Apply ``take`` over the flattened packed storage for plain tensor indices."""
    source = args[0]
    index = args[1]
    from .nested_tensor import NestedTensor

    if not isinstance(index, Tensor) or isinstance(index, NestedTensor):
        raise NotImplementedError(f"NestedTensor: {func} requires a Tensor index")
    if index.device != source._values.device:
        index = index.to(device=source._values.device)
    return func(source._values.reshape(-1), index, **kwargs)


@NestedTensorAtenRegistry.implement(aten.index_select.default)
def _index_select(func, args, kwargs):
    r"""Apply ``index_select`` with packed fast paths for batch and static dims."""
    source = args[0]
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_index = kwargs.pop("index", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError("index_select() got multiple values for argument 'dim'")
        dim = args[1]
    else:
        if kw_dim is _MISSING:
            raise TypeError("index_select() missing required argument 'dim'")
        dim = kw_dim
    if len(args) > 2:
        if kw_index is not _MISSING:
            raise TypeError("index_select() got multiple values for argument 'index'")
        index = args[2]
    else:
        if kw_index is _MISSING:
            raise TypeError("index_select() missing required argument 'index'")
        index = kw_index

    dim = _normalize_dim(dim, source.dim())
    batch_dim = _get_batch_dim(source)
    index_cpu = index.to(device=source._offsets.device, dtype=torch.long)
    index_device = index.to(device=source._values.device, dtype=torch.long)

    if dim == batch_dim:
        lengths = source._offsets[1:] - source._offsets[:-1]
        out_shape = source._physical_shape.index_select(0, index_cpu)
        out_lengths = lengths.index_select(0, index_cpu)
        out_offsets = torch.empty(
            (out_lengths.numel() + 1,), dtype=source._offsets.dtype, device=source._offsets.device
        )
        out_offsets[0] = 0
        if out_lengths.numel() > 0:
            out_offsets[1:] = torch.cumsum(out_lengths, dim=0)

        offsets_dev = source._offsets.to(device=source._values.device, dtype=torch.long)
        lengths_dev = offsets_dev[1:] - offsets_dev[:-1]
        starts = offsets_dev.index_select(0, index_device)
        picked_lengths = lengths_dev.index_select(0, index_device)
        picked_offsets = torch.empty(
            (picked_lengths.numel() + 1,),
            dtype=torch.long,
            device=source._values.device,
        )
        picked_offsets[0] = 0
        if picked_lengths.numel() > 0:
            picked_offsets[1:] = torch.cumsum(picked_lengths, dim=0)
        flat = torch.arange(picked_offsets[-1], device=source._values.device, dtype=torch.long)
        if flat.numel() == 0:
            out_values = source._values[:0]
        else:
            batch_idx = torch.searchsorted(picked_offsets[1:], flat, right=True)
            local_idx = flat - picked_offsets.index_select(0, batch_idx)
            gather_idx = starts.index_select(0, batch_idx) + local_idx
            out_values = source._values.index_select(0, gather_idx)

        return type(source)._from_packed(
            out_values,
            out_offsets,
            out_shape,
            batch_first=source.batch_first,
            padding_value=source.padding_value,
            mask_value=source.mask_value,
            pin_memory=source._pin_memory,
        )

    dim_adj = _translate_dim(source, dim)
    if dim_adj > 0 and source._values.dim() > dim_adj:
        out_values = func(source._values, dim_adj, index_device, **kwargs)
        return _packed_new_dim_size(source, out_values, dim_adj, index_device.numel())
    return per_element_fallback(func, (source, dim_adj, index_device), kwargs)


def _scatter_like(source, dim, index, src, apply_fn, op_name: str):
    r"""Apply scatter-style writes with a packed fast path for matching-offset static dims."""
    from .nested_tensor import NestedTensor

    dim = _normalize_dim(dim, source.dim())
    batch_dim = _get_batch_dim(source)
    if dim == batch_dim:
        raise ValueError(f"{op_name} along the batch dimension is not supported for NestedTensor.")
    dim_adj = _translate_dim(source, dim)

    aligned_index = source._maybe_exact_shape_nested_like(index)
    if aligned_index is not None:
        index = aligned_index
    aligned_src = source._maybe_exact_shape_nested_like(src)
    if aligned_src is not None:
        src = aligned_src

    if isinstance(index, NestedTensor) and len(source) != len(index):
        raise ValueError(
            f"{op_name}: NestedTensor batch length mismatch between input and index: "
            f"input={len(source)}, index={len(index)}"
        )
    if isinstance(src, NestedTensor) and len(source) != len(src):
        raise ValueError(
            f"{op_name}: NestedTensor batch length mismatch between input and source: "
            f"input={len(source)}, source={len(src)}"
        )

    if isinstance(index, NestedTensor) and dim_adj > 0 and source._values.dim() > dim_adj:
        # Writes along the ragged leading dim are not packed-safe: a padded fallback would let
        # invalid rows participate in the write. Restrict the fast path to static per-element dims.
        index_values = index._values.to(device=source._values.device, dtype=torch.long)
        if _structure_match(source, index):
            if isinstance(src, NestedTensor):
                if _structure_match(source, src):
                    src_values = src._values
                    if src_values.device != source._values.device:
                        src_values = src_values.to(device=source._values.device)
                    return _packed_like(source, apply_fn(source._values, dim_adj, index_values, src_values))
            else:
                src_values = src
                if isinstance(src_values, Tensor) and src_values.device != source._values.device:
                    src_values = src_values.to(device=source._values.device)
                return _packed_like(source, apply_fn(source._values, dim_adj, index_values, src_values))

    storage = []
    if isinstance(index, NestedTensor):
        indices = index._storage
    else:
        indices = tuple(index for _ in source._storage)
    if isinstance(src, NestedTensor):
        srcs = src._storage
    else:
        srcs = tuple(src for _ in source._storage)

    for tensor, idx, src_i in zip(source._storage, indices, srcs):
        if isinstance(idx, Tensor):
            idx = idx.to(device=tensor.device, dtype=torch.long)
        if isinstance(src_i, Tensor) and src_i.device != tensor.device:
            src_i = src_i.to(device=tensor.device)
        storage.append(apply_fn(tensor, dim_adj, idx, src_i))
    return type(source)(storage, **source._meta())


def _index_write_like(source, dim, index, src, apply_fn, op_name: str):
    r"""Apply index-write ops with a packed fast path for Tensor indices on static per-element dims."""
    from .nested_tensor import NestedTensor

    dim = _normalize_dim(dim, source.dim())
    batch_dim = _get_batch_dim(source)
    if dim == batch_dim:
        raise ValueError(f"{op_name} along the batch dimension is not supported for NestedTensor.")
    dim_adj = _translate_dim(source, dim)

    aligned_src = source._maybe_exact_shape_nested_like(src)
    if aligned_src is not None:
        src = aligned_src

    if not isinstance(index, Tensor) or isinstance(index, NestedTensor):
        raise NotImplementedError(f"NestedTensor: {op_name} requires a Tensor index")
    if isinstance(src, NestedTensor) and len(source) != len(src):
        raise ValueError(
            f"{op_name}: NestedTensor batch length mismatch between input and source: "
            f"input={len(source)}, source={len(src)}"
        )

    index_values = index.to(device=source._values.device, dtype=torch.long)
    if isinstance(src, NestedTensor) and dim_adj > 0 and source._values.dim() > dim_adj:
        # As with scatter, packed writes are only safe on static per-element dims.
        # The source layout must share offsets with the destination so row boundaries
        # remain aligned after concatenation.
        if _structure_match(source, src):
            src_values = src._values
            if src_values.device != source._values.device:
                src_values = src_values.to(device=source._values.device)
            return _packed_like(source, apply_fn(source._values, dim_adj, index_values, src_values))

    storage = []
    if isinstance(src, NestedTensor):
        srcs = src._storage
    else:
        srcs = tuple(src for _ in source._storage)
    for tensor, src_i in zip(source._storage, srcs):
        idx = index
        if idx.device != tensor.device:
            idx = idx.to(device=tensor.device)
        if isinstance(src_i, Tensor) and src_i.device != tensor.device:
            src_i = src_i.to(device=tensor.device)
        storage.append(apply_fn(tensor, dim_adj, idx, src_i))
    return type(source)(storage, **source._meta())


@NestedTensorAtenRegistry.implement(aten.scatter.src)
def _scatter_src(func, args, kwargs):
    r"""Apply ``scatter`` with Tensor/NestedTensor src via packed fast paths when layouts align."""
    source, dim, index, src = args[0], args[1], args[2], args[3]
    return _scatter_like(source, dim, index, src, lambda t, d, i, s: func(t, d, i, s, **kwargs), "scatter")


@NestedTensorAtenRegistry.implement(aten.scatter.value)
def _scatter_value(func, args, kwargs):
    r"""Apply scalar ``scatter`` with packed fast paths when index layouts align."""
    source, dim, index, value = args[0], args[1], args[2], args[3]
    return _scatter_like(source, dim, index, value, lambda t, d, i, s: func(t, d, i, s, **kwargs), "scatter")


@NestedTensorAtenRegistry.implement(aten.scatter_add.default)
def _scatter_add(func, args, kwargs):
    r"""Apply ``scatter_add`` with packed fast paths when index/source layouts align."""
    source, dim, index, src = args[0], args[1], args[2], args[3]
    return _scatter_like(source, dim, index, src, lambda t, d, i, s: func(t, d, i, s, **kwargs), "scatter_add")


@NestedTensorAtenRegistry.implement(aten.index_add.default)
def _index_add(func, args, kwargs):
    r"""Apply ``index_add`` with packed fast paths when the source layout aligns with the input."""
    source, dim, index, src = args[0], args[1], args[2], args[3]
    return _index_write_like(source, dim, index, src, lambda t, d, i, s: func(t, d, i, s, **kwargs), "index_add")


@NestedTensorAtenRegistry.implement(aten.index_copy.default)
def _index_copy(func, args, kwargs):
    r"""Apply ``index_copy`` with packed fast paths when the source layout aligns with the input."""
    source, dim, index, src = args[0], args[1], args[2], args[3]
    return _index_write_like(source, dim, index, src, lambda t, d, i, s: func(t, d, i, s, **kwargs), "index_copy")


def _is_integer_index_tensor(index) -> bool:
    from .nested_tensor import NestedTensor

    return (
        isinstance(index, Tensor)
        and not isinstance(index, NestedTensor)
        and index.dtype != torch.bool
        and not index.is_floating_point()
        and not index.is_complex()
    )


def _packed_index_put_indices(source: NestedTensor, indices):
    r"""
    Build flattened packed indices for shared advanced indexing over a leading dim prefix.

    Packed ``index_put`` is only safe when every element uses the same broadcasted index
    pattern. The ragged leading dim is offset per element; later indexed dims must already
    be static in the packed layout.
    """
    if source._physical_shape.size(1) == 0:
        return None
    if source._physical_shape.size(1) > 1 and source._values.dim() == 1:
        return None
    if not indices or len(indices) > source._physical_shape.size(1):
        return None
    if any(not _is_integer_index_tensor(idx) for idx in indices):
        return None

    device = source._values.device
    index_tensors = [idx.to(device=device, dtype=torch.long) for idx in indices]
    try:
        broadcasted = torch.broadcast_tensors(*index_tensors)
    except RuntimeError:
        return None

    batch = len(source)
    shape = broadcasted[0].shape
    expand_shape = (batch,) + shape
    packed_indices = []

    offsets = source._offsets.to(device=device, dtype=torch.long)
    lengths = offsets[1:] - offsets[:-1]
    view_shape = (batch,) + (1,) * len(shape)

    first = broadcasted[0].unsqueeze(0).expand(*expand_shape)
    first = torch.where(first < 0, first + lengths.view(*view_shape), first)
    if not torch.logical_and(first >= 0, first < lengths.view(*view_shape)).all():
        return None
    packed_indices.append((first + offsets[:-1].view(*view_shape)).reshape(-1))

    for dim, idx in enumerate(broadcasted[1:], start=1):
        dim_size = source._values.shape[dim]
        idx = idx.unsqueeze(0).expand(*expand_shape)
        idx = torch.where(idx < 0, idx + dim_size, idx)
        if not torch.logical_and(idx >= 0, idx < dim_size).all():
            return None
        packed_indices.append(idx.reshape(-1))

    return packed_indices, shape


def _packed_index_put_values(source: NestedTensor, values, broadcast_shape, indexed_dims: int):
    r"""Prepare values for the packed ``index_put`` fast path."""
    from .nested_tensor import NestedTensor

    trailing_shape = tuple(source._values.shape[indexed_dims:])
    batch = len(source)
    expanded_items = batch * math.prod(broadcast_shape)

    if isinstance(values, Tensor) and not isinstance(values, NestedTensor):
        if values.device != source._values.device:
            values = values.to(device=source._values.device)
        expected_shape = tuple(broadcast_shape) + trailing_shape
        if values.dim() == 0:
            return values
        if tuple(values.shape) != expected_shape:
            return None
        return values.unsqueeze(0).expand(batch, *values.shape).reshape(expanded_items, *trailing_shape)

    if not isinstance(values, NestedTensor):
        return None
    if len(values) != batch:
        raise ValueError(
            "index_put: NestedTensor batch length mismatch between input and values: "
            f"input={len(source)}, values={len(values)}"
        )

    expected_suffix = source._physical_shape[:, indexed_dims:]
    if broadcast_shape:
        expected_prefix = source._physical_shape.new_tensor(broadcast_shape).expand(batch, -1)
    else:
        expected_prefix = source._physical_shape[:, :0]
    expected_shape = torch.cat((expected_prefix, expected_suffix), dim=1)
    if not torch.equal(values._physical_shape, expected_shape):
        return None

    value_tensor = values._values
    if value_tensor.device != source._values.device:
        value_tensor = value_tensor.to(device=source._values.device)
    return value_tensor.reshape(expanded_items, *trailing_shape)


@NestedTensorAtenRegistry.implement(aten.index_put.default)
def _index_put(func, args, kwargs):
    r"""Apply ``index_put`` with packed fast paths for shared integer-tensor indices."""

    source = args[0]
    indices = args[1]
    values = args[2]
    if len(args) > 3:
        accumulate = args[3]
    else:
        accumulate = kwargs.get("accumulate", False)

    if not isinstance(indices, (tuple, list)):
        indices = [indices]
    from .nested_tensor import NestedTensor

    if any(isinstance(idx, NestedTensor) for idx in indices):
        raise NotImplementedError("NestedTensor: aten.index_put.default requires plain Tensor indices")

    packed = _packed_index_put_indices(source, indices)
    if packed is not None:
        packed_indices, broadcast_shape = packed
        value_tensor = _packed_index_put_values(source, values, broadcast_shape, len(indices))
        if value_tensor is not None:
            return _packed_like(source, func(source._values, packed_indices, value_tensor, accumulate))

    storage = []
    for i, tensor in enumerate(source._storage):
        value_i = values._storage[i] if isinstance(values, NestedTensor) else values
        if isinstance(value_i, Tensor) and value_i.device != tensor.device:
            value_i = value_i.to(device=tensor.device)
        per_tensor_indices = []
        for idx in indices:
            idx_i = idx
            if isinstance(idx_i, Tensor) and idx_i.device != tensor.device:
                idx_i = idx_i.to(device=tensor.device)
            per_tensor_indices.append(idx_i)
        storage.append(func(tensor, per_tensor_indices, value_i, accumulate))
    return type(source)(storage, **source._meta())


if hasattr(aten, "scatter_reduce"):

    @NestedTensorAtenRegistry.implement(aten.scatter_reduce.two)
    def _scatter_reduce(func, args, kwargs):
        r"""Apply ``scatter_reduce`` with packed fast paths when index/source layouts align."""
        source, dim, index, src, reduce = args[0], args[1], args[2], args[3], args[4]
        include_self = True if len(args) < 6 else args[5]
        call_kwargs = dict(kwargs)
        if "include_self" in call_kwargs:
            include_self = call_kwargs.pop("include_self")
        if "reduce" in call_kwargs:
            reduce = call_kwargs.pop("reduce")
        return _scatter_like(
            source,
            dim,
            index,
            src,
            lambda t, d, i, s: func(t, d, i, s, reduce, include_self=include_self, **call_kwargs),
            "scatter_reduce",
        )


# In-place variants of elementwise ops
_INPLACE_UNARY_OPS = [
    aten.relu_.default,
    aten.silu_.default,
    aten.hardswish_.default,
    aten.hardsigmoid_.default,
    aten.hardtanh_.default,
    aten.leaky_relu_.default,
    aten.elu_.default,
    aten.celu_.default,
    aten.selu_.default,
    aten.sigmoid_.default,
    aten.tanh_.default,
    aten.bernoulli_.float,
    aten.dropout_.default,
    aten.alpha_dropout_.default,
    aten.feature_alpha_dropout_.default,
]


def _inplace_unary_handler(func, args, kwargs):
    r"""Dispatch handler for in-place unary ops applied to _values."""
    source = args[0]
    func(source._values, *args[1:], **kwargs)
    source._cached_storage = None
    return source


_INPLACE_BINARY_OPS = [
    aten.add_.Tensor,
    aten.add_.Scalar,
    aten.sub_.Tensor,
    aten.sub_.Scalar,
    aten.mul_.Tensor,
    aten.mul_.Scalar,
    aten.div_.Tensor,
    aten.div_.Scalar,
    aten.div_.Tensor_mode,
    aten.div_.Scalar_mode,
    aten.remainder_.Tensor,
    aten.remainder_.Scalar,
    aten.fmod_.Tensor,
    aten.floor_divide_.Tensor,
    aten.pow_.Tensor,
    aten.pow_.Scalar,
    aten.bitwise_and_.Tensor,
    aten.bitwise_or_.Tensor,
    aten.bitwise_xor_.Tensor,
]


def _inplace_binary_handler(func, args, kwargs):
    r"""Dispatch handler for in-place binary ops applied to _values."""
    source = args[0]
    resolved = _resolve_other(source, args[1], func)
    func(source._values, resolved, *args[2:], **kwargs)
    source._cached_storage = None
    return source


# ---------------------------------------------------------------------------
# Shape-preserving unary-like ops (extra scalar/keyword args, operate on _values)
# ---------------------------------------------------------------------------

_UNARY_LIKE_OPS = [
    aten.clamp.default,
    aten.clamp_min.default,
    aten.clamp_max.default,
    aten.nan_to_num.default,
    aten.dropout.default,
    aten.alpha_dropout.default,
    aten.feature_alpha_dropout.default,
    aten.feature_dropout.default,
    aten.bernoulli.default,
]

# ---------------------------------------------------------------------------
# Tensor creation ops — preserve packing layout with new _values
# See also torch_functions.py for torch-level empty_like/zeros_like/ones_like/full_like.
# ---------------------------------------------------------------------------


_CREATION_OPS = [
    aten.empty_like.default,
    aten.zeros_like.default,
    aten.ones_like.default,
    aten.full_like.default,
]

# ---------------------------------------------------------------------------
# native_dropout — returns (output, mask) tuple, both as NestedTensor
# ---------------------------------------------------------------------------


@NestedTensorAtenRegistry.implement(aten.native_dropout.default)
def _native_dropout(func, args, kwargs):
    r"""Apply native dropout per element, returning (output, mask) as NestedTensors."""
    source = args[0]
    cls = type(source)
    if not source._storage:
        empty = cls([], **source._meta())
        return empty, empty

    outputs = []
    masks = []
    for t in source._storage:
        out, mask = func(t, *args[1:], **kwargs)
        outputs.append(out)
        masks.append(mask)
    return cls(outputs, **source._meta()), cls(masks, **source._meta())


# ---------------------------------------------------------------------------
# In-place ops that operate directly on _values
# ---------------------------------------------------------------------------


@NestedTensorAtenRegistry.implement(aten.copy_.default)
def _copy(func, args, kwargs):
    r"""In-place copy from src to dest, using packed ``_values`` only for exact layout matches."""
    from .nested_tensor import NestedTensor

    dest, src = args[0], args[1]
    if isinstance(src, NestedTensor) and _layout_match(dest, src):
        func(dest._values, src._values, *args[2:], **kwargs)
        dest._cached_storage = None
        return dest
    raise NotImplementedError(f"NestedTensor: {func} requires matching packed layout")


# ---------------------------------------------------------------------------
# Ternary ops — where, addcmul, addcdiv, lerp.Tensor (3 NT tensor args)
# ---------------------------------------------------------------------------


def _ternary_handler(func, args, kwargs):
    r"""Dispatch handler for ternary ops (where, addcmul, etc.) on packed _values."""
    a, b, c = args[0], args[1], args[2]
    from .nested_tensor import NestedTensor

    sources = [x for x in (a, b, c) if isinstance(x, NestedTensor)]
    if not sources:
        return func(*args, **kwargs)
    ref = sources[0]
    for other in sources[1:]:
        if len(other) != len(ref):
            raise ValueError(f"NestedTensor batch length mismatch for {func}: expected {len(ref)}, got {len(other)}")
    try:
        va = _resolve_ternary_other(ref, a, func)
        vb = _resolve_ternary_other(ref, b, func)
        vc = _resolve_ternary_other(ref, c, func)
    except NotImplementedError:
        # Preserve dense parity when a packed fast path cannot prove per-element
        # broadcasting semantics for every plain Tensor operand.
        return per_element_fallback(func, args, kwargs)
    return _packed_like(ref, func(va, vb, vc, **kwargs))


_TERNARY_OPS = [
    aten.where.self,
    aten.where.ScalarOther,
    aten.where.ScalarSelf,
    aten.where.Scalar,
    aten.addcmul.default,
    aten.addcdiv.default,
    aten.lerp.Tensor,
]

# ---------------------------------------------------------------------------
# Normalization ops — operate on packed _values
# ---------------------------------------------------------------------------


@NestedTensorAtenRegistry.implement(aten.native_layer_norm.default)
def _native_layer_norm(func, args, kwargs):
    r"""Dispatch handler for layer norm on packed _values."""
    source = args[0]
    output, mean, rstd = func(source._values, *args[1:], **kwargs)
    return _packed_like(source, output), mean, rstd


@NestedTensorAtenRegistry.implement(aten.native_layer_norm_backward.default)
def _native_layer_norm_backward(func, args, kwargs):
    r"""Dispatch handler for layer norm backward on packed _values."""
    from .nested_tensor import NestedTensor

    grad_out, input_ = args[0], args[1]
    sources = [a for a in (grad_out, input_) if isinstance(a, NestedTensor)]
    if not sources:
        return func(*args, **kwargs)
    if len(sources) == 2 and not _layout_match(sources[0], sources[1]):
        return per_element_fallback(func, args, kwargs)
    ref = sources[0]
    g = grad_out._values if isinstance(grad_out, NestedTensor) else grad_out
    i = input_._values if isinstance(input_, NestedTensor) else input_
    # args: grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask
    grad_input, grad_weight, grad_bias = func(g, i, *args[2:], **kwargs)
    return _packed_like(ref, grad_input), grad_weight, grad_bias


# ---------------------------------------------------------------------------
# Softmax / log_softmax — operate on packed _values
# ---------------------------------------------------------------------------


def _binary_unwrap_handler(func, args, kwargs):
    r"""Dispatch handler for backward ops that unwrap two NestedTensor args."""
    from .nested_tensor import NestedTensor

    a, b = args[0], args[1]
    sources = [x for x in (a, b) if isinstance(x, NestedTensor)]
    if not sources:
        return func(*args, **kwargs)
    if len(sources) == 2 and not _layout_match(sources[0], sources[1]):
        return per_element_fallback(func, args, kwargs)
    ref = sources[0]
    va = a._values if isinstance(a, NestedTensor) else a
    vb = b._values if isinstance(b, NestedTensor) else b
    return _packed_like(ref, func(va, vb, *args[2:], **kwargs))


def _softmax_handler(func, args, kwargs):
    r"""Dispatch handler for softmax/log_softmax that translates the dim argument."""
    source = args[0]
    dim_adj = _translate_dim(source, args[1])
    if dim_adj == 0:
        # Normalize each ragged row independently by masking padding with -inf.
        padded, _, _, batch_idx, local_idx, _ = _packed_to_padded(source, fill_value=float("-inf"))
        out_padded = func(padded, 1, *args[2:], **kwargs)
        return _packed_like(source, out_padded[batch_idx, local_idx])
    return _packed_like(source, func(source._values, dim_adj, *args[2:], **kwargs))


# ---------------------------------------------------------------------------
# Global reductions — reduce all of _values to a scalar (no dim argument)
# ---------------------------------------------------------------------------

_GLOBAL_REDUCTION_OPS = [
    aten.sum.default,
    aten.any.default,
    aten.all.default,
    aten.mean.default,
    aten.max.default,
    aten.min.default,
    aten.median.default,
    aten.nanmedian.default,
]


def _global_reduction_handler(func, args, kwargs):
    r"""Dispatch handler for global reductions (sum, mean, etc.) over all _values."""
    source = args[0]
    return func(source._values, **kwargs)


@NestedTensorAtenRegistry.implement(aten.linalg_vector_norm.default)
def _linalg_vector_norm(func, args, kwargs):
    r"""Handle vector-norm cases for NestedTensor with packed fast paths where semantics stay exact."""
    source = args[0]
    kw_ord = kwargs.pop("ord", _MISSING)
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_keepdim = kwargs.pop("keepdim", _MISSING)
    kw_dtype = kwargs.pop("dtype", _MISSING)
    if len(args) > 1:
        if kw_ord is not _MISSING:
            raise TypeError("got multiple values for argument 'ord'")
        ord_value = args[1]
    else:
        ord_value = 2 if kw_ord is _MISSING else kw_ord
    if len(args) > 2:
        if kw_dim is not _MISSING:
            raise TypeError("got multiple values for argument 'dim'")
        dim_arg = args[2]
    else:
        dim_arg = None if kw_dim is _MISSING else kw_dim
    if len(args) > 3:
        if kw_keepdim is not _MISSING:
            raise TypeError("got multiple values for argument 'keepdim'")
        keepdim = args[3]
    else:
        keepdim = False if kw_keepdim is _MISSING else kw_keepdim
    if len(args) > 4:
        if kw_dtype is not _MISSING:
            raise TypeError("got multiple values for argument 'dtype'")
        dtype = args[4]
    else:
        dtype = None if kw_dtype is _MISSING else kw_dtype

    if source._physical_shape.size(1) == 0:
        raise NotImplementedError(f"NestedTensor: {func} falls back for scalar elements")
    if source._physical_shape.size(1) > 1 and source._values.dim() == 1:
        raise NotImplementedError(f"NestedTensor: {func} requires non-flattened packed storage")

    dims = _parse_dims_arg(dim_arg)
    if len(dims) == 0:
        if not _vector_norm_zero_padding_safe(ord_value):
            raise NotImplementedError(f"NestedTensor: {func} requires zero-padding-safe ord for ragged reductions")
        padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=0)
        reduce_dims = list(range(1, padded.dim()))
        out_values = func(padded, ord_value, reduce_dims, keepdim, dtype=dtype, **kwargs)
        return _from_uniform_batched_output(source, out_values)

    if len(dims) != 1:
        raise NotImplementedError(f"NestedTensor: {func} only handles a single logical dimension")

    dim = _normalize_dim(dims[0], source.dim())
    batch_dim = _get_batch_dim(source)
    if dim == batch_dim:
        raise ValueError("linalg.norm along the batch dimension is not supported for NestedTensor.")

    dim_adj = _translate_dim(source, dim)
    if dim_adj == 0:
        if not _vector_norm_zero_padding_safe(ord_value):
            raise NotImplementedError(f"NestedTensor: {func} requires zero-padding-safe ord for ragged reductions")
        padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=0)
        out_values = func(padded, ord_value, [1], keepdim, dtype=dtype, **kwargs)
        return _from_uniform_batched_output(source, out_values)

    out_values = func(source._values, ord_value, [dim_adj], keepdim, dtype=dtype, **kwargs)
    return _reduce_non_ragged_packed(source, out_values, dim_adj, keepdim)


def _parse_dims_arg(dim_arg) -> tuple[int, ...]:
    if dim_arg is None:
        return ()
    if isinstance(dim_arg, int):
        return (dim_arg,)
    return tuple(dim_arg)


def _vector_norm_zero_padding_safe(ord_value) -> bool:
    r"""Return whether zero-padding preserves vector-norm semantics on ragged reductions."""
    if ord_value is None:
        return True
    if isinstance(ord_value, bool) or not isinstance(ord_value, (int, float)):
        return False
    ord_float = float(ord_value)
    if math.isnan(ord_float):
        return False
    return ord_float == 0.0 or ord_float > 0.0


def _order_stat_pair_reduction(source: NestedTensor, dim: int, keepdim: bool, apply):
    r"""
    Reduce order-statistic ops returning ``(values, indices)``.

    Packed fast paths only apply to static per-element dims. Ragged-dim reductions
    fall back to explicit per-element calls because padding cannot preserve
    order-statistic ranks the way max/min can.
    """
    dim = _normalize_dim(dim, source.dim())
    batch_dim = _get_batch_dim(source)

    if dim == batch_dim:
        values = []
        indices = []
        for tensor in source._storage:
            value, index = apply(tensor.reshape(-1), 0, False)
            values.append(value)
            indices.append(index)
        values_out = torch.stack(values)
        indices_out = torch.stack(indices)
        if keepdim:
            values_out = values_out.unsqueeze(batch_dim)
            indices_out = indices_out.unsqueeze(batch_dim)
        return values_out, indices_out

    dim_adj = _translate_dim(source, dim)
    if dim_adj == 0:
        values = []
        indices = []
        for tensor in source._storage:
            value, index = apply(tensor, dim_adj, keepdim)
            values.append(value)
            indices.append(index)
        return _try_stack(values, source), _try_stack(indices, source)

    values_out, indices_out = apply(source._values, dim_adj, keepdim)
    return (
        _reduce_non_ragged_packed(source, values_out, dim_adj, keepdim),
        _reduce_non_ragged_packed(source, indices_out, dim_adj, keepdim),
    )


@NestedTensorAtenRegistry.implement(aten.max.dim)
@NestedTensorAtenRegistry.implement(aten.min.dim)
def _max_min_dim_reduction(func, args, kwargs):
    r"""Handle ``max/min`` dim reductions, returning both values and indices."""
    source = args[0]
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_keepdim = kwargs.pop("keepdim", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError("got multiple values for argument 'dim'")
        dim = args[1]
    else:
        dim = kw_dim
    if dim is _MISSING:
        raise TypeError("missing required argument 'dim'")
    if len(args) > 2:
        if kw_keepdim is not _MISSING:
            raise TypeError("got multiple values for argument 'keepdim'")
        keepdim = args[2]
    else:
        keepdim = False if kw_keepdim is _MISSING else kw_keepdim

    dim = _normalize_dim(dim, source.dim())
    batch_dim = _get_batch_dim(source)
    largest = func is aten.max.dim

    if dim == batch_dim:
        values = torch.stack(
            [torch.ops.aten.max.default(t) if largest else torch.ops.aten.min.default(t) for t in source._storage]
        )
        indices = torch.stack(
            [torch.ops.aten.argmax.default(t) if largest else torch.ops.aten.argmin.default(t) for t in source._storage]
        )
        if keepdim:
            values = values.unsqueeze(batch_dim)
            indices = indices.unsqueeze(batch_dim)
        return values, indices

    dim_adj = _translate_dim(source, dim)
    if dim_adj == 0:
        fill_value = _topk_fill_value(source._values.dtype, largest=largest)
        padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=fill_value)
        return func(padded, 1, keepdim, **kwargs)

    values, indices = func(source._values, dim_adj, keepdim, **kwargs)
    return (
        _reduce_non_ragged_packed(source, values, dim_adj, keepdim),
        _reduce_non_ragged_packed(source, indices, dim_adj, keepdim),
    )


@NestedTensorAtenRegistry.implement(aten.argmax.default)
@NestedTensorAtenRegistry.implement(aten.argmin.default)
def _arg_extrema_reduction(func, args, kwargs):
    r"""Handle ``argmax/argmin`` for per-element global or dim reductions."""
    source = args[0]
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_keepdim = kwargs.pop("keepdim", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError("got multiple values for argument 'dim'")
        dim = args[1]
    else:
        dim = None if kw_dim is _MISSING else kw_dim
    if len(args) > 2:
        if kw_keepdim is not _MISSING:
            raise TypeError("got multiple values for argument 'keepdim'")
        keepdim = args[2]
    else:
        keepdim = False if kw_keepdim is _MISSING else kw_keepdim

    largest = func is aten.argmax.default

    if dim is None:
        output = torch.stack([func(t, **kwargs) for t in source._storage])
        if keepdim:
            output = output.unsqueeze(_get_batch_dim(source))
        return output

    dim = _normalize_dim(dim, source.dim())
    batch_dim = _get_batch_dim(source)
    if dim == batch_dim:
        output = torch.stack([func(t, **kwargs) for t in source._storage])
        if keepdim:
            output = output.unsqueeze(batch_dim)
        return output

    dim_adj = _translate_dim(source, dim)
    if dim_adj == 0:
        fill_value = _topk_fill_value(source._values.dtype, largest=largest)
        padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=fill_value)
        return func(padded, 1, keepdim, **kwargs)

    out_values = func(source._values, dim_adj, keepdim, **kwargs)
    return _reduce_non_ragged_packed(source, out_values, dim_adj, keepdim)


@NestedTensorAtenRegistry.implement(aten.kthvalue.default)
def _kthvalue_reduction(func, args, kwargs):
    r"""Handle ``kthvalue`` dim reductions on static packed dims."""
    source = args[0]
    kw_k = kwargs.pop("k", _MISSING)
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_keepdim = kwargs.pop("keepdim", _MISSING)
    if len(args) > 1:
        if kw_k is not _MISSING:
            raise TypeError("got multiple values for argument 'k'")
        k = args[1]
    else:
        k = kw_k
    if k is _MISSING:
        raise TypeError("missing required argument 'k'")
    if len(args) > 2:
        if kw_dim is not _MISSING:
            raise TypeError("got multiple values for argument 'dim'")
        dim = args[2]
    else:
        dim = -1 if kw_dim is _MISSING else kw_dim
    if len(args) > 3:
        if kw_keepdim is not _MISSING:
            raise TypeError("got multiple values for argument 'keepdim'")
        keepdim = args[3]
    else:
        keepdim = False if kw_keepdim is _MISSING else kw_keepdim

    return _order_stat_pair_reduction(source, dim, keepdim, lambda t, d, kd: func(t, k, d, kd, **kwargs))


@NestedTensorAtenRegistry.implement(aten.median.dim)
@NestedTensorAtenRegistry.implement(aten.nanmedian.dim)
def _median_dim_reduction(func, args, kwargs):
    r"""Handle ``median/nanmedian`` dim reductions on static packed dims."""
    source = args[0]
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_keepdim = kwargs.pop("keepdim", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError("got multiple values for argument 'dim'")
        dim = args[1]
    else:
        dim = kw_dim
    if dim is _MISSING:
        raise TypeError("missing required argument 'dim'")
    if len(args) > 2:
        if kw_keepdim is not _MISSING:
            raise TypeError("got multiple values for argument 'keepdim'")
        keepdim = args[2]
    else:
        keepdim = False if kw_keepdim is _MISSING else kw_keepdim

    return _order_stat_pair_reduction(source, dim, keepdim, lambda t, d, kd: func(t, d, kd, **kwargs))


@NestedTensorAtenRegistry.implement(aten.mode.default)
def _mode_reduction(func, args, kwargs):
    r"""Handle ``mode`` dim reductions on static packed dims."""
    source = args[0]
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_keepdim = kwargs.pop("keepdim", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError("got multiple values for argument 'dim'")
        dim = args[1]
    else:
        dim = -1 if kw_dim is _MISSING else kw_dim
    if len(args) > 2:
        if kw_keepdim is not _MISSING:
            raise TypeError("got multiple values for argument 'keepdim'")
        keepdim = args[2]
    else:
        keepdim = False if kw_keepdim is _MISSING else kw_keepdim

    return _order_stat_pair_reduction(source, dim, keepdim, lambda t, d, kd: func(t, d, kd, **kwargs))


@NestedTensorAtenRegistry.implement(aten.sum.dim_IntList)
@NestedTensorAtenRegistry.implement(aten.mean.dim)
def _sum_mean_dim_reduction(func, args, kwargs):
    r"""Handle ``sum/mean`` dim reductions on packed values for single logical dims."""
    source = args[0]
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_keepdim = kwargs.pop("keepdim", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError("got multiple values for argument 'dim'")
        dim_arg = args[1]
    else:
        dim_arg = None if kw_dim is _MISSING else kw_dim
    if len(args) > 2:
        if kw_keepdim is not _MISSING:
            raise TypeError("got multiple values for argument 'keepdim'")
        keepdim = args[2]
    else:
        keepdim = False if kw_keepdim is _MISSING else kw_keepdim
    dims = _parse_dims_arg(dim_arg)
    if len(dims) == 0:
        return func(source._values, dim_arg, keepdim, **kwargs)

    if len(dims) > 1:
        try:
            dims_adj = _translate_dims(source, dims)
        except ValueError as exc:
            raise NotImplementedError(f"NestedTensor: {func} with dim={dims} is not supported") from exc
        if source._values.dim() > 1 and all(dim_i > 0 for dim_i in dims_adj):
            out_values = func(source._values, list(dims_adj), keepdim, **kwargs)
            return _reduce_non_ragged_packed_dims(source, out_values, dims_adj, keepdim)
        # Ragged-including multi-dim reductions intentionally keep the generic
        # fallback semantics (including stack-on-uniform behavior).
        return per_element_fallback(func, (source, list(dims_adj), keepdim), kwargs)

    dim = _normalize_dim(dims[0], source.dim())
    batch_dim = _get_batch_dim(source)
    if dim == batch_dim:
        # Reducing batch dim follows existing NestedTensor semantics:
        # reduce each element globally, then stack.
        reduced = torch.stack([func(t, None, False, **kwargs) for t in source._storage])
        if keepdim:
            return reduced.unsqueeze(batch_dim)
        return reduced

    dim_adj = _translate_dim(source, dim)
    if dim_adj == 0:
        if func is aten.mean.dim:
            return per_element_fallback(func, (source, [dim_adj], keepdim), kwargs)
        padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=0)
        return func(padded, [1], keepdim, **kwargs)

    out_values = func(source._values, [dim_adj], keepdim, **kwargs)
    return _reduce_non_ragged_packed(source, out_values, dim_adj, keepdim)


@NestedTensorAtenRegistry.implement(aten.amax.default)
@NestedTensorAtenRegistry.implement(aten.amin.default)
def _extrema_dim_reduction(func, args, kwargs):
    r"""Handle ``amax/amin`` dim reductions on packed values for common dim patterns."""
    source = args[0]
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_keepdim = kwargs.pop("keepdim", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError("got multiple values for argument 'dim'")
        dim_arg = args[1]
    else:
        dim_arg = () if kw_dim is _MISSING else kw_dim
    if len(args) > 2:
        if kw_keepdim is not _MISSING:
            raise TypeError("got multiple values for argument 'keepdim'")
        keepdim = args[2]
    else:
        keepdim = False if kw_keepdim is _MISSING else kw_keepdim
    dims = _parse_dims_arg(dim_arg)
    if len(dims) == 0:
        return func(source._values, [], keepdim, **kwargs)

    if len(dims) > 1:
        try:
            dims_adj = _translate_dims(source, dims)
        except ValueError as exc:
            raise NotImplementedError(f"NestedTensor: {func} with dim={dims} is not supported") from exc
        if source._values.dim() > 1 and all(dim_i > 0 for dim_i in dims_adj):
            out_values = func(source._values, list(dims_adj), keepdim, **kwargs)
            return _reduce_non_ragged_packed_dims(source, out_values, dims_adj, keepdim)
        return per_element_fallback(func, (source, list(dims_adj), keepdim), kwargs)

    dim = _normalize_dim(dims[0], source.dim())
    batch_dim = _get_batch_dim(source)
    if dim == batch_dim:
        reduced = torch.stack([func(t, [], False, **kwargs) for t in source._storage])
        if keepdim:
            return reduced.unsqueeze(batch_dim)
        return reduced

    dim_adj = _translate_dim(source, dim)
    if dim_adj == 0:
        fill_value = _topk_fill_value(source._values.dtype, largest=(func is aten.amax.default))
        padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=fill_value)
        return func(padded, [1], keepdim, **kwargs)

    out_values = func(source._values, [dim_adj], keepdim, **kwargs)
    return _reduce_non_ragged_packed(source, out_values, dim_adj, keepdim)


@NestedTensorAtenRegistry.implement(aten.logsumexp.default)
def _logsumexp_dim_reduction(func, args, kwargs):
    r"""Handle ``logsumexp`` dim reductions with packed fastpaths for non-ragged dims."""
    source = args[0]
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_keepdim = kwargs.pop("keepdim", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError("got multiple values for argument 'dim'")
        dim_arg = args[1]
    else:
        dim_arg = () if kw_dim is _MISSING else kw_dim
    if len(args) > 2:
        if kw_keepdim is not _MISSING:
            raise TypeError("got multiple values for argument 'keepdim'")
        keepdim = args[2]
    else:
        keepdim = False if kw_keepdim is _MISSING else kw_keepdim
    dims = _parse_dims_arg(dim_arg)
    if len(dims) == 0:
        return func(source._values, [], keepdim, **kwargs)

    if len(dims) > 1:
        dims_adj = _translate_dims(source, dims)
        if source._values.dim() > 1 and all(dim_i > 0 for dim_i in dims_adj):
            out_values = func(source._values, list(dims_adj), keepdim, **kwargs)
            return _reduce_non_ragged_packed_dims(source, out_values, dims_adj, keepdim)
        if 0 in dims_adj:
            padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=float("-inf"))
            padded_dims = [1 if dim_i == 0 else dim_i for dim_i in dims_adj]
            return func(padded, padded_dims, keepdim, **kwargs)
        return per_element_fallback(func, (source, list(dims_adj), keepdim), kwargs)

    dim = _normalize_dim(dims[0], source.dim())
    batch_dim = _get_batch_dim(source)
    if dim == batch_dim:
        reduced = torch.stack([torch.logsumexp(t.reshape(-1), dim=0) for t in source._storage])
        if keepdim:
            return reduced.unsqueeze(batch_dim)
        return reduced

    dim_adj = _translate_dim(source, dim)
    if dim_adj == 0:
        padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=float("-inf"))
        return func(padded, [1], keepdim, **kwargs)

    out_values = func(source._values, [dim_adj], keepdim, **kwargs)
    return _reduce_non_ragged_packed(source, out_values, dim_adj, keepdim)


@NestedTensorAtenRegistry.implement(aten.nansum.default)
@NestedTensorAtenRegistry.implement(aten.nanmean.default)
def _nan_dim_reduction(func, args, kwargs):
    r"""Handle ``nansum/nanmean`` dim reductions with packed fastpaths for non-ragged dims."""
    source = args[0]
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_keepdim = kwargs.pop("keepdim", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError("got multiple values for argument 'dim'")
        dim_arg = args[1]
    else:
        dim_arg = None if kw_dim is _MISSING else kw_dim
    if len(args) > 2:
        if kw_keepdim is not _MISSING:
            raise TypeError("got multiple values for argument 'keepdim'")
        keepdim = args[2]
    else:
        keepdim = False if kw_keepdim is _MISSING else kw_keepdim
    dims = _parse_dims_arg(dim_arg)
    if len(dims) == 0:
        return func(source._values, None, keepdim, **kwargs)

    if len(dims) > 1:
        dims_adj = _translate_dims(source, dims)
        if source._values.dim() > 1 and all(dim_i > 0 for dim_i in dims_adj):
            out_values = func(source._values, list(dims_adj), keepdim, **kwargs)
            return _reduce_non_ragged_packed_dims(source, out_values, dims_adj, keepdim)
        if 0 in dims_adj and func is aten.nansum.default:
            padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=0)
            padded_dims = [1 if dim_i == 0 else dim_i for dim_i in dims_adj]
            return func(padded, padded_dims, keepdim, **kwargs)
        return per_element_fallback(func, (source, list(dims_adj), keepdim), kwargs)

    dim = _normalize_dim(dims[0], source.dim())
    batch_dim = _get_batch_dim(source)
    if dim == batch_dim:
        reduced = torch.stack([func(t, None, False, **kwargs) for t in source._storage])
        if keepdim:
            return reduced.unsqueeze(batch_dim)
        return reduced

    dim_adj = _translate_dim(source, dim)
    if dim_adj == 0:
        if func is aten.nansum.default:
            padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=0)
            return func(padded, [1], keepdim, **kwargs)
        return per_element_fallback(func, (source, [dim_adj], keepdim), kwargs)

    out_values = func(source._values, [dim_adj], keepdim, **kwargs)
    return _reduce_non_ragged_packed(source, out_values, dim_adj, keepdim)


@NestedTensorAtenRegistry.implement(aten.std.correction)
@NestedTensorAtenRegistry.implement(aten.var.correction)
def _variance_dim_reduction(func, args, kwargs):
    r"""Handle ``std/var`` correction reductions via packed fastpaths where valid."""
    source = args[0]
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_keepdim = kwargs.pop("keepdim", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError("got multiple values for argument 'dim'")
        dim_arg = args[1]
    else:
        dim_arg = None if kw_dim is _MISSING else kw_dim
    if len(args) > 2:
        if kw_keepdim is not _MISSING:
            raise TypeError("got multiple values for argument 'keepdim'")
        keepdim = args[2]
    else:
        keepdim = False if kw_keepdim is _MISSING else kw_keepdim
    dims = _parse_dims_arg(dim_arg)
    if len(dims) == 0:
        return func(source._values, None, keepdim=keepdim, **kwargs)

    if len(dims) > 1:
        dims_adj = _translate_dims(source, dims)
        if source._values.dim() > 1 and all(dim_i > 0 for dim_i in dims_adj):
            out_values = func(source._values, list(dims_adj), keepdim=keepdim, **kwargs)
            return _reduce_non_ragged_packed_dims(source, out_values, dims_adj, keepdim)
        kwargs["keepdim"] = keepdim
        return per_element_fallback(func, (source, list(dims_adj)), kwargs)

    dim = _normalize_dim(dims[0], source.dim())
    batch_dim = _get_batch_dim(source)
    if dim == batch_dim:
        reduced = torch.stack([func(t, None, keepdim=False, **kwargs) for t in source._storage])
        if keepdim:
            return reduced.unsqueeze(batch_dim)
        return reduced

    dim_adj = _translate_dim(source, dim)
    if dim_adj == 0:
        kwargs["keepdim"] = keepdim
        return per_element_fallback(func, (source, [dim_adj]), kwargs)

    out_values = func(source._values, [dim_adj], keepdim=keepdim, **kwargs)
    return _reduce_non_ragged_packed(source, out_values, dim_adj, keepdim)


@NestedTensorAtenRegistry.implement(aten.var_mean.correction)
def _var_mean_dim_reduction(func, args, kwargs):
    r"""Handle ``var_mean`` correction reductions via packed fastpaths where valid."""
    source = args[0]
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_keepdim = kwargs.pop("keepdim", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError("got multiple values for argument 'dim'")
        dim_arg = args[1]
    else:
        dim_arg = None if kw_dim is _MISSING else kw_dim
    if len(args) > 2:
        if kw_keepdim is not _MISSING:
            raise TypeError("got multiple values for argument 'keepdim'")
        keepdim = args[2]
    else:
        keepdim = False if kw_keepdim is _MISSING else kw_keepdim
    dims = _parse_dims_arg(dim_arg)
    if len(dims) == 0:
        out_var, out_mean = func(source._values, None, keepdim=keepdim, **kwargs)
        return out_var, out_mean

    if len(dims) > 1:
        dims_adj = _translate_dims(source, dims)
        if source._values.dim() > 1 and all(dim_i > 0 for dim_i in dims_adj):
            out_var, out_mean = func(source._values, list(dims_adj), keepdim=keepdim, **kwargs)
            return (
                _reduce_non_ragged_packed_dims(source, out_var, dims_adj, keepdim),
                _reduce_non_ragged_packed_dims(source, out_mean, dims_adj, keepdim),
            )
        kwargs["keepdim"] = keepdim
        return per_element_fallback(func, (source, list(dims_adj)), kwargs)

    dim = _normalize_dim(dims[0], source.dim())
    batch_dim = _get_batch_dim(source)
    if dim == batch_dim:
        vars_, means = [], []
        for tensor in source._storage:
            var_value, mean_value = func(tensor, None, keepdim=False, **kwargs)
            vars_.append(var_value)
            means.append(mean_value)
        out_var = torch.stack(vars_)
        out_mean = torch.stack(means)
        if keepdim:
            out_var = out_var.unsqueeze(batch_dim)
            out_mean = out_mean.unsqueeze(batch_dim)
        return out_var, out_mean

    dim_adj = _translate_dim(source, dim)
    if dim_adj == 0:
        kwargs["keepdim"] = keepdim
        return per_element_fallback(func, (source, [dim_adj]), kwargs)

    out_var, out_mean = func(source._values, [dim_adj], keepdim=keepdim, **kwargs)
    return (
        _reduce_non_ragged_packed(source, out_var, dim_adj, keepdim),
        _reduce_non_ragged_packed(source, out_mean, dim_adj, keepdim),
    )


@NestedTensorAtenRegistry.implement(aten.count_nonzero.dim_IntList)
def _count_nonzero_dim_reduction(func, args, kwargs):
    r"""Handle ``count_nonzero`` dim reductions on packed values for common dim patterns."""
    source = args[0]
    kw_dim = kwargs.pop("dim", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError("got multiple values for argument 'dim'")
        dim_arg = args[1]
    else:
        dim_arg = () if kw_dim is _MISSING else kw_dim
    dims = _parse_dims_arg(dim_arg)
    if len(dims) == 0:
        return aten.count_nonzero.default(source._values, **kwargs)

    if len(dims) > 1:
        dims_adj = _translate_dims(source, dims)
        if source._values.dim() > 1 and all(dim_i > 0 for dim_i in dims_adj):
            out_values = func(source._values, list(dims_adj), **kwargs)
            return _reduce_non_ragged_packed_dims(source, out_values, dims_adj, keepdim=False)
        if 0 in dims_adj:
            padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=0)
            padded_dims = [1 if dim_i == 0 else dim_i for dim_i in dims_adj]
            return func(padded, padded_dims, **kwargs)
        return per_element_fallback(func, (source, list(dims_adj)), kwargs)

    dim = _normalize_dim(dims[0], source.dim())
    batch_dim = _get_batch_dim(source)
    if dim == batch_dim:
        return torch.stack([torch.count_nonzero(t) for t in source._storage])

    dim_adj = _translate_dim(source, dim)
    if dim_adj == 0:
        padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=0)
        return func(padded, [1], **kwargs)

    out_values = func(source._values, [dim_adj], **kwargs)
    return _reduce_non_ragged_packed(source, out_values, dim_adj, keepdim=False)


# ---------------------------------------------------------------------------
# masked_fill — fast path when mask is a NestedTensor with matching packed layout
# ---------------------------------------------------------------------------


def _per_element_numel(source: NestedTensor) -> Tensor:
    r"""Return the number of scalar values contributed by each NestedTensor element."""
    batch_size = len(source)
    if batch_size == 0:
        return torch.empty((0,), dtype=source._offsets.dtype, device=source._offsets.device)
    if source._physical_shape.numel() == 0:
        return torch.ones((batch_size,), dtype=source._offsets.dtype, device=source._offsets.device)
    return torch.prod(source._physical_shape, dim=1)


def _per_element_true_counts(mask: NestedTensor) -> Tensor:
    r"""Count ``True`` values per element in packed mask storage."""
    if len(mask) == 0:
        return torch.empty((0,), dtype=torch.long, device=mask._values.device)
    if mask._values.dim() > 1:
        row_counts = mask._values.reshape(mask._values.shape[0], -1).to(dtype=torch.long).sum(dim=1)
    else:
        row_counts = mask._values.to(dtype=torch.long)
    prefix = torch.zeros((row_counts.numel() + 1,), dtype=torch.long, device=row_counts.device)
    prefix[1:] = torch.cumsum(row_counts, dim=0)
    offsets = mask._offsets.to(device=row_counts.device)
    return prefix.index_select(0, offsets[1:]) - prefix.index_select(0, offsets[:-1])


def _masked_scatter_source_consumption_matches(mask: NestedTensor, source: NestedTensor) -> bool:
    r"""
    Check whether packed ``masked_scatter`` preserves per-element source consumption.

    The packed path is only valid when each element of ``source`` contributes exactly as
    many scalars as the corresponding mask element consumes. Fake tensors do not expose
    concrete mask values, so this returns ``False`` under fake tensor mode.
    """
    if len(mask) != len(source) or _is_fake_tensor(mask._values) or _is_fake_tensor(source._values):
        return False
    source_numel = _per_element_numel(source).to(device=mask._values.device)
    mask_true = _per_element_true_counts(mask)
    return bool(torch.equal(source_numel, mask_true))


def _masked_fill_handler(func, args, kwargs):
    r"""Dispatch handler for masked_fill with matching-offset fast path."""
    from .nested_tensor import NestedTensor

    source, mask, value = args[0], args[1], args[2]
    if isinstance(mask, NestedTensor) and _layout_match(source, mask):
        return _packed_like(source, func(source._values, mask._values, value, **kwargs))
    raise NotImplementedError(f"NestedTensor: {func} requires mask with matching packed layout")


@NestedTensorAtenRegistry.implement(aten.masked_select.default)
def _masked_select_handler(func, args, kwargs):
    r"""Dispatch handler for masked_select with exact-shape matching-offset masks."""
    from .nested_tensor import NestedTensor

    source, mask = args[0], args[1]

    aligned_mask = source._maybe_exact_shape_nested_like(mask)
    if aligned_mask is not None:
        mask = aligned_mask

    if not isinstance(mask, NestedTensor):
        raise NotImplementedError(f"NestedTensor: {func} requires NestedTensor mask")
    if len(source) != len(mask):
        raise ValueError(
            "NestedTensor batch length mismatch between input and mask: " f"input={len(source)}, mask={len(mask)}"
        )
    if _is_fake_tensor(source._values) or _is_fake_tensor(mask._values):
        raise NotImplementedError(f"NestedTensor: {func} requires concrete mask values")
    if not _layout_match(source, mask):
        # Broadcasted masks are valid per element, but packed masked_select needs the exact
        # per-element True-counts from the stored mask to rebuild output boundaries.
        raise NotImplementedError(f"NestedTensor: {func} requires exact-shape masks with matching packed layout")

    mask_values = mask._values
    if mask_values.device != source._values.device:
        mask_values = mask_values.to(device=source._values.device)
    out_values = func(source._values, mask_values, **kwargs)
    counts = _per_element_true_counts(mask).to(device=source._offsets.device, dtype=source._offsets.dtype)
    out_offsets = torch.empty((counts.numel() + 1,), dtype=source._offsets.dtype, device=source._offsets.device)
    out_offsets[0] = 0
    if counts.numel() > 0:
        out_offsets[1:] = torch.cumsum(counts, dim=0)
    out_shape = counts.unsqueeze(1)
    return type(source)._from_packed(
        out_values,
        out_offsets,
        out_shape,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
    )


@NestedTensorAtenRegistry.implement(aten.nonzero.default)
def _nonzero_handler(func, args, kwargs):
    r"""Dispatch handler for ``nonzero(as_tuple=False)`` on non-flattened packed layouts."""
    source = args[0]
    batch_size = len(source)
    if batch_size == 0:
        from .nested_tensor import NestedTensor

        return NestedTensor([], device=source.device, dtype=torch.long, **source._meta(include_dtype=False))
    if _is_fake_tensor(source._values):
        raise NotImplementedError(f"NestedTensor: {func} requires concrete values")
    if source._physical_shape.size(1) > 1 and source._values.dim() == 1:
        # Flattened 1-D packing loses the per-dimension coordinate structure needed by
        # nonzero. Keep those cases on the explicit per-element path instead.
        raise NotImplementedError(f"NestedTensor: {func} requires non-flattened packed storage")

    packed_indices = func(source._values, **kwargs)
    input_ndim = source._physical_shape.size(1)
    if input_ndim == 0:
        batch_idx = packed_indices[:, 0]
        out_values = packed_indices[:, :0]
    else:
        batch_idx, local_idx0 = source._packed_batch_local_indices(
            packed_indices[:, 0],
            device=packed_indices.device,
            dtype=packed_indices.dtype,
        )
        out_values = torch.cat((local_idx0.unsqueeze(1), packed_indices[:, 1:]), dim=1)

    counts = torch.bincount(batch_idx, minlength=batch_size)
    counts_cpu = counts.to(device=source._offsets.device, dtype=source._offsets.dtype)
    out_offsets = torch.empty((batch_size + 1,), dtype=source._offsets.dtype, device=source._offsets.device)
    out_offsets[0] = 0
    out_offsets[1:] = torch.cumsum(counts_cpu, dim=0)
    rank_col = source._physical_shape.new_full((batch_size, 1), input_ndim)
    out_shape = torch.cat((counts_cpu.unsqueeze(1), rank_col), dim=1)
    return type(source)._from_packed(
        out_values,
        out_offsets,
        out_shape,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
    )


def _masked_scatter_handler(func, args, kwargs):
    r"""Dispatch handler for masked_scatter with exact per-element source-consumption boundaries."""
    from .nested_tensor import NestedTensor

    input_tensor, mask, source = args[0], args[1], args[2]

    aligned_mask = input_tensor._maybe_exact_shape_nested_like(mask)
    if aligned_mask is not None:
        mask = aligned_mask
    aligned_source = input_tensor._maybe_exact_shape_nested_like(source)
    if aligned_source is not None:
        source = aligned_source

    if not isinstance(mask, NestedTensor) or not isinstance(source, NestedTensor):
        raise NotImplementedError(f"NestedTensor: {func} requires NestedTensor mask and source")
    if len(input_tensor) != len(mask):
        raise ValueError(
            "NestedTensor batch length mismatch between input and mask: " f"input={len(input_tensor)}, mask={len(mask)}"
        )
    if len(input_tensor) != len(source):
        raise ValueError(
            "NestedTensor batch length mismatch between input and source: "
            f"input={len(input_tensor)}, source={len(source)}"
        )
    if not _layout_match(input_tensor, mask):
        # Broadcasted masks are valid per element but not packed-safe: the source stream would
        # need counts from the broadcasted mask, not the stored pre-broadcast values.
        raise NotImplementedError(f"NestedTensor: {func} requires exact-shape masks with matching packed layout")
    if not _masked_scatter_source_consumption_matches(mask, source):
        raise NotImplementedError(f"NestedTensor: {func} requires per-element source.numel() == mask.count_nonzero()")

    mask_values = mask._values
    if mask_values.device != input_tensor._values.device:
        mask_values = mask_values.to(device=input_tensor._values.device)
    source_values = source._values
    if source_values.device != input_tensor._values.device:
        source_values = source_values.to(device=input_tensor._values.device)
    return _packed_like(input_tensor, func(input_tensor._values, mask_values, source_values, **kwargs))


# ---------------------------------------------------------------------------
# RNG in-place ops — shape-preserving mutations on _values
# ---------------------------------------------------------------------------

_INPLACE_RNG_OPS = [
    aten.uniform_.default,
    aten.normal_.default,
]


# ---------------------------------------------------------------------------
# Random tensor creation ops — same pattern as empty_like/zeros_like
# ---------------------------------------------------------------------------

_RANDOM_CREATION_OPS = [
    aten.rand_like.default,
    aten.randn_like.default,
    aten.randint_like.default,
    aten.randint_like.low_dtype,
]


# ===========================================================================
# Bulk registration — all table-driven op → handler mappings in one place.
# @NestedTensorAtenRegistry.implement(...) decorators above handle 1:1 ops.
# ===========================================================================

for _op in ATEN_UNARY_ELEMENTWISE_OPS:
    NestedTensorAtenRegistry[_op] = _unary_handler
for _op in ATEN_BINARY_ELEMENTWISE_OPS:
    NestedTensorAtenRegistry[_op] = _binary_handler
for _op in _INPLACE_UNARY_OPS:
    NestedTensorAtenRegistry[_op] = _inplace_unary_handler
for _op in _INPLACE_BINARY_OPS:
    NestedTensorAtenRegistry[_op] = _inplace_binary_handler
for _op in _UNARY_LIKE_OPS + _CREATION_OPS:
    NestedTensorAtenRegistry[_op] = _unary_handler
for _op in _INPLACE_RNG_OPS:
    NestedTensorAtenRegistry[_op] = _inplace_unary_handler
for _op in _TERNARY_OPS:
    NestedTensorAtenRegistry[_op] = _ternary_handler
for _op in _GLOBAL_REDUCTION_OPS:
    NestedTensorAtenRegistry[_op] = _global_reduction_handler
NestedTensorAtenRegistry[aten.rms_norm.default] = _unary_handler
for _op in [aten._softmax.default, aten._log_softmax.default]:
    NestedTensorAtenRegistry[_op] = _softmax_handler
for _op in [aten._softmax_backward_data.default, aten._log_softmax_backward_data.default]:
    NestedTensorAtenRegistry[_op] = _binary_unwrap_handler
for _op in [aten.masked_fill.Scalar, aten.masked_fill.Tensor]:
    NestedTensorAtenRegistry[_op] = _masked_fill_handler
NestedTensorAtenRegistry[aten.masked_scatter.default] = _masked_scatter_handler


def _random_creation_handler(func, args, kwargs):
    r"""Create random tensors per element to preserve RNG parity with per-element eager calls."""
    source = args[0]
    cls = type(source)
    if not source._storage:
        return cls([], **source._meta())
    return cls([func(t, *args[1:], **kwargs) for t in source._storage], **source._meta())


for _op in _RANDOM_CREATION_OPS:
    NestedTensorAtenRegistry[_op] = _random_creation_handler
for _op in [aten.isin.Tensor_Tensor, aten.isin.Tensor_Scalar, aten.bucketize.Tensor]:
    NestedTensorAtenRegistry[_op] = _unary_handler
