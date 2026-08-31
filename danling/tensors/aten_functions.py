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
``__torch_dispatch__`` handlers for NestedTensor aten ops (**Level 1** dispatch).

This module implements the dispatch table that maps aten ops to optimized handlers
operating on the packed representation (_values, _offsets, _physical_shape).

Architecture:
    - Elementwise ops operate directly on ``_values`` (no unpack/repack overhead)
    - Structural ops (clone, detach, to_copy) operate on all inner tensors
    - Unregistered ops fall back to per-element application via ``_storage``
"""

from __future__ import annotations

import builtins
import math
from collections.abc import Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from .ops import (
    _MISSING,
    ATEN_BINARY_ELEMENTWISE_OPS,
    ATEN_UNARY_ELEMENTWISE_OPS,
    NestedTensorAtenRegistry,
    _binary_complementary_singleton_square,
    _binary_op_maybe_tensor,
    _broadcast_lower_rank_nested_to_values,
    _broadcast_nested_to_values,
    _check_execution_guard,
    _compile_unsupported,
    _complementary_singleton_square_operands,
    _dense_alignment,
    _dense_alignment_to_values,
    _dense_operand_for_element,
    _ExecutionGuardKind,
    _get_batch_dim,
    _is_compiling,
    _is_packed_identity,
    _maybe_align_dense_to_nested,
    _nested_like_elements,
    _normalize_dim,
    _physical_to_values_dim,
    _resolve_dense_for_values,
    _stack_or_nest,
    _translate_dim,
    _translate_dims,
)

if TYPE_CHECKING:
    from .nested_tensor import NestedTensor

aten = torch.ops.aten

#: Sentinel for a ternary operand that no packed resolution serves.
_UNRESOLVED = object()

try:
    from torch._subclasses.fake_tensor import is_fake as _torch_is_fake
except ImportError:
    _torch_is_fake = None


def _is_fake_tensor(tensor: Tensor) -> bool:
    if _torch_is_fake is None:
        return False
    return bool(_torch_is_fake(tensor))


def _offsets_match_identity_if_fake(a: Tensor, b: Tensor) -> bool:
    r"""
    Check if two offset tensors represent the same packing layout.

    Under fake tensor mode, requires object identity (conservative).
    Under eager mode, uses pointer identity as a fast-path before falling back
    to elementwise comparison.
    """
    if _is_fake_tensor(a) or _is_fake_tensor(b):
        return a is b
    with suppress(RuntimeError):
        if a.data_ptr() == b.data_ptr():
            return True
    if a.shape != b.shape:
        return False
    return bool(torch.equal(a, b))


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
        The inner serial execution helper (``_per_element_fallback_serial``) is
        ``@torch._dynamo.disable``, so any op that reaches the serial fallback
        body will exit a compiled graph. Register aten-level handlers in
        ``NestedTensorAtenRegistry`` for ops that must be compile-friendly.
    """
    from .nested_tensor import NestedTensor
    from .ops import _compile_unsupported, _is_compiling

    _check_execution_guard(_ExecutionGuardKind.EAGER_FALLBACK, "aten_functions.per_element_fallback")
    if _is_compiling():
        name = getattr(func, "__name__", None) or getattr(func, "_schema", None) or repr(func)
        _compile_unsupported(str(name), "would fall back to per-element eager execution")

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
            return source.packed_like(t)

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


def _apply_per_element_nested(source: NestedTensor, op, *, ragged_dims=_MISSING):
    r"""
    Apply ``op`` to each element and always rebuild a NestedTensor.

    Unlike ``per_element_fallback``, this helper is not ``@torch._dynamo.disable``.
    Use this only when we intentionally preserve NestedTensor output structure.
    """
    _check_execution_guard(_ExecutionGuardKind.STORAGE_MAP, "_apply_per_element_nested")
    cls = type(source)
    if len(source) == 0:
        return cls([], **source._meta(include_dtype=True))
    elements = source._unpack()
    outputs = tuple(op(t) for t in elements)
    meta = dict(source._meta())
    if source._ragged_dims_explicit:
        if ragged_dims is _MISSING:
            shape_preserving = all(output.shape == element.shape for output, element in zip(outputs, elements))
            meta["ragged_dims"] = source._ragged_dims if shape_preserving else None
        else:
            meta["ragged_dims"] = ragged_dims
    return cls(outputs, **meta)


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
        if source._has_same_structure(other):
            return other._values
        if len(source) != len(other):
            raise ValueError(
                "NestedTensor batch length mismatch between source and other: "
                f"source={len(source)}, other={len(other)}"
            )
        raise NotImplementedError(f"NestedTensor: {func} with mismatched packing layouts")
    device = source._values.device
    if isinstance(other, Tensor) and other.dim() > 0:
        aligned = _maybe_align_dense_to_nested(source, other)
        if aligned is not None and source._has_same_structure(aligned):
            values = aligned._values
            return values if values.device == device else values.to(device=device)
        candidate = other if other.device == device else other.to(device=device)
        resolved = _resolve_dense_for_values(source, candidate)
        if resolved is not None:
            return resolved
        raise NotImplementedError(
            f"NestedTensor: {func} with non-scalar Tensor operand that is neither shape-aligned nor "
            "broadcast-compatible with packed values"
        )
    return other


def _is_scalar_binary_overload(func) -> bool:
    return "Scalar" in str(getattr(func, "_overloadname", ""))


def _resolve_ternary_other(source, other):
    r"""
    Resolve a ternary-op operand against ``source``, or report that nothing packed applies.

    This accepts layout-aligned, singleton-prefix, and lower-rank NestedTensor
    operands plus the same dense tensor cases as ``_resolve_other``. A dense
    operand is read against the logical shape by
    :func:`_resolve_dense_for_values` and rewritten into packed axis order, which is what keeps
    a ``[B, 1, 1, C]`` operand a per-sample slab rather than an extra logical dimension.
    Returns ``_UNRESOLVED`` when the operand needs the per-element path; an operand whose shape
    has more than one reading raises from ``_resolve_dense_for_values`` and is not swallowed.
    """
    from .nested_tensor import NestedTensor

    device = source._values.device
    if isinstance(other, NestedTensor):
        if source._has_same_structure(other):
            values = other._values
            return values if values.device == device else values.to(device=device)
        values = _broadcast_nested_to_values(source, other)
        if values is None:
            values = _broadcast_lower_rank_nested_to_values(source, other)
        if values is None:
            return _UNRESOLVED
        return values if values.device == device else values.to(device=device)
    if isinstance(other, Tensor):
        if other.dim() == 0:
            return other if other.device == device else other.to(device=device)
        aligned = _maybe_align_dense_to_nested(source, other)
        if aligned is not None and source._has_same_structure(aligned):
            values = aligned._values
            return values if values.device == device else values.to(device=device)
        # Not ``_resolve_dense_for_values``: its packed-shape rule reads an operand shaped like
        # ``_values`` as elementwise on the packed rows, which for a ternary op would silently
        # accept a tensor that no element broadcasts against. Only the alignment applies here.
        candidate = other if other.device == device else other.to(device=device)
        reading = _dense_alignment(source, candidate)
        if reading is None:
            return _UNRESOLVED
        resolved = _dense_alignment_to_values(source, candidate, reading)
        return _UNRESOLVED if resolved is None else resolved
    return other


def _elementwise_binary_handler(func, args, kwargs):
    r"""Dispatch handler for elementwise binary ops on packed _values."""
    if not _is_scalar_binary_overload(func):
        return _binary_op_maybe_tensor(args[0], args[1], func, *args[2:], **kwargs)

    from .nested_tensor import NestedTensor

    lhs, rhs = args[0], args[1]
    extra = args[2:]
    if isinstance(lhs, NestedTensor):
        resolved = _resolve_other(lhs, rhs, func)
        return lhs._packed_like_unchecked(func(lhs._values, resolved, *extra, **kwargs))
    resolved = _resolve_other(rhs, lhs, func)
    return rhs._packed_like_unchecked(func(resolved, rhs._values, *extra, **kwargs))


# ---------------------------------------------------------------------------
# Elementwise unary ops — apply directly to _values
# ---------------------------------------------------------------------------


def _elementwise_unary_handler(func, args, kwargs):
    r"""Dispatch handler for elementwise unary ops applied to _values."""
    source = args[0]
    return source._packed_like_unchecked(func(source._values, *args[1:], **kwargs))


# ---------------------------------------------------------------------------
# Global reductions — reduce all of _values to a scalar (no dim argument)
# ---------------------------------------------------------------------------

ATEN_GLOBAL_REDUCTION_OPS = [
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


def _extract_dim_keepdim(args, kwargs, default_dim):
    r"""Parse (source, dim, keepdim) from aten dispatch args/kwargs."""
    source = args[0]
    kw_dim = kwargs.pop("dim", _MISSING)
    kw_keepdim = kwargs.pop("keepdim", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            raise TypeError("got multiple values for argument 'dim'")
        dim_arg = args[1]
    else:
        dim_arg = default_dim if kw_dim is _MISSING else kw_dim
    if len(args) > 2:
        if kw_keepdim is not _MISSING:
            raise TypeError("got multiple values for argument 'keepdim'")
        keepdim = args[2]
    else:
        keepdim = False if kw_keepdim is _MISSING else kw_keepdim
    return source, _parse_dims_arg(dim_arg), keepdim


def _dim_reduction_dispatch(func, source, dims, keepdim, kwargs, *, ragged_fill, keepdim_kw=False, none_dim):
    r"""
    Shared 4-way dispatch for single-value dim reductions.

    Args:
        ragged_fill: Fill value for fallback padded ragged-dim-0 path, or ``None`` for per-element fallback.
        keepdim_kw: If True, pass keepdim as keyword arg (for std/var correction schema).
        none_dim: What to pass as the dim argument for "reduce all elements" calls.
            ``None`` for ops like sum/mean, ``[]`` for ops like amax/amin.

    Branch structure:
    1. ``len(dims) == 0`` → global reduction on packed ``_values``
    2. ``len(dims) > 1`` → multi-dim: packed fast path on static dims, padded or fallback for ragged
    3. ``dim == batch_dim`` → stack per-element reductions
    4. single ragged dim → segment reduce when supported, otherwise padded/fallback
    5. ``dim_adj > 0`` (static) → apply directly to packed ``_values``
    """

    def _call(values_or_padded, dim_arg, kd):
        if keepdim_kw:
            return func(values_or_padded, dim_arg, keepdim=kd, **kwargs)
        return func(values_or_padded, dim_arg, kd, **kwargs)

    def _fallback(dim_arg, kd):
        if keepdim_kw:
            kwargs["keepdim"] = kd
            return per_element_fallback(func, (source, dim_arg), kwargs)
        return per_element_fallback(func, (source, dim_arg, kd), kwargs)

    if len(dims) == 0:
        return _call(source._values, none_dim, keepdim)

    if len(dims) > 1:
        try:
            dims_adj = _translate_dims(source, dims)
        except ValueError as exc:
            raise NotImplementedError(f"NestedTensor: {func} with dim={dims} is not supported") from exc
        values_dims = tuple(_physical_to_values_dim(source, dim_i) for dim_i in dims_adj)
        if all(dim_i is not None for dim_i in values_dims):
            return _reduce_non_ragged_packed_dims(
                source,
                _call(source._values, [int(dim_i) for dim_i in values_dims], keepdim),
                dims_adj,
                keepdim,
            )
        if not _is_packed_identity(source) or source._ragged_rank > 1:
            return _fallback(list(dims_adj), keepdim)
        if ragged_fill is not None and 0 in dims_adj:
            padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=ragged_fill)
            # Padded has shape [B, max_len, ...], so element dim d maps to padded dim d+1
            padded_dims = [dim_i + 1 for dim_i in dims_adj]
            output = _call(padded, padded_dims, keepdim)
            return _restore_multi_dim_batch_dim(source, output, dims_adj, keepdim)
        return _fallback(list(dims_adj), keepdim)

    dim = _normalize_dim(dims[0], source.dim())
    batch_dim = _get_batch_dim(source)
    if dim == batch_dim:
        reduced = torch.stack([_call(t, none_dim, False) for t in source._storage])
        if keepdim:
            return reduced.unsqueeze(batch_dim)
        return reduced

    dim_adj = _translate_dim(source, dim)
    values_dim = _physical_to_values_dim(source, dim_adj)
    if values_dim is not None:
        return _reduce_non_ragged_packed(
            source,
            _call(source._values, [values_dim], keepdim),
            dim_adj,
            keepdim,
        )
    if source._ragged_rank > 1:
        # Multiple ragged levels collapse into one packed dim in ``_values``, so a per-element dim index no
        # longer maps onto the packed layout; reduce per element, where each item still carries its full rank.
        return _fallback([dim_adj], keepdim)
    segment_reduced = _segment_reduce_ragged_dim(func, source, dim_adj, keepdim, kwargs)
    if segment_reduced is not None:
        return segment_reduced

    if not _is_packed_identity(source):
        if dim_adj in source._varying_dims:
            reduced = torch.stack([_call(t, [dim_adj], False) for t in source._unpack()])
            if keepdim:
                reduced = reduced.unsqueeze(dim)
            return reduced
        return _fallback([dim_adj], keepdim)
    if dim_adj == 0:
        if ragged_fill is None:
            # Reducing the variable-length dim always produces uniform elements.
            # Stack into a regular tensor rather than returning an NT.
            reduced = torch.stack([_call(t, [0], False) for t in source._storage])
            if keepdim:
                reduced = reduced.unsqueeze(dim)
            return reduced
        padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=ragged_fill)
        output = _call(padded, [1], keepdim)
        return _restore_segment_batch_dim(source, output, dim_adj, keepdim)

    return _reduce_non_ragged_packed(source, _call(source._values, [dim_adj], keepdim), dim_adj, keepdim)


@NestedTensorAtenRegistry.implement(aten.argmax.default)
@NestedTensorAtenRegistry.implement(aten.argmin.default)
def arg_extrema_reduction(func, args, kwargs):
    r"""Handle ``argmax/argmin`` for per-element global or dim reductions."""
    source, dims, keepdim = _extract_dim_keepdim(args, kwargs, None)
    largest = func is aten.argmax.default

    if not dims:
        output = torch.stack([func(t, **kwargs) for t in source._storage])
        if keepdim:
            output = output.unsqueeze(_get_batch_dim(source))
        return output

    dim = _normalize_dim(dims[0], source.dim())
    batch_dim = _get_batch_dim(source)
    if dim == batch_dim:
        output = torch.stack([func(t, **kwargs) for t in source._storage])
        if keepdim:
            output = output.unsqueeze(batch_dim)
        return output

    dim_adj = _translate_dim(source, dim)
    segment_indices = _segment_arg_extrema_ragged_dim(source, dim_adj, keepdim, largest=largest)
    if segment_indices is not None:
        return segment_indices

    values_dim = _physical_to_values_dim(source, dim_adj)
    if values_dim is None:
        if not _has_single_packed_ragged_dim(source, dim_adj):
            return per_element_fallback(func, (source, dim_adj, keepdim), kwargs)
        fill_value = _topk_fill_value(source._values.dtype, largest=largest)
        padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=fill_value)
        output = func(padded, 1 + dim_adj, keepdim, **kwargs)
        return _restore_segment_batch_dim(source, output, dim_adj, keepdim)

    out_values = func(source._values, values_dim, keepdim, **kwargs)
    return _reduce_non_ragged_packed(source, out_values, dim_adj, keepdim)


@NestedTensorAtenRegistry.implement(aten.count_nonzero.dim_IntList)
def count_nonzero_dim_reduction(func, args, kwargs):
    r"""Handle ``count_nonzero`` dim reductions on packed values for common dim patterns."""
    source, dims, _ = _extract_dim_keepdim(args, kwargs, ())
    if len(dims) == 0:
        return aten.count_nonzero.default(source._values, **kwargs)

    if len(dims) > 1:
        dims_adj = _translate_dims(source, dims)
        values_dims = tuple(_physical_to_values_dim(source, dim_i) for dim_i in dims_adj)
        if all(dim_i is not None for dim_i in values_dims):
            out_values = func(source._values, list(cast(tuple[int, ...], values_dims)), **kwargs)
            return _reduce_non_ragged_packed_dims(source, out_values, dims_adj, keepdim=False)
        if source._ragged_rank <= 1 and 0 in dims_adj and _is_packed_identity(source):
            padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=0)
            padded_dims = [1 + dim_i for dim_i in dims_adj]
            output = func(padded, padded_dims, **kwargs)
            return _restore_multi_dim_batch_dim(source, output, dims_adj, keepdim=False)
        return per_element_fallback(func, (source, list(dims_adj)), kwargs)

    dim = _normalize_dim(dims[0], source.dim())
    batch_dim = _get_batch_dim(source)
    if dim == batch_dim:
        return torch.stack([torch.count_nonzero(t) for t in source._storage])

    dim_adj = _translate_dim(source, dim)
    segment_counts = _segment_count_nonzero_ragged_dim(source, dim_adj)
    if segment_counts is not None:
        return segment_counts

    values_dim = _physical_to_values_dim(source, dim_adj)
    if values_dim is None:
        if not _has_single_packed_ragged_dim(source, dim_adj):
            return per_element_fallback(func, (source, [dim_adj]), kwargs)
        padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=0)
        output = func(padded, [1 + dim_adj], **kwargs)
        return _restore_segment_batch_dim(source, output, dim_adj, keepdim=False)

    out_values = func(source._values, [values_dim], **kwargs)
    return _reduce_non_ragged_packed(source, out_values, dim_adj, keepdim=False)


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
        return _stack_or_nest(values, source), _stack_or_nest(indices, source)

    values_out, indices_out = apply(source._values, dim_adj, keepdim)
    return (
        _reduce_non_ragged_packed(source, values_out, dim_adj, keepdim),
        _reduce_non_ragged_packed(source, indices_out, dim_adj, keepdim),
    )


@NestedTensorAtenRegistry.implement(aten.kthvalue.default)
def kthvalue_reduction(func, args, kwargs):
    r"""Handle ``kthvalue`` dim reductions on static packed dims."""
    source = args[0]
    kw_k = kwargs.pop("k", _MISSING)
    if len(args) > 1:
        if kw_k is not _MISSING:
            raise TypeError("got multiple values for argument 'k'")
        k = args[1]
    else:
        k = kw_k
    if k is _MISSING:
        raise TypeError("missing required argument 'k'")
    # Shift args so _extract_dim_keepdim sees (source, dim, keepdim) starting at args[2]
    _, dims, keepdim = _extract_dim_keepdim((source, *args[2:]), kwargs, -1)
    dim = dims[0] if dims else -1
    return _order_stat_pair_reduction(source, dim, keepdim, lambda t, d, kd: func(t, k, d, kd, **kwargs))


def _order_stat_dim_handler(func, args, kwargs, default_dim):
    r"""Shared handler for order-stat pair reductions (median, nanmedian, mode)."""
    source, dims, keepdim = _extract_dim_keepdim(args, kwargs, default_dim)
    if default_dim is _MISSING and not dims:
        raise TypeError("missing required argument 'dim'")
    dim = dims[0] if dims else default_dim
    return _order_stat_pair_reduction(source, dim, keepdim, lambda t, d, kd: func(t, d, kd, **kwargs))


@NestedTensorAtenRegistry.implement(aten.linalg_vector_norm.default)
def linalg_vector_norm(func, args, kwargs):
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
    dims = _parse_dims_arg(dim_arg)
    if len(dims) == 0:
        if not _vector_norm_zero_padding_safe(ord_value):
            return per_element_fallback(
                func,
                (source, ord_value, None, keepdim),
                {"dtype": dtype, **kwargs},
            )
        segment_norm = _segment_vector_norm_ragged_dim(source, ord_value, None, keepdim, dtype=dtype)
        if segment_norm is not None:
            return segment_norm
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
    values_dim = _physical_to_values_dim(source, dim_adj)
    if values_dim is None:
        if _vector_norm_zero_padding_safe(ord_value):
            segment_norm = _segment_vector_norm_ragged_dim(source, ord_value, dim_adj, keepdim, dtype=dtype)
            if segment_norm is not None:
                return segment_norm
            if _has_single_packed_ragged_dim(source, dim_adj):
                padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=0)
                out_values = func(padded, ord_value, [1 + dim_adj], keepdim, dtype=dtype, **kwargs)
                return _from_uniform_batched_output(source, out_values)
        return per_element_fallback(
            func,
            (source, ord_value, [dim_adj], keepdim),
            {"dtype": dtype, **kwargs},
        )

    out_values = func(source._values, ord_value, [values_dim], keepdim, dtype=dtype, **kwargs)
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


@NestedTensorAtenRegistry.implement(aten.max.dim)
@NestedTensorAtenRegistry.implement(aten.min.dim)
def max_min_dim_reduction(func, args, kwargs):
    r"""Handle ``max/min`` dim reductions, returning both values and indices."""
    source, dims, keepdim = _extract_dim_keepdim(args, kwargs, _MISSING)
    if not dims:
        raise TypeError("missing required argument 'dim'")
    dim = _normalize_dim(dims[0], source.dim())
    batch_dim = _get_batch_dim(source)
    largest = func is aten.max.dim

    if dim == batch_dim:
        pairs = [func(t.reshape(-1), 0, False, **kwargs) for t in source._unpack()]
        values = torch.stack([pair[0] for pair in pairs])
        indices = torch.stack([pair[1] for pair in pairs])
        if keepdim:
            values = values.unsqueeze(batch_dim)
            indices = indices.unsqueeze(batch_dim)
        return values, indices

    dim_adj = _translate_dim(source, dim)
    segment_pair = _segment_max_min_ragged_dim(source, dim_adj, keepdim, largest=largest)
    if segment_pair is not None:
        return segment_pair

    values_dim = _physical_to_values_dim(source, dim_adj)
    if values_dim is None:
        if not _has_single_packed_ragged_dim(source, dim_adj):
            return per_element_fallback(func, (source, dim_adj, keepdim), kwargs)
        fill_value = _topk_fill_value(source._values.dtype, largest=largest)
        padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=fill_value)
        values, indices = func(padded, 1 + dim_adj, keepdim, **kwargs)
        return (
            _restore_segment_batch_dim(source, values, dim_adj, keepdim),
            _restore_segment_batch_dim(source, indices, dim_adj, keepdim),
        )

    values, indices = func(source._values, values_dim, keepdim, **kwargs)
    return (
        _reduce_non_ragged_packed(source, values, dim_adj, keepdim),
        _reduce_non_ragged_packed(source, indices, dim_adj, keepdim),
    )


@NestedTensorAtenRegistry.implement(aten.var_mean.correction)
def var_mean_dim_reduction(func, args, kwargs):
    r"""Handle ``var_mean`` correction reductions via packed fastpaths where valid."""
    source, dims, keepdim = _extract_dim_keepdim(args, kwargs, None)
    if len(dims) == 0:
        out_var, out_mean = func(source._values, None, keepdim=keepdim, **kwargs)
        return out_var, out_mean

    if len(dims) > 1:
        dims_adj = _translate_dims(source, dims)
        values_dims = tuple(_physical_to_values_dim(source, dim_i) for dim_i in dims_adj)
        if all(dim_i is not None for dim_i in values_dims):
            out_var, out_mean = func(
                source._values,
                list(cast(tuple[int, ...], values_dims)),
                keepdim=keepdim,
                **kwargs,
            )
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
    values_dim = _physical_to_values_dim(source, dim_adj)
    if values_dim is None:
        kwargs["keepdim"] = keepdim
        return per_element_fallback(func, (source, [dim_adj]), kwargs)

    out_var, out_mean = func(source._values, [values_dim], keepdim=keepdim, **kwargs)
    return (
        _reduce_non_ragged_packed(source, out_var, dim_adj, keepdim),
        _reduce_non_ragged_packed(source, out_mean, dim_adj, keepdim),
    )


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


def _masked_scatter_supply_suffices(mask: NestedTensor, source: NestedTensor) -> bool:
    r"""
    Check that every element of ``source`` supplies the scalars its own mask element selects.

    Dense ``masked_scatter`` consumes a prefix of the source and ignores the rest, so a surplus
    is legal and only a shortfall is an error. Fake tensors expose no mask values to count, so
    tracing accepts the packed path and leaves the shortfall to be caught eagerly.
    """
    if len(mask) != len(source):
        return False
    if _is_fake_tensor(mask._values) or _is_fake_tensor(source._values):
        return True
    source_numel = _per_element_numel(source).to(device=mask._values.device)
    return bool(torch.all(_per_element_true_counts(mask) <= source_numel))


def _masked_scatter_packed_supported(input: NestedTensor, mask: NestedTensor, source: NestedTensor) -> bool:
    r"""
    Check that packed ``masked_scatter`` reads the source in the order dense semantics demand.

    Dense ``masked_scatter`` walks the destination in row-major order and takes the next source
    scalar at every selected position. Packed storage keeps each element contiguous but stores
    it under the packed permutation, so packed order and element row-major order are the same
    sequence only under an identity permutation. Under any other permutation the packed pass
    would hand the right values to the wrong positions and return without complaint.
    """
    if not input._has_same_layout(mask):
        # A broadcast mask is valid per element but not packed-safe: the source stream would
        # need counts from the broadcast mask, not from the stored pre-broadcast values.
        return False
    if not (_is_packed_identity(input) and _is_packed_identity(source)):
        return False
    return _masked_scatter_supply_suffices(mask, source)


def _packed_masked_scatter(input: NestedTensor, mask: NestedTensor, source: NestedTensor) -> Tensor:
    r"""
    Fill ``input``'s selected positions from ``source`` without unpacking either side.

    Each element consumes its own slice of the source, so the position a packed scalar reads
    from is its rank among the selected positions of its *element*: a global running count of
    selected positions, less the count standing at that element's first flat position. Both
    sides being packed, those flat boundaries are just offsets scaled by the static tail.

    The source buffer carries one spare trailing slot so unselected positions have somewhere
    harmless to point; without it an all-``False`` mask or an empty source would index an
    empty buffer.
    """
    from .segmented import align_rows

    values = input._values
    mask_values = mask._values.to(device=values.device)
    source_values = source._values.to(device=values.device, dtype=values.dtype)
    flat_mask = mask_values.reshape(-1)
    flat_source = source_values.reshape(-1)

    selected = flat_mask.to(torch.long).cumsum(0)
    rank = selected - flat_mask.to(torch.long)
    standing = torch.cat((selected.new_zeros(1), selected))

    offsets = input._offsets.to(device=values.device, dtype=torch.long)
    source_offsets = source._offsets.to(device=values.device, dtype=torch.long)
    element_rank = standing.index_select(0, offsets[:-1] * math.prod(values.shape[1:]))
    element_start = source_offsets[:-1] * math.prod(source_values.shape[1:])

    rows = align_rows(input.packed_batch_indices(device=values.device), mask_values).reshape(-1)
    read = element_start.index_select(0, rows) + rank - element_rank.index_select(0, rows)
    read = torch.where(flat_mask, read, torch.zeros_like(read))
    spare = torch.cat((flat_source, flat_source.new_zeros(1)))
    return torch.where(flat_mask, spare.index_select(0, read), values.reshape(-1)).reshape(values.shape)


def _plain_filled_by_nested_mask(source, mask, value, func, kwargs):
    r"""
    Fill a dense ``source`` by a ragged ``mask``, yielding a ragged result.

    ``__torch_function__`` also dispatches here when only the mask is nested, as in
    ``dense_bias.masked_fill(ragged_mask, -inf)``. Each element becomes
    ``source_element.masked_fill(mask_element, value)``: the source is sliced down to the
    element's ragged extent on the mask's varying dims while broadcast dims stay full, so a
    padded ``(B, H, N, N)`` attention bias filled by a ``(B, 1, N, N)`` ragged mask gives a
    ragged ``(H, n_i, n_i)`` result.
    """
    if not isinstance(source, Tensor) or source.dim() == 0:
        return type(mask)((func(source, element, value, **kwargs) for element in mask._storage), **mask._meta())
    batch_dim = 0 if mask.batch_first else 1
    if source.size(batch_dim) != len(mask):
        raise ValueError(
            "NestedTensor batch length mismatch between source and mask: "
            f"source={source.size(batch_dim)}, mask={len(mask)}"
        )
    varying = {int(dim) for dim in mask._varying_dims}
    results = []
    for index, element in enumerate(mask._storage):
        element_source = source.select(batch_dim, index)
        offset = element_source.dim() - element.dim()
        slices = [slice(None)] * element_source.dim()
        for dim in range(element.dim()):
            extent = int(element.shape[dim])
            if dim in varying or (extent != 1 and extent < int(element_source.shape[offset + dim])):
                slices[offset + dim] = slice(0, extent)
        results.append(func(element_source[tuple(slices)], element, value, **kwargs))
    return type(mask)(results, **mask._meta())


def _packed_masked_fill_supported(source: NestedTensor, mask: NestedTensor) -> bool:
    r"""Return whether ``mask`` broadcasts directly over ``source``'s packed rows."""
    if not source._has_same_structure(mask):
        return False
    if source._values.dim() != mask._values.dim():
        return False

    from torch.fx.experimental.symbolic_shapes import statically_known_true

    return all(
        statically_known_true(mask_size == 1) or statically_known_true(mask_size == source_size)
        for source_size, mask_size in zip(source._values.shape[1:], mask._values.shape[1:])
    )


def _masked_fill_handler(func, args, kwargs):
    r"""Dispatch handler for masked_fill: packed fast path + per-element broadcast fallback."""
    from .nested_tensor import NestedTensor

    source, mask, value = args[0], args[1], args[2]
    if not isinstance(source, NestedTensor) and isinstance(mask, NestedTensor):
        return _plain_filled_by_nested_mask(source, mask, value, func, kwargs)
    if isinstance(mask, NestedTensor) and _packed_masked_fill_supported(source, mask):
        return source._packed_like_unchecked(func(source._values, mask._values, value, **kwargs))
    aligned = source._maybe_exact_shape_nested_like(mask)
    if aligned is not None:
        mask = aligned
    if isinstance(mask, NestedTensor):
        if len(source) != len(mask):
            raise ValueError(
                "NestedTensor batch length mismatch between input and mask: " f"input={len(source)}, mask={len(mask)}"
            )
        return type(source)(
            (func(t, m, value, **kwargs) for t, m in zip(source._storage, mask._storage)),
            **source._meta(),
        )
    if not isinstance(mask, Tensor):
        mask = torch.as_tensor(mask, dtype=torch.bool, device=source._values.device)
    padded = source.tensor
    filled = func(padded, mask.to(device=padded.device), value, **kwargs)
    return source.nested_like(filled)


@NestedTensorAtenRegistry.implement(aten.masked_select.default)
def masked_select(func, args, kwargs):
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
    if not source._has_same_layout(mask):
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
        validate=False,
    )


@NestedTensorAtenRegistry.implement(aten.nonzero.default)
def nonzero(func, args, kwargs):
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
        validate=False,
    )


def _masked_scatter_handler(func, args, kwargs):
    r"""Dispatch handler for masked_scatter, reading the source stream in packed order."""
    from .nested_tensor import NestedTensor

    input_tensor, mask, source = args[0], args[1], args[2]
    if kwargs:
        raise TypeError(f"NestedTensor: {func} got unexpected arguments {kwargs!r}")

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
    if not _masked_scatter_packed_supported(input_tensor, mask, source):
        raise NotImplementedError(
            f"NestedTensor: {func} requires an exact-shape mask, an identity packed layout, "
            "and a source that supplies every position its own mask element selects"
        )
    return input_tensor._packed_like_unchecked(_packed_masked_scatter(input_tensor, mask, source))


# ---------------------------------------------------------------------------
# In-place ops that operate directly on _values
# ---------------------------------------------------------------------------


@NestedTensorAtenRegistry.implement(aten.copy_.default)
def copy(func, args, kwargs):
    r"""In-place copy from src to dest, using packed ``_values`` only for exact layout matches."""
    from .nested_tensor import NestedTensor

    dest, src = args[0], args[1]
    if isinstance(src, NestedTensor) and dest._has_same_layout(src):
        func(dest._values, src._values, *args[2:], **kwargs)
        dest._invalidate_transient_caches()
        return dest
    raise NotImplementedError(f"NestedTensor: {func} requires matching packed layout")


# ---------------------------------------------------------------------------
# In-place variants of elementwise ops
# ---------------------------------------------------------------------------
ATEN_INPLACE_UNARY_OPS = [
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
    source._invalidate_transient_caches()
    return source


ATEN_INPLACE_BINARY_OPS = [
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
    source._invalidate_transient_caches()
    return source


# ---------------------------------------------------------------------------
# Indexing read ops — packed fast paths when index layouts align.
# ---------------------------------------------------------------------------


@NestedTensorAtenRegistry.implement(aten.gather.default)
def gather(func, args, kwargs):
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

        if _single_ragged_gather_supported(source, index, dim_adj):
            index_values = index._values.to(device=source._values.device, dtype=torch.long)
            packed_index = _segmented_row_index(
                source,
                index_values,
                dim,
                "gather",
                row_batch_indices=index.packed_batch_indices(device=source._values.device),
            )
            out_values = func(source.concat, 0, packed_index, sparse_grad=sparse_grad, **kwargs)
            return _packed_gather_output(source, index, out_values)

        if _multi_ragged_outer_gather_supported(source, index, dim_adj):
            index_values = index._values.to(device=source._values.device, dtype=torch.long)
            packed_index = _segmented_outer_row_index(source, index, index_values, dim, "gather")
            out_values = func(source._values, 0, packed_index, sparse_grad=sparse_grad, **kwargs)
            return _packed_gather_output(source, index, out_values)

        if source._has_same_structure(index):
            # Gather runs on the packed values, so a per-element dim has to be mapped onto its
            # packed axis: the packed axes are the permuted per-element dims with every ragged dim
            # collapsed into axis 0, so the two numberings agree only under an identity layout. A
            # static dim keeps an axis of its own, whatever its per-element position; a ragged one
            # resolves to a row of the packed buffer, which a read can name outright even where a
            # write cannot, because the index says which row rather than which stride to take.
            values_dim = _packed_static_dim(source, dim_adj)
            if values_dim is not None:
                out_values = func(source._values, values_dim, index._values, sparse_grad=sparse_grad, **kwargs)
                return _packed_gather_output(source, index, out_values)
            index_values = index._values.to(device=source._values.device, dtype=torch.long)
            if _packed_inner_ragged_dim(source, dim_adj):
                packed_index = _segmented_row_index(source, index_values, dim, "gather")
                out_values = func(source._values, 0, packed_index, sparse_grad=sparse_grad, **kwargs)
                return _packed_gather_output(source, index, out_values)
            if _packed_outer_ragged_dim(source, dim_adj):
                packed_index = _segmented_outer_row_index(source, index, index_values, dim, "gather")
                out_values = func(source._values, 0, packed_index, sparse_grad=sparse_grad, **kwargs)
                return _packed_gather_output(source, index, out_values)

        # Nothing above claimed a packed axis, so report the per-sample loop rather than letting a
        # warm ``_storage`` cache carry it past the guards unannounced.
        _check_execution_guard(_ExecutionGuardKind.EAGER_FALLBACK, "aten_functions.gather")
        storage = []
        for tensor, idx in zip(source._storage, index._storage):
            if idx.device != tensor.device:
                idx = idx.to(device=tensor.device)
            storage.append(func(tensor, dim_adj, idx, sparse_grad=sparse_grad, **kwargs))
        return type(source)(storage, **source._meta())

    if isinstance(index, Tensor) and _shared_index_gather_supported(source, dim_adj, index):
        return _shared_index_gather(source, dim, dim_adj, index, func, sparse_grad, kwargs)

    _check_execution_guard(_ExecutionGuardKind.EAGER_FALLBACK, "aten_functions.gather")
    storage = []
    for tensor in source._storage:
        idx = index
        if isinstance(idx, Tensor) and idx.device != tensor.device:
            idx = idx.to(device=tensor.device)
        storage.append(func(tensor, dim_adj, idx, sparse_grad=sparse_grad, **kwargs))
    return type(source)(storage, **source._meta())


def _single_ragged_gather_supported(source: NestedTensor, index: NestedTensor, dim_adj: int) -> bool:
    r"""Whether one packed ragged axis can be addressed from a different index topology."""
    return (
        source.batch_first == index.batch_first
        and len(source._ragged_dims) == 1
        and source._ragged_dims == index._ragged_dims
        and source._permutation == index._permutation
        and source._values.dim() == index._values.dim()
        and _packed_inner_ragged_dim(source, dim_adj)
    )


def _multi_ragged_outer_gather_supported(source: NestedTensor, index: NestedTensor, dim_adj: int) -> bool:
    r"""Whether an index topology can select rows from the outer of two ragged dims."""
    return (
        source.batch_first == index.batch_first
        and source._ragged_dims == index._ragged_dims
        and len(source._ragged_dims) == 2
        and source._permutation == index._permutation
        and source._values.dim() == index._values.dim()
        and _packed_outer_ragged_dim(source, dim_adj)
        and _packed_outer_ragged_dim(index, dim_adj)
    )


def _shared_index_gather_supported(source: NestedTensor, dim_adj: int, index: Tensor) -> bool:
    r"""
    Whether one dense index can be replayed against every sample without unpacking the batch.

    A dense index is not split across the batch: every sample reads the same positions, so the
    result is uniform and each sample contributes the same number of packed rows. That only
    describes a packed buffer while a single ragged level leads the layout, and only while the
    index fits inside the shortest sample -- an index reaching past a segment would be answered
    out of its neighbour instead of raising, so the ranges are checked before the kernel runs.
    """
    ragged_dims = source._ragged_dims
    if len(ragged_dims) != 1 or not _has_single_packed_ragged_dim(source, ragged_dims[0]):
        return False
    rank = int(source._physical_shape.size(1))
    if index.dim() != rank or len(source) == 0:
        return False
    if _is_fake_tensor(source._values) or _is_fake_tensor(source._offsets):
        return False
    # Every dim the gather does not read is addressed positionally, so it has to fit the shortest
    # sample: ``gather`` reads the index as given rather than broadcasting it, and a ragged dim
    # that overran would be answered out of the next sample.
    lengths = source._offsets[1:] - source._offsets[:-1]
    sizes = source._physical_shape
    for dim in range(rank):
        if dim == dim_adj:
            continue
        limit = int(lengths.min()) if dim == ragged_dims[0] else int(sizes[:, dim].min())
        if int(index.shape[dim]) > limit:
            return False
    return True


def _shared_index_gather(source: NestedTensor, dim, dim_adj: int, index: Tensor, func, sparse_grad, kwargs):
    r"""Gather every sample with the same dense index, on the packed values."""
    from .segmented import align_rows

    ragged_dim = source._ragged_dims[0]
    batch_size = len(source)
    count = int(index.shape[ragged_dim])
    offsets = source._offsets.to(device=source._values.device, dtype=torch.long)
    # The index arrives in per-element order; the packed rows enumerate it in permutation order,
    # once per sample.
    packed_index = index.to(device=source._values.device, dtype=torch.long).permute(source._permutation)
    packed_index = (
        packed_index.unsqueeze(0).expand(batch_size, *packed_index.shape).reshape(-1, *packed_index.shape[1:])
    )
    starts = torch.repeat_interleave(offsets[:-1], count, output_size=batch_size * count)

    values_dim = _packed_static_dim(source, dim_adj)
    if values_dim is not None:
        # Only the first ``count`` rows of each sample can be read, so narrow to them and let the
        # kernel run on the static axis it was asked for.
        rows = starts + torch.arange(count, device=starts.device).repeat(batch_size)
        out_values = func(
            source._values.index_select(0, rows), values_dim, packed_index, sparse_grad=sparse_grad, **kwargs
        )
    else:
        lengths = torch.repeat_interleave(offsets[1:] - offsets[:-1], count, output_size=batch_size * count)
        _check_segment_index_bounds(packed_index, lengths, dim, "gather")
        out_values = func(
            source._values, 0, align_rows(starts, packed_index) + packed_index, sparse_grad=sparse_grad, **kwargs
        )

    element_shape = tuple(int(size) for size in index.shape)
    new_physical_shape = source._physical_shape.new_tensor(element_shape).reshape(1, -1).expand(batch_size, -1).clone()
    return _packed_with_shape(
        source,
        out_values,
        new_physical_shape,
        offsets=torch.arange(batch_size + 1, device=source._offsets.device, dtype=source._offsets.dtype) * count,
        permutation=source._permutation,
        packed_sizes=(count,) * batch_size,
        element_shapes=(element_shape,) * batch_size,
    )


def _gather_equivalent_index(source: NestedTensor, index: NestedTensor, dim_adj: int) -> bool:
    r"""
    Whether ``take_along_dim`` along ``dim_adj`` is exactly ``gather`` for these operands.

    ``take_along_dim`` broadcasts its two arguments against each other on every dimension but
    the one it reads, while ``gather`` reads the index as given. They coincide -- and the packed
    gather paths become available -- once nothing else is left to broadcast.
    """
    if source._values.dim() != index._values.dim():
        return False
    if _single_ragged_gather_supported(source, index, dim_adj) or _multi_ragged_outer_gather_supported(
        source, index, dim_adj
    ):
        rank = int(source._physical_shape.size(1))
        if rank != int(index._physical_shape.size(1)):
            return False
        retained_dims = [axis for axis in range(rank) if axis != dim_adj]
        source_shape = source._physical_shape[:, retained_dims]
        index_shape = index._physical_shape[:, retained_dims].to(device=source_shape.device)
        runtime_assert = _is_compiling() or _is_fake_tensor(source_shape) or _is_fake_tensor(index_shape)
        if not type(source)._meta_tensor_equal(
            source_shape,
            index_shape,
            "take_along_dim requires matching sizes outside the selected dimension",
            runtime_assert=runtime_assert,
        ):
            return False
    elif not source._has_same_structure(index):
        return False
    read_axis = _packed_static_dim(source, dim_adj)
    if read_axis is None:
        read_axis = 0
    return all(
        axis == read_axis or size == other
        for axis, (size, other) in enumerate(zip(source._values.shape, index._values.shape))
    )


def _packed_gather_output(source: NestedTensor, index: NestedTensor, out_values: Tensor) -> NestedTensor:
    r"""
    Rebuild the result of a packed gather, which is shaped like the index rather than the source.

    ``gather`` may be handed an index narrower than the source on every dim it does not read, so
    the output takes the index's element shapes. Carrying them across rather than recovering them
    from the physical shape also keeps an empty segment's rank, which trailing-zero trimming drops.
    """
    result = type(source)._from_packed(
        out_values,
        index._offsets,
        index._physical_shape,
        permutation=index._permutation,
        ragged_dims=index._ragged_dims if index._ragged_dims_explicit else None,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=torch.Size(index._logical_shape),
        packed_sizes=index._packed_sizes,
        element_shapes=index._element_shapes,
        ragged_offsets=index._persistent_ragged_offsets(),
        validate=False,
    )
    if torch.is_grad_enabled() and out_values.requires_grad:
        return result._packed_like_unchecked(out_values)
    return result


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
    if (
        isinstance(src, NestedTensor)
        and dim_adj > 0
        and source._values.dim() > dim_adj
        and source._has_same_structure(src)
    ):
        # As with scatter, packed writes are only safe on static per-element dims.
        # The source layout must share offsets with the destination so row boundaries
        # remain aligned after concatenation.
        src_values = src._values
        if src_values.device != source._values.device:
            src_values = src_values.to(device=source._values.device)
        return source._packed_like_unchecked(apply_fn(source._values, dim_adj, index_values, src_values))

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


@NestedTensorAtenRegistry.implement(aten.index_add.default)
def index_add(func, args, kwargs):
    r"""Apply ``index_add`` with packed fast paths when the source layout aligns with the input."""
    source, dim, index, src = args[0], args[1], args[2], args[3]
    return _index_write_like(source, dim, index, src, lambda t, d, i, s: func(t, d, i, s, **kwargs), "index_add")


@NestedTensorAtenRegistry.implement(aten.index_copy.default)
def index_copy(func, args, kwargs):
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
def index_put(func, args, kwargs):
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
            return source._packed_like_unchecked(func(source._values, packed_indices, value_tensor, accumulate))

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


@NestedTensorAtenRegistry.implement(aten.index_select.default)
def index_select(func, args, kwargs):
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
        if _is_fake_tensor(index_cpu):
            raise NotImplementedError(
                "NestedTensor: aten.index_select.default requires a concrete batch index to preserve layout metadata"
            )
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

        selected_indices = tuple(int(i) for i in index_cpu.tolist())
        selected_packed_sizes = None
        selected_element_shapes = None
        if source._packed_sizes is not None:
            selected_packed_sizes = tuple(source._packed_sizes[i] for i in selected_indices)
        if source._element_shapes is not None:
            selected_element_shapes = tuple(source._element_shapes[i] for i in selected_indices)

        return type(source)._from_packed(
            out_values,
            out_offsets,
            out_shape,
            permutation=source._permutation,
            ragged_dims=source._ragged_dims if source._ragged_dims_explicit else None,
            batch_first=source.batch_first,
            padding_value=source.padding_value,
            mask_value=source.mask_value,
            pin_memory=source._pin_memory,
            packed_sizes=selected_packed_sizes,
            element_shapes=selected_element_shapes,
            validate=False,
        )

    dim_adj = _translate_dim(source, dim)
    # ``dim_adj`` numbers per-element dims; ``_values`` numbers the permuted, ragged-collapsed
    # ones. Passing ``dim_adj`` straight through selects along whichever axis happens to sit at
    # that position and then stamps the *requested* dim with the new size, so the values and the
    # metadata disagree and the tensor only fails later, on unpacking.
    values_dim = _packed_static_dim(source, dim_adj)
    if values_dim is not None:
        out_values = func(source._values, values_dim, index_device, **kwargs)
        return _packed_new_dim_size(source, out_values, dim_adj, index_device.numel())
    if _packed_inner_ragged_dim(source, dim_adj):
        return _packed_segment_index_select(source, dim_adj, index_device)
    return per_element_fallback(func, (source, dim_adj, index_device), kwargs)


def _packed_segment_index_select(source: NestedTensor, dim_adj: int, index: Tensor) -> NestedTensor:
    r"""
    Select the same positions out of every segment along the innermost ragged dim.

    A packed row steps along that dim one position at a time, so the whole batch is one
    ``index_select`` over rows: every segment contributes its own starting row plus the caller's
    positions. Selecting ``k`` of them leaves each segment ``k`` rows long, so the dim survives
    with a uniform extent and the segment boundaries move to a multiple of ``k``.
    """
    offsets = _inner_segment_offsets(source)
    count = int(index.numel())
    if count and offsets.numel() > 1 and not (_is_fake_tensor(source._values) or _is_fake_tensor(index)):
        shortest = int((offsets[1:] - offsets[:-1]).min())
        if bool(((index < 0) | (index >= shortest)).any()):
            raise IndexError(f"index_select: index is out of bounds for dimension {dim_adj} with size {shortest}")
    rows = (offsets[:-1].unsqueeze(1) + index.reshape(1, -1)).reshape(-1)
    new_physical_shape, _, element_shapes = source._shape_meta_from_components(replace_dims={int(dim_adj): count})
    # One segment per sample under a single ragged level; under two, the level above says how
    # many segments each sample owns, and each of them is now ``count`` rows long.
    if source._ragged_rank <= 1:
        groups = torch.arange(len(source) + 1, device=source._offsets.device, dtype=source._offsets.dtype)
    else:
        groups = source._ragged_level_offsets(0)
    return _packed_with_shape(
        source,
        source._values.index_select(0, rows),
        new_physical_shape,
        source._logical_shape_from_components(replace_dims={int(dim_adj): count}),
        offsets=groups * count,
        permutation=source._permutation,
        # Recomputing the packed sizes without naming the ragged dims would read whichever dim
        # leads each element shape, which is a ragged one only under an identity layout.
        packed_sizes=None if element_shapes is None else source._packed_sizes_like(element_shapes, source._ragged_dims),
        element_shapes=element_shapes,
    )


#: Returned by :func:`_packed_scatter_src` for a ``src`` the packed buffer cannot express.
_NO_PACKED_SRC = object()


def _packed_inner_ragged_dim(source: NestedTensor, dim_adj: int) -> bool:
    r"""
    Whether packed dim 0 steps along ``dim_adj`` one position at a time.

    Packed rows enumerate the ragged dims in permutation order, so advancing one row advances
    the *innermost* ragged coordinate by one and leaves the outer ones alone. That is what lets
    a per-sample index along that dim be rebased onto packed dim 0 by adding the row its
    segment starts at. Any other ragged dim would need a stride that varies from sample to
    sample, which no single packed axis provides.
    """
    ragged = tuple(source._varying_dims)
    if not ragged or dim_adj != ragged[-1]:
        return False
    rank = int(source._physical_shape.size(1))
    return tuple(source._permutation or tuple(range(rank)))[: len(ragged)] == ragged


def _packed_scatter_src(source: NestedTensor, src):
    r"""
    Return the packed stand-in for ``src``, or :data:`_NO_PACKED_SRC` when it has none.

    A scalar writes the same value everywhere and needs no translation. A NestedTensor sharing
    ``source``'s structure is already row-aligned with it. A dense tensor is neither: the
    per-element path replays the whole tensor once per sample, which one packed buffer with one
    row per sample cannot express.
    """
    from .nested_tensor import NestedTensor

    if isinstance(src, NestedTensor):
        if not source._has_same_structure(src):
            return _NO_PACKED_SRC
        values = src._values
        return values if values.device == source._values.device else values.to(device=source._values.device)
    if isinstance(src, Tensor):
        return _NO_PACKED_SRC
    return src


def _inner_segment_offsets(source: NestedTensor) -> Tensor:
    r"""Return the offsets delimiting the innermost ragged level along packed dim 0."""
    offsets = source._offsets if source._ragged_rank <= 1 else source._ragged_level_offsets(-1)
    return offsets.to(device=source._values.device, dtype=torch.long)


def _check_segment_index_bounds(index_values: Tensor, lengths: Tensor, dim: int, op_name: str) -> None:
    r"""
    Reject an index that would be rebased out of its own segment.

    Rebasing an index onto packed dim 0 turns an out-of-range write into a write into the
    neighbouring sample: the row it lands on is a real row of the packed buffer, so the dense
    kernel accepts it and the corruption is silent. The range therefore has to be checked here,
    against the sample's own extent, rather than left to the kernel.
    """
    from .segmented import align_rows

    outside = (index_values < 0) | (index_values >= align_rows(lengths, index_values))
    in_bounds = ~outside.any()
    if _is_compiling() or _is_fake_tensor(in_bounds):
        torch._assert_async(in_bounds, f"{op_name}: index is out of bounds for dimension {dim}")
        return
    if bool(in_bounds):
        return
    position = int(torch.nonzero(outside.reshape(-1))[0])
    row = position // max(1, index_values[0].numel())
    raise IndexError(
        f"{op_name}: index {int(index_values.reshape(-1)[position])} is out of bounds for "
        f"dimension {dim} with size {int(lengths[row])}"
    )


def _segmented_row_index(
    source: NestedTensor,
    index_values: Tensor,
    dim: int,
    op_name: str,
    *,
    row_batch_indices: Tensor | None = None,
) -> Tensor:
    r"""
    Rebase per-sample indices along the innermost ragged dim onto packed dim 0.

    Every operator that addresses a position along that dim -- a scatter destination, a gather
    source -- needs the same translation, because the packed buffer numbers rows across the
    whole batch while the caller numbers them inside one sample.
    """
    from .segmented import align_rows, segment_row_bounds

    offsets = _inner_segment_offsets(source)
    if row_batch_indices is None:
        starts, lengths = segment_row_bounds(offsets, source._values.shape[0])
    else:
        starts = offsets[:-1].index_select(0, row_batch_indices)
        lengths = (offsets[1:] - offsets[:-1]).index_select(0, row_batch_indices)
    _check_segment_index_bounds(index_values, lengths, dim, op_name)
    return align_rows(starts, index_values) + index_values


def _packed_outer_ragged_dim(source: NestedTensor, dim_adj: int) -> bool:
    r"""
    Whether ``dim_adj`` is the outer dim of a doubly-ragged layout that leads the packed order.

    Advancing along that dim skips a whole inner segment, and inner segments differ in length
    from sample to sample, so no packed axis has the stride. A read can still be expressed --
    the index names the segment rather than a stride -- which is what
    :func:`_segmented_outer_row_index` builds.
    """
    ragged = tuple(source._varying_dims)
    if len(ragged) != 2 or dim_adj != ragged[0]:
        return False
    rank = int(source._physical_shape.size(1))
    return tuple(source._permutation or tuple(range(rank)))[:2] == ragged


def _segmented_outer_row_index(
    source: NestedTensor,
    index: NestedTensor,
    index_values: Tensor,
    dim: int,
    op_name: str,
) -> Tensor:
    r"""
    Resolve indices along the outer of two ragged dims to the packed rows they name.

    A packed row belongs to one sample and has an inner coordinate. Moving along the outer dim
    keeps that coordinate, but source and index may contain different numbers of outer rows. Use
    the index topology to recover the retained coordinate, then address the selected row inside
    the corresponding source sample.
    """
    from .segmented import align_rows

    device = source._values.device
    batch_index, local_index = index._packed_batch_local_indices(device=device, dtype=torch.long)
    _, inner_position = index._packed_varying_coords(
        batch_index,
        local_index,
        device=device,
        dtype=torch.long,
    )

    outer_dim, inner_dim = source._varying_dims
    source_shape = source._physical_shape.to(device=device, dtype=torch.long)
    index_shape = index._physical_shape.to(device=device, dtype=torch.long)
    source_outer = source_shape[:, outer_dim]
    source_inner = source_shape[:, inner_dim]
    index_inner = index_shape[:, inner_dim]

    inner_fits = (index_inner <= source_inner).all()
    if _is_compiling() or _is_fake_tensor(inner_fits):
        torch._assert_async(
            inner_fits,
            f"{op_name}: index shape exceeds the input outside dimension {dim}",
        )
    elif not bool(inner_fits):
        raise RuntimeError(f"{op_name}: index shape exceeds the input outside dimension {dim}")

    outer_lengths = source_outer.index_select(0, batch_index)
    _check_segment_index_bounds(index_values, outer_lengths, dim, op_name)
    sample_starts = source._offsets.to(device=device, dtype=torch.long).index_select(0, batch_index)
    inner_lengths = source_inner.index_select(0, batch_index)
    base_rows = sample_starts + inner_position
    return align_rows(base_rows, index_values) + index_values * align_rows(inner_lengths, index_values)


def _scatter_compile_safe(args: tuple, kwargs: dict[str, object]) -> bool:
    r"""Return whether the scatter family resolves to a packed axis for these operands."""
    from .nested_tensor import NestedTensor

    if len(args) < 4:
        return False
    source, dim, index, src = args[0], args[1], args[2], args[3]
    if not isinstance(index, NestedTensor) or _packed_scatter_src(source, src) is _NO_PACKED_SRC:
        return False
    try:
        dim = _normalize_dim(dim, source.dim())
        if dim == _get_batch_dim(source):
            return False
        dim_adj = _translate_dim(source, dim)
    except (TypeError, ValueError, IndexError):
        return False
    if not source._has_same_structure(index):
        return False
    return _packed_static_dim(source, dim_adj) is not None or _packed_inner_ragged_dim(source, dim_adj)


def _scatter_like(source, dim, index, src, apply_fn, op_name: str):
    r"""Apply scatter-style writes on the packed values whenever the written dim maps to one axis."""
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
            f"{op_name}: NestedTensor batch length mismatch between input and source: input={len(source)}, "
            f"source={len(src)}"
        )

    src_values = _packed_scatter_src(source, src)
    if isinstance(index, NestedTensor) and src_values is not _NO_PACKED_SRC and source._has_same_structure(index):
        index_values = index._values.to(device=source._values.device, dtype=torch.long)
        # ``dim_adj`` numbers per-element dimensions and the packed axes are the permuted,
        # ragged-collapsed ones, so the two agree only under an identity layout. Map through the
        # layout instead: a static dim keeps its own axis, and the innermost ragged dim becomes
        # packed dim 0 once its indices are rebased onto the sample's starting row.
        values_dim = _packed_static_dim(source, dim_adj)
        if values_dim is not None:
            return source._packed_like_unchecked(apply_fn(source._values, values_dim, index_values, src_values))
        if _packed_inner_ragged_dim(source, dim_adj):
            packed_index = _segmented_row_index(source, index_values, dim, op_name)
            return source._packed_like_unchecked(apply_fn(source._values, 0, packed_index, src_values))

    # Nothing above claimed a packed axis, so report the per-sample loop rather than letting a
    # warm ``_storage`` cache carry it past the guards unannounced.
    _check_execution_guard(_ExecutionGuardKind.EAGER_FALLBACK, f"aten_functions._scatter_like({op_name})")
    if _is_compiling():
        _compile_unsupported(op_name, "the written dimension does not resolve to a single packed axis")

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


@NestedTensorAtenRegistry.implement(
    aten.scatter_add.default,
    compile_safe=True,
    compile_guard=_scatter_compile_safe,
)
def scatter_add(func, args, kwargs):
    r"""Apply ``scatter_add`` with packed fast paths when index/source layouts align."""
    source, dim, index, src = args[0], args[1], args[2], args[3]
    return _scatter_like(source, dim, index, src, lambda t, d, i, s: func(t, d, i, s, **kwargs), "scatter_add")


@NestedTensorAtenRegistry.implement(
    aten.scatter.src,
    compile_safe=True,
    compile_guard=_scatter_compile_safe,
)
def scatter_src(func, args, kwargs):
    r"""Apply ``scatter`` with Tensor/NestedTensor src via packed fast paths when layouts align."""
    source, dim, index, src = args[0], args[1], args[2], args[3]
    return _scatter_like(source, dim, index, src, lambda t, d, i, s: func(t, d, i, s, **kwargs), "scatter")


@NestedTensorAtenRegistry.implement(
    aten.scatter.value,
    compile_safe=True,
    compile_guard=_scatter_compile_safe,
)
def scatter_value(func, args, kwargs):
    r"""Apply scalar ``scatter`` with packed fast paths when index layouts align."""
    source, dim, index, value = args[0], args[1], args[2], args[3]
    return _scatter_like(source, dim, index, value, lambda t, d, i, s: func(t, d, i, s, **kwargs), "scatter")


if hasattr(aten, "scatter_reduce"):

    @NestedTensorAtenRegistry.implement(
        aten.scatter_reduce.two,
        compile_safe=True,
        compile_guard=_scatter_compile_safe,
    )
    def scatter_reduce(func, args, kwargs):
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


@NestedTensorAtenRegistry.implement(aten.take.default)
def take(func, args, kwargs):
    r"""Apply ``take`` over the flattened packed storage for plain tensor indices."""
    source = args[0]
    index = args[1]
    from .nested_tensor import NestedTensor

    if not isinstance(index, Tensor) or isinstance(index, NestedTensor):
        raise NotImplementedError(f"NestedTensor: {func} requires a Tensor index")
    if index.device != source._values.device:
        index = index.to(device=source._values.device)
    return func(source._values.reshape(-1), index, **kwargs)


# Matrix multiply ops — apply to _values, update last dim of _physical_shape
# ---------------------------------------------------------------------------


def _packed_new_last_dim(source: NestedTensor, new_values: Tensor, new_last_dim) -> NestedTensor:
    r"""Rebuild a NestedTensor with a changed last dimension (e.g. after matmul)."""
    physical_rank = int(source._physical_shape.size(1))
    if physical_rank > 0:
        last_dim = physical_rank - 1
        if last_dim not in source._static_dims:
            raise NotImplementedError("Packed last-dimension updates require the last physical dim to be static.")
        return _packed_new_dim_size(source, new_values, last_dim, int(new_last_dim))

    new_physical_shape = source._physical_shape.clone()
    new_physical_shape = new_physical_shape.new_full((len(source), 1), new_last_dim)

    packed_sizes = None
    element_shapes = None
    if source._element_shapes is not None and isinstance(new_last_dim, int):
        element_shapes = tuple((int(new_last_dim),) for _ in source._element_shapes)
        packed_sizes = tuple(int(new_last_dim) for _ in source._element_shapes)

    new_outer_size = list(source._logical_shape)
    if new_outer_size:
        new_outer_size[-1] = new_last_dim
    return _packed_with_shape(
        source,
        new_values,
        new_physical_shape,
        torch.Size(new_outer_size),
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
        preserve_ragged_offsets=dim_adj not in source._ragged_dims,
    )


def _packed_with_static_tail_from_values(source: NestedTensor, new_values: Tensor) -> NestedTensor:
    r"""Rebuild after an elementwise broadcast that only resized packed static dimensions."""
    static_tail = tuple(int(size) for size in new_values.shape[1:])
    static_dims = source._static_dims
    if len(static_tail) != len(static_dims):
        raise RuntimeError(
            "NestedTensor packed broadcast changed the number of static dimensions: "
            f"expected {len(static_dims)}, got {len(static_tail)}"
        )
    replacements = {int(dim): size for dim, size in zip(static_dims, static_tail)}
    new_physical_shape, _, element_shapes = source._shape_meta_from_components(replace_dims=replacements)
    result = _packed_with_shape(
        source,
        new_values,
        new_physical_shape,
        source._logical_shape_from_components(replace_dims=replacements),
        permutation=source._permutation,
        packed_sizes=source._packed_sizes,
        element_shapes=element_shapes,
        preserve_ragged_offsets=True,
    )
    if source._cached_hierarchical_offsets is not None:
        result._cached_hierarchical_offsets = source._cached_hierarchical_offsets
    return result


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
    preserve_ragged_offsets: bool = False,
    force_explicit_ragged_dims: tuple[int, ...] | None = None,
) -> NestedTensor:
    r"""Rebuild a NestedTensor with explicit ``_physical_shape`` and logical shape."""
    source_offsets = offsets is None or offsets is source._offsets
    if offsets is None:
        offsets = source._offsets
    if new_logical_shape is None:
        new_logical_shape = type(source)._logical_shape_from_physical_shape(
            new_physical_shape, offsets, source.batch_first
        )
    output_permutation = permutation
    if output_permutation is None and int(new_physical_shape.size(1)) == len(source._permutation):
        output_permutation = source._permutation
    declared_ragged_dims = force_explicit_ragged_dims
    if declared_ragged_dims is None and source._ragged_dims_explicit and output_permutation is not None:
        output_ragged_rank = int(new_physical_shape.size(1)) - max(new_values.dim() - 1, 0)
        declared_ragged_dims = tuple(int(dim) for dim in output_permutation[:output_ragged_rank])
    ragged_offsets = None
    if (
        preserve_ragged_offsets
        and source_offsets
        and declared_ragged_dims is not None
        and type(source)._is_tensor_backed_layout(output_permutation, declared_ragged_dims)
    ):
        ragged_offsets = source._persistent_ragged_offsets()
        if ragged_offsets is None and len(declared_ragged_dims) == 1 and source_offsets:
            # A single packed ragged level is completely described by the sample
            # offsets.  View-like operations may therefore promote an inferred
            # layout to the tensor-backed compile contract without consulting
            # per-element Python shape caches.
            ragged_offsets = (offsets,)
    result = type(source)._from_packed(
        new_values,
        offsets,
        new_physical_shape,
        permutation=permutation,
        ragged_dims=declared_ragged_dims,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=torch.Size(new_logical_shape),
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
        ragged_offsets=ragged_offsets,
        validate=False,
    )
    return result._packed_like_unchecked(new_values)


def _packed_with_tail_from_values(source: NestedTensor, new_values: Tensor) -> NestedTensor:
    r"""
    Rebuild a NestedTensor by preserving per-element dim-0 lengths and using ``new_values`` static tail dims.

    This is used by packed fast paths whose outputs keep the ragged leading element dim
    but may change trailing per-element dimensions.
    """
    tail = tuple(int(x) for x in new_values.shape[1:])
    if source._physical_shape.size(1) == 0:
        if not tail:
            return source._packed_like_unchecked(new_values)
        batch_size = len(source)
        packed_width = tail[0]
        packed_values = new_values.reshape(batch_size * packed_width, *tail[1:])
        scalar_packed_sizes = tuple(packed_width for _ in range(batch_size))
        scalar_element_shapes = tuple(tail for _ in range(batch_size))
        out_shape = source._physical_shape.new_tensor(tail).reshape(1, -1).expand(batch_size, -1).clone()
        return _packed_with_shape(
            source,
            packed_values,
            out_shape,
            source._logical_shape_from_physical_dims(tail),
            offsets=_offsets_from_packed_sizes(source, scalar_packed_sizes),
            permutation=tuple(range(len(tail))),
            packed_sizes=scalar_packed_sizes,
            element_shapes=scalar_element_shapes,
        )

    static_dims = source._static_dims
    if len(tail) == len(static_dims):
        replacements = {int(dim): size for dim, size in zip(static_dims, tail)}
        out_shape, packed_sizes, element_shapes = source._shape_meta_from_components(replace_dims=replacements)
        return _packed_with_shape(
            source,
            new_values,
            out_shape,
            source._logical_shape_from_components(replace_dims=replacements),
            permutation=source._permutation,
            packed_sizes=packed_sizes,
            element_shapes=element_shapes,
            preserve_ragged_offsets=True,
        )

    leading_ragged = tuple(range(source._ragged_rank))
    if source._ragged_dims != leading_ragged or source._static_dims != tuple(
        range(source._ragged_rank, source._physical_shape.size(1))
    ):
        raise NotImplementedError(
            "Packed tail rank changes require leading ragged dimensions and a trailing static suffix."
        )

    out_shape, packed_sizes, element_shapes = source._shape_meta_from_components(
        keep_dims=source._ragged_dims,
        suffix=tail,
    )
    outer_size = source._logical_shape_from_components(keep_dims=source._ragged_dims, suffix=tail)
    return _packed_with_shape(
        source,
        new_values,
        out_shape,
        outer_size,
        permutation=tuple(range(source._ragged_rank + len(tail))),
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
        preserve_ragged_offsets=True,
    )


def _offsets_from_packed_sizes(source: NestedTensor, sizes: tuple[int, ...]) -> Tensor:
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + int(size))
    return source._offsets.new_tensor(offsets)


# ---------------------------------------------------------------------------
# Packed slicing.
#
# ``narrow``, ``split``, ``chunk``, ``tensor_split`` and ``unbind`` all reduce to "keep this
# span of this axis". Which packed axis that is depends entirely on the layout: the batch dim
# is a run of packed rows, a static per-element dim is an axis of its own, and the innermost
# ragged dim is a span *inside* every segment of packed dim 0.
# ---------------------------------------------------------------------------


def _resolved_packed_sizes(source: NestedTensor, op_name: str) -> tuple[int, ...]:
    r"""Return the packed row count of every sample, refusing a layout that cannot report it."""
    packed_sizes = source._packed_sizes
    if packed_sizes is not None:
        return tuple(int(size) for size in packed_sizes)
    if _is_fake_tensor(source._offsets) or _is_compiling():
        _compile_unsupported(op_name, "the batch dimension needs concrete packed sizes")
    return tuple(int(size) for size in (source._offsets[1:] - source._offsets[:-1]).tolist())


def _outer_size(
    source: NestedTensor,
    physical_shape: Tensor,
    offsets: Tensor,
    element_shapes: tuple[tuple[int, ...], ...] | None,
) -> torch.Size:
    r"""
    Report the logical shape of a rebuilt NestedTensor, from Python metadata where there is any.

    ``_logical_shape_from_physical_shape`` reads the per-dimension maximum off the shape tensor,
    which is a value read: under tracing it makes the *shape* of the result depend on data and
    the graph refuses. The same maximum is already in ``element_shapes`` whenever the layout
    carries them, and there it is plain Python.
    """
    if element_shapes is None:
        return type(source)._logical_shape_from_physical_shape(physical_shape, offsets, source.batch_first)
    batch_size = len(element_shapes)
    if batch_size == 0:
        return torch.Size((0,))
    rank = builtins.max(len(shape) for shape in element_shapes)
    dims = [
        builtins.max((shape[index] if index < len(shape) else 0) for shape in element_shapes) for index in range(rank)
    ]
    dims.insert(0 if source.batch_first else 1, batch_size)
    return torch.Size(dims)


def _packed_batch_slice(source: NestedTensor, start: int, length: int, op_name: str) -> NestedTensor:
    r"""
    Keep ``length`` samples starting at ``start``, addressing packed rows rather than elements.

    Every sample owns a contiguous run of packed dim 0, so a run of samples is a run of rows.
    The offsets have to be rebased on the first kept sample; everything else is a slice of the
    per-sample metadata.
    """
    if length <= 0:
        return source._empty_batch_like()
    sizes = _resolved_packed_sizes(source, op_name)
    row_start = sum(sizes[:start])
    row_count = sum(sizes[start : start + length])  # noqa: E203
    offsets = source._offsets[start : start + length + 1] - source._offsets[start]  # noqa: E203
    physical_shape = source._physical_shape[start : start + length]  # noqa: E203
    element_shapes = source._element_shapes
    sliced_shapes = None if element_shapes is None else element_shapes[start : start + length]  # noqa: E203
    return type(source)._from_packed(
        source._values.narrow(0, row_start, row_count),
        offsets,
        physical_shape,
        permutation=source._permutation,
        ragged_dims=source._ragged_dims if source._ragged_dims_explicit else None,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=_outer_size(source, physical_shape, offsets, sliced_shapes),
        packed_sizes=tuple(sizes[start : start + length]),  # noqa: E203
        element_shapes=sliced_shapes,
        validate=False,
    )


def _packed_static_slice(source: NestedTensor, dim_adj: int, start: int, length: int) -> NestedTensor:
    r"""Keep ``length`` positions of a static per-element dim, on the packed axis that carries it."""
    values_dim = _packed_static_dim(source, dim_adj)
    if values_dim is None:
        raise RuntimeError(f"NestedTensor dimension {dim_adj} is ragged and has no packed axis of its own")
    replacement = {dim_adj: length}
    shape, packed_sizes, element_shapes = source._shape_meta_from_components(replace_dims=replacement)
    return _packed_with_shape(
        source,
        source._values.narrow(values_dim, start, length),
        shape,
        source._logical_shape_from_components(replace_dims=replacement),
        permutation=source._permutation,
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
        preserve_ragged_offsets=True,
    )


def _packed_ragged_slice(
    source: NestedTensor,
    dim_adj: int,
    starts: Sequence[int],
    lengths: Sequence[int],
) -> NestedTensor:
    r"""
    Keep a per-sample span of the sole ragged dim, gathering the rows it names.

    The span may differ from sample to sample -- that is what ``chunk`` and ``tensor_split``
    produce on a ragged axis -- so the rows cannot be a single ``narrow``. They are still
    contiguous within each segment, so the gather index is the segment's start plus the
    sample's own offset plus a running position.
    """
    device = source._values.device
    offsets = source._offsets.to(device=device, dtype=torch.long)
    starts = tuple(int(start) for start in starts)
    lengths = tuple(int(length) for length in lengths)
    total = sum(lengths)
    new_offsets = _offsets_from_packed_sizes(source, lengths)
    if total == 0:
        rows = torch.empty((0,), device=device, dtype=torch.long)
    else:
        lengths_dev = torch.as_tensor(lengths, device=device, dtype=torch.long)
        starts_dev = torch.as_tensor(starts, device=device, dtype=torch.long)
        batch_index = torch.repeat_interleave(
            torch.arange(len(lengths), device=device, dtype=torch.long),
            lengths_dev,
            output_size=total,
        )
        local = torch.arange(total, device=device, dtype=torch.long) - new_offsets.to(device=device)[batch_index]
        rows = offsets[:-1][batch_index] + starts_dev[batch_index] + local
    physical_shape = source._physical_shape.clone()
    physical_shape[:, dim_adj] = physical_shape.new_tensor(lengths)
    element_shapes = source._element_shapes
    if element_shapes is not None:
        element_shapes = tuple(
            (*shape[:dim_adj], length, *shape[dim_adj + 1 :])  # noqa: E203
            for shape, length in zip(element_shapes, lengths)
        )
    return type(source)._from_packed(
        source._values.index_select(0, rows),
        new_offsets,
        physical_shape,
        permutation=source._permutation,
        ragged_dims=source._ragged_dims if source._ragged_dims_explicit else None,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=_outer_size(source, physical_shape, new_offsets, element_shapes),
        packed_sizes=lengths,
        element_shapes=element_shapes,
        validate=False,
    )


def _packed_uniform_ragged_slice(source: NestedTensor, dim_adj: int, start: int, length: int) -> NestedTensor:
    r"""
    Keep the same span of the sole ragged dim in every sample, reading no per-sample extent.

    A layout whose ``__tensor_flatten__`` context drops the Python shape metadata has no
    per-sample lengths to slice with, so :func:`_packed_ragged_slice` cannot serve it under
    tracing. It does not need to: a span that is the same everywhere leaves every sample with
    exactly ``length`` rows, which makes both the offsets and the new ragged extent constants.
    """
    device = source._values.device
    offsets = source._offsets.to(device=device, dtype=torch.long)
    rows = (offsets[:-1] + start).unsqueeze(1) + torch.arange(length, device=device, dtype=torch.long)
    physical_shape = source._physical_shape.clone()
    physical_shape[:, dim_adj] = length
    batch_size = source._offsets.shape[0] - 1
    new_offsets = torch.arange(batch_size + 1, dtype=source._offsets.dtype, device=source._offsets.device) * length
    return type(source)._from_packed(
        source._values.index_select(0, rows.reshape(-1)),
        new_offsets,
        physical_shape,
        permutation=source._permutation,
        ragged_dims=source._ragged_dims if source._ragged_dims_explicit else None,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=source._logical_shape_from_components(replace_dims={dim_adj: length}),
        packed_sizes=None,
        element_shapes=None,
        validate=False,
    )


def _packed_sole_ragged_dim(source: NestedTensor, dim_adj: int) -> bool:
    r"""Whether ``dim_adj`` is the layout's only ragged dim and leads the packed order.

    A slice along it renumbers packed dim 0 and nothing else, so the sample's new row count is
    the span's length. With a second ragged level the row count is a product of two extents and
    the span no longer describes it.
    """
    return source._ragged_rank == 1 and _packed_inner_ragged_dim(source, dim_adj)


def _packed_without_dim(source: NestedTensor, dim_adj: int, values: Tensor) -> NestedTensor:
    r"""Rebuild after a static per-element dim has been consumed, renumbering the layout."""
    keep_dims = tuple(dim for dim in range(int(source._physical_shape.size(1))) if dim != dim_adj)
    shape, packed_sizes, element_shapes = source._shape_meta_from_components(keep_dims=keep_dims)
    return _packed_with_shape(
        source,
        values,
        shape,
        source._logical_shape_from_components(keep_dims=keep_dims),
        permutation=tuple(dim - 1 if dim > dim_adj else dim for dim in source._permutation if dim != dim_adj),
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
        preserve_ragged_offsets=True,
    )


def _same_batch_meta(lhs: NestedTensor, rhs: NestedTensor) -> bool:
    if len(lhs) != len(rhs) or lhs.batch_first != rhs.batch_first:
        return False
    if _is_fake_tensor(lhs._offsets) or _is_fake_tensor(rhs._offsets):
        return lhs._packed_sizes is not None and lhs._packed_sizes == rhs._packed_sizes
    return _offsets_match_identity_if_fake(lhs._offsets, rhs._offsets)


def _packed_pair_indices_from_sizes(
    source: NestedTensor,
    sizes: tuple[int, ...],
) -> tuple[Tensor, Tensor, tuple[int, ...]]:
    pair_sizes = tuple(int(size) * int(size) for size in sizes)
    total = sum(pair_sizes)
    device = source._values.device
    sizes_tensor = torch.tensor(sizes, device=device, dtype=torch.long)
    pair_sizes_tensor = torch.tensor(pair_sizes, device=device, dtype=torch.long)
    seq = torch.repeat_interleave(
        torch.arange(len(sizes), device=device, dtype=torch.long),
        pair_sizes_tensor,
        output_size=total,
    )
    if total == 0:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return empty, empty, pair_sizes
    pair_starts = torch.cumsum(pair_sizes_tensor, 0) - pair_sizes_tensor
    local = torch.arange(total, device=device, dtype=torch.long) - pair_starts.index_select(0, seq)
    offsets = _offsets_from_packed_sizes(source, sizes).to(device=device, dtype=torch.long)
    lengths = sizes_tensor.index_select(0, seq)
    starts = offsets[:-1].index_select(0, seq)
    query = starts + torch.div(local, lengths, rounding_mode="floor")
    key = starts + torch.remainder(local, lengths)
    return query, key, pair_sizes


def _packed_pair_indices(source: NestedTensor) -> tuple[Tensor, Tensor, tuple[int, ...]] | None:
    if source._packed_sizes is None:
        return None
    return _packed_pair_indices_from_sizes(source, tuple(int(size) for size in source._packed_sizes))


def _jagged_outer_matmul_meta(lhs: NestedTensor, rhs: NestedTensor) -> tuple[int, tuple[int, ...]] | None:
    rank = int(lhs._physical_shape.size(1))
    if rank < 2 or int(rhs._physical_shape.size(1)) != rank:
        return None
    prefix_rank = rank - 2
    if lhs._varying_dims != (prefix_rank,) or rhs._varying_dims != (rank - 1,):
        return None
    if lhs._element_shapes is None or rhs._element_shapes is None or lhs._packed_sizes is None:
        return None
    if rhs._packed_sizes != lhs._packed_sizes or lhs._values.dim() != rank or rhs._values.dim() != rank:
        return None
    if tuple(lhs._values.shape[1:]) != tuple(rhs._values.shape[1:]) or not _same_batch_meta(lhs, rhs):
        return None
    for lhs_shape, rhs_shape in zip(lhs._element_shapes, rhs._element_shapes):
        if len(lhs_shape) != rank or len(rhs_shape) != rank:
            return None
        if lhs_shape[:prefix_rank] != rhs_shape[:prefix_rank]:
            return None
        if lhs_shape[prefix_rank] != rhs_shape[-1] or lhs_shape[-1] != rhs_shape[prefix_rank]:
            return None
    return prefix_rank, tuple(int(size) for size in lhs._packed_sizes)


def _jagged_contract_matmul_meta(lhs: NestedTensor, rhs: NestedTensor) -> tuple[int, tuple[int, ...]] | None:
    rank = int(rhs._physical_shape.size(1))
    if rank < 2 or int(lhs._physical_shape.size(1)) != rank:
        return None
    prefix_rank = rank - 2
    if len(lhs) != len(rhs) or lhs.batch_first != rhs.batch_first:
        return None
    if lhs._varying_dims != (prefix_rank, prefix_rank + 1) or rhs._varying_dims != (prefix_rank,):
        return None
    if lhs._element_shapes is None or rhs._element_shapes is None or rhs._packed_sizes is None:
        return None
    if lhs._values.dim() != rank - 1 or rhs._values.dim() != rank:
        return None
    if tuple(lhs._values.shape[1:]) != tuple(rhs._values.shape[1:-1]):
        return None
    pair_sizes = tuple(int(size) * int(size) for size in rhs._packed_sizes)
    if lhs._packed_sizes != pair_sizes:
        return None
    for lhs_shape, rhs_shape in zip(lhs._element_shapes, rhs._element_shapes):
        if len(lhs_shape) != rank or len(rhs_shape) != rank:
            return None
        if lhs_shape[:prefix_rank] != rhs_shape[:prefix_rank]:
            return None
        if lhs_shape[prefix_rank] != rhs_shape[prefix_rank] or lhs_shape[prefix_rank + 1] != rhs_shape[prefix_rank]:
            return None
    if not _is_fake_tensor(lhs._offsets) and not torch.equal(lhs._offsets, _offsets_from_packed_sizes(lhs, pair_sizes)):
        return None
    return prefix_rank, tuple(int(size) for size in rhs._packed_sizes)


def _packed_jagged_matmul_kind(lhs, rhs) -> str | None:
    from .nested_tensor import NestedTensor

    if not isinstance(lhs, NestedTensor) or not isinstance(rhs, NestedTensor):
        return None
    if _jagged_outer_matmul_meta(lhs, rhs) is not None:
        return "outer"
    if _jagged_contract_matmul_meta(lhs, rhs) is not None:
        return "contract"
    return None


def _packed_jagged_outer_matmul(lhs: NestedTensor, rhs: NestedTensor) -> NestedTensor | None:
    meta = _jagged_outer_matmul_meta(lhs, rhs)
    if meta is None:
        return None
    prefix_rank, _ = meta
    pair_indices = _packed_pair_indices(lhs)
    if pair_indices is None:
        return None
    query, key, pair_sizes = pair_indices
    values = (lhs._values.index_select(0, query) * rhs._values.index_select(0, key)).sum(-1)
    keep_dims = (*range(prefix_rank), prefix_rank, prefix_rank)
    shape, _, element_shapes = lhs._shape_meta_from_components(keep_dims=keep_dims)
    return _packed_with_shape(
        lhs,
        values,
        shape,
        lhs._logical_shape_from_components(keep_dims=keep_dims),
        offsets=_offsets_from_packed_sizes(lhs, pair_sizes),
        permutation=(prefix_rank, prefix_rank + 1, *range(prefix_rank)),
        packed_sizes=pair_sizes,
        element_shapes=element_shapes,
    )


def _packed_jagged_contract_matmul(lhs: NestedTensor, rhs: NestedTensor) -> NestedTensor | None:
    meta = _jagged_contract_matmul_meta(lhs, rhs)
    if meta is None:
        return None
    pair_indices = _packed_pair_indices(rhs)
    if pair_indices is None:
        return None
    query, key, _ = pair_indices
    values = torch.zeros_like(rhs._values).index_add(
        0, query, lhs._values.unsqueeze(-1) * rhs._values.index_select(0, key)
    )
    return rhs._packed_like_unchecked(values)


def _packed_jagged_matmul(lhs: NestedTensor, rhs: NestedTensor) -> NestedTensor | None:
    output = _packed_jagged_outer_matmul(lhs, rhs)
    if output is not None:
        return output
    return _packed_jagged_contract_matmul(lhs, rhs)


def _packed_square_softmax(
    source: NestedTensor,
    dim_adj: int,
    *,
    log: bool,
    half_to_float: bool = False,
) -> NestedTensor | None:
    rank = int(source._physical_shape.size(1))
    if rank < 2:
        return None
    prefix_rank = rank - 2
    if dim_adj != prefix_rank + 1 or source._varying_dims != (prefix_rank, prefix_rank + 1):
        return None
    if source._element_shapes is None or source._packed_sizes is None or source._values.dim() != rank - 1:
        return None
    sizes = []
    for shape in source._element_shapes:
        if len(shape) != rank or shape[prefix_rank] != shape[prefix_rank + 1]:
            return None
        sizes.append(int(shape[prefix_rank]))
    sizes_tuple = tuple(sizes)
    if tuple(size * size for size in sizes_tuple) != source._packed_sizes:
        return None
    query, _, _ = _packed_pair_indices_from_sizes(source, sizes_tuple)
    total = sum(sizes_tuple)
    source_values = source._values
    values = (
        source_values.float()
        if half_to_float or source_values.dtype in (torch.float16, torch.bfloat16)
        else source_values
    )
    tail = tuple(values.shape[1:])
    segment = query.reshape((-1, *([1] * len(tail)))).expand((-1, *tail))
    max_values = values.new_full((total, *tail), float("-inf"))
    max_values = max_values.scatter_reduce(0, segment, values, "amax", include_self=False)
    shifted = values - max_values.index_select(0, query)
    exp_values = torch.exp(shifted)
    sums = values.new_zeros((total, *tail)).index_add(0, query, exp_values)
    out_values = shifted - torch.log(sums.index_select(0, query)) if log else exp_values / sums.index_select(0, query)
    if not half_to_float and out_values.dtype != source_values.dtype:
        out_values = out_values.to(dtype=source_values.dtype)
    return source._packed_like_unchecked(out_values)


def _matmul_has_packed_path(lhs, rhs) -> bool:
    from .nested_tensor import NestedTensor

    if isinstance(lhs, NestedTensor):
        if isinstance(rhs, NestedTensor):
            return (
                (lhs._has_same_structure(rhs) and lhs._values.dim() > 2 and rhs._values.dim() > 2)
                or (_packed_jagged_matmul_kind(lhs, rhs) is not None)
                or (
                    lhs._values.dim() == rhs._values.dim() == 4
                    and _complementary_singleton_square_operands(lhs, rhs) is not None
                )
            )
        return isinstance(rhs, Tensor) and lhs._values.dim() >= 2 and rhs.dim() <= 2
    if isinstance(rhs, NestedTensor):
        if isinstance(lhs, Tensor) and lhs.dim() <= 2 and rhs._values.dim() > 2:
            return True
        if not isinstance(lhs, Tensor) or rhs._values.dim() != 2:
            return False
        if lhs.dim() == 2:
            return 0 not in rhs._varying_dims or (rhs._packed_sizes is not None and len(set(rhs._packed_sizes)) == 1)
        return (
            lhs.dim() > 2
            and rhs._packed_sizes is not None
            and len(set(rhs._packed_sizes)) == 1
            and rhs._physical_shape.size(1) == 2
            and rhs._element_shapes is not None
            and len({int(shape[1]) for shape in rhs._element_shapes}) == 1
        )
    return False


def _from_uniform_batched_output(source: NestedTensor, batched_values: Tensor) -> NestedTensor:
    r"""Wrap a batch-major tensor ``[B, *shape]`` as a NestedTensor with uniform per-element shape."""
    batch_size = len(source)
    elem_shape: tuple[int, ...] = tuple(int(x) for x in batched_values.shape[1:])
    permutation: tuple[int, ...]
    if not elem_shape:
        out_values = batched_values.reshape(batch_size)
        out_offsets = torch.arange(batch_size + 1, dtype=source._offsets.dtype, device=source._offsets.device)
        out_shape = source._physical_shape.new_empty((batch_size, 0))
        packed_sizes = tuple(1 for _ in range(batch_size))
        element_shapes = cast(tuple[tuple[int, ...], ...], tuple(() for _ in range(batch_size)))
        permutation = ()
    else:
        lengths = source._offsets.new_full((batch_size,), elem_shape[0])
        out_offsets = torch.empty((batch_size + 1,), dtype=source._offsets.dtype, device=source._offsets.device)
        out_offsets[0] = 0
        if lengths.numel() > 0:
            out_offsets[1:] = torch.cumsum(lengths, dim=0)
        out_shape = source._physical_shape.new_tensor(elem_shape).reshape(1, -1).expand(batch_size, -1).clone()
        out_values = batched_values.reshape(batch_size * elem_shape[0], *elem_shape[1:])
        packed_sizes = tuple(int(elem_shape[0]) for _ in range(batch_size))
        element_shapes = tuple(elem_shape for _ in range(batch_size))
        permutation = source._permutation_after_replacing_trailing_dims(
            max(source._physical_shape.size(1) - 1, 0), len(elem_shape[1:])
        )
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_physical_dims(elem_shape),
        offsets=out_offsets,
        permutation=permutation,
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
    )


def _reduce_non_ragged_packed(source: NestedTensor, out_values: Tensor, dim_adj: int, keepdim: bool):
    r"""Wrap non-ragged dim reductions on packed values as a NestedTensor."""
    if keepdim:
        out_shape, packed_sizes, element_shapes = source._shape_meta_from_components(replace_dims={dim_adj: 1})
    else:
        keep_dims = tuple(i for i in range(source._physical_shape.size(1)) if i != dim_adj)
        out_shape, packed_sizes, element_shapes = source._shape_meta_from_components(keep_dims=keep_dims)
    permutation = source._permutation if keepdim else source._project_permutation(keep_dims=keep_dims)
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        (
            source._logical_shape_from_components(replace_dims={dim_adj: 1})
            if keepdim
            else source._logical_shape_from_components(keep_dims=keep_dims)
        ),
        permutation=permutation,
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
        preserve_ragged_offsets=True,
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
            permutation=source._permutation,
            packed_sizes=packed_sizes,
            element_shapes=element_shapes,
            preserve_ragged_offsets=True,
        )

    keep_dims = tuple(i for i in range(source._physical_shape.size(1)) if i not in set(dims_adj))
    out_shape, packed_sizes, element_shapes = source._shape_meta_from_components(keep_dims=keep_dims)
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_components(keep_dims=keep_dims),
        permutation=source._project_permutation(keep_dims=keep_dims),
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
        preserve_ragged_offsets=True,
    )


def _has_single_packed_ragged_dim(source: NestedTensor, dim_adj: int) -> bool:
    rank = int(source._physical_shape.size(1))
    static_dims = tuple(dim for dim in range(rank) if dim != dim_adj)
    return (
        source._varying_dims == (dim_adj,)
        and source._static_dims == static_dims
        and tuple(source._permutation or tuple(range(rank))) == (dim_adj, *static_dims)
    )


def _has_segment_reducible_ragged_dim(source: NestedTensor, dim_adj: int) -> bool:
    r"""Whether one ragged dimension leads packed values for a segmented reduction.

    Unlike :func:`_has_single_packed_ragged_dim`, this permits the remaining
    static dimensions to use any valid packed order.  Segmented reductions run
    along packed axis 0 and restore that static tail to logical physical order
    before returning a dense batch-major result.
    """
    permutation = tuple(source._permutation or tuple(range(int(source._physical_shape.size(1)))))
    return source._varying_dims == (dim_adj,) and permutation[:1] == (dim_adj,)


def _restore_segment_batch_dim(
    source: NestedTensor,
    output: Tensor,
    dim_adj: int,
    keepdim: bool,
) -> Tensor:
    r"""Move a batch-major segment result to the logical batch position."""
    return _restore_multi_dim_batch_dim(source, output, (dim_adj,), keepdim)


def _restore_multi_dim_batch_dim(
    source: NestedTensor,
    output: Tensor,
    dims_adj,
    keepdim: bool,
) -> Tensor:
    r"""Restore the configured batch position after reducing physical dimensions."""
    physical_rank = int(source._physical_shape.size(1))
    output_physical_rank = physical_rank if keepdim else physical_rank - len({int(dim) for dim in dims_adj})
    if not source.batch_first and output_physical_rank > 0:
        return output.movedim(0, 1)
    return output


def _format_segment_reduction(
    source: NestedTensor,
    output: Tensor,
    dim_adj: int,
    keepdim: bool,
) -> Tensor:
    r"""Insert a kept physical dimension and restore the logical batch position."""
    if keepdim:
        output = output.unsqueeze(1 + dim_adj)
    return _restore_segment_batch_dim(source, output, dim_adj, keepdim)


def _format_permuted_segment_reduction(
    source: NestedTensor,
    output: Tensor,
    dim_adj: int,
    keepdim: bool,
) -> Tensor:
    r"""Restore a segmented output whose tail follows packed static order."""
    logical_static = tuple(dim for dim in range(int(source._physical_shape.size(1))) if dim != dim_adj)
    packed_static = source._static_dims
    if packed_static != logical_static:
        output = output.permute(0, *(1 + packed_static.index(dim) for dim in logical_static))
    return _format_segment_reduction(source, output, dim_adj, keepdim)


def _check_nonempty_extrema_segments(source: NestedTensor, func) -> None:
    r"""Match dense extrema errors when any ragged reduction segment is empty."""
    if source._packed_sizes is not None and any(int(size) == 0 for size in source._packed_sizes):
        op_name = "argmax" if func is aten.argmax.default else "argmin"
        if func in (aten.max.dim, aten.min.dim):
            op_name = "max" if func is aten.max.dim else "min"
        raise IndexError(f"{op_name}(): Expected reduction dim to have non-zero size.")


def _segment_sum(source: NestedTensor, values: Tensor, lengths: Tensor) -> Tensor:
    segment_reduce = getattr(torch, "segment_reduce", None)
    if segment_reduce is not None and values.dtype.is_floating_point and not values.dtype.is_complex:
        return segment_reduce(values, reduce="sum", lengths=lengths)
    out = values.new_zeros((len(source), *values.shape[1:]))
    batch_idx = source.packed_batch_indices(device=values.device)
    return out.index_add(0, batch_idx, values)


def _segment_extrema_values(source: NestedTensor, values: Tensor, lengths: Tensor, *, largest: bool) -> Tensor | None:
    if values.dtype.is_complex:
        return None
    segment_reduce = getattr(torch, "segment_reduce", None)
    if segment_reduce is not None and values.dtype.is_floating_point:
        return segment_reduce(values, reduce="max" if largest else "min", lengths=lengths)
    out = values.new_full((len(source), *values.shape[1:]), _topk_fill_value(values.dtype, largest=largest))
    batch_idx = source.packed_batch_indices(device=values.device)
    index = batch_idx.reshape(-1, *([1] * (values.dim() - 1))).expand_as(values)
    return out.scatter_reduce(0, index, values, "amax" if largest else "amin", include_self=False)


def _matches_extrema(values: Tensor, extrema: Tensor) -> Tensor:
    matches = values == extrema
    if values.dtype.is_floating_point:
        matches = matches | (torch.isnan(values) & torch.isnan(extrema))
    return matches


def _segment_extrema_indices(source: NestedTensor, values: Tensor, extrema: Tensor) -> Tensor:
    batch_idx = source.packed_batch_indices(device=values.device)
    local_idx = source.packed_local_indices(device=values.device)
    selected = extrema.index_select(0, batch_idx)
    matches = _matches_extrema(values, selected)
    if values.dim() > 1:
        local_idx = local_idx.reshape(-1, *([1] * (values.dim() - 1))).expand_as(values)
        batch_idx = batch_idx.reshape(-1, *([1] * (values.dim() - 1))).expand_as(values)
    sentinel = values.shape[0]
    candidates = torch.where(matches, local_idx, torch.full_like(local_idx, sentinel))
    indices = torch.full((len(source), *values.shape[1:]), sentinel, device=values.device, dtype=torch.long)
    return indices.scatter_reduce(0, batch_idx, candidates, "amin", include_self=True)


def _segment_arg_extrema_ragged_dim(
    source: NestedTensor,
    dim_adj: int,
    keepdim: bool,
    *,
    largest: bool,
) -> Tensor | None:
    if not _has_single_packed_ragged_dim(source, dim_adj):
        return None
    _check_nonempty_extrema_segments(source, aten.argmax.default if largest else aten.argmin.default)
    values = source._values
    offsets = source.ragged_level_offsets(0, device=values.device, dtype=torch.long)
    extrema = _segment_extrema_values(source, values, offsets[1:] - offsets[:-1], largest=largest)
    if extrema is None:
        return None
    indices = _segment_extrema_indices(source, values, extrema)
    return _format_segment_reduction(source, indices, dim_adj, keepdim)


def _segment_max_min_ragged_dim(
    source: NestedTensor,
    dim_adj: int,
    keepdim: bool,
    *,
    largest: bool,
) -> tuple[Tensor, Tensor] | None:
    if not _has_single_packed_ragged_dim(source, dim_adj):
        return None
    _check_nonempty_extrema_segments(source, aten.max.dim if largest else aten.min.dim)
    values = source._values
    offsets = source.ragged_level_offsets(0, device=values.device, dtype=torch.long)
    extrema = _segment_extrema_values(source, values, offsets[1:] - offsets[:-1], largest=largest)
    if extrema is None:
        return None
    indices = _segment_extrema_indices(source, values, extrema)
    starts = offsets[:-1].reshape(-1, *([1] * (indices.dim() - 1)))
    extrema = torch.gather(values, 0, indices + starts)
    return (
        _format_segment_reduction(source, extrema, dim_adj, keepdim),
        _format_segment_reduction(source, indices, dim_adj, keepdim),
    )


def _segment_count_nonzero_ragged_dim(source: NestedTensor, dim_adj: int) -> Tensor | None:
    if not _has_single_packed_ragged_dim(source, dim_adj):
        return None
    values = source._values.ne(0).to(dtype=torch.long)
    offsets = source.ragged_level_offsets(0, device=values.device, dtype=torch.long)
    output = _segment_sum(source, values, offsets[1:] - offsets[:-1])
    return _restore_segment_batch_dim(source, output, dim_adj, keepdim=False)


def _resolve_vector_norm_ord(ord_value) -> float | None:
    if ord_value is None:
        return 2.0
    if isinstance(ord_value, bool) or not isinstance(ord_value, (int, float)):
        return None
    ord_float = float(ord_value)
    if math.isnan(ord_float) or math.isinf(ord_float) or ord_float < 0.0:
        return None
    return ord_float


def _segment_vector_norm_ragged_dim(
    source: NestedTensor,
    ord_value,
    dim_adj: int | None,
    keepdim: bool,
    *,
    dtype: torch.dtype | None,
) -> NestedTensor | None:
    if dim_adj is not None and not _has_single_packed_ragged_dim(source, dim_adj):
        return None
    if dim_adj is None and not _has_single_packed_ragged_dim(source, 0):
        return None

    ord_float = _resolve_vector_norm_ord(ord_value)
    if ord_float is None or not source._values.dtype.is_floating_point or source._values.dtype.is_complex:
        return None

    values = source._values if dtype is None else source._values.to(dtype=dtype)
    offsets = source.ragged_level_offsets(0, device=values.device, dtype=torch.long)
    lengths = offsets[1:] - offsets[:-1]

    if ord_float == 0.0:
        reduced = values.ne(0).to(dtype=values.dtype)
    else:
        reduced = values.abs().pow(ord_float)

    if dim_adj is None and reduced.dim() > 1:
        reduced = reduced.reshape(reduced.shape[0], -1).sum(dim=1)

    out = _segment_sum(source, reduced, lengths)
    if ord_float != 0.0:
        out = out.pow(1.0 / ord_float)

    if dim_adj is None and keepdim:
        out = out.reshape(len(source), *([1] * int(source._physical_shape.size(1))))
    elif dim_adj is not None and keepdim:
        out = out.unsqueeze(1 + dim_adj)
    return _from_uniform_batched_output(source, out)


def _segment_reduce_ragged_dim(
    func,
    source: NestedTensor,
    dim_adj: int,
    keepdim: bool,
    kwargs,
) -> Tensor | None:
    if not _has_segment_reducible_ragged_dim(source, dim_adj):
        return None

    values = source._values
    batch_size = len(source)
    offsets = source.ragged_level_offsets(0, device=values.device, dtype=torch.long)
    lengths = offsets[1:] - offsets[:-1]
    segment_reduce = getattr(torch, "segment_reduce", None)

    if func is aten.sum.dim_IntList:
        dtype = kwargs.get("dtype")
        segment_values = values if dtype is None else values.to(dtype=dtype)
        if (
            segment_reduce is not None
            and segment_values.dtype.is_floating_point
            and not segment_values.dtype.is_complex
        ):
            out = segment_reduce(segment_values, reduce="sum", lengths=lengths)
            return _format_permuted_segment_reduction(source, out, dim_adj, keepdim)
        sample = func(values[:0], [0], False, **kwargs)
        out = sample.new_zeros((batch_size, *sample.shape))
        batch_idx = source.packed_batch_indices(device=values.device)
        add_values = values if values.dtype == out.dtype else values.to(dtype=out.dtype)
        out = out.index_add(0, batch_idx, add_values)
    elif func is aten.mean.dim:
        dtype = kwargs.get("dtype")
        segment_values = values if dtype is None else values.to(dtype=dtype)
        if (
            segment_reduce is not None
            and segment_values.dtype.is_floating_point
            and not segment_values.dtype.is_complex
        ):
            out = segment_reduce(segment_values, reduce="mean", lengths=lengths)
            return _format_permuted_segment_reduction(source, out, dim_adj, keepdim)
        sample = func(values[:0], [0], False, **kwargs)
        out = sample.new_zeros((batch_size, *sample.shape))
        batch_idx = source.packed_batch_indices(device=values.device)
        add_values = values if values.dtype == out.dtype else values.to(dtype=out.dtype)
        out = out.index_add(0, batch_idx, add_values)
        denominator = lengths.reshape(batch_size, *([1] * (out.dim() - 1)))
        out = out / denominator
    elif func in (aten.amax.default, aten.amin.default):
        if values.dtype.is_complex:
            return None
        largest = func is aten.amax.default
        if segment_reduce is not None and values.dtype.is_floating_point:
            out = segment_reduce(values, reduce="max" if largest else "min", lengths=lengths)
            return _format_permuted_segment_reduction(source, out, dim_adj, keepdim)
        fill_value = _topk_fill_value(values.dtype, largest=largest)
        out = values.new_full((batch_size, *values.shape[1:]), fill_value)
        batch_idx = source.packed_batch_indices(device=values.device)
        index = batch_idx.reshape(-1, *([1] * (values.dim() - 1))).expand_as(values)
        out = out.scatter_reduce(0, index, values, "amax" if largest else "amin", include_self=False)
    else:
        return None

    return _format_permuted_segment_reduction(source, out, dim_adj, keepdim)


def _packed_new_ragged_size(
    source: NestedTensor, new_values: Tensor, dim_adj: int, new_ragged_size: int
) -> NestedTensor:
    r"""Rebuild a NestedTensor when its one ragged dim takes a uniform size.

    Reconstructing from a list of elements instead would drop the packed permutation and, for
    an empty batch, take the dtype from ``_meta()`` — which would hand ``topk`` an index tensor
    in the *value* dtype.
    """
    batch_size = source._offsets.size(0) - 1
    # Keep offsets on the same device as the source metadata (CPU by design).
    new_offsets = torch.arange(batch_size + 1, dtype=torch.long, device=source._offsets.device) * new_ragged_size
    new_physical_shape = source._physical_shape.clone()
    if new_physical_shape.numel() > 0:
        new_physical_shape[:, dim_adj] = new_ragged_size
    return _packed_with_shape(
        source,
        new_values,
        new_physical_shape,
        source._logical_shape_from_components(replace_dims={int(dim_adj): int(new_ragged_size)}),
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


def _is_native_attention_layout(nt: NestedTensor) -> bool:
    r"""Return True when attention elements are stored as second-dim ragged packed values."""
    return (
        nt._physical_shape.size(1) == 3
        and nt._values.dim() == 3
        and nt._varying_dims == (1,)
        and nt._static_dims == (0, 2)
    )


def _sdpa_pack_native(nt: NestedTensor) -> tuple[Tensor, Tensor, int]:
    r"""
    Return the native varlen layout for ``(heads, seq_i, dim)`` elements:
    ``(sum_seq, heads, dim)`` plus cumulative sequence lengths.
    """
    if not _is_native_attention_layout(nt):
        raise ValueError("Native SDPA fast path requires elements shaped like (heads, seq, dim).")

    cumulative = nt.ragged_level_offsets(0, device=nt.device, dtype=torch.int32)
    if nt._element_shapes is not None and all(
        isinstance(shape[1], int) for shape in nt._element_shapes if len(shape) > 1
    ):
        max_seqlen = max((int(shape[1]) for shape in nt._element_shapes), default=0)
    else:
        lengths_cpu = nt._ragged_level_sizes(0)
        max_seqlen = int(lengths_cpu.max().item()) if lengths_cpu.numel() else 0
    return nt._values.contiguous(), cumulative, max_seqlen


def _sdpa_restore_native(attention: Tensor, query: NestedTensor) -> NestedTensor:
    r"""Restore fused-kernel output without unpacking per-element tensors."""
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


def _same_ragged_offsets(lhs: NestedTensor, rhs: NestedTensor) -> bool:
    r"""Return whether two native-attention tensors share the same sequence lengths."""
    if lhs._offsets is rhs._offsets:
        return True
    try:
        return lhs._offsets.data_ptr() == rhs._offsets.data_ptr()
    except RuntimeError:
        return False


def _pad_last_dim_for_flash(tensor: Tensor, alignment_size: int = 8) -> Tensor:
    r"""Pad the last dim for Flash Attention alignment requirements."""
    last_dim = tensor.size(-1)
    if last_dim % alignment_size == 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, alignment_size - (last_dim % alignment_size)))


def _flash_attention_forward_raw(
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
    return_debug_mask: bool,
    scale: float | None,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    seqused_k: Tensor | None = None,
    alibi_slopes: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""Run the underlying varlen FlashAttention op on packed dense values."""
    original_head_dim = query.size(-1)
    q_padded = _pad_last_dim_for_flash(query)
    k_padded = _pad_last_dim_for_flash(key)
    v_padded = _pad_last_dim_for_flash(value)
    softmax_scale = scale if scale is not None else original_head_dim**-0.5
    attention, logsumexp, rng_state, unused, debug_mask = aten._flash_attention_forward.default(
        q_padded,
        k_padded,
        v_padded,
        q_cumulative,
        k_cumulative,
        q_max,
        k_max,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=softmax_scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        seqused_k=seqused_k,
        alibi_slopes=alibi_slopes,
    )
    if attention.size(-1) != original_head_dim:
        attention = attention[..., :original_head_dim]
    return attention, logsumexp, rng_state, unused, debug_mask


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
    alibi_slopes: Tensor | None = None,
) -> Tensor:
    r"""Run varlen FlashAttention directly on packed ``(total_seq, heads, dim)`` values."""
    return _flash_attention_forward_raw(
        query,
        key,
        value,
        q_cumulative,
        k_cumulative,
        q_max,
        k_max,
        dropout_p=dropout_p,
        is_causal=is_causal,
        return_debug_mask=False,
        scale=scale,
        alibi_slopes=alibi_slopes,
    )[0]


def _sdpa_via_native_flash(
    query: NestedTensor,
    key: NestedTensor,
    value: NestedTensor,
    *,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    alibi_slopes: Tensor | None = None,
) -> NestedTensor:
    r"""Run SDPA directly on DanLing storage via varlen Flash Attention kernels."""
    q_values, q_cumulative, q_max = _sdpa_pack_native(query)
    if _same_ragged_offsets(query, key):
        k_values = key._values.contiguous()
        k_cumulative = q_cumulative
        k_max = q_max
    else:
        k_values, k_cumulative, k_max = _sdpa_pack_native(key)
    attention = _flash_attention_forward_values(
        q_values,
        k_values,
        value._values.contiguous(),
        q_cumulative,
        k_cumulative,
        q_max,
        k_max,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        alibi_slopes=alibi_slopes,
    )
    return _sdpa_restore_native(attention, query)


def _flash_attention_forward_compile_safe_inputs(args: tuple, kwargs: dict[str, object]) -> bool:
    r"""Return whether an ``aten._flash_attention_forward`` call stays on the packed native path."""
    from .nested_tensor import NestedTensor

    query = args[0] if len(args) > 0 else kwargs.get("query")
    key = args[1] if len(args) > 1 else kwargs.get("key")
    value = args[2] if len(args) > 2 else kwargs.get("value")
    if not isinstance(query, NestedTensor):
        return True
    cum_seq_q = args[3] if len(args) > 3 else kwargs.get("cum_seq_q")
    cum_seq_k = args[4] if len(args) > 4 else kwargs.get("cum_seq_k")
    max_q = cast(Any, args[5] if len(args) > 5 else kwargs.get("max_q", 0))
    max_k = cast(Any, args[6] if len(args) > 6 else kwargs.get("max_k", 0))
    if cum_seq_q is not None or cum_seq_k is not None:
        return False
    if int(max_q or 0) != 0 or int(max_k or 0) != 0:
        return False
    return (
        isinstance(key, NestedTensor)
        and isinstance(value, NestedTensor)
        and _is_native_attention_layout(query)
        and _is_native_attention_layout(key)
        and _is_native_attention_layout(value)
    )


@NestedTensorAtenRegistry.implement(
    aten._flash_attention_forward.default,
    compile_safe=True,
    compile_guard=_flash_attention_forward_compile_safe_inputs,
)
def flash_attention_forward(_func, args, kwargs):
    r"""Varlen FlashAttention on packed NestedTensors, with optional ALiBi ``alibi_slopes``."""
    from .nested_tensor import NestedTensor

    query = args[0] if len(args) > 0 else kwargs.get("query")
    key = args[1] if len(args) > 1 else kwargs.get("key")
    value = args[2] if len(args) > 2 else kwargs.get("value")
    cum_seq_q = args[3] if len(args) > 3 else kwargs.get("cum_seq_q")
    cum_seq_k = args[4] if len(args) > 4 else kwargs.get("cum_seq_k")
    max_q = args[5] if len(args) > 5 else kwargs.get("max_q", 0)
    max_k = args[6] if len(args) > 6 else kwargs.get("max_k", 0)
    dropout_p = float(args[7] if len(args) > 7 else kwargs.get("dropout_p", 0.0))
    is_causal = bool(args[8] if len(args) > 8 else kwargs.get("is_causal", False))
    return_debug_mask = bool(args[9] if len(args) > 9 else kwargs.get("return_debug_mask", False))
    scale = kwargs.get("scale")
    window_size_left = kwargs.get("window_size_left")
    window_size_right = kwargs.get("window_size_right")
    seqused_k = kwargs.get("seqused_k")
    alibi_slopes = kwargs.get("alibi_slopes")

    if not (isinstance(query, NestedTensor) and isinstance(key, NestedTensor) and isinstance(value, NestedTensor)):
        raise TypeError("DanLing _flash_attention_forward expects NestedTensor query, key, and value together.")
    if cum_seq_q is not None or cum_seq_k is not None or int(max_q or 0) != 0 or int(max_k or 0) != 0:
        raise ValueError("DanLing _flash_attention_forward derives cum_seq/max values from NestedTensor structure.")
    if not (
        len(query) == len(key) == len(value)
        and query.batch_first == key.batch_first == value.batch_first
        and _is_native_attention_layout(query)
        and _is_native_attention_layout(key)
        and _is_native_attention_layout(value)
    ):
        raise ValueError("DanLing _flash_attention_forward requires matching native attention NestedTensors.")

    q_values, q_cumulative, q_max = _sdpa_pack_native(query)
    if _same_ragged_offsets(query, key):
        k_values = key._values.contiguous()
        k_cumulative = q_cumulative
        k_max = q_max
    else:
        k_values, k_cumulative, k_max = _sdpa_pack_native(key)
    output, logsumexp, rng_state, unused, debug_mask = _flash_attention_forward_raw(
        q_values,
        k_values,
        value._values.contiguous(),
        q_cumulative,
        k_cumulative,
        q_max,
        k_max,
        dropout_p=dropout_p,
        is_causal=is_causal,
        return_debug_mask=return_debug_mask,
        scale=scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        seqused_k=seqused_k,
        alibi_slopes=alibi_slopes,
    )
    return _sdpa_restore_native(output, query), logsumexp, rng_state, unused, debug_mask


def _topk_fill_value(dtype: torch.dtype, largest: bool):
    if dtype.is_floating_point or dtype.is_complex:
        return float("-inf") if largest else float("inf")
    if dtype == torch.bool:
        return not largest
    info = torch.iinfo(dtype)
    return info.min if largest else info.max


@NestedTensorAtenRegistry.implement(aten.addmm.default)
def addmm(func, args, kwargs):
    r"""Dispatch handler for bias + matrix multiply (NT x dense) on packed _values."""
    from .nested_tensor import NestedTensor

    bias, mat1, mat2 = args[0], args[1], args[2]
    if isinstance(mat1, NestedTensor) and not isinstance(mat2, NestedTensor) and mat1._values.dim() == 2:
        new_values = func(bias, mat1._values, mat2, **kwargs)
        return _packed_new_last_dim(mat1, new_values, mat2.shape[1])
    if (
        not isinstance(mat1, NestedTensor)
        and isinstance(mat2, NestedTensor)
        and isinstance(mat1, Tensor)
        and mat1.dim() == 2
        and mat2._values.dim() == 2
    ):
        if 0 not in mat2._varying_dims:
            packed_bias = None
            if isinstance(bias, Tensor):
                if bias.dim() == 0:
                    packed_bias = bias
                elif bias.dim() == 1:
                    if bias.numel() == 1:
                        packed_bias = bias.reshape(1, 1)
                elif bias.dim() == 2 and bias.shape[1] == 1 and bias.shape[0] in (1, mat1.shape[0]):
                    packed_bias = bias.transpose(0, 1)
            if packed_bias is not None:
                new_values = func(packed_bias, mat2._values, mat1.transpose(0, 1), **kwargs)
                return _packed_new_dim_size(mat2, new_values, 0, int(mat1.shape[0]))
        if (
            mat2._packed_sizes is not None
            and len(set(mat2._packed_sizes)) == 1
            and mat2._physical_shape.size(1) == 2
            and mat2._element_shapes is not None
        ):
            cols = {int(shape[1]) for shape in mat2._element_shapes}
            if len(cols) == 1:
                col_dim = next(iter(cols))
                rhs_batched = mat2._values.view(len(mat2), mat2._packed_sizes[0], col_dim)
                bias_batched = (
                    bias.expand(len(mat2), *bias.shape) if isinstance(bias, Tensor) and bias.dim() >= 1 else bias
                )
                out = torch.baddbmm(
                    bias_batched,
                    mat1.unsqueeze(0).expand(len(mat2), *mat1.shape),
                    rhs_batched,
                    alpha=kwargs.get("alpha", 1),
                    beta=kwargs.get("beta", 1),
                )
                return _packed_new_dim_size(mat2, out.reshape(-1, col_dim), 0, int(mat1.shape[0]))
    raise NotImplementedError(f"NestedTensor: {func} requires a supported packed 2-D NestedTensor matrix operand")


@NestedTensorAtenRegistry.implement(aten.baddbmm.default)
def baddbmm(func, args, kwargs):
    r"""Dispatch handler for dense x NestedTensor batched addmm on packed _values."""
    from .nested_tensor import NestedTensor

    bias, batch1, batch2 = args[0], args[1], args[2]

    if (
        isinstance(batch1, Tensor)
        and not isinstance(batch1, NestedTensor)
        and isinstance(batch2, NestedTensor)
        and batch1.dim() == 3
        and batch2._values.dim() == 3
        and (
            (0 not in batch2._varying_dims and 1 not in batch2._varying_dims)
            or (
                batch2._packed_sizes is not None
                and len(set(batch2._packed_sizes)) == 1
                and batch2._physical_shape.size(1) > 1
                and 1 not in batch2._varying_dims
            )
        )
    ):
        # Transpose the bias to match the transposed packed computation
        packed_bias = bias
        if isinstance(bias, Tensor) and bias.dim() == 3:
            if bias.shape[1] == batch1.shape[1] and bias.shape[2] == 1 and bias.shape[0] in (1, batch1.shape[0]):
                packed_bias = bias.permute(0, 2, 1)
            else:
                raise NotImplementedError(
                    f"NestedTensor: {func} requires scalar bias or bias broadcastable with singleton ragged dim"
                )
        new_values = func(packed_bias, batch2._values.permute(1, 0, 2), batch1.transpose(1, 2), **kwargs).permute(
            1, 0, 2
        )
        return _packed_new_dim_size(batch2, new_values, 1, int(batch1.shape[1]))

    raise NotImplementedError(f"NestedTensor: {func} requires a supported packed dense x NT batched matmul")


# See also torch_functions.py::bmm for the torch-level handler (mismatched offsets).
@NestedTensorAtenRegistry.implement(aten.bmm.default)
def bmm(func, args, kwargs):
    r"""Dispatch handler for batched matrix multiply between two NestedTensors."""
    from .nested_tensor import NestedTensor

    mat1, mat2 = args[0], args[1]
    if isinstance(mat1, NestedTensor) and isinstance(mat2, NestedTensor) and mat1._has_same_structure(mat2):
        new_values = func(mat1._values, mat2._values, **kwargs)
        return _packed_new_last_dim(mat1, new_values, mat2._values.shape[-1])
    if (
        isinstance(mat1, Tensor)
        and not isinstance(mat1, NestedTensor)
        and isinstance(mat2, NestedTensor)
        and mat1.dim() == 3
        and mat2._values.dim() == 3
        and (
            (0 not in mat2._varying_dims and 1 not in mat2._varying_dims)
            or (
                mat2._packed_sizes is not None
                and len(set(mat2._packed_sizes)) == 1
                and mat2._physical_shape.size(1) > 1
                and 1 not in mat2._varying_dims
            )
        )
    ):
        new_values = torch.bmm(mat2._values.permute(1, 0, 2), mat1.transpose(1, 2)).permute(1, 0, 2)
        return _packed_new_dim_size(mat2, new_values, 1, int(mat1.shape[1]))
    raise NotImplementedError(f"NestedTensor: {func} requires two NTs with matching packed structure")


@NestedTensorAtenRegistry.implement(aten.diagonal.default)
def diagonal(func, args, kwargs):
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


@NestedTensorAtenRegistry.implement(aten._linalg_check_errors.default)
def linalg_check_errors(func, args, kwargs):
    r"""Dispatch handler for ``aten._linalg_check_errors`` that preserves its ``None`` return contract."""
    from .nested_tensor import NestedTensor

    info = args[0]
    if isinstance(info, NestedTensor):
        func(info._values, *args[1:], **kwargs)
        return None
    return func(*args, **kwargs)


@NestedTensorAtenRegistry.implement(aten.linalg_eigh.default)
@NestedTensorAtenRegistry.implement(aten._linalg_eigh.default)
def linalg_eigh(func, args, kwargs):
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
        source._packed_like_unchecked(eigvecs_values),
    )


@NestedTensorAtenRegistry.implement(aten.linalg_qr.default)
def linalg_qr(func, args, kwargs):
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


@NestedTensorAtenRegistry.implement(aten.linalg_solve.default)
def linalg_solve(func, args, kwargs):
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
            if mat_a._has_same_structure(mat_b) and mat_a._values.dim() > 2 and mat_b._values.dim() > 1:
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
def linalg_solve_ex(func, args, kwargs):
    r"""Dispatch handler for ``aten._linalg_solve_ex`` to support ``linalg_solve`` decomposition paths."""
    from .nested_tensor import NestedTensor

    mat_a, mat_b = args[0], args[1]
    if isinstance(mat_a, NestedTensor):
        if isinstance(mat_b, NestedTensor):
            if len(mat_a) != len(mat_b):
                raise ValueError(
                    "linalg.solve: NestedTensor batch length mismatch between input and B: "
                    f"input={len(mat_a)}, B={len(mat_b)}"
                )
            if mat_a._has_same_structure(mat_b) and mat_a._values.dim() > 2 and mat_b._values.dim() > 1:
                result, lu, pivots, info = func(mat_a._values, mat_b._values, *args[2:], **kwargs)
                return (
                    _packed_with_tail_from_values(mat_a, result),
                    mat_a._packed_like_unchecked(lu),
                    _packed_with_tail_from_values(mat_a, pivots),
                    _packed_with_tail_from_values(mat_a, info),
                )
            return per_element_fallback(func, args, kwargs)
        if isinstance(mat_b, Tensor) and mat_a._values.dim() > 2:
            result, lu, pivots, info = func(mat_a._values, mat_b, *args[2:], **kwargs)
            return (
                _packed_with_tail_from_values(mat_a, result),
                mat_a._packed_like_unchecked(lu),
                _packed_with_tail_from_values(mat_a, pivots),
                _packed_with_tail_from_values(mat_a, info),
            )
        return per_element_fallback(func, args, kwargs)

    if isinstance(mat_b, NestedTensor):
        return per_element_fallback(func, args, kwargs)

    return func(*args, **kwargs)


@NestedTensorAtenRegistry.implement(aten.linalg_svd.default)
@NestedTensorAtenRegistry.implement(aten._linalg_svd.default)
def linalg_svd(func, args, kwargs):
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


@NestedTensorAtenRegistry.implement(aten.matmul.default)
def matmul(func, args, kwargs):
    r"""Dispatch handler for matmul with packed fast paths when ragged dim remains element-local."""
    from .nested_tensor import NestedTensor

    lhs, rhs = args[0], args[1]

    if isinstance(lhs, NestedTensor):
        if isinstance(rhs, NestedTensor):
            jagged = _packed_jagged_matmul(lhs, rhs)
            if jagged is not None:
                return jagged
            if lhs._values.dim() == rhs._values.dim() == 4:
                square = _binary_complementary_singleton_square(lhs, rhs, torch.bmm, (), kwargs)
                if square is not None:
                    return square
            if lhs._has_same_structure(rhs) and lhs._values.dim() > 2 and rhs._values.dim() > 2:
                return _packed_with_tail_from_values(lhs, func(lhs._values, rhs._values, **kwargs))
            return per_element_fallback(func, args, kwargs)
        if isinstance(rhs, Tensor) and lhs._values.dim() >= 2 and rhs.dim() <= 2:
            return _packed_with_tail_from_values(lhs, func(lhs._values, rhs, **kwargs))
        return per_element_fallback(func, args, kwargs)

    if isinstance(rhs, NestedTensor):
        if isinstance(lhs, Tensor) and lhs.dim() == 2 and rhs._values.dim() == 2:
            if 0 not in rhs._varying_dims:
                new_values = torch.mm(rhs._values, lhs.transpose(0, 1))
                return _packed_new_dim_size(rhs, new_values, 0, int(lhs.shape[0]))
            if (
                rhs._packed_sizes is not None
                and len(set(rhs._packed_sizes)) == 1
                and rhs._physical_shape.size(1) == 2
                and rhs._element_shapes is not None
            ):
                cols = {int(shape[1]) for shape in rhs._element_shapes}
                if len(cols) == 1:
                    col_dim = next(iter(cols))
                    rhs_batched = rhs._values.view(len(rhs), rhs._packed_sizes[0], col_dim)
                    out_values = func(lhs, rhs_batched, **kwargs).reshape(-1, col_dim)
                    return _packed_new_dim_size(rhs, out_values, 0, int(lhs.shape[0]))
        if (
            isinstance(lhs, Tensor)
            and lhs.dim() > 2
            and rhs._values.dim() == 2
            and rhs._packed_sizes is not None
            and len(set(rhs._packed_sizes)) == 1
            and rhs._physical_shape.size(1) == 2
            and rhs._element_shapes is not None
        ):
            cols = {int(shape[1]) for shape in rhs._element_shapes}
            if len(cols) == 1:
                col_dim = next(iter(cols))
                rhs_batched = rhs._values.view(len(rhs), rhs._packed_sizes[0], col_dim)
                out_values = func(lhs.unsqueeze(0), rhs_batched.unsqueeze(1), **kwargs).reshape(
                    -1, *lhs.shape[-2:-1], col_dim
                )
                prefix = tuple(int(size) for size in lhs.shape[:-2]) + (int(lhs.shape[-2]),)
                shape, packed_sizes, element_shapes = rhs._shape_meta_from_components(prefix=prefix, keep_dims=(1,))
                return _packed_with_shape(
                    rhs,
                    out_values,
                    shape,
                    rhs._logical_shape_from_components(prefix=prefix, keep_dims=(1,)),
                    packed_sizes=packed_sizes,
                    element_shapes=element_shapes,
                )
        if isinstance(lhs, Tensor) and lhs.dim() <= 2 and rhs._values.dim() > 2:
            return _packed_with_tail_from_values(rhs, func(lhs, rhs._values, **kwargs))
        return per_element_fallback(func, args, kwargs)

    return func(*args, **kwargs)


@NestedTensorAtenRegistry.implement(aten.det.default)
@NestedTensorAtenRegistry.implement(aten.linalg_det.default)
def matrix_last2_to_scalar(func, args, kwargs):
    r"""Apply determinant-like ops and drop trailing matrix dims in metadata."""
    source = args[0]
    if source._values.dim() <= 2:
        if source._physical_shape.size(1) != 2:
            return _apply_per_element_nested(source, lambda t: func(t, *args[1:], **kwargs))

        device = source.device
        batch = len(source)
        rows = source._physical_shape[:, 0].to(device=device, dtype=torch.long)
        cols = source._physical_shape[:, 1].to(device=device, dtype=torch.long)
        max_rows, max_cols = source._max_physical_dims()

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


@NestedTensorAtenRegistry.implement(aten.triu.default)
@NestedTensorAtenRegistry.implement(aten.tril.default)
@NestedTensorAtenRegistry.implement(aten.matrix_exp.default)
@NestedTensorAtenRegistry.implement(aten.inverse.default)
@NestedTensorAtenRegistry.implement(aten.linalg_inv.default)
@NestedTensorAtenRegistry.implement(aten.linalg_cholesky.default)
def matrix_last2_unary(func, args, kwargs):
    r"""Apply matrix-style unary ops on packed values when ragged dim-0 is a batch axis."""
    source = args[0]
    if source._values.dim() <= 2:
        return _apply_per_element_nested(source, lambda t: func(t, *args[1:], **kwargs))
    return source._packed_like_unchecked(func(source._values, *args[1:], **kwargs))


@NestedTensorAtenRegistry.implement(aten.matrix_power.default)
@NestedTensorAtenRegistry.implement(aten.linalg_matrix_power.default)
def matrix_power(func, args, kwargs):
    r"""Apply matrix power on packed values when the ragged leading dim stays element-local."""
    source = args[0]
    if source._values.dim() <= 2:
        return _apply_per_element_nested(source, lambda t: func(t, *args[1:], **kwargs))
    return source._packed_like_unchecked(func(source._values, *args[1:], **kwargs))


# See also torch_functions.py::mm for the torch-level handler (mixed-type cases).
@NestedTensorAtenRegistry.implement(aten.mm.default)
def mm(func, args, kwargs):
    r"""Dispatch handler for matrix multiply (NT x dense) on packed _values."""
    from .nested_tensor import NestedTensor

    mat1, mat2 = args[0], args[1]
    if isinstance(mat1, NestedTensor) and not isinstance(mat2, NestedTensor) and mat1._values.dim() == 2:
        new_values = func(mat1._values, mat2, **kwargs)
        return _packed_new_last_dim(mat1, new_values, mat2.shape[1])
    if (
        not isinstance(mat1, NestedTensor)
        and isinstance(mat2, NestedTensor)
        and isinstance(mat1, Tensor)
        and mat1.dim() == 2
        and mat2._values.dim() == 2
    ):
        if 0 not in mat2._varying_dims:
            new_values = func(mat2._values, mat1.transpose(0, 1), **kwargs)
            return _packed_new_dim_size(mat2, new_values, 0, int(mat1.shape[0]))
        if (
            mat2._packed_sizes is not None
            and len(set(mat2._packed_sizes)) == 1
            and mat2._physical_shape.size(1) == 2
            and mat2._element_shapes is not None
        ):
            cols = {int(shape[1]) for shape in mat2._element_shapes}
            if len(cols) == 1:
                col_dim = next(iter(cols))
                rhs_batched = mat2._values.view(len(mat2), mat2._packed_sizes[0], col_dim)
                new_values = func(mat1, rhs_batched).reshape(-1, col_dim)
                return _packed_new_dim_size(mat2, new_values, 0, int(mat1.shape[0]))
    raise NotImplementedError(f"NestedTensor: {func} requires (NT × dense) with 2-D _values")


@NestedTensorAtenRegistry.implement(aten.trace.default)
def trace(func, args, kwargs):
    r"""Apply ``trace`` per element to preserve the underlying 2-D tensor semantics."""
    source = args[0]
    return _apply_per_element_nested(source, lambda t: func(t, **kwargs))


# ---------------------------------------------------------------------------
# Normalization ops — operate on packed _values
# ---------------------------------------------------------------------------


@NestedTensorAtenRegistry.implement(aten.native_layer_norm.default)
def native_layer_norm(func, args, kwargs):
    r"""Dispatch handler for layer norm on packed _values."""
    source = args[0]
    output, mean, rstd = func(source._values, *args[1:], **kwargs)
    return source._packed_like_unchecked(output), mean, rstd


@NestedTensorAtenRegistry.implement(aten.native_layer_norm_backward.default)
def native_layer_norm_backward(func, args, kwargs):
    r"""Dispatch handler for layer norm backward on packed _values."""
    from .nested_tensor import NestedTensor

    grad_out, input_ = args[0], args[1]
    sources = [a for a in (grad_out, input_) if isinstance(a, NestedTensor)]
    if not sources:
        return func(*args, **kwargs)
    if len(sources) == 2 and not sources[0]._has_same_layout(sources[1]):
        return per_element_fallback(func, args, kwargs)
    ref = sources[0]
    g = grad_out._values if isinstance(grad_out, NestedTensor) else grad_out
    i = input_._values if isinstance(input_, NestedTensor) else input_
    # args: grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask
    grad_input, grad_weight, grad_bias = func(g, i, *args[2:], **kwargs)
    return ref._packed_like_unchecked(grad_input), grad_weight, grad_bias


# ---------------------------------------------------------------------------
# Random tensor creation ops — same pattern as empty_like/zeros_like
# ---------------------------------------------------------------------------

ATEN_RANDOM_CREATION_OPS = [
    aten.rand_like.default,
    aten.randn_like.default,
    aten.randint_like.default,
    aten.randint_like.low_dtype,
]


# ---------------------------------------------------------------------------
# RNG in-place ops — shape-preserving mutations on _values
# ---------------------------------------------------------------------------

ATEN_INPLACE_RNG_OPS = [
    aten.uniform_.default,
    aten.normal_.default,
]


# ---------------------------------------------------------------------------
# Shape-preserving unary-like ops (extra scalar/keyword args, operate on _values)
# ---------------------------------------------------------------------------

ATEN_UNARY_LIKE_OPS = [
    aten.clamp.default,
    aten.clamp_min.default,
    aten.clamp_max.default,
    aten.nan_to_num.default,
    aten.alpha_dropout.default,
    aten.feature_alpha_dropout.default,
    aten.feature_dropout.default,
    aten.bernoulli.default,
]

# ---------------------------------------------------------------------------
# Shape/view ops — operate on packed _values and update metadata
# ---------------------------------------------------------------------------


@NestedTensorAtenRegistry.implement(aten.flatten.using_ints)
def flatten(func, args, kwargs):
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
        if source._ragged_rank >= 2 and start == 0 and end == 1:
            element_shapes = source._element_shapes
            if element_shapes is None:
                element_shapes = tuple(type(source)._trim_shape(shape) for shape in source._physical_shape.tolist())
            row_counts = tuple(int(shape[0]) for shape in element_shapes)
            if source.batch_first:
                row_pairs = tuple((batch, row) for batch, count in enumerate(row_counts) for row in range(count))
            else:
                max_rows = max(row_counts, default=0)
                row_pairs = tuple(
                    (batch, row) for row in range(max_rows) for batch, count in enumerate(row_counts) if row < count
                )
            row_shapes = tuple(element_shapes[batch][1:] for batch, _ in row_pairs)
            rank = int(source._physical_shape.size(1)) - 1
            if not row_shapes:
                shape = source._physical_shape.new_empty((0, rank))
                max_dims = tuple(
                    int(source._physical_shape[:, dim].max()) if source._physical_shape.size(0) else 0
                    for dim in range(1, rank + 1)
                )
                outer_size = (
                    torch.Size((0, *max_dims)) if source.batch_first else torch.Size((max_dims[0], 0, *max_dims[1:]))
                )
                return type(source)._from_packed(
                    source._values,
                    type(source)._offsets_from_sizes((), dtype=source._offsets.dtype),
                    shape,
                    permutation=tuple(range(rank)),
                    batch_first=source.batch_first,
                    padding_value=source.padding_value,
                    mask_value=source.mask_value,
                    pin_memory=source._pin_memory,
                    outer_size=outer_size,
                    packed_sizes=(),
                    element_shapes=(),
                    validate=False,
                )
            if source._permutation[:1] == (0,):
                row_varying, row_static = type(source)._pack_layout_from_element_shapes(row_shapes)
                packed_sizes = tuple(type(source)._packed_size_from_shape(shape, row_varying) for shape in row_shapes)
                offsets = type(source)._offsets_from_sizes(packed_sizes, dtype=source._offsets.dtype)
                row_block_sizes = []
                for element_shape in element_shapes:
                    block = 1
                    for dim in source._varying_dims:
                        if dim != 0:
                            block *= int(element_shape[dim])
                    row_block_sizes.append(block)
                if source.batch_first:
                    values = source._values
                else:
                    starts = [int(source._offsets[batch]) + row * row_block_sizes[batch] for batch, row in row_pairs]
                    starts_t = torch.tensor(starts, dtype=torch.long, device=source._values.device)
                    lengths_t = torch.tensor(packed_sizes, dtype=torch.long, device=source._values.device)
                    row_id = torch.repeat_interleave(
                        torch.arange(len(row_pairs), device=source._values.device),
                        lengths_t,
                    )
                    prefix = lengths_t.cumsum(0)
                    local = (
                        torch.arange(int(prefix[-1]), device=source._values.device) - prefix[row_id] + lengths_t[row_id]
                    )
                    values = source._values.index_select(0, starts_t[row_id] + local)
                shape = source._physical_shape.new_tensor(row_shapes)
                max_dims = tuple(max(shape[dim] for shape in row_shapes) for dim in range(len(row_shapes[0])))
                outer_size = (
                    torch.Size((len(row_shapes), *max_dims))
                    if source.batch_first
                    else torch.Size((max_dims[0], len(row_shapes), *max_dims[1:]))
                )
                return type(source)._from_packed(
                    values,
                    offsets,
                    shape,
                    permutation=row_varying + row_static,
                    batch_first=source.batch_first,
                    padding_value=source.padding_value,
                    mask_value=source.mask_value,
                    pin_memory=source._pin_memory,
                    outer_size=outer_size,
                    packed_sizes=packed_sizes,
                    element_shapes=row_shapes,
                    validate=False,
                )
            rows: list = []
            storage = source._storage
            if source.batch_first:
                for element in storage:
                    rows.extend(element.unbind(0))
            else:
                max_rows = max((element.shape[0] for element in storage), default=0)
                for row in range(max_rows):
                    rows.extend(element[row] for element in storage if row < element.shape[0])
            return type(source)(rows, **source._meta())
        return func(source.tensor, start_dim, end_dim, **kwargs)

    start_adj = _translate_dim(source, start)
    end_adj = _translate_dim(source, end)
    flattened_dims = tuple(range(start_adj, end_adj + 1))
    if any(dim not in source._static_dims for dim in flattened_dims):
        return per_element_fallback(func, (source, start_adj, end_adj), kwargs)
    packed_positions = tuple(source._static_dims.index(dim) for dim in flattened_dims)
    first_packed_position = packed_positions[0]
    if packed_positions != tuple(range(first_packed_position, first_packed_position + len(flattened_dims))):
        return per_element_fallback(func, (source, start_adj, end_adj), kwargs)

    values_start = 1 + first_packed_position
    values_end = values_start + len(flattened_dims) - 1
    out_values = func(source._values, values_start, values_end, **kwargs)
    merged = torch.prod(source._physical_shape[:, start_adj : end_adj + 1], dim=1, keepdim=True)
    out_shape = torch.cat(
        (source._physical_shape[:, :start_adj], merged, source._physical_shape[:, end_adj + 1 :]),
        dim=1,
    )
    physical_dims = list(source._max_physical_dims())
    physical_dims[start_adj : end_adj + 1] = [math.prod(physical_dims[start_adj : end_adj + 1])]
    out_packed_sizes = None
    out_element_shapes = None
    if source._element_shapes is not None:
        out_element_shapes = tuple(
            shape[:start_adj] + (math.prod(shape[start_adj : end_adj + 1]),) + shape[end_adj + 1 :]
            for shape in source._element_shapes
        )
        out_packed_sizes = source._packed_sizes_like(out_element_shapes)
    collapsed_rank = end_adj - start_adj
    shifted_permutation: list[int] = []
    flattened_inserted = False
    for physical_dim in source._permutation:
        if physical_dim < start_adj:
            shifted_permutation.append(physical_dim)
        elif physical_dim <= end_adj:
            if not flattened_inserted:
                shifted_permutation.append(start_adj)
                flattened_inserted = True
        else:
            shifted_permutation.append(physical_dim - collapsed_rank)
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_physical_dims(physical_dims),
        permutation=tuple(shifted_permutation),
        packed_sizes=out_packed_sizes,
        element_shapes=out_element_shapes,
        preserve_ragged_offsets=True,
    )


def _packed_metadata_permute(source: NestedTensor, tensor_dims: tuple[int, ...]) -> NestedTensor | None:
    r"""Relabel a logical per-element permutation when packed storage order is unchanged."""
    rank = int(source._physical_shape.size(1))
    if len(tensor_dims) != rank:
        return None
    out_shape = source._physical_shape[:, tensor_dims]
    out_element_shapes = None
    if source._element_shapes is not None:
        out_element_shapes = tuple(tuple(shape[dim] for dim in tensor_dims) for shape in source._element_shapes)

    old_to_new = {int(old_dim): new_dim for new_dim, old_dim in enumerate(tensor_dims)}
    new_packed_order = tuple(old_to_new[int(dim)] for dim in source._permutation)

    out_logical = source._logical_shape_from_physical_dims(
        tuple(source._max_physical_dims()[dim] for dim in tensor_dims)
    )
    return _packed_with_shape(
        source,
        source.concat,
        out_shape,
        out_logical,
        permutation=new_packed_order,
        packed_sizes=source._packed_sizes,
        element_shapes=out_element_shapes,
        preserve_ragged_offsets=True,
    )


@NestedTensorAtenRegistry.implement(aten.permute.default)
def permute(func, args, kwargs):
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
    metadata_only = _packed_metadata_permute(source, tensor_dims)
    if metadata_only is not None:
        return metadata_only
    if tensor_dims[0] != 0:
        # Ragged dim move stays per-element but should remain in the compile graph
        # when possible, so we avoid the dynamo-disabled generic fallback.
        return _apply_per_element_nested(source, lambda t: t.permute(*tensor_dims))
    out_values = func(source._values, list(tensor_dims), **kwargs)
    out_shape = source._physical_shape[:, tensor_dims]
    out_logical = source._logical_shape_from_physical_dims(
        tuple(source._max_physical_dims()[dim] for dim in tensor_dims)
    )
    out_packed_sizes = None
    out_element_shapes = None
    if source._element_shapes is not None:
        out_element_shapes = tuple(tuple(shape[dim] for dim in tensor_dims) for shape in source._element_shapes)
        out_packed_sizes = source._packed_sizes_like(out_element_shapes)
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        out_logical,
        packed_sizes=out_packed_sizes,
        element_shapes=out_element_shapes,
    )


@NestedTensorAtenRegistry.implement(aten.squeeze.default)
def squeeze_default(func, args, kwargs):
    r"""Squeeze all singleton per-element dims with a packed fastpath when ragged dim-0 is untouched."""
    source = args[0]
    rank = source._physical_shape.size(1)
    if rank == 0:
        return source._packed_like_unchecked(source._values)
    if source._element_shapes is None and (_is_compiling() or _is_fake_tensor(source._physical_shape)):
        _compile_unsupported(
            "aten.squeeze",
            "tensor-backed singleton-dimension analysis is not implemented",
        )

    # If any sample has ragged size 1, squeezing dim-0 is per-element.
    if source._physical_shape.size(0) > 0 and bool(torch.any(source._physical_shape[:, 0] == 1)):
        first_shape = source._element_shapes[0] if source._element_shapes else ()
        keep_dims = tuple(dim for dim, size in enumerate(first_shape) if int(size) != 1)
        if source._element_shapes is not None and all(
            tuple(dim for dim, size in enumerate(shape) if int(size) != 1) == keep_dims
            for shape in source._element_shapes
        ):
            ragged_dims = source._project_declared_ragged_dims(keep_dims=keep_dims)
        else:
            ragged_dims = None
        return _apply_per_element_nested(source, lambda t: t.squeeze(), ragged_dims=ragged_dims)

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
    physical_dims = [size for index, size in enumerate(source._max_physical_dims()) if not bool(squeeze_mask[index])]
    out_packed_sizes = None
    out_element_shapes = None
    if source._element_shapes is not None:
        squeeze_mask_list = tuple(bool(value) for value in squeeze_mask.tolist())
        out_element_shapes = tuple(
            tuple(size for index, size in enumerate(shape) if not squeeze_mask_list[index])
            for shape in source._element_shapes
        )
        out_packed_sizes = source._packed_sizes_like(out_element_shapes)
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_physical_dims(physical_dims),
        packed_sizes=out_packed_sizes,
        element_shapes=out_element_shapes,
    )


@NestedTensorAtenRegistry.implement(aten.squeeze.dim)
def squeeze_dim(func, args, kwargs):
    r"""Squeeze one logical dim; use packed fastpath for static per-element dims."""
    source = args[0]
    dim = _normalize_dim(args[1], source.dim())
    batch_dim = _get_batch_dim(source)
    if dim <= batch_dim:
        raise ValueError("Cannot squeeze the batch dimension or dimensions before it for NestedTensor.")

    dim_adj = _translate_dim(source, dim)
    values_dim = _physical_to_values_dim(source, dim_adj)
    if values_dim is not None:
        if source._values.shape[values_dim] != 1:
            return source
        return _packed_without_dim(source, dim_adj, func(source._values, values_dim, **kwargs))

    if source._element_shapes is not None:
        can_squeeze = all(dim_adj < len(shape) and int(shape[dim_adj]) == 1 for shape in source._element_shapes)
    else:
        if _is_compiling():
            _compile_unsupported("aten.squeeze.dim", "requires static element shape metadata")
        can_squeeze = bool(source._physical_shape[:, dim_adj].eq(1).all())
    if not can_squeeze:
        return source
    # The static case returned through packed values above. Squeezing a ragged
    # singleton is per-element because packed values collapse that dimension.
    keep_dims = tuple(dim for dim in range(source._physical_shape.size(1)) if dim != dim_adj)
    return _apply_per_element_nested(
        source,
        lambda t: t.squeeze(dim_adj),
        ragged_dims=source._project_declared_ragged_dims(keep_dims=keep_dims),
    )


def narrow_nested(source: NestedTensor, dim: int, start, length: int) -> NestedTensor:
    r"""
    Keep ``length`` positions of ``dim``, counting from ``start``.

    ``dim`` numbers the *logical* shape, whose batch dim has no per-element counterpart and
    whose remaining dims are the per-element ones. Handing that number to a dense kernel over
    ``_values`` reads whichever axis happens to sit there, which is a different dimension for
    every layout that is not the canonical leading-ragged one.

    Args:
        source: The NestedTensor to narrow.
        dim: The logical dimension to narrow.
        start: First position to keep. Negative counts from the end of ``dim``.
        length: How many positions to keep.

    Returns:
        NestedTensor: The narrowed result.
    """
    from .nested_tensor import NestedTensor

    if isinstance(start, Tensor):
        if start.numel() != 1:
            raise RuntimeError("narrow(): start must be a scalar")
        start = int(start.item())
    start = int(start)
    length = int(length)
    if length < 0:
        raise RuntimeError(f"narrow(): length must be non-negative, but got {length}")
    dim = _normalize_dim(dim, source.dim())
    if dim < 0 or dim >= source.dim():
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-source.dim()}, {source.dim() - 1}], but got {dim})"
        )
    batch_dim = _get_batch_dim(source)
    if dim == batch_dim:
        batch_size = len(source)
        if start < 0:
            start += batch_size
        if start < 0 or start + length > batch_size:
            raise RuntimeError(f"start ({start}) + length ({length}) exceeds dimension size ({batch_size}).")
        return _packed_batch_slice(source, start, length, "torch.narrow")

    dim_adj = _translate_dim(source, dim)
    values_dim = _packed_static_dim(source, dim_adj)
    if values_dim is not None:
        extent = int(source._values.shape[values_dim])
        resolved = start + extent if start < 0 else start
        if resolved < 0 or resolved + length > extent:
            raise RuntimeError(f"start ({start}) + length ({length}) exceeds dimension size ({extent}).")
        return _packed_static_slice(source, dim_adj, resolved, length)

    if _packed_sole_ragged_dim(source, dim_adj):
        if source._packed_sizes is None and _is_fake_tensor(source._offsets):
            if start < 0:
                _compile_unsupported("torch.narrow", "a negative start needs each sample's own extent")
            return _packed_uniform_ragged_slice(source, dim_adj, start, length)
        extents = _resolved_packed_sizes(source, "torch.narrow")
        starts = []
        for position, extent in enumerate(extents):
            resolved = start + extent if start < 0 else start
            if resolved < 0 or resolved + length > extent:
                raise RuntimeError(
                    f"start ({start}) + length ({length}) exceeds dimension size ({extent}) "
                    f"for NestedTensor element {position}."
                )
            starts.append(resolved)
        return _packed_ragged_slice(source, dim_adj, starts, [length] * len(starts))

    _check_execution_guard(_ExecutionGuardKind.EAGER_FALLBACK, "aten_functions.narrow_nested")
    if _is_compiling():
        _compile_unsupported("torch.narrow", "an outer ragged dim has no packed span")
    return NestedTensor(
        [element.narrow(dim_adj, start, length) for element in source._storage],
        **source._meta(),
    )


@NestedTensorAtenRegistry.implement(aten.narrow.default)
@NestedTensorAtenRegistry.implement(aten.narrow_copy.default)
def narrow(func, args, kwargs):
    r"""Narrow along a logical dim; the per-element index is not a packed axis."""
    source = args[0]
    dim = args[1] if len(args) > 1 else kwargs["dim"]
    start = args[2] if len(args) > 2 else kwargs["start"]
    length = args[3] if len(args) > 3 else kwargs["length"]
    return narrow_nested(source, dim, start, length)


@NestedTensorAtenRegistry.implement(aten.transpose.int)
def transpose(func, args, kwargs):
    r"""Transpose two non-batch logical dims, using packed storage only for static-dim swaps."""
    source = args[0]
    dim0 = _normalize_dim(args[1], source.dim())
    dim1 = _normalize_dim(args[2], source.dim())
    batch_dim = _get_batch_dim(source)
    if dim0 == batch_dim or dim1 == batch_dim:
        other = dim1 if dim0 == batch_dim else dim0
        seq_dim = 1 if source.batch_first else 0
        if other != seq_dim:
            raise ValueError("Cannot transpose the batch dimension with a non-sequence dimension for NestedTensor.")
        new_shape = list(source._logical_shape)
        new_shape[0], new_shape[1] = new_shape[1], new_shape[0]
        return type(source)._from_packed(
            source._values,
            source._offsets,
            source._physical_shape,
            batch_first=not source.batch_first,
            padding_value=source.padding_value,
            mask_value=source.mask_value,
            pin_memory=source._pin_memory,
            outer_size=torch.Size(new_shape),
            packed_sizes=source._packed_sizes,
            element_shapes=source._element_shapes,
            permutation=source._permutation,
            ragged_dims=source._ragged_dims if source._ragged_dims_explicit else None,
            ragged_offsets=source._persistent_ragged_offsets(),
            validate=False,
        )

    elem_dim0 = _translate_dim(source, dim0)
    elem_dim1 = _translate_dim(source, dim1)
    if elem_dim0 in source._varying_dims or elem_dim1 in source._varying_dims:
        tensor_dims = list(range(int(source._physical_shape.size(1))))
        tensor_dims[elem_dim0], tensor_dims[elem_dim1] = tensor_dims[elem_dim1], tensor_dims[elem_dim0]
        metadata_only = _packed_metadata_permute(source, tuple(tensor_dims))
        if metadata_only is not None:
            return metadata_only
        # Packed storage flattens ragged dimensions into the leading payload axis, so
        # swaps that touch them must happen per element to preserve shape semantics.
        return _apply_per_element_nested(source, lambda t: t.transpose(elem_dim0, elem_dim1))

    packed_dim0 = 1 + source._static_dims.index(elem_dim0)
    packed_dim1 = 1 + source._static_dims.index(elem_dim1)
    out_values = func(source._values, packed_dim0, packed_dim1, **kwargs)
    out_shape = source._physical_shape.clone()
    out_shape[:, [elem_dim0, elem_dim1]] = out_shape[:, [elem_dim1, elem_dim0]]
    physical_dims = list(source._max_physical_dims())
    physical_dims[elem_dim0], physical_dims[elem_dim1] = physical_dims[elem_dim1], physical_dims[elem_dim0]
    out_packed_sizes = None
    out_element_shapes = None
    if source._element_shapes is not None:
        transposed_shapes = []
        for shape in source._element_shapes:
            shape_list = list(shape)
            shape_list[elem_dim0], shape_list[elem_dim1] = shape_list[elem_dim1], shape_list[elem_dim0]
            transposed_shapes.append(tuple(shape_list))
        out_element_shapes = tuple(transposed_shapes)
        out_packed_sizes = source._packed_sizes_like(out_element_shapes)
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_physical_dims(physical_dims),
        packed_sizes=out_packed_sizes,
        element_shapes=out_element_shapes,
    )


@NestedTensorAtenRegistry.implement(aten.unflatten.int)
def unflatten(func, args, kwargs):
    r"""Unflatten static per-element dims on packed values and expand metadata."""
    source = args[0]
    dim = _normalize_dim(args[1], source.dim())
    sizes = args[2]
    batch_dim = _get_batch_dim(source)
    if dim <= batch_dim:
        raise ValueError("unflatten at or before the batch dimension is not supported for NestedTensor.")

    dim_adj = _translate_dim(source, dim)
    if dim_adj not in source._static_dims:
        # Unflattening a ragged dim can produce shape patterns that may collapse to a
        # plain Tensor when uniform across batch; keep generic fallback semantics.
        return per_element_fallback(func, (source, dim_adj, sizes), kwargs)

    values_dim = 1 + source._static_dims.index(dim_adj)
    out_values = func(source._values, values_dim, sizes, **kwargs)
    inserted_rank = out_values.dim() - source._values.dim() + 1
    resolved_sizes = out_values.shape[values_dim : values_dim + inserted_rank]
    inserted = source._physical_shape.new_tensor(resolved_sizes).unsqueeze(0).expand(source._physical_shape.size(0), -1)
    out_shape = torch.cat(
        (source._physical_shape[:, :dim_adj], inserted, source._physical_shape[:, dim_adj + 1 :]),
        dim=1,
    )
    physical_dims = list(source._max_physical_dims())
    physical_dims[dim_adj : dim_adj + 1] = [int(size) for size in resolved_sizes]
    out_packed_sizes = None
    out_element_shapes = None
    if source._element_shapes is not None:
        inserted_sizes = tuple(int(size) for size in resolved_sizes)
        out_element_shapes = tuple(
            shape[:dim_adj] + inserted_sizes + shape[dim_adj + 1 :] for shape in source._element_shapes
        )
        out_packed_sizes = source._packed_sizes_like(out_element_shapes)
    shifted_permutation: list[int] = []
    for physical_dim in source._permutation:
        if physical_dim < dim_adj:
            shifted_permutation.append(physical_dim)
        elif physical_dim == dim_adj:
            shifted_permutation.extend(range(dim_adj, dim_adj + inserted_rank))
        else:
            shifted_permutation.append(physical_dim + inserted_rank - 1)
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_physical_dims(physical_dims),
        permutation=tuple(shifted_permutation),
        packed_sizes=out_packed_sizes,
        element_shapes=out_element_shapes,
        preserve_ragged_offsets=True,
    )


@NestedTensorAtenRegistry.implement(aten.unsqueeze.default)
def unsqueeze(func, args, kwargs):
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
    ones = torch.ones_like(source._physical_shape[:, :1])
    out_shape = torch.cat(
        (source._physical_shape[:, :dim_adj], ones, source._physical_shape[:, dim_adj:]),
        dim=1,
    )
    physical_dims = list(source._max_physical_dims())
    physical_dims.insert(dim_adj, 1)
    shifted_varying = tuple(dim + 1 if dim >= dim_adj else dim for dim in source._varying_dims)
    shifted_static = tuple(dim + 1 if dim >= dim_adj else dim for dim in source._static_dims)
    packed_static = list(shifted_static)
    previous_dim = dim_adj - 1
    following_dim = dim_adj + 1
    if previous_dim in packed_static:
        packed_position = packed_static.index(previous_dim) + 1
    elif following_dim in packed_static:
        packed_position = packed_static.index(following_dim)
    else:
        previous_static = max((static_dim for static_dim in packed_static if static_dim < dim_adj), default=None)
        following_static = min((static_dim for static_dim in packed_static if static_dim > dim_adj), default=None)
        if previous_static is not None:
            packed_position = packed_static.index(previous_static) + 1
        elif following_static is not None:
            packed_position = packed_static.index(following_static)
        else:
            packed_position = 0
    packed_static.insert(packed_position, dim_adj)
    new_static = tuple(packed_static)
    packed_dim = 1 + packed_position
    out_values = func(source.concat, packed_dim, **kwargs)
    out_element_shapes = None
    if source._element_shapes is not None:
        out_element_shapes = tuple(shape[:dim_adj] + (1,) + shape[dim_adj:] for shape in source._element_shapes)
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_physical_dims(physical_dims),
        permutation=shifted_varying + new_static,
        packed_sizes=source._packed_sizes,
        element_shapes=out_element_shapes,
        preserve_ragged_offsets=True,
    )


@NestedTensorAtenRegistry.implement(aten.view.default, compile_safe=True)
@NestedTensorAtenRegistry.implement(aten.view_copy.default, compile_safe=True)
@NestedTensorAtenRegistry.implement(aten.reshape.default, compile_safe=True)
def view_like(func, args, kwargs):
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
        _check_execution_guard(_ExecutionGuardKind.STORAGE_MAP, f"{func}.view_like_rebuild")
        outputs = [func(t, list(s), **kwargs) for t, s in zip(source._unpack(), view_shapes)]
        return type(source)(outputs, **source._meta())

    rank = len(view_shapes[0])
    if any(-1 in s for s in view_shapes):
        return rebuild_per_element()

    if not all(len(s) == rank for s in view_shapes):
        return rebuild_per_element()

    if rank > 0:
        tail = view_shapes[0][1:]
        tails_uniform = all(s[1:] == tail for s in view_shapes[1:])
    else:
        tails_uniform = True

    if not tails_uniform:
        return rebuild_per_element()

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

    if packed_sizes == source._packed_sizes:
        out_offsets = source._offsets
    else:
        lengths_tensor = torch.as_tensor(lengths, dtype=source._offsets.dtype, device=source._offsets.device)
        out_offsets = torch.empty(
            (lengths_tensor.numel() + 1,), dtype=source._offsets.dtype, device=source._offsets.device
        )
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
    if len(sources) == 2 and not sources[0]._has_same_layout(sources[1]):
        return per_element_fallback(func, args, kwargs)
    ref = sources[0]
    va = a._values if isinstance(a, NestedTensor) else a
    vb = b._values if isinstance(b, NestedTensor) else b
    return ref._packed_like_unchecked(func(va, vb, *args[2:], **kwargs))


def _packed_varying_softmax_group_indices(
    source: NestedTensor,
    target_dim: int,
    batch_idx: Tensor,
    varying_coords: tuple[Tensor, ...],
) -> tuple[Tensor, int] | None:
    r"""Build compact group indices for a softmax over one varying dimension.

    Groups are laid out independently for each element using that element's
    actual non-target varying sizes.  This avoids allocating the batch-wide
    Cartesian product of global maxima for sparse or anti-correlated shapes.
    """
    element_shapes = source._element_shapes
    if element_shapes is None:
        if _is_compiling() or _is_fake_tensor(source._physical_shape):
            _compile_unsupported(
                "NestedTensor ragged softmax",
                "tensor-backed varying-dimension grouping is not implemented",
            )
        if _is_fake_tensor(source._physical_shape):
            return None
        element_shapes = tuple(tuple(int(size) for size in shape) for shape in source._physical_shape.tolist())

    excluded = tuple(
        (physical_dim, coord)
        for physical_dim, coord in zip(source._varying_dims, varying_coords)
        if physical_dim != target_dim
    )
    group_offsets = [0]
    for shape in element_shapes:
        target_size = int(shape[target_dim])
        group_count = math.prod(int(shape[physical_dim]) for physical_dim, _ in excluded)
        group_offsets.append(group_offsets[-1] + (group_count if target_size > 0 else 0))

    offsets = torch.tensor(group_offsets[:-1], dtype=batch_idx.dtype, device=batch_idx.device)
    local_group_idx = torch.zeros_like(batch_idx)
    if excluded:
        per_element_radices = torch.tensor(
            [[int(shape[physical_dim]) for physical_dim, _ in excluded] for shape in element_shapes],
            dtype=batch_idx.dtype,
            device=batch_idx.device,
        )
        for index, (_, coord) in enumerate(excluded):
            radix = per_element_radices[:, index].index_select(0, batch_idx)
            local_group_idx = local_group_idx * radix + coord

    return offsets.index_select(0, batch_idx) + local_group_idx, group_offsets[-1]


def _packed_varying_softmax(
    source: NestedTensor,
    target_dim: int,
    *,
    log: bool,
    half_to_float: bool,
) -> NestedTensor | None:
    r"""Apply softmax over one varying dimension without padding or unpacking."""
    varying_dims = source._varying_dims
    if target_dim not in varying_dims:
        return None

    source_values = source._values
    values = (
        source_values.float()
        if half_to_float or source_values.dtype in (torch.float16, torch.bfloat16)
        else source_values
    )
    if values.shape[0] == 0:
        return source._packed_like_unchecked(values if half_to_float else source_values)

    batch_idx, local_idx = source._packed_batch_local_indices(device=values.device, dtype=torch.long)
    varying_coords = source._packed_varying_coords(
        batch_idx,
        local_idx,
        device=values.device,
        dtype=torch.long,
    )
    groups = _packed_varying_softmax_group_indices(source, target_dim, batch_idx, varying_coords)
    if groups is None:
        return None
    group_idx, group_count = groups

    tail_rank = values.dim() - 1
    scatter_idx = group_idx.reshape(-1, *([1] * tail_rank)).expand_as(values)
    group_shape = (group_count, *values.shape[1:])
    maxima = values.new_full(group_shape, float("-inf"))
    maxima = maxima.scatter_reduce(0, scatter_idx, values, "amax", include_self=False)
    shifted = values - maxima.index_select(0, group_idx)
    exponentials = shifted.exp()
    sums = values.new_zeros(group_shape).index_add(0, group_idx, exponentials)
    gathered_sums = sums.index_select(0, group_idx)
    out_values = shifted - gathered_sums.log() if log else exponentials / gathered_sums
    if not half_to_float and out_values.dtype != source_values.dtype:
        out_values = out_values.to(dtype=source_values.dtype)
    return source._packed_like_unchecked(out_values)


def _softmax_handler(func, args, kwargs):
    r"""Dispatch handler for softmax/log_softmax that translates the dim argument."""
    source = args[0]
    dim_adj = _translate_dim(source, args[1])
    half_to_float = bool(args[2]) if len(args) > 2 else bool(kwargs.get("half_to_float", False))
    values_dim = _physical_to_values_dim(source, dim_adj)
    if values_dim is None:
        square = _packed_square_softmax(
            source,
            dim_adj,
            log=func is aten._log_softmax.default,
            half_to_float=half_to_float,
        )
        if square is not None:
            return square
        varying = _packed_varying_softmax(
            source,
            dim_adj,
            log=func is aten._log_softmax.default,
            half_to_float=half_to_float,
        )
        if varying is not None:
            return varying
        return _apply_per_element_nested(source, lambda t: func(t, dim_adj, *args[2:], **kwargs))
    return source._packed_like_unchecked(func(source._values, values_dim, *args[2:], **kwargs))


# ---------------------------------------------------------------------------
# Sorting / cumulative / reordering ops.
# Fast path: N-D packing with non-ragged target dims (operate on _values).
# Fallback: 1-D packing or ragged dim -> per-element path.
# ---------------------------------------------------------------------------


def _packed_static_dim(source: NestedTensor, dim_adj: int) -> int | None:
    r"""Return the packed axis for a static per-element dim, or None when it is ragged.

    A per-element dim is only its own packed axis when the packed permutation is the identity,
    so every handler that runs a dense kernel on ``_values`` has to map through the layout
    rather than reusing ``dim_adj``. The ragged dims collapse into packed axis 0 and get None;
    :func:`_has_single_packed_ragged_dim` decides whether the segmented path may claim them.
    """
    try:
        return _physical_to_values_dim(source, dim_adj)
    except (IndexError, RuntimeError):
        return None


def _sort_like_compile_safe(args: tuple, kwargs: dict[str, object]) -> bool:
    r"""Return whether sort/argsort/topk stay on packed compile-safe paths."""
    source = args[0]
    kw_dim = kwargs.get("dim", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            return False
        dim = args[1]
    else:
        dim = -1 if kw_dim is _MISSING else kw_dim
    try:
        dim_adj = _translate_dim(source, dim)
    except (TypeError, ValueError, IndexError):
        return False
    return _packed_static_dim(source, dim_adj) is not None or _has_single_packed_ragged_dim(source, dim_adj)


def _cumulative_compile_safe(args: tuple, kwargs: dict[str, object]) -> bool:
    r"""Return whether cumulative ops stay on packed compile-safe paths."""
    source = args[0]
    kw_dim = kwargs.get("dim", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            return False
        dim = args[1]
    else:
        if kw_dim is _MISSING:
            return False
        dim = kw_dim
    try:
        dim_adj = _translate_dim(source, dim)
    except (TypeError, ValueError, IndexError):
        return False
    # The ragged path is a data-dependent log-step loop, so only the static path may compile.
    return _packed_static_dim(source, dim_adj) is not None


def _cumsum_compile_safe(args: tuple, kwargs: dict[str, object]) -> bool:
    r"""Return whether cumsum stays on packed compile-safe paths."""
    source = args[0]
    kw_dim = kwargs.get("dim", _MISSING)
    if len(args) > 1:
        if kw_dim is not _MISSING:
            return False
        dim = args[1]
    else:
        if kw_dim is _MISSING:
            return False
        dim = kw_dim
    try:
        dim_adj = _translate_dim(source, dim)
    except (TypeError, ValueError, IndexError):
        return False
    return _packed_static_dim(source, dim_adj) is not None or _has_single_packed_ragged_dim(source, dim_adj)


def _segmented_cumsum_values(source: NestedTensor, *extra_args, **kwargs) -> Tensor:
    r"""Compute cumsum along the leading ragged dimension without padding."""
    values = source._values
    out_values = aten.cumsum.default(values, 0, *extra_args, **kwargs)
    if values.numel() == 0 or values.shape[0] == 0:
        return out_values

    offsets = source._offsets.to(device=values.device, dtype=torch.long)
    batch_size = offsets.numel() - 1
    if batch_size == 0:
        return out_values

    start_offsets = offsets[:-1]
    zero_prefix = out_values.new_zeros((1, *out_values.shape[1:]))
    prefix_indices = torch.clamp(start_offsets - 1, min=0)
    raw_prefix = out_values.index_select(0, prefix_indices)
    has_prefix = start_offsets > 0
    while has_prefix.dim() < raw_prefix.dim():
        has_prefix = has_prefix.unsqueeze(-1)
    segment_prefix = torch.where(has_prefix, raw_prefix, zero_prefix)

    prefix_delta = torch.cat((segment_prefix[:1], segment_prefix[1:] - segment_prefix[:-1]), dim=0)
    active = start_offsets < values.shape[0]
    while active.dim() < prefix_delta.dim():
        active = active.unsqueeze(-1)
    prefix_delta = torch.where(active, prefix_delta, torch.zeros_like(prefix_delta))
    safe_start_offsets = torch.where(start_offsets < values.shape[0], start_offsets, torch.zeros_like(start_offsets))

    correction = out_values.new_zeros(out_values.shape)
    correction = correction.index_add(0, safe_start_offsets, prefix_delta)
    return out_values - aten.cumsum.default(correction, 0, *extra_args, **kwargs)


def _cumprod_result_dtype(values_dtype: torch.dtype, dtype: torch.dtype | None) -> torch.dtype:
    r"""Return the dtype in which native per-segment cumprod must accumulate."""
    if dtype is not None:
        return dtype
    if values_dtype.is_floating_point or values_dtype.is_complex:
        return values_dtype
    # aten.cumprod promotes integral and boolean inputs when dtype is omitted.
    return torch.int64


def _flip_compile_safe(args: tuple, kwargs: dict[str, object]) -> bool:
    r"""Return whether flip stays on packed compile-safe paths."""
    source = args[0]
    kw_dims = kwargs.get("dims", _MISSING)
    if len(args) > 1:
        if kw_dims is not _MISSING:
            return False
        dims = args[1]
    else:
        dims = () if kw_dims is _MISSING else kw_dims
    if isinstance(dims, int):
        dims = (dims,)
    try:
        dims_adj = tuple(_translate_dim(source, dim) for dim in dims)
    except (TypeError, ValueError, IndexError):
        return False
    return source._values.dim() > 1 and all(dim in source._static_dims for dim in dims_adj)


def _topk_compile_safe(args: tuple, kwargs: dict[str, object]) -> bool:
    r"""Return whether topk stays on packed compile-safe paths."""
    source = args[0]
    kw_dim = kwargs.get("dim", _MISSING)
    if len(args) > 2:
        if kw_dim is not _MISSING:
            return False
        dim = args[2]
    else:
        dim = -1 if kw_dim is _MISSING else kw_dim
    try:
        dim_adj = _translate_dim(source, dim)
        values_dim = _physical_to_values_dim(source, dim_adj)
    except (TypeError, ValueError, IndexError):
        return False
    return source._ragged_rank <= 1 and values_dim is not None


@NestedTensorAtenRegistry.implement(
    aten.argsort.default,
    compile_safe=True,
    compile_guard=_sort_like_compile_safe,
)
@NestedTensorAtenRegistry.implement(
    aten.argsort.stable,
    compile_safe=True,
    compile_guard=_sort_like_compile_safe,
)
def argsort(func, args, kwargs):
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
    values_dim = _packed_static_dim(source, dim_adj)
    if values_dim is not None:
        return source._packed_like_unchecked(_call_argsort(source._values, values_dim))
    if _has_single_packed_ragged_dim(source, dim_adj):
        # Sort the packed values in place of a padded rectangle; see ``sort`` below for detail.
        from .segmented import segmented_sort_perm

        _, local = segmented_sort_perm(
            source._values,
            source._offsets,
            source.packed_batch_indices(),
            descending=descending,
        )
        return source._packed_like_unchecked(local)
    if stable_overload or stable is not None:
        return per_element_fallback(
            torch.ops.aten.argsort.stable,
            (source,),
            {"stable": bool(stable), "dim": dim_adj, "descending": descending, **kwargs},
        )
    return per_element_fallback(func, (source, dim_adj, descending), kwargs)


@NestedTensorAtenRegistry.implement(
    aten.cumsum.default,
    compile_safe=True,
    compile_guard=_cumsum_compile_safe,
)
@NestedTensorAtenRegistry.implement(
    aten.cumprod.default,
    compile_safe=True,
    compile_guard=_cumsum_compile_safe,
)
@NestedTensorAtenRegistry.implement(
    aten.logcumsumexp.default,
    compile_safe=True,
    compile_guard=_cumulative_compile_safe,
)
def cumulative(func, args, kwargs):
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
    values_dim = _packed_static_dim(source, dim_adj)
    if values_dim is not None:
        return source._packed_like_unchecked(func(source._values, values_dim, *extra_args, **kwargs))
    if _has_single_packed_ragged_dim(source, dim_adj):
        if func is aten.cumsum.default:
            return source._packed_like_unchecked(_segmented_cumsum_values(source, *extra_args, **kwargs))
        # cumsum returned above through its own packed path; only cumprod and logcumsumexp
        # arrive here. Neither has a usable inverse, so they need a real segmented scan rather
        # than the global-scan-and-correct trick cumsum uses.
        dtype = kwargs.pop("dtype", None)
        # Both schemas end at ``dtype``, which is keyword-only, so nothing else should arrive.
        # Refuse rather than drop, so a schema change surfaces here instead of silently.
        if extra_args or kwargs:
            raise TypeError(
                f"{func._schema.name.split('::')[-1]}() got unexpected arguments {extra_args!r}, {kwargs!r}"
            )
        if func is aten.cumprod.default:
            from .segmented import segmented_cumprod

            result_dtype = _cumprod_result_dtype(source._values.dtype, dtype)
            # Cast before accumulating, not after: that is the dense op's contract, and the two
            # orders are not numerically equivalent.
            values = source._values.to(result_dtype)
            return source._packed_like_unchecked(segmented_cumprod(values, source._offsets))
        if _is_compiling():
            _compile_unsupported(
                f"{func._schema.name.split('::')[-1]}",
                "ragged-dimension cumulative ops are eager-only under compile",
            )
        from .segmented import segmented_scan

        result_dtype = source._values.dtype if dtype is None else dtype
        scanned = segmented_scan(source._values.to(result_dtype), source.packed_batch_indices(), torch.logaddexp)
        return source._packed_like_unchecked(scanned)
    return per_element_fallback(func, (source, dim_adj, *extra_args), kwargs)


@NestedTensorAtenRegistry.implement(aten.dropout.default, compile_safe=True)
def dropout(func, args, kwargs):
    r"""Apply aten dropout on packed values, preserving eval-mode identity."""
    source = args[0]
    p = args[1] if len(args) > 1 else kwargs.get("p", 0.5)
    train = args[2] if len(args) > 2 else kwargs.get("train", True)
    if (not bool(train)) or float(p) == 0:
        return source
    return source._packed_like_unchecked(func(source._values, *args[1:], **kwargs))


@NestedTensorAtenRegistry.implement(
    aten.cummax.default,
    compile_safe=True,
    compile_guard=_cumulative_compile_safe,
)
@NestedTensorAtenRegistry.implement(
    aten.cummin.default,
    compile_safe=True,
    compile_guard=_cumulative_compile_safe,
)
def cumulative_pair(func, args, kwargs):
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
    values_dim = _packed_static_dim(source, dim_adj)
    if values_dim is not None:
        vals, idxs = func(source._values, values_dim, **kwargs)
        return source._packed_like_unchecked(vals), source._packed_like_unchecked(idxs)
    if _has_single_packed_ragged_dim(source, dim_adj):
        if _is_compiling():
            _compile_unsupported(
                f"{func._schema.name.split('::')[-1]}",
                "ragged-dimension cumulative ops are eager-only under compile",
            )
        from .segmented import segmented_arg_scan

        if kwargs:
            # aten.cummax/cummin.default take no keyword arguments besides `dim` (already
            # popped above), so this should be unreachable; raise rather than drop silently.
            raise TypeError(f"{func._schema.name.split('::')[-1]}() got unexpected arguments {kwargs!r}")
        largest = func is aten.cummax.default
        values, indices = segmented_arg_scan(
            source._values,
            source.packed_batch_indices(),
            source.packed_local_indices(device=source._values.device),
            largest=largest,
        )
        return source._packed_like_unchecked(values), source._packed_like_unchecked(indices)
    return per_element_fallback(func, (source, dim_adj), kwargs)


@NestedTensorAtenRegistry.implement(
    aten.flip.default,
    compile_safe=True,
    compile_guard=_flip_compile_safe,
)
def flip(func, args, kwargs):
    r"""Flip packed values when the requested dims have packed row/tail semantics."""
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
    varying_dims = tuple(dim for dim in dims_adj if dim in source._varying_dims)
    static_dims = tuple(dim for dim in dims_adj if dim in source._static_dims)
    if len(varying_dims) == 0 and all(dim in source._static_dims for dim in dims_adj):
        packed_dims = tuple(source._static_dims.index(dim) + 1 for dim in static_dims)
        return source._packed_like_unchecked(func(source._values, packed_dims, **kwargs))
    if len(set(varying_dims)) == 1 and _has_single_packed_ragged_dim(source, varying_dims[0]):
        if _is_compiling():
            _compile_unsupported("aten.flip.default", "ragged-dimension flip is eager-only under compile")
        values = source._values
        offsets = source.ragged_level_offsets(0, device=values.device, dtype=torch.long)
        lengths = offsets[1:] - offsets[:-1]
        batch_idx = source.packed_batch_indices(device=values.device)
        local_idx = source.packed_local_indices(device=values.device)
        gather = offsets[batch_idx] + lengths[batch_idx] - 1 - local_idx
        out_values = values.index_select(0, gather)
        if static_dims:
            packed_dims = tuple(source._static_dims.index(dim) + 1 for dim in static_dims)
            out_values = func(out_values, packed_dims, **kwargs)
        return source._packed_like_unchecked(out_values)
    return per_element_fallback(func, (source, dims_adj), kwargs)


@NestedTensorAtenRegistry.implement(aten.roll.default)
def roll(func, args, kwargs):
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
        if _is_compiling():
            _compile_unsupported("aten.roll.default", "dims=None flatten semantics are eager-only")
        return per_element_fallback(func, (source, shifts, []), kwargs)

    dims_adj = tuple(_translate_dim(source, dim) for dim in dims)
    if source._values.dim() > 1 and all(dim > 0 for dim in dims_adj):
        return source._packed_like_unchecked(func(source._values, shifts, list(dims_adj), **kwargs))
    if _is_compiling():
        _compile_unsupported("aten.roll.default", "only non-ragged roll dimensions are compile-safe")
    return per_element_fallback(func, (source, shifts, list(dims_adj)), kwargs)


@NestedTensorAtenRegistry.implement(aten.rot90.default)
def rot90(func, args, kwargs):
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
    k_mod = int(k) % 4

    if source._values.dim() > 1 and all(dim > 0 for dim in dims_adj):
        out_values = func(source._values, k, list(dims_adj), **kwargs)
        if k_mod % 2 == 0:
            return source._packed_like_unchecked(out_values)
        out_shape = source._physical_shape.clone()
        out_shape[:, [dims_adj[0], dims_adj[1]]] = out_shape[:, [dims_adj[1], dims_adj[0]]]
        out_logical = list(source._logical_shape)
        out_logical[dims_norm[0]], out_logical[dims_norm[1]] = out_logical[dims_norm[1]], out_logical[dims_norm[0]]
        out_packed_sizes = None
        out_element_shapes = None
        if source._element_shapes is not None:
            rotated_shapes = []
            for shape in source._element_shapes:
                shape_list = list(shape)
                shape_list[dims_adj[0]], shape_list[dims_adj[1]] = shape_list[dims_adj[1]], shape_list[dims_adj[0]]
                rotated_shapes.append(tuple(shape_list))
            out_element_shapes = tuple(rotated_shapes)
            out_packed_sizes = source._packed_sizes_like(out_element_shapes)
        return _packed_with_shape(
            source,
            out_values,
            out_shape,
            out_logical,
            packed_sizes=out_packed_sizes,
            element_shapes=out_element_shapes,
        )

    if _is_compiling():
        _compile_unsupported("aten.rot90.default", "only non-ragged rotation planes are compile-safe")
    return per_element_fallback(func, (source, k, list(dims_adj)), kwargs)


@NestedTensorAtenRegistry.implement(aten.searchsorted.Tensor)
def searchsorted_tensor(func, args, kwargs):
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
                return values._packed_like_unchecked(out_values)

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
            return values._packed_like_unchecked(out_values)

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


@NestedTensorAtenRegistry.implement(
    aten.sort.default,
    compile_safe=True,
    compile_guard=_sort_like_compile_safe,
)
@NestedTensorAtenRegistry.implement(
    aten.sort.stable,
    compile_safe=True,
    compile_guard=_sort_like_compile_safe,
)
def sort(func, args, kwargs):
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
    values_dim = _packed_static_dim(source, dim_adj)
    if values_dim is not None:
        vals, idxs = _call_sort(source._values, values_dim)
        return source._packed_like_unchecked(vals), source._packed_like_unchecked(idxs)
    if _has_single_packed_ragged_dim(source, dim_adj):
        # Sort the packed values in place of a padded rectangle. The segmented permutation is
        # stable, which is what the ``stable=True`` overload asks for and is harmless otherwise.
        from .segmented import segmented_sort_perm

        perm, local = segmented_sort_perm(
            source._values,
            source._offsets,
            source.packed_batch_indices(),
            descending=descending,
        )
        return source._packed_like_unchecked(torch.gather(source._values, 0, perm)), source._packed_like_unchecked(
            local
        )
    if stable_overload or stable is not None:
        return per_element_fallback(
            torch.ops.aten.sort.stable,
            (source,),
            {"stable": stable, "dim": dim_adj, "descending": descending, **kwargs},
        )
    return per_element_fallback(func, (source, dim_adj, descending), kwargs)


@NestedTensorAtenRegistry.implement(
    aten.topk.default,
    compile_safe=True,
    compile_guard=_topk_compile_safe,
)
def topk(func, args, kwargs):
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
    if source._ragged_rank > 1:
        # Multiple ragged levels collapse into one packed dim in ``_values``; take top-k per element instead.
        return per_element_fallback(func, (source, k, dim_adj, largest, sorted_output), kwargs)
    values_dim = _packed_static_dim(source, dim_adj)
    if values_dim is not None:
        vals, idxs = func(source._values, k, values_dim, largest, sorted_output, **kwargs)
        return (
            _packed_new_dim_size(source, vals, dim_adj, k),
            _packed_new_dim_size(source, idxs, dim_adj, k),
        )
    if _has_single_packed_ragged_dim(source, dim_adj):
        # Sort the packed segments instead of padding to a rectangle, then keep the first k of
        # each. k is checked against the shortest segment because a per-segment k has no dense
        # meaning. ``topk`` stays eager-only under compile (see ``_topk_compile_safe``) because
        # that check is data-dependent.
        from .segmented import segmented_sort_perm

        offsets = source._offsets.to(device=source._values.device, dtype=torch.long)
        lengths = offsets[1:] - offsets[:-1]
        k_value = int(k)
        if lengths.numel() > 0 and not (_is_compiling() or _is_fake_tensor(source._values)):
            shortest = int(lengths.min())
            if k_value > shortest:
                raise RuntimeError(f"selected index k out of range: k={k_value} exceeds shortest segment {shortest}")

        perm, local = segmented_sort_perm(
            source._values, source._offsets, source.packed_batch_indices(), descending=largest
        )
        # Every output segment now has the uniform width k. Selecting the survivors by their
        # offsets keeps the output shape a function of k alone; a boolean mask over the sorted
        # rows would instead make it depend on the data, which no graph can trace.
        keep = (offsets[:-1].view(-1, 1) + torch.arange(k_value, device=perm.device)).reshape(-1)
        return (
            _packed_new_ragged_size(
                source, torch.gather(source._values, 0, perm).index_select(0, keep), dim_adj, k_value
            ),
            _packed_new_ragged_size(source, local.index_select(0, keep), dim_adj, k_value),
        )
    return per_element_fallback(func, (source, k, dim_adj, largest, sorted_output), kwargs)


# ---------------------------------------------------------------------------
# Structured / shape-changing ops
# ---------------------------------------------------------------------------


@NestedTensorAtenRegistry.implement(aten.alias.default)
def alias(func, args, kwargs):
    r"""Create an alias of the NestedTensor sharing the same _values storage."""
    source = args[0]
    return source._packed_like_unchecked(func(source._values, **kwargs))


@NestedTensorAtenRegistry.implement(aten.clone.default)
def clone(func, args, kwargs):
    r"""Clone all internal tensors of a NestedTensor."""
    source = args[0]
    ragged_offsets = source._persistent_ragged_offsets()
    return type(source)._from_packed(
        source._values.clone(),
        source._offsets.clone(),
        source._physical_shape.clone(),
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=source._logical_shape,
        packed_sizes=source._packed_sizes,
        element_shapes=source._element_shapes,
        permutation=source._permutation,
        ragged_dims=source._ragged_dims if source._ragged_dims_explicit else None,
        ragged_offsets=tuple(level_offsets.clone() for level_offsets in ragged_offsets) if ragged_offsets else None,
        validate=False,
    )


@NestedTensorAtenRegistry.implement(aten.constant_pad_nd.default)
def constant_pad_nd(func, args, kwargs):
    r"""Compile-safe handler for constant padding on packed values when ragged dim is untouched."""
    source = args[0]
    pad = tuple(args[1])
    if len(pad) % 2 != 0:
        return per_element_fallback(func, args, kwargs)

    if len(pad) == 2:
        value = args[2] if len(args) > 2 else kwargs.get("value", 0)
        output = _constant_pad_packed_variable_last_dim(source, pad, value)
        if output is not None:
            return output

    padded_dims = len(pad) // 2
    # Packed fast path is valid only when padding targets trailing static dims.
    if source._values.dim() <= 1 or padded_dims >= source._values.dim():
        return per_element_fallback(func, args, kwargs)

    out_values = func(source._values, *args[1:], **kwargs)
    out_physical_shape = source._physical_shape.clone()
    for i in range(padded_dims):
        out_physical_shape[:, -(i + 1)] += pad[2 * i] + pad[2 * i + 1]
    out_logical = list(source._max_physical_dims())
    for i in range(padded_dims):
        out_logical[-(i + 1)] += int(pad[2 * i] + pad[2 * i + 1])
    return _packed_with_shape(
        source,
        out_values,
        out_physical_shape,
        source._logical_shape_from_physical_dims(out_logical),
        permutation=source._permutation,
    )


def _constant_pad_packed_variable_last_dim(source: NestedTensor, pad: tuple[int, int], value) -> NestedTensor | None:
    r"""Pad a packed ragged last physical dimension by inserting values between packed rows."""
    left, right = int(pad[0]), int(pad[1])
    if left < 0 or right < 0:
        return None
    if left == 0 and right == 0:
        return source

    rank = source._physical_shape.size(1)
    if rank < 1:
        return None
    target_dim = rank - 1
    if len(source._permutation) == 0 or int(source._permutation[0]) != target_dim:
        return None
    if source._varying_dims != (target_dim,):
        return None
    if source._element_shapes is not None and any(len(shape) != rank for shape in source._element_shapes):
        return None

    pad_width = left + right
    batch_steps = torch.arange(len(source) + 1, dtype=source._offsets.dtype, device=source._offsets.device)
    new_offsets = source._offsets + batch_steps * pad_width

    old_total = source._values.size(0)
    new_total = old_total + len(source) * pad_width
    output_values = source._values.new_full((new_total, *source._values.shape[1:]), value)
    batch_indices = source.packed_batch_indices(device=source._values.device)
    source_indices = torch.arange(old_total, device=source._values.device)
    destination_indices = source_indices + batch_indices * pad_width + left
    output_values.index_copy_(0, destination_indices, source._values)

    shape_tensor = source._physical_shape.clone()
    shape_tensor[:, target_dim] += pad_width
    element_shapes = None
    if source._element_shapes is not None:
        element_shapes = tuple(
            (*shape[:target_dim], shape[target_dim] + pad_width, *shape[target_dim + 1 :])
            for shape in source._element_shapes
        )
    if source._packed_sizes is not None:
        packed_sizes = tuple(int(size) + pad_width for size in source._packed_sizes)
    else:
        if _is_compiling() or _is_fake_tensor(source._offsets):
            _compile_unsupported(
                "aten.constant_pad_nd",
                "tensor-backed ragged padding metadata is not implemented",
            )
        packed_sizes = tuple(int(size) for size in (new_offsets[1:] - new_offsets[:-1]).tolist())

    outer_size = list(source._logical_shape)
    outer_size[-1] += pad_width
    return type(source)._from_packed(
        output_values,
        new_offsets,
        shape_tensor,
        permutation=source._permutation,
        ragged_dims=source._ragged_dims if source._ragged_dims_explicit else None,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=torch.Size(outer_size),
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
        validate=False,
    )


@NestedTensorAtenRegistry.implement(aten.detach.default)
def detach(func, args, kwargs):
    r"""Detach all internal tensors from the computation graph."""
    source = args[0]
    ragged_offsets = source._persistent_ragged_offsets()
    return type(source)._from_packed(
        source._values.detach(),
        source._offsets.detach(),
        source._physical_shape.detach(),
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=source._logical_shape,
        packed_sizes=source._packed_sizes,
        element_shapes=source._element_shapes,
        permutation=source._permutation,
        ragged_dims=source._ragged_dims if source._ragged_dims_explicit else None,
        ragged_offsets=tuple(level_offsets.detach() for level_offsets in ragged_offsets) if ragged_offsets else None,
        validate=False,
    )


@NestedTensorAtenRegistry.implement(aten.native_dropout.default)
def native_dropout(func, args, kwargs):
    r"""Apply native dropout per element, returning (output, mask) as NestedTensors."""
    source = args[0]
    cls = type(source)
    if len(source) == 0:
        empty = cls([], **source._meta())
        return empty, empty

    outputs = []
    masks = []
    for t in source._storage:
        out, mask = func(t, *args[1:], **kwargs)
        outputs.append(out)
        masks.append(mask)
    return cls(outputs, **source._meta()), cls(masks, **source._meta())


@NestedTensorAtenRegistry.implement(aten._to_copy.default)
def to_copy(func, args, kwargs):
    r"""
    Copy _values to a new dtype/device while preserving metadata tensors.

    Note: memory_format is applied to the packed _values buffer, not per-element.
    For non-contiguous formats like channels_last, the result may not have
    meaningful per-element layout since _values is a concatenation of
    variable-length elements.
    """
    source = args[0]
    # Offsets and _physical_shape stay on CPU — they are metadata, not compute tensors.
    return source.packed_like(func(source._values, **kwargs))


# ---------------------------------------------------------------------------
# Tensor creation ops — preserve packing layout with new _values
# See also torch_functions.py for torch-level empty_like/zeros_like/ones_like/full_like.
# ---------------------------------------------------------------------------


ATEN_CREATION_OPS = [
    aten.empty_like.default,
    aten.zeros_like.default,
    aten.ones_like.default,
    aten.full_like.default,
]


def _creation_like_handler(func, args, kwargs):
    r"""Create packed values while deriving layout and pinning from the actual result."""
    source = args[0]
    return source.packed_like(func(source._values, *args[1:], **kwargs))


# ---------------------------------------------------------------------------
# Ternary ops — where, addcmul, addcdiv, lerp.Tensor (3 NT tensor args)
# ---------------------------------------------------------------------------


def _ternary_per_element(func, args, kwargs, ref):
    r"""
    Replay a ternary op per element, giving each dense operand the slice that element meets.

    ``per_element_fallback`` substitutes elements for NestedTensors and passes every other
    argument through untouched, which for a dense operand carrying a batch axis hands the whole
    batch to every element and inflates the result's rank. Dense operands are aligned here the
    same way the packed resolver reads them, so the fallback answers what the packed path would.
    """
    from .nested_tensor import NestedTensor

    _check_execution_guard(_ExecutionGuardKind.EAGER_FALLBACK, "aten_functions._ternary_per_element")
    if _is_compiling():
        _compile_unsupported(str(getattr(func, "_schema", func)), "would fall back to per-element eager execution")
    operands = [operand._unpack() if isinstance(operand, NestedTensor) else operand for operand in args[:3]]
    results = []
    for index, anchor in enumerate(ref._unpack()):
        elements = []
        for operand in operands:
            if isinstance(operand, tuple):
                elements.append(operand[index])
            elif isinstance(operand, Tensor):
                # Nothing aligns: hand the operand over untouched so the dense kernel reports
                # the mismatch in its own words rather than inventing a reading for it.
                aligned = _dense_operand_for_element(ref, operand, index, anchor)
                elements.append(operand if aligned is None else aligned)
            else:
                elements.append(operand)
        results.append(func(*elements, *args[3:], **kwargs))
    return _nested_like_elements(ref, results)


def _ternary_handler(func, args, kwargs):
    r"""Dispatch handler for ternary ops (where, addcmul, etc.) on packed _values."""
    a, b, c = args[0], args[1], args[2]
    from .nested_tensor import NestedTensor

    sources = [x for x in (a, b, c) if isinstance(x, NestedTensor)]
    if not sources:
        return func(*args, **kwargs)
    first_ref = sources[0]
    for other in sources[1:]:
        if len(other) != len(first_ref):
            raise ValueError(
                f"NestedTensor batch length mismatch for {func}: expected {len(first_ref)}, got {len(other)}"
            )

    resolved = None
    ref = first_ref
    for candidate in sources:
        candidate_values = tuple(_resolve_ternary_other(candidate, operand) for operand in (a, b, c))
        if all(value is not _UNRESOLVED for value in candidate_values):
            ref = candidate
            resolved = candidate_values
            break

    if resolved is None:
        # Preserve dense parity when a packed fast path cannot prove per-element
        # broadcasting semantics for every plain Tensor operand.
        if len(ref) == 0:
            return per_element_fallback(func, args, kwargs)
        return _ternary_per_element(func, args, kwargs, ref)
    va, vb, vc = resolved
    out_values = func(va, vb, vc, **kwargs)
    if tuple(out_values.shape[1:]) == tuple(ref._values.shape[1:]):
        return ref._packed_like_unchecked(out_values)
    if out_values.dim() == ref._values.dim():
        return _packed_with_static_tail_from_values(ref, out_values)
    return _packed_with_tail_from_values(ref, out_values)


ATEN_TERNARY_OPS = [
    aten.where.self,
    aten.where.ScalarOther,
    aten.where.ScalarSelf,
    aten.where.Scalar,
    aten.addcmul.default,
    aten.addcdiv.default,
    aten.lerp.Tensor,
]


def _random_creation_handler(func, args, kwargs):
    r"""Create random tensors per element to preserve RNG parity with per-element eager calls."""
    source = args[0]
    cls = type(source)
    if len(source) == 0:
        return cls([], **source._meta())
    return cls([func(t, *args[1:], **kwargs) for t in source._storage], **source._meta())


def _make_dim_reduction_handler(ragged_fill, default_dim, keepdim_kw):
    r"""Factory for table-driven dim reduction handlers."""
    none_dim = [] if default_dim == () else None

    def _handler(func, args, kwargs):
        source, dims, keepdim = _extract_dim_keepdim(args, kwargs, default_dim)
        fill = ragged_fill
        if fill is ...:
            fill = _topk_fill_value(source._values.dtype, largest=(func is aten.amax.default))
        return _dim_reduction_dispatch(
            func, source, dims, keepdim, kwargs, ragged_fill=fill, keepdim_kw=keepdim_kw, none_dim=none_dim
        )

    return _handler


def _make_order_stat_handler(default_dim):
    r"""Factory for table-driven order-stat pair reduction handlers."""

    def _handler(func, args, kwargs):
        return _order_stat_dim_handler(func, args, kwargs, default_dim)

    return _handler


# ---------------------------------------------------------------------------
# Bulk registration — all table-driven op → handler mappings.
# @NestedTensorAtenRegistry.implement(...) decorators above handle 1:1 ops.
# ---------------------------------------------------------------------------

_ATEN_HANDLER_TABLE: list[tuple] = [
    # _binary_unwrap_handler — softmax/log_softmax backward
    *(
        (op, _binary_unwrap_handler)
        for op in [aten._softmax_backward_data.default, aten._log_softmax_backward_data.default]
    ),
    # _dim_reduction_handler — parameterised dim reductions
    #   args: (ragged_fill, default_dim, keepdim_kw)
    #   ragged_fill=None → per_element_fallback for ragged dim-0
    #   ragged_fill=... → dtype-dependent (amax/amin)
    (aten.amax.default, _make_dim_reduction_handler(..., (), False)),
    (aten.amin.default, _make_dim_reduction_handler(..., (), False)),
    (aten.logsumexp.default, _make_dim_reduction_handler(float("-inf"), (), False)),
    (aten.mean.dim, _make_dim_reduction_handler(None, None, False)),
    (aten.nanmean.default, _make_dim_reduction_handler(None, None, False)),
    (aten.nansum.default, _make_dim_reduction_handler(0, None, False)),
    (aten.std.correction, _make_dim_reduction_handler(None, None, True)),
    (aten.sum.dim_IntList, _make_dim_reduction_handler(0, None, False)),
    (aten.var.correction, _make_dim_reduction_handler(None, None, True)),
    # _elementwise_binary_handler
    *((op, _elementwise_binary_handler) for op in ATEN_BINARY_ELEMENTWISE_OPS),
    # _elementwise_unary_handler
    *((op, _elementwise_unary_handler) for op in ATEN_UNARY_ELEMENTWISE_OPS),
    *((op, _creation_like_handler) for op in ATEN_CREATION_OPS),
    *((op, _elementwise_unary_handler) for op in ATEN_UNARY_LIKE_OPS),
    *(
        (op, _elementwise_unary_handler)
        for op in [aten.bucketize.Tensor, aten.isin.Tensor_Scalar, aten.isin.Tensor_Tensor, aten.rms_norm.default]
    ),
    # _global_reduction_handler
    *((op, _global_reduction_handler) for op in ATEN_GLOBAL_REDUCTION_OPS),
    # _inplace_binary_handler
    *((op, _inplace_binary_handler) for op in ATEN_INPLACE_BINARY_OPS),
    # _inplace_unary_handler
    *((op, _inplace_unary_handler) for op in ATEN_INPLACE_RNG_OPS + ATEN_INPLACE_UNARY_OPS),
    # _masked_fill_handler
    *((op, _masked_fill_handler) for op in [aten.masked_fill.Scalar, aten.masked_fill.Tensor]),
    # _masked_scatter_handler
    (aten.masked_scatter.default, _masked_scatter_handler),
    # _order_stat_handler — parameterised order-stat pair reductions
    (aten.median.dim, _make_order_stat_handler(_MISSING)),
    (aten.mode.default, _make_order_stat_handler(-1)),
    (aten.nanmedian.dim, _make_order_stat_handler(_MISSING)),
    # _random_creation_handler — per-element for RNG parity
    *((op, _random_creation_handler) for op in ATEN_RANDOM_CREATION_OPS),
    # _softmax_handler
    *((op, _softmax_handler) for op in [aten._log_softmax.default, aten._softmax.default]),
    # _ternary_handler
    *((op, _ternary_handler) for op in ATEN_TERNARY_OPS),
]

for _op, _handler in _ATEN_HANDLER_TABLE:
    NestedTensorAtenRegistry.register(
        _op,
        _handler,
        compile_safe=_op not in {aten.roll.default, aten.rot90.default, *ATEN_RANDOM_CREATION_OPS},
    )
