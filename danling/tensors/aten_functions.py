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

import math
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from .ops import (
    _MISSING,
    ATEN_BINARY_ELEMENTWISE_OPS,
    ATEN_UNARY_ELEMENTWISE_OPS,
    NestedTensorAtenRegistry,
    _check_execution_guard,
    _compile_unsupported,
    _ExecutionGuardKind,
    _get_batch_dim,
    _is_compiling,
    _is_packed_identity,
    _maybe_align_dense_to_nested,
    _normalize_dim,
    _resolve_dense_for_values,
    _stack_or_nest,
    _translate_dim,
    _translate_dims,
)

if TYPE_CHECKING:
    from .nested_tensor import NestedTensor

aten = torch.ops.aten

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
    _check_execution_guard(_ExecutionGuardKind.STORAGE_MAP, "_apply_per_element_nested")
    cls = type(source)
    if len(source) == 0:
        return cls([], **source._meta(include_dtype=True))
    return cls((op(t) for t in source._unpack()), **source._meta())


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
        shape = source._physical_shape[idx].tolist()
        while shape and shape[-1] == 0:
            shape.pop()
        try:
            torch.broadcast_shapes(tuple(shape), candidate_shape)
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
        if source._has_same_structure(other):
            values = other._values
            if values.device != device:
                values = values.to(device=device)
            return values
        raise NotImplementedError(f"NestedTensor: {func} requires matching packed structure across all NT operands")
    if isinstance(other, Tensor):
        if other.dim() == 0:
            return other if other.device == device else other.to(device=device)
        aligned = _maybe_align_dense_to_nested(source, other)
        if aligned is not None and source._has_same_structure(aligned):
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


def _elementwise_binary_handler(func, args, kwargs):
    r"""Dispatch handler for elementwise binary ops on packed _values."""
    from .nested_tensor import NestedTensor

    lhs, rhs = args[0], args[1]
    extra = args[2:]
    if isinstance(lhs, NestedTensor):
        try:
            resolved = _resolve_other(lhs, rhs, func)
        except NotImplementedError:
            return per_element_fallback(func, args, kwargs)
        return _packed_like(lhs, func(lhs._values, resolved, *extra, **kwargs))
    # lhs is scalar/tensor, rhs is NestedTensor
    try:
        resolved = _resolve_other(rhs, lhs, func)
    except NotImplementedError:
        return per_element_fallback(func, args, kwargs)
    return _packed_like(rhs, func(resolved, rhs._values, *extra, **kwargs))


# ---------------------------------------------------------------------------
# Elementwise unary ops — apply directly to _values
# ---------------------------------------------------------------------------


def _elementwise_unary_handler(func, args, kwargs):
    r"""Dispatch handler for elementwise unary ops applied to _values."""
    source = args[0]
    return _packed_like(source, func(source._values, *args[1:], **kwargs))


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
        ragged_fill: Fill value for padded ragged-dim-0 path, or ``None`` for per-element fallback.
        keepdim_kw: If True, pass keepdim as keyword arg (for std/var correction schema).
        none_dim: What to pass as the dim argument for "reduce all elements" calls.
            ``None`` for ops like sum/mean, ``[]`` for ops like amax/amin.

    Branch structure:
    1. ``len(dims) == 0`` → global reduction on packed ``_values``
    2. ``len(dims) > 1`` → multi-dim: packed fast path on static dims, padded or fallback for ragged
    3. ``dim == batch_dim`` → stack per-element reductions
    4. ``dim_adj == 0`` (ragged) → padded with ``ragged_fill`` or fallback if ``None``
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
        if not _is_packed_identity(source):
            return _fallback(list(dims_adj), keepdim)
        if source._values.dim() > 1 and all(dim_i > 0 for dim_i in dims_adj):
            return _reduce_non_ragged_packed_dims(
                source, _call(source._values, list(dims_adj), keepdim), dims_adj, keepdim
            )
        if ragged_fill is not None and 0 in dims_adj:
            padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=ragged_fill)
            # Padded has shape [B, max_len, ...], so element dim d maps to padded dim d+1
            padded_dims = [dim_i + 1 for dim_i in dims_adj]
            return _call(padded, padded_dims, keepdim)
        return _fallback(list(dims_adj), keepdim)

    dim = _normalize_dim(dims[0], source.dim())
    batch_dim = _get_batch_dim(source)
    if dim == batch_dim:
        reduced = torch.stack([_call(t, none_dim, False) for t in source._storage])
        if keepdim:
            return reduced.unsqueeze(batch_dim)
        return reduced

    dim_adj = _translate_dim(source, dim)
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
        return _call(padded, [1], keepdim)

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
    if dim_adj == 0:
        fill_value = _topk_fill_value(source._values.dtype, largest=largest)
        padded, _, _, _, _, _ = _packed_to_padded(source, fill_value=fill_value)
        return func(padded, 1, keepdim, **kwargs)

    out_values = func(source._values, dim_adj, keepdim, **kwargs)
    return _reduce_non_ragged_packed(source, out_values, dim_adj, keepdim)


@NestedTensorAtenRegistry.implement(aten.count_nonzero.dim_IntList)
def count_nonzero_dim_reduction(func, args, kwargs):
    r"""Handle ``count_nonzero`` dim reductions on packed values for common dim patterns."""
    source, dims, _ = _extract_dim_keepdim(args, kwargs, ())
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


@NestedTensorAtenRegistry.implement(aten.var_mean.correction)
def var_mean_dim_reduction(func, args, kwargs):
    r"""Handle ``var_mean`` correction reductions via packed fastpaths where valid."""
    source, dims, keepdim = _extract_dim_keepdim(args, kwargs, None)
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
    r"""Dispatch handler for masked_fill: packed fast path + per-element broadcast fallback."""
    from .nested_tensor import NestedTensor

    source, mask, value = args[0], args[1], args[2]
    if isinstance(mask, NestedTensor) and source._has_same_layout(mask):
        return _packed_like(source, func(source._values, mask._values, value, **kwargs))
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
    if not input_tensor._has_same_layout(mask):
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

        if dim_adj > 0 and source._values.dim() > 1 and source._has_same_structure(index):
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
                packed_sizes=index._packed_sizes,
                element_shapes=index._element_shapes,
                validate=False,
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
            batch_first=source.batch_first,
            padding_value=source.padding_value,
            mask_value=source.mask_value,
            pin_memory=source._pin_memory,
            packed_sizes=selected_packed_sizes,
            element_shapes=selected_element_shapes,
            validate=False,
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
            f"{op_name}: NestedTensor batch length mismatch between input and source: input={len(source)}, "
            f"source={len(src)}"
        )

    if isinstance(index, NestedTensor) and dim_adj > 0 and source._values.dim() > dim_adj:
        # Writes along the ragged leading dim are not packed-safe: a padded fallback would let
        # invalid rows participate in the write. Restrict the fast path to static per-element dims.
        index_values = index._values.to(device=source._values.device, dtype=torch.long)
        if source._has_same_structure(index):
            if isinstance(src, NestedTensor):
                if source._has_same_structure(src):
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


@NestedTensorAtenRegistry.implement(aten.scatter_add.default)
def scatter_add(func, args, kwargs):
    r"""Apply ``scatter_add`` with packed fast paths when index/source layouts align."""
    source, dim, index, src = args[0], args[1], args[2], args[3]
    return _scatter_like(source, dim, index, src, lambda t, d, i, s: func(t, d, i, s, **kwargs), "scatter_add")


@NestedTensorAtenRegistry.implement(aten.scatter.src)
def scatter_src(func, args, kwargs):
    r"""Apply ``scatter`` with Tensor/NestedTensor src via packed fast paths when layouts align."""
    source, dim, index, src = args[0], args[1], args[2], args[3]
    return _scatter_like(source, dim, index, src, lambda t, d, i, s: func(t, d, i, s, **kwargs), "scatter")


@NestedTensorAtenRegistry.implement(aten.scatter.value)
def scatter_value(func, args, kwargs):
    r"""Apply scalar ``scatter`` with packed fast paths when index layouts align."""
    source, dim, index, value = args[0], args[1], args[2], args[3]
    return _scatter_like(source, dim, index, value, lambda t, d, i, s: func(t, d, i, s, **kwargs), "scatter")


if hasattr(aten, "scatter_reduce"):

    @NestedTensorAtenRegistry.implement(aten.scatter_reduce.two)
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
    new_physical_shape = source._physical_shape.clone()
    if new_physical_shape.size(1) == 0:
        new_physical_shape = new_physical_shape.new_full((len(source), 1), new_last_dim)
    else:
        new_physical_shape[:, -1] = new_last_dim

    packed_sizes = None
    element_shapes = None
    if source._element_shapes is not None and isinstance(new_last_dim, int):
        element_shapes = tuple(
            (*shape[:-1], int(new_last_dim)) if shape else (int(new_last_dim),) for shape in source._element_shapes
        )
        packed_sizes = source._packed_sizes_like(element_shapes)

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
    )


def _packed_like(source: NestedTensor, new_values: Tensor) -> NestedTensor:
    r"""Rebuild a NestedTensor from source metadata and a new packed value tensor."""
    result = _packed_with_shape(
        source,
        new_values,
        source._physical_shape,
        source._logical_shape,
        permutation=source._permutation,
        packed_sizes=source._packed_sizes,
        element_shapes=source._element_shapes,
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
) -> NestedTensor:
    r"""Rebuild a NestedTensor with explicit ``_physical_shape`` and logical shape."""
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
        validate=False,
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


def _offsets_from_packed_sizes(source: NestedTensor, sizes: tuple[int, ...]) -> Tensor:
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + int(size))
    return source._offsets.new_tensor(offsets)


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
    return _packed_like(rhs, values)


def _packed_jagged_matmul(lhs: NestedTensor, rhs: NestedTensor) -> NestedTensor | None:
    output = _packed_jagged_outer_matmul(lhs, rhs)
    if output is not None:
        return output
    return _packed_jagged_contract_matmul(lhs, rhs)


def _packed_square_softmax(source: NestedTensor, dim_adj: int, *, log: bool) -> NestedTensor | None:
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
    values = source._values
    tail = tuple(values.shape[1:])
    segment = query.reshape((-1, *([1] * len(tail)))).expand((-1, *tail))
    max_values = values.new_full((total, *tail), float("-inf"))
    max_values = max_values.scatter_reduce(0, segment, values, "amax", include_self=False)
    shifted = values - max_values.index_select(0, query)
    exp_values = torch.exp(shifted)
    sums = values.new_zeros((total, *tail)).index_add(0, query, exp_values)
    out_values = shifted - torch.log(sums.index_select(0, query)) if log else exp_values / sums.index_select(0, query)
    return _packed_like(source, out_values)


def _matmul_has_packed_path(lhs, rhs) -> bool:
    from .nested_tensor import NestedTensor

    if isinstance(lhs, NestedTensor):
        if isinstance(rhs, NestedTensor):
            return (lhs._has_same_structure(rhs) and lhs._values.dim() > 2 and rhs._values.dim() > 2) or (
                _packed_jagged_matmul_kind(lhs, rhs) is not None
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
        batched_values.shape,
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


def _needs_masked_topk_scores(dtype: torch.dtype) -> bool:
    return (not dtype.is_floating_point) and (not dtype.is_complex)


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
        _packed_like(source, eigvecs_values),
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
    return _packed_like(source, func(source._values, *args[1:], **kwargs))


@NestedTensorAtenRegistry.implement(aten.matrix_power.default)
@NestedTensorAtenRegistry.implement(aten.linalg_matrix_power.default)
def matrix_power(func, args, kwargs):
    r"""Apply matrix power on packed values when the ragged leading dim stays element-local."""
    source = args[0]
    if source._values.dim() <= 2:
        return _apply_per_element_nested(source, lambda t: func(t, *args[1:], **kwargs))
    return _packed_like(source, func(source._values, *args[1:], **kwargs))


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
    return _packed_like(source, output), mean, rstd


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
    return _packed_like(ref, grad_input), grad_weight, grad_bias


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
    if start_adj == 0:
        return per_element_fallback(func, (source, start_adj, end_adj), kwargs)

    if source._ragged_rank > 1:
        return per_element_fallback(func, (source, start_adj, end_adj), kwargs)

    out_values = func(source._values, start_adj, end_adj, **kwargs)
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
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_physical_dims(physical_dims),
        packed_sizes=out_packed_sizes,
        element_shapes=out_element_shapes,
    )


def _packed_metadata_permute(source: NestedTensor, tensor_dims: tuple[int, ...]) -> NestedTensor | None:
    r"""Relabel a logical per-element permutation when packed storage order is unchanged."""
    rank = int(source._physical_shape.size(1))
    if len(tensor_dims) != rank:
        return None
    old_packed_order = source._permutation
    if not old_packed_order:
        old_varying, old_static = type(source)._pack_layout_meta(source._physical_shape, source._element_shapes)
        old_packed_order = old_varying + old_static

    out_shape = source._physical_shape[:, tensor_dims]
    out_element_shapes = None
    if source._element_shapes is not None:
        out_element_shapes = tuple(tuple(shape[dim] for dim in tensor_dims) for shape in source._element_shapes)

    new_varying, new_static = type(source)._pack_layout_meta(out_shape, out_element_shapes)
    new_packed_order = new_varying + new_static
    new_packed_order_in_old_dims = tuple(tensor_dims[dim] for dim in new_packed_order)
    if new_packed_order_in_old_dims != old_packed_order:
        return None

    out_logical = source._logical_shape_from_physical_dims(
        tuple(source._max_physical_dims()[dim] for dim in tensor_dims)
    )
    return _packed_with_shape(
        source,
        source._values,
        out_shape,
        out_logical,
        permutation=new_packed_order,
        packed_sizes=source._packed_sizes,
        element_shapes=out_element_shapes,
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
    if source._element_shapes is not None:
        can_squeeze = all(dim_adj < len(shape) and int(shape[dim_adj]) == 1 for shape in source._element_shapes)
    else:
        if _is_compiling():
            _compile_unsupported("aten.squeeze.dim", "requires static element shape metadata")
        can_squeeze = bool(source._physical_shape[:, dim_adj].eq(1).all())
    if not can_squeeze:
        return source
    if dim_adj not in source._static_dims:
        # Squeezing ragged dims is per-element because packed values collapse them.
        return _apply_per_element_nested(source, lambda t: t.squeeze(dim_adj))

    values_dim = 1 + source._static_dims.index(dim_adj)
    out_values = func(source._values, values_dim, **kwargs)

    out_shape = torch.cat(
        (source._physical_shape[:, :dim_adj], source._physical_shape[:, dim_adj + 1 :]),
        dim=1,
    )
    physical_dims = list(source._max_physical_dims())
    del physical_dims[dim_adj]
    permutation = tuple(dim if dim < dim_adj else dim - 1 for dim in source._permutation if dim != dim_adj)
    out_packed_sizes = None
    out_element_shapes = None
    if source._element_shapes is not None:
        out_element_shapes = tuple(shape[:dim_adj] + shape[dim_adj + 1 :] for shape in source._element_shapes)
        out_packed_sizes = source._packed_sizes_like(out_element_shapes)
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_physical_dims(physical_dims),
        permutation=permutation,
        packed_sizes=out_packed_sizes,
        element_shapes=out_element_shapes,
    )


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
    return _packed_with_shape(
        source,
        out_values,
        out_shape,
        source._logical_shape_from_physical_dims(physical_dims),
        packed_sizes=out_packed_sizes,
        element_shapes=out_element_shapes,
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
    ones = torch.ones(
        (source._physical_shape.size(0), 1),
        dtype=source._physical_shape.dtype,
        device=source._physical_shape.device,
    )
    out_shape = torch.cat(
        (source._physical_shape[:, :dim_adj], ones, source._physical_shape[:, dim_adj:]),
        dim=1,
    )
    physical_dims = list(source._max_physical_dims())
    physical_dims.insert(dim_adj, 1)
    shifted_varying = tuple(dim + 1 if dim >= dim_adj else dim for dim in source._varying_dims)
    shifted_static = tuple(dim + 1 if dim >= dim_adj else dim for dim in source._static_dims)
    new_static = tuple(sorted((*shifted_static, dim_adj)))
    packed_dim = 1 + new_static.index(dim_adj)
    out_values = func(source._values, packed_dim, **kwargs)
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
    return _packed_like(ref, func(va, vb, *args[2:], **kwargs))


def _softmax_handler(func, args, kwargs):
    r"""Dispatch handler for softmax/log_softmax that translates the dim argument."""
    source = args[0]
    dim_adj = _translate_dim(source, args[1])
    if dim_adj in source._varying_dims:
        square = _packed_square_softmax(source, dim_adj, log=func is aten._log_softmax.default)
        if square is not None:
            return square
        if dim_adj == 0 and _is_packed_identity(source):
            padded, _, _, batch_idx, local_idx, _ = _packed_to_padded(source, fill_value=float("-inf"))
            out_padded = func(padded, 1, *args[2:], **kwargs)
            return _packed_like(source, out_padded[batch_idx, local_idx])
        return _apply_per_element_nested(source, lambda t: func(t, dim_adj, *args[2:], **kwargs))
    if dim_adj >= source._values.dim() or not _is_packed_identity(source):
        return _apply_per_element_nested(source, lambda t: func(t, dim_adj, *args[2:], **kwargs))
    return _packed_like(source, func(source._values, dim_adj, *args[2:], **kwargs))


# ---------------------------------------------------------------------------
# Sorting / cumulative / reordering ops.
# Fast path: N-D packing with non-ragged target dims (operate on _values).
# Fallback: 1-D packing or ragged dim -> per-element path.
# ---------------------------------------------------------------------------


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
    return source._values.dim() > 1 and dim_adj > 0


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
    return source._values.dim() > 1 and dim_adj > 0


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
    return dim_adj == 0 or (source._values.dim() > 1 and dim_adj > 0)


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
    return source._values.dim() > 1 and all(dim > 0 for dim in dims_adj)


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
    except (TypeError, ValueError, IndexError):
        return False
    return source._values.dim() > 1 and dim_adj > 0


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
    if source._values.dim() > 1 and dim_adj > 0:
        return _packed_like(source, _call_argsort(source._values, dim_adj))
    if dim_adj == 0:
        if _is_compiling():
            _compile_unsupported("aten.argsort.default", "ragged-dimension argsort is eager-only under compile")
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


@NestedTensorAtenRegistry.implement(
    aten.cumsum.default,
    compile_safe=True,
    compile_guard=_cumsum_compile_safe,
)
@NestedTensorAtenRegistry.implement(
    aten.cumprod.default,
    compile_safe=True,
    compile_guard=_cumulative_compile_safe,
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
    if source._values.dim() > 1 and dim_adj > 0:
        return _packed_like(source, func(source._values, dim_adj, *extra_args, **kwargs))
    if dim_adj == 0:
        if func is aten.cumsum.default:
            return _packed_like(source, _segmented_cumsum_values(source, *extra_args, **kwargs))
        if _is_compiling():
            _compile_unsupported(
                f"{func._schema.name.split('::')[-1]}",
                "ragged-dimension cumulative ops are eager-only under compile",
            )
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


@NestedTensorAtenRegistry.implement(aten.dropout.default, compile_safe=True)
def dropout(func, args, kwargs):
    r"""Apply aten dropout on packed values, preserving eval-mode identity."""
    source = args[0]
    p = args[1] if len(args) > 1 else kwargs.get("p", 0.5)
    train = args[2] if len(args) > 2 else kwargs.get("train", True)
    if (not bool(train)) or float(p) == 0:
        return source
    return _packed_like(source, func(source._values, *args[1:], **kwargs))


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
    if source._values.dim() > 1 and dim_adj > 0:
        vals, idxs = func(source._values, dim_adj, **kwargs)
        return _packed_like(source, vals), _packed_like(source, idxs)
    if dim_adj == 0:
        if _is_compiling():
            _compile_unsupported(
                f"{func._schema.name.split('::')[-1]}",
                "ragged-dimension cumulative ops are eager-only under compile",
            )
        largest = func is aten.cummax.default
        fill_value = _topk_fill_value(source._values.dtype, largest=largest)
        padded, _, _, batch_idx, local_idx, _ = _packed_to_padded(source, fill_value=fill_value)
        vals, idxs = func(padded, 1, **kwargs)
        return _packed_like(source, vals[batch_idx, local_idx]), _packed_like(source, idxs[batch_idx, local_idx])
    return per_element_fallback(func, (source, dim_adj), kwargs)


@NestedTensorAtenRegistry.implement(
    aten.flip.default,
    compile_safe=True,
    compile_guard=_flip_compile_safe,
)
def flip(func, args, kwargs):
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
    if not _is_packed_identity(source):
        return per_element_fallback(func, (source, dims_adj), kwargs)
    ragged_rank = source._ragged_rank
    if dims_adj and all(dim >= ragged_rank for dim in dims_adj):
        packed_dims = tuple(dim - ragged_rank + 1 for dim in dims_adj)
        return _packed_like(source, func(source._values, packed_dims, **kwargs))
    if ragged_rank == 1 and any(dim == 0 for dim in dims_adj):
        if _is_compiling():
            _compile_unsupported("aten.flip.default", "ragged-dimension flip is eager-only under compile")
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
        return _packed_like(source, func(source._values, shifts, list(dims_adj), **kwargs))
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
            return _packed_like(source, out_values)
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
    if source._values.dim() > 1 and dim_adj > 0:
        vals, idxs = _call_sort(source._values, dim_adj)
        return _packed_like(source, vals), _packed_like(source, idxs)
    if dim_adj == 0:
        if _is_compiling():
            _compile_unsupported("aten.sort.default", "ragged-dimension sort is eager-only under compile")
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
    if source._values.dim() > 1 and dim_adj > 0:
        vals, idxs = func(source._values, k, dim_adj, largest, sorted_output, **kwargs)
        return (
            _packed_new_dim_size(source, vals, dim_adj, k),
            _packed_new_dim_size(source, idxs, dim_adj, k),
        )
    if dim_adj == 0:
        if _is_compiling():
            _compile_unsupported("aten.topk.default", "ragged-dimension topk is eager-only under compile")
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


# ---------------------------------------------------------------------------
# Structured / shape-changing ops
# ---------------------------------------------------------------------------


@NestedTensorAtenRegistry.implement(aten.alias.default)
def alias(func, args, kwargs):
    r"""Create an alias of the NestedTensor sharing the same _values storage."""
    source = args[0]
    return _packed_like(source, source._values.alias())


@NestedTensorAtenRegistry.implement(aten.clone.default)
def clone(func, args, kwargs):
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
        packed_sizes=source._packed_sizes,
        element_shapes=source._element_shapes,
        permutation=source._permutation,
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
    old_sizes = source._offsets[1:] - source._offsets[:-1]
    batch_steps = torch.arange(len(source) + 1, dtype=source._offsets.dtype, device=source._offsets.device)
    new_offsets = source._offsets + batch_steps * pad_width

    old_total = source._values.size(0)
    new_total = old_total + len(source) * pad_width
    output_values = source._values.new_full((new_total, *source._values.shape[1:]), value)
    old_sizes_device = old_sizes.to(device=source._values.device)
    batch_indices = torch.arange(len(source), device=source._values.device).repeat_interleave(
        old_sizes_device,
        output_size=old_total,
    )
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
        packed_sizes = tuple(int(size) for size in (new_offsets[1:] - new_offsets[:-1]).tolist())

    outer_size = list(source._logical_shape)
    outer_size[-1] += pad_width
    return type(source)._from_packed(
        output_values,
        new_offsets,
        shape_tensor,
        permutation=source._permutation,
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
    return _packed_like(source, func(source._values, **kwargs))


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
    *((op, _elementwise_unary_handler) for op in ATEN_CREATION_OPS + ATEN_UNARY_LIKE_OPS),
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
