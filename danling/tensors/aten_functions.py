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
operating on the packed representation (_values, _offsets, _shape_tensor).

Architecture:
    - Elementwise ops operate directly on ``_values`` (no unpack/repack overhead)
    - Structural ops (clone, detach, to_copy) operate on all inner tensors
    - Unregistered ops fall back to per-element application via ``_storage``
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .ops import ATEN_BINARY_ELEMENTWISE_OPS, ATEN_UNARY_ELEMENTWISE_OPS, NestedTensorAtenRegistry, _translate_dim

if TYPE_CHECKING:
    from .nested_tensor import NestedTensor

aten = torch.ops.aten


def _offsets_match(a: Tensor, b: Tensor) -> bool:
    r"""
    Check if two offset tensors represent the same packing layout.

    Uses pointer identity as a fast-path before falling back to elementwise comparison.
    """
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


def try_packed_same_shape_fallback(func, args, kwargs):
    r"""
    Try a generic packed fallback for unregistered aten ops.

    This path is intentionally conservative:
    - all NestedTensor operands must share offsets
    - output tensor shapes must match packed ``_values`` exactly
    """
    from .nested_tensor import NestedTensor

    nested_args: list[NestedTensor] = []

    def collect_nested(obj):
        if isinstance(obj, NestedTensor):
            nested_args.append(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                collect_nested(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                collect_nested(item)

    collect_nested(args)
    if kwargs:
        collect_nested(kwargs)
    if not nested_args:
        return False, None

    ref = nested_args[0]
    if any(not _offsets_match(ref._offsets, other._offsets) for other in nested_args[1:]):
        return False, None

    def replace_nested_with_values(obj):
        if isinstance(obj, NestedTensor):
            return obj._values
        if isinstance(obj, tuple):
            return tuple(replace_nested_with_values(item) for item in obj)
        if isinstance(obj, list):
            return [replace_nested_with_values(item) for item in obj]
        if isinstance(obj, dict):
            return {k: replace_nested_with_values(v) for k, v in obj.items()}
        return obj

    packed_args = replace_nested_with_values(args)
    packed_kwargs = replace_nested_with_values(kwargs) if kwargs else {}

    try:
        packed_result = func(*packed_args, **packed_kwargs)
    except (TypeError, RuntimeError, ValueError, NotImplementedError):
        return False, None

    if isinstance(packed_result, Tensor):
        if packed_result.shape == ref._values.shape:
            return True, _packed_like(ref, packed_result)
        return False, None

    if isinstance(packed_result, tuple):
        wrapped = []
        for item in packed_result:
            if isinstance(item, Tensor) and item.shape == ref._values.shape:
                wrapped.append(_packed_like(ref, item))
            else:
                return False, None
        return True, tuple(wrapped)

    return False, None


@torch._dynamo.disable
def per_element_fallback(func, args, kwargs):
    r"""
    Fallback for unregistered ops: unpack to individual tensors, apply op, repack.

    Only used by ``__torch_dispatch__`` as a catch-all for ops without a registered handler.
    Registered handlers should raise ``NotImplementedError`` instead of calling this.

    Returns:
        If all per-element results are ``Tensor`` objects with the **same shape**,
        returns a plain ``torch.Tensor`` (via ``torch.stack``) rather than a
        ``NestedTensor``. This is intentional: uniform-shape outputs are regular
        tensors by definition and stacking them avoids unnecessary overhead.
        Callers that always require a ``NestedTensor`` result should check
        ``isinstance(result, NestedTensor)`` and wrap if needed.

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

    batch_size = len(source)
    if batch_size == 0:

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
        if all(r.shape == results[0].shape for r in results[1:]):
            return torch.stack(results)
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
                if all(e.shape == elements[0].shape for e in elements[1:]):
                    unpacked.append(torch.stack(elements))
                else:
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
    - NestedTensor with matching offsets → ``other._values``
    - Scalar or 0-dim tensor → ``other`` (broadcast-compatible with packed _values)
    - Anything else → raises ``NotImplementedError``
    """
    from .nested_tensor import NestedTensor

    if isinstance(other, NestedTensor):
        if _offsets_match(source._offsets, other._offsets):
            return other._values
        raise NotImplementedError(f"NestedTensor: {func} with mismatched packing layouts")
    if isinstance(other, Tensor) and other.dim() > 0:
        raise NotImplementedError(f"NestedTensor: {func} with non-scalar Tensor operand; convert to NestedTensor first")
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
    out_shape_tensor = source._shape_tensor.clone()
    for i in range(padded_dims):
        out_shape_tensor[:, -(i + 1)] += pad[2 * i] + pad[2 * i + 1]

    out_outer_size = list(source._logical_shape)
    for i in range(padded_dims):
        out_outer_size[-(i + 1)] = out_outer_size[-(i + 1)] + pad[2 * i] + pad[2 * i + 1]

    return type(source)._from_packed(
        out_values,
        source._offsets,
        out_shape_tensor,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=tuple(out_outer_size),
    )


@NestedTensorAtenRegistry.implement(aten.clone.default)
def _clone(func, args, kwargs):
    r"""Clone all internal tensors of a NestedTensor."""
    source = args[0]
    return type(source)._from_packed(
        source._values.clone(),
        source._offsets.clone(),
        source._shape_tensor.clone(),
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
        source._shape_tensor.detach(),
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
# Matrix multiply ops — apply to _values, update last dim of shape_tensor
# ---------------------------------------------------------------------------


def _packed_new_last_dim(source: NestedTensor, new_values: Tensor, new_last_dim: int) -> NestedTensor:
    r"""Rebuild a NestedTensor with a changed last dimension (e.g. after matmul)."""
    new_shape_tensor = source._shape_tensor.clone()
    new_shape_tensor[:, -1] = new_last_dim
    new_logical = list(source._logical_shape)
    new_logical[-1] = new_last_dim
    return type(source)._from_packed(
        new_values,
        source._offsets,
        new_shape_tensor,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=torch.Size(new_logical),
    )


def _packed_new_dim_size(source: NestedTensor, new_values: Tensor, dim_adj: int, new_dim_size: int) -> NestedTensor:
    r"""Rebuild a NestedTensor with a changed per-element dimension size."""
    new_shape_tensor = source._shape_tensor.clone()
    if new_shape_tensor.numel() > 0 and dim_adj < new_shape_tensor.size(1):
        new_shape_tensor[:, dim_adj] = new_dim_size
    new_logical = list(source._logical_shape)
    if new_logical:
        logical_dim = dim_adj + 1 if source.batch_first else (dim_adj if dim_adj == 0 else dim_adj + 1)
        if logical_dim < len(new_logical):
            new_logical[logical_dim] = new_dim_size
    return type(source)._from_packed(
        new_values,
        source._offsets,
        new_shape_tensor,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=torch.Size(new_logical),
    )


def _packed_like(source: NestedTensor, new_values: Tensor) -> NestedTensor:
    r"""Rebuild a NestedTensor from source metadata and a new packed value tensor."""
    return type(source)._from_packed(
        new_values,
        source._offsets,
        source._shape_tensor,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=source._logical_shape,
    )


def _packed_new_ragged_size(source: NestedTensor, new_values: Tensor, new_ragged_size) -> NestedTensor:
    r"""Rebuild a NestedTensor when per-element dim-0 size changes uniformly."""
    batch_size = source._offsets.size(0) - 1
    # Keep offsets on the same device as the source metadata (CPU by design).
    new_offsets = torch.arange(batch_size + 1, dtype=torch.long, device=source._offsets.device) * new_ragged_size
    new_shape_tensor = source._shape_tensor.clone()
    if new_shape_tensor.numel() > 0:
        new_shape_tensor[:, 0] = new_ragged_size
    new_logical = list(source._logical_shape)
    if source.batch_first:
        if len(new_logical) > 1:
            new_logical[1] = new_ragged_size
    elif len(new_logical) > 0:
        new_logical[0] = new_ragged_size
    return type(source)._from_packed(
        new_values,
        new_offsets,
        new_shape_tensor,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=torch.Size(new_logical),
    )


def _packed_to_padded(source: NestedTensor, *, fill_value) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
    r"""Convert packed values [sum(L_i), ...] into padded [B, max(L_i), ...] plus gather indices."""
    lengths = source._offsets[1:] - source._offsets[:-1]
    device = source._values.device
    lengths_dev = lengths.to(device=device, dtype=torch.long)
    with torch._C.DisableTorchFunctionSubclass():
        outer_size = torch.Tensor.size(source)
    ragged_dim = 1 if source.batch_first else 0
    batch_dim = 0 if source.batch_first else 1
    max_len = outer_size[ragged_dim] if len(outer_size) > ragged_dim else 0
    batch_size = outer_size[batch_dim] if len(outer_size) > batch_dim else 0
    padded_shape = (batch_size, max_len, *source._values.shape[1:])
    padded = torch.full(padded_shape, fill_value=fill_value, dtype=source._values.dtype, device=device)

    # Build packed->padded row indices without repeat_interleave so fullgraph
    # tracing is not blocked by data-dependent output-size ops.
    flat_idx = torch.arange(source._values.size(0), device=device, dtype=torch.long)
    offsets_dev = source._offsets.to(device=device, dtype=torch.long)
    batch_idx = torch.searchsorted(offsets_dev[1:], flat_idx, right=True)
    local_idx = flat_idx - offsets_dev[batch_idx]
    padded[batch_idx, local_idx] = source._values
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
    if (
        isinstance(mat1, NestedTensor)
        and isinstance(mat2, NestedTensor)
        and _offsets_match(mat1._offsets, mat2._offsets)
    ):
        new_values = func(mat1._values, mat2._values, **kwargs)
        return _packed_new_last_dim(mat1, new_values, mat2._values.shape[-1])
    raise NotImplementedError(f"NestedTensor: {func} requires two NTs with matching offsets")


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
    call_kwargs = dict(kwargs)
    dim = args[1] if len(args) > 1 else call_kwargs.pop("dim", -1)
    descending = args[2] if len(args) > 2 else call_kwargs.pop("descending", False)
    stable = call_kwargs.pop("stable", None)
    stable_overload = func is aten.sort.stable

    def _call_sort(tensor: Tensor, dim_value: int):
        if stable_overload or stable is not None:
            return torch.ops.aten.sort.stable(
                tensor, stable=stable, dim=dim_value, descending=descending, **call_kwargs
            )
        return func(tensor, dim_value, descending, **call_kwargs)

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
            {"stable": stable, "dim": dim_adj, "descending": descending, **call_kwargs},
        )
    return per_element_fallback(func, (source, dim_adj, descending), call_kwargs)


@NestedTensorAtenRegistry.implement(aten.argsort.default)
@NestedTensorAtenRegistry.implement(aten.argsort.stable)
def _argsort(func, args, kwargs):
    r"""Return sort indices along a non-ragged dim by operating on packed _values."""
    source = args[0]
    call_kwargs = dict(kwargs)
    dim = args[1] if len(args) > 1 else call_kwargs.pop("dim", -1)
    descending = args[2] if len(args) > 2 else call_kwargs.pop("descending", False)
    stable = call_kwargs.pop("stable", None)
    stable_overload = func is aten.argsort.stable

    def _call_argsort(tensor: Tensor, dim_value: int):
        if stable_overload or stable is not None:
            return torch.ops.aten.argsort.stable(
                tensor, stable=bool(stable), dim=dim_value, descending=descending, **call_kwargs
            )
        return func(tensor, dim_value, descending, **call_kwargs)

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
            {"stable": bool(stable), "dim": dim_adj, "descending": descending, **call_kwargs},
        )
    return per_element_fallback(func, (source, dim_adj, descending), call_kwargs)


@NestedTensorAtenRegistry.implement(aten.topk.default)
def _topk(func, args, kwargs):
    r"""Compute top-k along a non-ragged dim by operating on packed _values."""
    source = args[0]
    call_kwargs = dict(kwargs)
    k = args[1] if len(args) > 1 else call_kwargs.pop("k")
    dim = args[2] if len(args) > 2 else call_kwargs.pop("dim", -1)
    largest = args[3] if len(args) > 3 else call_kwargs.pop("largest", True)
    sorted_output = args[4] if len(args) > 4 else call_kwargs.pop("sorted", True)
    dim_adj = _translate_dim(source, dim)
    if source._values.dim() > 1 and dim_adj > 0:
        vals, idxs = func(source._values, k, dim_adj, largest, sorted_output, **call_kwargs)
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
        is_fake = False
        with suppress(ImportError):
            from torch._subclasses.fake_tensor import is_fake as _is_fake

            is_fake = _is_fake(source._values)
        if not is_fake:
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
            _, idxs = func(scores, k_value, 1, largest, sorted_output, **call_kwargs)
            vals = torch.gather(padded, 1, idxs)
        else:
            vals, idxs = func(padded, k_value, 1, largest, sorted_output, **call_kwargs)

        vals_packed = vals.reshape(-1, *vals.shape[2:])
        idxs_packed = idxs.reshape(-1, *idxs.shape[2:])
        return (
            _packed_new_ragged_size(source, vals_packed, k_value),
            _packed_new_ragged_size(source, idxs_packed, k_value),
        )
    return per_element_fallback(func, (source, k, dim_adj, largest, sorted_output), call_kwargs)


@NestedTensorAtenRegistry.implement(aten.cumsum.default)
@NestedTensorAtenRegistry.implement(aten.cumprod.default)
def _cumulative(func, args, kwargs):
    r"""Apply cumulative ops on packed _values when the target dim is static."""
    source = args[0]
    call_kwargs = dict(kwargs)
    dim = args[1] if len(args) > 1 else call_kwargs.pop("dim")
    dim_adj = _translate_dim(source, dim)
    extra_args = args[2:] if len(args) > 2 else ()
    if source._values.dim() > 1 and dim_adj > 0:
        return _packed_like(source, func(source._values, dim_adj, *extra_args, **call_kwargs))
    if dim_adj == 0:
        neutral = 0 if func is aten.cumsum.default else 1
        padded, _, _, batch_idx, local_idx, _ = _packed_to_padded(source, fill_value=neutral)
        out_padded = func(padded, 1, *extra_args, **call_kwargs)
        return _packed_like(source, out_padded[batch_idx, local_idx])
    return per_element_fallback(func, (source, dim_adj, *extra_args), call_kwargs)


@NestedTensorAtenRegistry.implement(aten.flip.default)
def _flip(func, args, kwargs):
    r"""Flip along non-ragged dims by operating directly on packed _values."""
    source = args[0]
    call_kwargs = dict(kwargs)
    dims = args[1] if len(args) > 1 else call_kwargs.pop("dims", ())
    if isinstance(dims, int):
        dims = (dims,)
    dims_adj = tuple(_translate_dim(source, dim) for dim in dims)
    if source._values.dim() > 1 and all(dim > 0 for dim in dims_adj):
        return _packed_like(source, func(source._values, dims_adj, **call_kwargs))
    if any(dim == 0 for dim in dims_adj):
        padded, _, lengths_dev, batch_idx, local_idx, max_len = _packed_to_padded(
            source, fill_value=source.padding_value
        )
        padded_dims = tuple(1 if dim == 0 else dim + 1 for dim in dims_adj)
        out_padded = func(padded, padded_dims, **call_kwargs)
        ragged_flips = sum(dim == 0 for dim in dims_adj)
        if ragged_flips % 2 == 1:
            row_idx = max_len - lengths_dev[batch_idx] + local_idx
        else:
            row_idx = local_idx
        return _packed_like(source, out_padded[batch_idx, row_idx])
    return per_element_fallback(func, (source, dims_adj), call_kwargs)


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
    r"""Apply native dropout to _values, returning (output, mask) as NestedTensors."""
    source = args[0]
    output, mask = func(source._values, *args[1:], **kwargs)
    return _packed_like(source, output), _packed_like(source, mask)


# ---------------------------------------------------------------------------
# In-place ops that operate directly on _values
# ---------------------------------------------------------------------------


@NestedTensorAtenRegistry.implement(aten.copy_.default)
def _copy(func, args, kwargs):
    r"""In-place copy from src to dest, using packed _values when offsets match."""
    from .nested_tensor import NestedTensor

    dest, src = args[0], args[1]
    if isinstance(src, NestedTensor) and _offsets_match(dest._offsets, src._offsets):
        func(dest._values, src._values, *args[2:], **kwargs)
        dest._cached_storage = None
        return dest
    raise NotImplementedError(f"NestedTensor: {func} requires matching offsets")


# ---------------------------------------------------------------------------
# Ternary ops — where, addcmul, addcdiv, lerp.Tensor (3 NT tensor args)
# ---------------------------------------------------------------------------


def _ternary_handler(func, args, kwargs):
    r"""Dispatch handler for ternary ops (where, addcmul, etc.) on packed _values."""
    from .nested_tensor import NestedTensor

    a, b, c = args[0], args[1], args[2]
    sources = [x for x in (a, b, c) if isinstance(x, NestedTensor)]
    if not sources:
        return func(*args, **kwargs)
    ref = sources[0]
    if all(_offsets_match(ref._offsets, s._offsets) for s in sources[1:]):
        va = a._values if isinstance(a, NestedTensor) else a
        vb = b._values if isinstance(b, NestedTensor) else b
        vc = c._values if isinstance(c, NestedTensor) else c
        return _packed_like(ref, func(va, vb, vc, **kwargs))
    raise NotImplementedError(f"NestedTensor: {func} requires matching offsets across all NT operands")


_TERNARY_OPS = [
    aten.where.self,
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
]


def _global_reduction_handler(func, args, kwargs):
    r"""Dispatch handler for global reductions (sum, mean, etc.) over all _values."""
    source = args[0]
    return func(source._values, **kwargs)


def _dimless_reduction_handler(func, args, kwargs):
    r"""Dispatch handler for amax/amin: global reduction only (no dim argument)."""
    source = args[0]
    dim = args[1] if len(args) > 1 else []
    if not dim:
        return func(source._values, **kwargs)
    raise NotImplementedError(f"NestedTensor: {func} with dim argument is not supported")


# ---------------------------------------------------------------------------
# masked_fill — fast path when mask is a NestedTensor with matching offsets
# ---------------------------------------------------------------------------


def _masked_fill_handler(func, args, kwargs):
    r"""Dispatch handler for masked_fill with matching-offset fast path."""
    from .nested_tensor import NestedTensor

    source, mask, value = args[0], args[1], args[2]
    if isinstance(mask, NestedTensor) and _offsets_match(source._offsets, mask._offsets):
        return _packed_like(source, func(source._values, mask._values, value, **kwargs))
    raise NotImplementedError(f"NestedTensor: {func} requires mask with matching offsets")


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
for _op in _UNARY_LIKE_OPS + _CREATION_OPS + _RANDOM_CREATION_OPS:
    NestedTensorAtenRegistry[_op] = _unary_handler
for _op in _INPLACE_RNG_OPS:
    NestedTensorAtenRegistry[_op] = _inplace_unary_handler
for _op in _TERNARY_OPS:
    NestedTensorAtenRegistry[_op] = _ternary_handler
for _op in _GLOBAL_REDUCTION_OPS:
    NestedTensorAtenRegistry[_op] = _global_reduction_handler
for _op in [aten.amax.default, aten.amin.default]:
    NestedTensorAtenRegistry[_op] = _dimless_reduction_handler
NestedTensorAtenRegistry[aten.rms_norm.default] = _unary_handler
for _op in [aten._softmax.default, aten._log_softmax.default]:
    NestedTensorAtenRegistry[_op] = _softmax_handler
for _op in [aten._softmax_backward_data.default, aten._log_softmax_backward_data.default]:
    NestedTensorAtenRegistry[_op] = _binary_unwrap_handler
for _op in [aten.masked_fill.Scalar, aten.masked_fill.Tensor]:
    NestedTensorAtenRegistry[_op] = _masked_fill_handler
for _op in [aten.isin.Tensor_Tensor, aten.isin.Tensor_Scalar, aten.bucketize.Tensor]:
    NestedTensorAtenRegistry[_op] = _unary_handler
