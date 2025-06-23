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

"""
``__torch_dispatch__`` handlers for NestedTensor aten ops.

This module implements the dispatch table that maps aten ops to optimized handlers
operating on the packed representation (_values, _offsets, _shape_tensor).

Architecture:
    - Elementwise ops operate directly on ``_values`` (no unpack/repack overhead)
    - Structural ops (clone, detach, to_copy) operate on all inner tensors
    - Unregistered ops fall back to per-element application via ``_storage``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .nested_tensor import NestedTensor

aten = torch.ops.aten

DISPATCH_TABLE: dict[Any, Any] = {}


def _register(op):
    def decorator(fn):
        DISPATCH_TABLE[op] = fn
        return fn

    return decorator


def _find_nested(*args) -> NestedTensor | None:
    from .nested_tensor import NestedTensor

    for a in args:
        if isinstance(a, NestedTensor):
            return a
        if isinstance(a, (list, tuple)):
            result = _find_nested(*a)
            if result is not None:
                return result
    return None


def _rebuild(source: NestedTensor, values: Tensor, offsets: Tensor, shape_tensor: Tensor) -> NestedTensor:
    from .nested_tensor import NestedTensor

    return NestedTensor._from_packed(
        values,
        offsets,
        shape_tensor,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        outer_size=source._logical_shape,
    )


def _from_values(source: NestedTensor, new_values: Tensor) -> NestedTensor:
    return _rebuild(source, new_values, source._offsets, source._shape_tensor)


def per_element_fallback(func, args, kwargs):
    """Fallback: unpack to individual tensors, apply op, repack."""
    from .nested_tensor import NestedTensor

    source = _find_nested(*args)
    if source is None:
        source = _find_nested(*kwargs.values()) if kwargs else None
    if source is None:
        return func(*args, **kwargs)

    batch_size = len(source)
    if batch_size == 0:
        return NestedTensor._from_packed(
            source._values,
            source._offsets,
            source._shape_tensor,
            batch_first=source.batch_first,
            padding_value=source.padding_value,
            mask_value=source.mask_value,
        )

    def replace_nested_with_element(obj, idx):
        if isinstance(obj, NestedTensor):
            return obj._storage[idx]
        if isinstance(obj, (list, tuple)):
            mapped = type(obj)(replace_nested_with_element(x, idx) for x in obj)
            return mapped
        return obj

    results = []
    for i in range(batch_size):
        elem_args = replace_nested_with_element(args, i)
        elem_kwargs = {k: replace_nested_with_element(v, i) for k, v in kwargs.items()} if kwargs else {}
        result = func(*elem_args, **elem_kwargs)
        results.append(result)

    if all(isinstance(r, Tensor) for r in results):
        try:
            return torch.stack(results)
        except (RuntimeError, ValueError):
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
                try:
                    unpacked.append(torch.stack(elements))
                except (RuntimeError, ValueError):
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

_ELEMENTWISE_UNARY_OPS = [
    aten.abs.default,
    aten.neg.default,
    aten.sign.default,
    aten.sgn.default,
    aten.ceil.default,
    aten.floor.default,
    aten.round.default,
    aten.trunc.default,
    aten.frac.default,
    aten.reciprocal.default,
    aten.sqrt.default,
    aten.rsqrt.default,
    aten.exp.default,
    aten.exp2.default,
    aten.expm1.default,
    aten.log.default,
    aten.log2.default,
    aten.log10.default,
    aten.log1p.default,
    aten.sin.default,
    aten.cos.default,
    aten.tan.default,
    aten.asin.default,
    aten.acos.default,
    aten.atan.default,
    aten.sinh.default,
    aten.cosh.default,
    aten.tanh.default,
    aten.asinh.default,
    aten.acosh.default,
    aten.atanh.default,
    aten.sigmoid.default,
    aten.logit.default,
    aten.relu.default,
    aten.gelu.default,
    aten.silu.default,
    aten.mish.default,
    aten.hardsigmoid.default,
    aten.hardswish.default,
    aten.hardtanh.default,
    aten.leaky_relu.default,
    aten.elu.default,
    aten.celu.default,
    aten.selu.default,
    aten.bitwise_not.default,
    aten.isnan.default,
    aten.isinf.default,
    aten.isfinite.default,
    aten.logical_not.default,
    aten.erf.default,
    aten.erfc.default,
    aten.erfinv.default,
    aten.positive.default,
    aten.zero_.default,
    aten.fill_.Scalar,
]


def _make_unary_handler(op):
    def handler(func, args, kwargs):
        source = args[0]
        new_values = func(source._values, *args[1:], **kwargs)
        return _from_values(source, new_values)

    return handler


for _op in _ELEMENTWISE_UNARY_OPS:
    DISPATCH_TABLE[_op] = _make_unary_handler(_op)

# ---------------------------------------------------------------------------
# Elementwise binary ops — apply directly to _values
# ---------------------------------------------------------------------------

_ELEMENTWISE_BINARY_OPS = [
    aten.add.Tensor,
    aten.sub.Tensor,
    aten.mul.Tensor,
    aten.div.Tensor,
    aten.div.Tensor_mode,
    aten.remainder.Tensor,
    aten.pow.Tensor_Tensor,
    aten.pow.Tensor_Scalar,
    aten.pow.Scalar,
    aten.atan2.default,
    aten.maximum.default,
    aten.minimum.default,
    aten.fmod.Tensor,
    aten.bitwise_and.Tensor,
    aten.bitwise_or.Tensor,
    aten.bitwise_xor.Tensor,
    aten.bitwise_left_shift.Tensor,
    aten.bitwise_right_shift.Tensor,
    aten.logical_and.default,
    aten.logical_or.default,
    aten.logical_xor.default,
    aten.eq.Tensor,
    aten.eq.Scalar,
    aten.ne.Tensor,
    aten.ne.Scalar,
    aten.lt.Tensor,
    aten.lt.Scalar,
    aten.le.Tensor,
    aten.le.Scalar,
    aten.gt.Tensor,
    aten.gt.Scalar,
    aten.ge.Tensor,
    aten.ge.Scalar,
    aten.add.Scalar,
    aten.sub.Scalar,
    aten.mul.Scalar,
    aten.div.Scalar,
    aten.div.Scalar_mode,
    aten.remainder.Scalar,
    aten.fmod.Scalar,
    aten.bitwise_and.Scalar,
    aten.bitwise_or.Scalar,
    aten.bitwise_xor.Scalar,
    aten.bitwise_left_shift.Tensor_Scalar,
    aten.bitwise_right_shift.Tensor_Scalar,
]


def _make_binary_handler(op):
    def handler(func, args, kwargs):
        from .nested_tensor import NestedTensor

        lhs, rhs = args[0], args[1]
        if isinstance(lhs, NestedTensor) and isinstance(rhs, NestedTensor):
            if len(lhs) != len(rhs):
                raise ValueError(f"NestedTensor batch length mismatch: {len(lhs)} vs {len(rhs)}")
            # Both are NestedTensors — if same packing layout, operate on _values directly
            if lhs._offsets.shape == rhs._offsets.shape and torch.equal(lhs._offsets, rhs._offsets):
                new_values = func(lhs._values, rhs._values, *args[2:], **kwargs)
                return _from_values(lhs, new_values)
            # Different packing layout — fall back to per-element
            return per_element_fallback(func, args, kwargs)
        elif isinstance(lhs, NestedTensor):
            # rhs is scalar or broadcastable tensor
            if isinstance(rhs, Tensor) and rhs.dim() > 0:
                # Non-scalar tensor — fall back to per-element for correctness
                return per_element_fallback(func, args, kwargs)
            new_values = func(lhs._values, rhs, *args[2:], **kwargs)
            return _from_values(lhs, new_values)
        else:
            # lhs is scalar/tensor, rhs is NestedTensor
            if isinstance(lhs, Tensor) and lhs.dim() > 0:
                return per_element_fallback(func, args, kwargs)
            new_values = func(lhs, rhs._values, *args[2:], **kwargs)
            return _from_values(rhs, new_values)

    return handler


for _op in _ELEMENTWISE_BINARY_OPS:
    DISPATCH_TABLE[_op] = _make_binary_handler(_op)


# ---------------------------------------------------------------------------
# Structural ops
# ---------------------------------------------------------------------------


@_register(aten.clone.default)
def _clone(func, args, kwargs):
    source = args[0]
    return _rebuild(source, source._values.clone(), source._offsets.clone(), source._shape_tensor.clone())


@_register(aten.detach.default)
def _detach(func, args, kwargs):
    source = args[0]
    return _rebuild(source, source._values.detach(), source._offsets.detach(), source._shape_tensor.detach())


@_register(aten._to_copy.default)
def _to_copy(func, args, kwargs):
    source = args[0]
    new_values = func(source._values, **kwargs)
    device = kwargs.get("device")
    new_offsets = source._offsets.to(device=device) if device else source._offsets
    new_shape_tensor = source._shape_tensor.to(device=device) if device else source._shape_tensor
    return _rebuild(source, new_values, new_offsets, new_shape_tensor)


@_register(aten.alias.default)
def _alias(func, args, kwargs):
    source = args[0]
    return _rebuild(source, source._values.alias(), source._offsets, source._shape_tensor)


# ---------------------------------------------------------------------------
# Matrix multiply ops — apply to _values, update last dim of shape_tensor
# ---------------------------------------------------------------------------


def _rebuild_new_last_dim(source: NestedTensor, new_values: Tensor, new_last_dim: int) -> NestedTensor:
    from .nested_tensor import NestedTensor

    new_shape = source._shape_tensor.clone()
    new_shape[:, -1] = new_last_dim
    return NestedTensor._from_packed(
        new_values,
        source._offsets,
        new_shape,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
    )


@_register(aten.mm.default)
def _mm(func, args, kwargs):
    from .nested_tensor import NestedTensor

    mat1, mat2 = args[0], args[1]
    if isinstance(mat1, NestedTensor) and not isinstance(mat2, NestedTensor) and mat1._values.dim() == 2:
        new_values = func(mat1._values, mat2, **kwargs)
        return _rebuild_new_last_dim(mat1, new_values, mat2.shape[1])
    return per_element_fallback(func, args, kwargs)


@_register(aten.addmm.default)
def _addmm(func, args, kwargs):
    from .nested_tensor import NestedTensor

    bias, mat1, mat2 = args[0], args[1], args[2]
    if isinstance(mat1, NestedTensor) and not isinstance(mat2, NestedTensor) and mat1._values.dim() == 2:
        new_values = func(bias, mat1._values, mat2, **kwargs)
        return _rebuild_new_last_dim(mat1, new_values, mat2.shape[1])
    return per_element_fallback(func, args, kwargs)


@_register(aten.bmm.default)
def _bmm(func, args, kwargs):
    from .nested_tensor import NestedTensor

    mat1, mat2 = args[0], args[1]
    if isinstance(mat1, NestedTensor) and isinstance(mat2, NestedTensor) and torch.equal(mat1._offsets, mat2._offsets):
        new_values = func(mat1._values, mat2._values, **kwargs)
        return _rebuild_new_last_dim(mat1, new_values, mat2._values.shape[-1])
    return per_element_fallback(func, args, kwargs)


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
]


def _make_inplace_unary_handler(op):
    def handler(func, args, kwargs):
        source = args[0]
        func(source._values, *args[1:], **kwargs)
        source._cache = {}
        return source

    return handler


for _op in _INPLACE_UNARY_OPS:
    DISPATCH_TABLE[_op] = _make_inplace_unary_handler(_op)


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
]


def _make_inplace_binary_handler(op):
    def handler(func, args, kwargs):
        from .nested_tensor import NestedTensor

        source = args[0]
        other = args[1]
        if isinstance(other, NestedTensor):
            if source._offsets.shape == other._offsets.shape and torch.equal(source._offsets, other._offsets):
                func(source._values, other._values, *args[2:], **kwargs)
            else:
                # Different layout — fall back
                return per_element_fallback(func, args, kwargs)
        elif isinstance(other, Tensor) and other.dim() > 0:
            return per_element_fallback(func, args, kwargs)
        else:
            func(source._values, other, *args[2:], **kwargs)
        source._cache = {}
        return source

    return handler


for _op in _INPLACE_BINARY_OPS:
    DISPATCH_TABLE[_op] = _make_inplace_binary_handler(_op)
