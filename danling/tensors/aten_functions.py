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

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .ops import ATEN_BINARY_ELEMENTWISE_OPS, ATEN_UNARY_ELEMENTWISE_OPS, NestedTensorAtenRegistry

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


@torch._dynamo.disable
def per_element_fallback(func, args, kwargs):
    r"""
    Fallback for unregistered ops: unpack to individual tensors, apply op, repack.

    Only used by ``__torch_dispatch__`` as a catch-all for ops without a registered handler.
    Registered handlers should raise ``NotImplementedError`` instead of calling this.
    """
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
    return type(source)._from_packed(
        func(source._values, *args[1:], **kwargs),
        source._offsets,
        source._shape_tensor,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=source._logical_shape,
    )


for _op in ATEN_UNARY_ELEMENTWISE_OPS:
    NestedTensorAtenRegistry[_op] = _unary_handler

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
        return type(lhs)._from_packed(
            func(lhs._values, resolved, *extra, **kwargs),
            lhs._offsets,
            lhs._shape_tensor,
            batch_first=lhs.batch_first,
            padding_value=lhs.padding_value,
            mask_value=lhs.mask_value,
            pin_memory=lhs._pin_memory,
            outer_size=lhs._logical_shape,
        )
    # lhs is scalar/tensor, rhs is NestedTensor
    resolved = _resolve_other(rhs, lhs, func)
    return type(rhs)._from_packed(
        func(resolved, rhs._values, *extra, **kwargs),
        rhs._offsets,
        rhs._shape_tensor,
        batch_first=rhs.batch_first,
        padding_value=rhs.padding_value,
        mask_value=rhs.mask_value,
        pin_memory=rhs._pin_memory,
        outer_size=rhs._logical_shape,
    )


for _op in ATEN_BINARY_ELEMENTWISE_OPS:
    NestedTensorAtenRegistry[_op] = _binary_handler


# ---------------------------------------------------------------------------
# Structural ops
# ---------------------------------------------------------------------------


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
    new_values = func(source._values, **kwargs)
    # Offsets and shape_tensor stay on CPU — they are metadata, not compute tensors.
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


@NestedTensorAtenRegistry.implement(aten.alias.default)
def _alias(func, args, kwargs):
    r"""Create an alias of the NestedTensor sharing the same _values storage."""
    source = args[0]
    return type(source)._from_packed(
        source._values.alias(),
        source._offsets,
        source._shape_tensor,
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=source._logical_shape,
    )


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
    return source


for _op in _INPLACE_UNARY_OPS:
    NestedTensorAtenRegistry[_op] = _inplace_unary_handler


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
    return source


for _op in _INPLACE_BINARY_OPS:
    NestedTensorAtenRegistry[_op] = _inplace_binary_handler


# ---------------------------------------------------------------------------
# Shape-preserving unary-like ops (extra scalar/keyword args, operate on _values)
# ---------------------------------------------------------------------------

_UNARY_LIKE_OPS = [
    aten.clamp.default,
    aten.clamp_min.default,
    aten.clamp_max.default,
    aten.nan_to_num.default,
]

for _op in _UNARY_LIKE_OPS:
    NestedTensorAtenRegistry[_op] = _unary_handler


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

for _op in _CREATION_OPS:
    NestedTensorAtenRegistry[_op] = _unary_handler


# ---------------------------------------------------------------------------
# native_dropout — returns (output, mask) tuple, both as NestedTensor
# ---------------------------------------------------------------------------


@NestedTensorAtenRegistry.implement(aten.native_dropout.default)
def _native_dropout(func, args, kwargs):
    r"""Apply native dropout to _values, returning (output, mask) as NestedTensors."""
    source = args[0]
    cls = type(source)
    output, mask = func(source._values, *args[1:], **kwargs)
    kw = dict(
        batch_first=source.batch_first,
        padding_value=source.padding_value,
        mask_value=source.mask_value,
        pin_memory=source._pin_memory,
        outer_size=source._logical_shape,
    )
    return (
        cls._from_packed(output, source._offsets, source._shape_tensor, **kw),
        cls._from_packed(mask, source._offsets, source._shape_tensor, **kw),
    )


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
        return type(ref)._from_packed(
            func(va, vb, vc, **kwargs),
            ref._offsets,
            ref._shape_tensor,
            batch_first=ref.batch_first,
            padding_value=ref.padding_value,
            mask_value=ref.mask_value,
            pin_memory=ref._pin_memory,
            outer_size=ref._logical_shape,
        )
    raise NotImplementedError(f"NestedTensor: {func} requires matching offsets across all NT operands")


_TERNARY_OPS = [
    aten.where.self,
    aten.addcmul.default,
    aten.addcdiv.default,
    aten.lerp.Tensor,
]

for _op in _TERNARY_OPS:
    NestedTensorAtenRegistry[_op] = _ternary_handler


# ---------------------------------------------------------------------------
# Normalization ops — operate on packed _values
# ---------------------------------------------------------------------------


@NestedTensorAtenRegistry.implement(aten.native_layer_norm.default)
def _native_layer_norm(func, args, kwargs):
    r"""Dispatch handler for layer norm on packed _values."""
    source = args[0]
    output, mean, rstd = func(source._values, *args[1:], **kwargs)
    return (
        type(source)._from_packed(
            output,
            source._offsets,
            source._shape_tensor,
            batch_first=source.batch_first,
            padding_value=source.padding_value,
            mask_value=source.mask_value,
            pin_memory=source._pin_memory,
            outer_size=source._logical_shape,
        ),
        mean,
        rstd,
    )


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
    grad_input = type(ref)._from_packed(
        grad_input,
        ref._offsets,
        ref._shape_tensor,
        batch_first=ref.batch_first,
        padding_value=ref.padding_value,
        mask_value=ref.mask_value,
        pin_memory=ref._pin_memory,
        outer_size=ref._logical_shape,
    )
    return grad_input, grad_weight, grad_bias


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
    return type(ref)._from_packed(
        func(va, vb, *args[2:], **kwargs),
        ref._offsets,
        ref._shape_tensor,
        batch_first=ref.batch_first,
        padding_value=ref.padding_value,
        mask_value=ref.mask_value,
        pin_memory=ref._pin_memory,
        outer_size=ref._logical_shape,
    )


for _op in [aten._softmax.default, aten._log_softmax.default]:
    NestedTensorAtenRegistry[_op] = _unary_handler

for _op in [aten._softmax_backward_data.default, aten._log_softmax_backward_data.default]:
    NestedTensorAtenRegistry[_op] = _binary_unwrap_handler


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


for _op in _GLOBAL_REDUCTION_OPS:
    NestedTensorAtenRegistry[_op] = _global_reduction_handler


def _dimless_reduction_handler(func, args, kwargs):
    r"""Dispatch handler for amax/amin: global reduction only (no dim argument)."""
    source = args[0]
    dim = args[1] if len(args) > 1 else []
    if not dim:
        return func(source._values, **kwargs)
    raise NotImplementedError(f"NestedTensor: {func} with dim argument is not supported")


for _op in [aten.amax.default, aten.amin.default]:
    NestedTensorAtenRegistry[_op] = _dimless_reduction_handler


# ---------------------------------------------------------------------------
# rms_norm — operate on packed _values (mirrors native_layer_norm)
# ---------------------------------------------------------------------------

NestedTensorAtenRegistry[aten.rms_norm.default] = _unary_handler


# ---------------------------------------------------------------------------
# masked_fill — fast path when mask is a NestedTensor with matching offsets
# ---------------------------------------------------------------------------


def _masked_fill_handler(func, args, kwargs):
    r"""Dispatch handler for masked_fill with matching-offset fast path."""
    from .nested_tensor import NestedTensor

    source, mask, value = args[0], args[1], args[2]
    if isinstance(mask, NestedTensor) and _offsets_match(source._offsets, mask._offsets):
        return type(source)._from_packed(
            func(source._values, mask._values, value, **kwargs),
            source._offsets,
            source._shape_tensor,
            batch_first=source.batch_first,
            padding_value=source.padding_value,
            mask_value=source.mask_value,
            pin_memory=source._pin_memory,
            outer_size=source._logical_shape,
        )
    raise NotImplementedError(f"NestedTensor: {func} requires mask with matching offsets")


for _op in [aten.masked_fill.Scalar, aten.masked_fill.Tensor]:
    NestedTensorAtenRegistry[_op] = _masked_fill_handler


# ---------------------------------------------------------------------------
# RNG in-place ops — shape-preserving mutations on _values
# ---------------------------------------------------------------------------

_INPLACE_RNG_OPS = [
    aten.uniform_.default,
    aten.normal_.default,
]

for _op in _INPLACE_RNG_OPS:
    NestedTensorAtenRegistry[_op] = _inplace_unary_handler


# ---------------------------------------------------------------------------
# Random tensor creation ops — same pattern as empty_like/zeros_like
# ---------------------------------------------------------------------------

_RANDOM_CREATION_OPS = [
    aten.rand_like.default,
    aten.randn_like.default,
    aten.randint_like.default,
    aten.randint_like.low_dtype,
]

for _op in _RANDOM_CREATION_OPS:
    NestedTensorAtenRegistry[_op] = _unary_handler


# ---------------------------------------------------------------------------
# isin / bucketize — elementwise on packed _values
# ---------------------------------------------------------------------------

for _op in [aten.isin.Tensor_Tensor, aten.isin.Tensor_Scalar, aten.bucketize.Tensor]:
    NestedTensorAtenRegistry[_op] = _unary_handler
