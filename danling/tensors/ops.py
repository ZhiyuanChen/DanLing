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
r"""Internal helpers shared across NestedTensor function registrations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ._streams import _run_on_streams

if TYPE_CHECKING:
    from .nested_tensor import NestedTensor


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TorchFuncRegistry(dict):
    r"""
    Plain dict mapping functions/ops to their NestedTensor handlers.

    Uses ``dict`` directly for O(1) lookup with minimal overhead (~30 ns)
    instead of chanfig.Registry (~700-2300 ns).

    Used for both ``__torch_function__`` (torch/nn ops) and
    ``__torch_dispatch__`` (aten ops) dispatch tables.
    """

    def implement(self, func: Callable) -> Callable:
        r"""Decorator to register a handler for *func*."""

        def wrapper(handler: Callable) -> Callable:
            self[func] = handler
            return handler

        return wrapper


#: ``__torch_function__`` dispatch table for ``torch.*`` and ``F.*`` ops.
NestedTensorFuncRegistry = TorchFuncRegistry()

#: ``__torch_dispatch__`` dispatch table for aten ops.
NestedTensorAtenRegistry = TorchFuncRegistry()


# ---------------------------------------------------------------------------
# Dimension Translation
# ---------------------------------------------------------------------------


def _get_batch_dim(input: NestedTensor) -> int:
    r"""Return the batch dimension index (0 if batch_first, else 1)."""
    return 0 if input.batch_first else 1


def _normalize_dim(dim: int, ndim: int) -> int:
    r"""Normalize a negative dimension index to positive."""
    if dim < 0:
        dim += ndim
    return dim


def _translate_dim(input: NestedTensor, dim: int) -> int:
    r"""Translate a NestedTensor dimension to a per-element dimension."""
    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        raise ValueError("Cannot translate the batch dimension for NestedTensor.")
    if input.batch_first:
        return dim - 1
    return dim if dim < batch_dim else dim - 1


def _translate_dims(input: NestedTensor, dims: Sequence[int]) -> tuple[int, ...]:
    r"""Translate multiple NestedTensor dimensions to per-element dimensions."""
    ndim = input.dim()
    batch_dim = _get_batch_dim(input)
    bf = input.batch_first
    result = []
    for d in dims:
        d = _normalize_dim(d, ndim)
        if d == batch_dim:
            raise ValueError("Cannot translate the batch dimension for NestedTensor.")
        result.append(d - 1 if bf else (d if d < batch_dim else d - 1))
    return tuple(result)


def _translate_non_batch_dim(input: NestedTensor, dim: int, *, name: str = "dim") -> int:
    r"""Translate a non-batch dimension, raising ValueError if dim is the batch dim."""
    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        raise ValueError(f"{name} along the batch dimension is not supported for NestedTensor.")
    return _translate_dim(input, dim)


# Storage Mapping


def _try_stack(values: Sequence, input: NestedTensor):
    r"""Stack results into a tensor, falling back to NestedTensor if shapes differ."""
    try:
        return torch.stack(list(values))
    except (RuntimeError, ValueError, TypeError):
        return type(input)(values, **input._meta())


@torch._dynamo.disable
def _map_storage(input: NestedTensor, fn):
    r"""Apply fn to each element in storage, using CUDA streams if available."""
    cls = type(input)
    elements = input._storage
    if not elements:
        return cls([], **input._meta(include_dtype=True))
    if not elements[0].is_cuda:
        return cls((fn(t) for t in elements), **input._meta())
    return cls(_run_on_streams(elements, fn), **input._meta())


def _map_storage_pair(input: NestedTensor, op, *args, **kwargs):
    r"""Apply *op* to every element, unpacking each 2-tuple result into two NestedTensors."""
    cls = type(input)
    if len(input) == 0:
        try:
            first_probe, second_probe = op(input._values, *args, **kwargs)
            first_dtype = first_probe.dtype if isinstance(first_probe, Tensor) else input.dtype
            second_dtype = second_probe.dtype if isinstance(second_probe, Tensor) else input.dtype
        except (TypeError, RuntimeError, ValueError):
            first_dtype = input.dtype
            second_dtype = input.dtype
        return cls([], dtype=first_dtype, **input._meta()), cls([], dtype=second_dtype, **input._meta())
    firsts, seconds = [], []
    for t in input._storage:
        a, b = op(t, *args, **kwargs)
        firsts.append(a)
        seconds.append(b)
    return cls(firsts, **input._meta()), cls(seconds, **input._meta())


# Binary & Ternary Operations


def _as_tensor_like(value, ref: Tensor) -> Tensor:
    r"""Convert value to a tensor on the same device as ref."""
    if isinstance(value, Tensor):
        return value.to(device=ref.device)
    return torch.as_tensor(value, device=ref.device, dtype=torch.result_type(ref, value))


def _ensure_nested_input(input, other, cls):
    r"""Ensure at least one argument is a NestedTensor, converting if needed."""
    if isinstance(input, cls):
        return input
    if isinstance(other, cls):
        return other.nested_like(input)
    raise ValueError("At least one argument must be a NestedTensor.")


def _binary_op_maybe_tensor(input, other, op, *extra_args, **extra_kwargs):
    r"""
    Apply a binary op between a NestedTensor and a tensor/scalar/NestedTensor.

    Performance notes:
    - **Scalar or 0-dim tensor ``other``**: O(1) — op runs directly on packed
      ``_values`` with no unpack/repack overhead. This is the common training path.
    - **Matched-offset NestedTensor ``other``**: O(1) — ``_offsets_match`` fast-path,
      op runs on ``_values`` directly.
    - **Mismatched-offset NestedTensor ``other``**: O(B) — iterates over ``_storage``
      and constructs a new NestedTensor from individual results.
    - **Dense tensor ``other`` with shape matching ``input.shape``**: converted via
      ``nested_like`` → ``tensor_mask`` internally, which has O(B * max_len) cost
      from padding the dense tensor to match the packed layout. Avoid in hot paths.
    """
    from .aten_functions import _offsets_match
    from .nested_tensor import NestedTensor

    # Normalize: input is always the NestedTensor
    cls = type(input) if isinstance(input, NestedTensor) else type(other)
    reverse = False
    if not isinstance(input, cls):
        reverse = True
        input, other = other, input

    if len(input) == 0:
        if isinstance(other, cls) and len(input) != len(other):
            raise ValueError(
                "NestedTensor batch length mismatch between input and other: " f"input={len(input)}, other={len(other)}"
            )
        if isinstance(other, cls):
            resolved = other._values
        else:
            resolved = _as_tensor_like(other, input._values)
        new_values = (
            op(resolved, input._values, *extra_args, **extra_kwargs)
            if reverse
            else op(input._values, resolved, *extra_args, **extra_kwargs)
        )
        return cls._from_packed(
            new_values,
            input._offsets,
            input._shape_tensor,
            batch_first=input.batch_first,
            padding_value=input.padding_value,
            mask_value=input.mask_value,
            pin_memory=input._pin_memory,
            outer_size=input._logical_shape,
        )

    # NT + scalar or 0-d tensor (most common in training)
    if not isinstance(other, Tensor) or other.dim() == 0:
        val = _as_tensor_like(other, input._values)
        new_values = (
            op(val, input._values, *extra_args, **extra_kwargs)
            if reverse
            else op(input._values, val, *extra_args, **extra_kwargs)
        )
        return cls._from_packed(
            new_values,
            input._offsets,
            input._shape_tensor,
            batch_first=input.batch_first,
            padding_value=input.padding_value,
            mask_value=input.mask_value,
            pin_memory=input._pin_memory,
            outer_size=input._logical_shape,
        )

    # Convert padded tensor to NT if shapes match
    if not isinstance(other, cls) and input.shape == other.shape:
        other = input.nested_like(other, strict=False)

    # NT + NT
    if isinstance(other, cls):
        if len(input) != len(other):
            raise ValueError(
                "NestedTensor batch length mismatch between input and other: " f"input={len(input)}, other={len(other)}"
            )
        lhs_v, rhs_v = (other._values, input._values) if reverse else (input._values, other._values)
        if _offsets_match(input._offsets, other._offsets):
            return cls._from_packed(
                op(lhs_v, rhs_v, *extra_args, **extra_kwargs),
                input._offsets,
                input._shape_tensor,
                batch_first=input.batch_first,
                padding_value=input.padding_value,
                mask_value=input.mask_value,
                pin_memory=input._pin_memory,
                outer_size=input._logical_shape,
            )
        lhs_s, rhs_s = (other._storage, input._storage) if reverse else (input._storage, other._storage)
        return cls(
            (op(x, y, *extra_args, **extra_kwargs) for x, y in zip(lhs_s, rhs_s)),
            **input._meta(),
        )

    # General tensor: per-element fallback
    elements = []
    for t in input._storage:
        if reverse:
            elements.append(op(_as_tensor_like(other, t), t, *extra_args, **extra_kwargs))
        else:
            elements.append(op(t, _as_tensor_like(other, t), *extra_args, **extra_kwargs))
    return cls(elements, **input._meta())


def _broadcast_storage(ref: NestedTensor, value):
    r"""Broadcast a value to match ref's per-element storage layout."""
    cls = type(ref)
    if isinstance(value, cls):
        if len(ref) != len(value):
            raise ValueError(
                "NestedTensor batch length mismatch between ref and value: " f"ref={len(ref)}, value={len(value)}"
            )
        return value._storage
    if isinstance(value, Tensor) and not isinstance(value, cls) and value.shape == ref.shape:
        return ref.nested_like(value, strict=False)._storage
    return [value] * len(ref)


def _ternary_op(layout_ref, input, tensor1, tensor2, op, **kwargs):
    r"""Apply a ternary op element-wise across three NestedTensor-compatible operands."""
    ref_elements = layout_ref._storage
    input_storage = _broadcast_storage(layout_ref, input)
    t1_storage = _broadcast_storage(layout_ref, tensor1)
    t2_storage = _broadcast_storage(layout_ref, tensor2)
    elements = []
    for t, x, t1, t2 in zip(ref_elements, input_storage, t1_storage, t2_storage):
        elements.append(
            op(
                _as_tensor_like(x, t),
                _as_tensor_like(t1, t),
                _as_tensor_like(t2, t),
                **kwargs,
            )
        )
    return type(layout_ref)(elements, **layout_ref._meta())


# Reductions


def _reduce_dim(
    input: NestedTensor,
    op,
    dim: int,
    keepdim: bool,
    *,
    dtype: torch.dtype | None = None,
    **op_kwargs,
):
    r"""Reduce a NestedTensor along a single dimension."""
    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dtype is not None:
        op_kwargs["dtype"] = dtype
    if dim == batch_dim:
        results = [op(t, **op_kwargs) for t in input._storage]
        output = torch.stack(results)
        if keepdim:
            return output.unsqueeze(batch_dim)
        return output
    dim_adj = _translate_dim(input, dim)
    results = [op(t, dim=dim_adj, keepdim=keepdim, **op_kwargs) for t in input._storage]
    return _try_stack(results, input)


def _reduce_dims_masked(
    input: NestedTensor,
    dims: Sequence[int],
    op,
    keepdim: bool,
    *,
    dtype: torch.dtype | None = None,
    fill_value: float | int | bool,
):
    r"""Reduce over multiple dims using a padded tensor with masked fill values."""
    dims = tuple(_normalize_dim(d, input.dim()) for d in dims)
    tensor, mask = input.tensor_mask
    valid = mask if not input.mask_value else ~mask
    while valid.dim() < tensor.dim():
        valid = valid.unsqueeze(-1)
    fill = torch.full_like(tensor, fill_value)
    data = torch.where(valid, tensor, fill)
    if dtype is not None:
        return op(data, dim=dims, keepdim=keepdim, dtype=dtype)
    return op(data, dim=dims, keepdim=keepdim)


def _reduce_none(input: NestedTensor, op, *, dtype: torch.dtype | None = None, keepdim: bool = False, **op_kwargs):
    r"""Reduce all elements to a scalar (no dim specified)."""
    if dtype is not None:
        op_kwargs["dtype"] = dtype
    result = op(input._values.reshape(-1), **op_kwargs)
    if keepdim:
        return result.reshape((1,) * input.dim())
    return result


def _reduce_none_pair(input: NestedTensor, op, *, dtype: torch.dtype | None = None, keepdim: bool = False, **op_kwargs):
    r"""Reduce all elements to a scalar pair (e.g. var_mean, no dim specified)."""
    if dtype is not None:
        op_kwargs["dtype"] = dtype
    a, b = op(input._values.reshape(-1), **op_kwargs)
    if keepdim:
        shape = (1,) * input.dim()
        return a.reshape(shape), b.reshape(shape)
    return a, b


def _reduce_multi_dim(
    input: NestedTensor,
    op,
    dims: Sequence[int],
    keepdim: bool,
    *,
    dtype: torch.dtype | None = None,
    **op_kwargs,
):
    r"""Per-element reduction over multiple non-batch dims."""
    dims = tuple(_normalize_dim(d, input.dim()) for d in dims)
    batch_dim = _get_batch_dim(input)
    if batch_dim in dims:
        raise NotImplementedError("Reduction over batch dim + other dims is not supported for NestedTensor.")
    dims_adj = _translate_dims(input, dims)
    op_kwargs["dim"] = dims_adj
    op_kwargs["keepdim"] = keepdim
    if dtype is not None:
        op_kwargs["dtype"] = dtype
    ret = [op(t, **op_kwargs) for t in input._storage]
    return _try_stack(ret, input)


_SENTINEL = object()


def _reduce(
    input: NestedTensor,
    op,
    dim,
    keepdim: bool,
    *,
    dtype: torch.dtype | None = None,
    fill_value=_SENTINEL,
    **op_kwargs,
):
    r"""Unified reduction dispatcher for NestedTensor."""
    if dim is None:
        return _reduce_none(input, op, dtype=dtype, keepdim=keepdim, **op_kwargs)
    if isinstance(dim, int):
        return _reduce_dim(input, op, dim, keepdim, dtype=dtype, **op_kwargs)
    dims = tuple(dim)
    if len(dims) == 1:
        return _reduce_dim(input, op, dims[0], keepdim, dtype=dtype, **op_kwargs)
    if fill_value is not _SENTINEL:
        return _reduce_dims_masked(input, dims, op, keepdim, dtype=dtype, fill_value=fill_value)
    return _reduce_multi_dim(input, op, dims, keepdim, dtype=dtype, **op_kwargs)


def _reduce_dim_pair(input: NestedTensor, op, dim: int, keepdim: bool, **op_kwargs):
    r"""Reduction returning a pair (e.g. values+indices) per element."""
    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        firsts, seconds = [], []
        for t in input._storage:
            a, b = op(t, **op_kwargs)
            firsts.append(a)
            seconds.append(b)
        first, second = torch.stack(firsts), torch.stack(seconds)
        if keepdim:
            first = first.unsqueeze(batch_dim)
            second = second.unsqueeze(batch_dim)
        return first, second
    dim_adj = _translate_dim(input, dim)
    firsts, seconds = [], []
    for t in input._storage:
        a, b = op(t, dim=dim_adj, keepdim=keepdim, **op_kwargs)
        firsts.append(a)
        seconds.append(b)
    return _try_stack(firsts, input), _try_stack(seconds, input)


# Concatenation Helpers


def _concat_apply(
    input: NestedTensor,
    op: Callable[[Tensor], Tensor],
    shape_fn: Callable[[torch.Size], torch.Size],
):
    r"""Apply op to concatenated storage and split back using shape_fn."""
    cls = type(input)
    if not input._storage:
        return cls([], **input._meta(include_dtype=True))

    concat, shapes = input.concatenate()
    if not shapes:
        return cls([], **input._meta(include_dtype=True))

    output = op(concat)
    output_shapes = tuple(shape_fn(shape) for shape in shapes)
    return cls.from_concatenated(output, output_shapes, **input._meta())


def _concat_apply_same_shape(input: NestedTensor, op: Callable[[Tensor], Tensor]):
    r"""Apply a shape-preserving op directly to packed _values."""
    if len(input) == 0:
        return type(input)([], **input._meta(include_dtype=True))
    return type(input)._from_packed(
        op(input._values),
        input._offsets,
        input._shape_tensor,
        batch_first=input.batch_first,
        padding_value=input.padding_value,
        mask_value=input.mask_value,
        pin_memory=input._pin_memory,
        outer_size=input._logical_shape,
    )


def _static_tensor_dims(input: NestedTensor) -> tuple[bool, ...]:
    r"""Return a tuple of bools indicating which per-element dims have uniform size."""
    st = input._shape_tensor
    if st.numel() == 0:
        return ()
    # A dim is static iff all elements have the same size along that dim.
    # Strip trailing zero-padded columns (from elements with fewer dims).
    ncols = st.size(1)
    result = []
    for d in range(ncols):
        col = st[:, d]
        if col.max().item() == 0:
            break  # trailing zero-pad
        result.append(col.min().item() == col.max().item())
    return tuple(result)


def _concat_dim_for_tensor_dim(input: NestedTensor, dim: int) -> int | None:
    r"""Map a per-element tensor dim to the corresponding concatenated tensor dim, or None."""
    if input._shape_tensor.numel() == 0:
        return None
    static_dims = _static_tensor_dims(input)
    if dim < 0:
        dim += len(static_dims)
    if dim < 0 or dim >= len(static_dims):
        raise IndexError(f"Dimension out of range for NestedTensor with {len(static_dims)} dims: {dim}")
    if dim == 0:
        return None
    if all(static_dims):
        return dim
    if not static_dims[dim]:
        return None
    static_indices = [i for i, is_static in enumerate(static_dims) if is_static]
    return 1 + static_indices.index(dim)


# ---------------------------------------------------------------------------
# Elementwise op lists
# ---------------------------------------------------------------------------
# Both the torch level (torch_functions.py, ~14x eager speedup) and aten level
# (aten_functions.py, torch.compile traceability) registration lists are
# maintained here so additions/removals stay in sync.

aten = torch.ops.aten

# -- Unary ops ---------------------------------------------------------------

TORCH_UNARY_ELEMENTWISE_OPS = [
    torch.abs,
    torch.neg,
    torch.sign,
    torch.sgn,
    torch.ceil,
    torch.floor,
    torch.round,
    torch.trunc,
    torch.frac,
    torch.reciprocal,
    torch.sqrt,
    torch.rsqrt,
    torch.exp,
    torch.exp2,
    torch.expm1,
    torch.log,
    torch.log2,
    torch.log10,
    torch.log1p,
    torch.sin,
    torch.cos,
    torch.tan,
    torch.asin,
    torch.acos,
    torch.atan,
    torch.sinh,
    torch.cosh,
    torch.tanh,
    torch.asinh,
    torch.acosh,
    torch.atanh,
    torch.sigmoid,
    torch.logit,
    torch.relu,
    torch.isnan,
    torch.isinf,
    torch.isfinite,
    torch.logical_not,
    torch.erf,
    torch.erfc,
    torch.erfinv,
    torch.positive,
    torch.bitwise_not,
]

ATEN_UNARY_ELEMENTWISE_OPS = [
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
    aten.isnan.default,
    aten.isinf.default,
    aten.isfinite.default,
    aten.logical_not.default,
    aten.erf.default,
    aten.erfc.default,
    aten.erfinv.default,
    aten.positive.default,
    aten.bitwise_not.default,
    # Aten-only: activations that go through F.* at torch level
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
    # Aten-only: in-place ops with no torch.* public API
    aten.zero_.default,
    aten.fill_.Scalar,
]

# -- Binary ops ---------------------------------------------------------------

TORCH_BINARY_ELEMENTWISE_OPS = [
    torch.add,
    torch.sub,
    torch.mul,
    torch.div,
    torch.true_divide,
    torch.floor_divide,
    torch.remainder,
    torch.fmod,
    torch.pow,
    torch.atan2,
    torch.maximum,
    torch.minimum,
    torch.eq,
    torch.ne,
    torch.gt,
    torch.ge,
    torch.lt,
    torch.le,
    torch.logical_and,
    torch.logical_or,
    torch.logical_xor,
    torch.bitwise_and,
    torch.bitwise_or,
    torch.bitwise_xor,
    torch.bitwise_left_shift,
    torch.bitwise_right_shift,
    torch.hypot,
    torch.logaddexp,
    torch.logaddexp2,
    torch.nextafter,
]

ATEN_BINARY_ELEMENTWISE_OPS = [
    aten.add.Tensor,
    aten.sub.Tensor,
    aten.mul.Tensor,
    aten.div.Tensor,
    aten.div.Tensor_mode,
    aten.floor_divide.default,
    aten.remainder.Tensor,
    aten.fmod.Tensor,
    aten.pow.Tensor_Tensor,
    aten.pow.Tensor_Scalar,
    aten.pow.Scalar,
    aten.atan2.default,
    aten.maximum.default,
    aten.minimum.default,
    aten.eq.Tensor,
    aten.eq.Scalar,
    aten.ne.Tensor,
    aten.ne.Scalar,
    aten.gt.Tensor,
    aten.gt.Scalar,
    aten.ge.Tensor,
    aten.ge.Scalar,
    aten.lt.Tensor,
    aten.lt.Scalar,
    aten.le.Tensor,
    aten.le.Scalar,
    aten.logical_and.default,
    aten.logical_or.default,
    aten.logical_xor.default,
    aten.bitwise_and.Tensor,
    aten.bitwise_or.Tensor,
    aten.bitwise_xor.Tensor,
    aten.bitwise_left_shift.Tensor,
    aten.bitwise_right_shift.Tensor,
    # Scalar overloads (same op, different aten dispatch key)
    aten.add.Scalar,
    aten.sub.Scalar,
    aten.mul.Scalar,
    aten.div.Scalar,
    aten.div.Scalar_mode,
    aten.floor_divide.Scalar,
    aten.remainder.Scalar,
    aten.fmod.Scalar,
    aten.bitwise_and.Scalar,
    aten.bitwise_or.Scalar,
    aten.bitwise_xor.Scalar,
    aten.bitwise_left_shift.Tensor_Scalar,
    aten.bitwise_right_shift.Tensor_Scalar,
    # Aten-only: backward activation ops (grad_output, self/output → grad_input)
    aten.gelu_backward.default,
    aten.silu_backward.default,
    aten.sigmoid_backward.default,
    aten.tanh_backward.default,
    aten.threshold_backward.default,
    aten.hardswish_backward.default,
    aten.hardsigmoid_backward.default,
    aten.leaky_relu_backward.default,
    aten.mish_backward.default,
    aten.native_dropout_backward.default,
    # lerp.Scalar is binary (third arg is a scalar weight, forwarded via *extra)
    aten.lerp.Scalar,
]
