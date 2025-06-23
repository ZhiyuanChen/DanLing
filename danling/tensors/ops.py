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
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .nested_tensor import NestedTensor


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_MISSING = object()  # Sentinel for 'argument not provided', distinct from None.


class _ExecutionGuardKind(Enum):
    ITERATION = auto()
    STORAGE_MAP = auto()
    EAGER_FALLBACK = auto()
    PADDED_MATERIALIZATION = auto()
    DENSE_REPACK = auto()


# Consider add `slots=True` when we deprecate Python 3.9
@dataclass(frozen=True)
class _ExecutionGuard:
    forbid_iteration: bool = False
    forbid_storage_map: bool = False
    forbid_eager_fallback: bool = False
    forbid_padded_materialization: bool = False
    forbid_dense_repack: bool = False


_EXECUTION_GUARD: ContextVar[_ExecutionGuard | None] = ContextVar("_EXECUTION_GUARD", default=None)


def _check_execution_guard(kind: _ExecutionGuardKind, detail: str) -> None:
    r"""Raise when a guarded slow path is touched inside ``nested_execution_guard``."""
    guard = _EXECUTION_GUARD.get()
    if guard is None:
        return
    if kind is _ExecutionGuardKind.ITERATION and guard.forbid_iteration:
        raise RuntimeError(f"NestedTensor hot path unexpectedly iterated storage via {detail}")
    if kind is _ExecutionGuardKind.STORAGE_MAP and guard.forbid_storage_map:
        raise RuntimeError(f"NestedTensor hot path unexpectedly used storage mapping via {detail}")
    if kind is _ExecutionGuardKind.EAGER_FALLBACK and guard.forbid_eager_fallback:
        raise RuntimeError(f"NestedTensor hot path unexpectedly entered eager fallback via {detail}")
    if kind is _ExecutionGuardKind.PADDED_MATERIALIZATION and guard.forbid_padded_materialization:
        raise RuntimeError(f"NestedTensor hot path unexpectedly materialized padded storage via {detail}")
    if kind is _ExecutionGuardKind.DENSE_REPACK and guard.forbid_dense_repack:
        raise RuntimeError(f"NestedTensor hot path unexpectedly repacked from dense storage via {detail}")


@contextmanager
def nested_execution_guard(
    *,
    forbid_iteration: bool = False,
    forbid_storage_map: bool = False,
    forbid_eager_fallback: bool = False,
    forbid_padded_materialization: bool = False,
    forbid_dense_repack: bool = False,
):
    r"""
    Temporarily forbid selected slow paths while exercising NestedTensor hot paths.

    This is intended for transformer-critical regression checks, where falling
    back to Python loops or padded materialization is considered a bug.
    """
    current = _EXECUTION_GUARD.get()
    merged = _ExecutionGuard(
        forbid_iteration=forbid_iteration or (current.forbid_iteration if current is not None else False),
        forbid_storage_map=forbid_storage_map or (current.forbid_storage_map if current is not None else False),
        forbid_eager_fallback=forbid_eager_fallback
        or (current.forbid_eager_fallback if current is not None else False),
        forbid_padded_materialization=forbid_padded_materialization
        or (current.forbid_padded_materialization if current is not None else False),
        forbid_dense_repack=forbid_dense_repack or (current.forbid_dense_repack if current is not None else False),
    )
    token: Token[_ExecutionGuard | None] = _EXECUTION_GUARD.set(merged)
    try:
        yield
    finally:
        _EXECUTION_GUARD.reset(token)


def _is_compiling() -> bool:
    r"""Return whether execution is currently happening under ``torch.compile`` tracing."""
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        if hasattr(compiler, "is_dynamo_compiling"):
            return bool(compiler.is_dynamo_compiling())
        if hasattr(compiler, "is_compiling"):
            return bool(compiler.is_compiling())
    return bool(torch._dynamo.is_compiling())


def _compile_unsupported(op_name: str, detail: str | None = None) -> None:
    r"""Raise a clear error for NestedTensor paths that are intentionally eager-only under compile."""
    suffix = f": {detail}" if detail else ""
    raise NotImplementedError(f"NestedTensor compile-safe path not implemented for {op_name}{suffix}")


class TorchFuncRegistry(dict):
    r"""
    Plain dict mapping functions/ops to their NestedTensor handlers.

    Uses ``dict`` directly for O(1) lookup with minimal overhead (~30 ns)
    instead of chanfig.Registry (~700-2300 ns).

    Used for both ``__torch_function__`` (torch/nn ops) and
    ``__torch_dispatch__`` (aten ops) dispatch tables.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compile_safe: dict[Callable, bool] = {}
        self._compile_guard: dict[Callable, Callable[[tuple, dict[str, object]], bool]] = {}

    def register(
        self,
        func: Callable,
        handler: Callable,
        *,
        compile_safe: bool = False,
        compile_guard: Callable[[tuple, dict[str, object]], bool] | None = None,
    ) -> Callable:
        r"""Register *handler* for *func* and record whether the path is compile-safe by default."""
        self[func] = handler
        self._compile_safe[func] = bool(compile_safe)
        if compile_guard is not None:
            self._compile_guard[func] = compile_guard
        else:
            self._compile_guard.pop(func, None)
        return handler

    def implement(
        self,
        func: Callable,
        *,
        compile_safe: bool = False,
        compile_guard: Callable[[tuple, dict[str, object]], bool] | None = None,
    ) -> Callable:
        r"""Decorator to register a handler for *func*."""

        def wrapper(handler: Callable) -> Callable:
            return self.register(func, handler, compile_safe=compile_safe, compile_guard=compile_guard)

        return wrapper

    def is_compile_safe(
        self, func: Callable, args: tuple | None = None, kwargs: dict[str, object] | None = None
    ) -> bool:
        r"""Return whether *func* is allowed to run while ``torch.compile`` is tracing."""
        if not bool(self._compile_safe.get(func, False)):
            return False
        guard = self._compile_guard.get(func)
        if guard is None or args is None:
            return True
        return bool(guard(args, kwargs or {}))

    def set_compile_safe(self, func: Callable, compile_safe: bool = True) -> None:
        r"""Update compile policy for an already-registered handler."""
        if func not in self:
            raise KeyError(f"{func} is not registered")
        self._compile_safe[func] = bool(compile_safe)

    def set_compile_guard(self, func: Callable, guard: Callable[[tuple, dict[str, object]], bool] | None) -> None:
        r"""Set or clear the runtime compile guard for an already-registered handler."""
        if func not in self:
            raise KeyError(f"{func} is not registered")
        if guard is None:
            self._compile_guard.pop(func, None)
        else:
            self._compile_guard[func] = guard

    def get_compile_guard(self, func: Callable) -> Callable[[tuple, dict[str, object]], bool] | None:
        r"""Return the runtime compile guard for *func*, if any."""
        return self._compile_guard.get(func)


def _bind_fn(handler: Callable, fn: Callable) -> Callable:
    r"""Bind ``_fn`` into a handler closure for table-driven registration."""

    def _bound(*args, **kwargs):
        return handler(*args, _fn=fn, **kwargs)

    name = getattr(fn, "__qualname__", getattr(fn, "__name__", None))
    _bound.__name__ = getattr(fn, "__name__", handler.__name__)
    _bound.__qualname__ = getattr(fn, "__qualname__", handler.__qualname__)
    _bound.__doc__ = f"NestedTensor override for ``{name}``. See the original for argument docs."
    _bound.__wrapped__ = fn  # type: ignore[attr-defined]
    return _bound


#: ``__torch_function__`` dispatch table for ``torch.*`` and ``F.*`` ops.
NestedTensorFuncRegistry = TorchFuncRegistry()

#: ``__torch_dispatch__`` dispatch table for aten ops.
NestedTensorAtenRegistry = TorchFuncRegistry()


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


def _maybe_align_dense_to_nested(ref: NestedTensor, value) -> NestedTensor | None:
    r"""
    Convert an exact-shape dense tensor to ``ref``'s NestedTensor layout.

    This is the shared policy boundary for dense-to-nested alignment:
    only dense tensors with logical shape exactly matching ``ref.shape`` are
    converted via ``_maybe_exact_shape_nested_like``.
    """
    cls = type(ref)
    if isinstance(value, cls):
        return value
    return ref._maybe_exact_shape_nested_like(value)


def _binary_op_maybe_tensor(input, other, op, *extra_args, **extra_kwargs):
    r"""
    Apply a binary op between a NestedTensor and a tensor/scalar/NestedTensor.

    Performance notes:
    - **Scalar or 0-dim tensor ``other``**: O(1) — op runs directly on packed
      ``_values`` with no unpack/repack overhead. This is the common training path.
    - **Matched-layout NestedTensor ``other``**: O(1) — packed-layout fast-path,
      op runs on ``_values`` directly.
    - **Mismatched-offset NestedTensor ``other``**: O(B) — iterates over ``_storage``
      and constructs a new NestedTensor from individual results.
    - **Dense tensor ``other`` with shape matching ``input.shape``**: converted via
      ``_maybe_exact_shape_nested_like`` internally, which has O(B * max_len) cost
      from repacking the dense tensor to match the packed layout. Avoid in hot paths.
    """
    from .aten_functions import _packed_like
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
        return _packed_like(input, new_values)

    # NT + scalar or 0-d tensor (most common in training)
    if not isinstance(other, Tensor) or other.dim() == 0:
        val = _as_tensor_like(other, input._values)
        new_values = (
            op(val, input._values, *extra_args, **extra_kwargs)
            if reverse
            else op(input._values, val, *extra_args, **extra_kwargs)
        )
        return _packed_like(input, new_values)

    # Convert padded tensor to NT if shapes match
    aligned_other = _maybe_align_dense_to_nested(input, other)
    if aligned_other is not None:
        other = aligned_other

    # NT + NT
    if isinstance(other, cls):
        if len(input) != len(other):
            raise ValueError(
                "NestedTensor batch length mismatch between input and other: " f"input={len(input)}, other={len(other)}"
            )
        lhs_v, rhs_v = (other._values, input._values) if reverse else (input._values, other._values)
        if input._has_same_structure(other):
            return _packed_like(input, op(lhs_v, rhs_v, *extra_args, **extra_kwargs))
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


def _binary_op_compile_safe(args: tuple, kwargs: dict[str, object]) -> bool:
    r"""Return whether a torch-level binary op stays on a packed fast path."""
    from .nested_tensor import NestedTensor

    if len(args) < 2:
        return True

    lhs, rhs = args[0], args[1]
    input = lhs if isinstance(lhs, NestedTensor) else rhs if isinstance(rhs, NestedTensor) else None
    other = rhs if input is lhs else lhs
    if input is None:
        return True

    if not isinstance(other, Tensor) or other.dim() == 0:
        return True

    aligned_other = _maybe_align_dense_to_nested(input, other)
    if aligned_other is not None:
        return input._has_same_structure(aligned_other)

    if isinstance(other, NestedTensor):
        return len(input) == len(other) and input._has_same_structure(other)

    return False


def _broadcast_storage(ref: NestedTensor, value):
    r"""Broadcast a value to match ref's per-element storage layout."""
    cls = type(ref)
    if isinstance(value, cls):
        if len(ref) != len(value):
            raise ValueError(
                "NestedTensor batch length mismatch between ref and value: " f"ref={len(ref)}, value={len(value)}"
            )
        return value._storage
    aligned = _maybe_align_dense_to_nested(ref, value)
    if aligned is not None:
        return aligned._storage
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


# Concatenation Helpers


def _concat_apply(
    input: NestedTensor,
    op: Callable[[Tensor], Tensor],
    shape_fn: Callable[[torch.Size], torch.Size],
):
    r"""Apply op to concatenated storage and split back using shape_fn."""
    cls = type(input)
    if len(input) == 0:
        return cls([], **input._meta(include_dtype=True))

    concat, shapes = input.concatenate()
    if not shapes:
        return cls([], **input._meta(include_dtype=True))

    output = op(concat)
    output_shapes = tuple(shape_fn(shape) for shape in shapes)
    return cls.from_concatenated(output, output_shapes, **input._meta())


def _concat_apply_same_shape(input: NestedTensor, op: Callable[[Tensor], Tensor]):
    r"""Apply a shape-preserving op directly to packed _values."""
    from .aten_functions import _packed_like

    if len(input) == 0:
        return type(input)([], **input._meta(include_dtype=True))
    return _packed_like(input, op(input._values))


def _static_dim_mask_from_element_shapes(
    element_shapes: tuple[tuple[int, ...], ...],
    physical_rank: int,
) -> tuple[bool, ...]:
    r"""Infer which per-element dims are static using cached Python metadata only."""
    if physical_rank <= 0 or not element_shapes:
        return ()

    static_dims: list[bool] = []
    for dim in range(physical_rank):
        sizes = [shape[dim] if dim < len(shape) else 0 for shape in element_shapes]
        if max(sizes) == 0:
            break
        first = sizes[0]
        static_dims.append(all(size == first for size in sizes))
    return tuple(static_dims)


def _concat_dim_for_tensor_dim(input: NestedTensor, dim: int) -> int | None:
    r"""Map a per-element tensor dim to the corresponding concatenated tensor dim, or None."""
    st = input._physical_shape
    if st.numel() == 0:
        return None
    if input._element_shapes is not None:
        static_dims = _static_dim_mask_from_element_shapes(input._element_shapes, int(st.size(1)))
    else:
        if _is_compiling():
            _compile_unsupported(
                "NestedTensor metadata analysis",
                "compile-safe dimension mapping requires cached python element_shapes metadata",
            )
        # Compute which per-element dims have uniform size across all elements.
        # A dim is static iff all elements have the same size along that dim.
        # Strip trailing zero-padded columns (from elements with fewer dims).
        ncols = st.size(1)
        static_dims_list: list[bool] = []
        for d in range(ncols):
            col = st[:, d]
            if col.max().item() == 0:
                break  # trailing zero-pad
            static_dims_list.append(col.min().item() == col.max().item())
        static_dims = tuple(static_dims_list)
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


def _batch_leading_valid_mask_from_sizes(
    sizes: Tensor,
    non_batch_shape: Sequence[int],
    *,
    device,
) -> Tensor:
    r"""Build a batch-leading boolean validity mask from per-element physical sizes."""
    batch_size = int(sizes.shape[0])
    non_batch_shape = tuple(int(size) for size in non_batch_shape)
    if batch_size == 0:
        return torch.empty((0, *non_batch_shape), dtype=torch.bool, device=device)
    if not non_batch_shape:
        return torch.ones((batch_size,), dtype=torch.bool, device=device)

    sizes = sizes.to(device=device, dtype=torch.long)
    valid = torch.ones((batch_size, *non_batch_shape), dtype=torch.bool, device=device)
    size_view = [batch_size] + [1] * len(non_batch_shape)
    for dim, max_size in enumerate(non_batch_shape):
        coord_shape = [1] * (len(non_batch_shape) + 1)
        coord_shape[dim + 1] = max_size
        coord = torch.arange(max_size, device=device, dtype=torch.long).view(coord_shape)
        valid &= coord < sizes[:, dim].view(size_view)
    return valid


# ---------------------------------------------------------------------------
# Elementwise op lists
# ---------------------------------------------------------------------------
# Both the torch level (torch_functions.py) and aten level
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
    torch.square,
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
    torch.digamma,
    torch.lgamma,
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
    aten.square.default,
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
    aten.digamma.default,
    aten.lgamma.default,
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
    return _stack_or_nest(results, input)


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
    return _stack_or_nest(ret, input)


def _reduce(
    input: NestedTensor,
    op,
    dim,
    keepdim: bool,
    *,
    dtype: torch.dtype | None = None,
    fill_value=_MISSING,
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
    if fill_value is not _MISSING:
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
    return _stack_or_nest(firsts, input), _stack_or_nest(seconds, input)


# Normalization & Validation Helpers


def _normalize_shape_tuple(normalized_shape) -> tuple[int, ...]:
    r"""Normalize a ``normalized_shape`` argument to a tuple."""
    return (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)


def _can_concat_normalize(input: NestedTensor, normalized_shape: tuple[int, ...]) -> bool:
    r"""Return whether normalization can run directly on packed ``_values``."""
    if not normalized_shape:
        return True
    element_shapes = input._element_shapes
    if element_shapes is None:
        element_shapes = tuple(tuple(int(size) for size in shape) for shape in input._physical_shape.tolist())
    if not element_shapes:
        return True
    ndim = len(normalized_shape)
    return all(len(shape) >= ndim and tuple(shape[-ndim:]) == normalized_shape for shape in element_shapes)


def _packed_layer_norm(
    input: NestedTensor,
    normalized_shape: tuple[int, ...],
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
):
    r"""Run layer norm on packed ``_values`` when the normalized tail is static."""
    if not _can_concat_normalize(input, normalized_shape):
        return None

    from .aten_functions import _packed_like

    try:
        output, _, _ = torch.ops.aten.native_layer_norm.default(input._values, normalized_shape, weight, bias, eps)
    except RuntimeError:
        return None
    return _packed_like(input, output)


def _packed_rms_norm(input: NestedTensor, normalized_shape: tuple[int, ...], weight: Tensor | None, eps):
    r"""Run RMS norm on packed ``_values`` when the normalized tail is static."""
    if not _can_concat_normalize(input, normalized_shape):
        return None

    from .aten_functions import _packed_like

    try:
        output = torch.ops.aten.rms_norm.default(input._values, normalized_shape, weight, eps)
    except RuntimeError:
        return None
    return _packed_like(input, output)


def _run_layer_norm(
    input: NestedTensor,
    normalized_shape,
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
    *,
    op_name: str,
    fallback: Callable[[Tensor, tuple[int, ...], Tensor | None, Tensor | None, float], Tensor],
):
    r"""Run layer norm with one shared NestedTensor policy boundary."""
    normalized = _normalize_shape_tuple(normalized_shape)
    output = _packed_layer_norm(input, normalized, weight, bias, eps)
    if output is not None:
        return output
    if _is_compiling():
        _compile_unsupported(op_name, "only static normalized tails that stay on packed storage are compile-safe")
    return _map_storage_serial(input, lambda t: fallback(t, normalized, weight, bias, eps))


def _run_rms_norm(
    input: NestedTensor,
    normalized_shape,
    weight: Tensor | None,
    eps: float | None,
    *,
    op_name: str,
    fallback: Callable[[Tensor, tuple[int, ...], Tensor | None, float | None], Tensor],
):
    r"""Run RMS norm with one shared NestedTensor policy boundary."""
    normalized = _normalize_shape_tuple(normalized_shape)
    output = _packed_rms_norm(input, normalized, weight, eps)
    if output is not None:
        return output
    if _is_compiling():
        _compile_unsupported(op_name, "only static normalized tails that stay on packed storage are compile-safe")
    return _map_storage_serial(input, lambda t: fallback(t, normalized, weight, eps))


def _validate_probability(p: float, *, error_type: type[Exception]) -> None:
    r"""Validate a dropout probability with the caller's API-specific exception type."""
    if p < 0.0 or p > 1.0:
        raise error_type(f"dropout probability has to be between 0 and 1, but got {p}")


# Storage Mapping


def _stack_or_nest(values: Sequence, input: NestedTensor):
    r"""Stack results into a tensor, falling back to NestedTensor if shapes differ."""
    values = list(values)
    if not values:
        return type(input)([], **input._meta(include_dtype=True))
    if all(isinstance(value, Tensor) for value in values):
        first = values[0]
        if all(value.shape == first.shape for value in values[1:]):
            return torch.stack(values)
    return type(input)(values, **input._meta())


@torch._dynamo.disable
def _map_storage_serial(input: NestedTensor, fn):
    r"""Apply fn to each element in storage serially."""
    _check_execution_guard(_ExecutionGuardKind.STORAGE_MAP, "_map_storage_serial")
    cls = type(input)
    elements = input._storage
    if not elements:
        return cls([], **input._meta(include_dtype=True))
    return cls((fn(t) for t in elements), **input._meta())


def _map_storage_pair(input: NestedTensor, op, *args, **kwargs):
    r"""Apply *op* to every element, unpacking each 2-tuple result into two NestedTensors."""
    _check_execution_guard(_ExecutionGuardKind.STORAGE_MAP, "_map_storage_pair")
    cls = type(input)
    if len(input) == 0:
        try:
            first_probe, second_probe = op(input._values, *args, **kwargs)
            first_dtype = first_probe.dtype if isinstance(first_probe, Tensor) else input.dtype
            second_dtype = second_probe.dtype if isinstance(second_probe, Tensor) else input.dtype
        except (TypeError, RuntimeError, ValueError):
            first_dtype = input.dtype
            second_dtype = input.dtype
        return (
            cls([], dtype=first_dtype, **input._meta(include_dtype=False)),
            cls([], dtype=second_dtype, **input._meta(include_dtype=False)),
        )
    firsts, seconds = [], []
    for t in input._storage:
        a, b = op(t, *args, **kwargs)
        firsts.append(a)
        seconds.append(b)
    return cls(firsts, **input._meta()), cls(seconds, **input._meta())
