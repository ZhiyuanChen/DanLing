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
from typing import TYPE_CHECKING, NamedTuple

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


def _logical_dim_for_element_dim(nt: NestedTensor, element_dim: int) -> int:
    r"""Map a per-element dimension to its position in the logical (padded) shape."""
    return element_dim if element_dim < _get_batch_dim(nt) else element_dim + 1


def _packed_static_extents(nt: NestedTensor) -> tuple[tuple[int, int], ...]:
    r"""Pair every static per-element dim with the extent its packed axis carries."""
    return tuple((dim, int(nt._values.shape[1 + axis])) for axis, dim in enumerate(nt._static_dims))


def _padded_ragged_extent(nt: NestedTensor, element_dim: int) -> int | None:
    r"""Return the padded extent of a ragged per-element dim, or None when it is unavailable."""
    from .aten_functions import _is_fake_tensor

    if nt._element_shapes is not None:
        return max(int(shape[element_dim]) for shape in nt._element_shapes)
    if _is_compiling() or _is_fake_tensor(nt._offsets):
        return None
    return int(nt.shape[_logical_dim_for_element_dim(nt, element_dim)])


def _dense_alignment_is_valid(nt: NestedTensor, aligned: tuple[int, ...]) -> bool:
    r"""Return whether a logical-shape alignment can be replayed on the packed values."""
    for dim, extent in _packed_static_extents(nt):
        size = aligned[_logical_dim_for_element_dim(nt, dim)]
        if size != 1 and extent != 1 and size != extent:
            return False
    for dim in nt._ragged_dims:
        size = aligned[_logical_dim_for_element_dim(nt, dim)]
        if size == 1:
            continue
        # A non-singleton extent addresses positions *along* a ragged axis, which is only a
        # meaning when the operand also spells out the batch it is positioned within: a purely
        # positional broadcast such as ``nt[B, ragged, H] + dense[1, L, H]`` is refused. Reading
        # it per row needs a row coordinate, which only a single ragged level has.
        if aligned[_get_batch_dim(nt)] != len(nt) or len(nt._ragged_dims) != 1:
            return False
        if size != _padded_ragged_extent(nt, dim):
            return False
    return True


def _logical_dense_alignment(nt: NestedTensor, shape: tuple[int, ...]) -> tuple[int, ...] | None:
    r"""
    Right-align a dense shape the way torch broadcasts, and say where the batch axis fell.

    An operand carries a batch axis only at full logical rank; anything shorter right-aligns
    onto the *per-element* dimensions and broadcasts identically into every sample. Those two
    coincide while the batch leads, and part company under ``batch_first=False``, where the
    logical shape holds the batch between element dims and a shorter operand would otherwise
    have to land on it.
    """
    batch_dim = _get_batch_dim(nt)
    rank = nt.dim()
    while len(shape) > rank and shape[0] == 1:
        shape = shape[1:]
    if len(shape) > rank:
        return None
    if len(shape) == rank:
        return shape if shape[batch_dim] in (1, len(nt)) else None
    aligned = list((1,) * (rank - 1 - len(shape)) + shape)
    aligned.insert(batch_dim, 1)
    return tuple(aligned)


def _metadata_dense_alignment(nt: NestedTensor, shape: tuple[int, ...]) -> tuple[int, ...] | None:
    r"""Read a dense shape as one static-tail slab per sample, expressed as a logical alignment.

    The reading exists only to let an operand address the batch, so it requires an axis that
    actually does: one leading axis of extent ``B`` on top of the static tail. Without that the
    reading would just be right-alignment with the ragged dims skipped, which is not a meaning
    any caller asked for and would collide with the logical reading on every shorter operand.
    A bare vector is a tail and never one scalar per sample, however its length compares to the
    batch size -- that collision is the cheapest one to make by accident. An operand at full
    logical rank has already named every dimension, so there is nothing left for this reading to
    supply and it steps aside rather than competing with what the caller spelled out.
    """
    static_dims = sorted(nt._static_dims)
    if len(shape) == nt.dim() or len(shape) < 2:
        return None
    if len(shape) <= len(static_dims) or shape[0] != len(nt):
        return None
    tail = shape[1:]
    # Singleton axes between the batch axis and the tail carry no positional information.
    while len(tail) > len(static_dims) and tail[0] == 1:
        tail = tail[1:]
    if len(tail) > len(static_dims):
        return None
    aligned = [1] * nt.dim()
    aligned[_get_batch_dim(nt)] = shape[0]
    tail = (1,) * (len(static_dims) - len(tail)) + tail
    for position, dim in enumerate(static_dims):
        aligned[_logical_dim_for_element_dim(nt, dim)] = tail[position]
    return tuple(aligned)


class _DenseReading(NamedTuple):
    r"""What a dense operand means against a NestedTensor.

    ``aligned`` places the operand's extents on the *logical* dimensions. ``batch_leads`` says
    the operand spells the batch first in its own buffer, which is what the one-slab-per-sample
    reading always does however the layout orders its dimensions: reaching logical order from
    there is a move, not a reinterpretation.
    """

    aligned: tuple[int, ...]
    batch_leads: bool


def _dense_alignment(nt: NestedTensor, other: Tensor) -> _DenseReading | None:
    r"""
    Decide what a dense operand means against ``nt``, or refuse when it means two things.

    Two readings can fit the same dense shape. The **logical** reading right-aligns the operand
    onto ``nt.shape`` exactly as :func:`torch.broadcast_shapes` would, so its leading axis lands
    on the batch dimension only when the operand carries an axis there. The **metadata** reading
    treats a rank-deficient operand as one static-tail slab per sample, which is how a
    ``[B, H, D]`` bias against ``[B, ragged, H, D]`` values is meant to be read; a ragged axis
    has no fixed extent, so this reading never places anything on one.

    Where the two agree there is nothing to decide. Where only one of them describes a shape the
    packed values can serve, that one is the answer. Where both describe different results the
    operand is genuinely ambiguous -- a ``[2, 1, D]`` operand against elements ``[H, ragged, D]``
    whose ``H`` is also 2 puts its leading axis on the batch under one reading and on ``H`` under
    the other -- and guessing is how a silently wrong answer gets produced, so it raises instead.
    """
    shape = tuple(int(size) for size in other.shape)
    logical = _logical_dense_alignment(nt, shape)
    if logical is not None and not _dense_alignment_is_valid(nt, logical):
        logical = None
    metadata = _metadata_dense_alignment(nt, shape)
    if metadata is not None and not _dense_alignment_is_valid(nt, metadata):
        metadata = None
    if logical is not None and metadata is not None and logical != metadata:
        raise NotImplementedError(
            f"NestedTensor: dense operand of shape {shape} is ambiguous against logical shape "
            f"{tuple(nt.shape)}: it right-aligns as {logical} and also reads as one slab per "
            f"sample {metadata}. Reshape the operand to the reading you mean."
        )
    if logical is not None:
        return _DenseReading(logical, False)
    return None if metadata is None else _DenseReading(metadata, True)


def _dense_reading_batch_first(nt: NestedTensor, reading: _DenseReading, other: Tensor) -> Tensor:
    r"""View an aligned dense operand as ``[batch, *element dims]``, in element order.

    The logical alignment is the operand's own axes with singletons inserted, so a reshape
    reaches it -- *except* under the one-slab-per-sample reading of a layout whose batch is not
    the leading logical dimension. There the alignment holds a static extent in front of the
    batch while the operand holds the batch in front of everything, and reinterpreting the
    buffer would hand each sample the slab of whichever sample shares its column.
    """
    aligned = reading.aligned
    batch_dim = _get_batch_dim(nt)
    batch_first = (aligned[batch_dim], *aligned[:batch_dim], *aligned[batch_dim + 1 :])
    if reading.batch_leads:
        return other.reshape(batch_first)
    return other.reshape(aligned).movedim(batch_dim, 0)


def _dense_alignment_to_values(nt: NestedTensor, other: Tensor, reading: _DenseReading) -> Tensor | None:
    r"""Rewrite an aligned dense operand into the packed axis order of ``nt._values``."""
    view = _dense_reading_batch_first(nt, reading, other)
    view = view.permute((0, *(1 + dim for dim in nt._permutation)))
    ragged_rank = len(nt._ragged_dims)
    batch_extent = int(view.shape[0])
    ragged_extents = tuple(int(size) for size in view.shape[1 : 1 + ragged_rank])
    tail = tuple(int(size) for size in view.shape[1 + ragged_rank :])
    if all(extent == 1 for extent in ragged_extents):
        view = view.reshape(batch_extent, *tail)
        if batch_extent == 1:
            return view
        return view.index_select(0, nt.packed_batch_indices(device=other.device))
    if ragged_rank != 1:
        return None
    ragged_extent = ragged_extents[0]
    rows = nt.packed_local_indices(0, device=other.device)
    if batch_extent != 1:
        rows = nt.packed_batch_indices(device=other.device) * ragged_extent + rows
    return view.reshape(batch_extent * ragged_extent, *tail).index_select(0, rows)


def _resolve_dense_for_values(nt: NestedTensor, other) -> Tensor | None:
    r"""
    Resolve a dense tensor into a form that can operate directly with ``_values``.

    The packed axes are the per-element dims permuted, with every ragged dim collapsed into axis
    0, so a dense operand cannot be handed to a kernel running on ``_values`` as it stands: it
    has to be read against the *logical* shape first and only then rewritten into packed order.
    :func:`_dense_alignment` does the reading and refuses ambiguous shapes; this function does
    the rewriting, one permutation plus at most one ``index_select``, for any packed layout.

    A dense operand shaped exactly like the packed values is elementwise on them, which is what
    a danling op that concatenated this same tensor produces (e.g. ``F.cross_entropy`` with
    ``reduction="none"``). That match covers packed dim 0, whose extent is data-dependent, so it
    cannot be a coincidental collision with a ragged maximum and is taken before any alignment.

    Returns ``None`` when no reading applies; raises when more than one does.
    """
    if not isinstance(other, Tensor) or other.dim() == 0:
        return None

    if len(nt) == 0 or nt._values.dim() == 0:
        return None

    if other.shape == nt._values.shape:
        return other

    reading = _dense_alignment(nt, other)
    if reading is None:
        return None
    return _dense_alignment_to_values(nt, other, reading)


def _dense_operand_for_element(input: NestedTensor, other: Tensor, index: int, element: Tensor) -> Tensor | None:
    r"""
    Align a dense operand to one element: pick that sample, then trim its padded ragged axes.

    This is the per-element counterpart of :func:`_dense_alignment` and reads a dense shape the
    same way: the batch dimension participates only when the operand carries an axis there, and
    an axis whose extent is the padded maximum of a ragged dim names positions along it, so the
    element takes the leading slice of that axis. A ``[B, *static_tail]`` operand is one slab per
    sample here too -- it is the same contract, and ``torch.broadcast_tensors`` reaching this
    helper rather than the packed resolver must not make it a different one. Returns ``None``
    when nothing aligns.
    """
    batch_dim = _get_batch_dim(input)
    reading = _dense_alignment(input, other)
    if reading is not None and reading.batch_leads:
        other = _dense_reading_batch_first(input, reading, other).select(0, index)
    if other.dim() == input.dim():
        extent = other.shape[batch_dim]
        if extent == len(input):
            other = other.select(batch_dim, index)
        elif extent == 1:
            other = other.select(batch_dim, 0)
        else:
            return None
    elif other.dim() > input.dim():
        return None
    if other.dim() != element.dim():
        return other
    trimmed = []
    for dim, (size, extent) in enumerate(zip(other.shape, element.shape)):
        if size == extent or size == 1:
            trimmed.append(slice(None))
        elif dim in input._varying_dims and size == _padded_ragged_extent(input, dim):
            trimmed.append(slice(0, int(extent)))
        else:
            return None
    return other[tuple(trimmed)]


def _nested_like_elements(input: NestedTensor, elements) -> NestedTensor:
    r"""
    Rebuild a NestedTensor from per-element results without losing a declared topology.

    ``ragged_dims`` is a *declaration*, not an observation: a batch whose samples happen to have
    equal extents, or a batch of one, offers nothing to re-infer it from, so rebuilding without
    it silently moves the ragged dim onto whichever dim leads the element shapes. Carry the
    declaration across whenever the results still have the rank it describes.
    """
    cls = type(input)
    elements = tuple(elements)
    rank = int(input._physical_shape.size(1))
    ragged_dims = (
        input._ragged_dims
        if input._ragged_dims_explicit and all(element.dim() == rank for element in elements)
        else None
    )
    return cls(
        elements,
        batch_first=input.batch_first,
        padding_value=input.padding_value,
        mask_value=input.mask_value,
        ragged_dims=ragged_dims,
    )


def _binary_per_element_dense(
    input,
    other,
    op,
    reverse,
    extra_args,
    extra_kwargs,
):
    r"""Apply a dense binary op per logical element (layout-correct for permuted NTs).

    Aligns ``other`` to each element by slicing its batch dim (or broadcasting when it has
    size 1 / no batch dim), then relies on standard tensor broadcasting against the element.
    Returns ``None`` (caller falls back) if shapes are incompatible or the op would change an
    element's shape.
    """
    _check_execution_guard(_ExecutionGuardKind.EAGER_FALLBACK, "ops._binary_per_element_dense")
    results = []
    for i, elem in enumerate(input._unpack()):
        other_i = _dense_operand_for_element(input, other, i, elem)
        if other_i is None:
            return None
        try:
            result = (
                op(other_i, elem, *extra_args, **extra_kwargs)
                if reverse
                else op(elem, other_i, *extra_args, **extra_kwargs)
            )
        except RuntimeError:
            return None
        if not isinstance(result, Tensor) or result.shape != elem.shape:
            return None
        results.append(result)
    return _nested_like_elements(input, results)


def _binary_dense_padded_broadcast(input, other, op, reverse, extra_args, extra_kwargs):
    r"""Apply a binary op when ``input`` broadcasts its singleton dims up to a dense ``other``.

    This is the symmetric counterpart of the exact-shape dense path: there a dense tensor whose
    logical shape *equals* ``input.shape`` is nested via ``nested_like``; here ``input`` carries
    size-1 dims (e.g. a ``[B, L, 1]`` mask) and ``other`` is the full padded shape it broadcasts
    up to (e.g. ``[B, L, H]``), so the result element shape is the broadcast ``[L, H]``.

    The gate ``broadcast_shapes(input.shape, other.shape) == other.shape`` is what keeps this from
    re-enabling *positional* dense broadcast (e.g. ``nt[B, ragged, H] + dense[1, L, H]``, which
    danling deliberately rejects): there ``other`` broadcasts on the batch dim, so it is not the
    full shape and the gate fails. Used only as a last resort, so the hot paths are unaffected.
    Returns ``None`` when ``other`` is not unambiguously alignable.
    """
    if not isinstance(other, Tensor) or other.dim() == 0:
        return None
    if input.dim() == 0 or other.dim() != input.dim():
        return None
    # Only the NestedTensor may broadcast up: `other` must be the full broadcast shape.
    try:
        broadcast_shape = torch.broadcast_shapes(tuple(input.shape), tuple(other.shape))
    except RuntimeError:
        return None
    if tuple(broadcast_shape) != tuple(other.shape):
        return None
    batch_dim = _get_batch_dim(input)
    if other.shape[batch_dim] != len(input):
        return None
    _check_execution_guard(_ExecutionGuardKind.EAGER_FALLBACK, "ops._binary_dense_padded_broadcast")
    varying_dims = input._varying_dims
    results = []
    for i, elem in enumerate(input._unpack()):
        other_i = other.select(batch_dim, i)
        if other_i.dim() != elem.dim():
            return None
        slices = []
        for dim, (size, elem_size) in enumerate(zip(other_i.shape, elem.shape)):
            if dim in varying_dims:
                if size == elem_size:
                    slices.append(slice(None))
                elif size > elem_size:
                    slices.append(slice(0, int(elem_size)))  # trim the ragged dim to this element
                else:
                    return None
            elif size == elem_size or elem_size == 1:
                slices.append(slice(None))  # static dim: equal, or element broadcasts up
            else:
                return None
        other_i = other_i[tuple(slices)]
        try:
            result = (
                op(other_i, elem, *extra_args, **extra_kwargs)
                if reverse
                else op(elem, other_i, *extra_args, **extra_kwargs)
            )
        except RuntimeError:
            return None
        if not isinstance(result, Tensor):
            return None
        results.append(result)
    return _nested_like_elements(input, results)


def _can_broadcast_nested_to(
    target: NestedTensor,
    source: NestedTensor,
) -> bool:
    r"""Return whether ``source`` can be aligned to ``target`` in packed coordinates.

    The supported layouts include an ordinary per-element broadcast such as
    ``(M_i, N_i, C) + (1, N_i, C)`` and an equal-rank view whose ragged axes
    have a different packed order, such as ``pair + pair.transpose(-2, -3)``.
    ``source`` may replace one or more ragged levels with singleton static
    dimensions while retaining the other ragged coordinates, or retain every
    ragged dimension in another order. Its remaining static dimensions may be
    singleton. Binary operators may also
    opt into broadcasting singleton static dimensions in ``target``, for
    example a unit-sample ``(1, N_i, 1)`` target and ``(1, N_i, C)`` source
    broadcasting to sampled ``(S, N_i, C)`` coordinates.
    Tensor-backed size equality is validated by
    :func:`_broadcast_nested_to_values`; this predicate intentionally uses only
    static layout descriptors so it is safe in a compile guard.
    """
    if target.batch_first != source.batch_first:
        return False
    if target._physical_shape.size(1) != source._physical_shape.size(1):
        return False

    target_ragged = target._ragged_dims
    source_ragged = source._ragged_dims
    if not source_ragged or len(source_ragged) > len(target_ragged):
        return False
    source_ragged_set = set(source_ragged)
    retained_target_ragged = tuple(dim for dim in target_ragged if dim in source_ragged_set)
    if len(source_ragged) == len(target_ragged):
        if source_ragged_set != set(target_ragged):
            return False
    elif retained_target_ragged != source_ragged:
        return False

    omitted_ragged = tuple(dim for dim in target_ragged if dim not in source_ragged_set)
    source_static = source._static_dims
    if any(dim not in source_static for dim in omitted_ragged):
        return False

    remaining_source_static = tuple(dim for dim in source_static if dim not in omitted_ragged)
    target_static = target._static_dims
    return len(remaining_source_static) == len(target_static) and set(remaining_source_static) == set(target_static)


def _broadcast_condition_matches(condition, *operands: NestedTensor) -> bool:
    r"""Check a retained-shape condition eagerly or preserve it symbolically."""
    from torch.fx.experimental.symbolic_shapes import statically_known_true

    from .aten_functions import _is_fake_tensor

    if statically_known_true(condition):
        return True
    if _is_compiling() or any(_is_fake_tensor(operand._values) for operand in operands):
        torch._check(condition, lambda: "Empty NestedTensor operands have incompatible retained extents")
        return True
    return False


def _broadcast_metadata_matches(
    target: NestedTensor,
    source: NestedTensor,
    omitted_ragged: tuple[int, ...],
    *,
    allow_target_static_broadcast: bool = False,
) -> bool:
    r"""Validate tensor-backed sizes for a packed NestedTensor broadcast."""
    from .aten_functions import _is_fake_tensor

    if len(target) == 0 and len(source) == 0:
        target_extents = target._max_physical_dims()
        source_extents = source._max_physical_dims()

        if any(not _broadcast_condition_matches(source_extents[dim] == 1, target, source) for dim in omitted_ragged):
            return False
        if any(
            not _broadcast_condition_matches(target_extents[dim] == source_extents[dim], target, source)
            for dim in source._ragged_dims
        ):
            return False
        for dim in target._static_dims:
            compatible = (source_extents[dim] == 1) | (target_extents[dim] == source_extents[dim])
            if allow_target_static_broadcast:
                compatible = compatible | (target_extents[dim] == 1)
            if not _broadcast_condition_matches(compatible, target, source):
                return False

    target_shape = target._physical_shape
    source_shape = source._physical_shape
    source_ragged = source._ragged_dims
    target_static = target._static_dims

    conditions: list[Tensor] = []
    conditions.extend(torch.all(source_shape.select(1, dim) == 1) for dim in omitted_ragged)
    conditions.extend(torch.all(target_shape.select(1, dim) == source_shape.select(1, dim)) for dim in source_ragged)
    for dim in target_static:
        source_sizes = source_shape.select(1, dim)
        target_sizes = target_shape.select(1, dim)
        compatible = (source_sizes == 1) | (target_sizes == source_sizes)
        if allow_target_static_broadcast:
            compatible = compatible | (target_sizes == 1)
        conditions.append(torch.all(compatible))
    if not conditions:
        return True

    valid = torch.stack(conditions).all()
    if _is_compiling() or _is_fake_tensor(valid):
        torch._assert_async(
            valid,
            "NestedTensor singleton ragged broadcast requires singleton omitted dimensions, "
            "matching ragged dimensions, and singleton or matching static dimensions",
        )
        return True
    return bool(valid)


def _empty_same_structure_broadcast_matches(lhs: NestedTensor, rhs: NestedTensor) -> bool:
    r"""Validate retained logical extents when an empty batch has no metadata rows."""
    if len(lhs) != 0 or len(rhs) != 0:
        return True

    lhs_extents = lhs._max_physical_dims()
    rhs_extents = rhs._max_physical_dims()

    for dim in range(len(lhs_extents)):
        if dim in lhs._ragged_dims:
            if not _broadcast_condition_matches(lhs_extents[dim] == rhs_extents[dim], lhs, rhs):
                return False
            continue
        compatible = (lhs_extents[dim] == 1) | (rhs_extents[dim] == 1) | (lhs_extents[dim] == rhs_extents[dim])
        if not _broadcast_condition_matches(compatible, lhs, rhs):
            return False
    return True


def _broadcast_nested_to_values(
    target: NestedTensor,
    source: NestedTensor,
    *,
    allow_target_static_broadcast: bool = False,
) -> Tensor | None:
    r"""Align ``source`` with ``target._values`` without padding or storage mapping."""
    if not _can_broadcast_nested_to(target, source):
        return None

    source_ragged_set = set(source._ragged_dims)
    omitted_ragged = tuple(dim for dim in target._ragged_dims if dim not in source_ragged_set)
    if not _broadcast_metadata_matches(
        target,
        source,
        omitted_ragged,
        allow_target_static_broadcast=allow_target_static_broadcast,
    ):
        return None

    device = source._values.device
    batch_indices = target.packed_batch_indices(device=device)
    _, target_local = target._packed_batch_local_indices(device=device, dtype=torch.long)
    target_coords = target._packed_varying_coords(
        batch_indices,
        target_local,
        device=device,
        dtype=torch.long,
    )
    coord_by_dim = dict(zip(target._ragged_dims, target_coords))
    source_shape = source._physical_shape.to(device=device, dtype=torch.long)
    source_local = torch.zeros_like(batch_indices)
    for dim in source._ragged_dims:
        radix = source_shape[:, dim].index_select(0, batch_indices)
        source_local = source_local * radix + coord_by_dim[dim]
    source_offsets = source.packed_offsets(device=device, dtype=source_local.dtype)
    source_indices = source_offsets[batch_indices] + source_local
    values = source._values.index_select(0, source_indices)

    source_static = source._static_dims
    omitted_axes = sorted((1 + source_static.index(dim) for dim in omitted_ragged), reverse=True)
    for axis in omitted_axes:
        values = values.select(axis, 0)

    remaining_source_static = tuple(dim for dim in source_static if dim not in omitted_ragged)
    if remaining_source_static != target._static_dims:
        permutation = (0, *(1 + remaining_source_static.index(dim) for dim in target._static_dims))
        values = values.permute(permutation)
    return values


def _has_same_ragged_structure(target: NestedTensor, source: NestedTensor) -> bool:
    r"""Return whether two NestedTensors number their packed rows identically."""
    from .aten_functions import _is_fake_tensor

    target_offsets = target._hierarchical_offsets or (target._offsets,)
    source_offsets = source._hierarchical_offsets or (source._offsets,)
    if len(target_offsets) != len(source_offsets):
        return False
    for lhs, rhs in zip(target_offsets, source_offsets):
        if lhs.shape != rhs.shape:
            return False
        if not _is_fake_tensor(lhs) and not _is_fake_tensor(rhs) and not bool(torch.equal(lhs, rhs)):
            return False
    return True


def _can_broadcast_lower_rank_nested_to(target: NestedTensor, source: NestedTensor) -> bool:
    r"""Return whether ``source``'s shorter elements right-align into ``target``'s.

    A NestedTensor broadcasts against another the way a dense tensor broadcasts against a dense
    one: its per-element dims right-align onto the wider operand's. That only reaches the packed
    buffers when the two share a row order, so the ragged dims have to land on ``target``'s own
    and the segment lengths have to agree; the remaining dims are then a permutation of a subset
    of ``target``'s static dims, which is a reshape away from ``target``'s tail.
    """
    if target.batch_first != source.batch_first:
        return False
    shift = int(target._physical_shape.size(1)) - int(source._physical_shape.size(1))
    if shift <= 0:
        return False
    if tuple(dim + shift for dim in source._ragged_dims) != target._ragged_dims:
        return False
    target_static = target._static_dims
    return all(dim + shift in target_static for dim in source._static_dims)


def _broadcast_lower_rank_nested_to_values(target: NestedTensor, source: NestedTensor) -> Tensor | None:
    r"""Rewrite ``source._values`` into ``target``'s packed tail without unpacking either."""
    if not _can_broadcast_lower_rank_nested_to(target, source):
        return None
    if not _has_same_ragged_structure(target, source):
        return None

    shift = int(target._physical_shape.size(1)) - int(source._physical_shape.size(1))
    if len(target) == 0 and len(source) == 0:
        target_extents = target._max_physical_dims()
        source_extents = source._max_physical_dims()
        for source_dim in source._ragged_dims:
            condition = target_extents[source_dim + shift] == source_extents[source_dim]
            if not _broadcast_condition_matches(condition, target, source):
                return None

    source_static = source._static_dims
    axes: list[int] = []
    tail: list[int] = []
    for dim in target._static_dims:
        origin = dim - shift
        if origin in source_static:
            axis = 1 + source_static.index(origin)
            axes.append(axis)
            tail.append(int(source._values.shape[axis]))
        else:
            tail.append(1)
    values = source._values.permute((0, *axes))
    return values.reshape(int(values.shape[0]), *tail)


def _single_element_logical_view(input: NestedTensor) -> Tensor:
    r"""View one packed element in logical dimension order without storage mapping."""
    element_shape = tuple(int(size) for size in input._physical_shape[0].tolist())
    packed_shape = tuple(element_shape[dim] for dim in input._permutation)
    packed_view = input._values.reshape(packed_shape)
    inverse_permutation = tuple(input._permutation.index(dim) for dim in range(len(input._permutation)))
    return packed_view.permute(inverse_permutation)


def _binary_single_element_nested_broadcast(
    target: NestedTensor,
    source: NestedTensor,
    op,
    *,
    source_first: bool,
    extra_args,
    extra_kwargs,
):
    r"""Run a one-element broadcast as one dense TensorIterator operation."""
    from .aten_functions import _is_fake_tensor

    if len(target) != 1 or len(source) != 1 or _is_compiling():
        return None
    if _is_fake_tensor(target._values) or _is_fake_tensor(source._values):
        return None
    if not _can_broadcast_nested_to(target, source):
        return None

    omitted_ragged = target._ragged_dims[: -len(source._ragged_dims)]
    if not _broadcast_metadata_matches(
        target,
        source,
        omitted_ragged,
        allow_target_static_broadcast=True,
    ):
        return None

    target_view = _single_element_logical_view(target)
    source_view = _single_element_logical_view(source)
    result = (
        op(source_view, target_view, *extra_args, **extra_kwargs)
        if source_first
        else op(target_view, source_view, *extra_args, **extra_kwargs)
    )
    packed_order = result.permute(target._permutation)
    packed_result = packed_order.flatten(0, target._ragged_rank - 1)
    if packed_result.shape[1:] == target._values.shape[1:]:
        return target._packed_like_unchecked(packed_result)

    from .aten_functions import _packed_with_static_tail_from_values

    return _packed_with_static_tail_from_values(target, packed_result)


def _binary_op_maybe_tensor(input, other, op, *extra_args, **extra_kwargs):
    r"""
    Apply a binary op between a NestedTensor and a tensor/scalar/NestedTensor.

    Performance notes:
    - **Scalar or 0-dim tensor ``other``**: O(1) — op runs directly on packed
      ``_values`` with no unpack/repack overhead. This is the common training path.
    - **Matched-layout NestedTensor ``other``**: O(1) — packed-layout fast-path,
      op runs on ``_values`` directly.
    - **Dense tensor ``other``**: read by ``_dense_alignment`` and rewritten into packed axis
      order, so a tail broadcast costs nothing and a per-sample operand costs one
      ``index_select``. This holds for permuted layouts too; the per-element loop below is
      reached only by shapes no packed reading serves.
    - **Mismatched-layout NestedTensor ``other``**: packed while one operand's elements
      right-align into the other's, and O(B) over ``_unpack()`` otherwise.
    """
    from .aten_functions import _packed_with_static_tail_from_values, _packed_with_tail_from_values
    from .nested_tensor import NestedTensor

    def _rebuild_from(reference, values):
        if tuple(values.shape[1:]) != tuple(reference._values.shape[1:]):
            if values.dim() == reference._values.dim():
                return _packed_with_static_tail_from_values(reference, values)
            return _packed_with_tail_from_values(reference, values)
        return reference._packed_like_unchecked(values)

    def _rebuild(values):
        return _rebuild_from(input, values)

    # Normalize: input is always the NestedTensor
    cls = type(input) if isinstance(input, NestedTensor) else type(other)
    reverse = False
    if not isinstance(input, cls):
        reverse = True
        input, other = other, input

    if len(input) == 0 and not isinstance(other, cls):
        resolved = _as_tensor_like(other, input._values)
        new_values = (
            op(resolved, input._values, *extra_args, **extra_kwargs)
            if reverse
            else op(input._values, resolved, *extra_args, **extra_kwargs)
        )
        return _rebuild(new_values)

    # NT + scalar or 0-d tensor (most common in training)
    if not isinstance(other, Tensor) or other.dim() == 0:
        val = _as_tensor_like(other, input._values)
        new_values = (
            op(val, input._values, *extra_args, **extra_kwargs)
            if reverse
            else op(input._values, val, *extra_args, **extra_kwargs)
        )
        return input._packed_like_unchecked(new_values)

    # Resolve dense operands directly to packed values when possible. ``_dense_alignment`` reads
    # the operand against the logical shape and ``_dense_alignment_to_values`` rewrites it into
    # packed axis order, so this serves every layout -- permuted ones included -- without
    # materializing padded storage or looping over elements.
    if not isinstance(other, cls):
        resolved = _resolve_dense_for_values(input, other)
        if resolved is not None:
            lhs, rhs = (resolved, input._values) if reverse else (input._values, resolved)
            return _rebuild(op(lhs, rhs, *extra_args, **extra_kwargs))

        # Convert padded tensor to NT if shapes match and no packed-value
        # resolver applies. This is a fallback, not the transformer hot path.
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
            if not _empty_same_structure_broadcast_matches(input, other):
                raise ValueError("Empty NestedTensor operands have incompatible retained logical extents")
            new_values = op(lhs_v, rhs_v, *extra_args, **extra_kwargs)
            # Fast path (the common elementwise case): the op preserved the static tail, so the ragged
            # metadata is reused as-is -- a single ``torch.Size`` comparison, no rebuild. Only when the
            # trailing (non-ragged) dims actually broadcast -- e.g. (N,1,5,1,3) - (N,K,1,5,3) -> (N,K,5,5,3)
            # -- is the tail re-derived, rather than silently copying the left operand's element shape.
            if new_values.shape[1:] == input._values.shape[1:]:
                return input._packed_like_unchecked(new_values)
            return _packed_with_static_tail_from_values(input, new_values)

        single_element = _binary_single_element_nested_broadcast(
            input,
            other,
            op,
            source_first=False,
            extra_args=extra_args,
            extra_kwargs=extra_kwargs,
        )
        if single_element is not None:
            return single_element

        single_element = _binary_single_element_nested_broadcast(
            other,
            input,
            op,
            source_first=True,
            extra_args=extra_args,
            extra_kwargs=extra_kwargs,
        )
        if single_element is not None:
            return single_element

        aligned_other = _broadcast_nested_to_values(input, other, allow_target_static_broadcast=True)
        if aligned_other is not None:
            new_values = op(input._values, aligned_other, *extra_args, **extra_kwargs)
            return _rebuild_from(input, new_values)

        aligned_input = _broadcast_nested_to_values(other, input, allow_target_static_broadcast=True)
        if aligned_input is not None:
            new_values = op(aligned_input, other._values, *extra_args, **extra_kwargs)
            return _rebuild_from(other, new_values)

        aligned_other = _broadcast_lower_rank_nested_to_values(input, other)
        if aligned_other is not None:
            lhs, rhs = (aligned_other, input._values) if reverse else (input._values, aligned_other)
            return _rebuild(op(lhs, rhs, *extra_args, **extra_kwargs))

        aligned_input = _broadcast_lower_rank_nested_to_values(other, input)
        if aligned_input is not None:
            lhs, rhs = (other._values, aligned_input) if reverse else (aligned_input, other._values)
            new_values = op(lhs, rhs, *extra_args, **extra_kwargs)
            if new_values.shape[1:] == other._values.shape[1:]:
                return other._packed_like_unchecked(new_values)
            return _packed_with_static_tail_from_values(other, new_values)

        if len(input) == 0:
            raise ValueError("Empty NestedTensor operands have incompatible retained layouts")

        # Re-derive elements with _unpack() rather than reading _storage: _cached_storage is filled on
        # first access and keeps whatever it saw, so a cache populated under no_grad (or below the
        # autograd layer) holds detached views that would silently cut autograd on this path.
        _check_execution_guard(_ExecutionGuardKind.EAGER_FALLBACK, "ops._binary_op_maybe_tensor")
        layout = other if int(other._physical_shape.size(1)) > int(input._physical_shape.size(1)) else input
        lhs_s, rhs_s = (other._unpack(), input._unpack()) if reverse else (input._unpack(), other._unpack())
        return _nested_like_elements(
            layout,
            [op(x, y, *extra_args, **extra_kwargs) for x, y in zip(lhs_s, rhs_s)],
        )

    # Nothing the packed buffers can serve. Pair each element with its own slice of ``other``
    # instead; these paths announce themselves through the eager-fallback guard. The loop reads
    # a positional ragged extent by trimming it per element, which is a wider contract than the
    # packed resolver's, so it stays where it has always been: permuted layouts only.
    if not _is_packed_identity(input):
        per_element = _binary_per_element_dense(input, other, op, reverse, extra_args, extra_kwargs)
        if per_element is not None:
            return per_element

    broadcast_up = _binary_dense_padded_broadcast(input, other, op, reverse, extra_args, extra_kwargs)
    if broadcast_up is not None:
        return broadcast_up

    # A dense operand led by the batch dim whose remaining dims only broadcast per element, such as
    # a (B, S, 1, C) noise term against a (B, 1, ragged_N, C) tensor, and whose result changes the
    # element shape. _unpack() again, for the same autograd reason as the nested-nested path above.
    if isinstance(other, Tensor) and other.dim() == input.dim() and other.shape[_get_batch_dim(input)] == len(input):
        _check_execution_guard(_ExecutionGuardKind.EAGER_FALLBACK, "ops._binary_op_maybe_tensor")
        pairs = zip(input._unpack(), other.unbind(_get_batch_dim(input)))
        return _nested_like_elements(
            input,
            [
                (
                    op(slice_, element, *extra_args, **extra_kwargs)
                    if reverse
                    else op(element, slice_, *extra_args, **extra_kwargs)
                )
                for element, slice_ in pairs
            ],
        )
    raise NotImplementedError(
        "NestedTensor binary op with non-scalar Tensor operand that is neither shape-aligned nor "
        f"broadcast-compatible with packed values: values shape {input._values.shape}, tensor shape {other.shape}"
    )


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

    if isinstance(other, NestedTensor):
        if len(input) != len(other):
            return False
        if input._has_same_structure(other):
            return True
        if _can_broadcast_nested_to(input, other) or _can_broadcast_nested_to(other, input):
            return True
        return _can_broadcast_lower_rank_nested_to(input, other) or _can_broadcast_lower_rank_nested_to(other, input)

    try:
        if _resolve_dense_for_values(input, other) is not None:
            return True
    except NotImplementedError:
        return False

    aligned_other = _maybe_align_dense_to_nested(input, other)
    if aligned_other is not None:
        return input._has_same_structure(aligned_other)

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

    if len(input) == 0:
        return type(input)([], **input._meta(include_dtype=True))
    return input._packed_like_unchecked(op(input._values))


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
    elif type(input)._is_tensor_backed_layout(input._permutation, input._ragged_dims):
        static_dims = (False, *(True for _ in range(1, int(st.size(1)))))
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


def _is_packed_identity(input: NestedTensor) -> bool:
    r"""Whether the packed layout matches the logical element order (identity permutation).

    Several packed fast paths use a per-element dim index (``logical - batch``) to index the
    packed ``_values`` tensor and to detect the ragged dim. That is only valid when the element
    layout has not been permuted (e.g. by ``transpose``/``movedim`` to a channel-first
    ``(B, C, L)`` layout, where the ragged dim moves to packed axis 0). For non-identity
    permutations, callers must fall back to the per-element path.
    """
    permutation = input._permutation
    if permutation is None:
        return True
    return tuple(permutation) == tuple(range(input._physical_shape.size(1)))


def _translate_dim(input: NestedTensor, dim: int) -> int:
    r"""Translate a NestedTensor dimension to a per-element dimension."""
    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        raise ValueError("Cannot translate the batch dimension for NestedTensor.")
    if input.batch_first:
        return dim - 1
    return dim if dim < batch_dim else dim - 1


def _physical_to_values_dim(input: NestedTensor, physical_dim: int) -> int | None:
    r"""Map a per-element physical dimension to its packed ``_values`` axis.

    Every varying dimension is collapsed into packed axis 0, so no individual
    varying dimension has a generic one-to-one packed axis. Static dimensions
    follow axis 0 in ``_static_dims`` (packed permutation) order.
    """
    physical_dim = int(physical_dim)
    rank = int(input._physical_shape.size(1))
    if physical_dim < 0 or physical_dim >= rank:
        raise IndexError(f"Physical dimension out of range for rank {rank}: {physical_dim}")
    if physical_dim in input._varying_dims:
        return None
    try:
        return 1 + input._static_dims.index(physical_dim)
    except ValueError as exc:
        raise RuntimeError(
            f"NestedTensor physical dimension {physical_dim} is absent from packed layout " f"{input._permutation}"
        ) from exc


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
    normalized = tuple(_normalize_dim(d, input.dim()) for d in dims)
    if fill_value is not _MISSING and _get_batch_dim(input) in normalized:
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
        if type(input)._is_tensor_backed_layout(input._permutation, input._ragged_dims):
            value_tail = tuple(input._values.shape[1:])
            ndim = len(normalized_shape)
            if ndim > len(value_tail):
                return False
            return tuple(value_tail[-ndim:]) == normalized_shape
        if _is_compiling():
            _compile_unsupported(
                "NestedTensor normalization",
                "compile-safe normalization requires tensor-backed or cached element shape metadata",
            )
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

    try:
        output, _, _ = torch.ops.aten.native_layer_norm.default(input._values, normalized_shape, weight, bias, eps)
    except RuntimeError:
        return None
    return input._packed_like_unchecked(output)


def _packed_rms_norm(input: NestedTensor, normalized_shape: tuple[int, ...], weight: Tensor | None, eps):
    r"""Run RMS norm on packed ``_values`` when the normalized tail is static."""
    if not _can_concat_normalize(input, normalized_shape):
        return None

    try:
        output = torch.ops.aten.rms_norm.default(input._values, normalized_shape, weight, eps)
    except RuntimeError:
        return None
    return input._packed_like_unchecked(output)


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
