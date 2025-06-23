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

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence, Tuple

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .nested_tensor import NestedTensor


def _get_batch_dim(input: NestedTensor) -> int:
    return 0 if input.batch_first else 1


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    return dim


def _translate_dim(input: NestedTensor, dim: int) -> int:
    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        raise ValueError("Cannot translate the batch dimension for NestedTensor.")
    if input.batch_first:
        return dim - 1
    return dim if dim < batch_dim else dim - 1


def _translate_dims(input: NestedTensor, dims: Sequence[int]) -> Tuple[int, ...]:
    dims = tuple(_normalize_dim(d, input.dim()) for d in dims)
    batch_dim = _get_batch_dim(input)
    if any(d == batch_dim for d in dims):
        raise ValueError("Cannot translate the batch dimension for NestedTensor.")
    return tuple(_translate_dim(input, d) for d in dims)


def _translate_non_batch_dim(input: NestedTensor, dim: int, *, name: str = "dim") -> int:
    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        raise ValueError(f"{name} along the batch dimension is not supported for NestedTensor.")
    return _translate_dim(input, dim)


def _stack_or_nested(values: Sequence, input: NestedTensor):
    from .nested_tensor import NestedTensor

    try:
        return torch.stack(list(values))
    except (RuntimeError, ValueError, TypeError):
        return NestedTensor(values, **input._state)


def _map_storage(input: NestedTensor, fn):
    from .nested_tensor import NestedTensor

    storage = input._storage
    if not storage or not storage[0].is_cuda:
        return NestedTensor((fn(t) for t in storage), **input._state)
    return _run_on_streams(storage, fn, input._state)


def _map_storage_pair(input, other, op):
    from .nested_tensor import NestedTensor

    input = _ensure_nested_input(input, other)
    if isinstance(other, NestedTensor):
        if len(input) != len(other):
            raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(other)}")
        storage = input._storage
        if storage and storage[0].is_cuda:
            return _run_on_streams(
                list(zip(storage, other._storage)),
                lambda pair: op(pair[0], pair[1]),
                input._state,
            )
        return NestedTensor((op(x, y) for x, y in zip(storage, other._storage)), **input._state)
    return NestedTensor((op(x, other) for x in input._storage), **input._state)


def _run_on_streams(items, fn, state, max_streams: int = 8):
    """Run fn on each item using a pool of CUDA streams for parallel kernel launches."""
    from .nested_tensor import NestedTensor

    n = len(items)
    n_streams = min(n, max_streams)
    current = torch.cuda.current_stream()
    streams = [torch.cuda.Stream() for _ in range(n_streams)]

    results = [None] * n
    for i, item in enumerate(items):
        s = streams[i % n_streams]
        s.wait_stream(current)
        with torch.cuda.stream(s):
            results[i] = fn(item)

    for s in streams:
        current.wait_stream(s)

    return NestedTensor(results, **state)


def _as_tensor_like(value, ref: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=ref.device)
    return torch.as_tensor(value, device=ref.device, dtype=torch.result_type(ref, value))


def _ensure_nested_input(input, other):
    from .nested_tensor import NestedTensor

    if isinstance(input, NestedTensor):
        return input
    if isinstance(other, NestedTensor):
        return other.nested_like(input)
    raise ValueError("At least one argument must be a NestedTensor.")


def _binary_op_maybe_tensor(input, other, op):
    from .dispatch import _from_values
    from .nested_tensor import NestedTensor

    reverse = False
    if not isinstance(input, NestedTensor) and isinstance(other, NestedTensor):
        if isinstance(input, Tensor):
            input = other.nested_like(input)
        else:
            reverse = True
            input, other = other, input

    input = _ensure_nested_input(input, other)
    if isinstance(other, Tensor) and not isinstance(other, NestedTensor) and input.shape == other.shape:
        other = input.nested_like(other, strict=False)
    if isinstance(other, NestedTensor):
        if len(input) != len(other):
            raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(other)}")
        # Fast path: same packing layout → operate on _values directly
        if torch.equal(input._offsets, other._offsets):
            return _from_values(input, op(input._values, other._values))
        return NestedTensor((op(x, y) for x, y in zip(input._storage, other._storage)), **input._state)

    # Scalar or 0-d tensor: apply directly to _values
    if not isinstance(other, Tensor) or other.dim() == 0:
        if len(input) == 0:
            return NestedTensor([], **input._state)
        if reverse:
            return _from_values(input, op(_as_tensor_like(other, input._values), input._values))
        return _from_values(input, op(input._values, _as_tensor_like(other, input._values)))

    # General tensor: per-element fallback
    storage = []
    for t in input._storage:
        if reverse:
            storage.append(op(_as_tensor_like(other, t), t))
        else:
            storage.append(op(t, _as_tensor_like(other, t)))
    return NestedTensor(storage, **input._state)


def _broadcast_storage(ref: NestedTensor, value):
    from .nested_tensor import NestedTensor

    if isinstance(value, NestedTensor):
        if len(ref) != len(value):
            raise ValueError(f"NestedTensor batch length mismatch: {len(ref)} vs {len(value)}")
        return value._storage
    if isinstance(value, Tensor) and not isinstance(value, NestedTensor) and value.shape == ref.shape:
        return ref.nested_like(value, strict=False)._storage
    return [value for _ in ref._storage]


def _ternary_op(ref, input, tensor1, tensor2, op, **kwargs):
    from .nested_tensor import NestedTensor

    input_storage = _broadcast_storage(ref, input)
    t1_storage = _broadcast_storage(ref, tensor1)
    t2_storage = _broadcast_storage(ref, tensor2)
    storage = []
    for t, x, t1, t2 in zip(ref._storage, input_storage, t1_storage, t2_storage):
        storage.append(
            op(
                _as_tensor_like(x, t),
                _as_tensor_like(t1, t),
                _as_tensor_like(t2, t),
                **kwargs,
            )
        )
    return NestedTensor(storage, **ref._state)


def _reduce_dim(
    input: NestedTensor,
    op,
    dim: int,
    keepdim: bool,
    *,
    dtype: torch.dtype | None = None,
):
    dim = _normalize_dim(dim, input.dim())
    batch_dim = _get_batch_dim(input)
    if dim == batch_dim:
        results = [op(t, dtype=dtype) if dtype is not None else op(t) for t in input._storage]
        output = torch.stack(results)
        if keepdim:
            return output.unsqueeze(0 if input.batch_first else 1)
        return output

    dim_adj = _translate_dim(input, dim)
    results = [
        op(t, dim=dim_adj, keepdim=keepdim, dtype=dtype) if dtype is not None else op(t, dim=dim_adj, keepdim=keepdim)
        for t in input._storage
    ]
    return _stack_or_nested(results, input)


def _reduce_dims_masked(
    input: NestedTensor,
    dims: Sequence[int],
    op,
    keepdim: bool,
    *,
    dtype: torch.dtype | None = None,
    fill_value: float | int | bool,
):
    dims = tuple(_normalize_dim(d, input.dim()) for d in dims)
    tensor = input.tensor
    mask = input.mask
    valid = mask if not input.mask_value else ~mask
    while valid.dim() < tensor.dim():
        valid = valid.unsqueeze(-1)
    fill = torch.full_like(tensor, fill_value)
    data = torch.where(valid, tensor, fill)
    if dtype is not None:
        return op(data, dim=dims, keepdim=keepdim, dtype=dtype)
    return op(data, dim=dims, keepdim=keepdim)


def _reduce_none(input: NestedTensor, op, *, dtype: torch.dtype | None = None):
    # For associative reductions, operate directly on packed _values
    if op in (torch.sum, torch.prod, torch.any, torch.all, torch.amax, torch.amin) and len(input) > 0:
        return op(input._values, dtype=dtype) if dtype is not None else op(input._values)
    results = [op(t, dtype=dtype) if dtype is not None else op(t) for t in input._storage]
    stacked = torch.stack(results)
    return op(stacked, dtype=dtype) if dtype is not None else op(stacked)


def _concat_apply(
    input: NestedTensor,
    op: Callable[[Tensor], Tensor],
    shape_fn: Callable[[torch.Size], torch.Size],
):
    from .nested_tensor import NestedTensor

    if not input._storage:
        return NestedTensor([], **input._state)

    concat, shapes = input.concatenate()
    if not shapes:
        return NestedTensor([], **input._state)

    output = op(concat)
    output_shapes = tuple(shape_fn(shape) for shape in shapes)
    return NestedTensor.from_concatenated(output, output_shapes, **input._state)


def _concat_apply_same_shape(input: NestedTensor, op: Callable[[Tensor], Tensor]):
    from .dispatch import _from_values

    if len(input) == 0:
        from .nested_tensor import NestedTensor

        return NestedTensor([], **input._state)
    return _from_values(input, op(input._values))


def _static_tensor_dims(input: NestedTensor) -> Tuple[bool, ...]:
    if not input._storage:
        return ()
    ndim = max(t.dim() for t in input._storage)
    static = []
    for dim in range(ndim):
        sizes = {t.size(dim) for t in input._storage}
        static.append(len(sizes) == 1)
    return tuple(static)


def _concat_dim_for_tensor_dim(input: NestedTensor, dim: int) -> int | None:
    if not input._storage:
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
