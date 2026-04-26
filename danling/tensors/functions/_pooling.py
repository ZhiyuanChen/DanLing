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
r"""Shared helpers for NestedTensor pooling handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from torch import Tensor

from ..ops import _check_execution_guard, _ExecutionGuardKind

if TYPE_CHECKING:
    from ..nested_tensor import NestedTensor

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency/runtime
    triton = None
    tl = None


def _per_element(input: NestedTensor, fn: Callable, *args, **kwargs) -> NestedTensor:
    _check_execution_guard(_ExecutionGuardKind.STORAGE_MAP, f"{fn.__name__}_per_element")
    cls = type(input)
    if len(input) == 0:
        return cls([], **input._meta(include_dtype=True))
    with torch._C.DisableTorchFunctionSubclass():
        return cls([fn(t, *args, **kwargs) for t in input._storage], **input._meta())


def _per_element_pair(input: NestedTensor, fn: Callable, *args, **kwargs) -> tuple[NestedTensor, NestedTensor]:
    _check_execution_guard(_ExecutionGuardKind.STORAGE_MAP, f"{fn.__name__}_per_element")
    cls = type(input)
    if len(input) == 0:
        empty = cls([], **input._meta(include_dtype=True))
        return empty, empty
    with torch._C.DisableTorchFunctionSubclass():
        outputs, indices = zip(*(fn(t, *args, return_indices=True, **kwargs) for t in input._storage))
    return cls(outputs, **input._meta()), cls(indices, **input._meta())


def _resolve_element_shapes(input: NestedTensor) -> tuple[tuple[int, ...], ...]:
    if input._element_shapes is not None:
        return input._element_shapes
    return tuple(type(input)._trim_shape(row) for row in input._physical_shape.tolist())


def _pool_output_size(size: int, kernel: int, stride: int, padding: int, dilation: int = 1) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


if triton is not None:

    @triton.jit
    def _pool_tile_batch(tile_to_batch_ptr, tile):
        return tl.load(tile_to_batch_ptr + tile)

else:  # pragma: no cover - optional dependency/runtime
    _pool_tile_batch = None


def _tile_to_batch(tile_counts: tuple[int, ...], *, device: torch.device) -> Tensor:
    tile_count = sum(int(count) for count in tile_counts)
    if tile_count == 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    return torch.repeat_interleave(
        torch.arange(len(tile_counts), dtype=torch.long, device=device),
        torch.tensor(tile_counts, dtype=torch.long, device=device),
    )


def _pool_block_size() -> tuple[int, int]:
    return 128, 32


def _from_pool_values(
    input: NestedTensor,
    output_values: Tensor,
    output_offsets: Tensor,
    output_shape_tensor: Tensor,
    output_packed_sizes: tuple[int, ...],
    output_shapes: tuple[tuple[int, ...], ...],
) -> NestedTensor:
    return type(input)._from_packed(
        output_values,
        output_offsets,
        output_shape_tensor,
        permutation=input._permutation,
        batch_first=input.batch_first,
        padding_value=input.padding_value,
        mask_value=input.mask_value,
        pin_memory=input._pin_memory,
        packed_sizes=output_packed_sizes,
        element_shapes=output_shapes,
    )
