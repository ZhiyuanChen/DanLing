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

r"""Shared helpers for NestedTensor convolution handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from torch import Tensor
from torch.nn import functional as F

from ..ops import _check_execution_guard, _ExecutionGuardKind

if TYPE_CHECKING:
    from ..nested_tensor import NestedTensor


def _per_element(input: NestedTensor, fn: Callable, *args, **kwargs) -> NestedTensor:
    _check_execution_guard(_ExecutionGuardKind.STORAGE_MAP, f"{fn.__name__}_per_element")
    cls = type(input)
    if len(input) == 0:
        return cls([], **input._meta(include_dtype=True))
    with torch._C.DisableTorchFunctionSubclass():
        return cls([fn(t, *args, **kwargs) for t in input._storage], **input._meta())


def _packed_pointwise_conv_transpose_channel_dim(
    input: NestedTensor,
    weight,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
    rank: int,
) -> int | None:
    r"""Return packed channel dim for pointwise transposed conv."""
    if groups != 1 or not isinstance(weight, Tensor) or weight.dim() != rank + 2:
        return None
    if tuple(int(size) for size in weight.shape[2:]) != (1,) * rank:
        return None

    if isinstance(stride, int):
        if stride != 1:
            return None
    elif not (isinstance(stride, tuple) and len(stride) == rank and all(value == 1 for value in stride)):
        return None

    if isinstance(dilation, int):
        if dilation != 1:
            return None
    elif not (isinstance(dilation, tuple) and len(dilation) == rank and all(value == 1 for value in dilation)):
        return None

    if isinstance(padding, int):
        if padding != 0:
            return None
    elif not (isinstance(padding, tuple) and len(padding) == rank and all(value == 0 for value in padding)):
        return None

    if isinstance(output_padding, int):
        if output_padding != 0:
            return None
    elif not (
        isinstance(output_padding, tuple)
        and len(output_padding) == rank
        and all(value == 0 for value in output_padding)
    ):
        return None

    if input._physical_shape.size(1) != rank + 1:
        return None
    if input._element_shapes is not None and any(len(shape) != rank + 1 for shape in input._element_shapes):
        return None

    in_channels = int(weight.shape[0])
    if not bool(torch.equal(input._physical_shape[:, 0], torch.full_like(input._physical_shape[:, 0], in_channels))):
        return None

    suffix_rank = input._values.dim() - 1
    if suffix_rank <= 0:
        return None
    static_dims = tuple(int(dim) for dim in input._permutation[-suffix_rank:])
    if 0 not in static_dims:
        return None
    channel_dim = 1 + static_dims.index(0)
    if int(input._values.shape[channel_dim]) != in_channels:
        return None
    return channel_dim


def _packed_pointwise_conv_transpose(
    input: NestedTensor,
    weight: Tensor,
    bias: Tensor | None,
    channel_dim: int,
) -> NestedTensor:
    out_channels = int(weight.shape[1])
    values = input._values
    moved = channel_dim != values.dim() - 1
    if moved:
        values = values.movedim(channel_dim, -1)
    output = F.linear(values.reshape(-1, int(weight.shape[0])), weight.flatten(2).squeeze(-1).T, bias)
    output = output.reshape(*values.shape[:-1], out_channels)
    if moved:
        output = output.movedim(-1, channel_dim).contiguous()

    shape_tensor = input._physical_shape.clone()
    shape_tensor[:, 0] = out_channels
    element_shapes = None
    if input._element_shapes is not None:
        element_shapes = tuple((out_channels, *shape[1:]) for shape in input._element_shapes)

    return type(input)._from_packed(
        output,
        input._offsets,
        shape_tensor,
        permutation=input._permutation,
        batch_first=input.batch_first,
        padding_value=input.padding_value,
        mask_value=input.mask_value,
        pin_memory=input._pin_memory,
        packed_sizes=input._packed_sizes,
        element_shapes=element_shapes,
    )


def _ceil_div(value: int, divisor: int) -> int:
    return -((-int(value)) // int(divisor))


def _conv_output_size(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def _conv_transpose_output_size(
    size: int,
    kernel: int,
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
) -> int:
    return (size - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1


def _resolve_element_shapes(input: NestedTensor) -> tuple[tuple[int, ...], ...]:
    if input._element_shapes is not None:
        return input._element_shapes
    return tuple(type(input)._trim_shape(row) for row in input._physical_shape.tolist())


def _can_use_spatial_tile_convolution(
    input: NestedTensor,
    weight: Tensor,
    groups: int,
    rank: int,
    input_channel_dim: int,
) -> bool:
    if groups != 1 or not isinstance(weight, Tensor) or weight.dim() != rank + 2:
        return False
    if input_channel_dim < 0 or input_channel_dim >= weight.dim():
        return False
    if not input._values.is_cuda:
        return False
    if weight.device != input._values.device:
        return False
    if weight.dtype != input._values.dtype:
        return False
    if input._physical_shape.size(1) != rank + 1:
        return False
    if input._element_shapes is not None and any(len(shape) != rank + 1 for shape in input._element_shapes):
        return False
    if tuple(int(dim) for dim in input._permutation) != (*range(1, rank + 1), 0) or input._values.dim() != 2:
        return False
    if not input._values.is_contiguous():
        return False

    in_channels = int(weight.shape[input_channel_dim])
    if int(input._values.shape[1]) != in_channels:
        return False
    return bool(torch.equal(input._physical_shape[:, 0], torch.full_like(input._physical_shape[:, 0], in_channels)))


def _valid_conv_bias(bias: Tensor | None, *, out_channels: int, device: torch.device, dtype: torch.dtype) -> bool:
    if bias is None:
        return True
    return (
        isinstance(bias, Tensor)
        and bias.dim() == 1
        and int(bias.shape[0]) == int(out_channels)
        and bias.device == device
        and bias.dtype == dtype
    )


def _has_cuda_tensors(*tensors: Tensor | None) -> bool:
    return all(tensor is not None and tensor.is_cuda for tensor in tensors)


def _spatial_tile_triton_block_size() -> int:
    return 1024


def _valid_output_padding(
    output_padding: tuple[int, ...],
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
) -> bool:
    return all(
        output_padding[dim] < stride[dim] or output_padding[dim] < dilation[dim] for dim in range(len(output_padding))
    )


def _spatial_tile_max_batch(max_tiles_per_batch: int | None, tile_count: int) -> int:
    if max_tiles_per_batch is None:
        return max(int(tile_count), 1)
    return max(int(max_tiles_per_batch), 1)


def _spatial_tile_length_occupancy(input_shapes: tuple[tuple[int, ...], ...]) -> float:
    if not input_shapes:
        return 0.0
    max_length = max(int(shape[1]) for shape in input_shapes)
    dense_length = len(input_shapes) * max(max_length, 1)
    real_length = sum(int(shape[1]) for shape in input_shapes)
    return float(real_length) / float(max(dense_length, 1))


def _spatial_tile_area_occupancy(input_shapes: tuple[tuple[int, ...], ...]) -> float:
    if not input_shapes:
        return 0.0
    max_h = max(int(shape[1]) for shape in input_shapes)
    max_w = max(int(shape[2]) for shape in input_shapes)
    dense_area = len(input_shapes) * max(max_h, 1) * max(max_w, 1)
    real_area = sum(int(shape[1]) * int(shape[2]) for shape in input_shapes)
    return float(real_area) / float(max(dense_area, 1))


def _prod(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _cap_spatial_tile_batch_for_channels(
    max_tiles_per_batch: int | None,
    *,
    in_channels: int,
    out_channels: int,
    input_tile_shape: tuple[int, ...],
    local_output_shape: tuple[int, ...],
) -> int:
    target_elements = 128 * 1024 * 1024
    input_elements = int(in_channels) * _prod(input_tile_shape)
    output_elements = int(out_channels) * _prod(local_output_shape)
    elements_per_tile = max(input_elements, output_elements, 1)
    channel_cap = max(1, target_elements // elements_per_tile)
    if max_tiles_per_batch is None:
        return channel_cap
    return max(1, min(int(max_tiles_per_batch), channel_cap))


def _new_spatial_tile2d_tensor(
    reference: Tensor,
    shape: tuple[int, int, int, int],
    *,
    channels_last: bool,
) -> Tensor:
    if channels_last and reference.is_cuda:
        return torch.empty(
            shape,
            device=reference.device,
            dtype=reference.dtype,
            memory_format=torch.channels_last,
        )
    return reference.new_empty(shape)


def _tile_output_intersection(
    local_start: int,
    valid: int,
    local_size: int,
) -> tuple[int, int, int, int] | None:
    dst0 = max(0, -local_start)
    dst1 = min(valid, local_size - local_start)
    if dst0 >= dst1:
        return None
    src0 = local_start + dst0
    src1 = local_start + dst1
    return dst0, dst1, src0, src1
