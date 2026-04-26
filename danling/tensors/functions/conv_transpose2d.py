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

r"""``conv_transpose2d`` handler for NestedTensor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from ._convolution import (
    _can_use_spatial_tile_convolution,
    _ceil_div,
    _conv_transpose_output_size,
    _has_cuda_tensors,
    _new_spatial_tile2d_tensor,
    _packed_pointwise_conv_transpose,
    _packed_pointwise_conv_transpose_channel_dim,
    _per_element,
    _resolve_element_shapes,
    _spatial_tile_area_occupancy,
    _spatial_tile_max_batch,
    _spatial_tile_triton_block_size,
    _tile_output_intersection,
    _valid_conv_bias,
    _valid_output_padding,
)

if TYPE_CHECKING:
    from ..nested_tensor import NestedTensor

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency/runtime
    triton = None
    tl = None


# ---------------------------------------------------------------------------
# NestedTensor conv_transpose2d dispatch
# ---------------------------------------------------------------------------


def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
    *,
    _fn=None,
):
    if len(input) != 0:
        channel_dim = _packed_pointwise_conv_transpose_channel_dim(
            input,
            weight,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
            2,
        )
        if channel_dim is not None:
            return _packed_pointwise_conv_transpose(input, weight, bias, channel_dim)
        output = _spatial_tile_conv_transpose2d(
            input,
            weight,
            bias,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
        )
        if output is not None:
            return output
    return _per_element(
        input,
        F.conv_transpose2d if _fn is None else _fn,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
    )


# ---------------------------------------------------------------------------
# Spatial tile CuDNN implementation
# ---------------------------------------------------------------------------


def _conv_transpose2d_output_meta(
    input: NestedTensor,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    output_padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], Tensor] | None:
    output_shapes = []
    output_packed_sizes = []
    for shape in _resolve_element_shapes(input):
        if len(shape) != 3:
            return None
        _, in_h, in_w = shape
        out_h = _conv_transpose_output_size(in_h, kernel_size[0], stride[0], padding[0], output_padding[0], dilation[0])
        out_w = _conv_transpose_output_size(in_w, kernel_size[1], stride[1], padding[1], output_padding[1], dilation[1])
        if out_h <= 0 or out_w <= 0:
            return None
        output_shapes.append((out_channels, out_h, out_w))
        output_packed_sizes.append(out_h * out_w)
    if not output_shapes:
        return None

    return tuple(output_shapes), tuple(output_packed_sizes), torch.tensor(output_shapes, dtype=torch.long)


def _conv_transpose2d_spatial_tiles(
    output_shapes: tuple[tuple[int, ...], ...],
    tile_shape: tuple[int, int],
) -> list[tuple[int, int, int, int, int]]:
    tile_h, tile_w = tile_shape
    tiles = []
    for batch, shape in enumerate(output_shapes):
        _, out_h, out_w = shape
        for out_y in range(0, out_h, tile_h):
            valid_h = min(tile_h, out_h - out_y)
            for out_x in range(0, out_w, tile_w):
                valid_w = min(tile_w, out_w - out_x)
                tiles.append((batch, out_y, out_x, valid_h, valid_w))
    return tiles


def _auto_spatial_tile2d_config(area_occupancy: float) -> tuple[tuple[int, int], int]:
    if area_occupancy <= 0.25:
        return (96, 96), 128
    return (64, 64), 128


def _auto_spatial_tile2d_batch(_tile_shape: tuple[int, int], _area_occupancy: float) -> int:
    return 128


def _auto_spatial_weight_tile2d_batch(_tile_shape: tuple[int, int], _area_occupancy: float) -> int:
    return 128


def _cap_spatial_tile2d_batch_for_channels(
    max_tiles_per_batch: int | None,
    *,
    in_channels: int,
    out_channels: int,
    input_tile_shape: tuple[int, int],
    local_output_shape: tuple[int, int],
) -> int:
    target_elements = 128 * 1024 * 1024
    input_elements = int(in_channels) * int(input_tile_shape[0]) * int(input_tile_shape[1])
    output_elements = int(out_channels) * int(local_output_shape[0]) * int(local_output_shape[1])
    elements_per_tile = max(input_elements, output_elements, 1)
    channel_cap = max(1, target_elements // elements_per_tile)
    if max_tiles_per_batch is None:
        return channel_cap
    return max(1, min(int(max_tiles_per_batch), channel_cap))


def _resolve_spatial_tile2d_config(
    input_shapes: tuple[tuple[int, ...], ...],
    tile_size: int | tuple[int, int] | str,
    max_tiles_per_batch: int | str | None,
) -> tuple[tuple[int, int], int | None] | None:
    area_occupancy = _spatial_tile_area_occupancy(input_shapes)
    if isinstance(tile_size, str):
        if tile_size != "auto":
            return None
        tile_shape, auto_batch = _auto_spatial_tile2d_config(area_occupancy)
    else:
        if isinstance(tile_size, int):
            parsed_tile_shape = (int(tile_size), int(tile_size))
        elif isinstance(tile_size, tuple) and len(tile_size) == 2:
            parsed_tile_shape = (int(tile_size[0]), int(tile_size[1]))
        else:
            return None
        if parsed_tile_shape[0] <= 0 or parsed_tile_shape[1] <= 0:
            return None
        tile_shape = parsed_tile_shape
        auto_batch = _auto_spatial_tile2d_batch(tile_shape, area_occupancy)

    if isinstance(max_tiles_per_batch, str):
        if max_tiles_per_batch != "auto":
            return None
        return tile_shape, auto_batch
    if max_tiles_per_batch is None:
        return tile_shape, None
    return tile_shape, max(int(max_tiles_per_batch), 1)


def _resolve_spatial_weight_tile2d_batch(
    input_shapes: tuple[tuple[int, ...], ...],
    tile_shape: tuple[int, int],
    weight_max_tiles_per_batch: int | str | None,
    default_chunk_size: int,
) -> int | None:
    if isinstance(weight_max_tiles_per_batch, str):
        if weight_max_tiles_per_batch == "same":
            return default_chunk_size
        if weight_max_tiles_per_batch != "auto":
            return None
        auto_batch = _auto_spatial_weight_tile2d_batch(tile_shape, _spatial_tile_area_occupancy(input_shapes))
        return min(default_chunk_size, auto_batch)
    if weight_max_tiles_per_batch is None:
        return max(default_chunk_size, 1)
    return max(int(weight_max_tiles_per_batch), 1)


if triton is not None:

    @triton.jit
    def _spatial_tile2d_pack_input_kernel(
        input_ptr,
        tile_ptr,
        tile_geometry_ptr,
        input_offsets_ptr,
        shape_meta_ptr,
        tile_offset,
        total,
        tile_stride_n: tl.constexpr,
        tile_stride_c: tl.constexpr,
        tile_stride_h: tl.constexpr,
        tile_stride_w: tl.constexpr,
        in_channels: tl.constexpr,
        input_tile_h: tl.constexpr,
        input_tile_w: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total
        channel = offsets % in_channels
        tmp = offsets // in_channels
        tile_x = tmp % input_tile_w
        tmp = tmp // input_tile_w
        tile_y = tmp % input_tile_h
        tile_local = tmp // input_tile_h
        tile = tile_offset + tile_local

        batch = tl.load(tile_geometry_ptr + tile * 11, mask=mask, other=0)
        origin_y = tl.load(tile_geometry_ptr + tile * 11 + 5, mask=mask, other=0)
        origin_x = tl.load(tile_geometry_ptr + tile * 11 + 6, mask=mask, other=0)
        copy_h = tl.load(tile_geometry_ptr + tile * 11 + 7, mask=mask, other=0)
        copy_w = tl.load(tile_geometry_ptr + tile * 11 + 8, mask=mask, other=0)
        in_w = tl.load(shape_meta_ptr + batch * 4 + 1, mask=mask, other=0)
        in_base = tl.load(input_offsets_ptr + batch, mask=mask, other=0)

        valid = mask & (tile_y < copy_h) & (tile_x < copy_w)
        input_y = origin_y + tile_y
        input_x = origin_x + tile_x
        input_pos = in_base + input_y * in_w + input_x
        values = tl.load(input_ptr + input_pos * in_channels + channel, mask=valid, other=0.0)
        tl.store(
            tile_ptr
            + tile_local * tile_stride_n
            + channel * tile_stride_c
            + tile_y * tile_stride_h
            + tile_x * tile_stride_w,
            values,
            mask=mask,
        )

    @triton.jit
    def _spatial_tile2d_scatter_output_kernel(
        tile_ptr,
        bias_ptr,
        output_ptr,
        tile_geometry_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        tile_offset,
        total,
        tile_stride_n: tl.constexpr,
        tile_stride_c: tl.constexpr,
        tile_stride_h: tl.constexpr,
        tile_stride_w: tl.constexpr,
        out_channels: tl.constexpr,
        tile_h: tl.constexpr,
        tile_w: tl.constexpr,
        local_h: tl.constexpr,
        local_w: tl.constexpr,
        has_bias: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total
        channel = offsets % out_channels
        tmp = offsets // out_channels
        region_x = tmp % tile_w
        tmp = tmp // tile_w
        region_y = tmp % tile_h
        tile_local = tmp // tile_h
        tile = tile_offset + tile_local

        batch = tl.load(tile_geometry_ptr + tile * 11, mask=mask, other=0)
        out_y = tl.load(tile_geometry_ptr + tile * 11 + 1, mask=mask, other=0)
        out_x = tl.load(tile_geometry_ptr + tile * 11 + 2, mask=mask, other=0)
        valid_h = tl.load(tile_geometry_ptr + tile * 11 + 3, mask=mask, other=0)
        valid_w = tl.load(tile_geometry_ptr + tile * 11 + 4, mask=mask, other=0)
        global_offset_y = tl.load(tile_geometry_ptr + tile * 11 + 9, mask=mask, other=0)
        global_offset_x = tl.load(tile_geometry_ptr + tile * 11 + 10, mask=mask, other=0)
        out_w = tl.load(shape_meta_ptr + batch * 4 + 3, mask=mask, other=0)
        out_base = tl.load(output_offsets_ptr + batch, mask=mask, other=0)

        valid = mask & (region_y < valid_h) & (region_x < valid_w)
        src_y = out_y + region_y - global_offset_y
        src_x = out_x + region_x - global_offset_x
        src_valid = valid & (src_y >= 0) & (src_y < local_h) & (src_x >= 0) & (src_x < local_w)
        values = tl.load(
            tile_ptr
            + tile_local * tile_stride_n
            + channel * tile_stride_c
            + src_y * tile_stride_h
            + src_x * tile_stride_w,
            mask=src_valid,
            other=0.0,
        )
        if has_bias:
            values += tl.load(bias_ptr + channel, mask=valid, other=0.0)
        output_pos = out_base + (out_y + region_y) * out_w + out_x + region_x
        tl.store(output_ptr + output_pos * out_channels + channel, values, mask=valid)

    @triton.jit
    def _spatial_tile2d_pack_grad_output_kernel(
        grad_output_ptr,
        tile_ptr,
        tile_geometry_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        tile_offset,
        total,
        grad_output_stride_n: tl.constexpr,
        grad_output_stride_c: tl.constexpr,
        tile_stride_n: tl.constexpr,
        tile_stride_c: tl.constexpr,
        tile_stride_h: tl.constexpr,
        tile_stride_w: tl.constexpr,
        out_channels: tl.constexpr,
        local_h: tl.constexpr,
        local_w: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total
        channel = offsets % out_channels
        tmp = offsets // out_channels
        local_x = tmp % local_w
        tmp = tmp // local_w
        local_y = tmp % local_h
        tile_local = tmp // local_h
        tile = tile_offset + tile_local

        batch = tl.load(tile_geometry_ptr + tile * 11, mask=mask, other=0)
        out_y = tl.load(tile_geometry_ptr + tile * 11 + 1, mask=mask, other=0)
        out_x = tl.load(tile_geometry_ptr + tile * 11 + 2, mask=mask, other=0)
        valid_h = tl.load(tile_geometry_ptr + tile * 11 + 3, mask=mask, other=0)
        valid_w = tl.load(tile_geometry_ptr + tile * 11 + 4, mask=mask, other=0)
        global_offset_y = tl.load(tile_geometry_ptr + tile * 11 + 9, mask=mask, other=0)
        global_offset_x = tl.load(tile_geometry_ptr + tile * 11 + 10, mask=mask, other=0)
        out_w = tl.load(shape_meta_ptr + batch * 4 + 3, mask=mask, other=0)
        out_base = tl.load(output_offsets_ptr + batch, mask=mask, other=0)

        global_y = global_offset_y + local_y
        global_x = global_offset_x + local_x
        valid = mask & (global_y >= out_y) & (global_y < out_y + valid_h)
        valid = valid & (global_x >= out_x) & (global_x < out_x + valid_w)
        output_pos = out_base + global_y * out_w + global_x
        values = tl.load(
            grad_output_ptr + output_pos * grad_output_stride_n + channel * grad_output_stride_c,
            mask=valid,
            other=0.0,
        )
        tl.store(
            tile_ptr
            + tile_local * tile_stride_n
            + channel * tile_stride_c
            + local_y * tile_stride_h
            + local_x * tile_stride_w,
            values,
            mask=mask,
        )

    @triton.jit
    def _spatial_tile2d_scatter_grad_input_kernel(
        tile_ptr,
        grad_input_ptr,
        tile_geometry_ptr,
        input_offsets_ptr,
        shape_meta_ptr,
        tile_offset,
        total,
        tile_stride_n: tl.constexpr,
        tile_stride_c: tl.constexpr,
        tile_stride_h: tl.constexpr,
        tile_stride_w: tl.constexpr,
        in_channels: tl.constexpr,
        input_tile_h: tl.constexpr,
        input_tile_w: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total
        channel = offsets % in_channels
        tmp = offsets // in_channels
        tile_x = tmp % input_tile_w
        tmp = tmp // input_tile_w
        tile_y = tmp % input_tile_h
        tile_local = tmp // input_tile_h
        tile = tile_offset + tile_local

        batch = tl.load(tile_geometry_ptr + tile * 11, mask=mask, other=0)
        origin_y = tl.load(tile_geometry_ptr + tile * 11 + 5, mask=mask, other=0)
        origin_x = tl.load(tile_geometry_ptr + tile * 11 + 6, mask=mask, other=0)
        copy_h = tl.load(tile_geometry_ptr + tile * 11 + 7, mask=mask, other=0)
        copy_w = tl.load(tile_geometry_ptr + tile * 11 + 8, mask=mask, other=0)
        in_w = tl.load(shape_meta_ptr + batch * 4 + 1, mask=mask, other=0)
        in_base = tl.load(input_offsets_ptr + batch, mask=mask, other=0)

        valid = mask & (tile_y < copy_h) & (tile_x < copy_w)
        values = tl.load(
            tile_ptr
            + tile_local * tile_stride_n
            + channel * tile_stride_c
            + tile_y * tile_stride_h
            + tile_x * tile_stride_w,
            mask=valid,
            other=0.0,
        )
        input_y = origin_y + tile_y
        input_x = origin_x + tile_x
        input_pos = in_base + input_y * in_w + input_x
        tl.atomic_add(grad_input_ptr + input_pos * in_channels + channel, values, sem="relaxed", mask=valid)

else:
    _spatial_tile2d_pack_input_kernel = None
    _spatial_tile2d_scatter_output_kernel = None
    _spatial_tile2d_pack_grad_output_kernel = None
    _spatial_tile2d_scatter_grad_input_kernel = None


def _conv_transpose2d_tile_origin_dim(
    out_start: int,
    in_size: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    origin = _ceil_div(out_start + padding - dilation * (kernel - 1), stride)
    return min(max(origin, 0), in_size)


def _conv_transpose2d_needed_length_dim(
    out_start: int,
    valid: int,
    in_size: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    origin = _conv_transpose2d_tile_origin_dim(out_start, in_size, kernel, stride, padding, dilation)
    end = (out_start + valid - 1 + padding) // stride + 1
    end = min(max(end, 0), in_size)
    return max(end - origin, 0)


def _conv_transpose2d_input_tile_shape(
    input_shapes: tuple[tuple[int, ...], ...],
    tiles: tuple[tuple[int, int, int, int, int], ...],
    *,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    max_h = max_w = 1
    for batch, out_y, out_x, valid_h, valid_w in tiles:
        _, in_h, in_w = input_shapes[batch]
        max_h = max(
            max_h,
            _conv_transpose2d_needed_length_dim(
                out_y, valid_h, in_h, kernel_size[0], stride[0], padding[0], dilation[0]
            ),
        )
        max_w = max(
            max_w,
            _conv_transpose2d_needed_length_dim(
                out_x, valid_w, in_w, kernel_size[1], stride[1], padding[1], dilation[1]
            ),
        )
    return max_h, max_w


def _conv_transpose2d_local_output_shape(
    input_tile_shape: tuple[int, int],
    *,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    # output_padding is a global output-size disambiguator.  Tile-local cuDNN
    # computes the uncropped transposed-convolution support, then scatter/crop
    # maps that support into the requested global output region.
    return (
        (input_tile_shape[0] - 1) * stride[0] + dilation[0] * (kernel_size[0] - 1) + 1,
        (input_tile_shape[1] - 1) * stride[1] + dilation[1] * (kernel_size[1] - 1) + 1,
    )


def _conv_transpose2d_tile_geometry(
    batch: int,
    out_y: int,
    out_x: int,
    input_shapes: tuple[tuple[int, ...], ...],
    input_tile_shape: tuple[int, int],
    *,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    _, in_h, in_w = input_shapes[batch]
    origin = (
        _conv_transpose2d_tile_origin_dim(out_y, in_h, kernel_size[0], stride[0], padding[0], dilation[0]),
        _conv_transpose2d_tile_origin_dim(out_x, in_w, kernel_size[1], stride[1], padding[1], dilation[1]),
    )
    copy_shape = (
        max(min(input_tile_shape[0], in_h - origin[0]), 0),
        max(min(input_tile_shape[1], in_w - origin[1]), 0),
    )
    global_offset = (
        origin[0] * stride[0] - padding[0],
        origin[1] * stride[1] - padding[1],
    )
    return origin, copy_shape, (in_h, in_w), global_offset


def _conv_transpose2d_tile_geometry_rows(
    input_shapes: tuple[tuple[int, ...], ...],
    tiles: tuple[tuple[int, int, int, int, int], ...],
    input_tile_shape: tuple[int, int],
    *,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[tuple[int, ...], ...]:
    rows = []
    for batch, out_y, out_x, valid_h, valid_w in tiles:
        origin, copy_shape, _, global_offset = _conv_transpose2d_tile_geometry(
            batch,
            out_y,
            out_x,
            input_shapes,
            input_tile_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        rows.append(
            (
                batch,
                out_y,
                out_x,
                valid_h,
                valid_w,
                origin[0],
                origin[1],
                copy_shape[0],
                copy_shape[1],
                global_offset[0],
                global_offset[1],
            )
        )
    return tuple(rows)


def _make_spatial_tile2d_input(
    input_values: Tensor,
    input_shapes: tuple[tuple[int, ...], ...],
    input_starts: tuple[int, ...],
    chunk: tuple[tuple[int, int, int, int, int], ...],
    *,
    in_channels: int,
    input_tile_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    channels_last: bool,
    tile_offset: int = 0,
    input_offsets_device: Tensor | None = None,
    shape_meta: Tensor | None = None,
    tile_geometry_meta: Tensor | None = None,
) -> Tensor:
    input_tile_h, input_tile_w = input_tile_shape
    tile_input = _new_spatial_tile2d_tensor(
        input_values,
        (len(chunk), in_channels, input_tile_h, input_tile_w),
        channels_last=channels_last,
    )

    if triton is not None and _has_cuda_tensors(input_offsets_device, shape_meta, tile_geometry_meta):
        block_size = _spatial_tile_triton_block_size()
        total = len(chunk) * input_tile_h * input_tile_w * in_channels
        grid = (triton.cdiv(total, block_size),)
        _spatial_tile2d_pack_input_kernel[grid](
            input_values,
            tile_input,
            tile_geometry_meta,
            input_offsets_device,
            shape_meta,
            int(tile_offset),
            total,
            tile_input.stride(0),
            tile_input.stride(1),
            tile_input.stride(2),
            tile_input.stride(3),
            in_channels,
            input_tile_h,
            input_tile_w,
            block_size,
        )
        return tile_input

    tile_input.zero_()

    for tile_index, (batch, out_y, out_x, _, _) in enumerate(chunk):
        origin, copy_shape, input_shape, _ = _conv_transpose2d_tile_geometry(
            batch,
            out_y,
            out_x,
            input_shapes,
            input_tile_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        copy_h, copy_w = copy_shape
        if copy_h == 0 or copy_w == 0:
            continue
        in_h, in_w = input_shape
        input_start = input_starts[batch]
        element = input_values[input_start : input_start + in_h * in_w].reshape(in_h, in_w, in_channels)
        tile_input[tile_index, :, :copy_h, :copy_w] = element[
            origin[0] : origin[0] + copy_h,
            origin[1] : origin[1] + copy_w,
        ].permute(2, 0, 1)

    return tile_input


def _scatter_spatial_tile2d_output(
    output_values: Tensor,
    tile_output: Tensor,
    bias: Tensor | None,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
    output_starts: tuple[int, ...],
    chunk: tuple[tuple[int, int, int, int, int], ...],
    *,
    out_channels: int,
    input_tile_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    tile_offset: int = 0,
    output_offsets_device: Tensor | None = None,
    shape_meta: Tensor | None = None,
    tile_geometry_meta: Tensor | None = None,
) -> None:
    local_h, local_w = (int(tile_output.shape[2]), int(tile_output.shape[3]))
    if triton is not None and _has_cuda_tensors(output_offsets_device, shape_meta, tile_geometry_meta):
        tile_h = max(int(tile[3]) for tile in chunk)
        tile_w = max(int(tile[4]) for tile in chunk)
        block_size = _spatial_tile_triton_block_size()
        total = len(chunk) * tile_h * tile_w * out_channels
        grid = (triton.cdiv(total, block_size),)
        _spatial_tile2d_scatter_output_kernel[grid](
            tile_output,
            bias if bias is not None else output_values,
            output_values,
            tile_geometry_meta,
            output_offsets_device,
            shape_meta,
            int(tile_offset),
            total,
            tile_output.stride(0),
            tile_output.stride(1),
            tile_output.stride(2),
            tile_output.stride(3),
            out_channels,
            tile_h,
            tile_w,
            local_h,
            local_w,
            bias is not None,
            block_size,
        )
        return

    for tile_index, (batch, out_y, out_x, valid_h, valid_w) in enumerate(chunk):
        _, out_h, out_w = output_shapes[batch]
        output_start = output_starts[batch]
        element_output = output_values[output_start : output_start + out_h * out_w].reshape(out_h, out_w, out_channels)
        region = element_output[out_y : out_y + valid_h, out_x : out_x + valid_w]
        if bias is None:
            region.zero_()
        else:
            region.copy_(bias.reshape(1, 1, out_channels).expand_as(region))

        _, _, _, global_offset = _conv_transpose2d_tile_geometry(
            batch,
            out_y,
            out_x,
            input_shapes,
            input_tile_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        y_hit = _tile_output_intersection(out_y - global_offset[0], valid_h, local_h)
        x_hit = _tile_output_intersection(out_x - global_offset[1], valid_w, local_w)
        if y_hit is None or x_hit is None:
            continue
        dst_y0, dst_y1, src_y0, src_y1 = y_hit
        dst_x0, dst_x1, src_x0, src_x1 = x_hit
        region[dst_y0:dst_y1, dst_x0:dst_x1] += tile_output[
            tile_index,
            :,
            src_y0:src_y1,
            src_x0:src_x1,
        ].permute(1, 2, 0)


def _make_spatial_tile2d_grad_output(
    grad_output_values: Tensor,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
    output_starts: tuple[int, ...],
    chunk: tuple[tuple[int, int, int, int, int], ...],
    *,
    out_channels: int,
    input_tile_shape: tuple[int, int],
    local_output_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    channels_last: bool,
    tile_offset: int = 0,
    output_offsets_device: Tensor | None = None,
    shape_meta: Tensor | None = None,
    tile_geometry_meta: Tensor | None = None,
) -> Tensor:
    local_h, local_w = local_output_shape
    tile_grad_output = _new_spatial_tile2d_tensor(
        grad_output_values,
        (len(chunk), out_channels, local_h, local_w),
        channels_last=channels_last,
    )

    if triton is not None and _has_cuda_tensors(output_offsets_device, shape_meta, tile_geometry_meta):
        block_size = _spatial_tile_triton_block_size()
        total = len(chunk) * local_h * local_w * out_channels
        grid = (triton.cdiv(total, block_size),)
        _spatial_tile2d_pack_grad_output_kernel[grid](
            grad_output_values,
            tile_grad_output,
            tile_geometry_meta,
            output_offsets_device,
            shape_meta,
            int(tile_offset),
            total,
            grad_output_values.stride(0),
            grad_output_values.stride(1),
            tile_grad_output.stride(0),
            tile_grad_output.stride(1),
            tile_grad_output.stride(2),
            tile_grad_output.stride(3),
            out_channels,
            local_h,
            local_w,
            block_size,
        )
        return tile_grad_output

    tile_grad_output.zero_()

    for tile_index, (batch, out_y, out_x, valid_h, valid_w) in enumerate(chunk):
        _, out_h, out_w = output_shapes[batch]
        output_start = output_starts[batch]
        element_grad = grad_output_values[output_start : output_start + out_h * out_w].reshape(
            out_h,
            out_w,
            out_channels,
        )
        _, _, _, global_offset = _conv_transpose2d_tile_geometry(
            batch,
            out_y,
            out_x,
            input_shapes,
            input_tile_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        y_hit = _tile_output_intersection(out_y - global_offset[0], valid_h, local_h)
        x_hit = _tile_output_intersection(out_x - global_offset[1], valid_w, local_w)
        if y_hit is None or x_hit is None:
            continue
        dst_y0, dst_y1, src_y0, src_y1 = y_hit
        dst_x0, dst_x1, src_x0, src_x1 = x_hit
        tile_grad_output[tile_index, :, src_y0:src_y1, src_x0:src_x1] = element_grad[
            out_y + dst_y0 : out_y + dst_y1,
            out_x + dst_x0 : out_x + dst_x1,
        ].permute(2, 0, 1)

    return tile_grad_output


def _scatter_spatial_tile2d_grad_input(
    grad_input_values: Tensor,
    tile_grad_input: Tensor,
    input_shapes: tuple[tuple[int, ...], ...],
    input_starts: tuple[int, ...],
    chunk: tuple[tuple[int, int, int, int, int], ...],
    *,
    in_channels: int,
    input_tile_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    tile_offset: int = 0,
    input_offsets_device: Tensor | None = None,
    shape_meta: Tensor | None = None,
    tile_geometry_meta: Tensor | None = None,
) -> None:
    if triton is not None and _has_cuda_tensors(input_offsets_device, shape_meta, tile_geometry_meta):
        input_tile_h, input_tile_w = input_tile_shape
        block_size = _spatial_tile_triton_block_size()
        total = len(chunk) * input_tile_h * input_tile_w * in_channels
        grid = (triton.cdiv(total, block_size),)
        _spatial_tile2d_scatter_grad_input_kernel[grid](
            tile_grad_input,
            grad_input_values,
            tile_geometry_meta,
            input_offsets_device,
            shape_meta,
            int(tile_offset),
            total,
            tile_grad_input.stride(0),
            tile_grad_input.stride(1),
            tile_grad_input.stride(2),
            tile_grad_input.stride(3),
            in_channels,
            input_tile_h,
            input_tile_w,
            block_size,
        )
        return

    for tile_index, (batch, out_y, out_x, _, _) in enumerate(chunk):
        origin, copy_shape, input_shape, _ = _conv_transpose2d_tile_geometry(
            batch,
            out_y,
            out_x,
            input_shapes,
            input_tile_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        copy_h, copy_w = copy_shape
        if copy_h == 0 or copy_w == 0:
            continue
        in_h, in_w = input_shape
        input_start = input_starts[batch]
        element_grad = grad_input_values[input_start : input_start + in_h * in_w].reshape(in_h, in_w, in_channels)
        element_grad[origin[0] : origin[0] + copy_h, origin[1] : origin[1] + copy_w] += tile_grad_input[
            tile_index,
            :,
            :copy_h,
            :copy_w,
        ].permute(1, 2, 0)


def _spatial_tile_conv_transpose2d_forward_values(
    input_values: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    *,
    output_size: int,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
    input_starts: tuple[int, ...],
    output_starts: tuple[int, ...],
    tiles: tuple[tuple[int, int, int, int, int], ...],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    input_tile_shape: tuple[int, int],
    max_tiles_per_batch: int,
    channels_last: bool,
    input_offsets_device: Tensor | None = None,
    output_offsets_device: Tensor | None = None,
    shape_meta: Tensor | None = None,
    tile_geometry_meta: Tensor | None = None,
) -> Tensor:
    in_channels = int(weight.shape[0])
    out_channels = int(weight.shape[1])
    kernel_size = (int(weight.shape[2]), int(weight.shape[3]))
    output_values = input_values.new_empty((output_size, out_channels))
    chunk_size = _spatial_tile_max_batch(max_tiles_per_batch, len(tiles))

    for chunk_start in range(0, len(tiles), chunk_size):
        chunk = tiles[chunk_start : chunk_start + chunk_size]
        tile_input = _make_spatial_tile2d_input(
            input_values,
            input_shapes,
            input_starts,
            chunk,
            in_channels=in_channels,
            input_tile_shape=input_tile_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            channels_last=channels_last,
            tile_offset=chunk_start,
            input_offsets_device=input_offsets_device,
            shape_meta=shape_meta,
            tile_geometry_meta=tile_geometry_meta,
        )
        tile_output = F.conv_transpose2d(
            tile_input,
            weight,
            None,
            stride=stride,
            padding=0,
            # Keep output_padding at zero for each tile.  Non-zero
            # output_padding only extends the global output shape; applying it
            # per tile would create extra local support and duplicate
            # contributions at tile boundaries.
            output_padding=0,
            groups=1,
            dilation=dilation,
        )
        _scatter_spatial_tile2d_output(
            output_values,
            tile_output,
            bias,
            input_shapes,
            output_shapes,
            output_starts,
            chunk,
            out_channels=out_channels,
            input_tile_shape=input_tile_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            tile_offset=chunk_start,
            output_offsets_device=output_offsets_device,
            shape_meta=shape_meta,
            tile_geometry_meta=tile_geometry_meta,
        )

    return output_values


def _spatial_tile_conv_transpose2d_input_grad(
    input_size: tuple[int, int, int, int],
    weight: Tensor,
    grad_output: Tensor,
    *,
    stride: tuple[int, int],
    dilation: tuple[int, int],
    channels_last: bool,
) -> Tensor:
    if channels_last and grad_output.is_cuda:
        input_meta = torch.empty(
            input_size,
            device=grad_output.device,
            dtype=grad_output.dtype,
            memory_format=torch.channels_last,
        )
    else:
        input_meta = grad_output.new_empty(input_size)
    return torch.ops.aten.convolution_backward(
        grad_output,
        input_meta,
        weight,
        None,
        stride,
        (0, 0),
        dilation,
        True,
        [0, 0],
        1,
        (True, False, False),
    )[0]


def _spatial_tile_conv_transpose2d_weight_grad(
    input: Tensor,
    weight: Tensor,
    grad_output: Tensor,
    *,
    stride: tuple[int, int],
    dilation: tuple[int, int],
) -> Tensor:
    weight_meta = torch.empty_strided(
        tuple(int(size) for size in weight.shape),
        tuple(int(stride_value) for stride_value in weight.stride()),
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight_meta,
        None,
        stride,
        (0, 0),
        dilation,
        True,
        [0, 0],
        1,
        (False, True, False),
    )[1]


class _SpatialTileConvTranspose2dCudnnFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_values: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        input_offsets_device: Tensor,
        output_offsets_device: Tensor,
        shape_meta: Tensor,
        tile_geometry_meta: Tensor,
        output_size: int,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shapes: tuple[tuple[int, ...], ...],
        input_starts: tuple[int, ...],
        output_starts: tuple[int, ...],
        tiles: tuple[tuple[int, int, int, int, int], ...],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        input_tile_shape: tuple[int, int],
        local_output_shape: tuple[int, int],
        max_tiles_per_batch: int,
        weight_max_tiles_per_batch: int,
        channels_last: bool,
    ) -> Tensor:
        weight = weight.contiguous()
        output_values = _spatial_tile_conv_transpose2d_forward_values(
            input_values,
            weight,
            bias,
            output_size=output_size,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_starts=input_starts,
            output_starts=output_starts,
            tiles=tiles,
            stride=stride,
            padding=padding,
            dilation=dilation,
            input_tile_shape=input_tile_shape,
            max_tiles_per_batch=max_tiles_per_batch,
            channels_last=channels_last,
            input_offsets_device=input_offsets_device,
            output_offsets_device=output_offsets_device,
            shape_meta=shape_meta,
            tile_geometry_meta=tile_geometry_meta,
        )
        ctx.save_for_backward(
            input_values,
            weight,
            input_offsets_device,
            output_offsets_device,
            shape_meta,
            tile_geometry_meta,
        )
        ctx.has_bias = bias is not None
        ctx.input_shapes = input_shapes
        ctx.output_shapes = output_shapes
        ctx.input_starts = input_starts
        ctx.output_starts = output_starts
        ctx.tiles = tiles
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.input_tile_shape = input_tile_shape
        ctx.local_output_shape = local_output_shape
        ctx.max_tiles_per_batch = max_tiles_per_batch
        ctx.weight_max_tiles_per_batch = weight_max_tiles_per_batch
        ctx.channels_last = channels_last
        return output_values

    @staticmethod
    def backward(ctx, grad_output_values: Tensor):
        input_values, weight, input_offsets_device, output_offsets_device, shape_meta, tile_geometry_meta = (
            ctx.saved_tensors
        )
        input_shapes = ctx.input_shapes
        output_shapes = ctx.output_shapes
        input_starts = ctx.input_starts
        output_starts = ctx.output_starts
        tiles = ctx.tiles
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        input_tile_shape = ctx.input_tile_shape
        local_output_shape = ctx.local_output_shape
        max_tiles_per_batch = ctx.max_tiles_per_batch
        weight_max_tiles_per_batch = ctx.weight_max_tiles_per_batch
        channels_last = ctx.channels_last
        in_channels = int(weight.shape[0])
        out_channels = int(weight.shape[1])
        kernel_size = (int(weight.shape[2]), int(weight.shape[3]))

        grad_input = torch.zeros_like(input_values) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        grad_bias = grad_output_values.sum(dim=0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        chunk_size = _spatial_tile_max_batch(max_tiles_per_batch, len(tiles))

        if grad_input is not None and (grad_weight is None or weight_max_tiles_per_batch != chunk_size):
            for chunk_start in range(0, len(tiles), chunk_size):
                chunk = tiles[chunk_start : chunk_start + chunk_size]
                tile_grad_output = _make_spatial_tile2d_grad_output(
                    grad_output_values,
                    input_shapes,
                    output_shapes,
                    output_starts,
                    chunk,
                    out_channels=out_channels,
                    input_tile_shape=input_tile_shape,
                    local_output_shape=local_output_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    channels_last=channels_last,
                    tile_offset=chunk_start,
                    output_offsets_device=output_offsets_device,
                    shape_meta=shape_meta,
                    tile_geometry_meta=tile_geometry_meta,
                )
                tile_grad_input = _spatial_tile_conv_transpose2d_input_grad(
                    (len(chunk), in_channels, input_tile_shape[0], input_tile_shape[1]),
                    weight,
                    tile_grad_output,
                    stride=stride,
                    dilation=dilation,
                    channels_last=channels_last,
                )
                _scatter_spatial_tile2d_grad_input(
                    grad_input,
                    tile_grad_input,
                    input_shapes,
                    input_starts,
                    chunk,
                    in_channels=in_channels,
                    input_tile_shape=input_tile_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    tile_offset=chunk_start,
                    input_offsets_device=input_offsets_device,
                    shape_meta=shape_meta,
                    tile_geometry_meta=tile_geometry_meta,
                )

        if grad_weight is not None and (grad_input is None or weight_max_tiles_per_batch != chunk_size):
            weight_chunk_size = _spatial_tile_max_batch(weight_max_tiles_per_batch, len(tiles))
            for chunk_start in range(0, len(tiles), weight_chunk_size):
                chunk = tiles[chunk_start : chunk_start + weight_chunk_size]
                tile_grad_output = _make_spatial_tile2d_grad_output(
                    grad_output_values,
                    input_shapes,
                    output_shapes,
                    output_starts,
                    chunk,
                    out_channels=out_channels,
                    input_tile_shape=input_tile_shape,
                    local_output_shape=local_output_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    channels_last=channels_last,
                    tile_offset=chunk_start,
                    output_offsets_device=output_offsets_device,
                    shape_meta=shape_meta,
                    tile_geometry_meta=tile_geometry_meta,
                )
                tile_input = _make_spatial_tile2d_input(
                    input_values,
                    input_shapes,
                    input_starts,
                    chunk,
                    in_channels=in_channels,
                    input_tile_shape=input_tile_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    channels_last=channels_last,
                    tile_offset=chunk_start,
                    input_offsets_device=input_offsets_device,
                    shape_meta=shape_meta,
                    tile_geometry_meta=tile_geometry_meta,
                )
                grad_weight += _spatial_tile_conv_transpose2d_weight_grad(
                    tile_input,
                    weight,
                    tile_grad_output,
                    stride=stride,
                    dilation=dilation,
                )

        if grad_input is not None and grad_weight is not None and weight_max_tiles_per_batch == chunk_size:
            for chunk_start in range(0, len(tiles), chunk_size):
                chunk = tiles[chunk_start : chunk_start + chunk_size]
                tile_grad_output = _make_spatial_tile2d_grad_output(
                    grad_output_values,
                    input_shapes,
                    output_shapes,
                    output_starts,
                    chunk,
                    out_channels=out_channels,
                    input_tile_shape=input_tile_shape,
                    local_output_shape=local_output_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    channels_last=channels_last,
                    tile_offset=chunk_start,
                    output_offsets_device=output_offsets_device,
                    shape_meta=shape_meta,
                    tile_geometry_meta=tile_geometry_meta,
                )
                tile_grad_input = _spatial_tile_conv_transpose2d_input_grad(
                    (len(chunk), in_channels, input_tile_shape[0], input_tile_shape[1]),
                    weight,
                    tile_grad_output,
                    stride=stride,
                    dilation=dilation,
                    channels_last=channels_last,
                )
                _scatter_spatial_tile2d_grad_input(
                    grad_input,
                    tile_grad_input,
                    input_shapes,
                    input_starts,
                    chunk,
                    in_channels=in_channels,
                    input_tile_shape=input_tile_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    tile_offset=chunk_start,
                    input_offsets_device=input_offsets_device,
                    shape_meta=shape_meta,
                    tile_geometry_meta=tile_geometry_meta,
                )
                tile_input = _make_spatial_tile2d_input(
                    input_values,
                    input_shapes,
                    input_starts,
                    chunk,
                    in_channels=in_channels,
                    input_tile_shape=input_tile_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    channels_last=channels_last,
                    tile_offset=chunk_start,
                    input_offsets_device=input_offsets_device,
                    shape_meta=shape_meta,
                    tile_geometry_meta=tile_geometry_meta,
                )
                grad_weight += _spatial_tile_conv_transpose2d_weight_grad(
                    tile_input,
                    weight,
                    tile_grad_output,
                    stride=stride,
                    dilation=dilation,
                )

        return (grad_input, grad_weight, grad_bias, *((None,) * 18))


def _spatial_tile_conv_transpose2d(
    input: NestedTensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride=1,
    padding=0,
    output_padding=0,
    groups: int = 1,
    dilation=1,
    *,
    tile_size: int | tuple[int, int] | str = "auto",
    max_tiles_per_batch: int | str | None = "auto",
    weight_max_tiles_per_batch: int | str | None = "same",
    channels_last: bool = True,
) -> NestedTensor | None:
    r"""Run packed CHW ragged ``conv_transpose2d`` as fixed-size 2D cuDNN tiles."""
    if triton is None or not _can_use_spatial_tile_convolution(input, weight, groups, rank=2, input_channel_dim=0):
        return None

    if isinstance(stride, int):
        stride_pair = (int(stride), int(stride))
    elif isinstance(stride, tuple) and len(stride) == 2:
        stride_pair = (int(stride[0]), int(stride[1]))
    else:
        return None
    if isinstance(padding, int):
        padding_pair = (int(padding), int(padding))
    elif isinstance(padding, tuple) and len(padding) == 2:
        padding_pair = (int(padding[0]), int(padding[1]))
    else:
        return None
    if isinstance(output_padding, int):
        output_padding_pair = (int(output_padding), int(output_padding))
    elif isinstance(output_padding, tuple) and len(output_padding) == 2:
        output_padding_pair = (int(output_padding[0]), int(output_padding[1]))
    else:
        return None
    if isinstance(dilation, int):
        dilation_pair = (int(dilation), int(dilation))
    elif isinstance(dilation, tuple) and len(dilation) == 2:
        dilation_pair = (int(dilation[0]), int(dilation[1]))
    else:
        return None
    if any(value <= 0 for value in (*stride_pair, *dilation_pair)):
        return None
    if any(value < 0 for value in (*padding_pair, *output_padding_pair)):
        return None
    if not _valid_output_padding(output_padding_pair, stride_pair, dilation_pair):
        return None

    out_channels = int(weight.shape[1])
    in_channels = int(weight.shape[0])
    kernel_size = (int(weight.shape[2]), int(weight.shape[3]))
    if any(size <= 0 for size in kernel_size):
        return None
    if not _valid_conv_bias(bias, out_channels=out_channels, device=input._values.device, dtype=input._values.dtype):
        return None

    output_meta = _conv_transpose2d_output_meta(
        input,
        out_channels,
        kernel_size,
        stride_pair,
        padding_pair,
        output_padding_pair,
        dilation_pair,
    )
    if output_meta is None:
        return None
    output_shapes, output_packed_sizes, output_shape_tensor = output_meta
    output_offsets = type(input)._offsets_from_sizes(output_packed_sizes, dtype=torch.long)
    total_out = int(output_offsets[-1].item())
    if total_out == 0:
        output_values = input._values.new_empty((total_out, out_channels))
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

    input_shapes = tuple(_resolve_element_shapes(input))
    requires_grad = torch.is_grad_enabled() and (
        input._values.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    )

    resolved_tile_config = _resolve_spatial_tile2d_config(input_shapes, tile_size, max_tiles_per_batch)
    if resolved_tile_config is None:
        return None
    tile_shape, resolved_max_tiles = resolved_tile_config
    if tile_size == "auto":
        max_output_shape = (
            max(int(shape[1]) for shape in output_shapes),
            max(int(shape[2]) for shape in output_shapes),
        )
        tile_shape = (min(tile_shape[0], max_output_shape[0]), min(tile_shape[1], max_output_shape[1]))
        if max_tiles_per_batch == "auto":
            resolved_max_tiles = _auto_spatial_tile2d_batch(tile_shape, _spatial_tile_area_occupancy(input_shapes))
    if any(size <= 0 for size in tile_shape):
        return None

    tiles = tuple(_conv_transpose2d_spatial_tiles(output_shapes, tile_shape))
    input_tile_shape = _conv_transpose2d_input_tile_shape(
        input_shapes,
        tiles,
        kernel_size=kernel_size,
        stride=stride_pair,
        padding=padding_pair,
        dilation=dilation_pair,
    )
    if any(size <= 0 for size in input_tile_shape):
        return None
    local_output_shape = _conv_transpose2d_local_output_shape(
        input_tile_shape,
        kernel_size=kernel_size,
        stride=stride_pair,
        dilation=dilation_pair,
    )
    if any(size <= 0 for size in local_output_shape):
        return None
    resolved_max_tiles = _cap_spatial_tile2d_batch_for_channels(
        resolved_max_tiles,
        in_channels=in_channels,
        out_channels=out_channels,
        input_tile_shape=input_tile_shape,
        local_output_shape=local_output_shape,
    )

    device = input._values.device
    shape_meta = torch.tensor(
        [
            (int(input_shape[1]), int(input_shape[2]), int(output_shape[1]), int(output_shape[2]))
            for input_shape, output_shape in zip(input_shapes, output_shapes)
        ],
        device=device,
        dtype=torch.int32,
    )
    tile_geometry_rows = _conv_transpose2d_tile_geometry_rows(
        input_shapes,
        tiles,
        input_tile_shape,
        kernel_size=kernel_size,
        stride=stride_pair,
        padding=padding_pair,
        dilation=dilation_pair,
    )
    tile_geometry_meta = torch.tensor(tile_geometry_rows, device=device, dtype=torch.int32)
    input_offsets_device = input._offsets.to(device=device, non_blocking=True)
    output_offsets_device = output_offsets.to(device=device, non_blocking=True)

    input_starts = tuple(int(offset) for offset in input._offsets[:-1].tolist())
    output_starts = tuple(int(offset) for offset in output_offsets[:-1].tolist())
    chunk_size = _spatial_tile_max_batch(resolved_max_tiles, len(tiles))
    resolved_weight_max_tiles = _resolve_spatial_weight_tile2d_batch(
        input_shapes,
        tile_shape,
        weight_max_tiles_per_batch,
        chunk_size,
    )
    if resolved_weight_max_tiles is None:
        return None
    use_channels_last = bool(channels_last and input._values.is_cuda)

    if requires_grad:
        output_values = _SpatialTileConvTranspose2dCudnnFunction.apply(
            input._values,
            weight,
            bias,
            input_offsets_device,
            output_offsets_device,
            shape_meta,
            tile_geometry_meta,
            total_out,
            input_shapes,
            output_shapes,
            input_starts,
            output_starts,
            tiles,
            stride_pair,
            padding_pair,
            dilation_pair,
            input_tile_shape,
            local_output_shape,
            chunk_size,
            resolved_weight_max_tiles,
            use_channels_last,
        )
    else:
        output_values = _spatial_tile_conv_transpose2d_forward_values(
            input._values,
            weight.contiguous(),
            bias,
            output_size=total_out,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_starts=input_starts,
            output_starts=output_starts,
            tiles=tiles,
            stride=stride_pair,
            padding=padding_pair,
            dilation=dilation_pair,
            input_tile_shape=input_tile_shape,
            max_tiles_per_batch=chunk_size,
            channels_last=use_channels_last,
            input_offsets_device=input_offsets_device,
            output_offsets_device=output_offsets_device,
            shape_meta=shape_meta,
            tile_geometry_meta=tile_geometry_meta,
        )

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
