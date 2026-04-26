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

r"""``conv_transpose1d`` handler for NestedTensor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from ._convolution import (
    _can_use_spatial_tile_convolution,
    _cap_spatial_tile_batch_for_channels,
    _ceil_div,
    _conv_transpose_output_size,
    _has_cuda_tensors,
    _packed_pointwise_conv_transpose,
    _packed_pointwise_conv_transpose_channel_dim,
    _per_element,
    _resolve_element_shapes,
    _spatial_tile_length_occupancy,
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
# NestedTensor conv_transpose1d dispatch
# ---------------------------------------------------------------------------


def conv_transpose1d(
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
            1,
        )
        if channel_dim is not None:
            return _packed_pointwise_conv_transpose(input, weight, bias, channel_dim)
        output = _spatial_tile_conv_transpose1d(
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
        F.conv_transpose1d if _fn is None else _fn,
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


def _conv_transpose1d_output_meta(
    input: NestedTensor,
    out_channels: int,
    kernel_size: tuple[int],
    stride: tuple[int],
    padding: tuple[int],
    output_padding: tuple[int],
    dilation: tuple[int],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], Tensor] | None:
    output_shapes = []
    output_packed_sizes = []
    for shape in _resolve_element_shapes(input):
        if len(shape) != 2:
            return None
        _, in_length = shape
        out_length = _conv_transpose_output_size(
            in_length,
            kernel_size[0],
            stride[0],
            padding[0],
            output_padding[0],
            dilation[0],
        )
        if out_length <= 0:
            return None
        output_shapes.append((out_channels, out_length))
        output_packed_sizes.append(out_length)
    if not output_shapes:
        return None

    return tuple(output_shapes), tuple(output_packed_sizes), torch.tensor(output_shapes, dtype=torch.long)


def _conv_transpose1d_spatial_tiles(
    output_shapes: tuple[tuple[int, ...], ...],
    tile_length: int,
) -> list[tuple[int, int, int]]:
    tiles = []
    for batch, shape in enumerate(output_shapes):
        _, out_length = shape
        for out_x in range(0, out_length, tile_length):
            valid_length = min(tile_length, out_length - out_x)
            tiles.append((batch, out_x, valid_length))
    return tiles


def _auto_spatial_tile1d_config(length_occupancy: float) -> tuple[int, int]:
    tile_length = 1024
    return tile_length, _auto_spatial_tile1d_batch(tile_length, length_occupancy)


def _auto_spatial_tile1d_batch(tile_length: int, _length_occupancy: float) -> int:
    target_tile_positions = 65536
    return max(16, min(128, target_tile_positions // max(int(tile_length), 1)))


def _resolve_spatial_tile1d_config(
    input_shapes: tuple[tuple[int, ...], ...],
    tile_size: int | str,
    max_tiles_per_batch: int | str | None,
) -> tuple[int, int | None] | None:
    length_occupancy = _spatial_tile_length_occupancy(input_shapes)
    if isinstance(tile_size, str):
        if tile_size != "auto":
            return None
        tile_length, auto_batch = _auto_spatial_tile1d_config(length_occupancy)
    else:
        if not isinstance(tile_size, int) or tile_size <= 0:
            return None
        tile_length = int(tile_size)
        auto_batch = _auto_spatial_tile1d_batch(tile_length, length_occupancy)

    if isinstance(max_tiles_per_batch, str):
        if max_tiles_per_batch != "auto":
            return None
        return tile_length, auto_batch
    if max_tiles_per_batch is None:
        return tile_length, None
    return tile_length, max(int(max_tiles_per_batch), 1)


def _resolve_spatial_weight_tile1d_batch(
    input_shapes: tuple[tuple[int, ...], ...],
    tile_length: int,
    weight_max_tiles_per_batch: int | str | None,
    default_chunk_size: int,
) -> int | None:
    if isinstance(weight_max_tiles_per_batch, str):
        if weight_max_tiles_per_batch == "same":
            return default_chunk_size
        if weight_max_tiles_per_batch != "auto":
            return None
        auto_batch = _auto_spatial_tile1d_batch(tile_length, _spatial_tile_length_occupancy(input_shapes))
        return min(default_chunk_size, auto_batch)
    if weight_max_tiles_per_batch is None:
        return max(default_chunk_size, 1)
    return max(int(weight_max_tiles_per_batch), 1)


def _conv_transpose1d_tile_origin(
    out_start: int,
    in_size: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    origin = _ceil_div(out_start + padding - dilation * (kernel - 1), stride)
    return min(max(origin, 0), in_size)


def _conv_transpose1d_needed_length(
    out_start: int,
    valid: int,
    in_size: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    origin = _conv_transpose1d_tile_origin(out_start, in_size, kernel, stride, padding, dilation)
    end = (out_start + valid - 1 + padding) // stride + 1
    end = min(max(end, 0), in_size)
    return max(end - origin, 0)


def _conv_transpose1d_input_tile_shape(
    input_shapes: tuple[tuple[int, ...], ...],
    tiles: tuple[tuple[int, int, int], ...],
    *,
    kernel_size: tuple[int],
    stride: tuple[int],
    padding: tuple[int],
    dilation: tuple[int],
) -> tuple[int]:
    max_length = 1
    for batch, out_x, valid_length in tiles:
        _, in_length = input_shapes[batch]
        max_length = max(
            max_length,
            _conv_transpose1d_needed_length(
                out_x,
                valid_length,
                in_length,
                kernel_size[0],
                stride[0],
                padding[0],
                dilation[0],
            ),
        )
    return (max_length,)


def _conv_transpose1d_local_output_shape(
    input_tile_shape: tuple[int],
    *,
    kernel_size: tuple[int],
    stride: tuple[int],
    dilation: tuple[int],
) -> tuple[int]:
    # output_padding is a global output-size disambiguator.  Tile-local cuDNN
    # computes the uncropped transposed-convolution support, then scatter/crop
    # maps that support into the requested global output region.
    return ((input_tile_shape[0] - 1) * stride[0] + dilation[0] * (kernel_size[0] - 1) + 1,)


def _conv_transpose1d_tile_geometry(
    batch: int,
    out_x: int,
    input_shapes: tuple[tuple[int, ...], ...],
    input_tile_shape: tuple[int],
    *,
    kernel_size: tuple[int],
    stride: tuple[int],
    padding: tuple[int],
    dilation: tuple[int],
) -> tuple[int, int, int, int]:
    _, in_length = input_shapes[batch]
    origin = _conv_transpose1d_tile_origin(out_x, in_length, kernel_size[0], stride[0], padding[0], dilation[0])
    copy_length = max(min(input_tile_shape[0], in_length - origin), 0)
    global_offset = origin * stride[0] - padding[0]
    return origin, copy_length, in_length, global_offset


def _conv_transpose1d_tile_geometry_rows(
    input_shapes: tuple[tuple[int, ...], ...],
    tiles: tuple[tuple[int, int, int], ...],
    input_tile_shape: tuple[int],
    *,
    kernel_size: tuple[int],
    stride: tuple[int],
    padding: tuple[int],
    dilation: tuple[int],
) -> tuple[tuple[int, ...], ...]:
    rows = []
    for batch, out_x, valid_length in tiles:
        origin, copy_length, _, global_offset = _conv_transpose1d_tile_geometry(
            batch,
            out_x,
            input_shapes,
            input_tile_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        rows.append((batch, out_x, valid_length, origin, copy_length, global_offset))
    return tuple(rows)


def _new_spatial_tile1d_tensor(
    reference: Tensor,
    shape: tuple[int, int, int],
    *,
    channels_last: bool,
) -> Tensor:
    if channels_last and reference.is_cuda:
        return torch.empty(
            (shape[0], shape[1], 1, shape[2]),
            device=reference.device,
            dtype=reference.dtype,
            memory_format=torch.channels_last,
        ).squeeze(2)
    return reference.new_empty(shape)


if triton is not None:

    @triton.jit
    def _spatial_tile1d_pack_input_kernel(
        input_ptr,
        tile_ptr,
        tile_geometry_ptr,
        input_offsets_ptr,
        tile_offset,
        total,
        tile_stride_n: tl.constexpr,
        tile_stride_c: tl.constexpr,
        tile_stride_l: tl.constexpr,
        in_channels: tl.constexpr,
        input_tile_length: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total
        channel = offsets % in_channels
        tmp = offsets // in_channels
        tile_x = tmp % input_tile_length
        tile_local = tmp // input_tile_length
        tile = tile_offset + tile_local

        batch = tl.load(tile_geometry_ptr + tile * 6, mask=mask, other=0)
        origin = tl.load(tile_geometry_ptr + tile * 6 + 3, mask=mask, other=0)
        copy_length = tl.load(tile_geometry_ptr + tile * 6 + 4, mask=mask, other=0)
        input_base = tl.load(input_offsets_ptr + batch, mask=mask, other=0)

        valid = mask & (tile_x < copy_length)
        input_pos = input_base + origin + tile_x
        values = tl.load(input_ptr + input_pos * in_channels + channel, mask=valid, other=0.0)
        tl.store(
            tile_ptr + tile_local * tile_stride_n + channel * tile_stride_c + tile_x * tile_stride_l,
            values,
            mask=mask,
        )

    @triton.jit
    def _spatial_tile1d_scatter_output_kernel(
        tile_ptr,
        bias_ptr,
        output_ptr,
        tile_geometry_ptr,
        output_offsets_ptr,
        tile_offset,
        total,
        tile_stride_n: tl.constexpr,
        tile_stride_c: tl.constexpr,
        tile_stride_l: tl.constexpr,
        out_channels: tl.constexpr,
        tile_length: tl.constexpr,
        local_length: tl.constexpr,
        has_bias: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total
        channel = offsets % out_channels
        tmp = offsets // out_channels
        region_x = tmp % tile_length
        tile_local = tmp // tile_length
        tile = tile_offset + tile_local

        batch = tl.load(tile_geometry_ptr + tile * 6, mask=mask, other=0)
        out_x = tl.load(tile_geometry_ptr + tile * 6 + 1, mask=mask, other=0)
        valid_length = tl.load(tile_geometry_ptr + tile * 6 + 2, mask=mask, other=0)
        global_offset = tl.load(tile_geometry_ptr + tile * 6 + 5, mask=mask, other=0)
        output_base = tl.load(output_offsets_ptr + batch, mask=mask, other=0)

        valid = mask & (region_x < valid_length)
        src_x = out_x + region_x - global_offset
        src_valid = valid & (src_x >= 0) & (src_x < local_length)
        values = tl.load(
            tile_ptr + tile_local * tile_stride_n + channel * tile_stride_c + src_x * tile_stride_l,
            mask=src_valid,
            other=0.0,
        )
        if has_bias:
            values += tl.load(bias_ptr + channel, mask=valid, other=0.0)
        output_pos = output_base + out_x + region_x
        tl.store(output_ptr + output_pos * out_channels + channel, values, mask=valid)

    @triton.jit
    def _spatial_tile1d_pack_grad_output_kernel(
        grad_output_ptr,
        tile_ptr,
        tile_geometry_ptr,
        output_offsets_ptr,
        tile_offset,
        total,
        grad_output_stride_n: tl.constexpr,
        grad_output_stride_c: tl.constexpr,
        tile_stride_n: tl.constexpr,
        tile_stride_c: tl.constexpr,
        tile_stride_l: tl.constexpr,
        out_channels: tl.constexpr,
        local_length: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total
        channel = offsets % out_channels
        tmp = offsets // out_channels
        local_x = tmp % local_length
        tile_local = tmp // local_length
        tile = tile_offset + tile_local

        batch = tl.load(tile_geometry_ptr + tile * 6, mask=mask, other=0)
        out_x = tl.load(tile_geometry_ptr + tile * 6 + 1, mask=mask, other=0)
        valid_length = tl.load(tile_geometry_ptr + tile * 6 + 2, mask=mask, other=0)
        global_offset = tl.load(tile_geometry_ptr + tile * 6 + 5, mask=mask, other=0)
        output_base = tl.load(output_offsets_ptr + batch, mask=mask, other=0)

        global_x = global_offset + local_x
        valid = mask & (global_x >= out_x) & (global_x < out_x + valid_length)
        output_pos = output_base + global_x
        values = tl.load(
            grad_output_ptr + output_pos * grad_output_stride_n + channel * grad_output_stride_c,
            mask=valid,
            other=0.0,
        )
        tl.store(
            tile_ptr + tile_local * tile_stride_n + channel * tile_stride_c + local_x * tile_stride_l,
            values,
            mask=mask,
        )

    @triton.jit
    def _spatial_tile1d_scatter_grad_input_kernel(
        tile_ptr,
        grad_input_ptr,
        tile_geometry_ptr,
        input_offsets_ptr,
        tile_offset,
        total,
        tile_stride_n: tl.constexpr,
        tile_stride_c: tl.constexpr,
        tile_stride_l: tl.constexpr,
        in_channels: tl.constexpr,
        input_tile_length: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total
        channel = offsets % in_channels
        tmp = offsets // in_channels
        tile_x = tmp % input_tile_length
        tile_local = tmp // input_tile_length
        tile = tile_offset + tile_local

        batch = tl.load(tile_geometry_ptr + tile * 6, mask=mask, other=0)
        origin = tl.load(tile_geometry_ptr + tile * 6 + 3, mask=mask, other=0)
        copy_length = tl.load(tile_geometry_ptr + tile * 6 + 4, mask=mask, other=0)
        input_base = tl.load(input_offsets_ptr + batch, mask=mask, other=0)

        valid = mask & (tile_x < copy_length)
        values = tl.load(
            tile_ptr + tile_local * tile_stride_n + channel * tile_stride_c + tile_x * tile_stride_l,
            mask=valid,
            other=0.0,
        )
        input_pos = input_base + origin + tile_x
        tl.atomic_add(grad_input_ptr + input_pos * in_channels + channel, values, sem="relaxed", mask=valid)

else:  # pragma: no cover - optional dependency/runtime
    _spatial_tile1d_pack_input_kernel = None
    _spatial_tile1d_scatter_output_kernel = None
    _spatial_tile1d_pack_grad_output_kernel = None
    _spatial_tile1d_scatter_grad_input_kernel = None


def _make_spatial_tile1d_input(
    input_values: Tensor,
    input_shapes: tuple[tuple[int, ...], ...],
    input_starts: tuple[int, ...],
    chunk: tuple[tuple[int, int, int], ...],
    *,
    in_channels: int,
    input_tile_shape: tuple[int],
    kernel_size: tuple[int],
    stride: tuple[int],
    padding: tuple[int],
    dilation: tuple[int],
    channels_last: bool,
    tile_offset: int = 0,
    input_offsets_device: Tensor | None = None,
    tile_geometry_meta: Tensor | None = None,
) -> Tensor:
    tile_input = _new_spatial_tile1d_tensor(
        input_values,
        (len(chunk), in_channels, input_tile_shape[0]),
        channels_last=channels_last,
    )

    if triton is not None and _has_cuda_tensors(input_offsets_device, tile_geometry_meta):
        block_size = _spatial_tile_triton_block_size()
        total = len(chunk) * input_tile_shape[0] * in_channels
        grid = (triton.cdiv(total, block_size),)
        _spatial_tile1d_pack_input_kernel[grid](
            input_values,
            tile_input,
            tile_geometry_meta,
            input_offsets_device,
            int(tile_offset),
            total,
            tile_input.stride(0),
            tile_input.stride(1),
            tile_input.stride(2),
            in_channels,
            input_tile_shape[0],
            block_size,
        )
        return tile_input

    tile_input.zero_()
    for tile_index, (batch, out_x, _) in enumerate(chunk):
        origin, copy_length, in_length, _ = _conv_transpose1d_tile_geometry(
            batch,
            out_x,
            input_shapes,
            input_tile_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        if copy_length == 0:
            continue
        input_start = input_starts[batch]
        element = input_values[input_start : input_start + in_length].reshape(in_length, in_channels)
        tile_input[tile_index, :, :copy_length] = element[origin : origin + copy_length].transpose(0, 1)

    return tile_input


def _spatial_tile_conv_transpose1d_tile_forward(
    tile_input: Tensor,
    weight: Tensor,
    *,
    stride: tuple[int],
    dilation: tuple[int],
    channels_last: bool,
) -> Tensor:
    # Keep output_padding at zero for each tile.  Non-zero output_padding only
    # extends the global output shape; applying it per tile would create extra
    # local support and duplicate contributions at tile boundaries.
    if channels_last and tile_input.is_cuda:
        return F.conv_transpose2d(
            tile_input.unsqueeze(2),
            weight.unsqueeze(2).contiguous(),
            None,
            stride=(1, stride[0]),
            padding=(0, 0),
            output_padding=(0, 0),
            groups=1,
            dilation=(1, dilation[0]),
        ).squeeze(2)
    return F.conv_transpose1d(
        tile_input,
        weight,
        None,
        stride=stride,
        padding=0,
        output_padding=0,
        groups=1,
        dilation=dilation,
    )


def _scatter_spatial_tile1d_output(
    output_values: Tensor,
    tile_output: Tensor,
    bias: Tensor | None,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
    output_starts: tuple[int, ...],
    chunk: tuple[tuple[int, int, int], ...],
    *,
    out_channels: int,
    input_tile_shape: tuple[int],
    kernel_size: tuple[int],
    stride: tuple[int],
    padding: tuple[int],
    dilation: tuple[int],
    tile_offset: int = 0,
    output_offsets_device: Tensor | None = None,
    tile_geometry_meta: Tensor | None = None,
) -> None:
    local_length = int(tile_output.shape[2])
    if triton is not None and _has_cuda_tensors(output_offsets_device, tile_geometry_meta):
        tile_length = max(int(tile[2]) for tile in chunk)
        block_size = _spatial_tile_triton_block_size()
        total = len(chunk) * tile_length * out_channels
        grid = (triton.cdiv(total, block_size),)
        _spatial_tile1d_scatter_output_kernel[grid](
            tile_output,
            bias if bias is not None else output_values,
            output_values,
            tile_geometry_meta,
            output_offsets_device,
            int(tile_offset),
            total,
            tile_output.stride(0),
            tile_output.stride(1),
            tile_output.stride(2),
            out_channels,
            tile_length,
            local_length,
            bias is not None,
            block_size,
        )
        return

    for tile_index, (batch, out_x, valid_length) in enumerate(chunk):
        _, out_length = output_shapes[batch]
        output_start = output_starts[batch]
        element_output = output_values[output_start : output_start + out_length].reshape(out_length, out_channels)
        region = element_output[out_x : out_x + valid_length]
        if bias is None:
            region.zero_()
        else:
            region.copy_(bias.reshape(1, out_channels).expand_as(region))

        _, _, _, global_offset = _conv_transpose1d_tile_geometry(
            batch,
            out_x,
            input_shapes,
            input_tile_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        hit = _tile_output_intersection(out_x - global_offset, valid_length, local_length)
        if hit is None:
            continue
        dst_x0, dst_x1, src_x0, src_x1 = hit
        region[dst_x0:dst_x1] += tile_output[tile_index, :, src_x0:src_x1].transpose(0, 1)


def _make_spatial_tile1d_grad_output(
    grad_output_values: Tensor,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
    output_starts: tuple[int, ...],
    chunk: tuple[tuple[int, int, int], ...],
    *,
    out_channels: int,
    input_tile_shape: tuple[int],
    local_output_shape: tuple[int],
    kernel_size: tuple[int],
    stride: tuple[int],
    padding: tuple[int],
    dilation: tuple[int],
    channels_last: bool,
    tile_offset: int = 0,
    output_offsets_device: Tensor | None = None,
    tile_geometry_meta: Tensor | None = None,
) -> Tensor:
    tile_grad_output = _new_spatial_tile1d_tensor(
        grad_output_values,
        (len(chunk), out_channels, local_output_shape[0]),
        channels_last=channels_last,
    )

    if triton is not None and _has_cuda_tensors(output_offsets_device, tile_geometry_meta):
        block_size = _spatial_tile_triton_block_size()
        total = len(chunk) * local_output_shape[0] * out_channels
        grid = (triton.cdiv(total, block_size),)
        _spatial_tile1d_pack_grad_output_kernel[grid](
            grad_output_values,
            tile_grad_output,
            tile_geometry_meta,
            output_offsets_device,
            int(tile_offset),
            total,
            grad_output_values.stride(0),
            grad_output_values.stride(1),
            tile_grad_output.stride(0),
            tile_grad_output.stride(1),
            tile_grad_output.stride(2),
            out_channels,
            local_output_shape[0],
            block_size,
        )
        return tile_grad_output

    tile_grad_output.zero_()
    for tile_index, (batch, out_x, valid_length) in enumerate(chunk):
        _, out_length = output_shapes[batch]
        output_start = output_starts[batch]
        element_grad = grad_output_values[output_start : output_start + out_length].reshape(out_length, out_channels)
        _, _, _, global_offset = _conv_transpose1d_tile_geometry(
            batch,
            out_x,
            input_shapes,
            input_tile_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        hit = _tile_output_intersection(out_x - global_offset, valid_length, local_output_shape[0])
        if hit is None:
            continue
        dst_x0, dst_x1, src_x0, src_x1 = hit
        tile_grad_output[tile_index, :, src_x0:src_x1] = element_grad[out_x + dst_x0 : out_x + dst_x1].transpose(0, 1)

    return tile_grad_output


def _scatter_spatial_tile1d_grad_input(
    grad_input_values: Tensor,
    tile_grad_input: Tensor,
    input_shapes: tuple[tuple[int, ...], ...],
    input_starts: tuple[int, ...],
    chunk: tuple[tuple[int, int, int], ...],
    *,
    in_channels: int,
    input_tile_shape: tuple[int],
    kernel_size: tuple[int],
    stride: tuple[int],
    padding: tuple[int],
    dilation: tuple[int],
    tile_offset: int = 0,
    input_offsets_device: Tensor | None = None,
    tile_geometry_meta: Tensor | None = None,
) -> None:
    if triton is not None and _has_cuda_tensors(input_offsets_device, tile_geometry_meta):
        block_size = _spatial_tile_triton_block_size()
        total = len(chunk) * input_tile_shape[0] * in_channels
        grid = (triton.cdiv(total, block_size),)
        _spatial_tile1d_scatter_grad_input_kernel[grid](
            tile_grad_input,
            grad_input_values,
            tile_geometry_meta,
            input_offsets_device,
            int(tile_offset),
            total,
            tile_grad_input.stride(0),
            tile_grad_input.stride(1),
            tile_grad_input.stride(2),
            in_channels,
            input_tile_shape[0],
            block_size,
        )
        return

    for tile_index, (batch, out_x, _) in enumerate(chunk):
        origin, copy_length, in_length, _ = _conv_transpose1d_tile_geometry(
            batch,
            out_x,
            input_shapes,
            input_tile_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        if copy_length == 0:
            continue
        input_start = input_starts[batch]
        element_grad = grad_input_values[input_start : input_start + in_length].reshape(in_length, in_channels)
        element_grad[origin : origin + copy_length] += tile_grad_input[
            tile_index,
            :,
            :copy_length,
        ].transpose(0, 1)


def _spatial_tile_conv_transpose1d_forward_values(
    input_values: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    *,
    output_size: int,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
    input_starts: tuple[int, ...],
    output_starts: tuple[int, ...],
    tiles: tuple[tuple[int, int, int], ...],
    stride: tuple[int],
    padding: tuple[int],
    dilation: tuple[int],
    input_tile_shape: tuple[int],
    max_tiles_per_batch: int,
    channels_last: bool,
    input_offsets_device: Tensor | None = None,
    output_offsets_device: Tensor | None = None,
    tile_geometry_meta: Tensor | None = None,
) -> Tensor:
    in_channels = int(weight.shape[0])
    out_channels = int(weight.shape[1])
    kernel_size = (int(weight.shape[2]),)
    output_values = input_values.new_empty((output_size, out_channels))
    chunk_size = _spatial_tile_max_batch(max_tiles_per_batch, len(tiles))

    for chunk_start in range(0, len(tiles), chunk_size):
        chunk = tiles[chunk_start : chunk_start + chunk_size]
        tile_input = _make_spatial_tile1d_input(
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
            tile_geometry_meta=tile_geometry_meta,
        )
        tile_output = _spatial_tile_conv_transpose1d_tile_forward(
            tile_input,
            weight,
            stride=stride,
            dilation=dilation,
            channels_last=channels_last,
        )
        _scatter_spatial_tile1d_output(
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
            tile_geometry_meta=tile_geometry_meta,
        )

    return output_values


def _spatial_tile_conv_transpose1d_input_grad(
    input_size: tuple[int, int, int],
    weight: Tensor,
    grad_output: Tensor,
    *,
    stride: tuple[int],
    dilation: tuple[int],
    channels_last: bool,
) -> Tensor:
    if channels_last and grad_output.is_cuda:
        input_meta = torch.empty(
            (input_size[0], input_size[1], 1, input_size[2]),
            device=grad_output.device,
            dtype=grad_output.dtype,
            memory_format=torch.channels_last,
        )
        return torch.ops.aten.convolution_backward(
            grad_output.unsqueeze(2),
            input_meta,
            weight.unsqueeze(2).contiguous(),
            None,
            (1, stride[0]),
            (0, 0),
            (1, dilation[0]),
            True,
            [0],
            1,
            (True, False, False),
        )[0].squeeze(2)
    input_meta = grad_output.new_empty(input_size)
    return torch.ops.aten.convolution_backward(
        grad_output,
        input_meta,
        weight,
        None,
        stride,
        (0,),
        dilation,
        True,
        [0],
        1,
        (True, False, False),
    )[0]


def _spatial_tile_conv_transpose1d_weight_grad(
    input: Tensor,
    weight: Tensor,
    grad_output: Tensor,
    *,
    stride: tuple[int],
    dilation: tuple[int],
    channels_last: bool,
) -> Tensor:
    if channels_last and input.is_cuda:
        grad_weight = torch.ops.aten.convolution_backward(
            grad_output.unsqueeze(2),
            input.unsqueeze(2),
            weight.unsqueeze(2).contiguous(),
            None,
            (1, stride[0]),
            (0, 0),
            (1, dilation[0]),
            True,
            [0],
            1,
            (False, True, False),
        )[1]
        return grad_weight.squeeze(2)
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
        (0,),
        dilation,
        True,
        [0],
        1,
        (False, True, False),
    )[1]


class _SpatialTileConvTranspose1dCudnnFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_values: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        input_offsets_device: Tensor,
        output_offsets_device: Tensor,
        tile_geometry_meta: Tensor,
        output_size: int,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shapes: tuple[tuple[int, ...], ...],
        input_starts: tuple[int, ...],
        output_starts: tuple[int, ...],
        tiles: tuple[tuple[int, int, int], ...],
        stride: tuple[int],
        padding: tuple[int],
        dilation: tuple[int],
        input_tile_shape: tuple[int],
        local_output_shape: tuple[int],
        max_tiles_per_batch: int,
        weight_max_tiles_per_batch: int,
        channels_last: bool,
    ) -> Tensor:
        weight = weight.contiguous()
        output_values = _spatial_tile_conv_transpose1d_forward_values(
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
            tile_geometry_meta=tile_geometry_meta,
        )
        ctx.save_for_backward(input_values, weight, input_offsets_device, output_offsets_device, tile_geometry_meta)
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
        input_values, weight, input_offsets_device, output_offsets_device, tile_geometry_meta = ctx.saved_tensors
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
        kernel_size = (int(weight.shape[2]),)

        grad_input = torch.zeros_like(input_values) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        grad_bias = grad_output_values.sum(dim=0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        chunk_size = _spatial_tile_max_batch(max_tiles_per_batch, len(tiles))

        if grad_input is not None and (grad_weight is None or weight_max_tiles_per_batch != chunk_size):
            for chunk_start in range(0, len(tiles), chunk_size):
                chunk = tiles[chunk_start : chunk_start + chunk_size]
                tile_grad_output = _make_spatial_tile1d_grad_output(
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
                    tile_geometry_meta=tile_geometry_meta,
                )
                tile_grad_input = _spatial_tile_conv_transpose1d_input_grad(
                    (len(chunk), in_channels, input_tile_shape[0]),
                    weight,
                    tile_grad_output,
                    stride=stride,
                    dilation=dilation,
                    channels_last=channels_last,
                )
                _scatter_spatial_tile1d_grad_input(
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
                    tile_geometry_meta=tile_geometry_meta,
                )

        if grad_weight is not None and (grad_input is None or weight_max_tiles_per_batch != chunk_size):
            weight_chunk_size = _spatial_tile_max_batch(weight_max_tiles_per_batch, len(tiles))
            for chunk_start in range(0, len(tiles), weight_chunk_size):
                chunk = tiles[chunk_start : chunk_start + weight_chunk_size]
                tile_grad_output = _make_spatial_tile1d_grad_output(
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
                    tile_geometry_meta=tile_geometry_meta,
                )
                tile_input = _make_spatial_tile1d_input(
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
                    tile_geometry_meta=tile_geometry_meta,
                )
                grad_weight += _spatial_tile_conv_transpose1d_weight_grad(
                    tile_input,
                    weight,
                    tile_grad_output,
                    stride=stride,
                    dilation=dilation,
                    channels_last=channels_last,
                )

        if grad_input is not None and grad_weight is not None and weight_max_tiles_per_batch == chunk_size:
            for chunk_start in range(0, len(tiles), chunk_size):
                chunk = tiles[chunk_start : chunk_start + chunk_size]
                tile_grad_output = _make_spatial_tile1d_grad_output(
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
                    tile_geometry_meta=tile_geometry_meta,
                )
                tile_grad_input = _spatial_tile_conv_transpose1d_input_grad(
                    (len(chunk), in_channels, input_tile_shape[0]),
                    weight,
                    tile_grad_output,
                    stride=stride,
                    dilation=dilation,
                    channels_last=channels_last,
                )
                _scatter_spatial_tile1d_grad_input(
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
                    tile_geometry_meta=tile_geometry_meta,
                )
                tile_input = _make_spatial_tile1d_input(
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
                    tile_geometry_meta=tile_geometry_meta,
                )
                grad_weight += _spatial_tile_conv_transpose1d_weight_grad(
                    tile_input,
                    weight,
                    tile_grad_output,
                    stride=stride,
                    dilation=dilation,
                    channels_last=channels_last,
                )

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _spatial_tile_conv_transpose1d(
    input: NestedTensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride=1,
    padding=0,
    output_padding=0,
    groups: int = 1,
    dilation=1,
    *,
    tile_size: int | str = "auto",
    max_tiles_per_batch: int | str | None = "auto",
    weight_max_tiles_per_batch: int | str | None = "same",
    channels_last: bool = True,
) -> NestedTensor | None:
    r"""Run packed CL ragged ``conv_transpose1d`` as fixed-size 1D cuDNN tiles."""
    if triton is None or not _can_use_spatial_tile_convolution(input, weight, groups, rank=1, input_channel_dim=0):
        return None

    if not isinstance(stride, int) or not isinstance(padding, int):
        return None
    if not isinstance(output_padding, int) or not isinstance(dilation, int):
        return None
    stride_single = (int(stride),)
    padding_single = (int(padding),)
    output_padding_single = (int(output_padding),)
    dilation_single = (int(dilation),)
    if stride_single[0] <= 0 or dilation_single[0] <= 0 or padding_single[0] < 0 or output_padding_single[0] < 0:
        return None
    if not _valid_output_padding(output_padding_single, stride_single, dilation_single):
        return None

    in_channels = int(weight.shape[0])
    out_channels = int(weight.shape[1])
    kernel_size = (int(weight.shape[2]),)
    if kernel_size[0] <= 0:
        return None
    if not _valid_conv_bias(bias, out_channels=out_channels, device=input._values.device, dtype=input._values.dtype):
        return None

    output_meta = _conv_transpose1d_output_meta(
        input,
        out_channels,
        kernel_size,
        stride_single,
        padding_single,
        output_padding_single,
        dilation_single,
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
    resolved_tile_config = _resolve_spatial_tile1d_config(input_shapes, tile_size, max_tiles_per_batch)
    if resolved_tile_config is None:
        return None
    tile_length, resolved_max_tiles = resolved_tile_config
    if tile_size == "auto":
        max_output_length = max(int(shape[1]) for shape in output_shapes)
        if tile_length > max_output_length:
            tile_length = max_output_length
            if max_tiles_per_batch == "auto":
                resolved_max_tiles = _auto_spatial_tile1d_batch(
                    tile_length,
                    _spatial_tile_length_occupancy(input_shapes),
                )
    if tile_length <= 0:
        return None

    tiles = tuple(_conv_transpose1d_spatial_tiles(output_shapes, tile_length))
    input_tile_shape = _conv_transpose1d_input_tile_shape(
        input_shapes,
        tiles,
        kernel_size=kernel_size,
        stride=stride_single,
        padding=padding_single,
        dilation=dilation_single,
    )
    if input_tile_shape[0] <= 0:
        return None
    local_output_shape = _conv_transpose1d_local_output_shape(
        input_tile_shape,
        kernel_size=kernel_size,
        stride=stride_single,
        dilation=dilation_single,
    )
    if local_output_shape[0] <= 0:
        return None
    resolved_max_tiles = _cap_spatial_tile_batch_for_channels(
        resolved_max_tiles,
        in_channels=in_channels,
        out_channels=out_channels,
        input_tile_shape=input_tile_shape,
        local_output_shape=local_output_shape,
    )

    input_starts = tuple(int(offset) for offset in input._offsets[:-1].tolist())
    output_starts = tuple(int(offset) for offset in output_offsets[:-1].tolist())
    chunk_size = _spatial_tile_max_batch(resolved_max_tiles, len(tiles))
    resolved_weight_max_tiles = _resolve_spatial_weight_tile1d_batch(
        input_shapes,
        tile_length,
        weight_max_tiles_per_batch,
        chunk_size,
    )
    if resolved_weight_max_tiles is None:
        return None

    device = input._values.device
    tile_geometry_rows = _conv_transpose1d_tile_geometry_rows(
        input_shapes,
        tiles,
        input_tile_shape,
        kernel_size=kernel_size,
        stride=stride_single,
        padding=padding_single,
        dilation=dilation_single,
    )
    tile_geometry_meta = torch.tensor(tile_geometry_rows, device=device, dtype=torch.int64)
    input_offsets_device = input._offsets.to(device=device, non_blocking=True)
    output_offsets_device = output_offsets.to(device=device, non_blocking=True)
    requires_grad = torch.is_grad_enabled() and (
        input._values.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    )
    use_channels_last = bool(channels_last and input._values.is_cuda)

    if requires_grad:
        output_values = _SpatialTileConvTranspose1dCudnnFunction.apply(
            input._values,
            weight,
            bias,
            input_offsets_device,
            output_offsets_device,
            tile_geometry_meta,
            total_out,
            input_shapes,
            output_shapes,
            input_starts,
            output_starts,
            tiles,
            stride_single,
            padding_single,
            dilation_single,
            input_tile_shape,
            local_output_shape,
            chunk_size,
            resolved_weight_max_tiles,
            use_channels_last,
        )
    else:
        output_values = _spatial_tile_conv_transpose1d_forward_values(
            input._values,
            weight.contiguous(),
            bias,
            output_size=total_out,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_starts=input_starts,
            output_starts=output_starts,
            tiles=tiles,
            stride=stride_single,
            padding=padding_single,
            dilation=dilation_single,
            input_tile_shape=input_tile_shape,
            max_tiles_per_batch=chunk_size,
            channels_last=use_channels_last,
            input_offsets_device=input_offsets_device,
            output_offsets_device=output_offsets_device,
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
