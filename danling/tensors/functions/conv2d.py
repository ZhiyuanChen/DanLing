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
r"""``conv2d`` handlers for NestedTensor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from ..ops import _check_execution_guard, _ExecutionGuardKind
from ._convolution import (
    _can_use_spatial_tile_convolution,
    _conv_output_size,
    _has_cuda_tensors,
    _resolve_element_shapes,
    _spatial_tile_area_occupancy,
    _spatial_tile_max_batch,
    _spatial_tile_triton_block_size,
    _valid_conv_bias,
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
# NestedTensor conv2d dispatch
# ---------------------------------------------------------------------------


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, _fn=None):
    r"""Use packed pointwise or spatial tile conv2d, otherwise fall back to per-element conv."""
    if len(input) != 0:
        channel_dim = _packed_pointwise_conv2d_channel_dim(input, weight, stride, padding, dilation, groups)
        if channel_dim is not None:
            return _packed_pointwise_conv2d(input, weight, bias, channel_dim)
        output = _spatial_tile_conv2d(input, weight, bias, stride, padding, dilation, groups)
        if output is not None:
            return output
    return _per_element_conv2d(input, weight, bias, stride, padding, dilation, groups, _fn=_fn)


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------


def _packed_pointwise_conv2d_channel_dim(input: NestedTensor, weight, stride, padding, dilation, groups) -> int | None:
    r"""Return packed channel dim when ``conv2d`` can run as one packed linear op."""
    if groups != 1 or not isinstance(weight, Tensor) or weight.dim() != 4:
        return None
    if tuple(int(size) for size in weight.shape[2:]) != (1, 1):
        return None
    if (stride != 1 and stride != (1, 1)) or (dilation != 1 and dilation != (1, 1)):
        return None
    if isinstance(padding, str):
        padding_pair = (0, 0) if padding in {"same", "valid"} else None
    elif isinstance(padding, int):
        padding_pair = (int(padding), int(padding))
    elif isinstance(padding, tuple) and len(padding) == 2:
        padding_pair = (int(padding[0]), int(padding[1]))
    else:
        padding_pair = None
    if padding_pair != (0, 0):
        return None
    if input._physical_shape.size(1) != 3:
        return None
    if input._element_shapes is not None and any(len(shape) != 3 for shape in input._element_shapes):
        return None

    in_channels = int(weight.shape[1])
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


def _packed_pointwise_conv2d(
    input: NestedTensor,
    weight: Tensor,
    bias: Tensor | None,
    channel_dim: int,
) -> NestedTensor:
    r"""Run 1x1 conv2d over all packed valid pixels with one dense linear op."""
    out_channels = int(weight.shape[0])
    values = input._values
    moved = channel_dim != values.dim() - 1
    if moved:
        values = values.movedim(channel_dim, -1)
    output = F.linear(values.reshape(-1, int(weight.shape[1])), weight[:, :, 0, 0], bias)
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


def _per_element_conv2d(input: NestedTensor, *args, _fn=None, **kwargs) -> NestedTensor:
    _check_execution_guard(_ExecutionGuardKind.STORAGE_MAP, "_per_element_conv2d")
    cls = type(input)
    if len(input) == 0:
        return cls([], **input._meta(include_dtype=True))
    conv2d = F.conv2d if _fn is None else _fn
    with torch._C.DisableTorchFunctionSubclass():
        results = [conv2d(t, *args, **kwargs) for t in input._storage]
    return cls(results, **input._meta())


# ---------------------------------------------------------------------------
# Spatial tile CuDNN implementation
# ---------------------------------------------------------------------------


def _conv2d_output_meta(
    input: NestedTensor,
    out_channels: int,
    kernel_h: int,
    kernel_w: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], Tensor] | None:
    output_shapes = []
    output_packed_sizes = []
    for shape in _resolve_element_shapes(input):
        if len(shape) != 3:
            return None
        _, in_h, in_w = shape
        out_h = _conv_output_size(in_h, kernel_h, stride[0], padding[0], dilation[0])
        out_w = _conv_output_size(in_w, kernel_w, stride[1], padding[1], dilation[1])
        if out_h <= 0 or out_w <= 0:
            return None
        out_hw = out_h * out_w
        output_shapes.append((out_channels, out_h, out_w))
        output_packed_sizes.append(out_hw)
    if not output_shapes:
        return None

    output_shape_tensor = torch.tensor(output_shapes, dtype=torch.long)
    return tuple(output_shapes), tuple(output_packed_sizes), output_shape_tensor


def _conv2d_spatial_tiles(
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


def _auto_spatial_tile2d_config(_area_occupancy: float) -> tuple[tuple[int, int], int]:
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
    output_tile_shape: tuple[int, int],
) -> int:
    target_elements = 128 * 1024 * 1024
    input_elements = int(in_channels) * int(input_tile_shape[0]) * int(input_tile_shape[1])
    output_elements = int(out_channels) * int(output_tile_shape[0]) * int(output_tile_shape[1])
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


if triton is not None:

    @triton.jit
    def _spatial_tile_pack_input_kernel(
        input_ptr,
        tile_ptr,
        tile_meta_ptr,
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
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        padding_h: tl.constexpr,
        padding_w: tl.constexpr,
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

        batch = tl.load(tile_meta_ptr + tile * 5, mask=mask, other=0)
        out_y = tl.load(tile_meta_ptr + tile * 5 + 1, mask=mask, other=0)
        out_x = tl.load(tile_meta_ptr + tile * 5 + 2, mask=mask, other=0)
        in_h = tl.load(shape_meta_ptr + batch * 4, mask=mask, other=0)
        in_w = tl.load(shape_meta_ptr + batch * 4 + 1, mask=mask, other=0)
        in_base = tl.load(input_offsets_ptr + batch, mask=mask, other=0)

        input_y = out_y * stride_h - padding_h + tile_y
        input_x = out_x * stride_w - padding_w + tile_x
        valid = mask & (input_y >= 0) & (input_y < in_h) & (input_x >= 0) & (input_x < in_w)
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

    # Retained for tuning; the production path currently favors the flat copy kernels.
    @triton.jit
    def _spatial_tile_pack_input_channels_last_kernel(
        input_ptr,
        tile_ptr,
        tile_meta_ptr,
        input_offsets_ptr,
        shape_meta_ptr,
        tile_offset,
        tile_count: tl.constexpr,
        tile_stride_n: tl.constexpr,
        tile_stride_c: tl.constexpr,
        tile_stride_h: tl.constexpr,
        tile_stride_w: tl.constexpr,
        in_channels: tl.constexpr,
        input_tile_h: tl.constexpr,
        input_tile_w: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        padding_h: tl.constexpr,
        padding_w: tl.constexpr,
        block_hw: tl.constexpr,
        block_c: tl.constexpr,
    ):
        tile_local = tl.program_id(0)
        spatial = tl.program_id(1) * block_hw + tl.arange(0, block_hw)
        channel = tl.program_id(2) * block_c + tl.arange(0, block_c)
        tile = tile_offset + tile_local

        batch = tl.load(tile_meta_ptr + tile * 5)
        out_y = tl.load(tile_meta_ptr + tile * 5 + 1)
        out_x = tl.load(tile_meta_ptr + tile * 5 + 2)
        in_h = tl.load(shape_meta_ptr + batch * 4)
        in_w = tl.load(shape_meta_ptr + batch * 4 + 1)
        in_base = tl.load(input_offsets_ptr + batch)

        tile_y = spatial // input_tile_w
        tile_x = spatial - tile_y * input_tile_w
        input_y = out_y * stride_h - padding_h + tile_y
        input_x = out_x * stride_w - padding_w + tile_x
        spatial_mask = spatial < input_tile_h * input_tile_w
        channel_mask = channel < in_channels
        valid = spatial_mask & (input_y >= 0) & (input_y < in_h) & (input_x >= 0) & (input_x < in_w)
        input_pos = in_base + input_y * in_w + input_x
        values = tl.load(
            input_ptr + input_pos[:, None] * in_channels + channel[None, :],
            mask=valid[:, None] & channel_mask[None, :],
            other=0.0,
        )
        tl.store(
            tile_ptr
            + tile_local * tile_stride_n
            + channel[None, :] * tile_stride_c
            + tile_y[:, None] * tile_stride_h
            + tile_x[:, None] * tile_stride_w,
            values,
            mask=spatial_mask[:, None] & channel_mask[None, :] & (tile_local < tile_count),
        )

    @triton.jit
    def _spatial_tile_pack_input_local_kernel(
        input_ptr,
        tile_ptr,
        tile_meta_ptr,
        input_offsets_ptr,
        shape_meta_ptr,
        tile_offset,
        tile_stride_n: tl.constexpr,
        tile_stride_c: tl.constexpr,
        tile_stride_h: tl.constexpr,
        tile_stride_w: tl.constexpr,
        in_channels: tl.constexpr,
        input_tile_h: tl.constexpr,
        input_tile_w: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        padding_h: tl.constexpr,
        padding_w: tl.constexpr,
        block_size: tl.constexpr,
    ):
        tile_local = tl.program_id(0)
        offsets = tl.program_id(1) * block_size + tl.arange(0, block_size)
        mask = offsets < input_tile_h * input_tile_w * in_channels
        channel = offsets % in_channels
        pixel = offsets // in_channels
        tile_y = pixel // input_tile_w
        tile_x = pixel - tile_y * input_tile_w
        tile = tile_offset + tile_local

        batch = tl.load(tile_meta_ptr + tile * 5)
        out_y = tl.load(tile_meta_ptr + tile * 5 + 1)
        out_x = tl.load(tile_meta_ptr + tile * 5 + 2)
        in_h = tl.load(shape_meta_ptr + batch * 4)
        in_w = tl.load(shape_meta_ptr + batch * 4 + 1)
        in_base = tl.load(input_offsets_ptr + batch)

        input_y = out_y * stride_h - padding_h + tile_y
        input_x = out_x * stride_w - padding_w + tile_x
        valid = mask & (input_y >= 0) & (input_y < in_h) & (input_x >= 0) & (input_x < in_w)
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
    def _spatial_tile_pack_output_kernel(
        output_ptr,
        tile_ptr,
        tile_meta_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        tile_offset,
        total,
        output_stride_n: tl.constexpr,
        output_stride_c: tl.constexpr,
        tile_stride_n: tl.constexpr,
        tile_stride_c: tl.constexpr,
        tile_stride_h: tl.constexpr,
        tile_stride_w: tl.constexpr,
        out_channels: tl.constexpr,
        tile_h: tl.constexpr,
        tile_w: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total
        channel = offsets % out_channels
        tmp = offsets // out_channels
        local_x = tmp % tile_w
        tmp = tmp // tile_w
        local_y = tmp % tile_h
        tile_local = tmp // tile_h
        tile = tile_offset + tile_local

        batch = tl.load(tile_meta_ptr + tile * 5, mask=mask, other=0)
        out_y = tl.load(tile_meta_ptr + tile * 5 + 1, mask=mask, other=0)
        out_x = tl.load(tile_meta_ptr + tile * 5 + 2, mask=mask, other=0)
        valid_h = tl.load(tile_meta_ptr + tile * 5 + 3, mask=mask, other=0)
        valid_w = tl.load(tile_meta_ptr + tile * 5 + 4, mask=mask, other=0)
        out_w = tl.load(shape_meta_ptr + batch * 4 + 3, mask=mask, other=0)
        out_base = tl.load(output_offsets_ptr + batch, mask=mask, other=0)

        valid = mask & (local_y < valid_h) & (local_x < valid_w)
        output_pos = out_base + (out_y + local_y) * out_w + out_x + local_x
        values = tl.load(output_ptr + output_pos * output_stride_n + channel * output_stride_c, mask=valid, other=0.0)
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
    def _spatial_tile_pack_output_channels_last_kernel(
        output_ptr,
        tile_ptr,
        tile_meta_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        tile_offset,
        tile_count: tl.constexpr,
        output_stride_n: tl.constexpr,
        output_stride_c: tl.constexpr,
        tile_stride_n: tl.constexpr,
        tile_stride_c: tl.constexpr,
        tile_stride_h: tl.constexpr,
        tile_stride_w: tl.constexpr,
        out_channels: tl.constexpr,
        tile_h: tl.constexpr,
        tile_w: tl.constexpr,
        block_hw: tl.constexpr,
        block_c: tl.constexpr,
    ):
        tile_local = tl.program_id(0)
        spatial = tl.program_id(1) * block_hw + tl.arange(0, block_hw)
        channel = tl.program_id(2) * block_c + tl.arange(0, block_c)
        tile = tile_offset + tile_local

        batch = tl.load(tile_meta_ptr + tile * 5)
        out_y = tl.load(tile_meta_ptr + tile * 5 + 1)
        out_x = tl.load(tile_meta_ptr + tile * 5 + 2)
        valid_h = tl.load(tile_meta_ptr + tile * 5 + 3)
        valid_w = tl.load(tile_meta_ptr + tile * 5 + 4)
        out_w = tl.load(shape_meta_ptr + batch * 4 + 3)
        out_base = tl.load(output_offsets_ptr + batch)

        local_y = spatial // tile_w
        local_x = spatial - local_y * tile_w
        spatial_mask = spatial < tile_h * tile_w
        channel_mask = channel < out_channels
        valid = spatial_mask & (local_y < valid_h) & (local_x < valid_w)
        output_pos = out_base + (out_y + local_y) * out_w + out_x + local_x
        values = tl.load(
            output_ptr + output_pos[:, None] * output_stride_n + channel[None, :] * output_stride_c,
            mask=valid[:, None] & channel_mask[None, :],
            other=0.0,
        )
        tl.store(
            tile_ptr
            + tile_local * tile_stride_n
            + channel[None, :] * tile_stride_c
            + local_y[:, None] * tile_stride_h
            + local_x[:, None] * tile_stride_w,
            values,
            mask=spatial_mask[:, None] & channel_mask[None, :] & (tile_local < tile_count),
        )

    @triton.jit
    def _spatial_tile_pack_output_local_kernel(
        output_ptr,
        tile_ptr,
        tile_meta_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        tile_offset,
        output_stride_n: tl.constexpr,
        output_stride_c: tl.constexpr,
        tile_stride_n: tl.constexpr,
        tile_stride_c: tl.constexpr,
        tile_stride_h: tl.constexpr,
        tile_stride_w: tl.constexpr,
        out_channels: tl.constexpr,
        tile_h: tl.constexpr,
        tile_w: tl.constexpr,
        block_size: tl.constexpr,
    ):
        tile_local = tl.program_id(0)
        offsets = tl.program_id(1) * block_size + tl.arange(0, block_size)
        mask = offsets < tile_h * tile_w * out_channels
        channel = offsets % out_channels
        pixel = offsets // out_channels
        local_y = pixel // tile_w
        local_x = pixel - local_y * tile_w
        tile = tile_offset + tile_local

        batch = tl.load(tile_meta_ptr + tile * 5)
        out_y = tl.load(tile_meta_ptr + tile * 5 + 1)
        out_x = tl.load(tile_meta_ptr + tile * 5 + 2)
        valid_h = tl.load(tile_meta_ptr + tile * 5 + 3)
        valid_w = tl.load(tile_meta_ptr + tile * 5 + 4)
        out_w = tl.load(shape_meta_ptr + batch * 4 + 3)
        out_base = tl.load(output_offsets_ptr + batch)

        valid = mask & (local_y < valid_h) & (local_x < valid_w)
        output_pos = out_base + (out_y + local_y) * out_w + out_x + local_x
        values = tl.load(output_ptr + output_pos * output_stride_n + channel * output_stride_c, mask=valid, other=0.0)
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
    def _spatial_tile_scatter_output_kernel(
        tile_ptr,
        bias_ptr,
        output_ptr,
        tile_meta_ptr,
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
        has_bias: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total
        channel = offsets % out_channels
        tmp = offsets // out_channels
        local_x = tmp % tile_w
        tmp = tmp // tile_w
        local_y = tmp % tile_h
        tile_local = tmp // tile_h
        tile = tile_offset + tile_local

        batch = tl.load(tile_meta_ptr + tile * 5, mask=mask, other=0)
        out_y = tl.load(tile_meta_ptr + tile * 5 + 1, mask=mask, other=0)
        out_x = tl.load(tile_meta_ptr + tile * 5 + 2, mask=mask, other=0)
        valid_h = tl.load(tile_meta_ptr + tile * 5 + 3, mask=mask, other=0)
        valid_w = tl.load(tile_meta_ptr + tile * 5 + 4, mask=mask, other=0)
        out_w = tl.load(shape_meta_ptr + batch * 4 + 3, mask=mask, other=0)
        out_base = tl.load(output_offsets_ptr + batch, mask=mask, other=0)

        valid = mask & (local_y < valid_h) & (local_x < valid_w)
        values = tl.load(
            tile_ptr
            + tile_local * tile_stride_n
            + channel * tile_stride_c
            + local_y * tile_stride_h
            + local_x * tile_stride_w,
            mask=valid,
            other=0.0,
        )
        if has_bias:
            values += tl.load(bias_ptr + channel, mask=valid, other=0.0)
        output_pos = out_base + (out_y + local_y) * out_w + out_x + local_x
        tl.store(output_ptr + output_pos * out_channels + channel, values, mask=valid)

    @triton.jit
    def _spatial_tile_scatter_output_channels_last_kernel(
        tile_ptr,
        bias_ptr,
        output_ptr,
        tile_meta_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        tile_offset,
        tile_count: tl.constexpr,
        tile_stride_n: tl.constexpr,
        tile_stride_c: tl.constexpr,
        tile_stride_h: tl.constexpr,
        tile_stride_w: tl.constexpr,
        out_channels: tl.constexpr,
        tile_h: tl.constexpr,
        tile_w: tl.constexpr,
        has_bias: tl.constexpr,
        block_hw: tl.constexpr,
        block_c: tl.constexpr,
    ):
        tile_local = tl.program_id(0)
        spatial = tl.program_id(1) * block_hw + tl.arange(0, block_hw)
        channel = tl.program_id(2) * block_c + tl.arange(0, block_c)
        tile = tile_offset + tile_local

        batch = tl.load(tile_meta_ptr + tile * 5)
        out_y = tl.load(tile_meta_ptr + tile * 5 + 1)
        out_x = tl.load(tile_meta_ptr + tile * 5 + 2)
        valid_h = tl.load(tile_meta_ptr + tile * 5 + 3)
        valid_w = tl.load(tile_meta_ptr + tile * 5 + 4)
        out_w = tl.load(shape_meta_ptr + batch * 4 + 3)
        out_base = tl.load(output_offsets_ptr + batch)

        local_y = spatial // tile_w
        local_x = spatial - local_y * tile_w
        spatial_mask = spatial < tile_h * tile_w
        channel_mask = channel < out_channels
        valid = spatial_mask & (local_y < valid_h) & (local_x < valid_w) & (tile_local < tile_count)
        values = tl.load(
            tile_ptr
            + tile_local * tile_stride_n
            + channel[None, :] * tile_stride_c
            + local_y[:, None] * tile_stride_h
            + local_x[:, None] * tile_stride_w,
            mask=valid[:, None] & channel_mask[None, :],
            other=0.0,
        )
        if has_bias:
            values += tl.load(bias_ptr + channel, mask=channel_mask, other=0.0)[None, :]
        output_pos = out_base + (out_y + local_y) * out_w + out_x + local_x
        tl.store(
            output_ptr + output_pos[:, None] * out_channels + channel[None, :],
            values,
            mask=valid[:, None] & channel_mask[None, :],
        )

    @triton.jit
    def _spatial_tile_scatter_output_local_kernel(
        tile_ptr,
        bias_ptr,
        output_ptr,
        tile_meta_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        tile_offset,
        tile_stride_n: tl.constexpr,
        tile_stride_c: tl.constexpr,
        tile_stride_h: tl.constexpr,
        tile_stride_w: tl.constexpr,
        out_channels: tl.constexpr,
        tile_h: tl.constexpr,
        tile_w: tl.constexpr,
        has_bias: tl.constexpr,
        block_size: tl.constexpr,
    ):
        tile_local = tl.program_id(0)
        offsets = tl.program_id(1) * block_size + tl.arange(0, block_size)
        mask = offsets < tile_h * tile_w * out_channels
        channel = offsets % out_channels
        pixel = offsets // out_channels
        local_y = pixel // tile_w
        local_x = pixel - local_y * tile_w
        tile = tile_offset + tile_local

        batch = tl.load(tile_meta_ptr + tile * 5)
        out_y = tl.load(tile_meta_ptr + tile * 5 + 1)
        out_x = tl.load(tile_meta_ptr + tile * 5 + 2)
        valid_h = tl.load(tile_meta_ptr + tile * 5 + 3)
        valid_w = tl.load(tile_meta_ptr + tile * 5 + 4)
        out_w = tl.load(shape_meta_ptr + batch * 4 + 3)
        out_base = tl.load(output_offsets_ptr + batch)

        valid = mask & (local_y < valid_h) & (local_x < valid_w)
        values = tl.load(
            tile_ptr
            + tile_local * tile_stride_n
            + channel * tile_stride_c
            + local_y * tile_stride_h
            + local_x * tile_stride_w,
            mask=valid,
            other=0.0,
        )
        if has_bias:
            values += tl.load(bias_ptr + channel, mask=valid, other=0.0)
        output_pos = out_base + (out_y + local_y) * out_w + out_x + local_x
        tl.store(output_ptr + output_pos * out_channels + channel, values, mask=valid)

    @triton.jit
    def _spatial_tile_scatter_input_kernel(
        tile_ptr,
        input_ptr,
        tile_meta_ptr,
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
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        padding_h: tl.constexpr,
        padding_w: tl.constexpr,
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

        batch = tl.load(tile_meta_ptr + tile * 5, mask=mask, other=0)
        out_y = tl.load(tile_meta_ptr + tile * 5 + 1, mask=mask, other=0)
        out_x = tl.load(tile_meta_ptr + tile * 5 + 2, mask=mask, other=0)
        in_h = tl.load(shape_meta_ptr + batch * 4, mask=mask, other=0)
        in_w = tl.load(shape_meta_ptr + batch * 4 + 1, mask=mask, other=0)
        in_base = tl.load(input_offsets_ptr + batch, mask=mask, other=0)

        input_y = out_y * stride_h - padding_h + tile_y
        input_x = out_x * stride_w - padding_w + tile_x
        valid = mask & (input_y >= 0) & (input_y < in_h) & (input_x >= 0) & (input_x < in_w)
        values = tl.load(
            tile_ptr
            + tile_local * tile_stride_n
            + channel * tile_stride_c
            + tile_y * tile_stride_h
            + tile_x * tile_stride_w,
            mask=valid,
            other=0.0,
        )
        input_pos = in_base + input_y * in_w + input_x
        tl.atomic_add(input_ptr + input_pos * in_channels + channel, values, sem="relaxed", mask=valid)

    @triton.jit
    def _spatial_tile_scatter_input_channels_last_kernel(
        tile_ptr,
        input_ptr,
        tile_meta_ptr,
        input_offsets_ptr,
        shape_meta_ptr,
        tile_offset,
        tile_count: tl.constexpr,
        tile_stride_n: tl.constexpr,
        tile_stride_c: tl.constexpr,
        tile_stride_h: tl.constexpr,
        tile_stride_w: tl.constexpr,
        in_channels: tl.constexpr,
        input_tile_h: tl.constexpr,
        input_tile_w: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        padding_h: tl.constexpr,
        padding_w: tl.constexpr,
        block_hw: tl.constexpr,
        block_c: tl.constexpr,
    ):
        tile_local = tl.program_id(0)
        spatial = tl.program_id(1) * block_hw + tl.arange(0, block_hw)
        channel = tl.program_id(2) * block_c + tl.arange(0, block_c)
        tile = tile_offset + tile_local

        batch = tl.load(tile_meta_ptr + tile * 5)
        out_y = tl.load(tile_meta_ptr + tile * 5 + 1)
        out_x = tl.load(tile_meta_ptr + tile * 5 + 2)
        in_h = tl.load(shape_meta_ptr + batch * 4)
        in_w = tl.load(shape_meta_ptr + batch * 4 + 1)
        in_base = tl.load(input_offsets_ptr + batch)

        tile_y = spatial // input_tile_w
        tile_x = spatial - tile_y * input_tile_w
        input_y = out_y * stride_h - padding_h + tile_y
        input_x = out_x * stride_w - padding_w + tile_x
        spatial_mask = spatial < input_tile_h * input_tile_w
        channel_mask = channel < in_channels
        valid = (
            spatial_mask
            & (input_y >= 0)
            & (input_y < in_h)
            & (input_x >= 0)
            & (input_x < in_w)
            & (tile_local < tile_count)
        )
        values = tl.load(
            tile_ptr
            + tile_local * tile_stride_n
            + channel[None, :] * tile_stride_c
            + tile_y[:, None] * tile_stride_h
            + tile_x[:, None] * tile_stride_w,
            mask=valid[:, None] & channel_mask[None, :],
            other=0.0,
        )
        input_pos = in_base + input_y * in_w + input_x
        tl.atomic_add(
            input_ptr + input_pos[:, None] * in_channels + channel[None, :],
            values,
            sem="relaxed",
            mask=valid[:, None] & channel_mask[None, :],
        )

    @triton.jit
    def _spatial_tile_scatter_input_local_kernel(
        tile_ptr,
        input_ptr,
        tile_meta_ptr,
        input_offsets_ptr,
        shape_meta_ptr,
        tile_offset,
        tile_stride_n: tl.constexpr,
        tile_stride_c: tl.constexpr,
        tile_stride_h: tl.constexpr,
        tile_stride_w: tl.constexpr,
        in_channels: tl.constexpr,
        input_tile_h: tl.constexpr,
        input_tile_w: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        padding_h: tl.constexpr,
        padding_w: tl.constexpr,
        block_size: tl.constexpr,
    ):
        tile_local = tl.program_id(0)
        offsets = tl.program_id(1) * block_size + tl.arange(0, block_size)
        mask = offsets < input_tile_h * input_tile_w * in_channels
        channel = offsets % in_channels
        pixel = offsets // in_channels
        tile_y = pixel // input_tile_w
        tile_x = pixel - tile_y * input_tile_w
        tile = tile_offset + tile_local

        batch = tl.load(tile_meta_ptr + tile * 5)
        out_y = tl.load(tile_meta_ptr + tile * 5 + 1)
        out_x = tl.load(tile_meta_ptr + tile * 5 + 2)
        in_h = tl.load(shape_meta_ptr + batch * 4)
        in_w = tl.load(shape_meta_ptr + batch * 4 + 1)
        in_base = tl.load(input_offsets_ptr + batch)

        input_y = out_y * stride_h - padding_h + tile_y
        input_x = out_x * stride_w - padding_w + tile_x
        valid = mask & (input_y >= 0) & (input_y < in_h) & (input_x >= 0) & (input_x < in_w)
        values = tl.load(
            tile_ptr
            + tile_local * tile_stride_n
            + channel * tile_stride_c
            + tile_y * tile_stride_h
            + tile_x * tile_stride_w,
            mask=valid,
            other=0.0,
        )
        input_pos = in_base + input_y * in_w + input_x
        tl.atomic_add(input_ptr + input_pos * in_channels + channel, values, sem="relaxed", mask=valid)

else:
    _spatial_tile_pack_input_kernel = None
    _spatial_tile_pack_output_kernel = None
    _spatial_tile_scatter_output_kernel = None
    _spatial_tile_scatter_input_kernel = None
    _spatial_tile_pack_input_channels_last_kernel = None
    _spatial_tile_pack_output_channels_last_kernel = None
    _spatial_tile_scatter_output_channels_last_kernel = None
    _spatial_tile_scatter_input_channels_last_kernel = None
    _spatial_tile_pack_input_local_kernel = None
    _spatial_tile_pack_output_local_kernel = None
    _spatial_tile_scatter_output_local_kernel = None
    _spatial_tile_scatter_input_local_kernel = None


def _spatial_tile2d_channels_last_blocks(channels: int) -> tuple[int, int]:
    block_c = min(max(16, 1 << (int(channels) - 1).bit_length()), 64)
    return 128, block_c


def _spatial_tile2d_use_local_copy_kernel() -> bool:
    return False


def _spatial_tile2d_copy_geometry(
    batch: int,
    out_y: int,
    out_x: int,
    input_shapes: tuple[tuple[int, ...], ...],
    input_tile_shape: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> tuple[int, int, int, int, int, int, int, int]:
    _, in_h, in_w = input_shapes[batch]
    input_tile_h, input_tile_w = input_tile_shape
    raw_y0 = out_y * stride[0] - padding[0]
    raw_x0 = out_x * stride[1] - padding[1]
    raw_y1 = raw_y0 + input_tile_h
    raw_x1 = raw_x0 + input_tile_w
    src_y0 = max(raw_y0, 0)
    src_x0 = max(raw_x0, 0)
    src_y1 = min(raw_y1, in_h)
    src_x1 = min(raw_x1, in_w)
    dst_y0 = src_y0 - raw_y0
    dst_x0 = src_x0 - raw_x0
    copy_h = max(src_y1 - src_y0, 0)
    copy_w = max(src_x1 - src_x0, 0)
    return src_y0, src_x0, dst_y0, dst_x0, copy_h, copy_w, in_h, in_w


def _make_spatial_tile2d_input(
    input_values: Tensor,
    input_shapes: tuple[tuple[int, ...], ...],
    input_starts: tuple[int, ...],
    chunk: list[tuple[int, int, int, int, int]] | tuple[tuple[int, int, int, int, int], ...],
    *,
    in_channels: int,
    input_tile_shape: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    channels_last: bool,
    tile_offset: int = 0,
    input_offsets_device: Tensor | None = None,
    shape_meta: Tensor | None = None,
    tile_meta: Tensor | None = None,
) -> Tensor:
    input_tile_h, input_tile_w = input_tile_shape
    tile_input = _new_spatial_tile2d_tensor(
        input_values,
        (len(chunk), in_channels, input_tile_h, input_tile_w),
        channels_last=channels_last,
    )

    if triton is not None and _has_cuda_tensors(input_offsets_device, shape_meta, tile_meta):
        if (
            channels_last
            and _spatial_tile2d_use_local_copy_kernel()
            and tile_input.stride(1) == 1
            and _spatial_tile_pack_input_local_kernel is not None
        ):
            block_size = _spatial_tile_triton_block_size()
            grid: tuple[int, ...] = (len(chunk), triton.cdiv(input_tile_h * input_tile_w * in_channels, block_size))
            _spatial_tile_pack_input_local_kernel[grid](
                input_values,
                tile_input,
                tile_meta,
                input_offsets_device,
                shape_meta,
                int(tile_offset),
                tile_input.stride(0),
                tile_input.stride(1),
                tile_input.stride(2),
                tile_input.stride(3),
                in_channels,
                input_tile_h,
                input_tile_w,
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                block_size,
            )
            return tile_input

        if channels_last and tile_input.stride(1) == 1 and _spatial_tile_pack_input_channels_last_kernel is not None:
            block_hw, block_c = _spatial_tile2d_channels_last_blocks(in_channels)
            grid = (
                len(chunk),
                triton.cdiv(input_tile_h * input_tile_w, block_hw),
                triton.cdiv(in_channels, block_c),
            )
            _spatial_tile_pack_input_channels_last_kernel[grid](
                input_values,
                tile_input,
                tile_meta,
                input_offsets_device,
                shape_meta,
                int(tile_offset),
                len(chunk),
                tile_input.stride(0),
                tile_input.stride(1),
                tile_input.stride(2),
                tile_input.stride(3),
                in_channels,
                input_tile_h,
                input_tile_w,
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                block_hw,
                block_c,
            )
            return tile_input

        block_size = _spatial_tile_triton_block_size()
        total = len(chunk) * input_tile_h * input_tile_w * in_channels
        grid = (triton.cdiv(total, block_size),)
        _spatial_tile_pack_input_kernel[grid](
            input_values,
            tile_input,
            tile_meta,
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
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            block_size,
        )
        return tile_input

    tile_input.zero_()

    for tile_index, (batch, out_y, out_x, _, _) in enumerate(chunk):
        src_y0, src_x0, dst_y0, dst_x0, copy_h, copy_w, in_h, in_w = _spatial_tile2d_copy_geometry(
            batch,
            out_y,
            out_x,
            input_shapes,
            input_tile_shape,
            stride,
            padding,
        )
        if copy_h == 0 or copy_w == 0:
            continue
        input_start = input_starts[batch]
        element = input_values[input_start : input_start + in_h * in_w].reshape(in_h, in_w, in_channels)
        tile_input[
            tile_index,
            :,
            dst_y0 : dst_y0 + copy_h,
            dst_x0 : dst_x0 + copy_w,
        ] = element[
            src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w
        ].permute(2, 0, 1)

    return tile_input


def _make_spatial_tile2d_grad_output(
    grad_output_values: Tensor,
    output_shapes: tuple[tuple[int, ...], ...],
    output_starts: tuple[int, ...],
    chunk: list[tuple[int, int, int, int, int]] | tuple[tuple[int, int, int, int, int], ...],
    *,
    out_channels: int,
    tile_shape: tuple[int, int],
    channels_last: bool,
    tile_offset: int = 0,
    output_offsets_device: Tensor | None = None,
    shape_meta: Tensor | None = None,
    tile_meta: Tensor | None = None,
) -> Tensor:
    tile_h, tile_w = tile_shape
    tile_grad_output = _new_spatial_tile2d_tensor(
        grad_output_values,
        (len(chunk), out_channels, tile_h, tile_w),
        channels_last=channels_last,
    )

    if triton is not None and _has_cuda_tensors(output_offsets_device, shape_meta, tile_meta):
        if (
            channels_last
            and _spatial_tile2d_use_local_copy_kernel()
            and tile_grad_output.stride(1) == 1
            and _spatial_tile_pack_output_local_kernel is not None
        ):
            block_size = _spatial_tile_triton_block_size()
            grid: tuple[int, ...] = (len(chunk), triton.cdiv(tile_h * tile_w * out_channels, block_size))
            _spatial_tile_pack_output_local_kernel[grid](
                grad_output_values,
                tile_grad_output,
                tile_meta,
                output_offsets_device,
                shape_meta,
                int(tile_offset),
                grad_output_values.stride(0),
                grad_output_values.stride(1),
                tile_grad_output.stride(0),
                tile_grad_output.stride(1),
                tile_grad_output.stride(2),
                tile_grad_output.stride(3),
                out_channels,
                tile_h,
                tile_w,
                block_size,
            )
            return tile_grad_output

        if (
            channels_last
            and tile_grad_output.stride(1) == 1
            and _spatial_tile_pack_output_channels_last_kernel is not None
        ):
            block_hw, block_c = _spatial_tile2d_channels_last_blocks(out_channels)
            grid = (
                len(chunk),
                triton.cdiv(tile_h * tile_w, block_hw),
                triton.cdiv(out_channels, block_c),
            )
            _spatial_tile_pack_output_channels_last_kernel[grid](
                grad_output_values,
                tile_grad_output,
                tile_meta,
                output_offsets_device,
                shape_meta,
                int(tile_offset),
                len(chunk),
                grad_output_values.stride(0),
                grad_output_values.stride(1),
                tile_grad_output.stride(0),
                tile_grad_output.stride(1),
                tile_grad_output.stride(2),
                tile_grad_output.stride(3),
                out_channels,
                tile_h,
                tile_w,
                block_hw,
                block_c,
            )
            return tile_grad_output

        block_size = _spatial_tile_triton_block_size()
        total = len(chunk) * tile_h * tile_w * out_channels
        grid = (triton.cdiv(total, block_size),)
        _spatial_tile_pack_output_kernel[grid](
            grad_output_values,
            tile_grad_output,
            tile_meta,
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
            tile_h,
            tile_w,
            block_size,
        )
        return tile_grad_output

    tile_grad_output.zero_()

    for tile_index, (batch, out_y, out_x, valid_h, valid_w) in enumerate(chunk):
        _, out_h, out_w = output_shapes[batch]
        output_start = output_starts[batch]
        element = grad_output_values[output_start : output_start + out_h * out_w].reshape(
            out_h,
            out_w,
            out_channels,
        )
        tile_grad_output[tile_index, :, :valid_h, :valid_w] = element[
            out_y : out_y + valid_h,
            out_x : out_x + valid_w,
        ].permute(2, 0, 1)

    return tile_grad_output


def _scatter_spatial_tile2d_grad_input(
    grad_input_values: Tensor,
    tile_grad_input: Tensor,
    input_shapes: tuple[tuple[int, ...], ...],
    input_starts: tuple[int, ...],
    chunk: list[tuple[int, int, int, int, int]] | tuple[tuple[int, int, int, int, int], ...],
    *,
    in_channels: int,
    input_tile_shape: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    tile_offset: int = 0,
    input_offsets_device: Tensor | None = None,
    shape_meta: Tensor | None = None,
    tile_meta: Tensor | None = None,
    channels_last: bool = False,
) -> None:
    if triton is not None and _has_cuda_tensors(input_offsets_device, shape_meta, tile_meta):
        input_tile_h, input_tile_w = input_tile_shape
        if (
            channels_last
            and _spatial_tile2d_use_local_copy_kernel()
            and tile_grad_input.stride(1) == 1
            and _spatial_tile_scatter_input_local_kernel is not None
        ):
            block_size = _spatial_tile_triton_block_size()
            grid: tuple[int, ...] = (len(chunk), triton.cdiv(input_tile_h * input_tile_w * in_channels, block_size))
            _spatial_tile_scatter_input_local_kernel[grid](
                tile_grad_input,
                grad_input_values,
                tile_meta,
                input_offsets_device,
                shape_meta,
                int(tile_offset),
                tile_grad_input.stride(0),
                tile_grad_input.stride(1),
                tile_grad_input.stride(2),
                tile_grad_input.stride(3),
                in_channels,
                input_tile_h,
                input_tile_w,
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                block_size,
            )
            return

        if (
            channels_last
            and tile_grad_input.stride(1) == 1
            and _spatial_tile_scatter_input_channels_last_kernel is not None
        ):
            block_hw, block_c = _spatial_tile2d_channels_last_blocks(in_channels)
            grid = (
                len(chunk),
                triton.cdiv(input_tile_h * input_tile_w, block_hw),
                triton.cdiv(in_channels, block_c),
            )
            _spatial_tile_scatter_input_channels_last_kernel[grid](
                tile_grad_input,
                grad_input_values,
                tile_meta,
                input_offsets_device,
                shape_meta,
                int(tile_offset),
                len(chunk),
                tile_grad_input.stride(0),
                tile_grad_input.stride(1),
                tile_grad_input.stride(2),
                tile_grad_input.stride(3),
                in_channels,
                input_tile_h,
                input_tile_w,
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                block_hw,
                block_c,
            )
            return

        block_size = _spatial_tile_triton_block_size()
        total = len(chunk) * input_tile_h * input_tile_w * in_channels
        grid = (triton.cdiv(total, block_size),)
        _spatial_tile_scatter_input_kernel[grid](
            tile_grad_input,
            grad_input_values,
            tile_meta,
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
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            block_size,
        )
        return

    for tile_index, (batch, out_y, out_x, _, _) in enumerate(chunk):
        src_y0, src_x0, dst_y0, dst_x0, copy_h, copy_w, in_h, in_w = _spatial_tile2d_copy_geometry(
            batch,
            out_y,
            out_x,
            input_shapes,
            input_tile_shape,
            stride,
            padding,
        )
        if copy_h == 0 or copy_w == 0:
            continue
        input_start = input_starts[batch]
        element_grad = grad_input_values[input_start : input_start + in_h * in_w].reshape(in_h, in_w, in_channels)
        element_grad[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w] += tile_grad_input[
            tile_index,
            :,
            dst_y0 : dst_y0 + copy_h,
            dst_x0 : dst_x0 + copy_w,
        ].permute(1, 2, 0)


def _scatter_spatial_tile2d_output(
    output_values: Tensor,
    tile_output: Tensor,
    bias: Tensor | None,
    output_shapes: tuple[tuple[int, ...], ...],
    output_starts: tuple[int, ...],
    chunk: list[tuple[int, int, int, int, int]] | tuple[tuple[int, int, int, int, int], ...],
    *,
    out_channels: int,
    tile_offset: int = 0,
    output_offsets_device: Tensor | None = None,
    shape_meta: Tensor | None = None,
    tile_meta: Tensor | None = None,
    channels_last: bool = False,
) -> None:
    if triton is not None and _has_cuda_tensors(output_offsets_device, shape_meta, tile_meta):
        tile_h = int(tile_output.shape[2])
        tile_w = int(tile_output.shape[3])
        if (
            _spatial_tile2d_use_local_copy_kernel()
            and tile_output.stride(1) == 1
            and _spatial_tile_scatter_output_local_kernel is not None
        ):
            block_size = _spatial_tile_triton_block_size()
            grid: tuple[int, ...] = (len(chunk), triton.cdiv(tile_h * tile_w * out_channels, block_size))
            _spatial_tile_scatter_output_local_kernel[grid](
                tile_output,
                bias if bias is not None else output_values,
                output_values,
                tile_meta,
                output_offsets_device,
                shape_meta,
                int(tile_offset),
                tile_output.stride(0),
                tile_output.stride(1),
                tile_output.stride(2),
                tile_output.stride(3),
                out_channels,
                tile_h,
                tile_w,
                bias is not None,
                block_size,
            )
            return

        if (
            channels_last
            and tile_output.stride(1) == 1
            and _spatial_tile_scatter_output_channels_last_kernel is not None
        ):
            block_hw, block_c = _spatial_tile2d_channels_last_blocks(out_channels)
            grid = (
                len(chunk),
                triton.cdiv(tile_h * tile_w, block_hw),
                triton.cdiv(out_channels, block_c),
            )
            _spatial_tile_scatter_output_channels_last_kernel[grid](
                tile_output,
                bias if bias is not None else output_values,
                output_values,
                tile_meta,
                output_offsets_device,
                shape_meta,
                int(tile_offset),
                len(chunk),
                tile_output.stride(0),
                tile_output.stride(1),
                tile_output.stride(2),
                tile_output.stride(3),
                out_channels,
                tile_h,
                tile_w,
                bias is not None,
                block_hw,
                block_c,
            )
            return

        block_size = _spatial_tile_triton_block_size()
        total = len(chunk) * tile_h * tile_w * out_channels
        grid = (triton.cdiv(total, block_size),)
        _spatial_tile_scatter_output_kernel[grid](
            tile_output,
            bias if bias is not None else output_values,
            output_values,
            tile_meta,
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
            bias is not None,
            block_size,
        )
        return

    for tile_index, (batch, out_y, out_x, valid_h, valid_w) in enumerate(chunk):
        _, out_h, out_w = output_shapes[batch]
        output_start = output_starts[batch]
        element_output = output_values[output_start : output_start + out_h * out_w].reshape(
            out_h,
            out_w,
            out_channels,
        )
        element_output[out_y : out_y + valid_h, out_x : out_x + valid_w] = tile_output[
            tile_index,
            :,
            :valid_h,
            :valid_w,
        ].permute(1, 2, 0)
        if bias is not None:
            element_output[out_y : out_y + valid_h, out_x : out_x + valid_w] += bias


def _spatial_tile_conv2d_forward_values(
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
    tile_shape: tuple[int, int],
    input_tile_shape: tuple[int, int],
    max_tiles_per_batch: int,
    channels_last: bool,
    input_offsets_device: Tensor | None = None,
    output_offsets_device: Tensor | None = None,
    shape_meta: Tensor | None = None,
    tile_meta: Tensor | None = None,
) -> Tensor:
    out_channels = int(weight.shape[0])
    in_channels = int(weight.shape[1])
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
            stride=stride,
            padding=padding,
            channels_last=channels_last,
            tile_offset=chunk_start,
            input_offsets_device=input_offsets_device,
            shape_meta=shape_meta,
            tile_meta=tile_meta,
        )
        tile_output = F.conv2d(
            tile_input,
            weight,
            None,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=1,
        )
        _scatter_spatial_tile2d_output(
            output_values,
            tile_output,
            bias,
            output_shapes,
            output_starts,
            chunk,
            out_channels=out_channels,
            tile_offset=chunk_start,
            output_offsets_device=output_offsets_device,
            shape_meta=shape_meta,
            tile_meta=tile_meta,
            channels_last=channels_last,
        )

    return output_values


def _spatial_tile_conv2d_input_grad(
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
        False,
        [0],
        1,
        (True, False, False),
    )[0]


def _spatial_tile_conv2d_weight_grad(
    input: Tensor,
    weight: Tensor,
    grad_output: Tensor,
    *,
    stride: tuple[int, int],
    dilation: tuple[int, int],
) -> Tensor:
    weight_meta = torch.empty_strided(
        tuple(int(size) for size in weight.shape),
        tuple(int(stride) for stride in weight.stride()),
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
        False,
        [0],
        1,
        (False, True, False),
    )[1]


class _SpatialTileConv2dCudnnFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_values: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        input_offsets_device: Tensor,
        output_offsets_device: Tensor,
        shape_meta: Tensor,
        tile_meta: Tensor,
        output_size: int,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shapes: tuple[tuple[int, ...], ...],
        input_starts: tuple[int, ...],
        output_starts: tuple[int, ...],
        tiles: tuple[tuple[int, int, int, int, int], ...],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        tile_shape: tuple[int, int],
        input_tile_shape: tuple[int, int],
        max_tiles_per_batch: int,
        weight_max_tiles_per_batch: int,
        channels_last: bool,
    ) -> Tensor:
        weight = weight.contiguous()
        output_values = _spatial_tile_conv2d_forward_values(
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
            tile_shape=tile_shape,
            input_tile_shape=input_tile_shape,
            max_tiles_per_batch=max_tiles_per_batch,
            channels_last=channels_last,
            input_offsets_device=input_offsets_device,
            output_offsets_device=output_offsets_device,
            shape_meta=shape_meta,
            tile_meta=tile_meta,
        )
        ctx.save_for_backward(
            input_values,
            weight,
            input_offsets_device,
            output_offsets_device,
            shape_meta,
            tile_meta,
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
        ctx.tile_shape = tile_shape
        ctx.input_tile_shape = input_tile_shape
        ctx.max_tiles_per_batch = max_tiles_per_batch
        ctx.weight_max_tiles_per_batch = weight_max_tiles_per_batch
        ctx.channels_last = channels_last
        return output_values

    @staticmethod
    def backward(ctx, grad_output_values: Tensor):
        input_values, weight, input_offsets_device, output_offsets_device, shape_meta, tile_meta = ctx.saved_tensors[:6]
        input_shapes = ctx.input_shapes
        output_shapes = ctx.output_shapes
        input_starts = ctx.input_starts
        output_starts = ctx.output_starts
        tiles = ctx.tiles
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        tile_shape = ctx.tile_shape
        input_tile_shape = ctx.input_tile_shape
        max_tiles_per_batch = ctx.max_tiles_per_batch
        weight_max_tiles_per_batch = ctx.weight_max_tiles_per_batch
        channels_last = ctx.channels_last
        in_channels = int(weight.shape[1])
        out_channels = int(weight.shape[0])

        grad_input = torch.zeros_like(input_values) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        grad_bias = grad_output_values.sum(dim=0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        chunk_size = _spatial_tile_max_batch(max_tiles_per_batch, len(tiles))

        if grad_input is not None and (grad_weight is None or weight_max_tiles_per_batch != chunk_size):
            for chunk_start in range(0, len(tiles), chunk_size):
                chunk = tiles[chunk_start : chunk_start + chunk_size]
                tile_grad_output = _make_spatial_tile2d_grad_output(
                    grad_output_values,
                    output_shapes,
                    output_starts,
                    chunk,
                    out_channels=out_channels,
                    tile_shape=tile_shape,
                    channels_last=channels_last,
                    tile_offset=chunk_start,
                    output_offsets_device=output_offsets_device,
                    shape_meta=shape_meta,
                    tile_meta=tile_meta,
                )
                tile_grad_input = _spatial_tile_conv2d_input_grad(
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
                    stride=stride,
                    padding=padding,
                    tile_offset=chunk_start,
                    input_offsets_device=input_offsets_device,
                    shape_meta=shape_meta,
                    tile_meta=tile_meta,
                    channels_last=channels_last,
                )

        if grad_weight is not None and (grad_input is None or weight_max_tiles_per_batch != chunk_size):
            weight_chunk_size = _spatial_tile_max_batch(weight_max_tiles_per_batch, len(tiles))
            for chunk_start in range(0, len(tiles), weight_chunk_size):
                chunk = tiles[chunk_start : chunk_start + weight_chunk_size]
                tile_grad_output = _make_spatial_tile2d_grad_output(
                    grad_output_values,
                    output_shapes,
                    output_starts,
                    chunk,
                    out_channels=out_channels,
                    tile_shape=tile_shape,
                    channels_last=channels_last,
                    tile_offset=chunk_start,
                    output_offsets_device=output_offsets_device,
                    shape_meta=shape_meta,
                    tile_meta=tile_meta,
                )
                tile_input = _make_spatial_tile2d_input(
                    input_values,
                    input_shapes,
                    input_starts,
                    chunk,
                    in_channels=in_channels,
                    input_tile_shape=input_tile_shape,
                    stride=stride,
                    padding=padding,
                    channels_last=channels_last,
                    tile_offset=chunk_start,
                    input_offsets_device=input_offsets_device,
                    shape_meta=shape_meta,
                    tile_meta=tile_meta,
                )
                grad_weight += _spatial_tile_conv2d_weight_grad(
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
                    output_shapes,
                    output_starts,
                    chunk,
                    out_channels=out_channels,
                    tile_shape=tile_shape,
                    channels_last=channels_last,
                    tile_offset=chunk_start,
                    output_offsets_device=output_offsets_device,
                    shape_meta=shape_meta,
                    tile_meta=tile_meta,
                )
                tile_grad_input = _spatial_tile_conv2d_input_grad(
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
                    stride=stride,
                    padding=padding,
                    tile_offset=chunk_start,
                    input_offsets_device=input_offsets_device,
                    shape_meta=shape_meta,
                    tile_meta=tile_meta,
                    channels_last=channels_last,
                )
                tile_input = _make_spatial_tile2d_input(
                    input_values,
                    input_shapes,
                    input_starts,
                    chunk,
                    in_channels=in_channels,
                    input_tile_shape=input_tile_shape,
                    stride=stride,
                    padding=padding,
                    channels_last=channels_last,
                    tile_offset=chunk_start,
                    input_offsets_device=input_offsets_device,
                    shape_meta=shape_meta,
                    tile_meta=tile_meta,
                )
                grad_weight += _spatial_tile_conv2d_weight_grad(
                    tile_input,
                    weight,
                    tile_grad_output,
                    stride=stride,
                    dilation=dilation,
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
            None,
        )


def _spatial_tile_conv2d(
    input: NestedTensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    *,
    tile_size: int | tuple[int, int] | str = "auto",
    max_tiles_per_batch: int | str | None = "auto",
    weight_max_tiles_per_batch: int | str | None = "same",
    channels_last: bool = True,
) -> NestedTensor | None:
    r"""
    Run ragged ``conv2d`` by batching fixed-size spatial tiles through PyTorch/cuDNN.

    Each tile covers a fixed output-space window plus the input halo required by the
    convolution. cuDNN still sees a regular dense batch, while padding waste is bounded
    by tile size instead of the largest image in the NestedTensor batch.
    """
    if triton is None or not _can_use_spatial_tile_convolution(input, weight, groups, rank=2, input_channel_dim=1):
        return None
    if not isinstance(stride, int) or not isinstance(padding, int) or not isinstance(dilation, int):
        return None
    stride_pair = (int(stride), int(stride))
    padding_pair = (int(padding), int(padding))
    dilation_pair = (int(dilation), int(dilation))
    if any(value <= 0 for value in (*stride_pair, *dilation_pair)):
        return None
    if padding_pair[0] < 0 or padding_pair[1] < 0:
        return None

    out_channels = int(weight.shape[0])
    in_channels = int(weight.shape[1])
    kernel_h = int(weight.shape[2])
    kernel_w = int(weight.shape[3])
    if not _valid_conv_bias(bias, out_channels=out_channels, device=input._values.device, dtype=input._values.dtype):
        return None
    output_meta = _conv2d_output_meta(
        input,
        out_channels,
        kernel_h,
        kernel_w,
        stride_pair,
        padding_pair,
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

    input_shapes = _resolve_element_shapes(input)
    resolved_tile_config = _resolve_spatial_tile2d_config(input_shapes, tile_size, max_tiles_per_batch)
    if resolved_tile_config is None:
        return None
    tile_shape, resolved_max_tiles = resolved_tile_config
    input_starts = tuple(int(offset) for offset in input._offsets[:-1].tolist())
    output_starts = tuple(int(offset) for offset in output_offsets[:-1].tolist())
    device = input._values.device
    shape_meta = torch.tensor(
        [
            (int(input_shape[1]), int(input_shape[2]), int(output_shape[1]), int(output_shape[2]))
            for input_shape, output_shape in zip(input_shapes, output_shapes)
        ],
        device=device,
        dtype=torch.int32,
    )
    input_offsets_device = input._offsets.to(device=device, non_blocking=True)
    output_offsets_device = output_offsets.to(device=device, non_blocking=True)
    tile_h, tile_w = tile_shape
    input_tile_h = (tile_h - 1) * stride_pair[0] + dilation_pair[0] * (kernel_h - 1) + 1
    input_tile_w = (tile_w - 1) * stride_pair[1] + dilation_pair[1] * (kernel_w - 1) + 1
    if input_tile_h <= 0 or input_tile_w <= 0:
        return None
    input_tile_shape = (input_tile_h, input_tile_w)
    resolved_max_tiles = _cap_spatial_tile2d_batch_for_channels(
        resolved_max_tiles,
        in_channels=in_channels,
        out_channels=out_channels,
        input_tile_shape=input_tile_shape,
        output_tile_shape=tile_shape,
    )

    tiles = tuple(_conv2d_spatial_tiles(output_shapes, tile_shape))
    tile_meta = torch.tensor(tiles, device=device, dtype=torch.int32)
    chunk_size = _spatial_tile_max_batch(resolved_max_tiles, len(tiles))
    resolved_weight_max_tiles = _resolve_spatial_weight_tile2d_batch(
        tuple(input_shapes),
        tile_shape,
        weight_max_tiles_per_batch,
        chunk_size,
    )
    if resolved_weight_max_tiles is None:
        return None
    use_channels_last = bool(channels_last and input._values.is_cuda)
    requires_grad = torch.is_grad_enabled() and (
        input._values.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    )

    if requires_grad:
        output_values = _SpatialTileConv2dCudnnFunction.apply(
            input._values,
            weight,
            bias,
            input_offsets_device,
            output_offsets_device,
            shape_meta,
            tile_meta,
            total_out,
            tuple(input_shapes),
            output_shapes,
            input_starts,
            output_starts,
            tiles,
            stride_pair,
            padding_pair,
            dilation_pair,
            tile_shape,
            input_tile_shape,
            chunk_size,
            resolved_weight_max_tiles,
            use_channels_last,
        )
    else:
        output_values = _spatial_tile_conv2d_forward_values(
            input._values,
            weight,
            bias,
            output_size=total_out,
            input_shapes=tuple(input_shapes),
            output_shapes=output_shapes,
            input_starts=input_starts,
            output_starts=output_starts,
            tiles=tiles,
            stride=stride_pair,
            padding=padding_pair,
            dilation=dilation_pair,
            tile_shape=tile_shape,
            input_tile_shape=input_tile_shape,
            max_tiles_per_batch=chunk_size,
            channels_last=use_channels_last,
            input_offsets_device=input_offsets_device,
            output_offsets_device=output_offsets_device,
            shape_meta=shape_meta,
            tile_meta=tile_meta,
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
