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
r"""Fused linear spatial copies using the existing convolution copy kernels."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils._python_dispatch import _get_current_dispatch_mode

from ..ops import _is_compiling

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None

if triton is not None:

    @triton.jit
    def _coordinate_is_covered(
        coordinate, padding, stride: tl.constexpr, dilation: tl.constexpr, kernel: tl.constexpr, output_size
    ):
        covered = tl.full(coordinate.shape, False, tl.int1)
        for index in tl.static_range(kernel):
            shifted = coordinate + padding - index * dilation
            covered |= (shifted >= 0) & (shifted % stride == 0) & (shifted // stride < output_size)
        return covered

    @triton.jit(
        do_not_specialize=[
            "tile_offset",
            "tile_stride_n",
            "tile_stride_c",
            "tile_stride_h",
            "tile_stride_w",
            "input_tile_h",
            "input_tile_w",
        ]
    )
    def _spatial_tile_pack_input_local_kernel(
        input_ptr,
        tile_ptr,
        tile_meta_ptr,
        input_offsets_ptr,
        shape_meta_ptr,
        tile_offset,
        tile_stride_n,
        tile_stride_c,
        tile_stride_h,
        tile_stride_w,
        in_channels: tl.constexpr,
        input_tile_h,
        input_tile_w,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        padding_h: tl.constexpr,
        padding_w: tl.constexpr,
        kernel_h: tl.constexpr,
        kernel_w: tl.constexpr,
        dilation_h: tl.constexpr,
        dilation_w: tl.constexpr,
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
        if stride_h != 1 or stride_w != 1 or dilation_h != 1 or dilation_w != 1:
            out_h = tl.load(shape_meta_ptr + batch * 4 + 2)
            out_w = tl.load(shape_meta_ptr + batch * 4 + 3)
            valid &= _coordinate_is_covered(input_y, padding_h, stride_h, dilation_h, kernel_h, out_h)
            valid &= _coordinate_is_covered(input_x, padding_w, stride_w, dilation_w, kernel_w, out_w)
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

    @triton.jit(
        do_not_specialize=[
            "tile_offset",
            "output_stride_n",
            "output_stride_c",
            "tile_stride_n",
            "tile_stride_c",
            "tile_stride_h",
            "tile_stride_w",
            "tile_h",
            "tile_w",
        ]
    )
    def _spatial_tile_pack_output_local_kernel(
        output_ptr,
        tile_ptr,
        tile_meta_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        tile_offset,
        output_stride_n,
        output_stride_c,
        tile_stride_n,
        tile_stride_c,
        tile_stride_h,
        tile_stride_w,
        out_channels: tl.constexpr,
        tile_h,
        tile_w,
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

    @triton.jit(
        do_not_specialize=[
            "tile_offset",
            "tile_stride_n",
            "tile_stride_c",
            "tile_stride_h",
            "tile_stride_w",
            "tile_h",
            "tile_w",
        ]
    )
    def _spatial_tile_scatter_output_local_kernel(
        tile_ptr,
        bias_ptr,
        output_ptr,
        tile_meta_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        tile_offset,
        tile_stride_n,
        tile_stride_c,
        tile_stride_h,
        tile_stride_w,
        out_channels: tl.constexpr,
        tile_h,
        tile_w,
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

    @triton.jit(
        do_not_specialize=[
            "tile_offset",
            "tile_stride_n",
            "tile_stride_c",
            "tile_stride_h",
            "tile_stride_w",
            "input_tile_h",
            "input_tile_w",
        ]
    )
    def _spatial_tile_scatter_input_local_kernel(
        tile_ptr,
        input_ptr,
        tile_meta_ptr,
        input_offsets_ptr,
        shape_meta_ptr,
        tile_offset,
        tile_stride_n,
        tile_stride_c,
        tile_stride_h,
        tile_stride_w,
        in_channels: tl.constexpr,
        input_tile_h,
        input_tile_w,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        padding_h: tl.constexpr,
        padding_w: tl.constexpr,
        kernel_h: tl.constexpr,
        kernel_w: tl.constexpr,
        dilation_h: tl.constexpr,
        dilation_w: tl.constexpr,
        accumulate: tl.constexpr,
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
        if stride_h != 1 or stride_w != 1 or dilation_h != 1 or dilation_w != 1:
            out_h = tl.load(shape_meta_ptr + batch * 4 + 2)
            out_w = tl.load(shape_meta_ptr + batch * 4 + 3)
            valid &= _coordinate_is_covered(input_y, padding_h, stride_h, dilation_h, kernel_h, out_h)
            valid &= _coordinate_is_covered(input_x, padding_w, stride_w, dilation_w, kernel_w, out_w)
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
        target = input_ptr + input_pos * in_channels + channel
        if accumulate:
            tl.atomic_add(target, values, sem="relaxed", mask=valid)
        else:
            tl.store(target, values, mask=valid)


def _copy_context_is_safe(values):
    return (
        type(values) is Tensor
        and values.is_cuda
        and not _is_compiling()
        and _get_current_dispatch_mode() is None
        and not torch._C._are_functorch_transforms_active()
        and not torch.jit.is_tracing()
        and not torch._is_functional_tensor(values)
        and torch.autograd.forward_ad._current_level < 0
        and not torch.are_deterministic_algorithms_enabled()
    )


def available(values, stride, dilation, count, input_h, input_w, output_h, output_w, out_channels):
    if (
        triton is None
        or not _copy_context_is_safe(values)
        or values.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        or not values.is_contiguous()
        or values.is_conj()
        or values.is_neg()
    ):
        return False
    # These existing kernels use 32-bit tile-local linear indices.
    return (
        max(count * 5, count * input_h * input_w * values.shape[1], count * output_h * output_w * out_channels)
        <= 2147483647
    )


def make_metadata(
    spatial, output_spatial, offsets, output_offsets, tile_grid, tile_offsets, count, tile_h, tile_w, device
):
    positions = torch.arange(count, device=spatial.device)
    owner = torch.bucketize(positions, tile_offsets[1:], right=True)
    local = positions - tile_offsets.index_select(0, owner)
    columns = tile_grid[:, 1].index_select(0, owner)
    y = torch.div(local, columns, rounding_mode="floor") * tile_h
    x = local.remainder(columns) * tile_w
    sizes = output_spatial.index_select(0, owner)
    height = (sizes[:, 0] - y).clamp(min=0, max=tile_h)
    width = (sizes[:, 1] - x).clamp(min=0, max=tile_w)
    tile_meta = torch.stack((owner, y, x, height, width), dim=1).to(device=device, dtype=torch.long)
    shape_meta = torch.cat((spatial, output_spatial), dim=1).to(device=device, dtype=torch.long)
    return (
        offsets.contiguous().to(device=device, dtype=torch.long),
        output_offsets.contiguous().to(device=device, dtype=torch.long),
        shape_meta,
        tile_meta,
    )


def _rows(offsets, shapes, tiles, height, width, padding, output=False, stride=(1, 1), kernel=(1, 1), dilation=(1, 1)):
    device = offsets.device
    owner = tiles[:, 0]
    y = tiles[:, 1, None, None] * stride[0] + torch.arange(height, device=device)[None, :, None] - padding[0]
    x = tiles[:, 2, None, None] * stride[1] + torch.arange(width, device=device)[None, None, :] - padding[1]
    dims = shapes[:, 2:] if output else shapes[:, :2]
    selected = dims.index_select(0, owner)
    valid = (y >= 0) & (x >= 0) & (y < selected[:, 0, None, None]) & (x < selected[:, 1, None, None])
    if output:
        valid = valid & (torch.arange(height, device=device)[None, :, None] < tiles[:, 3, None, None])
        valid = valid & (torch.arange(width, device=device)[None, None, :] < tiles[:, 4, None, None])
    elif stride != (1, 1) or dilation != (1, 1):
        output_sizes = shapes[:, 2:].index_select(0, owner)
        candidate_y = y[:, :, 0, None] + padding[0] - torch.arange(kernel[0], device=device) * dilation[0]
        candidate_x = x[:, 0, :, None] + padding[1] - torch.arange(kernel[1], device=device) * dilation[1]
        covered_y = (
            (candidate_y >= 0)
            & (candidate_y.remainder(stride[0]) == 0)
            & (torch.div(candidate_y, stride[0], rounding_mode="floor") < output_sizes[:, 0, None, None])
        ).any(-1)
        covered_x = (
            (candidate_x >= 0)
            & (candidate_x.remainder(stride[1]) == 0)
            & (torch.div(candidate_x, stride[1], rounding_mode="floor") < output_sizes[:, 1, None, None])
        ).any(-1)
        valid = valid & covered_y[:, :, None] & covered_x[:, None, :]
    rows = offsets.index_select(0, owner)[:, None, None] + y * selected[:, 1, None, None] + x
    return rows.reshape(-1), valid.reshape(-1)


class _InputCopy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values,
        offsets,
        shapes,
        meta,
        height,
        width,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        kernel_h,
        kernel_w,
        dilation_h,
        dilation_w,
    ):
        count, channels = meta.shape[0], values.shape[1]
        result = torch.empty(
            (count, channels, height, width),
            device=values.device,
            dtype=values.dtype,
            memory_format=torch.channels_last,
        )
        _spatial_tile_pack_input_local_kernel[(count, triton.cdiv(height * width * channels, 1024))](
            values,
            result,
            meta,
            offsets,
            shapes,
            0,
            *result.stride(),
            channels,
            height,
            width,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            kernel_h,
            kernel_w,
            dilation_h,
            dilation_w,
            1024,
        )
        ctx.save_for_backward(offsets, shapes, meta)
        ctx.packed_shape = tuple(values.shape)
        ctx.height, ctx.width, ctx.padding = height, width, (pad_h, pad_w)
        ctx.stride, ctx.kernel, ctx.dilation = (stride_h, stride_w), (kernel_h, kernel_w), (dilation_h, dilation_w)
        return result

    @staticmethod
    def backward(ctx, gradient):
        offsets, shapes, meta = ctx.saved_tensors
        total, channels = ctx.packed_shape
        gradient = gradient.resolve_conj().resolve_neg()
        if torch.is_grad_enabled() or not _copy_context_is_safe(gradient):
            rows, valid = _rows(
                offsets,
                shapes,
                meta,
                ctx.height,
                ctx.width,
                ctx.padding,
                stride=ctx.stride,
                kernel=ctx.kernel,
                dilation=ctx.dilation,
            )
            safe = torch.where(valid, rows, torch.arange(rows.numel(), device=rows.device).remainder(total))
            values = gradient.permute(0, 2, 3, 1).reshape(-1, channels)
            values = torch.where(valid[:, None], values, 0)
            result = gradient.new_zeros((total, channels)).index_add(0, safe, values)
        else:
            result = gradient.new_zeros((total, channels))
            count = meta.shape[0]
            _spatial_tile_scatter_input_local_kernel[(count, triton.cdiv(ctx.height * ctx.width * channels, 1024))](
                gradient,
                result,
                meta,
                offsets,
                shapes,
                0,
                *gradient.stride(),
                channels,
                ctx.height,
                ctx.width,
                *ctx.stride,
                *ctx.padding,
                *ctx.kernel,
                *ctx.dilation,
                count != shapes.shape[0],
                1024,
            )
        return (result,) + (None,) * 13


class _OutputCopy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, offsets, shapes, meta, total):
        count, channels, height, width = values.shape
        result = values.new_empty((total, channels))
        _spatial_tile_scatter_output_local_kernel[(count, triton.cdiv(height * width * channels, 1024))](
            values, result, result, meta, offsets, shapes, 0, *values.stride(), channels, height, width, False, 1024
        )
        ctx.save_for_backward(offsets, shapes, meta)
        ctx.tile_shape = tuple(values.shape)
        return result

    @staticmethod
    def backward(ctx, gradient):
        offsets, shapes, meta = ctx.saved_tensors
        count, channels, height, width = ctx.tile_shape
        gradient = gradient.resolve_conj().resolve_neg()
        if torch.is_grad_enabled() or not _copy_context_is_safe(gradient):
            rows, valid = _rows(offsets, shapes, meta, height, width, (0, 0), output=True)
            safe = torch.where(valid, rows, torch.arange(rows.numel(), device=rows.device).remainder(gradient.shape[0]))
            result = gradient.index_select(0, safe)
            result = torch.where(valid[:, None], result, 0)
            result = result.reshape(count, height, width, channels).permute(0, 3, 1, 2)
        else:
            result = torch.empty(
                ctx.tile_shape, dtype=gradient.dtype, device=gradient.device, memory_format=torch.channels_last
            )
            block_size = 512
            _spatial_tile_pack_output_local_kernel[(count, triton.cdiv(height * width * channels, block_size))](
                gradient,
                result,
                meta,
                offsets,
                shapes,
                0,
                *gradient.stride(),
                *result.stride(),
                channels,
                height,
                width,
                block_size,
            )
        return result, None, None, None, None


def pack(values, offsets, shapes, meta, height, width, padding, stride=(1, 1), kernel=(1, 1), dilation=(1, 1)):
    return _InputCopy.apply(values, offsets, shapes, meta, height, width, *padding, *stride, *kernel, *dilation)


def unpack(values, offsets, shapes, meta, total):
    return _OutputCopy.apply(values, offsets, shapes, meta, total)
