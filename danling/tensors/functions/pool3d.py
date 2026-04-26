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
r"""``avg_pool3d`` and ``max_pool3d`` handlers for NestedTensor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from ._pooling import (
    _from_pool_values,
    _per_element,
    _per_element_pair,
    _pool_block_size,
    _pool_output_size,
    _pool_tile_batch,
    _resolve_element_shapes,
    _tile_to_batch,
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
# NestedTensor pool3d dispatch
# ---------------------------------------------------------------------------


def avg_pool3d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
    *,
    _fn=None,
):
    output = _packed_avg_pool3d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    if output is not None:
        return output
    return _per_element(
        input,
        F.avg_pool3d if _fn is None else _fn,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )


def max_pool3d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
    *,
    _fn=None,
):
    fn = F.max_pool3d if _fn is None else _fn
    if not return_indices:
        output = _packed_max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)
        if output is not None:
            return output
    if return_indices:
        return _per_element_pair(input, fn, kernel_size, stride, padding, dilation, ceil_mode)
    return _per_element(input, fn, kernel_size, stride, padding, dilation, ceil_mode, return_indices)


# ---------------------------------------------------------------------------
# Packed CUDA pooling implementation
# ---------------------------------------------------------------------------


def _can_use_packed_pool3d(input: NestedTensor) -> bool:
    if len(input) == 0 or triton is None or not input._values.is_cuda:
        return False
    if input._physical_shape.size(1) != 4:
        return False
    if input._element_shapes is not None and any(len(shape) != 4 for shape in input._element_shapes):
        return False
    if tuple(int(dim) for dim in input._permutation) != (1, 2, 3, 0) or input._values.dim() != 2:
        return False

    channels = int(input._physical_shape[0, 0])
    if int(input._values.shape[1]) != channels:
        return False
    return bool(torch.equal(input._physical_shape[:, 0], torch.full_like(input._physical_shape[:, 0], channels)))


def _pool3d_output_meta(
    input: NestedTensor,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int] = (1, 1, 1),
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], Tensor] | None:
    output_shapes = []
    output_packed_sizes = []
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    padding_d, padding_h, padding_w = padding
    dilation_d, dilation_h, dilation_w = dilation
    for shape in _resolve_element_shapes(input):
        if len(shape) != 4:
            return None
        channels, depth, height, width = shape
        out_d = _pool_output_size(depth, kernel_d, stride_d, padding_d, dilation_d)
        out_h = _pool_output_size(height, kernel_h, stride_h, padding_h, dilation_h)
        out_w = _pool_output_size(width, kernel_w, stride_w, padding_w, dilation_w)
        if out_d <= 0 or out_h <= 0 or out_w <= 0:
            return None
        output_shapes.append((channels, out_d, out_h, out_w))
        output_packed_sizes.append(out_d * out_h * out_w)
    if not output_shapes:
        return None
    return tuple(output_shapes), tuple(output_packed_sizes), torch.tensor(output_shapes, dtype=torch.long)


def _pool3d_tile_counts(output_shapes: tuple[tuple[int, ...], ...], tile_size: int) -> tuple[int, ...]:
    return tuple(
        (int(shape[1]) * int(shape[2]) * int(shape[3]) + tile_size - 1) // tile_size for shape in output_shapes
    )


def _make_pool3d_metadata(
    input: NestedTensor,
    output_shapes: tuple[tuple[int, ...], ...],
    output_offsets: Tensor,
    tile_counts: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    device = input._values.device
    input_shapes = _resolve_element_shapes(input)
    shape_meta = torch.tensor(
        [
            (
                int(input_shape[1]),
                int(input_shape[2]),
                int(input_shape[3]),
                int(output_shape[1]),
                int(output_shape[2]),
                int(output_shape[3]),
            )
            for input_shape, output_shape in zip(input_shapes, output_shapes)
        ],
        device=device,
        dtype=torch.int64,
    )
    tile_offsets = type(input)._offsets_from_sizes(tile_counts, dtype=torch.long)
    return (
        input._offsets.to(device=device, non_blocking=True),
        output_offsets.to(device=device, non_blocking=True),
        shape_meta,
        tile_offsets.to(device=device, non_blocking=True),
        _tile_to_batch(tile_counts, device=device),
    )


if triton is not None:

    @triton.jit
    def _avg_pool3d_forward_kernel(
        input_ptr,
        output_ptr,
        input_offsets_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        tile_offsets_ptr,
        tile_to_batch_ptr,
        batch_count: tl.constexpr,
        channels: tl.constexpr,
        kernel_d: tl.constexpr,
        kernel_h: tl.constexpr,
        kernel_w: tl.constexpr,
        stride_d: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        padding_d: tl.constexpr,
        padding_h: tl.constexpr,
        padding_w: tl.constexpr,
        count_include_pad: tl.constexpr,
        divisor_override: tl.constexpr,
        block_l: tl.constexpr,
        block_c: tl.constexpr,
    ):
        tile = tl.program_id(0)
        channel_block = tl.program_id(1)
        batch = _pool_tile_batch(tile_to_batch_ptr, tile)
        out_start = (tile - tl.load(tile_offsets_ptr + batch)) * block_l
        in_d = tl.load(shape_meta_ptr + batch * 6)
        in_h = tl.load(shape_meta_ptr + batch * 6 + 1)
        in_w = tl.load(shape_meta_ptr + batch * 6 + 2)
        out_d = tl.load(shape_meta_ptr + batch * 6 + 3)
        out_h = tl.load(shape_meta_ptr + batch * 6 + 4)
        out_w = tl.load(shape_meta_ptr + batch * 6 + 5)
        out_hw = out_h * out_w
        out_dhw = out_d * out_hw
        input_base = tl.load(input_offsets_ptr + batch)
        output_base = tl.load(output_offsets_ptr + batch)
        offsets_l = tl.arange(0, block_l)
        offsets_c = channel_block * block_c + tl.arange(0, block_c)
        out_flat = out_start + offsets_l
        out_z = out_flat // out_hw
        out_rem = out_flat - out_z * out_hw
        out_y = out_rem // out_w
        out_x = out_rem - out_y * out_w
        acc = tl.zeros((block_l, block_c), dtype=tl.float32)
        valid_count = tl.zeros((block_l,), dtype=tl.float32)

        for kernel_z in range(0, kernel_d):
            in_z = out_z * stride_d - padding_d + kernel_z
            valid_z = (in_z >= 0) & (in_z < in_d)
            for kernel_y in range(0, kernel_h):
                in_y = out_y * stride_h - padding_h + kernel_y
                valid_zy = valid_z & (in_y >= 0) & (in_y < in_h)
                for kernel_x in range(0, kernel_w):
                    in_x = out_x * stride_w - padding_w + kernel_x
                    valid = (out_flat < out_dhw) & valid_zy & (in_x >= 0) & (in_x < in_w)
                    safe_in_z = tl.where(valid, in_z, 0)
                    safe_in_y = tl.where(valid, in_y, 0)
                    safe_in_x = tl.where(valid, in_x, 0)
                    input_index = (safe_in_z * in_h + safe_in_y) * in_w + safe_in_x
                    values = tl.load(
                        input_ptr + (input_base + input_index[:, None]) * channels + offsets_c[None, :],
                        mask=valid[:, None] & (offsets_c[None, :] < channels),
                        other=0.0,
                    )
                    acc += values
                    valid_count += tl.where(valid, 1.0, 0.0)

        if divisor_override > 0:
            divisor = divisor_override + tl.zeros((block_l,), dtype=tl.float32)
        elif count_include_pad:
            divisor = kernel_d * kernel_h * kernel_w + tl.zeros((block_l,), dtype=tl.float32)
        else:
            divisor = valid_count
        result = acc / divisor[:, None]
        output_mask = (out_flat[:, None] < out_dhw) & (offsets_c[None, :] < channels)
        tl.store(
            output_ptr + (output_base + out_flat[:, None]) * channels + offsets_c[None, :],
            result,
            mask=output_mask,
        )

    @triton.jit
    def _avg_pool3d_backward_kernel(
        grad_output_ptr,
        grad_input_ptr,
        input_offsets_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        tile_offsets_ptr,
        tile_to_batch_ptr,
        batch_count: tl.constexpr,
        channels: tl.constexpr,
        kernel_d: tl.constexpr,
        kernel_h: tl.constexpr,
        kernel_w: tl.constexpr,
        stride_d: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        padding_d: tl.constexpr,
        padding_h: tl.constexpr,
        padding_w: tl.constexpr,
        count_include_pad: tl.constexpr,
        divisor_override: tl.constexpr,
        non_overlapping: tl.constexpr,
        block_l: tl.constexpr,
        block_c: tl.constexpr,
    ):
        tile = tl.program_id(0)
        channel_block = tl.program_id(1)
        batch = _pool_tile_batch(tile_to_batch_ptr, tile)
        out_start = (tile - tl.load(tile_offsets_ptr + batch)) * block_l
        in_d = tl.load(shape_meta_ptr + batch * 6)
        in_h = tl.load(shape_meta_ptr + batch * 6 + 1)
        in_w = tl.load(shape_meta_ptr + batch * 6 + 2)
        out_d = tl.load(shape_meta_ptr + batch * 6 + 3)
        out_h = tl.load(shape_meta_ptr + batch * 6 + 4)
        out_w = tl.load(shape_meta_ptr + batch * 6 + 5)
        out_hw = out_h * out_w
        out_dhw = out_d * out_hw
        input_base = tl.load(input_offsets_ptr + batch)
        output_base = tl.load(output_offsets_ptr + batch)
        offsets_l = tl.arange(0, block_l)
        offsets_c = channel_block * block_c + tl.arange(0, block_c)
        out_flat = out_start + offsets_l
        out_z = out_flat // out_hw
        out_rem = out_flat - out_z * out_hw
        out_y = out_rem // out_w
        out_x = out_rem - out_y * out_w
        valid_count = tl.zeros((block_l,), dtype=tl.float32)

        for kernel_z in range(0, kernel_d):
            in_z = out_z * stride_d - padding_d + kernel_z
            valid_z = (in_z >= 0) & (in_z < in_d)
            for kernel_y in range(0, kernel_h):
                in_y = out_y * stride_h - padding_h + kernel_y
                valid_zy = valid_z & (in_y >= 0) & (in_y < in_h)
                for kernel_x in range(0, kernel_w):
                    in_x = out_x * stride_w - padding_w + kernel_x
                    valid = (out_flat < out_dhw) & valid_zy & (in_x >= 0) & (in_x < in_w)
                    valid_count += tl.where(valid, 1.0, 0.0)

        if divisor_override > 0:
            divisor = divisor_override + tl.zeros((block_l,), dtype=tl.float32)
        elif count_include_pad:
            divisor = kernel_d * kernel_h * kernel_w + tl.zeros((block_l,), dtype=tl.float32)
        else:
            divisor = valid_count
        grad = (
            tl.load(
                grad_output_ptr + (output_base + out_flat[:, None]) * channels + offsets_c[None, :],
                mask=(out_flat[:, None] < out_dhw) & (offsets_c[None, :] < channels),
                other=0.0,
            )
            / divisor[:, None]
        )

        for kernel_z in range(0, kernel_d):
            in_z = out_z * stride_d - padding_d + kernel_z
            valid_z = (in_z >= 0) & (in_z < in_d)
            for kernel_y in range(0, kernel_h):
                in_y = out_y * stride_h - padding_h + kernel_y
                valid_zy = valid_z & (in_y >= 0) & (in_y < in_h)
                for kernel_x in range(0, kernel_w):
                    in_x = out_x * stride_w - padding_w + kernel_x
                    valid = (out_flat < out_dhw) & valid_zy & (in_x >= 0) & (in_x < in_w)
                    safe_in_z = tl.where(valid, in_z, 0)
                    safe_in_y = tl.where(valid, in_y, 0)
                    safe_in_x = tl.where(valid, in_x, 0)
                    input_index = (safe_in_z * in_h + safe_in_y) * in_w + safe_in_x
                    grad_input_ptrs = (
                        grad_input_ptr + (input_base + input_index[:, None]) * channels + offsets_c[None, :]
                    )
                    grad_mask = valid[:, None] & (offsets_c[None, :] < channels)
                    if non_overlapping:
                        tl.store(grad_input_ptrs, grad, mask=grad_mask)
                    else:
                        tl.atomic_add(grad_input_ptrs, grad, sem="relaxed", mask=grad_mask)

    @triton.jit
    def _max_pool3d_forward_kernel(
        input_ptr,
        output_ptr,
        indices_ptr,
        input_offsets_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        tile_offsets_ptr,
        tile_to_batch_ptr,
        batch_count: tl.constexpr,
        channels: tl.constexpr,
        kernel_d: tl.constexpr,
        kernel_h: tl.constexpr,
        kernel_w: tl.constexpr,
        stride_d: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        padding_d: tl.constexpr,
        padding_h: tl.constexpr,
        padding_w: tl.constexpr,
        dilation_d: tl.constexpr,
        dilation_h: tl.constexpr,
        dilation_w: tl.constexpr,
        save_indices: tl.constexpr,
        block_l: tl.constexpr,
        block_c: tl.constexpr,
    ):
        tile = tl.program_id(0)
        channel_block = tl.program_id(1)
        batch = _pool_tile_batch(tile_to_batch_ptr, tile)
        out_start = (tile - tl.load(tile_offsets_ptr + batch)) * block_l
        in_d = tl.load(shape_meta_ptr + batch * 6)
        in_h = tl.load(shape_meta_ptr + batch * 6 + 1)
        in_w = tl.load(shape_meta_ptr + batch * 6 + 2)
        out_d = tl.load(shape_meta_ptr + batch * 6 + 3)
        out_h = tl.load(shape_meta_ptr + batch * 6 + 4)
        out_w = tl.load(shape_meta_ptr + batch * 6 + 5)
        out_hw = out_h * out_w
        out_dhw = out_d * out_hw
        input_base = tl.load(input_offsets_ptr + batch)
        output_base = tl.load(output_offsets_ptr + batch)
        offsets_l = tl.arange(0, block_l)
        offsets_c = channel_block * block_c + tl.arange(0, block_c)
        out_flat = out_start + offsets_l
        out_z = out_flat // out_hw
        out_rem = out_flat - out_z * out_hw
        out_y = out_rem // out_w
        out_x = out_rem - out_y * out_w
        max_values = tl.full((block_l, block_c), -float("inf"), dtype=tl.float32)
        max_indices = tl.zeros((block_l, block_c), dtype=tl.int64)

        for kernel_z in range(0, kernel_d):
            in_z = out_z * stride_d - padding_d + kernel_z * dilation_d
            valid_z = (in_z >= 0) & (in_z < in_d)
            for kernel_y in range(0, kernel_h):
                in_y = out_y * stride_h - padding_h + kernel_y * dilation_h
                valid_zy = valid_z & (in_y >= 0) & (in_y < in_h)
                for kernel_x in range(0, kernel_w):
                    in_x = out_x * stride_w - padding_w + kernel_x * dilation_w
                    valid = (out_flat < out_dhw) & valid_zy & (in_x >= 0) & (in_x < in_w)
                    safe_in_z = tl.where(valid, in_z, 0)
                    safe_in_y = tl.where(valid, in_y, 0)
                    safe_in_x = tl.where(valid, in_x, 0)
                    input_index = (safe_in_z * in_h + safe_in_y) * in_w + safe_in_x
                    values = tl.load(
                        input_ptr + (input_base + input_index[:, None]) * channels + offsets_c[None, :],
                        mask=valid[:, None] & (offsets_c[None, :] < channels),
                        other=-float("inf"),
                    )
                    better = values > max_values
                    max_values = tl.where(better, values, max_values)
                    max_indices = tl.where(better, input_index[:, None], max_indices)

        output_mask = (out_flat[:, None] < out_dhw) & (offsets_c[None, :] < channels)
        tl.store(
            output_ptr + (output_base + out_flat[:, None]) * channels + offsets_c[None, :],
            max_values,
            mask=output_mask,
        )
        if save_indices:
            tl.store(
                indices_ptr + (output_base + out_flat[:, None]) * channels + offsets_c[None, :],
                max_indices,
                mask=output_mask,
            )

    @triton.jit
    def _max_pool3d_backward_indices_kernel(
        grad_output_ptr,
        grad_input_ptr,
        indices_ptr,
        input_offsets_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        tile_offsets_ptr,
        tile_to_batch_ptr,
        batch_count: tl.constexpr,
        channels: tl.constexpr,
        block_l: tl.constexpr,
        block_c: tl.constexpr,
    ):
        tile = tl.program_id(0)
        channel_block = tl.program_id(1)
        batch = _pool_tile_batch(tile_to_batch_ptr, tile)
        out_start = (tile - tl.load(tile_offsets_ptr + batch)) * block_l
        out_d = tl.load(shape_meta_ptr + batch * 6 + 3)
        out_h = tl.load(shape_meta_ptr + batch * 6 + 4)
        out_w = tl.load(shape_meta_ptr + batch * 6 + 5)
        out_dhw = out_d * out_h * out_w
        input_base = tl.load(input_offsets_ptr + batch)
        output_base = tl.load(output_offsets_ptr + batch)
        offsets_l = tl.arange(0, block_l)
        offsets_c = channel_block * block_c + tl.arange(0, block_c)
        out_flat = out_start + offsets_l
        mask = (out_flat[:, None] < out_dhw) & (offsets_c[None, :] < channels)
        indices = tl.load(
            indices_ptr + (output_base + out_flat[:, None]) * channels + offsets_c[None, :],
            mask=mask,
            other=0,
        )
        grad = tl.load(
            grad_output_ptr + (output_base + out_flat[:, None]) * channels + offsets_c[None, :],
            mask=mask,
            other=0.0,
        )
        grad_input_ptrs = grad_input_ptr + (input_base + indices) * channels + offsets_c[None, :]
        tl.atomic_add(grad_input_ptrs, grad, sem="relaxed", mask=mask)

else:  # pragma: no cover - optional dependency/runtime
    _avg_pool3d_forward_kernel = None
    _avg_pool3d_backward_kernel = None
    _max_pool3d_forward_kernel = None
    _max_pool3d_backward_indices_kernel = None


class _PackedAvgPool3dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_values: Tensor,
        input_offsets: Tensor,
        output_offsets: Tensor,
        shape_meta: Tensor,
        tile_offsets: Tensor,
        tile_to_batch: Tensor,
        tile_count: int,
        batch_count: int,
        output_size: int,
        channels: int,
        kernel_d: int,
        kernel_h: int,
        kernel_w: int,
        stride_d: int,
        stride_h: int,
        stride_w: int,
        padding_d: int,
        padding_h: int,
        padding_w: int,
        count_include_pad: bool,
        divisor_override: int,
    ) -> Tensor:
        output = input_values.new_empty((output_size, channels))
        block_l, block_c = _pool_block_size()
        grid = (tile_count, triton.cdiv(channels, block_c))
        _avg_pool3d_forward_kernel[grid](
            input_values,
            output,
            input_offsets,
            output_offsets,
            shape_meta,
            tile_offsets,
            tile_to_batch,
            batch_count,
            channels,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
            count_include_pad,
            divisor_override,
            block_l,
            block_c,
        )
        ctx.save_for_backward(input_offsets, output_offsets, shape_meta, tile_offsets, tile_to_batch)
        ctx.tile_count = tile_count
        ctx.batch_count = batch_count
        ctx.input_shape = tuple(int(size) for size in input_values.shape)
        ctx.channels = channels
        ctx.kernel_d = kernel_d
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_d = stride_d
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.padding_d = padding_d
        ctx.padding_h = padding_h
        ctx.padding_w = padding_w
        ctx.count_include_pad = count_include_pad
        ctx.divisor_override = divisor_override
        ctx.non_overlapping = stride_d >= kernel_d and stride_h >= kernel_h and stride_w >= kernel_w
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input_offsets, output_offsets, shape_meta, tile_offsets, tile_to_batch = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = grad_output.new_zeros(ctx.input_shape)
        block_l, block_c = _pool_block_size()
        grid = (ctx.tile_count, triton.cdiv(ctx.channels, block_c))
        _avg_pool3d_backward_kernel[grid](
            grad_output,
            grad_input,
            input_offsets,
            output_offsets,
            shape_meta,
            tile_offsets,
            tile_to_batch,
            ctx.batch_count,
            ctx.channels,
            ctx.kernel_d,
            ctx.kernel_h,
            ctx.kernel_w,
            ctx.stride_d,
            ctx.stride_h,
            ctx.stride_w,
            ctx.padding_d,
            ctx.padding_h,
            ctx.padding_w,
            ctx.count_include_pad,
            ctx.divisor_override,
            ctx.non_overlapping,
            block_l,
            block_c,
        )
        return (grad_input, *([None] * 20))


class _PackedMaxPool3dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_values: Tensor,
        input_offsets: Tensor,
        output_offsets: Tensor,
        shape_meta: Tensor,
        tile_offsets: Tensor,
        tile_to_batch: Tensor,
        tile_count: int,
        batch_count: int,
        output_size: int,
        channels: int,
        kernel_d: int,
        kernel_h: int,
        kernel_w: int,
        stride_d: int,
        stride_h: int,
        stride_w: int,
        padding_d: int,
        padding_h: int,
        padding_w: int,
        dilation_d: int,
        dilation_h: int,
        dilation_w: int,
    ) -> Tensor:
        output = input_values.new_empty((output_size, channels))
        max_input_size = int((shape_meta[:, 0] * shape_meta[:, 1] * shape_meta[:, 2]).max().item())
        save_indices = bool(input_values.requires_grad)
        indices_dtype = torch.int32 if max_input_size < 2**31 else torch.long
        indices = (
            torch.empty((output_size, channels), device=input_values.device, dtype=indices_dtype)
            if save_indices
            else output
        )
        block_l, block_c = _pool_block_size()
        grid = (tile_count, triton.cdiv(channels, block_c))
        _max_pool3d_forward_kernel[grid](
            input_values,
            output,
            indices,
            input_offsets,
            output_offsets,
            shape_meta,
            tile_offsets,
            tile_to_batch,
            batch_count,
            channels,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
            dilation_d,
            dilation_h,
            dilation_w,
            save_indices,
            block_l,
            block_c,
        )
        ctx.save_for_backward(indices, input_offsets, output_offsets, shape_meta, tile_offsets, tile_to_batch)
        ctx.tile_count = tile_count
        ctx.batch_count = batch_count
        ctx.input_shape = tuple(int(size) for size in input_values.shape)
        ctx.channels = channels
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        indices, input_offsets, output_offsets, shape_meta, tile_offsets, tile_to_batch = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = grad_output.new_zeros(ctx.input_shape)
        block_l, block_c = _pool_block_size()
        grid = (ctx.tile_count, triton.cdiv(ctx.channels, block_c))
        _max_pool3d_backward_indices_kernel[grid](
            grad_output,
            grad_input,
            indices,
            input_offsets,
            output_offsets,
            shape_meta,
            tile_offsets,
            tile_to_batch,
            ctx.batch_count,
            ctx.channels,
            block_l,
            block_c,
        )
        return (grad_input, *([None] * 21))


def _packed_avg_pool3d(
    input: NestedTensor,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
) -> NestedTensor | None:
    if ceil_mode or not _can_use_packed_pool3d(input):
        return None
    if not isinstance(kernel_size, int) or (stride is not None and not isinstance(stride, int)):
        return None
    if not isinstance(padding, int):
        return None
    kernel = (int(kernel_size), int(kernel_size), int(kernel_size))
    stride_value = kernel_size if stride is None else int(stride)
    stride_triple = (stride_value, stride_value, stride_value)
    padding_triple = (int(padding), int(padding), int(padding))
    divisor = 0 if divisor_override is None else int(divisor_override)
    if any(value <= 0 for value in (*kernel, *stride_triple)):
        return None
    if (
        min(padding_triple) < 0
        or padding_triple[0] > kernel[0] // 2
        or padding_triple[1] > kernel[1] // 2
        or padding_triple[2] > kernel[2] // 2
    ):
        return None
    if divisor <= 0 and divisor_override is not None:
        return None
    output_meta = _pool3d_output_meta(input, kernel, stride_triple, padding_triple)
    if output_meta is None:
        return None
    output_shapes, output_packed_sizes, output_shape_tensor = output_meta
    output_offsets = type(input)._offsets_from_sizes(output_packed_sizes, dtype=torch.long)
    total_out = int(output_offsets[-1].item())
    channels = int(input._values.shape[1])
    tile_counts = _pool3d_tile_counts(output_shapes, _pool_block_size()[0])
    input_offsets, output_offsets_device, shape_meta, tile_offsets, tile_to_batch = _make_pool3d_metadata(
        input,
        output_shapes,
        output_offsets,
        tile_counts,
    )
    output_values = _PackedAvgPool3dFunction.apply(
        input._values,
        input_offsets,
        output_offsets_device,
        shape_meta,
        tile_offsets,
        tile_to_batch,
        sum(tile_counts),
        len(tile_counts),
        total_out,
        channels,
        kernel[0],
        kernel[1],
        kernel[2],
        stride_triple[0],
        stride_triple[1],
        stride_triple[2],
        padding_triple[0],
        padding_triple[1],
        padding_triple[2],
        bool(count_include_pad),
        divisor,
    )
    return _from_pool_values(
        input, output_values, output_offsets, output_shape_tensor, output_packed_sizes, output_shapes
    )


def _packed_max_pool3d(
    input: NestedTensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
) -> NestedTensor | None:
    if ceil_mode or not _can_use_packed_pool3d(input):
        return None
    if not isinstance(kernel_size, int) or (stride is not None and not isinstance(stride, int)):
        return None
    if not isinstance(padding, int) or not isinstance(dilation, int):
        return None
    kernel = (int(kernel_size), int(kernel_size), int(kernel_size))
    stride_value = kernel_size if stride is None else int(stride)
    stride_triple = (stride_value, stride_value, stride_value)
    padding_triple = (int(padding), int(padding), int(padding))
    dilation_triple = (int(dilation), int(dilation), int(dilation))
    if any(value <= 0 for value in (*kernel, *stride_triple, *dilation_triple)):
        return None
    if (
        min(padding_triple) < 0
        or padding_triple[0] > kernel[0] // 2
        or padding_triple[1] > kernel[1] // 2
        or padding_triple[2] > kernel[2] // 2
    ):
        return None
    output_meta = _pool3d_output_meta(input, kernel, stride_triple, padding_triple, dilation_triple)
    if output_meta is None:
        return None
    output_shapes, output_packed_sizes, output_shape_tensor = output_meta
    output_offsets = type(input)._offsets_from_sizes(output_packed_sizes, dtype=torch.long)
    total_out = int(output_offsets[-1].item())
    channels = int(input._values.shape[1])
    tile_counts = _pool3d_tile_counts(output_shapes, _pool_block_size()[0])
    input_offsets, output_offsets_device, shape_meta, tile_offsets, tile_to_batch = _make_pool3d_metadata(
        input,
        output_shapes,
        output_offsets,
        tile_counts,
    )
    output_values = _PackedMaxPool3dFunction.apply(
        input._values,
        input_offsets,
        output_offsets_device,
        shape_meta,
        tile_offsets,
        tile_to_batch,
        sum(tile_counts),
        len(tile_counts),
        total_out,
        channels,
        kernel[0],
        kernel[1],
        kernel[2],
        stride_triple[0],
        stride_triple[1],
        stride_triple[2],
        padding_triple[0],
        padding_triple[1],
        padding_triple[2],
        dilation_triple[0],
        dilation_triple[1],
        dilation_triple[2],
    )
    return _from_pool_values(
        input, output_values, output_offsets, output_shape_tensor, output_packed_sizes, output_shapes
    )
