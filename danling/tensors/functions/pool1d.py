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
r"""``avg_pool1d`` and ``max_pool1d`` handlers for NestedTensor."""

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
# NestedTensor pool1d dispatch
# ---------------------------------------------------------------------------


def avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, *, _fn=None):
    output = _packed_avg_pool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    if output is not None:
        return output
    return _per_element(
        input,
        F.avg_pool1d if _fn is None else _fn,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
    )


def max_pool1d(
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
    fn = F.max_pool1d if _fn is None else _fn
    if not return_indices:
        output = _packed_max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode)
        if output is not None:
            return output
    if return_indices:
        return _per_element_pair(input, fn, kernel_size, stride, padding, dilation, ceil_mode)
    return _per_element(input, fn, kernel_size, stride, padding, dilation, ceil_mode, return_indices)


# ---------------------------------------------------------------------------
# Packed CUDA pooling implementation
# ---------------------------------------------------------------------------


def _can_use_packed_pool1d(input: NestedTensor) -> bool:
    if len(input) == 0 or triton is None or not input._values.is_cuda:
        return False
    if input._physical_shape.size(1) != 2:
        return False
    if input._element_shapes is not None and any(len(shape) != 2 for shape in input._element_shapes):
        return False
    if tuple(int(dim) for dim in input._permutation) != (1, 0) or input._values.dim() != 2:
        return False

    channels = int(input._physical_shape[0, 0])
    if int(input._values.shape[1]) != channels:
        return False
    return bool(torch.equal(input._physical_shape[:, 0], torch.full_like(input._physical_shape[:, 0], channels)))


def _pool1d_output_meta(
    input: NestedTensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], Tensor] | None:
    output_shapes = []
    output_packed_sizes = []
    for shape in _resolve_element_shapes(input):
        if len(shape) != 2:
            return None
        channels, length = shape
        out_length = _pool_output_size(length, kernel_size, stride, padding, dilation)
        if out_length <= 0:
            return None
        output_shapes.append((channels, out_length))
        output_packed_sizes.append(out_length)
    if not output_shapes:
        return None
    return tuple(output_shapes), tuple(output_packed_sizes), torch.tensor(output_shapes, dtype=torch.long)


def _pool1d_tile_counts(output_shapes: tuple[tuple[int, ...], ...], tile_length: int) -> tuple[int, ...]:
    return tuple((int(shape[1]) + tile_length - 1) // tile_length for shape in output_shapes)


def _make_pool1d_metadata(
    input: NestedTensor,
    output_shapes: tuple[tuple[int, ...], ...],
    output_offsets: Tensor,
    tile_counts: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    device = input._values.device
    input_shapes = _resolve_element_shapes(input)
    shape_meta = torch.tensor(
        [(int(input_shape[1]), int(output_shape[1])) for input_shape, output_shape in zip(input_shapes, output_shapes)],
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
    def _avg_pool1d_forward_kernel(
        input_ptr,
        output_ptr,
        input_offsets_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        tile_offsets_ptr,
        tile_to_batch_ptr,
        batch_count: tl.constexpr,
        channels: tl.constexpr,
        kernel_size: tl.constexpr,
        stride: tl.constexpr,
        padding: tl.constexpr,
        count_include_pad: tl.constexpr,
        divisor_override: tl.constexpr,
        block_l: tl.constexpr,
        block_c: tl.constexpr,
    ):
        tile = tl.program_id(0)
        channel_block = tl.program_id(1)
        batch = _pool_tile_batch(tile_to_batch_ptr, tile)
        out_start = (tile - tl.load(tile_offsets_ptr + batch)) * block_l
        in_length = tl.load(shape_meta_ptr + batch * 2)
        out_length = tl.load(shape_meta_ptr + batch * 2 + 1)
        input_base = tl.load(input_offsets_ptr + batch)
        output_base = tl.load(output_offsets_ptr + batch)
        offsets_l = tl.arange(0, block_l)
        offsets_c = channel_block * block_c + tl.arange(0, block_c)
        out_x = out_start + offsets_l
        acc = tl.zeros((block_l, block_c), dtype=tl.float32)
        valid_count = tl.zeros((block_l,), dtype=tl.float32)

        for kernel_x in range(0, kernel_size):
            in_x = out_x * stride - padding + kernel_x
            valid = (out_x < out_length) & (in_x >= 0) & (in_x < in_length)
            safe_in_x = tl.where(valid, in_x, 0)
            values = tl.load(
                input_ptr + (input_base + safe_in_x[:, None]) * channels + offsets_c[None, :],
                mask=valid[:, None] & (offsets_c[None, :] < channels),
                other=0.0,
            )
            acc += values
            valid_count += tl.where(valid, 1.0, 0.0)

        if divisor_override > 0:
            divisor = divisor_override + tl.zeros((block_l,), dtype=tl.float32)
        elif count_include_pad:
            divisor = kernel_size + tl.zeros((block_l,), dtype=tl.float32)
        else:
            divisor = valid_count
        result = acc / divisor[:, None]
        output_mask = (out_x[:, None] < out_length) & (offsets_c[None, :] < channels)
        tl.store(output_ptr + (output_base + out_x[:, None]) * channels + offsets_c[None, :], result, mask=output_mask)

    @triton.jit
    def _avg_pool1d_backward_kernel(
        grad_output_ptr,
        grad_input_ptr,
        input_offsets_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        tile_offsets_ptr,
        tile_to_batch_ptr,
        batch_count: tl.constexpr,
        channels: tl.constexpr,
        kernel_size: tl.constexpr,
        stride: tl.constexpr,
        padding: tl.constexpr,
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
        in_length = tl.load(shape_meta_ptr + batch * 2)
        out_length = tl.load(shape_meta_ptr + batch * 2 + 1)
        input_base = tl.load(input_offsets_ptr + batch)
        output_base = tl.load(output_offsets_ptr + batch)
        offsets_l = tl.arange(0, block_l)
        offsets_c = channel_block * block_c + tl.arange(0, block_c)
        out_x = out_start + offsets_l
        valid_count = tl.zeros((block_l,), dtype=tl.float32)

        for kernel_x in range(0, kernel_size):
            in_x = out_x * stride - padding + kernel_x
            valid = (out_x < out_length) & (in_x >= 0) & (in_x < in_length)
            valid_count += tl.where(valid, 1.0, 0.0)

        if divisor_override > 0:
            divisor = divisor_override + tl.zeros((block_l,), dtype=tl.float32)
        elif count_include_pad:
            divisor = kernel_size + tl.zeros((block_l,), dtype=tl.float32)
        else:
            divisor = valid_count

        grad = (
            tl.load(
                grad_output_ptr + (output_base + out_x[:, None]) * channels + offsets_c[None, :],
                mask=(out_x[:, None] < out_length) & (offsets_c[None, :] < channels),
                other=0.0,
            )
            / divisor[:, None]
        )
        for kernel_x in range(0, kernel_size):
            in_x = out_x * stride - padding + kernel_x
            valid = (out_x < out_length) & (in_x >= 0) & (in_x < in_length)
            safe_in_x = tl.where(valid, in_x, 0)
            grad_input_ptrs = grad_input_ptr + (input_base + safe_in_x[:, None]) * channels + offsets_c[None, :]
            grad_mask = valid[:, None] & (offsets_c[None, :] < channels)
            if non_overlapping:
                tl.store(grad_input_ptrs, grad, mask=grad_mask)
            else:
                tl.atomic_add(grad_input_ptrs, grad, sem="relaxed", mask=grad_mask)

    @triton.jit
    def _max_pool1d_forward_kernel(
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
        kernel_size: tl.constexpr,
        stride: tl.constexpr,
        padding: tl.constexpr,
        dilation: tl.constexpr,
        save_indices: tl.constexpr,
        block_l: tl.constexpr,
        block_c: tl.constexpr,
    ):
        tile = tl.program_id(0)
        channel_block = tl.program_id(1)
        batch = _pool_tile_batch(tile_to_batch_ptr, tile)
        out_start = (tile - tl.load(tile_offsets_ptr + batch)) * block_l
        in_length = tl.load(shape_meta_ptr + batch * 2)
        out_length = tl.load(shape_meta_ptr + batch * 2 + 1)
        input_base = tl.load(input_offsets_ptr + batch)
        output_base = tl.load(output_offsets_ptr + batch)
        offsets_l = tl.arange(0, block_l)
        offsets_c = channel_block * block_c + tl.arange(0, block_c)
        out_x = out_start + offsets_l
        max_values = tl.full((block_l, block_c), -float("inf"), dtype=tl.float32)
        max_indices = tl.zeros((block_l, block_c), dtype=tl.int64)

        for kernel_x in range(0, kernel_size):
            in_x = out_x * stride - padding + kernel_x * dilation
            valid = (out_x < out_length) & (in_x >= 0) & (in_x < in_length)
            safe_in_x = tl.where(valid, in_x, 0)
            values = tl.load(
                input_ptr + (input_base + safe_in_x[:, None]) * channels + offsets_c[None, :],
                mask=valid[:, None] & (offsets_c[None, :] < channels),
                other=-float("inf"),
            )
            better = values > max_values
            max_values = tl.where(better, values, max_values)
            max_indices = tl.where(better, safe_in_x[:, None], max_indices)

        output_mask = (out_x[:, None] < out_length) & (offsets_c[None, :] < channels)
        tl.store(
            output_ptr + (output_base + out_x[:, None]) * channels + offsets_c[None, :],
            max_values,
            mask=output_mask,
        )
        if save_indices:
            tl.store(
                indices_ptr + (output_base + out_x[:, None]) * channels + offsets_c[None, :],
                max_indices,
                mask=output_mask,
            )

    @triton.jit
    def _max_pool1d_backward_indices_kernel(
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
        out_length = tl.load(shape_meta_ptr + batch * 2 + 1)
        input_base = tl.load(input_offsets_ptr + batch)
        output_base = tl.load(output_offsets_ptr + batch)
        offsets_l = tl.arange(0, block_l)
        offsets_c = channel_block * block_c + tl.arange(0, block_c)
        out_x = out_start + offsets_l
        mask = (out_x[:, None] < out_length) & (offsets_c[None, :] < channels)
        indices = tl.load(
            indices_ptr + (output_base + out_x[:, None]) * channels + offsets_c[None, :],
            mask=mask,
            other=0,
        )
        grad = tl.load(
            grad_output_ptr + (output_base + out_x[:, None]) * channels + offsets_c[None, :],
            mask=mask,
            other=0.0,
        )
        grad_input_ptrs = grad_input_ptr + (input_base + indices) * channels + offsets_c[None, :]
        tl.atomic_add(grad_input_ptrs, grad, sem="relaxed", mask=mask)

    @triton.jit
    def _max_pool1d_backward_kernel(
        input_ptr,
        grad_output_ptr,
        grad_input_ptr,
        input_offsets_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        tile_offsets_ptr,
        tile_to_batch_ptr,
        batch_count: tl.constexpr,
        channels: tl.constexpr,
        kernel_size: tl.constexpr,
        stride: tl.constexpr,
        padding: tl.constexpr,
        dilation: tl.constexpr,
        block_l: tl.constexpr,
        block_c: tl.constexpr,
    ):
        tile = tl.program_id(0)
        channel_block = tl.program_id(1)
        batch = _pool_tile_batch(tile_to_batch_ptr, tile)
        out_start = (tile - tl.load(tile_offsets_ptr + batch)) * block_l
        in_length = tl.load(shape_meta_ptr + batch * 2)
        out_length = tl.load(shape_meta_ptr + batch * 2 + 1)
        input_base = tl.load(input_offsets_ptr + batch)
        output_base = tl.load(output_offsets_ptr + batch)
        offsets_l = tl.arange(0, block_l)
        offsets_c = channel_block * block_c + tl.arange(0, block_c)
        out_x = out_start + offsets_l
        max_values = tl.full((block_l, block_c), -float("inf"), dtype=tl.float32)
        max_indices = tl.zeros((block_l, block_c), dtype=tl.int64)

        for kernel_x in range(0, kernel_size):
            in_x = out_x * stride - padding + kernel_x * dilation
            valid = (out_x < out_length) & (in_x >= 0) & (in_x < in_length)
            safe_in_x = tl.where(valid, in_x, 0)
            values = tl.load(
                input_ptr + (input_base + safe_in_x[:, None]) * channels + offsets_c[None, :],
                mask=valid[:, None] & (offsets_c[None, :] < channels),
                other=-float("inf"),
            )
            better = values > max_values
            max_values = tl.where(better, values, max_values)
            max_indices = tl.where(better, safe_in_x[:, None], max_indices)

        grad = tl.load(
            grad_output_ptr + (output_base + out_x[:, None]) * channels + offsets_c[None, :],
            mask=(out_x[:, None] < out_length) & (offsets_c[None, :] < channels),
            other=0.0,
        )
        grad_input_ptrs = grad_input_ptr + (input_base + max_indices) * channels + offsets_c[None, :]
        grad_mask = (out_x[:, None] < out_length) & (offsets_c[None, :] < channels)
        tl.atomic_add(grad_input_ptrs, grad, sem="relaxed", mask=grad_mask)

else:  # pragma: no cover - optional dependency/runtime
    _avg_pool1d_forward_kernel = None
    _avg_pool1d_backward_kernel = None
    _max_pool1d_forward_kernel = None
    _max_pool1d_backward_indices_kernel = None
    _max_pool1d_backward_kernel = None


class _PackedAvgPool1dFunction(torch.autograd.Function):
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
        kernel_size: int,
        stride: int,
        padding: int,
        count_include_pad: bool,
        divisor_override: int,
    ) -> Tensor:
        output = input_values.new_empty((output_size, channels))
        block_l, block_c = _pool_block_size()
        grid = (tile_count, triton.cdiv(channels, block_c))
        _avg_pool1d_forward_kernel[grid](
            input_values,
            output,
            input_offsets,
            output_offsets,
            shape_meta,
            tile_offsets,
            tile_to_batch,
            batch_count,
            channels,
            kernel_size,
            stride,
            padding,
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
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.count_include_pad = count_include_pad
        ctx.divisor_override = divisor_override
        ctx.non_overlapping = stride >= kernel_size
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input_offsets, output_offsets, shape_meta, tile_offsets, tile_to_batch = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = grad_output.new_zeros(ctx.input_shape)
        block_l, block_c = _pool_block_size()
        grid = (ctx.tile_count, triton.cdiv(ctx.channels, block_c))
        _avg_pool1d_backward_kernel[grid](
            grad_output,
            grad_input,
            input_offsets,
            output_offsets,
            shape_meta,
            tile_offsets,
            tile_to_batch,
            ctx.batch_count,
            ctx.channels,
            ctx.kernel_size,
            ctx.stride,
            ctx.padding,
            ctx.count_include_pad,
            ctx.divisor_override,
            ctx.non_overlapping,
            block_l,
            block_c,
        )
        return (grad_input, *([None] * 14))


class _PackedMaxPool1dFunction(torch.autograd.Function):
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
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
    ) -> Tensor:
        output = input_values.new_empty((output_size, channels))
        save_indices = bool(input_values.requires_grad and int(shape_meta[:, 0].max().item()) < 2**31)
        indices = (
            torch.empty((output_size, channels), device=input_values.device, dtype=torch.int32)
            if save_indices
            else output
        )
        block_l, block_c = _pool_block_size()
        grid = (tile_count, triton.cdiv(channels, block_c))
        _max_pool1d_forward_kernel[grid](
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
            kernel_size,
            stride,
            padding,
            dilation,
            save_indices,
            block_l,
            block_c,
        )
        if save_indices:
            ctx.save_for_backward(indices, input_offsets, output_offsets, shape_meta, tile_offsets, tile_to_batch)
        else:
            ctx.save_for_backward(input_values, input_offsets, output_offsets, shape_meta, tile_offsets, tile_to_batch)
        ctx.use_indices = save_indices
        ctx.tile_count = tile_count
        ctx.batch_count = batch_count
        ctx.input_shape = tuple(int(size) for size in input_values.shape)
        ctx.channels = channels
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_output = grad_output.contiguous()
        block_l, block_c = _pool_block_size()
        grid = (ctx.tile_count, triton.cdiv(ctx.channels, block_c))
        if ctx.use_indices:
            indices, input_offsets, output_offsets, shape_meta, tile_offsets, tile_to_batch = ctx.saved_tensors
            grad_input = grad_output.new_zeros(ctx.input_shape)
            _max_pool1d_backward_indices_kernel[grid](
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
            return (grad_input, *([None] * 13))

        input_values, input_offsets, output_offsets, shape_meta, tile_offsets, tile_to_batch = ctx.saved_tensors
        grad_input = torch.zeros_like(input_values)
        _max_pool1d_backward_kernel[grid](
            input_values,
            grad_output,
            grad_input,
            input_offsets,
            output_offsets,
            shape_meta,
            tile_offsets,
            tile_to_batch,
            ctx.batch_count,
            ctx.channels,
            ctx.kernel_size,
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            block_l,
            block_c,
        )
        return (grad_input, *([None] * 13))


def _packed_avg_pool1d(
    input: NestedTensor,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
) -> NestedTensor | None:
    if ceil_mode or not _can_use_packed_pool1d(input):
        return None
    if not isinstance(kernel_size, int) or (stride is not None and not isinstance(stride, int)):
        return None
    if not isinstance(padding, int):
        return None
    kernel = int(kernel_size)
    stride_single = kernel if stride is None else int(stride)
    padding_single = int(padding)
    divisor = 0 if divisor_override is None else int(divisor_override)
    if kernel <= 0 or stride_single <= 0 or padding_single < 0 or padding_single > kernel // 2:
        return None
    if divisor <= 0 and divisor_override is not None:
        return None
    output_meta = _pool1d_output_meta(input, kernel, stride_single, padding_single)
    if output_meta is None:
        return None
    output_shapes, output_packed_sizes, output_shape_tensor = output_meta
    output_offsets = type(input)._offsets_from_sizes(output_packed_sizes, dtype=torch.long)
    total_out = int(output_offsets[-1].item())
    channels = int(input._values.shape[1])
    tile_counts = _pool1d_tile_counts(output_shapes, _pool_block_size()[0])
    input_offsets, output_offsets_device, shape_meta, tile_offsets, tile_to_batch = _make_pool1d_metadata(
        input,
        output_shapes,
        output_offsets,
        tile_counts,
    )
    output_values = _PackedAvgPool1dFunction.apply(
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
        kernel,
        stride_single,
        padding_single,
        bool(count_include_pad),
        divisor,
    )
    return _from_pool_values(
        input, output_values, output_offsets, output_shape_tensor, output_packed_sizes, output_shapes
    )


def _packed_max_pool1d(
    input: NestedTensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
) -> NestedTensor | None:
    if ceil_mode or not _can_use_packed_pool1d(input):
        return None
    if not isinstance(kernel_size, int) or (stride is not None and not isinstance(stride, int)):
        return None
    if not isinstance(padding, int) or not isinstance(dilation, int):
        return None
    kernel = int(kernel_size)
    stride_single = kernel if stride is None else int(stride)
    padding_single = int(padding)
    dilation_single = int(dilation)
    if kernel <= 0 or stride_single <= 0 or dilation_single <= 0:
        return None
    if padding_single < 0 or padding_single > kernel // 2:
        return None
    output_meta = _pool1d_output_meta(input, kernel, stride_single, padding_single, dilation_single)
    if output_meta is None:
        return None
    output_shapes, output_packed_sizes, output_shape_tensor = output_meta
    output_offsets = type(input)._offsets_from_sizes(output_packed_sizes, dtype=torch.long)
    total_out = int(output_offsets[-1].item())
    channels = int(input._values.shape[1])
    tile_counts = _pool1d_tile_counts(output_shapes, _pool_block_size()[0])
    input_offsets, output_offsets_device, shape_meta, tile_offsets, tile_to_batch = _make_pool1d_metadata(
        input,
        output_shapes,
        output_offsets,
        tile_counts,
    )
    output_values = _PackedMaxPool1dFunction.apply(
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
        kernel,
        stride_single,
        padding_single,
        dilation_single,
    )
    return _from_pool_values(
        input, output_values, output_offsets, output_shape_tensor, output_packed_sizes, output_shapes
    )
