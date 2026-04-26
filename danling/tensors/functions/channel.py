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
r"""Channel and pixel shuffle handlers for NestedTensor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from ..ops import _check_execution_guard, _ExecutionGuardKind

if TYPE_CHECKING:
    from ..nested_tensor import NestedTensor

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency/runtime
    triton = None
    tl = None


# ---------------------------------------------------------------------------
# NestedTensor channel dispatch
# ---------------------------------------------------------------------------


def channel_shuffle(input, groups: int, *, _fn=None):
    output = _channel_shuffle_uniform(input, groups)
    if output is not None:
        return output
    output = _packed_channel_shuffle_height(input, groups)
    if output is not None:
        return output
    output = _channel_shuffle_packed(input, groups)
    if output is not None:
        return output
    return _per_element(input, F.channel_shuffle if _fn is None else _fn, groups)


def pixel_shuffle(input, upscale_factor: int, *, _fn=None):
    output = _packed_pixel2d(input, upscale_factor, shuffle=True)
    if output is not None:
        return output
    values = _uniform_dense_values(input)
    if values is not None:
        output = F.pixel_shuffle(values, upscale_factor)
        return _from_uniform_values(input, output, tuple(int(size) for size in output.shape[1:]))
    return _per_element(input, F.pixel_shuffle if _fn is None else _fn, upscale_factor)


def pixel_unshuffle(input, downscale_factor: int, *, _fn=None):
    output = _packed_pixel2d(input, downscale_factor, shuffle=False)
    if output is not None:
        return output
    values = _uniform_dense_values(input)
    if values is not None:
        output = F.pixel_unshuffle(values, downscale_factor)
        return _from_uniform_values(input, output, tuple(int(size) for size in output.shape[1:]))
    return _per_element(input, F.pixel_unshuffle if _fn is None else _fn, downscale_factor)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _uniform_element_shape(input: NestedTensor) -> tuple[int, ...] | None:
    rank = input._physical_shape.size(1)
    if tuple(int(dim) for dim in input._permutation) != tuple(range(rank)):
        return None
    if not input._element_shapes or len(set(input._element_shapes)) != 1:
        return None
    shape = tuple(int(size) for size in input._element_shapes[0])
    expected = (len(input) * shape[0], *shape[1:]) if shape else (len(input),)
    if tuple(int(size) for size in input._values.shape) != expected:
        return None
    return shape


def _uniform_dense_values(input: NestedTensor) -> torch.Tensor | None:
    shape = _uniform_element_shape(input)
    if shape is None:
        return None
    return input._values.reshape(len(input), *shape)


def _uniform_shape_tensor(input: NestedTensor, element_shape: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor([element_shape] * len(input), dtype=torch.long)


def _from_uniform_values(input: NestedTensor, values: torch.Tensor, element_shape: tuple[int, ...]) -> NestedTensor:
    packed_size = int(element_shape[0]) if element_shape else 1
    packed_values = values.reshape(len(input) * packed_size, *element_shape[1:])
    return type(input)._from_packed(
        packed_values,
        type(input)._offsets_from_sizes((packed_size,) * len(input), dtype=torch.long),
        _uniform_shape_tensor(input, element_shape),
        permutation=tuple(range(len(element_shape))),
        batch_first=input.batch_first,
        padding_value=input.padding_value,
        mask_value=input.mask_value,
        pin_memory=input._pin_memory,
        packed_sizes=(packed_size,) * len(input),
        element_shapes=tuple(element_shape for _ in range(len(input))),
    )


def _per_element(input: NestedTensor, fn, *args, **kwargs) -> NestedTensor:
    _check_execution_guard(_ExecutionGuardKind.STORAGE_MAP, f"{fn.__name__}_per_element")
    cls = type(input)
    if len(input) == 0:
        return cls([], **input._meta(include_dtype=True))
    with torch._C.DisableTorchFunctionSubclass():
        return cls([fn(t, *args, **kwargs) for t in input._storage], **input._meta())


def _packed_channel_dim_for_physical_dim(input: NestedTensor, physical_dim: int) -> int | None:
    suffix_rank = input._values.dim() - 1
    if suffix_rank <= 0:
        return None
    static_dims = tuple(int(dim) for dim in input._permutation[-suffix_rank:])
    if physical_dim not in static_dims:
        return None
    return 1 + static_dims.index(physical_dim)


def _channel_shuffle_uniform(input: NestedTensor, groups: int) -> NestedTensor | None:
    values = _uniform_dense_values(input)
    if values is None:
        return None
    rank = input._physical_shape.size(1)
    if rank < 2:
        return None

    leading = values.shape[:2]
    tail = values.shape[2:]
    shuffled = F.channel_shuffle(values.reshape(int(leading[0] * leading[1]), *tail), groups)
    output = shuffled.reshape(*leading, *shuffled.shape[1:])
    return _from_uniform_values(input, output, tuple(int(size) for size in output.shape[1:]))


def _channel_shuffle_packed(input: NestedTensor, groups: int) -> NestedTensor | None:
    if groups <= 0:
        return None
    if input._physical_shape.size(1) < 2:
        return None
    if input._element_shapes is not None and any(len(shape) < 2 for shape in input._element_shapes):
        return None

    channel_dim = _packed_channel_dim_for_physical_dim(input, 1)
    if channel_dim is None:
        return None
    values = input._values
    moved = channel_dim != 1
    if moved:
        values = values.movedim(channel_dim, 1)
    channels = int(values.shape[1])
    if channels % groups != 0:
        return None
    output = values.reshape(values.shape[0], groups, channels // groups, *values.shape[2:])
    output = output.transpose(1, 2).reshape_as(values)
    if moved:
        output = output.movedim(1, channel_dim).contiguous()

    return type(input)._from_packed(
        output,
        input._offsets,
        input._physical_shape.clone(),
        permutation=input._permutation,
        batch_first=input.batch_first,
        padding_value=input.padding_value,
        mask_value=input.mask_value,
        pin_memory=input._pin_memory,
        packed_sizes=input._packed_sizes,
        element_shapes=input._element_shapes,
    )


def _can_use_packed_channel_shuffle_height(input: NestedTensor, groups: int) -> bool:
    if triton is None or len(input) == 0 or not input._values.is_cuda or groups <= 0:
        return False
    if input._physical_shape.size(1) != 3 or input._values.dim() != 2:
        return False
    if input._element_shapes is not None and any(len(shape) != 3 for shape in input._element_shapes):
        return False
    if tuple(int(dim) for dim in input._permutation) != (1, 2, 0):
        return False
    channels = int(input._values.shape[1])
    if not bool(torch.equal(input._physical_shape[:, 0], torch.full_like(input._physical_shape[:, 0], channels))):
        return False
    return bool(torch.all(input._physical_shape[:, 1].remainder(groups) == 0))


def _channel_shuffle_block_size() -> tuple[int, int]:
    return 128, 64


if triton is not None:

    @triton.jit
    def _channel_shuffle_height_kernel(
        input_ptr,
        output_ptr,
        offsets_ptr,
        shape_ptr,
        groups: tl.constexpr,
        channels: tl.constexpr,
        block_l: tl.constexpr,
        block_c: tl.constexpr,
    ):
        tile = tl.program_id(0)
        channel_block = tl.program_id(1)
        batch = tl.program_id(2)
        out_start = tile * block_l
        height = tl.load(shape_ptr + batch * 3 + 1)
        width = tl.load(shape_ptr + batch * 3 + 2)
        size = height * width
        base = tl.load(offsets_ptr + batch)
        offsets_l = out_start + tl.arange(0, block_l)
        offsets_c = channel_block * block_c + tl.arange(0, block_c)
        out_y = offsets_l // width
        out_x = offsets_l - out_y * width
        channels_per_group = height // groups
        in_y = (out_y % groups) * channels_per_group + out_y // groups
        values = tl.load(
            input_ptr + (base + in_y[:, None] * width + out_x[:, None]) * channels + offsets_c[None, :],
            mask=(offsets_l[:, None] < size) & (offsets_c[None, :] < channels),
            other=0.0,
        )
        tl.store(
            output_ptr + (base + offsets_l[:, None]) * channels + offsets_c[None, :],
            values,
            mask=(offsets_l[:, None] < size) & (offsets_c[None, :] < channels),
        )

    @triton.jit
    def _channel_unshuffle_height_kernel(
        input_ptr,
        output_ptr,
        offsets_ptr,
        shape_ptr,
        groups: tl.constexpr,
        channels: tl.constexpr,
        block_l: tl.constexpr,
        block_c: tl.constexpr,
    ):
        tile = tl.program_id(0)
        channel_block = tl.program_id(1)
        batch = tl.program_id(2)
        in_start = tile * block_l
        height = tl.load(shape_ptr + batch * 3 + 1)
        width = tl.load(shape_ptr + batch * 3 + 2)
        size = height * width
        base = tl.load(offsets_ptr + batch)
        offsets_l = in_start + tl.arange(0, block_l)
        offsets_c = channel_block * block_c + tl.arange(0, block_c)
        in_y = offsets_l // width
        in_x = offsets_l - in_y * width
        channels_per_group = height // groups
        group = in_y // channels_per_group
        index_in_group = in_y - group * channels_per_group
        out_y = index_in_group * groups + group
        values = tl.load(
            input_ptr + (base + out_y[:, None] * width + in_x[:, None]) * channels + offsets_c[None, :],
            mask=(offsets_l[:, None] < size) & (offsets_c[None, :] < channels),
            other=0.0,
        )
        tl.store(
            output_ptr + (base + offsets_l[:, None]) * channels + offsets_c[None, :],
            values,
            mask=(offsets_l[:, None] < size) & (offsets_c[None, :] < channels),
        )


class _PackedChannelShuffleHeightFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: Tensor,
        offsets: Tensor,
        shape: Tensor,
        groups: int,
        channels: int,
        batch: int,
        max_hw: int,
        block_l: int,
        block_c: int,
    ) -> Tensor:
        output = torch.empty_like(values)
        grid = (triton.cdiv(max_hw, block_l), triton.cdiv(channels, block_c), batch)
        _channel_shuffle_height_kernel[grid](values, output, offsets, shape, groups, channels, block_l, block_c)

        ctx.save_for_backward(offsets, shape)
        ctx.groups = groups
        ctx.channels = channels
        ctx.batch = batch
        ctx.max_hw = max_hw
        ctx.block_l = block_l
        ctx.block_c = block_c
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        offsets, shape = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(grad_output)
        grid = (triton.cdiv(ctx.max_hw, ctx.block_l), triton.cdiv(ctx.channels, ctx.block_c), ctx.batch)
        _channel_unshuffle_height_kernel[grid](
            grad_output,
            grad_input,
            offsets,
            shape,
            ctx.groups,
            ctx.channels,
            ctx.block_l,
            ctx.block_c,
        )
        return grad_input, None, None, None, None, None, None, None, None


def _packed_channel_shuffle_height(input: NestedTensor, groups: int) -> NestedTensor | None:
    if not _can_use_packed_channel_shuffle_height(input, groups):
        return None
    input_shapes = _resolve_element_shapes(input)
    channels = int(input._values.shape[1])
    block_l, block_c = _channel_shuffle_block_size()
    device = input._values.device
    output_values = _PackedChannelShuffleHeightFunction.apply(
        input._values,
        input._offsets.to(device=device, non_blocking=True),
        input._physical_shape.to(device=device, non_blocking=True),
        int(groups),
        channels,
        len(input_shapes),
        max(int(height) * int(width) for _, height, width in input_shapes),
        block_l,
        block_c,
    )
    return type(input)._from_packed(
        output_values,
        input._offsets,
        input._physical_shape.clone(),
        permutation=input._permutation,
        batch_first=input.batch_first,
        padding_value=input.padding_value,
        mask_value=input.mask_value,
        pin_memory=input._pin_memory,
        packed_sizes=input._packed_sizes,
        element_shapes=input._element_shapes,
    )


# ---------------------------------------------------------------------------
# Packed pixel shuffle implementation
# ---------------------------------------------------------------------------


def _can_use_packed_pixel2d(input: NestedTensor, channels: int) -> bool:
    if triton is None or len(input) == 0 or not input._values.is_cuda:
        return False
    if input._physical_shape.size(1) != 3:
        return False
    if input._element_shapes is not None and any(len(shape) != 3 for shape in input._element_shapes):
        return False
    if tuple(int(dim) for dim in input._permutation) != (1, 2, 0) or input._values.dim() != 2:
        return False
    if int(input._values.shape[1]) != channels:
        return False
    return bool(torch.equal(input._physical_shape[:, 0], torch.full_like(input._physical_shape[:, 0], channels)))


def _resolve_element_shapes(input: NestedTensor) -> tuple[tuple[int, ...], ...]:
    if input._element_shapes is not None:
        return input._element_shapes
    return tuple(type(input)._trim_shape(row) for row in input._physical_shape.tolist())


def _pixel_output_meta(
    input: NestedTensor,
    factor: int,
    *,
    shuffle: bool,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], Tensor] | None:
    if factor <= 0:
        return None
    factor2 = factor * factor
    output_shapes = []
    output_packed_sizes = []
    for shape in _resolve_element_shapes(input):
        if len(shape) != 3:
            return None
        channels, height, width = shape
        if shuffle:
            if channels % factor2 != 0:
                return None
            output_shape = (channels // factor2, height * factor, width * factor)
        else:
            if height % factor != 0 or width % factor != 0:
                return None
            output_shape = (channels * factor2, height // factor, width // factor)
        output_shapes.append(output_shape)
        output_packed_sizes.append(output_shape[1] * output_shape[2])
    if not output_shapes:
        return None
    return tuple(output_shapes), tuple(output_packed_sizes), torch.tensor(output_shapes, dtype=torch.long)


def _pixel_shape_meta(
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
    *,
    device: torch.device,
) -> Tensor:
    return torch.tensor(
        [(in_h, in_w, out_h, out_w) for (_, in_h, in_w), (_, out_h, out_w) in zip(input_shapes, output_shapes)],
        device=device,
        dtype=torch.int64,
    )


def _pixel_block_size() -> tuple[int, int]:
    return 128, 32


if triton is not None:

    @triton.jit
    def _pixel_shuffle_batch_grid_kernel(
        input_ptr,
        output_ptr,
        input_offsets_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        in_channels: tl.constexpr,
        out_channels: tl.constexpr,
        factor: tl.constexpr,
        block_l: tl.constexpr,
        block_c: tl.constexpr,
    ):
        tile = tl.program_id(0)
        channel_block = tl.program_id(1)
        batch = tl.program_id(2)
        out_start = tile * block_l
        in_w = tl.load(shape_meta_ptr + batch * 4 + 1)
        out_h = tl.load(shape_meta_ptr + batch * 4 + 2)
        out_w = tl.load(shape_meta_ptr + batch * 4 + 3)
        out_hw = out_h * out_w
        input_base = tl.load(input_offsets_ptr + batch)
        output_base = tl.load(output_offsets_ptr + batch)
        offsets_l = tl.arange(0, block_l)
        offsets_c = channel_block * block_c + tl.arange(0, block_c)
        out_flat = out_start + offsets_l
        out_y = out_flat // out_w
        out_x = out_flat - out_y * out_w
        in_y = out_y // factor
        in_x = out_x // factor
        inner_y = out_y - in_y * factor
        inner_x = out_x - in_x * factor
        in_c = offsets_c * factor * factor + inner_y[:, None] * factor + inner_x[:, None]
        valid = out_flat < out_hw
        values = tl.load(
            input_ptr + (input_base + in_y[:, None] * in_w + in_x[:, None]) * in_channels + in_c,
            mask=valid[:, None] & (offsets_c[None, :] < out_channels),
            other=0.0,
        )
        tl.store(
            output_ptr + (output_base + out_flat[:, None]) * out_channels + offsets_c[None, :],
            values,
            mask=valid[:, None] & (offsets_c[None, :] < out_channels),
        )

    @triton.jit
    def _pixel_unshuffle_batch_grid_kernel(
        input_ptr,
        output_ptr,
        input_offsets_ptr,
        output_offsets_ptr,
        shape_meta_ptr,
        in_channels: tl.constexpr,
        out_channels: tl.constexpr,
        factor: tl.constexpr,
        block_l: tl.constexpr,
        block_c: tl.constexpr,
    ):
        tile = tl.program_id(0)
        channel_block = tl.program_id(1)
        batch = tl.program_id(2)
        out_start = tile * block_l
        in_w = tl.load(shape_meta_ptr + batch * 4 + 1)
        out_h = tl.load(shape_meta_ptr + batch * 4 + 2)
        out_w = tl.load(shape_meta_ptr + batch * 4 + 3)
        out_hw = out_h * out_w
        input_base = tl.load(input_offsets_ptr + batch)
        output_base = tl.load(output_offsets_ptr + batch)
        offsets_l = tl.arange(0, block_l)
        offsets_c = channel_block * block_c + tl.arange(0, block_c)
        out_flat = out_start + offsets_l
        out_y = out_flat // out_w
        out_x = out_flat - out_y * out_w
        in_c = offsets_c // (factor * factor)
        inner = offsets_c - in_c * factor * factor
        inner_y = inner // factor
        inner_x = inner - inner_y * factor
        in_y = out_y[:, None] * factor + inner_y[None, :]
        in_x = out_x[:, None] * factor + inner_x[None, :]
        valid = out_flat < out_hw
        values = tl.load(
            input_ptr + (input_base + in_y * in_w + in_x) * in_channels + in_c[None, :],
            mask=valid[:, None] & (offsets_c[None, :] < out_channels),
            other=0.0,
        )
        tl.store(
            output_ptr + (output_base + out_flat[:, None]) * out_channels + offsets_c[None, :],
            values,
            mask=valid[:, None] & (offsets_c[None, :] < out_channels),
        )


class _PackedPixelShuffleFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: Tensor,
        input_offsets: Tensor,
        output_offsets: Tensor,
        shape_meta: Tensor,
        inverse_shape_meta: Tensor,
        factor: int,
        shuffle: bool,
        in_channels: int,
        out_channels: int,
        input_total: int,
        output_total: int,
        batch: int,
        max_input_hw: int,
        max_output_hw: int,
        block_l: int,
        block_c: int,
    ) -> Tensor:
        output = values.new_empty((output_total, out_channels))
        grid = (triton.cdiv(max_output_hw, block_l), triton.cdiv(out_channels, block_c), batch)
        if shuffle:
            _pixel_shuffle_batch_grid_kernel[grid](
                values,
                output,
                input_offsets,
                output_offsets,
                shape_meta,
                in_channels,
                out_channels,
                factor,
                block_l,
                block_c,
            )
        else:
            _pixel_unshuffle_batch_grid_kernel[grid](
                values,
                output,
                input_offsets,
                output_offsets,
                shape_meta,
                in_channels,
                out_channels,
                factor,
                block_l,
                block_c,
            )

        ctx.save_for_backward(input_offsets, output_offsets, inverse_shape_meta)
        ctx.factor = factor
        ctx.shuffle = shuffle
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels
        ctx.input_total = input_total
        ctx.batch = batch
        ctx.max_input_hw = max_input_hw
        ctx.block_l = block_l
        ctx.block_c = block_c
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input_offsets, output_offsets, inverse_shape_meta = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = grad_output.new_empty((ctx.input_total, ctx.in_channels))
        grid = (triton.cdiv(ctx.max_input_hw, ctx.block_l), triton.cdiv(ctx.in_channels, ctx.block_c), ctx.batch)
        if ctx.shuffle:
            _pixel_unshuffle_batch_grid_kernel[grid](
                grad_output,
                grad_input,
                output_offsets,
                input_offsets,
                inverse_shape_meta,
                ctx.out_channels,
                ctx.in_channels,
                ctx.factor,
                ctx.block_l,
                ctx.block_c,
            )
        else:
            _pixel_shuffle_batch_grid_kernel[grid](
                grad_output,
                grad_input,
                output_offsets,
                input_offsets,
                inverse_shape_meta,
                ctx.out_channels,
                ctx.in_channels,
                ctx.factor,
                ctx.block_l,
                ctx.block_c,
            )
        return grad_input, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def _packed_pixel2d(input: NestedTensor, factor: int, *, shuffle: bool) -> NestedTensor | None:
    if factor <= 0:
        return None
    if input._values.dim() < 2:
        return None
    in_channels = int(input._values.shape[1])
    if shuffle and in_channels % (factor * factor) != 0:
        return None
    if not _can_use_packed_pixel2d(input, in_channels):
        return None
    meta = _pixel_output_meta(input, factor, shuffle=shuffle)
    if meta is None:
        return None

    output_shapes, output_packed_sizes, output_shape_tensor = meta
    input_shapes = _resolve_element_shapes(input)
    out_channels = int(output_shapes[0][0])
    device = input._values.device
    input_offsets = input._offsets.to(device=device, non_blocking=True)
    output_offsets = type(input)._offsets_from_sizes(output_packed_sizes, dtype=torch.long)
    output_offsets_device = output_offsets.to(device=device, non_blocking=True)
    shape_meta = _pixel_shape_meta(input_shapes, output_shapes, device=device)
    inverse_shape_meta = _pixel_shape_meta(output_shapes, input_shapes, device=device)
    block_l, block_c = _pixel_block_size()
    output_values = _PackedPixelShuffleFunction.apply(
        input._values,
        input_offsets,
        output_offsets_device,
        shape_meta,
        inverse_shape_meta,
        factor,
        shuffle,
        in_channels,
        out_channels,
        int(input._values.shape[0]),
        int(output_offsets[-1]),
        len(output_shapes),
        max(height * width for _, height, width in input_shapes),
        max(height * width for _, height, width in output_shapes),
        block_l,
        block_c,
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
