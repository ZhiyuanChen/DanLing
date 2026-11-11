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
r"""``conv1d`` handlers for NestedTensor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from ..ops import _check_execution_guard, _ExecutionGuardKind, _is_compiling, _per_element_unsupported
from ._convolution import _has_static_input_channels

if TYPE_CHECKING:
    from ..nested_tensor import NestedTensor


def _unwrap_single_conv_arg(value):
    if isinstance(value, (tuple, list)) and len(value) == 1:
        return value[0]
    return value


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, _fn=None):
    r"""Run pointwise or spatial convolutions on packed sample positions."""
    stride = _unwrap_single_conv_arg(stride)
    padding = _unwrap_single_conv_arg(padding)
    dilation = _unwrap_single_conv_arg(dilation)
    if len(input) != 0:
        channel_dim = _packed_pointwise_conv1d_channel_dim(input, weight, stride, padding, dilation, groups)
        if channel_dim is not None:
            return _packed_pointwise_conv1d(input, weight, bias, channel_dim, stride)
        if _can_use_packed_conv1d(input, weight, bias, stride, padding, dilation, groups):
            if stride == 1 and weight.shape[1] == 1 and groups == input._packed_values.shape[1]:
                return _packed_depthwise_conv1d(input, weight, bias, padding, dilation, groups)
            return _packed_spatial_conv1d(input, weight, bias, stride, padding, dilation, groups)
    return _per_element_conv1d(input, weight, bias, stride, padding, dilation, groups, _fn=_fn)


def _conv1d_padding(padding, kernel_size, stride, dilation) -> tuple[int, int] | None:
    if isinstance(padding, str):
        if padding == "valid":
            return 0, 0
        if padding != "same" or stride != 1:
            return None
        total = dilation * (kernel_size - 1)
        return total // 2, total - total // 2
    if isinstance(padding, bool) or not isinstance(padding, (int, torch.SymInt)) or padding < 0:
        return None
    return padding, padding


def _can_use_packed_conv1d(input, weight, bias, stride, padding, dilation, groups) -> bool:
    r"""Whether the input has one packed spatial axis and a static channel axis."""
    stride = _unwrap_single_conv_arg(stride)
    padding = _unwrap_single_conv_arg(padding)
    dilation = _unwrap_single_conv_arg(dilation)
    if (
        isinstance(stride, bool)
        or not isinstance(stride, (int, torch.SymInt))
        or stride <= 0
        or isinstance(dilation, bool)
        or not isinstance(dilation, (int, torch.SymInt))
        or dilation <= 0
        or isinstance(groups, bool)
        or not isinstance(groups, (int, torch.SymInt))
        or groups <= 0
        or not isinstance(weight, Tensor)
        or weight.dim() != 3
        or weight.shape[1] <= 0
        or weight.shape[2] <= 0
        or weight.shape[0] <= 0
        or weight.shape[0] % groups != 0
        or _conv1d_padding(padding, weight.shape[2], stride, dilation) is None
    ):
        return False
    if (
        input._physical_shape.size(0) == 0
        or input._physical_shape.size(1) != 2
        or input._ragged_dims != (1,)
        or input._permutation != (1, 0)
    ):
        return False
    if not _has_static_input_channels(input, int(weight.shape[1]) * int(groups), rank=1):
        return False
    values = input._packed_values
    if values.dim() != 2 or values.device != weight.device:
        return False
    if bias is not None and (
        not isinstance(bias, Tensor)
        or bias.dim() != 1
        or bias.shape[0] != weight.shape[0]
        or bias.device != values.device
    ):
        return False
    return True


def _packed_spatial_conv1d(input: NestedTensor, weight: Tensor, bias: Tensor | None, stride, padding, dilation, groups):
    r"""Use height-one spatial tiles and the shared native convolution engine."""
    from .conv2d import _packed_spatial_conv2d

    return _packed_spatial_conv2d(
        input,
        weight.unsqueeze(2),
        bias,
        stride=(1, stride),
        padding=padding if isinstance(padding, str) else (0, padding),
        dilation=(1, dilation),
        groups=groups,
        tile_size=(1, 1024),
    )


def _packed_depthwise_conv1d(input: NestedTensor, weight: Tensor, bias: Tensor | None, padding, dilation, groups):
    r"""Convolve only real stride-one depthwise output windows."""
    from ..aten_functions import _packed_with_shape

    padding_pair = _conv1d_padding(padding, weight.shape[2], 1, dilation)
    assert padding_pair is not None
    left, right = padding_pair
    values = input.concat
    channels = values.shape[1]
    out_channels = weight.shape[0]
    batch_size = len(input)
    kernel_extent = dilation * (weight.shape[2] - 1) + 1
    input_lengths = input._offsets[1:] - input._offsets[:-1]
    output_lengths = input_lengths + left + right - kernel_extent + 1
    torch._assert_async(torch.all(input_lengths > 0), "conv1d requires a positive input length for every sample")
    torch._assert_async(torch.all(output_lengths > 0), "conv1d kernel size cannot exceed the padded input length")
    shift = left + right - kernel_extent + 1
    same_lengths = shift == 0
    cpu_output_offsets = (
        input._offsets
        if same_lengths
        else input._offsets + torch.arange(batch_size + 1, dtype=input._offsets.dtype) * shift
    )
    packed_sizes = None
    element_shapes = None
    if not _is_compiling() and input._element_shapes is not None:
        # Standalone FakeTensor has no data-derived scalar sizes. Its validated
        # shape cache still describes exactly the same output geometry.
        packed_sizes = tuple(shape[1] + shift for shape in input._element_shapes)
        element_shapes = tuple((out_channels, length) for length in packed_sizes)
    total = values.shape[0] + batch_size * shift

    device_type = values.device.type
    if not values.is_meta and torch.is_autocast_enabled(device_type):
        dtype = torch.get_autocast_dtype(device_type)
        eligible = (torch.float16, torch.bfloat16, torch.float32)
        values = values.to(dtype) if values.dtype in eligible else values
        weight = weight.to(dtype) if weight.dtype in eligible else weight
        if bias is not None and bias.dtype in eligible:
            bias = bias.to(dtype)
    if values.dtype != weight.dtype or (bias is not None and bias.dtype != values.dtype):
        raise RuntimeError("conv1d input, weight and bias must have the same dtype")
    if values.dtype == torch.bool or (values.is_cuda and not (values.is_floating_point() or values.is_complex())):
        raise RuntimeError(f"conv1d is not implemented for dtype {values.dtype}")

    offsets = input.packed_offsets(device=values.device)
    output_offsets = cpu_output_offsets.to(device=values.device)
    positions = torch.arange(total, device=values.device)
    batch = torch.bucketize(positions, output_offsets[1:], right=True)
    local = positions - output_offsets.index_select(0, batch)
    coordinates = local.unsqueeze(1) + torch.arange(weight.shape[2], device=values.device) * dilation - left
    lengths = (offsets[1:] - offsets[:-1]).index_select(0, batch).unsqueeze(1)
    valid = (coordinates >= 0) & (coordinates < lengths)
    rows = torch.where(valid, offsets.index_select(0, batch).unsqueeze(1) + coordinates, values.shape[0])

    # Only real output windows enter the dot products. Padding uses an independent
    # zero row, so its adjoint cannot flow into a neighbouring sample.
    extended = torch.cat((values, values.new_zeros((1, channels))))
    multiplier = out_channels // channels
    accumulation = torch.float32 if values.dtype in (torch.float16, torch.bfloat16) else values.dtype
    factors = weight.to(accumulation).reshape(channels, multiplier, weight.shape[2]).permute(2, 0, 1)
    output = values.new_zeros((), dtype=accumulation)
    for tap in range(weight.shape[2]):
        window = extended.index_select(0, rows[:, tap]).unsqueeze(-1)
        output = torch.addcmul(output, window, factors[tap].unsqueeze(0))
    if bias is not None:
        output = output + bias.to(accumulation).reshape(channels, multiplier)
    output = output.reshape(total, out_channels).to(values.dtype)

    physical_shape = input._physical_shape.clone()
    physical_shape[:, 0] = out_channels
    physical_shape[:, 1] = output_lengths
    max_length = input._max_physical_dims()[1] + shift
    return _packed_with_shape(
        input,
        output,
        physical_shape,
        input._logical_shape_from_components(replace_dims={0: out_channels, 1: max_length}),
        offsets=cpu_output_offsets,
        permutation=(1, 0),
        force_explicit_ragged_dims=(1,),
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
        preserve_ragged_offsets=same_lengths,
    )


def _packed_pointwise_conv1d_channel_dim(input: NestedTensor, weight, stride, padding, dilation, groups) -> int | None:
    r"""Return packed channel dim when ``conv1d`` can run as one packed linear op."""
    if any(isinstance(value, bool) for value in (stride, padding, dilation, groups)):
        return None
    if groups != 1 or not isinstance(weight, Tensor) or weight.dim() != 3:
        return None
    if int(weight.shape[2]) != 1:
        return None
    if not isinstance(stride, (int, torch.SymInt)) or stride <= 0 or dilation != 1:
        return None
    if stride != 1 and (input._ragged_dims != (1,) or input._permutation != (1, 0)):
        return None
    if _conv1d_padding(padding, 1, stride, dilation) != (0, 0):
        return None
    in_channels = int(weight.shape[1])
    if not _has_static_input_channels(input, in_channels, rank=1):
        return None

    suffix_rank = input.concat.dim() - 1
    if suffix_rank <= 0:
        return None
    static_dims = tuple(int(dim) for dim in input._permutation[-suffix_rank:])
    if 0 not in static_dims:
        return None
    channel_dim = 1 + static_dims.index(0)
    if int(input.concat.shape[channel_dim]) != in_channels:
        return None
    return channel_dim


def _packed_pointwise_conv1d(
    input: NestedTensor,
    weight: Tensor,
    bias: Tensor | None,
    channel_dim: int,
    stride=1,
) -> NestedTensor:
    r"""Run 1x1 conv1d over all packed valid positions with one dense linear op."""
    out_channels = int(weight.shape[0])
    values = input.concat
    offsets = input._offsets
    packed_sizes = input._packed_sizes
    element_shapes: tuple[tuple[int, ...], ...] | None = None
    replace_dims = {0: out_channels}
    shape_tensor = input._physical_shape.clone()
    shape_tensor[:, 0] = out_channels
    torch._assert_async(torch.all(shape_tensor[:, 1] > 0), "conv1d requires a positive input length for every sample")
    if stride != 1:
        from ..nested_tensor import _layout_scalar

        lengths = torch.div(shape_tensor[:, 1] - 1, stride, rounding_mode="floor") + 1
        shape_tensor[:, 1] = lengths
        offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        packed_sizes = None
        if not _is_compiling() and input._element_shapes is not None:
            packed_sizes = tuple((shape[1] - 1) // stride + 1 for shape in input._element_shapes)
            element_shapes = tuple((out_channels, length) for length in packed_sizes)
        max_length = (input._max_physical_dims()[1] - 1) // stride + 1
        if len(input) == 1:
            total = max_length
        else:
            total = sum(packed_sizes) if packed_sizes is not None else _layout_scalar(offsets[-1])
        torch._check(total >= len(input))
        output_offsets = offsets.to(device=values.device)
        positions = torch.arange(total, device=values.device)
        owner = torch.bucketize(positions, output_offsets[1:], right=True)
        rows = (
            input.packed_offsets(device=values.device).index_select(0, owner)
            + (positions - output_offsets.index_select(0, owner)) * stride
        )
        values = values.index_select(0, rows)
        replace_dims[1] = max_length
    elif input._element_shapes is not None:
        element_shapes = tuple((out_channels, *shape[1:]) for shape in input._element_shapes)

    moved = channel_dim != values.dim() - 1
    if moved:
        values = values.movedim(channel_dim, -1)
    output = F.linear(values.reshape(-1, int(weight.shape[1])), weight[:, :, 0], bias)
    output = output.reshape(*values.shape[:-1], out_channels)
    if moved:
        output = output.movedim(-1, channel_dim).contiguous()

    from ..aten_functions import _packed_with_shape

    return _packed_with_shape(
        input,
        output,
        shape_tensor,
        input._logical_shape_from_components(replace_dims=replace_dims),
        offsets=offsets,
        permutation=input._permutation,
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
        ragged_offsets=None if stride == 1 else (offsets,),
        preserve_ragged_offsets=stride == 1,
    )


def _per_element_conv1d(input: NestedTensor, *args, _fn=None, **kwargs) -> NestedTensor:
    _check_execution_guard(_ExecutionGuardKind.STORAGE_MAP, "_per_element_conv1d")
    _per_element_unsupported(input, "conv1d")
    cls = type(input)
    if len(input) == 0:
        return cls([], **input._meta(include_dtype=True))
    conv1d = F.conv1d if _fn is None else _fn
    with torch._C.DisableTorchFunctionSubclass():
        results = [conv1d(t, *args, **kwargs) for t in input._storage]
    return cls(results, **input._meta())
