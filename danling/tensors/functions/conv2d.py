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
from torch.utils._python_dispatch import _get_current_dispatch_mode

from ..ops import _check_execution_guard, _ExecutionGuardKind, _is_compiling, _per_element_unsupported
from ._convolution import _has_static_input_channels

if TYPE_CHECKING:
    from ..nested_tensor import NestedTensor


def _has_size_hint(value: int | torch.SymInt) -> bool:
    return not isinstance(value, torch.SymInt) or value.node.hint is not None


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, _fn=None):
    r"""Convolve packed spatial tiles with each sample's independent boundary."""
    if len(input) != 0:
        channel_dim = _packed_pointwise_conv2d_channel_dim(input, weight, stride, padding, dilation, groups)
        if channel_dim is not None:
            return _packed_pointwise_conv2d(input, weight, bias, channel_dim)
        if _can_use_packed_conv2d(input, weight, bias, stride, padding, dilation, groups):
            return _packed_spatial_conv2d(input, weight, bias, stride, padding, dilation, groups)
    return _per_element_conv2d(input, weight, bias, stride, padding, dilation, groups, _fn=_fn)


def _conv2d_pair(value) -> tuple[int, int] | None:
    if isinstance(value, (int, torch.SymInt)) and not isinstance(value, bool):
        return int(value), int(value)
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(item, (int, torch.SymInt)) and not isinstance(item, bool) for item in value)
    ):
        return int(value[0]), int(value[1])
    return None


def _conv2d_parameters(weight, stride, padding, dilation):
    stride = _conv2d_pair(stride)
    dilation = _conv2d_pair(dilation)
    if stride is None or dilation is None or any(value <= 0 for value in (*stride, *dilation)):
        return None
    if isinstance(padding, str):
        if padding == "valid":
            padding_before = padding_after = (0, 0)
        elif padding == "same" and stride == (1, 1):
            total = tuple(dilation[axis] * (int(weight.shape[axis + 2]) - 1) for axis in range(2))
            padding_before = tuple(value // 2 for value in total)
            padding_after = tuple(value - before for value, before in zip(total, padding_before))
        else:
            return None
    else:
        pair = _conv2d_pair(padding)
        if pair is None or any(value < 0 for value in pair):
            return None
        padding_before = padding_after = pair
    return stride, padding_before, padding_after, dilation


def _can_use_packed_conv2d(input, weight, bias, stride, padding, dilation, groups) -> bool:
    r"""Whether static channels and two packed spatial axes define this convolution."""
    if (
        not isinstance(weight, Tensor)
        or weight.dim() != 4
        or not isinstance(groups, (int, torch.SymInt))
        or isinstance(groups, bool)
        or groups <= 0
        or weight.shape[0] <= 0
        or weight.shape[0] % groups != 0
        or any(size <= 0 for size in weight.shape[1:])
        or input._physical_shape.size(0) == 0
        or input._physical_shape.size(1) != 3
        or input._ragged_dims != (1, 2)
        or input._permutation != (1, 2, 0)
    ):
        return False
    values = input._packed_values
    if values.dim() != 2 or values.device != weight.device:
        return False
    if not _has_static_input_channels(input, int(weight.shape[1]) * int(groups), rank=2):
        return False
    if bias is not None and (
        not isinstance(bias, Tensor)
        or bias.dim() != 1
        or bias.shape[0] != weight.shape[0]
        or bias.device != values.device
    ):
        return False
    if _is_compiling() and not all(_has_size_hint(extent) for extent in input._max_physical_dims()[1:]):
        return False
    return _conv2d_parameters(weight, stride, padding, dilation) is not None


def _packed_pointwise_conv2d_channel_dim(input: NestedTensor, weight, stride, padding, dilation, groups) -> int | None:
    r"""Return packed channel dim when ``conv2d`` can run as one packed linear op."""
    if isinstance(groups, bool) or groups != 1 or not isinstance(weight, Tensor) or weight.dim() != 4:
        return None
    if tuple(int(size) for size in weight.shape[2:]) != (1, 1):
        return None
    parameters = _conv2d_parameters(weight, stride, padding, dilation)
    if parameters != ((1, 1), (0, 0), (0, 0), (1, 1)):
        return None
    in_channels = int(weight.shape[1])
    if not _has_static_input_channels(input, in_channels, rank=2):
        return None

    suffix_rank = input.values().dim() - 1
    if suffix_rank <= 0:
        return None
    static_dims = tuple(int(dim) for dim in input._permutation[-suffix_rank:])
    if 0 not in static_dims:
        return None
    channel_dim = 1 + static_dims.index(0)
    if int(input.values().shape[channel_dim]) != in_channels:
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
    values = input.values()
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

    from ..aten_functions import _packed_with_shape

    return _packed_with_shape(
        input,
        output,
        shape_tensor,
        input._logical_shape_from_components(replace_dims={0: out_channels}),
        permutation=input._permutation,
        packed_sizes=input._packed_sizes,
        element_shapes=element_shapes,
        preserve_ragged_offsets=True,
    )


def _per_element_conv2d(input: NestedTensor, *args, _fn=None, **kwargs) -> NestedTensor:
    _check_execution_guard(_ExecutionGuardKind.STORAGE_MAP, "_per_element_conv2d")
    _per_element_unsupported(input, "conv2d")
    cls = type(input)
    if len(input) == 0:
        return cls([], **input._meta(include_dtype=True))
    conv2d = F.conv2d if _fn is None else _fn
    with torch._C.DisableTorchFunctionSubclass():
        results = [conv2d(t, *args, **kwargs) for t in input._storage]
    return cls(results, **input._meta())


def _packed_spatial_conv2d(input, weight, bias, stride, padding, dilation, groups, *, tile_size=(64, 64)):
    r"""Convolve bounded native tiles with a capacity derived from packed input sizes."""
    from ..aten_functions import _is_fake_tensor, _packed_with_shape
    from ..nested_tensor import _layout_scalar

    stride, before, after, dilation = _conv2d_parameters(weight, stride, padding, dilation)
    groups = int(groups)
    kernel_h, kernel_w = int(weight.shape[2]), int(weight.shape[3])
    out_channels = int(weight.shape[0])
    shape = input._physical_shape
    rank_one = shape.shape[1] == 2
    if rank_one:
        spatial = torch.cat((torch.ones_like(shape[:, :1]), shape[:, 1:]), dim=1)
    else:
        spatial = shape[:, 1:]
    torch._assert_async(torch.all(spatial > 0), "conv2d input spatial dimensions must be positive")
    preserves_spatial = stride == (1, 1) and all(
        before[axis] + after[axis] == dilation[axis] * (int(weight.shape[axis + 2]) - 1) for axis in range(2)
    )
    if preserves_spatial:
        output_spatial = spatial
        output_shape, packed_sizes, element_shapes = input._shape_meta_from_components(replace_dims={0: out_channels})
        if rank_one and element_shapes is not None:
            element_shapes = tuple((channels, 1, length) for channels, length in element_shapes)
    else:
        shrink = spatial.new_tensor(
            [before[axis] + after[axis] - dilation[axis] * (int(weight.shape[axis + 2]) - 1) - 1 for axis in range(2)]
        )
        output_spatial = torch.div(spatial + shrink, spatial.new_tensor(stride), rounding_mode="floor") + 1
        torch._assert_async(torch.all(output_spatial > 0), "conv2d output spatial dimensions must be positive")
        output_shape = torch.cat(
            (torch.full_like(shape[:, :1], out_channels), output_spatial[:, 1:] if rank_one else output_spatial), dim=1
        )
        element_shapes = None
        packed_sizes = None
    values = input.values()
    native_autograd = (
        not _is_compiling()
        and (torch._C._are_functorch_transforms_active() or torch.autograd.forward_ad._current_level >= 0)
    ) or (
        values.is_complex()
        and torch.is_grad_enabled()
        and (values.requires_grad or weight.requires_grad or bias is not None and bias.requires_grad)
    )
    if native_autograd:
        # A single real output per tile retains native complex/forward-AD
        # derivatives without ever differentiating discarded output windows.
        tile_size = (1, 1)
    if not preserves_spatial and not _is_compiling() and input._element_shapes is not None:
        # Standalone FakeTensor retains concrete layout schema but cannot read tensor data.
        element_shapes = tuple(
            (
                out_channels,
                *(
                    (
                        extent[axis + 1]
                        + before[axis]
                        + after[axis]
                        - dilation[axis] * (int(weight.shape[axis + 2]) - 1)
                        - 1
                    )
                    // stride[axis]
                    + 1
                    for axis in range(2)
                ),
            )
            for extent in (
                tuple((channels, 1, length) for channels, length in input._element_shapes)
                if rank_one
                else input._element_shapes
            )
        )
        packed_sizes = tuple(extent[1] * extent[2] for extent in element_shapes)
    if preserves_spatial:
        output_offsets = input._offsets
        total_output = values.shape[0]
        ragged_offsets = input._persistent_ragged_offsets()
    else:
        output_counts = output_spatial.prod(1)
        output_offsets = torch.cat((output_counts.new_zeros(1), output_counts.cumsum(0)))
        total_output = sum(packed_sizes) if packed_sizes is not None else _layout_scalar(output_offsets[-1])
        torch._check(total_output >= 1)
        if rank_one:
            ragged_offsets = (output_offsets,)
        else:
            heights = output_spatial[:, 0]
            input_ragged_offsets = input._persistent_ragged_offsets()
            preserves_height = stride[0] == 1 and before[0] + after[0] == dilation[0] * (kernel_h - 1)
            if preserves_height and input_ragged_offsets is not None:
                row_offsets = input_ragged_offsets[0]
                row_split_count = input_ragged_offsets[1].numel()
            else:
                row_offsets = torch.cat((heights.new_zeros(1), heights.cumsum(0)))
                # Bind the returned row-split length itself, rather than an internal size plus one.
                row_split_count = (
                    sum(extent[1] for extent in element_shapes) + 1
                    if element_shapes is not None
                    else _layout_scalar(row_offsets[-1] + 1)
                )
            torch._check(row_split_count >= 2)
            widths = torch.repeat_interleave(output_spatial[:, 1], heights, output_size=row_split_count - 1)
            pixel_offsets = torch.cat((widths.new_zeros(1), widths.cumsum(0)))
            ragged_offsets = (row_offsets, pixel_offsets)

    max_spatial = input._max_physical_dims()
    if rank_one:
        max_spatial = (max_spatial[0], 1, max_spatial[1])
    max_output = tuple(
        (max_spatial[axis + 1] + before[axis] + after[axis] - dilation[axis] * (int(weight.shape[axis + 2]) - 1) - 1)
        // stride[axis]
        + 1
        for axis in range(2)
    )
    torch._check(max_output[0] >= 1)
    torch._check(max_output[1] >= 1)
    tile_h, tile_w = min(tile_size[0], max_output[0]), min(tile_size[1], max_output[1])
    batch_size = shape.shape[0]
    effective_h = dilation[0] * (kernel_h - 1)
    effective_w = dilation[1] * (kernel_w - 1)
    extra_h = max(before[0] + after[0] - effective_h + stride[0] - 1, 0)
    extra_w = max(before[1] + after[1] - effective_w + stride[1] - 1, 0)
    tile_shape = spatial.new_tensor((tile_h, tile_w))
    tile_grid = torch.div(output_spatial + tile_shape - 1, tile_shape, rounding_mode="floor")
    tile_counts = tile_grid.prod(1)
    tile_offsets = torch.cat((tile_counts.new_zeros(1), tile_counts.cumsum(0)))
    grid_bound = batch_size * ((max_output[0] + tile_h - 1) // tile_h) * ((max_output[1] + tile_w - 1) // tile_w)
    if element_shapes is not None:
        capacity = sum(
            ((height + tile_h - 1) // tile_h) * ((width + tile_w - 1) // tile_w) for _, height, width in element_shapes
        )
    elif (
        not _is_compiling()
        and _get_current_dispatch_mode() is None
        and not torch._C._are_functorch_transforms_active()
        and not _is_fake_tensor(tile_offsets)
    ):
        # Real CPU metadata gives the exact tile count without synchronizing CUDA.
        capacity = _layout_scalar(tile_offsets[-1])
    elif _has_size_hint(values.shape[0]):
        area_numerator = (
            values.shape[0]
            + extra_h * batch_size * max_spatial[2]
            + extra_w * batch_size * max_spatial[1]
            + batch_size * extra_h * extra_w
        )
        area_bound = (area_numerator + stride[0] * stride[1] - 1) // (stride[0] * stride[1])
        # The integer tile count is bounded by sum((h+t_h-1)*(w+t_w-1))/(t_h*t_w).
        capacity = (
            area_bound
            + (tile_w - 1) * batch_size * max_output[0]
            + (tile_h - 1) * batch_size * max_output[1]
            + batch_size * (tile_h - 1) * (tile_w - 1)
        ) // (tile_h * tile_w)
        capacity = min(capacity, grid_bound)
    else:
        # A prior ragged operation can make the exact packed count data-dependent.
        # Its logical spatial maxima still provide a backed bound for the same tiles.
        capacity = grid_bound
    if not native_autograd and torch.is_grad_enabled() and (values.requires_grad or weight.requires_grad):
        tile_ids_cpu = torch.arange(capacity, device=spatial.device)
        raw_owner_cpu = torch.bucketize(tile_ids_cpu, tile_offsets[1:], right=True)
        owner_cpu = raw_owner_cpu.clamp(max=batch_size - 1)
        local_cpu = tile_ids_cpu - tile_offsets.index_select(0, owner_cpu)
        columns_cpu = tile_grid[:, 1].index_select(0, owner_cpu)
        row_cpu = torch.div(local_cpu, columns_cpu, rounding_mode="floor")
        output_y_cpu = row_cpu * tile_h
        output_x_cpu = (local_cpu - row_cpu * columns_cpu) * tile_w
        remaining = output_spatial.index_select(0, owner_cpu) - torch.stack((output_y_cpu, output_x_cpu), dim=1)
        output_tile_shapes = torch.minimum(remaining.clamp(min=0), tile_shape)
        output_tile_shapes = torch.where((raw_owner_cpu < batch_size)[:, None], output_tile_shapes, 0)
    else:
        output_tile_shapes = spatial.new_empty((0, 2))
    from . import _conv2d_copy

    input_h = (tile_h - 1) * stride[0] + effective_h + 1
    input_w = (tile_w - 1) * stride[1] + effective_w + 1
    use_copy = _conv2d_copy.available(
        values, stride, dilation, capacity, input_h, input_w, tile_h, tile_w, out_channels
    )
    if use_copy:
        copy_in_offsets, copy_out_offsets, copy_shapes, copy_tiles = _conv2d_copy.make_metadata(
            spatial,
            output_spatial,
            input._offsets,
            output_offsets,
            tile_grid,
            tile_offsets,
            capacity,
            tile_h,
            tile_w,
            values.device,
        )
        tiles = _conv2d_copy.pack(
            values,
            copy_in_offsets,
            copy_shapes,
            copy_tiles,
            input_h,
            input_w,
            before,
            stride,
            (kernel_h, kernel_w),
            dilation,
        )
    else:
        device = values.device
        input_spatial_device = input.element_sizes(device=device)[:, 1:]
        if rank_one:
            input_spatial_device = torch.cat((torch.ones_like(input_spatial_device), input_spatial_device), dim=1)
        tile_offsets_device = tile_offsets.to(device=device)
        tile_grid_device = tile_grid.to(device=device)
        tile_ids = torch.arange(capacity, device=device)
        raw_owner = torch.bucketize(tile_ids, tile_offsets_device[1:], right=True)
        real_tile = raw_owner < batch_size
        tile_owner = raw_owner.clamp(max=batch_size - 1)
        local_tile = tile_ids - tile_offsets_device.index_select(0, tile_owner)
        tile_columns = tile_grid_device[:, 1].index_select(0, tile_owner)
        tile_y = torch.div(local_tile, tile_columns, rounding_mode="floor") * tile_h
        tile_x = local_tile.remainder(tile_columns) * tile_w
        input_h = (tile_h - 1) * stride[0] + effective_h + 1
        input_w = (tile_w - 1) * stride[1] + effective_w + 1
        y = tile_y[:, None, None] * stride[0] + torch.arange(input_h, device=device)[None, :, None] - before[0]
        x = tile_x[:, None, None] * stride[1] + torch.arange(input_w, device=device)[None, None, :] - before[1]
        spatial_device = input_spatial_device.index_select(0, tile_owner)
        valid = (
            real_tile[:, None, None]
            & (y >= 0)
            & (y < spatial_device[:, 0, None, None])
            & (x >= 0)
            & (x < spatial_device[:, 1, None, None])
        )
        if stride != (1, 1) or dilation != (1, 1):
            # Unused activation coordinates must not enter zero-weighted extra output windows.
            output_for_tile = output_spatial.to(device=device).index_select(0, tile_owner)
            candidates_y = (
                y[:, :, 0, None] + before[0] - torch.arange(kernel_h, device=device)[None, None, :] * dilation[0]
            )
            candidates_x = (
                x[:, 0, :, None] + before[1] - torch.arange(kernel_w, device=device)[None, None, :] * dilation[1]
            )
            covered_y = (
                (candidates_y >= 0)
                & (candidates_y.remainder(stride[0]) == 0)
                & (torch.div(candidates_y, stride[0], rounding_mode="floor") < output_for_tile[:, 0, None, None])
            ).any(-1)
            covered_x = (
                (candidates_x >= 0)
                & (candidates_x.remainder(stride[1]) == 0)
                & (torch.div(candidates_x, stride[1], rounding_mode="floor") < output_for_tile[:, 1, None, None])
            ).any(-1)
            valid = valid & covered_y[:, :, None] & covered_x[:, None, :]
        input_offsets = input.packed_offsets(device=device)
        indices = input_offsets.index_select(0, tile_owner)[:, None, None] + y * spatial_device[:, 1, None, None] + x
        flat_valid = valid.reshape(-1)
        distributed = torch.arange(indices.numel(), device=device).remainder(values.shape[0])
        gather_indices = torch.where(flat_valid, indices.reshape(-1), distributed)
        tiles = values.index_select(0, gather_indices)
        tiles = torch.where(flat_valid[:, None], tiles, 0)
        tiles = tiles.reshape(capacity, input_h, input_w, values.shape[1]).permute(0, 3, 1, 2)
        if not values.is_cuda:
            tiles = tiles.contiguous()
    from ._convolution import _tile_convolution

    convolved = _tile_convolution(
        tiles, weight, bias, output_tile_shapes, stride, dilation, groups, native_autograd=native_autograd
    )
    if use_copy:
        output_values = _conv2d_copy.unpack(convolved, copy_out_offsets, copy_shapes, copy_tiles, total_output)
    else:
        convolved = convolved.permute(0, 2, 3, 1).reshape(-1, out_channels)

        if preserves_spatial:
            output_offsets_device = input_offsets
            output_spatial_device = input_spatial_device
            owner = input.packed_batch_indices(device=device)
        else:
            output_offsets_device = output_offsets.to(device=device)
            output_spatial_device = output_spatial.to(device=device)
            owner = torch.repeat_interleave(
                torch.arange(batch_size, device=device), output_counts.to(device=device), output_size=total_output
            )
        local_output = torch.arange(total_output, device=device) - output_offsets_device.index_select(0, owner)
        output_width = output_spatial_device[:, 1].index_select(0, owner)
        output_y = torch.div(local_output, output_width, rounding_mode="floor")
        output_x = local_output.remainder(output_width)
        output_tile = (
            tile_offsets_device.index_select(0, owner)
            + torch.div(output_y, tile_h, rounding_mode="floor") * tile_grid_device[:, 1].index_select(0, owner)
            + torch.div(output_x, tile_w, rounding_mode="floor")
        )
        row = (output_tile * tile_h + output_y.remainder(tile_h)) * tile_w + output_x.remainder(tile_w)
        output_values = convolved.index_select(0, row)
    if rank_one:
        if element_shapes is not None:
            element_shapes = tuple((channels, width) for channels, _, width in element_shapes)
        logical_shape = input._logical_shape_from_components(replace_dims={0: out_channels, 1: max_output[1]})
        permutation, ragged_dims = (1, 0), (1,)
    else:
        logical_shape = input._logical_shape_from_components(
            replace_dims={0: out_channels, 1: max_output[0], 2: max_output[1]}
        )
        permutation, ragged_dims = (1, 2, 0), (1, 2)
    return _packed_with_shape(
        input,
        output_values,
        output_shape,
        logical_shape,
        offsets=output_offsets,
        packed_sizes=packed_sizes,
        element_shapes=element_shapes,
        permutation=permutation,
        force_explicit_ragged_dims=ragged_dims,
        ragged_offsets=ragged_offsets,
    )
