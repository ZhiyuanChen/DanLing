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

r"""Packed segmented primitives.

Operators that act along a ragged dimension have historically materialized a dense padded
tensor, paying ``O(B x max_len)`` memory to run a dense kernel and mask the padding back
out. These primitives do the same work on the packed values directly.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


@torch.library.custom_op("danling::_inferred_one_hot", mutates_args=(), schema="(Tensor values) -> Tensor")
def inferred_one_hot(values: Tensor) -> Tensor:
    """Infer the categorical width at runtime without reading tensor data in Dynamo."""
    return F.one_hot(values)


@inferred_one_hot.register_fake
def _inferred_one_hot_fake(values: Tensor) -> Tensor:
    classes = torch.library.get_ctx().new_dynamic_size(min=1)
    return values.new_empty((*values.shape, classes))


def _unpack_sample(values: Tensor, shape: list[int], permutation: list[int]) -> Tensor:
    inverse = [permutation.index(dim) for dim in range(len(shape))]
    return values.reshape([shape[dim] for dim in permutation]).permute(inverse)


def _pack_sample(values: Tensor, permutation: list[int], ragged_rank: int) -> Tensor:
    ordered = values.permute(permutation)
    leading = 1
    for dim in range(ragged_rank):
        leading *= ordered.shape[dim]
    return ordered.reshape(leading, *ordered.shape[ragged_rank:])


@torch.library.custom_op(
    "danling::_segmented_pdist",
    mutates_args=(),
    schema="(Tensor values, Tensor offsets, Tensor shapes, SymInt[] permutation, float p) -> Tensor",
)
def segmented_pdist(values: Tensor, offsets: Tensor, shapes: Tensor, permutation: list[int], p: float) -> Tensor:
    """Preserve native pdist's condensed ordering and per-sample matrix widths."""
    splits = offsets.tolist()
    outputs = [
        F.pdist(_unpack_sample(values[start:end], shape, permutation).contiguous(), p=p)
        for start, end, shape in zip(splits[:-1], splits[1:], shapes.tolist())
    ]
    return torch.cat(outputs) if outputs else values.new_empty((0,))


@segmented_pdist.register_fake
def _segmented_pdist_fake(values: Tensor, offsets: Tensor, shapes: Tensor, permutation: list[int], p: float) -> Tensor:
    del offsets, shapes, permutation, p
    return values.new_empty((torch.library.get_ctx().new_dynamic_size(min=0),))


@torch.library.custom_op(
    "danling::_segmented_pdist_backward",
    mutates_args=(),
    schema="(Tensor grad, Tensor values, Tensor offsets, Tensor shapes, SymInt[] permutation, float p, "
    "Tensor distances) -> Tensor",
)
def _segmented_pdist_backward(
    grad: Tensor,
    values: Tensor,
    offsets: Tensor,
    shapes: Tensor,
    permutation: list[int],
    p: float,
    distances: Tensor,
) -> Tensor:
    return _pdist_native_backward(grad, values, offsets, shapes, permutation, p, distances)


def _pdist_native_backward(grad, values, offsets, shapes, permutation, p, distances):
    splits = offsets.tolist()
    results = []
    position = 0
    ragged_rank = len(permutation) - values.dim() + 1
    for start, end, shape in zip(splits[:-1], splits[1:], shapes.tolist()):
        count = shape[0] * (shape[0] - 1) // 2
        sample = _unpack_sample(values[start:end], shape, permutation).contiguous()
        # Native pdist backward can dereference an empty pair buffer for fewer than two rows.
        vjp = (
            torch.ops.aten._pdist_backward.default(
                grad[position : position + count].contiguous(), sample, p, distances[position : position + count]
            )
            if count and shape[1]
            else torch.zeros_like(sample)
        )
        results.append(_pack_sample(vjp, permutation, ragged_rank))
        position += count
    return torch.cat(results) if results else torch.zeros_like(values)


@_segmented_pdist_backward.register_fake
def _segmented_pdist_backward_fake(
    grad: Tensor,
    values: Tensor,
    offsets: Tensor,
    shapes: Tensor,
    permutation: list[int],
    p: float,
    distances: Tensor,
) -> Tensor:
    del grad, offsets, shapes, permutation, p, distances
    return torch.empty_like(values)


def _pdist_setup_context(ctx, inputs, output):
    values, offsets, shapes, permutation, p = inputs
    ctx.save_for_backward(values, offsets, shapes, output)
    ctx.permutation = permutation
    ctx.p = p


def _pdist_backward(ctx, grad):
    values, offsets, shapes, distances = ctx.saved_tensors
    backward = _pdist_native_backward if torch.is_grad_enabled() else _segmented_pdist_backward
    return backward(grad, values, offsets, shapes, ctx.permutation, ctx.p, distances), None, None, None, None


torch.library.register_autograd(segmented_pdist, _pdist_backward, setup_context=_pdist_setup_context)


def _segmented_fractional_max_pool_impl(
    values: Tensor,
    offsets: Tensor,
    shapes: Tensor,
    output_shapes: Tensor,
    permutation: list[int],
    spatial_rank: int,
    kernel: list[int],
    static_shape: list[int],
    random_samples: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run native stochastic windows without introducing a padded spatial envelope."""
    pool = F.fractional_max_pool2d_with_indices if spatial_rank == 2 else F.fractional_max_pool3d_with_indices
    ragged_rank = len(permutation) - values.dim() + 1
    splits = offsets.tolist()
    width = math.prod(values.shape[1:])
    outputs, indices, source_indices = [], [], []
    for start, end, shape, output_shape in zip(splits[:-1], splits[1:], shapes.tolist(), output_shapes.tolist()):
        sample = _unpack_sample(values[start:end], shape, permutation)
        output, index = pool(sample, kernel, output_size=output_shape[-spatial_rank:], _random_samples=random_samples)
        spatial_size = math.prod(shape[-spatial_rank:])
        planes = math.prod(shape[:-spatial_rank])
        plane = torch.arange(planes, device=values.device).reshape(*shape[:-spatial_rank], *([1] * spatial_rank))
        logical_index = index + plane * spatial_size
        source = torch.full_like(index, start * width)
        for dim in range(len(shape)):
            logical_stride = math.prod(shape[dim + 1 :])
            packed_stride = math.prod(shape[axis] for axis in permutation[permutation.index(dim) + 1 :])
            source = source + ((logical_index // logical_stride) % shape[dim]) * packed_stride
        outputs.append(_pack_sample(output, permutation, ragged_rank))
        indices.append(_pack_sample(index, permutation, ragged_rank))
        source_indices.append(_pack_sample(source, permutation, ragged_rank))
    if not outputs:
        empty = values.new_empty((0, *static_shape))
        return empty, empty.to(torch.long), empty.to(torch.long)
    return torch.cat(outputs), torch.cat(indices), torch.cat(source_indices)


_FRACTIONAL_POOL_LIBRARY = torch.library.Library("danling", "FRAGMENT")
_FRACTIONAL_POOL_LIBRARY.define(
    "_segmented_fractional_max_pool(Tensor values, Tensor offsets, Tensor shapes, Tensor output_shapes, "
    "SymInt[] permutation, SymInt spatial_rank, SymInt[] kernel, SymInt[] static_shape, "
    "Tensor? random_samples) -> (Tensor, Tensor, Tensor)",
    tags=(torch.Tag.pt2_compliant_tag, torch.Tag.needs_fixed_stride_order, torch.Tag.nondeterministic_seeded),
)
_FRACTIONAL_POOL_LIBRARY.impl(
    "_segmented_fractional_max_pool", _segmented_fractional_max_pool_impl, "CompositeExplicitAutograd"
)
segmented_fractional_max_pool = torch.ops.danling._segmented_fractional_max_pool.default


@torch.library.register_fake("danling::_segmented_fractional_max_pool")
def _segmented_fractional_max_pool_fake(
    values: Tensor,
    offsets: Tensor,
    shapes: Tensor,
    output_shapes: Tensor,
    permutation: list[int],
    spatial_rank: int,
    kernel: list[int],
    static_shape: list[int],
    random_samples: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    del offsets, shapes, spatial_rank, kernel, random_samples
    ragged_rank = len(permutation) - values.dim() + 1
    rows = torch.library.get_ctx().new_dynamic_size(min=0) if ragged_rank else output_shapes.shape[0]
    output = values.new_empty((rows, *static_shape))
    return output, output.to(torch.long), output.to(torch.long)


def _fractional_pool_setup_context(ctx, inputs, output):
    ctx.shape = inputs[0].shape
    ctx.save_for_backward(output[2])
    ctx.mark_non_differentiable(output[1], output[2])


def _fractional_pool_backward(ctx, grad, grad_indices, grad_source_indices):
    del grad_indices, grad_source_indices
    (indices,) = ctx.saved_tensors
    flat = grad.new_zeros(ctx.shape).reshape(-1)
    result = flat.scatter_add(0, indices.reshape(-1), grad.reshape(-1)).reshape(ctx.shape)
    return result, None, None, None, None, None, None, None, None


torch.library.register_autograd(
    "danling::_segmented_fractional_max_pool", _fractional_pool_backward, setup_context=_fractional_pool_setup_context
)


class PackedVectorNorm(torch.autograd.Function):
    """Grouped vector norms with native zero and tied-extremum derivatives."""

    generate_vmap_rule = True

    @staticmethod
    def forward(values, groups, group_count, order):
        carrier = values.float() if values.dtype in (torch.float16, torch.bfloat16) else values
        magnitudes = carrier.abs()
        shape = (group_count, *magnitudes.shape[1:])
        if order in (float("inf"), float("-inf")):
            largest = order == float("inf")
            result = magnitudes.new_full(shape, float("-inf") if largest else float("inf"))
            index = groups.reshape(-1, *([1] * (values.dim() - 1))).expand_as(magnitudes)
            result = result.scatter_reduce(0, index, magnitudes, "amax" if largest else "amin", include_self=False)
        else:
            powers = magnitudes.ne(0).to(magnitudes.dtype) if order == 0 else magnitudes.pow(order)
            result = powers.new_zeros(shape).index_add(0, groups, powers)
            if order != 0:
                result = result.pow(1.0 / order)
        return result.to(values.real.dtype)

    @staticmethod
    def setup_context(ctx, inputs, output):
        values, groups, _, ctx.order = inputs
        ctx.save_for_backward(values, groups, output)
        ctx.save_for_forward(values, groups, output)

    @staticmethod
    def _derivative(values, groups, norms, order):
        selected = norms.index_select(0, groups)
        if order in (float("inf"), float("-inf")):
            ties = values.abs() == selected
            counts = norms.new_zeros(norms.shape).index_add(0, groups, ties.to(norms.dtype))
            return torch.where(ties, values.sgn() / counts.index_select(0, groups), 0)
        if order == 2:
            return (values / selected).masked_fill(selected == 0, 0)
        zero = (values == 0) | (selected == 0)
        safe_norms = torch.where(selected == 0, 1, selected)
        powers = values.abs().pow(order - 1).masked_fill(values == 0, 0)
        derivative = values.sgn() * powers / safe_norms.pow(order - 1)
        return torch.where(zero, 0, derivative)

    @staticmethod
    def backward(ctx, grad):
        if ctx.order == 0:
            return None, None, None, None
        values, groups, norms = ctx.saved_tensors
        return (
            grad.index_select(0, groups) * PackedVectorNorm._derivative(values, groups, norms, ctx.order),
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, tangent, groups_tangent, count_tangent, order_tangent):
        del groups_tangent, count_tangent, order_tangent
        values, groups, norms = ctx.saved_tensors
        if ctx.order == 0:
            return torch.zeros_like(norms)
        derivative = PackedVectorNorm._derivative(values, groups, norms, ctx.order)
        contributions = (derivative.conj() * tangent).real
        return torch.zeros_like(norms).index_add(0, groups, contributions)
