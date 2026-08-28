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

import torch
from torch import Tensor


def _segment_lengths(offsets: Tensor, total: int) -> list[int]:
    r"""Validate packed offsets and return concrete runtime segment lengths."""
    if offsets.dim() != 1:
        raise ValueError("segmented offsets must be one-dimensional")
    if offsets.device.type != "cpu":
        raise ValueError("segmented offsets must be a CPU tensor")
    if offsets.dtype != torch.long:
        raise TypeError("segmented offsets must use torch.int64")
    positions = [int(offset) for offset in offsets.tolist()]
    if not positions or positions[0] != 0:
        raise ValueError("segmented offsets must start at zero")
    if any(left > right for left, right in zip(positions, positions[1:])):
        raise ValueError("segmented offsets must be nondecreasing")
    if positions[-1] != total:
        raise ValueError("segmented offsets must cover the packed values exactly")
    return [right - left for left, right in zip(positions, positions[1:])]


@torch.library.custom_op("danling::_segmented_cumprod_backward", mutates_args=())
def segmented_cumprod_backward(
    grad_output: Tensor,
    values: Tensor,
    offsets: Tensor,
    output: Tensor,
) -> Tensor:
    r"""Apply native cumprod's first-order VJP independently to packed segments."""
    lengths = _segment_lengths(offsets, values.shape[0])
    if grad_output.shape != values.shape or output.shape != values.shape:
        raise ValueError("segmented cumprod values, output, and gradient must have the same shape")
    grad_values = torch.empty_like(values)
    start = 0
    for length in lengths:
        if length:
            segment = values.narrow(0, start, length)
            grad_segment = torch.ops.aten.cumprod_backward.default(
                grad_output.narrow(0, start, length),
                segment,
                0,
                output.narrow(0, start, length),
            )
            grad_values.narrow(0, start, length).copy_(grad_segment)
        start += length
    return grad_values


@segmented_cumprod_backward.register_fake
def _segmented_cumprod_backward_fake(
    grad_output: Tensor,
    values: Tensor,
    offsets: Tensor,
    output: Tensor,
) -> Tensor:
    del grad_output, offsets, output
    return torch.empty_like(values)


@torch.library.custom_op("danling::_segmented_cumprod", mutates_args=())
def segmented_cumprod(values: Tensor, offsets: Tensor) -> Tensor:
    r"""Run native cumprod independently over packed variable-length segments.

    Calling the device's native kernel per segment preserves its accumulation order, including
    device-specific rounding, underflow, zero, and non-finite behavior. The custom-op boundary
    keeps the data-dependent segment loop opaque to full-graph compilation without padding.
    """
    lengths = _segment_lengths(offsets, values.shape[0])
    output = torch.empty_like(values)
    start = 0
    for length in lengths:
        if length:
            torch.ops.aten.cumprod.out(
                values.narrow(0, start, length),
                0,
                out=output.narrow(0, start, length),
            )
        start += length
    return output


@segmented_cumprod.register_fake
def _segmented_cumprod_fake(values: Tensor, offsets: Tensor) -> Tensor:
    del offsets
    return torch.empty_like(values)


def _segmented_cumprod_setup_context(ctx, inputs, output) -> None:
    values, offsets = inputs
    ctx.save_for_backward(values, offsets, output)
    ctx.set_materialize_grads(False)


def _segmented_cumprod_autograd_backward(ctx, grad_output):
    if grad_output is None:
        return None, None
    values, offsets, output = ctx.saved_tensors
    return segmented_cumprod_backward(grad_output, values, offsets, output), None


torch.library.register_autograd(
    segmented_cumprod,
    _segmented_cumprod_autograd_backward,
    setup_context=_segmented_cumprod_setup_context,
)


def align_rows(rows: Tensor, values: Tensor) -> Tensor:
    r"""
    Broadcast a per-row vector across the static tail of ``values``.

    Packed metadata — segment ids, local positions — carries one entry per packed row, while
    the values it describes are ``[N, *tail]``. Torch aligns a ``[N]`` vector to the *trailing*
    axis, so it has to be reshaped to lead before it is expanded: left as it is, a tail whose
    width happens to equal ``N`` broadcasts legally and silently answers the wrong question,
    and any other width raises an opaque size mismatch.

    Args:
        rows: One entry per packed row.
        values: The packed values ``rows`` describes.

    Returns:
        Tensor: ``rows`` with ``values``' shape, or ``rows`` itself when there is no tail.
    """
    if values.dim() <= 1:
        return rows
    return rows.view(-1, *([1] * (values.dim() - 1))).expand_as(values)


def segmented_sort_perm(
    values: Tensor,
    offsets: Tensor,
    batch_idx: Tensor,
    *,
    descending: bool = False,
) -> tuple[Tensor, Tensor]:
    r"""
    Sort each segment of a packed tensor along packed dim 0.

    Two stable global sorts compose into a segmented one: sort by value, then stably sort by
    segment id. Because the second pass is stable it regroups by segment while preserving the
    ordering the first pass established, so segments come out contiguous and internally sorted.

    Args:
        values: Packed values, sorted along dim 0. A static tail is sorted per column.
        offsets: Segment offsets, length ``batch + 1``.
        batch_idx: Segment id of every packed row.
        descending: Sort each segment in descending order.

    Returns:
        tuple[Tensor, Tensor]: The permutation into ``values``, and the same permutation
        expressed as per-segment indices.
    """
    segments = align_rows(batch_idx, values)
    by_value = torch.argsort(values, dim=0, stable=True, descending=descending)
    by_segment = torch.argsort(torch.gather(segments, 0, by_value), dim=0, stable=True)
    perm = torch.gather(by_value, 0, by_segment)
    return perm, perm - offsets.to(perm.device)[torch.gather(segments, 0, perm)]


def _shift_down(values: Tensor, step: int) -> Tensor:
    r"""Return ``values`` shifted ``step`` rows later, zero-filled at the front."""
    shifted = torch.zeros_like(values)
    shifted[step:] = values[:-step]
    return shifted


def _same_segment(batch_idx: Tensor, step: int, rank: int) -> Tensor:
    r"""Mask rows whose partner ``step`` back belongs to the same segment."""
    same = torch.zeros(batch_idx.shape[0], dtype=torch.bool, device=batch_idx.device)
    same[step:] = batch_idx[step:] == batch_idx[:-step]
    return same.view(-1, *([1] * (rank - 1)))


def segmented_scan(values: Tensor, batch_idx: Tensor, combine) -> Tensor:
    r"""
    Inclusive scan of ``combine`` within each segment.

    A log-step Hillis-Steele scan, masked so a row only combines with a partner in its own
    segment. The mask is what makes this a segmented scan rather than a global one, and it is
    needed because the trick ``cumsum`` uses — scan globally, then subtract the running total
    at each segment start — requires an inverse. ``cummax`` has none and ``cumprod``'s is
    division, which is unsafe across zeros.

    A log-step scan combines whole blocks rather than one element at a time, so an intermediate
    result can leave the representable range even when every prefix is inside it. Callers whose
    ``combine`` can overflow — ``mul`` can, ``logaddexp`` cannot — should widen the dtype before
    scanning and narrow the result afterwards.

    Args:
        values: Packed values, scanned along dim 0. A static tail is scanned per column.
        batch_idx: Segment id of every packed row.
        combine: The associative operator to scan with.

    Returns:
        Tensor: The inclusive scan, with ``values``' shape.
    """
    total = values.shape[0]
    if total == 0:
        return values.clone()
    result = values.clone()
    step = 1
    while step < total:
        mask = _same_segment(batch_idx, step, result.dim())
        result = torch.where(mask, combine(result, _shift_down(result, step)), result)
        step *= 2
    return result


def segmented_arg_scan(
    values: Tensor,
    batch_idx: Tensor,
    local_idx: Tensor,
    *,
    largest: bool,
) -> tuple[Tensor, Tensor]:
    r"""
    Running extremum within each segment, with the per-segment index that produced it.

    The same masked log-step scan as :func:`segmented_scan`, carrying the index alongside the
    value so it follows whichever operand wins. ``candidate`` is the earlier window and
    ``running`` the later one, so a strict comparison keeps the *latest* tied index — which is
    what ``cummax`` and ``cummin`` report. NaN is absorbing for both, and a strict comparison
    is false against NaN, so NaN has to win explicitly or a later ordinary value would
    overwrite it.

    Args:
        values: Packed values, scanned along dim 0. A static tail is scanned per column.
        batch_idx: Segment id of every packed row.
        local_idx: Position of every packed row within its own segment.
        largest: Track the running maximum rather than the running minimum.

    Returns:
        tuple[Tensor, Tensor]: The running extremum and the per-segment index that produced it,
        both with ``values``' shape.
    """
    indices = align_rows(local_idx, values)
    total = values.shape[0]
    if total == 0:
        return values.clone(), indices.clone()
    running, indices = values.clone(), indices.clone()
    nan_absorbs = values.is_floating_point()
    step = 1
    while step < total:
        candidate, candidate_idx = _shift_down(running, step), _shift_down(indices, step)
        better = candidate > running if largest else candidate < running
        if nan_absorbs:
            better = better | (candidate.isnan() & ~running.isnan())
        take = _same_segment(batch_idx, step, running.dim()) & better
        running = torch.where(take, candidate, running)
        indices = torch.where(take, candidate_idx, indices)
        step *= 2
    return running, indices
