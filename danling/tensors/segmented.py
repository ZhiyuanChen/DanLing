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


def _segmented_cdist_lengths(row_lengths: Tensor, column_lengths: Tensor) -> tuple[list[int], list[int]]:
    r"""Validate CPU length metadata and return concrete runtime segment sizes."""
    if row_lengths.dim() != 1 or column_lengths.dim() != 1:
        raise ValueError("segmented cdist lengths must be one-dimensional")
    if row_lengths.shape != column_lengths.shape:
        raise ValueError("segmented cdist length tensors must contain the same number of samples")
    if row_lengths.device.type != "cpu" or column_lengths.device.type != "cpu":
        raise ValueError("segmented cdist lengths must be CPU tensors")
    if row_lengths.dtype != torch.long or column_lengths.dtype != torch.long:
        raise TypeError("segmented cdist lengths must use torch.int64")
    rows = [int(length) for length in row_lengths.tolist()]
    columns = [int(length) for length in column_lengths.tolist()]
    if any(length < 0 for length in rows) or any(length < 0 for length in columns):
        raise ValueError("segmented cdist lengths must be non-negative")
    return rows, columns


def _validate_segmented_cdist_operands(
    left: Tensor,
    right: Tensor,
    rows: list[int],
    columns: list[int],
    compute_mode: int,
) -> None:
    if left.dim() != 2 or right.dim() != 2:
        raise ValueError("segmented cdist operands must be two-dimensional")
    if left.device != right.device or left.dtype != right.dtype:
        raise ValueError("segmented cdist operands must have the same device and dtype")
    if left.shape[1] != right.shape[1]:
        raise ValueError("segmented cdist operands must have the same feature width")
    if sum(rows) != left.shape[0] or sum(columns) != right.shape[0]:
        raise ValueError("segmented cdist lengths must cover their packed operands exactly")
    if compute_mode not in (0, 1, 2):
        raise ValueError(f"invalid segmented cdist compute_mode code: {compute_mode}")


def _mm_cdist_backward(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
    distances: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Differentiate the Euclidean MM path through its composite matrix formula.

    Dense cdist's Euclidean MM path is composite-autograd. Writing the same
    derivative as two matrix products preserves the live matmul precision
    policy, supports FP16/BF16 where native ``_cdist_backward`` does not, and
    needs only the distance matrix rather than a ``[P, R, M]`` difference.
    """
    scale = torch.where(distances != 0, grad / distances, torch.zeros_like(grad))
    grad_left = left * scale.sum(dim=1, keepdim=True) - scale @ right
    grad_right = right * scale.sum(dim=0).unsqueeze(1) - scale.mT @ left
    return grad_left, grad_right


@torch.library.custom_op("danling::_segmented_cdist_backward", mutates_args=())
def segmented_cdist_backward(
    grad_output: Tensor,
    left: Tensor,
    right: Tensor,
    row_lengths: Tensor,
    column_lengths: Tensor,
    distances: Tensor,
    p: float,
    compute_mode: int,
) -> tuple[Tensor, Tensor]:
    r"""First-order VJP for :func:`segmented_cdist`, kept opaque to AOTAutograd."""
    rows, columns = _segmented_cdist_lengths(row_lengths, column_lengths)
    _validate_segmented_cdist_operands(left, right, rows, columns, compute_mode)
    total_pairs = sum(row * column for row, column in zip(rows, columns))
    if grad_output.numel() != total_pairs or distances.numel() != total_pairs:
        raise ValueError("segmented cdist gradients and distances must cover every output pair")

    grad_left = torch.zeros_like(left)
    grad_right = torch.zeros_like(right)
    left_start = right_start = output_start = 0
    for row_count, column_count in zip(rows, columns):
        pair_count = row_count * column_count
        if row_count and column_count and left.shape[1]:
            left_segment = left[left_start : left_start + row_count]
            right_segment = right[right_start : right_start + column_count]
            grad_segment = grad_output[output_start : output_start + pair_count].reshape(row_count, column_count)
            distance_segment = distances[output_start : output_start + pair_count].reshape(row_count, column_count)
            uses_mm = p == 2 and (compute_mode == 1 or (compute_mode == 0 and (row_count > 25 or column_count > 25)))
            if uses_mm:
                left_vjp, right_vjp = _mm_cdist_backward(
                    grad_segment,
                    left_segment,
                    right_segment,
                    distance_segment,
                )
                grad_left[left_start : left_start + row_count].copy_(left_vjp)
                grad_right[right_start : right_start + column_count].copy_(right_vjp)
            else:
                torch.ops.aten._cdist_backward.out(
                    grad_segment.unsqueeze(0),
                    left_segment.unsqueeze(0),
                    right_segment.unsqueeze(0),
                    p,
                    distance_segment.unsqueeze(0),
                    out=grad_left[left_start : left_start + row_count].unsqueeze(0),
                )
                torch.ops.aten._cdist_backward.out(
                    grad_segment.mT.unsqueeze(0),
                    right_segment.unsqueeze(0),
                    left_segment.unsqueeze(0),
                    p,
                    distance_segment.mT.unsqueeze(0),
                    out=grad_right[right_start : right_start + column_count].unsqueeze(0),
                )
        left_start += row_count
        right_start += column_count
        output_start += pair_count
    return grad_left, grad_right


@segmented_cdist_backward.register_fake
def _segmented_cdist_backward_fake(
    grad_output: Tensor,
    left: Tensor,
    right: Tensor,
    row_lengths: Tensor,
    column_lengths: Tensor,
    distances: Tensor,
    p: float,
    compute_mode: int,
) -> tuple[Tensor, Tensor]:
    del grad_output, row_lengths, column_lengths, distances, p, compute_mode
    return torch.empty_like(left), torch.empty_like(right)


@torch.library.custom_op("danling::_segmented_cdist", mutates_args=())
def segmented_cdist(
    left: Tensor,
    right: Tensor,
    row_lengths: Tensor,
    column_lengths: Tensor,
    p: float,
    compute_mode: int,
) -> Tensor:
    r"""Run native cdist independently over packed variable-length segments.

    The operator is a single graph node with a data-dependent packed output
    extent. At runtime each segment retains PyTorch's optimized direct/MM
    selection and dtype behavior. No padded batch, cross-sample matrix, pair
    index tensor, or feature-expanded Cartesian gather is materialized.
    """
    rows, columns = _segmented_cdist_lengths(row_lengths, column_lengths)
    _validate_segmented_cdist_operands(left, right, rows, columns, compute_mode)
    output = left.new_empty((sum(row * column for row, column in zip(rows, columns)),))
    left_start = right_start = output_start = 0
    for row_count, column_count in zip(rows, columns):
        pair_count = row_count * column_count
        torch.ops.aten._cdist_forward.out(
            left[left_start : left_start + row_count],
            right[right_start : right_start + column_count],
            p,
            compute_mode,
            out=output[output_start : output_start + pair_count].view(row_count, column_count),
        )
        left_start += row_count
        right_start += column_count
        output_start += pair_count
    return output


@segmented_cdist.register_fake
def _segmented_cdist_fake(
    left: Tensor,
    right: Tensor,
    row_lengths: Tensor,
    column_lengths: Tensor,
    p: float,
    compute_mode: int,
) -> Tensor:
    del right, row_lengths, column_lengths, p, compute_mode
    total_pairs = torch.library.get_ctx().new_dynamic_size(min=0)
    return left.new_empty((total_pairs,))


def _segmented_cdist_setup_context(ctx, inputs, output) -> None:
    left, right, row_lengths, column_lengths, p, compute_mode = inputs
    ctx.save_for_backward(left, right, row_lengths, column_lengths, output)
    ctx.p = p
    ctx.compute_mode = compute_mode
    ctx.set_materialize_grads(False)


def _segmented_cdist_autograd_backward(ctx, grad_output):
    if grad_output is None:
        return None, None, None, None, None, None
    left, right, row_lengths, column_lengths, distances = ctx.saved_tensors
    grad_left, grad_right = segmented_cdist_backward(
        grad_output,
        left,
        right,
        row_lengths,
        column_lengths,
        distances,
        ctx.p,
        ctx.compute_mode,
    )
    return grad_left, grad_right, None, None, None, None


torch.library.register_autograd(
    segmented_cdist,
    _segmented_cdist_autograd_backward,
    setup_context=_segmented_cdist_setup_context,
)


def _segment_lengths(offsets: Tensor, total: int) -> list[int]:
    r"""Validate packed offsets and return concrete runtime segment lengths."""
    positions = [int(offset) for offset in offsets.tolist()]
    if not positions or positions[0] != 0 or positions[-1] != total:
        raise ValueError("segmented offsets must cover the packed values exactly")
    return [right - left for left, right in zip(positions, positions[1:])]


@torch.library.custom_op("danling::_segmented_cumprod_backward", mutates_args=())
def segmented_cumprod_backward(grad_output: Tensor, values: Tensor, offsets: Tensor, output: Tensor) -> Tensor:
    lengths = _segment_lengths(offsets, values.shape[0])
    grad_values = torch.empty_like(values)
    start = 0
    for length in lengths:
        if length:
            segment = values.narrow(0, start, length)
            grad_segment = torch.ops.aten.cumprod_backward.default(
                grad_output.narrow(0, start, length), segment, 0, output.narrow(0, start, length)
            )
            grad_values.narrow(0, start, length).copy_(grad_segment)
        start += length
    return grad_values


@segmented_cumprod_backward.register_fake
def _segmented_cumprod_backward_fake(grad_output: Tensor, values: Tensor, offsets: Tensor, output: Tensor) -> Tensor:
    del grad_output, offsets, output
    return torch.empty_like(values)


@torch.library.custom_op("danling::_segmented_cumprod", mutates_args=())
def segmented_cumprod(values: Tensor, offsets: Tensor) -> Tensor:
    lengths = _segment_lengths(offsets, values.shape[0])
    output = torch.empty_like(values)
    start = 0
    for length in lengths:
        if length:
            torch.ops.aten.cumprod.out(values.narrow(0, start, length), 0, out=output.narrow(0, start, length))
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
