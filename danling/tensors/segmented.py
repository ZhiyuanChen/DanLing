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
    if values.dim() > 1:
        segments = batch_idx.view(-1, *([1] * (values.dim() - 1))).expand_as(values)
    else:
        segments = batch_idx
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
    value so it follows whichever operand wins. A strict comparison keeps the earliest index on
    ties, which is what ``cummax`` and ``cummin`` report.
    """
    total = values.shape[0]
    if total == 0:
        return values.clone(), local_idx.clone()
    running, indices = values.clone(), local_idx.clone()
    step = 1
    while step < total:
        candidate, candidate_idx = _shift_down(running, step), _shift_down(indices, step)
        better = candidate > running if largest else candidate < running
        take = _same_segment(batch_idx, step, running.dim()) & better
        running = torch.where(take, candidate, running)
        indices = torch.where(take, candidate_idx, indices)
        step *= 2
    return running, indices
