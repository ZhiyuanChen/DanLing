# DanLing
# Copyright (C) 2022-Present  DanLing

# This program is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from collections.abc import Sequence
from typing import Tuple

import torch
from torch import Tensor


def tensor_mask(
    tensors: Sequence,
    size: torch.Size,
    *,
    batch_first: bool = True,
    padding_value: float = 0.0,
    mask_value: bool = False,
) -> Tuple[Tensor, Tensor]:
    r"""
    Build a padded tensor and corresponding tensor mask with a sequence of tensors and desired size.

    Args:
        tensors: sequence of tensors to be padded.
        size: desired size of the padded tensor (and mask tensor).
        batch_first: whether to put the batch dimension in the first dimension.
            Defaults to `True`.
        padding_value: padding value in the padded tensor.
            Defaults to `0.0`.
        mask_value: mask value in the mask tensor.
            Defaults to `False`.

    Returns:
        (tuple[Tensor, Tensor]): padded tensor and corresponding tensor mask

    Examples:
        >>> tensor_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        >>> tensor, mask = tensor_mask(tensor_list, (2, 3))
        >>> tensor
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> mask
        tensor([[ True,  True,  True],
                [ True,  True, False]])
    """

    tensor = torch.full(size, fill_value=padding_value, dtype=tensors[0].dtype, device=tensors[0].device)
    mask = torch.full(size, fill_value=mask_value, dtype=torch.bool, device=tensors[0].device)
    for i, t in enumerate(tensors):
        tensor[i][tuple(slice(0, t.shape[dim]) for dim in range(len(size) - 1))] = t  # type: ignore
        mask[i][tuple(slice(0, t.shape[dim]) for dim in range(len(size) - 1))] = not mask_value
    if not batch_first:
        tensor, mask = tensor.transpose(0, 1), mask.transpose(0, 1)
    return tensor, mask


def pad_tensor(tensors: Sequence, size: torch.Size, *, batch_first: bool = True, padding_value: float = 0.0) -> Tensor:
    r"""
    Pads a tensor with a sequence of tensors and desired size.

    Args:
        tensors: sequence of tensors to be padded.
        size: desired size of the padded tensor (and mask tensor).
        batch_first: whether to put the batch dimension in the first dimension.
            Defaults to `True`.
        mask_value: mask value in the mask tensor.
            Defaults to `False`.

    Returns:
        (Tensor): padded tensor
    """

    ret = torch.full(size, fill_value=padding_value, dtype=tensors[0].dtype, device=tensors[0].device)
    for i, t in enumerate(tensors):
        ret[i][tuple(slice(0, t.shape[dim]) for dim in range(len(size) - 1))] = t  # type: ignore
    if not batch_first:
        ret = ret.transpose(0, 1)
    return ret


def mask_tensor(tensors: Sequence, size: torch.Size, *, batch_first: bool = True, mask_value: bool = False) -> Tensor:
    r"""
    Build a tensor mask with a sequence of tensors and desired size.

    Args:
        tensors: sequence of tensors to be padded.
        size: desired size of the padded tensor (and mask tensor).
        batch_first: whether to put the batch dimension in the first dimension.
            Defaults to `True`.
        mask_value: mask value in the mask tensor.
            Defaults to `False`.

    Returns:
        (Tensor): tensor mask
    """

    ret = torch.full(size, fill_value=mask_value, dtype=torch.bool, device=tensors[0].device)
    for i, t in enumerate(tensors):
        ret[i][tuple(slice(0, t.shape[dim]) for dim in range(len(size) - 1))] = not mask_value
    if not batch_first:
        ret = ret.transpose(0, 1)
    return ret
