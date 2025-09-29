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

# pylint: disable=redefined-builtin
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from danling.tensors import NestedTensor

from .functional.utils import infer_task


def base_preprocess(
    input: Tensor | NestedTensor | Sequence,
    target: Tensor | NestedTensor | Sequence,
    ignore_index: int | None = None,
    ignore_nan: bool = False,
):
    """
    Basic preprocessing function for metric inputs and targets.

    This function handles common preprocessing tasks for metric computation:
    1. Converting inputs/targets to tensors or nested tensors
    2. Handling nested tensors by concatenating them
    3. Removing ignored indices (e.g., padding)
    4. Removing NaN values
    5. Properly reshaping tensors for metric functions

    Args:
        input: Model predictions or outputs
            - Can be Tensor, NestedTensor, or a sequence that can be converted to tensor
        target: Ground truth labels/values
            - Can be Tensor, NestedTensor, or a sequence that can be converted to tensor
        ignore_index: Value in target to ignore (e.g., -100 for padding in classification tasks)
        ignore_nan: Whether to remove NaN values (useful for regression tasks)

    Returns:
        tuple: (processed_input, processed_target) as tensors ready for metric computation

    Examples:
        >>> # Basic usage with tensors
        >>> input = torch.tensor([0.1, 0.8, 0.6, 0.2])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> proc_input, proc_target = base_preprocess(input, target)
        >>> proc_input.shape, proc_target.shape
        (torch.Size([4]), torch.Size([4]))

        >>> # Ignoring a specific value
        >>> input = torch.tensor([0.1, 0.8, 0.6, 0.2])
        >>> target = torch.tensor([0, -100, 1, 0])
        >>> proc_input, proc_target = base_preprocess(input, target, ignore_index=-100)
        >>> proc_input.shape, proc_target.shape
        (torch.Size([3]), torch.Size([3]))

        >>> # Working with lists
        >>> input = [0.1, 0.8, 0.6, 0.2]
        >>> target = [0, 1, 1, 0]
        >>> proc_input, proc_target = base_preprocess(input, target)
        >>> proc_input.shape, proc_target.shape
        (torch.Size([4]), torch.Size([4]))
    """
    input = _coerce_to_tensor_like(input)
    target = _coerce_to_tensor_like(target)

    input, target = _align_nested_tensors(input, target)

    if ignore_index is not None:
        input, target = _apply_mask(input, target, target != ignore_index)
    if ignore_nan:
        input, target = _apply_mask(input, target, ~torch.isnan(target))

    if input.numel() == target.numel():
        return input.squeeze(), target.squeeze()
    return input, target


def _coerce_to_tensor_like(value: Tensor | NestedTensor | Sequence) -> Tensor | NestedTensor:
    if isinstance(value, (Tensor, NestedTensor)):
        return value
    if isinstance(value, Sequence):
        try:
            return torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError):
            return NestedTensor(value)
    raise TypeError(f"Unsupported input type: {type(value)}")


def _align_nested_tensors(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
) -> tuple[Tensor, Tensor]:
    input_is_nested = isinstance(input, NestedTensor)
    target_is_nested = isinstance(target, NestedTensor)

    if input_is_nested or target_is_nested:
        if input_is_nested and not target_is_nested:
            target = input.nested_like(target, strict=False)
        elif target_is_nested and not input_is_nested:
            input = target.nested_like(input, strict=False)
        if isinstance(input, NestedTensor):
            input = input.concat
        if isinstance(target, NestedTensor):
            target = target.concat

    if not isinstance(input, Tensor) or not isinstance(target, Tensor):
        raise TypeError("base_preprocess expects tensors after alignment")
    return input, target


def _apply_mask(input: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    if mask.dtype is not torch.bool:
        mask = mask.to(dtype=torch.bool)
    return input[mask], target[mask]


def preprocess_regression(
    input: Tensor | NestedTensor | Sequence,
    target: Tensor | NestedTensor | Sequence,
    num_outputs: int = 1,
    ignore_nan: bool = True,
):
    input, target = base_preprocess(input, target, ignore_nan=False, ignore_index=None)
    if not target.is_floating_point():
        raise TypeError(f"Regression targets must be floating point tensors, but got {target.dtype}.")
    input = input.to(target.dtype)

    if num_outputs <= 0:
        raise ValueError(f"num_outputs must be positive, but got {num_outputs!r}")

    if num_outputs == 1:
        if ignore_nan:
            input, target = _apply_mask(input, target, ~torch.isnan(target))
        return input.flatten(), target.flatten()

    if target.numel() % num_outputs != 0 or input.numel() % num_outputs != 0:
        raise ValueError(
            "Regression inputs cannot be reshaped to the requested num_outputs. "
            f"Got input.numel()={input.numel()}, target.numel()={target.numel()}, num_outputs={num_outputs}."
        )

    input = input.reshape(-1, num_outputs)
    target = target.reshape(-1, num_outputs)
    if ignore_nan:
        row_mask = ~torch.isnan(target).any(dim=-1)
        input, target = input[row_mask], target[row_mask]
    return input, target


def preprocess_classification(
    input: Tensor | NestedTensor | Sequence,
    target: Tensor | NestedTensor | Sequence,
    task: str | None = None,
    num_labels: int | None = None,
    num_classes: int | None = None,
    ignore_index: int | None = -100,
):
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        return preprocess_binary(input, target, ignore_index=ignore_index)
    if task == "multiclass":
        return preprocess_multiclass(input, target, num_classes=num_classes, ignore_index=ignore_index)  # type: ignore
    if task == "multilabel":
        return preprocess_multilabel(input, target, num_labels=num_labels, ignore_index=ignore_index)  # type: ignore
    raise ValueError(f"Invalid task: {task}")


def preprocess_binary(
    input: Tensor | NestedTensor | Sequence,
    target: Tensor | NestedTensor | Sequence,
    ignore_index: int | None = -100,
):
    input, target = base_preprocess(input, target, ignore_index=None, ignore_nan=False)
    input, target = input.flatten(), target.flatten()
    if ignore_index is not None:
        mask = target != ignore_index
        input, target = input[mask], target[mask]
    if input.numel() > 0 and (input.max() > 1 or input.min() < 0):
        input = input.sigmoid()
    return input, target


def preprocess_multiclass(
    input: Tensor | NestedTensor | Sequence,
    target: Tensor | NestedTensor | Sequence,
    num_classes: int,
    ignore_index: int | None = -100,
):
    input, target = base_preprocess(input, target, ignore_index=None, ignore_nan=False)
    input, target = input.reshape(-1, num_classes), target.flatten()
    if ignore_index is not None:
        mask = target != ignore_index
        input, target = input[mask], target[mask]
    if input.numel() > 0 and (input.max() > 1 or input.min() < 0):
        input = input.softmax(dim=-1)
    return input, target


def preprocess_multilabel(
    input: Tensor | NestedTensor | Sequence,
    target: Tensor | NestedTensor | Sequence,
    num_labels: int,
    ignore_index: int | None = -100,
):
    input, target = base_preprocess(input, target, ignore_index=None, ignore_nan=False)
    input, target = input.reshape(-1, num_labels), target.reshape(-1, num_labels)
    if ignore_index is not None:
        # Preserve per-label ignore markers for partially valid samples, but
        # drop rows that are entirely ignored to avoid storing pure padding.
        row_mask = (target != ignore_index).any(dim=-1)
        input, target = input[row_mask], target[row_mask]
    if input.numel() > 0 and (input.max() > 1 or input.min() < 0):
        input = input.sigmoid()
    return input, target
