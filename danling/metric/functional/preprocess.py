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
from copy import copy
from functools import wraps
from inspect import signature
from typing import Callable

import torch
from torch import Tensor

from danling.tensor import NestedTensor

from .utils import infer_task


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
    if not isinstance(input, (Tensor, NestedTensor)):
        try:
            input = torch.tensor(input)
        except ValueError:
            input = NestedTensor(input)
    if not isinstance(target, (Tensor, NestedTensor)):
        try:
            target = torch.tensor(target)
        except ValueError:
            target = NestedTensor(target)
    if isinstance(input, NestedTensor) or isinstance(target, NestedTensor):
        if isinstance(input, NestedTensor) and isinstance(target, Tensor):
            target = input.nested_like(target, strict=False)
        elif isinstance(target, NestedTensor) and isinstance(input, Tensor):
            input = target.nested_like(input, strict=False)
        input, target = input.concat, target.concat
    if ignore_index is not None:
        mask = target != ignore_index
        input, target = input[mask], target[mask]
    if ignore_nan:
        mask = ~(torch.isnan(target))
        input, target = input[mask], target[mask]
    if input.numel() == target.numel():
        return input.squeeze(), target.squeeze()
    return input, target


def preprocess_regression(
    input: Tensor | NestedTensor | Sequence,
    target: Tensor | NestedTensor | Sequence,
    num_outputs: int = 1,
    ignore_nan: bool = True,
    ignore_index: int | None = None,
):
    input, target = base_preprocess(input, target, ignore_nan=ignore_nan)
    if num_outputs > 1:
        return input.view(-1, num_outputs), target.view(-1, num_outputs)
    input = input.to(target.dtype)
    return input.flatten(), target.flatten()


def preprocess_classification(
    input: Tensor | NestedTensor | Sequence,
    target: Tensor | NestedTensor | Sequence,
    task: str | None = None,
    num_labels: int | None = None,
    num_classes: int | None = None,
    ignore_index: int | None = -100,
    ignore_nan: bool = False,
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
    ignore_nan: bool = False,
):
    input, target = base_preprocess(input, target, ignore_index=ignore_index)
    input, target = input.flatten(), target.flatten()
    if input.max() > 1 or input.min() < 0:
        input = input.sigmoid()
    return input, target


def preprocess_multiclass(
    input: Tensor | NestedTensor | Sequence,
    target: Tensor | NestedTensor | Sequence,
    num_classes: int,
    ignore_index: int | None = -100,
    ignore_nan: bool = False,
):
    input, target = base_preprocess(input, target, ignore_index=ignore_index)
    input, target = input.view(-1, num_classes), target.flatten()
    if input.max() > 1 or input.min() < 0:
        input = input.softmax(dim=-1)
    return input, target


def preprocess_multilabel(
    input: Tensor | NestedTensor | Sequence,
    target: Tensor | NestedTensor | Sequence,
    num_labels: int,
    ignore_index: int | None = -100,
    ignore_nan: bool = False,
):
    input, target = base_preprocess(input, target, ignore_index=ignore_index)
    input, target = input.view(-1, num_labels), target.view(-1, num_labels)
    if input.max() > 1 or input.min() < 0:
        input = input.sigmoid()
    return input, target


def with_preprocess(preprocess_fn: Callable, **default_kwargs):
    """
    Decorator to apply preprocessing to metric functions.

    This decorator wraps metric functions to handle preprocessing of inputs before the metric is calculated.
    It handles common preprocessing tasks like ignoring certain values or converting input formats,
    making the metric functions more robust and easier to use.

    Args:
        preprocess_fn: The preprocessing function to apply
        **default_kwargs: Default values for the preprocessing function's parameters

    Returns:
        Decorated function with preprocessing capability

    Examples:
        >>> import torch
        >>> @with_preprocess(preprocess_binary, ignore_index=-100)
        ... def my_regression_metric(input, target):
        ...     # Assumes input and target are already preprocessed
        ...     return input.mean() / target.mean()

        >>> # When called, preprocessing is automatically applied
        >>> result = my_regression_metric(torch.rand(4), torch.rand(4))

        >>> # Preprocessing can be disabled
        >>> result = my_regression_metric(torch.rand(4), torch.rand(4), preprocess=False)

        >>> # Additional preprocessing parameters can be passed
        >>> result = my_regression_metric(torch.rand(4), torch.rand(4), ignore_index=-1)

    Notes:
        - The wrapper extracts parameters needed for preprocessing from **kwargs
        - If preprocess=False is passed, no preprocessing is applied
        - All other parameters are passed through to the metric function
    """
    preprocess_params = set(signature(preprocess_fn).parameters.keys()) - {"input", "target"}

    def decorator(metric_fn: Callable) -> Callable:

        @wraps(metric_fn)
        def wrapper(
            input: Tensor | NestedTensor | Sequence,
            target: Tensor | NestedTensor | Sequence,
            *,
            preprocess: bool = True,
            **kwargs,
        ):
            metric_kwargs = {k: v for k, v in kwargs.items() if k not in default_kwargs}

            if not preprocess:
                return metric_fn(input, target, **metric_kwargs)

            preprocess_kwargs = copy(default_kwargs)
            for key in preprocess_params:
                if key in kwargs:
                    preprocess_kwargs[key] = kwargs[key]
            input, target = preprocess_fn(input, target, **preprocess_kwargs)
            return metric_fn(input, target, **metric_kwargs)

        return wrapper

    return decorator
