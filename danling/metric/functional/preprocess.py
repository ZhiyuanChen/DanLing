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
        if isinstance(target, NestedTensor) and isinstance(input, Tensor):
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

    Args:
        preprocess_fn: The preprocessing function to apply
        **default_kwargs: Default values for the preprocessing function's parameters

    Returns:
        Decorated function with preprocessing capability
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
