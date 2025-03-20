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


def infer_task(num_classes: int | None, num_labels: int | None):
    if num_classes is not None and num_labels is not None:
        raise ValueError("Only one of `num_classes` or `num_labels` can be provided.")
    if num_classes is not None:
        return "multiclass"
    if num_labels is not None:
        return "multilabel"
    return "binary"


def preprocess(
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
    input, target = preprocess(input, target, ignore_nan=ignore_nan)
    if num_outputs > 1:
        return input.view(-1, num_outputs), target.view(-1, num_outputs)
    return input.view(-1), target.view(-1)


def preprocess_binary(
    input: Tensor | NestedTensor | Sequence,
    target: Tensor | NestedTensor | Sequence,
    ignore_index: int | None = -100,
    ignore_nan: bool = False,
):
    input, target = preprocess(input, target, ignore_index=ignore_index)
    input, target = input.view(-1), target.view(-1)
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
    input, target = preprocess(input, target, ignore_index=ignore_index)
    input, target = input.view(-1, num_classes), target.view(-1)
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
    input, target = preprocess(input, target, ignore_index=ignore_index)
    input, target = input.view(-1, num_labels), target.view(-1, num_labels)
    if input.max() > 1 or input.min() < 0:
        input = input.sigmoid()
    return input, target
