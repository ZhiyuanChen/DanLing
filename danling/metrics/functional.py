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

# pylint: disable=redefined-builtin
from __future__ import annotations

import torch
from lazy_imports import try_import
from torch import Tensor

from danling.tensors import NestedTensor

with try_import() as lazy_import:
    from torcheval.metrics import functional as tef
    from torchmetrics import functional as tmf


def auroc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    weight: Tensor | None = None,
    average: str | None = "macro",
    num_labels: int | None = None,
    num_classes: int | None = None,
    task_weights: Tensor | None = None,
    ignored_index: int = -100,
    **kwargs,
):
    if num_classes and num_labels:
        raise ValueError("Only one of num_classes or num_labels can be specified, but not both")
    input, target = preprocess(input, target, ignored_index=ignored_index)
    if num_labels is None and num_classes is None:
        return tef.binary_auroc(input=input, target=target, weight=weight, **kwargs)
    if num_classes is None:
        ret = tef.binary_auroc(input=input.T, target=target.T, num_tasks=num_labels, weight=weight, **kwargs)
        if task_weights is not None:
            return ret @ task_weights.double()
        return ret.mean()
    if num_labels is None:
        return tef.multiclass_auroc(input=input, target=target, num_classes=num_classes, average=average, **kwargs)
    raise ValueError("Could not infer the type of the task. Only one of `num_labels`, `num_classes` is allowed.")


def auprc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    average: str | None = "macro",
    num_labels: int | None = None,
    num_classes: int | None = None,
    task_weights: Tensor | None = None,
    ignored_index: int = -100,
    **kwargs,
):
    if num_classes and num_labels:
        raise ValueError("Only one of num_classes or num_labels can be specified, but not both")
    input, target = preprocess(input, target, ignored_index=ignored_index)
    if num_labels is None and num_classes is None:
        return tef.binary_auprc(input=input, target=target, **kwargs)
    if num_classes is None:
        ret = tef.multilabel_auprc(input=input, target=target, num_labels=num_labels, average=average, **kwargs)
        if task_weights is not None:
            return ret @ task_weights.double()
        return ret.mean()
    if num_labels is None:
        return tef.multiclass_auprc(input=input, target=target, num_classes=num_classes, average=average, **kwargs)
    raise ValueError("Could not infer the type of the task. Only one of `num_labels`, `num_classes` is allowed.")


def accuracy(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    average: str | None = "micro",
    num_labels: int | None = None,
    num_classes: int | None = None,
    ignored_index: int = -100,
    **kwargs,
):
    if num_classes and num_labels:
        raise ValueError("Only one of num_classes or num_labels can be specified, but not both")
    input, target = preprocess(input, target, ignored_index=ignored_index)
    if num_labels is None and num_classes is None:
        return tef.binary_accuracy(input=input, target=target, threshold=threshold, **kwargs)
    if num_classes is None:
        return tef.multilabel_accuracy(input=input, target=target, threshold=threshold, **kwargs)
    if num_labels is None:
        return tef.multiclass_accuracy(input=input, target=target, num_classes=num_classes, average=average, **kwargs)
    raise ValueError("Could not infer the type of the task. Only one of `num_labels`, `num_classes` is allowed.")


def matthews_corrcoef(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    num_labels: int | None = None,
    num_classes: int | None = None,
    ignored_index: int = -100,
):
    lazy_import.check()
    if num_classes and num_labels:
        raise ValueError("Only one of num_classes or num_labels can be specified, but not both")
    task = "binary"
    if num_classes:
        task = "multiclass"
    if num_labels:
        task = "multilabel"
    input, target = preprocess(input, target, ignored_index=ignored_index)
    try:
        return tmf.matthews_corrcoef(
            input, target, task, threshold=threshold, num_classes=num_classes, num_labels=num_labels
        )
    except ValueError:
        return torch.tensor(0, dtype=float).to(input.device)


def pearson(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
):
    lazy_import.check()
    input, target = preprocess(input, target)
    try:
        return tmf.pearson_corrcoef(input, target)
    except ValueError:
        return torch.tensor(0, dtype=float).to(input.device)


def spearman(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
):
    lazy_import.check()
    input, target = preprocess(input, target)
    try:
        return tmf.spearman_corrcoef(input, target)
    except ValueError:
        return torch.tensor(0, dtype=float).to(input.device)


def r2_score(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    multioutput: str = "uniform_average",
    num_regressors: int = 0,
):
    input, target = preprocess(input, target)
    try:
        return tef.r2_score(input, target, multioutput=multioutput, num_regressors=num_regressors)
    except ValueError:
        return torch.tensor(0, dtype=float).to(input.device)


def mse(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
):
    input, target = preprocess(input, target)
    return tef.mean_squared_error(input, target)


def rmse(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
):
    input, target = preprocess(input, target)
    return mse(input, target).sqrt()


def preprocess(input: Tensor | NestedTensor, target: Tensor | NestedTensor, ignored_index: int | None = None):
    if isinstance(input, NestedTensor) or isinstance(target, NestedTensor):
        if isinstance(input, NestedTensor) and isinstance(target, Tensor):
            target = input.nested_like(target, strict=False)
        if isinstance(target, NestedTensor) and isinstance(input, Tensor):
            input = target.nested_like(input, strict=False)
        input, target = torch.cat(input.storage()), torch.cat(target.storage())
    if ignored_index is not None:
        input, target = input[target != ignored_index], target[target != ignored_index]
    return input, target
