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
# mypy: disable-error-code="arg-type"
from __future__ import annotations

import torch
from lazy_imports import try_import
from torch import Tensor

from danling.tensors import NestedTensor

with try_import() as te:
    from torcheval.metrics import functional as tef
with try_import() as tm:
    from torchmetrics import functional as tmf

from .preprocesses import (
    infer_task,
    preprocess_binary,
    preprocess_multiclass,
    preprocess_multilabel,
    preprocess_regression,
)


def auroc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    weight: Tensor | None = None,
    average: str | None = "macro",
    num_labels: int | None = None,
    num_classes: int | None = None,
    task_weights: Tensor | None = None,
    task: str | None = None,
    preprocess: bool = True,
    ignored_index: int | None = -100,
    **kwargs,
):
    te.check()
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        if preprocess:
            input, target = preprocess_binary(input, target, ignored_index=ignored_index)
        return tef.binary_auroc(input=input, target=target, weight=weight, **kwargs)
    if task == "multiclass":
        if preprocess:
            input, target = preprocess_multiclass(input, target, num_classes, ignored_index=ignored_index)
        return tef.multiclass_auroc(input=input, target=target, num_classes=num_classes, average=average, **kwargs)
    if task == "multilabel":
        if preprocess:
            input, target = preprocess_multilabel(input, target, num_labels, ignored_index=ignored_index)
        ret = tef.binary_auroc(input=input.T, target=target.T, num_tasks=num_labels, weight=weight, **kwargs)
        if task_weights is not None:
            return ret @ task_weights.double()
        return ret.mean()
    raise ValueError(f"Task should be one of binary, multiclass, or multilabel, but got {task}")


def auprc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    average: str | None = "macro",
    num_labels: int | None = None,
    num_classes: int | None = None,
    task_weights: Tensor | None = None,
    task: str | None = None,
    preprocess: bool = True,
    ignored_index: int | None = -100,
    **kwargs,
):
    te.check()
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        if preprocess:
            input, target = preprocess_binary(input, target, ignored_index=ignored_index)
        return tef.binary_auprc(input=input, target=target, **kwargs)
    if task == "multiclass":
        if preprocess:
            input, target = preprocess_multiclass(input, target, num_classes, ignored_index=ignored_index)
        return tef.multiclass_auprc(input=input, target=target, num_classes=num_classes, average=average, **kwargs)
    if task == "multilabel":
        if preprocess:
            input, target = preprocess_multilabel(input, target, num_labels, ignored_index=ignored_index)
        ret = tef.multilabel_auprc(input=input, target=target, num_labels=num_labels, average=average, **kwargs)
        if task_weights is not None:
            return ret @ task_weights.double()
        return ret.mean()
    raise ValueError(f"Task should be one of binary, multiclass, or multilabel, but got {task}")


def f1_score(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    average: str | None = "micro",
    num_labels: int | None = None,
    num_classes: int | None = None,
    task_weights: Tensor | None = None,
    task: str | None = None,
    preprocess: bool = True,
    ignored_index: int | None = -100,
    **kwargs,
):
    te.check()
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        if preprocess:
            input, target = preprocess_binary(input, target, ignored_index=ignored_index)
        return tef.binary_f1_score(input=input, target=target, threshold=threshold, **kwargs)
    if task == "multiclass":
        if preprocess:
            input, target = preprocess_multiclass(input, target, num_classes, ignored_index=ignored_index)
        return tef.multiclass_f1_score(input=input, target=target, num_classes=num_classes, average=average, **kwargs)
    if task == "multilabel":
        if preprocess:
            input, target = preprocess_multilabel(input, target, num_labels, ignored_index=ignored_index)
        ret = tmf.classification.multilabel_f1_score(input, target, num_labels=num_labels, average=average, **kwargs)
        if task_weights is not None:
            return ret @ task_weights.double()
        return ret.mean()
    raise ValueError(f"Task should be one of binary, multiclass, or multilabel, but got {task}")


def accuracy(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    average: str | None = "micro",
    num_labels: int | None = None,
    num_classes: int | None = None,
    task: str | None = None,
    preprocess: bool = True,
    ignored_index: int | None = -100,
    **kwargs,
):
    te.check()
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        if preprocess:
            input, target = preprocess_binary(input, target, ignored_index=ignored_index)
        return tef.binary_accuracy(input=input, target=target, threshold=threshold, **kwargs)
    if num_labels is None:
        if preprocess:
            input, target = preprocess_multiclass(input, target, num_classes, ignored_index=ignored_index)
        return tef.multiclass_accuracy(input=input, target=target, num_classes=num_classes, average=average, **kwargs)
    if num_classes is None:
        if preprocess:
            input, target = preprocess_multilabel(input, target, num_labels, ignored_index=ignored_index)
        return tef.multilabel_accuracy(input=input, target=target, threshold=threshold, **kwargs)
    raise ValueError(f"Task should be one of binary, multiclass, or multilabel, but got {task}")


def mcc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    num_labels: int | None = None,
    num_classes: int | None = None,
    task: str | None = None,
    preprocess: bool = True,
    ignored_index: int | None = -100,
    **kwargs,
):
    tm.check()
    if task is None:
        task = infer_task(num_classes, num_labels)
    if preprocess:
        if task == "binary":
            input, target = preprocess_binary(input, target, ignored_index=ignored_index)
        elif task == "multiclass":
            input, target = preprocess_multiclass(input, target, num_classes, ignored_index=ignored_index)
        elif task == "multilabel":
            input, target = preprocess_multilabel(input, target, num_labels, ignored_index=ignored_index)
    try:
        return tmf.matthews_corrcoef(
            input, target, task, threshold=threshold, num_classes=num_classes, num_labels=num_labels, **kwargs
        )
    except:  # noqa
        return torch.tensor(0, dtype=float).to(input.device)


def pearson(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    num_outputs: int = 1,
    task_weights: Tensor | None = None,
    preprocess: bool = True,
    **kwargs,
):
    tm.check()
    if preprocess:
        input, target = preprocess_regression(input, target, num_outputs=num_outputs)
    try:
        ret = tmf.pearson_corrcoef(input, target, **kwargs)
        if task_weights is not None:
            return ret @ task_weights.double()
        return ret.mean()
    except ValueError:
        return torch.tensor(0, dtype=float).to(input.device)


def spearman(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    num_outputs: int = 1,
    task_weights: Tensor | None = None,
    preprocess: bool = True,
    **kwargs,
):
    tm.check()
    if preprocess:
        input, target = preprocess_regression(input, target, num_outputs=num_outputs)
    try:
        ret = tmf.spearman_corrcoef(input, target, **kwargs)
        if task_weights is not None:
            return ret @ task_weights.double()
        return ret.mean()
    except ValueError:
        return torch.tensor(0, dtype=float).to(input.device)


def r2_score(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    num_outputs: int = 1,
    task_weights: Tensor | None = None,
    multioutput: str = "raw_values",
    preprocess: bool = True,
    **kwargs,
):
    te.check()
    if preprocess:
        input, target = preprocess_regression(input, target, num_outputs=num_outputs)
    try:
        ret = tef.r2_score(input, target, multioutput=multioutput, **kwargs)
        if multioutput != "raw_values":
            return ret
        if task_weights is not None:
            return ret @ task_weights.double()
        return ret.mean()
    except ValueError:
        return torch.tensor(0, dtype=float).to(input.device)


def mse(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    num_outputs: int = 1,
    task_weights: Tensor | None = None,
    multioutput: str = "raw_values",
    preprocess: bool = True,
    **kwargs,
):
    te.check()
    if preprocess:
        input, target = preprocess_regression(input, target, num_outputs=num_outputs)
    ret = tef.mean_squared_error(input, target, **kwargs)
    if multioutput != "raw_values":
        return ret
    if task_weights is not None:
        return ret @ task_weights.double()
    return ret.mean()


def rmse(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    num_outputs: int = 1,
    task_weights: Tensor | None = None,
    multioutput: str = "raw_values",
    preprocess: bool = True,
    **kwargs,
):
    te.check()
    if preprocess:
        input, target = preprocess_regression(input, target, num_outputs=num_outputs)
    ret = tef.mean_squared_error(input, target, **kwargs).sqrt()
    if multioutput != "raw_values":
        return ret
    if task_weights is not None:
        return ret @ task_weights.double()
    return ret.mean()
