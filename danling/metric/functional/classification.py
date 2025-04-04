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
# mypy: disable-error-code="arg-type"
from __future__ import annotations

from typing import Callable

import torch
from lazy_imports import try_import
from torch import Tensor

from danling.tensor import NestedTensor

with try_import() as tm:
    from torchmetrics import functional as tmf

from warnings import warn

from .binary import binary_accuracy, binary_auprc, binary_auroc, binary_f1_score
from .multiclass import multiclass_accuracy, multiclass_auprc, multiclass_auroc, multiclass_f1_score
from .multilabel import multilabel_accuracy, multilabel_auprc, multilabel_auroc, multilabel_f1_score
from .preprocess import preprocess_classification, with_preprocess
from .utils import infer_task


def accuracy(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    average: str | None = "macro",
    num_labels: int | None = None,
    num_classes: int | None = None,
    task: str | None = None,
    preprocess: bool | Callable = True,
    ignore_index: int | None = -100,
    **kwargs,
):
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        return binary_accuracy(
            input, target, threshold=threshold, preprocess=preprocess, ignore_index=ignore_index, **kwargs
        )
    if task == "multiclass":
        return multiclass_accuracy(
            input,
            target,
            num_classes=num_classes,
            average=average,
            preprocess=preprocess,
            ignore_index=ignore_index,
            **kwargs,
        )
    if task == "multilabel":
        return multilabel_accuracy(
            input,
            target,
            num_labels=num_labels,
            preprocess=preprocess,
            ignore_index=ignore_index,
            **kwargs,
        )
    raise ValueError(f"Task should be one of binary, multiclass, or multilabel, but got {task}")


def auprc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    average: str | None = "macro",
    num_labels: int | None = None,
    num_classes: int | None = None,
    task: str | None = None,
    preprocess: bool | Callable = True,
    ignore_index: int | None = -100,
    **kwargs,
):
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        return binary_auprc(input, target, preprocess=preprocess, ignore_index=ignore_index, **kwargs)
    if task == "multiclass":
        return multiclass_auprc(
            input,
            target,
            num_classes=num_classes,
            average=average,
            preprocess=preprocess,
            ignore_index=ignore_index,
            **kwargs,
        )
    if task == "multilabel":
        return multilabel_auprc(
            input,
            target,
            num_labels=num_labels,
            average=average,
            preprocess=preprocess,
            ignore_index=ignore_index,
            **kwargs,
        )
    raise ValueError(f"Task should be one of binary, multiclass, or multilabel, but got {task}")


def auroc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    average: str | None = "macro",
    num_labels: int | None = None,
    num_classes: int | None = None,
    task: str | None = None,
    preprocess: bool | Callable = True,
    ignore_index: int | None = -100,
    **kwargs,
):
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        return binary_auroc(input, target, preprocess=preprocess, ignore_index=ignore_index, **kwargs)
    if task == "multiclass":
        return multiclass_auroc(
            input,
            target,
            num_classes=num_classes,
            average=average,
            preprocess=preprocess,
            ignore_index=ignore_index,
            **kwargs,
        )
    if task == "multilabel":
        return multilabel_auroc(
            input, target, num_labels=num_labels, preprocess=preprocess, ignore_index=ignore_index, **kwargs
        )
    raise ValueError(f"Task should be one of binary, multiclass, or multilabel, but got {task}")


def f1_score(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    average: str | None = "macro",
    num_labels: int | None = None,
    num_classes: int | None = None,
    task: str | None = None,
    preprocess: bool | Callable = True,
    ignore_index: int | None = -100,
    **kwargs,
):
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        return binary_f1_score(
            input, target, threshold=threshold, preprocess=preprocess, ignore_index=ignore_index, **kwargs
        )
    if task == "multiclass":
        return multiclass_f1_score(
            input,
            target,
            num_classes=num_classes,
            average=average,
            preprocess=preprocess,
            ignore_index=ignore_index,
            **kwargs,
        )
    if task == "multilabel":
        return multilabel_f1_score(
            input,
            target,
            num_labels=num_labels,
            average=average,
            preprocess=preprocess,
            ignore_index=ignore_index,
            **kwargs,
        )
    raise ValueError(f"Task should be one of binary, multiclass, or multilabel, but got {task}")


@with_preprocess(preprocess_classification, ignore_index=-100)
def mcc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    num_labels: int | None = None,
    num_classes: int | None = None,
    task: str | None = None,
    **kwargs,
):
    tm.check()
    if task is None:
        task = infer_task(num_classes, num_labels)
    try:
        return tmf.matthews_corrcoef(
            input, target, task, threshold=threshold, num_classes=num_classes, num_labels=num_labels, **kwargs
        )
    except Exception as e:  # noqa
        warn(f"{e} encountered will computing MCC with {input} and {target}")
        return torch.tensor(0, dtype=torch.float32).to(input.device)
