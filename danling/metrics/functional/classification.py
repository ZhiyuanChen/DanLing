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

from typing import TYPE_CHECKING

import torch
from lazy_imports import try_import
from torch import Tensor

from danling.tensors import NestedTensor

from .utils import infer_task

with try_import() as tm:
    from torchmetrics.functional import classification as tmcls

from .utils import MetricFunc

if TYPE_CHECKING:  # pragma: no cover
    from ..state import MetricState


class mcc(MetricFunc):
    def __init__(
        self,
        *,
        task: str = "binary",
        num_classes: int | None = None,
        num_labels: int | None = None,
        threshold: float = 0.5,
        ignore_index: int | None = -100,
        name: str | None = "mcc",
    ) -> None:
        self.task = task
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.threshold = threshold
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            preds_targets=True,
            task=task,
            num_classes=num_classes,
            num_labels=num_labels,
            threshold=threshold,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return torch.tensor(float("nan"))
        tm.check()
        return tmcls.matthews_corrcoef(
            state.preds,
            state.targets,
            task=self.task,
            threshold=self.threshold,
            num_classes=self.num_classes,
            num_labels=self.num_labels,
            ignore_index=self.ignore_index,
        )


def accuracy(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    average: str | None = "macro",
    k: int = 1,
    num_labels: int | None = None,
    num_classes: int | None = None,
    task: str | None = None,
    ignore_index: int | None = -100,
    **kwargs,
):
    tm.check()
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        return tmcls.binary_accuracy(input, target, threshold=threshold, ignore_index=ignore_index, **kwargs)
    if task == "multiclass":
        return tmcls.multiclass_accuracy(
            input,
            target,
            num_classes=num_classes,
            average=average,
            top_k=k,
            ignore_index=ignore_index,
            **kwargs,
        )
    if task == "multilabel":
        return tmcls.multilabel_accuracy(
            input,
            target,
            threshold=threshold,
            num_labels=num_labels,
            average=average,
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
    ignore_index: int | None = -100,
    **kwargs,
):
    tm.check()
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        return tmcls.binary_average_precision(input, target, ignore_index=ignore_index, **kwargs)
    if task == "multiclass":
        return tmcls.multiclass_average_precision(
            input,
            target,
            num_classes=num_classes,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
    if task == "multilabel":
        return tmcls.multilabel_average_precision(
            input,
            target,
            num_labels=num_labels,
            average=average,
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
    ignore_index: int | None = -100,
    **kwargs,
):
    tm.check()
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        return tmcls.binary_auroc(input, target, ignore_index=ignore_index, **kwargs)
    if task == "multiclass":
        return tmcls.multiclass_auroc(
            input,
            target,
            num_classes=num_classes,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
    if task == "multilabel":
        return tmcls.multilabel_auroc(
            input,
            target,
            num_labels=num_labels,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
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
    ignore_index: int | None = -100,
    **kwargs,
):
    return fbeta_score(
        input,
        target,
        beta=1.0,
        threshold=threshold,
        average=average,
        num_labels=num_labels,
        num_classes=num_classes,
        task=task,
        ignore_index=ignore_index,
        **kwargs,
    )


def fbeta_score(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    beta: float,
    threshold: float = 0.5,
    average: str | None = "macro",
    num_labels: int | None = None,
    num_classes: int | None = None,
    task: str | None = None,
    ignore_index: int | None = -100,
    **kwargs,
):
    tm.check()
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        return tmcls.binary_fbeta_score(
            input,
            target,
            beta=beta,
            threshold=threshold,
            ignore_index=ignore_index,
            **kwargs,
        )
    if task == "multiclass":
        return tmcls.multiclass_fbeta_score(
            input,
            target,
            beta=beta,
            num_classes=num_classes,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
    if task == "multilabel":
        return tmcls.multilabel_fbeta_score(
            input,
            target,
            beta=beta,
            threshold=threshold,
            num_labels=num_labels,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
    raise ValueError(f"Task should be one of binary, multiclass, or multilabel, but got {task}")


def precision(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    average: str | None = "macro",
    k: int = 1,
    num_labels: int | None = None,
    num_classes: int | None = None,
    task: str | None = None,
    ignore_index: int | None = -100,
    **kwargs,
):
    tm.check()
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        return tmcls.binary_precision(input, target, threshold=threshold, ignore_index=ignore_index, **kwargs)
    if task == "multiclass":
        return tmcls.multiclass_precision(
            input,
            target,
            num_classes=num_classes,
            average=average,
            top_k=k,
            ignore_index=ignore_index,
            **kwargs,
        )
    if task == "multilabel":
        return tmcls.multilabel_precision(
            input,
            target,
            threshold=threshold,
            num_labels=num_labels,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
    raise ValueError(f"Task should be one of binary, multiclass, or multilabel, but got {task}")


def recall(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    average: str | None = "macro",
    k: int = 1,
    num_labels: int | None = None,
    num_classes: int | None = None,
    task: str | None = None,
    ignore_index: int | None = -100,
    **kwargs,
):
    tm.check()
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        return tmcls.binary_recall(input, target, threshold=threshold, ignore_index=ignore_index, **kwargs)
    if task == "multiclass":
        return tmcls.multiclass_recall(
            input,
            target,
            num_classes=num_classes,
            average=average,
            top_k=k,
            ignore_index=ignore_index,
            **kwargs,
        )
    if task == "multilabel":
        return tmcls.multilabel_recall(
            input,
            target,
            threshold=threshold,
            num_labels=num_labels,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
    raise ValueError(f"Task should be one of binary, multiclass, or multilabel, but got {task}")


def specificity(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    average: str | None = "macro",
    k: int = 1,
    num_labels: int | None = None,
    num_classes: int | None = None,
    task: str | None = None,
    ignore_index: int | None = -100,
    **kwargs,
):
    tm.check()
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        return tmcls.binary_specificity(input, target, threshold=threshold, ignore_index=ignore_index, **kwargs)
    if task == "multiclass":
        return tmcls.multiclass_specificity(
            input,
            target,
            num_classes=num_classes,
            average=average,
            top_k=k,
            ignore_index=ignore_index,
            **kwargs,
        )
    if task == "multilabel":
        return tmcls.multilabel_specificity(
            input,
            target,
            threshold=threshold,
            num_labels=num_labels,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
    raise ValueError(f"Task should be one of binary, multiclass, or multilabel, but got {task}")


def balanced_accuracy(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    average: str | None = "macro",
    k: int = 1,
    num_labels: int | None = None,
    num_classes: int | None = None,
    task: str | None = None,
    ignore_index: int | None = -100,
    **kwargs,
):
    tm.check()
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        rec = tmcls.binary_recall(input, target, threshold=threshold, ignore_index=ignore_index, **kwargs)
        spec = tmcls.binary_specificity(input, target, threshold=threshold, ignore_index=ignore_index, **kwargs)
        return 0.5 * (rec + spec)
    if task == "multiclass":
        return tmcls.multiclass_recall(
            input,
            target,
            num_classes=num_classes,
            average=average,
            top_k=k,
            ignore_index=ignore_index,
            **kwargs,
        )
    if task == "multilabel":
        rec = tmcls.multilabel_recall(
            input,
            target,
            threshold=threshold,
            num_labels=num_labels,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
        spec = tmcls.multilabel_specificity(
            input,
            target,
            threshold=threshold,
            num_labels=num_labels,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
        return 0.5 * (rec + spec)
    raise ValueError(f"Task should be one of binary, multiclass, or multilabel, but got {task}")


def jaccard_index(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    average: str | None = "macro",
    num_labels: int | None = None,
    num_classes: int | None = None,
    task: str | None = None,
    ignore_index: int | None = -100,
    **kwargs,
):
    tm.check()
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        return tmcls.binary_jaccard_index(input, target, threshold=threshold, ignore_index=ignore_index, **kwargs)
    if task == "multiclass":
        return tmcls.multiclass_jaccard_index(
            input,
            target,
            num_classes=num_classes,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
    if task == "multilabel":
        return tmcls.multilabel_jaccard_index(
            input,
            target,
            threshold=threshold,
            num_labels=num_labels,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
    raise ValueError(f"Task should be one of binary, multiclass, or multilabel, but got {task}")


def iou(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    average: str | None = "macro",
    num_labels: int | None = None,
    num_classes: int | None = None,
    task: str | None = None,
    ignore_index: int | None = -100,
    **kwargs,
):
    return jaccard_index(
        input,
        target,
        threshold=threshold,
        average=average,
        num_labels=num_labels,
        num_classes=num_classes,
        task=task,
        ignore_index=ignore_index,
        **kwargs,
    )


def hamming_loss(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    average: str | None = "macro",
    k: int = 1,
    num_labels: int | None = None,
    num_classes: int | None = None,
    task: str | None = None,
    ignore_index: int | None = -100,
    **kwargs,
):
    tm.check()
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task == "binary":
        return tmcls.binary_hamming_distance(input, target, threshold=threshold, ignore_index=ignore_index, **kwargs)
    if task == "multiclass":
        return tmcls.multiclass_hamming_distance(
            input,
            target,
            num_classes=num_classes,
            average=average,
            top_k=k,
            ignore_index=ignore_index,
            **kwargs,
        )
    if task == "multilabel":
        return tmcls.multilabel_hamming_distance(
            input,
            target,
            threshold=threshold,
            num_labels=num_labels,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
    raise ValueError(f"Task should be one of binary, multiclass, or multilabel, but got {task}")
