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

from .utils import infer_task, require_standard_multiclass_balanced_accuracy

with try_import() as tm:
    from torchmetrics.functional import classification as tmcls

from .utils import MetricFunc

if TYPE_CHECKING:  # pragma: no cover
    from ..state import MetricState


def _resolve_task(task: str | None, num_classes: int | None, num_labels: int | None) -> str:
    if task is None:
        task = infer_task(num_classes, num_labels)
    if task not in {"binary", "multiclass", "multilabel"}:
        raise ValueError(f"Task should be one of binary, multiclass, or multilabel, but got {task}")
    return task


def _preprocess_for_task(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    *,
    task: str,
    num_classes: int | None,
    num_labels: int | None,
    ignore_index: int | None,
) -> tuple[Tensor, Tensor]:
    from ..preprocess import preprocess_binary, preprocess_multiclass, preprocess_multilabel

    if task == "binary":
        return preprocess_binary(input, target, ignore_index=ignore_index)
    if task == "multiclass":
        if num_classes is None:
            raise ValueError("num_classes is required for multiclass metrics")
        return preprocess_multiclass(input, target, num_classes=num_classes, ignore_index=ignore_index)
    if num_labels is None:
        raise ValueError("num_labels is required for multilabel metrics")
    return preprocess_multilabel(input, target, num_labels=num_labels, ignore_index=ignore_index)


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
    resolved_task = _resolve_task(task, num_classes, num_labels)
    preds, targets = _preprocess_for_task(
        input,
        target,
        task=resolved_task,
        num_classes=num_classes,
        num_labels=num_labels,
        ignore_index=ignore_index,
    )
    if resolved_task == "binary":
        return tmcls.binary_accuracy(preds, targets, threshold=threshold, ignore_index=ignore_index, **kwargs)
    if resolved_task == "multiclass":
        return tmcls.multiclass_accuracy(
            preds,
            targets,
            num_classes=num_classes,
            average=average,
            top_k=k,
            ignore_index=ignore_index,
            **kwargs,
        )
    return tmcls.multilabel_accuracy(
        preds,
        targets,
        threshold=threshold,
        num_labels=num_labels,
        average=average,
        ignore_index=ignore_index,
        **kwargs,
    )


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
    resolved_task = _resolve_task(task, num_classes, num_labels)
    preds, targets = _preprocess_for_task(
        input,
        target,
        task=resolved_task,
        num_classes=num_classes,
        num_labels=num_labels,
        ignore_index=ignore_index,
    )
    if resolved_task == "binary":
        return tmcls.binary_average_precision(preds, targets, ignore_index=ignore_index, **kwargs)
    if resolved_task == "multiclass":
        return tmcls.multiclass_average_precision(
            preds,
            targets,
            num_classes=num_classes,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
    return tmcls.multilabel_average_precision(
        preds,
        targets,
        num_labels=num_labels,
        average=average,
        ignore_index=ignore_index,
        **kwargs,
    )


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
    resolved_task = _resolve_task(task, num_classes, num_labels)
    preds, targets = _preprocess_for_task(
        input,
        target,
        task=resolved_task,
        num_classes=num_classes,
        num_labels=num_labels,
        ignore_index=ignore_index,
    )
    if resolved_task == "binary":
        return tmcls.binary_auroc(preds, targets, ignore_index=ignore_index, **kwargs)
    if resolved_task == "multiclass":
        return tmcls.multiclass_auroc(
            preds,
            targets,
            num_classes=num_classes,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
    return tmcls.multilabel_auroc(
        preds,
        targets,
        num_labels=num_labels,
        average=average,
        ignore_index=ignore_index,
        **kwargs,
    )


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
    resolved_task = _resolve_task(task, num_classes, num_labels)
    preds, targets = _preprocess_for_task(
        input,
        target,
        task=resolved_task,
        num_classes=num_classes,
        num_labels=num_labels,
        ignore_index=ignore_index,
    )
    if resolved_task == "binary":
        return tmcls.binary_fbeta_score(
            preds,
            targets,
            beta=beta,
            threshold=threshold,
            ignore_index=ignore_index,
            **kwargs,
        )
    if resolved_task == "multiclass":
        return tmcls.multiclass_fbeta_score(
            preds,
            targets,
            beta=beta,
            num_classes=num_classes,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
    return tmcls.multilabel_fbeta_score(
        preds,
        targets,
        beta=beta,
        threshold=threshold,
        num_labels=num_labels,
        average=average,
        ignore_index=ignore_index,
        **kwargs,
    )


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
    resolved_task = _resolve_task(task, num_classes, num_labels)
    preds, targets = _preprocess_for_task(
        input,
        target,
        task=resolved_task,
        num_classes=num_classes,
        num_labels=num_labels,
        ignore_index=ignore_index,
    )
    if resolved_task == "binary":
        return tmcls.binary_precision(preds, targets, threshold=threshold, ignore_index=ignore_index, **kwargs)
    if resolved_task == "multiclass":
        return tmcls.multiclass_precision(
            preds,
            targets,
            num_classes=num_classes,
            average=average,
            top_k=k,
            ignore_index=ignore_index,
            **kwargs,
        )
    return tmcls.multilabel_precision(
        preds,
        targets,
        threshold=threshold,
        num_labels=num_labels,
        average=average,
        ignore_index=ignore_index,
        **kwargs,
    )


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
    resolved_task = _resolve_task(task, num_classes, num_labels)
    preds, targets = _preprocess_for_task(
        input,
        target,
        task=resolved_task,
        num_classes=num_classes,
        num_labels=num_labels,
        ignore_index=ignore_index,
    )
    if resolved_task == "binary":
        return tmcls.binary_recall(preds, targets, threshold=threshold, ignore_index=ignore_index, **kwargs)
    if resolved_task == "multiclass":
        return tmcls.multiclass_recall(
            preds,
            targets,
            num_classes=num_classes,
            average=average,
            top_k=k,
            ignore_index=ignore_index,
            **kwargs,
        )
    return tmcls.multilabel_recall(
        preds,
        targets,
        threshold=threshold,
        num_labels=num_labels,
        average=average,
        ignore_index=ignore_index,
        **kwargs,
    )


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
    resolved_task = _resolve_task(task, num_classes, num_labels)
    preds, targets = _preprocess_for_task(
        input,
        target,
        task=resolved_task,
        num_classes=num_classes,
        num_labels=num_labels,
        ignore_index=ignore_index,
    )
    if resolved_task == "binary":
        return tmcls.binary_specificity(
            preds,
            targets,
            threshold=threshold,
            ignore_index=ignore_index,
            **kwargs,
        )
    if resolved_task == "multiclass":
        return tmcls.multiclass_specificity(
            preds,
            targets,
            num_classes=num_classes,
            average=average,
            top_k=k,
            ignore_index=ignore_index,
            **kwargs,
        )
    return tmcls.multilabel_specificity(
        preds,
        targets,
        threshold=threshold,
        num_labels=num_labels,
        average=average,
        ignore_index=ignore_index,
        **kwargs,
    )


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
    resolved_task = _resolve_task(task, num_classes, num_labels)
    preds, targets = _preprocess_for_task(
        input,
        target,
        task=resolved_task,
        num_classes=num_classes,
        num_labels=num_labels,
        ignore_index=ignore_index,
    )
    if resolved_task == "binary":
        return 0.5 * (
            tmcls.binary_recall(preds, targets, threshold=threshold, ignore_index=ignore_index, **kwargs)
            + tmcls.binary_specificity(preds, targets, threshold=threshold, ignore_index=ignore_index, **kwargs)
        )
    if resolved_task == "multiclass":
        require_standard_multiclass_balanced_accuracy(average, k)
        return tmcls.multiclass_recall(
            preds,
            targets,
            num_classes=num_classes,
            average=average,
            top_k=k,
            ignore_index=ignore_index,
            **kwargs,
        )
    return 0.5 * (
        tmcls.multilabel_recall(
            preds,
            targets,
            threshold=threshold,
            num_labels=num_labels,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
        + tmcls.multilabel_specificity(
            preds,
            targets,
            threshold=threshold,
            num_labels=num_labels,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
    )


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
    resolved_task = _resolve_task(task, num_classes, num_labels)
    preds, targets = _preprocess_for_task(
        input,
        target,
        task=resolved_task,
        num_classes=num_classes,
        num_labels=num_labels,
        ignore_index=ignore_index,
    )
    if resolved_task == "binary":
        return tmcls.binary_jaccard_index(preds, targets, threshold=threshold, ignore_index=ignore_index, **kwargs)
    if resolved_task == "multiclass":
        return tmcls.multiclass_jaccard_index(
            preds,
            targets,
            num_classes=num_classes,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
    return tmcls.multilabel_jaccard_index(
        preds,
        targets,
        threshold=threshold,
        num_labels=num_labels,
        average=average,
        ignore_index=ignore_index,
        **kwargs,
    )


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
    resolved_task = _resolve_task(task, num_classes, num_labels)
    preds, targets = _preprocess_for_task(
        input,
        target,
        task=resolved_task,
        num_classes=num_classes,
        num_labels=num_labels,
        ignore_index=ignore_index,
    )
    if resolved_task == "binary":
        return tmcls.binary_hamming_distance(
            preds,
            targets,
            threshold=threshold,
            ignore_index=ignore_index,
            **kwargs,
        )
    if resolved_task == "multiclass":
        return tmcls.multiclass_hamming_distance(
            preds,
            targets,
            num_classes=num_classes,
            average=average,
            top_k=k,
            ignore_index=ignore_index,
            **kwargs,
        )
    return tmcls.multilabel_hamming_distance(
        preds,
        targets,
        threshold=threshold,
        num_labels=num_labels,
        average=average,
        ignore_index=ignore_index,
        **kwargs,
    )
