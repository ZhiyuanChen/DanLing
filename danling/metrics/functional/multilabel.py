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

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import torch
from lazy_imports import try_import
from torch import Tensor

from .utils import MetricFunc, reduce_fbeta_metric, reduce_precomputed_metric, reduce_ratio_metric

with try_import() as tm:
    from torchmetrics.functional import classification as tmcls

if TYPE_CHECKING:  # pragma: no cover
    from ..state import MetricState


def _empty_state(state: MetricState) -> bool:
    return state.preds.numel() == 0 or state.targets.numel() == 0


def _call_tm_multilabel(
    state: MetricState,
    metric_fn: Callable[..., Tensor],
    *,
    num_labels: int | None,
    ignore_index: int | None,
    threshold: float | None = None,
    average: str | None = None,
    use_threshold: bool = False,
    use_average: bool = False,
    beta: float | None = None,
) -> Tensor:
    if _empty_state(state):
        return torch.tensor(float("nan"))
    if num_labels is None:
        raise ValueError("num_labels must be set for multilabel metrics.")

    tm.check()
    kwargs: dict[str, object] = {"num_labels": num_labels, "ignore_index": ignore_index}
    if use_threshold:
        kwargs["threshold"] = threshold
    if use_average:
        kwargs["average"] = average
    if beta is not None:
        kwargs["beta"] = beta
    return metric_fn(state.preds, state.targets, **kwargs)


class multilabel_accuracy(MetricFunc):
    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: str | None = "macro",
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "acc",
    ) -> None:
        self.num_labels = num_labels
        self.threshold = threshold
        self.average = average
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            confmat=True,
            task="multilabel",
            num_labels=num_labels,
            threshold=threshold,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is None:
            return _call_tm_multilabel(
                state,
                tmcls.multilabel_accuracy,
                num_labels=self.num_labels,
                ignore_index=self.ignore_index,
                threshold=self.threshold,
                average=self.average,
                use_threshold=True,
                use_average=True,
            )

        tp, fp, fn, tn, support, total = state.multilabel_stats
        return reduce_ratio_metric(tp + tn, total, self.average, support)


class multilabel_auprc(MetricFunc):
    def __init__(
        self,
        num_labels: int,
        average: str | None = "macro",
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "auprc",
    ) -> None:
        self.num_labels = num_labels
        self.average = average
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            preds_targets=True,
            task="multilabel",
            num_labels=num_labels,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        return _call_tm_multilabel(
            state,
            tmcls.multilabel_average_precision,
            num_labels=self.num_labels,
            ignore_index=self.ignore_index,
            average=self.average,
            use_average=True,
        )


class multilabel_auroc(MetricFunc):
    def __init__(
        self,
        num_labels: int,
        average: str | None = "macro",
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "auroc",
    ) -> None:
        self.num_labels = num_labels
        self.average = average
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            preds_targets=True,
            task="multilabel",
            num_labels=num_labels,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        return _call_tm_multilabel(
            state,
            tmcls.multilabel_auroc,
            num_labels=self.num_labels,
            ignore_index=self.ignore_index,
            average=self.average,
            use_average=True,
        )


class multilabel_precision(MetricFunc):
    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: str | None = "macro",
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "precision",
    ) -> None:
        self.num_labels = num_labels
        self.threshold = threshold
        self.average = average
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            confmat=True,
            task="multilabel",
            num_labels=num_labels,
            threshold=threshold,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is None:
            return _call_tm_multilabel(
                state,
                tmcls.multilabel_precision,
                num_labels=self.num_labels,
                ignore_index=self.ignore_index,
                threshold=self.threshold,
                average=self.average,
                use_threshold=True,
                use_average=True,
            )

        tp, fp, _, _, support, _ = state.multilabel_stats
        return reduce_ratio_metric(tp, tp + fp, self.average, support)


class multilabel_recall(MetricFunc):
    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: str | None = "macro",
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "recall",
    ) -> None:
        self.num_labels = num_labels
        self.threshold = threshold
        self.average = average
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            confmat=True,
            task="multilabel",
            num_labels=num_labels,
            threshold=threshold,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is None:
            return _call_tm_multilabel(
                state,
                tmcls.multilabel_recall,
                num_labels=self.num_labels,
                ignore_index=self.ignore_index,
                threshold=self.threshold,
                average=self.average,
                use_threshold=True,
                use_average=True,
            )

        tp, _, fn, _, support, _ = state.multilabel_stats
        return reduce_ratio_metric(tp, tp + fn, self.average, support)


class multilabel_specificity(MetricFunc):
    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: str | None = "macro",
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "specificity",
    ) -> None:
        self.num_labels = num_labels
        self.threshold = threshold
        self.average = average
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            confmat=True,
            task="multilabel",
            num_labels=num_labels,
            threshold=threshold,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is None:
            return _call_tm_multilabel(
                state,
                tmcls.multilabel_specificity,
                num_labels=self.num_labels,
                ignore_index=self.ignore_index,
                threshold=self.threshold,
                average=self.average,
                use_threshold=True,
                use_average=True,
            )

        _, fp, _, tn, support, _ = state.multilabel_stats
        return reduce_ratio_metric(tn, tn + fp, self.average, support)


class multilabel_balanced_accuracy(MetricFunc):
    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: str | None = "macro",
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "balanced_accuracy",
    ) -> None:
        self.num_labels = num_labels
        self.threshold = threshold
        self.average = average
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            confmat=True,
            task="multilabel",
            num_labels=num_labels,
            threshold=threshold,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is None:
            recall = _call_tm_multilabel(
                state,
                tmcls.multilabel_recall,
                num_labels=self.num_labels,
                ignore_index=self.ignore_index,
                threshold=self.threshold,
                average=self.average,
                use_threshold=True,
                use_average=True,
            )
            specificity = _call_tm_multilabel(
                state,
                tmcls.multilabel_specificity,
                num_labels=self.num_labels,
                ignore_index=self.ignore_index,
                threshold=self.threshold,
                average=self.average,
                use_threshold=True,
                use_average=True,
            )
            return 0.5 * (recall + specificity)

        tp, fp, fn, tn, support, _ = state.multilabel_stats
        recall = tp / (tp + fn).clamp(min=1)
        specificity = tn / (tn + fp).clamp(min=1)
        balanced = 0.5 * (recall + specificity)
        return reduce_precomputed_metric(
            balanced,
            self.average,
            support,
            micro_value=0.5 * (tp.sum() / (tp + fn).sum().clamp(min=1) + tn.sum() / (tn + fp).sum().clamp(min=1)),
        )


class multilabel_jaccard_index(MetricFunc):
    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: str | None = "macro",
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "jaccard",
    ) -> None:
        self.num_labels = num_labels
        self.threshold = threshold
        self.average = average
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            confmat=True,
            task="multilabel",
            num_labels=num_labels,
            threshold=threshold,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is None:
            return _call_tm_multilabel(
                state,
                tmcls.multilabel_jaccard_index,
                num_labels=self.num_labels,
                ignore_index=self.ignore_index,
                threshold=self.threshold,
                average=self.average,
                use_threshold=True,
                use_average=True,
            )

        tp, fp, fn, _, support, _ = state.multilabel_stats
        return reduce_ratio_metric(tp, tp + fp + fn, self.average, support)


class multilabel_iou(multilabel_jaccard_index):
    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: str | None = "macro",
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "iou",
    ) -> None:
        super().__init__(
            num_labels=num_labels,
            threshold=threshold,
            average=average,
            ignore_index=ignore_index,
            name=name,
        )


class multilabel_hamming_loss(MetricFunc):
    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: str | None = "macro",
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "hamming_loss",
    ) -> None:
        self.num_labels = num_labels
        self.threshold = threshold
        self.average = average
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            confmat=True,
            task="multilabel",
            num_labels=num_labels,
            threshold=threshold,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is None:
            return _call_tm_multilabel(
                state,
                tmcls.multilabel_hamming_distance,
                num_labels=self.num_labels,
                ignore_index=self.ignore_index,
                threshold=self.threshold,
                average=self.average,
                use_threshold=True,
                use_average=True,
            )

        _, fp, fn, _, support, total = state.multilabel_stats
        return reduce_ratio_metric(fp + fn, total, self.average, support)


class multilabel_fbeta_score(MetricFunc):
    def __init__(
        self,
        num_labels: int,
        beta: float = 1.0,
        threshold: float = 0.5,
        average: str | None = "macro",
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "fbeta",
    ) -> None:
        self.num_labels = num_labels
        self.beta = beta
        self.threshold = threshold
        self.average = average
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            confmat=True,
            task="multilabel",
            num_labels=num_labels,
            threshold=threshold,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is None:
            return _call_tm_multilabel(
                state,
                tmcls.multilabel_fbeta_score,
                num_labels=self.num_labels,
                ignore_index=self.ignore_index,
                threshold=self.threshold,
                average=self.average,
                use_threshold=True,
                use_average=True,
                beta=self.beta,
            )

        tp, fp, fn, _, support, _ = state.multilabel_stats
        return reduce_fbeta_metric(tp, fp, fn, self.beta, self.average, support)


class multilabel_f1_score(multilabel_fbeta_score):
    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: str | None = "macro",
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "f1",
    ) -> None:
        super().__init__(
            num_labels=num_labels,
            beta=1.0,
            threshold=threshold,
            average=average,
            ignore_index=ignore_index,
            name=name,
        )
