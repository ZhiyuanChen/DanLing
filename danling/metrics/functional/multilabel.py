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

from .utils import MetricFunc

with try_import() as tm:
    from torchmetrics.functional import classification as tmcls

if TYPE_CHECKING:  # pragma: no cover
    from ..state import MetricState


def _multilabel_stats(confmat: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    confmat = confmat.to(torch.float)
    tn = confmat[:, 0, 0]
    fp = confmat[:, 0, 1]
    fn = confmat[:, 1, 0]
    tp = confmat[:, 1, 1]
    support = tp + fn
    total = tp + tn + fp + fn
    return tp, fp, fn, tn, support, total


def _reduce(
    values: Tensor,
    average: str | None,
    support: Tensor,
    *,
    micro_num: Tensor,
    micro_den: Tensor,
) -> Tensor:
    if average in (None, "none"):
        return values
    if average == "macro":
        return values.mean()
    if average == "weighted":
        return (values * support).sum() / support.sum().clamp(min=1)
    if average == "micro":
        return micro_num / micro_den.clamp(min=1)
    raise ValueError(f"Invalid average value: {average!r}")


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

        tp, fp, fn, tn, support, total = _multilabel_stats(confmat)
        acc = (tp + tn) / total.clamp(min=1)
        return _reduce(
            acc,
            self.average,
            support,
            micro_num=(tp + tn).sum(),
            micro_den=total.sum(),
        )


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

        tp, fp, _, _, support, _ = _multilabel_stats(confmat)
        precision = tp / (tp + fp).clamp(min=1)
        return _reduce(
            precision,
            self.average,
            support,
            micro_num=tp.sum(),
            micro_den=(tp + fp).sum(),
        )


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

        tp, _, fn, _, support, _ = _multilabel_stats(confmat)
        recall = tp / (tp + fn).clamp(min=1)
        return _reduce(
            recall,
            self.average,
            support,
            micro_num=tp.sum(),
            micro_den=(tp + fn).sum(),
        )


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

        _, fp, _, tn, support, _ = _multilabel_stats(confmat)
        specificity = tn / (tn + fp).clamp(min=1)
        return _reduce(
            specificity,
            self.average,
            support,
            micro_num=tn.sum(),
            micro_den=(tn + fp).sum(),
        )


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

        tp, fp, fn, tn, support, _ = _multilabel_stats(confmat)
        recall = tp / (tp + fn).clamp(min=1)
        specificity = tn / (tn + fp).clamp(min=1)
        balanced = 0.5 * (recall + specificity)
        return _reduce(
            balanced,
            self.average,
            support,
            micro_num=0.5 * (tp.sum() / (tp + fn).sum().clamp(min=1) + tn.sum() / (tn + fp).sum().clamp(min=1)),
            micro_den=torch.tensor(1.0, device=balanced.device),
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

        tp, fp, fn, _, support, _ = _multilabel_stats(confmat)
        iou = tp / (tp + fp + fn).clamp(min=1)
        return _reduce(
            iou,
            self.average,
            support,
            micro_num=tp.sum(),
            micro_den=(tp + fp + fn).sum(),
        )


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

        _, fp, fn, _, support, total = _multilabel_stats(confmat)
        hamming = (fp + fn) / total.clamp(min=1)
        return _reduce(
            hamming,
            self.average,
            support,
            micro_num=(fp + fn).sum(),
            micro_den=total.sum(),
        )


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

        tp, fp, fn, _, support, _ = _multilabel_stats(confmat)
        beta_square = self.beta * self.beta
        numerator = (1.0 + beta_square) * tp
        denominator = numerator + beta_square * fn + fp
        fbeta = numerator / denominator.clamp(min=1)
        return _reduce(
            fbeta,
            self.average,
            support,
            micro_num=numerator.sum(),
            micro_den=denominator.sum(),
        )


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
