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

from typing import TYPE_CHECKING, Optional

import torch
from lazy_imports import try_import
from torch import Tensor

from .utils import (
    MetricFunc,
    reduce_fbeta_metric,
    reduce_precomputed_metric,
    reduce_ratio_metric,
    require_standard_multiclass_balanced_accuracy,
)

with try_import() as te:
    from torchmetrics.functional import classification as tmcls

if TYPE_CHECKING:  # pragma: no cover
    from ..state import MetricState


def _empty_state(state: MetricState) -> bool:
    return state.preds.numel() == 0 or state.targets.numel() == 0


def _present_mask(support: Tensor, fp: Tensor) -> Tensor:
    return (support + fp) > 0


def _call_tm_multiclass(
    state: MetricState,
    metric_fn,
    *,
    num_classes: int | None,
    ignore_index: int | None,
    average: str | None = None,
    top_k: int | None = None,
    use_average: bool = False,
    use_top_k: bool = False,
    beta: float | None = None,
) -> Tensor:
    if _empty_state(state):
        return torch.tensor(float("nan"))
    if num_classes is None:
        raise ValueError("num_classes is required for multiclass metrics")

    te.check()
    kwargs: dict[str, object] = {"num_classes": num_classes, "ignore_index": ignore_index}
    if use_average:
        kwargs["average"] = average
    if use_top_k:
        kwargs["top_k"] = top_k
    if beta is not None:
        kwargs["beta"] = beta
    return metric_fn(state.preds, state.targets, **kwargs)


class multiclass_accuracy(MetricFunc):
    def __init__(
        self,
        num_classes: int,
        average: str | None = "macro",
        k: int = 1,
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "acc",
    ) -> None:
        self.num_classes = num_classes
        self.average = average
        self.k = k
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            preds_targets=k != 1,
            confmat=True,
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is not None and self.k == 1:
            tp, fp, fn, _, support, _ = state.multiclass_stats
            return reduce_ratio_metric(tp, tp + fn, self.average, support, present=_present_mask(support, fp))

        return _call_tm_multiclass(
            state,
            tmcls.multiclass_accuracy,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average=self.average,
            top_k=self.k,
            use_average=True,
            use_top_k=True,
        )


class multiclass_auprc(MetricFunc):
    def __init__(
        self,
        num_classes: int,
        average: str | None = "macro",
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "auprc",
    ) -> None:
        self.num_classes = num_classes
        self.average = average
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            preds_targets=True,
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        return _call_tm_multiclass(
            state,
            tmcls.multiclass_average_precision,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average=self.average,
            use_average=True,
        )


class multiclass_auroc(MetricFunc):
    def __init__(
        self,
        num_classes: int,
        average: str | None = "macro",
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "auroc",
    ) -> None:
        self.num_classes = num_classes
        self.average = average
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            preds_targets=True,
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        return _call_tm_multiclass(
            state,
            tmcls.multiclass_auroc,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average=self.average,
            use_average=True,
        )


class multiclass_precision(MetricFunc):
    def __init__(
        self,
        num_classes: int,
        average: str | None = "macro",
        k: int = 1,
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "precision",
    ) -> None:
        self.num_classes = num_classes
        self.average = average
        self.k = k
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            preds_targets=k != 1,
            confmat=True,
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is not None and self.k == 1:
            tp, fp, _, _, support, _ = state.multiclass_stats
            return reduce_ratio_metric(tp, tp + fp, self.average, support, present=_present_mask(support, fp))

        return _call_tm_multiclass(
            state,
            tmcls.multiclass_precision,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average=self.average,
            top_k=self.k,
            use_average=True,
            use_top_k=True,
        )


class multiclass_recall(MetricFunc):
    def __init__(
        self,
        num_classes: int,
        average: str | None = "macro",
        k: int = 1,
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "recall",
    ) -> None:
        self.num_classes = num_classes
        self.average = average
        self.k = k
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            preds_targets=k != 1,
            confmat=True,
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is not None and self.k == 1:
            tp, fp, fn, _, support, _ = state.multiclass_stats
            return reduce_ratio_metric(tp, tp + fn, self.average, support, present=_present_mask(support, fp))

        return _call_tm_multiclass(
            state,
            tmcls.multiclass_recall,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average=self.average,
            top_k=self.k,
            use_average=True,
            use_top_k=True,
        )


class multiclass_specificity(MetricFunc):
    def __init__(
        self,
        num_classes: int,
        average: str | None = "macro",
        k: int = 1,
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "specificity",
    ) -> None:
        self.num_classes = num_classes
        self.average = average
        self.k = k
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            preds_targets=k != 1,
            confmat=True,
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is not None and self.k == 1:
            _, fp, _, tn, support, _ = state.multiclass_stats
            return reduce_ratio_metric(tn, tn + fp, self.average, support, present=_present_mask(support, fp))

        return _call_tm_multiclass(
            state,
            tmcls.multiclass_specificity,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average=self.average,
            top_k=self.k,
            use_average=True,
            use_top_k=True,
        )


class multiclass_balanced_accuracy(MetricFunc):
    """
    Metric function version of multiclass balanced accuracy.

    For multiclass classification, balanced accuracy is the class-balanced recall.
    Only the standard multiclass definition is supported: `average="macro"` with `k=1`.
    """

    def __init__(
        self,
        num_classes: int,
        average: str | None = "macro",
        k: int = 1,
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "balanced_accuracy",
    ) -> None:
        require_standard_multiclass_balanced_accuracy(average, k)
        self.num_classes = num_classes
        self.average = average
        self.k = k
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            preds_targets=False,
            confmat=True,
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is not None:
            tp, fp, fn, _, support, _ = state.multiclass_stats
            return reduce_ratio_metric(tp, tp + fn, self.average, support, present=_present_mask(support, fp))

        return _call_tm_multiclass(
            state,
            tmcls.multiclass_recall,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average=self.average,
            top_k=self.k,
            use_average=True,
            use_top_k=True,
        )


class multiclass_jaccard_index(MetricFunc):
    def __init__(
        self,
        num_classes: int,
        average: str | None = "macro",
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "jaccard",
    ) -> None:
        self.num_classes = num_classes
        self.average = average
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            confmat=True,
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is None:
            return _call_tm_multiclass(
                state,
                tmcls.multiclass_jaccard_index,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
                average=self.average,
                use_average=True,
            )

        tp, fp, fn, _, support, _ = state.multiclass_stats
        return reduce_ratio_metric(tp, tp + fp + fn, self.average, support, present=_present_mask(support, fp))


class multiclass_iou(multiclass_jaccard_index):
    def __init__(
        self,
        num_classes: int,
        average: str | None = "macro",
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "iou",
    ) -> None:
        super().__init__(num_classes=num_classes, average=average, ignore_index=ignore_index, name=name)


class multiclass_hamming_loss(MetricFunc):
    def __init__(
        self,
        num_classes: int,
        average: str | None = "macro",
        k: int = 1,
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "hamming_loss",
    ) -> None:
        self.num_classes = num_classes
        self.average = average
        self.k = k
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            preds_targets=k != 1,
            confmat=True,
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is not None and self.k == 1:
            tp, fp, fn, _, support, total = state.multiclass_stats
            hamming = 1.0 - (tp / (tp + fn).clamp(min=1))
            return reduce_precomputed_metric(
                hamming,
                self.average,
                support,
                micro_value=fn.sum() / total.clamp(min=1),
                present=_present_mask(support, fp),
            )

        return _call_tm_multiclass(
            state,
            tmcls.multiclass_hamming_distance,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average=self.average,
            top_k=self.k,
            use_average=True,
            use_top_k=True,
        )


class multiclass_fbeta_score(MetricFunc):
    def __init__(
        self,
        num_classes: int,
        beta: float = 1.0,
        average: str | None = "macro",
        k: int = 1,
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "fbeta",
    ) -> None:
        self.num_classes = num_classes
        self.beta = beta
        self.average = average
        self.k = k
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            preds_targets=k != 1,
            confmat=True,
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is not None and self.k == 1:
            tp, fp, fn, _, support, _ = state.multiclass_stats
            return reduce_fbeta_metric(
                tp,
                fp,
                fn,
                self.beta,
                self.average,
                support,
                present=_present_mask(support, fp),
            )

        return _call_tm_multiclass(
            state,
            tmcls.multiclass_fbeta_score,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average=self.average,
            top_k=self.k,
            use_average=True,
            use_top_k=True,
            beta=self.beta,
        )


class multiclass_f1_score(multiclass_fbeta_score):
    def __init__(
        self,
        num_classes: int,
        average: str | None = "macro",
        k: int = 1,
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "f1",
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            beta=1.0,
            average=average,
            k=k,
            ignore_index=ignore_index,
            name=name,
        )
