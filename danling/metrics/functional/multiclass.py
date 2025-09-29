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

from .utils import MetricFunc

with try_import() as te:
    from torchmetrics.functional import classification as tmcls

if TYPE_CHECKING:  # pragma: no cover
    from ..state import MetricState


def _multiclass_stats(confmat: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    confmat = confmat.to(torch.float)
    tp = torch.diag(confmat)
    support = confmat.sum(dim=1)
    pred_support = confmat.sum(dim=0)
    total = confmat.sum()
    fn = support - tp
    fp = pred_support - tp
    tn = total - (tp + fp + fn)
    return tp, fp, fn, tn, support, total


def _reduce(
    values: Tensor,
    average: str | None,
    support: Tensor,
    *,
    micro_num: Tensor,
    micro_den: Tensor,
    present: Tensor | None = None,
) -> Tensor:
    if average in (None, "none"):
        return values
    if average == "macro":
        if present is None:
            return values.mean()
        if present.any():
            return values[present].mean()
        return torch.zeros((), dtype=values.dtype, device=values.device)
    if average == "weighted":
        return (values * support).sum() / support.sum().clamp(min=1)
    if average == "micro":
        return micro_num / micro_den.clamp(min=1)
    raise ValueError(f"Invalid average value: {average!r}")


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
            tp, fp, fn, _, support, total = _multiclass_stats(confmat)
            acc = tp / (tp + fn).clamp(min=1)
            present = (support + fp) > 0
            return _reduce(acc, self.average, support, micro_num=tp.sum(), micro_den=total, present=present)

        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return torch.tensor(float("nan"))
        te.check()
        return tmcls.multiclass_accuracy(
            state.preds,
            state.targets,
            num_classes=self.num_classes,
            average=self.average,
            top_k=self.k,
            ignore_index=self.ignore_index,
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
        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return torch.tensor(float("nan"))
        te.check()
        return tmcls.multiclass_average_precision(
            state.preds,
            state.targets,
            num_classes=self.num_classes,
            average=self.average,
            ignore_index=self.ignore_index,
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
        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return torch.tensor(float("nan"))
        te.check()
        return tmcls.multiclass_auroc(
            state.preds,
            state.targets,
            num_classes=self.num_classes,
            average=self.average,
            ignore_index=self.ignore_index,
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
            tp, fp, _, _, support, _ = _multiclass_stats(confmat)
            precision = tp / (tp + fp).clamp(min=1)
            present = (support + fp) > 0
            return _reduce(
                precision,
                self.average,
                support,
                micro_num=tp.sum(),
                micro_den=(tp + fp).sum(),
                present=present,
            )

        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return torch.tensor(float("nan"))
        te.check()
        return tmcls.multiclass_precision(
            state.preds,
            state.targets,
            num_classes=self.num_classes,
            average=self.average,
            top_k=self.k,
            ignore_index=self.ignore_index,
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
            tp, fp, fn, _, support, _ = _multiclass_stats(confmat)
            recall = tp / (tp + fn).clamp(min=1)
            present = (support + fp) > 0
            return _reduce(
                recall,
                self.average,
                support,
                micro_num=tp.sum(),
                micro_den=(tp + fn).sum(),
                present=present,
            )

        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return torch.tensor(float("nan"))
        te.check()
        return tmcls.multiclass_recall(
            state.preds,
            state.targets,
            num_classes=self.num_classes,
            average=self.average,
            top_k=self.k,
            ignore_index=self.ignore_index,
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
            tp, fp, fn, tn, support, _ = _multiclass_stats(confmat)
            specificity = tn / (tn + fp).clamp(min=1)
            present = (support + fp) > 0
            return _reduce(
                specificity,
                self.average,
                support,
                micro_num=tn.sum(),
                micro_den=(tn + fp).sum(),
                present=present,
            )

        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return torch.tensor(float("nan"))
        te.check()
        return tmcls.multiclass_specificity(
            state.preds,
            state.targets,
            num_classes=self.num_classes,
            average=self.average,
            top_k=self.k,
            ignore_index=self.ignore_index,
        )


class multiclass_balanced_accuracy(MetricFunc):
    def __init__(
        self,
        num_classes: int,
        average: str | None = "macro",
        k: int = 1,
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "balanced_accuracy",
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
            tp, fp, fn, _, support, _ = _multiclass_stats(confmat)
            recall = tp / (tp + fn).clamp(min=1)
            present = (support + fp) > 0
            return _reduce(
                recall,
                self.average,
                support,
                micro_num=tp.sum(),
                micro_den=(tp + fn).sum(),
                present=present,
            )

        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return torch.tensor(float("nan"))
        te.check()
        return tmcls.multiclass_recall(
            state.preds,
            state.targets,
            num_classes=self.num_classes,
            average=self.average,
            top_k=self.k,
            ignore_index=self.ignore_index,
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
            if state.preds.numel() == 0 or state.targets.numel() == 0:
                return torch.tensor(float("nan"))
            te.check()
            return tmcls.multiclass_jaccard_index(
                state.preds,
                state.targets,
                num_classes=self.num_classes,
                average=self.average,
                ignore_index=self.ignore_index,
            )

        tp, fp, fn, _, support, _ = _multiclass_stats(confmat)
        iou = tp / (tp + fp + fn).clamp(min=1)
        present = (support + fp) > 0
        return _reduce(
            iou,
            self.average,
            support,
            micro_num=tp.sum(),
            micro_den=(tp + fp + fn).sum(),
            present=present,
        )


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
            tp, fp, fn, _, support, total = _multiclass_stats(confmat)
            recall = tp / (tp + fn).clamp(min=1)
            hamming = 1.0 - recall
            present = (support + fp) > 0
            return _reduce(
                hamming,
                self.average,
                support,
                micro_num=fn.sum(),
                micro_den=total,
                present=present,
            )

        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return torch.tensor(float("nan"))
        te.check()
        return tmcls.multiclass_hamming_distance(
            state.preds,
            state.targets,
            num_classes=self.num_classes,
            average=self.average,
            top_k=self.k,
            ignore_index=self.ignore_index,
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
            tp, fp, fn, _, support, _ = _multiclass_stats(confmat)
            beta_square = self.beta * self.beta
            numerator = (1.0 + beta_square) * tp
            denominator = numerator + beta_square * fn + fp
            fbeta = numerator / denominator.clamp(min=1)
            present = (support + fp) > 0
            return _reduce(
                fbeta,
                self.average,
                support,
                micro_num=numerator.sum(),
                micro_den=denominator.sum(),
                present=present,
            )

        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return torch.tensor(float("nan"))
        te.check()
        return tmcls.multiclass_fbeta_score(
            state.preds,
            state.targets,
            num_classes=self.num_classes,
            beta=self.beta,
            average=self.average,
            top_k=self.k,
            ignore_index=self.ignore_index,
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
