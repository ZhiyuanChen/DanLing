# DanLing
# Copyright (C) 2022-Present  DanLing

# This file is part of DanLing.

# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0
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
from torch import Tensor

# NOTE: Functional imports live here to keep metric definitions colocated with metric logic
from torchmetrics.functional.classification import binary_accuracy as tm_binary_accuracy  # type: ignore[import-untyped]
from torchmetrics.functional.classification import binary_auroc as tm_binary_auroc
from torchmetrics.functional.classification import binary_average_precision as tm_binary_auprc
from torchmetrics.functional.classification import binary_fbeta_score as tm_binary_fbeta
from torchmetrics.functional.classification import binary_hamming_distance as tm_binary_hamming_loss
from torchmetrics.functional.classification import binary_jaccard_index as tm_binary_jaccard_index
from torchmetrics.functional.classification import binary_precision as tm_binary_precision
from torchmetrics.functional.classification import binary_recall as tm_binary_recall
from torchmetrics.functional.classification import binary_specificity as tm_binary_specificity

from .utils import MetricFunc

if TYPE_CHECKING:  # pragma: no cover
    from ..state import MetricState


class binary_accuracy(MetricFunc):
    """
    Metric function version of binary accuracy.

    Relies on a confusion matrix if available, otherwise falls back to
    torchmetrics' functional binary accuracy implementation.
    """

    def __init__(self, threshold: float = 0.5, ignore_index: int | None = -100, *, name: Optional[str] = "acc") -> None:
        self.threshold = threshold
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            confmat=True,
            task="binary",
            threshold=threshold,
            num_classes=2,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is None:
            if state.preds.numel() == 0 or state.targets.numel() == 0:
                return torch.tensor(float("nan"))
            return tm_binary_accuracy(
                state.preds,
                state.targets,
                threshold=self.threshold,
                ignore_index=self.ignore_index,
            )

        correct = torch.diag(confmat).sum()
        total = confmat.sum()
        return correct / total if total > 0 else torch.tensor(float("nan"), device=confmat.device)


class binary_auroc(MetricFunc):
    """
    Metric function version of binary AUROC.

    Requires raw predictions and targets; does not use confusion matrix.
    """

    def __init__(self, ignore_index: int | None = -100, *, name: Optional[str] = "auroc") -> None:
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            preds_targets=True,
            task="binary",
            num_classes=2,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return torch.tensor(float("nan"))
        return tm_binary_auroc(state.preds, state.targets, ignore_index=self.ignore_index)


class binary_auprc(MetricFunc):
    """
    Metric function version of binary AUPRC.
    """

    def __init__(self, ignore_index: int | None = -100, *, name: Optional[str] = "auprc") -> None:
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            preds_targets=True,
            task="binary",
            num_classes=2,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return torch.tensor(float("nan"))
        return tm_binary_auprc(state.preds, state.targets, ignore_index=self.ignore_index)


class binary_precision(MetricFunc):
    """
    Metric function version of binary precision.
    """

    def __init__(
        self, threshold: float = 0.5, ignore_index: int | None = -100, *, name: Optional[str] = "precision"
    ) -> None:
        self.threshold = threshold
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            confmat=True,
            task="binary",
            threshold=threshold,
            num_classes=2,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is None:
            if state.preds.numel() == 0 or state.targets.numel() == 0:
                return torch.tensor(float("nan"))
            return tm_binary_precision(
                state.preds,
                state.targets,
                threshold=self.threshold,
                ignore_index=self.ignore_index,
            )

        tp = confmat[1, 1]
        fp = confmat[0, 1]
        denom = (tp + fp).clamp(min=1)
        return tp / denom


class binary_recall(MetricFunc):
    """
    Metric function version of binary recall.
    """

    def __init__(
        self, threshold: float = 0.5, ignore_index: int | None = -100, *, name: Optional[str] = "recall"
    ) -> None:
        self.threshold = threshold
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            confmat=True,
            task="binary",
            threshold=threshold,
            num_classes=2,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is None:
            if state.preds.numel() == 0 or state.targets.numel() == 0:
                return torch.tensor(float("nan"))
            return tm_binary_recall(
                state.preds,
                state.targets,
                threshold=self.threshold,
                ignore_index=self.ignore_index,
            )

        tp = confmat[1, 1]
        fn = confmat[1, 0]
        denom = (tp + fn).clamp(min=1)
        return tp / denom


class binary_specificity(MetricFunc):
    """
    Metric function version of binary specificity.
    """

    def __init__(
        self, threshold: float = 0.5, ignore_index: int | None = -100, *, name: Optional[str] = "specificity"
    ) -> None:
        self.threshold = threshold
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            confmat=True,
            task="binary",
            threshold=threshold,
            num_classes=2,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is None:
            if state.preds.numel() == 0 or state.targets.numel() == 0:
                return torch.tensor(float("nan"))
            return tm_binary_specificity(
                state.preds,
                state.targets,
                threshold=self.threshold,
                ignore_index=self.ignore_index,
            )

        tn = confmat[0, 0]
        fp = confmat[0, 1]
        denom = (tn + fp).clamp(min=1)
        return tn / denom


class binary_balanced_accuracy(MetricFunc):
    """
    Metric function version of binary balanced accuracy.
    """

    def __init__(
        self, threshold: float = 0.5, ignore_index: int | None = -100, *, name: Optional[str] = "balanced_accuracy"
    ) -> None:
        self.threshold = threshold
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            confmat=True,
            task="binary",
            threshold=threshold,
            num_classes=2,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is None:
            if state.preds.numel() == 0 or state.targets.numel() == 0:
                return torch.tensor(float("nan"))
            recall = tm_binary_recall(
                state.preds,
                state.targets,
                threshold=self.threshold,
                ignore_index=self.ignore_index,
            )
            specificity = tm_binary_specificity(
                state.preds,
                state.targets,
                threshold=self.threshold,
                ignore_index=self.ignore_index,
            )
            return 0.5 * (recall + specificity)

        tp = confmat[1, 1]
        fn = confmat[1, 0]
        tn = confmat[0, 0]
        fp = confmat[0, 1]
        tpr = tp / (tp + fn).clamp(min=1)
        tnr = tn / (tn + fp).clamp(min=1)
        return 0.5 * (tpr + tnr)


class binary_jaccard_index(MetricFunc):
    """
    Metric function version of binary Jaccard index (IoU).
    """

    def __init__(
        self, threshold: float = 0.5, ignore_index: int | None = -100, *, name: Optional[str] = "jaccard"
    ) -> None:
        self.threshold = threshold
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            confmat=True,
            task="binary",
            threshold=threshold,
            num_classes=2,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is None:
            if state.preds.numel() == 0 or state.targets.numel() == 0:
                return torch.tensor(float("nan"))
            return tm_binary_jaccard_index(
                state.preds,
                state.targets,
                threshold=self.threshold,
                ignore_index=self.ignore_index,
            )

        tp = confmat[1, 1]
        fp = confmat[0, 1]
        fn = confmat[1, 0]
        return tp / (tp + fp + fn).clamp(min=1)


class binary_iou(binary_jaccard_index):
    """
    Alias of binary Jaccard index.
    """

    def __init__(self, threshold: float = 0.5, ignore_index: int | None = -100, *, name: Optional[str] = "iou") -> None:
        super().__init__(threshold=threshold, ignore_index=ignore_index, name=name)


class binary_hamming_loss(MetricFunc):
    """
    Metric function version of binary hamming loss.
    """

    def __init__(
        self, threshold: float = 0.5, ignore_index: int | None = -100, *, name: Optional[str] = "hamming_loss"
    ) -> None:
        self.threshold = threshold
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            confmat=True,
            task="binary",
            threshold=threshold,
            num_classes=2,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is None:
            if state.preds.numel() == 0 or state.targets.numel() == 0:
                return torch.tensor(float("nan"))
            return tm_binary_hamming_loss(
                state.preds,
                state.targets,
                threshold=self.threshold,
                ignore_index=self.ignore_index,
            )

        fp = confmat[0, 1]
        fn = confmat[1, 0]
        total = confmat.sum()
        return (fp + fn) / total if total > 0 else torch.tensor(float("nan"), device=confmat.device)


class binary_fbeta(MetricFunc):
    """
    Metric function version of binary F-beta score.
    """

    def __init__(
        self,
        beta: float = 1.0,
        threshold: float = 0.5,
        ignore_index: int | None = -100,
        *,
        name: Optional[str] = "fbeta",
    ) -> None:
        self.beta = beta
        self.threshold = threshold
        self.ignore_index = ignore_index
        super().__init__(
            name=name,
            confmat=True,
            task="binary",
            threshold=threshold,
            num_classes=2,
            ignore_index=ignore_index,
        )

    def __call__(self, state: MetricState) -> Tensor | float:
        confmat = state.confmat
        if confmat is None:
            if state.preds.numel() == 0 or state.targets.numel() == 0:
                return torch.tensor(float("nan"))
            return tm_binary_fbeta(
                state.preds,
                state.targets,
                beta=self.beta,
                threshold=self.threshold,
                ignore_index=self.ignore_index,
            )

        tp = confmat[1, 1]
        fp = confmat[0, 1]
        fn = confmat[1, 0]
        beta_square = self.beta * self.beta
        numerator = (1 + beta_square) * tp
        denominator = numerator + beta_square * fn + fp
        return numerator / denominator.clamp(min=1)


class binary_f1(binary_fbeta):
    """
    Metric function version of binary F1 score.
    """

    def __init__(self, threshold: float = 0.5, ignore_index: int | None = -100, *, name: Optional[str] = "f1") -> None:
        super().__init__(beta=1.0, threshold=threshold, ignore_index=ignore_index, name=name)
