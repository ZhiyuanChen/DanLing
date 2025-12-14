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
from torchmetrics.functional.classification import binary_f1_score as tm_binary_f1

try:
    from torchmetrics.functional.classification.accuracy import _accuracy_reduce  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _accuracy_reduce = None

from .utils import Artifact, MetricFunc

if TYPE_CHECKING:  # pragma: no cover
    from ..metrics import Metrics


class binary_accuracy(MetricFunc):
    """
    Metric function version of binary accuracy.

    Relies on a confusion matrix if available, otherwise falls back to
    torchmetrics' functional binary accuracy implementation.
    """

    def __init__(self, threshold: float = 0.5, *, name: Optional[str] = "acc") -> None:
        self.threshold = threshold
        super().__init__(name=name, artifact=Artifact(confmat=True, task="binary", threshold=threshold, num_classes=2))

    def __call__(self, metrics: Metrics) -> Tensor | float:
        confmat = metrics.confmat
        if confmat is None:
            if metrics.preds.numel() == 0 or metrics.targets.numel() == 0:
                return torch.tensor(float("nan"))
            return tm_binary_accuracy(metrics.preds, metrics.targets, threshold=self.threshold)

        if _accuracy_reduce is None:  # pragma: no cover
            correct = torch.diag(confmat).sum()
            total = confmat.sum()
            return correct / total if total > 0 else torch.tensor(float("nan"))

        tn = confmat[0, 0]
        fp = confmat[0, 1]
        fn = confmat[1, 0]
        tp = confmat[1, 1]
        return _accuracy_reduce(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            average="micro",
            multidim_average="global",
            multilabel=False,
            top_k=1,
        )


class binary_auroc(MetricFunc):
    """
    Metric function version of binary AUROC.

    Requires raw predictions and targets; does not use confusion matrix.
    """

    def __init__(self, *, name: Optional[str] = "auroc") -> None:
        super().__init__(name=name, artifact=Artifact(preds_targets=True, task="binary", num_classes=2))

    def __call__(self, metrics: Metrics) -> Tensor | float:
        if metrics.preds.numel() == 0 or metrics.targets.numel() == 0:
            return torch.tensor(float("nan"))
        return tm_binary_auroc(metrics.preds, metrics.targets)


class binary_auprc(MetricFunc):
    """
    Metric function version of binary AUPRC.
    """

    def __init__(self, *, name: Optional[str] = "auprc") -> None:
        super().__init__(name=name, artifact=Artifact(preds_targets=True, task="binary", num_classes=2))

    def __call__(self, metrics: Metrics) -> Tensor | float:
        if metrics.preds.numel() == 0 or metrics.targets.numel() == 0:
            return torch.tensor(float("nan"))
        return tm_binary_auprc(metrics.preds, metrics.targets)


class binary_f1(MetricFunc):
    """
    Metric function version of binary F1 score.

    Uses confusion matrix when available; otherwise falls back to torchmetrics functional.
    """

    def __init__(self, threshold: float = 0.5, *, name: Optional[str] = "f1") -> None:
        self.threshold = threshold
        super().__init__(name=name, artifact=Artifact(confmat=True, task="binary", threshold=threshold, num_classes=2))

    def __call__(self, metrics: Metrics) -> Tensor | float:
        confmat = metrics.confmat
        if confmat is None:
            if metrics.preds.numel() == 0 or metrics.targets.numel() == 0:
                return torch.tensor(float("nan"))
            return tm_binary_f1(metrics.preds, metrics.targets, threshold=self.threshold)

        tp = confmat[1, 1]
        fp = confmat[0, 1]
        fn = confmat[1, 0]
        denom = (2 * tp + fp + fn).clamp(min=1)
        return (2 * tp) / denom
