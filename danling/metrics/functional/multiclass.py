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

from .utils import Artifact, MetricFunc

with try_import() as te:
    from torchmetrics.functional import classification as tmcls

if TYPE_CHECKING:  # pragma: no cover
    from ..metrics import Metrics


class multiclass_accuracy(MetricFunc):
    def __init__(self, num_classes: int, average: str | None = "macro", *, name: Optional[str] = "acc") -> None:
        self.num_classes = num_classes
        self.average = average
        super().__init__(name=name, artifact=Artifact(preds_targets=True, task="multiclass", num_classes=num_classes))

    def __call__(self, metrics: "Metrics") -> Tensor | float:
        if metrics.preds.numel() == 0 or metrics.targets.numel() == 0:
            return torch.tensor(float("nan"))
        te.check()
        return tmcls.multiclass_accuracy(
            metrics.preds, metrics.targets, num_classes=self.num_classes, average=self.average
        )


class multiclass_auprc(MetricFunc):
    def __init__(self, num_classes: int, average: str | None = "macro", *, name: Optional[str] = "auprc") -> None:
        self.num_classes = num_classes
        self.average = average
        super().__init__(name=name, artifact=Artifact(preds_targets=True, task="multiclass", num_classes=num_classes))

    def __call__(self, metrics: "Metrics") -> Tensor | float:
        if metrics.preds.numel() == 0 or metrics.targets.numel() == 0:
            return torch.tensor(float("nan"))
        te.check()
        return tmcls.multiclass_average_precision(
            metrics.preds, metrics.targets, num_classes=self.num_classes, average=self.average
        )


class multiclass_auroc(MetricFunc):
    def __init__(self, num_classes: int, average: str | None = "macro", *, name: Optional[str] = "auroc") -> None:
        self.num_classes = num_classes
        self.average = average
        super().__init__(name=name, artifact=Artifact(preds_targets=True, task="multiclass", num_classes=num_classes))

    def __call__(self, metrics: "Metrics") -> Tensor | float:
        if metrics.preds.numel() == 0 or metrics.targets.numel() == 0:
            return torch.tensor(float("nan"))
        te.check()
        return tmcls.multiclass_auroc(
            metrics.preds, metrics.targets, num_classes=self.num_classes, average=self.average
        )


class multiclass_f1_score(MetricFunc):
    def __init__(self, num_classes: int, average: str | None = "macro", *, name: Optional[str] = "f1") -> None:
        self.num_classes = num_classes
        self.average = average
        super().__init__(name=name, artifact=Artifact(preds_targets=True, task="multiclass", num_classes=num_classes))

    def __call__(self, metrics: "Metrics") -> Tensor | float:
        if metrics.preds.numel() == 0 or metrics.targets.numel() == 0:
            return torch.tensor(float("nan"))
        te.check()
        return tmcls.multiclass_f1_score(
            metrics.preds, metrics.targets, num_classes=self.num_classes, average=self.average
        )
