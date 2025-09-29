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
from __future__ import annotations

from torch import Tensor

from ..state import MetricState


class MetricFunc:
    """
    Base class for metric functions with declared state requirements.

    Metric functions behave like callables via ``__call__`` and carry metadata
    so that metrics containers know which shared state to maintain.
    """

    name: str = ""
    preds_targets: bool = False
    confmat: bool = False
    task: str | None = None
    num_classes: int | None = None
    num_labels: int | None = None
    threshold: float | None = None
    ignore_index: int | None = None

    def __init__(
        self,
        *,
        name: str | None = None,
        preds_targets: bool = False,
        confmat: bool = False,
        task: str | None = None,
        num_classes: int | None = None,
        num_labels: int | None = None,
        threshold: float | None = None,
        ignore_index: int | None = None,
    ) -> None:
        self.name = name or self.name or self.__class__.__name__
        self.preds_targets = preds_targets
        self.confmat = confmat
        self.task = task
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.threshold = threshold
        self.ignore_index = ignore_index

    def __call__(self, state: MetricState) -> Tensor | float:  # pragma: no cover - interface stub
        raise NotImplementedError


def infer_task(num_classes: int | None, num_labels: int | None):
    if num_classes is not None and num_labels is not None:
        raise ValueError("Only one of `num_classes` or `num_labels` can be provided.")
    if num_classes is not None:
        return "multiclass"
    if num_labels is not None:
        return "multilabel"
    return "binary"
