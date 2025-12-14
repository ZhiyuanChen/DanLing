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

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from torch import Tensor

if TYPE_CHECKING:
    from ..metrics import Metrics


@dataclass(frozen=True)
class Artifact:
    preds_targets: bool = False
    confmat: bool = False
    task: str | None = None
    num_classes: int | None = None
    num_labels: int | None = None
    threshold: float | None = None

    def merge(self, other: Artifact) -> Artifact:
        def _merge_scalar(current: Any, incoming: Any, label: str):
            if current is None:
                return incoming
            if incoming is None or current == incoming:
                return current
            raise ValueError(f"Conflicting artifact requirement for {label}: {current} vs {incoming}")

        return Artifact(
            preds_targets=self.preds_targets or other.preds_targets,
            confmat=self.confmat or other.confmat,
            task=_merge_scalar(self.task, other.task, "task"),
            num_classes=_merge_scalar(self.num_classes, other.num_classes, "num_classes"),
            num_labels=_merge_scalar(self.num_labels, other.num_labels, "num_labels"),
            threshold=_merge_scalar(self.threshold, other.threshold, "threshold"),
        )

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> Artifact:
        return replace(cls(), **kwargs)


class MetricFunc:
    """
    Base class for metric functions with declared artifact needs.

    Metric functions behave like callables via ``__call__`` and carry metadata
    so that the Metrics container knows which artifacts to maintain.
    """

    name: str = ""
    artifact: Artifact

    def __init__(self, *, name: str | None = None, artifact: Artifact | None = None) -> None:
        self.name = name or self.name or self.__class__.__name__
        self.artifact = artifact or Artifact()

    def __call__(self, metrics: Metrics) -> Tensor | float:  # pragma: no cover - interface stub
        raise NotImplementedError


def infer_task(num_classes: int | None, num_labels: int | None):
    if num_classes is not None and num_labels is not None:
        raise ValueError("Only one of `num_classes` or `num_labels` can be provided.")
    if num_classes is not None:
        return "multiclass"
    if num_labels is not None:
        return "multilabel"
    return "binary"
