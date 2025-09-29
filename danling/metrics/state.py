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

from collections.abc import Sequence
from typing import Any

from torch import Tensor


class MetricState:
    def __init__(self, preds: Any, targets: Any, confmat: Tensor | None = None) -> None:
        self.preds = preds
        self.targets = targets
        self.confmat = confmat

    @classmethod
    def from_requirements(cls, preds: Any, targets: Any, requirements: Any = None) -> MetricState:
        return cls(preds=preds, targets=targets, confmat=cls.compute_confmat(preds, targets, requirements))

    @staticmethod
    def _requirement_get(requirements: Any, name: str, default: Any = None) -> Any:
        if requirements is None:
            return default
        if isinstance(requirements, dict):
            return requirements.get(name, default)
        return getattr(requirements, name, default)

    @staticmethod
    def _merge_scalar(current: Any, incoming: Any, label: str) -> Any:
        if current is None:
            return incoming
        if incoming is None or current == incoming:
            return current
        raise ValueError(f"Conflicting metric requirement for {label}: {current} vs {incoming}")

    @classmethod
    def collect_requirements(cls, metric_funcs: Sequence[Any], *, require_nonempty: bool = False) -> dict[str, Any]:
        if require_nonempty and not metric_funcs:
            raise ValueError("At least one metric function is required.")

        requirements: dict[str, Any] = {
            "preds_targets": False,
            "confmat": False,
            "task": None,
            "num_classes": None,
            "num_labels": None,
            "threshold": None,
            "ignore_index": None,
        }

        for metric in metric_funcs:
            if cls._requirement_get(metric, "preds_targets", False):
                requirements["preds_targets"] = True
            if cls._requirement_get(metric, "confmat", False):
                requirements["confmat"] = True
                requirements["task"] = cls._merge_scalar(
                    requirements["task"], cls._requirement_get(metric, "task"), "task"
                )
                requirements["num_classes"] = cls._merge_scalar(
                    requirements["num_classes"], cls._requirement_get(metric, "num_classes"), "num_classes"
                )
                requirements["num_labels"] = cls._merge_scalar(
                    requirements["num_labels"], cls._requirement_get(metric, "num_labels"), "num_labels"
                )
                requirements["threshold"] = cls._merge_scalar(
                    requirements["threshold"], cls._requirement_get(metric, "threshold"), "threshold"
                )
                requirements["ignore_index"] = cls._merge_scalar(
                    requirements["ignore_index"], cls._requirement_get(metric, "ignore_index"), "ignore_index"
                )

        if requirements["confmat"] and requirements["task"] is None:
            raise ValueError("Confusion matrix computation requires a task to be specified.")
        return requirements

    @classmethod
    def compute_confmat(cls, preds: Any, targets: Any, requirements: Any = None) -> Tensor | None:
        if not cls._requirement_get(requirements, "confmat", False):
            return None
        if not isinstance(preds, Tensor) or not isinstance(targets, Tensor):
            return None
        task = cls._requirement_get(requirements, "task")
        if task is None:
            raise ValueError("Confusion matrix computation requires a task to be specified.")

        from torchmetrics.functional.classification import confusion_matrix as tm_confusion_matrix

        kwargs = {"task": task}
        num_classes = cls._requirement_get(requirements, "num_classes")
        num_labels = cls._requirement_get(requirements, "num_labels")
        threshold = cls._requirement_get(requirements, "threshold")
        ignore_index = cls._requirement_get(requirements, "ignore_index")

        if num_classes is not None:
            kwargs["num_classes"] = num_classes
        if num_labels is not None:
            kwargs["num_labels"] = num_labels
        if threshold is not None:
            kwargs["threshold"] = threshold
        if ignore_index is not None:
            kwargs["ignore_index"] = ignore_index
        return tm_confusion_matrix(preds, targets, **kwargs)
