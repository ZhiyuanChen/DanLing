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

from operator import index
from typing import Literal

from chanfig import Registry as Registry_

from .factory import binary_metrics, multiclass_metrics, multilabel_metrics, regression_metrics
from .global_metrics import GlobalMetrics
from .stream_metrics import StreamMetrics


class Registry(Registry_):
    case_sensitive = False

    @staticmethod
    def _require_positive_int(name: str, value: int | None) -> int:
        if value is None:
            raise ValueError(f"{name} is required")
        value = index(value)
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer, but got {value!r}")
        return value

    def build(
        self,
        type: str,
        mode: Literal["global", "stream"] = "global",
        num_labels: int | None = None,
        num_classes: int | None = None,
        num_outputs: int | None = None,
        **kwargs,
    ) -> GlobalMetrics | StreamMetrics:
        type = type.lower()
        if type == "multilabel":
            if num_classes is not None and num_labels is not None and num_classes != num_labels:
                raise ValueError(
                    f"num_classes and num_labels must match when both are provided, got {num_classes} and {num_labels}"
                )
            num_labels = self._require_positive_int("num_labels", num_labels)
            return self.init(self.lookup(type), mode=mode, num_labels=num_labels, **kwargs)
        if type == "multiclass":
            if num_classes is not None and num_labels is not None and num_classes != num_labels:
                raise ValueError(
                    f"num_classes and num_labels must match when both are provided, got {num_classes} and {num_labels}"
                )
            if num_classes is None:
                num_classes = num_labels
            num_classes = self._require_positive_int("num_classes", num_classes)
            return self.init(self.lookup(type), mode=mode, num_classes=num_classes, **kwargs)
        if type == "regression":
            if num_outputs is not None and num_labels is not None and num_outputs != num_labels:
                raise ValueError(
                    f"num_outputs and num_labels must match when both are provided, got {num_outputs} and {num_labels}"
                )
            if num_outputs is None:
                num_outputs = num_labels if num_labels is not None else 1
            num_outputs = self._require_positive_int("num_outputs", num_outputs)
            return self.init(self.lookup(type), mode=mode, num_outputs=num_outputs, **kwargs)
        return self.init(self.lookup(type), mode=mode, **kwargs)


METRICS = Registry(key="type")
METRICS.register(binary_metrics, "binary")
METRICS.register(multiclass_metrics, "multiclass")
METRICS.register(multilabel_metrics, "multilabel")
METRICS.register(regression_metrics, "regression")
