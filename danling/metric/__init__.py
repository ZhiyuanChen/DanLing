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

from .average_meter import AverageMeter, AverageMeters, MultiTaskAverageMeters
from .factory import binary_metrics, multiclass_metrics, multilabel_metrics, regression_metrics
from .metric_meter import MetricMeter, MetricMeters, MultiTaskMetricMeters
from .metrics import Metrics, MultiTaskMetrics
from .preprocesses import (
    preprocess,
    preprocess_binary,
    preprocess_multiclass,
    preprocess_multilabel,
    preprocess_regression,
)
from .registry import METRICS

__all__ = [
    "Metrics",
    "MultiTaskMetrics",
    "MetricMeter",
    "MetricMeters",
    "MultiTaskMetricMeters",
    "AverageMeter",
    "AverageMeters",
    "MultiTaskAverageMeters",
    "METRICS",
    "regression_metrics",
    "binary_metrics",
    "multiclass_metrics",
    "multilabel_metrics",
    "preprocess",
    "preprocess_binary",
    "preprocess_multiclass",
    "preprocess_multilabel",
    "preprocess_regression",
]
