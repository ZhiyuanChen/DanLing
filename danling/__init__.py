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

from lazy_imports import try_import

from .metric import (
    METRICS,
    AverageMeter,
    AverageMeters,
    MetricMeter,
    MetricMeters,
    MultiTaskAverageMeters,
    MultiTaskMetricMeters,
)
from .optim import OPTIMIZERS, SCHEDULERS, LRScheduler
from .runner import AccelerateRunner, BaseRunner, Config, DeepSpeedRunner, Runner, TorchRunner
from .tensor import NestedTensor, PNTensor, tensor
from .utils import (
    catch,
    debug,
    ensure_dir,
    flexible_decorator,
    is_json_serializable,
    load,
    load_pandas,
    method_cache,
    save,
)

with try_import():
    from .metric import Metrics, MultiTaskMetrics

__all__ = [
    "Config",
    "Runner",
    "BaseRunner",
    "OPTIMIZERS",
    "SCHEDULERS",
    "LRScheduler",
    "TorchRunner",
    "DeepSpeedRunner",
    "AccelerateRunner",
    "METRICS",
    "Metrics",
    "MultiTaskMetrics",
    "MetricMeter",
    "MetricMeters",
    "MultiTaskMetricMeters",
    "AverageMeter",
    "AverageMeters",
    "MultiTaskAverageMeters",
    "NestedTensor",
    "PNTensor",
    "tensor",
    "save",
    "load",
    "load_pandas",
    "catch",
    "debug",
    "flexible_decorator",
    "method_cache",
    "ensure_dir",
    "is_json_serializable",
]
