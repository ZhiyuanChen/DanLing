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

from .data import to_device
from .metrics import (
    METRICS,
    AverageMeter,
    AverageMeters,
    MetricMeter,
    StreamMetrics,
)
from .optim import OPTIMIZERS, SCHEDULERS, LRScheduler
from .runners import (
    BaseRunner,
    DeepSpeedRunner,
    DtppRunner,
    FsdpRunner,
    Runner,
    RunnerConfig,
    RunnerState,
    TorchRunner,
    TpppRunner,
)
from .tensors import NestedTensor, PNTensor, tensor
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
    from .metrics import GlobalMetrics, MultiTaskMetrics

__all__ = [
    "RunnerConfig",
    "Runner",
    "BaseRunner",
    "RunnerState",
    "OPTIMIZERS",
    "SCHEDULERS",
    "LRScheduler",
    "TorchRunner",
    "DeepSpeedRunner",
    "FsdpRunner",
    "TpppRunner",
    "DtppRunner",
    "METRICS",
    "GlobalMetrics",
    "MultiTaskMetrics",
    "MetricMeter",
    "StreamMetrics",
    "AverageMeter",
    "AverageMeters",
    "NestedTensor",
    "PNTensor",
    "tensor",
    "to_device",
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
