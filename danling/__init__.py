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

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .data import to_device
    from .metrics import (
        METRICS,
        AverageMeter,
        AverageMeters,
        GlobalMetrics,
        MetricMeter,
        MultiTaskMetrics,
        StreamMetrics,
    )
    from .optim import OPTIMIZERS, SCHEDULERS, LRScheduler
    from .runners import (
        BaseRunner,
        DeepSpeedRunner,
        ParallelRunner,
        Runner,
        RunnerConfig,
        RunnerState,
        TorchRunner,
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
    "ParallelRunner",
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

_LAZY_IMPORTS = {
    "RunnerConfig": (".runners", "RunnerConfig"),
    "Runner": (".runners", "Runner"),
    "BaseRunner": (".runners", "BaseRunner"),
    "RunnerState": (".runners", "RunnerState"),
    "OPTIMIZERS": (".optim", "OPTIMIZERS"),
    "SCHEDULERS": (".optim", "SCHEDULERS"),
    "LRScheduler": (".optim", "LRScheduler"),
    "TorchRunner": (".runners", "TorchRunner"),
    "DeepSpeedRunner": (".runners", "DeepSpeedRunner"),
    "ParallelRunner": (".runners", "ParallelRunner"),
    "METRICS": (".metrics", "METRICS"),
    "GlobalMetrics": (".metrics", "GlobalMetrics"),
    "MultiTaskMetrics": (".metrics", "MultiTaskMetrics"),
    "MetricMeter": (".metrics", "MetricMeter"),
    "StreamMetrics": (".metrics", "StreamMetrics"),
    "AverageMeter": (".metrics", "AverageMeter"),
    "AverageMeters": (".metrics", "AverageMeters"),
    "NestedTensor": (".tensors", "NestedTensor"),
    "PNTensor": (".tensors", "PNTensor"),
    "tensor": (".tensors", "tensor"),
    "to_device": (".data", "to_device"),
    "save": (".utils", "save"),
    "load": (".utils", "load"),
    "load_pandas": (".utils", "load_pandas"),
    "catch": (".utils", "catch"),
    "debug": (".utils", "debug"),
    "flexible_decorator": (".utils", "flexible_decorator"),
    "method_cache": (".utils", "method_cache"),
    "ensure_dir": (".utils", "ensure_dir"),
    "is_json_serializable": (".utils", "is_json_serializable"),
}

_LAZY_MODULES = frozenset({"data", "metrics", "optim", "runners", "tensors", "utils"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        value = import_module(f".{name}", __name__)
    else:
        try:
            module_name, attribute_name = _LAZY_IMPORTS[name]
        except KeyError:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
        value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_IMPORTS, *_LAZY_MODULES})
