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

from math import nan
from typing import Dict, Type

import torch
from torch import Tensor
from torch import distributed as dist

from danling.utils import get_world_size

from .utils import MetricsDict, RoundDict

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self


class AverageMeter:
    r"""
    A lightweight utility to compute and store running averages of values.

    AverageMeter provides an efficient way to track running statistics (current value, sum, count, average)
    with minimal memory overhead and support for distributed environments.

    Attributes:
        val: Most recent value added to the meter
        bat: Most recent value, synchronized across distributed processes
        avg: Running average of all values, weighted by counts
        sum: Sum of all values added to the meter
        count: Total count of values added (considering weights)

    Examples:
        >>> meter = AverageMeter()
        >>> meter.update(0.7)
        >>> meter.val
        0.7
        >>> meter.bat  # Same as val in non-distributed settings
        0.7
        >>> meter.avg
        0.7
        >>> meter.update(0.9)
        >>> meter.val
        0.9
        >>> meter.avg
        0.8
        >>> meter.sum
        1.6
        >>> meter.count
        2
        >>> # Weighted update
        >>> meter.update(value=0.5, n=3)
        >>> meter.avg
        0.62
        >>> meter.reset()
        AverageMeter(val=nan, avg=nan)

    See Also:
        - [`MetricMeter`][danling.metric.metric_meter.MetricMeter]:
            Memory-efficient metric tracker that averages metrics batch-by-batch.
    """

    v: float = 0.0
    n: int = 1
    sum: float = 0.0
    count: int = 0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> Self:
        r"""
        Resets the meter.
        """

        self.v = 0.0
        self.n = 1
        self.sum = 0.0
        self.count = 0
        return self

    def update(self, value: float | int, n: int = 1) -> None:
        r"""
        Updates the average and current value in the meter.

        Args:
            value: Value to be added to the average.
            n: Number of values to be added.
        """

        self.v = value
        self.n = n
        self.sum += value * n
        self.count += n

    def value(self) -> float:
        if self.count == 0:
            return nan
        return self.v

    def batch(self) -> float:
        world_size = get_world_size()
        if world_size <= 1:
            return self.value()
        synced_tensor = torch.tensor([self.v, self.n], dtype=torch.float64).cuda()
        dist.all_reduce(synced_tensor)
        val, count = synced_tensor
        if count == 0:
            return nan
        return (val / count).item()

    def average(self) -> float:
        world_size = get_world_size()
        if world_size <= 1:
            return self.sum / self.count if self.count != 0 else nan
        synced_tensor = torch.tensor([self.sum, self.count], dtype=torch.float64).cuda()
        dist.all_reduce(synced_tensor)
        val, count = synced_tensor
        if count == 0:
            return nan
        return (val / count).item()

    @property
    def val(self) -> float:
        return self.value()

    @property
    def bat(self) -> float:
        return self.batch()

    @property
    def avg(self) -> float:
        return self.average()

    def __format__(self, format_spec: str) -> str:
        return f"{self.val.__format__(format_spec)} ({self.avg.__format__(format_spec)})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(val={self.val}, avg={self.avg})"


class AverageMeters(MetricsDict):
    r"""
    Manages multiple average meters in one object.

    Examples:
        >>> meters = AverageMeters()
        >>> meters.update({"loss": 0.6, "auroc": 0.7, "r2": 0.8})
        >>> f"{meters:.4f}"
        'loss: 0.6000 (0.6000)\tauroc: 0.7000 (0.7000)\tr2: 0.8000 (0.8000)'
        >>> meters['loss'].update(value=0.9, n=1)
        >>> f"{meters:.4f}"
        'loss: 0.9000 (0.7500)\tauroc: 0.7000 (0.7000)\tr2: 0.8000 (0.8000)'
        >>> meters.sum.dict()
        {'loss': 1.5, 'auroc': 0.7, 'r2': 0.8}
        >>> meters.count.dict()
        {'loss': 2, 'auroc': 1, 'r2': 1}
        >>> meters.reset()
        AverageMeters(...)
        >>> f"{meters:.4f}"
        'loss: nan (nan)\tauroc: nan (nan)\tr2: nan (nan)'

    See Also:
        - [`MetricMeters`][danling.metric.metric_meter.MetricMeters]:
            Memory-efficient metric tracker that averages multiple metrics batch-by-batch.
    """

    def __init__(self, default_factory: Type[AverageMeter] = AverageMeter, **meters) -> None:
        super().__init__(default_factory=default_factory, **meters)
        for name, meter in self.items():
            if not isinstance(meter, AverageMeter):
                raise ValueError(f"Expected {name} to be an instance of AverageMeter, but got {type(meter)}")

    @property
    def sum(self) -> RoundDict[str, float]:
        return RoundDict({key: meter.sum for key, meter in self.all_items()})

    @property
    def count(self) -> RoundDict[str, int]:
        return RoundDict({key: meter.count for key, meter in self.all_items()})

    def update(self, *args: Dict, **values: int | float) -> None:  # pylint: disable=W0237
        r"""
        Updates the average and current value in all meters.

        Args:
            values: Dict of values to be added to the average.
            n: Number of values to be added.

        Raises:
            ValueError: If the value is not an instance of (int, float).
        """  # noqa: E501

        if args:
            if len(args) > 1:
                raise ValueError("Expected only one positional argument, but got multiple.")
            values = args[0].update(values) or args[0] if values else args[0]

        for meter, value in values.items():
            if isinstance(value, Tensor):
                value = value.item()
            if not isinstance(value, (int, float)):
                raise ValueError(f"Expected values to be int or float, but got {type(value)}")
            self[meter].update(value)

    def set(self, name: str, meter: AverageMeter) -> None:  # pylint: disable=W0237
        if not isinstance(meter, AverageMeter):
            raise ValueError(f"Expected meter to be an instance of AverageMeter, but got {type(meter)}")
        super().set(name, meter)
