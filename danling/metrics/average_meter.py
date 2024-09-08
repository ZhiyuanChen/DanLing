# DanLing
# Copyright (C) 2022-Present  DanLing

# This program is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Type

from chanfig import FlatDict, NestedDict
from torch import distributed as dist

from danling.utils import get_world_size

from .utils import MetricsDict, MultiTaskDict


class AverageMeter:
    r"""
    Computes and stores the average and current value.

    Attributes:
        val: Results of current batch on current device.
        bat: Results of current batch on all devices.
        avg: Results of all results on all devices.
        sum: Sum of values.
        count: Number of values.

    See Also:
        [`MetricMeter`]: Average Meter with metric function built-in.
        [`AverageMeters`]: Manage multiple average meters in one object.
        [`MultiTaskAverageMeters`]: Manage multiple average meters in one object with multi-task support.

    Examples:
        >>> meter = AverageMeter()
        >>> meter.update(0.7)
        >>> meter.val
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
        >>> meter.reset()
        >>> meter.val
        0
        >>> meter.avg
        nan
    """

    val: float = 0
    n: float = 1
    sum: float = 0
    count: float = 0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        r"""
        Resets the meter.
        """

        self.val = 0
        self.n = 1
        self.sum = 0
        self.count = 0

    def update(self, value, n: float = 1) -> None:
        r"""
        Updates the average and current value in the meter.

        Args:
            value: Value to be added to the average.
            n: Number of values to be added.
        """

        self.val = value
        self.n = n
        self.sum += value * n
        self.count += n

    def value(self):
        return self.val

    def batch(self):
        world_size = get_world_size()
        if world_size == 1:
            return self.val / self.n if self.n != 0 else float("nan")
        synced_tuple = [None for _ in range(world_size)]
        dist.all_gather_object(synced_tuple, (self.val * self.n, self.n))
        val, n = zip(*synced_tuple)
        count = sum(n)
        if count == 0:
            return float("nan")
        return sum(val) / count

    def average(self):
        world_size = get_world_size()
        if world_size == 1:
            return self.sum / self.count if self.count != 0 else float("nan")
        synced_tuple = [None for _ in range(world_size)]
        dist.all_gather_object(synced_tuple, (self.sum, self.count))
        val, n = zip(*synced_tuple)
        count = sum(n)
        if count == 0:
            return float("nan")
        return sum(val) / count

    @property
    def bat(self):
        return self.batch()

    @property
    def avg(self):
        return self.average()

    def __format__(self, format_spec) -> str:
        return f"{self.val.__format__(format_spec)} ({self.avg.__format__(format_spec)})"


class AverageMeters(MetricsDict):
    r"""
    Manages multiple average meters in one object.

    See Also:
        [`AverageMeter`]: Computes and stores the average and current value.
        [`MultiTaskAverageMeters`]: Manage multiple average meters in one object with multi-task support.
        [`MetricMeters`]: Manage multiple metric meters in one object.

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
        >>> f"{meters:.4f}"
        'loss: 0.0000 (nan)\tauroc: 0.0000 (nan)\tr2: 0.0000 (nan)'
    """

    def __init__(self, *args, default_factory: Type[AverageMeter] = AverageMeter, **kwargs) -> None:
        for meter in args:
            if not isinstance(meter, AverageMeter):
                raise ValueError(f"Expected meter to be an instance of AverageMeter, but got {type(meter)}")
        for name, meter in kwargs.items():
            if not isinstance(meter, AverageMeter):
                raise ValueError(f"Expected {name} to be an instance of AverageMeter, but got {type(meter)}")
        super().__init__(*args, default_factory=default_factory, **kwargs)

    @property
    def sum(self) -> FlatDict[str, float]:
        return FlatDict({key: meter.sum for key, meter in self.all_items()})

    @property
    def count(self) -> FlatDict[str, int]:
        return FlatDict({key: meter.count for key, meter in self.all_items()})

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
            if not isinstance(value, (int, float)):
                raise ValueError(f"Expected values to be int or float, but got {type(value)}")
            self[meter].update(value)

    def set(self, name: str, meter: AverageMeter) -> None:  # pylint: disable=W0237
        if not isinstance(meter, AverageMeter):
            raise ValueError(f"Expected meter to be an instance of AverageMeter, but got {type(meter)}")
        super().set(name, meter)


class MultiTaskAverageMeters(MultiTaskDict):
    r"""
    Manages multiple average meters in one object with multi-task support.

    See Also:
        [`AverageMeter`]: Computes and stores the average and current value.
        [`AverageMeters`]: Manage multiple average meters in one object.
        [`MetricMeters`]: Manage multiple metric meters in one object.

    Examples:
        >>> meters = MultiTaskAverageMeters()
        >>> meters.update({"loss": 0.6, "dataset1.cls.auroc": 0.7, "dataset1.reg.r2": 0.8, "dataset2.r2": 0.9})
        >>> f"{meters:.4f}"
        'loss: 0.6000 (0.6000)\ndataset1.cls.auroc: 0.7000 (0.7000)\ndataset1.reg.r2: 0.8000 (0.8000)\ndataset2.r2: 0.9000 (0.9000)'
        >>> meters['loss'].update(0.9, n=1)
        >>> f"{meters:.4f}"
        'loss: 0.9000 (0.7500)\ndataset1.cls.auroc: 0.7000 (0.7000)\ndataset1.reg.r2: 0.8000 (0.8000)\ndataset2.r2: 0.9000 (0.9000)'
        >>> meters.sum.dict()
        {'loss': 1.5, 'dataset1': {'cls': {'auroc': 0.7}, 'reg': {'r2': 0.8}}, 'dataset2': {'r2': 0.9}}
        >>> meters.count.dict()
        {'loss': 2, 'dataset1': {'cls': {'auroc': 1}, 'reg': {'r2': 1}}, 'dataset2': {'r2': 1}}
        >>> meters.reset()
        >>> f"{meters:.4f}"
        'loss: 0.0000 (nan)\ndataset1.cls.auroc: 0.0000 (nan)\ndataset1.reg.r2: 0.0000 (nan)\ndataset2.r2: 0.0000 (nan)'
        >>> meters = MultiTaskAverageMeters(return_average=True)
        >>> meters.update({"loss": 0.6, "dataset1.a.auroc": 0.7, "dataset1.b.auroc": 0.8, "dataset2.auroc": 0.9})
        >>> f"{meters:.4f}"
        'loss: 0.6000 (0.6000)\ndataset1.a.auroc: 0.7000 (0.7000)\ndataset1.b.auroc: 0.8000 (0.8000)\ndataset2.auroc: 0.9000 (0.9000)'
        >>> meters.update({"loss": 0.9, "dataset1.a.auroc": 0.8, "dataset1.b.auroc": 0.9, "dataset2.auroc": 1.0})
        >>> f"{meters:.4f}"
        'loss: 0.9000 (0.7500)\ndataset1.a.auroc: 0.8000 (0.7500)\ndataset1.b.auroc: 0.9000 (0.8500)\ndataset2.auroc: 1.0000 (0.9500)'
    """  # noqa: E501

    @property
    def sum(self) -> NestedDict[str, float]:
        return NestedDict({key: meter.sum for key, meter in self.all_items()})

    @property
    def count(self) -> NestedDict[str, int]:
        return NestedDict({key: meter.count for key, meter in self.all_items()})

    def update(self, *args: Dict, **values: float) -> None:  # pylint: disable=W0237
        r"""
        Updates the average and current value in all meters.

        Args:
            values: Dict of values to be added to the average.
            n: Number of values to be added.

        Raises:
            ValueError: If the value is not an instance of (int, float, Mapping).
        """  # noqa: E501

        if args:
            if len(args) > 1:
                raise ValueError("Expected only one positional argument, but got multiple.")
            values = args[0].update(values) or args[0] if values else args[0]

        for meter, value in values.items():
            if not isinstance(value, (int, float, Mapping)):
                raise ValueError(f"Expected values to be int, float, or a Mapping, but got {type(value)}")
            self[meter].update(value)

    # evil hack, as the default_factory must not be set to make `NestedDict` happy
    # this have some side effects, it will break attribute style intermediate nested dict auto creation
    # but everything has a price
    def get(self, name: Any, default=None) -> Any:
        if not name.startswith("_") and not name.endswith("_"):
            return self.setdefault(name, AverageMeter())
        return super().get(name, default)

    def set(self, name: str, meter: AverageMeter | AverageMeters) -> None:  # pylint: disable=W0237
        if not isinstance(meter, (AverageMeter, AverageMeters)):
            raise ValueError(
                f"Expected meter to be an instance of AverageMeter or AverageMeters, but got {type(meter)}"
            )
        super().set(name, meter)
