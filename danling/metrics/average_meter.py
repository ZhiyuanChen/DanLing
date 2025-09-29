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
from typing import Dict
from warnings import warn

import torch
from torch import Tensor
from torch import device as torch_device
from torch import distributed as dist

from danling.utils import get_world_size

from .utils import MetersBase, RoundDict

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
        device: Device used when synchronising statistics across processes

    Args:
        device: Optional device used for distributed reductions. When not provided,
            the device is detected automatically when synchronisation happens.

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
        - [`MetricMeter`][danling.metrics.metric_meter.MetricMeter]:
            Memory-efficient metric tracker that averages metrics batch-by-batch.
    """

    v: float = 0.0
    n: int = 0
    sum: float = 0.0
    count: int = 0

    def __init__(self, *, device: torch.device | str | None = None) -> None:
        self.device = torch_device(device) if device is not None else None
        self.reset()

    def reset(self, *, device: torch.device | str | None = None) -> Self:
        r"""
        Resets the meter.
        """

        if device is not None:
            self.device = torch_device(device)
        self.v = 0.0
        self.n = 0
        self.sum = 0.0
        self.count = 0
        return self

    def update(self, value: float | int | Tensor, n: int = 1) -> None:
        r"""
        Updates the average and current value in the meter.

        Args:
            value: Value to be added to the average.
            n: Number of values to be added.
        """

        if isinstance(value, Tensor):
            if self.device is None:
                self.device = value.device
            value = value.item()
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
        device = self._infer_device()
        synced_tensor = torch.tensor([self.v * self.n, self.n], dtype=torch.float64, device=device)
        dist.all_reduce(synced_tensor)
        val, count = synced_tensor.tolist()
        if count == 0:
            return nan
        return val / count

    def average(self) -> float:
        world_size = get_world_size()
        if world_size <= 1:
            return self.sum / self.count if self.count != 0 else nan
        device = self._infer_device()
        synced_tensor = torch.tensor([self.sum, self.count], dtype=torch.float64, device=device)
        dist.all_reduce(synced_tensor)
        val, count = synced_tensor.tolist()
        if count == 0:
            return nan
        return val / count

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

    def _infer_device(self) -> torch.device:
        if self.device is not None:
            return self.device

        device: torch.device | None = None

        if dist.is_available() and dist.is_initialized():
            backend = dist.get_backend()
            if backend in {"nccl", "cuda"} and torch.cuda.is_available():
                try:
                    index = torch.cuda.current_device()
                except (AssertionError, RuntimeError):
                    index = 0
                device = torch.device("cuda", index)
            elif backend in {"gloo", "mpi"}:
                device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                try:
                    index = torch.cuda.current_device()
                except (AssertionError, RuntimeError):
                    index = 0
                device = torch.device("cuda", index)
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        if device is None:
            warn("Failed to infer device, defaulting to CPU.")
            device = torch.device("cpu")

        self.device = device
        return device


class AverageMeters(MetersBase):
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
        - [`MetricMeters`][danling.metrics.metric_meter.MetricMeters]:
            Memory-efficient metric tracker that averages multiple metrics batch-by-batch.
    """

    meter_cls = AverageMeter

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
