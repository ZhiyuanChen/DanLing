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

from collections.abc import Sequence
from typing import Callable, Optional

from torch import Tensor

from ..tensors import NestedTensor
from .average_meter import AverageMeter, AverageMeters
from .functional import preprocess


class MetricMeter(AverageMeter):
    r"""
    Computes metrics and averages them over time.

    Attributes:
        metric: Metric function for computing the value.
        ignored_index: Index to be ignored in the computation.
        val: Results of current batch on current device.
        bat: Results of current batch on all devices.
        avg: Results of all results on all devices.
        sum: Sum of values.
        count: Number of values.

    See Also:
        [`AverageMeter`]: Average meter for computed values.
        [`MetricMeters`]: Manage multiple metric meters in one object.

    Examples:
        >>> from danling.metrics.functional import accuracy
        >>> meter = MetricMeter(accuracy)
        >>> meter.update([0.1, 0.8, 0.6, 0.2], [0, 1, 0, 0])
        >>> meter.val
        0.75
        >>> meter.avg
        0.75
        >>> meter.update([0.1, 0.7, 0.3, 0.2, 0.8, 0.4], [0, 1, 1, 0, 0, 1])
        >>> meter.val
        0.5
        >>> meter.avg
        0.6
        >>> meter.sum
        6.0
        >>> meter.count
        10
        >>> meter.reset()
        >>> meter.val
        0
        >>> meter.avg
        nan
    """

    metric: Callable
    ignored_index: Optional[int] = None

    def __init__(self, metric: Callable, ignored_index: int | None = None) -> None:
        self.metric = metric
        self.ignored_index = ignored_index
        super().__init__()

    def reset(self) -> None:
        r"""
        Resets the meter.
        """

        self.val = 0
        self.n = 1
        self.sum = 0
        self.count = 0

    def update(  # type: ignore[override] # pylint: disable=W0237
        self,
        input: Tensor | NestedTensor | Sequence,  # pylint: disable=W0622
        target: Tensor | NestedTensor | Sequence,
    ) -> None:
        r"""
        Updates the average and current value in the meter.

        Args:
            value: Value to be added to the average.
            n: Number of values to be added.
        """

        input, target = preprocess(input, target, ignored_index=self.ignored_index)
        super().update(self.metric(input, target).item(), n=len(input))


class MetricMeters(AverageMeters):
    r"""
    Manages multiple metric meters in one object.

    Attributes:
        ignored_index: Index to be ignored in the computation.
            Defaults to None.

    See Also:
        [`MetricMeter`]: Computes metrics and averages them over time.
        [`AverageMeters`]: Average meters for computed values.

    >>> from danling.metrics.functional import accuracy, auroc, auprc
    >>> meters = MetricMeters(acc=accuracy, auroc=auroc, auprc=auprc)
    >>> meters.update([0.1, 0.8, 0.6, 0.2], [0, 1, 0, 0])
    >>> meters.sum.dict()
    {'acc': 3.0, 'auroc': 4.0, 'auprc': 4.0}
    >>> meters.count.dict()
    {'acc': 4, 'auroc': 4, 'auprc': 4}
    >>> meters['auroc'].update([0.2, 0.8], [0, 1])
    >>> meters.sum.dict()
    {'acc': 3.0, 'auroc': 6.0, 'auprc': 4.0}
    >>> meters.count.dict()
    {'acc': 4, 'auroc': 6, 'auprc': 4}
    >>> meters.update([[0.1, 0.7, 0.3, 0.2], [0.8, 0.4]], [[0, 0, 1, 0], [0, 0]])
    >>> meters.sum.dict()
    {'acc': 6.0, 'auroc': 8.4, 'auprc': 5.5}
    >>> meters.count.dict()
    {'acc': 10, 'auroc': 12, 'auprc': 10}
    >>> meters['auroc'].update([0.4, 0.8, 0.6, 0.2], [0, 1, 1, 0])
    >>> meters.avg.dict()
    {'acc': 0.6, 'auroc': 0.775, 'auprc': 0.55}
    >>> meters.update(dict(loss=""))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    TypeError: ...update() missing 1 required positional argument: 'target'
    """

    ignored_index = None

    def __init__(self, *args, ignored_index: int | None = None, **kwargs) -> None:
        self.setattr("ignored_index", ignored_index)
        for meter in args:
            if callable(meter):
                meter = MetricMeter(meter, ignored_index=self.ignored_index)
            if not isinstance(meter, MetricMeter):
                raise ValueError(f"Expected meter to be an instance of MetricMeter, but got {type(meter)}")
        for name, meter in kwargs.items():
            if callable(meter):
                kwargs[name] = meter = MetricMeter(meter, ignored_index=self.ignored_index)
            if not isinstance(meter, MetricMeter):
                raise ValueError(f"Expected {name} to be an instance of MetricMeter, but got {type(meter)}")
        if ignored_index is not None:
            self.setattr("ignored_index", ignored_index)
        super().__init__(*args, default_factory=None, **kwargs)  # type: ignore[arg-type]

    def update(  # type: ignore[override] # pylint: disable=W0221
        self,
        input: Tensor | NestedTensor | Sequence,  # pylint: disable=W0622
        target: Tensor | NestedTensor | Sequence,
    ) -> None:
        r"""
        Updates the average and current value in all meters.

        Args:
            input: Input values to compute the metrics.
            target: Target values to compute the metrics.
        """

        input, target = preprocess(input, target, ignored_index=self.ignored_index)
        for meter in self.values():
            meter.update(input, target)

    def set(self, name: str, meter: MetricMeter | Callable) -> None:  # type: ignore[override] # pylint: disable=W0237
        if callable(meter):
            meter = MetricMeter(meter, ignored_index=self.ignored_index)
        if not isinstance(meter, MetricMeter):
            raise ValueError(f"Expected meter to be an instance of MetricMeter, but got {type(meter)}")
        super().set(name, meter)

    def __repr__(self):
        keys = tuple(i for i in self.keys())
        return f"{self.__class__.__name__}{keys}"
