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

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional, Tuple

from torch import Tensor

from ..tensors import NestedTensor
from .average_meter import AverageMeter, AverageMeters, MultiTaskAverageMeters
from .preprocesses import preprocess as default_preprocess
from .utils import MultiTaskDict


class MetricMeter(AverageMeter):
    r"""
    Computes metrics and averages them over time.

    Attributes:
        metric: Metric function for computing the value.
        ignore_index: Index to be ignored in the computation.
        ignore_nan: Whether to ignore NaN values in the computation.
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
        0.0
        >>> meter.avg
        nan
    """

    metric: Callable
    preprocess: Optional[Callable] = None
    ignore_index: int = -100
    ignore_nan: bool = True

    def __init__(
        self,
        metric: Callable,
        preprocess: Callable | None = default_preprocess,
        ignore_index: int | None = None,
        ignore_nan: bool | None = None,
    ) -> None:
        self.metric = metric
        self.preprocess = preprocess
        if ignore_index is not None:
            self.ignore_index = ignore_index
        if ignore_nan is not None:
            self.ignore_nan = ignore_nan
        super().__init__()

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
        if self.preprocess is not None:
            input, target = self.preprocess(input, target, ignore_index=self.ignore_index, ignore_nan=self.ignore_nan)
        n = len(input)
        super().update(self.metric(input, target).item() * n, n=n)


class MetricMeters(AverageMeters):
    r"""
    Manages multiple metric meters in one object.

    Attributes:
        ignore_index: Index to be ignored in the computation.
            Defaults to None.
        ignore_nan: Whether to ignore NaN values in the computation.
            Defaults to False.

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

    preprocess = None
    ignore_index = -100
    ignore_nan = True

    def __init__(
        self,
        *args,
        ignore_index: int | None = None,
        ignore_nan: bool | None = None,
        preprocess: Callable | None = None,
        **meters,
    ) -> None:
        if args:
            from .metrics import Metrics

            if len(args) == 1 and isinstance(args[0], Metrics):
                metrics = args[0]
                for name, metric in metrics.metrics.items():
                    meters.setdefault(name, metric)
                if preprocess is None:
                    preprocess = metrics.preprocess
                if ignore_index is None:
                    ignore_index = metrics.ignore_index
                if ignore_nan is None:
                    ignore_nan = metrics.ignore_nan
            else:
                for metric in args:
                    if not callable(metric):
                        raise ValueError(f"Expected metric to be callable, but got {type(metric)}")
                    meters.setdefault(metric.__name__, metric)
        if preprocess is None:
            preprocess = default_preprocess
        self.setattr("preprocess", preprocess)
        if ignore_index is not None:
            self.setattr("ignore_index", ignore_index)
        if ignore_nan is not None:
            self.setattr("ignore_nan", ignore_nan)
        for name, meter in meters.items():
            if callable(meter):
                meters[name] = meter = MetricMeter(
                    meter, preprocess=None, ignore_index=self.ignore_index, ignore_nan=self.ignore_nan
                )
            if not isinstance(meter, MetricMeter):
                raise ValueError(f"Expected {name} to be an instance of MetricMeter, but got {type(meter)}")
        super().__init__(default_factory=None, **meters)  # type: ignore[arg-type]

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

        if self.preprocess is not None:
            input, target = self.preprocess(input, target, ignore_index=self.ignore_index, ignore_nan=self.ignore_nan)
        for meter in self.values():
            meter.update(input, target)

    def set(self, name: str, meter: MetricMeter | Callable) -> None:  # type: ignore[override] # pylint: disable=W0237
        if callable(meter):
            meter = MetricMeter(meter, preprocess=None, ignore_index=self.ignore_index, ignore_nan=self.ignore_nan)
        if not isinstance(meter, MetricMeter):
            raise ValueError(f"Expected meter to be an instance of MetricMeter, but got {type(meter)}")
        super().set(name, meter)

    def __repr__(self):
        keys = tuple(i for i in self.keys())
        return f"{self.__class__.__name__}{keys}"


class MultiTaskMetricMeters(MultiTaskAverageMeters):
    r"""
    Examples:
        >>> from danling.metrics.functional import accuracy
        >>> metrics = MultiTaskMetricMeters()
        >>> metrics.dataset1.cls = MetricMeters(acc=accuracy)
        >>> metrics.dataset2 = MetricMeters(acc=accuracy)
        >>> metrics
        MultiTaskMetricMeters(<class 'danling.metrics.metric_meter.MultiTaskMetricMeters'>,
          ('dataset1'): MultiTaskMetricMeters(<class 'danling.metrics.metric_meter.MultiTaskMetricMeters'>,
            ('cls'): MetricMeters('acc',)
          )
          ('dataset2'): MetricMeters('acc',)
        )
        >>> metrics.update({"dataset1.cls": {"input": [0.2, 0.4, 0.5, 0.7], "target": [0, 1, 0, 1]}, "dataset2": ([0.1, 0.4, 0.6, 0.8], [1, 0, 0, 0])})
        >>> f"{metrics:.4f}"
        'dataset1.cls: acc: 0.5000 (0.5000)\ndataset2: acc: 0.2500 (0.2500)'
        >>> metrics.setattr("return_average", True)
        >>> metrics.update({"dataset1.cls": [[0.1, 0.4, 0.6, 0.8], [0, 0, 1, 0]], "dataset2": {"input": [0.2, 0.3, 0.5, 0.7], "target": [0, 0, 0, 1]}})
        >>> f"{metrics:.4f}"
        'dataset1.cls: acc: 0.7500 (0.6250)\ndataset2: acc: 0.7500 (0.5000)'
        >>> metrics.update(dict(loss=""))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ValueError: Metric loss not found in ...
    """  # noqa: E501

    def __init__(self, *args, **kwargs):
        super().__init__(*args, default_factory=MultiTaskMetricMeters, **kwargs)

    def update(  # type: ignore[override] # pylint: disable=W0221
        self,
        values: Mapping[str, Tuple[Tensor | NestedTensor | Sequence, Tensor | NestedTensor | Sequence]],
    ) -> None:
        r"""
        Updates the average and current value in all meters.

        Args:
            input: Input values to compute the metrics.
            target: Target values to compute the metrics.
        """

        for metric, value in values.items():
            if metric not in self:
                raise ValueError(f"Metric {metric} not found in {self}")
            if isinstance(self[metric], MultiTaskMetricMeters):
                for met in self[metric].all_values():
                    if isinstance(value, Mapping):
                        met.update(**value)
                    elif isinstance(value, Sequence):
                        met.update(*value)
                    else:
                        raise ValueError(f"Expected value to be a Mapping or Sequence, but got {type(value)}")
            elif isinstance(self[metric], (MetricMeters, MetricMeter)):
                if isinstance(value, Mapping):
                    self[metric].update(**value)
                elif isinstance(value, Sequence):
                    self[metric].update(*value)
                else:
                    raise ValueError(f"Expected value to be a Mapping or Sequence, but got {type(value)}")
            else:
                raise ValueError(
                    f"Expected {metric} to be an instance of MultiTaskMetricMeters, MetricMeters, "
                    f"or MetricMeter, but got {type(self[metric])}"
                )

    # MultiTaskAverageMeters.get is hacked
    def get(self, name: Any, default=None) -> Any:
        return MultiTaskDict.get(self, name, default)

    def set(  # pylint: disable=W0237
        self,
        name: str,
        metric: MetricMeter | MetricMeters | Callable,  # type: ignore[override]
    ) -> None:
        from .metrics import Metrics

        if isinstance(metric, Metrics):
            metric = MetricMeters(preprocess=metric.preprocess, ignore_index=metric.ignore_index, **metric.metrics)
        elif callable(metric):
            metric = MetricMeter(metric)
        elif not isinstance(metric, (MetricMeter, MetricMeters)):
            raise ValueError(
                f"Expected {metric} to be an instance of MetricMeter or MetricMeters, but got {type(metric)}"
            )
        super().set(name, metric)
