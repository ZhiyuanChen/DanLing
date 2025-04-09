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
from functools import partial
from inspect import signature
from typing import Any, Callable, Optional, Tuple

from torch import Tensor

from danling.tensor import NestedTensor

from .average_meter import AverageMeter, AverageMeters, MultiTaskAverageMeters
from .utils import MultiTaskDict


class MetricMeter(AverageMeter):
    r"""
    A memory-efficient metric tracker that computes and averages metrics across batches.

    MetricMeter applies a metric function to each batch and maintains running averages
    without storing the complete history of predictions and labels. This makes it ideal for
    metrics that can be meaningfully averaged across batches (like accuracy or loss).

    Attributes:
        metric: The metric function to compute on each batch
        preprocess: Optional preprocessing function to apply to inputs and targets
        ignore_index: Value to ignore in classification tasks (e.g., -100 for padding)
        ignore_nan: Whether to ignore NaN values in regression tasks
        val: Result from the most recent batch
        bat: Result from the most recent batch, synchronized across devices
        avg: Weighted average of all results so far
        sum: Running sum of (metric × batch_size) values
        count: Running sum of batch sizes

    Args:
        metric: Function that computes a metric given input and target tensors
        preprocess: Function to preprocess inputs before computing the metric
        ignore_index: Value to ignore in classification tasks
        ignore_nan: Whether to ignore NaN values in regression tasks

    Examples:
        >>> from danling.metric.functional import accuracy
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
        MetricMeter(accuracy)
        >>> meter.val
        nan
        >>> meter.avg
        nan

    Notes:
        - MetricMeter is more memory-efficient than [`Metrics`][danling.metric.metrics.Metrics]
          because it only stores running statistics
        - Only suitable for metrics that can be meaningfully averaged batch-by-batch
        - Not suitable for metrics like AUROC that need the entire dataset
        - The metric function should accept input and target tensors and return a scalar value
        - For multiple metrics, use [`MetricMeters`][danling.metric.metric_meter.MetricMeters]

    See Also:
        - [`AverageMeter`][danling.metric.average_meter.AverageMeter]:
            A lightweight utility to compute and store running averages of values.
    """

    metric: Callable
    preprocess: Optional[Callable] = None
    ignore_index: int = -100
    ignore_nan: bool = True

    def __init__(
        self,
        metric: Callable,
        preprocess: Callable | None = None,
        ignore_index: int | None = None,
        ignore_nan: bool | None = None,
    ) -> None:
        super().__init__()
        self.preprocess = preprocess
        if ignore_index is not None:
            self.ignore_index = ignore_index
        if ignore_nan is not None:
            self.ignore_nan = ignore_nan
        if preprocess is not None and "preprocess" in signature(metric).parameters:
            metric = partial(metric, preprocess=None)
        self.metric = metric

    def update(  # type: ignore[override] # pylint: disable=W0237
        self,
        input: Tensor | NestedTensor | Sequence,  # pylint: disable=W0622
        target: Tensor | NestedTensor | Sequence,
        preprocessed: bool = False,
    ) -> None:
        r"""
        Updates the average and current value in the meter.

        Args:
            value: Value to be added to the average.
            n: Number of values to be added.
        """
        if self.preprocess is not None and not preprocessed:
            input, target = self.preprocess(input, target, ignore_index=self.ignore_index, ignore_nan=self.ignore_nan)
        n = len(input)
        value = self.metric(input, target)
        if isinstance(value, Tensor):
            value = value.item()
        super().update(value=value, n=n)

    def __repr__(self):
        metric = self.metric
        if isinstance(metric, partial):
            metric = metric.func
        return f"{self.__class__.__name__}({metric.__name__})"


class MetricMeters(AverageMeters):
    r"""
    A container for managing multiple MetricMeter instances with shared preprocessing.

    MetricMeters allows you to organize and track multiple metrics in a unified interface,
    with consistent preprocessing applied to all inputs before computing each metric.
    This is particularly useful when you want to track several metrics that can be
    meaningfully averaged across batches.

    Attributes:
        preprocess: Shared preprocessing function for all meters
        ignore_index: Value to ignore in classification tasks
        ignore_nan: Whether to ignore NaN values in regression tasks
        val: Dictionary of current values from all meters
        avg: Dictionary of running averages from all meters
        sum: Dictionary of sums from all meters
        count: Dictionary of counts from all meters

    Args:
        *args: Either metric functions or a Metrics instance to extract metrics from
        preprocess: Preprocessing function to apply to inputs before computing metrics
        ignore_index: Value to ignore in classification tasks
        ignore_nan: Whether to ignore NaN values in regression tasks
        **meters: Named MetricMeter instances or metric functions

    Examples:
        >>> from danling.metric.functional import accuracy, auroc, auprc, base_preprocess
        >>> meters = MetricMeters(acc=accuracy, auroc=auroc, auprc=auprc, preprocess=base_preprocess)
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
        >>> meters.sum.round(2).dict()
        {'acc': 6.0, 'auroc': 8.4, 'auprc': 5.5}
        >>> meters.count.dict()
        {'acc': 10, 'auroc': 12, 'auprc': 10}
        >>> meters['auroc'].update([0.4, 0.8, 0.6, 0.2], [0, 1, 1, 0])
        >>> meters.avg.round(4).dict()
        {'acc': 0.6, 'auroc': 0.775, 'auprc': 0.55}
        >>> meters.update(dict(loss=""))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        TypeError: ...update() missing 1 required positional argument: 'target'

    Notes:
        - `MetricMeters` manages multiple `MetricMeter` instances with shared preprocessing
        - Each metric is computed independently but uses the same inputs
        - All meters are updated simultaneously when you call `update()`
        - Individual meters can be accessed like dictionary items or attributes

    See Also:
        - [`AverageMeters`][danling.metric.average_meter.AverageMeters]:
            A container for managing multiple average meters in one object.
        - [`Metrics`][danling.metric.metrics.Metrics]:
            Metric tracker that stores the complete prediction and target history.
    """

    preprocess = None
    ignore_index = -100
    ignore_nan = True

    def __init__(
        self,
        *args,
        preprocess: Callable | None = None,
        ignore_index: int | None = None,
        ignore_nan: bool | None = None,
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
        self.setattr("preprocess", preprocess)
        if ignore_index is not None:
            self.setattr("ignore_index", ignore_index)
        if ignore_nan is not None:
            self.setattr("ignore_nan", ignore_nan)
        for name, meter in meters.items():
            if callable(meter):
                meters[name] = meter = self._preprocess_callable(meter)
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

        preprocessed = False
        if self.preprocess is not None:
            input, target = self.preprocess(input, target, ignore_index=self.ignore_index, ignore_nan=self.ignore_nan)
            preprocessed = True
        if (
            isinstance(input, (Tensor, NestedTensor))
            and isinstance(target, (Tensor, NestedTensor))
            and input.ndim == target.ndim + 1
        ):
            input = input.squeeze(-1)
        if isinstance(input, (Tensor, NestedTensor)):
            input = input.detach().to("cpu")
        if isinstance(target, (Tensor, NestedTensor)):
            target = target.detach().to("cpu")
        for meter in self.values():
            meter.update(input, target, preprocessed=preprocessed)

    def set(self, name: str, meter: MetricMeter | Callable) -> None:  # type: ignore[override] # pylint: disable=W0237
        if callable(meter):
            meter = self._preprocess_callable(meter)
        if not isinstance(meter, MetricMeter):
            raise ValueError(f"Expected meter to be an instance of MetricMeter, but got {type(meter)}")
        super().set(name, meter)

    def _preprocess_callable(self, func: Callable) -> MetricMeter:
        if not callable(func):
            raise ValueError(f"Expected func to be callable, but got {type(func)}")
        if "preprocess" in signature(func).parameters:
            func = partial(func, preprocess=self.preprocess is None)
        return MetricMeter(func, preprocess=self.preprocess, ignore_index=self.ignore_index, ignore_nan=self.ignore_nan)

    def __repr__(self):
        keys = tuple(i for i in self.keys())
        return f"{self.__class__.__name__}{keys}"


class MultiTaskMetricMeters(MultiTaskAverageMeters):
    r"""
    Examples:
        >>> from danling.metric.functional import accuracy
        >>> metrics = MultiTaskMetricMeters()
        >>> metrics.dataset1.cls = MetricMeters(acc=accuracy)
        >>> metrics.dataset2 = MetricMeters(acc=accuracy)
        >>> metrics
        MultiTaskMetricMeters(<class 'danling.metric.metric_meter.MultiTaskMetricMeters'>,
          ('dataset1'): MultiTaskMetricMeters(<class 'danling.metric.metric_meter.MultiTaskMetricMeters'>,
            ('cls'): MetricMeters('acc',)
          )
          ('dataset2'): MetricMeters('acc',)
        )
        >>> metrics.update({"dataset1.cls": {"input": [0.2, 0.4, 0.6, 0.7], "target": [0, 1, 0, 1]}, "dataset2": ([0.1, 0.4, 0.6, 0.8], [1, 0, 0, 0])})
        >>> f"{metrics:.4f}"
        'dataset1.cls: acc: 0.5000 (0.5000)\ndataset2: acc: 0.2500 (0.2500)'
        >>> metrics.setattr("return_average", True)
        >>> metrics.update({"dataset1.cls": [[0.1, 0.4, 0.6, 0.8], [0, 0, 1, 0]], "dataset2": {"input": [0.2, 0.3, 0.6, 0.7], "target": [0, 0, 0, 1]}})
        >>> f"{metrics:.4f}"
        'dataset1.cls: acc: 0.7500 (0.6250)\ndataset2: acc: 0.7500 (0.5000)'
        >>> metrics.update(dict(loss=""))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ValueError: Metric loss not found in ...

    Notes:
        - `MultiTaskMetricMeters` manages nested hierarchies of MetricMeters for multiple tasks/datasets
        - Supports hierarchical access using dot notation or dictionary-style access
        - All metrics are updated simultaneously with a single `update()` call
        - Provides a structured way to track metrics across different tasks or model components

    See Also:
        - [`MultiTaskAverageMeters`][danling.metric.average_meter.MultiTaskAverageMeters]:
            A container for managing multiple average meters in one object for multi-task learning.
        - [`MultiTaskMetrics`][danling.metric.metrics.MultiTaskMetrics]:
            Metric tracker that stores the complete prediction and target history for multi-task learning.
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
        meters: MetricMeter | MetricMeters | Callable,  # type: ignore[override]
    ) -> None:
        from .metrics import Metrics

        if isinstance(meters, Metrics):
            meters = MetricMeters(meters)
        elif callable(meters):
            meters = MetricMeter(meters)
        if not isinstance(meters, (MetricMeter, MetricMeters)):
            raise ValueError(
                f"Expected {meters} to be an instance of MetricMeter or MetricMeters, but got {type(meters)}"
            )
        super().set(name, meters)
