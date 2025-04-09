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

from collections.abc import Sequence
from functools import partial
from typing import Callable

from torch import Tensor

from danling.tensor import NestedTensor

from .average_meter import AverageMeter, AverageMeters
from .preprocess import base_preprocess


class MetricMeter(AverageMeter):
    r"""
    A memory-efficient metric tracker that computes and averages metrics across batches.

    MetricMeter applies a metric function to each batch and maintains running averages
    without storing the complete history of predictions and labels. This makes it ideal for
    metrics that can be meaningfully averaged across batches (like accuracy or loss).

    Attributes:
        metric: The metric function to compute on each batch
        preprocess: Optional preprocessing function to apply to inputs and targets
        val: Result from the most recent batch
        bat: Result from the most recent batch, synchronized across devices
        avg: Weighted average of all results so far
        sum: Running sum of (metric × batch_size) values
        count: Running sum of batch sizes

    Args:
        metric: Function that computes a metric given input and target tensors
        preprocess: Function to preprocess inputs before computing the metric

    Examples:
        >>> import torch
        >>> from danling.metric.functional import accuracy
        >>> meter = MetricMeter(accuracy)
        >>> meter.update(torch.tensor([0.1, 0.8, 0.6, 0.2]), torch.tensor([0, 1, 0, 0]))
        >>> meter.val
        0.75
        >>> meter.avg
        0.75
        >>> meter.update(torch.tensor([0.1, 0.7, 0.3, 0.2, 0.8, 0.4]), torch.tensor([0, 1, 1, 0, 0, 1]))
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

    def __init__(
        self,
        metric: Callable,
    ) -> None:
        super().__init__()
        if not callable(metric):
            raise ValueError(f"Expected metric to be callable, but got {type(metric)}")
        self.metric = metric

    def update(  # type: ignore[override] # pylint: disable=W0237
        self,
        input: Tensor | NestedTensor,  # pylint: disable=W0622
        target: Tensor | NestedTensor,
    ) -> None:
        r"""
        Updates the average and current value in the meter.

        Args:
            value: Value to be added to the average.
            n: Number of values to be added.
        """
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
        val: Dictionary of current values from all meters
        avg: Dictionary of running averages from all meters
        sum: Dictionary of sums from all meters
        count: Dictionary of counts from all meters

    Args:
        *args: Either metric functions or a Metrics instance to extract metrics from
        preprocess: Preprocessing function to apply to inputs before computing metrics
        **meters: Named MetricMeter instances or metric functions

    Examples:
        >>> import torch
        >>> from danling.metric.functional import accuracy, auroc, auprc
        >>> meters = MetricMeters(acc=accuracy, auroc=auroc, auprc=auprc)
        >>> meters.update([0.1, 0.8, 0.6, 0.2], [0, 1, 0, 0])
        >>> meters.sum.dict()
        {'acc': 3.0, 'auroc': 4.0, 'auprc': 4.0}
        >>> meters.count.dict()
        {'acc': 4, 'auroc': 4, 'auprc': 4}
        >>> meters['auroc'].update(torch.tensor([0.2, 0.8]), torch.tensor([0, 1]))
        >>> meters.sum.dict()
        {'acc': 3.0, 'auroc': 6.0, 'auprc': 4.0}
        >>> meters.count.dict()
        {'acc': 4, 'auroc': 6, 'auprc': 4}
        >>> meters.update([[0.1, 0.7, 0.3, 0.2], [0.8, 0.4]], [[0, 0, 1, 0], [0, 0]])
        >>> meters.sum.round(2).dict()
        {'acc': 6.0, 'auroc': 8.4, 'auprc': 5.5}
        >>> meters.count.dict()
        {'acc': 10, 'auroc': 12, 'auprc': 10}
        >>> meters['auroc'].update(torch.tensor([0.4, 0.8, 0.6, 0.2]), torch.tensor([0, 1, 1, 0]))
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

    preprocess = base_preprocess
    default_cls = MetricMeter

    def __init__(
        self,
        *args,
        preprocess: Callable = base_preprocess,
        **meters,
    ) -> None:
        if args:
            from .metrics import Metrics

            if len(args) == 1 and isinstance(args[0], Metrics):
                metrics = args[0]
                for name, metric in metrics.metrics.items():
                    meters.setdefault(name, metric)
                if preprocess is base_preprocess:
                    preprocess = metrics.preprocess
            else:
                for metric in args:
                    if not callable(metric):
                        raise ValueError(f"Expected metric to be callable, but got {type(metric)}")
                    meters.setdefault(metric.__name__, metric)
        self.setattr("preprocess", preprocess)
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

        input, target = self.preprocess(input, target)  # type: ignore[arg-type]
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
            meter.update(input, target)

    def set(self, name: str, meter: MetricMeter | Callable) -> None:  # type: ignore[override] # pylint: disable=W0237
        if callable(meter):
            meter = self.getattr("default_cls", MetricMeter)(meter)
        if not isinstance(meter, MetricMeter):
            raise ValueError(f"Expected meter to be an instance of MetricMeter, but got {type(meter)}")
        super().set(name, meter)

    def __repr__(self):
        keys = tuple(i for i in self.keys())
        return f"{self.__class__.__name__}{keys}"
