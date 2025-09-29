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

import torch
from torch import Tensor

from danling.tensors import NestedTensor

from .average_meter import AverageMeter, AverageMeters
from .functional.utils import MetricFunc
from .preprocess import base_preprocess
from .state import MetricState
from .utils import infer_metric_name, iter_metric_funcs, merge_metric_entries


class MetricMeter(AverageMeter):
    r"""
    A memory-efficient metric tracker that computes and averages metrics across batches.

    MetricMeter applies a metric function to each batch and maintains running averages
    without storing the complete history of predictions and labels. This makes it ideal for
    metrics that can be meaningfully averaged across batches (like accuracy or loss).

    Attributes:
        metric: The metric function to compute on each batch
        preprocess: Optional preprocessing function applied before the metric
        val: Result from the most recent batch on the current rank
        bat: Synchronized metric result for the current step
        avg: Weighted average of all results so far
        sum: Running sum of (metric × batch_size) values
        count: Running sum of batch sizes

    Args:
        metric: Function that computes a metric given input and target tensors
        preprocess: Optional preprocessing function to apply before computing the metric

    Examples:
        >>> import torch
        >>> from danling.metrics.functional import accuracy
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
        - MetricMeter is more memory-efficient than [`GlobalMetrics`][danling.metrics.global_metrics.GlobalMetrics]
          because it only stores running statistics
        - Only suitable for metrics that can be meaningfully averaged batch-by-batch
        - Not suitable for metrics like AUROC that need the entire dataset
        - Metrics are evaluated once per update; batch-vs-sample semantics are determined by the metric itself
        - Stream metrics may return tensors; tensor outputs are averaged elementwise across batches
        - `MetricFunc` descriptors receive [`MetricState`][danling.metrics.MetricState]
        - Plain callables receive preprocessed `input` / `target` tensors
        - For multiple metrics, use [`StreamMetrics`][danling.metrics.stream_metrics.StreamMetrics]

    See Also:
        - [`AverageMeter`][danling.metrics.average_meter.AverageMeter]:
            A lightweight utility to compute and store running averages of values.
    """

    metric: Callable | MetricFunc
    output_name: str | None = None

    # Construction
    def __init__(
        self,
        metric: Callable | MetricFunc,
        *,
        preprocess: Callable | None = None,
        device: torch.device | str | None = None,
        distributed: bool = True,
    ) -> None:
        super().__init__(device=device, distributed=distributed)
        if not callable(metric):
            raise ValueError(f"Expected metric to be callable, but got {type(metric)}")
        self.metric = metric
        self.preprocess = preprocess
        self._requirements = MetricState.collect_requirements((metric,)) if isinstance(metric, MetricFunc) else None

    # Mutation
    def update(  # type: ignore[override] # pylint: disable=W0237
        self,
        input: Tensor | NestedTensor,  # pylint: disable=W0622
        target: Tensor | NestedTensor,
        *,
        n: int | None = None,
    ) -> None:
        r"""
        Updates the average and current value in the meter.

        Args:
            input: Prediction tensor or nested tensor.
            target: Ground-truth tensor or nested tensor.
            n: Optional number of samples represented by this update. When omitted,
                the batch size is inferred from the inputs.
        """

        if self.preprocess is not None:
            input, target = self.preprocess(input, target)

        self._update_state(self._build_state(input, target), n=n)

    # Internal helpers
    def _update_state(self, state: MetricState, *, n: int | None = None) -> None:
        if n is None:
            try:
                n = len(state.preds)
            except TypeError:
                n = 1

        value = self._compute_metric(state)
        super().update(value=value, n=n)

    def _compute_metric(self, state: MetricState) -> Tensor | float:
        if isinstance(self.metric, MetricFunc):
            return self._normalize_value(self.metric(state))
        return self._normalize_value(self.metric(state.preds, state.targets))

    def _build_state(
        self,
        input: Tensor | NestedTensor | Sequence,
        target: Tensor | NestedTensor | Sequence,
    ) -> MetricState:
        return MetricState.from_requirements(input, target, self._requirements)

    @staticmethod
    def _normalize_value(value: Tensor | float | int) -> Tensor | float:
        if isinstance(value, Tensor):
            if value.numel() == 0:
                return torch.tensor(float("nan"))
            if value.numel() == 1:
                return value.item()
            return value.detach()
        return float(value)

    def __repr__(self):
        metric = self.metric
        if isinstance(metric, MetricFunc):
            return f"{self.__class__.__name__}({metric.name})"
        if isinstance(metric, partial):
            metric = metric.func
        return f"{self.__class__.__name__}({metric.__name__})"


class StreamMetrics(AverageMeters):
    r"""
    A container for managing multiple MetricMeter instances with shared preprocessing.

    StreamMetrics allows you to organize and track multiple metrics in a unified interface,
    with consistent preprocessing applied to all inputs before computing each metric.
    This is particularly useful when you want to track several metrics that can be
    meaningfully averaged across batches.

    Attributes:
        preprocess: Shared preprocessing function for all meters
        val: Dictionary of current local values from all meters
        bat: Dictionary of synchronized current-step values from all meters
        avg: Dictionary of running averages from all meters
        sum: Dictionary of sums from all meters
        count: Dictionary of counts from all meters

    Args:
        *args: Metric functions to register as meters
        preprocess: Preprocessing function to apply to inputs before computing metrics
        **meters: Named MetricMeter instances or metric functions

    Examples:
        >>> import torch
        >>> from danling.metrics.functional import accuracy
        >>> meters = StreamMetrics(acc=accuracy)
        >>> meters.update([0.1, 0.8, 0.6, 0.2], [0, 1, 0, 0])
        >>> round(meters.val["acc"], 4)
        0.75
        >>> round(meters.avg["acc"], 4)
        0.75
        >>> meters["acc"].update(torch.tensor([0.2, 0.8]), torch.tensor([0, 1]))
        >>> meters.count["acc"]
        6
        >>> meters.update(dict(loss=""))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        TypeError: ...update() missing 1 required positional argument: 'target'

    Notes:
        - `StreamMetrics` manages multiple `MetricMeter` instances with shared preprocessing
        - Each metric is computed independently but uses the same inputs
        - All meters are updated simultaneously when you call `update()`
        - Individual meters can be accessed like dictionary items or attributes
        - Metrics are evaluated once per update; batch-vs-sample semantics are determined by the metric itself
        - Tensor-valued metrics are preserved and averaged elementwise across batches
        - Built-in `MetricFunc` stream values may be approximate rather than exact dataset-level metrics

    See Also:
        - [`AverageMeters`][danling.metrics.average_meter.AverageMeters]:
            A container for managing multiple average meters in one object.
        - [`GlobalMetrics`][danling.metrics.global_metrics.GlobalMetrics]:
            Metric tracker that stores the complete prediction and target history.
    """

    preprocess = base_preprocess
    meter_cls = MetricMeter  # type: ignore[assignment]

    # Construction
    def __init__(
        self,
        *metric_funcs,
        preprocess: Callable = base_preprocess,
        distributed: bool = True,
        device: torch.device | str | None = None,
        **meters,
    ) -> None:
        self.setattr("_initializing_meters", True)
        self.setattr("_requirements", None)
        positional: list[tuple[str, Callable | MetricMeter]] = []
        for metric in iter_metric_funcs(metric_funcs):
            if not callable(metric):
                raise ValueError(f"Expected metric to be callable, but got {type(metric)}")
            positional.append((infer_metric_name(metric), metric))

        named: dict[str, Callable | MetricMeter] = {}
        for name, metric in meters.items():
            if not isinstance(metric, MetricMeter) and not callable(metric):
                raise ValueError(f"Expected metric to be callable or MetricMeter, but got {type(metric)}")
            named[name] = metric

        meters = merge_metric_entries(positional, named)
        self.setattr("preprocess", preprocess)
        self.setattr("distributed", distributed)
        self.setattr("device", torch.device(device) if device is not None else None)
        try:
            super().__init__(**meters)
        finally:
            self.setattr("_initializing_meters", False)
        self._refresh_requirements()

    # Meter registration
    def _coerce_metric(self, value: Callable | MetricFunc | MetricMeter) -> MetricMeter:
        meter_cls: type[MetricMeter] = getattr(type(self), "meter_cls", MetricMeter)
        if isinstance(value, meter_cls):
            value.preprocess = None
            if self.device is not None:
                value.device = self.device
            value.distributed = self.distributed
            value._requirements = (
                MetricState.collect_requirements((value.metric,)) if isinstance(value.metric, MetricFunc) else None
            )
            return value
        if callable(value):
            return meter_cls(value, preprocess=None, device=self.device, distributed=self.distributed)
        raise ValueError(f"Expected meter to be an instance of {meter_cls.__name__}, but got {type(value)}")

    def _coerce_meter(self, value):  # type: ignore[override]
        return self._coerce_metric(value)

    def set(self, name, value) -> None:  # type: ignore[override]
        super().set(name, value)
        if not self.getattr("_initializing_meters", False):
            self._refresh_requirements()

    # Mutation
    def update(  # type: ignore[override] # pylint: disable=W0221
        self,
        input: Tensor | NestedTensor | Sequence,  # pylint: disable=W0622
        target: Tensor | NestedTensor | Sequence,
        *,
        n: int | None = None,
    ) -> None:
        r"""
        Updates the average and current value in all meters.

        Args:
            input: Input values to compute the metrics.
            target: Target values to compute the metrics.
            n: Optional number of samples represented by this update. Defaults to
                the inferred batch size.
        """

        input, target = self.preprocess(input, target)  # type: ignore[arg-type]
        if isinstance(input, (Tensor, NestedTensor)):
            input = input.detach()
        if isinstance(target, (Tensor, NestedTensor)):
            target = target.detach()
        state = self._build_state(input, target)
        for meter in self.values():
            if isinstance(meter, MetricMeter):
                meter._update_state(state, n=n)
            else:
                meter.update(input, target, n=n)

    # Internal helpers
    def _collect_requirements_from_meters(self):
        metric_funcs = []
        for meter in self.values():
            if isinstance(meter, MetricMeter) and isinstance(meter.metric, MetricFunc):
                metric_funcs.append(meter.metric)
        if not metric_funcs:
            return None
        return MetricState.collect_requirements(metric_funcs)

    def _refresh_requirements(self) -> None:
        self.setattr("_requirements", self._collect_requirements_from_meters())

    def _build_state(
        self,
        input: Tensor | NestedTensor | Sequence,
        target: Tensor | NestedTensor | Sequence,
    ) -> MetricState:
        return MetricState.from_requirements(input, target, self._requirements)

    # Formatting helpers
    def __repr__(self):
        keys = tuple(i for i in self.keys())
        return f"{self.__class__.__name__}{keys}"
