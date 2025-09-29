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
from typing import Callable, Literal

from torch import Tensor

from danling.tensors import NestedTensor

from .global_metrics import GlobalMetrics
from .stream_metrics import MetricMeter, StreamMetrics
from .utils import MultiTaskBase, infer_metric_name


class MultiTaskMetrics(MultiTaskBase):
    r"""
    Examples:
        >>> from danling.metrics.functional import accuracy
        >>> metrics = MultiTaskMetrics(aggregate="macro")
        >>> metrics.dataset1 = StreamMetrics(acc=accuracy)
        >>> metrics.dataset2 = StreamMetrics(acc=accuracy)
        >>> metrics.update({"dataset1": {"input": [0.2, 0.4, 0.6, 0.7], "target": [0, 1, 0, 1]}, "dataset2": ([0.1, 0.4, 0.6, 0.8], [1, 0, 0, 0])})
        >>> f"{metrics:.4f}"
        'dataset1: acc: 0.5000 (0.5000)\ndataset2: acc: 0.2500 (0.2500)'
        >>> metrics.update({"dataset1": ([0.1, 0.4, 0.6, 0.8], [0, 0, 1, 0]), "dataset2": {"input": [0.2, 0.3, 0.6, 0.7], "target": [0, 0, 0, 1]}})
        >>> round(metrics.avg["aggregate"]["acc"], 4)
        0.5625
        >>> metrics.update(dict(loss=""))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ValueError: Task loss not found in ...

    Notes:
        - `MultiTaskMetrics` manages a flat collection of task-level metric containers
        - All task containers are updated simultaneously with a single `update()` call
        - Aggregation mode is configured at construction time via `aggregate=...`
        - `aggregate="macro"` gives equal task weight, `aggregate="micro"` weights by sample count,
          and `aggregate="weighted"` uses explicit `aggregate_weights`
        - Aggregate outputs are matched by exact relative metric path across tasks
        - Provides a structured way to track metrics across different tasks or model components

    See Also:
        - [`GlobalMetrics`][danling.metrics.global_metrics.GlobalMetrics]:
            Exact metrics container that stores prediction and target history.
        - [`StreamMetrics`][danling.metrics.stream_metrics.StreamMetrics]:
            Streaming metrics container for hot-path metric tracking.
    """  # noqa: E501

    def __init__(
        self,
        *args,
        aggregate: Literal["macro", "micro", "weighted"] | None = None,
        aggregate_weights: Mapping[str, float | int | Tensor] | None = None,
        **kwargs,
    ):
        super().__init__(*args, aggregate=aggregate, aggregate_weights=aggregate_weights, **kwargs)

    def update(  # type: ignore[override] # pylint: disable=W0221
        self,
        values: Mapping[str, Mapping[str, Tensor | NestedTensor | Sequence] | Sequence],
    ) -> None:
        r"""
        Updates all task metric containers.

        Args:
            values: Mapping from task names to update payloads.
                Mapping payloads are forwarded as keyword arguments to the
                child container's `update()`; sequence payloads are forwarded
                positionally.
        """

        for task, payload in values.items():
            if task not in self:
                raise ValueError(f"Task {task} not found in {self}")
            task_metrics = self[task]
            if isinstance(payload, Mapping):
                task_metrics.update(**payload)
            elif isinstance(payload, Sequence):
                task_metrics.update(*payload)
            else:
                raise ValueError(
                    f"Expected payload for task {task} to be a Mapping or Sequence, but got {type(payload)}"
                )

    def set(  # pylint: disable=W0237
        self,
        name: str,
        task_metrics: GlobalMetrics | MetricMeter | StreamMetrics | Callable,  # type: ignore[override]
    ) -> None:
        if callable(task_metrics) and not isinstance(task_metrics, (GlobalMetrics, StreamMetrics, MetricMeter)):
            task_metrics = MetricMeter(task_metrics)
        if isinstance(task_metrics, MetricMeter):
            task_metrics.output_name = self._metric_output_name(name, task_metrics)
        if not isinstance(task_metrics, (GlobalMetrics, StreamMetrics, MetricMeter)):
            raise ValueError(
                f"Expected task_metrics for {name} to be an instance of GlobalMetrics, "
                f"StreamMetrics, MetricMeter, or a callable, but got {type(task_metrics)}"
            )
        super().set(name, task_metrics)

    @staticmethod
    def _metric_output_name(task_name: str, metric: MetricMeter) -> str:
        output_name = getattr(metric, "output_name", None)
        if isinstance(output_name, str) and output_name not in {"", "<lambda>", "__call__"}:
            return output_name

        try:
            inferred = infer_metric_name(metric.metric)
        except ValueError:
            return task_name
        if inferred in {"", "<lambda>", "__call__"}:
            return task_name
        return inferred
