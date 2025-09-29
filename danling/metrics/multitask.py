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
from typing import Callable

from torch import Tensor

from danling.tensors import NestedTensor

from .global_metrics import GlobalMetrics
from .stream_metrics import MetricMeter, StreamMetrics
from .utils import MultiTaskBase


class MultiTaskMetrics(MultiTaskBase):
    r"""
    Examples:
        >>> from danling.metrics.functional import accuracy
        >>> metrics = MultiTaskMetrics()
        >>> metrics.dataset1 = StreamMetrics(acc=accuracy)
        >>> metrics.dataset2 = StreamMetrics(acc=accuracy)
        >>> metrics.update({"dataset1": {"input": [0.2, 0.4, 0.6, 0.7], "target": [0, 1, 0, 1]}, "dataset2": ([0.1, 0.4, 0.6, 0.8], [1, 0, 0, 0])})
        >>> f"{metrics:.4f}"
        'dataset1: acc: 0.5000 (0.5000)\ndataset2: acc: 0.2500 (0.2500)'
        >>> metrics.setattr("return_average", True)
        >>> metrics.update({"dataset1": ([0.1, 0.4, 0.6, 0.8], [0, 0, 1, 0]), "dataset2": {"input": [0.2, 0.3, 0.6, 0.7], "target": [0, 0, 0, 1]}})
        >>> round(metrics.avg["average"]["acc"], 4)
        0.5625
        >>> metrics.update(dict(loss=""))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ValueError: Metric loss not found in ...

    Notes:
        - `MultiTaskMetrics` manages nested hierarchies of StreamMetrics for multiple tasks/datasets
        - Supports hierarchical access using dot notation or dictionary-style access
        - All metrics are updated simultaneously with a single `update()` call
        - Provides a structured way to track metrics across different tasks or model components

    See Also:
        - [`MultiTaskMetrics`][danling.metrics.multitask.MultiTaskMetrics]:
            Metric tracker that stores the complete prediction and target history for multi-task learning.
    """  # noqa: E501

    def __init__(self, *args, **kwargs):
        super().__init__(*args, default_factory=MultiTaskMetrics, **kwargs)

    def update(  # type: ignore[override] # pylint: disable=W0221
        self,
        values: Mapping[str, Mapping[str, Tensor | NestedTensor | Sequence]],
    ) -> None:
        r"""
        Updates the average and current value in all metrics.

        Args:
            values: A dictionary mapping metric names to their values.
        """

        for metric, value in values.items():
            if metric not in self:
                raise ValueError(f"Metric {metric} not found in {self}")
            if isinstance(self[metric], MultiTaskMetrics):
                for name, met in self[metric].items():
                    if name in value:
                        val = value[name]
                        if isinstance(val, Mapping):
                            met.update(**val)
                        elif isinstance(val, Sequence):
                            met.update(*val)
                        else:
                            raise ValueError(f"Expected value to be a Mapping or Sequence, but got {type(val)}")
            elif isinstance(self[metric], (GlobalMetrics, StreamMetrics, MetricMeter)):
                if isinstance(value, Mapping):
                    self[metric].update(**value)
                elif isinstance(value, Sequence):
                    self[metric].update(*value)
                else:
                    raise ValueError(f"Expected value to be a Mapping or Sequence, but got {type(value)}")
            else:
                raise ValueError(
                    f"Expected {metric} to be an instance of MultiTaskMetrics, GlobalMetrics, "
                    f"StreamMetrics, or MetricMeter, but got {type(self[metric])}"
                )

    def set(  # pylint: disable=W0237
        self,
        name: str,
        metric: MultiTaskMetrics | MetricMeter | StreamMetrics | Callable,  # type: ignore[override]
    ) -> None:
        if not isinstance(metric, (MultiTaskMetrics, GlobalMetrics, StreamMetrics, MetricMeter)):
            raise ValueError(
                f"Expected {metric} to be an instance of MultiTaskMetrics, GlobalMetrics, "
                f"StreamMetrics, or MetricMeter, but got {type(metric)}"
            )
        super().set(name, metric)
