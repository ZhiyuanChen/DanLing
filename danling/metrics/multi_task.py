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

# pylint: disable=redefined-builtin
from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import Tensor
from torcheval.metrics import Metric as EvalMetric
from torchmetrics import Metric as TorchMetric

from danling.tensors import NestedTensor

from .metric_meter import MetricMeter
from .metrics import Metrics
from .utils import MultiTaskDict


class MultiTaskMetrics(MultiTaskDict):
    r"""
    Examples:
        >>> from danling.metrics.functional import auroc, auprc, pearson, spearman, accuracy, mcc
        >>> from torcheval import
        >>> metrics = MultiTaskMetrics()
        >>> metrics.dataset1.cls = Metrics(auroc=auroc, auprc=auprc)
        >>> metrics.dataset1.reg = Metrics(pearson=pearson, spearman=spearman)
        >>> metrics.dataset2 = Metrics(auroc=auroc, auprc=auprc)
        >>> metrics
        MultiTaskMetrics(<class 'danling.metrics.multi_task.MultiTaskMetrics'>,
          ('dataset1'): MultiTaskMetrics(<class 'danling.metrics.multi_task.MultiTaskMetrics'>,
            ('cls'): Metrics('auroc', 'auprc')
            ('reg'): Metrics('pearson', 'spearman')
          )
          ('dataset2'): Metrics('auroc', 'auprc')
        )
        >>> metrics.update({"dataset1.cls": {"input": [0.2, 0.4, 0.5, 0.7], "target": [0, 1, 0, 1]}, "dataset1.reg": {"input": [0.1, 0.4, 0.6, 0.8], "target": [0.2, 0.3, 0.5, 0.7]}, "dataset2": {"input": [0.1, 0.4, 0.6, 0.8], "target": [0, 1, 0, 1]}})
        >>> f"{metrics:.4f}"
        'dataset1.cls: auroc: 0.7500 (0.7500)\tauprc: 0.8333 (0.8333)\ndataset1.reg: pearson: 0.9691 (0.9691)\tspearman: 1.0000 (1.0000)\ndataset2: auroc: 0.7500 (0.7500)\tauprc: 0.8333 (0.8333)'
        >>> metrics.setattr("return_average", True)
        >>> metrics.update({"dataset1.cls": {"input": [0.1, 0.4, 0.6, 0.8], "target": [0, 0, 1, 0]}, "dataset1.reg": {"input": [0.2, 0.3, 0.5, 0.7], "target": [0.2, 0.4, 0.6, 0.8]}, "dataset2": {"input": [0.2, 0.3, 0.5, 0.7], "target": [0, 0, 1, 0]}})
        >>> f"{metrics:.4f}"
        'dataset1.cls: auroc: 0.6667 (0.7000)\tauprc: 0.5000 (0.5556)\ndataset1.reg: pearson: 0.9898 (0.9146)\tspearman: 1.0000 (0.9222)\ndataset2: auroc: 0.6667 (0.7333)\tauprc: 0.5000 (0.7000)'
        >>> metrics.update({"dataset1": {"cls": {"input": [0.1, 0.4, 0.6, 0.8], "target": [1, 0, 1, 0]}}})
        >>> f"{metrics:.4f}"
        'dataset1.cls: auroc: 0.2500 (0.5286)\tauprc: 0.5000 (0.4789)\ndataset1.reg: pearson: 0.9898 (0.9146)\tspearman: 1.0000 (0.9222)\ndataset2: auroc: 0.6667 (0.7333)\tauprc: 0.5000 (0.7000)'
        >>> metrics.update(dict(loss=""))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ValueError: Metric loss not found in ...
    """  # noqa: E501

    def __init__(self, *args, **kwargs):
        super().__init__(*args, default_factory=MultiTaskMetrics, **kwargs)

    def update(
        self,
        values: Mapping[str, Mapping[str, Tensor | NestedTensor | Sequence]],
    ) -> None:
        r"""
        Updates the average and current value in all metrics.

        Args:
            values: Dict of values to be added to the average.

        Raises:
            ValueError: If the value is not an instance of (Mapping).
        """

        for metric, value in values.items():
            if metric not in self:
                raise ValueError(f"Metric {metric} not found in {self}")
            if isinstance(self[metric], MultiTaskMetrics):
                for name, met in self[metric].items():
                    if name in value:
                        val = value[name]
                        if isinstance(value, Mapping):
                            met.update(**val)
                        elif isinstance(value, Sequence):
                            met.update(*val)
                        else:
                            raise ValueError(f"Expected value to be a Mapping or Sequence, but got {type(value)}")
            elif isinstance(self[metric], (Metrics, MetricMeter, TorchMetric, EvalMetric)):
                if isinstance(value, Mapping):
                    self[metric].update(**value)
                elif isinstance(value, Sequence):
                    self[metric].update(*value)
                else:
                    raise ValueError(f"Expected value to be a Mapping or Sequence, but got {type(value)}")
            else:
                raise ValueError(
                    f"Expected {metric} to be an instance of MultiTaskMetrics, Metrics, or Metric, "
                    "but got {type(self[metric])}"
                )

    def set(  # pylint: disable=W0237
        self,
        name: str,
        metric: Metrics | MetricMeter | TorchMetric | EvalMetric,  # type: ignore[override]
    ) -> None:
        if not isinstance(metric, (Metrics, MetricMeter, TorchMetric, EvalMetric)):
            raise ValueError(
                f"Expected {metric} to be an instance of Metrics, MetricMeter, torcheval.Metric, or torcemetric.Metric",
                f"but got {type(metric)}",
            )
        super().set(name, metric)
