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

# pylint: disable=redefined-builtin
from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from inspect import signature
from math import nan
from typing import Callable, Iterable, Optional

import torch
from chanfig import DefaultDict, FlatDict, NestedDict
from torch import Tensor
from torch import distributed as dist
from torcheval.metrics import Metric

from danling.tensor import NestedTensor
from danling.utils import flist, get_world_size

from .utils import MultiTaskDict

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self


class Metrics(Metric):
    r"""
    A comprehensive metric tracking system that maintains the complete history of predictions and labels.

    Metrics is designed for computing evaluations that require access to the entire dataset history,
    such as AUROC, Pearson correlation, or other metrics that cannot be meaningfully averaged batch-by-batch.

    Attributes:
        metrics: A dictionary of metric functions to be computed
        preprocess: Optional preprocessing function to apply to inputs and targets
        ignore_index: Value to ignore in classification tasks (e.g., -100 for padding)
        ignore_nan: Whether to ignore NaN values in regression tasks
        val: Metrics computed on the current batch only
        avg: Metrics computed on all accumulated data
        input: The input tensor from the latest batch
        target: The target tensor from the latest batch
        inputs: Concatenation of all input tensors seen so far
        targets: Concatenation of all target tensors seen so far

    Args:
        *args: A single mapping of metrics or callable metric functions
        device: Device to store tensors on
        ignore_index: Value to ignore in classification tasks
        ignore_nan: Whether to ignore NaN values in regression tasks
        preprocess: Function to preprocess inputs before computing metrics
        **metrics: Named metric functions to compute

    Examples:
        >>> from danling.metric.functional import auroc, auprc, base_preprocess
        >>> metrics = Metrics(auroc=auroc, auprc=auprc)
        >>> metrics
        Metrics('auroc', 'auprc')
        >>> metrics.update([0.2, 0.3, 0.5, 0.7], [0, 1, 0, 1])
        >>> metrics.input  # predicted values of current batch
        tensor([0.2000, 0.3000, 0.5000, 0.7000])
        >>> metrics.target  # ground truth of current batch
        tensor([0, 1, 0, 1])
        >>> metrics.inputs  # predicted values of all data
        tensor([0.2000, 0.3000, 0.5000, 0.7000])
        >>> metrics.targets  # ground truth of all data
        tensor([0, 1, 0, 1])
        >>> metrics.val  # Metrics of current batch on current device
        NestedDict(
          ('auroc'): 0.75
          ('auprc'): 0.8333333730697632
        )
        >>> metrics.avg  # Metrics of all data on all devices
        NestedDict(
          ('auroc'): 0.75
          ('auprc'): 0.8333333730697632
        )
        >>> metrics.update([0.1, 0.4, 0.6, 0.8], [0, 0, 1, 0])
        >>> metrics.input  # predicted values of current batch
        tensor([0.1000, 0.4000, 0.6000, 0.8000])
        >>> metrics.target  # ground truth of current batch
        tensor([0, 0, 1, 0])
        >>> metrics.inputs  # predicted values of all data
        tensor([0.2000, 0.3000, 0.5000, 0.7000, 0.1000, 0.4000, 0.6000, 0.8000])
        >>> metrics.targets  # ground truth of all data
        tensor([0, 1, 0, 1, 0, 0, 1, 0])
        >>> metrics.val  # Metrics of current batch on current device
        NestedDict(
          ('auroc'): 0.6666666666666666
          ('auprc'): 0.5
        )
        >>> metrics.avg  # Metrics of all data on all devices
        NestedDict(
          ('auroc'): 0.6666666666666666
          ('auprc'): 0.5555555820465088
        )
        >>> f"{metrics:.4f}"
        'auroc: 0.6667 (0.6667)\tauprc: 0.5000 (0.5556)'
        >>> metrics = Metrics(auroc=auroc, auprc=auprc, preprocess=base_preprocess)
        >>> metrics.update([[0.1, 0.4, 0.6, 0.8], [0.1, 0.4, 0.6]], [[0, -100, 1, 0], [0, -100, 1]])
        >>> metrics.input, metrics.target
        (tensor([0.1000, 0.6000, 0.8000, 0.1000, 0.6000]), tensor([0, 1, 0, 0, 1]))

    Notes:
        - `Metrics` stores the complete prediction and target history, which is memory-intensive
          but necessary for metrics like AUROC that operate on the entire dataset.
        - For metrics that can be meaningfully averaged batch-by-batch (like accuracy),
          consider using [`MetricMeter`][danling.metric.metric_meter.MetricMeter] for better memory efficiency.
        - The `ignore_index` parameter is useful for handling masked or padded values in classification tasks.
        - The `ignore_nan` parameter helps handle missing values in regression tasks.
        - All metrics are synchronized across devices in distributed training environments.

    See Also:
        - [`MetricMeters`][danling.metric.metric_meter.MetricMeters]:
            Memory-efficient metric tracker that averages multiple metrics batch-by-batch.
    """

    metrics: FlatDict[str, Callable]
    preprocess: Optional[Callable] = None
    ignore_index: int = -100
    ignore_nan: bool = True
    _input: Tensor
    _target: Tensor
    _inputs: Tensor
    _targets: Tensor

    def __init__(
        self,
        *args,
        device: torch.device | None = None,
        ignore_index: int | None = None,
        ignore_nan: bool | None = None,
        preprocess: Callable | None = None,
        **metrics: Callable,
    ):
        super().__init__(device=device)
        self.metrics = FlatDict()
        self._add_state("_input", torch.empty(0))
        self._add_state("_target", torch.empty(0))
        self._add_state("_inputs", torch.empty(0))
        self._add_state("_targets", torch.empty(0))
        self.world_size = get_world_size()
        if args:
            from .metric_meter import MetricMeters

            if len(args) == 1 and isinstance(args[0], MetricMeters):
                meters = args[0]
                for name, meter in meters.items():
                    metrics.setdefault(name, meter.metric)
                if preprocess is None:
                    preprocess = meters.getattr("preprocess")
                if ignore_index is None:
                    ignore_index = meters.ignore_index
                if ignore_nan is None:
                    ignore_nan = meters.ignore_nan
            else:
                for metric in args:
                    if not callable(metric):
                        raise ValueError(f"Expected metric to be callable, but got {type(metric)}")
                    metrics.setdefault(metric.__name__, metric)
        self.preprocess = preprocess
        if ignore_index is not None:
            self.ignore_index = ignore_index
        if ignore_nan is not None:
            self.ignore_nan = ignore_nan
        for name, metric in metrics.items():
            self.metrics[name] = self._preprocess_callable(metric)

    def update(self, input: Tensor | NestedTensor | Sequence, target: Tensor | NestedTensor | Sequence) -> None:
        # convert input and target to Tensor if they are not
        if not isinstance(input, (Tensor, NestedTensor)):
            try:
                input = torch.tensor(input)
            except ValueError:
                input = NestedTensor(input)
        if not isinstance(target, (Tensor, NestedTensor)):
            try:
                target = torch.tensor(target)
            except ValueError:
                target = NestedTensor(target)
        if input.ndim == target.ndim + 1:
            input = input.squeeze(-1)
        # convert input and target to NestedTensor if one of them is
        if isinstance(input, NestedTensor) or isinstance(target, NestedTensor):
            if isinstance(target, NestedTensor) and isinstance(input, NestedTensor):
                input, target = input.concat, target.concat
            elif isinstance(input, NestedTensor):
                input, mask = input.concat, input.mask
                target = target[mask]
            elif isinstance(target, NestedTensor):
                target, mask = target.concat, target.mask
                input = input[mask]
            else:
                raise ValueError(f"Unknown input and target: {input}, {target}")
        if self.preprocess is not None:
            input, target = self.preprocess(input, target, ignore_index=self.ignore_index, ignore_nan=self.ignore_nan)
        if self.world_size > 1:
            input, target = self._sync(input), self._sync(target)
        input, target = input.detach().to(self.device), target.detach().to(self.device)
        self._input = input
        self._target = target
        self._inputs = torch.cat([self._inputs, input]).to(input.dtype)
        self._targets = torch.cat([self._targets, target]).to(target.dtype)

    def value(self) -> NestedDict[str, float | flist]:
        return self.calculate(self.input, self.target)

    def average(self) -> NestedDict[str, float | flist]:
        return self.calculate(self.inputs, self.targets)

    def compute(self) -> NestedDict[str, float | flist]:
        return self.average()

    @property
    def val(self) -> NestedDict[str, float | flist]:
        return self.value()

    @property
    def avg(self) -> NestedDict[str, float | flist]:
        return self.average()

    @torch.inference_mode()
    def calculate(self, input: Tensor, target: Tensor) -> NestedDict[str, flist | float]:
        if (
            isinstance(input, (Tensor, NestedTensor))
            and input.numel() == 0 == target.numel()
            or isinstance(input, (list, dict))
            and len(input) == 0 == len(target)
        ):
            return NestedDict({name: nan for name in self.metrics.keys()})
        ret = NestedDict()
        for name, metric in self.metrics.items():
            score = self._calculate(metric, input, target)
            if isinstance(score, Mapping):
                ret.merge(score)
            else:
                ret[name] = score
        return ret

    @torch.inference_mode()
    def _calculate(self, metric, input: Tensor, target: Tensor) -> flist | float:
        score = metric(input, target)
        if isinstance(score, Tensor):
            return score.item() if score.numel() == 1 else flist(score.tolist())
        return score

    @torch.inference_mode()
    def merge_state(self, metrics: Iterable):
        raise NotImplementedError()

    def _sync(self, tensor: Tensor):
        local_size = torch.tensor([tensor.shape[0]], dtype=torch.int64, device=tensor.device)
        size_list = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        dist.all_gather(size_list, local_size)
        sizes = torch.cat(size_list)
        max_size = sizes.max()

        padded_tensor = torch.empty((max_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        padded_tensor[: tensor.shape[0]] = tensor
        gathered_tensors = [torch.empty_like(padded_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_tensors, padded_tensor)
        slices = [gathered_tensors[i][: sizes[i]] for i in range(self.world_size) if sizes[i] > 0]
        return torch.cat(slices, dim=0)

    @property
    def input(self) -> Tensor:
        return self._input

    @property
    def target(self) -> Tensor:
        return self._target

    @property
    def inputs(self) -> Tensor:
        return self._inputs

    @property
    def targets(self) -> Tensor:
        return self._targets

    def __repr__(self):
        keys = tuple(i for i in self.metrics.keys())
        return f"{self.__class__.__name__}{keys}"

    def __format__(self, format_spec):
        val, avg = self.value(), self.average()
        return "\t".join(
            [f"{key}: {val[key].__format__(format_spec)} ({avg[key].__format__(format_spec)})" for key in self.metrics]
        )

    def reset(self: Self) -> Self:  # pragma: no cover
        r"""
        Reset the metric state variables to their default value.
        The tensors in the default values are also moved to the device of
        the last ``self.to(device)`` call.
        """
        for state_name, default in self._state_name_to_default.items():
            if isinstance(default, Tensor):
                setattr(self, state_name, default.clone().to(self.device))
            elif isinstance(default, list):
                setattr(
                    self,
                    state_name,
                    flist(tensor.clone().to(self.device) for tensor in default),
                )
            elif isinstance(default, dict):
                setattr(
                    self,
                    state_name,
                    DefaultDict(
                        lambda: torch.tensor(0.0, device=self.device),
                        {key: tensor.clone().to(self.device) for key, tensor in default.items()},
                    ),
                )
            elif isinstance(default, (int, float)):
                setattr(self, state_name, default)
            else:
                raise TypeError(
                    f"Invalid type for default value for {state_name}. Received {type(default)},"
                    "but expected ``Tensor``, a list of ``Tensor``,"
                    "a dictionary with ``Tensor``, int, or float."
                )
        return self

    def _preprocess_callable(self, func: Callable) -> Callable:
        if not callable(func):
            raise ValueError(f"Expected func to be callable, but got {type(func)}")
        if "preprocess" not in signature(func).parameters:
            return func
        return partial(func, preprocess=self.preprocess is None)


class ScoreMetrics(Metrics):  # pylint: disable=abstract-method
    r"""
    `ScoreMetrics` is a subclass of Metrics that supports scoring.

    Score is a single value that best represents the performance of the model.
    It is the core metrics that we use to compare different models.
    For example, in classification, we usually use auroc as the score.

    `ScoreMetrics` requires two additional arguments: `score_name` and `best_fn`.
    `score_name` is the name of the metric that we use to compute the score.
    `best_fn` is a function that takes a list of values and returns the best value.
    `best_fn` is only not used by `ScoreMetrics`, it is meant to be accessed by other classes.

    Attributes:
        score_name: The name of the metric that we use to compute the score.
        best_fn: A function that takes a list of values and returns the best value.

    Args:
        *args: A single mapping of metrics.
        score_name: The name of the metric that we use to compute the score. Defaults to the first metric.
        best_fn: A function that takes a list of values and returns the best value. Defaults to `max`.
        **metrics: Metrics.

    Notes:
        - `ScoreMetrics` adds the ability to designate one metric as the "score" metric
        - The score metric is typically used for model selection or early stopping
        - `best_fn` determines how to select the "best" score (e.g., max for accuracy, min for loss)
        - Access the score using `metrics.batch_score` or `metrics.average_score`
    """

    _score_name: str
    _best_fn: Callable

    def __init__(
        self, *args, score_name: str | None = None, best_fn: Callable | None = max, **metrics: NestedDict[str, Callable]
    ):
        super().__init__(*args, **metrics)
        self.score_name = score_name or next(iter(self.metrics.keys()))
        self.metric = self.metrics[self.score_name]
        self.best_fn = best_fn or max

    def get_score(self, scope: str) -> float | flist:
        if scope == "batch":
            return self.batch_score
        if scope == "average":
            return self.average_score
        raise ValueError(f"Unknown scope: {scope}")

    @property
    def batch_score(self) -> float | flist:
        return self._calculate(self.metric, self.input, self.target)

    @property
    def average_score(self) -> float | flist:
        return self._calculate(self.metric, self.inputs, self.targets)

    @property
    def score_name(self) -> str:
        return self._score_name

    @score_name.setter
    def score_name(self, name) -> None:
        if name not in self.metrics:
            raise ValueError(f"score_name must be in {self.metrics.keys()}, but got {name}")
        self._score_name = name

    @property
    def best_fn(self) -> Callable:
        return self._best_fn

    @best_fn.setter
    def best_fn(self, fn: Callable) -> None:
        if not callable(fn):
            raise ValueError(f"best_fn must be callable, but got {type(fn)}")
        self._best_fn = fn


class MultiTaskMetrics(MultiTaskDict):
    r"""
    A container for managing multiple `Metrics` for multiple tasks.

    Typically, we have multiple tasks, and each task has multiple metrics.
    For example, a multi-task model might have a classification task and a regression task.
    We want to compute auroc and accuracy for the classification task,
    and pearson and rmse for the regression task.

    `MultiTaskMetrics` is a mapping from task names to `Metrics` instances.

    Examples:
        >>> from danling.metric.functional import auroc, auprc, pearson, spearman, accuracy, mcc
        >>> metrics = MultiTaskMetrics()
        >>> metrics.dataset1.cls = Metrics(auroc=auroc, auprc=auprc)
        >>> metrics.dataset1.reg = Metrics(pearson=pearson, spearman=spearman)
        >>> metrics.dataset2 = Metrics(auroc=auroc, auprc=auprc)
        >>> metrics  # doctest: +ELLIPSIS
        MultiTaskMetrics(...
          ('dataset1'): MultiTaskMetrics(...
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

    Notes:
        - `MultiTaskMetrics` is a container for managing multiple `Metrics` instances
        - Each task has its own `Metrics` instance for tracking task-specific metrics
        - Updates are passed as a dictionary mapping task names to (input, target) pairs
        - Access values using attribute notation: `metrics.task_name.metric_name`

    See Also:
        - [`MultiTaskMetricMeters`][danling.metric.metric_meter.MultiTaskMetricMeters]:
            Memory-efficient metric tracker that averages multiple metrics batch-by-batch for multi-task learning.
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
            elif isinstance(self[metric], (Metrics, Metric)):
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
        metric: Metrics | Metric,  # type: ignore[override]
    ) -> None:
        from .metric_meter import MetricMeters

        if isinstance(metric, MetricMeters):
            metric = Metrics(
                preprocess=metric.preprocess,
                ignore_index=metric.ignore_index,
                ignore_nan=metric.ignore_nan,
                **{name: meter.metric for name, meter in metric.items()},
            )
        if not isinstance(metric, (Metrics, Metric)):
            raise ValueError(f"Expected {metric} to be an instance of Metrics or Metric, but got {type(metric)}")
        super().set(name, metric)
