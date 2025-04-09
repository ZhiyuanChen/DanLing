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
from math import nan
from typing import Callable, Iterable

import torch
from chanfig import DefaultDict, FlatDict
from torch import Tensor
from torch import distributed as dist
from torcheval.metrics import Metric

from danling.tensor import NestedTensor
from danling.utils import flist, get_world_size

from .preprocess import base_preprocess
from .utils import RoundDict

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
        val: Metrics computed on the current batch only
        avg: Metrics computed on all accumulated data
        input: The input tensor from the latest batch
        target: The target tensor from the latest batch
        inputs: Concatenation of all input tensors seen so far
        targets: Concatenation of all target tensors seen so far

    Args:
        *args: A single mapping of metrics or callable metric functions
        device: Device to store tensors on
        preprocess: Function to preprocess inputs before computing metrics
        **metrics: Named metric functions to compute

    Examples:
        >>> from danling.metric.functional import auroc, auprc
        >>> from danling.metric.preprocess import preprocess_binary
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
        RoundDict(
          ('auroc'): 0.75
          ('auprc'): 0.8333333730697632
        )
        >>> metrics.avg  # Metrics of all data on all devices
        RoundDict(
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
        >>> metrics.val.round(4)  # Metrics of current batch on current device
        RoundDict(
          ('auroc'): 0.6667
          ('auprc'): 0.5
        )
        >>> metrics.avg.round(4)  # Metrics of all data on all devices
        RoundDict(
          ('auroc'): 0.6667
          ('auprc'): 0.5556
        )
        >>> f"{metrics:.4f}"
        'auroc: 0.6667 (0.6667)\tauprc: 0.5000 (0.5556)'
        >>> metrics = Metrics(auroc=auroc, auprc=auprc, preprocess=preprocess_binary)
        >>> metrics.update([[0.1, 0.4, 0.6, 0.8], [0.1, 0.4, 0.6]], [[0, -100, 1, 0], [0, -100, 1]])
        >>> metrics.input, metrics.target
        (tensor([0.1000, 0.6000, 0.8000, 0.1000, 0.6000]), tensor([0, 1, 0, 0, 1]))

    Notes:
        - `Metrics` stores the complete prediction and target history, which is memory-intensive
          but necessary for metrics like AUROC that operate on the entire dataset.
        - For metrics that can be meaningfully averaged batch-by-batch (like accuracy),
          consider using [`MetricMeter`][danling.metric.metric_meter.MetricMeter] for better memory efficiency.
        - All metrics are synchronized across devices in distributed training environments.

    See Also:
        - [`MetricMeters`][danling.metric.metric_meter.MetricMeters]:
            Memory-efficient metric tracker that averages multiple metrics batch-by-batch.
    """

    metrics: FlatDict[str, Callable]
    preprocess: Callable = base_preprocess
    _input: Tensor
    _target: Tensor
    _inputs: Tensor
    _targets: Tensor

    def __init__(
        self,
        *args,
        device: torch.device | None = None,
        preprocess: Callable = base_preprocess,
        **metrics: Callable,
    ):
        super().__init__(device=device)
        self._add_state("_input", torch.empty(0))
        self._add_state("_target", torch.empty(0))
        self._add_state("_inputs", torch.empty(0))
        self._add_state("_targets", torch.empty(0))
        if args:
            from .metric_meter import MetricMeters

            if len(args) == 1 and isinstance(args[0], MetricMeters):
                meters = args[0]
                for name, meter in meters.items():
                    metrics.setdefault(name, meter.metric)
                if preprocess is base_preprocess:
                    preprocess = meters.getattr("preprocess")
            else:
                for metric in args:
                    if not callable(metric):
                        raise ValueError(f"Expected metric to be callable, but got {type(metric)}")
                    metrics.setdefault(metric.__name__, metric)
        self.metrics = FlatDict(**metrics)
        self.preprocess = preprocess

    def update(self, input: Tensor | NestedTensor | Sequence, target: Tensor | NestedTensor | Sequence) -> None:
        input, target = self.preprocess(input, target)
        world_size = get_world_size()
        if world_size > 1:
            input, target = self._sync(input, world_size), self._sync(target, world_size)
        if (
            isinstance(input, (Tensor, NestedTensor))
            and isinstance(target, (Tensor, NestedTensor))
            and input.ndim == target.ndim + 1
        ):
            input = input.squeeze(-1)
        if isinstance(input, (Tensor, NestedTensor)):
            input = input.detach().to(self.device)
        if isinstance(target, (Tensor, NestedTensor)):
            target = target.detach().to(self.device)
        self._input = input
        self._target = target
        if self._inputs.numel() == 0 or self._targets.numel() == 0:
            self._inputs = input
            self._targets = target
        else:
            self._inputs = torch.cat([self._inputs, input])
            self._targets = torch.cat([self._targets, target])

    def value(self) -> RoundDict[str, float | flist]:
        return self.calculate(self.input, self.target)

    def average(self) -> RoundDict[str, float | flist]:
        return self.calculate(self.inputs, self.targets)

    def compute(self) -> RoundDict[str, float | flist]:
        return self.average()

    @property
    def val(self) -> RoundDict[str, float | flist]:
        return self.value()

    @property
    def avg(self) -> RoundDict[str, float | flist]:
        return self.average()

    @torch.inference_mode()
    def calculate(self, input: Tensor, target: Tensor) -> RoundDict[str, flist | float]:
        if (
            isinstance(input, (Tensor, NestedTensor))
            and input.numel() == 0 == target.numel()
            or isinstance(input, (list, dict))
            and len(input) == 0 == len(target)
        ):
            return RoundDict({name: nan for name in self.metrics.keys()})
        ret = RoundDict()
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

    def _sync(self, tensor: Tensor, world_size: int):
        local_size = torch.tensor([tensor.shape[0]], dtype=torch.int64, device=tensor.device)
        size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        sizes = torch.cat(size_list)
        max_size = sizes.max()

        padded_tensor = torch.empty((max_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        padded_tensor[: tensor.shape[0]] = tensor
        gathered_tensors = [torch.empty_like(padded_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, padded_tensor)
        slices = [gathered_tensors[i][: sizes[i]] for i in range(world_size) if sizes[i] > 0]
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
        self, *args, score_name: str | None = None, best_fn: Callable | None = max, **metrics: FlatDict[str, Callable]
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
