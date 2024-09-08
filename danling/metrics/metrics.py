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
from math import nan
from typing import Callable, Iterable, Optional

import torch
from chanfig import DefaultDict, FlatDict, NestedDict
from torch import Tensor
from torch import distributed as dist
from torcheval.metrics import Metric

from danling.tensors import NestedTensor
from danling.utils import flist, get_world_size

from .preprocesses import preprocess as default_preprocess
from .utils import MultiTaskDict

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self


class Metrics(Metric):
    r"""
    Metric class wraps around multiple metrics that share the same states.

    Typically, there are many metrics that we want to compute for a single task.
    For example, we usually needs to compute `pearson` and `spearman` for a regression task.
    Unlike `accuracy`, which can uses an average meter to compute the average accuracy,
    `pearson` and `spearman` cannot be computed by averaging the results of multiple batches.
    They need access to all the data to compute the correct results.
    And saving all intermediate results for each tasks is quite inefficient.

    `Metrics` solves this problem by maintaining a shared state for multiple metric functions.

    Attributes:
        metrics: A dictionary of metrics to be computed.A
        ignored_index: Index to be ignored in the computation.
        val: Metric results of current batch on current device.
        bat: Metric results of current batch on all devices.
        avg: Metric results of all results on all devices.
        input: The input tensor of latest batch.
        target: The target tensor of latest batch.
        inputs: All input tensors.
        targets: All target tensors.

    Args:
        *args: A single mapping of metrics.
        **metrics: Metrics.

    Examples:
        >>> from danling.metrics.functional import auroc, auprc
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
        >>> metrics.bat  # Metrics of current batch on all devices
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
        >>> metrics.bat  # Metrics of current batch on all devices
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
        >>> metrics = Metrics(auroc=auroc, auprc=auprc, ignored_index=-100)
        >>> metrics.update([[0.1, 0.4, 0.6, 0.8], [0.1, 0.4, 0.6]], [[0, -100, 1, 0], [0, -100, 1]])
        >>> metrics.input, metrics.target
        (PNTensor([0.1000, 0.6000, 0.8000, 0.1000, 0.6000]), PNTensor([0, 1, 0, 0, 1]))
    """

    metrics: FlatDict[str, Callable]
    preprocess: Callable
    ignored_index: Optional[int] = None
    _input: Tensor
    _target: Tensor
    _inputs: flist
    _targets: flist
    _input_buffer: flist
    _target_buffer: flist
    score_name: str
    best_fn: Callable
    merge_dict: bool = True
    return_nested: bool = False

    def __init__(
        self,
        *args,
        merge_dict: bool | None = None,
        return_nested: bool | None = None,
        device: torch.device | None = None,
        ignored_index: int | None = None,
        preprocess: Callable = default_preprocess,
        **metrics: Callable,
    ):
        super().__init__(device=device)
        self._add_state("_input", torch.empty(0))
        self._add_state("_target", torch.empty(0))
        self._add_state("_inputs", flist())
        self._add_state("_targets", flist())
        self._add_state("_input_buffer", flist())
        self._add_state("_target_buffer", flist())
        self.metrics = FlatDict(*args, **metrics)
        self.preprocess = preprocess
        if merge_dict is not None:
            self.merge_dict = merge_dict
        if return_nested is not None:
            self.return_nested = return_nested
        self.ignored_index = ignored_index

    @torch.inference_mode()
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
            if isinstance(input, NestedTensor) and isinstance(target, Tensor):
                target = input.nested_like(target, strict=False)
            if isinstance(target, NestedTensor) and isinstance(input, Tensor):
                input = target.nested_like(input, strict=False)
        # remove ignored index
        if self.ignored_index is not None:
            if isinstance(input, NestedTensor):
                indices = [i != self.ignored_index for i in target.storage()]
                input = NestedTensor([t[i] for t, i in zip(input.storage(), indices)])
                target = NestedTensor([t[i] for t, i in zip(target.storage(), indices)])
            else:
                input, target = input[target != self.ignored_index], target[target != self.ignored_index]
        # update internal state
        if isinstance(input, NestedTensor):
            self._input = input
            self._input_buffer.extend(input.cpu().storage())  # type: ignore[union-attr]
            self._target = target
            self._target_buffer.extend(target.cpu().storage())  # type: ignore[union-attr]
        else:
            self._input = input
            self._input_buffer.append(input.cpu())  # type: ignore[union-attr]
            self._target = target
            self._target_buffer.append(target.cpu())  # type: ignore[union-attr]

    def compute(self) -> NestedDict[str, float | flist]:
        return self.calculate(self.inputs.to(self.device), self.targets.to(self.device))

    def value(self) -> NestedDict[str, float | flist]:
        input = self._input.concat if isinstance(self._input, NestedTensor) else self._input
        target = self._target.concat if isinstance(self._target, NestedTensor) else self._target
        return self.calculate(input, target)

    def batch(self) -> NestedDict[str, float | flist]:
        return self.calculate(self.input, self.target)

    def average(self) -> NestedDict[str, float | flist]:
        return self.calculate(self.inputs.to(self.device), self.targets.to(self.device))

    @property
    def val(self) -> NestedDict[str, float | flist]:
        return self.value()

    @property
    def bat(self) -> NestedDict[str, float | flist]:
        return self.batch()

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
        input, target = self.preprocess(input, target, ignored_index=self.ignored_index)
        for name, metric in self.metrics.items():
            score = self._calculate(metric, input, target, preprocess=False)
            if isinstance(score, Mapping):
                if self.merge_dict:
                    ret.merge(score)
                else:
                    for n, s in score.items():
                        ret[f"{name}.{n}"] = s
            else:
                ret[name] = score
        return ret

    @torch.inference_mode()
    def _calculate(self, metric, input: Tensor, target: Tensor, preprocess: bool = True) -> flist | float:
        if preprocess:
            input, target = self.preprocess(input, target, ignored_index=self.ignored_index)
        score = metric(input, target)
        if isinstance(score, Tensor):
            return score.item() if score.numel() == 1 else flist(score.tolist())
        return score

    @torch.inference_mode()
    def merge_state(self, metrics: Iterable):
        raise NotImplementedError()

    # Due to an issue with PyTorch, we cannot decorate input/target with @torch.inference_mode()
    # Otherwise, we will encounter the following error when using "gloo" backend:
    # Inplace update to inference tensor outside InferenceMode is not allowed
    @property
    def input(self):
        world_size = get_world_size()
        if world_size == 1:
            if isinstance(self._input, NestedTensor) and not self.return_nested:
                return self._input.concat
            return self._input
        if isinstance(self._input, Tensor):
            synced_tensor = [torch.zeros_like(self._input) for _ in range(world_size)]
            dist.all_gather(synced_tensor, self._input)
            return torch.cat(synced_tensor, 0)
        if isinstance(self._input, NestedTensor):
            synced_tensors = [None for _ in range(world_size)]
            dist.all_gather_object(synced_tensors, self._input.storage())
            synced_tensors = flist(i.to(self.device) for j in synced_tensors for i in j)
            try:
                return torch.cat(synced_tensors, 0)
            except RuntimeError:
                input = NestedTensor(synced_tensors)
                if self.return_nested:
                    return input
                return input.concat
        raise ValueError(f"Expected _input to be a Tensor or a NestedTensor, but got {type(self._input)}")

    @property
    def target(self):
        world_size = get_world_size()
        if world_size == 1:
            if isinstance(self._target, NestedTensor) and not self.return_nested:
                return self._target.concat
            return self._target
        if isinstance(self._target, Tensor):
            synced_tensor = [torch.zeros_like(self._target) for _ in range(world_size)]
            dist.all_gather(synced_tensor, self._target)
            return torch.cat(synced_tensor, 0)
        if isinstance(self._target, NestedTensor):
            synced_tensors = [None for _ in range(world_size)]
            dist.all_gather_object(synced_tensors, self._target.storage())
            synced_tensors = flist(i.to(self.device) for j in synced_tensors for i in j)
            try:
                return torch.cat(synced_tensors, 0)
            except RuntimeError:
                target = NestedTensor(synced_tensors)
                if self.return_nested:
                    return target
                return target.concat
        raise ValueError(f"Expected _target to be a Tensor or a NestedTensor, but got {type(self._target)}")

    @property
    def inputs(self):
        if not self._inputs and not self._input_buffer:
            return torch.empty(0)
        if self._input_buffer:
            world_size = get_world_size()
            if world_size > 1:
                synced_tensors = [None for _ in range(world_size)]
                dist.all_gather_object(synced_tensors, self._input_buffer)
                self._inputs.extend([i for j in synced_tensors for i in j])
            else:
                self._inputs.extend(self._input_buffer)
            self._input_buffer = flist()
        try:
            return torch.cat(self._inputs, 0)
        except RuntimeError:
            inputs = NestedTensor(self._inputs)
            if self.return_nested:
                return inputs
            return inputs.concat

    @property
    def targets(self):
        if not self._targets and not self._target_buffer:
            return torch.empty(0)
        if self._target_buffer:
            world_size = get_world_size()
            if world_size > 1:
                synced_tensors = [None for _ in range(world_size)]
                dist.all_gather_object(synced_tensors, self._target_buffer)
                self._targets.extend([i for j in synced_tensors for i in j])
            else:
                self._targets.extend(self._target_buffer)
            self._target_buffer = flist()
        try:
            return torch.cat(self._targets, 0)
        except RuntimeError:
            targets = NestedTensor(self._targets)
            if self.return_nested:
                return targets
            return targets.concat

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
    """

    _score_name: str
    best_fn: Callable

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


class MultiTaskMetrics(MultiTaskDict):
    r"""
    Examples:
        >>> from danling.metrics.functional import auroc, auprc, pearson, spearman, accuracy, mcc
        >>> metrics = MultiTaskMetrics()
        >>> metrics.dataset1.cls = Metrics(auroc=auroc, auprc=auprc)
        >>> metrics.dataset1.reg = Metrics(pearson=pearson, spearman=spearman)
        >>> metrics.dataset2 = Metrics(auroc=auroc, auprc=auprc)
        >>> metrics
        MultiTaskMetrics(<class 'danling.metrics.metrics.MultiTaskMetrics'>,
          ('dataset1'): MultiTaskMetrics(<class 'danling.metrics.metrics.MultiTaskMetrics'>,
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

    def update(self, values: Mapping[str, Mapping[str, Tensor | NestedTensor | Sequence]]) -> None:
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
        if not isinstance(metric, (Metrics, Metric)):
            raise ValueError(f"Expected {metric} to be an instance of Metrics or Metric, but got {type(metric)}")
        super().set(name, metric)
