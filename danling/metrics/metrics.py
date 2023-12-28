# pylint: disable=redefined-builtin
from __future__ import annotations

from collections.abc import Mapping
from math import nan
from typing import Any, Callable, Iterable

import torch
from chanfig import FlatDict
from torch import Tensor
from torch import distributed as dist
from torcheval.metrics import Metric

from danling.tensors import NestedTensor


def world_size() -> int:
    r"""Return the number of processes in the current process group."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


class flist(list):
    def __format__(self, *args, **kwargs):
        return " ".join([x.__format__(*args, **kwargs) for x in self])


class Metrics(Metric):
    r"""
    Metric class wraps around multiple metrics that share the same states.

    Typically, there are many metrics that we want to compute for a single task.
    For example, we usually needs to compute `accuracy`, `auroc`, `auprc` for a classification task.
    Computing them one by one is inefficient, especially when evaluating in a distributed environment.

    To solve this problem, Metrics maintains a shared state for multiple metric functions.

    Attributes:
        metrics: A dictionary of metrics to be computed.
        input: The input tensor of latest batch.
        target: The target tensor of latest batch.
        inputs: All input tensors.
        targets: All target tensors.

    Args:
        *args: A single mapping of metrics.
        **metrics: Metrics.

    Examples:
        >>> from danling.metrics import auroc, auprc
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
        >>> metrics.comp  # Metrics of current batch on current device
        FlatDict(
          ('auroc'): 0.75
          ('auprc'): 0.8333333730697632
        )
        >>> metrics.val  # Metrics of current batch on all devices
        FlatDict(
          ('auroc'): 0.75
          ('auprc'): 0.8333333730697632
        )
        >>> metrics.avg  # Metrics of all data on all devices
        FlatDict(
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
        >>> metrics.comp  # Metrics of current batch on current device
        FlatDict(
          ('auroc'): 0.6666666666666666
          ('auprc'): 0.5
        )
        >>> metrics.val  # Metrics of current batch on all devices
        FlatDict(
          ('auroc'): 0.6666666666666666
          ('auprc'): 0.5
        )
        >>> metrics.avg  # Metrics of all data on all devices
        FlatDict(
          ('auroc'): 0.6666666666666666
          ('auprc'): 0.5555555820465088
        )
    """

    metrics: FlatDict[str, Callable]
    _input: Tensor
    _target: Tensor
    _inputs: list[Tensor]
    _targets: list[Tensor]
    _input_buffer: list[Tensor]
    _target_buffer: list[Tensor]
    score_name: str
    best_fn: Callable
    merge_dict: bool = True

    def __init__(
        self,
        *args,
        merge_dict: bool | None = None,
        device: torch.device | None = None,
        **metrics: FlatDict[str, Callable]
    ):
        super().__init__(device=device)
        self._add_state("_input", torch.empty(0))
        self._add_state("_target", torch.empty(0))
        self._add_state("_inputs", [])
        self._add_state("_targets", [])
        self._add_state("_input_buffer", [])
        self._add_state("_target_buffer", [])
        self.metrics = FlatDict(*args, **metrics)
        if merge_dict is not None:
            self.merge_dict = merge_dict

    @torch.inference_mode()
    def update(self, input: Any, target: Any) -> None:  # pylint: disable=W0221
        if isinstance(input, NestedTensor):
            self._input = input
            self._input_buffer.extend(input.cpu().storage())
        else:
            if not isinstance(input, torch.Tensor):
                input = torch.tensor(input)
            self._input = input
            self._input_buffer.append(input.cpu())
        if isinstance(target, NestedTensor):
            self._target = target
            self._target_buffer.extend(target.cpu().storage())
        else:
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            self._target = target
            self._target_buffer.append(target.cpu())

    def compute(self) -> FlatDict[str, float]:
        return self.comp

    def value(self) -> FlatDict[str, float]:
        return self.val

    def average(self) -> FlatDict[str, float]:
        return self.avg

    @property
    def comp(self) -> FlatDict[str, float]:
        return self.calculate(self._input, self._target)

    @property
    def val(self) -> FlatDict[str, float]:
        return self.calculate(self.input, self.target)

    @property
    def avg(self) -> FlatDict[str, float]:
        return self.calculate(self.inputs.to(self.device), self.targets.to(self.device))

    @torch.inference_mode()
    def calculate(self, input: Tensor, target: Tensor) -> flist | float:
        if input.numel() == 0 == target.numel():
            return FlatDict({name: nan for name in self.metrics.keys()})
        ret = FlatDict()
        for name, metric in self.metrics.items():
            score = self._calculate(metric, input, target)
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
    def _calculate(self, metric, input: Tensor, target: Tensor) -> flist | float:
        if input.numel() == 0 == target.numel():
            return FlatDict({name: nan for name in self.metrics.keys()})
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
        if world_size() == 1:
            return self._input
        if isinstance(self._input, Tensor):
            synced_tensor = [torch.zeros_like(self._input) for _ in range(dist.get_world_size())]
            dist.all_gather(synced_tensor, self._input)
            return torch.cat(synced_tensor, 0)
        if isinstance(self._input, NestedTensor):
            synced_tensors = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(synced_tensors, self._input.storage())
            synced_tensors = [i for j in synced_tensors for i in j]
            try:
                return torch.cat(synced_tensors, 0)
            except RuntimeError:
                return synced_tensors
        raise ValueError(f"Expected _input to be a Tensor or a NestedTensor, but got {type(self._input)}")

    @property
    def target(self):
        if world_size() == 1:
            return self._target
        if isinstance(self._target, Tensor):
            synced_tensor = [torch.zeros_like(self._target) for _ in range(dist.get_world_size())]
            dist.all_gather(synced_tensor, self._target)
            return torch.cat(synced_tensor, 0)
        if isinstance(self._target, NestedTensor):
            synced_tensors = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(synced_tensors, self._target.storage())
            synced_tensors = [i for j in synced_tensors for i in j]
            try:
                return torch.cat(synced_tensors, 0)
            except RuntimeError:
                return synced_tensors
        raise ValueError(f"Expected _target to be a Tensor or a NestedTensor, but got {type(self._target)}")

    @property
    def inputs(self):
        if not self._inputs and not self._input_buffer:
            return torch.empty(0)
        if self._input_buffer:
            if world_size() > 1:
                synced_tensors = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(synced_tensors, self._input_buffer)
                self._inputs.extend([i for j in synced_tensors for i in j])
            else:
                self._inputs.extend(self._input_buffer)
            self._input_buffer = []
        # if isinstance(self._input, NestedTensor):
        #     return NestedTensor(self._inputs)
        try:
            return torch.cat(self._inputs, 0)
        except RuntimeError:
            return self._inputs

    @property
    def targets(self):
        if not self._targets and not self._target_buffer:
            return torch.empty(0)
        if self._target_buffer:
            if world_size() > 1:
                synced_tensors = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(synced_tensors, self._target_buffer)
                self._targets.extend([i for j in synced_tensors for i in j])
            else:
                self._targets.extend(self._target_buffer)
            self._target_buffer = []
        # if isinstance(self._target, NestedTensor):
        #     return NestedTensor(self._targets)
        try:
            return torch.cat(self._targets, 0)
        except RuntimeError:
            return self._targets

    def __repr__(self):
        keys = tuple(i for i in self.metrics.keys())
        return f"{self.__class__.__name__}{keys}"

    def __format__(self, format_spec):
        val, avg = self.compute(), self.average()
        return "\n".join(
            [f"{key}: {val[key].__format__(format_spec)} ({avg[key].__format__(format_spec)})" for key in self.metrics]
        )


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
