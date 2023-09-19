# pylint: disable=E1101,W0622

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


class flist(list):  # pylint: disable=R0903
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

    def __init__(self, *args, merge_dict: bool | None = None, **metrics: FlatDict[str, Callable]):
        super().__init__()
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
    def update(self, input: Any, target: Any) -> None:
        if isinstance(input, NestedTensor):
            self._input = input
            self._input_buffer.extend(input.to(self.device).storage)
        else:
            if not isinstance(input, torch.Tensor):
                input = torch.tensor(input)
            self._input = input
            self._input_buffer.append(input.to(self.device))
        if isinstance(target, NestedTensor):
            self._target = target
            self._target_buffer.extend(target.to(self.device).storage)
        else:
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            self._target = target
            self._target_buffer.append(target.to(self.device))

    def compute(self) -> FlatDict[str, float]:
        return self.comp

    def value(self) -> FlatDict[str, float]:
        return self.val

    def average(self) -> FlatDict[str, float]:
        return self.avg

    @property
    def comp(self) -> FlatDict[str, float]:
        return self._compute(self._input, self._target)

    @property
    def val(self) -> FlatDict[str, float]:
        return self._compute(self.input, self.target)

    @property
    def avg(self) -> FlatDict[str, float]:
        return self._compute(self.inputs, self.targets)

    @torch.inference_mode()
    def _compute(self, input: Tensor, target: Tensor) -> flist | float:
        if input.numel() == 0 == target.numel():
            return FlatDict({name: nan for name in self.metrics.keys()})
        ret = FlatDict()
        for name, metric in self.metrics.items():
            score = metric(input, target)
            if isinstance(score, Tensor):
                ret[name] = score.item() if score.numel() == 1 else flist(score.tolist())
            elif isinstance(score, Mapping):
                if self.merge_dict:
                    ret.merge(score)
                else:
                    for n, s in score:
                        ret[f"{name}.{n}"] = s
            else:
                ret[name] = score
        return ret

    @torch.inference_mode()
    def merge_state(self, metrics: Iterable):
        raise NotImplementedError()

    @property
    @torch.inference_mode()
    def input(self):
        if world_size() == 1:
            return self._input
        if isinstance(self._input, Tensor):
            synced_tensor = [torch.zeros_like(self._input) for _ in range(dist.get_world_size())]
            dist.all_gather(synced_tensor, self._input)
            return torch.cat(synced_tensor, 0)
        if isinstance(self._input, NestedTensor):
            synced_tensors = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(synced_tensors, self._input.storage)
            return NestedTensor([i for j in synced_tensors for i in j])
        raise ValueError(f"Expected input to be a Tensor or a NestedTensor, but got {type(self._input)}")

    @property
    @torch.inference_mode()
    def target(self):
        if world_size() == 1:
            return self._target
        if isinstance(self._target, Tensor):
            synced_tensor = [torch.zeros_like(self._target) for _ in range(dist.get_world_size())]
            dist.all_gather(synced_tensor, self._target)
            return torch.cat(synced_tensor, 0)
        if isinstance(self._target, NestedTensor):
            synced_tensors = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(synced_tensors, self._target.storage)
            return NestedTensor([i for j in synced_tensors for i in j])

    @property
    @torch.inference_mode()
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
        if isinstance(self._input, NestedTensor):
            return NestedTensor(self._inputs)
        return torch.cat(self._inputs, 0)

    @property
    @torch.inference_mode()
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
        if isinstance(self._target, NestedTensor):
            return NestedTensor(self._targets)
        return torch.cat(self._targets, 0)

    def __repr__(self):
        keys = tuple(i for i in self.metrics.keys())
        return f"{self.__class__.__name__}{keys}"

    def __format__(self, format_spec):
        val, avg = self.compute(), self.average()
        return "\n".join(
            [f"{key}: {val[key].__format__(format_spec)} ({avg[key].__format__(format_spec)})" for key in self.metrics]
        )


class ScoreMetrics(Metrics):
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

    score_name: str
    best_fn: Callable

    def __init__(
        self, *args, score_name: str | None = None, best_fn: Callable | None = max, **metrics: FlatDict[str, Callable]
    ):
        super().__init__(*args, **metrics)
        self.score_name = score_name or next(iter(self.metrics.keys()))
        self.metric = self.metrics[self.score_name]
        self.best_fn = best_fn or max

    def score(self, scope: str) -> float | flist:
        if scope == "batch":
            return self.batch_score()
        if scope == "average":
            return self.average_score()
        raise ValueError(f"Unknown scope: {scope}")

    def batch_score(self) -> float | flist:
        return self.calculate(self.metric, self.input, self.target)

    def average_score(self) -> float | flist:
        return self.calculate(self.metric, self.inputs, self.targets)
