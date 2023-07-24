from functools import partial
from math import nan
from typing import Any, Callable, Iterable, List, Mapping, Optional, Union

import torch
from chanfig import FlatDict
from torch import Tensor
from torch import distributed as dist
from torcheval.metrics import Metric
from torcheval.metrics import functional as tef
from torchmetrics import functional as tmf


class flist(list):  # pylint: disable=R0903
    def __format__(self, *args, **kwargs):
        return " ".join([x.__format__(*args, **kwargs) for x in self])


class Metrics(Metric):
    r"""
    Metric class that wraps around multiple metrics.

    Typically, there are many metrics that we want to compute.
    Computing them one by one is inefficient, especially when evaluating in distributed environment.
    This class wraps around multiple metrics and computes them at the same time.

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
    _inputs: List[Tensor]
    _targets: List[Tensor]
    index: str
    best_fn: Callable

    def __init__(self, *args, **metrics: FlatDict[str, Callable]):
        self.metrics = FlatDict()
        super().__init__()
        self._add_state("_input", torch.empty(0))  # pylint: disable=E1101
        self._add_state("_target", torch.empty(0))  # pylint: disable=E1101
        self._add_state("_inputs", [])
        self._add_state("_targets", [])
        if len(args) == 1 and isinstance(args[0], Mapping):
            self.metrics.merge(args[0])
        elif len(args) != 0:
            raise ValueError("Metrics only accepts a single mapping as positional argument")
        self.metrics.merge(metrics)

    @torch.inference_mode()
    def update(self, input: Any, target: Any) -> None:  # pylint: disable=W0622
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input)  # pylint: disable=E1101
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)  # pylint: disable=E1101
        input, target = input.to(self.device), target.to(self.device)
        self._input, self._target = input, target
        self._inputs.append(input)
        self._targets.append(target)

    @property
    def val(self) -> FlatDict[str, float]:
        return self.compute()

    @property
    def avg(self) -> FlatDict[str, float]:
        return self.average()

    def compute(self) -> FlatDict[str, float]:
        ret = FlatDict()
        for name, metric in self.metrics.items():
            ret[name] = self.calculate(metric, self.input, self.target)
        return ret

    def average(self) -> FlatDict[str, float]:
        ret = FlatDict()
        for name, metric in self.metrics.items():
            ret[name] = self.calculate(metric, self.inputs, self.targets)
        return ret

    @staticmethod
    @torch.inference_mode()
    def calculate(func, input: Tensor, target: Tensor) -> Union[flist, float]:  # pylint: disable=W0622
        if input.numel() == 0 == target.numel():
            return nan
        score = func(input, target)
        return score.item() if score.numel() == 1 else flist(score.tolist())

    @torch.inference_mode()
    def merge_state(self, metrics: Iterable):
        raise NotImplementedError()

    @property
    @torch.inference_mode()
    def input(self):
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return self._input
        synced_input = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(synced_input, self._input)
        return torch.cat([t.to(self.device) for t in synced_input], 0)  # pylint: disable=E1101

    @property
    @torch.inference_mode()
    def target(self):
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return self._target
        synced_target = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(synced_target, self._target)
        return torch.cat([t.to(self.device) for t in synced_target], 0)  # pylint: disable=E1101

    @property
    @torch.inference_mode()
    def inputs(self):
        if not self._inputs:
            return torch.empty(0)  # pylint: disable=E1101
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return torch.cat(self._inputs, 0)  # pylint: disable=E1101
        synced_inputs = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(synced_inputs, self._inputs)
        return torch.cat([t.to(self.device) for i in synced_inputs for t in i], 0)  # pylint: disable=E1101

    @property
    @torch.inference_mode()
    def targets(self):
        if not self._targets:
            return torch.empty(0)  # pylint: disable=E1101
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return torch.cat(self._targets, 0)  # pylint: disable=E1101
        synced_targets = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(synced_targets, self._targets)
        return torch.cat([t.to(self.device) for i in synced_targets for t in i], 0)  # pylint: disable=E1101

    def __repr__(self):
        keys = tuple(i for i in self.metrics.keys())
        return f"{self.__class__.__name__}{keys}"

    def __format__(self, format_spec):
        val, avg = self.compute(), self.average()
        return "\n".join(
            [f"{key}: {val[key].__format__(format_spec)} ({avg[key].__format__(format_spec)})" for key in self.metrics]
        )


class IndexMetrics(Metrics):
    r"""
    IndexMetrics is a subclass of Metrics that supports scoring.

    Score is a single value that best represents the performance of the model.
    It is the core metrics that we use to compare different models.
    For example, in classification, we usually use auroc as the score.

    IndexMetrics requires two additional arguments: `index` and `best_fn`.
    `index` is the name of the metric that we use to compute the score.
    `best_fn` is a function that takes a list of values and returns the best value.
    `best_fn` is only not used by IndexMetrics, it is meant to be accessed by other classes.

    Attributes:
        index: The name of the metric that we use to compute the score.
        best_fn: A function that takes a list of values and returns the best value.

    Args:
        *args: A single mapping of metrics.
        index: The name of the metric that we use to compute the score. Defaults to the first metric.
        best_fn: A function that takes a list of values and returns the best value. Defaults to `max`.
        **metrics: Metrics.
    """

    index: str
    best_fn: Callable

    def __init__(
        self, *args, index: Optional[str] = None, best_fn: Optional[Callable] = max, **metrics: FlatDict[str, Callable]
    ):
        super().__init__(*args, **metrics)
        self.index = index or next(iter(self.metrics.keys()))
        self.metric = self.metrics[self.index]
        self.best_fn = best_fn or max

    def score(self, scope: str) -> Union[float, flist]:
        if scope == "batch":
            return self.batch_score()
        if scope == "average":
            return self.average_score()
        raise ValueError(f"Unknown scope: {scope}")

    def batch_score(self) -> Union[float, flist]:
        return self.calculate(self.metric, self.input, self.target)

    def average_score(self) -> Union[float, flist]:
        return self.calculate(self.metric, self.inputs, self.targets)


def binary_metrics():
    return Metrics(auroc=tef.binary_auroc, auprc=tef.binary_auprc, acc=tef.binary_accuracy)


def multiclass_metrics(num_classes: int):
    auroc = partial(tef.multiclass_auroc, num_classes=num_classes)
    auprc = partial(tef.multiclass_auprc, num_classes=num_classes)
    acc = partial(tef.multiclass_accuracy, num_classes=num_classes)
    return Metrics(auroc=auroc, auprc=auprc, acc=acc)


def multilabel_metrics(num_labels: int):
    auroc = partial(tmf.classification.multilabel_auroc, num_labels=num_labels)
    auprc = partial(tef.multilabel_auprc, num_labels=num_labels)
    return Metrics(auroc=auroc, auprc=auprc, acc=tef.multilabel_accuracy)


def regression_metrics():
    return Metrics(
        pearson=tmf.pearson_corrcoef,
        spearman=tmf.spearman_corrcoef,
        r2=tef.r2_score,
        mse=tef.mean_squared_error,
    )
