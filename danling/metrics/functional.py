# pylint: disable=redefined-builtin
from __future__ import annotations

import torch
from torch import Tensor
from torcheval.metrics import functional as tef
from torchmetrics import functional as tmf

from danling.tensors import NestedTensor


def auroc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    weight: Tensor | None = None,
    average: str | None = "macro",
    num_labels: int | None = None,
    num_classes: int | None = None,
    task_weights: Tensor | None = None,
    **kwargs
):
    if isinstance(input, NestedTensor):
        input = torch.cat(input.storage())
    if isinstance(target, NestedTensor):
        target = torch.cat(target.storage())
    if num_labels is None and num_classes is None:
        return tef.binary_auroc(input=input, target=target, weight=weight, **kwargs)
    if num_classes is None:
        ret = tef.binary_auroc(input=input.T, target=target.T, num_tasks=num_labels, weight=weight, **kwargs)
        if task_weights is not None:
            return ret @ task_weights.double()
        return ret.mean()
    if num_labels is None:
        return tef.multiclass_auroc(input=input, target=target, num_classes=num_classes, average=average, **kwargs)
    raise ValueError("Could not infer the type of the task. Only one of `num_labels`, `num_classes` is allowed.")


def auprc(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    average: str | None = "macro",
    num_labels: int | None = None,
    num_classes: int | None = None,
    task_weights: Tensor | None = None,
    **kwargs
):
    if isinstance(input, NestedTensor):
        input = torch.cat(input.storage())
    if isinstance(target, NestedTensor):
        target = torch.cat(target.storage())
    if num_labels is None and num_classes is None:
        return tef.binary_auprc(input=input, target=target, **kwargs)
    if num_classes is None:
        ret = tef.multilabel_auprc(input=input, target=target, num_labels=num_labels, average=average, **kwargs)
        if task_weights is not None:
            return ret @ task_weights.double()
        return ret.mean()
    if num_labels is None:
        return tef.multiclass_auprc(input=input, target=target, num_classes=num_classes, average=average, **kwargs)
    raise ValueError("Could not infer the type of the task. Only one of `num_labels`, `num_classes` is allowed.")


def accuracy(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
    threshold: float = 0.5,
    average: str | None = "micro",
    num_labels: int | None = None,
    num_classes: int | None = None,
    **kwargs
):
    if isinstance(input, NestedTensor):
        input = torch.cat(input.storage())
    if isinstance(target, NestedTensor):
        target = torch.cat(target.storage())
    if num_labels is None and num_classes is None:
        return tef.binary_accuracy(input=input, target=target, threshold=threshold, **kwargs)
    if num_classes is None:
        return tef.multilabel_accuracy(input=input, target=target, threshold=threshold, **kwargs)
    if num_labels is None:
        return tef.multiclass_accuracy(input=input, target=target, num_classes=num_classes, average=average, **kwargs)
    raise ValueError("Could not infer the type of the task. Only one of `num_labels`, `num_classes` is allowed.")


def pearson(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
):
    if isinstance(input, NestedTensor):
        input = torch.cat(input.storage())
    if isinstance(target, NestedTensor):
        target = torch.cat(target.storage())
    return tmf.pearson_corrcoef(input, target)


def spearman(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
):
    if isinstance(input, NestedTensor):
        input = torch.cat(input.storage())
    if isinstance(target, NestedTensor):
        target = torch.cat(target.storage())
    return tmf.spearman_corrcoef(input, target)


def r2_score(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
):
    if isinstance(input, NestedTensor):
        input = torch.cat(input.storage())
    if isinstance(target, NestedTensor):
        target = torch.cat(target.storage())
    return tef.r2_score(input, target)


def mse(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
):
    if isinstance(input, NestedTensor):
        input = torch.cat(input.storage())
    if isinstance(target, NestedTensor):
        target = torch.cat(target.storage())
    return tef.mean_squared_error(input, target)


def rmse(
    input: Tensor | NestedTensor,
    target: Tensor | NestedTensor,
):
    if isinstance(input, NestedTensor):
        input = torch.cat(input.storage())
    if isinstance(target, NestedTensor):
        target = torch.cat(target.storage())
    return tef.mean_squared_error(input, target).sqrt()
