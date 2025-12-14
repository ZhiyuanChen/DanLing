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
from dataclasses import dataclass
from math import nan
from typing import Callable, Iterable, Optional

import torch
from chanfig import DefaultDict, FlatDict
from torch import Tensor
from torch import distributed as dist

from danling.tensors import NestedTensor
from danling.utils import flist, get_world_size

from .functional.utils import Artifact, MetricFunc
from .preprocess import base_preprocess, preprocess_binary
from .utils import RoundDict

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self


class MetricRequirementError(RuntimeError):
    """Raised when a descriptor cannot be computed due to missing artifacts."""


@dataclass
class ArtifactPlan:
    need_preds_targets: bool = False
    need_confmat: bool = False
    task: Optional[str] = None
    num_classes: Optional[int] = None
    num_labels: Optional[int] = None
    threshold: Optional[float] = None

    @classmethod
    def from_functions(cls, funcs: Sequence[MetricFunc]) -> ArtifactPlan:
        if not funcs:
            raise ValueError("Metrics requires at least one metric function.")
        merged = Artifact()
        for func in funcs:
            merged = merged.merge(func.artifact)
        plan = cls(
            need_preds_targets=merged.preds_targets,
            need_confmat=merged.confmat,
            task=merged.task,
            num_classes=merged.num_classes,
            num_labels=merged.num_labels,
            threshold=merged.threshold,
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        if self.need_confmat and self.task is None:
            raise ValueError("Confusion matrix computation requires a task to be specified.")


class _MetricsView:
    """
    Lightweight view that exposes the expected interface to descriptors.

    This allows us to switch between last-batch artifacts (value) and
    aggregated artifacts (average) without mutating container state.
    """

    def __init__(self, container: Metrics, *, use_last: bool) -> None:
        self.container = container
        self.use_last = use_last

    @property
    def preds(self) -> Tensor:
        if self.use_last or not self.container.plan.need_preds_targets:
            return self.container._last_preds  # noqa: SLF001
        return self.container.preds

    @property
    def targets(self) -> Tensor:
        if self.use_last or not self.container.plan.need_preds_targets:
            return self.container._last_targets  # noqa: SLF001
        return self.container.targets

    @property
    def confmat(self) -> Tensor | None:
        if self.use_last and self.container._last_confmat is not None:  # noqa: SLF001
            return self.container._last_confmat  # noqa: SLF001
        return self.container.confmat

    @property
    def plan(self) -> ArtifactPlan:
        return self.container.plan


class Metrics:
    """
    Data container for metrics descriptors.

    The container aggregates required artifacts (preds/targets, confusion
    matrix, running stats) only once, synchronises them across processes,
    and lets descriptors compute metric values without duplicating work.
    """

    def __init__(
        self,
        metric_funcs: Sequence[MetricFunc],
        *,
        preprocess: Callable = preprocess_binary,
        distributed: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        for func in metric_funcs:
            if not isinstance(func, MetricFunc):
                raise ValueError(f"Expected metric functions to be MetricFunc instances, got {type(func)}")
        self.metrics = list(metric_funcs)
        self.plan = ArtifactPlan.from_functions(self.metrics)
        self.preprocess = preprocess
        self.distributed = distributed
        self.device = torch.device(device) if device is not None else None

        self._preds: list[Tensor] = []
        self._targets: list[Tensor] = []
        self._confmat: Tensor | None = None
        self._last_confmat: Tensor | None = None
        self._last_preds: Tensor = torch.empty(0)
        self._last_targets: Tensor = torch.empty(0)

        self._artifact_version = 0
        self._cache: dict[str, tuple[int, Tensor | float]] = {}
        self._synced = False

    def update(self, input: Tensor | NestedTensor | Sequence, target: Tensor | NestedTensor | Sequence) -> None:
        input, target = self.preprocess(input, target)
        if isinstance(input, NestedTensor):
            input = input.concat
        if isinstance(target, NestedTensor):
            target = target.concat

        self._last_preds = input
        self._last_targets = target

        if self.plan.need_preds_targets:
            self._preds.append(self._detach_to_device(input))
            self._targets.append(self._detach_to_device(target))
        if self.plan.need_confmat:
            batch_confmat = self._compute_confmat(input, target)
            self._last_confmat = batch_confmat
            self._confmat = batch_confmat if self._confmat is None else self._confmat + batch_confmat

        self._artifact_version += 1
        self._cache.clear()
        self._synced = False

    def sync(self) -> None:
        if self._synced or not self.distributed:
            return

        world_size = get_world_size()
        if world_size <= 1:
            self._synced = True
            return

        if self.plan.need_preds_targets and self._preds:
            preds = self._gather_tensor(torch.cat(self._preds, dim=0), world_size)
            targets = self._gather_tensor(torch.cat(self._targets, dim=0), world_size)
            self._preds = [preds]
            self._targets = [targets]

        if self.plan.need_confmat and self._confmat is not None:
            self._confmat = self._all_reduce(self._confmat)

        self._artifact_version += 1
        self._cache.clear()
        self._synced = True

    def value(self) -> RoundDict[str, Tensor | float]:
        view = _MetricsView(self, use_last=True)
        return RoundDict({func.name: self._run_metric(func, view, cache=False) for func in self.metrics})

    def average(self) -> RoundDict[str, Tensor | float]:
        self.sync()
        view = _MetricsView(self, use_last=False)
        return RoundDict({func.name: self._run_metric(func, view, cache=True) for func in self.metrics})

    def compute(self) -> RoundDict[str, Tensor | float]:
        return self.average()

    @property
    def val(self) -> RoundDict[str, Tensor | float]:
        return self.value()

    @property
    def avg(self) -> RoundDict[str, Tensor | float]:
        return self.average()

    @property
    def preds(self) -> Tensor:
        if self._preds:
            return torch.cat(self._preds, dim=0)
        return torch.empty(0, device=self.device or "cpu")

    @property
    def targets(self) -> Tensor:
        if self._targets:
            return torch.cat(self._targets, dim=0)
        return torch.empty(0, device=self.device or "cpu")

    @property
    def confmat(self) -> Tensor | None:
        return self._confmat

    def reset(self) -> Self:
        self._preds.clear()
        self._targets.clear()
        self._confmat = None
        self._last_confmat = None
        self._last_preds = torch.empty(0)
        self._last_targets = torch.empty(0)
        self._artifact_version = 0
        self._cache.clear()
        self._synced = False
        return self

    def __repr__(self) -> str:  # pragma: no cover - repr convenience
        keys = tuple(func.name for func in self.metrics)
        return f"{self.__class__.__name__}{keys}"

    def __format__(self, format_spec: str) -> str:
        val, avg = self.value(), self.average()
        return "\t".join(
            f"{key}: {val[key].__format__(format_spec)} ({avg[key].__format__(format_spec)})" for key in val
        )

    def _run_metric(self, func: MetricFunc, view: _MetricsView, cache: bool) -> Tensor | float:
        if cache:
            cached = self._cache.get(func.name)
            if cached and cached[0] == self._artifact_version:
                return cached[1]

        value = func(view)

        if cache:
            self._cache[func.name] = (self._artifact_version, value)
        return value

    def _compute_confmat(self, input: Tensor, target: Tensor) -> Tensor:
        if self.plan.task is None:
            raise MetricRequirementError("Confusion matrix requested but no task specified.")

        from torchmetrics.functional.classification import confusion_matrix as tm_confusion_matrix

        kwargs = {"task": self.plan.task}
        if self.plan.num_classes is not None:
            kwargs["num_classes"] = self.plan.num_classes
        if self.plan.num_labels is not None:
            kwargs["num_labels"] = self.plan.num_labels
        if self.plan.threshold is not None:
            kwargs["threshold"] = self.plan.threshold

        return tm_confusion_matrix(input, target, **kwargs)

    def _detach_to_device(self, tensor: Tensor) -> Tensor:
        output = tensor.detach()
        if self.device is not None:
            output = output.to(self.device)
        return output

    def _gather_tensor(self, tensor: Tensor, world_size: int) -> Tensor:
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

    def _all_reduce(self, tensor: Tensor) -> Tensor:
        dist.all_reduce(tensor)
        return tensor
