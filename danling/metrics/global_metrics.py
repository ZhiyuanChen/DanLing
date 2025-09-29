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

from collections.abc import Sequence
from typing import Callable

import torch
from torch import Tensor
from torch import distributed as dist

from danling.tensors import NestedTensor
from danling.utils import get_world_size

from .functional.utils import MetricFunc
from .preprocess import base_preprocess
from .state import MetricState
from .utils import RoundDict, iter_metric_funcs, merge_metric_entries

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self


class MetricRequirementError(RuntimeError):
    """Raised when a descriptor cannot be computed due to missing artifacts."""


class GlobalMetrics:
    """
    Data container for metrics descriptors.

    The container aggregates required artifacts (preds/targets, confusion
    matrix, running stats) only once, synchronises them across processes,
    and lets descriptors compute metric values without duplicating work.
    """

    def __init__(
        self,
        *metric_funcs,
        preprocess: Callable = base_preprocess,
        distributed: bool = True,
        device: torch.device | str | None = None,
        **meters,
    ) -> None:
        positional: list[tuple[str, MetricFunc]] = []
        for metric in iter_metric_funcs(metric_funcs):
            metric = self._coerce_metric(metric)
            positional.append((metric.name, metric))

        named: dict[str, MetricFunc] = {}
        for name, metric in meters.items():
            named[name] = self._coerce_metric(metric)

        metric_map = merge_metric_entries(positional, named)
        self.metrics = metric_map
        self.requirements = MetricState.collect_requirements(tuple(self.metrics.values()), require_nonempty=True)
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

    @staticmethod
    def _coerce_metric(value: MetricFunc) -> MetricFunc:
        if not isinstance(value, MetricFunc):
            raise ValueError(f"Expected metric functions to be MetricFunc instances, got {type(value)}")
        return value

    def update(self, input: Tensor | NestedTensor | Sequence, target: Tensor | NestedTensor | Sequence) -> None:
        input, target = self.preprocess(input, target)
        if isinstance(input, NestedTensor):
            input = input.concat
        if isinstance(target, NestedTensor):
            target = target.concat

        self._last_preds = input
        self._last_targets = target

        if self.requirements["preds_targets"]:
            self._preds.append(self._detach_to_device(input))
            self._targets.append(self._detach_to_device(target))
        if self.requirements["confmat"]:
            batch_confmat = MetricState.compute_confmat(input, target, self.requirements)
            if batch_confmat is None:
                raise MetricRequirementError("Confusion matrix requested but required tensors are not available.")
            self._last_confmat = batch_confmat
            self._confmat = batch_confmat if self._confmat is None else self._confmat + batch_confmat

        self._artifact_version += 1
        self._cache.clear()
        self._synced = False

    def sync(self) -> None:
        if self._synced:
            return

        changed = self._compact_local_preds_targets()

        if self.distributed:
            world_size = get_world_size()
            if world_size > 1:
                if self.requirements["preds_targets"] and self._preds:
                    self._preds = [self._gather_tensor(self._preds[0], world_size)]
                    self._targets = [self._gather_tensor(self._targets[0], world_size)]
                    changed = True

                if self.requirements["confmat"] and self._confmat is not None:
                    self._confmat = self._all_reduce(self._confmat)
                    changed = True

        if changed:
            self._artifact_version += 1
            self._cache.clear()
        self._synced = True

    def value(self) -> RoundDict:
        state = self._last_state()
        return RoundDict(
            {name: self._run_metric(name, func, state, cache=False) for name, func in self.metrics.items()}
        )

    def average(self) -> RoundDict:
        self.sync()
        state = self._average_state()
        return RoundDict({name: self._run_metric(name, func, state, cache=True) for name, func in self.metrics.items()})

    def compute(self) -> RoundDict:
        return self.average()

    @property
    def val(self) -> RoundDict:
        return self.value()

    @property
    def avg(self) -> RoundDict:
        return self.average()

    @property
    def preds(self) -> Tensor:
        if self._preds:
            if len(self._preds) == 1:
                return self._preds[0]
            return torch.cat(self._preds, dim=0)
        return torch.empty(0, device=self.device or "cpu")

    @property
    def targets(self) -> Tensor:
        if self._targets:
            if len(self._targets) == 1:
                return self._targets[0]
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
        keys = tuple(self.metrics.keys())
        return f"{self.__class__.__name__}{keys}"

    def __format__(self, format_spec: str) -> str:
        val, avg = self.value(), self.average()
        return "\t".join(
            f"{key}: {val[key].__format__(format_spec)} ({avg[key].__format__(format_spec)})" for key in val
        )

    def _last_state(self) -> MetricState:
        confmat = self._last_confmat if self._last_confmat is not None else self.confmat
        return MetricState(preds=self._last_preds, targets=self._last_targets, confmat=confmat)

    def _average_state(self) -> MetricState:
        return MetricState(preds=self.preds, targets=self.targets, confmat=self.confmat)

    def _run_metric(self, name: str, func: MetricFunc, state: MetricState, cache: bool) -> Tensor | float:
        if cache:
            cached = self._cache.get(name)
            if cached and cached[0] == self._artifact_version:
                return cached[1]

        value = func(state)

        if cache:
            self._cache[name] = (self._artifact_version, value)
        return value

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

    def _compact_local_preds_targets(self) -> bool:
        if not self.requirements["preds_targets"] or not self._preds:
            return False
        if len(self._preds) == 1 and len(self._targets) == 1:
            return False

        self._preds = [torch.cat(self._preds, dim=0)]
        self._targets = [torch.cat(self._targets, dim=0)]
        return True
