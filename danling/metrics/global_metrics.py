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
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor
from torch import distributed as dist
from typing_extensions import Self

from danling.tensors import NestedTensor
from danling.utils import get_world_size

from .functional.utils import MetricFunc
from .preprocess import base_preprocess
from .state import MetricState
from .utils import RoundDict, infer_device, iter_metric_funcs, merge_metric_entries


class MetricRequirementError(RuntimeError):
    """Raised when a descriptor cannot be computed due to missing artifacts."""


@dataclass
class _ArtifactState:
    preds: list[Tensor] = field(default_factory=list)
    targets: list[Tensor] = field(default_factory=list)
    pending_preds: list[Tensor] = field(default_factory=list)
    pending_targets: list[Tensor] = field(default_factory=list)
    confmat: Tensor | None = None
    last_confmat: Tensor | None = None
    last_preds: Tensor = field(default_factory=lambda: torch.empty(0))
    last_targets: Tensor = field(default_factory=lambda: torch.empty(0))
    preds_template: Tensor | None = None
    targets_template: Tensor | None = None


@dataclass
class _SyncState:
    pred_chunks: list[Tensor] | None = None
    target_chunks: list[Tensor] | None = None
    preds: Tensor | None = None
    targets: Tensor | None = None
    confmat: Tensor | None = None
    count: int | None = None
    synced: bool = False
    world_size: int | None = None


class GlobalMetrics:
    """
    Data container for metrics descriptors.

    The container aggregates required artifacts (preds/targets, confusion
    matrix, running stats) only once, synchronises them across processes,
    and lets descriptors compute metric values without duplicating work.
    """

    _local_n: int = 0
    _local_count: int = 0

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

        self._artifacts = _ArtifactState()
        self._sync = _SyncState()
        self._local_n = 0
        self._local_count = 0

        self._artifact_version = 0
        self._cache: dict[str, tuple[int, Tensor | float]] = {}

    # Construction
    @staticmethod
    def _coerce_metric(value: MetricFunc) -> MetricFunc:
        if not isinstance(value, MetricFunc):
            raise ValueError(f"Expected metric functions to be MetricFunc instances, got {type(value)}")
        return value

    # Lifecycle
    def update(self, input: Tensor | NestedTensor | Sequence, target: Tensor | NestedTensor | Sequence) -> None:
        artifacts = self._artifacts
        input, target = self.preprocess(input, target)
        if isinstance(input, NestedTensor):
            input = input.concat
        if isinstance(target, NestedTensor):
            target = target.concat

        artifacts.last_preds = input.detach()
        artifacts.last_targets = target.detach()
        self._local_n = self._infer_batch_count(target)
        self._local_count += self._local_n

        if self.requirements["preds_targets"]:
            stored_input = self._detach_to_device(input)
            stored_target = self._detach_to_device(target)
            artifacts.preds.append(stored_input)
            artifacts.targets.append(stored_target)
            artifacts.pending_preds.append(stored_input)
            artifacts.pending_targets.append(stored_target)
            artifacts.preds_template = self._empty_artifact(stored_input)
            artifacts.targets_template = self._empty_artifact(stored_target)
        if self.requirements["confmat"]:
            batch_confmat = MetricState.compute_confmat(input, target, self.requirements)
            if batch_confmat is None:
                raise MetricRequirementError("Confusion matrix requested but required tensors are not available.")
            artifacts.last_confmat = batch_confmat
            artifacts.confmat = batch_confmat if artifacts.confmat is None else artifacts.confmat + batch_confmat

        self._artifact_version += 1
        self._cache.clear()
        self._mark_sync_stale()

    def sync(self) -> None:
        artifacts = self._artifacts
        sync = self._sync
        world_size = get_world_size() if self.distributed else 1
        if sync.synced and sync.world_size == world_size:
            return

        synced_pred_chunks = sync.pred_chunks
        synced_target_chunks = sync.target_chunks
        synced_preds = sync.preds if sync.preds is not None else self._local_preds()
        synced_targets = sync.targets if sync.targets is not None else self._local_targets()
        synced_confmat = artifacts.confmat

        if self.distributed:
            if world_size > 1:
                if self.requirements["preds_targets"]:
                    if self._requires_full_artifact_resync(world_size):
                        local_pred_tensor = self._local_artifact(artifacts.preds, artifacts.preds_template)
                        local_target_tensor = self._local_artifact(artifacts.targets, artifacts.targets_template)
                        synced_pred_chunks = self._gather_tensor_chunks(local_pred_tensor, world_size)
                        synced_target_chunks = self._gather_tensor_chunks(local_target_tensor, world_size)
                    else:
                        delta_pred_tensor = self._local_artifact(artifacts.pending_preds, artifacts.preds_template)
                        delta_target_tensor = self._local_artifact(
                            artifacts.pending_targets, artifacts.targets_template
                        )
                        delta_pred_chunks = self._gather_tensor_chunks(delta_pred_tensor, world_size)
                        delta_target_chunks = self._gather_tensor_chunks(delta_target_tensor, world_size)
                        synced_pred_chunks = self._append_tensor_chunks(sync.pred_chunks, delta_pred_chunks)
                        synced_target_chunks = self._append_tensor_chunks(sync.target_chunks, delta_target_chunks)
                    synced_preds = self._concat_tensor_chunks(synced_pred_chunks)
                    synced_targets = self._concat_tensor_chunks(synced_target_chunks)

                if self.requirements["confmat"]:
                    local_confmat = artifacts.confmat if artifacts.confmat is not None else self._empty_confmat()
                    synced_confmat, synced_count = self._all_reduce_confmat_count(local_confmat, self._local_count)
                    if synced_count == 0:
                        synced_confmat = None
                else:
                    synced_count = None
            else:
                if self.requirements["preds_targets"]:
                    synced_pred_chunks = [self._local_preds()]
                    synced_target_chunks = [self._local_targets()]
                    synced_preds = synced_pred_chunks[0]
                    synced_targets = synced_target_chunks[0]
                synced_count = self._local_count
        else:
            if self.requirements["preds_targets"]:
                synced_pred_chunks = [self._local_preds()]
                synced_target_chunks = [self._local_targets()]
                synced_preds = synced_pred_chunks[0]
                synced_targets = synced_target_chunks[0]
            synced_count = self._local_count
        sync.pred_chunks = synced_pred_chunks if self.requirements["preds_targets"] else None
        sync.target_chunks = synced_target_chunks if self.requirements["preds_targets"] else None
        sync.preds = synced_preds if self.requirements["preds_targets"] else None
        sync.targets = synced_targets if self.requirements["preds_targets"] else None
        sync.confmat = synced_confmat if self.requirements["confmat"] else None
        if world_size == 1:
            sync.count = self._local_count
        elif sync.targets is not None:
            sync.count = self._infer_batch_count(sync.targets)
        elif synced_count is not None:
            sync.count = synced_count
        else:
            sync.count = None
        artifacts.pending_preds.clear()
        artifacts.pending_targets.clear()
        self._cache.clear()
        sync.synced = True
        sync.world_size = world_size

    def reset(self) -> Self:
        self._artifacts = _ArtifactState()
        self._clear_sync_state()
        self._local_n = 0
        self._local_count = 0
        self._artifact_version = 0
        self._cache.clear()
        return self

    # Public reductions
    def value(self) -> RoundDict:
        state = self._last_state()
        return RoundDict(
            {name: self._run_metric(name, func, state, cache=False) for name, func in self.metrics.items()}
        )

    def batch(self) -> RoundDict:
        world_size = self._current_world_size()
        if world_size == 1:
            return self.value()

        if not self.requirements["preds_targets"] and not self.requirements["confmat"]:
            return self._approximate_batch_values()

        state, _ = self._batch_state(world_size)
        return RoundDict(
            {name: self._run_metric(name, func, state, cache=False) for name, func in self.metrics.items()}
        )

    def average(self) -> RoundDict:
        self.sync()
        state = self._average_state()
        return RoundDict({name: self._run_metric(name, func, state, cache=True) for name, func in self.metrics.items()})

    # Public aliases
    @property
    def val(self) -> RoundDict:
        return self.value()

    @property
    def bat(self) -> RoundDict:
        return self.batch()

    @property
    def avg(self) -> RoundDict:
        return self.average()

    # Public artifact accessors
    @property
    def preds(self) -> Tensor:
        if self._should_expose_synced_state() and self._sync.preds is not None:
            return self._sync.preds
        return self._local_preds()

    @property
    def targets(self) -> Tensor:
        if self._should_expose_synced_state() and self._sync.targets is not None:
            return self._sync.targets
        return self._local_targets()

    @property
    def confmat(self) -> Tensor | None:
        if self._should_expose_synced_state() and self._sync.confmat is not None:
            return self._sync.confmat
        return self._artifacts.confmat

    # Public state accessors
    @property
    def n(self) -> int:
        return self._local_n

    @property
    def count(self) -> int:
        return self._local_count

    # Formatting helpers
    def __repr__(self) -> str:  # pragma: no cover - repr convenience
        keys = tuple(self.metrics.keys())
        return f"{self.__class__.__name__}{keys}"

    def __format__(self, format_spec: str) -> str:
        val = self.value()
        state = self._local_average_state()
        avg = RoundDict({name: self._run_metric(name, func, state, cache=False) for name, func in self.metrics.items()})
        return "\t".join(
            f"{key}: {val[key].__format__(format_spec)} ({avg[key].__format__(format_spec)})" for key in val
        )

    # State builders
    def _last_state(self) -> MetricState:
        artifacts = self._artifacts
        return MetricState(preds=artifacts.last_preds, targets=artifacts.last_targets, confmat=artifacts.last_confmat)

    def _local_average_state(self) -> MetricState:
        return MetricState(preds=self._local_preds(), targets=self._local_targets(), confmat=self._artifacts.confmat)

    def _average_state(self) -> MetricState:
        preds = self._sync.preds if self._sync.preds is not None else self._local_preds()
        targets = self._sync.targets if self._sync.targets is not None else self._local_targets()
        confmat = self._sync.confmat if self._sync.confmat is not None else self._artifacts.confmat
        return MetricState(preds=preds, targets=targets, confmat=confmat)

    def _batch_state(self, world_size: int) -> tuple[MetricState, int]:
        artifacts = self._artifacts
        synced_preds = artifacts.last_preds
        synced_targets = artifacts.last_targets
        synced_confmat = artifacts.last_confmat
        synced_count: int | None = None

        if self.requirements["preds_targets"]:
            local_pred_tensor = self._current_batch_artifact(artifacts.last_preds, artifacts.preds_template)
            local_target_tensor = self._current_batch_artifact(artifacts.last_targets, artifacts.targets_template)
            synced_preds = self._gather_tensor(local_pred_tensor, world_size)
            synced_targets = self._gather_tensor(local_target_tensor, world_size)
            synced_count = self._infer_batch_count(synced_targets)

        if self.requirements["confmat"]:
            local_confmat = artifacts.last_confmat if artifacts.last_confmat is not None else self._empty_confmat()
            synced_confmat, confmat_count = self._all_reduce_confmat_count(local_confmat, self._local_n)
            if confmat_count == 0:
                synced_confmat = None
            if synced_count is None:
                synced_count = confmat_count

        if synced_count is None:
            synced_count = self._all_reduce_count(self._local_n)

        return MetricState(preds=synced_preds, targets=synced_targets, confmat=synced_confmat), synced_count

    def _run_metric(self, name: str, func: MetricFunc, state: MetricState, cache: bool) -> Tensor | float:
        if cache:
            cached = self._cache.get(name)
            if cached and cached[0] == self._artifact_version:
                return cached[1]

        value = func(state)

        if cache:
            self._cache[name] = (self._artifact_version, value)
        return value

    # Local artifact helpers
    def _local_preds(self) -> Tensor:
        preds = self._artifacts.preds
        if preds:
            if len(preds) == 1:
                return preds[0]
            return torch.cat(preds, dim=0)
        return torch.empty(0, device=self.device or "cpu")

    def _local_targets(self) -> Tensor:
        targets = self._artifacts.targets
        if targets:
            if len(targets) == 1:
                return targets[0]
            return torch.cat(targets, dim=0)
        return torch.empty(0, device=self.device or "cpu")

    def _detach_to_device(self, tensor: Tensor) -> Tensor:
        output = tensor.detach()
        if self.device is not None:
            output = output.to(self.device)
        return output

    def _local_artifact(self, tensors: list[Tensor], template: Tensor | None) -> Tensor | None:
        if not tensors:
            return template
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=0)

    def _approximate_batch_values(self) -> RoundDict:
        local_values = self.value()
        device = self._sync_device()
        local_count = float(self._local_n)
        names: list[str] = []
        shapes: list[torch.Size] = []
        tensor_flags: list[bool] = []
        total_numel = 1

        for name, value in local_values.items():
            tensor_value = torch.as_tensor(value, dtype=torch.float64, device=device)
            names.append(name)
            shapes.append(tensor_value.shape)
            tensor_flags.append(isinstance(value, Tensor))
            total_numel += tensor_value.numel()

        reduced = torch.zeros(total_numel, dtype=torch.float64, device=device)
        reduced[0] = local_count

        offset = 1
        for name in names:
            tensor_value = torch.as_tensor(local_values[name], dtype=torch.float64, device=device).reshape(-1)
            numel = tensor_value.numel()
            if local_count > 0:
                reduced[offset : offset + numel] = tensor_value * local_count
            offset += numel

        dist.all_reduce(reduced)

        total_count = int(round(reduced[0].item()))

        batch_values = RoundDict()
        offset = 1
        for name, shape, is_tensor in zip(names, shapes, tensor_flags):
            numel = int(torch.Size(shape).numel())
            values = reduced[offset : offset + numel]
            offset += numel
            if total_count == 0:
                reduced_value = torch.full(shape, float("nan"), dtype=torch.float64, device=device)
            else:
                reduced_value = (values / total_count).reshape(shape)
            batch_values[name] = reduced_value if is_tensor else reduced_value.item()

        return batch_values

    # Distributed synchronization helpers
    def _gather_tensor(self, tensor: Tensor | None, world_size: int) -> Tensor:
        gathered_chunks = self._gather_tensor_chunks(tensor, world_size)
        return self._concat_tensor_chunks(gathered_chunks)

    def _gather_tensor_chunks(self, tensor: Tensor | None, world_size: int) -> list[Tensor]:
        device = self._sync_device()
        if tensor is not None:
            tensor = self._tensor_on_sync_device(tensor, device=device)
        local_size = torch.tensor([tensor.shape[0] if tensor is not None else -1], dtype=torch.int64, device=device)
        size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        sizes = torch.cat(size_list)

        if (sizes < 0).any():
            tensor = self._gather_tensor_with_metadata(tensor, world_size)
            sizes = sizes.clamp_min(0)

        if tensor is None:
            return [torch.empty(0, device=device) for _ in range(world_size)]

        return self._gather_tensor_chunks_data(tensor, sizes, world_size)

    def _gather_tensor_chunks_data(self, tensor: Tensor, sizes: Tensor, world_size: int) -> list[Tensor]:
        max_size = int(sizes.max().item())

        padded_tensor = torch.empty((max_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        padded_tensor.zero_()
        if tensor.shape[0] > 0:
            padded_tensor[: tensor.shape[0]] = tensor
        gathered_tensors = [torch.empty_like(padded_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, padded_tensor)
        empty = torch.empty((0, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        return [gathered_tensors[i][: sizes[i]] if sizes[i] > 0 else empty.clone() for i in range(world_size)]

    def _gather_tensor_with_metadata(self, tensor: Tensor | None, world_size: int) -> Tensor | None:
        metadata = None if tensor is None else (tuple(tensor.shape[1:]), str(tensor.dtype))
        metadata_list: list[tuple[tuple[int, ...], str] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(metadata_list, metadata)
        reference = next((item for item in metadata_list if item is not None), None)
        if reference is None:
            return tensor
        if tensor is not None:
            return tensor

        reference_shape, reference_dtype = reference
        return torch.empty(
            (0, *reference_shape),
            dtype=getattr(torch, reference_dtype.removeprefix("torch.")),
            device=self._sync_device(),
        )

    def _all_reduce_confmat_count(self, tensor: Tensor, count: int) -> tuple[Tensor, int]:
        tensor = self._tensor_on_sync_device(tensor)
        reduced = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
        reduced[:-1] = tensor.reshape(-1)
        reduced[-1] = count
        dist.all_reduce(reduced)
        return reduced[:-1].reshape_as(tensor), int(round(float(reduced[-1].item())))

    def _all_reduce_count(self, count: int) -> int:
        device = self._sync_device()
        reduced = torch.tensor([float(count)], dtype=torch.float64, device=device)
        dist.all_reduce(reduced)
        return int(round(reduced.item()))

    def _empty_artifact(self, tensor: Tensor) -> Tensor:
        return torch.empty((0, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)

    def _current_batch_artifact(self, tensor: Tensor, template: Tensor | None) -> Tensor | None:
        if tensor.numel() > 0 or tensor.ndim > 1:
            return tensor
        if template is not None:
            return template
        return None

    def _concat_tensor_chunks(self, chunks: list[Tensor] | None) -> Tensor:
        if not chunks:
            return torch.empty(0, device=self._sync_device())
        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks, dim=0)

    def _append_tensor_chunks(self, base: list[Tensor] | None, delta: list[Tensor]) -> list[Tensor]:
        if base is None:
            return delta
        return [torch.cat((current, update), dim=0) for current, update in zip(base, delta)]

    def _tensor_on_sync_device(self, tensor: Tensor, *, device: torch.device | None = None) -> Tensor:
        sync_device = self._sync_device() if device is None else device
        if tensor.device == sync_device:
            return tensor
        return tensor.to(sync_device)

    def _mark_sync_stale(self) -> None:
        self._sync.count = None
        self._sync.synced = False

    def _clear_sync_state(self) -> None:
        self._sync = _SyncState()

    # Generic helpers
    def _empty_confmat(self) -> Tensor:
        task = self.requirements["task"]
        device = self._sync_device()

        if task == "binary":
            return torch.zeros((2, 2), dtype=torch.long, device=device)

        if task == "multiclass":
            num_classes = self.requirements["num_classes"]
            return torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)

        if task == "multilabel":
            num_labels = self.requirements["num_labels"]
            return torch.zeros((num_labels, 2, 2), dtype=torch.long, device=device)

        raise MetricRequirementError(f"Unsupported confusion matrix task: {task!r}")

    def _sync_device(self) -> torch.device:
        if not (dist.is_available() and dist.is_initialized()) and self.device is not None:
            return self.device
        return infer_device()

    def _current_world_size(self) -> int:
        if not self.distributed:
            return 1
        return get_world_size()

    def _should_expose_synced_state(self) -> bool:
        return self._sync.synced and self._sync.world_size == self._current_world_size()

    def _requires_full_artifact_resync(self, world_size: int) -> bool:
        return (
            self._sync.world_size != world_size
            or self._sync.pred_chunks is None
            or self._sync.target_chunks is None
            or not self._chunks_match_template(self._sync.pred_chunks, self._artifacts.preds_template)
            or not self._chunks_match_template(self._sync.target_chunks, self._artifacts.targets_template)
        )

    @staticmethod
    def _infer_batch_count(target: Tensor) -> int:
        if target.ndim == 0:
            return 1
        return int(target.shape[0])

    @staticmethod
    def _chunks_match_template(chunks: list[Tensor], template: Tensor | None) -> bool:
        if template is None:
            return True
        return all(chunk.dtype == template.dtype and chunk.shape[1:] == template.shape[1:] for chunk in chunks)
