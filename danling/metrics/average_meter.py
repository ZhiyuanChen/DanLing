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

from __future__ import annotations

from collections.abc import Mapping
from math import nan

import torch
from torch import Tensor
from torch import device as torch_device
from torch import distributed as dist
from typing_extensions import Self

from danling.utils import get_world_size

from .utils import MetersBase, RoundDict, infer_device


class AverageMeter:
    r"""
    A lightweight utility to compute and store running averages of values.

    AverageMeter provides an efficient way to track running statistics (current value, sum, count, average)
    with minimal memory overhead and optional distributed averaging.
    Scalar values stay scalar. Tensor values are preserved end to end as long as
    each update for the meter has the same shape.

    Attributes:
        val: Most recent local value added to the meter
        bat: Synchronized metric value for the current step
        avg: Running average of all values, weighted by counts
        sum: Sum of all values added to the meter
        count: Total count of values added (considering weights)
        device: Device used when synchronising running averages across processes

    Args:
        device: Optional device used for distributed reductions. When not provided,
            the device is detected automatically when synchronisation happens.

    Examples:
        >>> meter = AverageMeter()
        >>> meter.update(0.7)
        >>> meter.val
        0.7
        >>> meter.bat  # Same as val in non-distributed mode
        0.7
        >>> meter.avg
        0.7
        >>> meter.update(0.9)
        >>> meter.val
        0.9
        >>> meter.avg
        0.8
        >>> meter.sum
        1.6
        >>> meter.count
        2
        >>> # Weighted update
        >>> meter.update(value=0.5, n=3)
        >>> meter.avg
        0.62
        >>> meter.reset()
        AverageMeter(val=nan, avg=nan)

    See Also:
        - [`MetricMeter`][danling.metrics.stream_metrics.MetricMeter]:
            Memory-efficient metric tracker that averages metrics batch-by-batch.
    """

    _local_value: float | Tensor = 0.0
    _local_n: int = 0
    _local_sum: float | Tensor = 0.0
    _local_count: int = 0

    def __init__(self, *, device: torch.device | str | None = None, distributed: bool = True) -> None:
        self.distributed = distributed
        self.device = torch_device(device) if device is not None else None
        self.reset()

    # Lifecycle
    def reset(self, *, device: torch.device | str | None = None) -> Self:
        r"""
        Resets the meter.
        """

        if device is not None:
            self.device = torch_device(device)
        self._local_value = 0.0
        self._local_n = 0
        self._local_sum = 0.0
        self._local_count = 0
        self._tensor_template = None
        return self

    # Mutation
    def update(self, value: float | int | Tensor, n: int = 1) -> None:
        r"""
        Updates the average and current value in the meter.

        Args:
            value: Value to be added to the average.
            n: Number of values to be added.
        """

        if isinstance(value, Tensor):
            if value.numel() == 1:
                if self.device is None:
                    self.device = value.device
                value = float(value.detach().item())
            else:
                if self._local_count > 0 and not isinstance(self._local_sum, Tensor):
                    raise ValueError("AverageMeter cannot mix scalar and tensor values.")

                value = value.detach().to(dtype=torch.float64)
                if self.device is None:
                    self.device = value.device

                if isinstance(self._local_sum, Tensor):
                    if value.shape != self._local_sum.shape:
                        raise ValueError(
                            "AverageMeter requires consistent tensor shapes, "
                            f"but got {tuple(value.shape)} after {tuple(self._local_sum.shape)}."
                        )
                    if value.device != self._local_sum.device:
                        self._local_sum = self._local_sum.to(value.device)
                        if isinstance(self._local_value, Tensor):
                            self._local_value = self._local_value.to(value.device)
                        if self._tensor_template is not None:
                            self._tensor_template = self._tensor_template.to(value.device)
                else:
                    self._local_sum = torch.zeros_like(value, dtype=torch.float64, device=value.device)

                self._tensor_template = torch.empty_like(value, dtype=torch.float64, device=value.device)
                self._local_value = value
                if n > 0:
                    self._local_sum.add_(value * n)
        if not isinstance(value, Tensor):
            if self._tensor_template is not None:
                if n == 0:
                    tensor_value = torch.full_like(self._tensor_template, float(value), dtype=torch.float64)
                    self._local_value = tensor_value
                    self._local_n = 0
                    return
                raise ValueError("AverageMeter cannot mix tensor and scalar values.")
            value = float(value)
            if isinstance(self._local_sum, Tensor):
                tensor_value = torch.tensor(value, dtype=self._local_sum.dtype, device=self._local_sum.device)
                self._local_value = tensor_value
                if n > 0:
                    self._local_sum.add_(tensor_value * n)
            else:
                self._local_value = value
                if n > 0:
                    self._local_sum += value * n
        self._local_n = n
        if n > 0:
            self._local_count += n

    # Public reductions
    def value(self) -> float | Tensor:
        if self._local_count == 0:
            empty_tensor = self._empty_tensor_value()
            if empty_tensor is not None:
                return empty_tensor
            return nan
        return self._local_value

    def batch(self) -> float | Tensor:
        world_size = self._current_world_size()
        if world_size == 1:
            return self.value()

        if self._tensor_template is not None:
            return self._tensor_batch()

        device = self._sync_device()
        synced_tensor = torch.tensor([0.0, float(self._local_n)], dtype=torch.float64, device=device)
        if self._local_n > 0:
            if isinstance(self._local_value, Tensor):
                synced_tensor[0] = self._local_value.to(device=device, dtype=torch.float64) * self._local_n
            else:
                synced_tensor[0] = float(self._local_value) * self._local_n
        dist.all_reduce(synced_tensor)
        total, count = synced_tensor.tolist()
        if count == 0:
            return nan
        return total / count

    def average(self) -> float | Tensor:
        world_size = self._current_world_size()
        if world_size == 1:
            return self._local_average()
        if self._tensor_template is not None:
            return self._tensor_average()
        device = self._sync_device()
        synced_tensor = torch.tensor([0.0, float(self._local_count)], dtype=torch.float64, device=device)
        if isinstance(self._local_sum, Tensor):
            synced_tensor[0] = self._local_sum.to(device=device, dtype=torch.float64)
        else:
            synced_tensor[0] = float(self._local_sum)
        dist.all_reduce(synced_tensor)
        val, count = synced_tensor.tolist()
        if count == 0:
            return nan
        return val / count

    # Public aliases
    @property
    def val(self) -> float | Tensor:
        return self.value()

    @property
    def bat(self) -> float | Tensor:
        return self.batch()

    @property
    def avg(self) -> float | Tensor:
        return self.average()

    # Public state accessors
    @property
    def n(self) -> int:
        return self._local_n

    @property
    def sum(self) -> float | Tensor:
        return self._local_sum

    @property
    def count(self) -> int:
        return self._local_count

    # Formatting helpers
    def __format__(self, format_spec: str) -> str:
        value = self.value()
        average = self._local_average()
        return f"{self._format_value(value, format_spec)} ({self._format_value(average, format_spec)})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(val={self.value()}, avg={self._local_average()})"

    # Internal helpers
    def _local_average(self) -> float | Tensor:
        if self._local_count == 0:
            empty_tensor = self._empty_tensor_value()
            if empty_tensor is not None:
                return empty_tensor
            return nan
        if isinstance(self._local_sum, Tensor) and self._tensor_template is not None:
            return self._local_sum / self._local_count
        if isinstance(self._local_sum, Tensor):
            return (self._local_sum / self._local_count).item()
        return self._local_sum / self._local_count

    def _current_world_size(self) -> int:
        if not self.distributed:
            return 1
        return get_world_size()

    def _sync_device(self) -> torch.device:
        if not (dist.is_available() and dist.is_initialized()) and self.device is not None:
            return self.device
        return infer_device()

    def _distributed_tensor_reduction(
        self,
        tensor: Tensor | None,
        count: int,
        *,
        scale_by_count: bool,
        template: Tensor | None = None,
    ) -> Tensor:
        world_size = self._current_world_size()
        device = self._sync_device()
        if template is None:
            template = self._resolve_tensor_template(world_size, device)
        else:
            template = template.to(device=device, dtype=torch.float64)
        if template is None:
            return torch.tensor(float("nan"), dtype=torch.float64, device=device)

        reduced = torch.zeros(template.numel() + 1, dtype=torch.float64, device=device)
        if count > 0 and tensor is not None:
            tensor = tensor.to(device=device, dtype=torch.float64)
            if tensor.shape != template.shape:
                raise ValueError(
                    "AverageMeter requires consistent tensor shapes across ranks, "
                    f"but got local shape {tuple(tensor.shape)} and expected {tuple(template.shape)}."
                )
            reduced[:-1] = ((tensor * count) if scale_by_count else tensor).reshape(-1)
        reduced[-1] = float(count)
        dist.all_reduce(reduced)

        total_count = int(round(reduced[-1].item()))
        if total_count == 0:
            return torch.full(template.shape, float("nan"), dtype=torch.float64, device=device)
        return (reduced[:-1] / total_count).reshape(template.shape)

    def _tensor_batch(self, template: Tensor | None = None) -> Tensor:
        return self._distributed_tensor_reduction(
            self._local_value if isinstance(self._local_value, Tensor) else None,
            self._local_n,
            scale_by_count=True,
            template=template,
        )

    def _tensor_average(self, template: Tensor | None = None) -> Tensor:
        local_sum = self._local_sum if isinstance(self._local_sum, Tensor) else None
        return self._distributed_tensor_reduction(
            local_sum,
            self._local_count,
            scale_by_count=False,
            template=template,
        )

    def _empty_tensor_value(self) -> Tensor | None:
        if self._tensor_template is None:
            return None
        return torch.full_like(self._tensor_template, float("nan"), dtype=torch.float64)

    def _resolve_tensor_template(self, world_size: int, device: torch.device) -> Tensor | None:
        if world_size == 1:
            if self._tensor_template is None:
                return None
            return self._tensor_template.to(device=device, dtype=torch.float64)

        if not (dist.is_available() and dist.is_initialized()):
            if self._tensor_template is None:
                return None
            return self._tensor_template.to(device=device, dtype=torch.float64)

        metadata = None
        if self._tensor_template is not None:
            metadata = (tuple(self._tensor_template.shape), str(self._tensor_template.dtype))
        metadata_list: list[tuple[tuple[int, ...], str] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(metadata_list, metadata)
        references = [item for item in metadata_list if item is not None]
        if not references:
            return None
        if any(item != references[0] for item in references[1:]):
            raise ValueError(f"AverageMeter received inconsistent tensor metadata across ranks: {references!r}")

        shape, dtype_name = references[0]
        template = torch.empty(shape, dtype=getattr(torch, dtype_name.removeprefix("torch.")), device=device)
        template_device = self.device if self.device is not None else device
        self._tensor_template = template.to(device=template_device)
        return template

    @staticmethod
    def _format_value(value: float | Tensor, format_spec: str) -> str:
        if isinstance(value, Tensor):
            if value.numel() == 1:
                return value.item().__format__(format_spec)
            return str(value)
        return value.__format__(format_spec)

    def _tensor_spec(self) -> tuple[tuple[int, ...], str] | None:
        if self._tensor_template is None:
            return None
        return tuple(self._tensor_template.shape), str(self._tensor_template.dtype)


class AverageMeters(MetersBase):
    r"""
    Manages multiple average meters in one object.

    Examples:
        >>> meters = AverageMeters()
        >>> meters.update({"loss": 0.6, "auroc": 0.7, "r2": 0.8})
        >>> f"{meters:.4f}"
        'loss: 0.6000 (0.6000)\tauroc: 0.7000 (0.7000)\tr2: 0.8000 (0.8000)'
        >>> meters['loss'].update(value=0.9, n=1)
        >>> f"{meters:.4f}"
        'loss: 0.9000 (0.7500)\tauroc: 0.7000 (0.7000)\tr2: 0.8000 (0.8000)'
        >>> meters.sum.dict()
        {'loss': 1.5, 'auroc': 0.7, 'r2': 0.8}
        >>> meters.count.dict()
        {'loss': 2, 'auroc': 1, 'r2': 1}
        >>> meters.reset()
        AverageMeters(...)
        >>> f"{meters:.4f}"
        'loss: nan (nan)\tauroc: nan (nan)\tr2: nan (nan)'

    See Also:
        - [`StreamMetrics`][danling.metrics.stream_metrics.StreamMetrics]:
            Memory-efficient metric tracker that averages multiple metrics batch-by-batch.
    """

    meter_cls = AverageMeter  # type: ignore[assignment]

    # Aggregate state accessors
    @property
    def n(self) -> RoundDict[str, int]:
        return RoundDict({key: meter.n for key, meter in self.all_items()})

    @property
    def sum(self) -> RoundDict[str, float | Tensor]:
        return RoundDict({key: meter.sum for key, meter in self.all_items()})

    @property
    def count(self) -> RoundDict[str, int]:
        return RoundDict({key: meter.count for key, meter in self.all_items()})

    # Public reductions
    def batch(self) -> RoundDict[str, float | Tensor]:
        items = list(self.all_items())
        sync_names = [name for name, meter in items if meter._current_world_size() > 1]
        if not sync_names:
            return super().batch()

        device = self[sync_names[0]]._sync_device()
        tensor_templates = self._resolved_tensor_templates(sync_names, device)
        tensor_sync_names = set(tensor_templates)
        scalar_sync_names = [name for name in sync_names if name not in tensor_sync_names]
        if not scalar_sync_names:
            return RoundDict(
                {
                    name: (
                        meter._tensor_batch(tensor_templates.get(name)) if name in tensor_sync_names else meter.batch()
                    )
                    for name, meter in items
                }
            )

        reduced = torch.zeros(len(scalar_sync_names) * 2, dtype=torch.float64, device=device)
        sync_indices = {name: idx for idx, name in enumerate(scalar_sync_names)}

        for name in scalar_sync_names:
            meter = self[name]
            offset = sync_indices[name] * 2
            if meter._local_n > 0:
                if isinstance(meter._local_value, Tensor):
                    reduced[offset] = meter._local_value.to(device=device, dtype=torch.float64) * meter._local_n
                else:
                    reduced[offset] = float(meter._local_value) * meter._local_n
            reduced[offset + 1] = float(meter._local_n)

        dist.all_reduce(reduced)

        batches: dict[str, float | Tensor] = {}
        for name, meter in items:
            sync_index = sync_indices.get(name)
            if sync_index is None:
                batches[name] = (
                    meter._tensor_batch(tensor_templates.get(name)) if name in tensor_sync_names else meter.batch()
                )
                continue

            total, count = reduced[sync_index * 2 : sync_index * 2 + 2].tolist()
            batches[name] = nan if count == 0 else total / count

        return RoundDict(batches)

    def average(self) -> RoundDict[str, float | Tensor]:
        items = list(self.all_items())
        sync_names = [name for name, meter in items if meter._current_world_size() > 1]
        if not sync_names:
            return super().average()

        device = self[sync_names[0]]._sync_device()
        tensor_templates = self._resolved_tensor_templates(sync_names, device)
        tensor_sync_names = set(tensor_templates)
        scalar_sync_names = [name for name in sync_names if name not in tensor_sync_names]
        if not scalar_sync_names:
            return RoundDict(
                {
                    name: (
                        meter._tensor_average(tensor_templates.get(name))
                        if name in tensor_sync_names
                        else meter.average()
                    )
                    for name, meter in items
                }
            )

        reduced = torch.zeros(len(scalar_sync_names) * 2, dtype=torch.float64, device=device)
        sync_indices = {name: idx for idx, name in enumerate(scalar_sync_names)}

        for name in scalar_sync_names:
            meter = self[name]
            offset = sync_indices[name] * 2
            if isinstance(meter._local_sum, Tensor):
                reduced[offset] = meter._local_sum.to(device=device, dtype=torch.float64)
            else:
                reduced[offset] = float(meter._local_sum)
            reduced[offset + 1] = float(meter._local_count)

        dist.all_reduce(reduced)

        averages: dict[str, float | Tensor] = {}
        for name, meter in items:
            sync_index = sync_indices.get(name)
            if sync_index is None:
                averages[name] = (
                    meter._tensor_average(tensor_templates.get(name)) if name in tensor_sync_names else meter.average()
                )
                continue

            total, count = reduced[sync_index * 2 : sync_index * 2 + 2].tolist()
            averages[name] = nan if count == 0 else total / count

        return RoundDict(averages)

    def _resolved_tensor_templates(self, sync_names: list[str], device: torch.device) -> dict[str, Tensor]:
        if not sync_names:
            return {}

        local_specs = {name: self[name]._tensor_spec() for name in sync_names}
        if not (dist.is_available() and dist.is_initialized()):
            return {
                name: self[name]._resolve_tensor_template(self[name]._current_world_size(), device)
                for name, spec in local_specs.items()
                if spec is not None
            }

        world_size = self[sync_names[0]]._current_world_size()
        gathered_specs: list[dict[str, tuple[tuple[int, ...], str] | None]] = [{} for _ in range(world_size)]
        dist.all_gather_object(gathered_specs, local_specs)

        templates: dict[str, Tensor] = {}
        for name in sync_names:
            references: list[tuple[tuple[int, ...], str]] = []
            for spec in gathered_specs:
                reference = spec.get(name)
                if reference is not None:
                    references.append(reference)
            if not references:
                continue
            if any(reference != references[0] for reference in references[1:]):
                raise ValueError(
                    f"AverageMeters received inconsistent tensor metadata for meter {name!r}: {references!r}"
                )

            shape, dtype_name = references[0]
            template = torch.empty(shape, dtype=getattr(torch, dtype_name.removeprefix("torch.")), device=device)
            meter = self[name]
            template_device = meter.device if meter.device is not None else device
            meter._tensor_template = template.to(device=template_device, dtype=torch.float64)
            templates[name] = template.to(dtype=torch.float64)
        return templates

    # Mutation
    def update(
        self, *args: Mapping[str, int | float | Tensor], **values: int | float | Tensor
    ) -> None:  # pylint: disable=W0237
        r"""
        Updates the average and current value in all meters.

        Args:
            values: Mapping or keyword values to be added to the corresponding meters.
        """  # noqa: E501

        if args:
            if len(args) > 1:
                raise ValueError("Expected only one positional argument, but got multiple.")
            values = dict(args[0]) | values

        for meter, value in values.items():
            self[meter].update(value)
