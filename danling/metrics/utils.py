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

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from math import isfinite, isnan
from typing import Any, Generic, Literal
from warnings import warn

import torch
from chanfig import DefaultDict, FlatDict, NestedDict
from torch import Tensor
from torch import device as torch_device
from torch import distributed as dist
from typing_extensions import Self, TypeVar

K = TypeVar("K", bound=str, default=str)
V = TypeVar("V", default=Any)
TMetric = TypeVar("TMetric")
Reduction = Literal["value", "batch", "average"]


def infer_device(device: torch.device | str | None = None) -> torch.device:
    if device is not None:
        return torch_device(device)

    inferred_device: torch.device | None = None

    if dist.is_available() and dist.is_initialized():
        backend = dist.get_backend()
        if backend in {"nccl", "cuda"} and torch.cuda.is_available():
            try:
                index = torch.cuda.current_device()
            except (AssertionError, RuntimeError):
                index = 0
            inferred_device = torch.device("cuda", index)
        elif backend in {"gloo", "mpi"}:
            inferred_device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            try:
                index = torch.cuda.current_device()
            except (AssertionError, RuntimeError):
                index = 0
            inferred_device = torch.device("cuda", index)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            inferred_device = torch.device("mps")
        else:
            inferred_device = torch.device("cpu")

    if inferred_device is None:
        warn("Failed to infer device, defaulting to CPU.")
        inferred_device = torch.device("cpu")

    return inferred_device


def iter_metric_funcs(metric_funcs: Sequence[Any]):
    for metric in metric_funcs:
        if isinstance(metric, Sequence) and not isinstance(metric, (str, bytes)):
            yield from metric
        else:
            yield metric


def infer_metric_name(metric: Callable[..., Any]) -> str:
    name = getattr(metric, "__name__", None)
    if name is None:
        name = getattr(metric, "name", None)
    if name is None and isinstance(metric, partial):
        name = getattr(metric.func, "__name__", None)
    if name is None:
        raise ValueError("Unable to infer metric name from positional metric; pass it as a keyword argument instead.")
    return name


def merge_metric_entries(
    positional: Sequence[tuple[str, TMetric]],
    named: Mapping[str, TMetric],
) -> dict[str, TMetric]:
    named_metrics: dict[str, TMetric] = {}
    for name, metric in positional:
        named_metrics.setdefault(name, metric)
    for name, metric in named.items():
        named_metrics[name] = metric
    return named_metrics


class RoundDict(NestedDict, Generic[K, V]):

    def round(self, ndigits: int = 4) -> Self:
        for key, value in self.all_items():
            self[key] = self._round_value(value, ndigits)
        return self

    def __round__(self, ndigits: int = 4) -> Self:
        dict = self.empty_like()
        for key, value in self.all_items():
            dict[key] = self._round_value(value, ndigits)
        return dict

    @staticmethod
    def _round_value(value: Any, ndigits: int):
        if isinstance(value, Tensor):
            return value.round(decimals=ndigits)
        return round(value, ndigits)


class MetersBase(DefaultDict):
    r"""Base container for collections of meter objects.

    Subclasses can provide a `meter_cls` attribute to enforce the type of
    values stored in the dictionary and customise how callable objects are
    converted into meters.
    """

    meter_cls = None

    def __init__(self, *args: Mapping[str, Any] | None, default_factory=None, **meters: Any) -> None:
        meter_cls = getattr(type(self), "meter_cls", None)
        factory = default_factory if default_factory is not None else meter_cls
        super().__init__(default_factory=factory)

        initial: dict[str, Any] = {}
        if args:
            if len(args) > 1:
                raise TypeError("MetersBase accepts at most one positional mapping argument.")
            mapping = args[0]
            if mapping is not None:
                initial.update(dict(mapping))
        if meters:
            initial.update(meters)
        for name, meter in initial.items():
            self.set(name, meter)

    def set(self, name: Any, value: Any) -> None:
        super().set(name, self._coerce_meter(value))

    # Construction helpers
    def _coerce_meter(self, value: Any):
        meter_cls = getattr(self, "meter_cls", None)
        if meter_cls is None or isinstance(value, meter_cls):
            return value
        raise ValueError(f"Expected value to be an instance of {meter_cls.__name__}, but got {type(value)}")

    # Public reductions
    def value(self) -> RoundDict[str, float | Tensor]:
        return RoundDict({key: metric.value() for key, metric in self.all_items()})

    def batch(self) -> RoundDict[str, float | Tensor]:
        return RoundDict({key: metric.batch() for key, metric in self.all_items()})

    def average(self) -> RoundDict[str, float | Tensor]:
        return RoundDict({key: metric.average() for key, metric in self.all_items()})

    # Public aliases
    @property
    def val(self) -> RoundDict[str, float | Tensor]:
        return self.value()

    @property
    def bat(self) -> RoundDict[str, float | Tensor]:
        return self.batch()

    @property
    def avg(self) -> RoundDict[str, float | Tensor]:
        return self.average()

    # Lifecycle
    def reset(self) -> Self:
        for metric in self.all_values():
            metric.reset()
        return self

    # Formatting helpers
    def __format__(self, format_spec: str) -> str:
        return "\t".join(f"{key}: {metric.__format__(format_spec)}" for key, metric in self.all_items())


class MultiTaskBase(FlatDict):
    r"""
    Container that groups meters for multiple tasks and aggregates them.
    """

    def __init__(
        self,
        *args,
        aggregate: Literal["macro", "micro", "weighted"] | None = None,
        aggregate_weights: Mapping[str, float | int | Tensor] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if aggregate not in {None, "macro", "micro", "weighted"}:
            raise ValueError(f"aggregate must be one of None, 'macro', 'micro', or 'weighted', but got {aggregate!r}")
        if aggregate == "weighted":
            if aggregate_weights is None:
                raise ValueError("aggregate_weights is required when aggregate='weighted'")
        elif aggregate_weights is not None:
            raise ValueError("aggregate_weights is only supported when aggregate='weighted'")
        self.setattr("_aggregate", aggregate)
        self.setattr(
            "_aggregate_weights",
            (
                None
                if aggregate_weights is None
                else {
                    str(name): self._normalize_weight_value(weight, label=f"aggregate weight for task {name!r}")
                    for name, weight in aggregate_weights.items()
                }
            ),
        )

    # Normalization helpers
    @classmethod
    def _normalize_metric_output(
        cls, name: str, metrics: Any, value: Mapping[str, float | Tensor] | float | Tensor
    ) -> RoundDict[str, float | Tensor]:
        if isinstance(value, Mapping):
            if isinstance(value, RoundDict):
                return value
            return RoundDict(value)
        output_name = getattr(metrics, "output_name", None)
        if output_name is not None:
            output_name = str(output_name)
            if output_name in {"", "<lambda>", "__call__"}:
                output_name = None
        if output_name is None:
            output_name = name
        return RoundDict({output_name: value})

    @classmethod
    def _flatten_mapping(cls, mapping: Mapping[str, Any]) -> dict[str, Any]:
        if hasattr(mapping, "all_items"):
            return dict(mapping.all_items())

        flat: dict[str, Any] = {}
        for key, value in mapping.items():
            path = str(key)
            if isinstance(value, Mapping):
                for nested_key, nested_value in cls._flatten_mapping(value).items():
                    flat[f"{path}.{nested_key}"] = nested_value
            else:
                flat[path] = value
        return flat

    @staticmethod
    def _is_nan_value(value: Any) -> bool:
        if isinstance(value, Tensor):
            return bool(value.isnan().all().item())
        try:
            return isnan(value)
        except TypeError:
            return False

    @staticmethod
    def _to_averageable(value: Any, *, label: str) -> float | Tensor:
        if isinstance(value, Tensor):
            return value.detach().to(dtype=torch.float64)
        return float(value)

    @staticmethod
    def _normalize_weight_value(value: Any, *, label: str) -> float:
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise ValueError(f"{label} must be scalar, but got shape {tuple(value.shape)}")
            value = float(value.item())
        else:
            value = float(value)
        if not isfinite(value) or value < 0:
            raise ValueError(f"{label} must be a non-negative finite scalar, but got {value!r}")
        return value

    @staticmethod
    def _reduction_weight_attr(reduction: Reduction) -> str:
        if reduction in {"value", "batch"}:
            return "n"
        if reduction == "average":
            return "count"
        raise ValueError(f"Unsupported reduction: {reduction!r}")

    @classmethod
    def _weight_for_path(
        cls,
        task_name: str,
        path: str,
        weight_source: Mapping[str, Any] | Any,
        *,
        label_prefix: str,
    ) -> float:
        if isinstance(weight_source, Mapping):
            flat_weights = cls._flatten_mapping(weight_source)
            if path not in flat_weights:
                raise ValueError(f"{label_prefix} is missing a weight for metric '{task_name}.{path}'")
            return cls._normalize_weight_value(
                flat_weights[path], label=f"{label_prefix} for metric '{task_name}.{path}'"
            )
        return cls._normalize_weight_value(weight_source, label=f"{label_prefix} for task {task_name!r}")

    @staticmethod
    def _sync_weights(weights: list[float]) -> list[float]:
        if not weights or not (dist.is_available() and dist.is_initialized()):
            return weights
        device = infer_device()
        reduced = torch.tensor(weights, dtype=torch.float64, device=device)
        dist.all_reduce(reduced)
        return reduced.tolist()

    # Public reductions
    def _collect_output(self, reduction: Reduction) -> RoundDict[str, float | Tensor]:
        output = RoundDict()
        for key, metrics in self.all_items():
            value = self._normalize_metric_output(key, metrics, getattr(metrics, reduction)())
            if all(self._is_nan_value(v) for v in value.all_values()):
                continue
            output[key] = value
        aggregate = self.getattr("_aggregate", None)
        if aggregate is not None:
            output["aggregate"] = self.compute_aggregate(output, reduction)
        return output

    def value(self) -> RoundDict[str, float | Tensor]:
        return self._collect_output("value")

    def batch(self) -> RoundDict[str, float | Tensor]:
        return self._collect_output("batch")

    def average(self) -> RoundDict[str, float | Tensor]:
        return self._collect_output("average")

    def compute_aggregate(
        self,
        output: RoundDict[str, float | Tensor],
        reduction: Reduction,
    ) -> RoundDict[str, float | Tensor]:
        aggregate = self.getattr("_aggregate", None)
        if aggregate is None:
            return RoundDict()
        if aggregate == "macro":
            return self.compute_average(output)
        if aggregate == "micro":
            return self.compute_weighted_average(output, reduction=reduction, mode="micro")
        return self.compute_weighted_average(output, reduction=reduction, mode="weighted")

    def compute_average(self, output: RoundDict[str, float | Tensor]) -> RoundDict[str, float | Tensor]:
        totals: dict[str, float | Tensor] = {}
        counts: dict[str, int] = {}

        for task_name, task_output in output.items():
            if task_name == "aggregate":
                continue
            flat_output = self._flatten_mapping(task_output)
            for path, value in flat_output.items():
                if self._is_nan_value(value):
                    continue
                averageable_value = self._to_averageable(value, label=f"metric '{task_name}.{path}'")
                if path not in totals:
                    totals[path] = (
                        averageable_value.clone() if isinstance(averageable_value, Tensor) else averageable_value
                    )
                else:
                    total = totals[path]
                    if isinstance(averageable_value, Tensor):
                        if not isinstance(total, Tensor):
                            raise ValueError(f"metric '{path}' mixes scalar and tensor outputs across tasks")
                        if total.shape != averageable_value.shape:
                            raise ValueError(
                                f"metric '{path}' has inconsistent tensor shapes across tasks: "
                                f"{tuple(total.shape)} vs {tuple(averageable_value.shape)}"
                            )
                        totals[path] = total + averageable_value
                    else:
                        if isinstance(total, Tensor):
                            raise ValueError(f"metric '{path}' mixes tensor and scalar outputs across tasks")
                        totals[path] = total + averageable_value
                counts[path] = counts.get(path, 0) + 1

        average = RoundDict()
        for path, total in totals.items():
            average[path] = total / counts[path]
        return average

    def compute_weighted_average(
        self,
        output: RoundDict[str, float | Tensor],
        *,
        reduction: Reduction,
        mode: Literal["micro", "weighted"],
    ) -> RoundDict[str, float | Tensor]:
        if mode == "weighted":
            task_weights = self.getattr("_aggregate_weights", None)
            if task_weights is None:
                raise ValueError("aggregate_weights is required when aggregate='weighted'")
            unknown_tasks = set(task_weights) - {str(name) for name in self.keys()}
            if unknown_tasks:
                raise ValueError(f"aggregate_weights contains unknown tasks: {sorted(unknown_tasks)!r}")
        else:
            task_weights = None

        weighted_entries: list[tuple[str, float | Tensor, float]] = []
        for task_name, task_output in output.items():
            if task_name == "aggregate":
                continue

            flat_output = self._flatten_mapping(task_output)
            if mode == "micro":
                weight_attr = self._reduction_weight_attr(reduction)
                if not hasattr(self[task_name], weight_attr):
                    raise ValueError(
                        f"micro aggregate requires task {task_name!r} to expose {weight_attr!r} sample counts"
                    )
                weight_source = getattr(self[task_name], weight_attr)
                label_prefix = "sample weight"
            else:
                if task_name not in task_weights:
                    raise ValueError(f"aggregate_weights is missing a weight for task {task_name!r}")
                weight_source = task_weights[task_name]
                label_prefix = "aggregate weight"

            for path, value in flat_output.items():
                if self._is_nan_value(value):
                    continue
                averageable_value = self._to_averageable(value, label=f"metric '{task_name}.{path}'")
                weight = self._weight_for_path(task_name, path, weight_source, label_prefix=label_prefix)
                weighted_entries.append((path, averageable_value, weight))

        if mode == "micro" and reduction in {"batch", "average"}:
            synced_weights = self._sync_weights([weight for _, _, weight in weighted_entries])
        else:
            synced_weights = [weight for _, _, weight in weighted_entries]

        totals: dict[str, float | Tensor] = {}
        weight_totals: dict[str, float] = {}
        for (path, averageable_value, _), weight in zip(weighted_entries, synced_weights):
            if weight <= 0:
                continue

            if path not in totals:
                totals[path] = averageable_value * weight
            else:
                total = totals[path]
                if isinstance(averageable_value, Tensor):
                    if not isinstance(total, Tensor):
                        raise ValueError(f"metric '{path}' mixes scalar and tensor outputs across tasks")
                    if total.shape != averageable_value.shape:
                        raise ValueError(
                            f"metric '{path}' has inconsistent tensor shapes across tasks: "
                            f"{tuple(total.shape)} vs {tuple(averageable_value.shape)}"
                        )
                    totals[path] = total + averageable_value * weight
                else:
                    if isinstance(total, Tensor):
                        raise ValueError(f"metric '{path}' mixes tensor and scalar outputs across tasks")
                    totals[path] = total + averageable_value * weight
            weight_totals[path] = weight_totals.get(path, 0.0) + weight

        average = RoundDict()
        for path, total in totals.items():
            total_weight = weight_totals[path]
            if total_weight > 0:
                average[path] = total / total_weight
        return average

    # Public aliases
    @property
    def val(self) -> RoundDict[str, float | Tensor]:
        return self.value()

    @property
    def bat(self) -> RoundDict[str, float | Tensor]:
        return self.batch()

    @property
    def avg(self) -> RoundDict[str, float | Tensor]:
        return self.average()

    # Lifecycle
    def reset(self) -> Self:
        for metric in self.all_values():
            metric.reset()
        return self

    # Formatting helpers
    def __format__(self, format_spec: str) -> str:
        return "\n".join(f"{key}: {metric.__format__(format_spec)}" for key, metric in self.all_items())
