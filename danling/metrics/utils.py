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
from math import isnan
from typing import Any, Generic, Union

from chanfig import DefaultDict, NestedDict
from typing_extensions import Self, TypeVar

from ..utils import flist

try:
    from typing import Self  # noqa: F811
except ImportError:
    from typing_extensions import Self  # noqa: F811


K = TypeVar("K", bound=str, default=str)
V = TypeVar("V", float, int, flist, Union[float, flist], Union[float, int], default=Union[float, flist])
TMetric = TypeVar("TMetric")


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
            self[key] = round(value, ndigits)
        return self

    def __round__(self, ndigits: int = 4) -> Self:
        dict = self.empty_like()
        for key, value in self.all_items():
            dict[key] = round(value, ndigits)
        return dict


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

    def _coerce_meter(self, value: Any):
        meter_cls = getattr(self, "meter_cls", None)
        if meter_cls is None or isinstance(value, meter_cls):
            return value
        raise ValueError(f"Expected value to be an instance of {meter_cls.__name__}, but got {type(value)}")

    def value(self) -> RoundDict[str, float]:
        return RoundDict({key: metric.value() for key, metric in self.all_items()})

    def batch(self) -> RoundDict[str, float]:
        return RoundDict({key: metric.batch() for key, metric in self.all_items()})

    def average(self) -> RoundDict[str, float]:
        return RoundDict({key: metric.average() for key, metric in self.all_items()})

    @property
    def val(self) -> RoundDict[str, float]:
        return self.value()

    @property
    def bat(self) -> RoundDict[str, float]:
        return self.batch()

    @property
    def avg(self) -> RoundDict[str, float]:
        return self.average()

    def reset(self) -> Self:
        for metric in self.all_values():
            metric.reset()
        return self

    def __format__(self, format_spec: str) -> str:
        return "\t".join(f"{key}: {metric.__format__(format_spec)}" for key, metric in self.all_items())


class MultiTaskBase(NestedDict):
    r"""
    Container that groups meters for multiple tasks and aggregates them.
    """

    return_average = False

    def __init__(self, *args, return_average: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setattr("return_average", return_average)

    def value(self) -> RoundDict[str, float]:
        output = RoundDict()
        for key, metrics in self.all_items():
            value = metrics.value()
            if all(isnan(v) for v in value.values()):
                continue
            output[key] = value
        if self.getattr("return_average", False):
            output["average"] = self.compute_average(output)
        return output

    def batch(self) -> RoundDict[str, float]:
        output = RoundDict()
        for key, metrics in self.all_items():
            value = metrics.batch()
            if all(isnan(v) for v in value.values()):
                continue
            output[key] = value
        if self.getattr("return_average", False):
            output["average"] = self.compute_average(output)
        return output

    def average(self) -> RoundDict[str, float]:
        output = RoundDict()
        for key, metrics in self.all_items():
            value = metrics.average()
            if all(isnan(v) for v in value.values()):
                continue
            output[key] = value
        if self.getattr("return_average", False):
            output["average"] = self.compute_average(output)
        return output

    def compute_average(self, output: RoundDict[str, float]) -> RoundDict[str, float]:
        average = DefaultDict(default_factory=list)
        for key, metric in output.all_items():
            average[key.rsplit(".", 1)[-1]].append(metric)
        return RoundDict({key: sum(values) / len(values) for key, values in average.items()})

    @property
    def val(self) -> RoundDict[str, float]:
        return self.value()

    @property
    def bat(self) -> RoundDict[str, float]:
        return self.batch()

    @property
    def avg(self) -> RoundDict[str, float]:
        return self.average()

    def reset(self) -> Self:
        for metric in self.all_values():
            metric.reset()
        return self

    def __format__(self, format_spec: str) -> str:
        return "\n".join(f"{key}: {metric.__format__(format_spec)}" for key, metric in self.all_items())
