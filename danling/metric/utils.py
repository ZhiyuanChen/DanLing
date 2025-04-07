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

from typing import Generic, Union

from chanfig import DefaultDict, NestedDict
from typing_extensions import TypeVar

from ..utils import flist

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self


K = TypeVar("K", bound=str, default=str)
V = TypeVar("V", float, int, flist, Union[float, flist], Union[float, int], default=Union[float, flist])


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


class MetricsDict(DefaultDict):
    r"""
    A `MetricsDict` for better support for `AverageMeters`.
    """

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


class MultiTaskDict(NestedDict):
    r"""
    A `MultiTaskDict` for better multi-task support For `MultiTaskAverageMeters` and `MultiTaskMetrics`.
    """

    return_average = False

    def __init__(self, *args, return_average: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setattr("return_average", return_average)

    def value(self) -> RoundDict[str, float]:
        output = RoundDict({key: metric.value() for key, metric in self.all_items()})
        if self.getattr("return_average", False):
            average = DefaultDict(default_factory=list)
            for key, metric in output.all_items():
                average[key.rsplit(".", 1)[-1]].append(metric)
            output["average"] = RoundDict({key: sum(values) / len(values) for key, values in average.items()})
        return output

    def batch(self) -> RoundDict[str, float]:
        output = RoundDict({key: metric.batch() for key, metric in self.all_items()})
        if self.getattr("return_average", False):
            average = DefaultDict(default_factory=list)
            for key, metric in output.all_items():
                average[key.rsplit(".", 1)[-1]].append(metric)
            output["average"] = RoundDict({key: sum(values) / len(values) for key, values in average.items()})
        return output

    def average(self) -> RoundDict[str, float]:
        output = RoundDict({key: metric.average() for key, metric in self.all_items()})
        if self.getattr("return_average", False):
            average = DefaultDict(default_factory=list)
            for key, metric in output.all_items():
                average[key.rsplit(".", 1)[-1]].append(metric)
            output["average"] = RoundDict({key: sum(values) / len(values) for key, values in average.items()})
        return output

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
