# DanLing
# Copyright (C) 2022-Present  DanLing

# This program is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from __future__ import annotations

from chanfig import DefaultDict, NestedDict


class MetricsDict(DefaultDict):
    r"""
    A `MetricsDict` for better support for `AverageMeters`.
    """

    def value(self) -> NestedDict[str, float]:
        return NestedDict({key: metric.value() for key, metric in self.all_items()})

    def batch(self) -> NestedDict[str, float]:
        return NestedDict({key: metric.batch() for key, metric in self.all_items()})

    def average(self) -> NestedDict[str, float]:
        return NestedDict({key: metric.average() for key, metric in self.all_items()})

    @property
    def val(self) -> NestedDict[str, float]:
        return self.value()

    @property
    def bat(self) -> NestedDict[str, float]:
        return self.batch()

    @property
    def avg(self) -> NestedDict[str, float]:
        return self.average()

    def reset(self) -> None:
        for metric in self.all_values():
            metric.reset()

    def __format__(self, format_spec) -> str:
        return "\t".join(f"{key}: {metric.__format__(format_spec)}" for key, metric in self.all_items())


class MultiTaskDict(NestedDict):
    r"""
    A `MultiTaskDict` for better multi-task support For `MultiTaskAverageMeters` and `MultiTaskMetrics`.
    """

    return_average = False

    def __init__(self, *args, return_average: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setattr("return_average", return_average)

    def value(self) -> NestedDict[str, float]:
        output = NestedDict({key: metric.value() for key, metric in self.all_items()})
        if self.getattr("return_average", False):
            average = DefaultDict(default_factory=list)
            for key, metric in output.all_items():
                average[key.rsplit(".", 1)[-1]].append(metric)
            output["average"] = NestedDict({key: sum(values) / len(values) for key, values in average.items()})
        return output

    def batch(self) -> NestedDict[str, float]:
        output = NestedDict({key: metric.batch() for key, metric in self.all_items()})
        if self.getattr("return_average", False):
            average = DefaultDict(default_factory=list)
            for key, metric in output.all_items():
                average[key.rsplit(".", 1)[-1]].append(metric)
            output["average"] = NestedDict({key: sum(values) / len(values) for key, values in average.items()})
        return output

    def average(self) -> NestedDict[str, float]:
        output = NestedDict({key: metric.average() for key, metric in self.all_items()})
        if self.getattr("return_average", False):
            average = DefaultDict(default_factory=list)
            for key, metric in output.all_items():
                average[key.rsplit(".", 1)[-1]].append(metric)
            output["average"] = NestedDict({key: sum(values) / len(values) for key, values in average.items()})
        return output

    @property
    def val(self) -> NestedDict[str, float]:
        return self.value()

    @property
    def bat(self) -> NestedDict[str, float]:
        return self.batch()

    @property
    def avg(self) -> NestedDict[str, float]:
        return self.average()

    def reset(self) -> None:
        for metric in self.all_values():
            metric.reset()

    def __format__(self, format_spec) -> str:
        return "\n".join(f"{key}: {metric.__format__(format_spec)}" for key, metric in self.all_items())
