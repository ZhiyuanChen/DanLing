# DanLing
# Copyright (C) 2022-Present  DanLing

# This file is part of DanLing.

# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from __future__ import annotations

from functools import partial
from math import isnan

import pytest

from danling.metrics.utils import (
    MetersBase,
    MultiTaskBase,
    RoundDict,
    infer_metric_name,
    iter_metric_funcs,
    merge_metric_entries,
)


class _DummyMeter:
    def __init__(self, value: float = 0.0):
        self.value_store = value
        self.reset_calls = 0

    def value(self) -> float:
        return self.value_store

    def batch(self) -> float:
        return self.value_store + 1

    def average(self) -> float:
        return self.value_store + 2

    def reset(self):
        self.reset_calls += 1
        return self

    def __format__(self, format_spec: str) -> str:
        return format(self.value_store, format_spec)


class _DummyMeters(MetersBase):
    meter_cls = _DummyMeter


class _StubMetric:
    def __init__(self, val: dict[str, float], bat: dict[str, float] | None = None, avg: dict[str, float] | None = None):
        self._val = val
        self._bat = bat or val
        self._avg = avg or val
        self.reset_calls = 0

    def value(self):
        return RoundDict(self._val)

    def batch(self):
        return RoundDict(self._bat)

    def average(self):
        return RoundDict(self._avg)

    def reset(self):
        self.reset_calls += 1
        return self

    def __format__(self, format_spec: str) -> str:
        return f"{self._val}"


def test_iter_metric_funcs_flattens_sequences_but_not_strings():
    assert list(iter_metric_funcs([1, [2, 3], (4, 5), "ab", b"cd"])) == [1, 2, 3, 4, 5, "ab", b"cd"]


def test_infer_metric_name_uses_name_attribute_and_partial():
    def fn(a, b):
        return a + b

    class NamedCallable:
        def __init__(self):
            self.name = "named_metric"

        def __call__(self, a, b):
            return a + b

    class NamelessCallable:
        def __call__(self, a, b):
            return a + b

    assert infer_metric_name(fn) == "fn"
    assert infer_metric_name(NamedCallable()) == "named_metric"
    assert infer_metric_name(partial(fn, 1)) == "fn"

    with pytest.raises(ValueError, match="Unable to infer metric name"):
        infer_metric_name(NamelessCallable())


def test_merge_metric_entries_prefers_first_positional_then_named_override():
    positional = [("acc", 1), ("acc", 2), ("loss", 3)]
    merged = merge_metric_entries(positional, {"acc": 4, "f1": 5})
    assert merged == {"acc": 4, "loss": 3, "f1": 5}


def test_meters_base_init_validation_and_operations():
    with pytest.raises(TypeError, match="at most one positional mapping argument"):
        _DummyMeters({}, {})

    meters = _DummyMeters({"a": _DummyMeter(1.0)}, b=_DummyMeter(2.0))
    assert set(meters.keys()) == {"a", "b"}

    with pytest.raises(ValueError, match="Expected value to be an instance of _DummyMeter"):
        meters.set("bad", 1)  # type: ignore[arg-type]

    assert meters.val["a"] == 1.0
    assert meters.bat["a"] == 2.0
    assert meters.avg["a"] == 3.0
    assert "a: 1.00" in format(meters, ".2f")

    meters.reset()
    assert meters["a"].reset_calls == 1
    assert meters["b"].reset_calls == 1


def test_multitask_base_skips_all_nan_and_computes_average():
    metrics = MultiTaskBase(return_average=True)
    metrics["task1"] = _StubMetric({"acc": 0.8, "f1": 0.6}, {"acc": 0.7, "f1": 0.5}, {"acc": 0.9, "f1": 0.7})
    metrics["task2"] = _StubMetric({"acc": float("nan"), "f1": float("nan")})
    metrics["task3"] = _StubMetric({"acc": 0.4, "f1": 0.2}, {"acc": 0.3, "f1": 0.1}, {"acc": 0.5, "f1": 0.3})

    value = metrics.value()
    assert "task2" not in value
    assert value["average"]["acc"] == pytest.approx(0.6)
    assert value["average"]["f1"] == pytest.approx(0.4)

    batch = metrics.batch()
    assert batch["average"]["acc"] == pytest.approx(0.5)
    assert batch["average"]["f1"] == pytest.approx(0.3)

    average = metrics.average()
    assert average["average"]["acc"] == pytest.approx(0.7)
    assert average["average"]["f1"] == pytest.approx(0.5)

    formatted = format(metrics, ".2f")
    assert "task1:" in formatted
    assert "\n" in formatted

    metrics.reset()
    assert metrics["task1"].reset_calls == 1
    assert metrics["task2"].reset_calls == 1
    assert metrics["task3"].reset_calls == 1


def test_multitask_base_without_average_flag():
    metrics = MultiTaskBase(return_average=False)
    metrics["task"] = _StubMetric({"score": 1.0})
    output = metrics.value()
    assert "average" not in output
    assert output["task"]["score"] == 1.0


def test_multitask_base_all_nan_outputs_empty():
    metrics = MultiTaskBase(return_average=True)
    metrics["task"] = _StubMetric({"score": float("nan")})
    output = metrics.value()
    assert list(output.keys()) == ["average"]
    assert output["average"] == {}
    assert isnan(metrics["task"].value()["score"])
