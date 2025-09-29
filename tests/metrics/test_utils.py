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
    def __init__(
        self,
        val: dict[str, float],
        bat: dict[str, float] | None = None,
        avg: dict[str, float] | None = None,
        *,
        n: dict[str, float] | float = 1,
        count: dict[str, float] | float = 1,
    ):
        self._val = val
        self._bat = bat or val
        self._avg = avg or val
        self.n = n
        self.count = count
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


def _build_aggregate_stub_metrics(
    *,
    aggregate: str,
    aggregate_weights: dict[str, float] | None = None,
) -> MultiTaskBase:
    metrics = MultiTaskBase(aggregate=aggregate, aggregate_weights=aggregate_weights)
    metrics["task1"] = _StubMetric(
        {"acc": 0.8, "f1": 0.6},
        {"acc": 0.7, "f1": 0.5},
        {"acc": 0.9, "f1": 0.7},
        n={"acc": 8, "f1": 8},
        count={"acc": 20, "f1": 20},
    )
    metrics["task2"] = _StubMetric(
        {"acc": 0.4, "f1": 0.2},
        {"acc": 0.3, "f1": 0.1},
        {"acc": 0.5, "f1": 0.3},
        n={"acc": 2, "f1": 2},
        count={"acc": 10, "f1": 10},
    )
    return metrics


class TestMetricUtilities:
    def test_iter_funcs_flattens_sequences_not_strings(self):
        assert list(iter_metric_funcs([1, [2, 3], (4, 5), "ab", b"cd"])) == [1, 2, 3, 4, 5, "ab", b"cd"]

    def test_infer_name_uses_function_name(self):
        def fn(a, b):
            return a + b

        assert infer_metric_name(fn) == "fn"

    def test_infer_name_uses_name_attribute(self):
        class NamedCallable:
            def __init__(self):
                self.name = "named_metric"

            def __call__(self, a, b):
                return a + b

        assert infer_metric_name(NamedCallable()) == "named_metric"

    def test_infer_name_uses_partial_function(self):
        def fn(a, b):
            return a + b

        assert infer_metric_name(partial(fn, 1)) == "fn"

    def test_infer_name_rejects_nameless_callable(self):
        class NamelessCallable:
            def __call__(self, a, b):
                return a + b

        with pytest.raises(ValueError, match="Unable to infer metric name"):
            infer_metric_name(NamelessCallable())

    def test_merge_entries_prefers_named_override(self):
        positional = [("acc", 1), ("acc", 2), ("loss", 3)]
        merged = merge_metric_entries(positional, {"acc": 4, "f1": 5})
        assert merged == {"acc": 4, "loss": 3, "f1": 5}

    def test_round_dict_rounds_in_place(self):
        values = RoundDict({"a": 1.23456, "b": 9.87654})
        rounded_in_place = values.round(2)
        assert rounded_in_place is values
        assert values["a"] == 1.23
        assert values["b"] == 9.88

    def test_round_dict_dunder_round_returns_copy(self):
        original = RoundDict({"x": 3.14159, "y": 2.71828})
        rounded_copy = round(original, 3)
        assert rounded_copy["x"] == 3.142
        assert rounded_copy["y"] == 2.718
        assert original["x"] == 3.14159


class TestMetersBase:
    def test_rejects_multiple_positional_mappings(self):
        with pytest.raises(TypeError, match="at most one positional mapping argument"):
            _DummyMeters({}, {})

    def test_rejects_invalid_meter_assignment(self):
        meters = _DummyMeters({"a": _DummyMeter(1.0)}, b=_DummyMeter(2.0))
        with pytest.raises(ValueError, match="Expected value to be an instance of _DummyMeter"):
            meters.set("bad", 1)  # type: ignore[arg-type]

    def test_exposes_reduction_aliases(self):
        meters = _DummyMeters({"a": _DummyMeter(1.0)}, b=_DummyMeter(2.0))
        assert set(meters.keys()) == {"a", "b"}
        assert meters.val["a"] == 1.0
        assert meters.bat["a"] == 2.0
        assert meters.avg["a"] == 3.0
        assert "a: 1.00" in format(meters, ".2f")

    def test_reset_resets_all_children(self):
        meters = _DummyMeters({"a": _DummyMeter(1.0)}, b=_DummyMeter(2.0))
        meters.reset()
        assert meters["a"].reset_calls == 1
        assert meters["b"].reset_calls == 1


class TestMultiTaskBase:
    def test_skips_tasks_with_all_nan_outputs(self):
        metrics = MultiTaskBase(aggregate="macro")
        metrics["task1"] = _StubMetric({"acc": 0.8, "f1": 0.6})
        metrics["task2"] = _StubMetric({"acc": float("nan"), "f1": float("nan")})

        value = metrics.value()

        assert "task1" in value
        assert "task2" not in value

    @pytest.mark.parametrize(
        ("aggregate", "aggregate_weights", "expected"),
        [
            (
                "macro",
                None,
                {
                    "value": {"acc": 0.6, "f1": 0.4},
                    "batch": {"acc": 0.5, "f1": 0.3},
                    "average": {"acc": 0.7, "f1": 0.5},
                },
            ),
            (
                "micro",
                None,
                {
                    "value": {"acc": 0.72, "f1": 0.52},
                    "batch": {"acc": 0.62, "f1": 0.42},
                    "average": {"acc": 0.7666666667, "f1": 0.5666666667},
                },
            ),
            (
                "weighted",
                {"task1": 4.0, "task2": 1.0},
                {
                    "value": {"acc": 0.72, "f1": 0.52},
                    "batch": {"acc": 0.62, "f1": 0.42},
                    "average": {"acc": 0.82, "f1": 0.62},
                },
            ),
        ],
    )
    def test_aggregates_matching_metrics(self, aggregate, aggregate_weights, expected):
        metrics = _build_aggregate_stub_metrics(aggregate=aggregate, aggregate_weights=aggregate_weights)

        for reduction, aggregate_output in {
            "value": metrics.value()["aggregate"],
            "batch": metrics.batch()["aggregate"],
            "average": metrics.average()["aggregate"],
        }.items():
            assert aggregate_output["acc"] == pytest.approx(expected[reduction]["acc"])
            assert aggregate_output["f1"] == pytest.approx(expected[reduction]["f1"])

    def test_format_includes_each_task(self):
        metrics = MultiTaskBase(aggregate="macro")
        metrics["task1"] = _StubMetric({"acc": 0.8})
        metrics["task2"] = _StubMetric({"acc": 0.4})

        formatted = format(metrics, ".2f")
        assert "task1:" in formatted
        assert "\n" in formatted

    def test_reset_resets_all_children(self):
        metrics = MultiTaskBase(aggregate="macro")
        metrics["task1"] = _StubMetric({"acc": 0.8})
        metrics["task2"] = _StubMetric({"acc": float("nan")})
        metrics["task3"] = _StubMetric({"acc": 0.4})

        metrics.reset()
        assert metrics["task1"].reset_calls == 1
        assert metrics["task2"].reset_calls == 1
        assert metrics["task3"].reset_calls == 1

    def test_omits_aggregate_output_when_disabled(self):
        metrics = MultiTaskBase(aggregate=None)
        metrics["task"] = _StubMetric({"score": 1.0})
        output = metrics.value()
        assert "aggregate" not in output
        assert output["task"]["score"] == 1.0

    def test_averages_by_exact_metric_path(self):
        metrics = MultiTaskBase(aggregate="macro")
        metrics["task1"] = _StubMetric({"head1": {"acc": 0.8}}, n={"head1": {"acc": 2}}, count={"head1": {"acc": 2}})
        metrics["task2"] = _StubMetric({"head2": {"acc": 0.4}}, n={"head2": {"acc": 6}}, count={"head2": {"acc": 6}})

        output = metrics.value()
        assert output["aggregate"]["head1"]["acc"] == pytest.approx(0.8)
        assert output["aggregate"]["head2"]["acc"] == pytest.approx(0.4)
        assert "acc" not in output["aggregate"]

    def test_empty_aggregate_when_all_outputs_are_nan(self):
        metrics = MultiTaskBase(aggregate="macro")
        metrics["task"] = _StubMetric({"score": float("nan")})
        output = metrics.value()
        assert list(output.keys()) == ["aggregate"]
        assert output["aggregate"] == {}
        assert isnan(metrics["task"].value()["score"])

    def test_rejects_unsupported_aggregate_mode(self):
        with pytest.raises(ValueError, match="aggregate must be one of None, 'macro', 'micro', or 'weighted'"):
            MultiTaskBase(aggregate="median")  # type: ignore[arg-type]

    def test_rejects_weighted_without_aggregate_weights(self):
        with pytest.raises(ValueError, match="aggregate_weights is required when aggregate='weighted'"):
            MultiTaskBase(aggregate="weighted")

    def test_rejects_aggregate_weights_without_weighted(self):
        with pytest.raises(ValueError, match="aggregate_weights is only supported when aggregate='weighted'"):
            MultiTaskBase(aggregate="macro", aggregate_weights={"task": 1.0})

    def test_weighted_rejects_missing_task_weights(self):
        metrics = MultiTaskBase(aggregate="weighted", aggregate_weights={"task1": 1.0})
        metrics["task1"] = _StubMetric({"acc": 1.0})
        metrics["task2"] = _StubMetric({"acc": 0.0})

        with pytest.raises(ValueError, match="aggregate_weights is missing a weight for task 'task2'"):
            metrics.value()
