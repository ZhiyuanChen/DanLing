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

import random

import pytest
import torch
from torch.testing import assert_close

from danling.metrics.factory import binary_metrics
from danling.metrics.functional import binary_accuracy
from danling.metrics.global_metrics import GlobalMetrics
from danling.metrics.multitask import MultiTaskMetrics
from danling.metrics.preprocess import preprocess_binary
from danling.metrics.stream_metrics import MetricMeter, StreamMetrics

from .utils import process_group, run_distributed


def _distributed_multitask_worker(rank: int, world_size: int):
    with process_group("gloo", rank, world_size):
        counts = {"a": 1, "b": 9} if rank == 0 else {"a": 9, "b": 1}

        global_metrics = MultiTaskMetrics(aggregate="macro")
        global_metrics.a = GlobalMetrics(
            [binary_accuracy()], preprocess=preprocess_binary, distributed=True, device="cpu"
        )
        global_metrics.b = GlobalMetrics(
            [binary_accuracy()], preprocess=preprocess_binary, distributed=True, device="cpu"
        )
        global_metrics.update(
            {
                "a": (torch.full((counts["a"],), 10.0), torch.ones(counts["a"], dtype=torch.long)),
                "b": (torch.full((counts["b"],), 10.0), torch.zeros(counts["b"], dtype=torch.long)),
            }
        )

        global_batch = global_metrics.bat
        global_average = global_metrics.average()
        assert global_batch["a"]["acc"] == pytest.approx(1.0)
        assert global_batch["b"]["acc"] == pytest.approx(0.0)
        assert global_batch["aggregate"]["acc"] == pytest.approx(0.5)
        assert global_average["a"]["acc"] == pytest.approx(1.0)
        assert global_average["b"]["acc"] == pytest.approx(0.0)
        assert global_average["aggregate"]["acc"] == pytest.approx(0.5)

        stream_metrics = MultiTaskMetrics(aggregate="macro")
        stream_metrics.a = StreamMetrics(
            score=lambda input, target: input.float().mean(), distributed=True, device="cpu"
        )
        stream_metrics.b = StreamMetrics(
            score=lambda input, target: input.float().mean(), distributed=True, device="cpu"
        )
        stream_metrics.update(
            {
                "a": (torch.full((counts["a"],), 0.1), torch.zeros(counts["a"])),
                "b": (torch.full((counts["b"],), 0.9), torch.zeros(counts["b"])),
            }
        )

        stream_batch = stream_metrics.batch()
        stream_average = stream_metrics.average()
        assert stream_batch["a"]["score"] == pytest.approx(0.1)
        assert stream_batch["b"]["score"] == pytest.approx(0.9)
        assert stream_batch["aggregate"]["score"] == pytest.approx(0.5)
        assert stream_average["a"]["score"] == pytest.approx(0.1)
        assert stream_average["b"]["score"] == pytest.approx(0.9)
        assert stream_average["aggregate"]["score"] == pytest.approx(0.5)


def _distributed_multitask_micro_worker(rank: int, world_size: int):
    with process_group("gloo", rank, world_size):
        counts = {"a": 1, "b": 9} if rank == 0 else {"a": 1, "b": 1}

        global_metrics = MultiTaskMetrics(aggregate="micro")
        global_metrics.a = GlobalMetrics(
            [binary_accuracy()], preprocess=preprocess_binary, distributed=True, device="cpu"
        )
        global_metrics.b = GlobalMetrics(
            [binary_accuracy()], preprocess=preprocess_binary, distributed=True, device="cpu"
        )
        global_metrics.update(
            {
                "a": (torch.full((counts["a"],), 10.0), torch.ones(counts["a"], dtype=torch.long)),
                "b": (torch.full((counts["b"],), 10.0), torch.zeros(counts["b"], dtype=torch.long)),
            }
        )

        global_batch = global_metrics.bat
        global_average = global_metrics.average()
        assert global_batch["aggregate"]["acc"] == pytest.approx(1.0 / 6.0)
        assert global_average["aggregate"]["acc"] == pytest.approx(1.0 / 6.0)

        stream_metrics = MultiTaskMetrics(aggregate="micro")
        stream_metrics.a = StreamMetrics(
            score=lambda input, target: input.float().mean(), distributed=True, device="cpu"
        )
        stream_metrics.b = StreamMetrics(
            score=lambda input, target: input.float().mean(), distributed=True, device="cpu"
        )
        stream_metrics.update(
            {
                "a": (torch.full((counts["a"],), 0.1), torch.zeros(counts["a"])),
                "b": (torch.full((counts["b"],), 0.9), torch.zeros(counts["b"])),
            }
        )

        stream_batch = stream_metrics.batch()
        stream_average = stream_metrics.average()
        assert stream_batch["aggregate"]["score"] == pytest.approx((0.1 * 2 + 0.9 * 10) / 12)
        assert stream_average["aggregate"]["score"] == pytest.approx((0.1 * 2 + 0.9 * 10) / 12)


class TestMultiTaskMetrics:
    def test_accepts_sequence_payloads(self):
        random.seed(0)
        torch.random.manual_seed(0)
        metrics = MultiTaskMetrics()
        metrics.a = binary_metrics(mode="stream")
        metrics.b = binary_metrics(mode="stream")

        metrics.update(
            {
                "a": (torch.randn(8), torch.randint(2, (8,))),
                "b": (torch.randn(8), torch.randint(2, (8,))),
            }
        )

        assert {"acc", "f1"} <= set(metrics.avg["a"].keys())
        assert {"acc", "f1"} <= set(metrics.avg["b"].keys())

    def test_accepts_mapping_payloads(self):
        metrics = MultiTaskMetrics()
        metrics.binary = GlobalMetrics([binary_accuracy()], preprocess=preprocess_binary, distributed=False)

        metrics.update(
            {
                "binary": {
                    "input": torch.tensor([10.0, -10.0], dtype=torch.float32),
                    "target": torch.tensor([1, 0], dtype=torch.long),
                }
            }
        )

        assert metrics.avg["binary"]["acc"] == pytest.approx(1.0)

    def test_rejects_unknown_task_updates(self):
        metrics = MultiTaskMetrics()
        metrics.score = lambda input, target: (input == target).float().mean()

        with pytest.raises(ValueError, match="Task missing not found"):
            metrics.update({"missing": (torch.tensor([1.0]), torch.tensor([1.0]))})

    def test_rejects_invalid_payload_types(self):
        metrics = MultiTaskMetrics()
        metrics.score = lambda input, target: (input == target).float().mean()

        with pytest.raises(ValueError, match="Mapping or Sequence"):
            metrics.update({"score": object()})

    def test_rejects_nested_multitask_children(self):
        metrics = MultiTaskMetrics()

        with pytest.raises(ValueError, match="Expected .* callable"):
            metrics.group = MultiTaskMetrics()

    def test_coerces_callable_to_metric_meter(self):
        metrics = MultiTaskMetrics()
        metrics.score = lambda input, target: (input == target).float().mean()

        assert isinstance(metrics["score"], MetricMeter)

        metrics.update({"score": (torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0]))})
        assert metrics.avg["score"]["score"] == pytest.approx(0.5)

    def test_keeps_explicit_metric_meter_output_name(self):
        meter = MetricMeter(lambda input, target: (input == target).float().mean())
        meter.output_name = "accuracy"
        metrics = MultiTaskMetrics()
        metrics.score = meter

        metrics.update({"score": (torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0]))})

        assert metrics["score"].output_name == "accuracy"
        assert metrics.avg["score"]["accuracy"] == pytest.approx(0.5)

    def test_uses_task_name_when_metric_name_is_unknown(self):
        class CallableMetric:
            def __call__(self, input, target):
                return (input == target).float().mean()

        metrics = MultiTaskMetrics()
        metrics.score = CallableMetric()

        metrics.update({"score": (torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0]))})

        assert metrics["score"].output_name == "score"
        assert metrics.avg["score"]["score"] == pytest.approx(0.5)

    def test_macro_averages_shared_scalar_metric_names(self):
        def shared_score(input, target):
            del target
            return input.float().mean()

        metrics = MultiTaskMetrics(aggregate="macro")
        metrics.a = shared_score
        metrics.b = MetricMeter(shared_score)

        metrics.update(
            {
                "a": (torch.ones(1), torch.zeros(1)),
                "b": (torch.zeros(3), torch.zeros(3)),
            }
        )

        assert metrics.avg["aggregate"]["shared_score"] == pytest.approx(0.5)

    def test_macro_averages_shared_tensor_metric_names(self):
        def shared_vector(input, target):
            del target
            return input.float().mean(dim=0)

        metrics = MultiTaskMetrics(aggregate="macro")
        metrics.a = StreamMetrics(vector=shared_vector, distributed=False)
        metrics.b = StreamMetrics(vector=shared_vector, distributed=False)

        metrics.update(
            {
                "a": (torch.tensor([[1.0, 3.0], [3.0, 5.0]]), torch.zeros(2, 2)),
                "b": (torch.tensor([[0.0, 2.0], [2.0, 4.0]]), torch.zeros(2, 2)),
            }
        )

        assert_close(metrics.avg["aggregate"]["vector"], torch.tensor([1.5, 3.5], dtype=torch.float64))

    def test_micro_averages_by_sample_count(self):
        def shared_score(input, target):
            del target
            return input.float().mean()

        metrics = MultiTaskMetrics(aggregate="micro")
        metrics.a = shared_score
        metrics.b = MetricMeter(shared_score)

        metrics.update(
            {
                "a": (torch.ones(1), torch.zeros(1)),
                "b": (torch.zeros(3), torch.zeros(3)),
            }
        )

        assert metrics.avg["aggregate"]["shared_score"] == pytest.approx(0.25)

    def test_weighted_averages_by_explicit_weight(self):
        def shared_score(input, target):
            del target
            return input.float().mean()

        metrics = MultiTaskMetrics(aggregate="weighted", aggregate_weights={"a": 1.0, "b": 3.0})
        metrics.a = shared_score
        metrics.b = MetricMeter(shared_score)

        metrics.update(
            {
                "a": (torch.ones(1), torch.zeros(1)),
                "b": (torch.zeros(3), torch.zeros(3)),
            }
        )

        assert metrics.avg["aggregate"]["shared_score"] == pytest.approx(0.25)

    def test_set_invalid_does_not_mutate(self):
        metrics = MultiTaskMetrics()
        assert list(metrics.keys()) == []

        with pytest.raises(ValueError):
            metrics.set("invalid", 1)

        assert list(metrics.keys()) == []


class TestDistributedMultiTaskMetrics:
    def test_macro_averages_matching_metrics(self):
        run_distributed(_distributed_multitask_worker, world_size=2)

    def test_micro_averages_matching_metrics(self):
        run_distributed(_distributed_multitask_micro_worker, world_size=2)
