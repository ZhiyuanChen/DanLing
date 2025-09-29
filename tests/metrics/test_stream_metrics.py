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

import pytest
import torch
from torch.testing import assert_close

from danling.metrics.factory import regression_metrics
from danling.metrics.functional import (
    binary_accuracy,
    mse,
    multiclass_accuracy,
    multilabel_accuracy,
)
from danling.metrics.global_metrics import GlobalMetrics
from danling.metrics.preprocess import (
    preprocess_binary,
    preprocess_multiclass,
    preprocess_multilabel,
    preprocess_regression,
)
from danling.metrics.stream_metrics import MetricMeter, StreamMetrics

from .utils import ATOL, RTOL, process_group, require_nccl_cuda, run_distributed


def assert_metric_dict_close(actual, expected) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        actual_value = torch.as_tensor(actual[key])
        expected_value = torch.as_tensor(expected[key])
        if actual_value.device.type != "cpu":
            actual_value = actual_value.cpu()
        if expected_value.device.type != "cpu":
            expected_value = expected_value.cpu()
        assert_close(
            actual_value,
            expected_value,
            rtol=RTOL,
            atol=ATOL,
            check_dtype=False,
            equal_nan=True,
        )


CHUNKED_UPDATE_CASES = [
    pytest.param(
        binary_accuracy(),
        preprocess_binary,
        [
            (torch.tensor([0.10, 0.90, 0.80, 0.40]), torch.tensor([0, 1, 1, 1])),
            (torch.tensor([0.20, 0.70, 0.30]), torch.tensor([0, 0, 1])),
            (torch.tensor([0.55, 0.45]), torch.tensor([1, 0])),
        ],
        id="binary",
    ),
    pytest.param(
        multiclass_accuracy(num_classes=3, average="micro"),
        partial(preprocess_multiclass, num_classes=3),
        [
            (
                torch.tensor(
                    [
                        [3.0, 1.0, 0.0],
                        [0.1, 2.0, 0.2],
                        [0.2, 0.5, 2.0],
                    ]
                ),
                torch.tensor([0, 1, 1]),
            ),
            (
                torch.tensor(
                    [
                        [0.3, 2.0, 0.1],
                        [0.1, 0.2, 3.0],
                        [2.0, 0.1, 0.1],
                        [0.5, 0.6, 0.7],
                    ]
                ),
                torch.tensor([1, 2, 0, 0]),
            ),
        ],
        id="multiclass",
    ),
    pytest.param(
        multilabel_accuracy(num_labels=3, average="micro"),
        partial(preprocess_multilabel, num_labels=3),
        [
            (
                torch.tensor(
                    [
                        [0.90, 0.10, 0.80],
                        [0.20, 0.70, 0.30],
                    ]
                ),
                torch.tensor([[1, 0, 1], [0, 1, 0]]),
            ),
            (
                torch.tensor(
                    [
                        [0.60, 0.60, 0.40],
                        [0.10, 0.20, 0.90],
                        [0.80, 0.30, 0.20],
                    ]
                ),
                torch.tensor([[1, 0, 0], [0, 0, 1], [1, 0, 1]]),
            ),
        ],
        id="multilabel",
    ),
    pytest.param(
        mse(num_outputs=2),
        partial(preprocess_regression, num_outputs=2, ignore_nan=True),
        [
            (
                torch.tensor([[1.0, 2.0]]),
                torch.tensor([[float("nan"), float("nan")]]),
            ),
            (
                torch.tensor([[1.0, 2.0], [3.0, 5.0]]),
                torch.tensor([[1.5, 1.0], [2.0, 4.0]]),
            ),
            (
                torch.tensor([[2.0, 4.0], [4.0, 8.0], [8.0, 16.0]]),
                torch.tensor([[1.0, 1.0], [1.0, 1.0], [8.0, 10.0]]),
            ),
        ],
        id="regression",
    ),
]


def _distributed_stream_metrics_worker(rank: int, world_size: int):
    with process_group("gloo", rank, world_size):
        metrics = StreamMetrics(acc=binary_accuracy(), distributed=True, device="cpu")
        if rank == 0:
            logits = torch.tensor([10.0, 10.0], dtype=torch.float32)
            target = torch.tensor([1, 1], dtype=torch.long)
        else:
            logits = torch.tensor([-10.0, -10.0, -10.0, -10.0], dtype=torch.float32)
            target = torch.tensor([1, 1, 1, 1], dtype=torch.long)

        metrics.update(logits, target)

        expected_local = 1.0 if rank == 0 else 0.0
        assert float(torch.as_tensor(metrics.val["acc"]).cpu()) == pytest.approx(expected_local)
        assert metrics.count["acc"] == len(target)

        expected_synced = 2.0 / 6.0
        assert float(torch.as_tensor(metrics.bat["acc"]).cpu()) == pytest.approx(expected_synced)
        assert float(torch.as_tensor(metrics.avg["acc"]).cpu()) == pytest.approx(expected_synced)


def _distributed_stream_metrics_nccl_worker(rank: int, world_size: int):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    with process_group("nccl", rank, world_size):
        metrics = StreamMetrics(acc=binary_accuracy(), distributed=True, device=device)
        if rank == 0:
            logits = torch.tensor([10.0, 10.0], dtype=torch.float32, device=device)
            target = torch.tensor([1, 1], dtype=torch.long, device=device)
        else:
            logits = torch.tensor([-10.0, -10.0, -10.0, -10.0], dtype=torch.float32, device=device)
            target = torch.tensor([1, 1, 1, 1], dtype=torch.long, device=device)

        metrics.update(logits, target)

        expected_local = 1.0 if rank == 0 else 0.0
        assert metrics.val["acc"] == pytest.approx(expected_local)
        assert metrics.count["acc"] == len(target)

        expected_synced = 2.0 / 6.0
        assert metrics.bat["acc"] == pytest.approx(expected_synced)
        assert metrics.avg["acc"] == pytest.approx(expected_synced)


def _distributed_tensor_stream_metrics_empty_rank_worker(rank: int, world_size: int):
    def vector_metric(input: torch.Tensor, target: torch.Tensor):
        if input.numel() == 0:
            return torch.tensor(float("nan"))
        return (input - target).abs().mean(dim=0)

    with process_group("gloo", rank, world_size):
        metrics = StreamMetrics(
            vector=vector_metric,
            preprocess=lambda input, target: preprocess_regression(input, target, num_outputs=2, ignore_nan=True),
            distributed=True,
            device="cpu",
        )

        if rank == 0:
            input = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
            target = torch.tensor([[float("nan"), float("nan")]], dtype=torch.float32)
        else:
            input = torch.tensor([[2.0, 4.0], [4.0, 8.0]], dtype=torch.float32)
            target = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)

        metrics.update(input, target)

        expected = torch.tensor([2.0, 5.0], dtype=torch.float64)
        assert_close(metrics.bat["vector"], expected, rtol=RTOL, atol=ATOL)
        assert_close(metrics.avg["vector"], expected, rtol=RTOL, atol=ATOL)


class TestMetricMeter:
    def test_rejects_non_callable_metric(self):
        with pytest.raises(ValueError, match="Expected metric to be callable"):
            MetricMeter(1)  # type: ignore[arg-type]

    def test_uses_preprocessed_inputs(self):
        preprocess_called = {"count": 0}

        def preprocess(input: torch.Tensor, target: torch.Tensor):
            preprocess_called["count"] += 1
            return input + 1, target

        meter = MetricMeter(lambda input, target: (input == target).float().mean(), preprocess=preprocess)
        meter.update(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0]))
        assert preprocess_called["count"] == 1
        assert meter.count == 2

    def test_scalar_inputs_count_as_one(self):
        meter = MetricMeter(lambda input, target: float(input.item() == target.item()))
        meter.update(torch.tensor(1.0), torch.tensor(1.0))
        assert meter.count == 1
        assert meter.avg == pytest.approx(1.0)

    def test_repr_for_metricfunc(self):
        meter_metricfunc = MetricMeter(binary_accuracy())
        assert repr(meter_metricfunc) == "MetricMeter(acc)"

    def test_repr_for_function(self):
        def plain_metric(input: torch.Tensor, target: torch.Tensor):
            return (input == target).float().mean()

        meter_function = MetricMeter(plain_metric)
        assert repr(meter_function) == "MetricMeter(plain_metric)"

    def test_repr_for_partial(self):
        def plain_metric(input: torch.Tensor, target: torch.Tensor):
            return (input == target).float().mean()

        meter_partial = MetricMeter(partial(plain_metric))
        assert repr(meter_partial) == "MetricMeter(plain_metric)"


class TestStreamMetrics:
    @pytest.mark.parametrize("metric,preprocess,batches", CHUNKED_UPDATE_CASES)
    def test_chunked_updates(self, metric, preprocess, batches, device):
        stream = StreamMetrics(metric, preprocess=preprocess, distributed=False, device=device)
        global_metrics = GlobalMetrics(metric, preprocess=preprocess, distributed=False, device=device)

        for input, target in batches:
            input = input.to(device)
            target = target.to(device)
            stream.update(input, target)
            global_metrics.update(input, target)

            assert_metric_dict_close(stream.value(), global_metrics.value())

        assert_metric_dict_close(stream.average(), global_metrics.average())
        assert stream.count[next(iter(stream.keys()))] == global_metrics.count

    def test_reset_clears_state(self):
        metrics = StreamMetrics(binary_accuracy(), preprocess=preprocess_binary, distributed=False)

        metrics.update(torch.tensor([0.2, 0.8, 0.9]), torch.tensor([0, 1, 0]))
        assert metrics.count["acc"] == 3

        assert metrics.reset() is metrics
        assert metrics.count["acc"] == 0
        assert torch.isnan(torch.as_tensor(metrics.value()["acc"]))
        assert torch.isnan(torch.as_tensor(metrics.average()["acc"]))

    def test_reset_allows_reuse(self):
        metrics = StreamMetrics(binary_accuracy(), preprocess=preprocess_binary, distributed=False)
        metrics.update(torch.tensor([0.2, 0.8, 0.9]), torch.tensor([0, 1, 0]))
        metrics.reset()

        metrics.update(torch.tensor([0.2, 0.8]), torch.tensor([0, 1]))
        assert metrics.count["acc"] == 2
        assert metrics.value()["acc"] == pytest.approx(1.0)
        assert metrics.average()["acc"] == pytest.approx(1.0)

    def test_updates_callable_metrics_from_batch_inputs(self):
        calls = {"count": 0}

        def batch_metric(input: torch.Tensor, target: torch.Tensor):
            calls["count"] += 1
            assert input.shape == torch.Size([4])
            assert target.shape == torch.Size([4])
            return (input.round() == target).float().mean()

        metrics = StreamMetrics(sample=batch_metric)
        input = torch.tensor([0.1, 0.9, 0.2, 0.7])
        target = torch.tensor([0.0, 1.0, 1.0, 1.0])
        metrics.update(input, target)
        assert calls["count"] == 1
        assert_close(
            torch.as_tensor(metrics.val["sample"]),
            torch.tensor(0.75),
            rtol=RTOL,
            atol=ATOL,
            check_dtype=False,
        )

    def test_rejects_invalid_metric_inputs(self):
        with pytest.raises(ValueError, match="Expected metric to be callable"):
            StreamMetrics(1)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Expected metric to be callable or MetricMeter"):
            StreamMetrics(bad=1)  # type: ignore[arg-type]

    def test_repr_lists_metric_names(self):
        metrics = StreamMetrics(acc=lambda input, target: (input == target).float().mean())
        assert repr(metrics) == "StreamMetrics('acc',)"

    def test_accepts_metric_meter_instances(self):
        meter = MetricMeter(binary_accuracy())
        metrics = StreamMetrics(acc=meter, preprocess=preprocess_binary, device="cpu")
        metrics.update(torch.tensor([0.2, 0.8]), torch.tensor([0, 1]))

        assert metrics["acc"] is meter
        assert metrics.avg["acc"] == pytest.approx(1.0)

    def test_default_preprocess_accepts_column_vector_binary(self):
        metrics = StreamMetrics(acc=binary_accuracy())
        input = torch.tensor([[0.2], [0.8]], requires_grad=True)
        target = torch.tensor([0, 1])
        metrics.update(input, target)

        assert_close(torch.as_tensor(metrics.val["acc"]), torch.tensor(1.0), rtol=RTOL, atol=ATOL, check_dtype=False)
        assert metrics.count["acc"] == 2

    def test_value_preserves_tensor_metric_outputs(self):
        metrics = StreamMetrics(vector=lambda input, target: torch.stack([input.float().mean(), target.float().mean()]))
        metrics.update(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0]))

        assert_close(metrics.val["vector"], torch.tensor([0.5, 1.0], dtype=torch.float64))

    def test_average_preserves_tensor_metric_outputs(self):
        metrics = StreamMetrics(vector=lambda input, target: torch.stack([input.float().mean(), target.float().mean()]))
        metrics.update(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0]))

        assert_close(metrics.avg["vector"], torch.tensor([0.5, 1.0], dtype=torch.float64))

    def test_empty_updates_do_not_poison_average(self):
        metrics = regression_metrics(num_outputs=2, mode="stream", distributed=False)
        metrics.update(
            torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            torch.tensor([[float("nan"), float("nan")]], dtype=torch.float32),
        )

        assert metrics.count["mse"] == 0
        assert isinstance(metrics.val["mse"], torch.Tensor)
        assert isinstance(metrics.avg["mse"], torch.Tensor)
        assert metrics.val["mse"].shape == torch.Size([2])
        assert metrics.avg["mse"].shape == torch.Size([2])
        assert torch.isnan(metrics.val["mse"]).all()
        assert torch.isnan(metrics.avg["mse"]).all()

        metrics.update(
            torch.tensor([[2.0, 4.0], [4.0, 8.0]], dtype=torch.float32),
            torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
        )

        assert_close(metrics.avg["mse"], torch.tensor([5.0, 29.0], dtype=torch.float64), rtol=RTOL, atol=ATOL)
        assert_close(
            metrics.avg["rmse"],
            torch.tensor([5.0, 29.0], dtype=torch.float64).sqrt(),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_accepts_explicit_zero_n_for_empty_updates(self):
        metrics = regression_metrics(num_outputs=2, mode="stream", distributed=False)
        metrics.update(
            torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            torch.tensor([[float("nan"), float("nan")]], dtype=torch.float32),
            n=0,
        )

        assert metrics.count["mse"] == 0
        assert isinstance(metrics.val["mse"], torch.Tensor)
        assert metrics.val["mse"].shape == torch.Size([2])
        assert torch.isnan(metrics.val["mse"]).all()

    def test_ignored_only_batch(self):
        metrics = StreamMetrics(
            acc=binary_accuracy(ignore_index=-100),
            preprocess=partial(preprocess_binary, ignore_index=-100),
            distributed=False,
        )

        metrics.update(torch.tensor([0.1, 0.9]), torch.tensor([-100, -100]))
        assert metrics.count["acc"] == 0
        assert torch.isnan(torch.as_tensor(metrics.val["acc"]))
        assert torch.isnan(torch.as_tensor(metrics.avg["acc"]))

        metrics.update(torch.tensor([0.1, 0.9]), torch.tensor([0, 1]))
        assert metrics.count["acc"] == 2
        assert metrics.val["acc"] == pytest.approx(1.0)
        assert metrics.avg["acc"] == pytest.approx(1.0)

    def test_distributed_sync_aggregates_across_ranks(self):
        run_distributed(_distributed_stream_metrics_worker, world_size=2)

    def test_tensor_sync_handles_empty_rank(self):
        run_distributed(_distributed_tensor_stream_metrics_empty_rank_worker, world_size=2)

    def test_nccl_sync_smoke(self):
        require_nccl_cuda(world_size=2)
        run_distributed(_distributed_stream_metrics_nccl_worker, world_size=2)
