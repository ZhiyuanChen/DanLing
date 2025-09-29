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

import inspect
from functools import partial
from math import isnan

import pytest
import torch
from torch.testing import assert_close

from danling.metrics.average_meter import AverageMeter
from danling.metrics.functional import binary_accuracy, binary_auroc, binary_f1
from danling.metrics.global_metrics import GlobalMetrics
from danling.metrics.preprocess import preprocess_binary
from danling.metrics.state import MetricState
from danling.metrics.stream_metrics import MetricMeter, StreamMetrics
from danling.tensors import NestedTensor

from .helpers import ATOL, RTOL


def test_global_and_stream_metrics_have_identical_constructor_signature():
    assert inspect.signature(GlobalMetrics.__init__) == inspect.signature(StreamMetrics.__init__)


def test_constructor_metric_resolution_rules_are_shared():
    def stream_a(input: torch.Tensor, target: torch.Tensor):
        return (input == target).float().mean()

    def stream_b(input: torch.Tensor, target: torch.Tensor):
        return (input == target).float().mean() * 0.5

    def stream_c(input: torch.Tensor, target: torch.Tensor):
        return (input == target).float().mean() * 0.25

    stream_a.__name__ = "shared"
    stream_b.__name__ = "shared"
    stream_c.__name__ = "shared"

    stream_metrics = StreamMetrics([stream_a, stream_b], shared=stream_c)
    assert list(stream_metrics.keys()) == ["shared"]
    assert stream_metrics["shared"].metric is stream_c

    global_a = binary_accuracy(name="shared")
    global_b = binary_f1(name="shared")
    global_c = binary_auroc(name="shared")

    global_metrics = GlobalMetrics(
        [global_a, global_b],
        preprocess=preprocess_binary,
        distributed=False,
        shared=global_c,
    )
    assert list(global_metrics.metrics.keys()) == ["shared"]
    assert global_metrics.metrics["shared"] is global_c


def test_stream_metrics_compute_samplewise_then_average():
    def sample_metric(input: torch.Tensor, target: torch.Tensor):
        assert input.shape[0] == 1
        assert target.shape[0] == 1
        return (input.round() == target).float().mean()

    metrics = StreamMetrics(sample=sample_metric)
    input = torch.tensor([0.1, 0.9, 0.2, 0.7])
    target = torch.tensor([0.0, 1.0, 1.0, 1.0])
    metrics.update(input, target)
    assert_close(torch.as_tensor(metrics.val["sample"]), torch.tensor(0.75), rtol=RTOL, atol=ATOL, check_dtype=False)


def test_metric_meter_rejects_non_callable_metric():
    with pytest.raises(ValueError, match="Expected metric to be callable"):
        MetricMeter(1)  # type: ignore[arg-type]


def test_metric_meter_update_with_preprocess_and_scalar_state_n_inference():
    preprocess_called = {"count": 0}

    def preprocess(input: torch.Tensor, target: torch.Tensor):
        preprocess_called["count"] += 1
        return input + 1, target

    meter = MetricMeter(lambda input, target: (input == target).float().mean(), preprocess=preprocess)
    meter.update(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0]))
    assert preprocess_called["count"] == 1
    assert meter.count == 2

    scalar_state = MetricState(preds=torch.tensor(1.0), targets=torch.tensor(1.0))
    meter._update_state(scalar_state)
    assert meter.count == 3


def test_metric_meter_update_state_validates_n():
    meter = MetricMeter(lambda input, target: (input == target).float().mean())
    state = MetricState(preds=torch.tensor([1.0]), targets=torch.tensor([1.0]))
    with pytest.raises(ValueError, match="n must be a number"):
        meter._update_state(state, n="1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n must be a positive integer"):
        meter._update_state(state, n=0)


def test_metric_meter_samplewise_fallback_to_batch_metric():
    def metric(input: torch.Tensor, target: torch.Tensor):
        if input.shape[0] == 1:
            raise RuntimeError("force samplewise fallback")
        return (input.round() == target).float().mean()

    meter = MetricMeter(metric)
    meter.update(torch.tensor([0.1, 0.9, 0.2, 0.7]), torch.tensor([0.0, 1.0, 1.0, 1.0]))
    assert_close(torch.as_tensor(meter.val), torch.tensor(0.75), rtol=RTOL, atol=ATOL, check_dtype=False)


def test_metric_meter_iter_sample_pairs_and_scalar_conversion():
    assert MetricMeter._iter_sample_pairs(torch.tensor(1.0), torch.tensor([1.0])) is None
    assert MetricMeter._iter_sample_pairs(torch.tensor([1.0, 2.0]), torch.tensor([1.0])) is None

    input_nt = NestedTensor([torch.tensor([1.0]), torch.tensor([2.0])])
    target_nt = NestedTensor([torch.tensor([1.0])])
    assert MetricMeter._iter_sample_pairs(input_nt, target_nt) is None
    assert MetricMeter._iter_sample_pairs([1, 2], [1]) is None

    pairs = MetricMeter._iter_sample_pairs(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
    assert pairs is not None
    assert len(list(pairs)) == 2

    assert isnan(MetricMeter._to_scalar(torch.tensor([])))
    assert MetricMeter._to_scalar(torch.tensor([2.0])) == 2.0
    assert MetricMeter._to_scalar(torch.tensor([2.0, 4.0])) == 3.0
    assert MetricMeter._to_scalar(5) == 5.0


def test_metric_meter_repr_for_metricfunc_partial_and_function():
    meter_metricfunc = MetricMeter(binary_accuracy())
    assert repr(meter_metricfunc) == "MetricMeter(acc)"

    def plain_metric(input: torch.Tensor, target: torch.Tensor):
        return (input == target).float().mean()

    meter_function = MetricMeter(plain_metric)
    assert repr(meter_function) == "MetricMeter(plain_metric)"

    meter_partial = MetricMeter(partial(plain_metric))
    assert repr(meter_partial) == "MetricMeter(plain_metric)"


def test_metric_meter_build_state_and_call_metric_paths():
    metric_func_meter = MetricMeter(binary_accuracy())
    metric_state = metric_func_meter._build_state(torch.tensor([0.9, 0.1]), torch.tensor([1, 0]))
    assert metric_state.confmat is not None
    assert isinstance(metric_func_meter._call_metric(metric_state), (torch.Tensor, float))

    callable_meter = MetricMeter(lambda input, target: (input == target).float().mean())
    callable_state = callable_meter._build_state(torch.tensor([1, 0]), torch.tensor([1, 1]))
    assert callable_state.confmat is None
    assert isinstance(callable_meter._call_metric(callable_state), (torch.Tensor, float))


def test_stream_metrics_constructor_validation_and_repr():
    with pytest.raises(ValueError, match="Expected metric to be callable"):
        StreamMetrics(1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Expected metric to be callable or MetricMeter"):
        StreamMetrics(bad=1)  # type: ignore[arg-type]

    metrics = StreamMetrics(acc=lambda input, target: (input == target).float().mean())
    assert repr(metrics) == "StreamMetrics('acc',)"


def test_stream_metrics_coerce_metric_paths():
    metrics = StreamMetrics(acc=lambda input, target: (input == target).float().mean(), device="cpu")

    raw_meter = MetricMeter(binary_accuracy(), preprocess=preprocess_binary)
    coerced = metrics._coerce_metric(raw_meter)
    assert coerced is raw_meter
    assert coerced.preprocess is None
    assert coerced.device == torch.device("cpu")
    assert coerced._requirements is not None

    callable_meter = metrics._coerce_metric(lambda input, target: (input == target).float().mean())
    assert isinstance(callable_meter, MetricMeter)
    assert callable_meter.preprocess is None
    assert callable_meter.device == torch.device("cpu")

    with pytest.raises(ValueError, match="Expected meter to be an instance of MetricMeter"):
        metrics._coerce_metric(1)  # type: ignore[arg-type]


def test_stream_metrics_collect_requirements_and_build_state():
    metrics = StreamMetrics(acc=binary_accuracy(), fn=lambda input, target: (input == target).float().mean())
    requirements = metrics._collect_requirements()
    assert requirements is not None
    assert requirements["confmat"] is True
    assert requirements["task"] == "binary"

    state = metrics._build_state(torch.tensor([0.9, 0.1]), torch.tensor([1, 0]))
    assert state.confmat is not None

    callable_only_metrics = StreamMetrics(fn=lambda input, target: (input == target).float().mean())
    assert callable_only_metrics._collect_requirements() is None


def test_stream_metrics_update_squeeze_and_detach_for_metric_meter():
    metrics = StreamMetrics(acc=binary_accuracy())
    input = torch.tensor([[0.2], [0.8]], requires_grad=True)
    target = torch.tensor([0, 1])
    metrics.update(input, target)

    assert_close(torch.as_tensor(metrics.val["acc"]), torch.tensor(1.0), rtol=RTOL, atol=ATOL, check_dtype=False)
    assert metrics.count["acc"] == 2


class _NonMetricMeter(AverageMeter):
    def __init__(self, metric, *, preprocess=None, device=None):
        super().__init__(device=device)
        self.metric = metric
        self.preprocess = preprocess
        self.calls = []

    def update(self, input, target, *, n=None):  # type: ignore[override]
        self.calls.append((input, target, n))
        super().update(0.0, n=n or 1)


class _NonMetricStreamMetrics(StreamMetrics):
    meter_cls = _NonMetricMeter


def test_stream_metrics_update_non_metric_meter_branch():
    metrics = _NonMetricStreamMetrics(
        score=lambda input, target: 0.0,
        preprocess=lambda input, target: (input, target),
    )
    input = torch.tensor([[1.0], [0.0]], requires_grad=True)
    target = torch.tensor([1.0, 0.0])
    metrics.update(input, target, n=2)

    meter = metrics["score"]
    call_input, call_target, call_n = meter.calls[0]
    assert call_n == 2
    assert isinstance(call_input, torch.Tensor)
    assert isinstance(call_target, torch.Tensor)
    assert call_input.shape == torch.Size([2])
    assert call_input.requires_grad is False
    assert call_target.requires_grad is False
