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
from types import SimpleNamespace

import pytest
import torch

from danling.metrics.factory import (
    binary_metrics,
    multiclass_metrics,
    multilabel_metrics,
    regression_metrics,
)
from danling.metrics.global_metrics import GlobalMetrics
from danling.metrics.preprocess import preprocess_binary
from danling.metrics.stream_metrics import StreamMetrics


@pytest.fixture(
    params=[
        (binary_metrics, {}, (8,), (8,)),
        (multiclass_metrics, {"num_classes": 5}, (8, 5), (8,)),
        (multilabel_metrics, {"num_labels": 5}, (8, 5), (8, 5)),
        (regression_metrics, {"num_outputs": 1}, (8,), (8,)),
        (regression_metrics, {"num_outputs": 2}, (8, 2), (8, 2)),
    ],
    ids=["binary", "multiclass", "multilabel", "regression", "multioutput"],
)
def factory_case(request):
    factory, options, input_shape, target_shape = request.param
    regression = factory is regression_metrics
    targets = torch.randn(target_shape) if regression else torch.randint(options.get("num_classes", 2), target_shape)
    return SimpleNamespace(
        build=partial(factory, **options, distributed=False),
        inputs=torch.randn(input_shape),
        targets=targets,
        names={"pearson", "spearman", "r2", "mse", "rmse"} if regression else {"auroc", "auprc", "acc", "f1", "mcc"},
    )


class TestMetricFactories:
    @pytest.mark.parametrize("mode,expected", [(None, GlobalMetrics), ("stream", StreamMetrics)])
    def test_mode(self, factory_case, mode, expected):
        metrics = factory_case.build(**({} if mode is None else {"mode": mode}))
        assert isinstance(metrics, expected)

    def test_default_metrics(self, factory_case):
        metrics = factory_case.build(mode="stream")
        metrics.update(factory_case.inputs, factory_case.targets)
        assert factory_case.names <= metrics.avg.keys()

    @pytest.mark.parametrize("mode", ["global", "stream"])
    def test_preprocess_option(self, mode):
        metrics = binary_metrics(mode=mode, distributed=False, preprocess=preprocess_binary)
        metrics.update(torch.tensor([-10.0, 10.0]), torch.tensor([0, 1]))
        assert metrics.avg["acc"] == pytest.approx(1.0)

    def test_multioutput(self):
        metrics = regression_metrics(num_outputs=2, mode="stream", distributed=False)
        metrics.update(torch.randn(8, 2), torch.randn(8, 2))
        assert isinstance(metrics.avg["pearson"], float)
        assert isinstance(metrics.avg["spearman"], float)
        assert isinstance(metrics.avg["r2"], float)
        assert metrics.avg["mse"].shape == torch.Size([2])
        assert metrics.avg["rmse"].shape == torch.Size([2])
