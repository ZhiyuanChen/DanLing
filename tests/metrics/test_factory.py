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


def test_exact_factories_accept_runtime_kwargs():
    assert isinstance(binary_metrics(distributed=False), GlobalMetrics)
    assert isinstance(multiclass_metrics(num_classes=4, distributed=False), GlobalMetrics)
    assert isinstance(multilabel_metrics(num_labels=4, distributed=False), GlobalMetrics)
    assert isinstance(regression_metrics(num_outputs=2, distributed=False), GlobalMetrics)


def test_factories_accept_preprocess_constructor_kwarg():
    global_metrics = binary_metrics(distributed=False, preprocess=preprocess_binary)
    assert isinstance(global_metrics, GlobalMetrics)
    global_metrics.update(torch.randn(8), torch.randint(2, (8,)))

    stream_metrics = binary_metrics(mode="stream", preprocess=preprocess_binary)
    assert isinstance(stream_metrics, StreamMetrics)
    stream_metrics.update(torch.randn(8), torch.randint(2, (8,)))


def test_stream_metrics_factories_cover_all_tasks():
    metrics = binary_metrics(mode="stream")
    metrics.update(torch.randn(8), torch.randint(2, (8,)))
    assert isinstance(metrics, StreamMetrics)
    assert {"auroc", "auprc", "acc", "f1", "mcc"} <= set(metrics.avg.keys())

    metrics = multiclass_metrics(num_classes=5, mode="stream")
    metrics.update(torch.randn(8, 5), torch.randint(5, (8,)))
    assert {"auroc", "auprc", "acc", "f1", "mcc"} <= set(metrics.avg.keys())

    metrics = multilabel_metrics(num_labels=5, mode="stream")
    metrics.update(torch.randn(8, 5), torch.randint(2, (8, 5)))
    assert {"auroc", "auprc", "acc", "f1", "mcc"} <= set(metrics.avg.keys())

    metrics = regression_metrics(num_outputs=2, mode="stream")
    metrics.update(torch.randn(8, 2), torch.randn(8, 2))
    assert {"pearson", "spearman", "r2", "mse", "rmse"} <= set(metrics.avg.keys())


def test_stream_factories_match_global_metric_names():
    assert set(binary_metrics(distributed=False).avg.keys()) == set(binary_metrics(mode="stream").avg.keys())
    assert set(multiclass_metrics(num_classes=5, distributed=False).avg.keys()) == set(
        multiclass_metrics(num_classes=5, mode="stream").avg.keys()
    )
    assert set(multilabel_metrics(num_labels=5, distributed=False).avg.keys()) == set(
        multilabel_metrics(num_labels=5, mode="stream").avg.keys()
    )
    assert set(regression_metrics(num_outputs=2, distributed=False).avg.keys()) == set(
        regression_metrics(num_outputs=2, mode="stream").avg.keys()
    )
