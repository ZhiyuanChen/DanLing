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


class TestMetricFactories:
    def test_global_is_default(self):
        assert isinstance(binary_metrics(distributed=False), GlobalMetrics)
        assert isinstance(multiclass_metrics(num_classes=4, distributed=False), GlobalMetrics)
        assert isinstance(multilabel_metrics(num_labels=4, distributed=False), GlobalMetrics)
        assert isinstance(regression_metrics(num_outputs=2, distributed=False), GlobalMetrics)

    def test_stream_respects_explicit_mode(self):
        assert isinstance(binary_metrics(mode="stream", distributed=False), StreamMetrics)
        assert isinstance(multiclass_metrics(num_classes=4, mode="stream", distributed=False), StreamMetrics)
        assert isinstance(multilabel_metrics(num_labels=4, mode="stream", distributed=False), StreamMetrics)
        assert isinstance(regression_metrics(num_outputs=1, mode="stream", distributed=False), StreamMetrics)
        assert isinstance(regression_metrics(num_outputs=2, mode="stream", distributed=False), StreamMetrics)

    def test_forward_preprocess_constructor_kwarg(self):
        global_metrics = binary_metrics(distributed=False, preprocess=preprocess_binary)
        assert isinstance(global_metrics, GlobalMetrics)
        global_metrics.update(torch.randn(8), torch.randint(2, (8,)))

        stream_metrics = binary_metrics(mode="stream", preprocess=preprocess_binary)
        assert isinstance(stream_metrics, StreamMetrics)
        stream_metrics.update(torch.randn(8), torch.randint(2, (8,)))

    def test_stream_builds_default_metrics_for_all_tasks(self):
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

        metrics = regression_metrics(num_outputs=1, mode="stream")
        metrics.update(torch.randn(8), torch.randn(8))
        assert {"pearson", "spearman", "r2", "mse", "rmse"} <= set(metrics.avg.keys())

    def test_stream_regression_preserves_multioutput(self):
        metrics = regression_metrics(num_outputs=2, mode="stream", distributed=False)
        metrics.update(torch.randn(8, 2), torch.randn(8, 2))
        assert isinstance(metrics.avg["pearson"], float)
        assert isinstance(metrics.avg["spearman"], float)
        assert isinstance(metrics.avg["r2"], float)
        assert metrics.avg["mse"].shape == torch.Size([2])
        assert metrics.avg["rmse"].shape == torch.Size([2])
