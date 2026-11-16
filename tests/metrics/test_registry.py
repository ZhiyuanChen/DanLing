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

import pytest
import torch

from danling.metrics.global_metrics import GlobalMetrics
from danling.metrics.registry import METRICS
from danling.metrics.stream_metrics import StreamMetrics


class TestMetricsRegistry:
    @pytest.mark.parametrize(
        "specification,options,expected",
        [
            ("BINARY", {}, GlobalMetrics),
            ("binary", {"mode": "stream"}, StreamMetrics),
            ({"type": "MULTICLASS", "num_classes": 3, "mode": "global"}, {"mode": "stream"}, StreamMetrics),
            (None, {"type": "binary"}, GlobalMetrics),
        ],
        ids=["case", "mode", "mapping", "keyword"],
    )
    def test_build(self, specification, options, expected):
        original = dict(specification) if isinstance(specification, dict) else specification
        args = () if specification is None else (specification,)
        assert isinstance(METRICS.build(*args, distributed=False, **options), expected)
        assert specification == original

    def test_metric_argument(self):
        from danling.metrics.functional import binary_accuracy

        metrics = METRICS.build("binary", binary_accuracy(), distributed=False)
        metrics.update(torch.tensor([0.1, 0.9]), torch.tensor([0, 1]))
        result = metrics.average()
        assert set(result) == {"acc"}
        assert result["acc"] == pytest.approx(1.0)

    def test_class_dimension_alias(self):
        metrics = METRICS.build("multiclass", num_labels=5, distributed=False)
        metrics.update(torch.tensor([[10.0, 0, 0, 0, 0], [0, 10.0, 0, 0, 0]]), torch.tensor([0, 1]))
        assert metrics.average()["acc"] == pytest.approx(1.0)

    @pytest.mark.parametrize("options,width", [({}, 1), ({"num_labels": 3}, 3)])
    def test_output_dimension(self, options, width):
        metrics = METRICS.build("regression", distributed=False, **options)
        metrics.update(torch.zeros(2, width), torch.ones(2, width))
        result = metrics.average()
        expected = torch.ones(width) if width > 1 else torch.tensor(1.0)
        for name in ("mse", "rmse"):
            torch.testing.assert_close(torch.as_tensor(result[name]).float(), expected)

    @pytest.mark.parametrize(
        "task,options,error",
        [
            ("multiclass", {}, "num_classes is required"),
            ("multilabel", {}, "num_labels is required"),
            ("multiclass", {"num_classes": 4, "num_labels": 5}, "num_classes and num_labels must match"),
            ("regression", {"num_outputs": 2, "num_labels": 3}, "num_outputs and num_labels must match"),
            ("multiclass", {"num_classes": 0}, "num_classes must be a positive integer"),
            ("multilabel", {"num_labels": 0}, "num_labels must be a positive integer"),
            ("regression", {"num_outputs": 0}, "num_outputs must be a positive integer"),
        ],
    )
    def test_invalid_dimensions(self, task, options, error):
        with pytest.raises(ValueError, match=error):
            METRICS.build(task, distributed=False, **options)

    def test_numpy_dimension(self):
        np = pytest.importorskip("numpy")
        metrics = METRICS.build("multiclass", num_classes=np.int64(5), distributed=False)
        metrics.update(torch.tensor([[10.0, 0, 0, 0, 0]]), torch.tensor([0]))
        assert metrics.average()["acc"] == pytest.approx(1.0)

    def test_unknown(self):
        with pytest.raises(ValueError, match="Component missing is not registered."):
            METRICS.build("missing")
