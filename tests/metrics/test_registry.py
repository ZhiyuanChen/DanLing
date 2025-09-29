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

from danling.metrics.global_metrics import GlobalMetrics
from danling.metrics.registry import METRICS
from danling.metrics.stream_metrics import StreamMetrics


class TestMetricsRegistry:
    def test_build_is_case_insensitive(self):
        assert isinstance(METRICS.build("BINARY", distributed=False), GlobalMetrics)

    def test_build_respects_mode(self):
        assert isinstance(METRICS.build("BINARY", distributed=False), GlobalMetrics)
        assert isinstance(METRICS.build("binary", mode="stream", distributed=False), StreamMetrics)

    def test_maps_num_labels_to_num_classes_for_multiclass(self):
        metrics = METRICS.build("multiclass", num_labels=5, distributed=False)
        assert isinstance(metrics, GlobalMetrics)
        assert metrics.requirements["num_classes"] == 5

    def test_requires_num_classes_for_multiclass(self):
        with pytest.raises(ValueError, match="num_classes is required"):
            METRICS.build("multiclass", distributed=False)

    def test_requires_num_labels_for_multilabel(self):
        with pytest.raises(ValueError, match="num_labels is required"):
            METRICS.build("multilabel", distributed=False)

    def test_rejects_conflicting_multiclass_dimensions(self):
        with pytest.raises(ValueError, match="num_classes and num_labels must match"):
            METRICS.build("multiclass", num_classes=4, num_labels=5, distributed=False)

    def test_maps_num_labels_to_num_outputs_for_regression(self):
        metrics = METRICS.build("regression", num_labels=3, distributed=False)
        assert isinstance(metrics, GlobalMetrics)
        assert metrics.metrics["mse"].num_outputs == 3
        assert metrics.metrics["rmse"].num_outputs == 3

    def test_regression_defaults_to_one_output(self):
        metrics = METRICS.build("regression", distributed=False)
        assert isinstance(metrics, GlobalMetrics)
        assert metrics.metrics["mse"].num_outputs == 1
        assert metrics.metrics["rmse"].num_outputs == 1

    def test_rejects_conflicting_regression_dimensions(self):
        with pytest.raises(ValueError, match="num_outputs and num_labels must match"):
            METRICS.build("regression", num_outputs=2, num_labels=3, distributed=False)

    def test_accepts_numpy_integer_dimensions(self):
        np = pytest.importorskip("numpy")

        metrics = METRICS.build("multiclass", num_classes=np.int64(5), distributed=False)

        assert isinstance(metrics, GlobalMetrics)
        assert metrics.requirements["num_classes"] == 5

    @pytest.mark.parametrize(
        ("task", "kwargs", "error"),
        [
            ("multiclass", {"num_classes": 0}, "num_classes must be a positive integer"),
            ("multilabel", {"num_labels": 0}, "num_labels must be a positive integer"),
            ("regression", {"num_outputs": 0}, "num_outputs must be a positive integer"),
        ],
    )
    def test_rejects_non_positive_dimensions(self, task, kwargs, error):
        with pytest.raises(ValueError, match=error):
            METRICS.build(task, distributed=False, **kwargs)

    def test_build_raises_for_unknown_component(self):
        with pytest.raises(ValueError, match="Component missing is not registered."):
            METRICS.build("missing")
