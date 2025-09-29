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
import torchmetrics.functional as tmf
from torch.testing import assert_close

from danling.metrics.functional.regression import mae, mse, pearson, r2_score, rmse, spearman
from danling.metrics.state import MetricState

ATOL = 1e-6
RTOL = 1e-5

PREDS = torch.tensor(
    [
        [0.10, 1.20],
        [0.25, 1.35],
        [0.40, 1.60],
        [0.55, 1.85],
        [0.70, 2.00],
        [0.85, 2.25],
        [1.00, 2.50],
        [1.15, 2.70],
    ]
)
TARGETS = torch.tensor(
    [
        [0.20, 1.10],
        [0.35, 1.30],
        [0.50, 1.55],
        [0.65, 1.70],
        [0.80, 1.95],
        [0.95, 2.10],
        [1.10, 2.40],
        [1.25, 2.60],
    ]
)


def assert_metric_close(actual, expected) -> None:
    assert_close(
        torch.as_tensor(actual),
        torch.as_tensor(expected),
        rtol=RTOL,
        atol=ATOL,
        check_dtype=False,
    )


def test_regression_metrics_match_torchmetrics() -> None:
    state = MetricState(preds=PREDS, targets=TARGETS)

    expected_pearson = tmf.pearson_corrcoef(PREDS, TARGETS).mean()
    expected_spearman = tmf.spearman_corrcoef(PREDS, TARGETS).mean()
    expected_r2 = tmf.r2_score(PREDS, TARGETS)
    expected_mse = tmf.mean_squared_error(PREDS, TARGETS, squared=True, num_outputs=2)
    expected_rmse = tmf.mean_squared_error(PREDS, TARGETS, squared=False, num_outputs=2)
    expected_mae = tmf.mean_absolute_error(PREDS, TARGETS, num_outputs=2)

    assert_metric_close(pearson()(state), expected_pearson)
    assert_metric_close(spearman()(state), expected_spearman)
    assert_metric_close(r2_score()(state), expected_r2)
    assert_metric_close(mse(num_outputs=2)(state), expected_mse)
    assert_metric_close(rmse(num_outputs=2)(state), expected_rmse)
    assert_metric_close(mae(num_outputs=2)(state), expected_mae)


def test_regression_multioutput_modes_for_correlations() -> None:
    state = MetricState(preds=PREDS, targets=TARGETS)

    pearson_raw = tmf.pearson_corrcoef(PREDS, TARGETS)
    spearman_raw = tmf.spearman_corrcoef(PREDS, TARGETS)

    assert_metric_close(pearson(multioutput="raw_values")(state), pearson_raw)
    assert_metric_close(pearson(multioutput="uniform_average")(state), pearson_raw.mean())
    assert_metric_close(
        pearson(multioutput="variance_weighted")(state),
        pearson_raw.mul(1 / pearson_raw.var(dim=0)).sum(dim=0),
    )

    assert_metric_close(spearman(multioutput="raw_values")(state), spearman_raw)
    assert_metric_close(spearman(multioutput="uniform_average")(state), spearman_raw.mean())
    assert_metric_close(
        spearman(multioutput="variance_weighted")(state),
        spearman_raw.mul(1 / spearman_raw.var(dim=0)).sum(dim=0),
    )


def test_regression_multioutput_rejects_invalid_mode() -> None:
    state = MetricState(preds=PREDS, targets=TARGETS)
    with pytest.raises(ValueError, match="Invalid multioutput value"):
        pearson(multioutput="invalid")(state)
    with pytest.raises(ValueError, match="Invalid multioutput value"):
        spearman(multioutput="invalid")(state)


@pytest.mark.parametrize(
    "metric",
    [
        pearson(),
        spearman(),
        r2_score(),
        mse(num_outputs=2),
        rmse(num_outputs=2),
        mae(num_outputs=2),
    ],
)
def test_regression_metrics_return_nan_on_empty_state(metric) -> None:
    empty = MetricState(preds=torch.tensor([]), targets=torch.tensor([]))
    assert torch.isnan(torch.as_tensor(metric(empty)))
