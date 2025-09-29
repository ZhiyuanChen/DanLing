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

from collections.abc import Callable

import pytest
import torch
from torch.testing import assert_close
from torchmetrics.functional import classification as tmfc

from danling.metrics.functional.binary import (
    binary_accuracy,
    binary_auprc,
    binary_auroc,
    binary_balanced_accuracy,
    binary_f1,
    binary_fbeta,
    binary_hamming_loss,
    binary_iou,
    binary_jaccard_index,
    binary_precision,
    binary_recall,
    binary_specificity,
)
from danling.metrics.state import MetricState

ATOL = 1e-6
RTOL = 1e-5

PREDS = torch.tensor([0.10, 0.90, 0.85, 0.20, 0.40, 0.60, 0.95, 0.05])
TARGETS = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0])


def assert_metric_close(actual, expected) -> None:
    assert_close(
        torch.as_tensor(actual),
        torch.as_tensor(expected),
        rtol=RTOL,
        atol=ATOL,
        check_dtype=False,
    )


def build_confmat_state(metric) -> MetricState:
    requirements = MetricState.collect_requirements((metric,))
    return MetricState.from_requirements(PREDS, TARGETS, requirements)


CONF_MAT_CASES: list[tuple[str, Callable, Callable]] = [
    ("acc", binary_accuracy(), lambda p, t: tmfc.binary_accuracy(p, t, threshold=0.5, ignore_index=-100)),
    ("precision", binary_precision(), lambda p, t: tmfc.binary_precision(p, t, threshold=0.5, ignore_index=-100)),
    ("recall", binary_recall(), lambda p, t: tmfc.binary_recall(p, t, threshold=0.5, ignore_index=-100)),
    (
        "specificity",
        binary_specificity(),
        lambda p, t: tmfc.binary_specificity(p, t, threshold=0.5, ignore_index=-100),
    ),
    (
        "balanced_accuracy",
        binary_balanced_accuracy(),
        lambda p, t: 0.5
        * (
            tmfc.binary_recall(p, t, threshold=0.5, ignore_index=-100)
            + tmfc.binary_specificity(p, t, threshold=0.5, ignore_index=-100)
        ),
    ),
    ("jaccard", binary_jaccard_index(), lambda p, t: tmfc.binary_jaccard_index(p, t, threshold=0.5, ignore_index=-100)),
    ("iou", binary_iou(), lambda p, t: tmfc.binary_jaccard_index(p, t, threshold=0.5, ignore_index=-100)),
    (
        "hamming_loss",
        binary_hamming_loss(),
        lambda p, t: tmfc.binary_hamming_distance(p, t, threshold=0.5, ignore_index=-100),
    ),
    (
        "fbeta",
        binary_fbeta(beta=2.0),
        lambda p, t: tmfc.binary_fbeta_score(p, t, beta=2.0, threshold=0.5, ignore_index=-100),
    ),
    ("f1", binary_f1(), lambda p, t: tmfc.binary_f1_score(p, t, threshold=0.5, ignore_index=-100)),
]


@pytest.mark.parametrize("name,metric,expected_fn", CONF_MAT_CASES)
def test_binary_confmat_metrics_match_torchmetrics(name: str, metric, expected_fn: Callable) -> None:
    del name
    confmat_state = build_confmat_state(metric)
    fallback_state = MetricState(preds=PREDS, targets=TARGETS, confmat=None)
    expected = expected_fn(PREDS, TARGETS)

    assert_metric_close(metric(confmat_state), expected)
    assert_metric_close(metric(fallback_state), expected)


@pytest.mark.parametrize(
    "metric,expected_fn",
    [
        (binary_auroc(), lambda p, t: tmfc.binary_auroc(p, t, ignore_index=-100)),
        (binary_auprc(), lambda p, t: tmfc.binary_average_precision(p, t, ignore_index=-100)),
    ],
)
def test_binary_preds_target_metrics_match_torchmetrics(metric, expected_fn: Callable) -> None:
    state = MetricState(preds=PREDS, targets=TARGETS)
    assert_metric_close(metric(state), expected_fn(PREDS, TARGETS))


@pytest.mark.parametrize(
    "metric",
    [
        binary_accuracy(),
        binary_precision(),
        binary_recall(),
        binary_specificity(),
        binary_balanced_accuracy(),
        binary_jaccard_index(),
        binary_iou(),
        binary_hamming_loss(),
        binary_fbeta(beta=2.0),
        binary_f1(),
        binary_auroc(),
        binary_auprc(),
    ],
)
def test_binary_metrics_return_nan_on_empty_state(metric) -> None:
    empty_state = MetricState(preds=torch.tensor([]), targets=torch.tensor([]), confmat=None)
    assert torch.isnan(torch.as_tensor(metric(empty_state)))


def test_binary_metrics_return_nan_on_zero_confmat_totals() -> None:
    confmat = torch.zeros((2, 2))
    state = MetricState(preds=torch.tensor([]), targets=torch.tensor([]), confmat=confmat)
    assert torch.isnan(torch.as_tensor(binary_accuracy()(state)))
    assert torch.isnan(torch.as_tensor(binary_hamming_loss()(state)))


def test_binary_alias_and_inheritance() -> None:
    assert isinstance(binary_f1(), binary_fbeta)
    assert binary_iou().name == "iou"
