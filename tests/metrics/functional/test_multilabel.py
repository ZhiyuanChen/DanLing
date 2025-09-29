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

from danling.metrics.functional.multilabel import (
    multilabel_accuracy,
    multilabel_auprc,
    multilabel_auroc,
    multilabel_balanced_accuracy,
    multilabel_f1_score,
    multilabel_fbeta_score,
    multilabel_hamming_loss,
    multilabel_iou,
    multilabel_jaccard_index,
    multilabel_precision,
    multilabel_recall,
    multilabel_specificity,
)
from danling.metrics.state import MetricState

ATOL = 1e-6
RTOL = 1e-5
NUM_LABELS = 3

PREDS = torch.tensor(
    [
        [0.90, 0.10, 0.80],
        [0.20, 0.70, 0.30],
        [0.60, 0.60, 0.40],
        [0.10, 0.20, 0.90],
        [0.80, 0.30, 0.20],
        [0.30, 0.80, 0.70],
    ]
)
TARGETS = torch.tensor(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 1],
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


def build_confmat_state(metric) -> MetricState:
    requirements = MetricState.collect_requirements((metric,))
    return MetricState.from_requirements(PREDS, TARGETS, requirements)


CONF_MAT_CASES: list[tuple[str, Callable, Callable]] = [
    (
        "acc",
        multilabel_accuracy(num_labels=NUM_LABELS),
        lambda p, t: tmfc.multilabel_accuracy(
            p, t, num_labels=NUM_LABELS, threshold=0.5, average="macro", ignore_index=-100
        ),
    ),
    (
        "precision",
        multilabel_precision(num_labels=NUM_LABELS),
        lambda p, t: tmfc.multilabel_precision(
            p, t, num_labels=NUM_LABELS, threshold=0.5, average="macro", ignore_index=-100
        ),
    ),
    (
        "recall",
        multilabel_recall(num_labels=NUM_LABELS),
        lambda p, t: tmfc.multilabel_recall(
            p, t, num_labels=NUM_LABELS, threshold=0.5, average="macro", ignore_index=-100
        ),
    ),
    (
        "specificity",
        multilabel_specificity(num_labels=NUM_LABELS),
        lambda p, t: tmfc.multilabel_specificity(
            p, t, num_labels=NUM_LABELS, threshold=0.5, average="macro", ignore_index=-100
        ),
    ),
    (
        "balanced_accuracy",
        multilabel_balanced_accuracy(num_labels=NUM_LABELS),
        lambda p, t: 0.5
        * (
            tmfc.multilabel_recall(p, t, num_labels=NUM_LABELS, threshold=0.5, average="macro", ignore_index=-100)
            + tmfc.multilabel_specificity(
                p, t, num_labels=NUM_LABELS, threshold=0.5, average="macro", ignore_index=-100
            )
        ),
    ),
    (
        "jaccard",
        multilabel_jaccard_index(num_labels=NUM_LABELS),
        lambda p, t: tmfc.multilabel_jaccard_index(
            p, t, num_labels=NUM_LABELS, threshold=0.5, average="macro", ignore_index=-100
        ),
    ),
    (
        "iou",
        multilabel_iou(num_labels=NUM_LABELS),
        lambda p, t: tmfc.multilabel_jaccard_index(
            p, t, num_labels=NUM_LABELS, threshold=0.5, average="macro", ignore_index=-100
        ),
    ),
    (
        "hamming_loss",
        multilabel_hamming_loss(num_labels=NUM_LABELS),
        lambda p, t: tmfc.multilabel_hamming_distance(
            p, t, num_labels=NUM_LABELS, threshold=0.5, average="macro", ignore_index=-100
        ),
    ),
    (
        "fbeta",
        multilabel_fbeta_score(num_labels=NUM_LABELS, beta=2.0),
        lambda p, t: tmfc.multilabel_fbeta_score(
            p, t, num_labels=NUM_LABELS, beta=2.0, threshold=0.5, average="macro", ignore_index=-100
        ),
    ),
    (
        "f1",
        multilabel_f1_score(num_labels=NUM_LABELS),
        lambda p, t: tmfc.multilabel_f1_score(
            p, t, num_labels=NUM_LABELS, threshold=0.5, average="macro", ignore_index=-100
        ),
    ),
]


@pytest.mark.parametrize("name,metric,expected_fn", CONF_MAT_CASES)
def test_multilabel_confmat_metrics_match_torchmetrics(name: str, metric, expected_fn: Callable) -> None:
    del name
    confmat_state = build_confmat_state(metric)
    fallback_state = MetricState(preds=PREDS, targets=TARGETS, confmat=None)
    expected = expected_fn(PREDS, TARGETS)

    assert_metric_close(metric(confmat_state), expected)
    assert_metric_close(metric(fallback_state), expected)


@pytest.mark.parametrize(
    "metric,expected_fn",
    [
        (
            multilabel_auroc(num_labels=NUM_LABELS),
            lambda p, t: tmfc.multilabel_auroc(p, t, num_labels=NUM_LABELS, average="macro", ignore_index=-100),
        ),
        (
            multilabel_auprc(num_labels=NUM_LABELS),
            lambda p, t: tmfc.multilabel_average_precision(
                p, t, num_labels=NUM_LABELS, average="macro", ignore_index=-100
            ),
        ),
    ],
)
def test_multilabel_preds_target_metrics_match_torchmetrics(metric, expected_fn: Callable) -> None:
    state = MetricState(preds=PREDS, targets=TARGETS)
    assert_metric_close(metric(state), expected_fn(PREDS, TARGETS))


@pytest.mark.parametrize("average", [None, "micro", "weighted"])
def test_multilabel_reduce_modes_are_supported(average) -> None:
    metric = multilabel_precision(num_labels=NUM_LABELS, average=average)
    state = build_confmat_state(metric)
    expected = tmfc.multilabel_precision(
        PREDS,
        TARGETS,
        num_labels=NUM_LABELS,
        threshold=0.5,
        average=average,
        ignore_index=-100,
    )
    assert_metric_close(metric(state), expected)


def test_multilabel_reduce_rejects_invalid_average() -> None:
    metric = multilabel_precision(num_labels=NUM_LABELS, average="invalid")
    state = build_confmat_state(metric)
    with pytest.raises(ValueError, match="Invalid average value"):
        metric(state)


@pytest.mark.parametrize(
    "metric",
    [
        multilabel_accuracy(num_labels=NUM_LABELS),
        multilabel_auroc(num_labels=NUM_LABELS),
        multilabel_auprc(num_labels=NUM_LABELS),
        multilabel_precision(num_labels=NUM_LABELS),
        multilabel_recall(num_labels=NUM_LABELS),
        multilabel_specificity(num_labels=NUM_LABELS),
        multilabel_balanced_accuracy(num_labels=NUM_LABELS),
        multilabel_jaccard_index(num_labels=NUM_LABELS),
        multilabel_iou(num_labels=NUM_LABELS),
        multilabel_hamming_loss(num_labels=NUM_LABELS),
        multilabel_fbeta_score(num_labels=NUM_LABELS, beta=2.0),
        multilabel_f1_score(num_labels=NUM_LABELS),
    ],
)
def test_multilabel_metrics_return_nan_on_empty_state(metric) -> None:
    empty_state = MetricState(preds=torch.tensor([]), targets=torch.tensor([]), confmat=None)
    assert torch.isnan(torch.as_tensor(metric(empty_state)))


def test_multilabel_alias_and_inheritance() -> None:
    assert isinstance(multilabel_f1_score(num_labels=NUM_LABELS), multilabel_fbeta_score)
    assert multilabel_iou(num_labels=NUM_LABELS).name == "iou"
