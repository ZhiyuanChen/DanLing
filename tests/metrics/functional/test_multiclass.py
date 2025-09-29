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

from danling.metrics.functional.multiclass import (
    multiclass_accuracy,
    multiclass_auprc,
    multiclass_auroc,
    multiclass_balanced_accuracy,
    multiclass_f1_score,
    multiclass_fbeta_score,
    multiclass_hamming_loss,
    multiclass_iou,
    multiclass_jaccard_index,
    multiclass_precision,
    multiclass_recall,
    multiclass_specificity,
)
from danling.metrics.state import MetricState

ATOL = 1e-6
RTOL = 1e-5
NUM_CLASSES = 3

PREDS = torch.tensor(
    [
        [0.90, 0.05, 0.05],
        [0.10, 0.80, 0.10],
        [0.20, 0.20, 0.60],
        [0.10, 0.70, 0.20],
        [0.60, 0.20, 0.20],
        [0.20, 0.30, 0.50],
        [0.15, 0.70, 0.15],
        [0.25, 0.15, 0.60],
    ]
)
TARGETS = torch.tensor([0, 1, 2, 1, 0, 2, 1, 2])


def assert_metric_close(actual, expected) -> None:
    assert_close(
        torch.as_tensor(actual),
        torch.as_tensor(expected),
        rtol=RTOL,
        atol=ATOL,
        check_dtype=False,
    )


def build_confmat_state(metric, device) -> MetricState:
    requirements = MetricState.collect_requirements((metric,))
    return MetricState.from_requirements(PREDS.to(device), TARGETS.to(device), requirements)


CONF_MAT_CASES: list[tuple[str, Callable, Callable]] = [
    (
        "acc",
        multiclass_accuracy(num_classes=NUM_CLASSES),
        lambda p, t: tmfc.multiclass_accuracy(
            p, t, num_classes=NUM_CLASSES, average="macro", top_k=1, ignore_index=-100
        ),
    ),
    (
        "precision",
        multiclass_precision(num_classes=NUM_CLASSES),
        lambda p, t: tmfc.multiclass_precision(
            p, t, num_classes=NUM_CLASSES, average="macro", top_k=1, ignore_index=-100
        ),
    ),
    (
        "recall",
        multiclass_recall(num_classes=NUM_CLASSES),
        lambda p, t: tmfc.multiclass_recall(p, t, num_classes=NUM_CLASSES, average="macro", top_k=1, ignore_index=-100),
    ),
    (
        "specificity",
        multiclass_specificity(num_classes=NUM_CLASSES),
        lambda p, t: tmfc.multiclass_specificity(
            p, t, num_classes=NUM_CLASSES, average="macro", top_k=1, ignore_index=-100
        ),
    ),
    (
        "balanced_accuracy",
        multiclass_balanced_accuracy(num_classes=NUM_CLASSES),
        lambda p, t: tmfc.multiclass_recall(p, t, num_classes=NUM_CLASSES, average="macro", top_k=1, ignore_index=-100),
    ),
    (
        "jaccard",
        multiclass_jaccard_index(num_classes=NUM_CLASSES),
        lambda p, t: tmfc.multiclass_jaccard_index(p, t, num_classes=NUM_CLASSES, average="macro", ignore_index=-100),
    ),
    (
        "iou",
        multiclass_iou(num_classes=NUM_CLASSES),
        lambda p, t: tmfc.multiclass_jaccard_index(p, t, num_classes=NUM_CLASSES, average="macro", ignore_index=-100),
    ),
    (
        "hamming_loss",
        multiclass_hamming_loss(num_classes=NUM_CLASSES),
        lambda p, t: tmfc.multiclass_hamming_distance(
            p, t, num_classes=NUM_CLASSES, average="macro", top_k=1, ignore_index=-100
        ),
    ),
    (
        "fbeta",
        multiclass_fbeta_score(num_classes=NUM_CLASSES, beta=2.0),
        lambda p, t: tmfc.multiclass_fbeta_score(
            p, t, num_classes=NUM_CLASSES, beta=2.0, average="macro", top_k=1, ignore_index=-100
        ),
    ),
    (
        "f1",
        multiclass_f1_score(num_classes=NUM_CLASSES),
        lambda p, t: tmfc.multiclass_f1_score(
            p, t, num_classes=NUM_CLASSES, average="macro", top_k=1, ignore_index=-100
        ),
    ),
]


class TestMulticlassMetrics:
    @pytest.mark.parametrize("name,metric,expected_fn", CONF_MAT_CASES)
    def test_confmat_metrics(self, name: str, metric, expected_fn: Callable, device) -> None:
        del name
        preds, targets = PREDS.to(device), TARGETS.to(device)
        confmat_state = build_confmat_state(metric, device)
        fallback_state = MetricState(preds=preds, targets=targets, confmat=None)
        expected = expected_fn(preds, targets)

        assert_metric_close(metric(confmat_state), expected)
        assert_metric_close(metric(fallback_state), expected)

    @pytest.mark.parametrize(
        "metric,expected_fn",
        [
            (
                multiclass_auroc(num_classes=NUM_CLASSES),
                lambda p, t: tmfc.multiclass_auroc(
                    p,
                    t,
                    num_classes=NUM_CLASSES,
                    average="macro",
                    ignore_index=-100,
                ),
            ),
            (
                multiclass_auprc(num_classes=NUM_CLASSES),
                lambda p, t: tmfc.multiclass_average_precision(
                    p,
                    t,
                    num_classes=NUM_CLASSES,
                    average="macro",
                    ignore_index=-100,
                ),
            ),
        ],
    )
    def test_preds_target_metrics(self, metric, expected_fn: Callable, device) -> None:
        preds = PREDS.to(device)
        targets = TARGETS.to(device)
        state = MetricState(preds=preds, targets=targets)
        assert_metric_close(metric(state), expected_fn(preds, targets))

    def test_topk_accuracy(self, device) -> None:
        preds = PREDS.to(device)
        targets = TARGETS.to(device)
        metric = multiclass_accuracy(num_classes=NUM_CLASSES, k=2)
        state = build_confmat_state(metric, device)
        expected = tmfc.multiclass_accuracy(
            preds,
            targets,
            num_classes=NUM_CLASSES,
            average="macro",
            top_k=2,
            ignore_index=-100,
        )
        assert_metric_close(metric(state), expected)

    @pytest.mark.parametrize(
        "metric,expected_fn",
        [
            (
                multiclass_accuracy(num_classes=NUM_CLASSES),
                lambda p, t: tmfc.multiclass_accuracy(
                    p, t, num_classes=NUM_CLASSES, average="macro", top_k=1, ignore_index=-100
                ),
            ),
            (
                multiclass_precision(num_classes=NUM_CLASSES),
                lambda p, t: tmfc.multiclass_precision(
                    p, t, num_classes=NUM_CLASSES, average="macro", top_k=1, ignore_index=-100
                ),
            ),
            (
                multiclass_recall(num_classes=NUM_CLASSES),
                lambda p, t: tmfc.multiclass_recall(
                    p, t, num_classes=NUM_CLASSES, average="macro", top_k=1, ignore_index=-100
                ),
            ),
            (
                multiclass_jaccard_index(num_classes=NUM_CLASSES),
                lambda p, t: tmfc.multiclass_jaccard_index(
                    p, t, num_classes=NUM_CLASSES, average="macro", ignore_index=-100
                ),
            ),
            (
                multiclass_f1_score(num_classes=NUM_CLASSES),
                lambda p, t: tmfc.multiclass_f1_score(
                    p, t, num_classes=NUM_CLASSES, average="macro", top_k=1, ignore_index=-100
                ),
            ),
        ],
    )
    def test_absent_class(self, metric, expected_fn: Callable, device) -> None:
        preds = torch.tensor(
            [
                [0.90, 0.10, 0.00],
                [0.80, 0.10, 0.10],
                [0.20, 0.70, 0.10],
                [0.30, 0.60, 0.10],
            ],
            device=device,
        )
        targets = torch.tensor([0, 0, 1, 1], device=device)
        requirements = MetricState.collect_requirements((metric,))
        confmat_state = MetricState.from_requirements(preds, targets, requirements)
        fallback_state = MetricState(preds=preds, targets=targets, confmat=None)
        expected = expected_fn(preds, targets)

        assert_metric_close(metric(confmat_state), expected)
        assert_metric_close(metric(fallback_state), expected)

    @pytest.mark.parametrize("average", [None, "micro", "weighted"])
    def test_reduce_modes(self, average, device) -> None:
        preds = PREDS.to(device)
        targets = TARGETS.to(device)
        metric = multiclass_precision(num_classes=NUM_CLASSES, average=average)
        state = build_confmat_state(metric, device)
        expected = tmfc.multiclass_precision(
            preds,
            targets,
            num_classes=NUM_CLASSES,
            average=average,
            top_k=1,
            ignore_index=-100,
        )
        assert_metric_close(metric(state), expected)

    def test_reduce_rejects_invalid_average(self, device) -> None:
        metric = multiclass_precision(num_classes=NUM_CLASSES, average="invalid")
        state = build_confmat_state(metric, device)
        with pytest.raises(ValueError, match="Invalid average value"):
            metric(state)

    @pytest.mark.parametrize("average", [None, "micro", "weighted"])
    def test_balanced_accuracy_rejects_non_macro_average(self, average) -> None:
        with pytest.raises(ValueError, match="only supports average='macro'"):
            multiclass_balanced_accuracy(num_classes=NUM_CLASSES, average=average)

    def test_balanced_accuracy_rejects_top_k(self) -> None:
        with pytest.raises(ValueError, match="only supports k=1"):
            multiclass_balanced_accuracy(num_classes=NUM_CLASSES, k=2)

    @pytest.mark.parametrize(
        "metric",
        [
            multiclass_accuracy(num_classes=NUM_CLASSES),
            multiclass_auroc(num_classes=NUM_CLASSES),
            multiclass_auprc(num_classes=NUM_CLASSES),
            multiclass_precision(num_classes=NUM_CLASSES),
            multiclass_recall(num_classes=NUM_CLASSES),
            multiclass_specificity(num_classes=NUM_CLASSES),
            multiclass_balanced_accuracy(num_classes=NUM_CLASSES),
            multiclass_jaccard_index(num_classes=NUM_CLASSES),
            multiclass_iou(num_classes=NUM_CLASSES),
            multiclass_hamming_loss(num_classes=NUM_CLASSES),
            multiclass_fbeta_score(num_classes=NUM_CLASSES, beta=2.0),
            multiclass_f1_score(num_classes=NUM_CLASSES),
        ],
    )
    def test_return_nan_on_empty_state(self, metric, device) -> None:
        empty_state = MetricState(
            preds=torch.tensor([], device=device),
            targets=torch.tensor([], device=device),
            confmat=None,
        )
        assert torch.isnan(torch.as_tensor(metric(empty_state)))

    def test_f1_alias(self) -> None:
        assert isinstance(multiclass_f1_score(num_classes=NUM_CLASSES), multiclass_fbeta_score)

    def test_iou_name(self) -> None:
        assert multiclass_iou(num_classes=NUM_CLASSES).name == "iou"

    def test_stats_are_cached_per_state(self, device) -> None:
        state = build_confmat_state(multiclass_precision(num_classes=NUM_CLASSES), device)
        first = state.multiclass_stats
        multiclass_precision(num_classes=NUM_CLASSES)(state)
        multiclass_recall(num_classes=NUM_CLASSES)(state)
        multiclass_specificity(num_classes=NUM_CLASSES)(state)
        second = state.multiclass_stats

        assert first is second
