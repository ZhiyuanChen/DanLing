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
from torch import Tensor
from torch.testing import assert_close
from torchmetrics.functional import classification as tmfc

from danling import NestedTensor
from danling.metrics.functional.classification import (
    accuracy,
    auprc,
    auroc,
    balanced_accuracy,
    f1_score,
    fbeta_score,
    hamming_loss,
    iou,
    jaccard_index,
    mcc,
    precision,
    recall,
    specificity,
)
from danling.metrics.preprocess import preprocess_binary, preprocess_multiclass, preprocess_multilabel
from danling.metrics.state import MetricState

ATOL = 1e-6
RTOL = 1e-5

BINARY_PREDS = torch.tensor([0.1, 0.9, 0.8, 0.2, 0.4, 0.6])
BINARY_TARGETS = torch.tensor([0, 1, 1, 0, 1, 0])

MULTICLASS_PREDS = torch.tensor(
    [
        [0.90, 0.05, 0.05],
        [0.10, 0.80, 0.10],
        [0.20, 0.20, 0.60],
        [0.10, 0.70, 0.20],
        [0.60, 0.20, 0.20],
        [0.10, 0.20, 0.70],
    ]
)
MULTICLASS_TARGETS = torch.tensor([0, 1, 2, 0, 1, 2])
NUM_CLASSES = 3

MULTILABEL_PREDS = torch.tensor(
    [
        [0.90, 0.10, 0.80],
        [0.20, 0.70, 0.30],
        [0.60, 0.60, 0.40],
        [0.10, 0.20, 0.90],
        [0.80, 0.30, 0.20],
        [0.30, 0.80, 0.70],
    ]
)
MULTILABEL_TARGETS = torch.tensor(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 1],
    ]
)
NUM_LABELS = 3


def assert_metric_close(actual: Tensor | float, expected: Tensor | float) -> None:
    assert_close(
        torch.as_tensor(actual),
        torch.as_tensor(expected),
        rtol=RTOL,
        atol=ATOL,
        check_dtype=False,
    )


DISPATCH_CASES: list[tuple[str, Callable, dict, Callable, Callable, Callable]] = [
    (
        "accuracy",
        accuracy,
        {"k": 2},
        lambda p, t: tmfc.binary_accuracy(p, t, threshold=0.5, ignore_index=-100),
        lambda p, t: tmfc.multiclass_accuracy(
            p,
            t,
            num_classes=NUM_CLASSES,
            average="macro",
            top_k=2,
            ignore_index=-100,
        ),
        lambda p, t: tmfc.multilabel_accuracy(
            p,
            t,
            threshold=0.5,
            num_labels=NUM_LABELS,
            average="macro",
            ignore_index=-100,
        ),
    ),
    (
        "auprc",
        auprc,
        {},
        lambda p, t: tmfc.binary_average_precision(p, t, ignore_index=-100),
        lambda p, t: tmfc.multiclass_average_precision(
            p,
            t,
            num_classes=NUM_CLASSES,
            average="macro",
            ignore_index=-100,
        ),
        lambda p, t: tmfc.multilabel_average_precision(
            p,
            t,
            num_labels=NUM_LABELS,
            average="macro",
            ignore_index=-100,
        ),
    ),
    (
        "auroc",
        auroc,
        {},
        lambda p, t: tmfc.binary_auroc(p, t, ignore_index=-100),
        lambda p, t: tmfc.multiclass_auroc(
            p,
            t,
            num_classes=NUM_CLASSES,
            average="macro",
            ignore_index=-100,
        ),
        lambda p, t: tmfc.multilabel_auroc(
            p,
            t,
            num_labels=NUM_LABELS,
            average="macro",
            ignore_index=-100,
        ),
    ),
    (
        "f1",
        f1_score,
        {},
        lambda p, t: tmfc.binary_f1_score(p, t, threshold=0.5, ignore_index=-100),
        lambda p, t: tmfc.multiclass_f1_score(
            p,
            t,
            num_classes=NUM_CLASSES,
            average="macro",
            ignore_index=-100,
        ),
        lambda p, t: tmfc.multilabel_f1_score(
            p,
            t,
            threshold=0.5,
            num_labels=NUM_LABELS,
            average="macro",
            ignore_index=-100,
        ),
    ),
    (
        "fbeta",
        fbeta_score,
        {"beta": 2.0},
        lambda p, t: tmfc.binary_fbeta_score(p, t, beta=2.0, threshold=0.5, ignore_index=-100),
        lambda p, t: tmfc.multiclass_fbeta_score(
            p,
            t,
            beta=2.0,
            num_classes=NUM_CLASSES,
            average="macro",
            ignore_index=-100,
        ),
        lambda p, t: tmfc.multilabel_fbeta_score(
            p,
            t,
            beta=2.0,
            threshold=0.5,
            num_labels=NUM_LABELS,
            average="macro",
            ignore_index=-100,
        ),
    ),
    (
        "precision",
        precision,
        {"k": 2},
        lambda p, t: tmfc.binary_precision(p, t, threshold=0.5, ignore_index=-100),
        lambda p, t: tmfc.multiclass_precision(
            p,
            t,
            num_classes=NUM_CLASSES,
            average="macro",
            top_k=2,
            ignore_index=-100,
        ),
        lambda p, t: tmfc.multilabel_precision(
            p,
            t,
            threshold=0.5,
            num_labels=NUM_LABELS,
            average="macro",
            ignore_index=-100,
        ),
    ),
    (
        "recall",
        recall,
        {"k": 2},
        lambda p, t: tmfc.binary_recall(p, t, threshold=0.5, ignore_index=-100),
        lambda p, t: tmfc.multiclass_recall(
            p,
            t,
            num_classes=NUM_CLASSES,
            average="macro",
            top_k=2,
            ignore_index=-100,
        ),
        lambda p, t: tmfc.multilabel_recall(
            p,
            t,
            threshold=0.5,
            num_labels=NUM_LABELS,
            average="macro",
            ignore_index=-100,
        ),
    ),
    (
        "specificity",
        specificity,
        {"k": 2},
        lambda p, t: tmfc.binary_specificity(p, t, threshold=0.5, ignore_index=-100),
        lambda p, t: tmfc.multiclass_specificity(
            p,
            t,
            num_classes=NUM_CLASSES,
            average="macro",
            top_k=2,
            ignore_index=-100,
        ),
        lambda p, t: tmfc.multilabel_specificity(
            p,
            t,
            threshold=0.5,
            num_labels=NUM_LABELS,
            average="macro",
            ignore_index=-100,
        ),
    ),
    (
        "balanced_accuracy",
        balanced_accuracy,
        {},
        lambda p, t: 0.5
        * (
            tmfc.binary_recall(p, t, threshold=0.5, ignore_index=-100)
            + tmfc.binary_specificity(p, t, threshold=0.5, ignore_index=-100)
        ),
        lambda p, t: tmfc.multiclass_recall(
            p,
            t,
            num_classes=NUM_CLASSES,
            average="macro",
            top_k=1,
            ignore_index=-100,
        ),
        lambda p, t: 0.5
        * (
            tmfc.multilabel_recall(
                p,
                t,
                threshold=0.5,
                num_labels=NUM_LABELS,
                average="macro",
                ignore_index=-100,
            )
            + tmfc.multilabel_specificity(
                p,
                t,
                threshold=0.5,
                num_labels=NUM_LABELS,
                average="macro",
                ignore_index=-100,
            )
        ),
    ),
    (
        "jaccard_index",
        jaccard_index,
        {},
        lambda p, t: tmfc.binary_jaccard_index(p, t, threshold=0.5, ignore_index=-100),
        lambda p, t: tmfc.multiclass_jaccard_index(
            p,
            t,
            num_classes=NUM_CLASSES,
            average="macro",
            ignore_index=-100,
        ),
        lambda p, t: tmfc.multilabel_jaccard_index(
            p,
            t,
            threshold=0.5,
            num_labels=NUM_LABELS,
            average="macro",
            ignore_index=-100,
        ),
    ),
    (
        "iou",
        iou,
        {},
        lambda p, t: tmfc.binary_jaccard_index(p, t, threshold=0.5, ignore_index=-100),
        lambda p, t: tmfc.multiclass_jaccard_index(
            p,
            t,
            num_classes=NUM_CLASSES,
            average="macro",
            ignore_index=-100,
        ),
        lambda p, t: tmfc.multilabel_jaccard_index(
            p,
            t,
            threshold=0.5,
            num_labels=NUM_LABELS,
            average="macro",
            ignore_index=-100,
        ),
    ),
    (
        "hamming_loss",
        hamming_loss,
        {"k": 2},
        lambda p, t: tmfc.binary_hamming_distance(p, t, threshold=0.5, ignore_index=-100),
        lambda p, t: tmfc.multiclass_hamming_distance(
            p,
            t,
            num_classes=NUM_CLASSES,
            average="macro",
            top_k=2,
            ignore_index=-100,
        ),
        lambda p, t: tmfc.multilabel_hamming_distance(
            p,
            t,
            threshold=0.5,
            num_labels=NUM_LABELS,
            average="macro",
            ignore_index=-100,
        ),
    ),
]


class TestClassificationDispatch:
    @pytest.mark.parametrize(
        "name,metric_fn,metric_kwargs,expected_binary,expected_multiclass,expected_multilabel",
        DISPATCH_CASES,
    )
    def test_function_dispatch(
        self,
        name: str,
        metric_fn: Callable,
        metric_kwargs: dict,
        expected_binary: Callable,
        expected_multiclass: Callable,
        expected_multilabel: Callable,
    ):
        del name
        actual_binary = metric_fn(BINARY_PREDS, BINARY_TARGETS, **metric_kwargs)
        assert_metric_close(actual_binary, expected_binary(BINARY_PREDS, BINARY_TARGETS))

        actual_multiclass = metric_fn(
            MULTICLASS_PREDS,
            MULTICLASS_TARGETS,
            num_classes=NUM_CLASSES,
            **metric_kwargs,
        )
        assert_metric_close(actual_multiclass, expected_multiclass(MULTICLASS_PREDS, MULTICLASS_TARGETS))

        actual_multilabel = metric_fn(
            MULTILABEL_PREDS,
            MULTILABEL_TARGETS,
            num_labels=NUM_LABELS,
            **metric_kwargs,
        )
        assert_metric_close(actual_multilabel, expected_multilabel(MULTILABEL_PREDS, MULTILABEL_TARGETS))

    @pytest.mark.parametrize(
        "metric_fn,metric_kwargs",
        [
            (accuracy, {}),
            (auprc, {}),
            (auroc, {}),
            (f1_score, {}),
            (fbeta_score, {"beta": 2.0}),
            (precision, {}),
            (recall, {}),
            (specificity, {}),
            (balanced_accuracy, {}),
            (jaccard_index, {}),
            (iou, {}),
            (hamming_loss, {}),
        ],
    )
    def test_dispatch_rejects_invalid_task(self, metric_fn: Callable, metric_kwargs: dict):
        with pytest.raises(ValueError, match="Task should be one of binary, multiclass, or multilabel"):
            metric_fn(BINARY_PREDS, BINARY_TARGETS, task="invalid", **metric_kwargs)

    @pytest.mark.parametrize("average", [None, "micro", "weighted"])
    def test_multiclass_balanced_accuracy_rejects_non_macro_average(self, average) -> None:
        with pytest.raises(ValueError, match="only supports average='macro'"):
            balanced_accuracy(MULTICLASS_PREDS, MULTICLASS_TARGETS, num_classes=NUM_CLASSES, average=average)

    def test_multiclass_balanced_accuracy_rejects_top_k(self) -> None:
        with pytest.raises(ValueError, match="only supports k=1"):
            balanced_accuracy(MULTICLASS_PREDS, MULTICLASS_TARGETS, num_classes=NUM_CLASSES, k=2)


class TestClassificationPreprocess:
    @pytest.mark.parametrize(
        "metric_fn,expected_fn",
        [
            (
                accuracy,
                lambda p, t: tmfc.binary_accuracy(p, t, threshold=0.5, ignore_index=-100),
            ),
            (
                balanced_accuracy,
                lambda p, t: 0.5
                * (
                    tmfc.binary_recall(p, t, threshold=0.5, ignore_index=-100)
                    + tmfc.binary_specificity(p, t, threshold=0.5, ignore_index=-100)
                ),
            ),
        ],
    )
    def test_binary_nested_tensor(self, metric_fn: Callable, expected_fn: Callable) -> None:
        input = NestedTensor([torch.tensor([-2.0, 2.0]), torch.tensor([1.0])])
        target = NestedTensor([torch.tensor([0, 1]), torch.tensor([1])])

        preds, targets = preprocess_binary(input, target)
        actual = metric_fn(input, target)

        assert_metric_close(actual, expected_fn(preds, targets))

    @pytest.mark.parametrize(
        "metric_fn,expected_fn",
        [
            (
                accuracy,
                lambda p, t: tmfc.multiclass_accuracy(
                    p,
                    t,
                    num_classes=NUM_CLASSES,
                    average="macro",
                    top_k=1,
                    ignore_index=-100,
                ),
            ),
            (
                balanced_accuracy,
                lambda p, t: tmfc.multiclass_recall(
                    p,
                    t,
                    num_classes=NUM_CLASSES,
                    average="macro",
                    top_k=1,
                    ignore_index=-100,
                ),
            ),
        ],
    )
    def test_multiclass_column_target(self, metric_fn: Callable, expected_fn: Callable) -> None:
        input = torch.tensor([[3.0, 1.0, 0.0], [0.1, 2.0, 0.2]])
        target = torch.tensor([[0], [1]])

        preds, targets = preprocess_multiclass(input, target, num_classes=NUM_CLASSES)
        actual = metric_fn(input, target, num_classes=NUM_CLASSES)

        assert_metric_close(actual, expected_fn(preds, targets))

    @pytest.mark.parametrize(
        "metric_fn,expected_fn",
        [
            (
                accuracy,
                lambda p, t: tmfc.multilabel_accuracy(
                    p,
                    t,
                    threshold=0.5,
                    num_labels=NUM_LABELS,
                    average="macro",
                    ignore_index=-100,
                ),
            ),
            (
                balanced_accuracy,
                lambda p, t: 0.5
                * (
                    tmfc.multilabel_recall(
                        p,
                        t,
                        threshold=0.5,
                        num_labels=NUM_LABELS,
                        average="macro",
                        ignore_index=-100,
                    )
                    + tmfc.multilabel_specificity(
                        p,
                        t,
                        threshold=0.5,
                        num_labels=NUM_LABELS,
                        average="macro",
                        ignore_index=-100,
                    )
                ),
            ),
        ],
    )
    def test_multilabel_ignore_index(self, metric_fn: Callable, expected_fn: Callable) -> None:
        input = torch.tensor(
            [
                [3.0, -2.0, 1.0],
                [0.2, -0.8, 0.6],
                [1.0, 2.0, -1.0],
            ]
        )
        target = torch.tensor(
            [
                [1, -100, 0],
                [-100, -100, -100],
                [0, 1, -100],
            ]
        )

        preds, targets = preprocess_multilabel(input, target, num_labels=NUM_LABELS, ignore_index=-100)
        actual = metric_fn(input, target, num_labels=NUM_LABELS, ignore_index=-100)

        assert_metric_close(actual, expected_fn(preds, targets))

    @pytest.mark.parametrize(
        "metric_fn,metric_kwargs,input,target,preprocess,expected_fn",
        [
            pytest.param(
                auroc,
                {},
                NestedTensor([torch.tensor([-2.0, 2.0]), torch.tensor([1.0])]),
                NestedTensor([torch.tensor([0, 1]), torch.tensor([1])]),
                preprocess_binary,
                lambda p, t: tmfc.binary_auroc(p, t, ignore_index=-100),
                id="binary-auroc-nested",
            ),
            pytest.param(
                auprc,
                {},
                NestedTensor([torch.tensor([-2.0, 2.0]), torch.tensor([1.0])]),
                NestedTensor([torch.tensor([0, 1]), torch.tensor([1])]),
                preprocess_binary,
                lambda p, t: tmfc.binary_average_precision(p, t, ignore_index=-100),
                id="binary-auprc-nested",
            ),
            pytest.param(
                fbeta_score,
                {"beta": 2.0},
                NestedTensor([torch.tensor([-2.0, 2.0]), torch.tensor([1.0])]),
                NestedTensor([torch.tensor([0, 1]), torch.tensor([1])]),
                preprocess_binary,
                lambda p, t: tmfc.binary_fbeta_score(p, t, beta=2.0, threshold=0.5, ignore_index=-100),
                id="binary-fbeta-nested",
            ),
            pytest.param(
                precision,
                {"num_classes": NUM_CLASSES, "k": 2},
                torch.tensor([[3.0, 1.0, 0.0], [0.1, 2.0, 0.2]]),
                torch.tensor([[0], [1]]),
                lambda input, target: preprocess_multiclass(input, target, num_classes=NUM_CLASSES),
                lambda p, t: tmfc.multiclass_precision(
                    p,
                    t,
                    num_classes=NUM_CLASSES,
                    average="macro",
                    top_k=2,
                    ignore_index=-100,
                ),
                id="multiclass-precision-column-target",
            ),
            pytest.param(
                hamming_loss,
                {"num_classes": NUM_CLASSES, "k": 2},
                torch.tensor([[3.0, 1.0, 0.0], [0.1, 2.0, 0.2]]),
                torch.tensor([[0], [1]]),
                lambda input, target: preprocess_multiclass(input, target, num_classes=NUM_CLASSES),
                lambda p, t: tmfc.multiclass_hamming_distance(
                    p,
                    t,
                    num_classes=NUM_CLASSES,
                    average="macro",
                    top_k=2,
                    ignore_index=-100,
                ),
                id="multiclass-hamming-column-target",
            ),
            pytest.param(
                specificity,
                {"num_labels": NUM_LABELS, "ignore_index": -100},
                torch.tensor(
                    [
                        [3.0, -2.0, 1.0],
                        [0.2, -0.8, 0.6],
                        [1.0, 2.0, -1.0],
                    ]
                ),
                torch.tensor(
                    [
                        [1, -100, 0],
                        [-100, -100, -100],
                        [0, 1, -100],
                    ]
                ),
                lambda input, target: preprocess_multilabel(input, target, num_labels=NUM_LABELS, ignore_index=-100),
                lambda p, t: tmfc.multilabel_specificity(
                    p,
                    t,
                    threshold=0.5,
                    num_labels=NUM_LABELS,
                    average="macro",
                    ignore_index=-100,
                ),
                id="multilabel-specificity-ignore-index",
            ),
            pytest.param(
                jaccard_index,
                {"num_labels": NUM_LABELS, "ignore_index": -100},
                torch.tensor(
                    [
                        [3.0, -2.0, 1.0],
                        [0.2, -0.8, 0.6],
                        [1.0, 2.0, -1.0],
                    ]
                ),
                torch.tensor(
                    [
                        [1, -100, 0],
                        [-100, -100, -100],
                        [0, 1, -100],
                    ]
                ),
                lambda input, target: preprocess_multilabel(input, target, num_labels=NUM_LABELS, ignore_index=-100),
                lambda p, t: tmfc.multilabel_jaccard_index(
                    p,
                    t,
                    threshold=0.5,
                    num_labels=NUM_LABELS,
                    average="macro",
                    ignore_index=-100,
                ),
                id="multilabel-jaccard-ignore-index",
            ),
        ],
    )
    def test_preprocess_metric_families(
        self,
        metric_fn: Callable,
        metric_kwargs: dict,
        input,
        target,
        preprocess: Callable,
        expected_fn: Callable,
    ) -> None:
        preds, targets = preprocess(input, target)
        actual = metric_fn(input, target, **metric_kwargs)

        assert_metric_close(actual, expected_fn(preds, targets))

    @pytest.mark.parametrize(
        "input,target,metric_kwargs,preprocess,expected_error",
        [
            (
                torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
                torch.tensor([[0, 1], [1, 0]]),
                {},
                preprocess_binary,
                "Expected input to be at least 2D when multidim_average is set to `samplewise`",
            ),
            (
                torch.tensor(
                    [
                        [[0.9, 0.1], [0.1, 0.8], [0.0, 0.1]],
                        [[0.1, 0.7], [0.2, 0.2], [0.7, 0.1]],
                    ],
                    dtype=torch.float32,
                ),
                torch.tensor([[0, 1], [2, 0]]),
                {"num_classes": NUM_CLASSES},
                lambda input, target: preprocess_multiclass(input, target, num_classes=NUM_CLASSES),
                "the shape of `preds` should be at least 3D when multidim_average is set to `samplewise`",
            ),
        ],
    )
    def test_accuracy_forwards_kwargs_after_preprocess(
        self,
        input: Tensor,
        target: Tensor,
        metric_kwargs: dict,
        preprocess: Callable,
        expected_error: str,
    ) -> None:
        preds, targets = preprocess(input, target)

        with pytest.raises(ValueError, match=expected_error):
            tmfc.accuracy(
                preds,
                targets,
                task="binary" if not metric_kwargs else "multiclass",
                multidim_average="samplewise",
                **metric_kwargs,
            )

        with pytest.raises(ValueError, match=expected_error):
            accuracy(input, target, multidim_average="samplewise", **metric_kwargs)


class TestMatthewsCorrcoef:
    def test_metric_func(self):
        for task, preds, targets, kwargs in [
            ("binary", BINARY_PREDS, BINARY_TARGETS, {}),
            ("multiclass", MULTICLASS_PREDS, MULTICLASS_TARGETS, {"num_classes": NUM_CLASSES}),
            ("multilabel", MULTILABEL_PREDS, MULTILABEL_TARGETS, {"num_labels": NUM_LABELS}),
        ]:
            metric = mcc(task=task, **kwargs)
            state = MetricState(preds=preds, targets=targets)
            expected = tmfc.matthews_corrcoef(preds, targets, task=task, threshold=0.5, ignore_index=-100, **kwargs)
            assert_metric_close(metric(state), expected)

    def test_returns_nan_on_empty_state(self):
        metric = mcc(task="binary")
        state = MetricState(preds=torch.tensor([]), targets=torch.tensor([]))
        assert torch.isnan(torch.as_tensor(metric(state)))
