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

import torch
import torchmetrics.functional as tmf
from torch.testing import assert_close
from torchmetrics.functional import classification as tmfc
from torchmetrics.functional import matthews_corrcoef

from danling.metrics.functional import (
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
    mae,
    mcc,
    mse,
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
    pearson,
    r2_score,
    rmse,
    spearman,
)
from danling.metrics.global_metrics import GlobalMetrics
from danling.metrics.preprocess import (
    preprocess_binary,
    preprocess_multiclass,
    preprocess_multilabel,
    preprocess_regression,
)

ATOL = 1e-6
RTOL = 1e-5


def build_function_map(
    task: str,
    *,
    num_classes: int = 10,
    num_labels: int = 10,
    num_outputs: int = 1,
    average: str = "macro",
    ignore_index: int | None = -100,
):
    if task == "binary":
        return {
            "auroc": partial(tmfc.binary_auroc, ignore_index=ignore_index),
            "auprc": partial(tmfc.binary_average_precision, ignore_index=ignore_index),
            "acc": partial(tmfc.binary_accuracy, ignore_index=ignore_index),
            "f1": partial(tmfc.binary_f1_score, ignore_index=ignore_index),
            "mcc": partial(matthews_corrcoef, task="binary", ignore_index=ignore_index),
        }
    if task == "multiclass":
        return {
            "auroc": partial(
                tmfc.multiclass_auroc,
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
            "auprc": partial(
                tmfc.multiclass_average_precision,
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
            "acc": partial(
                tmfc.multiclass_accuracy, num_classes=num_classes, average=average, ignore_index=ignore_index
            ),
            "f1": partial(
                tmfc.multiclass_f1_score, num_classes=num_classes, average=average, ignore_index=ignore_index
            ),
            "mcc": partial(matthews_corrcoef, task="multiclass", num_classes=num_classes, ignore_index=ignore_index),
        }
    if task == "multilabel":
        return {
            "auroc": partial(tmfc.multilabel_auroc, num_labels=num_labels, average=average, ignore_index=ignore_index),
            "auprc": partial(
                tmfc.multilabel_average_precision,
                num_labels=num_labels,
                average=average,
                ignore_index=ignore_index,
            ),
            "acc": partial(tmfc.multilabel_accuracy, num_labels=num_labels, ignore_index=ignore_index),
            "f1": partial(tmfc.multilabel_f1_score, num_labels=num_labels, average=average, ignore_index=ignore_index),
            "mcc": partial(matthews_corrcoef, task="multilabel", num_labels=num_labels, ignore_index=ignore_index),
        }
    if task == "regression":
        return {
            "pearson": lambda p, t: tmf.pearson_corrcoef(p, t).mean(),
            "spearman": lambda p, t: tmf.spearman_corrcoef(p, t).mean(),
            "r2": tmf.r2_score,
            "mse": partial(tmf.mean_squared_error, squared=True, num_outputs=num_outputs),
            "rmse": partial(tmf.mean_squared_error, squared=False, num_outputs=num_outputs),
        }
    raise ValueError(f"Unsupported task: {task}")


def build_prf_function_map(
    task: str,
    *,
    beta: float = 2.0,
    num_classes: int = 10,
    num_labels: int = 10,
    average: str = "macro",
    ignore_index: int | None = -100,
):
    if task == "binary":
        return {
            "precision": partial(tmfc.binary_precision, ignore_index=ignore_index),
            "recall": partial(tmfc.binary_recall, ignore_index=ignore_index),
            "fbeta": partial(tmfc.binary_fbeta_score, beta=beta, ignore_index=ignore_index),
            "f1": partial(tmfc.binary_f1_score, ignore_index=ignore_index),
        }
    if task == "multiclass":
        return {
            "precision": partial(
                tmfc.multiclass_precision,
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
            "recall": partial(
                tmfc.multiclass_recall, num_classes=num_classes, average=average, ignore_index=ignore_index
            ),
            "fbeta": partial(
                tmfc.multiclass_fbeta_score,
                num_classes=num_classes,
                average=average,
                beta=beta,
                ignore_index=ignore_index,
            ),
            "f1": partial(
                tmfc.multiclass_f1_score, num_classes=num_classes, average=average, ignore_index=ignore_index
            ),
        }
    if task == "multilabel":
        return {
            "precision": partial(
                tmfc.multilabel_precision, num_labels=num_labels, average=average, ignore_index=ignore_index
            ),
            "recall": partial(
                tmfc.multilabel_recall, num_labels=num_labels, average=average, ignore_index=ignore_index
            ),
            "fbeta": partial(
                tmfc.multilabel_fbeta_score,
                num_labels=num_labels,
                average=average,
                beta=beta,
                ignore_index=ignore_index,
            ),
            "f1": partial(tmfc.multilabel_f1_score, num_labels=num_labels, average=average, ignore_index=ignore_index),
        }
    raise ValueError(f"Unsupported task: {task}")


def build_structural_function_map(
    task: str,
    *,
    num_classes: int = 10,
    num_labels: int = 10,
    average: str = "macro",
    ignore_index: int | None = -100,
):
    if task == "binary":
        return {
            "specificity": partial(tmfc.binary_specificity, ignore_index=ignore_index),
            "balanced_accuracy": lambda p, t: 0.5
            * (
                tmfc.binary_recall(p, t, ignore_index=ignore_index)
                + tmfc.binary_specificity(p, t, ignore_index=ignore_index)
            ),
            "jaccard": partial(tmfc.binary_jaccard_index, ignore_index=ignore_index),
            "iou": partial(tmfc.binary_jaccard_index, ignore_index=ignore_index),
            "hamming_loss": partial(tmfc.binary_hamming_distance, ignore_index=ignore_index),
        }
    if task == "multiclass":
        return {
            "specificity": partial(
                tmfc.multiclass_specificity,
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
            "balanced_accuracy": partial(
                tmfc.multiclass_recall,
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
            "jaccard": partial(
                tmfc.multiclass_jaccard_index,
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
            "iou": partial(
                tmfc.multiclass_jaccard_index,
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
            "hamming_loss": partial(
                tmfc.multiclass_hamming_distance,
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
        }
    if task == "multilabel":
        return {
            "specificity": partial(
                tmfc.multilabel_specificity,
                num_labels=num_labels,
                average=average,
                ignore_index=ignore_index,
            ),
            "balanced_accuracy": lambda p, t: 0.5
            * (
                tmfc.multilabel_recall(p, t, num_labels=num_labels, average=average, ignore_index=ignore_index)
                + tmfc.multilabel_specificity(p, t, num_labels=num_labels, average=average, ignore_index=ignore_index)
            ),
            "jaccard": partial(
                tmfc.multilabel_jaccard_index,
                num_labels=num_labels,
                average=average,
                ignore_index=ignore_index,
            ),
            "iou": partial(
                tmfc.multilabel_jaccard_index,
                num_labels=num_labels,
                average=average,
                ignore_index=ignore_index,
            ),
            "hamming_loss": partial(
                tmfc.multilabel_hamming_distance,
                num_labels=num_labels,
                average=average,
                ignore_index=ignore_index,
            ),
        }
    raise ValueError(f"Unsupported task: {task}")


def make_binary_metrics(*, distributed: bool = False):
    funcs = [
        binary_auroc(),
        binary_auprc(),
        binary_accuracy(),
        binary_f1(),
        mcc(task="binary"),
    ]
    return GlobalMetrics(funcs, preprocess=preprocess_binary, distributed=distributed)


def make_multiclass_metrics(num_classes: int, average: str = "macro", *, distributed: bool = False):
    funcs = [
        multiclass_auroc(num_classes=num_classes, average=average),
        multiclass_auprc(num_classes=num_classes, average=average),
        multiclass_accuracy(num_classes=num_classes, average=average),
        multiclass_f1_score(num_classes=num_classes, average=average),
        mcc(task="multiclass", num_classes=num_classes),
    ]
    return GlobalMetrics(
        funcs, preprocess=partial(preprocess_multiclass, num_classes=num_classes), distributed=distributed
    )


def make_multilabel_metrics(num_labels: int, average: str = "macro", *, distributed: bool = False):
    funcs = [
        multilabel_auroc(num_labels=num_labels, average=average),
        multilabel_auprc(num_labels=num_labels, average=average),
        multilabel_accuracy(num_labels=num_labels),
        multilabel_f1_score(num_labels=num_labels, average=average),
        mcc(task="multilabel", num_labels=num_labels),
    ]
    return GlobalMetrics(
        funcs, preprocess=partial(preprocess_multilabel, num_labels=num_labels), distributed=distributed
    )


def make_regression_metrics(num_outputs: int = 1, *, distributed: bool = False):
    funcs = [
        pearson(),
        spearman(),
        r2_score(),
        mse(num_outputs=num_outputs),
        rmse(num_outputs=num_outputs),
    ]
    return GlobalMetrics(
        funcs,
        preprocess=partial(preprocess_regression, num_outputs=num_outputs, ignore_nan=True),
        distributed=distributed,
    )


def make_binary_prf_metrics(beta: float = 2.0, *, distributed: bool = False):
    funcs = [binary_precision(), binary_recall(), binary_fbeta(beta=beta), binary_f1()]
    return GlobalMetrics(funcs, preprocess=preprocess_binary, distributed=distributed)


def make_multiclass_prf_metrics(num_classes: int, beta: float = 2.0, *, distributed: bool = False):
    funcs = [
        multiclass_precision(num_classes=num_classes),
        multiclass_recall(num_classes=num_classes),
        multiclass_fbeta_score(num_classes=num_classes, beta=beta),
        multiclass_f1_score(num_classes=num_classes),
    ]
    return GlobalMetrics(
        funcs, preprocess=partial(preprocess_multiclass, num_classes=num_classes), distributed=distributed
    )


def make_multilabel_prf_metrics(num_labels: int, beta: float = 2.0, *, distributed: bool = False):
    funcs = [
        multilabel_precision(num_labels=num_labels),
        multilabel_recall(num_labels=num_labels),
        multilabel_fbeta_score(num_labels=num_labels, beta=beta),
        multilabel_f1_score(num_labels=num_labels),
    ]
    return GlobalMetrics(
        funcs, preprocess=partial(preprocess_multilabel, num_labels=num_labels), distributed=distributed
    )


def make_binary_structural_metrics(*, distributed: bool = False):
    funcs = [
        binary_specificity(),
        binary_balanced_accuracy(),
        binary_jaccard_index(),
        binary_iou(),
        binary_hamming_loss(),
    ]
    return GlobalMetrics(funcs, preprocess=preprocess_binary, distributed=distributed)


def make_multiclass_structural_metrics(num_classes: int, average: str = "macro", *, distributed: bool = False):
    funcs = [
        multiclass_specificity(num_classes=num_classes, average=average),
        multiclass_balanced_accuracy(num_classes=num_classes, average=average),
        multiclass_jaccard_index(num_classes=num_classes, average=average),
        multiclass_iou(num_classes=num_classes, average=average),
        multiclass_hamming_loss(num_classes=num_classes, average=average),
    ]
    return GlobalMetrics(
        funcs, preprocess=partial(preprocess_multiclass, num_classes=num_classes), distributed=distributed
    )


def make_multilabel_structural_metrics(num_labels: int, average: str = "macro", *, distributed: bool = False):
    funcs = [
        multilabel_specificity(num_labels=num_labels, average=average),
        multilabel_balanced_accuracy(num_labels=num_labels, average=average),
        multilabel_jaccard_index(num_labels=num_labels, average=average),
        multilabel_iou(num_labels=num_labels, average=average),
        multilabel_hamming_loss(num_labels=num_labels, average=average),
    ]
    return GlobalMetrics(
        funcs, preprocess=partial(preprocess_multilabel, num_labels=num_labels), distributed=distributed
    )


def make_regression_mae_metric(num_outputs: int = 1, *, distributed: bool = False):
    funcs = [mae(num_outputs=num_outputs)]
    return GlobalMetrics(
        funcs,
        preprocess=partial(preprocess_regression, num_outputs=num_outputs, ignore_nan=True),
        distributed=distributed,
    )


def assert_metric_outputs(metric_values, function_map, preds, targets):
    for key, func in function_map.items():
        actual = torch.as_tensor(metric_values[key])
        expected = torch.as_tensor(func(preds, targets))
        assert_close(actual, expected, rtol=RTOL, atol=ATOL, check_dtype=False)
