# DanLing
# Copyright (C) 2022-Present  DanLing

# This file is part of DanLing.

# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from __future__ import annotations

from functools import partial

from lazy_imports import try_import

from .functional.preprocess import (
    preprocess_binary,
    preprocess_multiclass,
    preprocess_multilabel,
    preprocess_regression,
)
from .metric_meter import MetricMeters

with try_import() as lazy_import:
    from .functional import (
        binary_accuracy,
        binary_auprc,
        binary_auroc,
        binary_f1_score,
        mcc,
        multiclass_accuracy,
        multiclass_auprc,
        multiclass_auroc,
        multiclass_f1_score,
        multilabel_accuracy,
        multilabel_auprc,
        multilabel_auroc,
        multilabel_f1_score,
        pearson,
        r2_score,
        rmse,
        spearman,
    )
    from .metrics import Metrics


def binary_metrics(ignore_index: int | None = -100, **kwargs):
    lazy_import.check()
    return Metrics(
        auroc=partial(binary_auroc, preprocess=False),
        auprc=partial(binary_auprc, preprocess=False),
        acc=partial(binary_accuracy, preprocess=False),
        f1=partial(binary_f1_score, preprocess=False),
        mcc=partial(mcc, task="binary", preprocess=False),
        preprocess=partial(preprocess_binary, ignore_index=ignore_index),
        **kwargs,
    )


def multiclass_metrics(num_classes: int, average: str = "macro", ignore_index: int | None = -100, **kwargs):
    lazy_import.check()
    return Metrics(
        auroc=partial(multiclass_auroc, num_classes=num_classes, average=average, preprocess=False),
        auprc=partial(multiclass_auprc, num_classes=num_classes, average=average, preprocess=False),
        acc=partial(multiclass_accuracy, num_classes=num_classes, average=average, preprocess=False),
        f1=partial(multiclass_f1_score, num_classes=num_classes, average=average, preprocess=False),
        mcc=partial(mcc, task="multiclass", num_classes=num_classes, preprocess=False),
        preprocess=partial(preprocess_multiclass, num_classes=num_classes, ignore_index=ignore_index),
        **kwargs,
    )


def multiclass_metric_meters(num_classes: int, average: str = "macro", ignore_index: int | None = -100, **kwargs):
    lazy_import.check()
    return MetricMeters(
        acc=partial(multiclass_accuracy, num_classes=num_classes, average=average, preprocess=False),
        f1=partial(multiclass_f1_score, num_classes=num_classes, average=average, preprocess=False),
        preprocess=partial(preprocess_multiclass, num_classes=num_classes, ignore_index=ignore_index),
        **kwargs,
    )


def multilabel_metrics(num_labels: int, average: str = "macro", ignore_index: int | None = -100, **kwargs):
    lazy_import.check()
    return Metrics(
        auroc=partial(multilabel_auroc, num_labels=num_labels, average=average, preprocess=False),
        auprc=partial(multilabel_auprc, num_labels=num_labels, average=average, preprocess=False),
        acc=partial(multilabel_accuracy, num_labels=num_labels, preprocess=False),
        f1=partial(multilabel_f1_score, num_labels=num_labels, average=average, preprocess=False),
        mcc=partial(mcc, task="multilabel", num_labels=num_labels, preprocess=False),
        preprocess=partial(preprocess_multilabel, num_labels=num_labels, ignore_index=ignore_index),
        **kwargs,
    )


def multilabel_metric_meters(num_labels: int, average: str = "macro", ignore_index: int | None = -100, **kwargs):
    lazy_import.check()
    return MetricMeters(
        acc=partial(multilabel_accuracy, num_labels=num_labels, average=average, preprocess=False),
        f1=partial(multilabel_f1_score, num_labels=num_labels, average=average, preprocess=False),
        preprocess=partial(preprocess_multilabel, num_labels=num_labels, ignore_index=ignore_index),
        **kwargs,
    )


def regression_metrics(num_outputs: int = 1, ignore_nan: bool = True, **kwargs):
    lazy_import.check()
    return Metrics(
        pearson=partial(pearson, num_outputs=num_outputs, preprocess=False),
        spearman=partial(spearman, num_outputs=num_outputs, preprocess=False),
        r2=partial(r2_score, num_outputs=num_outputs, preprocess=False),
        rmse=partial(rmse, num_outputs=num_outputs, preprocess=False),
        preprocess=partial(preprocess_regression, num_outputs=num_outputs, ignore_nan=ignore_nan),
        **kwargs,
    )
