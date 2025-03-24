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

from .metric_meter import MetricMeters
from .preprocesses import preprocess_binary, preprocess_multiclass, preprocess_multilabel, preprocess_regression

with try_import() as lazy_import:
    from .functional import accuracy, auprc, auroc, f1_score, mcc, pearson, r2_score, rmse, spearman
    from .metrics import Metrics


def binary_metrics(ignore_index: int | None = -100, **kwargs):
    lazy_import.check()
    return Metrics(
        auroc=partial(auroc, task="binary", preprocess=False),
        auprc=partial(auprc, task="binary", preprocess=False),
        acc=partial(accuracy, task="binary", preprocess=False),
        mcc=partial(mcc, task="binary", preprocess=False),
        f1=partial(f1_score, task="binary", preprocess=False),
        preprocess=partial(preprocess_binary, ignore_index=ignore_index),
        **kwargs,
    )


def multiclass_metrics(num_classes: int, ignore_index: int | None = -100, **kwargs):
    lazy_import.check()
    return Metrics(
        auroc=partial(auroc, task="multiclass", num_classes=num_classes, preprocess=False),
        auprc=partial(auprc, task="multiclass", num_classes=num_classes, preprocess=False),
        acc=partial(accuracy, task="multiclass", num_classes=num_classes, preprocess=False),
        mcc=partial(mcc, task="multiclass", num_classes=num_classes, preprocess=False),
        f1=partial(f1_score, task="multiclass", num_classes=num_classes, preprocess=False),
        preprocess=partial(preprocess_multiclass, num_classes=num_classes, ignore_index=ignore_index),
        **kwargs,
    )


def multiclass_metric_meters(num_classes: int, ignore_index: int | None = -100, **kwargs):
    lazy_import.check()
    return MetricMeters(
        acc=partial(accuracy, task="multiclass", num_classes=num_classes, preprocess=False),
        f1=partial(f1_score, task="multiclass", num_classes=num_classes, preprocess=False),
        preprocess=partial(preprocess_multiclass, num_classes=num_classes, ignore_index=ignore_index),
        **kwargs,
    )


def multilabel_metrics(num_labels: int, ignore_index: int | None = -100, **kwargs):
    lazy_import.check()
    return Metrics(
        auroc=partial(auroc, task="multilabel", num_labels=num_labels, preprocess=False),
        auprc=partial(auprc, task="multilabel", num_labels=num_labels, preprocess=False),
        acc=partial(accuracy, task="multilabel", num_labels=num_labels, preprocess=False),
        mcc=partial(mcc, task="multilabel", num_labels=num_labels, preprocess=False),
        f1=partial(f1_score, task="multilabel", num_labels=num_labels, preprocess=False),
        preprocess=partial(preprocess_multilabel, num_labels=num_labels, ignore_index=ignore_index),
        **kwargs,
    )


def multilabel_metric_meters(num_labels: int, ignore_index: int | None = -100, **kwargs):
    lazy_import.check()
    return MetricMeters(
        acc=partial(accuracy, task="multilabel", num_labels=num_labels, preprocess=False),
        f1=partial(f1_score, task="multilabel", num_labels=num_labels, preprocess=False),
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
