# DanLing
# Copyright (C) 2022-Present  DanLing

# This program is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from functools import partial

from lazy_imports import try_import

from .average_meter import AverageMeter, AverageMeters, MultiTaskAverageMeters

with try_import() as lazy_import:
    from .functional import accuracy, auprc, auroc, matthews_corrcoef, pearson, r2_score, rmse, spearman
    from .metrics import Metrics, MultiTaskMetrics

__all__ = [
    "Metrics",
    "MultiTaskMetrics",
    "AverageMeter",
    "AverageMeters",
    "MultiTaskAverageMeters",
    "regression_metrics",
    "binary_metrics",
    "multiclass_metrics",
    "multilabel_metrics",
]


def binary_metrics(**kwargs):
    lazy_import.check()
    return Metrics(auroc=auroc, auprc=auprc, acc=accuracy, mcc=matthews_corrcoef, **kwargs)


def multiclass_metrics(num_classes: int, **kwargs):
    lazy_import.check()
    p_mcc = partial(matthews_corrcoef, num_classes=num_classes)
    p_auroc = partial(auroc, num_classes=num_classes)
    p_auprc = partial(auprc, num_classes=num_classes)
    p_acc = partial(accuracy, num_classes=num_classes)
    return Metrics(auroc=p_auroc, auprc=p_auprc, acc=p_acc, mcc=p_mcc, **kwargs)


def multilabel_metrics(num_labels: int, **kwargs):
    lazy_import.check()
    p_mcc = partial(matthews_corrcoef, num_labels=num_labels)
    p_auroc = partial(auroc, num_labels=num_labels)
    p_auprc = partial(auprc, num_labels=num_labels)
    p_acc = partial(accuracy, num_labels=num_labels)
    return Metrics(auroc=p_auroc, auprc=p_auprc, acc=p_acc, mcc=p_mcc, **kwargs)


def regression_metrics(**kwargs):
    lazy_import.check()
    return Metrics(
        pearson=pearson,
        spearman=spearman,
        r2=r2_score,
        rmse=rmse,
        **kwargs,
    )
