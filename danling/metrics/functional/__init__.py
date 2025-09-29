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

from .binary import (
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
from .classification import (
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
from .multiclass import (
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
from .multilabel import (
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
from .regression import mae, mse, pearson, r2_score, rmse, spearman
from .utils import MetricFunc

__all__ = [
    "MetricFunc",
    "binary_accuracy",
    "binary_auprc",
    "binary_auroc",
    "binary_balanced_accuracy",
    "binary_f1",
    "binary_fbeta",
    "binary_hamming_loss",
    "binary_iou",
    "binary_jaccard_index",
    "binary_precision",
    "binary_recall",
    "binary_specificity",
    "accuracy",
    "auprc",
    "auroc",
    "balanced_accuracy",
    "f1_score",
    "fbeta_score",
    "hamming_loss",
    "iou",
    "jaccard_index",
    "precision",
    "recall",
    "specificity",
    "mcc",
    "mae",
    "mse",
    "pearson",
    "r2_score",
    "rmse",
    "spearman",
    "multiclass_accuracy",
    "multiclass_auprc",
    "multiclass_auroc",
    "multiclass_balanced_accuracy",
    "multiclass_f1_score",
    "multiclass_fbeta_score",
    "multiclass_hamming_loss",
    "multiclass_iou",
    "multiclass_jaccard_index",
    "multiclass_precision",
    "multiclass_recall",
    "multiclass_specificity",
    "multilabel_accuracy",
    "multilabel_auprc",
    "multilabel_auroc",
    "multilabel_balanced_accuracy",
    "multilabel_f1_score",
    "multilabel_fbeta_score",
    "multilabel_hamming_loss",
    "multilabel_iou",
    "multilabel_jaccard_index",
    "multilabel_precision",
    "multilabel_recall",
    "multilabel_specificity",
]
