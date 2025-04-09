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

from .binary import binary_accuracy, binary_auprc, binary_auroc, binary_f1_score
from .classification import accuracy, auprc, auroc, f1_score, mcc
from .multiclass import multiclass_accuracy, multiclass_auprc, multiclass_auroc, multiclass_f1_score
from .multilabel import multilabel_accuracy, multilabel_auprc, multilabel_auroc, multilabel_f1_score
from .regression import mse, pearson, r2_score, rmse, spearman

__all__ = [
    "accuracy",
    "auprc",
    "auroc",
    "f1_score",
    "mcc",
    "mse",
    "pearson",
    "r2_score",
    "rmse",
    "spearman",
    "binary_accuracy",
    "binary_auprc",
    "binary_auroc",
    "binary_f1_score",
    "multiclass_accuracy",
    "multiclass_auprc",
    "multiclass_auroc",
    "multiclass_f1_score",
    "multilabel_accuracy",
    "multilabel_auprc",
    "multilabel_auroc",
    "multilabel_f1_score",
]
