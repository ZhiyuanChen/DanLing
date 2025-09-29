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

from collections.abc import Callable
from functools import partial
from typing import Literal, Sequence

from lazy_imports import try_import

from .preprocess import (
    preprocess_binary,
    preprocess_multiclass,
    preprocess_multilabel,
    preprocess_regression,
)
from .stream_metrics import StreamMetrics

with try_import() as lazy_import:
    from .functional import (
        MetricFunc,
        binary_accuracy,
        binary_auprc,
        binary_auroc,
        binary_f1,
        mcc,
        mse,
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
    from .global_metrics import GlobalMetrics

Mode = Literal["global", "stream"]


def _normalize_mode(mode: str) -> Mode:
    normalized = mode.lower()
    if normalized not in {"global", "stream"}:
        raise ValueError(f"mode must be either 'global' or 'stream', but got {mode!r}")
    return normalized  # type: ignore[return-value]


def _validate_metric_funcs(metric_funcs: Sequence[object]) -> None:
    for metric in metric_funcs:
        if not isinstance(metric, MetricFunc):
            raise ValueError(f"Expected metrics to be MetricFunc instances, got {type(metric)}")


def _build_metrics(
    mode: Mode,
    *,
    default_metric_funcs: Sequence[MetricFunc],
    custom_metrics: dict[str, MetricFunc],
    preprocess: Callable,
    custom_preprocess: Callable | None,
    metrics_funcs: Sequence[MetricFunc],
    distributed: bool,
    device,
):
    positional_metric_funcs = list(metrics_funcs) if metrics_funcs else list(default_metric_funcs)
    named_metrics = dict(custom_metrics)
    _validate_metric_funcs([*positional_metric_funcs, *named_metrics.values()])

    metrics_cls = GlobalMetrics if mode == "global" else StreamMetrics

    ctor_kwargs = {
        "preprocess": custom_preprocess or preprocess,
        "distributed": distributed,
        "device": device,
        **named_metrics,
    }
    return metrics_cls(*positional_metric_funcs, **ctor_kwargs)


def binary_metrics(
    *metrics_funcs: MetricFunc,
    mode: str = "global",
    ignore_index: int | None = -100,
    distributed: bool = True,
    device=None,
    preprocess: Callable | None = None,
    **metrics,
):
    """
    Build task-standard binary metrics.

    Args:
        mode: `"global"` for exact dataset-level metrics, `"stream"` for streaming batch-averaged metrics.
        *metrics_funcs: Custom metric functions. When provided, defaults are not added.
        ignore_index: Value in target to ignore.
        distributed: Whether global metrics should synchronise across processes.
        device: Optional storage device for global artifacts.
        preprocess: Optional preprocess override passed to the metrics constructor.
        **metrics: Custom metrics as MetricFunc descriptors.
    """
    lazy_import.check()
    mode = _normalize_mode(mode)

    default_metric_funcs = [
        binary_auroc(ignore_index=ignore_index),
        binary_auprc(ignore_index=ignore_index),
        binary_accuracy(ignore_index=ignore_index),
        binary_f1(ignore_index=ignore_index),
        mcc(task="binary", ignore_index=ignore_index),
    ]
    return _build_metrics(
        mode,
        default_metric_funcs=default_metric_funcs,
        custom_metrics=metrics,
        preprocess=partial(preprocess_binary, ignore_index=ignore_index),
        custom_preprocess=preprocess,
        metrics_funcs=metrics_funcs,
        distributed=distributed,
        device=device,
    )


def multiclass_metrics(
    num_classes: int,
    average: str = "macro",
    *metrics_funcs: MetricFunc,
    mode: str = "global",
    ignore_index: int | None = -100,
    distributed: bool = True,
    device=None,
    preprocess: Callable | None = None,
    **metrics,
):
    """
    Build task-standard multiclass metrics.

    Args:
        num_classes: Number of classes in the task.
        average: Averaging mode for multiclass metrics.
        mode: `"global"` or `"stream"`.
        *metrics_funcs: Custom metric functions. When provided, defaults are not added.
        ignore_index: Value in target to ignore.
    """
    lazy_import.check()
    mode = _normalize_mode(mode)

    default_metric_funcs = [
        multiclass_auroc(num_classes=num_classes, average=average, ignore_index=ignore_index),
        multiclass_auprc(num_classes=num_classes, average=average, ignore_index=ignore_index),
        multiclass_accuracy(num_classes=num_classes, average=average, ignore_index=ignore_index),
        multiclass_f1_score(num_classes=num_classes, average=average, ignore_index=ignore_index),
        mcc(task="multiclass", num_classes=num_classes, ignore_index=ignore_index),
    ]
    return _build_metrics(
        mode,
        default_metric_funcs=default_metric_funcs,
        custom_metrics=metrics,
        preprocess=partial(preprocess_multiclass, num_classes=num_classes, ignore_index=ignore_index),
        custom_preprocess=preprocess,
        metrics_funcs=metrics_funcs,
        distributed=distributed,
        device=device,
    )


def multilabel_metrics(
    num_labels: int,
    average: str = "macro",
    *metrics_funcs: MetricFunc,
    mode: str = "global",
    ignore_index: int | None = -100,
    distributed: bool = True,
    device=None,
    preprocess: Callable | None = None,
    **metrics,
):
    """
    Build task-standard multilabel metrics.

    Args:
        num_labels: Number of labels in the task.
        average: Averaging mode for multilabel metrics.
        mode: `"global"` or `"stream"`.
        *metrics_funcs: Custom metric functions. When provided, defaults are not added.
        ignore_index: Value in target to ignore.
    """
    lazy_import.check()
    mode = _normalize_mode(mode)

    default_metric_funcs = [
        multilabel_auroc(num_labels=num_labels, average=average, ignore_index=ignore_index),
        multilabel_auprc(num_labels=num_labels, average=average, ignore_index=ignore_index),
        multilabel_accuracy(num_labels=num_labels, average=average, ignore_index=ignore_index),
        multilabel_f1_score(num_labels=num_labels, average=average, ignore_index=ignore_index),
        mcc(task="multilabel", num_labels=num_labels, ignore_index=ignore_index),
    ]
    return _build_metrics(
        mode,
        default_metric_funcs=default_metric_funcs,
        custom_metrics=metrics,
        preprocess=partial(preprocess_multilabel, num_labels=num_labels, ignore_index=ignore_index),
        custom_preprocess=preprocess,
        metrics_funcs=metrics_funcs,
        distributed=distributed,
        device=device,
    )


def regression_metrics(
    num_outputs: int = 1,
    ignore_nan: bool = True,
    *metrics_funcs: MetricFunc,
    mode: str = "global",
    distributed: bool = True,
    device=None,
    preprocess: Callable | None = None,
    **metrics,
):
    """
    Build task-standard regression metrics.

    Args:
        num_outputs: Number of regression outputs.
        ignore_nan: Whether to mask NaNs in targets.
        mode: `"global"` or `"stream"`.
        *metrics_funcs: Custom metric functions. When provided, defaults are not added.
    """
    lazy_import.check()
    mode = _normalize_mode(mode)

    default_metric_funcs = [
        pearson(),
        spearman(),
        r2_score(),
        mse(num_outputs=num_outputs),
        rmse(num_outputs=num_outputs),
    ]
    return _build_metrics(
        mode,
        default_metric_funcs=default_metric_funcs,
        custom_metrics=metrics,
        preprocess=partial(preprocess_regression, num_outputs=num_outputs, ignore_nan=ignore_nan),
        custom_preprocess=preprocess,
        metrics_funcs=metrics_funcs,
        distributed=distributed,
        device=device,
    )
