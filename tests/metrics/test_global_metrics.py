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

import os
import random
import socket
from functools import partial

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn
from torch.testing import assert_close
from torchmetrics.functional import classification as tmfc

from danling import NestedTensor
from danling.metrics.functional import (
    binary_accuracy,
    binary_auroc,
    multiclass_accuracy,
    multiclass_balanced_accuracy,
    multiclass_f1_score,
    multiclass_hamming_loss,
    multiclass_iou,
    multiclass_jaccard_index,
    multiclass_precision,
    multiclass_recall,
    multiclass_specificity,
    multilabel_accuracy,
    multilabel_balanced_accuracy,
    multilabel_f1_score,
    multilabel_hamming_loss,
    multilabel_iou,
    multilabel_jaccard_index,
    multilabel_precision,
    multilabel_recall,
    multilabel_specificity,
)
from danling.metrics.global_metrics import GlobalMetrics
from danling.metrics.preprocess import preprocess_binary, preprocess_multiclass, preprocess_multilabel

from .helpers import (
    ATOL,
    RTOL,
    assert_metric_outputs,
    build_function_map,
    make_binary_metrics,
    make_multiclass_metrics,
    make_multilabel_metrics,
    make_regression_metrics,
)


def test_binary_metrics_value_and_average():
    random.seed(0)
    torch.random.manual_seed(0)
    metrics = make_binary_metrics(distributed=False)
    function_map = build_function_map("binary")
    preds, targets = [], []

    for _ in range(4):
        logits = torch.randn(8)
        raw_target = torch.randint(2, (8,))
        pred, target = preprocess_binary(logits, raw_target)
        preds.append(pred)
        targets.append(target)
        metrics.update(logits, raw_target)

        assert_metric_outputs(metrics.value(), function_map, metrics._last_preds, metrics._last_targets)
        assert_metric_outputs(metrics.average(), function_map, metrics.preds, metrics.targets)

    assert_close(metrics.preds, torch.cat(preds), rtol=RTOL, atol=ATOL)
    assert_close(metrics.targets, torch.cat(targets), rtol=RTOL, atol=ATOL)


def test_multiclass_metrics_value_and_average():
    random.seed(0)
    torch.random.manual_seed(0)
    num_classes = 10
    metrics = make_multiclass_metrics(num_classes=num_classes, distributed=False)
    function_map = build_function_map("multiclass", num_classes=num_classes)
    preds, targets = [], []

    for _ in range(4):
        logits = torch.randn(8, num_classes)
        target = torch.randint(num_classes, (8,))
        pred, target = preprocess_multiclass(logits, target, num_classes=num_classes)
        preds.append(pred)
        targets.append(target)
        metrics.update(logits, target)

        assert_metric_outputs(metrics.value(), function_map, metrics._last_preds, metrics._last_targets)
        assert_metric_outputs(metrics.average(), function_map, metrics.preds, metrics.targets)

    assert_close(metrics.preds, torch.cat(preds), rtol=RTOL, atol=ATOL)
    assert_close(metrics.targets, torch.cat(targets), rtol=RTOL, atol=ATOL)


def test_multiclass_accuracy_supports_k():
    random.seed(0)
    torch.random.manual_seed(0)
    num_classes = 6
    k = 3
    metrics = GlobalMetrics(
        [multiclass_accuracy(num_classes=num_classes, k=k)],
        preprocess=partial(preprocess_multiclass, num_classes=num_classes),
        distributed=False,
    )
    preds, targets = [], []

    for _ in range(4):
        logits = torch.randn(8, num_classes)
        target = torch.randint(num_classes, (8,))
        pred, target = preprocess_multiclass(logits, target, num_classes=num_classes)
        preds.append(pred)
        targets.append(target)
        metrics.update(logits, target)

        expected_last = tmfc.multiclass_accuracy(
            metrics._last_preds,
            metrics._last_targets,
            num_classes=num_classes,
            average="macro",
            top_k=k,
        )
        assert_close(
            torch.as_tensor(metrics.value()["acc"]),
            torch.as_tensor(expected_last),
            rtol=RTOL,
            atol=ATOL,
            check_dtype=False,
        )

    expected_avg = tmfc.multiclass_accuracy(
        torch.cat(preds),
        torch.cat(targets),
        num_classes=num_classes,
        average="macro",
        top_k=k,
    )
    assert_close(
        torch.as_tensor(metrics.average()["acc"]),
        torch.as_tensor(expected_avg),
        rtol=RTOL,
        atol=ATOL,
        check_dtype=False,
    )


def test_multilabel_metrics_value_and_average():
    random.seed(0)
    torch.random.manual_seed(0)
    num_labels = 10
    metrics = make_multilabel_metrics(num_labels=num_labels, distributed=False)
    function_map = build_function_map("multilabel", num_labels=num_labels)
    preds, targets = [], []

    for _ in range(4):
        logits = torch.randn(8, num_labels)
        target = torch.randint(2, (8, num_labels))
        pred, target = preprocess_multilabel(logits, target, num_labels=num_labels)
        preds.append(pred)
        targets.append(target)
        metrics.update(logits, target)

        assert_metric_outputs(metrics.value(), function_map, metrics._last_preds, metrics._last_targets)
        assert_metric_outputs(metrics.average(), function_map, metrics.preds, metrics.targets)

    assert_close(metrics.preds, torch.cat(preds), rtol=RTOL, atol=ATOL)
    assert_close(metrics.targets, torch.cat(targets), rtol=RTOL, atol=ATOL)


def test_regression_metrics_with_nested_tensors():
    random.seed(0)
    torch.random.manual_seed(0)
    num_outputs = 2
    metrics = make_regression_metrics(num_outputs=num_outputs, distributed=False)
    function_map = build_function_map("regression", num_outputs=num_outputs)
    preds, targets = [], []

    lengths_list = [(2, 3, 5), (7, 11)]
    for lengths in lengths_list:
        pred_list, target_list = [], []
        for length in lengths:
            pred_list.append(torch.randn(length, num_outputs))
            target_list.append(torch.randn(length, num_outputs))
        preds.extend(pred_list)
        targets.extend(target_list)
        pred_nt, target_nt = NestedTensor(pred_list), NestedTensor(target_list)
        metrics.update(pred_nt, target_nt)

        assert_metric_outputs(metrics.value(), function_map, metrics._last_preds, metrics._last_targets)
        assert_metric_outputs(metrics.average(), function_map, metrics.preds, metrics.targets)

    assert_close(metrics.preds, torch.cat(preds), rtol=RTOL, atol=ATOL)
    assert_close(metrics.targets, torch.cat(targets), rtol=RTOL, atol=ATOL)


def test_average_compacts_buffers_once():
    metrics = make_binary_metrics(distributed=False)
    for _ in range(3):
        logits = torch.randn(6)
        target = torch.randint(2, (6,))
        metrics.update(logits, target)

    assert len(metrics._preds) == 3
    assert len(metrics._targets) == 3

    _ = metrics.average()
    assert len(metrics._preds) == 1
    assert len(metrics._targets) == 1

    version = metrics._artifact_version
    first = metrics.average()
    second = metrics.average()
    assert metrics._artifact_version == version
    for key in first:
        assert_close(torch.as_tensor(first[key]), torch.as_tensor(second[key]), rtol=RTOL, atol=ATOL, check_dtype=False)


def test_global_metrics_support_keyword_metric_descriptors():
    metrics = GlobalMetrics(
        preprocess=preprocess_binary,
        distributed=False,
        auroc=binary_auroc(),
        acc=binary_accuracy(),
    )
    metrics.update(torch.randn(8), torch.randint(2, (8,)))
    assert {"auroc", "acc"} <= set(metrics.avg.keys())


def test_multiclass_confmat_metrics_with_ignore_index():
    random.seed(0)
    torch.random.manual_seed(0)
    num_classes = 7
    ignore_index = -100
    average = "macro"
    metrics = GlobalMetrics(
        [
            multiclass_accuracy(num_classes=num_classes, average=average, ignore_index=ignore_index),
            multiclass_precision(num_classes=num_classes, average=average, ignore_index=ignore_index),
            multiclass_recall(num_classes=num_classes, average=average, ignore_index=ignore_index),
            multiclass_specificity(num_classes=num_classes, average=average, ignore_index=ignore_index),
            multiclass_balanced_accuracy(num_classes=num_classes, average=average, ignore_index=ignore_index),
            multiclass_jaccard_index(num_classes=num_classes, average=average, ignore_index=ignore_index),
            multiclass_iou(num_classes=num_classes, average=average, ignore_index=ignore_index),
            multiclass_hamming_loss(num_classes=num_classes, average=average, ignore_index=ignore_index),
            multiclass_f1_score(num_classes=num_classes, average=average, ignore_index=ignore_index),
        ],
        preprocess=partial(preprocess_multiclass, num_classes=num_classes, ignore_index=ignore_index),
        distributed=False,
    )
    function_map = {
        "acc": partial(tmfc.multiclass_accuracy, num_classes=num_classes, average=average, ignore_index=ignore_index),
        "precision": partial(
            tmfc.multiclass_precision,
            num_classes=num_classes,
            average=average,
            ignore_index=ignore_index,
        ),
        "recall": partial(tmfc.multiclass_recall, num_classes=num_classes, average=average, ignore_index=ignore_index),
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
            tmfc.multiclass_jaccard_index, num_classes=num_classes, average=average, ignore_index=ignore_index
        ),
        "hamming_loss": partial(
            tmfc.multiclass_hamming_distance,
            num_classes=num_classes,
            average=average,
            ignore_index=ignore_index,
        ),
        "f1": partial(tmfc.multiclass_f1_score, num_classes=num_classes, average=average, ignore_index=ignore_index),
    }
    preds, targets = [], []

    for _ in range(4):
        logits = torch.randn(16, num_classes)
        target = torch.randint(num_classes, (16,))
        target[torch.rand(16) < 0.25] = ignore_index
        pred, processed_target = preprocess_multiclass(
            logits, target, num_classes=num_classes, ignore_index=ignore_index
        )
        preds.append(pred)
        targets.append(processed_target)
        metrics.update(logits, target)
        assert_metric_outputs(metrics.value(), function_map, metrics._last_preds, metrics._last_targets)
        assert_metric_outputs(metrics.average(), function_map, torch.cat(preds), torch.cat(targets))


def test_multilabel_confmat_metrics_with_elementwise_ignore_index():
    random.seed(0)
    torch.random.manual_seed(0)
    num_labels = 6
    ignore_index = -100
    average = "macro"
    metrics = GlobalMetrics(
        [
            multilabel_accuracy(num_labels=num_labels, average=average, ignore_index=ignore_index),
            multilabel_precision(num_labels=num_labels, average=average, ignore_index=ignore_index),
            multilabel_recall(num_labels=num_labels, average=average, ignore_index=ignore_index),
            multilabel_specificity(num_labels=num_labels, average=average, ignore_index=ignore_index),
            multilabel_balanced_accuracy(num_labels=num_labels, average=average, ignore_index=ignore_index),
            multilabel_jaccard_index(num_labels=num_labels, average=average, ignore_index=ignore_index),
            multilabel_iou(num_labels=num_labels, average=average, ignore_index=ignore_index),
            multilabel_hamming_loss(num_labels=num_labels, average=average, ignore_index=ignore_index),
            multilabel_f1_score(num_labels=num_labels, average=average, ignore_index=ignore_index),
        ],
        preprocess=partial(preprocess_multilabel, num_labels=num_labels, ignore_index=ignore_index),
        distributed=False,
    )
    function_map = {
        "acc": partial(tmfc.multilabel_accuracy, num_labels=num_labels, average=average, ignore_index=ignore_index),
        "precision": partial(
            tmfc.multilabel_precision, num_labels=num_labels, average=average, ignore_index=ignore_index
        ),
        "recall": partial(tmfc.multilabel_recall, num_labels=num_labels, average=average, ignore_index=ignore_index),
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
            tmfc.multilabel_jaccard_index, num_labels=num_labels, average=average, ignore_index=ignore_index
        ),
        "iou": partial(
            tmfc.multilabel_jaccard_index, num_labels=num_labels, average=average, ignore_index=ignore_index
        ),
        "hamming_loss": partial(
            tmfc.multilabel_hamming_distance,
            num_labels=num_labels,
            average=average,
            ignore_index=ignore_index,
        ),
        "f1": partial(tmfc.multilabel_f1_score, num_labels=num_labels, average=average, ignore_index=ignore_index),
    }
    preds, targets = [], []

    for _ in range(4):
        logits = torch.randn(16, num_labels)
        target = torch.randint(2, (16, num_labels))
        target[torch.rand_like(target.float()) < 0.2] = ignore_index
        pred, processed_target = preprocess_multilabel(logits, target, num_labels=num_labels, ignore_index=ignore_index)
        preds.append(pred)
        targets.append(processed_target)
        metrics.update(logits, target)
        assert_metric_outputs(metrics.value(), function_map, metrics._last_preds, metrics._last_targets)
        assert_metric_outputs(metrics.average(), function_map, torch.cat(preds), torch.cat(targets))


def test_binary_metrics_distributed_sync():
    _run_distributed(_distributed_binary_worker, world_size=2)


def _run_distributed(func, world_size: int = 2):
    if not _can_bind_localhost():
        pytest.skip("Local TCP sockets are unavailable in this environment.")

    random.seed(0)
    torch.random.manual_seed(0)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(29531)
    spawn(func, args=(world_size,), nprocs=world_size)


def _can_bind_localhost() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
    except OSError:
        return False
    return True


def _distributed_binary_worker(rank: int, world_size: int):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(1234 + rank)
    metrics = make_binary_metrics(distributed=True)
    function_map = build_function_map("binary")

    for step in range(2):
        length = 4 + rank + step
        logits = torch.randn(length)
        target = torch.randint(2, (length,))
        metrics.update(logits, target)

        assert_metric_outputs(metrics.value(), function_map, metrics._last_preds, metrics._last_targets)
        assert_metric_outputs(metrics.average(), function_map, metrics.preds, metrics.targets)

    dist.destroy_process_group()
