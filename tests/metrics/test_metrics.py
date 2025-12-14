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

import os
import random
from functools import partial

import pytest
import torch
import torchmetrics.functional as tmf
from torch import distributed as dist
from torch.multiprocessing import spawn
from torch.testing import assert_close
from torchmetrics.functional import classification as tmfc
from torchmetrics.functional import matthews_corrcoef

from danling import NestedTensor
from danling.metrics.functions import (
    MSE,
    RMSE,
    BinaryAccuracy,
    BinaryAUPRC,
    BinaryAUROC,
    BinaryF1Score,
    BinaryMCC,
    MulticlassAccuracy,
    MulticlassAUPRC,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassMCC,
    MultilabelAccuracy,
    MultilabelAUPRC,
    MultilabelAUROC,
    MultilabelF1Score,
    MultilabelMCC,
    Pearson,
)
from danling.metrics.functions import R2Score as MetricR2Score
from danling.metrics.functions import (
    Spearman,
)
from danling.metrics.metrics import ClassificationMetrics, RegressionMetrics
from danling.metrics.preprocess import (
    preprocess_binary,
    preprocess_multiclass,
    preprocess_multilabel,
    preprocess_regression,
)

ATOL = 1e-6
RTOL = 1e-5


def _ensure_multidim(metric, value: str = "global"):
    if not hasattr(metric, "multidim_average"):
        setattr(metric, "multidim_average", value)
    return metric


def build_function_map(
    task: str, num_classes: int = 10, num_labels: int = 10, num_outputs: int = 1, average: str = "macro"
):
    if task == "binary":
        return {
            "auroc": tmfc.binary_auroc,
            "auprc": tmfc.binary_average_precision,
            "acc": partial(tmfc.binary_accuracy, multidim_average="global"),
            "f1": partial(tmfc.binary_f1_score, multidim_average="global"),
            "mcc": partial(matthews_corrcoef, task="binary"),
        }
    if task == "multiclass":
        return {
            "auroc": partial(tmfc.multiclass_auroc, num_classes=num_classes, average=average),
            "auprc": partial(tmfc.multiclass_average_precision, num_classes=num_classes, average=average),
            "acc": partial(
                tmfc.multiclass_accuracy, num_classes=num_classes, average=average, multidim_average="global"
            ),
            "f1": partial(
                tmfc.multiclass_f1_score, num_classes=num_classes, average=average, multidim_average="global"
            ),
            "mcc": partial(matthews_corrcoef, task="multiclass", num_classes=num_classes),
        }
    if task == "multilabel":
        return {
            "auroc": partial(tmfc.multilabel_auroc, num_labels=num_labels, average=average),
            "auprc": partial(tmfc.multilabel_average_precision, num_labels=num_labels, average=average),
            "acc": partial(tmfc.multilabel_accuracy, num_labels=num_labels, average=average, multidim_average="global"),
            "f1": partial(tmfc.multilabel_f1_score, num_labels=num_labels, average=average, multidim_average="global"),
            "mcc": partial(matthews_corrcoef, task="multilabel", num_labels=num_labels),
        }
    if task == "regression":
        return {
            "pearson": lambda p, t: tmf.pearson_corrcoef(p, t).mean(),
            "spearman": lambda p, t: tmf.spearman_corrcoef(p, t).mean(),
            "r2": lambda p, t: tmf.r2_score(p, t),
            "mse": lambda p, t: tmf.mean_squared_error(p, t, squared=True, num_outputs=num_outputs),
            "rmse": lambda p, t: tmf.mean_squared_error(p, t, squared=False, num_outputs=num_outputs),
        }
    raise ValueError(f"Unsupported task: {task}")


def make_binary_metrics(threshold: float = 0.5, *, distributed: bool = False):
    acc = _ensure_multidim(BinaryAccuracy(threshold=threshold, multidim_average="global"))
    f1 = _ensure_multidim(BinaryF1Score(threshold=threshold, multidim_average="global"))
    funcs = [
        acc,
        f1,
        BinaryAUROC(),
        BinaryAUPRC(),
        BinaryMCC(threshold=threshold),
    ]
    return ClassificationMetrics(funcs, preprocess=preprocess_binary, distributed=distributed)


def make_multiclass_metrics(num_classes: int, average: str = "macro", *, distributed: bool = False):
    acc = _ensure_multidim(MulticlassAccuracy(num_classes=num_classes, average=average, multidim_average="global"))
    f1 = _ensure_multidim(MulticlassF1Score(num_classes=num_classes, average=average, multidim_average="global"))
    funcs = [
        MulticlassAUROC(num_classes=num_classes, average=average),
        MulticlassAUPRC(num_classes=num_classes, average=average),
        acc,
        f1,
        MulticlassMCC(num_classes=num_classes),
    ]
    return ClassificationMetrics(
        funcs, preprocess=partial(preprocess_multiclass, num_classes=num_classes), distributed=distributed
    )


def make_multilabel_metrics(num_labels: int, average: str = "macro", *, distributed: bool = False):
    acc = _ensure_multidim(MultilabelAccuracy(num_labels=num_labels, average=average, multidim_average="global"))
    f1 = _ensure_multidim(MultilabelF1Score(num_labels=num_labels, average=average, multidim_average="global"))
    funcs = [
        MultilabelAUROC(num_labels=num_labels, average=average),
        MultilabelAUPRC(num_labels=num_labels, average=average),
        acc,
        f1,
        MultilabelMCC(num_labels=num_labels),
    ]
    return ClassificationMetrics(
        funcs, preprocess=partial(preprocess_multilabel, num_labels=num_labels), distributed=distributed
    )


def make_regression_metrics(num_outputs: int = 1, *, distributed: bool = False):
    funcs = [
        Pearson(),
        Spearman(),
        MetricR2Score(),
        MSE(num_outputs=num_outputs),
        RMSE(num_outputs=num_outputs),
    ]
    return RegressionMetrics(
        funcs,
        preprocess=partial(preprocess_regression, num_outputs=num_outputs, ignore_nan=True),
        distributed=distributed,
    )


def test_descriptor_metrics_binary_accuracy():
    metrics = ClassificationMetrics(
        [BinaryAccuracy(threshold=0.5, multidim_average="global")], preprocess=preprocess_binary, distributed=False
    )
    first_pred = torch.tensor([0.2, 0.7, 0.9, 0.1])
    first_target = torch.tensor([0, 1, 1, 0])
    metrics.update(first_pred, first_target)
    expected_first = tmfc.binary_accuracy(first_pred, first_target)
    assert_close(torch.as_tensor(metrics.val["acc"]), expected_first)
    assert_close(torch.as_tensor(metrics.avg["acc"]), expected_first)

    second_pred = torch.tensor([0.6, 0.4, 0.2, 0.9])
    second_target = torch.tensor([1, 0, 0, 1])
    metrics.update(second_pred, second_target)
    expected_all = tmfc.binary_accuracy(torch.cat([first_pred, second_pred]), torch.cat([first_target, second_target]))
    assert_close(torch.as_tensor(metrics.avg["acc"]), expected_all)
    assert metrics.confmat is not None


def test_descriptor_metrics_combined_binary():
    metrics = make_binary_metrics()
    preds = torch.tensor([0.1, 0.6, 0.9, 0.3, 0.8])
    targets = torch.tensor([0, 1, 1, 0, 1])
    metrics.update(preds, targets)

    assert metrics.plan.confmat is True
    assert metrics.plan.need_preds_targets is True
    value = metrics.avg
    assert_close(torch.as_tensor(value["acc"]), tmfc.binary_accuracy(preds, targets))
    assert_close(torch.as_tensor(value["f1"]), tmfc.binary_f1_score(preds, targets))
    assert_close(torch.as_tensor(value["auroc"]), tmfc.binary_auroc(preds, targets), check_dtype=False)
    assert_close(torch.as_tensor(value["auprc"]), tmfc.binary_average_precision(preds, targets), check_dtype=False)
    assert_close(torch.as_tensor(value["mcc"]), matthews_corrcoef(preds, targets, task="binary"))


def test_tensor_regression():
    random.seed(0)
    torch.random.manual_seed(0)
    metrics = make_regression_metrics()
    function_map = build_function_map("regression")
    preds, targets = [], []
    for _ in range(10):
        pred = torch.randn(8, 1)
        target = torch.randn(8)
        preds.append(pred)
        targets.append(target)
        metrics.update(pred, target)
        value, average = metrics.value(), metrics.average()
        assert torch.allclose(pred.flatten(), metrics._last_preds.flatten(), rtol=RTOL, atol=ATOL)
        assert torch.allclose(target.flatten(), metrics._last_targets.flatten(), rtol=RTOL, atol=ATOL)
        for key, func in function_map.items():
            assert_close(torch.as_tensor(value[key]), torch.as_tensor(func(metrics._last_preds, metrics._last_targets)))
        for key, func in function_map.items():
            assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))
    assert torch.allclose(torch.cat(preds).flatten(), metrics.preds.flatten(), rtol=RTOL, atol=ATOL)
    assert torch.allclose(torch.cat(targets).flatten(), metrics.targets.flatten(), rtol=RTOL, atol=ATOL)
    average = metrics.average()
    for key, func in function_map.items():
        assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))


def test_nested_tensor_regression():
    random.seed(0)
    torch.random.manual_seed(0)
    num_outputs = 2
    metrics = make_regression_metrics(num_outputs=num_outputs)
    function_map = build_function_map("regression", num_outputs=num_outputs)
    preds, targets = [], []
    lengths_list = [(2, 3, 5, 7), (11, 13, 17, 19)]
    for lengths in lengths_list:
        pred_list, target_list = [], []
        for length in lengths:
            pred_list.append(torch.randn(length, num_outputs))
            target_list.append(torch.randn(length, num_outputs))
        preds.extend(pred_list)
        targets.extend(target_list)
        pred_nt, target_nt = NestedTensor(pred_list), NestedTensor(target_list)
        metrics.update(pred_nt.tensor, target_nt)
        value, average = metrics.value(), metrics.average()
        assert torch.allclose(pred_nt.concat, metrics._last_preds, rtol=RTOL, atol=ATOL)
        assert torch.allclose(target_nt.concat, metrics._last_targets, rtol=RTOL, atol=ATOL)
        for key, func in function_map.items():
            assert_close(torch.as_tensor(value[key]), torch.as_tensor(func(metrics._last_preds, metrics._last_targets)))
        for key, func in function_map.items():
            assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))
    assert torch.allclose(torch.cat(preds), metrics.preds, rtol=RTOL, atol=ATOL)
    assert torch.allclose(torch.cat(targets), metrics.targets, rtol=RTOL, atol=ATOL)
    average = metrics.average()
    for key, func in function_map.items():
        assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))


def test_tensor_binary():
    random.seed(0)
    torch.random.manual_seed(0)
    metrics = make_binary_metrics(distributed=False)
    function_map = build_function_map("binary")
    preds, targets = [], []
    for _ in range(10):
        logits = torch.randn(8)
        pred = logits.sigmoid()
        target = torch.randint(2, (8,))
        preds.append(pred)
        targets.append(target)
        metrics.update(logits, target)
        value, average = metrics.value(), metrics.average()
        assert torch.allclose(pred, metrics._last_preds, rtol=RTOL, atol=ATOL)
        assert torch.allclose(target, metrics._last_targets, rtol=RTOL, atol=ATOL)
        for key, func in function_map.items():
            assert_close(torch.as_tensor(value[key]), torch.as_tensor(func(metrics._last_preds, metrics._last_targets)))
        for key, func in function_map.items():
            assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))
    assert torch.allclose(torch.cat(preds), metrics.preds, rtol=RTOL, atol=ATOL)
    assert torch.allclose(torch.cat(targets), metrics.targets, rtol=RTOL, atol=ATOL)
    average = metrics.average()
    for key, func in function_map.items():
        assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))


def test_nested_tensor_binary():
    random.seed(0)
    torch.random.manual_seed(0)
    metrics = make_binary_metrics(distributed=False)
    function_map = build_function_map("binary")
    preds, targets = [], []
    lengths_list = [(2, 3, 5, 7), (11, 13, 17, 19)]
    for lengths in lengths_list:
        pred_list, target_list = [], []
        for length in lengths:
            pred_list.append(torch.randn(length).sigmoid())
            target_list.append(torch.randint(2, (length,)))
        preds.extend(pred_list)
        targets.extend(target_list)
        pred_nt, target_nt = NestedTensor(pred_list), NestedTensor(target_list)
        metrics.update(pred_nt, target_nt.tensor)
        value, average = metrics.value(), metrics.average()
        assert torch.allclose(pred_nt.concat, metrics._last_preds, rtol=RTOL, atol=ATOL)
        assert torch.allclose(target_nt.concat, metrics._last_targets, rtol=RTOL, atol=ATOL)
        for key, func in function_map.items():
            assert_close(torch.as_tensor(value[key]), torch.as_tensor(func(metrics._last_preds, metrics._last_targets)))
        for key, func in function_map.items():
            assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))
    assert torch.allclose(torch.cat(preds), metrics.preds, rtol=RTOL, atol=ATOL)
    assert torch.allclose(torch.cat(targets), metrics.targets, rtol=RTOL, atol=ATOL)
    average = metrics.average()
    for key, func in function_map.items():
        assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))


def test_tensor_multiclass():
    random.seed(0)
    torch.random.manual_seed(0)
    num_classes = 10
    metrics = make_multiclass_metrics(num_classes=num_classes, distributed=False)
    function_map = build_function_map("multiclass", num_classes=num_classes)
    preds, targets = [], []
    for _ in range(10):
        logits = torch.randn(8, num_classes)
        pred = logits.softmax(-1)
        target = torch.randint(num_classes, (8,))
        preds.append(pred)
        targets.append(target)
        metrics.update(logits, target)
        value, average = metrics.value(), metrics.average()
        assert torch.allclose(pred, metrics._last_preds, rtol=RTOL, atol=ATOL)
        assert torch.allclose(target, metrics._last_targets, rtol=RTOL, atol=ATOL)
        for key, func in function_map.items():
            assert_close(torch.as_tensor(value[key]), torch.as_tensor(func(metrics._last_preds, metrics._last_targets)))
        for key, func in function_map.items():
            assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))
    metrics_inputs, metrics_targets = metrics.preds, metrics.targets
    assert torch.allclose(torch.cat(preds), metrics_inputs, rtol=RTOL, atol=ATOL)
    assert torch.allclose(torch.cat(targets), metrics_targets, rtol=RTOL, atol=ATOL)
    average = metrics.average()
    for key, func in function_map.items():
        assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))


def test_nested_tensor_multiclass():
    random.seed(0)
    torch.random.manual_seed(0)
    num_classes = 10
    metrics = make_multiclass_metrics(num_classes=num_classes, distributed=False)
    function_map = build_function_map("multiclass", num_classes=num_classes)
    preds, targets = [], []
    lengths_list = [(2, 3, 5, 7), (11, 13, 17, 19)]
    for lengths in lengths_list:
        pred_list, target_list = [], []
        for length in lengths:
            pred_list.append(torch.randn(length, num_classes).softmax(-1))
            target_list.append(torch.randint(num_classes, (length,)))
        preds.extend(pred_list)
        targets.extend(target_list)
        pred_nt, target_nt = NestedTensor(pred_list), NestedTensor(target_list)
        metrics.update(pred_nt, target_nt)
        value, average = metrics.value(), metrics.average()
        assert torch.allclose(pred_nt.concat, metrics._last_preds, rtol=RTOL, atol=ATOL)
        assert torch.allclose(target_nt.concat, metrics._last_targets, rtol=RTOL, atol=ATOL)
        for key, func in function_map.items():
            assert_close(torch.as_tensor(value[key]), torch.as_tensor(func(metrics._last_preds, metrics._last_targets)))
        for key, func in function_map.items():
            assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))
    assert torch.allclose(torch.cat(preds), metrics.preds, rtol=RTOL, atol=ATOL)
    assert torch.allclose(torch.cat(targets), metrics.targets, rtol=RTOL, atol=ATOL)
    average = metrics.average()
    for key, func in function_map.items():
        assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))


def test_tensor_multilabel():
    random.seed(0)
    torch.random.manual_seed(0)
    num_labels = 10
    metrics = make_multilabel_metrics(num_labels=num_labels, distributed=False)
    function_map = build_function_map("multilabel", num_labels=num_labels)
    preds, targets = [], []
    for _ in range(10):
        logits = torch.randn(8, num_labels)
        pred = logits.sigmoid()
        target = torch.randint(2, (8, num_labels))
        preds.append(pred)
        targets.append(target)
        metrics.update(logits, target)
        value, average = metrics.value(), metrics.average()
        assert torch.allclose(pred, metrics._last_preds, rtol=RTOL, atol=ATOL)
        assert torch.allclose(target, metrics._last_targets, rtol=RTOL, atol=ATOL)
        for key, func in function_map.items():
            assert_close(torch.as_tensor(value[key]), torch.as_tensor(func(metrics._last_preds, metrics._last_targets)))
        for key, func in function_map.items():
            assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))
    metrics_inputs, metrics_targets = metrics.preds, metrics.targets
    assert torch.allclose(torch.cat(preds), metrics_inputs, rtol=RTOL, atol=ATOL)
    assert torch.allclose(torch.cat(targets), metrics_targets, rtol=RTOL, atol=ATOL)
    average = metrics.average()
    for key, func in function_map.items():
        assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))


def test_nested_tensor_multilabel():
    random.seed(0)
    torch.random.manual_seed(0)
    num_labels = 10
    metrics = make_multilabel_metrics(num_labels=num_labels, distributed=False)
    function_map = build_function_map("multilabel", num_labels=num_labels)
    preds, targets = [], []
    lengths_list = [(2, 3, 5, 7), (11, 13, 17, 19)]
    for lengths in lengths_list:
        pred_list, target_list = [], []
        for length in lengths:
            pred_list.append(torch.randn(length, num_labels).sigmoid())
            target_list.append(torch.randint(2, (length, num_labels)))
        preds.extend(pred_list)
        targets.extend(target_list)
        pred_nt, target_nt = NestedTensor(pred_list), NestedTensor(target_list)
        metrics.update(pred_nt, target_nt)
        value, average = metrics.value(), metrics.average()
        assert torch.allclose(pred_nt.concat, metrics._last_preds, rtol=RTOL, atol=ATOL)
        assert torch.allclose(target_nt.concat, metrics._last_targets, rtol=RTOL, atol=ATOL)
        for key, func in function_map.items():
            assert_close(torch.as_tensor(value[key]), torch.as_tensor(func(metrics._last_preds, metrics._last_targets)))
        for key, func in function_map.items():
            assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))
    assert torch.allclose(torch.cat(preds), metrics.preds, rtol=RTOL, atol=ATOL)
    assert torch.allclose(torch.cat(targets), metrics.targets, rtol=RTOL, atol=ATOL)
    average = metrics.average()
    for key, func in function_map.items():
        assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))


def _test_distributed(func: callable, world_size: int = 8):
    random.seed(0)
    torch.random.manual_seed(0)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(29501)
    spawn(func, args=(world_size,), nprocs=world_size)


def test_tensor_binary_distributed(world_size: int = 8):
    _test_distributed(_test_tensor_binary, world_size)


def _test_tensor_binary(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    metrics = make_binary_metrics(distributed=True)
    function_map = build_function_map("binary")
    length = 8
    for _ in range(8):
        pred = torch.randn(length).sigmoid()
        target = torch.randint(2, (length,))
        metrics.update(pred, target)
        value, average = metrics.value(), metrics.average()
        for key, func in function_map.items():
            assert_close(torch.as_tensor(value[key]), torch.as_tensor(func(metrics._last_preds, metrics._last_targets)))
        for key, func in function_map.items():
            assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))
    final_avg = metrics.average()
    for key, func in function_map.items():
        assert_close(torch.as_tensor(final_avg[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))

    dist.destroy_process_group()


def test_nested_tensor_binary_distributed(world_size: int = 8):
    _test_distributed(_test_nested_tensor_binary, world_size)


def _test_nested_tensor_binary(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    metrics = make_binary_metrics(distributed=True)
    function_map = build_function_map("binary")
    lengths_list = [[2, 3, 5, 7], [11, 13, 17]]
    if rank == 0:
        lengths_list[-1].append(19)
    for lengths in lengths_list:
        pred_list, target_list = [], []
        for length in lengths:
            pred_list.append(torch.randn(length).sigmoid())
            target_list.append(torch.randint(2, (length,)))
        pred_nt, target_nt = NestedTensor(pred_list), NestedTensor(target_list)
        metrics.update(pred_nt, target_nt)
        value, average = metrics.value(), metrics.average()
        for key, func in function_map.items():
            assert_close(torch.as_tensor(value[key]), torch.as_tensor(func(metrics._last_preds, metrics._last_targets)))
        for key, func in function_map.items():
            assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))
    final_avg = metrics.average()
    for key, func in function_map.items():
        assert_close(torch.as_tensor(final_avg[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))

    dist.destroy_process_group()


def test_nested_tensor_regression_distributed(world_size: int = 8):
    _test_distributed(_test_nested_tensor_regression, world_size)


def _test_nested_tensor_regression(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    metrics = make_regression_metrics(distributed=True)
    function_map = build_function_map("regression")
    lengths_list = [[2, 3, 5, 7], [11, 13, 17]]
    if rank == 0:
        lengths_list[-1].append(19)
    for lengths in lengths_list:
        pred_list, target_list = [], []
        for length in lengths:
            pred_list.append(torch.randn(length))
            target_list.append(torch.randn(length))
        pred_nt, target_nt = NestedTensor(pred_list), NestedTensor(target_list)
        metrics.update(pred_nt, target_nt)
        value, average = metrics.value(), metrics.average()
        for key, func in function_map.items():
            assert_close(torch.as_tensor(value[key]), torch.as_tensor(func(metrics._last_preds, metrics._last_targets)))
        for key, func in function_map.items():
            assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))
    final_avg = metrics.average()
    for key, func in function_map.items():
        assert_close(torch.as_tensor(final_avg[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))

    dist.destroy_process_group()


def test_nested_tensor_multi_regression_distributed(world_size: int = 8):
    _test_distributed(_test_nested_tensor_multi_regression, world_size)


def _test_nested_tensor_multi_regression(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    num_outputs = 8
    metrics = make_regression_metrics(num_outputs=num_outputs, distributed=True)
    function_map = build_function_map("regression", num_outputs=num_outputs)
    lengths_list = [[2, 3, 5, 7], [11, 13, 17]]
    if rank == 0:
        lengths_list[-1].append(19)
    for lengths in lengths_list:
        pred_list, target_list = [], []
        for length in lengths:
            pred_list.append(torch.randn(length, num_outputs))
            target_list.append(torch.randn(length, num_outputs))
        pred_nt, target_nt = NestedTensor(pred_list), NestedTensor(target_list)
        metrics.update(pred_nt, target_nt)
        value, average = metrics.value(), metrics.average()
        for key, func in function_map.items():
            assert_close(torch.as_tensor(value[key]), torch.as_tensor(func(metrics._last_preds, metrics._last_targets)))
        for key, func in function_map.items():
            assert_close(torch.as_tensor(average[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))
    final_avg = metrics.average()
    for key, func in function_map.items():
        assert_close(torch.as_tensor(final_avg[key]), torch.as_tensor(func(metrics.preds, metrics.targets)))

    dist.destroy_process_group()
