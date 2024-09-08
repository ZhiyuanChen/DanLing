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

import os
import random
from functools import partial

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn
from torcheval.metrics import (
    BinaryAccuracy,
    BinaryAUPRC,
    BinaryAUROC,
    BinaryF1Score,
    MulticlassAccuracy,
    MulticlassAUPRC,
    MulticlassAUROC,
    MulticlassF1Score,
    MultilabelAccuracy,
    MultilabelAUPRC,
    R2Score,
)
from torcheval.metrics.functional import (
    binary_accuracy,
    binary_auprc,
    binary_auroc,
    binary_f1_score,
    multiclass_accuracy,
    multiclass_auprc,
    multiclass_auroc,
    multiclass_f1_score,
    multilabel_accuracy,
    multilabel_auprc,
)
from torchmetrics.functional import matthews_corrcoef
from torchmetrics.functional.classification import multilabel_auroc, multilabel_f1_score
from torchmetrics.functional.regression import pearson_corrcoef, r2_score, spearman_corrcoef
from torchmetrics.regression import SpearmanCorrCoef

from danling import NestedTensor
from danling.metrics import Metrics, binary_metrics, multiclass_metrics, multilabel_metrics, regression_metrics
from danling.metrics.functional import accuracy, auprc, auroc, f1_score, mcc
from danling.metrics.metrics import ScoreMetrics


def demo_dict_metric_func(input, target):
    """
    Normally DanLing Metrics takes a dict of function in constructor calls.
    The function should return a single float value representing the score.

    In rare cases, some metric functions may share internal variables,
    and the best way for them is to use one function to calculate all metrics.

    DanLing Metrics allows functions that return a dict for multiple scores.
    """

    return {
        "auroc": auroc(input, target),
        "auprc": auprc(input, target),
        "acc": accuracy(input, target),
        "mcc": mcc(input, target),
        "f1": f1_score(input, target),
    }


def build_metric_map(task: str, num_classes: int = 10, num_labels: int = 10, num_outputs: int = 1):
    if task == "binary":
        return {
            "auroc": BinaryAUROC(),
            "auprc": BinaryAUPRC(),
            "acc": BinaryAccuracy(),
            "f1": BinaryF1Score(),
        }
    if task == "multiclass":
        return {
            "auroc": MulticlassAUROC(num_classes=num_classes),
            "auprc": MulticlassAUPRC(num_classes=num_classes),
            "acc": MulticlassAccuracy(num_classes=num_classes),
            "f1": MulticlassF1Score(num_classes=num_classes),
        }
    if task == "multilabel":
        return {
            "auprc": MultilabelAUPRC(num_labels=num_labels),
            "acc": MultilabelAccuracy(),
        }
    if task == "regression":
        return {
            "spearman": SpearmanCorrCoef(num_outputs=num_outputs),
            "r2": R2Score(),
        }


def build_function_map(task: str, num_classes: int = 10, num_labels: int = 10, num_outputs: int = 1):
    if task == "binary":
        return {
            "auroc": binary_auroc,
            "auprc": binary_auprc,
            "acc": binary_accuracy,
            "f1": binary_f1_score,
            "mcc": partial(matthews_corrcoef, task="binary"),
        }
    if task == "multiclass":
        return {
            "auroc": partial(multiclass_auroc, num_classes=num_classes),
            "auprc": partial(multiclass_auprc, num_classes=num_classes),
            "acc": partial(multiclass_accuracy, num_classes=num_classes),
            "f1": partial(multiclass_f1_score, num_classes=num_classes),
            "mcc": partial(matthews_corrcoef, task="multiclass", num_classes=num_classes),
        }
    if task == "multilabel":
        return {
            "auroc": partial(multilabel_auroc, num_labels=num_labels),
            "auprc": partial(multilabel_auprc, num_labels=num_labels),
            "acc": multilabel_accuracy,
            "f1": partial(multilabel_f1_score, num_labels=num_labels, average="micro"),
            "mcc": partial(matthews_corrcoef, task="multilabel", num_labels=num_labels),
        }
    if task == "regression":
        return {
            "pearson": pearson_corrcoef,
            "spearman": spearman_corrcoef,
            "r2": r2_score,
        }


class TestMetricsSingle:
    epsilon = 1e-6

    def test_empty(self):
        metrics = Metrics()
        assert (metrics.inputs == torch.empty(0)).all()
        assert (metrics.targets == torch.empty(0)).all()

    def test_score_metrics(self):
        random.seed(0)
        torch.random.manual_seed(0)
        metrics = ScoreMetrics(auroc=auroc, auprc=auprc, acc=accuracy)
        score_name = "auroc"
        assert metrics.score_name == score_name
        for _ in range(10):
            pred = torch.randn(8).sigmoid()
            target = torch.randint(2, (8,))
            metrics.update(pred, target)
            assert metrics.batch_score == metrics.val[score_name] == metrics.get_score("batch")
            assert metrics.average_score == metrics.avg[score_name] == metrics.get_score("average")
        with pytest.raises(ValueError):
            metrics.get_score("total")
        with pytest.raises(ValueError):
            metrics.score_name = "f1"

    def test_tensor_regression(self):
        random.seed(0)
        torch.random.manual_seed(0)
        metrics = regression_metrics()
        metric_map = build_metric_map("regression")
        function_map = build_function_map("regression")
        preds, targets = [], []
        for _ in range(10):
            pred = torch.randn(8, 1)
            target = torch.randn(8)
            preds.append(pred)
            targets.append(target)
            metrics.update(pred, target)
            for metric in metric_map.values():
                metric.update(pred.view(-1), target.view(-1))
            value, average = metrics.value(), metrics.average()
            assert (pred.view(-1) == metrics.input.view(-1)).all()
            assert (target.view(-1) == metrics.target.view(-1)).all()
            for key, func in function_map.items():
                assert value[key] - func(pred.view(-1), target.view(-1)) < self.epsilon
            for key, metric in metric_map.items():
                assert average[key] - metric.compute() < self.epsilon
        assert (torch.cat(preds).view(-1) == metrics.inputs).all()
        assert (torch.cat(targets).view(-1) == metrics.targets).all()
        for key, metric in metric_map.items():
            assert average[key] - metric.compute() < self.epsilon

    def test_nested_tensor_regression(self):
        random.seed(0)
        torch.random.manual_seed(0)
        num_outputs = 2
        metrics = regression_metrics(num_outputs=num_outputs)
        metric_map = build_metric_map("regression", num_outputs=num_outputs)
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
            pred, target = torch.cat(pred_list), torch.cat(target_list)
            metrics.update(pred_nt, target_nt)
            for metric in metric_map.values():
                metric.update(pred.view(-1, num_outputs), target.view(-1, num_outputs))
            value, average = metrics.value(), metrics.average()
            assert (pred_nt == metrics._input).all()
            assert (target_nt == metrics._target).all()
            pred = torch.cat(pred_list)
            target = torch.cat(target_list)
            for key, func in function_map.items():
                assert value[key] - func(pred, target).mean() < self.epsilon
            for key, metric in metric_map.items():
                assert average[key] - metric.compute().mean() < self.epsilon
        assert (torch.cat(preds) == metrics.inputs).all()
        assert (torch.cat(targets) == metrics.targets).all()
        for key, metric in metric_map.items():
            assert average[key] - metric.compute().mean() < self.epsilon

    def test_tensor_binary(self):
        random.seed(0)
        torch.random.manual_seed(0)
        merge_metrics = Metrics(func=demo_dict_metric_func, merge_dict=True)
        metrics = binary_metrics()
        metric_map = build_metric_map("binary")
        function_map = build_function_map("binary")
        preds, targets = [], []
        for _ in range(10):
            logits = torch.randn(8)
            pred = logits.sigmoid()
            target = torch.randint(2, (8,))
            preds.append(pred)
            targets.append(target)
            metrics.update(logits, target)
            merge_metrics.update(pred, target)
            for metric in metric_map.values():
                metric.update(pred, target)
            value, average = metrics.value(), metrics.average()
            metrics_input, metrics_target = metrics.preprocess(metrics.input, metrics.target)
            assert (pred == metrics_input).all()
            assert (target == metrics_target).all()
            for key, func in function_map.items():
                assert value[key] - func(pred, target) < self.epsilon
            for key, metric in metric_map.items():
                assert average[key] - metric.compute() < self.epsilon
            assert metrics.avg == merge_metrics.avg
        metrics_inputs, metrics_targets = metrics.preprocess(metrics.inputs, metrics.targets)
        assert (torch.cat(preds) - metrics_inputs < self.epsilon).all()
        assert (torch.cat(targets) - metrics_targets < self.epsilon).all()
        for key, metric in metric_map.items():
            assert average[key] - metric.compute() < self.epsilon
        assert metrics.avg == merge_metrics.avg

    def test_nested_tensor_binary(self):
        random.seed(0)
        torch.random.manual_seed(0)
        merge_metrics = Metrics(func=demo_dict_metric_func, merge_dict=False)
        metrics = binary_metrics()
        metric_map = build_metric_map("binary")
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
            pred, target = torch.cat(pred_list), torch.cat(target_list)
            metrics.update(pred_nt, target_nt)
            merge_metrics.update(pred_nt, target_nt)
            for metric in metric_map.values():
                metric.update(pred, target)
            value, batch, average = metrics.value(), metrics.batch(), metrics.average()
            assert (pred_nt == metrics._input).all()
            assert (target_nt == metrics._target).all()
            pred = torch.cat(pred_list)
            target = torch.cat(target_list)
            for key, func in function_map.items():
                assert batch[key] - func(pred, target) < self.epsilon
            for key, metric in metric_map.items():
                assert average[key] - metric.compute() < self.epsilon
        assert (torch.cat(preds) == metrics.inputs).all()
        assert (torch.cat(targets) == metrics.targets).all()
        for key, metric in metric_map.items():
            assert average[key] - metric.compute() < self.epsilon
        ret, dict_ret = metrics.average(), merge_metrics.average()
        for key, value in ret.items():
            assert dict_ret[f"func.{key}"] == value

    def test_tensor_multiclass(self):
        random.seed(0)
        torch.random.manual_seed(0)
        num_classes = 10
        metrics = multiclass_metrics(num_classes=num_classes)
        metric_map = build_metric_map("multiclass", num_classes=num_classes)
        function_map = build_function_map("multiclass", num_classes=num_classes)
        preds, targets = [], []
        for _ in range(10):
            logits = torch.randn(8, num_classes)
            pred = logits.softmax(-1)
            target = torch.randint(num_classes, (8,))
            preds.append(pred)
            targets.append(target)
            metrics.update(logits, target)
            for metric in metric_map.values():
                metric.update(pred, target)
            value, average = metrics.value(), metrics.average()
            metrics_input, metrics_target = metrics.preprocess(metrics.input, metrics.target)
            assert (pred == metrics_input).all()
            assert (target == metrics_target).all()
            for key, func in function_map.items():
                assert value[key] - func(pred, target) < self.epsilon
            for key, metric in metric_map.items():
                assert average[key] - metric.compute() < self.epsilon
        metrics_inputs, metrics_targets = metrics.preprocess(metrics.inputs, metrics.targets)
        assert (torch.cat(preds) - metrics_inputs < self.epsilon).all()
        assert (torch.cat(targets) - metrics_targets < self.epsilon).all()
        for key, metric in metric_map.items():
            assert average[key] - metric.compute() < self.epsilon

    def test_nested_tensor_multiclass(self):
        random.seed(0)
        torch.random.manual_seed(0)
        num_classes = 10
        metrics = multiclass_metrics(num_classes=num_classes)
        metric_map = build_metric_map("multiclass", num_classes=num_classes)
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
            pred, target = torch.cat(pred_list), torch.cat(target_list)
            metrics.update(pred_nt, target_nt)
            for metric in metric_map.values():
                metric.update(pred, target)
            value, average = metrics.value(), metrics.average()
            assert (pred_nt == metrics._input).all()
            assert (target_nt == metrics._target).all()
            pred = torch.cat(pred_list)
            target = torch.cat(target_list)
            for key, func in function_map.items():
                assert value[key] - func(pred, target) < self.epsilon
            for key, metric in metric_map.items():
                assert average[key] - metric.compute() < self.epsilon
        assert (torch.cat(preds) == metrics.inputs).all()
        assert (torch.cat(targets) == metrics.targets).all()
        for key, metric in metric_map.items():
            assert average[key] - metric.compute() < self.epsilon

    def test_tensor_multilabel(self):
        random.seed(0)
        torch.random.manual_seed(0)
        num_labels = 10
        metrics = multilabel_metrics(num_labels=num_labels)
        metric_map = build_metric_map("multilabel", num_labels=num_labels)
        function_map = build_function_map("multilabel", num_labels=num_labels)
        preds, targets = [], []
        for _ in range(10):
            logits = torch.randn(8, num_labels)
            pred = logits.sigmoid()
            target = torch.randint(2, (8, num_labels))
            preds.append(pred)
            targets.append(target)
            metrics.update(logits, target)
            for metric in metric_map.values():
                metric.update(pred, target)
            value, average = metrics.value(), metrics.average()
            metrics_input, metrics_target = metrics.preprocess(metrics.input, metrics.target)
            assert (pred == metrics_input).all()
            assert (target == metrics_target).all()
            for key, func in function_map.items():
                assert value[key] - func(pred, target) < self.epsilon
            for key, metric in metric_map.items():
                assert average[key] - metric.compute() < self.epsilon
        metrics_inputs, metrics_targets = metrics.preprocess(metrics.inputs, metrics.targets)
        assert (torch.cat(preds) - metrics_inputs < self.epsilon).all()
        assert (torch.cat(targets) - metrics_targets < self.epsilon).all()
        for key, metric in metric_map.items():
            assert average[key] - metric.compute() < self.epsilon

    def test_nested_tensor_multilabel(self):
        random.seed(0)
        torch.random.manual_seed(0)
        num_labels = 10
        metrics = multilabel_metrics(num_labels=num_labels)
        metric_map = build_metric_map("multilabel", num_labels=num_labels)
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
            pred, target = torch.cat(pred_list), torch.cat(target_list)
            metrics.update(pred_nt, target_nt)
            for metric in metric_map.values():
                metric.update(pred, target)
            value, average = metrics.value(), metrics.average()
            assert (pred_nt == metrics._input).all()
            assert (target_nt == metrics._target).all()
            pred = torch.cat(pred_list)
            target = torch.cat(target_list)
            for key, func in function_map.items():
                assert value[key] - func(pred, target) < self.epsilon
            for key, metric in metric_map.items():
                assert average[key] - metric.compute() < self.epsilon
        assert (torch.cat(preds) == metrics.inputs).all()
        assert (torch.cat(targets) == metrics.targets).all()
        for key, metric in metric_map.items():
            assert average[key] - metric.compute() < self.epsilon


class TestMetricsDistributed:
    epsilon = 1e-6

    def test_tensor_binary(self, world_size: int = 8):
        self._test_distributed(self._test_tensor_binary, world_size)

    def _test_tensor_binary(self, rank, world_size):
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        metrics = binary_metrics()
        metric_map = build_metric_map("binary")
        function_map = build_function_map("binary")
        preds, targets = [], []
        length = 8
        for _ in range(8):
            # PyTorch will complain if requires_grad is not True
            pred = torch.randn(length, requires_grad=True).sigmoid()
            target = torch.randint(2, (length,), dtype=torch.float, requires_grad=True)
            preds.append(pred)
            targets.append(target)
            metrics.update(pred, target)
            assert (pred == metrics.input[length * rank : length * (rank + 1)]).all()  # noqa: E203
            assert (target == metrics.target[length * rank : length * (rank + 1)]).all()  # noqa: E203
            value, batch, average = metrics.value(), metrics.batch(), metrics.average()
            for key, func in function_map.items():
                assert value[key] - func(pred, target) < self.epsilon
            pred = torch.cat(self._gather_tensors([pred], world_size))
            target = torch.cat(self._gather_tensors([target], world_size))
            for key, func in function_map.items():
                assert batch[key] - func(pred, target) < self.epsilon
            for key, metric in metric_map.items():
                metric.update(pred, target)
                assert average[key] - metric.compute() < self.epsilon
        pred = torch.cat(self._all_gather(preds, world_size))
        target = torch.cat(self._all_gather(targets, world_size))
        average = metrics.average()
        for key, func in function_map.items():
            assert average[key] - func(pred, target) < self.epsilon

        dist.destroy_process_group()

    def test_nested_tensor_binary(self, world_size: int = 8):
        self._test_distributed(self._test_nested_tensor_binary, world_size)

    def _test_nested_tensor_binary(self, rank, world_size):
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # cum_length = 0
        metrics = binary_metrics()
        metric_map = build_metric_map("binary")
        function_map = build_function_map("binary")
        preds, targets = [], []
        lengths_list = [[2, 3, 5, 7], [11, 13, 17]]
        if rank == 0:
            lengths_list[-1].append(19)
        # for iter, lengths in enumerate(lengths_list):
        for lengths in lengths_list:
            pred_list, target_list = [], []
            for length in lengths:
                pred_list.append(torch.randn(length).sigmoid())
                target_list.append(torch.randint(2, (length,)))
            preds.extend(pred_list)
            targets.extend(target_list)
            pred_nt, target_nt = NestedTensor(pred_list), NestedTensor(target_list)
            pred, target = torch.cat(pred_list), torch.cat(target_list)
            metrics.update(pred_nt, target_nt)
            assert (pred_nt == metrics._input).all()
            assert (target_nt == metrics._target).all()
            value, batch, average = metrics.value(), metrics.batch(), metrics.average()
            for key, func in function_map.items():
                assert value[key] - func(pred, target) < self.epsilon
            pred = torch.cat(self._all_gather(pred_list, world_size))
            target = torch.cat(self._all_gather(target_list, world_size))
            for key, func in function_map.items():
                assert batch[key] - func(pred, target) < self.epsilon
            for key, metric in metric_map.items():
                metric.update(pred, target)
                assert average[key] - metric.compute() < self.epsilon
        pred = torch.cat(self._all_gather(preds, world_size))
        target = torch.cat(self._all_gather(targets, world_size))
        average = metrics.average()
        for key, func in function_map.items():
            assert average[key] - func(pred, target) < self.epsilon
        for key, metric in metric_map.items():
            assert average[key] - metric.compute() < self.epsilon

        dist.destroy_process_group()

    def test_nested_tensor_regression(self, world_size: int = 8):
        self._test_distributed(self._test_nested_tensor_regression, world_size)

    def _test_nested_tensor_regression(self, rank, world_size):
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # cum_length = 0
        metrics = regression_metrics()
        metric_map = build_metric_map("regression")
        function_map = build_function_map("regression")
        preds, targets = [], []
        lengths_list = [[2, 3, 5, 7], [11, 13, 17]]
        if rank == 0:
            lengths_list[-1].append(19)
        # for iter, lengths in enumerate(lengths_list):
        for lengths in lengths_list:
            pred_list, target_list = [], []
            for length in lengths:
                pred_list.append(torch.randn(length))
                target_list.append(torch.randn(length))
            preds.extend(pred_list)
            targets.extend(target_list)
            pred_nt, target_nt = NestedTensor(pred_list), NestedTensor(target_list)
            pred, target = torch.cat(pred_list), torch.cat(target_list)
            metrics.update(pred_nt, target_nt)
            assert (pred_nt == metrics._input).all()
            assert (target_nt == metrics._target).all()
            value, batch, average = metrics.value(), metrics.batch(), metrics.average()
            for key, func in function_map.items():
                assert value[key] - func(pred, target) < self.epsilon
            pred = torch.cat(self._all_gather(pred_list, world_size))
            target = torch.cat(self._all_gather(target_list, world_size))
            for key, func in function_map.items():
                assert batch[key] - func(pred, target) < self.epsilon
            for key, metric in metric_map.items():
                metric.update(pred, target)
                assert average[key] - metric.compute() < self.epsilon
        pred = torch.cat(self._all_gather(preds, world_size))
        target = torch.cat(self._all_gather(targets, world_size))
        average = metrics.average()
        for key, func in function_map.items():
            assert average[key] - func(pred, target) < self.epsilon

        dist.destroy_process_group()

    def test_nested_tensor_multi_regression(self, world_size: int = 8):
        self._test_distributed(self._test_nested_tensor_multi_regression, world_size)

    def _test_nested_tensor_multi_regression(self, rank, world_size):
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # cum_length = 0
        num_outputs = 8
        metrics = regression_metrics(num_outputs=num_outputs)
        metric_map = build_metric_map("regression", num_outputs=num_outputs)
        function_map = build_function_map("regression", num_outputs=num_outputs)
        preds, targets = [], []
        lengths_list = [[2, 3, 5, 7], [11, 13, 17]]
        if rank == 0:
            lengths_list[-1].append(19)
        # for iter, lengths in enumerate(lengths_list):
        for lengths in lengths_list:
            pred_list, target_list = [], []
            for length in lengths:
                pred_list.append(torch.randn(length, num_outputs))
                target_list.append(torch.randn(length, num_outputs))
            preds.extend(pred_list)
            targets.extend(target_list)
            pred_nt, target_nt = NestedTensor(pred_list), NestedTensor(target_list)
            pred, target = torch.cat(pred_list), torch.cat(target_list)
            metrics.update(pred_nt, target_nt)
            assert (pred_nt == metrics._input).all()
            assert (target_nt == metrics._target).all()
            value, batch, average = metrics.value(), metrics.batch(), metrics.average()
            for key, func in function_map.items():
                assert value[key] - func(pred, target).mean() < self.epsilon
            pred = torch.cat(self._all_gather(pred_list, world_size))
            target = torch.cat(self._all_gather(target_list, world_size))
            for key, func in function_map.items():
                assert batch[key] - func(pred, target).mean() < self.epsilon
            for key, metric in metric_map.items():
                metric.update(pred.view(-1, num_outputs), target.view(-1, num_outputs))
                assert average[key] - metric.compute().mean() < self.epsilon
        pred = torch.cat(self._all_gather(preds, world_size))
        target = torch.cat(self._all_gather(targets, world_size))
        average = metrics.average()
        for key, func in function_map.items():
            assert average[key] - func(pred, target).mean() < self.epsilon

        dist.destroy_process_group()

    def _test_distributed(self, func: callable, world_size: int = 8):
        random.seed(0)
        torch.random.manual_seed(0)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(29501)
        spawn(func, args=(world_size,), nprocs=world_size)

    def _gather_tensors(self, tensor, world_size):
        tensor = torch.cat(tensor)
        synced_tensor = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(synced_tensor, tensor)
        return synced_tensor

    def _all_gather(self, tensors, world_size):
        synced_tensor = [None for _ in range(world_size)]
        dist.all_gather_object(synced_tensor, tensors)
        return [i for t in synced_tensor for i in t]
