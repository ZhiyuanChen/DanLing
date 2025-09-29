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

import random
from functools import partial

import pytest
import torch
from torch.testing import assert_close
from torchmetrics.functional import classification as tmfc

from danling import NestedTensor
from danling.metrics.functional import (
    MetricFunc,
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

from .utils import (
    ATOL,
    RTOL,
    assert_metric_outputs,
    build_function_map,
    make_binary_metrics,
    make_multiclass_metrics,
    make_multilabel_metrics,
    make_regression_metrics,
    process_group,
    require_nccl_cuda,
    run_distributed,
)


def _updated_binary_metrics(num_batches: int = 3, batch_size: int = 6):
    metrics = make_binary_metrics(distributed=False)
    preds, targets = [], []
    for _ in range(num_batches):
        logits = torch.randn(batch_size)
        raw_target = torch.randint(2, (batch_size,))
        pred, target = preprocess_binary(logits, raw_target)
        preds.append(pred)
        targets.append(target)
        metrics.update(logits, raw_target)
    return metrics, torch.cat(preds), torch.cat(targets)


class TestGlobalMetricsLocalBehavior:
    def test_rejects_non_metricfunc_descriptors(self):
        with pytest.raises(ValueError, match="MetricFunc instances"):
            GlobalMetrics([1], preprocess=preprocess_binary, distributed=False)  # type: ignore[list-item]

        with pytest.raises(ValueError, match="MetricFunc instances"):
            GlobalMetrics(preprocess=preprocess_binary, distributed=False, bad=1)  # type: ignore[arg-type]

    def test_binary_value(self):
        random.seed(0)
        torch.random.manual_seed(0)
        metrics = make_binary_metrics(distributed=False)
        function_map = build_function_map("binary")

        for _ in range(4):
            logits = torch.randn(8)
            raw_target = torch.randint(2, (8,))
            pred, target = preprocess_binary(logits, raw_target)
            metrics.update(logits, raw_target)

            assert_metric_outputs(metrics.value(), function_map, pred, target)

    def test_binary_average(self):
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

            average = metrics.average()
            assert_metric_outputs(average, function_map, metrics.preds, metrics.targets)

        assert_close(metrics.preds, torch.cat(preds), rtol=RTOL, atol=ATOL)
        assert_close(metrics.targets, torch.cat(targets), rtol=RTOL, atol=ATOL)

    def test_multiclass_value(self):
        random.seed(0)
        torch.random.manual_seed(0)
        num_classes = 10
        metrics = make_multiclass_metrics(num_classes=num_classes, distributed=False)
        function_map = build_function_map("multiclass", num_classes=num_classes)

        for _ in range(4):
            logits = torch.randn(8, num_classes)
            raw_target = torch.randint(num_classes, (8,))
            pred, target = preprocess_multiclass(logits, raw_target, num_classes=num_classes)
            metrics.update(logits, raw_target)

            assert_metric_outputs(metrics.value(), function_map, pred, target)

    def test_multiclass_average(self):
        random.seed(0)
        torch.random.manual_seed(0)
        num_classes = 10
        metrics = make_multiclass_metrics(num_classes=num_classes, distributed=False)
        function_map = build_function_map("multiclass", num_classes=num_classes)
        preds, targets = [], []

        for _ in range(4):
            logits = torch.randn(8, num_classes)
            raw_target = torch.randint(num_classes, (8,))
            pred, target = preprocess_multiclass(logits, raw_target, num_classes=num_classes)
            preds.append(pred)
            targets.append(target)
            metrics.update(logits, raw_target)

            average = metrics.average()
            assert_metric_outputs(average, function_map, metrics.preds, metrics.targets)

        assert_close(metrics.preds, torch.cat(preds), rtol=RTOL, atol=ATOL)
        assert_close(metrics.targets, torch.cat(targets), rtol=RTOL, atol=ATOL)

    def test_multiclass_topk_value(self):
        random.seed(0)
        torch.random.manual_seed(0)
        num_classes = 6
        k = 3
        metrics = GlobalMetrics(
            [multiclass_accuracy(num_classes=num_classes, k=k)],
            preprocess=partial(preprocess_multiclass, num_classes=num_classes),
            distributed=False,
        )

        for _ in range(4):
            logits = torch.randn(8, num_classes)
            target = torch.randint(num_classes, (8,))
            pred, target = preprocess_multiclass(logits, target, num_classes=num_classes)
            metrics.update(logits, target)

            expected_last = tmfc.multiclass_accuracy(
                pred,
                target,
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

    def test_multiclass_topk_average(self):
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

    def test_multilabel_value(self):
        random.seed(0)
        torch.random.manual_seed(0)
        num_labels = 10
        metrics = make_multilabel_metrics(num_labels=num_labels, distributed=False)
        function_map = build_function_map("multilabel", num_labels=num_labels)

        for _ in range(4):
            logits = torch.randn(8, num_labels)
            raw_target = torch.randint(2, (8, num_labels))
            pred, target = preprocess_multilabel(logits, raw_target, num_labels=num_labels)
            metrics.update(logits, raw_target)

            assert_metric_outputs(metrics.value(), function_map, pred, target)

    def test_multilabel_average(self):
        random.seed(0)
        torch.random.manual_seed(0)
        num_labels = 10
        metrics = make_multilabel_metrics(num_labels=num_labels, distributed=False)
        function_map = build_function_map("multilabel", num_labels=num_labels)
        preds, targets = [], []

        for _ in range(4):
            logits = torch.randn(8, num_labels)
            raw_target = torch.randint(2, (8, num_labels))
            pred, target = preprocess_multilabel(logits, raw_target, num_labels=num_labels)
            preds.append(pred)
            targets.append(target)
            metrics.update(logits, raw_target)

            average = metrics.average()
            assert_metric_outputs(average, function_map, metrics.preds, metrics.targets)

        assert_close(metrics.preds, torch.cat(preds), rtol=RTOL, atol=ATOL)
        assert_close(metrics.targets, torch.cat(targets), rtol=RTOL, atol=ATOL)

    def test_regression_value_with_nested_tensors(self):
        random.seed(0)
        torch.random.manual_seed(0)
        num_outputs = 2
        metrics = make_regression_metrics(num_outputs=num_outputs, distributed=False)
        function_map = build_function_map("regression", num_outputs=num_outputs)

        lengths_list = [(2, 3, 5), (7, 11)]
        for lengths in lengths_list:
            pred_list, target_list = [], []
            for length in lengths:
                pred_list.append(torch.randn(length, num_outputs))
                target_list.append(torch.randn(length, num_outputs))
            pred_nt, target_nt = NestedTensor(pred_list), NestedTensor(target_list)
            metrics.update(pred_nt, target_nt)

            assert_metric_outputs(metrics.value(), function_map, torch.cat(pred_list), torch.cat(target_list))

    def test_regression_average_with_nested_tensors(self):
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

            average = metrics.average()
            assert_metric_outputs(average, function_map, metrics.preds, metrics.targets)

        assert_close(metrics.preds, torch.cat(preds), rtol=RTOL, atol=ATOL)
        assert_close(metrics.targets, torch.cat(targets), rtol=RTOL, atol=ATOL)

    def test_average_preserves_public_artifacts(self):
        metrics, expected_preds, expected_targets = _updated_binary_metrics()

        _ = metrics.average()
        assert_close(metrics.preds, expected_preds, rtol=RTOL, atol=ATOL)
        assert_close(metrics.targets, expected_targets, rtol=RTOL, atol=ATOL)

    def test_average_is_idempotent(self):
        metrics, _, _ = _updated_binary_metrics()

        first = metrics.average()
        second = metrics.average()
        for key in first:
            assert_close(
                torch.as_tensor(first[key]),
                torch.as_tensor(second[key]),
                rtol=RTOL,
                atol=ATOL,
                check_dtype=False,
            )

    def test_reset_clears_state(self):
        metrics = make_binary_metrics(distributed=False)
        metrics.update(torch.tensor([0.2, 0.8, 0.9]), torch.tensor([0, 1, 0]))

        assert metrics.n == 3
        assert metrics.count == 3
        assert metrics.confmat is not None

        assert metrics.reset() is metrics
        assert metrics.n == 0
        assert metrics.count == 0
        assert metrics.preds.numel() == 0
        assert metrics.targets.numel() == 0
        assert metrics.confmat is None

        for value in metrics.value().values():
            assert torch.isnan(torch.as_tensor(value)).all()
        for value in metrics.average().values():
            assert torch.isnan(torch.as_tensor(value)).all()

    def test_reset_allows_reuse(self):
        metrics = make_binary_metrics(distributed=False)
        metrics.update(torch.tensor([0.2, 0.8, 0.9]), torch.tensor([0, 1, 0]))
        metrics.reset()

        metrics.update(torch.tensor([0.2, 0.8]), torch.tensor([0, 1]))
        assert metrics.n == 2
        assert metrics.count == 2
        assert metrics.value()["acc"] == pytest.approx(1.0)

    def test_update_detaches_last_batch(self):
        class _DetachedMetric(MetricFunc):
            def __init__(self) -> None:
                super().__init__(name="detached", preds_targets=True)

            def __call__(self, state):
                return torch.tensor(float(state.preds.requires_grad or state.targets.requires_grad))

        metrics = GlobalMetrics(
            [_DetachedMetric()],
            preprocess=preprocess_binary,
            distributed=False,
        )
        logits = torch.tensor([0.2, 0.8], requires_grad=True)
        target = torch.tensor([0, 1])

        metrics.update(logits, target)

        assert metrics.value()["detached"] == pytest.approx(0.0)

    def test_preserves_post_preprocess_shapes(self):
        class _ShapeMetric(MetricFunc):
            def __init__(self) -> None:
                super().__init__(name="shape", preds_targets=True)

            def __call__(self, state):
                return torch.tensor(float(state.preds.shape[-1]))

        class _DetachedMetric(MetricFunc):
            def __init__(self) -> None:
                super().__init__(name="detached", preds_targets=True)

            def __call__(self, state):
                return torch.tensor(float(state.preds.requires_grad or state.targets.requires_grad))

        metrics = GlobalMetrics(
            [_ShapeMetric(), _DetachedMetric()],
            preprocess=lambda input, target: (input, target),
            distributed=False,
        )
        input = torch.tensor([[1.0], [0.0]], requires_grad=True)
        target = torch.tensor([1.0, 0.0])

        metrics.update(input, target)

        assert metrics.value()["shape"] == pytest.approx(1.0)
        assert metrics.value()["detached"] == pytest.approx(0.0)

    def test_accepts_nested_tensors(self):
        class _AbsoluteErrorMetric(MetricFunc):
            def __init__(self) -> None:
                super().__init__(name="mae", preds_targets=True)

            def __call__(self, state):
                return (state.preds - state.targets).abs().mean()

        pred_tensors = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0]]),
        ]
        target_tensors = [
            torch.tensor([[1.5, 1.0], [2.0, 5.0]]),
            torch.tensor([[4.0, 8.0]]),
        ]
        metrics = GlobalMetrics(
            [_AbsoluteErrorMetric()], preprocess=lambda input, target: (input, target), distributed=False
        )

        metrics.update(NestedTensor(pred_tensors), NestedTensor(target_tensors))

        expected_preds = torch.cat(pred_tensors)
        expected_targets = torch.cat(target_tensors)
        expected = (expected_preds - expected_targets).abs().mean()
        assert_close(metrics.preds, expected_preds, rtol=RTOL, atol=ATOL)
        assert_close(metrics.targets, expected_targets, rtol=RTOL, atol=ATOL)
        assert_close(torch.as_tensor(metrics.value()["mae"]), torch.as_tensor(expected), rtol=RTOL, atol=ATOL)
        assert_close(torch.as_tensor(metrics.average()["mae"]), torch.as_tensor(expected), rtol=RTOL, atol=ATOL)

    def test_accepts_keyword_metric_descriptors(self):
        metrics = GlobalMetrics(
            preprocess=preprocess_binary,
            distributed=False,
            auroc=binary_auroc(),
            acc=binary_accuracy(),
        )
        metrics.update(torch.randn(8), torch.randint(2, (8,)))
        assert {"auroc", "acc"} <= set(metrics.avg.keys())

    def test_multiclass_confmat_with_ignore_index(self):
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
            "acc": partial(
                tmfc.multiclass_accuracy, num_classes=num_classes, average=average, ignore_index=ignore_index
            ),
            "precision": partial(
                tmfc.multiclass_precision,
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
            "recall": partial(
                tmfc.multiclass_recall, num_classes=num_classes, average=average, ignore_index=ignore_index
            ),
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
            "f1": partial(
                tmfc.multiclass_f1_score,
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
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
            assert_metric_outputs(metrics.value(), function_map, pred, processed_target)
            assert_metric_outputs(metrics.average(), function_map, torch.cat(preds), torch.cat(targets))

    def test_multilabel_confmat_with_elementwise_ignore_index(self):
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
            "recall": partial(
                tmfc.multilabel_recall, num_labels=num_labels, average=average, ignore_index=ignore_index
            ),
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
            pred, processed_target = preprocess_multilabel(
                logits, target, num_labels=num_labels, ignore_index=ignore_index
            )
            preds.append(pred)
            targets.append(processed_target)
            metrics.update(logits, target)
            assert_metric_outputs(metrics.value(), function_map, pred, processed_target)
            assert_metric_outputs(metrics.average(), function_map, torch.cat(preds), torch.cat(targets))


class TestDistributedGlobalMetrics:
    def test_binary_average_syncs_across_ranks(self):
        run_distributed(_distributed_binary_worker, world_size=2)

    def test_multiclass_average_syncs_across_ranks(self):
        run_distributed(_distributed_multiclass_worker, world_size=2)

    def test_multilabel_average_syncs_across_ranks(self):
        run_distributed(_distributed_multilabel_worker, world_size=2)

    def test_binary_sync_handles_empty_rank(self):
        run_distributed(_distributed_binary_empty_rank_worker, world_size=2)

    def test_batch_syncs_current_step(self):
        run_distributed(_distributed_global_batch_worker, world_size=2)

    def test_public_artifacts_sync_after_average(self):
        run_distributed(_distributed_global_public_artifacts_worker, world_size=2)

    def test_batch_without_declared_artifacts(self):
        run_distributed(_distributed_global_stateless_batch_worker, world_size=2)

    def test_exact_metrics_handle_empty_rank(self):
        run_distributed(_distributed_global_exact_empty_rank_worker, world_size=2)

    def test_binary_average_syncs_across_ranks_with_nccl(self):
        require_nccl_cuda(world_size=2)
        run_distributed(_distributed_binary_nccl_worker, world_size=2)


def _distributed_binary_worker(rank: int, world_size: int):
    with process_group("gloo", rank, world_size):
        torch.manual_seed(1234 + rank)
        metrics = make_binary_metrics(distributed=True)
        function_map = build_function_map("binary")

        for step in range(2):
            length = 4 + rank + step
            logits = torch.randn(length)
            target = torch.randint(2, (length,))
            pred, processed_target = preprocess_binary(logits, target)
            metrics.update(logits, target)

            assert_metric_outputs(metrics.value(), function_map, pred, processed_target)
            average = metrics.average()
            assert_metric_outputs(average, function_map, metrics.preds, metrics.targets)


def _distributed_multiclass_worker(rank: int, world_size: int):
    with process_group("gloo", rank, world_size):
        num_classes = 5
        torch.manual_seed(2234 + rank)
        metrics = make_multiclass_metrics(num_classes=num_classes, distributed=True)
        function_map = build_function_map("multiclass", num_classes=num_classes)

        for step in range(2):
            length = 4 + rank + step
            logits = torch.randn(length, num_classes)
            target = torch.randint(num_classes, (length,))
            pred, processed_target = preprocess_multiclass(logits, target, num_classes=num_classes)
            metrics.update(logits, target)

            assert_metric_outputs(metrics.value(), function_map, pred, processed_target)
            average = metrics.average()
            assert_metric_outputs(average, function_map, metrics.preds, metrics.targets)


def _distributed_multilabel_worker(rank: int, world_size: int):
    with process_group("gloo", rank, world_size):
        num_labels = 5
        torch.manual_seed(3234 + rank)
        metrics = make_multilabel_metrics(num_labels=num_labels, distributed=True)
        function_map = build_function_map("multilabel", num_labels=num_labels)

        for step in range(2):
            length = 4 + rank + step
            logits = torch.randn(length, num_labels)
            target = torch.randint(2, (length, num_labels))
            pred, processed_target = preprocess_multilabel(logits, target, num_labels=num_labels)
            metrics.update(logits, target)

            assert_metric_outputs(metrics.value(), function_map, pred, processed_target)
            average = metrics.average()
            assert_metric_outputs(average, function_map, metrics.preds, metrics.targets)


def _distributed_binary_empty_rank_worker(rank: int, world_size: int):
    with process_group("gloo", rank, world_size):
        metrics = GlobalMetrics([binary_accuracy()], preprocess=preprocess_binary, distributed=True, device="cpu")

        if rank == 1:
            logits = torch.tensor([0.2, 0.8, 0.7], dtype=torch.float32)
            target = torch.tensor([0, 1, 1], dtype=torch.long)
            metrics.update(logits, target)

        average = metrics.average()

        assert average["acc"] == pytest.approx(1.0)
        assert metrics.confmat is not None
        assert_close(metrics.confmat, torch.tensor([[1, 0], [0, 2]], dtype=torch.long), rtol=RTOL, atol=ATOL)


def _distributed_global_batch_worker(rank: int, world_size: int):
    with process_group("gloo", rank, world_size):
        metrics = GlobalMetrics(
            [binary_accuracy(), binary_auroc()],
            preprocess=preprocess_binary,
            distributed=True,
            device="cpu",
        )

        if rank == 0:
            logits = torch.tensor([0.2, 0.8], dtype=torch.float32)
            target = torch.tensor([0, 1], dtype=torch.long)
        else:
            logits = torch.tensor([0.9], dtype=torch.float32)
            target = torch.tensor([0], dtype=torch.long)

        metrics.update(logits, target)
        batch = metrics.batch()
        expected_preds = torch.tensor([0.2, 0.8, 0.9], dtype=torch.float32)
        expected_targets = torch.tensor([0, 1, 0], dtype=torch.long)

        assert metrics.value()["acc"] == pytest.approx(1.0 if rank == 0 else 0.0)
        assert batch["acc"] == pytest.approx(2.0 / 3.0)
        assert_close(
            torch.as_tensor(batch["auroc"]),
            torch.as_tensor(tmfc.binary_auroc(expected_preds, expected_targets, ignore_index=-100)),
            rtol=RTOL,
            atol=ATOL,
            check_dtype=False,
        )


def _distributed_global_public_artifacts_worker(rank: int, world_size: int):
    with process_group("gloo", rank, world_size):
        metrics = GlobalMetrics(
            [binary_accuracy(), binary_auroc()],
            preprocess=preprocess_binary,
            distributed=True,
            device="cpu",
        )

        if rank == 0:
            logits = torch.tensor([0.2, 0.8], dtype=torch.float32)
            target = torch.tensor([0, 1], dtype=torch.long)
        else:
            logits = torch.tensor([0.6, 0.4, 0.9], dtype=torch.float32)
            target = torch.tensor([1, 0, 1], dtype=torch.long)

        local_pred, local_target = preprocess_binary(logits, target)
        metrics.update(logits, target)

        expected_local_confmat = torch.zeros((2, 2), dtype=torch.long)
        for pred_value, target_value in zip((local_pred >= 0.5).long(), local_target.long()):
            expected_local_confmat[target_value, pred_value] += 1

        assert_close(metrics.preds, local_pred, rtol=RTOL, atol=ATOL)
        assert_close(metrics.targets, local_target, rtol=RTOL, atol=ATOL)
        assert metrics.confmat is not None
        assert_close(metrics.confmat, expected_local_confmat, rtol=RTOL, atol=ATOL)

        average = metrics.average()
        expected_preds = torch.tensor([0.2, 0.8, 0.6, 0.4, 0.9], dtype=torch.float32)
        expected_targets = torch.tensor([0, 1, 1, 0, 1], dtype=torch.long)
        expected_confmat = torch.tensor([[2, 0], [0, 3]], dtype=torch.long)

        assert_close(metrics.preds, expected_preds, rtol=RTOL, atol=ATOL)
        assert_close(metrics.targets, expected_targets, rtol=RTOL, atol=ATOL)
        assert metrics.confmat is not None
        assert_close(metrics.confmat, expected_confmat, rtol=RTOL, atol=ATOL)
        assert_close(
            torch.as_tensor(average["auroc"]),
            torch.as_tensor(tmfc.binary_auroc(expected_preds, expected_targets, ignore_index=-100)),
            rtol=RTOL,
            atol=ATOL,
            check_dtype=False,
        )


def _distributed_global_stateless_batch_worker(rank: int, world_size: int):
    class _ScalarMetric(MetricFunc):
        def __init__(self, value: float) -> None:
            super().__init__(name="scalar")
            self.local_value = value

        def __call__(self, state):
            del state
            return self.local_value

    class _VectorMetric(MetricFunc):
        def __init__(self, value: list[float]) -> None:
            super().__init__(name="vector")
            self.local_value = torch.tensor(value, dtype=torch.float32)

        def __call__(self, state):
            del state
            return self.local_value

    with process_group("gloo", rank, world_size):
        scalar = 0.25 if rank == 0 else 0.75
        vector = [0.25, 0.5] if rank == 0 else [1.0, 1.5]
        metrics = GlobalMetrics(
            [_ScalarMetric(scalar), _VectorMetric(vector)],
            preprocess=lambda input, target: (input, target),
            distributed=True,
            device="cpu",
        )

        if rank == 0:
            input = torch.tensor([1.0, 2.0])
            target = torch.tensor([0.0, 1.0])
        else:
            input = torch.tensor([3.0])
            target = torch.tensor([1.0])

        metrics.update(input, target)
        batch = metrics.batch()

        assert batch["scalar"] == pytest.approx((0.25 * 2 + 0.75) / 3)
        assert_close(batch["vector"], torch.tensor([0.5, 5.0 / 6.0], dtype=torch.float64), rtol=RTOL, atol=ATOL)


def _distributed_global_exact_empty_rank_worker(rank: int, world_size: int):
    with process_group("gloo", rank, world_size):
        metrics = GlobalMetrics([binary_auroc()], preprocess=preprocess_binary, distributed=True, device="cpu")

        if rank == 1:
            logits = torch.tensor([0.2, 0.8, 0.7], dtype=torch.float32)
            target = torch.tensor([0, 1, 1], dtype=torch.long)
            metrics.update(logits, target)

        batch = metrics.batch()
        average = metrics.average()
        expected_preds = torch.tensor([0.2, 0.8, 0.7], dtype=torch.float32)
        expected_targets = torch.tensor([0, 1, 1], dtype=torch.long)
        expected = tmfc.binary_auroc(expected_preds, expected_targets, ignore_index=-100)

        assert_close(
            torch.as_tensor(batch["auroc"]), torch.as_tensor(expected), rtol=RTOL, atol=ATOL, check_dtype=False
        )
        assert_close(
            torch.as_tensor(average["auroc"]),
            torch.as_tensor(expected),
            rtol=RTOL,
            atol=ATOL,
            check_dtype=False,
        )
        assert_close(metrics.preds, expected_preds, rtol=RTOL, atol=ATOL)
        assert_close(metrics.targets, expected_targets, rtol=RTOL, atol=ATOL)


def _distributed_binary_nccl_worker(rank: int, world_size: int):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    with process_group("nccl", rank, world_size):
        metrics = GlobalMetrics([binary_accuracy()], preprocess=preprocess_binary, distributed=True, device=device)

        if rank == 0:
            logits = torch.tensor([10.0, 10.0], dtype=torch.float32, device=device)
            target = torch.tensor([1, 1], dtype=torch.long, device=device)
        else:
            logits = torch.tensor([-10.0, -10.0, -10.0, -10.0], dtype=torch.float32, device=device)
            target = torch.tensor([1, 1, 1, 1], dtype=torch.long, device=device)

        metrics.update(logits, target)

        expected_local = 1.0 if rank == 0 else 0.0
        assert float(torch.as_tensor(metrics.value()["acc"]).cpu()) == pytest.approx(expected_local)

        average = metrics.average()

        assert float(torch.as_tensor(average["acc"]).cpu()) == pytest.approx(2.0 / 6.0)
        assert metrics.confmat is not None
        assert_close(
            metrics.confmat,
            torch.tensor([[0, 0], [4, 2]], dtype=torch.long, device=device),
            rtol=RTOL,
            atol=ATOL,
        )
