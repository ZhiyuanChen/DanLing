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

import torch
from pytest import raises
from torcheval.metrics.functional import (
    binary_accuracy,
    binary_auprc,
    binary_auroc,
    binary_f1_score,
    mean_squared_error,
    multiclass_accuracy,
    multiclass_auprc,
    multiclass_auroc,
    multiclass_f1_score,
)
from torcheval.metrics.functional import r2_score as tef_r2_score
from torchmetrics.functional import matthews_corrcoef, pearson_corrcoef, spearman_corrcoef
from torchmetrics.functional.classification import multilabel_accuracy, multilabel_auroc
from torchmetrics.functional.classification import multilabel_average_precision as multilabel_auprc
from torchmetrics.functional.classification import multilabel_f1_score

from danling.metric.functional import accuracy, auprc, auroc, f1_score, mcc, mse, pearson, r2_score, rmse, spearman
from danling.metric.functional.preprocess import (
    base_preprocess,
    preprocess_binary,
    preprocess_multiclass,
    preprocess_multilabel,
    preprocess_regression,
)
from danling.metric.functional.utils import infer_task
from danling.tensor import NestedTensor

torch.manual_seed(0)

ATOL = 1e-6


def test_auroc():
    # Binary classification
    pred = torch.rand(20)
    target = torch.randint(0, 2, (20,))
    expected = binary_auroc(pred, target)
    actual = auroc(pred, target, task="binary")
    assert abs(expected - actual) < ATOL

    # Multiclass classification
    num_classes = 4
    pred = torch.rand(15, num_classes)
    pred = pred / pred.sum(dim=1, keepdim=True)
    target = torch.randint(0, num_classes, (15,))
    expected = multiclass_auroc(pred, target, num_classes=num_classes, average="macro")
    actual = auroc(pred, target, num_classes=num_classes, average="macro")
    assert abs(expected - actual) < ATOL

    # Multilabel classification
    num_labels = 3
    pred = torch.rand(10, num_labels)
    target = torch.randint(0, 2, (10, num_labels))
    expected = multilabel_auroc(pred, target, num_labels=num_labels, average="macro")
    actual = auroc(pred, target, num_labels=num_labels, average="macro")
    assert abs(expected - actual) < ATOL

    # Test with NestedTensor
    input_list = [torch.rand(3), torch.rand(5), torch.rand(2)]
    target_list = [torch.randint(0, 2, (3,)), torch.randint(0, 2, (5,)), torch.randint(0, 2, (2,))]
    input_nested = NestedTensor(input_list)
    target_nested = NestedTensor(target_list)
    input_tensor = torch.cat(input_list)
    target_tensor = torch.cat(target_list)
    expected = auroc(input_tensor, target_tensor, task="binary")
    actual = auroc(input_nested, target_nested, task="binary")
    assert abs(expected - actual) < ATOL

    # Test with ignore_index
    target_with_ignore = target_tensor.clone()
    target_with_ignore[target_with_ignore == 0] = -100
    valid_mask = target_with_ignore != -100
    expected = auroc(input_tensor[valid_mask], target_tensor[valid_mask], task="binary")
    actual = auroc(input_tensor, target_with_ignore, task="binary", ignore_index=-100)
    assert abs(expected - actual) < ATOL


def test_auprc():
    # Binary classification
    pred = torch.rand(20)
    target = torch.randint(0, 2, (20,))
    expected = binary_auprc(pred, target)
    actual = auprc(pred, target, task="binary")
    assert abs(expected - actual) < ATOL

    # Multiclass classification
    num_classes = 4
    pred = torch.rand(15, num_classes)
    pred = pred / pred.sum(dim=1, keepdim=True)
    target = torch.randint(0, num_classes, (15,))
    expected = multiclass_auprc(pred, target, num_classes=num_classes, average="macro")
    actual = auprc(pred, target, num_classes=num_classes, average="macro")
    assert abs(expected - actual) < ATOL

    # Multilabel classification
    num_labels = 3
    pred = torch.rand(10, num_labels)
    target = torch.randint(0, 2, (10, num_labels))

    try:
        expected = multilabel_auprc(pred, target, num_labels=num_labels, average="macro")
        actual = auprc(pred, target, num_labels=num_labels, average="macro")
        assert abs(expected - actual) < ATOL
    except (ImportError, AttributeError):
        # Fallback if multilabel_auprc isn't available
        label_auprcs = []
        for i in range(num_labels):
            label_auprc = auprc(pred[:, i], target[:, i], task="binary")
            label_auprcs.append(label_auprc)
        expected = sum(label_auprcs) / len(label_auprcs)
        actual = auprc(pred, target, num_labels=num_labels, average="macro")
        assert abs(expected - actual) < ATOL

    # Test with NestedTensor
    input_list = [torch.rand(3, num_labels), torch.rand(5, num_labels), torch.rand(2, num_labels)]
    target_list = [
        torch.randint(0, 2, (3, num_labels)),
        torch.randint(0, 2, (5, num_labels)),
        torch.randint(0, 2, (2, num_labels)),
    ]
    input_nested = NestedTensor(input_list)
    target_nested = NestedTensor(target_list)
    input_tensor = torch.cat(input_list)
    target_tensor = torch.cat(target_list)
    expected = auprc(input_tensor, target_tensor, num_labels=num_labels)
    actual = auprc(input_nested, target_nested, num_labels=num_labels)
    assert abs(expected - actual) < ATOL


def test_f1_score():
    # Binary classification
    pred = torch.rand(20)
    target = torch.randint(0, 2, (20,))
    expected = binary_f1_score(pred, target, threshold=0.5)
    actual = f1_score(pred, target, threshold=0.5, task="binary")
    assert abs(expected - actual) < ATOL

    # Multiclass classification
    num_classes = 4
    pred = torch.rand(15, num_classes)
    pred = pred / pred.sum(dim=1, keepdim=True)
    target = torch.randint(0, num_classes, (15,))
    expected = multiclass_f1_score(pred, target, num_classes=num_classes, average="micro")
    actual = f1_score(pred, target, num_classes=num_classes, average="micro")
    assert abs(expected - actual) < ATOL

    # Multilabel classification
    num_labels = 3
    pred = torch.rand(10, num_labels)
    target = torch.randint(0, 2, (10, num_labels))
    expected = multilabel_f1_score(pred, target, num_labels=num_labels, average="micro")
    actual = f1_score(pred, target, num_labels=num_labels, average="micro")
    assert abs(expected - actual) < ATOL

    # Test with NestedTensor
    input_list = [torch.rand(3), torch.rand(5), torch.rand(2)]
    target_list = [torch.randint(0, 2, (3,)), torch.randint(0, 2, (5,)), torch.randint(0, 2, (2,))]
    input_nested = NestedTensor(input_list)
    target_nested = NestedTensor(target_list)
    input_tensor = torch.cat(input_list)
    target_tensor = torch.cat(target_list)
    expected = f1_score(input_tensor, target_tensor, task="binary")
    actual = f1_score(input_nested, target_nested, task="binary")
    assert abs(expected - actual) < ATOL

    # Test with ignore_index
    target_with_ignore = target_tensor.clone()
    target_with_ignore[target_with_ignore == 0] = -100
    valid_mask = target_with_ignore != -100
    expected = f1_score(input_tensor[valid_mask], target_tensor[valid_mask], task="binary")
    actual = f1_score(input_tensor, target_with_ignore, task="binary", ignore_index=-100)
    assert abs(expected - actual) < ATOL


def test_accuracy(threshold=0.5):
    # Binary classification
    pred = torch.rand(20)
    target = torch.randint(0, 2, (20,))
    expected = binary_accuracy(pred, target, threshold=threshold)
    actual = accuracy(pred, target, threshold=threshold, task="binary")
    assert abs(expected - actual) < ATOL

    # Multiclass classification
    num_classes = 4
    pred = torch.rand(15, num_classes)
    pred = pred / pred.sum(dim=1, keepdim=True)
    target = torch.randint(0, num_classes, (15,))
    expected = multiclass_accuracy(pred, target, num_classes=num_classes, average="micro")
    actual = accuracy(pred, target, num_classes=num_classes, average="micro")
    assert abs(expected - actual) < ATOL

    # Multilabel classification
    num_labels = 3
    pred = torch.rand(10, num_labels)
    target = torch.randint(0, 2, (10, num_labels))
    expected = multilabel_accuracy(pred, target, num_labels=num_labels, threshold=threshold)
    actual = accuracy(pred, target, num_labels=num_labels, threshold=threshold)
    assert abs(expected - actual) < ATOL

    # Test with NestedTensor
    input_list = [torch.rand(3, num_labels), torch.rand(5, num_labels), torch.rand(2, num_labels)]
    target_list = [
        torch.randint(0, 2, (3, num_labels)),
        torch.randint(0, 2, (5, num_labels)),
        torch.randint(0, 2, (2, num_labels)),
    ]
    input_nested = NestedTensor(input_list)
    target_nested = NestedTensor(target_list)
    input_tensor = torch.cat(input_list)
    target_tensor = torch.cat(target_list)
    expected = accuracy(input_tensor, target_tensor, num_labels=num_labels)
    actual = accuracy(input_nested, target_nested, num_labels=num_labels)
    assert abs(expected - actual) < ATOL


def test_mcc():
    # Binary classification
    pred = torch.rand(20)
    target = torch.randint(0, 2, (20,))
    expected = matthews_corrcoef(pred, target, task="binary", threshold=0.5)
    actual = mcc(pred, target, task="binary", threshold=0.5)
    assert abs(expected - actual) < ATOL

    # Multiclass classification
    num_classes = 4
    pred = torch.rand(15, num_classes)
    pred = pred / pred.sum(dim=1, keepdim=True)
    target = torch.randint(0, num_classes, (15,))
    expected = matthews_corrcoef(pred, target, task="multiclass", num_classes=num_classes)
    actual = mcc(pred, target, task="multiclass", num_classes=num_classes)
    assert abs(expected - actual) < ATOL

    # Multilabel classification
    num_labels = 3
    pred = torch.rand(10, num_labels)
    target = torch.randint(0, 2, (10, num_labels))
    expected = matthews_corrcoef(pred, target, task="multilabel", num_labels=num_labels)
    actual = mcc(pred, target, task="multilabel", num_labels=num_labels)
    assert abs(expected - actual) < ATOL

    # Test with NestedTensor
    input_list = [torch.rand(3, num_labels), torch.rand(5, num_labels), torch.rand(2, num_labels)]
    target_list = [
        torch.randint(0, 2, (3, num_labels)),
        torch.randint(0, 2, (5, num_labels)),
        torch.randint(0, 2, (2, num_labels)),
    ]
    input_nested = NestedTensor(input_list)
    target_nested = NestedTensor(target_list)
    input_tensor = torch.cat(input_list)
    target_tensor = torch.cat(target_list)
    expected = mcc(input_tensor, target_tensor, task="multilabel", num_labels=num_labels)
    actual = mcc(input_nested, target_nested, task="multilabel", num_labels=num_labels)
    assert abs(expected - actual) < ATOL


def test_pearson():
    # Single output case
    pred = torch.randn(20)
    target = torch.randn(20)
    expected = pearson_corrcoef(pred, target)
    actual = pearson(pred, target)
    assert abs(expected - actual) < ATOL

    # Multi-output case
    num_outputs = 3
    pred = torch.randn(15, num_outputs)
    target = torch.randn(15, num_outputs)
    expected = pearson_corrcoef(pred, target).mean()
    actual = pearson(pred, target, num_outputs=num_outputs)
    assert abs(expected - actual) < ATOL

    # Test with NestedTensor
    input_list = [torch.randn(3), torch.randn(5), torch.randn(2)]
    target_list = [torch.randn(3), torch.randn(5), torch.randn(2)]
    input_nested = NestedTensor(input_list)
    target_nested = NestedTensor(target_list)
    input_tensor = torch.cat(input_list)
    target_tensor = torch.cat(target_list)
    expected = pearson(input_tensor, target_tensor)
    actual = pearson(input_nested, target_nested)
    assert abs(expected - actual) < ATOL


def test_spearman():
    # Single output case
    pred = torch.randn(20)
    target = torch.randn(20)
    expected = spearman_corrcoef(pred, target)
    actual = spearman(pred, target)
    assert abs(expected - actual) < ATOL

    # Multi-output case
    num_outputs = 3
    pred = torch.randn(15, num_outputs)
    target = torch.randn(15, num_outputs)
    expected = spearman_corrcoef(pred, target).mean()
    actual = spearman(pred, target, num_outputs=num_outputs)
    assert abs(expected - actual) < ATOL

    # Test with NestedTensor
    input_list = [torch.randn(3), torch.randn(5), torch.randn(2)]
    target_list = [torch.randn(3), torch.randn(5), torch.randn(2)]
    input_nested = NestedTensor(input_list)
    target_nested = NestedTensor(target_list)
    input_tensor = torch.cat(input_list)
    target_tensor = torch.cat(target_list)
    expected = spearman(input_tensor, target_tensor)
    actual = spearman(input_nested, target_nested)
    assert abs(expected - actual) < ATOL


def test_r2_score():
    # Single output case
    pred = torch.randn(20)
    target = torch.randn(20)
    expected = tef_r2_score(pred, target)
    actual = r2_score(pred, target)
    assert abs(expected - actual) < ATOL

    # Multi-output case
    num_outputs = 3
    pred = torch.randn(15, num_outputs)
    target = torch.randn(15, num_outputs)
    expected = tef_r2_score(pred, target, multioutput="raw_values").mean()
    actual = r2_score(pred, target, num_outputs=num_outputs)
    assert abs(expected - actual) < ATOL

    # Test with NestedTensor
    input_list = [torch.randn(3), torch.randn(5), torch.randn(2)]
    target_list = [torch.randn(3), torch.randn(5), torch.randn(2)]
    input_nested = NestedTensor(input_list)
    target_nested = NestedTensor(target_list)
    input_tensor = torch.cat(input_list)
    target_tensor = torch.cat(target_list)
    expected = r2_score(input_tensor, target_tensor)
    actual = r2_score(input_nested, target_nested)
    assert abs(expected - actual) < ATOL


def test_mse():
    # Single output case
    pred = torch.randn(20)
    target = torch.randn(20)
    expected = mean_squared_error(pred, target)
    actual = mse(pred, target)
    assert abs(expected - actual) < ATOL

    # Multi-output case
    num_outputs = 3
    pred = torch.randn(15, num_outputs)
    target = torch.randn(15, num_outputs)
    expected = mean_squared_error(pred, target).mean()
    actual = mse(pred, target, num_outputs=num_outputs)
    assert abs(expected - actual) < ATOL

    # Test with NestedTensor
    input_list = [torch.randn(3), torch.randn(5), torch.randn(2)]
    target_list = [torch.randn(3), torch.randn(5), torch.randn(2)]
    input_nested = NestedTensor(input_list)
    target_nested = NestedTensor(target_list)
    input_tensor = torch.cat(input_list)
    target_tensor = torch.cat(target_list)
    expected = mse(input_tensor, target_tensor)
    actual = mse(input_nested, target_nested)
    assert abs(expected - actual) < ATOL


def test_rmse():
    # Single output case
    pred = torch.randn(20)
    target = torch.randn(20)
    expected = torch.sqrt(mean_squared_error(pred, target))
    actual = rmse(pred, target)
    assert abs(expected - actual) < ATOL

    # Multi-output case
    num_outputs = 3
    pred = torch.randn(15, num_outputs)
    target = torch.randn(15, num_outputs)
    expected = torch.sqrt(mean_squared_error(pred, target)).mean()
    actual = rmse(pred, target, num_outputs=num_outputs)
    assert abs(expected - actual) < ATOL

    # Test with NestedTensor
    input_list = [torch.randn(3), torch.randn(5), torch.randn(2)]
    target_list = [torch.randn(3), torch.randn(5), torch.randn(2)]
    input_nested = NestedTensor(input_list)
    target_nested = NestedTensor(target_list)
    input_tensor = torch.cat(input_list)
    target_tensor = torch.cat(target_list)
    expected = rmse(input_tensor, target_tensor)
    actual = rmse(input_nested, target_nested)
    assert abs(expected - actual) < ATOL


def test_infer_task():
    assert infer_task(None, None) == "binary"
    assert infer_task(10, None) == "multiclass"
    assert infer_task(None, 5) == "multilabel"

    with raises(ValueError):
        infer_task(10, 5)


def test_base_preprocess():
    # Basic preprocessing
    input_tensor = torch.tensor([0.1, 0.5, 0.9])
    target_tensor = torch.tensor([0, 1, 1])
    processed_input, processed_target = base_preprocess(input_tensor, target_tensor)
    assert torch.allclose(processed_input, input_tensor)
    assert torch.allclose(processed_target, target_tensor)

    # List inputs
    input_list = [0.1, 0.5, 0.9]
    target_list = [0, 1, 1]
    processed_input, processed_target = base_preprocess(input_list, target_list)
    assert torch.allclose(processed_input, torch.tensor(input_list))
    assert torch.allclose(processed_target, torch.tensor(target_list))

    # Ignore index
    input_tensor = torch.tensor([0.1, 0.5, 0.9, 0.3])
    target_tensor = torch.tensor([0, 1, -100, 0])
    processed_input, processed_target = base_preprocess(input_tensor, target_tensor, ignore_index=-100)
    assert processed_input.shape == torch.Size([3])
    assert processed_target.shape == torch.Size([3])
    assert torch.allclose(processed_input, torch.tensor([0.1, 0.5, 0.3]))
    assert torch.allclose(processed_target, torch.tensor([0, 1, 0]))

    # Ignore NaN
    input_tensor = torch.tensor([0.1, 0.5, 0.3, 0.1])
    target_tensor = torch.tensor([0.0, 1.0, 1.0, float("nan")])
    processed_input, processed_target = base_preprocess(input_tensor, target_tensor, ignore_nan=True)
    assert processed_input.shape == torch.Size([3])
    assert processed_target.shape == torch.Size([3])
    assert torch.allclose(processed_input, torch.tensor([0.1, 0.5, 0.3]))
    assert torch.allclose(processed_target, torch.tensor([0.0, 1.0, 1.0]))

    # NestedTensor
    input_nested = NestedTensor([torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4, 0.5])])
    target_nested = NestedTensor([torch.tensor([0, 1]), torch.tensor([1, 0, 1])])
    processed_input, processed_target = base_preprocess(input_nested, target_nested)
    assert torch.allclose(processed_input, input_nested.concat)
    assert torch.allclose(processed_target, target_nested.concat)

    # Error case: mismatched tensor and nested tensor
    target_tensor = torch.tensor([0, 1, 1, 0, 1])
    with raises(ValueError):
        processed_input, processed_target = base_preprocess(input_nested, target_tensor)

    # Nested list
    input_list = [[0.1, 0.2], [0.3, 0.4]]
    target_list = [[0, 1], [1, 0]]
    processed_input, processed_target = base_preprocess(input_list, target_list)
    assert isinstance(processed_input, torch.Tensor)
    assert isinstance(processed_target, torch.Tensor)
    assert processed_input.shape == torch.Size([2, 2])
    assert processed_target.shape == torch.Size([2, 2])

    # Tensor and NestedTensor
    input_batch_tensor = torch.tensor([[0.1, 0.2, 0.0], [0.3, 0.4, 0.5]])
    expected_result = torch.cat([input_batch_tensor[0, :2], input_batch_tensor[1, :3]])
    processed_input, processed_target = base_preprocess(input_batch_tensor, target_nested)
    assert torch.allclose(processed_input, expected_result)
    assert torch.allclose(processed_target, target_nested.concat)


def test_task_specific_preprocess():
    # Regression
    input_tensor = torch.randn(10)
    target_tensor = torch.randn(10)
    processed_input, processed_target = preprocess_regression(input_tensor, target_tensor)
    assert processed_input.shape == torch.Size([10])
    assert processed_target.shape == torch.Size([10])

    # Multi-output regression
    input_tensor = torch.randn(5, 3)
    target_tensor = torch.randn(5, 3)
    processed_input, processed_target = preprocess_regression(input_tensor, target_tensor, num_outputs=3)
    assert processed_input.shape == torch.Size([5, 3])
    assert processed_target.shape == torch.Size([5, 3])

    # Regression with NaN
    input_tensor = torch.tensor([0.1, 0.2, 0.3])
    target_tensor = torch.tensor([0.1, 0.2, float("nan")])
    processed_input, processed_target = preprocess_regression(input_tensor, target_tensor, ignore_nan=True)
    assert processed_input.shape == torch.Size([2])
    assert processed_target.shape == torch.Size([2])
    assert torch.allclose(processed_input, torch.tensor([0.1, 0.2]))
    assert torch.allclose(processed_target, torch.tensor([0.1, 0.2]))

    # Binary classification
    input_tensor = torch.tensor([0.1, 0.7, 0.4])
    target_tensor = torch.tensor([0, 1, 0])
    processed_input, processed_target = preprocess_binary(input_tensor, target_tensor)
    assert processed_input.shape == torch.Size([3])
    assert processed_target.shape == torch.Size([3])
    assert torch.allclose(processed_input, input_tensor)
    assert torch.allclose(processed_target, target_tensor)

    # Binary classification with normalization
    input_tensor = torch.tensor([-1.0, 2.0, 0.5])
    target_tensor = torch.tensor([0, 1, 0])
    processed_input, processed_target = preprocess_binary(input_tensor, target_tensor)
    assert processed_input.shape == torch.Size([3])
    assert processed_target.shape == torch.Size([3])
    assert processed_input.min() >= 0.0 and processed_input.max() <= 1.0
    assert torch.allclose(processed_target, target_tensor)

    # Multiclass classification
    input_tensor = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4], [0.6, 0.3, 0.1]])
    target_tensor = torch.tensor([1, 2, 0])
    processed_input, processed_target = preprocess_multiclass(input_tensor, target_tensor, num_classes=3)
    assert processed_input.shape == torch.Size([3, 3])
    assert processed_target.shape == torch.Size([3])
    assert torch.allclose(processed_input, input_tensor)
    assert torch.allclose(processed_target, target_tensor)

    # Multiclass with normalization
    input_tensor = torch.tensor([[-1.0, 2.0, 0.5], [0.0, -3.0, 1.0], [1.0, 2.0, -1.0]])
    target_tensor = torch.tensor([1, 2, 1])
    processed_input, processed_target = preprocess_multiclass(input_tensor, target_tensor, num_classes=3)
    assert processed_input.shape == torch.Size([3, 3])
    assert processed_target.shape == torch.Size([3])
    assert (processed_input >= 0.0).all() and (processed_input <= 1.0).all()
    assert torch.allclose(processed_input.sum(dim=1), torch.ones(3))
    assert torch.allclose(processed_target, target_tensor)

    # Multilabel classification
    input_tensor = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4], [0.6, 0.3, 0.1]])
    target_tensor = torch.tensor([[0, 1, 0], [1, 0, 1], [1, 0, 0]])
    processed_input, processed_target = preprocess_multilabel(input_tensor, target_tensor, num_labels=3)
    assert processed_input.shape == torch.Size([3, 3])
    assert processed_target.shape == torch.Size([3, 3])
    assert torch.allclose(processed_input, input_tensor)
    assert torch.allclose(processed_target, target_tensor)

    # Multilabel with normalization
    input_tensor = torch.tensor([[-1.0, 2.0, 0.5], [0.0, -3.0, 1.0], [1.0, 2.0, -1.0]])
    target_tensor = torch.tensor([[0, 1, 0], [1, 0, 1], [1, 0, 0]])
    processed_input, processed_target = preprocess_multilabel(input_tensor, target_tensor, num_labels=3)
    assert processed_input.shape == torch.Size([3, 3])
    assert processed_target.shape == torch.Size([3, 3])
    assert (processed_input >= 0.0).all() and (processed_input <= 1.0).all()
    assert torch.allclose(processed_target, target_tensor)
