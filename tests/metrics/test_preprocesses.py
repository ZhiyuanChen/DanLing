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

import pytest
import torch
from pytest import raises

from danling.metrics.preprocesses import (
    infer_task,
    preprocess,
    preprocess_binary,
    preprocess_multiclass,
    preprocess_multilabel,
    preprocess_regression,
)
from danling.tensors import NestedTensor


def test_infer_task():
    assert infer_task(None, None) == "binary"
    assert infer_task(10, None) == "multiclass"
    assert infer_task(None, 5) == "multilabel"

    with pytest.raises(ValueError):
        infer_task(10, 5)


def test_preprocess():
    # Basic preprocessing
    input_tensor = torch.tensor([0.1, 0.5, 0.9])
    target_tensor = torch.tensor([0, 1, 1])
    processed_input, processed_target = preprocess(input_tensor, target_tensor)
    assert torch.allclose(processed_input, input_tensor)
    assert torch.allclose(processed_target, target_tensor)

    # List inputs
    input_list = [0.1, 0.5, 0.9]
    target_list = [0, 1, 1]
    processed_input, processed_target = preprocess(input_list, target_list)
    assert torch.allclose(processed_input, torch.tensor(input_list))
    assert torch.allclose(processed_target, torch.tensor(target_list))

    # Ignore index
    input_tensor = torch.tensor([0.1, 0.5, 0.9, 0.3])
    target_tensor = torch.tensor([0, 1, -100, 0])
    processed_input, processed_target = preprocess(input_tensor, target_tensor, ignore_index=-100)
    assert processed_input.shape == torch.Size([3])
    assert processed_target.shape == torch.Size([3])
    assert torch.allclose(processed_input, torch.tensor([0.1, 0.5, 0.3]))
    assert torch.allclose(processed_target, torch.tensor([0, 1, 0]))

    # Ignore NaN
    input_tensor = torch.tensor([0.1, 0.5, 0.3, 0.1])
    target_tensor = torch.tensor([0.0, 1.0, 1.0, float("nan")])
    processed_input, processed_target = preprocess(input_tensor, target_tensor, ignore_nan=True)
    assert processed_input.shape == torch.Size([3])
    assert processed_target.shape == torch.Size([3])
    assert torch.allclose(processed_input, torch.tensor([0.1, 0.5, 0.3]))
    assert torch.allclose(processed_target, torch.tensor([0.0, 1.0, 1.0]))

    # NestedTensor
    input_nested = NestedTensor([torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4, 0.5])])
    target_nested = NestedTensor([torch.tensor([0, 1]), torch.tensor([1, 0, 1])])
    processed_input, processed_target = preprocess(input_nested, target_nested)
    assert torch.allclose(processed_input, input_nested.concat)
    assert torch.allclose(processed_target, target_nested.concat)

    # Error case: mismatched tensor and nested tensor
    target_tensor = torch.tensor([0, 1, 1, 0, 1])
    with raises(ValueError):
        processed_input, processed_target = preprocess(input_nested, target_tensor)

    # Nested list
    input_list = [[0.1, 0.2], [0.3, 0.4]]
    target_list = [[0, 1], [1, 0]]
    processed_input, processed_target = preprocess(input_list, target_list)
    assert isinstance(processed_input, torch.Tensor)
    assert isinstance(processed_target, torch.Tensor)
    assert processed_input.shape == torch.Size([2, 2])
    assert processed_target.shape == torch.Size([2, 2])

    # Tensor and NestedTensor
    input_batch_tensor = torch.tensor([[0.1, 0.2, 0.0], [0.3, 0.4, 0.5]])
    expected_result = torch.cat([input_batch_tensor[0, :2], input_batch_tensor[1, :3]])
    processed_input, processed_target = preprocess(input_batch_tensor, target_nested)
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
