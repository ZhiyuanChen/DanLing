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

import pytest
import torch
from pytest import raises

from danling.metrics.preprocess import (
    base_preprocess,
    preprocess_binary,
    preprocess_classification,
    preprocess_multiclass,
    preprocess_multilabel,
    preprocess_regression,
)
from danling.tensors import NestedTensor

ATOL = 1e-6


class TestBasePreprocess:
    def test_idempotent_for_dense_tensors(self):
        input_tensor = torch.tensor([0.1, 0.5, 0.9])
        target_tensor = torch.tensor([0, 1, 1])

        once_input, once_target = base_preprocess(input_tensor, target_tensor)
        twice_input, twice_target = base_preprocess(once_input, once_target)

        assert torch.allclose(twice_input, once_input)
        assert torch.allclose(twice_target, once_target)

    def test_preserves_tensor_inputs(self):
        input_tensor = torch.tensor([0.1, 0.5, 0.9])
        target_tensor = torch.tensor([0, 1, 1])
        processed_input, processed_target = base_preprocess(input_tensor, target_tensor)
        assert torch.allclose(processed_input, input_tensor)
        assert torch.allclose(processed_target, target_tensor)

    def test_converts_list_inputs(self):
        input_list = [0.1, 0.5, 0.9]
        target_list = [0, 1, 1]
        processed_input, processed_target = base_preprocess(input_list, target_list)
        assert torch.allclose(processed_input, torch.tensor(input_list))
        assert torch.allclose(processed_target, torch.tensor(target_list))

    def test_applies_ignore_index(self):
        input_tensor = torch.tensor([0.1, 0.5, 0.9, 0.3])
        target_tensor = torch.tensor([0, 1, -100, 0])
        processed_input, processed_target = base_preprocess(input_tensor, target_tensor, ignore_index=-100)
        assert processed_input.shape == torch.Size([3])
        assert processed_target.shape == torch.Size([3])
        assert torch.allclose(processed_input, torch.tensor([0.1, 0.5, 0.3]))
        assert torch.allclose(processed_target, torch.tensor([0, 1, 0]))

    def test_applies_ignore_nan(self):
        input_tensor = torch.tensor([0.1, 0.5, 0.3, 0.1])
        target_tensor = torch.tensor([0.0, 1.0, 1.0, float("nan")])
        processed_input, processed_target = base_preprocess(input_tensor, target_tensor, ignore_nan=True)
        assert processed_input.shape == torch.Size([3])
        assert processed_target.shape == torch.Size([3])
        assert torch.allclose(processed_input, torch.tensor([0.1, 0.5, 0.3]))
        assert torch.allclose(processed_target, torch.tensor([0.0, 1.0, 1.0]))

    def test_flattens_nested_tensors(self):
        input_nested = NestedTensor([torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4, 0.5])])
        target_nested = NestedTensor([torch.tensor([0, 1]), torch.tensor([1, 0, 1])])
        processed_input, processed_target = base_preprocess(input_nested, target_nested)
        assert torch.allclose(processed_input, input_nested.concat)
        assert torch.allclose(processed_target, target_nested.concat)

    def test_rejects_mismatched_dense_and_nested_inputs(self):
        input_nested = NestedTensor([torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4, 0.5])])
        target_tensor = torch.tensor([0, 1, 1, 0, 1])
        with raises(ValueError):
            base_preprocess(input_nested, target_tensor)

    def test_converts_nested_lists_to_dense_tensors(self):
        input_list = [[0.1, 0.2], [0.3, 0.4]]
        target_list = [[0, 1], [1, 0]]
        processed_input, processed_target = base_preprocess(input_list, target_list)
        assert isinstance(processed_input, torch.Tensor)
        assert isinstance(processed_target, torch.Tensor)
        assert processed_input.shape == torch.Size([2, 2])
        assert processed_target.shape == torch.Size([2, 2])

    def test_handles_dense_inputs_with_nested_targets(self):
        input_batch_tensor = torch.tensor([[0.1, 0.2, 0.0], [0.3, 0.4, 0.5]])
        target_nested = NestedTensor([torch.tensor([0, 1]), torch.tensor([1, 0, 1])])
        expected_result = torch.cat([input_batch_tensor[0, :2], input_batch_tensor[1, :3]])
        processed_input, processed_target = base_preprocess(input_batch_tensor, target_nested)
        assert torch.allclose(processed_input, expected_result)
        assert torch.allclose(processed_target, target_nested.concat)

    def test_flattens_dense_and_nested_batches_equivalently(self):
        input_nested = NestedTensor([torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4, 0.5])])
        target_nested = NestedTensor([torch.tensor([0, 1]), torch.tensor([1, 0, 1])])

        nested_input, nested_target = base_preprocess(input_nested, target_nested)
        dense_input, dense_target = base_preprocess(input_nested.concat, target_nested.concat)

        assert torch.allclose(nested_input, dense_input)
        assert torch.allclose(nested_target, dense_target)


class TestRegressionPreprocess:
    def test_flattens_nested_and_dense_equivalently(self):
        input_nested = NestedTensor([torch.tensor([[0.1, 0.2], [0.3, 0.4]]), torch.tensor([[0.5, 0.6]])])
        target_nested = NestedTensor([torch.tensor([[1.0, 1.1], [1.2, 1.3]]), torch.tensor([[1.4, 1.5]])])

        nested_input, nested_target = preprocess_regression(input_nested, target_nested, num_outputs=2)
        dense_input, dense_target = preprocess_regression(input_nested.concat, target_nested.concat, num_outputs=2)

        assert torch.allclose(nested_input, dense_input)
        assert torch.allclose(nested_target, dense_target)

    def test_preserves_single_output_shape(self):
        input_tensor = torch.tensor([0.1, -0.2, 0.3, -0.4])
        target_tensor = torch.tensor([0.0, 0.2, 0.4, 0.6])
        processed_input, processed_target = preprocess_regression(input_tensor, target_tensor)
        assert processed_input.shape == torch.Size([4])
        assert processed_target.shape == torch.Size([4])

    def test_preserves_multioutput_shape(self):
        input_tensor = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        target_tensor = torch.tensor([[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
        processed_input, processed_target = preprocess_regression(input_tensor, target_tensor, num_outputs=3)
        assert processed_input.shape == torch.Size([2, 3])
        assert processed_target.shape == torch.Size([2, 3])

    def test_ignore_nan_filters_single_output_elements(self):
        input_tensor = torch.tensor([0.1, 0.2, 0.3])
        target_tensor = torch.tensor([0.1, 0.2, float("nan")])
        processed_input, processed_target = preprocess_regression(input_tensor, target_tensor, ignore_nan=True)
        assert processed_input.shape == torch.Size([2])
        assert processed_target.shape == torch.Size([2])
        assert torch.allclose(processed_input, torch.tensor([0.1, 0.2]))
        assert torch.allclose(processed_target, torch.tensor([0.1, 0.2]))

    def test_multioutput_drops_rows_with_nan_targets(self):
        input_tensor = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        target_tensor = torch.tensor([[1.0, float("nan")], [2.0, 20.0], [float("nan"), 30.0]])

        processed_input, processed_target = preprocess_regression(
            input_tensor,
            target_tensor,
            num_outputs=2,
            ignore_nan=True,
        )

        assert processed_input.shape == torch.Size([1, 2])
        assert processed_target.shape == torch.Size([1, 2])
        assert torch.allclose(processed_input, torch.tensor([[2.0, 20.0]]))
        assert torch.allclose(processed_target, torch.tensor([[2.0, 20.0]]))

    def test_multioutput_all_nan_rows_return_empty(self):
        input_tensor = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
        target_tensor = torch.tensor([[float("nan"), 10.0], [2.0, float("nan")]])

        processed_input, processed_target = preprocess_regression(
            input_tensor,
            target_tensor,
            num_outputs=2,
            ignore_nan=True,
        )

        assert processed_input.shape == torch.Size([0, 2])
        assert processed_target.shape == torch.Size([0, 2])

    def test_ignore_nan_false_preserves_rows_and_nans(self):
        input_tensor = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
        target_tensor = torch.tensor([[1.0, float("nan")], [2.0, 20.0]])

        processed_input, processed_target = preprocess_regression(
            input_tensor,
            target_tensor,
            num_outputs=2,
            ignore_nan=False,
        )

        assert torch.allclose(processed_input, input_tensor)
        assert torch.allclose(processed_target, target_tensor, equal_nan=True)

    def test_rejects_non_floating_targets(self):
        with raises(TypeError, match="floating point tensors"):
            preprocess_regression(
                torch.tensor([0.1, 0.2, 0.3]),
                torch.tensor([0, 1, 0], dtype=torch.long),
            )

    def test_reshapes_noncontiguous_multioutput_inputs(self):
        input_tensor = torch.arange(20, dtype=torch.float32).reshape(5, 4)[:, ::2]
        target_tensor = torch.arange(20, dtype=torch.float32).reshape(5, 4)[:, 1::2]

        assert input_tensor.is_contiguous() is False
        assert target_tensor.is_contiguous() is False

        processed_input, processed_target = preprocess_regression(input_tensor, target_tensor, num_outputs=2)

        assert processed_input.shape == torch.Size([5, 2])
        assert processed_target.shape == torch.Size([5, 2])
        assert torch.allclose(processed_input, input_tensor)
        assert torch.allclose(processed_target, target_tensor)


class TestBinaryPreprocess:
    def test_idempotent_for_probabilities(self):
        input_tensor = torch.tensor([0.1, 0.7, 0.4])
        target_tensor = torch.tensor([0, 1, 0])

        once_input, once_target = preprocess_binary(input_tensor, target_tensor)
        twice_input, twice_target = preprocess_binary(once_input, once_target)

        assert torch.allclose(twice_input, once_input)
        assert torch.allclose(twice_target, once_target)

    def test_preserves_probabilities(self):
        input_tensor = torch.tensor([0.1, 0.7, 0.4])
        target_tensor = torch.tensor([0, 1, 0])
        processed_input, processed_target = preprocess_binary(input_tensor, target_tensor)
        assert processed_input.shape == torch.Size([3])
        assert processed_target.shape == torch.Size([3])
        assert torch.allclose(processed_input, input_tensor)
        assert torch.allclose(processed_target, target_tensor)

    def test_normalizes_logits(self):
        input_tensor = torch.tensor([-1.0, 2.0, 0.5])
        target_tensor = torch.tensor([0, 1, 0])
        processed_input, processed_target = preprocess_binary(input_tensor, target_tensor)
        assert processed_input.shape == torch.Size([3])
        assert processed_target.shape == torch.Size([3])
        assert processed_input.min() >= 0.0 and processed_input.max() <= 1.0
        assert torch.allclose(processed_target, target_tensor)

    def test_ignore_index_filters_examples(self):
        input_tensor = torch.tensor([0.1, 0.7, 0.4, 0.9])
        target_tensor = torch.tensor([0, -100, 0, 1])
        processed_input, processed_target = preprocess_binary(input_tensor, target_tensor, ignore_index=-100)
        assert processed_input.shape == torch.Size([3])
        assert processed_target.shape == torch.Size([3])
        assert torch.allclose(processed_input, torch.tensor([0.1, 0.4, 0.9]))
        assert torch.allclose(processed_target, torch.tensor([0, 0, 1]))


class TestMulticlassPreprocess:
    def test_idempotent_for_normalized_probabilities(self):
        input_tensor = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4], [0.6, 0.3, 0.1]])
        target_tensor = torch.tensor([1, 2, 0])

        once_input, once_target = preprocess_multiclass(input_tensor, target_tensor, num_classes=3)
        twice_input, twice_target = preprocess_multiclass(once_input, once_target, num_classes=3)

        assert torch.allclose(twice_input, once_input)
        assert torch.allclose(twice_target, once_target)

    def test_preserves_probability_inputs(self):
        input_tensor = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4], [0.6, 0.3, 0.1]])
        target_tensor = torch.tensor([1, 2, 0])
        processed_input, processed_target = preprocess_multiclass(input_tensor, target_tensor, num_classes=3)
        assert processed_input.shape == torch.Size([3, 3])
        assert processed_target.shape == torch.Size([3])
        assert torch.allclose(processed_input, input_tensor)
        assert torch.allclose(processed_target, target_tensor)

    def test_normalizes_logits(self):
        input_tensor = torch.tensor([[-1.0, 2.0, 0.5], [0.0, -3.0, 1.0], [1.0, 2.0, -1.0]])
        target_tensor = torch.tensor([1, 2, 1])
        processed_input, processed_target = preprocess_multiclass(input_tensor, target_tensor, num_classes=3)
        assert processed_input.shape == torch.Size([3, 3])
        assert processed_target.shape == torch.Size([3])
        assert (processed_input >= 0.0).all() and (processed_input <= 1.0).all()
        assert torch.allclose(processed_input.sum(dim=1), torch.ones(3))
        assert torch.allclose(processed_target, target_tensor)

    def test_ignore_index_filters_examples(self):
        input_tensor = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4], [0.6, 0.3, 0.1]])
        target_tensor = torch.tensor([1, -100, 0])
        processed_input, processed_target = preprocess_multiclass(
            input_tensor, target_tensor, num_classes=3, ignore_index=-100
        )
        assert processed_input.shape == torch.Size([2, 3])
        assert processed_target.shape == torch.Size([2])
        assert torch.allclose(processed_input, torch.tensor([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]]))
        assert torch.allclose(processed_target, torch.tensor([1, 0]))

    def test_reshapes_noncontiguous_inputs(self):
        input_tensor = torch.arange(20, dtype=torch.float32).reshape(5, 4)[:, ::2]
        target_tensor = torch.tensor([0, 1, 0, 1, 0])

        assert input_tensor.is_contiguous() is False

        processed_input, processed_target = preprocess_multiclass(input_tensor, target_tensor, num_classes=2)

        assert processed_input.shape == torch.Size([5, 2])
        assert processed_target.shape == torch.Size([5])
        assert torch.allclose(processed_input, input_tensor.softmax(dim=-1))
        assert torch.allclose(processed_target, target_tensor)


class TestMultilabelPreprocess:
    def test_idempotent_for_probabilities(self):
        input_tensor = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4], [0.6, 0.3, 0.1]])
        target_tensor = torch.tensor([[0, 1, 0], [1, 0, 1], [1, 0, 0]])

        once_input, once_target = preprocess_multilabel(input_tensor, target_tensor, num_labels=3)
        twice_input, twice_target = preprocess_multilabel(once_input, once_target, num_labels=3)

        assert torch.allclose(twice_input, once_input)
        assert torch.allclose(twice_target, once_target)

    def test_preserves_probability_inputs(self):
        input_tensor = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4], [0.6, 0.3, 0.1]])
        target_tensor = torch.tensor([[0, 1, 0], [1, 0, 1], [1, 0, 0]])
        processed_input, processed_target = preprocess_multilabel(input_tensor, target_tensor, num_labels=3)
        assert processed_input.shape == torch.Size([3, 3])
        assert processed_target.shape == torch.Size([3, 3])
        assert torch.allclose(processed_input, input_tensor)
        assert torch.allclose(processed_target, target_tensor)

    def test_normalizes_logits(self):
        input_tensor = torch.tensor([[-1.0, 2.0, 0.5], [0.0, -3.0, 1.0], [1.0, 2.0, -1.0]])
        target_tensor = torch.tensor([[0, 1, 0], [1, 0, 1], [1, 0, 0]])
        processed_input, processed_target = preprocess_multilabel(input_tensor, target_tensor, num_labels=3)
        assert processed_input.shape == torch.Size([3, 3])
        assert processed_target.shape == torch.Size([3, 3])
        assert (processed_input >= 0.0).all() and (processed_input <= 1.0).all()
        assert torch.allclose(processed_target, target_tensor)

    def test_ignore_index_drops_fully_ignored_rows(self):
        input_tensor = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4], [0.6, 0.3, 0.1]])
        target_tensor = torch.tensor([[0, -100, 0], [-100, -100, -100], [1, 0, 0]])
        processed_input, processed_target = preprocess_multilabel(
            input_tensor,
            target_tensor,
            num_labels=3,
            ignore_index=-100,
        )
        assert processed_input.shape == torch.Size([2, 3])
        assert processed_target.shape == torch.Size([2, 3])
        assert torch.allclose(processed_input, torch.tensor([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]]))
        assert torch.allclose(processed_target, torch.tensor([[0, -100, 0], [1, 0, 0]]))

    def test_reshapes_noncontiguous_inputs(self):
        input_tensor = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2]
        target_tensor = torch.tensor(
            [
                [0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0],
            ],
            dtype=torch.long,
        )[:, ::2]

        assert input_tensor.is_contiguous() is False
        assert target_tensor.is_contiguous() is False

        processed_input, processed_target = preprocess_multilabel(input_tensor, target_tensor, num_labels=3)

        assert processed_input.shape == torch.Size([4, 3])
        assert processed_target.shape == torch.Size([4, 3])
        assert torch.allclose(processed_input, input_tensor.sigmoid())
        assert torch.allclose(processed_target, target_tensor)


class TestClassificationPreprocess:
    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({}, preprocess_binary),
            ({"task": "multiclass", "num_classes": 3}, preprocess_multiclass),
            ({"task": "multilabel", "num_labels": 2}, preprocess_multilabel),
        ],
    )
    def test_dispatches_to_task_preprocessors(self, kwargs, expected):
        if expected is preprocess_binary:
            input_tensor = torch.tensor([0.1, 0.7, 0.4])
            target_tensor = torch.tensor([0, 1, 0])
        elif expected is preprocess_multiclass:
            input_tensor = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]])
            target_tensor = torch.tensor([1, 2])
        else:
            input_tensor = torch.tensor([[0.1, 0.7], [0.3, 0.4]])
            target_tensor = torch.tensor([[0, 1], [1, 0]])

        actual_input, actual_target = preprocess_classification(input_tensor, target_tensor, **kwargs)
        expected_input, expected_target = expected(
            input_tensor, target_tensor, **{k: v for k, v in kwargs.items() if k != "task"}
        )

        assert torch.allclose(actual_input, expected_input)
        assert torch.allclose(actual_target, expected_target)

    def test_infers_multiclass_from_num_classes(self):
        input_tensor = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]])
        target_tensor = torch.tensor([1, 2])

        actual_input, actual_target = preprocess_classification(input_tensor, target_tensor, num_classes=3)
        expected_input, expected_target = preprocess_multiclass(input_tensor, target_tensor, num_classes=3)

        assert torch.allclose(actual_input, expected_input)
        assert torch.allclose(actual_target, expected_target)

    def test_infers_multilabel_from_num_labels(self):
        input_tensor = torch.tensor([[0.1, 0.7], [0.3, 0.4]])
        target_tensor = torch.tensor([[0, 1], [1, 0]])

        actual_input, actual_target = preprocess_classification(input_tensor, target_tensor, num_labels=2)
        expected_input, expected_target = preprocess_multilabel(input_tensor, target_tensor, num_labels=2)

        assert torch.allclose(actual_input, expected_input)
        assert torch.allclose(actual_target, expected_target)

    def test_rejects_invalid_task(self):
        with raises(ValueError, match="Invalid task"):
            preprocess_classification(torch.tensor([0.1]), torch.tensor([0]), task="invalid")


class TestTaskPreprocessors:
    def test_return_empty_tensors_for_all_ignored_batches(self):
        binary_input, binary_target = preprocess_binary(
            torch.randn(4),
            torch.full((4,), -100),
            ignore_index=-100,
        )
        assert binary_input.shape == binary_target.shape == torch.Size([0])

        multiclass_input, multiclass_target = preprocess_multiclass(
            torch.randn(4, 3),
            torch.full((4,), -100),
            num_classes=3,
            ignore_index=-100,
        )
        assert multiclass_input.shape == torch.Size([0, 3])
        assert multiclass_target.shape == torch.Size([0])

        multilabel_input, multilabel_target = preprocess_multilabel(
            torch.randn(4, 2),
            torch.full((4, 2), -100),
            num_labels=2,
            ignore_index=-100,
        )
        assert multilabel_input.shape == torch.Size([0, 2])
        assert multilabel_target.shape == torch.Size([0, 2])

        regression_input, regression_target = preprocess_regression(
            torch.randn(2, 3),
            torch.tensor([[1.0, float("nan"), 2.0], [3.0, 4.0, 5.0]]),
            num_outputs=3,
            ignore_nan=True,
        )
        assert regression_input.shape == regression_target.shape
