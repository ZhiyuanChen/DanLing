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
    @pytest.mark.parametrize("tensor", [True, False], ids=["tensor", "list"])
    @pytest.mark.parametrize(
        "inputs,targets", [([0.1, 0.5, 0.9], [0, 1, 1]), ([[0.1, 0.2], [0.3, 0.4]], [[0, 1], [1, 0]])]
    )
    def test_dense_inputs(self, tensor, inputs, targets):
        expected = torch.tensor(inputs), torch.tensor(targets)
        source = expected if tensor else (inputs, targets)
        actual = base_preprocess(*source)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(base_preprocess(*actual), actual)

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
        assert torch.allclose(processed_input, input_nested.values())
        assert torch.allclose(processed_target, target_nested.values())

    def test_mismatched_inputs(self):
        input_nested = NestedTensor([torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4, 0.5])])
        target_tensor = torch.tensor([0, 1, 1, 0, 1])
        with raises(ValueError):
            base_preprocess(input_nested, target_tensor)

    def test_dense_nested_targets(self):
        input_batch_tensor = torch.tensor([[0.1, 0.2, 0.0], [0.3, 0.4, 0.5]])
        target_nested = NestedTensor([torch.tensor([0, 1]), torch.tensor([1, 0, 1])])
        expected_result = torch.cat([input_batch_tensor[0, :2], input_batch_tensor[1, :3]])
        processed_input, processed_target = base_preprocess(input_batch_tensor, target_nested)
        assert torch.allclose(processed_input, expected_result)
        assert torch.allclose(processed_target, target_nested.values())

    def test_dense_nested_alignment(self):
        input_nested = NestedTensor([torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4, 0.5])])
        target_nested = NestedTensor([torch.tensor([0, 1]), torch.tensor([1, 0, 1])])

        nested_input, nested_target = base_preprocess(input_nested, target_nested)
        dense_input, dense_target = base_preprocess(input_nested.values(), target_nested.values())

        assert torch.allclose(nested_input, dense_input)
        assert torch.allclose(nested_target, dense_target)


class TestRegressionPreprocess:
    def test_flattens_nested_and_dense_equivalently(self):
        input_nested = NestedTensor([torch.tensor([[0.1, 0.2], [0.3, 0.4]]), torch.tensor([[0.5, 0.6]])])
        target_nested = NestedTensor([torch.tensor([[1.0, 1.1], [1.2, 1.3]]), torch.tensor([[1.4, 1.5]])])

        nested_input, nested_target = preprocess_regression(input_nested, target_nested, num_outputs=2)
        dense_input, dense_target = preprocess_regression(input_nested.values(), target_nested.values(), num_outputs=2)

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

    def test_single_output_nan(self):
        input_tensor = torch.tensor([0.1, 0.2, 0.3])
        target_tensor = torch.tensor([0.1, 0.2, float("nan")])
        processed_input, processed_target = preprocess_regression(input_tensor, target_tensor, ignore_nan=True)
        assert processed_input.shape == torch.Size([2])
        assert processed_target.shape == torch.Size([2])
        assert torch.allclose(processed_input, torch.tensor([0.1, 0.2]))
        assert torch.allclose(processed_target, torch.tensor([0.1, 0.2]))

    def test_multioutput_nan_rows(self):
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

    def test_retain_nan_rows(self):
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
    def test_probabilities(self):
        inputs = torch.tensor([0.1, 0.7, 0.4])
        targets = torch.tensor([0, 1, 0])
        actual = preprocess_binary(inputs, targets)
        repeated = preprocess_binary(*actual)
        torch.testing.assert_close(actual, (inputs, targets))
        torch.testing.assert_close(repeated, actual)

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
    def test_probabilities(self):
        inputs = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4], [0.6, 0.3, 0.1]])
        targets = torch.tensor([1, 2, 0])
        actual = preprocess_multiclass(inputs, targets, num_classes=3)
        repeated = preprocess_multiclass(*actual, num_classes=3)
        torch.testing.assert_close(actual, (inputs, targets))
        torch.testing.assert_close(repeated, actual)

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

    @pytest.mark.parametrize(
        "scores,labels,class_dim",
        [
            ([[[10.0, 0.0, 10.0], [0.0, 10.0, 0.0]]], [[0, 1, 0]], None),
            ([[[9.0, 8.0], [10.0, 7.0]]], [[1, 0]], 1),
        ],
        ids=["inferred", "explicit"],
    )
    def test_class_axis(self, scores, labels, class_dim):
        options = {} if class_dim is None else {"class_dim": class_dim}
        preds, target = preprocess_multiclass(torch.tensor(scores), torch.tensor(labels), num_classes=2, **options)
        torch.testing.assert_close(preds.argmax(-1), target)


class TestMultilabelPreprocess:
    def test_probabilities(self):
        inputs = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4], [0.6, 0.3, 0.1]])
        targets = torch.tensor([[0, 1, 0], [1, 0, 1], [1, 0, 0]])
        actual = preprocess_multilabel(inputs, targets, num_labels=3)
        repeated = preprocess_multilabel(*actual, num_labels=3)
        torch.testing.assert_close(actual, (inputs, targets))
        torch.testing.assert_close(repeated, actual)

    def test_normalizes_logits(self):
        input_tensor = torch.tensor([[-1.0, 2.0, 0.5], [0.0, -3.0, 1.0], [1.0, 2.0, -1.0]])
        target_tensor = torch.tensor([[0, 1, 0], [1, 0, 1], [1, 0, 0]])
        processed_input, processed_target = preprocess_multilabel(input_tensor, target_tensor, num_labels=3)
        assert processed_input.shape == torch.Size([3, 3])
        assert processed_target.shape == torch.Size([3, 3])
        assert (processed_input >= 0.0).all() and (processed_input <= 1.0).all()
        assert torch.allclose(processed_target, target_tensor)

    def test_fully_ignored_rows(self):
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
    @pytest.mark.parametrize(
        "preprocess,options,input_shape,target_shape,ignored,expected_shapes",
        [
            (preprocess_binary, {"ignore_index": -100}, (4,), (4,), -100, ((0,), (0,))),
            (preprocess_multiclass, {"num_classes": 3, "ignore_index": -100}, (4, 3), (4,), -100, ((0, 3), (0,))),
            (preprocess_multilabel, {"num_labels": 2, "ignore_index": -100}, (4, 2), (4, 2), -100, ((0, 2), (0, 2))),
            (
                preprocess_regression,
                {"num_outputs": 2, "ignore_nan": True},
                (2, 2),
                (2, 2),
                float("nan"),
                ((0, 2), (0, 2)),
            ),
        ],
        ids=["binary", "multiclass", "multilabel", "regression"],
    )
    def test_ignored_batches(self, preprocess, options, input_shape, target_shape, ignored, expected_shapes):
        actual = preprocess(torch.randn(input_shape), torch.full(target_shape, ignored), **options)
        assert tuple(tuple(value.shape) for value in actual) == expected_shapes
