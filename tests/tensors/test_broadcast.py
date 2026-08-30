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

r"""Elementwise broadcast parity against per-element dense PyTorch operations."""

from __future__ import annotations

import math

import pytest
import torch

from danling.tensors import NestedTensor, nested_execution_guard


def build_elements(shapes, offset=0.0):
    return [
        torch.arange(offset + 1.0, offset + 1.0 + float(math.prod(shape))).reshape(shape) + 1000.0 * position
        for position, shape in enumerate(shapes)
    ]


def operand_for(shape, offset=1.0):
    return torch.arange(offset, offset + float(math.prod(shape))).reshape(shape)


def assert_elements_close(output, expected):
    assert isinstance(output, NestedTensor)
    assert len(output) == len(expected)
    for actual, reference in zip(output, expected):
        assert actual.shape == reference.shape
        torch.testing.assert_close(actual.to(reference.dtype), reference)


def projected_case():
    reference = NestedTensor(
        [torch.randn(2, 3, 5), torch.randn(2, 4, 5)],
        ragged_dims=(1,),
    )
    return reference.packed_with_static_tail(torch.randn(7, 2, 9))


BROADCAST_CASES = [
    (
        "leading-ragged",
        [(2, 5), (4, 5)],
        {"ragged_dims": (0,)},
        (5,),
    ),
    (
        "middle-ragged",
        [(3, 2, 5), (3, 4, 5)],
        {"ragged_dims": (1,)},
        (1, 5),
    ),
    (
        "trailing-ragged",
        [(4, 2), (4, 5)],
        {"ragged_dims": (1,)},
        (4, 1),
    ),
    (
        "multi-ragged",
        [(2, 3, 5), (3, 4, 5)],
        {"ragged_dims": (0, 1)},
        (5,),
    ),
    (
        "batch-second",
        [(2, 5), (4, 5)],
        {"batch_first": False, "ragged_dims": (0,)},
        (5,),
    ),
]


def broadcast_cases_named(*names):
    return [case for case in BROADCAST_CASES if case[0] in names]


def broadcast_case_params(cases=None):
    cases = BROADCAST_CASES if cases is None else cases
    return pytest.mark.parametrize(
        ("case_id", "shapes", "metadata", "operand_shape"),
        cases,
        ids=[case[0] for case in cases],
    )


class TestDenseBroadcast:

    @broadcast_case_params()
    def test_shared_tail_operand(self, case_id, shapes, metadata, operand_shape):
        elements = build_elements(shapes)
        operand = operand_for(operand_shape)

        output = NestedTensor(elements, **metadata) + operand

        assert_elements_close(output, [element + operand for element in elements])

    @pytest.mark.parametrize(
        "operation",
        [torch.div, torch.maximum, torch.gt],
        ids=["div", "maximum", "greater"],
    )
    def test_other_binary_operations(self, operation):
        elements = build_elements([(3, 2, 5), (3, 4, 5)])
        operand = operand_for((1, 5))

        output = operation(NestedTensor(elements, ragged_dims=(1,)), operand)

        assert_elements_close(output, [operation(element, operand) for element in elements])

    @pytest.mark.parametrize(
        ("elements", "metadata", "operand", "expected"),
        [
            (
                build_elements([(2, 5), (4, 5)]),
                {"ragged_dims": (0,)},
                operand_for((2, 1, 5)),
                lambda elements, operand: [element + operand[index] for index, element in enumerate(elements)],
            ),
            (
                build_elements([(3, 2, 5), (3, 4, 5)]),
                {"ragged_dims": (1,)},
                operand_for((2, 3, 1, 5)),
                lambda elements, operand: [element + operand[index] for index, element in enumerate(elements)],
            ),
            (
                [torch.arange(float(2 * length * 3)).reshape(2, length, 3) for length in (3, 5, 2)],
                {"batch_first": False, "ragged_dims": (1,)},
                torch.arange(18.0).reshape(3, 2, 3),
                lambda elements, operand: [
                    element + operand[index].unsqueeze(1) for index, element in enumerate(elements)
                ],
            ),
        ],
        ids=["canonical", "middle-ragged", "batch-second"],
    )
    def test_per_sample_dense_operand(self, elements, metadata, operand, expected):
        output = NestedTensor(elements, **metadata) + operand

        assert_elements_close(output, expected(elements, operand))

    @broadcast_case_params(broadcast_cases_named("leading-ragged", "batch-second"))
    def test_reversed_operand_order(self, case_id, shapes, metadata, operand_shape):
        elements = build_elements(shapes)
        operand = operand_for(operand_shape)

        output = operand - NestedTensor(elements, **metadata)

        assert_elements_close(output, [operand - element for element in elements])

    def test_scalar_and_packed_shaped_operands(self):
        elements = build_elements([(2, 5), (4, 5)])
        nested = NestedTensor(elements, ragged_dims=(0,))
        assert_elements_close(nested + 2.5, [element + 2.5 for element in elements])

        packed_operand = torch.arange(float(nested.concat.numel())).reshape(nested.concat.shape)
        torch.testing.assert_close((nested + packed_operand).concat, nested.concat + packed_operand)

    def test_projected_layout(self):
        projected = projected_case()
        elements = list(projected)
        operand = torch.randn(9)

        output = projected + operand

        assert_elements_close(output, [element + operand for element in elements])


class TestAmbiguousOperands:

    def test_ambiguous_shape_raises(self):
        nested = NestedTensor(
            [torch.randn(2, length, 5) for length in (3, 4)],
            ragged_dims=(1,),
        )

        with pytest.raises(NotImplementedError, match="ambiguous"):
            nested + torch.randn(2, 1, 5)

    def test_explicit_spellings_disambiguate_shared_and_per_sample_operands(self):
        elements = [torch.randn(2, length, 5) for length in (3, 4)]
        nested = NestedTensor(elements, ragged_dims=(1,))
        ambiguous = torch.randn(2, 1, 5)
        per_sample = ambiguous.reshape(2, 1, 1, 5)
        shared = ambiguous.reshape(1, 2, 1, 5)

        assert_elements_close(
            nested + per_sample,
            [element + per_sample[index] for index, element in enumerate(elements)],
        )
        assert_elements_close(nested + shared, [element + shared[0] for element in elements])

    def test_batch_sized_static_tail_is_not_mistaken_for_a_batch_axis(self):
        elements = [torch.zeros(length, 2, 4) for length in (2, 3)]
        operand = torch.arange(8.0).reshape(2, 4)

        output = NestedTensor(elements) + operand

        assert_elements_close(output, [element + operand for element in elements])


class TestBroadcastTensors:

    def test_per_sample_operand(self):
        elements = [torch.ones(3, 5), torch.ones(4, 5)]
        operand = torch.arange(10.0).reshape(2, 5)

        _, spread = torch.broadcast_tensors(NestedTensor(elements), operand)

        assert_elements_close(
            spread,
            [operand[index].expand(rows, 5) for index, rows in enumerate((3, 4))],
        )

    def test_shared_tail_operand(self):
        elements = [torch.zeros(rows, 2, 4) for rows in (2, 3)]
        operand = torch.arange(8.0).reshape(2, 4)

        _, spread = torch.broadcast_tensors(NestedTensor(elements), operand)

        assert_elements_close(spread, [operand.expand_as(element) for element in elements])


class TestLayoutPreservation:

    @pytest.mark.parametrize("lengths", [(3, 3), (3,), (0, 4)], ids=["equal", "single", "empty"])
    def test_declared_ragged_dimension_is_preserved(self, lengths):
        elements = [torch.randn(2, length, 5) for length in lengths]
        nested = NestedTensor(elements, ragged_dims=(1,))
        operand = torch.randn(5)

        output = nested + operand

        assert output.ragged_dims == (1,)
        assert_elements_close(output, [element + operand for element in elements])
        assert output.shape == nested.shape

    def test_source_derived_empty_batch_preserves_topology(self):
        nested = NestedTensor(
            [torch.randn(2, length, 5) for length in (3, 4)],
            ragged_dims=(1,),
        )[:0]

        output = nested + torch.randn(5)

        assert output.shape == nested.shape
        assert output.ragged_dims == (1,)


def expanding_ragged_operands(shapes, *, requires_grad=False):
    target_elements = [torch.randn(rows, columns, 1, requires_grad=requires_grad) for rows, columns in shapes]
    source_elements = [torch.randn(1, columns, 4, requires_grad=requires_grad) for _, columns in shapes]
    return (
        NestedTensor(target_elements, ragged_dims=(0, 1)),
        NestedTensor(source_elements, ragged_dims=(1,)),
        target_elements,
        source_elements,
    )


class TestNestedBroadcast:

    @pytest.mark.parametrize(
        "shapes",
        [((2, 3),), ((2, 3), (3, 2))],
        ids=["single-sample", "batch"],
    )
    @pytest.mark.parametrize("source_first", [False, True])
    def test_source_expands_target_static_tail_with_gradient(self, shapes, source_first):
        target, source, target_elements, source_elements = expanding_ragged_operands(
            shapes,
            requires_grad=True,
        )
        expected_elements = [
            source_element - target_element if source_first else target_element - source_element
            for target_element, source_element in zip(target_elements, source_elements)
        ]

        output = source - target if source_first else target - source

        actual_gradients = torch.autograd.grad(
            output.concat.square().sum(),
            (*target_elements, *source_elements),
        )
        expected_gradients = torch.autograd.grad(
            sum(element.square().sum() for element in expected_elements),
            (*target_elements, *source_elements),
        )
        assert_elements_close(output, expected_elements)
        assert output.ragged_dims == (0, 1)
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, expected)

    def test_source_derived_empty_batch(self):
        target, source, _, _ = expanding_ragged_operands(((2, 3), (2, 3)))

        output = target[:0] - source[:0]

        assert output.shape == (0, 2, 3, 4)
        assert output.ragged_dims == (0, 1)

    def test_zero_width_static_tail(self):
        target_element = torch.empty(2, 3, 1)
        source_element = torch.empty(1, 3, 0)
        target = NestedTensor([target_element], ragged_dims=(0, 1))
        source = NestedTensor([source_element], ragged_dims=(1,))

        output = target + source

        assert_elements_close(output, [target_element + source_element])

    def test_empty_lower_rank_mismatch_raises(self):
        wide = NestedTensor(
            [torch.empty(3, 2, 5), torch.empty(3, 4, 5)],
            ragged_dims=(1,),
        )[:0]
        narrow = NestedTensor(
            [torch.empty(3, 5), torch.empty(5, 5)],
            ragged_dims=(0,),
        )[:0]

        with pytest.raises(ValueError):
            wide - narrow

    @pytest.mark.parametrize("narrow_first", [False, True])
    def test_lower_rank_nested_operand(self, narrow_first):
        wide_elements = [torch.randn(3, length, 5) for length in (2, 4)]
        narrow_elements = [torch.randn(length, 5) for length in (2, 4)]
        wide = NestedTensor(wide_elements, ragged_dims=(1,))
        narrow = NestedTensor(narrow_elements, ragged_dims=(0,))

        output = narrow + wide if narrow_first else wide + narrow

        assert_elements_close(
            output,
            (
                [a + b for a, b in zip(narrow_elements, wide_elements)]
                if narrow_first
                else [a + b for a, b in zip(wide_elements, narrow_elements)]
            ),
        )
        assert output.ragged_dims == (1,)

    def test_mismatched_nested_lengths_raise(self):
        wide = NestedTensor(
            [torch.randn(3, length, 5) for length in (2, 4)],
            ragged_dims=(1,),
        )
        narrow = NestedTensor(
            [torch.randn(length, 5) for length in (2, 3)],
            ragged_dims=(0,),
        )

        with pytest.raises(RuntimeError):
            wide + narrow


class TestWhereAndTernary:

    @staticmethod
    def sampled_operands(lengths, *, condition_spelling, requires_grad=False):
        samples = 4
        channels = 3
        coordinate = NestedTensor(
            [torch.empty(length, channels) for length in lengths],
            ragged_dims=(0,),
        )
        unit_template = coordinate.unsqueeze(-3)
        sampled_template = unit_template.expand(-1, samples, -1, -1)
        unit_values = torch.randn_like(unit_template.concat, requires_grad=requires_grad)
        sampled_values = torch.randn_like(sampled_template.concat, requires_grad=requires_grad)
        unit = unit_template.packed_like(unit_values)
        sampled = sampled_template.packed_like(sampled_values)

        mask_template = NestedTensor(
            [torch.empty(length, dtype=torch.bool) for length in lengths],
            ragged_dims=(0,),
        )
        mask_values = torch.arange(sum(lengths)).remainder(2).eq(0)
        condition = mask_template.packed_like(mask_values)
        if condition_spelling == "lower-rank":
            condition = condition.unsqueeze(-1)
        else:
            condition = condition.unsqueeze(-2).unsqueeze(-1)
        return condition, unit, sampled, unit_values, sampled_values

    @staticmethod
    def packed_condition(condition):
        values = condition.concat
        return values.unsqueeze(1) if values.dim() == 2 else values

    @pytest.mark.parametrize("condition_spelling", ["lower-rank", "unit-sample"])
    def test_nested_where_matches_values_and_gradients(self, condition_spelling):
        condition, unit, sampled, unit_values, sampled_values = self.sampled_operands(
            (3, 5),
            condition_spelling=condition_spelling,
            requires_grad=True,
        )

        output = torch.where(condition, unit, sampled)

        expected = torch.where(self.packed_condition(condition), unit_values, sampled_values)
        cotangent = torch.randn_like(expected)
        actual_gradients = torch.autograd.grad(
            output.concat,
            (unit_values, sampled_values),
            cotangent,
        )
        expected_gradients = torch.autograd.grad(
            expected,
            (unit_values, sampled_values),
            cotangent,
        )
        torch.testing.assert_close(output.concat, expected)
        for actual, reference in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, reference)

    def test_nested_where_supports_fake_tensor(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        condition, unit, sampled, _, _ = self.sampled_operands(
            (3, 5),
            condition_spelling="lower-rank",
        )
        mode = fake_tensor_mod.FakeTensorMode()

        with mode:
            output = torch.where(
                mode.from_tensor(condition),
                mode.from_tensor(unit),
                mode.from_tensor(sampled),
            )

        assert fake_tensor_mod.is_fake(output.concat)
        assert output.shape == (2, 4, 5, 3)

    def test_nested_where_dynamic_fullgraph_with_gradient(self):
        def consume(condition, unit, sampled):
            return torch.where(condition, unit, sampled).concat.square().sum()

        compiled = torch.compile(
            consume,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        for lengths in ((2, 3), (3, 5)):
            condition, unit, sampled, unit_values, sampled_values = self.sampled_operands(
                lengths,
                condition_spelling="lower-rank",
                requires_grad=True,
            )
            expected_values = torch.where(
                self.packed_condition(condition),
                unit_values,
                sampled_values,
            )

            output = compiled(condition, unit, sampled)

            expected = expected_values.square().sum()
            actual_gradients = torch.autograd.grad(output, (unit_values, sampled_values))
            expected_gradients = torch.autograd.grad(expected, (unit_values, sampled_values))
            torch.testing.assert_close(output, expected)
            for actual, reference in zip(actual_gradients, expected_gradients):
                torch.testing.assert_close(actual, reference)

    def test_where_with_dense_tail_and_condition(self):
        elements = [torch.randn(3, length, 5) for length in (2, 4)]
        nested = NestedTensor(elements, ragged_dims=(1,))
        tail = torch.rand(5)
        dense_condition = torch.rand(2, 1, 1, 5) > 0.5

        assert_elements_close(
            torch.where(nested > 0, nested, tail),
            [torch.where(element > 0, element, tail) for element in elements],
        )
        assert_elements_close(
            torch.where(dense_condition, nested, 0.0),
            [torch.where(dense_condition[index], element, torch.zeros(())) for index, element in enumerate(elements)],
        )

    @pytest.mark.parametrize("operation", ["addcmul", "addcdiv", "lerp"])
    def test_addc_family_per_sample_operand(self, operation):
        elements = [torch.rand(3, length, 5) + 1.0 for length in (2, 4)]
        nested = NestedTensor(elements, ragged_dims=(1,))
        operand = torch.rand(2, 1, 1, 1)

        if operation == "lerp":
            output = torch.lerp(nested, nested * 2, operand)
            expected = [torch.lerp(element, element * 2, operand[index]) for index, element in enumerate(elements)]
        elif operation == "addcmul":
            output = torch.addcmul(nested, nested, operand, value=2.0)
            expected = [
                torch.addcmul(element, element, operand[index], value=2.0) for index, element in enumerate(elements)
            ]
        else:
            output = torch.addcdiv(nested, nested, operand, value=2.0)
            expected = [
                torch.addcdiv(element, element, operand[index], value=2.0) for index, element in enumerate(elements)
            ]

        assert_elements_close(output, expected)


class TestInPlace:

    @pytest.mark.parametrize("operation", ["add_", "sub_", "mul_"])
    def test_matches_out_of_place_operation(self, operation):
        elements = [torch.randn(3, length, 5) for length in (2, 4)]
        nested = NestedTensor(elements, ragged_dims=(1,))
        operand = torch.rand(5)
        expected = [getattr(element.clone(), operation)(operand) for element in elements]
        target = nested.clone()

        getattr(target, operation)(operand)

        assert_elements_close(target, expected)

    def test_per_sample_operand(self):
        elements = [torch.randn(3, length, 5) for length in (2, 4)]
        operand = torch.rand(2, 1, 1, 5)
        target = NestedTensor(elements, ragged_dims=(1,))

        target.add_(operand)

        assert_elements_close(
            target,
            [element + operand[index] for index, element in enumerate(elements)],
        )


class TestNestedViewAlignment:

    @staticmethod
    def _pair(lengths, *, requires_grad=False):
        template = NestedTensor(
            [torch.empty(length, length, 3) for length in lengths],
            ragged_dims=(0, 1),
        )
        values = torch.randn_like(template.concat, requires_grad=requires_grad)
        return template, values

    @staticmethod
    def _symmetrize(values, lengths):
        elements = []
        for length, packed in zip(lengths, values.split([length * length for length in lengths])):
            pair = packed.reshape(length, length, 3)
            elements.append((pair + pair.transpose(0, 1)).flatten(0, 1))
        return torch.cat(elements)

    def test_square_pair_adds_its_transpose_without_leaving_packed_execution(self):
        lengths = (2, 3)
        template, values = self._pair(lengths)
        pair = template.packed_like(values)

        with nested_execution_guard(
            forbid_iteration=True,
            forbid_storage_map=True,
            forbid_eager_fallback=True,
            forbid_padded_materialization=True,
            forbid_dense_repack=True,
        ):
            output = pair + pair.transpose(-2, -3)

        torch.testing.assert_close(output.concat, self._symmetrize(values, lengths))

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_square_pair_transpose_dynamic_fullgraph_vjp(self):
        def symmetrize(structure, packed):
            pair = structure.packed_like(packed)
            return (pair + pair.transpose(-2, -3)).concat

        compiled = torch.compile(
            symmetrize,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        for lengths in ((2, 3), (1, 4, 2)):
            template, values = self._pair(lengths, requires_grad=True)
            reference_values = values.detach().clone().requires_grad_()

            output = compiled(template, values)
            expected = self._symmetrize(reference_values, lengths)
            cotangent = torch.randn_like(expected)
            (actual_gradient,) = torch.autograd.grad(output, values, cotangent)
            (expected_gradient,) = torch.autograd.grad(expected, reference_values, cotangent)

            torch.testing.assert_close(output, expected)
            torch.testing.assert_close(actual_gradient, expected_gradient)


class TestCompile:

    def test_nested_static_tail_expansion_fullgraph(self):
        def consume(target, source):
            output = target - source
            return output.concat, output.element_sizes()

        compiled = torch.compile(
            consume,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        for shapes in (((2, 3), (3, 2)), ((3, 2), (1, 5))):
            target, source, target_elements, source_elements = expanding_ragged_operands(shapes)
            values, sizes = compiled(target, source)
            expected = NestedTensor(
                [
                    target_element - source_element
                    for target_element, source_element in zip(target_elements, source_elements)
                ],
                ragged_dims=(0, 1),
            )
            torch.testing.assert_close(values, expected.concat)
            torch.testing.assert_close(sizes, expected.element_sizes())

        target, source, _, _ = expanding_ragged_operands(((2, 3), (2, 3)))
        values, sizes = compiled(target[:0], source[:0])
        assert values.shape == (0, 4)
        assert sizes.shape == (0, 3)

    def test_dense_tail_broadcast_fullgraph(self):
        elements = [torch.randn(3, length, 5) for length in (2, 4)]
        nested = NestedTensor(elements, ragged_dims=(1,))
        operand = torch.rand(5)
        compiled = torch.compile(
            lambda input_, other: input_ + other,
            backend="aot_eager",
            fullgraph=True,
        )

        output = compiled(nested, operand)

        assert_elements_close(output, [element + operand for element in elements])

    def test_projected_layout_fullgraph(self):
        projected = projected_case()
        elements = list(projected)
        operand = torch.rand(9)
        compiled = torch.compile(
            lambda input_, other: input_ + other,
            backend="aot_eager",
            fullgraph=True,
        )

        output = compiled(projected, operand)

        assert_elements_close(output, [element + operand for element in elements])


class TestAutograd:

    def test_dense_operand_gradient_matches_per_element_reference(self):
        leaves = [torch.randn(3, length, 5, requires_grad=True) for length in (2, 4)]
        operand = torch.randn(5, requires_grad=True)

        (NestedTensor(leaves, ragged_dims=(1,)) * operand).concat.sum().backward()

        actual_leaf_gradients = [leaf.grad.clone() for leaf in leaves]
        actual_operand_gradient = operand.grad.clone()

        reference_leaves = [leaf.detach().clone().requires_grad_() for leaf in leaves]
        reference_operand = operand.detach().clone().requires_grad_()
        sum((leaf * reference_operand).sum() for leaf in reference_leaves).backward()

        for actual, reference in zip(actual_leaf_gradients, reference_leaves):
            torch.testing.assert_close(actual, reference.grad)
        torch.testing.assert_close(actual_operand_gradient, reference_operand.grad)
