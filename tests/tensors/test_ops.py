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

r"""Public behaviors shared by NestedTensor operations."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from danling.tensors import NestedTensor, nested_execution_guard

NT = NestedTensor


class TestDropoutValidation:

    def test_dropout_probability_error_types(self):
        nt = NT(
            [
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([4.0, 5.0]),
            ]
        )

        with pytest.raises(RuntimeError, match="between 0 and 1"):
            torch.dropout(nt, p=-0.1, train=False)

        with pytest.raises(ValueError, match="between 0 and 1"):
            F.dropout(nt, p=-0.1, training=False)


class TestDenseBinaryOperands:
    r"""Dense operands that pair with a NestedTensor without materializing padding."""

    def test_dense_operand_matching_concat_shape_is_elementwise(self):
        nt = NT([torch.randn(2, 3), torch.randn(4, 3)])
        dense = torch.randn_like(nt.concat)

        output = nt * dense

        torch.testing.assert_close(output.concat, nt.concat * dense)
        assert [tuple(element.shape) for element in output] == [(2, 3), (4, 3)]

    def test_per_element_dense_broadcast(self):
        # (B, 1, ragged_N, C) against a (B, S, 1, C) term: each element pairs with its own slice.
        nt = NT([torch.randn(1, 2, 3), torch.randn(1, 4, 3)])
        dense = torch.randn(2, 5, 1, 3)

        output = nt + dense
        assert isinstance(output, NestedTensor)
        assert len(output) == 2
        for index, element in enumerate(output):
            torch.testing.assert_close(element, nt[index] + dense[index])

    def test_gather_keeps_grad_after_no_grad_iteration(self):
        values = torch.randn(5, requires_grad=True)
        nt = NT([values[:2], values[2:]])
        with torch.no_grad():
            _ = tuple(nt)

        index = NT([torch.tensor([1, 0]), torch.tensor([2, 1, 0])])
        output = torch.gather(nt, 1, index)
        output.concat.sum().backward()

        torch.testing.assert_close(output.concat, torch.cat((values[:2].flip(0), values[2:].flip(0))))
        torch.testing.assert_close(values.grad, torch.ones_like(values))


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


class TestComplementarySingletonBroadcast:

    @staticmethod
    def _operands(lengths=(2, 3), channels=4):
        row_template = NestedTensor(
            [torch.empty(length, 1, channels) for length in lengths],
            ragged_dims=(0,),
        )
        column_template = NestedTensor(
            [torch.empty(1, length, channels) for length in lengths],
            ragged_dims=(1,),
        )
        row_values = torch.randn_like(row_template.concat, requires_grad=True)
        column_values = torch.randn_like(column_template.concat, requires_grad=True)
        return row_template, column_template, row_values, column_values

    @staticmethod
    def _dense_result(operation, row_values, column_values, lengths, reverse):
        elements = []
        for row, column in zip(row_values.split(lengths), column_values.split(lengths)):
            column = column.transpose(0, 1)
            elements.append(operation(column, row) if reverse else operation(row, column))
        return torch.cat([element.flatten(0, 1) for element in elements])

    @pytest.mark.parametrize(
        ("operation", "reverse"),
        [(torch.add, False), (torch.sub, True)],
        ids=("add", "column-minus-row"),
    )
    def test_values_and_vjp_match_per_element(self, operation, reverse):
        lengths = (2, 3)
        row_template, column_template, row_values, column_values = self._operands(lengths)
        row = row_template.packed_like(row_values)
        column = column_template.packed_like(column_values)

        output = operation(column, row) if reverse else operation(row, column)

        reference_row = row_values.detach().clone().requires_grad_()
        reference_column = column_values.detach().clone().requires_grad_()
        expected = self._dense_result(operation, reference_row, reference_column, lengths, reverse)
        cotangent = torch.randn_like(expected)
        actual_gradients = torch.autograd.grad(output.concat, (row_values, column_values), cotangent)
        expected_gradients = torch.autograd.grad(expected, (reference_row, reference_column), cotangent)
        torch.testing.assert_close(output.concat, expected)
        assert output.ragged_dims == (0, 1)
        assert output.element_sizes().tolist() == [[length, length, 4] for length in lengths]
        for actual, reference in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, reference)

    def test_supports_fake_tensor(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        row_template, column_template, _, _ = self._operands()
        mode = fake_tensor_mod.FakeTensorMode()
        row = row_template.packed_like(mode.from_tensor(torch.empty_like(row_template.concat)))
        column = column_template.packed_like(mode.from_tensor(torch.empty_like(column_template.concat)))

        output = row + column

        assert fake_tensor_mod.is_fake(output.concat)
        assert output.concat.shape == (13, 4)
        assert output.shape == (2, 3, 3, 4)
        assert output.ragged_dims == (0, 1)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_aot_eager_fullgraph_vjp_matches_per_element(self):
        lengths = (2, 3)
        row_template, column_template, row_values, column_values = self._operands(lengths)
        reference_row = row_values.detach().clone().requires_grad_()
        reference_column = column_values.detach().clone().requires_grad_()

        def subtract(row_structure, column_structure, packed_row, packed_column):
            row = row_structure.packed_like(packed_row)
            column = column_structure.packed_like(packed_column)
            return (column - row).concat

        compiled = torch.compile(subtract, backend="aot_eager", fullgraph=True)
        output = compiled(row_template, column_template, row_values, column_values)
        expected = self._dense_result(torch.sub, reference_row, reference_column, lengths, True)
        cotangent = torch.randn_like(expected)
        actual_gradients = torch.autograd.grad(output, (row_values, column_values), cotangent)
        expected_gradients = torch.autograd.grad(expected, (reference_row, reference_column), cotangent)
        torch.testing.assert_close(output, expected)
        for actual, reference in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, reference)

    @staticmethod
    def _static_prefix_input(lengths, *, requires_grad=True):
        return [torch.randn(length, requires_grad=requires_grad) for length in lengths]

    @staticmethod
    def _static_prefix_reference(parts, prefix=4):
        elements = []
        for part in parts:
            expanded = part.unsqueeze(0).expand(prefix, -1)
            elements.append(expanded.unsqueeze(-1) + expanded.unsqueeze(-2))
        return NestedTensor(elements, ragged_dims=(1, 2)).concat

    def test_static_prefix_matches_per_element_under_strict_execution(self):
        lengths = (2, 3)
        parts = self._static_prefix_input(lengths)
        reference_parts = [part.detach().clone().requires_grad_() for part in parts]
        source = NestedTensor(parts, ragged_dims=(0,))

        with nested_execution_guard(
            forbid_iteration=True,
            forbid_storage_map=True,
            forbid_eager_fallback=True,
            forbid_padded_materialization=True,
            forbid_dense_repack=True,
        ):
            expanded = source.unsqueeze(-2).expand(-1, 4, -1)
            output = expanded.unsqueeze(-1) + expanded.unsqueeze(-2)

        expected = self._static_prefix_reference(reference_parts)
        cotangent = torch.randn_like(expected)
        actual_gradients = torch.autograd.grad(output.concat, parts, cotangent)
        expected_gradients = torch.autograd.grad(expected, reference_parts, cotangent)

        assert output.ragged_dims == (1, 2)
        assert output.element_sizes().tolist() == [[4, length, length] for length in lengths]
        torch.testing.assert_close(output.concat, expected)
        for actual, reference in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, reference)

    def test_static_prefix_supports_fake_tensor(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        symbolic_shapes = pytest.importorskip("torch.fx.experimental.symbolic_shapes")
        lengths = (2, 3)
        source = NestedTensor([torch.empty(length) for length in lengths], ragged_dims=(0,))
        shape_env = symbolic_shapes.ShapeEnv(allow_dynamic_output_shape_ops=True)
        with fake_tensor_mod.FakeTensorMode(shape_env=shape_env) as mode:
            fake_source = mode.from_tensor(source)
            expanded = fake_source.unsqueeze(-2).expand(-1, 4, -1)
            output = expanded.unsqueeze(-1) + expanded.unsqueeze(-2)

        assert fake_tensor_mod.is_fake(output.concat)
        assert output.shape[:2] == (2, 4)
        assert output.shape[-2] == output.shape[-1]
        assert output.ragged_dims == (1, 2)
        assert output.element_sizes().shape == (2, 3)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_static_prefix_dynamic_fullgraph_vjp(self):
        def square(source):
            expanded = source.unsqueeze(-2).expand(-1, 4, -1)
            return torch.relu(expanded.unsqueeze(-1) + expanded.unsqueeze(-2))

        compiled = torch.compile(square, backend="aot_eager", fullgraph=True, dynamic=True)
        for lengths in ((2, 3), (1, 4, 2)):
            parts = self._static_prefix_input(lengths)
            reference_parts = [part.detach().clone().requires_grad_() for part in parts]
            source = NestedTensor(parts, ragged_dims=(0,))

            output = compiled(source)
            expected = self._static_prefix_reference(reference_parts).relu()
            cotangent = torch.randn_like(expected)
            actual_gradients = torch.autograd.grad(output.concat, parts, cotangent)
            expected_gradients = torch.autograd.grad(expected, reference_parts, cotangent)

            assert output.ragged_dims == (1, 2)
            assert output.element_sizes().tolist() == [[4, length, length] for length in lengths]
            torch.testing.assert_close(output.concat, expected)
            for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
                torch.testing.assert_close(actual, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_static_prefix_channel_tail_dynamic_fullgraph_vjp(self):
        def square(source):
            return torch.relu(source.unsqueeze(-2) + source.unsqueeze(-3))

        compiled = torch.compile(square, backend="aot_eager", fullgraph=True, dynamic=True)
        for lengths in ((2, 3), (1, 4, 2)):
            parts = [torch.randn(2, length, 3, requires_grad=True) for length in lengths]
            reference_parts = [part.detach().clone().requires_grad_() for part in parts]
            source = NestedTensor(parts, ragged_dims=(1,))

            output = compiled(source)
            expected_parts = [torch.relu(part.unsqueeze(-2) + part.unsqueeze(-3)) for part in reference_parts]
            expected = NestedTensor(expected_parts, ragged_dims=(1, 2)).concat
            cotangent = torch.randn_like(expected)
            actual_gradients = torch.autograd.grad(output.concat, parts, cotangent)
            expected_gradients = torch.autograd.grad(expected, reference_parts, cotangent)

            assert output.ragged_dims == (1, 2)
            assert output.element_sizes().tolist() == [[2, length, length, 3] for length in lengths]
            torch.testing.assert_close(output.concat, expected)
            for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
                torch.testing.assert_close(actual, reference)

    @pytest.mark.parametrize("operation", (torch.add, torch.sub, torch.mul))
    @pytest.mark.parametrize("prefix", (False, True))
    @pytest.mark.parametrize("batch_first", (False, True))
    def test_rectangular_complementary_values_and_vjps(self, device, operation, prefix, batch_first):
        rows, columns = (1, 3, 2), (3, 1, 4)
        row_dim = int(prefix)
        left = tuple(
            torch.randn(*((2,) if prefix else ()), n, 1, 4, device=device, dtype=torch.float64, requires_grad=True)
            for n in rows
        )
        right = tuple(
            torch.randn(*((1,) if prefix else ()), 1, n, 4, device=device, dtype=torch.float64, requires_grad=True)
            for n in columns
        )
        row = NestedTensor(left, ragged_dims=(row_dim,), batch_first=batch_first)
        column = NestedTensor(right, ragged_dims=(row_dim + 1,), batch_first=batch_first)
        for reverse in (False, True):
            with nested_execution_guard(
                forbid_eager_fallback=True,
                forbid_storage_map=True,
                forbid_iteration=True,
                forbid_padded_materialization=True,
                forbid_dense_repack=True,
            ):
                output = operation(column, row) if reverse else operation(row, column)
            expected = tuple(operation(b, a) if reverse else operation(a, b) for a, b in zip(left, right))
            actual = output.unbind(0 if batch_first else 1)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert output.ragged_dims == (row_dim, row_dim + 1)
            assert output.element_sizes().tolist() == [list(value.shape) for value in expected]
            assert output.packed_offsets().tolist() == [0, 3, 6, 14]
            assert output.ragged_level_offsets(0).tolist() == [0, 1, 4, 6]
            assert output.ragged_level_offsets(1).tolist() == [0, 3, 4, 5, 6, 10, 14]
            weights = tuple(
                torch.arange(value.numel(), device=device, dtype=value.dtype).reshape_as(value) for value in expected
            )
            actual_loss = sum((value * weight).sum() for value, weight in zip(actual, weights))
            expected_loss = sum((value * weight).sum() for value, weight in zip(expected, weights))
            observed = torch.autograd.grad(actual_loss, (*left, *right), retain_graph=True)
            wanted = torch.autograd.grad(expected_loss, (*left, *right), retain_graph=True)
            torch.testing.assert_close(observed, wanted, atol=1e-12, rtol=1e-12)

    def test_rectangular_complementary_zero_extents(self, device):
        left = tuple(torch.randn(n, 1, 2, device=device, requires_grad=True) for n in (0, 2, 1))
        right = tuple(torch.randn(1, n, 2, device=device, requires_grad=True) for n in (3, 0, 2))
        with nested_execution_guard(forbid_eager_fallback=True, forbid_storage_map=True):
            output = NestedTensor(left, ragged_dims=(0,)) + NestedTensor(right, ragged_dims=(1,))
        expected = tuple(a + b for a, b in zip(left, right))
        torch.testing.assert_close(output.unbind(), expected)
        assert output.ragged_dims == (0, 1)
        torch.testing.assert_close(
            torch.autograd.grad(output.concat.sum(), (*left, *right)),
            torch.autograd.grad(sum(value.sum() for value in expected), (*left, *right)),
        )


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


def _guard():
    return nested_execution_guard(
        forbid_iteration=True,
        forbid_storage_map=True,
        forbid_eager_fallback=True,
        forbid_padded_materialization=True,
        forbid_dense_repack=True,
    )


def _parts(shapes, device, offset=0.0):
    return [
        (
            torch.arange(math.prod(shape), dtype=torch.float64, device=device).reshape(shape) / 17 + offset + i + 1
        ).requires_grad_()
        for i, shape in enumerate(shapes)
    ]


def _compare(target_parts, source_parts, target_order, source_order, operation, source_first, batch_first=True):
    target = NestedTensor(target_parts, ragged_dims=target_order, batch_first=batch_first)
    source = NestedTensor(source_parts, ragged_dims=source_order, batch_first=batch_first)
    expected_parts = [
        operation(mask, value) if source_first else operation(value, mask)
        for value, mask in zip(target_parts, source_parts)
    ]
    expected = NestedTensor(expected_parts, ragged_dims=target_order, batch_first=batch_first)
    weights = torch.linspace(-1, 2, expected.concat.numel(), device=expected.device, dtype=torch.float64)
    weights = weights.reshape(expected.concat.shape)
    leaves = (*target_parts, *source_parts)
    with _guard():
        output = operation(source, target) if source_first else operation(target, source)
        actual_gradients = torch.autograd.grad(output.concat, leaves, weights)
    expected_gradients = torch.autograd.grad(expected.concat, leaves, weights)
    assert output.ragged_dims == target_order
    assert output.batch_first is batch_first
    torch.testing.assert_close(output.element_sizes(), expected.element_sizes())
    with torch.no_grad():
        torch.testing.assert_close(output.concat, expected.concat, atol=1e-12, rtol=1e-12)
    for actual, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("orders", [((0, 1), (1, 0)), ((1, 0), (0, 1))])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize(
    "operation,source_first",
    [(torch.mul, False), (torch.sub, False), (torch.sub, True), (torch.div, False), (torch.div, True)],
    ids=["multiply", "target-minus-mask", "mask-minus-target", "target-div-mask", "mask-div-target"],
)
def test_retained_ragged_singletons_preserve_values_operand_order_and_vjp(
    device, orders, axis, batch_first, operation, source_first
):
    shapes = ((2, 3, 4), (4, 2, 4))
    mask_shapes = [tuple(1 if dim == axis else size for dim, size in enumerate(shape)) for shape in shapes]
    _compare(
        _parts(shapes, device),
        _parts(mask_shapes, device, 0.5),
        *orders,
        operation,
        source_first,
        batch_first,
    )


@pytest.mark.parametrize("orders", [((1, 2), (2, 1)), ((2, 1), (1, 2))])
@pytest.mark.parametrize("axis", [1, 2])
def test_nonleading_ragged_singletons_keep_static_prefix_and_expand_features(device, orders, axis):
    shapes = ((2, 2, 3, 1), (2, 4, 2, 1))
    masks = [tuple(1 if dim == axis else 4 if dim == 3 else size for dim, size in enumerate(shape)) for shape in shapes]
    _compare(_parts(shapes, device), _parts(masks, device, 0.5), *orders, torch.sub, True)


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_additional_retained_axis_can_broadcast_in_only_one_sample(device, order):
    _compare(
        _parts(((2, 3, 4), (4, 5, 4)), device),
        _parts(((1, 3, 4), (1, 1, 4)), device, 0.5),
        order,
        tuple(reversed(order)),
        torch.mul,
        False,
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_zero_volume_elements_and_empty_batch_retain_the_full_operand_shape(device, axis):
    shapes = ((0, 3, 4), (2, 3, 4)) if axis == 0 else ((3, 0, 4), (3, 2, 4))
    masks = [tuple(1 if dim == axis else size for dim, size in enumerate(shape)) for shape in shapes]
    target = NestedTensor(_parts(shapes, device), ragged_dims=(0, 1))
    source = NestedTensor(_parts(masks, device), ragged_dims=(0, 1))
    expected = NestedTensor([a * b for a, b in zip(target, source)], ragged_dims=(0, 1))
    empty_target, empty_source = target[:0], source[:0]
    with _guard():
        output = target * source
        empty_output = empty_source - empty_target
    torch.testing.assert_close(output.concat, expected.concat)
    torch.testing.assert_close(output.element_sizes(), expected.element_sizes())
    assert empty_output.shape == empty_target.shape
    assert empty_output.ragged_dims == (0, 1)
    assert empty_output.concat.shape == empty_target.concat.shape


def test_incompatible_retained_extent_is_rejected(device):
    target = NestedTensor(_parts(((2, 3, 4), (4, 5, 4)), device), ragged_dims=(0, 1))
    source = NestedTensor(_parts(((1, 2, 4), (1, 5, 4)), device), ragged_dims=(0, 1))
    with pytest.raises((ValueError, RuntimeError)):
        target * source


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("source_first", [False, True])
def test_declared_singletons_compile_without_equal_offset_guards(axis, source_first):
    torch.compiler.reset()
    compiled = torch.compile(
        lambda full, mask: (mask - full).concat if source_first else (full - mask).concat,
        backend="aot_eager",
        fullgraph=True,
        dynamic=True,
    )
    try:
        for lengths in ((2, 3), (4, 2)):
            shapes = [(length, length, 4) for length in lengths]
            masks = [tuple(1 if dim == axis else size for dim, size in enumerate(shape)) for shape in shapes]
            target_parts, source_parts = _parts(shapes, "cpu"), _parts(masks, "cpu", 0.5)
            target = NestedTensor(target_parts, ragged_dims=(0, 1))
            source = NestedTensor(source_parts, ragged_dims=(0, 1))
            expected = NestedTensor(
                [b - a if source_first else a - b for a, b in zip(target_parts, source_parts)], ragged_dims=(0, 1)
            )
            with _guard():
                values = compiled(target, source)
                gradients = torch.autograd.grad(values.square().sum(), (*target_parts, *source_parts))
            expected_gradients = torch.autograd.grad(expected.concat.square().sum(), (*target_parts, *source_parts))
            torch.testing.assert_close(values, expected.concat)
            for actual, reference in zip(gradients, expected_gradients):
                torch.testing.assert_close(actual, reference)
    finally:
        torch.compiler.reset()
