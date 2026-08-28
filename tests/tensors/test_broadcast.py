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








class TestLayoutPreservation:


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

