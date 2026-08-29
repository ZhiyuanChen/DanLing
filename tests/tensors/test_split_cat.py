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

r"""Split/cat-family parity tests against per-element dense PyTorch operations."""

from __future__ import annotations

import math

import pytest
import torch

from danling.tensors import NestedTensor

NT = NestedTensor


def logical_dim(element_dim: int, batch_first: bool) -> int:
    if batch_first or element_dim > 0:
        return element_dim + 1
    return element_dim


def batch_dim_of(batch_first: bool) -> int:
    return 0 if batch_first else 1


def build_elements(shapes, offset: float = 0.0):
    return [
        torch.arange(float(math.prod(shape))).reshape(shape) + 1000.0 * position + offset
        for position, shape in enumerate(shapes)
    ]


def assert_matches(output, expected):
    assert isinstance(output, NestedTensor)
    assert len(output) == len(expected)
    for actual, reference in zip(output, expected):
        assert actual.shape == reference.shape
        torch.testing.assert_close(actual, reference)


def assert_parts_match(parts, expected_parts):
    assert len(parts) == len(expected_parts)
    for part, expected in zip(parts, expected_parts):
        assert_matches(part, expected)


def projected_case():
    return NT([torch.randn(2, 3, 4), torch.randn(2, 5, 4), torch.randn(2, 2, 4)]).permute(0, 3, 1, 2)


CASES = [
    ("leading-ragged", [(3, 4), (5, 4), (2, 4)], {"ragged_dims": (0,)}),
    ("trailing-ragged", [(4, 3), (4, 5), (4, 2)], {"ragged_dims": (1,)}),
    ("multi-ragged", [(3, 4, 2), (5, 2, 2), (2, 3, 2)], {"ragged_dims": (0, 1)}),
    ("batch-second", [(3, 4), (5, 4), (2, 4)], {"batch_first": False}),
    ("empty-segment", [(0, 4), (5, 4)], {"ragged_dims": (0,)}),
]


def cases_named(*names):
    return [case for case in CASES if case[0] in names]


def case_params(cases=None):
    cases = CASES if cases is None else cases
    return pytest.mark.parametrize(
        ("case_id", "shapes", "metadata"),
        cases,
        ids=[case[0] for case in cases],
    )


class TestNarrow:

    @case_params(cases_named("leading-ragged", "batch-second", "empty-segment"))
    def test_batch_dimension(self, case_id, shapes, metadata):
        elements = build_elements(shapes)
        batch_dim = batch_dim_of(metadata.get("batch_first", True))

        output = torch.narrow(NT(elements, **metadata), batch_dim, 1, 1)

        assert_matches(output, elements[1:2])

    @case_params(cases_named("leading-ragged", "trailing-ragged", "multi-ragged", "batch-second"))
    def test_element_dimensions(self, case_id, shapes, metadata):
        elements = build_elements(shapes)
        batch_first = metadata.get("batch_first", True)
        for element_dim in range(len(shapes[0])):
            length = min(2, min(shape[element_dim] for shape in shapes))

            output = torch.narrow(
                NT(elements, **metadata),
                logical_dim(element_dim, batch_first),
                0,
                length,
            )

            assert_matches(
                output,
                [torch.narrow(element, element_dim, 0, length) for element in elements],
            )

    def test_method_and_negative_indices(self):
        elements = build_elements([(3, 4), (5, 4), (2, 4)])

        output = NT(elements).narrow(-1, -2, 2)

        assert_matches(output, [torch.narrow(element, 1, -2, 2) for element in elements])

    def test_projected_layout(self):
        projected = projected_case()
        elements = list(projected)

        output = torch.narrow(projected, 2, 0, 2)

        assert_matches(output, [torch.narrow(element, 1, 0, 2) for element in elements])

    def test_invalid_span_or_dimension_raises(self):
        nested = NT([torch.randn(2, 4), torch.randn(5, 4)])
        with pytest.raises(RuntimeError):
            torch.narrow(nested, 1, 0, 4)
        with pytest.raises(IndexError):
            torch.narrow(nested, 3, 0, 1)


class TestSplitFamily:

    @case_params(cases_named("leading-ragged", "batch-second"))
    def test_split_batch_dimension(self, case_id, shapes, metadata):
        elements = build_elements(shapes)
        nested = NT(elements, **metadata)
        batch_dim = batch_dim_of(metadata.get("batch_first", True))

        parts = torch.split(nested, 2, dim=batch_dim)
        index_parts = torch.split(torch.arange(len(elements)), 2)
        expected = [[elements[int(index)] for index in indices] for indices in index_parts]

        assert_parts_match(parts, expected)

    @pytest.mark.parametrize("operation", ["chunk", "tensor_split"])
    def test_chunk_and_tensor_split_batch_dimension(self, operation):
        elements = build_elements([(3, 4), (5, 4), (2, 4)])
        function = getattr(torch, operation)

        parts = function(NT(elements), 2, dim=0)
        index_parts = function(torch.arange(len(elements)), 2)

        assert_parts_match(parts, [[elements[int(index)] for index in indices] for indices in index_parts])

    @case_params(cases_named("leading-ragged", "trailing-ragged", "multi-ragged", "batch-second"))
    def test_split_element_dimensions(self, case_id, shapes, metadata):
        elements = build_elements(shapes)
        nested = NT(elements, **metadata)
        batch_first = metadata.get("batch_first", True)
        for element_dim in range(len(shapes[0])):
            references = [torch.split(element, 2, dim=element_dim) for element in elements]
            dim = logical_dim(element_dim, batch_first)
            if len({len(reference) for reference in references}) > 1:
                with pytest.raises(ValueError, match="uniform per-element split counts"):
                    torch.split(nested, 2, dim=dim)
                continue

            parts = torch.split(nested, 2, dim=dim)

            assert_parts_match(parts, [list(part) for part in zip(*references)])

    def test_chunk_and_tensor_split_element_dimensions(self):
        elements = build_elements([(3, 4), (5, 4), (2, 4)])
        nested = NT(elements)

        chunked = torch.chunk(nested, 2, dim=2)
        chunk_references = [torch.chunk(element, 2, dim=1) for element in elements]
        assert_parts_match(chunked, [list(part) for part in zip(*chunk_references)])

        split = torch.tensor_split(nested, 3, dim=1)
        split_references = [torch.tensor_split(element, 3, dim=0) for element in elements]
        assert_parts_match(split, [list(part) for part in zip(*split_references)])

    def test_tensor_split_preserves_zero_width_and_source_derived_empty_shapes(self):
        elements = [torch.arange(2.0), torch.arange(10.0, 13.0)]
        nested = NT(elements, ragged_dims=(0,))

        parts = torch.tensor_split(nested, [0], dim=1)

        references = [torch.tensor_split(element, [0]) for element in elements]
        assert_parts_match(parts, [list(part) for part in zip(*references)])
        empty_parts = torch.tensor_split(nested[:0], [0], dim=1)
        assert [part.shape for part in empty_parts] == [part[:0].shape for part in parts]

    def test_split_sections_and_errors(self):
        elements = build_elements([(3, 4), (5, 4), (2, 4)])
        parts = torch.split(NT(elements), [1, 2], dim=0)
        assert_parts_match(parts, [elements[:1], elements[1:]])

        with pytest.raises(ValueError):
            torch.split(NT(elements), [1], dim=0)

    def test_vsplit_hsplit_and_dsplit_axes(self):
        elements = build_elements([(4, 6), (4, 6)])
        nested = NT(elements, ragged_dims=(1,))

        assert_parts_match(torch.vsplit(nested, 2), [elements[:1], elements[1:]])
        assert_parts_match(
            torch.hsplit(nested, 2),
            [[element[:2] for element in elements], [element[2:] for element in elements]],
        )
        assert_parts_match(
            torch.dsplit(nested, 2),
            [[element[:, :3] for element in elements], [element[:, 3:] for element in elements]],
        )


class TestUnbind:

    def test_batch_and_static_dimensions(self):
        elements = build_elements([(3, 4), (5, 4), (2, 4)])
        nested = NT(elements, ragged_dims=(0,))

        batch_parts = torch.unbind(nested, dim=0)
        for actual, reference in zip(batch_parts, elements):
            torch.testing.assert_close(actual, reference)

        static_parts = torch.unbind(nested, dim=2)
        references = [torch.unbind(element, dim=1) for element in elements]
        assert_parts_match(static_parts, [list(part) for part in zip(*references)])

    def test_ragged_dimension_is_refused(self):
        nested = NT(build_elements([(3, 4), (5, 4)]), ragged_dims=(0,))

        with pytest.raises(NotImplementedError, match="ragged"):
            torch.unbind(nested, dim=1)


class TestCat:

    @case_params(
        cases_named(
            "leading-ragged",
            "trailing-ragged",
            "multi-ragged",
            "batch-second",
            "empty-segment",
        )
    )
    def test_batch_dimension(self, case_id, shapes, metadata):
        left = build_elements(shapes)
        right = build_elements(shapes, offset=7.0)
        batch_dim = batch_dim_of(metadata.get("batch_first", True))

        output = torch.cat((NT(left, **metadata), NT(right, **metadata)), dim=batch_dim)

        assert_matches(output, [*left, *right])

    @case_params(cases_named("leading-ragged", "trailing-ragged", "multi-ragged", "batch-second"))
    def test_element_dimensions(self, case_id, shapes, metadata):
        left = build_elements(shapes)
        right = build_elements(shapes, offset=7.0)
        batch_first = metadata.get("batch_first", True)
        for element_dim in range(len(shapes[0])):

            output = torch.cat(
                (NT(left, **metadata), NT(right, **metadata)),
                dim=logical_dim(element_dim, batch_first),
            )

            assert_matches(
                output,
                [torch.cat((a, b), dim=element_dim) for a, b in zip(left, right)],
            )

    def test_projected_layout(self):
        projected = projected_case()
        elements = list(projected)

        output = torch.cat((projected, projected), dim=2)

        assert_matches(output, [torch.cat((element, element), dim=1) for element in elements])


    def test_batch_dimension_validates_static_extents(self):
        left = NT([torch.randn(3, 4), torch.randn(5, 4)])
        right = NT([torch.randn(3, 6), torch.randn(5, 6)])

        with pytest.raises(ValueError, match="cannot make element dimension"):
            torch.cat((left, right), dim=0)

    @pytest.mark.parametrize(
        "groups",
        [
            [[(4, 3), (4, 5)], [(4, 2)]],
            [[(3, 4), (5, 4)], [(4, 3), (4, 5)]],
        ],
        ids=["single-sample", "different-inferred-orders"],
    )
    def test_batch_dimension_merges_compatible_inferred_layouts(self, groups):
        elements = []
        operands = []
        for position, shapes in enumerate(groups):
            group = build_elements(shapes, offset=7.0 * position)
            elements.extend(group)
            operands.append(NT(group))

        assert_matches(torch.cat(operands, dim=0), elements)

    def test_accepts_a_dense_batch_operand(self):
        elements = build_elements([(4, 3), (4, 5)])
        dense = build_elements([(4, 2)], offset=7.0)[0]

        assert_matches(torch.cat((NT(elements), dense), dim=0), [*elements, dense])

    def test_refuses_to_invent_a_ragged_dimension(self):
        left = build_elements([(2, 3), (1, 3)])
        right = build_elements([(2, 4), (1, 4)], offset=7.0)

        with pytest.raises(ValueError, match="no operand varies"):
            torch.cat((NT(left), NT(right)), dim=0)


class TestStack:

    @pytest.mark.parametrize("batch_first", [False, True])
    def test_element_dimension(self, batch_first):
        shapes = [(3, 4), (5, 4), (2, 4)]
        left = build_elements(shapes)
        right = build_elements(shapes, offset=7.0)
        dim = 2

        output = torch.stack(
            (NT(left, batch_first=batch_first), NT(right, batch_first=batch_first)),
            dim=dim,
        )

        assert_matches(output, [torch.stack((a, b), dim=1) for a, b in zip(left, right)])

    def test_batch_dimension_is_refused(self):
        nested = NT(build_elements([(3, 4), (5, 4)]))

        with pytest.raises(NotImplementedError, match="batch dimension"):
            torch.stack((nested, nested), dim=0)

    def test_common_stack_aliases(self):
        left = build_elements([(3, 4), (5, 4)])
        right = build_elements([(3, 4), (5, 4)], offset=7.0)
        nested_left, nested_right = NT(left), NT(right)

        assert_matches(torch.vstack((nested_left, nested_right)), [*left, *right])
        assert_matches(
            torch.hstack((nested_left, nested_right)),
            [torch.cat((a, b), dim=0) for a, b in zip(left, right)],
        )
        assert_matches(
            torch.dstack((nested_left, nested_right)),
            [torch.cat((a, b), dim=1) for a, b in zip(left, right)],
        )


class TestMethodAndAliasSpellings:

    def test_method_forms(self):
        elements = build_elements([(3, 4), (5, 4)])
        nested = NT(elements)

        assert_matches(nested.narrow(0, 0, 1), elements[:1])
        assert_parts_match(nested.split(1, dim=0), [[elements[0]], [elements[1]]])
        assert_parts_match(nested.chunk(2, dim=0), [[elements[0]], [elements[1]]])
        assert_parts_match(nested.tensor_split(2, dim=0), [[elements[0]], [elements[1]]])
        for actual, reference in zip(nested.unbind(0), elements):
            torch.testing.assert_close(actual, reference)

    def test_tensor_split_accepts_tensor_indices(self):
        elements = build_elements([(3, 4), (5, 4)])

        parts = torch.tensor_split(NT(elements), torch.tensor([1]), dim=0)

        assert_parts_match(parts, [elements[:1], elements[1:]])

    def test_concat_aliases(self):
        left = build_elements([(3, 4), (5, 4)])
        right = build_elements([(3, 4), (5, 4)], offset=7.0)

        for alias in (torch.concat, torch.concatenate):
            assert_matches(alias((NT(left), NT(right)), dim=0), [*left, *right])


class TestCompile:

    def test_narrow_and_split_fullgraph(self):
        elements = build_elements([(3, 4), (5, 4), (2, 4)])
        nested = NT(elements, ragged_dims=(0,))
        compiled_narrow = torch.compile(
            lambda input_: torch.narrow(input_, 1, 0, 2),
            backend="aot_eager",
            fullgraph=True,
        )
        compiled_split = torch.compile(
            lambda input_: torch.split(input_, 2, dim=2),
            backend="aot_eager",
            fullgraph=True,
        )

        output = compiled_narrow(nested)
        parts = compiled_split(nested)

        assert_matches(output, [torch.narrow(element, 0, 0, 2) for element in elements])
        references = [torch.split(element, 2, dim=1) for element in elements]
        assert_parts_match(parts, [list(part) for part in zip(*references)])

    def test_cat_and_stack_fullgraph(self):
        shapes = [(3, 4), (5, 4), (2, 4)]
        left = build_elements(shapes)
        right = build_elements(shapes, offset=7.0)
        compiled_cat = torch.compile(
            lambda a, b: torch.cat((a, b), dim=0),
            backend="aot_eager",
            fullgraph=True,
        )
        compiled_stack = torch.compile(
            lambda a, b: torch.stack((a, b), dim=2),
            backend="aot_eager",
            fullgraph=True,
        )

        cat_output = compiled_cat(NT(left, ragged_dims=(0,)), NT(right, ragged_dims=(0,)))
        stack_output = compiled_stack(NT(left, ragged_dims=(0,)), NT(right, ragged_dims=(0,)))

        assert_matches(cat_output, [*left, *right])
        assert_matches(
            stack_output,
            [torch.stack((a, b), dim=1) for a, b in zip(left, right)],
        )

    def test_dynamic_batch_cat(self):
        compiled = torch.compile(
            lambda a, b: torch.cat((a, b), dim=0),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        for lengths in ((3, 5), (2, 7)):
            left = [torch.randn(length, 4) for length in lengths]
            right = [torch.randn(length + 1, 4) for length in lengths]

            output = compiled(
                NT(left, ragged_dims=(0,)),
                NT(right, ragged_dims=(0,)),
            )

            assert_matches(output, [*left, *right])



class TestAutograd:

    @staticmethod
    def packed(lengths, width=4):
        reference = NT([torch.zeros(length, width) for length in lengths])
        values = torch.randn(sum(lengths), width, requires_grad=True)
        return reference.packed_with_lengths(values, torch.tensor(lengths)), values

    def test_narrow_ragged_dimension(self):
        nested, values = self.packed((3, 5))

        torch.narrow(nested, 1, 0, 2).sum().backward()

        expected = torch.zeros(8, 4)
        expected[0:2] = 1.0
        expected[3:5] = 1.0
        torch.testing.assert_close(values.grad, expected)

    def test_split_static_dimension(self):
        nested, values = self.packed((3, 5))

        torch.split(nested, 2, dim=2)[0].sum().backward()

        expected = torch.zeros(8, 4)
        expected[:, :2] = 1.0
        torch.testing.assert_close(values.grad, expected)

    @pytest.mark.parametrize("dim", [0, 1], ids=["batch", "ragged"])
    def test_cat(self, dim):
        left, left_values = self.packed((3, 5))
        right, right_values = self.packed((2, 4))

        torch.cat((left, right), dim=dim).sum().backward()

        torch.testing.assert_close(left_values.grad, torch.ones(8, 4))
        torch.testing.assert_close(right_values.grad, torch.ones(6, 4))
