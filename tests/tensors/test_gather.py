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

r"""Gather-family parity tests against per-element dense PyTorch operations."""

from __future__ import annotations

import math

import pytest
import torch

from danling.tensors import NestedTensor, nested_execution_guard

NT = NestedTensor


def logical_dim(element_dim: int, batch_first: bool) -> int:
    if batch_first or element_dim > 0:
        return element_dim + 1
    return element_dim


def build_elements(shapes):
    return [
        torch.arange(float(math.prod(shape))).reshape(shape) + 1000.0 * position
        for position, shape in enumerate(shapes)
    ]


def build_indices(shapes, element_dim, *, seed=0, index_shapes=None):
    generator = torch.Generator().manual_seed(seed)
    indices = []
    for shape, index_shape in zip(shapes, index_shapes or shapes):
        extent = shape[element_dim]
        if extent == 0:
            indices.append(torch.zeros(index_shape, dtype=torch.long))
        else:
            indices.append(torch.randint(0, extent, tuple(index_shape), generator=generator))
    return indices


def select_index(shapes, element_dim):
    extent = min(shape[element_dim] for shape in shapes)
    return torch.tensor(list(reversed(range(min(extent, 2)))), dtype=torch.long)


def assert_matches(output, expected):
    assert isinstance(output, NestedTensor)
    assert len(output) == len(expected)
    for actual, reference in zip(output, expected):
        assert actual.shape == reference.shape
        torch.testing.assert_close(actual, reference)


def projected_case():
    sampled = NT(
        [torch.randn(2, 3, 4), torch.randn(2, 5, 4), torch.randn(2, 2, 4)],
        ragged_dims=(1,),
    )
    rows = sampled.concat.shape[0]
    tail = torch.arange(float(rows * 2 * 6)).reshape(rows, 2, 6)
    return sampled.packed_with_static_tail(tail)


def cross_topology_row_case(truth_lengths, selected_rows, *, prefix=2, requires_grad=True):
    elements = [torch.randn(prefix, length, 3, requires_grad=requires_grad) for length in truth_lengths]
    return elements, NT(elements, ragged_dims=(1,)), NT(selected_rows, ragged_dims=(0,))


CASES = [
    ("leading-ragged", [(3, 4), (5, 4), (2, 4)], 0, {}),
    ("trailing-ragged", [(4, 3), (4, 5), (4, 2)], 1, {"ragged_dims": (1,)}),
    ("multi-ragged-inner", [(3, 4, 2), (5, 2, 2), (2, 3, 2)], 1, {"ragged_dims": (0, 1)}),
    ("multi-ragged-outer", [(3, 4, 2), (5, 2, 2), (2, 3, 2)], 0, {"ragged_dims": (0, 1)}),
    ("static-tail", [(2, 3, 4), (2, 5, 4), (2, 2, 4)], 2, {"ragged_dims": (1,)}),
    ("batch-second", [(3, 4), (5, 4), (2, 4)], 0, {"batch_first": False}),
    ("empty-segment", [(1,), (0,), (4,)], 0, {"ragged_dims": (0,)}),
]


def cases_named(*names):
    return [case for case in CASES if case[0] in names]


def case_params(cases=None):
    cases = CASES if cases is None else cases
    return pytest.mark.parametrize(
        ("case_id", "shapes", "element_dim", "metadata"),
        cases,
        ids=[case[0] for case in cases],
    )


class TestGatherParity:

    @case_params(
        cases_named(
            "leading-ragged",
            "trailing-ragged",
            "multi-ragged-inner",
            "multi-ragged-outer",
            "static-tail",
            "batch-second",
            "empty-segment",
        )
    )
    def test_gather(self, case_id, shapes, element_dim, metadata):
        elements = build_elements(shapes)
        indices = build_indices(shapes, element_dim)
        dim = logical_dim(element_dim, metadata.get("batch_first", True))

        output = torch.gather(NT(elements, **metadata), dim, NT(indices, **metadata))

        assert_matches(
            output,
            [torch.gather(element, element_dim, index) for element, index in zip(elements, indices)],
        )

    @case_params(cases_named("leading-ragged", "trailing-ragged", "batch-second"))
    def test_take_along_dim(self, case_id, shapes, element_dim, metadata):
        elements = build_elements(shapes)
        indices = build_indices(shapes, element_dim)
        dim = logical_dim(element_dim, metadata.get("batch_first", True))

        output = torch.take_along_dim(NT(elements, **metadata), NT(indices, **metadata), dim=dim)

        assert_matches(
            output,
            [torch.take_along_dim(element, index, dim=element_dim) for element, index in zip(elements, indices)],
        )

    @case_params(cases_named("leading-ragged", "multi-ragged-outer", "static-tail", "batch-second"))
    def test_index_select(self, case_id, shapes, element_dim, metadata):
        elements = build_elements(shapes)
        index = select_index(shapes, element_dim)
        dim = logical_dim(element_dim, metadata.get("batch_first", True))

        output = torch.index_select(NT(elements, **metadata), dim, index)

        assert_matches(
            output,
            [torch.index_select(element, element_dim, index) for element in elements],
        )

    def test_method_forms(self):
        shapes = [(3, 4), (5, 4), (2, 4)]
        elements = build_elements(shapes)
        indices = build_indices(shapes, 0)
        nested, index = NT(elements), NT(indices)

        assert_matches(
            nested.gather(1, index),
            [torch.gather(element, 0, idx) for element, idx in zip(elements, indices)],
        )
        assert_matches(
            torch.gather(nested, 1, index, sparse_grad=True),
            [torch.gather(element, 0, idx, sparse_grad=True) for element, idx in zip(elements, indices)],
        )
        assert_matches(
            nested.take_along_dim(index, dim=1),
            [torch.take_along_dim(element, idx, dim=0) for element, idx in zip(elements, indices)],
        )
        selected = torch.tensor([1, 0])
        assert_matches(
            nested.index_select(1, selected),
            [torch.index_select(element, 0, selected) for element in elements],
        )

    def test_projected_layout(self):
        projected = projected_case()
        elements = list(projected)
        shapes = [tuple(element.shape) for element in elements]
        indices = build_indices(shapes, 1)
        index = NT(indices, ragged_dims=(1,))
        selected = select_index(shapes, 1)

        assert_matches(
            torch.gather(projected, 2, index),
            [torch.gather(element, 1, idx) for element, idx in zip(elements, indices)],
        )
        assert_matches(
            torch.take_along_dim(projected, index, dim=2),
            [torch.take_along_dim(element, idx, dim=1) for element, idx in zip(elements, indices)],
        )
        assert_matches(
            torch.index_select(projected, 2, selected),
            [torch.index_select(element, 1, selected) for element in elements],
        )

    def test_gather_index_may_have_a_different_output_shape(self):
        shapes = [(3, 4), (5, 4), (2, 4)]
        index_shapes = [(2, 2), (6, 2), (1, 2)]
        elements = build_elements(shapes)
        indices = build_indices(shapes, 0, index_shapes=index_shapes)

        output = torch.gather(NT(elements), 1, NT(indices))

        assert_matches(
            output,
            [torch.gather(element, 0, index) for element, index in zip(elements, indices)],
        )




class TestNarrowIndex:

    def test_different_source_and_index_shapes(self):
        source_shapes = [(3, 4), (5, 4), (2, 4)]
        index_shapes = [(2, 2), (6, 2), (1, 2)]
        elements = build_elements(source_shapes)
        indices = build_indices(source_shapes, 0, index_shapes=index_shapes)
        index = NT(indices, ragged_dims=(0,))

        output = torch.gather(NT(elements, ragged_dims=(0,)), 1, index)
        expected = [torch.gather(element, 0, sample_index) for element, sample_index in zip(elements, indices)]

        assert output.shape == index.shape
        assert output.ragged_dims == index.ragged_dims
        torch.testing.assert_close(output.element_sizes(), index.element_sizes())
        for actual, reference in zip(output, expected, strict=True):
            torch.testing.assert_close(actual, reference)

    def test_different_shapes_with_fake_tensors(self):
        fake_tensor = pytest.importorskip("torch._subclasses.fake_tensor")
        source = NT([torch.empty(3, 4), torch.empty(5, 4), torch.empty(2, 4)], ragged_dims=(0,))
        index = NT(
            [
                torch.empty(2, 2, dtype=torch.long),
                torch.empty(6, 2, dtype=torch.long),
                torch.empty(1, 2, dtype=torch.long),
            ],
            ragged_dims=(0,),
        )

        with fake_tensor.FakeTensorMode() as mode:
            fake_source = source.packed_like(mode.from_tensor(source.concat))
            fake_index = index.packed_like(mode.from_tensor(index.concat))
            output = torch.gather(fake_source, 1, fake_index)

        assert fake_tensor.is_fake(output.concat)
        assert output.shape == fake_index.shape
        assert output.ragged_dims == fake_index.ragged_dims
        assert output.concat.shape == fake_index.concat.shape

    @pytest.mark.parametrize("operation", ["gather", "take_along_dim"])
    def test_multi_ragged_row_selection(self, operation):
        source_shapes = [(4, 3, 2), (3, 2, 2)]
        elements = build_elements(source_shapes)
        rows = [torch.tensor([3, 1]), torch.tensor([2])]
        indices = [row[:, None, None].expand(-1, shape[1], shape[2]) for row, shape in zip(rows, source_shapes)]
        source = NT(elements, ragged_dims=(0, 1))
        index = NT(indices, ragged_dims=(0, 1))

        if operation == "gather":
            output = torch.gather(source, 1, index)
            expected = [torch.gather(element, 0, sample_index) for element, sample_index in zip(elements, indices)]
        else:
            output = torch.take_along_dim(source, index, dim=1)
            expected = [
                torch.take_along_dim(element, sample_index, dim=0) for element, sample_index in zip(elements, indices)
            ]

        assert output.shape == index.shape
        assert output.ragged_dims == index.ragged_dims
        torch.testing.assert_close(output.element_sizes(), index.element_sizes())
        for actual, reference in zip(output, expected, strict=True):
            torch.testing.assert_close(actual, reference)

    @pytest.mark.parametrize("operation", ["gather", "take_along_dim"])
    def test_multi_ragged_row_selection_with_fake_tensors(self, operation):
        fake_tensor = pytest.importorskip("torch._subclasses.fake_tensor")
        symbolic_shapes = pytest.importorskip("torch.fx.experimental.symbolic_shapes")
        source = NT([torch.empty(4, 3, 2), torch.empty(3, 2, 2)], ragged_dims=(0, 1))
        index = NT(
            [
                torch.empty(2, 3, 2, dtype=torch.long),
                torch.empty(1, 2, 2, dtype=torch.long),
            ],
            ragged_dims=(0, 1),
        )

        shape_env = symbolic_shapes.ShapeEnv(allow_dynamic_output_shape_ops=True)
        with fake_tensor.FakeTensorMode(shape_env=shape_env) as mode:
            fake_source = mode.from_tensor(source)
            fake_index = mode.from_tensor(index)
            if operation == "gather":
                output = torch.gather(fake_source, 1, fake_index)
            else:
                output = torch.take_along_dim(fake_source, fake_index, dim=1)

        assert fake_tensor.is_fake(output.concat)
        assert output.shape == fake_index.shape
        assert output.ragged_dims == fake_index.ragged_dims
        assert output.concat.shape == fake_index.concat.shape


class TestCrossTopologyRowSelection:

    @staticmethod
    def select(operation, source, index):
        index = index.unsqueeze(1).unsqueeze(-1).expand(-1, source.shape[1], -1, source.shape[-1])
        if operation == "gather":
            return torch.gather(source, 2, index)
        return torch.take_along_dim(source, index, dim=2)

    @staticmethod
    def dense_reference(operation, elements, index):
        index = [
            sample_index[None, :, None].expand(element.shape[0], -1, element.shape[-1])
            for element, sample_index in zip(elements, index)
        ]
        if operation == "gather":
            return [torch.gather(element, 1, sample_index) for element, sample_index in zip(elements, index)]
        return [torch.take_along_dim(element, sample_index, dim=1) for element, sample_index in zip(elements, index)]

    @pytest.mark.parametrize("operation", ["gather", "take_along_dim"])
    def test_static_prefix_matches_dense_reference_without_fallback(self, operation):
        elements, source, index = cross_topology_row_case(
            (4, 3),
            (torch.tensor([3, 1]), torch.tensor([2])),
        )

        with nested_execution_guard(
            forbid_iteration=True,
            forbid_storage_map=True,
            forbid_eager_fallback=True,
            forbid_padded_materialization=True,
            forbid_dense_repack=True,
        ):
            output = self.select(operation, source, index)

        expected = self.dense_reference(operation, elements, list(index))
        assert output.ragged_dims == (1,)
        assert_matches(output, expected)

    @pytest.mark.parametrize("operation", ["gather", "take_along_dim"])
    def test_static_prefix_with_fake_tensors(self, operation):
        fake_tensor = pytest.importorskip("torch._subclasses.fake_tensor")
        symbolic_shapes = pytest.importorskip("torch.fx.experimental.symbolic_shapes")
        _, source, index = cross_topology_row_case(
            (4, 3),
            (torch.tensor([3, 1]), torch.tensor([2])),
        )

        shape_env = symbolic_shapes.ShapeEnv(allow_dynamic_output_shape_ops=True)
        with fake_tensor.FakeTensorMode(shape_env=shape_env) as mode:
            fake_source = mode.from_tensor(source)
            fake_index = mode.from_tensor(index)
            expanded_index = (
                fake_index.unsqueeze(1).unsqueeze(-1).expand(-1, fake_source.shape[1], -1, fake_source.shape[-1])
            )
            output = self.select(operation, fake_source, fake_index)

        assert fake_tensor.is_fake(output.concat)
        assert output.ragged_dims == (1,)
        assert output.shape == expanded_index.shape
        assert output.element_sizes().shape == expanded_index.element_sizes().shape


class TestDenseIndex:

    def test_batch_shaped_dense_index(self):
        shapes = [(3, 4), (5, 4), (2, 4)]
        elements = build_elements(shapes)
        nested = NT(elements)
        dense = torch.zeros(nested.shape, dtype=torch.long)
        for position, shape in enumerate(shapes):
            dense[position, : shape[0], : shape[1]] = 1

        output = torch.gather(nested, 2, dense)

        expected = [
            torch.gather(element, 1, dense[position, : shape[0], : shape[1]])
            for position, (element, shape) in enumerate(zip(elements, shapes))
        ]
        assert_matches(output, expected)

    @pytest.mark.parametrize(
        ("dim", "dense"),
        [
            (1, torch.zeros(2, 4, dtype=torch.long)),
            (2, torch.tensor([[3, 2, 1, 0]])),
        ],
        ids=["ragged-dim", "static-dim"],
    )
    def test_shared_dense_index(self, dim, dense):
        elements = build_elements([(3, 4), (5, 4), (2, 4)])

        output = torch.gather(NT(elements), dim, dense)

        assert_matches(
            output,
            [torch.gather(element, dim - 1, dense) for element in elements],
        )

    def test_index_taller_than_a_sample_is_valid_when_reading_the_ragged_dim(self):
        elements = build_elements([(3, 4), (5, 4), (2, 4)])
        dense = torch.ones(6, 4, dtype=torch.long)

        output = torch.gather(NT(elements), 1, dense)

        assert_matches(output, [torch.gather(element, 0, dense) for element in elements])


class TestGatherErrors:

    def test_segment_local_indices_do_not_cross_samples(self):
        elements = [torch.arange(12.0).reshape(3, 4), torch.arange(8.0).reshape(2, 4) + 100]
        indices = [torch.full((3, 4), 2, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long)]

        output = torch.gather(NT(elements), 1, NT(indices))

        assert_matches(
            output,
            [torch.gather(element, 0, index) for element, index in zip(elements, indices)],
        )

    @pytest.mark.parametrize("bad_index", [-1, 2])
    def test_out_of_range_ragged_index_raises(self, bad_index):
        elements = [torch.zeros(3, 4), torch.zeros(2, 4)]
        indices = [torch.zeros(3, 4, dtype=torch.long), torch.full((2, 4), bad_index, dtype=torch.long)]

        with pytest.raises(IndexError, match="out of bounds"):
            torch.gather(NT(elements), 1, NT(indices))

    @pytest.mark.parametrize("bad_index", [-1, 5])
    def test_index_select_out_of_range_raises(self, bad_index):
        elements = [torch.zeros(3, 4), torch.zeros(2, 4)]

        with pytest.raises(IndexError, match="out of bounds"):
            torch.index_select(NT(elements), 1, torch.tensor([bad_index]))

    @pytest.mark.parametrize("operation", ["gather", "take_along_dim"])
    def test_nested_index_cannot_address_the_batch_dimension(self, operation):
        elements = [torch.zeros(3, 4), torch.zeros(2, 4)]
        indices = [torch.zeros(3, 4, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long)]

        with pytest.raises(ValueError, match="batch dimension"):
            if operation == "gather":
                torch.gather(NT(elements), 0, NT(indices))
            else:
                torch.take_along_dim(NT(elements), NT(indices), dim=0)

    def test_nested_index_batch_lengths_must_match(self):
        elements = [torch.zeros(3, 4), torch.zeros(2, 4)]
        indices = [torch.zeros(3, 4, dtype=torch.long)]

        with pytest.raises(ValueError, match="batch length mismatch"):
            torch.gather(NT(elements), 1, NT(indices))


class TestAdditionalSpellings:

    def test_take_along_dim_without_a_dimension_flattens_each_element(self):
        elements = build_elements([(3, 4), (5, 4), (2, 4)])
        indices = [torch.tensor([0, 3, 5]), torch.tensor([1, 2]), torch.tensor([7])]

        output = torch.take_along_dim(NT(elements), NT(indices), dim=None)

        assert_matches(
            output,
            [torch.take_along_dim(element, index, dim=None) for element, index in zip(elements, indices)],
        )

    def test_take_along_dim_broadcasts_a_dense_index(self):
        elements = build_elements([(3, 4), (5, 4), (2, 4)])
        dense = torch.zeros((1, 4), dtype=torch.long)

        for output in (
            torch.take_along_dim(NT(elements), dense, dim=1),
            NT(elements).take_along_dim(dense, dim=1),
        ):
            assert_matches(
                output,
                [torch.take_along_dim(element, dense, dim=0) for element in elements],
            )

    def test_index_select_on_the_batch_dimension(self):
        elements = build_elements([(3, 4), (5, 4), (2, 4)])
        index = torch.tensor([2, 0, 2])

        output = torch.index_select(NT(elements), 0, index)

        assert_matches(output, [elements[position] for position in index.tolist()])


class TestGatherBounds:

    def test_out_of_range_index_raises(self):
        bases = [torch.zeros(2, 4), torch.full((3, 4), 100.0)]
        indices = [torch.full((1, 4), 2, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long)]
        message = r"gather: index 2 is out of bounds for dimension 1 with size 2"
        with pytest.raises(IndexError, match=message):
            torch.gather(NT(bases), 1, NT(indices))


class TestGatherAutograd:

    def test_gather_gradient_matches_dense_reference(self):
        shapes = [(3, 4), (5, 4), (2, 4)]
        elements = [element.requires_grad_() for element in build_elements(shapes)]
        indices = build_indices(shapes, 0)

        torch.gather(NT(elements), 1, NT(indices)).concat.sum().backward()

        for element, index in zip(elements, indices):
            reference = element.detach().clone().requires_grad_()
            torch.gather(reference, 0, index).sum().backward()
            torch.testing.assert_close(element.grad, reference.grad)

    def test_take_along_dim_gradient_matches_dense_reference(self):
        shapes = [(3, 4), (5, 4), (2, 4)]
        elements = [element.requires_grad_() for element in build_elements(shapes)]
        indices = build_indices(shapes, 0)

        torch.take_along_dim(NT(elements), NT(indices), dim=1).concat.sum().backward()

        for element, index in zip(elements, indices):
            reference = element.detach().clone().requires_grad_()
            torch.take_along_dim(reference, index, dim=0).sum().backward()
            torch.testing.assert_close(element.grad, reference.grad)

    def test_index_select_gradient_matches_dense_reference(self):
        elements = [element.requires_grad_() for element in build_elements([(3, 4), (5, 4), (2, 4)])]
        index = torch.tensor([1, 0])

        torch.index_select(NT(elements), 1, index).concat.sum().backward()

        for element in elements:
            reference = element.detach().clone().requires_grad_()
            torch.index_select(reference, 0, index).sum().backward()
            torch.testing.assert_close(element.grad, reference.grad)

    def test_batch_index_select_accumulates_repeated_gradients(self):
        elements = [element.requires_grad_() for element in build_elements([(3, 4), (5, 4), (2, 4)])]

        torch.index_select(NT(elements), 0, torch.tensor([2, 0, 2])).concat.sum().backward()

        torch.testing.assert_close(elements[2].grad, torch.full((2, 4), 2.0))
        torch.testing.assert_close(elements[0].grad, torch.ones(3, 4))
        torch.testing.assert_close(elements[1].grad, torch.zeros(5, 4))


class TestGatherCompile:

    def test_gather_fullgraph(self):
        shapes = [(2, 3, 4), (2, 5, 4)]
        elements = build_elements(shapes)
        indices = build_indices(shapes, 1)
        compiled = torch.compile(
            lambda input_, index: torch.gather(input_, 2, index),
            backend="aot_eager",
            fullgraph=True,
        )

        output = compiled(
            NT(elements, ragged_dims=(1,)),
            NT(indices, ragged_dims=(1,)),
        )

        assert_matches(
            output,
            [torch.gather(element, 1, index) for element, index in zip(elements, indices)],
        )

    def test_index_select_fullgraph(self):
        shapes = [(3, 4), (5, 4), (2, 4)]
        elements = build_elements(shapes)
        index = torch.tensor([1, 0])
        compiled = torch.compile(
            lambda input_, index_: torch.index_select(input_, 1, index_),
            backend="aot_eager",
            fullgraph=True,
        )

        output = compiled(NT(elements, ragged_dims=(0,)), index)

        assert_matches(output, [torch.index_select(element, 0, index) for element in elements])

    def test_projected_gather_fullgraph(self):
        projected = projected_case()
        elements = list(projected)
        indices = build_indices([tuple(element.shape) for element in elements], 1)
        compiled = torch.compile(
            lambda input_, index: torch.gather(input_, 2, index),
            backend="aot_eager",
            fullgraph=True,
        )

        output = compiled(projected, NT(indices, ragged_dims=(1,)))

        assert_matches(
            output,
            [torch.gather(element, 1, index) for element, index in zip(elements, indices)],
        )

    @pytest.mark.parametrize("operation", ["gather", "take_along_dim"])
    def test_multi_ragged_row_selection_dynamic_fullgraph(self, operation):
        def select(source, index):
            if operation == "gather":
                return torch.gather(source, 1, index)
            return torch.take_along_dim(source, index, dim=1)

        cases = [
            ([(4, 3, 2), (3, 2, 2)], [torch.tensor([3, 1]), torch.tensor([2])]),
            ([(5, 2, 2), (2, 4, 2)], [torch.tensor([4, 2, 0]), torch.tensor([1])]),
        ]
        compiled = torch.compile(select, backend="aot_eager", fullgraph=True, dynamic=True)

        for source_shapes, rows in cases:
            indices = [row[:, None, None].expand(-1, shape[1], shape[2]) for row, shape in zip(rows, source_shapes)]
            elements = build_elements(source_shapes)
            source = NT(elements, ragged_dims=(0, 1))
            index = NT(indices, ragged_dims=(0, 1))

            actual = compiled(source, index)
            if operation == "gather":
                expected = [torch.gather(element, 0, sample_index) for element, sample_index in zip(elements, indices)]
            else:
                expected = [
                    torch.take_along_dim(element, sample_index, dim=0)
                    for element, sample_index in zip(elements, indices)
                ]
            assert actual.ragged_dims == index.ragged_dims
            torch.testing.assert_close(actual.element_sizes(), index.element_sizes())
            assert_matches(actual, expected)

    def test_different_shapes_aot_eager_vjp_and_runtime_bounds(self):
        source_lengths = (3, 5, 2)
        index_lengths = (2, 6, 1)
        source_template = NT([torch.empty(length, 4) for length in source_lengths], ragged_dims=(0,))
        index_template = NT(
            [torch.empty(length, 2, dtype=torch.long) for length in index_lengths],
            ragged_dims=(0,),
        )
        indices = build_indices(
            [(length, 4) for length in source_lengths],
            0,
            index_shapes=[(length, 2) for length in index_lengths],
        )
        packed_indices = torch.cat(indices)

        def gather_values(source_structure, source_values, index_structure, index_values):
            source = source_structure.packed_like(source_values)
            index = index_structure.packed_like(index_values)
            return torch.gather(source, 1, index).concat

        compiled = torch.compile(gather_values, backend="aot_eager", fullgraph=True)
        source_values = torch.randn_like(source_template.concat, requires_grad=True)
        actual = compiled(source_template, source_values, index_template, packed_indices)
        source_elements = source_values.split(source_lengths)
        expected = torch.cat(
            [torch.gather(element, 0, sample_index) for element, sample_index in zip(source_elements, indices)]
        )
        cotangent = torch.randn_like(actual)
        actual_grad = torch.autograd.grad(actual, source_values, cotangent, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, source_values, cotangent)[0]

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad, expected_grad)

        out_of_bounds = packed_indices.clone()
        out_of_bounds[0, 0] = source_lengths[0]
        with pytest.raises(RuntimeError, match="gather: index is out of bounds"):
            compiled(source_template, source_values, index_template, out_of_bounds)
