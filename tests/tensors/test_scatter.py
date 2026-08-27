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

r"""Scatter-family parity tests against per-element dense PyTorch operations."""

from __future__ import annotations

import pytest
import torch

from danling.tensors import NestedTensor

NT = NestedTensor


def logical_dim(element_dim: int, batch_first: bool) -> int:
    if batch_first or element_dim > 0:
        return element_dim + 1
    return element_dim


def build_case(shapes, element_dim, *, seed=0, duplicates=False, **metadata):
    generator = torch.Generator().manual_seed(seed)
    bases, indices, sources = [], [], []
    for position, shape in enumerate(shapes):
        count = int(torch.tensor(shape).prod()) if shape else 1
        bases.append(torch.arange(float(count)).reshape(shape) + 1000.0 * position)
        sources.append(-torch.arange(1.0, count + 1.0).reshape(shape) - 1000.0 * position)
        extent = shape[element_dim]
        if extent == 0:
            indices.append(torch.zeros(shape, dtype=torch.long))
        elif duplicates:
            indices.append(torch.randint(0, extent, shape, generator=generator))
        else:
            indices.append(torch.rand(shape, generator=generator).argsort(dim=element_dim))
    return (
        NT(bases, **metadata),
        NT(indices, **metadata),
        NT(sources, **metadata),
        bases,
        indices,
        sources,
    )


def assert_matches(output, expected):
    assert isinstance(output, NestedTensor)
    assert len(output) == len(expected)
    for actual, reference in zip(output, expected):
        assert actual.shape == reference.shape
        torch.testing.assert_close(actual, reference)


CASES = [
    ("leading-ragged", [(3, 4), (2, 4)], 0, {}),
    ("trailing-ragged", [(4, 3), (4, 2)], 1, {"ragged_dims": (1,)}),
    ("multi-ragged-inner", [(3, 4, 2), (2, 3, 2)], 1, {"ragged_dims": (0, 1)}),
    ("multi-ragged-outer", [(3, 4, 2), (2, 3, 2)], 0, {"ragged_dims": (0, 1)}),
    ("static-tail", [(2, 3, 4), (2, 2, 4)], 2, {"ragged_dims": (1,)}),
    ("batch-second", [(3, 4), (2, 4)], 0, {"batch_first": False}),
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


class TestScatterParity:

    @case_params(
        cases_named(
            "leading-ragged",
            "trailing-ragged",
            "multi-ragged-inner",
            "multi-ragged-outer",
            "static-tail",
            "batch-second",
        )
    )
    def test_scatter_tensor_source(self, case_id, shapes, element_dim, metadata):
        nested, index, source, bases, indices, sources = build_case(shapes, element_dim, **metadata)
        dim = logical_dim(element_dim, metadata.get("batch_first", True))

        output = torch.scatter(nested, dim, index, source)

        expected = [torch.scatter(base, element_dim, idx, src) for base, idx, src in zip(bases, indices, sources)]
        assert_matches(output, expected)

    def test_scatter_scalar_source(self):
        shapes = [(3, 4), (2, 4)]
        nested, index, _, bases, indices, _ = build_case(shapes, 0)

        output = torch.scatter(nested, 1, index, -7.5)

        assert_matches(output, [torch.scatter(base, 0, idx, -7.5) for base, idx in zip(bases, indices)])

    @case_params(cases_named("leading-ragged", "trailing-ragged", "batch-second"))
    def test_scatter_add(self, case_id, shapes, element_dim, metadata):
        nested, index, source, bases, indices, sources = build_case(
            shapes,
            element_dim,
            duplicates=True,
            **metadata,
        )
        dim = logical_dim(element_dim, metadata.get("batch_first", True))

        output = torch.scatter_add(nested, dim, index, source)

        expected = [torch.scatter_add(base, element_dim, idx, src) for base, idx, src in zip(bases, indices, sources)]
        assert_matches(output, expected)

    @pytest.mark.parametrize("reduce", ["sum", "prod", "amax", "amin"])
    @pytest.mark.parametrize("include_self", [False, True])
    def test_scatter_reduce(self, reduce, include_self):
        shapes = [(3, 4), (2, 4)]
        nested, index, source, bases, indices, sources = build_case(shapes, 0, duplicates=True)

        output = torch.scatter_reduce(
            nested,
            1,
            index,
            source,
            reduce=reduce,
            include_self=include_self,
        )

        expected = [
            torch.scatter_reduce(base, 0, idx, src, reduce=reduce, include_self=include_self)
            for base, idx, src in zip(bases, indices, sources)
        ]
        assert_matches(output, expected)


class TestScatterErrors:

    def test_segment_local_indices_do_not_cross_samples(self):
        bases = [torch.arange(24.0).reshape(2, 3, 4), torch.arange(16.0).reshape(2, 2, 4)]
        indices = [torch.full((2, 3, 4), 2, dtype=torch.long), torch.zeros(2, 2, 4, dtype=torch.long)]
        sources = [torch.full((2, 3, 4), 7.0), torch.full((2, 2, 4), 8.0)]

        output = torch.scatter(NT(bases), 2, NT(indices), NT(sources))

        assert_matches(
            output,
            [torch.scatter(base, 1, idx, src) for base, idx, src in zip(bases, indices, sources)],
        )

    @pytest.mark.parametrize("bad_index", [-1, 2])
    def test_out_of_range_ragged_index_raises(self, bad_index):
        bases = [torch.zeros(3, 4), torch.zeros(2, 4)]
        indices = [torch.zeros(3, 4, dtype=torch.long), torch.full((2, 4), bad_index, dtype=torch.long)]
        sources = [torch.ones(3, 4), torch.ones(2, 4)]

        with pytest.raises(IndexError, match="out of bounds"):
            torch.scatter(NT(bases), 1, NT(indices), NT(sources))

    def test_batch_dimension_is_refused(self):
        nested = NT([torch.zeros(3, 4), torch.zeros(2, 4)])
        index = NT([torch.zeros(3, 4, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long)])
        source = NT([torch.ones(3, 4), torch.ones(2, 4)])

        with pytest.raises(ValueError, match="batch dimension"):
            torch.scatter(nested, 0, index, source)


class TestScatterCompile:

    def test_scatter_fullgraph(self):
        shapes = [(2, 3, 4), (2, 2, 4)]
        nested, index, source, bases, indices, sources = build_case(
            shapes,
            1,
            ragged_dims=(1,),
        )
        compiled = torch.compile(
            lambda input_, indices_, source_: torch.scatter(input_, 2, indices_, source_),
            backend="aot_eager",
            fullgraph=True,
        )

        output = compiled(nested, index, source)

        assert_matches(
            output,
            [torch.scatter(base, 1, idx, src) for base, idx, src in zip(bases, indices, sources)],
        )


MASKED_CASES = [
    ("leading-ragged", [(3, 4), (2, 4)], {}),
    ("trailing-ragged", [(2, 3), (2, 2)], {"ragged_dims": (1,)}),
    ("empty-segment", [(1, 4), (0, 4)], {}),
    ("batch-second", [(3, 4), (2, 4)], {"batch_first": False}),
]


def build_masked_case(shapes, *, exact_source):
    generator = torch.Generator().manual_seed(7)
    bases, masks, sources = [], [], []
    for position, shape in enumerate(shapes):
        count = int(torch.tensor(shape).prod()) if shape else 1
        bases.append(torch.zeros(shape))
        mask = torch.rand(shape, generator=generator) < 0.5
        masks.append(mask)
        supply = int(mask.sum()) if exact_source else count
        sources.append(torch.arange(1.0, supply + 1.0) + 100.0 * position)
    return bases, masks, sources


class TestMaskedScatter:

    @pytest.mark.parametrize(
        ("case_id", "shapes", "metadata"),
        MASKED_CASES,
        ids=[case[0] for case in MASKED_CASES],
    )
    def test_matches_dense_reference(self, case_id, shapes, metadata):
        bases, masks, sources = build_masked_case(shapes, exact_source=True)
        source_metadata = {key: value for key, value in metadata.items() if key != "ragged_dims"}

        output = torch.masked_scatter(
            NT(bases, **metadata),
            NT(masks, **metadata),
            NT(sources, **source_metadata),
        )

        assert_matches(
            output,
            [torch.masked_scatter(base, mask, source) for base, mask, source in zip(bases, masks, sources)],
        )

    def test_oversized_source_matches_dense_reference(self):
        shapes = [(3, 4), (2, 4)]
        bases, masks, sources = build_masked_case(shapes, exact_source=False)

        output = torch.masked_scatter(NT(bases), NT(masks), NT(sources))

        assert_matches(
            output,
            [torch.masked_scatter(base, mask, source) for base, mask, source in zip(bases, masks, sources)],
        )

    def test_fullgraph(self):
        bases = [torch.zeros(3, 4), torch.zeros(2, 4)]
        masks = [
            torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=torch.bool),
            torch.tensor([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=torch.bool),
        ]
        sources = [torch.arange(1.0, 7.0), torch.arange(11.0, 15.0)]
        compiled = torch.compile(
            lambda input_, mask_, source_: torch.masked_scatter(input_, mask_, source_),
            backend="aot_eager",
            fullgraph=True,
        )

        output = compiled(
            NT(bases, ragged_dims=(0,)),
            NT(masks, ragged_dims=(0,)),
            NT(sources, ragged_dims=(0,)),
        )

        assert_matches(
            output,
            [torch.masked_scatter(base, mask, source) for base, mask, source in zip(bases, masks, sources)],
        )

    def test_short_source_raises(self):
        bases = [torch.zeros(3), torch.zeros(2)]
        masks = [torch.tensor([True, True, True]), torch.tensor([True, False])]
        sources = [torch.tensor([1.0]), torch.tensor([2.0])]

        with pytest.raises(RuntimeError, match="source < number of ones in mask"):
            torch.masked_scatter(NT(bases), NT(masks), NT(sources))


class TestMaskedScatterAutograd:

    @staticmethod
    def build(exact_supply):
        torch.manual_seed(0)
        leaves = [torch.randn(3, 4, requires_grad=True), torch.randn(5, 4, requires_grad=True)]
        elements = [leaf * 1.0 for leaf in leaves]
        masks = [leaf.detach() > 0 for leaf in leaves]
        counts = [int(mask.sum()) for mask in masks] if exact_supply else [mask.numel() for mask in masks]
        supply = torch.randn(sum(counts), requires_grad=True)
        source_elements = [supply[: counts[0]], supply[counts[0] :]]
        return (
            leaves,
            supply,
            elements,
            masks,
            source_elements,
            NT(elements, ragged_dims=(0,)),
            NT(masks, ragged_dims=(0,)),
            NT(source_elements, ragged_dims=(0,)),
        )

    @pytest.mark.parametrize("exact_supply", [False, True], ids=["surplus", "exact"])
    def test_gradient_matches_per_element_reference(self, exact_supply):
        leaves, supply, _, _, _, nested, mask, source = self.build(exact_supply)
        torch.masked_scatter(nested, mask, source).concat.sum().backward()
        actual = [leaf.grad.clone() for leaf in leaves] + [supply.grad.clone()]

        leaves, supply, elements, masks, sources, _, _, _ = self.build(exact_supply)
        sum(
            torch.masked_scatter(element, mask, source).sum() for element, mask, source in zip(elements, masks, sources)
        ).backward()
        expected = [leaf.grad.clone() for leaf in leaves] + [supply.grad.clone()]

        for actual_gradient, expected_gradient in zip(actual, expected):
            torch.testing.assert_close(actual_gradient, expected_gradient)
