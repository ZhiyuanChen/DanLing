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

r"""User-visible parity tests for packed segmented primitives."""

import pytest
import torch

from danling.tensors import NestedTensor
from danling.tensors.segmented import align_rows, segmented_arg_scan, segmented_scan, segmented_sort_perm


def lengths_case(shape_fn):
    return NestedTensor([torch.randn(*shape_fn(length)) for length in (3, 5, 2, 1)])


class TestSegmentedSortPerm:

    @pytest.mark.parametrize("descending", [False, True])
    def test_matches_per_element_argsort(self, descending):
        nested = lengths_case(lambda length: (length,))
        offsets = nested.packed_offsets()

        permutation, local_indices = segmented_sort_perm(
            nested.concat,
            offsets,
            nested.packed_batch_indices(),
            descending=descending,
        )

        expected_indices = torch.cat([torch.argsort(element, stable=True, descending=descending) for element in nested])
        expected_values = torch.cat(
            [torch.sort(element, stable=True, descending=descending).values for element in nested]
        )
        assert torch.equal(local_indices, expected_indices)
        torch.testing.assert_close(nested.concat[permutation], expected_values)

    def test_sorts_each_static_tail_column(self):
        nested = lengths_case(lambda length: (length, 4))
        _, local_indices = segmented_sort_perm(
            nested.concat,
            nested.packed_offsets(),
            nested.packed_batch_indices(),
        )
        expected = torch.cat([torch.argsort(element, dim=0, stable=True) for element in nested])
        assert torch.equal(local_indices, expected)

    def test_is_stable_on_ties(self):
        nested = NestedTensor([torch.zeros(3), torch.zeros(4)])
        _, local_indices = segmented_sort_perm(
            nested.concat,
            nested.packed_offsets(),
            nested.packed_batch_indices(),
        )
        assert torch.equal(local_indices, torch.cat([torch.arange(len(element)) for element in nested]))


class TestAlignRows:

    def test_broadcasts_by_row_across_a_static_tail(self):
        rows = torch.arange(5)
        aligned = align_rows(rows, torch.randn(5, 5, 2))

        assert aligned.shape == (5, 5, 2)
        assert torch.equal(aligned, rows.view(5, 1, 1).expand(5, 5, 2))
        assert torch.equal(align_rows(rows, torch.randn(5)), rows)


class TestSegmentedScan:

    @pytest.mark.parametrize(
        ("combine", "reference"),
        [
            (torch.mul, lambda value: torch.cumprod(value, 0)),
            (torch.logaddexp, lambda value: torch.logcumsumexp(value, 0)),
        ],
        ids=["cumprod", "logcumsumexp"],
    )
    def test_matches_per_element_scan(self, combine, reference):
        nested = lengths_case(lambda length: (length, 3))

        output = segmented_scan(nested.concat, nested.packed_batch_indices(), combine)

        torch.testing.assert_close(output, torch.cat([reference(element) for element in nested]))

    @pytest.mark.parametrize("largest", [False, True], ids=["cummin", "cummax"])
    def test_arg_scan_matches_values_and_indices(self, largest):
        nested = lengths_case(lambda length: (length, 4))

        values, indices = segmented_arg_scan(
            nested.concat,
            nested.packed_batch_indices(),
            nested.packed_local_indices(),
            largest=largest,
        )

        references = [(torch.cummax(element, 0) if largest else torch.cummin(element, 0)) for element in nested]
        torch.testing.assert_close(values, torch.cat([reference.values for reference in references]))
        assert torch.equal(indices, torch.cat([reference.indices for reference in references]))

    @pytest.mark.parametrize("largest", [False, True], ids=["cummin", "cummax"])
    def test_arg_scan_matches_ties_and_nan_semantics(self, largest):
        nested = NestedTensor(
            [
                torch.tensor([0.0, 0.0, float("nan"), 2.0]),
                torch.tensor([3.0, 3.0, 1.0]),
            ]
        )

        values, indices = segmented_arg_scan(
            nested.concat,
            nested.packed_batch_indices(),
            nested.packed_local_indices(),
            largest=largest,
        )

        references = [(torch.cummax(element, 0) if largest else torch.cummin(element, 0)) for element in nested]
        torch.testing.assert_close(
            values,
            torch.cat([reference.values for reference in references]),
            equal_nan=True,
        )
        assert torch.equal(indices, torch.cat([reference.indices for reference in references]))

    def test_empty_input(self):
        empty = torch.zeros(0, 3)
        batch = torch.zeros(0, dtype=torch.long)

        scanned = segmented_scan(empty, batch, torch.mul)
        values, indices = segmented_arg_scan(empty, batch, batch, largest=True)

        assert scanned.shape == (0, 3)
        assert values.shape == (0, 3)
        assert indices.shape == (0, 3)
