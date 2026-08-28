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
from danling.tensors.segmented import segmented_sort_perm


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




