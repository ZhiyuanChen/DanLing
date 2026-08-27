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

from danling.tensors import NestedTensor


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










