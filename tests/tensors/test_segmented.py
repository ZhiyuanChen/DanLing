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
from danling.tensors.segmented import (
    align_rows,
    segmented_arg_scan,
    segmented_cdist,
    segmented_cdist_backward,
    segmented_cumprod,
    segmented_cumprod_backward,
    segmented_scan,
    segmented_sort_perm,
)


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


class TestSegmentedCumprodComplex:

    @pytest.mark.parametrize("conjugate", [False, True])
    def test_compiled_complex_gradient_matches_native(self, device, conjugate):
        values = torch.randn(5, 2, dtype=torch.complex128, device=device, requires_grad=True)
        source = values.conj() if conjugate else values
        parts = source.split((3, 0, 2))
        nested = NestedTensor(parts, ragged_dims=(0,))
        compiled = torch.compile(lambda data: data.cumprod(dim=1).concat, backend="aot_eager", fullgraph=True)
        actual = compiled(nested)
        expected = torch.cat([part.cumprod(0) for part in parts])
        cotangent = torch.randn_like(actual)
        actual_gradient = torch.autograd.grad(actual, values, cotangent, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, values, cotangent)[0]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_gradient, expected_gradient)


class TestSegmentedHigherDerivatives:

    @pytest.mark.parametrize("backend", [None, "aot_eager"])
    @pytest.mark.parametrize("case", ["zeros", "complex", "nonfinite", "empty"])
    def test_cumprod_vjp_derivative_matches_native(self, device, backend, case):
        dtype = torch.complex128 if case == "complex" else torch.float64
        data = [0.0, 2.0, 3.0, 4.0, 0.0]
        if case == "nonfinite":
            data[1] = float("inf")
        values = torch.tensor(data, dtype=dtype, device=device, requires_grad=True)
        if case == "complex":
            values = (values + 0.5j).detach().requires_grad_()
        if case == "empty":
            values = values[:0].detach().requires_grad_()
        lengths = (3, 0, 2) if case != "empty" else (0, 0, 0)
        offsets = torch.tensor([0, lengths[0], lengths[0], sum(lengths)])
        output = torch.cat([part.cumprod(0) for part in values.split(lengths)]).detach().requires_grad_()
        cotangent = torch.randn_like(values, requires_grad=True)

        def run(grad, data, result):
            return segmented_cumprod_backward(grad, data, offsets, result)

        if backend is not None:
            run = torch.compile(run, backend=backend, fullgraph=True, dynamic=True)
        actual = run(cotangent, values, output)
        with torch.no_grad():
            expected_value = torch.cat(
                [
                    torch.ops.aten.cumprod_backward.default(grad, data, 0, result)
                    for grad, data, result in zip(
                        cotangent.split(lengths), values.split(lengths), output.split(lengths)
                    )
                ]
            )
        expected = torch.cat(
            [
                torch.ops.aten.cumprod_backward.default(grad, data, 0, result)
                for grad, data, result in zip(cotangent.split(lengths), values.split(lengths), output.split(lengths))
            ]
        )
        direction = torch.randn_like(values)
        actual_gradients = torch.autograd.grad(actual, (cotangent, values, output), direction, allow_unused=True)
        expected_gradients = torch.autograd.grad(expected, (cotangent, values, output), direction, allow_unused=True)

        torch.testing.assert_close(actual, expected_value, equal_nan=True)
        for argument, actual_gradient, expected_gradient in zip(
            (cotangent, values, output), actual_gradients, expected_gradients
        ):
            if expected_gradient is None:
                expected_gradient = torch.zeros_like(argument)
            torch.testing.assert_close(actual_gradient, expected_gradient, equal_nan=True)

    @pytest.mark.parametrize("backend", [None, "aot_eager"])
    def test_euclidean_cdist_vjp_passes_gradcheck(self, device, backend):
        left = torch.randn(3, 2, dtype=torch.float64, device=device, requires_grad=True)
        right = torch.randn(3, 2, dtype=torch.float64, device=device, requires_grad=True)
        row_lengths, column_lengths = torch.tensor([2, 1]), torch.tensor([1, 2])
        distances = (torch.rand(4, dtype=torch.float64, device=device) + 1).requires_grad_()
        cotangent = torch.randn_like(distances, requires_grad=True)

        def run(grad, lhs, rhs, result):
            return segmented_cdist_backward(grad, lhs, rhs, row_lengths, column_lengths, result, 2.0, 1)

        if backend is not None:
            run = torch.compile(run, backend=backend, fullgraph=True, dynamic=True)
        assert torch.autograd.gradcheck(run, (cotangent, left, right, distances), fast_mode=True)

    @pytest.mark.parametrize("operation", ["cumprod", "cdist"])
    def test_compiled_forward_preserves_native_third_derivatives(self, device, operation):
        values = torch.tensor([0.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64, device=device, requires_grad=True)
        reference = values.detach().clone().requires_grad_()
        offsets = torch.tensor([0, 3, 5])
        if operation == "cumprod":

            def run(data):
                return segmented_cumprod(data, offsets)

            expected = torch.cat([part.cumprod(0) for part in reference.split((3, 2))])
        else:
            right = torch.randn(3, 2, dtype=torch.float64, device=device)
            row_lengths, column_lengths = torch.tensor([3, 2]), torch.tensor([1, 2])

            def run(data):
                return segmented_cdist(data[:, None].expand(-1, 2), right, row_lengths, column_lengths, 2.0, 1)

            expected = torch.cat(
                [
                    torch.cdist(lhs[:, None].expand(-1, 2), rhs, compute_mode="use_mm_for_euclid_dist").flatten()
                    for lhs, rhs in zip(reference.split((3, 2)), right.split((1, 2)))
                ]
            )
        compiled = torch.compile(run, backend="eager", fullgraph=True, dynamic=True)
        actual = compiled(values)
        torch.testing.assert_close(actual, expected)
        for _ in range(3):
            actual = torch.autograd.grad(actual.sum(), values, create_graph=True)[0]
            expected = torch.autograd.grad(expected.sum(), reference, create_graph=True)[0]
            torch.testing.assert_close(actual, expected)
