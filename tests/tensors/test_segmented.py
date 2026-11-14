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
            nested.values(),
            offsets,
            nested.packed_batch_indices(),
            descending=descending,
        )

        expected_indices = torch.cat([torch.argsort(element, stable=True, descending=descending) for element in nested])
        expected_values = torch.cat(
            [torch.sort(element, stable=True, descending=descending).values for element in nested]
        )
        assert torch.equal(local_indices, expected_indices)
        torch.testing.assert_close(nested.values()[permutation], expected_values)

    def test_sorts_each_static_tail_column(self):
        nested = lengths_case(lambda length: (length, 4))
        _, local_indices = segmented_sort_perm(
            nested.values(),
            nested.packed_offsets(),
            nested.packed_batch_indices(),
        )
        expected = torch.cat([torch.argsort(element, dim=0, stable=True) for element in nested])
        assert torch.equal(local_indices, expected)

    def test_is_stable_on_ties(self):
        nested = NestedTensor([torch.zeros(3), torch.zeros(4)])
        _, local_indices = segmented_sort_perm(
            nested.values(),
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

        output = segmented_scan(nested.values(), nested.packed_batch_indices(), combine)

        torch.testing.assert_close(output, torch.cat([reference(element) for element in nested]))

    @pytest.mark.parametrize("largest", [False, True], ids=["cummin", "cummax"])
    def test_arg_scan_matches_values_and_indices(self, largest):
        nested = lengths_case(lambda length: (length, 4))

        values, indices = segmented_arg_scan(
            nested.values(),
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
            nested.values(),
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


def _cumprod_transform(scan, transform):
    def loss(values, offsets):
        return scan(values, offsets).real.sum()

    gradient = torch.func.grad(loss)

    def tangent(values, offsets):
        return torch.func.jvp(lambda data: scan(data, offsets), (values,), (torch.ones_like(values),))[1]

    if transform == "grad":
        return gradient
    if transform == "jvp":
        return tangent
    if transform == "grad_grad":
        return torch.func.grad(lambda values, offsets: gradient(values, offsets).real.sum())
    if transform == "grad_jvp":
        return torch.func.grad(lambda values, offsets: tangent(values, offsets).real.sum())
    parent = gradient if transform == "jvp_grad" else tangent
    return lambda values, offsets: torch.func.jvp(
        lambda data: parent(data, offsets), (values,), (torch.ones_like(values),)
    )[1]


def _assert_cumprod_transform_close(actual, expected):
    if actual.is_complex():
        actual = torch.view_as_real(actual.resolve_conj())
        expected = torch.view_as_real(expected.resolve_conj())
    torch.testing.assert_close(actual, expected, equal_nan=True)


class TestSegmentedCumprodComplex:

    def test_nested_vmap_jvp(self, device):
        values = torch.arange(3.0, device=device).expand(2, 2, 3).clone()
        offsets = torch.tensor([0, 1, 3])

        def native(data):
            return torch.cat([part.cumprod(0) for part in data.split((1, 2))])

        def tangent(scan, data):
            return torch.func.jvp(scan, (data,), (torch.ones_like(data),))[1]

        actual = torch.vmap(torch.vmap(lambda data: tangent(lambda value: segmented_cumprod(value, offsets), data)))(
            values
        )
        expected = torch.vmap(torch.vmap(lambda data: tangent(native, data)))(values)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    @pytest.mark.parametrize("gradient", [False, True])
    def test_compiled_native_forward(self, device, backend, gradient):
        values = torch.arange(3.0, device=device, dtype=torch.float64)
        offsets = torch.tensor([0, 1, 3])

        def native(data, partition):
            del partition
            return torch.cat([part.cumprod(0) for part in data.split((1, 2))])

        actual_run = (
            torch.func.grad(lambda data, partition: segmented_cumprod(data, partition).sum())
            if gradient
            else segmented_cumprod
        )
        expected_run = torch.func.grad(lambda data, partition: native(data, partition).sum()) if gradient else native
        actual_run = torch.compile(actual_run, backend=backend, fullgraph=True)
        native_run = torch.compile(expected_run, backend=backend, fullgraph=True)
        forward_ad = torch.autograd.forward_ad
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(values, torch.ones_like(values))
            try:
                native_run(dual, offsets)
            except torch._dynamo.exc.Unsupported as error:
                if "Attempted to wrap a dual tensor input" not in str(error):
                    raise
                pytest.skip("torch.compile rejects native forward-AD dual inputs")
            # Input support is separate from the derivative oracle: older Inductor
            # accepts duals but drops native tangents, so compare with eager AD.
            actual = forward_ad.unpack_dual(actual_run(dual, offsets))
            expected = forward_ad.unpack_dual(expected_run(dual, offsets))
            torch.testing.assert_close(actual.primal, expected.primal)
            assert actual.tangent is not None and expected.tangent is not None
            torch.testing.assert_close(actual.tangent, expected.tangent)

    @pytest.mark.parametrize("lengths", [(1, 1, 1), (0, 0, 0), (1, 2)])
    def test_native_forward_over_reverse(self, device, lengths):
        values = torch.arange(sum(lengths), device=device, dtype=torch.float64)
        source = NestedTensor(values.split(lengths))
        forward_ad = torch.autograd.forward_ad
        with forward_ad.dual_level():
            dense_dual = forward_ad.make_dual(values, torch.ones_like(values))
            expected = torch.func.grad(lambda data: torch.cat([part.cumprod(0) for part in data.split(lengths)]).sum())(
                dense_dual
            )
            dual = forward_ad.make_dual(source, torch.ones_like(source))
            actual = torch.func.grad(lambda data: data.cumprod(1).sum())(dual)
            for result, native in ((actual, expected), (actual * float("inf"), expected * float("inf"))):
                primal, tangent = forward_ad.unpack_dual(result)
                expected_primal, expected_tangent = forward_ad.unpack_dual(native)
                torch.testing.assert_close(primal.values(), expected_primal, equal_nan=True)
                assert (tangent is None) == (expected_tangent is None)
                if tangent is not None:
                    torch.testing.assert_close(tangent.values(), expected_tangent, equal_nan=True)

    @pytest.mark.parametrize("forward_api", ["func", "native"])
    @pytest.mark.parametrize("create_graph", [False, True])
    def test_forward_reverse_context(self, device, forward_api, create_graph):
        values = torch.tensor([0.0, 2.0, 3.0], device=device, dtype=torch.float64, requires_grad=True)
        reference = values.detach().clone().requires_grad_()
        if forward_api == "func":
            offsets = torch.tensor([0, 3])
            actual = torch.func.jvp(
                lambda data: segmented_cumprod(data, offsets), (values,), (torch.ones_like(values),)
            )
            expected = torch.func.jvp(lambda data: data.cumprod(0), (reference,), (torch.ones_like(reference),))
        else:
            forward_ad = torch.autograd.forward_ad
            source = NestedTensor([values], ragged_dims=(0,))
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(source, torch.ones_like(source))
                primal, tangent = forward_ad.unpack_dual(dual.cumprod(1))
            actual = (primal.values(), tangent.values())
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(reference, torch.ones_like(reference))
                expected = forward_ad.unpack_dual(dual.cumprod(0))
        for result, native in zip(actual, expected):
            _assert_cumprod_transform_close(result, native)
        _assert_cumprod_transform_close(
            torch.autograd.grad(actual[1].sum(), values, create_graph=create_graph)[0],
            torch.autograd.grad(expected[1].sum(), reference, create_graph=create_graph)[0],
        )

    @pytest.mark.parametrize("event", ["replayed_prefix", "new_backward"])
    def test_anomaly_context(self, device, event):
        results, targets = [], []
        for nested in (False, True):
            values = torch.tensor(
                [1.0, 2.0, 3.0] if event == "replayed_prefix" else [0.0, 2.0, 3.0],
                device=device,
                dtype=torch.float64,
                requires_grad=event == "new_backward",
            )
            direction = torch.full_like(
                values, float("nan") if event == "replayed_prefix" else 1.0, requires_grad=event == "replayed_prefix"
            )
            offsets = torch.tensor([0, 3])

            def scan(data, offsets=offsets, nested=nested):
                return segmented_cumprod(data, offsets) if nested else data.cumprod(0)

            def tangent(data, scan=scan, direction=direction):
                return torch.func.jvp(scan, (data,), (direction,))[1]

            with torch.autograd.set_detect_anomaly(False):
                result = (
                    torch.func.grad(lambda data: tangent(data).sum())(values)
                    if event == "replayed_prefix"
                    else tangent(values)
                )
            results.append(result)
            targets.append(direction if event == "replayed_prefix" else values)
        with torch.autograd.set_detect_anomaly(True):
            if event == "replayed_prefix":
                torch.testing.assert_close(
                    torch.autograd.grad(results[0].sum(), targets[0]), torch.autograd.grad(results[1].sum(), targets[1])
                )
            else:
                for result, target in zip(results, targets):
                    with pytest.raises(RuntimeError, match="returned nan"):
                        torch.autograd.grad(result.sum(), target)

    @pytest.mark.parametrize(
        "transform,case,batch_size,values_dim,offsets_dim",
        [
            ("grad", "finite", 2, None, 0),
            ("jvp", "zeros", 2, 0, 0),
            ("grad_grad", "complex", 2, -1, -1),
            ("grad_jvp", "nonfinite", 2, None, 0),
            ("jvp_grad", "nonfinite", 2, 0, 0),
            ("jvp_jvp", "complex", 2, -1, -1),
            ("grad", "empty", 2, None, 0),
            ("jvp_grad", "zeros", 0, 0, 0),
            ("grad_jvp", "complex", 0, None, 0),
            ("jvp_jvp", "finite", 0, -1, -1),
        ],
    )
    def test_mapped_offsets(self, device, transform, case, batch_size, values_dim, offsets_dim):
        dtype = torch.complex128 if case == "complex" else torch.float64
        base = torch.arange(1.0, 6.0, device=device, dtype=torch.float64).to(dtype).unsqueeze(-1).expand(-1, 2).clone()
        if case == "zeros":
            base[[0, 3]] = 0
        elif case == "nonfinite":
            base[1, 0] = float("nan")
            base[-1, 1] = float("inf")
        elif case == "complex":
            base = base + 0.3j
        elif case == "empty":
            base = base[:0]
        batch = torch.stack((base, base * 2))[:batch_size]
        lengths = ((0, 0, 0),) * 2 if case == "empty" else ((2, 0, 3), (3, 0, 2))
        splits = torch.tensor([[0, sizes[0], sizes[0], sum(sizes)] for sizes in lengths])[:batch_size]
        values = base if values_dim is None else batch.movedim(0, values_dim)
        offsets = splits.movedim(0, offsets_dim)
        actual = torch.vmap(_cumprod_transform(segmented_cumprod, transform), in_dims=(values_dim, offsets_dim))(
            values, offsets
        )
        references = []
        for index in range(batch_size):

            def native(data, _offsets, sizes=lengths[index]):
                return torch.cat([part.cumprod(0) for part in data.split(sizes)])

            references.append(_cumprod_transform(native, transform)(base if values_dim is None else batch[index], None))
        expected = torch.stack(references) if references else base.new_empty((0, *base.shape))
        _assert_cumprod_transform_close(actual, expected)
        if batch_size and case == "finite" and transform == "grad" and values_dim is None:
            assert not torch.equal(expected[0], expected[1])

    @pytest.mark.parametrize(
        "backend,transform,dtype",
        [
            ("aot_eager", "grad", torch.float64),
            ("inductor", "jvp_grad", torch.float64),
            ("aot_eager", "jvp", torch.complex128),
            ("inductor", "grad_jvp", torch.complex128),
        ],
    )
    def test_mapped_offsets_compile(self, device, backend, transform, dtype):
        torch.compiler.reset()
        operation = torch.vmap(_cumprod_transform(segmented_cumprod, transform))
        run = torch.compile(operation, backend=backend, fullgraph=True)
        base = torch.tensor([1.0, 0.0, 3.0, 4.0, 5.0], device=device, dtype=dtype)
        if dtype.is_complex:
            base = base + 0.3j
        values = base.expand(2, 2, -1).clone().transpose(1, 2)
        for nonfinite in (False,) if dtype.is_complex else (False, True):
            offsets = torch.tensor([[0, 2, 2, 5], [0, 3, 3, 5]])
            if nonfinite:
                values[0, 1] = float("nan")
                values[1, -1] = -float("inf")
                offsets = offsets.flip(0)
            actual = run(values, offsets)
            expected = []
            for index, sizes in enumerate(((3, 0, 2), (2, 0, 3)) if nonfinite else ((2, 0, 3), (3, 0, 2))):

                def native(data, _offsets, sizes=sizes):
                    return torch.cat([part.cumprod(0) for part in data.split(sizes)])

                expected.append(_cumprod_transform(native, transform)(values[index], None))
            _assert_cumprod_transform_close(actual, torch.stack(expected))

    @pytest.mark.parametrize(
        "transform,mutated", [("grad", "values"), ("jvp", "offsets"), ("jvp", "direction"), ("grad_jvp", "direction")]
    )
    def test_mapped_offsets_versions(self, device, transform, mutated):
        values = torch.randn(2, 5, 2, device=device, dtype=torch.float64, requires_grad=mutated != "direction")
        offsets = torch.tensor([[0, 2, 2, 5], [0, 3, 3, 5]])
        direction = torch.ones_like(values, requires_grad=mutated == "direction")
        if transform == "grad_jvp":
            values = (
                torch.tensor(
                    [[1.0, 2.0, 1e-308, 1e-308, 1e308], [1e-308, 1e-308, 1e308, 1.0, 2.0]],
                    device=device,
                    dtype=torch.float64,
                )
                .unsqueeze(-1)
                .expand(-1, -1, 2)
                .clone()
            )
            direction = torch.full_like(values, 1e-308, requires_grad=True)

        def directional(scan, data, vector):
            def tangent(data):
                return torch.func.jvp(scan, (data,), (vector,))[1]

            return torch.func.grad(lambda data: tangent(data).sum())(data) if transform == "grad_jvp" else tangent(data)

        if transform != "grad":

            def tangent(data, splits, vector):
                return directional(lambda data: segmented_cumprod(data, splits), data, vector)

            derivative = torch.vmap(tangent)(values, offsets, direction)
        else:
            derivative = torch.vmap(_cumprod_transform(segmented_cumprod, transform))(values, offsets)
        if mutated == "direction":
            native = []
            for data, vector, sizes in zip(values, direction, ((2, 0, 3), (3, 0, 2))):

                def scan(data, sizes=sizes):
                    return torch.cat([part.cumprod(0) for part in data.split(sizes)])

                native.append(directional(scan, data, vector))
            expected = torch.stack(native)
        with torch.no_grad():
            {"values": values, "offsets": offsets, "direction": direction}[mutated].add_(1)
        if mutated == "direction":
            deterministic = torch.are_deterministic_algorithms_enabled()
            warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
            fill = torch.utils.deterministic.fill_uninitialized_memory
            try:
                torch.use_deterministic_algorithms(True)
                torch.utils.deterministic.fill_uninitialized_memory = True
                with torch.autograd.set_detect_anomaly(True):
                    probe = torch.ones_like(derivative)
                    if transform == "grad_jvp":
                        probe.zero_()
                        probe[0, 2] = probe[1, 0] = 1
                    torch.testing.assert_close(
                        torch.autograd.grad(derivative, direction, probe),
                        torch.autograd.grad(expected, direction, probe),
                    )
            finally:
                torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
                torch.utils.deterministic.fill_uninitialized_memory = fill
            return
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            torch.autograd.grad(derivative.sum(), values)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    @pytest.mark.parametrize("values_dim,offsets_dim,out_dim", [(0, 0, 0), (-1, -1, -1), (None, 0, 0)])
    def test_empty_mapped_offsets(self, device, dtype, values_dim, offsets_dim, out_dim):
        base = torch.randn((5, 2) if values_dim is None else (0, 5, 2), device=device, dtype=dtype)
        values = (base if values_dim is None else base.movedim(0, values_dim)).requires_grad_()
        offsets = torch.tensor([0, 2, 5]).expand(0, -1).movedim(0, offsets_dim)

        def actual(data):
            return torch.vmap(segmented_cumprod, in_dims=(values_dim, offsets_dim), out_dims=out_dim)(data, offsets)

        def native(data):
            def scan(value, _offsets):
                return torch.cat([part.cumprod(0) for part in value.split((2, 3))])

            return torch.vmap(scan, in_dims=(values_dim, offsets_dim), out_dims=out_dim)(data, offsets)

        torch.testing.assert_close(actual(values), native(values))
        torch.testing.assert_close(
            torch.autograd.grad(actual(values).real.sum(), values),
            torch.autograd.grad(native(values).real.sum(), values),
        )
        torch.testing.assert_close(
            torch.func.jvp(actual, (values,), (torch.ones_like(values),)),
            torch.func.jvp(native, (values,), (torch.ones_like(values),)),
        )

    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    def test_empty_mapped_offsets_compiled(self, device, backend):
        torch.compiler.reset()
        values = torch.empty(0, 5, 2, device=device, dtype=torch.float64, requires_grad=True)
        offsets = torch.tensor([0, 2, 5]).expand(0, -1)
        run = torch.compile(torch.vmap(segmented_cumprod), backend=backend, fullgraph=True)
        output = run(values, offsets)
        assert output.shape == values.shape
        torch.testing.assert_close(torch.autograd.grad(output.sum(), values)[0], torch.zeros_like(values))

    @pytest.mark.parametrize("case", ["zeros", "complex", "empty"])
    def test_transforms(self, device, case):
        dtype = torch.complex128 if case == "complex" else torch.float64
        values = torch.tensor([0.0, 2.0, 3.0, 4.0, 0.0], device=device, dtype=dtype)
        if case == "complex":
            values = values + 0.5j
        lengths = (3, 0, 2) if case != "empty" else (0, 0, 0)
        if case == "empty":
            values = values[:0]
        offsets = torch.tensor([0, lengths[0], lengths[0], sum(lengths)])
        direction = torch.randn_like(values)

        def actual(data):
            return segmented_cumprod(data, offsets)

        def reference(data):
            return torch.cat([part.cumprod(0) for part in data.split(lengths)])

        torch.testing.assert_close(
            torch.func.jvp(actual, (values,), (direction,)), torch.func.jvp(reference, (values,), (direction,))
        )
        torch.testing.assert_close(
            torch.func.grad(lambda data: actual(data).abs().square().sum())(values),
            torch.func.grad(lambda data: reference(data).abs().square().sum())(values),
        )
        batch = torch.stack((values, values + 1))
        torch.testing.assert_close(torch.vmap(actual)(batch), torch.stack([reference(part) for part in batch]))
        second_lengths = (2, 0, 3) if case != "empty" else lengths
        split_batch = torch.stack(
            (offsets, torch.tensor([0, second_lengths[0], second_lengths[0], sum(second_lengths)]))
        )
        expected = torch.stack(
            [
                torch.cat([part.cumprod(0) for part in data.split(sizes)])
                for data, sizes in zip(batch, (lengths, second_lengths))
            ]
        )
        torch.testing.assert_close(torch.vmap(segmented_cumprod)(batch, split_batch), expected)

    @pytest.mark.parametrize("conjugate", [False, True])
    def test_compiled_complex_gradient_matches_native(self, device, conjugate):
        values = torch.randn(5, 2, dtype=torch.complex128, device=device, requires_grad=True)
        source = values.conj() if conjugate else values
        parts = source.split((3, 0, 2))
        nested = NestedTensor(parts, ragged_dims=(0,))
        compiled = torch.compile(lambda data: data.cumprod(dim=1).values(), backend="aot_eager", fullgraph=True)
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
