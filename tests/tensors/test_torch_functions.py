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

import itertools
import math
import operator
from functools import partial
from itertools import accumulate

import pytest
import torch
from torch.nn import functional as F

from danling.tensors import NestedTensor
from tests.tensors.utils import assert_close, low_precision_cuda_tolerances, nested_rand, ragged_shapes

NT = NestedTensor


class TestDiff:
    @pytest.mark.parametrize("order", [0, 1, 2, 6])
    @pytest.mark.parametrize("boundaries", [False, True])
    def test_ragged_values_and_gradients(self, order, boundaries):
        parts = [torch.randn(length, dtype=torch.float64, requires_grad=True) for length in (3, 5)]
        refs = [part.detach().clone().requires_grad_() for part in parts]
        before = torch.randn(2, 1, dtype=torch.float64, requires_grad=True) if boundaries else None
        after = torch.randn(2, 1, dtype=torch.float64, requires_grad=True) if boundaries else None
        ref_before = before.detach().clone().requires_grad_() if boundaries else None
        ref_after = after.detach().clone().requires_grad_() if boundaries else None
        result = torch.diff(NT(parts, ragged_dims=(0,)), n=order, prepend=before, append=after)
        expected = [
            torch.diff(
                part,
                n=order,
                prepend=ref_before[i] if boundaries else None,
                append=ref_after[i] if boundaries else None,
            )
            for i, part in enumerate(refs)
        ]
        torch.testing.assert_close(result.unbind(), expected)
        variables = (*parts, before, after) if boundaries and order else tuple(parts)
        references = (*refs, ref_before, ref_after) if boundaries and order else tuple(refs)
        actual_grad = torch.autograd.grad(sum(part.square().sum() for part in result.unbind()), variables)
        expected_grad = torch.autograd.grad(sum(part.square().sum() for part in expected), references)
        torch.testing.assert_close(actual_grad, expected_grad)

    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    def test_boolean_and_static_axis_compilation(self, device, backend):
        torch.compiler.reset()
        parts = [
            torch.tensor([[True, False, True], [False, False, True]], device=device),
            torch.tensor([[True, True, False]], device=device),
        ]
        source = NT(parts, ragged_dims=(0,))
        run = torch.compile(lambda value: value.diff(dim=-1), backend=backend, dynamic=True, fullgraph=True)
        torch.testing.assert_close(run(source).unbind(), [part.diff(dim=-1) for part in parts])

    def test_nonleading_batch_and_argument_errors(self):
        parts = [torch.randn(length, 3, dtype=torch.float64, requires_grad=True) for length in (3, 5)]
        source = NT(parts, ragged_dims=(0,), batch_first=False)
        result = torch.diff(input=source, dim=0)
        expected = [part.diff(dim=0) for part in parts]
        torch.testing.assert_close(result.unbind(dim=1), expected)
        actual_grad = torch.autograd.grad(sum(x.square().sum() for x in result.unbind(dim=1)), parts, retain_graph=True)
        expected_grad = torch.autograd.grad(sum(x.square().sum() for x in expected), parts)
        torch.testing.assert_close(actual_grad, expected_grad)
        unchanged = torch.diff(input=source, n=0, dim=100)
        torch.testing.assert_close(unchanged.unbind(dim=1), parts)
        with torch.no_grad():
            unchanged.add_(1)
        torch.testing.assert_close(source.unbind(dim=1), parts)
        for boundary in ("prepend", "append"):
            invalid = {boundary: torch.tensor(0.0, dtype=source.dtype)}
            with pytest.raises(IndexError):
                torch.diff(input=source, n=0, dim=100, **invalid)
            with pytest.raises(RuntimeError):
                torch.diff(input=source, n=0, dim=0, **invalid)
        with pytest.raises(IndexError):
            torch.diff(input=source, dim=-4)
        with pytest.raises(RuntimeError, match="order must be non-negative"):
            torch.diff(input=source, n=-1)
        torch.testing.assert_close(torch.diff(input=source, dim=0, out=None).unbind(dim=1), expected)
        with pytest.raises(RuntimeError, match="automatic differentiation"):
            torch.diff(input=source, dim=0, out=NT([torch.empty_like(part) for part in expected], batch_first=False))

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("order", [1, 6])
    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    def test_static_axis_gradients(self, device, batch_first, order, backend):
        torch.compiler.reset()
        parts = [torch.randn(length, 4, device=device, dtype=torch.float64, requires_grad=True) for length in (3, 5)]
        source = NT(parts, ragged_dims=(0,), batch_first=batch_first)
        run = torch.compile(
            lambda value: torch.diff(input=value, dim=-1, n=order), backend=backend, dynamic=True, fullgraph=True
        )
        output = run(source).unbind(dim=0 if batch_first else 1)
        expected = [part.diff(dim=-1, n=order) for part in parts]
        torch.testing.assert_close(output, expected)
        actual_grad = torch.autograd.grad(sum(x.square().sum() for x in output), parts, retain_graph=True)
        expected_grad = torch.autograd.grad(sum(x.square().sum() for x in expected), parts)
        torch.testing.assert_close(actual_grad, expected_grad)

    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    @pytest.mark.parametrize(
        "case",
        [
            "sequence",
            "batch-second",
            "permuted",
            "dense-boundaries",
            "nt-boundaries",
            "empty-features",
            "varying-features",
        ],
    )
    def test_dynamic_ragged(self, device, backend, case):
        torch.compiler.reset()
        multi = case in {"permuted", "dense-boundaries", "nt-boundaries"}
        batch_first = case != "batch-second"
        ragged = (1, 0) if case == "permuted" else (0, 1) if multi else (0,)
        physical_dim = 1 if case in {"permuted", "varying-features"} else 0
        dim = physical_dim + 1 if batch_first else 0
        order = 6 if case == "batch-second" else 2
        has_boundaries = case in {
            "batch-second",
            "dense-boundaries",
            "nt-boundaries",
            "empty-features",
            "varying-features",
        }

        def difference(value, before, after):
            return torch.diff(value, n=order, dim=dim, prepend=before, append=after)

        run = torch.compile(difference, backend=backend, dynamic=True, fullgraph=True)
        for lengths in ((3, 5), (0, 2)):
            columns = (0, 0) if case == "empty-features" else (2, 4) if multi else (3, 3)
            parts = [
                torch.randn(n, c, device=device, dtype=torch.float64, requires_grad=True)
                for n, c in zip(lengths, columns)
            ]
            references = [value.detach().clone().requires_grad_() for value in parts]
            before = after = None
            ref_before = ref_after = None
            if case == "varying-features":
                bp = [
                    torch.randn(n, p, device=device, dtype=torch.float64, requires_grad=True)
                    for n, p in zip(lengths, (1, 2))
                ]
                ref_before = [value.detach().clone().requires_grad_() for value in bp]
                before = NT(bp)
                after = torch.randn(2, max(lengths), 1, device=device, dtype=torch.float64, requires_grad=True)
                ra = after.detach().clone().requires_grad_()
                ref_after = [ra[i, :n] for i, n in enumerate(lengths)]
                variables, reference_variables = (*parts, *bp, after), (*references, *ref_before, ra)
            elif case == "nt-boundaries":
                bp = [
                    torch.randn(n, c, device=device, dtype=torch.float64, requires_grad=True)
                    for n, c in zip((1, 2), columns)
                ]
                ap = [torch.randn(1, c, device=device, dtype=torch.float64, requires_grad=True) for c in columns]
                ref_before = [value.detach().clone().requires_grad_() for value in bp]
                ref_after = [value.detach().clone().requires_grad_() for value in ap]
                before, after = NT(bp, ragged_dims=(1, 0)), NT(ap)
                variables = (*parts, *bp, *ap)
                reference_variables = (*references, *ref_before, *ref_after)
            elif has_boundaries:
                shape = (2, 1, max(columns)) if batch_first else (1, 2, max(columns))
                before = torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True)
                after = torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True)
                rb = before.detach().clone().requires_grad_()
                ra = after.detach().clone().requires_grad_()
                ref_before = [rb[i, :, :c] if batch_first else rb[:, i, :c] for i, c in enumerate(columns)]
                ref_after = [ra[i, :, :c] if batch_first else ra[:, i, :c] for i, c in enumerate(columns)]
                variables, reference_variables = (*parts, before, after), (*references, rb, ra)
            else:
                variables, reference_variables = tuple(parts), tuple(references)
            output = run(NT(parts, ragged_dims=ragged, batch_first=batch_first), before, after)
            expected = [
                torch.diff(
                    value,
                    n=order,
                    dim=physical_dim,
                    prepend=ref_before[i] if ref_before is not None else None,
                    append=ref_after[i] if ref_after is not None else None,
                )
                for i, value in enumerate(references)
            ]
            torch.testing.assert_close(output.unbind(0 if batch_first else 1), expected)
            torch.testing.assert_close(
                torch.autograd.grad(output.concat.square().sum(), variables),
                torch.autograd.grad(sum(value.square().sum() for value in expected), reference_variables),
            )

    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    @pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.float64, torch.complex64])
    def test_ragged_dtypes(self, device, backend, dtype):
        torch.compiler.reset()
        differentiable = dtype.is_floating_point or dtype.is_complex
        parts = [
            (
                torch.randn(n, 3, device=device, dtype=dtype, requires_grad=True)
                if differentiable
                else (torch.arange(n * 3, device=device).reshape(n, 3).square() * 37).to(dtype)
            )
            for n in (1, 4)
        ]
        if dtype == torch.float64:
            with torch.no_grad():
                parts[0].fill_(float("nan"))
        refs = [value.detach().clone().requires_grad_(differentiable) for value in parts]
        run = torch.compile(lambda value: value.diff(n=2, dim=1), backend=backend, dynamic=True, fullgraph=True)
        output = run(NT(parts, ragged_dims=(0,)))
        expected = [value.diff(n=2, dim=0) for value in refs]
        torch.testing.assert_close(output.unbind(), expected)
        if differentiable:
            torch.testing.assert_close(
                torch.autograd.grad(output.concat.abs().square().sum(), parts),
                torch.autograd.grad(sum(value.abs().square().sum() for value in expected), refs),
            )

    @pytest.mark.parametrize("backend", [None, "aot_eager", "inductor"])
    @pytest.mark.parametrize(
        "case", ["sequence", "multi", "batch-second", "uniform", "boundaries", "mixed", "zero-features", "empty"]
    )
    def test_batch(self, device, backend, case):
        torch.compiler.reset()
        batch_first = case != "batch-second"
        order = 2 if case in {"multi", "boundaries", "mixed", "empty"} else 1
        axis = 0 if batch_first else 1

        def difference(value, before, after):
            return value.diff(n=order, dim=axis, prepend=before, append=after)

        run = (
            difference if backend is None else torch.compile(difference, backend=backend, dynamic=True, fullgraph=True)
        )
        for lengths in ((2, 1, 3), (0, 1, 0, 1)):
            if case == "multi":
                shapes = [(2, 1, 3), (1, 3, 3), (2, 1, 3)] if len(lengths) == 3 else [(3, 1, 3), (1, 2, 3)] * 2
                ragged = (1, 0)
            else:
                if case in {"boundaries", "mixed"}:
                    lengths = (1, 3, 1) if len(lengths) == 3 else (1, 2, 1, 2)
                elif case == "uniform":
                    lengths = (3,) * len(lengths)
                elif case == "empty":
                    lengths = (3,)
                shapes = [(n, 0 if case == "zero-features" else 3) for n in lengths]
                ragged = () if case == "uniform" else (0,)
            dtype = torch.bfloat16 if case == "mixed" else torch.float64
            parts = [torch.randn(shape, device=device, dtype=dtype, requires_grad=True) for shape in shapes]
            refs = [part.detach().clone().requires_grad_() for part in parts]
            source = NT(parts, ragged_dims=ragged, batch_first=batch_first)
            variables, references = list(parts), list(refs)
            before = after = None
            expected = refs
            if case in {"boundaries", "mixed"}:
                prefix = torch.randn(max(lengths), 3, device=device, dtype=torch.float64, requires_grad=True)
                suffix = torch.randn(
                    1,
                    max(lengths),
                    3,
                    device=device,
                    dtype=torch.float32 if case == "mixed" else torch.float64,
                    requires_grad=True,
                )
                rb, ra = (value.detach().clone().requires_grad_() for value in (prefix, suffix))
                before, after = NT([prefix], ragged_dims=(0,)), suffix
                variables.extend((prefix, suffix))
                references.extend((rb, ra))
                expected = [value.to(torch.float64) for value in [rb, *refs, ra[0]]]
            if case == "empty":
                source, expected = source[:0], []
            for _ in range(order):
                expected = [right - left for left, right in zip(expected, expected[1:])]
            output = run(source, before, after)
            torch.testing.assert_close(output.unbind(axis), expected)
            loss = (
                sum(value.square().sum() for value in expected) if expected else sum(value[:0].sum() for value in refs)
            )
            actual_grad = torch.autograd.grad(output.concat.square().sum(), variables)
            expected_grad = torch.autograd.grad(loss, references)
            for actual, expected in zip(actual_grad, expected_grad):
                options = {"rtol": 0, "atol": 0} if actual.dtype == torch.bfloat16 else {}
                torch.testing.assert_close(actual, expected, **options)

    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_batch_errors(self, device, order):
        source = NT([torch.ones(n, device=device) for n in (2, 3)], ragged_dims=(0,))
        with pytest.raises(RuntimeError, match="broadcastable"):
            source.diff(n=order, dim=0)

    def test_positions(self, device):
        torch.compiler.reset()
        parts = [torch.arange(n, device=device) for n in (3, 5)]
        before = torch.full((2, 1), -1, device=device, dtype=torch.long)
        run = torch.compile(
            lambda value, prefix: torch.diff(value, dim=-1, prepend=prefix),
            backend="inductor",
            dynamic=True,
            fullgraph=True,
        )
        output = run(NT(parts, ragged_dims=(0,)), before)
        torch.testing.assert_close(output.unbind(), [torch.ones_like(value) for value in parts])

    @pytest.mark.parametrize(
        "layout", ["nested", "empty", "resize", "dense", "dense-empty", "dense-resize", "strided", "self", "rank"]
    )
    def test_out(self, device, layout):
        parts = [torch.randn(n, 4, device=device, dtype=torch.float64) for n in (3, 5)]
        source = NT(parts, ragged_dims=(0,))
        expected = NT([part.diff(dim=-1) for part in parts], ragged_dims=(0,))
        if layout == "self":
            out = source
        elif layout == "rank":
            out = NT([torch.zeros(1, 2, n, device=device, dtype=source.dtype) for n in (3, 5)], ragged_dims=(2,))
        elif layout in {"nested", "empty", "resize"}:
            lengths = (3, 5) if layout == "nested" else (0, 0) if layout == "empty" else (1, 2)
            out = NT([torch.zeros(n, 3, device=device, dtype=source.dtype) for n in lengths], ragged_dims=(0,))
        elif layout == "dense-empty":
            out = torch.empty(0, device=device, dtype=source.dtype)
        elif layout == "dense-resize":
            out = torch.empty(1, device=device, dtype=source.dtype)
        elif layout == "strided":
            out = torch.empty(3, 5, 2, device=device, dtype=source.dtype).permute(2, 1, 0)
        else:
            out = torch.empty(2, 5, 3, device=device, dtype=source.dtype)
        nested = isinstance(out, NT)
        alias = out.concat.detach().view(-1) if nested else out.detach()
        strides = out.stride() if not nested and layout == "strided" else None
        assert torch.diff(source, dim=-1, out=out) is out
        if nested:
            torch.testing.assert_close(out.unbind(), expected.unbind())
            retained = min(alias.numel(), out.concat.numel())
            torch.testing.assert_close(alias[:retained], out.concat.reshape(-1)[:retained])
            if layout == "rank":
                torch.compiler.reset()
                run = torch.compile(lambda value: value + 1, backend="aot_eager", dynamic=True, fullgraph=True)
                torch.testing.assert_close(run(out).unbind(), [part + 1 for part in expected.unbind()])
        else:
            torch.testing.assert_close(out, expected.tensor)
            if strides is not None:
                assert out.stride() == strides
                torch.testing.assert_close(alias, expected.tensor)

    @pytest.mark.parametrize(
        "order,dtype,accepted", [(1, torch.float32, True), (1, torch.int64, False), (0, torch.float32, False)]
    )
    def test_out_dtype(self, device, order, dtype, accepted):
        parts = [torch.randn(n, 4, device=device, dtype=torch.float64) for n in (3, 5)]
        source = NT(parts, ragged_dims=(0,))
        out = NT([torch.empty(n, 4 - order, device=device, dtype=dtype) for n in (3, 5)], ragged_dims=(0,))
        if not accepted:
            with pytest.raises(RuntimeError):
                torch.diff(source, n=order, dim=-1, out=out)
            return
        assert torch.diff(source, n=order, dim=-1, out=out) is out
        torch.testing.assert_close(out.unbind(), [part.diff(n=order, dim=-1).to(dtype) for part in parts])

    @pytest.mark.parametrize("case", ["input", "boundary", "out", "no_grad", "no_grad_resize", "n0", "n0_resize"])
    def test_out_grad(self, device, case):
        order = 0 if case.startswith("n0") else 1
        parts = [
            torch.randn(
                n,
                4,
                device=device,
                dtype=torch.float64,
                requires_grad=case in {"input", "no_grad", "no_grad_resize", "n0", "n0_resize"},
            )
            for n in (3, 5)
        ]
        source = NT(parts, ragged_dims=(0,))
        before = (
            torch.randn(2, 1, 4, device=device, dtype=source.dtype, requires_grad=True) if case == "boundary" else None
        )
        expected = [
            part.detach().diff(n=order, dim=0, prepend=before[i].detach() if before is not None else None)
            for i, part in enumerate(parts)
        ]
        out_parts = [
            torch.randn_like(part, requires_grad=case in {"out", "no_grad", "no_grad_resize", "n0", "n0_resize"})
            for part in (parts if case == "no_grad_resize" else expected[::-1] if case == "n0_resize" else expected)
        ]
        out = NT(out_parts, ragged_dims=(0,)).sin()
        if case == "n0_resize":
            previous = [part.detach().clone() for part in out.unbind()]
            with pytest.raises(RuntimeError, match="resize"):
                torch.diff(source, n=order, dim=1, out=out)
            torch.testing.assert_close(out.unbind(), previous)
            return
        if case in {"input", "boundary", "out"}:
            with pytest.raises(RuntimeError):
                torch.diff(source, n=order, dim=1, prepend=before, out=out)
            return
        with torch.set_grad_enabled(not case.startswith("no_grad")):
            assert torch.diff(source, n=order, dim=1, out=out) is out
        torch.testing.assert_close(out.unbind(), expected)
        if order == 0:
            actual = torch.autograd.grad(out.concat.sum(), (*parts, *out_parts))
            torch.testing.assert_close(
                actual,
                (*[torch.ones_like(part) for part in parts], *[torch.zeros_like(part) for part in out_parts]),
            )

    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("backend", [None, "aot_eager", "inductor"])
    def test_out_saved(self, device, order, backend):
        torch.compiler.reset()
        parts = [torch.randn(n, 4, device=device, dtype=torch.float64) for n in (3, 5)]
        if backend is None:
            leaves = [torch.randn(n, 4 - order, device=device, dtype=torch.float64, requires_grad=True) for n in (3, 5)]
            out = NT(leaves, ragged_dims=(0,)).sin()
            values = out.concat
        else:
            leaf = torch.randn(2, 5, 4 - order, device=device, dtype=torch.float64, requires_grad=True)
            leaves = [leaf]
            out = leaf.sin()
            values = out
        saved = values.square().sum()
        alias = values.detach().requires_grad_()
        alias_loss = alias.square().sum()

        def difference(value, target):
            return torch.diff(value, n=order, dim=-1, out=target)

        run = (
            difference if backend is None else torch.compile(difference, backend=backend, dynamic=True, fullgraph=True)
        )
        with torch.no_grad():
            assert run(NT(parts, ragged_dims=(0,)), out) is out
        with pytest.raises(RuntimeError, match="modified"):
            torch.autograd.grad(saved, leaves)
        with pytest.raises(RuntimeError, match="modified"):
            torch.autograd.grad(alias_loss, alias)

    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    @pytest.mark.parametrize("dense", [False, True])
    @pytest.mark.parametrize("axis,order", [(-1, 0), (-1, 1), (1, 1), (0, 1)])
    def test_out_compile(self, device, backend, dense, axis, order):
        torch.compiler.reset()
        run = torch.compile(
            lambda value, out: torch.diff(value, n=order, dim=axis, out=out),
            backend=backend,
            dynamic=True,
            fullgraph=True,
        )
        for lengths in ((1, 5), (1, 4)):
            parts = [
                torch.randn(n, 4, device=device, dtype=torch.float64, requires_grad=dense and order == 0)
                for n in lengths
            ]
            expected = NT(
                (
                    [parts[1] - parts[0]]
                    if axis == 0
                    else [part.diff(n=order, dim=0 if axis == 1 else axis) for part in parts]
                ),
                ragged_dims=(0,),
            )
            out = (
                torch.empty_like(expected.tensor)
                if dense
                else NT([torch.empty_like(part) for part in expected.unbind()], ragged_dims=(0,))
            )
            if not dense:
                with pytest.raises(torch._dynamo.exc.Unsupported, match="packed alias"):
                    run(NT(parts, ragged_dims=(0,)), out)
                continue
            assert run(NT(parts, ragged_dims=(0,)), out) is out
            torch.testing.assert_close(out, expected.tensor)
            if order == 0:
                torch.testing.assert_close(
                    torch.autograd.grad(out.square().sum(), parts),
                    torch.autograd.grad(sum(part.square().sum() for part in parts), parts),
                )


_BOOLEAN_REDUCTION_CASES = [
    pytest.param(((2, 3), (4, 3)), (0,), 1, 0, "any", False, id="leading-ragged-any"),
    pytest.param(((2, 3), (4, 3)), (0,), 2, 1, "all", False, id="single-static-all"),
    pytest.param(((2, 4, 2), (3, 2, 2)), (0, 1), 1, 0, "any", False, id="multi-first-any"),
    pytest.param(((2, 4, 2), (3, 2, 2)), (0, 1), 1, 0, "all", True, id="multi-first-all-keepdim"),
    pytest.param(((2, 4, 2), (3, 2, 2)), (0, 1), 2, 1, "all", False, id="multi-second-all"),
    pytest.param(((2, 4, 2), (3, 2, 2)), (0, 1), 2, 1, "any", True, id="multi-second-any-keepdim"),
    pytest.param(((2, 4, 2), (3, 2, 2)), (0, 1), 3, 2, "all", True, id="multi-static-all-keepdim"),
]

_BOOLEAN_GRAPH_CASES = [
    case
    for case in _BOOLEAN_REDUCTION_CASES
    if case.id in {"leading-ragged-any", "multi-first-all-keepdim", "multi-second-any-keepdim"}
]


def reference_options(source: NestedTensor) -> dict:
    r"""Return public construction options for an elementwise reference."""
    return {
        "batch_first": source.batch_first,
        "padding_value": source.padding_value,
        "mask_value": source.mask_value,
    }


def boolean_reduction_elements(shapes, operation, device=None):
    elements = [torch.full(shape, operation == "all", device=device, dtype=torch.bool) for shape in shapes]
    for element in elements:
        element[(0,) * element.dim()] = operation == "any"
    return elements


def ragged_reduction_reference(parts, ragged_dims, element_dim, keepdim):
    if all(part.shape == parts[0].shape for part in parts[1:]):
        return torch.stack(parts)
    output_ragged_dims = tuple(
        axis if keepdim else axis - (axis > element_dim) for axis in ragged_dims if axis != element_dim
    )
    return NT(parts, ragged_dims=output_ragged_dims)


def _run_or_expect_unsupported(nested_call, tensor_call):
    try:
        reference = tensor_call()
    except RuntimeError as error:
        with pytest.raises(type(error)):
            nested_call()
        return None
    return nested_call(), reference


class TestInPlace:

    @pytest.mark.parametrize("backend", [None, "aot_eager"])
    def test_dense(self, device, backend):
        torch.compiler.reset()
        parts = [torch.randn(3, 4, device=device, dtype=torch.float64, requires_grad=True) for _ in range(2)]
        bias = torch.randn(2, 3, device=device, dtype=torch.float64, requires_grad=True)
        references = [part.detach().clone().requires_grad_() for part in parts]
        reference_bias = bias.detach().clone().requires_grad_()
        output = NT(parts, ragged_dims=(0,)).sin()
        other = bias.cos().view(2, 3, 1)

        def mutate(left, right):
            return left.add_(right)

        run = mutate if backend is None else torch.compile(mutate, backend=backend, fullgraph=True)
        assert run(output, other) is output
        expected = [part.sin().add_(operand) for part, operand in zip(references, reference_bias.cos().view(2, 3, 1))]
        torch.testing.assert_close(output.unbind(), expected)
        probes = [torch.randn_like(part) for part in expected]
        torch.testing.assert_close(
            torch.autograd.grad(sum((part * probe).sum() for part, probe in zip(output, probes)), (*parts, bias)),
            torch.autograd.grad(
                sum((part * probe).sum() for part, probe in zip(expected, probes)), (*references, reference_bias)
            ),
        )

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_scalar_gradient(self, batch_first):
        torch.compiler.reset()
        parts = [torch.randn(length, dtype=torch.float64, requires_grad=True) for length in (3, 5)]
        bias = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        source = NT(parts, ragged_dims=(0,), padding_value=123.0, batch_first=batch_first).sin()
        operation = torch.compile(lambda left, right: left.add_(right), backend="aot_eager", fullgraph=True)
        result = operation(source, bias)
        gradients = torch.autograd.grad(result.concat.sum(), (*parts, bias))
        torch.testing.assert_close(gradients, (*[part.cos() for part in parts], torch.tensor(8.0, dtype=torch.float64)))

    @pytest.mark.parametrize(
        ("operation", "backend", "vector"),
        [
            pytest.param(lambda left, right: left.add_(right), None, False, id="method"),
            pytest.param(torch.ops.aten.add_.Tensor, None, False, id="aten"),
            pytest.param(operator.iadd, None, False, id="iadd"),
            pytest.param(operator.isub, None, False, id="isub"),
            pytest.param(operator.imul, None, False, id="imul"),
            pytest.param(operator.itruediv, None, False, id="idiv"),
            pytest.param(operator.iadd, "aot_eager", False, id="compiled"),
            pytest.param(operator.iadd, "aot_eager", None, id="compiled-same"),
            pytest.param(operator.iadd, "aot_eager", True, id="vector"),
        ],
    )
    def test_broadcast(self, device, operation, backend, vector):
        torch._dynamo.reset()
        shapes = ((2, 3), (4, 5))
        parts = [torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
        operand_shapes = shapes if vector is None else ((3,), (5,)) if vector else ((1, 3), (4, 1))
        operands = [
            torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True) for shape in operand_shapes
        ]
        references = [part.detach().clone().requires_grad_() for part in parts]
        reference_operands = [part.detach().clone().requires_grad_() for part in operands]
        output = NT(parts, ragged_dims=(0, 1)).sin()
        other = NT(operands, ragged_dims=(0,) if vector else (0, 1)).cos() + 2
        alias = output.detach()
        _ = alias.tensor

        def mutate(left, right):
            return operation(left, right)

        run = mutate if backend is None else torch.compile(mutate, backend=backend, fullgraph=True)
        assert run(output, other) is output

        expected = [operation(part.sin(), operand.cos() + 2) for part, operand in zip(references, reference_operands)]
        torch.testing.assert_close(output.unbind(), expected)
        torch.testing.assert_close(alias.tensor, output.tensor)
        probes = [torch.randn_like(part) for part in expected]
        torch.testing.assert_close(
            torch.autograd.grad(sum((part * probe).sum() for part, probe in zip(output, probes)), (*parts, *operands)),
            torch.autograd.grad(
                sum((part * probe).sum() for part, probe in zip(expected, probes)), (*references, *reference_operands)
            ),
        )

    def test_overlap(self, device):
        parts = [torch.randn(size, size, device=device) for size in (2, 3)]
        source = NT(parts, ragged_dims=(0, 1))
        expected = [part.clone() for part in parts]
        with pytest.raises(RuntimeError, match="memory|overlap"):
            expected[0].add_(expected[0].T)
        with pytest.raises(RuntimeError, match="memory|overlap"):
            source.add_(source.transpose(-1, -2))
        torch.testing.assert_close(source.unbind(), expected)
        scalar = NT([torch.ones(1, 1, device=device)], ragged_dims=(0, 1))
        scalar.add_(scalar.transpose(-1, -2))
        torch.testing.assert_close(scalar[0], torch.tensor([[2.0]], device=device))


class TestArithmeticFunctions:

    def test_dense_left_mul_sampled_layout_values_and_vjp(self):
        template = NT(
            [torch.empty(2, 4, 3), torch.empty(2, 6, 3)],
            ragged_dims=(1,),
        )
        values = torch.randn_like(template.concat, requires_grad=True)
        scale = torch.randn(2, 2, 1, 1)
        nested = template.packed_like(values)

        dense_left = scale * nested
        nested_left = nested * scale

        cotangent = torch.randn_like(values)
        dense_left_grad = torch.autograd.grad(dense_left.concat, values, cotangent, retain_graph=True)[0]
        nested_left_grad = torch.autograd.grad(nested_left.concat, values, cotangent)[0]
        assert_close(dense_left.concat, nested_left.concat)
        assert_close(dense_left_grad, nested_left_grad)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_dense_left_mul_sampled_layout_compiles_with_vjp(self):
        def consume(template, values, scale):
            nested = template.packed_like(values)
            return (scale * nested).concat, (nested * scale).concat

        compiled = torch.compile(consume, backend="aot_eager", fullgraph=True, dynamic=True)
        template = NT([torch.empty(2, length, 3) for length in (4, 6)], ragged_dims=(1,))
        values = torch.randn_like(template.concat, requires_grad=True)
        eager_values = values.detach().clone().requires_grad_()
        scale = torch.randn(2, 2, 1, 1)
        dense_left, nested_left = compiled(template, values, scale)
        eager_dense_left, eager_nested_left = consume(template, eager_values, scale)

        assert_close(dense_left, eager_dense_left)
        assert_close(nested_left, eager_nested_left)
        loss = dense_left.square().sum() + nested_left.square().sum()
        eager_loss = eager_dense_left.square().sum() + eager_nested_left.square().sum()
        expected_gradient = torch.autograd.grad(eager_loss, eager_values)[0]
        actual_gradient = torch.autograd.grad(loss, values)[0]
        assert_close(actual_gradient, expected_gradient)

    def test_dense_broadcast_expands_only_static_tail(self):
        nested = NT([torch.randn(2, 1), torch.randn(3, 1)])

        output = nested + torch.ones(1, 4)

        assert [element.shape for element in output] == [torch.Size((2, 4)), torch.Size((3, 4))]
        assert_close(output.concat, nested.concat + torch.ones(1, 4))

        single = NT([torch.randn(1, 3)])
        with pytest.raises(NotImplementedError, match="neither shape-aligned nor broadcast-compatible"):
            torch.add(single, torch.ones(4, 3))

    def test_add_converts_tensor_to_nested(self, device, float_dtype):
        nt = NestedTensor([torch.ones(2, device=device, dtype=float_dtype)])
        tensor = torch.ones_like(nt.tensor)
        output = torch.add(nt, tensor)
        assert_close(output, nt.tensor + tensor)

    def test_div_converts_other_to_nested(self, device, float_dtype):
        nt = NestedTensor([torch.ones(2, device=device, dtype=float_dtype) * 4])
        other = torch.full_like(nt.tensor, 2, device=device, dtype=float_dtype)
        output = torch.div(other, nt)
        assert_close(output, other / nt.tensor)

    def test_pow_with_scalar_and_nested_exponent(self, device, float_dtype):
        base = NestedTensor([torch.arange(1.0, 3.0, device=device, dtype=float_dtype)])
        output = torch.pow(base, 2)
        reference = base.tensor.pow(2)
        assert_close(output, reference)
        exponent = torch.full_like(base.tensor, 2, device=device, dtype=float_dtype)
        output = torch.pow(base, exponent)
        reference = base.tensor.pow(exponent)
        assert_close(output, reference)

    def test_sub_converts_tensor_argument(self, device, float_dtype):
        nt = NestedTensor([torch.ones(2, device=device, dtype=float_dtype)])
        tensor = torch.zeros_like(nt.tensor)
        output = torch.sub(tensor, nt)
        assert_close(output, tensor - nt.tensor)

    def test_add_supports_multi_ragged_layouts(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(6, device=device, dtype=float_dtype).reshape(2, 3),
                torch.arange(4, device=device, dtype=float_dtype).reshape(1, 4),
            ]
        )
        other = NT([torch.ones_like(t) for t in nt], **reference_options(nt))
        output = torch.add(nt, other)
        reference = NT([torch.add(x, y) for x, y in zip(nt, other)], **reference_options(nt))
        assert_close(output, reference)

    def test_add_padded_dense(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(5, 8, device=device, dtype=float_dtype),
                torch.randn(3, 8, device=device, dtype=float_dtype),
            ]
        )
        dense = torch.randn(*nt.shape, device=device, dtype=float_dtype)
        output = nt + dense
        for i, elem in enumerate(nt):
            expected = elem + dense[i, : elem.shape[0]]
            assert_close(output[i], expected)

        output_rev = dense + nt
        for i, elem in enumerate(nt):
            expected = dense[i, : elem.shape[0]] + elem
            assert_close(output_rev[i], expected)

    def test_add_padded_dense_image_like(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, 3, device=device, dtype=float_dtype),
                torch.randn(2, 5, 5, device=device, dtype=float_dtype),
            ]
        )
        dense = torch.zeros(*nt.shape, device=device, dtype=float_dtype)
        dense[0, :, :, 3:] = -1000
        output = nt + dense
        reference = nt.nested_like(nt.tensor + dense)
        assert_close(output, reference)

    def test_add_rejects_positional_dense_broadcast(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(5, 8, device=device, dtype=float_dtype),
                torch.randn(3, 8, device=device, dtype=float_dtype),
            ]
        )
        dense = torch.randn(1, 5, 8, device=device, dtype=float_dtype)
        with pytest.raises(NotImplementedError, match="non-scalar Tensor operand"):
            _ = nt + dense

    def test_add_dense_broadcastable_with_values(self, device, float_dtype):
        """NT[B, var_seq, D] + dense[D] — broadcast across packed dim."""
        nt = NT(
            [
                torch.randn(5, 8, device=device, dtype=float_dtype),
                torch.randn(3, 8, device=device, dtype=float_dtype),
            ]
        )
        bias = torch.randn(8, device=device, dtype=float_dtype)
        output = nt + bias
        for i, elem in enumerate(nt):
            assert_close(output[i], elem + bias)

    def test_dense_tail_broadcast_can_expand_static_dim(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(5, 1, device=device, dtype=float_dtype),
                torch.randn(3, 1, device=device, dtype=float_dtype),
            ]
        )
        bias = torch.randn(8, device=device, dtype=float_dtype)
        output = nt + bias
        reference = NT([elem + bias for elem in nt], **reference_options(nt))
        assert_close(output, reference)

    @pytest.mark.parametrize(
        ("lhs_shapes", "rhs_shapes"),
        [
            (((2, 2, 1), (3, 3, 1)), ((2, 2, 4), (3, 3, 4))),
            (((2, 2, 1), (2, 3, 1)), ((2, 2, 4), (2, 3, 4))),
        ],
        ids=("multi-ragged", "permuted-ragged"),
    )
    def test_nested_static_broadcast_values_and_shapes(
        self,
        device,
        float_dtype,
        lhs_shapes,
        rhs_shapes,
    ):
        lhs_parts = [torch.randn(shape, device=device, dtype=float_dtype) for shape in lhs_shapes]
        rhs_parts = [torch.randn(shape, device=device, dtype=float_dtype) for shape in rhs_shapes]
        lhs = NT(lhs_parts)
        rhs = NT(rhs_parts)
        bias = torch.randn(4, device=device, dtype=float_dtype)

        for output, reference_parts in (
            (torch.add(lhs, rhs), [x + y for x, y in zip(lhs_parts, rhs_parts)]),
            (torch.add(lhs, bias), [x + bias for x in lhs_parts]),
        ):
            assert_close(output, NT(reference_parts, **reference_options(lhs)))
            assert [tuple(element.shape) for element in output] == list(rhs_shapes)
            assert output.ragged_dims == rhs.ragged_dims

    def test_nested_singleton_ragged_broadcast_values_and_vjp(self, device, float_dtype):
        shapes = ((2, 3), (4, 5))
        channels = 4
        lhs_parts = [torch.randn(m, n, channels, device=device, dtype=float_dtype) for m, n in shapes]
        rhs_parts = [torch.randn(n, channels, device=device, dtype=float_dtype) for _m, n in shapes]
        lhs_values = torch.cat([part.reshape(-1, channels) for part in lhs_parts]).requires_grad_()
        rhs_values = torch.cat(rhs_parts).requires_grad_()
        lhs = NT([torch.empty_like(part) for part in lhs_parts], ragged_dims=(0, 1)).packed_like(lhs_values)
        rhs = NT([torch.empty_like(part) for part in rhs_parts], ragged_dims=(0,)).packed_like(rhs_values).unsqueeze(-3)

        added = lhs + rhs
        lhs_minus_rhs = lhs - rhs
        rhs_minus_lhs = rhs - lhs

        expected_add = torch.cat([(left + right).reshape(-1, channels) for left, right in zip(lhs_parts, rhs_parts)])
        expected_sub = torch.cat([(left - right).reshape(-1, channels) for left, right in zip(lhs_parts, rhs_parts)])
        assert_close(added.concat, expected_add)
        assert_close(lhs_minus_rhs.concat, expected_sub)
        assert_close(rhs_minus_lhs.concat, -expected_sub)
        assert added.ragged_dims == lhs.ragged_dims
        assert_close(added.element_sizes(), lhs.element_sizes())

        lhs_grad, rhs_grad = torch.autograd.grad(added.concat.sum(), (lhs_values, rhs_values))
        expected_rhs_grad = torch.cat([torch.full_like(right, m) for (m, _n), right in zip(shapes, rhs_parts)])
        assert_close(lhs_grad, torch.ones_like(lhs_values))
        assert_close(rhs_grad, expected_rhs_grad)

    def test_nested_broadcast_merges_ragged_axes_fullgraph_and_vjp(self, device):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")

        def multiply(left, right):
            return left * right, right * left

        compiled = torch.compile(multiply, backend="aot_eager", fullgraph=True, dynamic=True)
        for lhs_shapes, rhs_shapes in (
            (((3, 1, 2, 3), (3, 1, 2, 3)), ((3, 2, 2, 3), (3, 4, 2, 3))),
            (((1, 1, 2, 3), (4, 1, 2, 3)), ((4, 3, 2, 3), (4, 2, 2, 3))),
        ):
            lhs_parts = [torch.randn(shape, device=device, dtype=torch.float64) for shape in lhs_shapes]
            rhs_parts = [torch.randn(shape, device=device, dtype=torch.float64) for shape in rhs_shapes]
            lhs_template, rhs_template = NT(lhs_parts, ragged_dims=(0,)), NT(rhs_parts, ragged_dims=(1,))
            lhs_values, rhs_values = (
                template.concat.detach().requires_grad_() for template in (lhs_template, rhs_template)
            )
            lhs, rhs = lhs_template.packed_like(lhs_values), rhs_template.packed_like(rhs_values)
            lhs_reference = [part.detach().requires_grad_() for part in lhs_parts]
            rhs_reference = [part.detach().requires_grad_() for part in rhs_parts]
            expected = NT([left * right for left, right in zip(lhs_reference, rhs_reference)], ragged_dims=(0, 1))
            with fake_tensor_mod.FakeTensorMode() as mode:
                fake_outputs = multiply(mode.from_tensor(lhs_template), mode.from_tensor(rhs_template))
            for output in fake_outputs:
                assert output.ragged_dims == (0, 1)
                assert output.concat.shape == expected.concat.shape

            for operation in (multiply, compiled):
                output, reversed_output = operation(lhs, rhs)
                for actual in (output, reversed_output):
                    assert actual.ragged_dims == (0, 1)
                    assert_close(actual.element_sizes(), expected.element_sizes())
                    assert_close(actual.concat, expected.concat)
                weights = torch.randn_like(expected.concat)
                actual_grads = torch.autograd.grad(
                    (output.concat, reversed_output.concat), (lhs_values, rhs_values), (weights, weights)
                )
                reference_grads = torch.autograd.grad(
                    expected.concat, (*lhs_reference, *rhs_reference), weights, retain_graph=True
                )
                assert_close(actual_grads[0], 2 * NT(reference_grads[:2], ragged_dims=(0,)).concat)
                assert_close(actual_grads[1], 2 * NT(reference_grads[2:], ragged_dims=(1,)).concat)

    def test_dense_vector_broadcast_rejects_ragged_final_physical_dim(self, device, float_dtype):
        parts = [
            torch.arange(4, device=device, dtype=float_dtype).reshape(4, 1),
            torch.arange(16, device=device, dtype=float_dtype).reshape(4, 4),
        ]
        nt = NT(parts)
        bias = torch.arange(10, 50, 10, device=device, dtype=float_dtype)
        for lhs, rhs in ((nt, bias), (bias, nt)):
            with pytest.raises(NotImplementedError, match="broadcast-compatible"):
                torch.add(lhs, rhs)

    def test_mul_dense_channel_broadcast(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(3, 4, device=device, dtype=float_dtype),
                torch.randn(3, 7, device=device, dtype=float_dtype),
            ]
        )
        scale = torch.randn(1, 3, 1, device=device, dtype=float_dtype)
        output = nt * scale
        reference = NT([t * scale[0] for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_wrapped_ops_preserve_state(self):
        nt = NestedTensor(
            [torch.tensor([1, 2]), torch.tensor([3])],
            batch_first=False,
            padding_value=-5,
            mask_value=True,
        )
        output = torch.add(nt, 1)
        assert output.batch_first is False
        assert output.padding_value == -5
        assert output.mask_value is True
        reference = NT([t + 1 for t in nt], **reference_options(nt))
        assert_close(output, reference)


class TestCompile:

    @pytest.mark.parametrize("native", [False, True])
    def test_captured_shape(self, native):
        parts = [torch.randn(length, 4, requires_grad=True) for length in (2, 4)]
        source = NT(parts, ragged_dims=(0,))
        shape = source.shape

        def operation(value, target=shape):
            if native:
                return torch.ops.aten.reshape.default(value, target)
            return torch.reshape(value, target)

        compiled = torch.compile(operation, backend="aot_eager", fullgraph=True, dynamic=True)
        output = compiled(source)
        torch.testing.assert_close(output.unbind(0), tuple(parts))
        torch.testing.assert_close(
            torch.autograd.grad(output.concat.square().sum(), parts), tuple(2 * part for part in parts)
        )
        changed = [torch.randn(length, 4, requires_grad=True) for length in (3, 4)]
        changed_output = compiled(NT(changed, ragged_dims=(0,)))
        torch.testing.assert_close(changed_output.unbind(0), tuple(changed))
        torch.testing.assert_close(
            torch.autograd.grad(changed_output.concat.square().sum(), changed), tuple(2 * part for part in changed)
        )
        torch.testing.assert_close(compiled(source, tuple(map(int, shape))).unbind(0), tuple(parts))
        dense = torch.randn(2, 4, 4, requires_grad=True)
        dense_output = compiled(dense)
        torch.testing.assert_close(dense_output, dense)
        torch.testing.assert_close(torch.autograd.grad(dense_output.square().sum(), dense)[0], 2 * dense)

    @pytest.fixture(autouse=True)
    def isolate_compiler(self):
        # Independent tests can otherwise exhaust the shared torch function wrapper cache.
        torch.compiler.reset()
        yield
        torch.compiler.reset()

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compile_det(self):
        torch.manual_seed(1016)
        nt = NT([torch.randn(size, size) for size in (3, 4)])
        compiled = torch.compile(torch.linalg.det, backend="inductor", fullgraph=True)
        output = compiled(nt)
        reference = NT([torch.linalg.det(element) for element in nt], **reference_options(nt))
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compile_view_rebuild(self, device):
        nt = NT([torch.randn(length, 3, device=device) for length in (2, 4)])

        def reshape(value):
            return value.view(-1, 3).unflatten(2, (1, 3)).squeeze(2)

        output = torch.compile(reshape, backend="inductor", fullgraph=True)(nt)
        reference = NT(
            [element.view(-1, 3).unflatten(1, (1, 3)).squeeze(1) for element in nt],
            **reference_options(nt),
        )
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compile_convolution(self):
        nt = NT([torch.randn(3, length) for length in (17, 29)])
        weight = torch.randn(5, 3, 1)
        bias = torch.randn(5)
        compiled = torch.compile(F.conv1d, backend="inductor", fullgraph=True)
        assert_close(compiled(nt, weight, bias), F.conv1d(nt, weight, bias))

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compile_dropout(self):
        nt = NT(
            [
                torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
                torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
            ]
        )
        dropout_train_fn = torch.compile(
            lambda x: torch.dropout(x, p=0.2, train=True),
            backend="inductor",
            fullgraph=True,
        )
        dropout_comp = dropout_train_fn(nt)
        assert [tuple(element.shape) for element in dropout_comp] == [tuple(element.shape) for element in nt]
        assert torch.all((dropout_comp.concat == 0) | (dropout_comp.concat == nt.concat / 0.8))

    @pytest.mark.skipif(
        not hasattr(torch, "compile") or not hasattr(torch, "rms_norm"),
        reason="torch.compile or torch.rms_norm not available",
    )
    def test_compile_rms_norm(self):
        nt = NT(
            [
                torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
                torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
            ]
        )
        compiled = torch.compile(lambda x: torch.rms_norm(x, (2,)), backend="inductor", fullgraph=True)
        result = compiled(nt)
        reference = NT([torch.rms_norm(t, (2,)) for t in nt], **reference_options(nt))
        assert_close(result, reference)


class TestCumulativeOps:

    def test_cummax_batch_dim_not_supported(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, 0.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0, 1.0], device=device, dtype=float_dtype),
            ]
        )
        with pytest.raises(ValueError):
            torch.cummax(nt, dim=0)

    @pytest.mark.parametrize("op", [torch.cummax, torch.cummin])
    def test_cummax_cummin_returns_values_and_indices(self, op, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, 0.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0, 1.0], device=device, dtype=float_dtype),
            ]
        )
        output = op(nt, dim=1)
        reference = tuple(NT([op(t, dim=0)[idx] for t in nt], **reference_options(nt)) for idx in range(2))
        assert isinstance(output, tuple)
        assert_close(output[0], reference[0])
        assert_close(output[1], reference[1])

    def test_cumulative_batch_dim_not_supported(self, device, float_dtype):
        nt = NestedTensor([torch.tensor([1.0, 2.0], device=device, dtype=float_dtype)])
        with pytest.raises(ValueError):
            torch.cumsum(nt, dim=0)
        with pytest.raises(ValueError):
            torch.cumprod(nt, dim=0)

        nt_bf_false = NestedTensor([torch.tensor([1.0, 2.0], device=device, dtype=float_dtype)], batch_first=False)
        with pytest.raises(ValueError):
            torch.cumsum(nt_bf_false, dim=1)
        with pytest.raises(ValueError):
            torch.cumprod(nt_bf_false, dim=1)

    def test_cumulative_functions(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 2, 3], device=device, dtype=float_dtype),
                torch.tensor([4, 5], device=device, dtype=float_dtype),
            ]
        )
        output = torch.cumsum(nt, dim=1)
        reference = torch.tensor([[1, 3, 6], [4, 9, 0]], device=device, dtype=float_dtype)
        assert_close(output, reference)
        output = torch.cumprod(nt, dim=1)
        reference = torch.tensor([[1, 2, 6], [4, 20, 0]], device=device, dtype=float_dtype)
        assert_close(output, reference)

    def test_logcumsumexp(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(3, device=device, dtype=float_dtype),
                torch.randn(2, device=device, dtype=float_dtype),
            ]
        )
        output = torch.logcumsumexp(nt, dim=1)
        reference = NT([torch.logcumsumexp(t, dim=0) for t in nt], **reference_options(nt))
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

        with pytest.raises(ValueError):
            torch.logcumsumexp(nt, dim=0)


class TestDimensionTransforms:

    def test_moveaxis(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
            ]
        )
        output = torch.moveaxis(nt, 1, 2)
        reference = torch.moveaxis(nt.tensor, 1, 2)
        assert_close(output, reference)

    @pytest.mark.parametrize(
        "shapes,ragged,batch_first",
        [(((2, 1, 3), (3, 1, 3)), (0,), True), (((2, 1, 3),), (1,), True), (((1, 2, 3), (1, 4, 3)), (1,), False)],
        ids=["static", "packed-singleton", "before-batch"],
    )
    def test_squeeze_default(self, device, float_dtype, shapes, ragged, batch_first):
        parts = [torch.randn(shape, device=device, dtype=float_dtype, requires_grad=True) for shape in shapes]
        references = [part.detach().clone().requires_grad_() for part in parts]
        source = NT(parts, ragged_dims=ragged, batch_first=batch_first)
        output = torch.squeeze(source)
        expected = [part.squeeze() for part in references]
        assert output.batch_first
        torch.testing.assert_close(output.unbind(), expected)
        gradients = torch.autograd.grad(output.square().sum(), parts)
        reference_gradients = torch.autograd.grad(sum(part.square().sum() for part in expected), references)
        torch.testing.assert_close(gradients, reference_gradients)

    @pytest.mark.parametrize("native", [False, True])
    def test_unbind_before_batch(self, device, native):
        parts = [torch.randn(2, n, 3, device=device, dtype=torch.float64, requires_grad=True) for n in (2, 4)]
        references = [part.detach().clone().requires_grad_() for part in parts]
        source = NT(parts, ragged_dims=(1,), batch_first=False)
        output = torch.ops.aten.unbind.int(self=source, dim=0) if native else source.unbind(0)
        expected = [[part.select(0, index) for part in references] for index in range(2)]
        for actual, reference in zip(output, expected):
            assert actual.batch_first
            assert actual.shape == (2, 4, 3)
            torch.testing.assert_close(actual.unbind(), reference)
        gradients = torch.autograd.grad(sum(part.square().sum() for part in output), parts)
        reference_gradients = torch.autograd.grad(
            sum(part.square().sum() for group in expected for part in group), references
        )
        torch.testing.assert_close(gradients, reference_gradients)
        with torch.inference_mode():
            output = torch.ops.aten.unbind.int(self=source, dim=0) if native else source.unbind(0)
            for actual, reference in zip(output, expected):
                assert actual.batch_first
                torch.testing.assert_close(actual.unbind(), reference)

    def test_squeeze_default_with_all_ragged_ones(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(1, 2, device=device, dtype=float_dtype),
                torch.randn(1, 2, device=device, dtype=float_dtype),
            ]
        )
        output = torch.squeeze(nt)
        reference = NT([torch.squeeze(t) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_squeeze_unsqueeze_round_trip(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(1, 3, device=device, dtype=float_dtype),
                torch.randn(1, 3, device=device, dtype=float_dtype),
            ]
        )
        squeezed = torch.squeeze(nt, dim=1)
        output = torch.unsqueeze(squeezed, dim=2)
        reference = torch.unsqueeze(torch.squeeze(nt.tensor, dim=1), dim=2)
        assert_close(output, reference)

    def test_squeeze_trailing_static_dim_after_ragged_dims(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(4, 4, 1, device=device, dtype=float_dtype),
                torch.randn(2, 2, 1, device=device, dtype=float_dtype),
            ]
        )
        output = torch.squeeze(nt, dim=-1)
        reference = NT([torch.squeeze(t, dim=-1) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_squeeze_sample_axis_values_and_vjp(self, device, float_dtype):
        template = NT(
            [
                torch.empty(1, 2, 3, device=device, dtype=float_dtype),
                torch.empty(1, 5, 3, device=device, dtype=float_dtype),
            ],
            ragged_dims=(1,),
        )
        values = torch.randn_like(template.concat, requires_grad=True)
        sampled = template.packed_like(values)
        expected = values.squeeze(1)

        output = sampled.squeeze(-3)

        cotangent = torch.randn_like(expected)
        actual_grad = torch.autograd.grad(output.concat, values, cotangent)[0]
        expected_grad = torch.autograd.grad(expected, values, cotangent)[0]
        assert output.ragged_dims == (0,)
        assert output.concat.shape == (7, 3)
        assert_close(output.concat, expected)
        assert_close(actual_grad, expected_grad)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_squeeze_sample_axis_compiles_with_vjp(self, device):
        compiled = torch.compile(
            lambda template, values: template.packed_like(values).squeeze(-3).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        template = NT([torch.empty(1, length, 3, device=device) for length in (2, 3)], ragged_dims=(1,))
        values = torch.randn_like(template.concat, requires_grad=True)
        expected = values.squeeze(1)
        output = compiled(template, values)
        cotangent = torch.randn_like(expected)
        actual_grad = torch.autograd.grad(output, values, cotangent)[0]
        expected_grad = torch.autograd.grad(expected, values, cotangent)[0]
        assert_close(output, expected)
        assert_close(actual_grad, expected_grad)

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("method", [False, True])
    def test_transpose_saved_alias(self, device, batch_first, method):
        class SavedInput(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                ctx.save_for_backward(value)
                return value.concat.sum()

            @staticmethod
            def backward(ctx, gradient):
                (value,) = ctx.saved_tensors
                return torch.ones_like(value) * gradient

        source = NT(
            [torch.randn(length, 4, device=device, requires_grad=True) for length in (3, 5)],
            ragged_dims=(0,),
            batch_first=batch_first,
        ).sin()
        loss = SavedInput.apply(source)
        view = source.transpose(0, 1) if method else torch.transpose(source, 0, 1)
        view.add_(1)
        with pytest.raises(RuntimeError, match="modified.*inplace|version"):
            loss.backward()

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_batch_sequence_transpose_compiled_vjp(self, device, batch_first):
        def project_and_transpose(source, weight, bias):
            return F.linear(source, weight, bias).transpose(0, 1)

        compiled = torch.compile(project_and_transpose, backend="aot_eager", fullgraph=True, dynamic=True)
        for lengths in ((3, 5), (4, 6)):
            parts = [torch.randn(length, 4, device=device, requires_grad=True) for length in lengths]
            references = [part.detach().clone().requires_grad_() for part in parts]
            weight = torch.randn(6, 4, device=device, requires_grad=True)
            bias = torch.randn(6, device=device, requires_grad=True)
            reference_weight = weight.detach().clone().requires_grad_()
            reference_bias = bias.detach().clone().requires_grad_()
            actual = compiled(NT(parts, ragged_dims=(0,), batch_first=batch_first), weight, bias)
            expected = project_and_transpose(
                NT(references, ragged_dims=(0,), batch_first=batch_first), reference_weight, reference_bias
            )
            assert actual.batch_first is not batch_first
            torch.testing.assert_close(actual.concat, expected.concat)
            torch.testing.assert_close(
                torch.autograd.grad(actual.concat.square().sum(), (*parts, weight, bias)),
                torch.autograd.grad(expected.concat.square().sum(), (*references, reference_weight, reference_bias)),
            )

    def test_transpose_swaps_last_two_element_dims(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(4, 5, 7, device=device, dtype=float_dtype),
                torch.randn(4, 4, 6, device=device, dtype=float_dtype),
            ]
        )

        output = nt.transpose(-1, -2)
        alias = nt.mT
        reference = NT([tensor.transpose(-1, -2) for tensor in nt], **reference_options(nt))

        assert_close(output, reference)
        assert_close(alias, reference)
        assert output.shape == torch.Size([2, 4, 7, 5])

    @pytest.mark.parametrize(
        ("nested_transform", "element_transform"),
        [
            (lambda value: value.squeeze(2), lambda value: value.squeeze(1)),
            (lambda value: value.movedim(1, 3), lambda value: value.movedim(0, 2)),
        ],
        ids=("view", "axis-move"),
    )
    def test_view_like_ops_values_and_vjp(self, device, float_dtype, nested_transform, element_transform):
        leaves = [
            torch.randn(2, 1, 3, device=device, dtype=float_dtype, requires_grad=True),
            torch.randn(3, 1, 3, device=device, dtype=float_dtype, requires_grad=True),
        ]
        references = [leaf.detach().clone().requires_grad_() for leaf in leaves]

        output = nested_transform(NT(leaves))
        expected = [element_transform(reference) for reference in references]
        weights = [torch.randn_like(element) for element in expected]
        actual_loss = sum((element * weight).sum() for element, weight in zip(output, weights))
        expected_loss = sum((element * weight).sum() for element, weight in zip(expected, weights))
        actual_gradients = torch.autograd.grad(actual_loss, leaves)
        expected_gradients = torch.autograd.grad(expected_loss, references)

        for actual, reference in zip(output, expected):
            assert_close(actual, reference)
        for actual, reference in zip(actual_gradients, expected_gradients):
            assert_close(actual, reference)

    def test_swapaxes_batch_dim_raises(self):
        nt = NestedTensor([torch.ones(2, 2)])
        with pytest.raises(ValueError):
            torch.swapaxes(nt, 0, 1)

    def test_swapaxes(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
            ]
        )
        output = torch.swapaxes(nt, 1, 2)
        reference = torch.swapaxes(nt.tensor, 1, 2)
        assert_close(output, reference)

    def test_torch_permute_accepts_dims_sequence(self, device, float_dtype):
        nt = NestedTensor([torch.randn(2, 3, 4, device=device, dtype=float_dtype)])
        dims = (0, 3, 2, 1)
        reference = nt.permute(*dims)
        output = torch.permute(nt, dims)
        assert_close(output, reference)
        assert output.shape == reference.shape

    def test_unsqueeze_before_batch_preserves_samples(self):
        nt = NestedTensor([torch.tensor([1, 2]), torch.tensor([3, 4, 5])])
        output = torch.unsqueeze(nt, dim=0)
        assert output.batch_first is False
        assert_close(output.tensor, nt.tensor.unsqueeze(0))

    @pytest.mark.parametrize("transform", ["grad", "jvp", "vmap"])
    @pytest.mark.parametrize("factory", [False, True])
    @pytest.mark.parametrize("dim", [0, -1])
    def test_unsqueeze_transforms(self, device, transform, factory, dim):
        values = torch.randn(5, 2, device=device, dtype=torch.float64)
        direction = torch.randn_like(values)
        source = values if factory else NT(values.split((2, 3)), ragged_dims=(0,))
        vector = direction if factory else NT(direction.split((2, 3)), ragged_dims=(0,))

        def loss(value):
            if factory:
                value = NT(value.split((2, 3)), ragged_dims=(0,))
            return torch.unsqueeze(value, dim).sin().square().sum()

        def reference(value):
            return value.sin().square().sum()

        if transform == "grad":
            actual = torch.func.grad(loss)(source)
            actual = actual if factory else actual.concat
            expected = torch.func.grad(reference)(values)
        elif transform == "jvp":
            actual = torch.func.jvp(loss, (source,), (vector,))
            expected = torch.func.jvp(reference, (values,), (direction,))
        else:
            actual = torch.vmap(loss, in_dims=-1)(source)
            expected = torch.vmap(reference, in_dims=-1)(values)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("entry", ["function", "method"])
    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    @pytest.mark.parametrize("dim", [0, 1])
    def test_unsqueeze_grad(self, device, entry, backend, dim):
        torch.compiler.reset()
        parts = [torch.tensor(value, device=device, dtype=torch.float64, requires_grad=True) for value in (2.0, 3.0)]
        references = [part.detach().clone().requires_grad_() for part in parts]

        def view(value):
            return torch.unsqueeze(value, dim) if entry == "function" else value.unsqueeze(dim)

        compiled = torch.compile(view, backend=backend, fullgraph=True)
        output = compiled(NT(parts))
        expected = torch.stack(references).unsqueeze(dim)
        torch.testing.assert_close(output.tensor, expected)
        torch.testing.assert_close(
            torch.autograd.grad(output.square().sum(), parts),
            torch.autograd.grad(expected.square().sum(), references),
        )

    def test_unsqueeze_batch_first_false_respects_batch_dim(self):
        nt = NestedTensor([torch.tensor([1, 2])], batch_first=False)
        with pytest.raises(ValueError):
            torch.unsqueeze(nt, dim=0)
        with pytest.raises(ValueError):
            torch.unsqueeze(nt, dim=1)
        output = torch.unsqueeze(nt, dim=2)
        assert output.batch_first is False
        assert output.tensor.shape == torch.Size([2, 1, 1])

    def test_unsqueeze_after_moved_static_axis_rbf_values_and_vjp(self, device, float_dtype):
        lengths = (3, 5)
        template = NT(
            [torch.empty(length, 5, 2, device=device) for length in lengths],
            ragged_dims=(0,),
        )
        values = torch.randn_like(template.concat, dtype=float_dtype, requires_grad=True)
        source = template.packed_like(values)
        centers = torch.linspace(2, 22, 2, device=device, dtype=float_dtype)

        moved = source.movedim(2, 3)
        expanded = moved.unsqueeze(-1)
        output = torch.exp(-(((expanded - centers) / 10) ** 2)).flatten(start_dim=-2)

        parts = values.split(lengths)
        references = tuple(
            torch.exp(-(((part.movedim(1, 2).unsqueeze(-1) - centers) / 10) ** 2)).flatten(start_dim=-2)
            for part in parts
        )
        expected = torch.cat(tuple(reference.movedim(1, 2) for reference in references))
        cotangent = torch.randn_like(expected)
        actual_grad = torch.autograd.grad(output.concat, values, cotangent, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, values, cotangent)[0]

        assert output.shape == torch.Size((2, 5, 2, 10))
        assert output.ragged_dims == (0,)
        assert_close(output.concat, expected)
        assert_close(actual_grad, expected_grad)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_unsqueeze_after_moved_static_axis_compiles(self, device):
        def run(template, values, centers):
            source = template.packed_like(values).movedim(2, 3).unsqueeze(-1)
            return torch.exp(-(((source - centers) / 10) ** 2)).flatten(start_dim=-2).concat

        compiled = torch.compile(run, backend="aot_eager", fullgraph=True, dynamic=True)
        centers = torch.linspace(2, 22, 2, device=device)
        lengths = (2, 3)
        template = NT([torch.empty(length, 5, 2, device=device) for length in lengths], ragged_dims=(0,))
        values = torch.randn_like(template.concat, requires_grad=True)
        output = compiled(template, values, centers)
        references = tuple(
            torch.exp(-(((part.movedim(1, 2).unsqueeze(-1) - centers) / 10) ** 2)).flatten(start_dim=-2)
            for part in values.split(lengths)
        )
        expected = torch.cat(tuple(reference.movedim(1, 2) for reference in references))
        cotangent = torch.randn_like(expected)
        actual_grad = torch.autograd.grad(output, values, cotangent, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, values, cotangent)[0]
        assert_close(output, expected)
        assert_close(actual_grad, expected_grad)

    def test_movedim(self, device, float_dtype):
        a = torch.randn(3, 4, 5, device=device, dtype=float_dtype)
        b = torch.randn(2, 4, 5, device=device, dtype=float_dtype)
        nt = NT([a, b])
        # dim -1 -> dim 1 in NestedTensor means last -> second
        result = torch.movedim(nt, -1, 1)
        ref = torch.moveaxis(nt, -1, 1)
        assert_close(result[0], ref[0])
        assert_close(result[1], ref[1])

    def test_pairwise_unsqueeze_multiply_builds_square_map(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(26, 8, device=device, dtype=float_dtype),
                torch.randn(14, 8, device=device, dtype=float_dtype),
                torch.randn(9, 8, device=device, dtype=float_dtype),
            ]
        )
        output = nt.unsqueeze(1) * nt.unsqueeze(2)
        reference = NT(
            [torch.unsqueeze(t, 0) * torch.unsqueeze(t, 1) for t in nt],
            batch_first=nt.batch_first,
            padding_value=nt.padding_value,
            mask_value=nt.mask_value,
        )
        assert_close(output, reference)
        assert output.shape == torch.Size([3, 26, 26, 8])
        assert [tuple(element.shape) for element in output] == [(26, 26, 8), (14, 14, 8), (9, 9, 8)]

    def test_pairwise_unsqueeze_multiply_from_tensor_mask_preserves_square_metadata(self, device, float_dtype):
        dense = torch.randn(1, 5, 3, device=device, dtype=float_dtype)
        mask = torch.ones(1, 5, device=device, dtype=torch.bool)
        nt = NestedTensor.from_tensor_mask(dense, mask)

        output = nt.unsqueeze(1) * nt.unsqueeze(2)
        reference = torch.unsqueeze(dense, 1) * torch.unsqueeze(dense, 2)

        assert output.shape == torch.Size([1, 5, 5, 3])
        assert [tuple(element.shape) for element in output] == [(5, 5, 3)]
        assert_close(output, reference)

        channels_first = torch.transpose(output, 1, 3)
        normalized = torch.nn.functional.group_norm(channels_first, 1)
        assert normalized.shape == torch.Size([1, 3, 5, 5])
        assert_close(normalized, torch.nn.functional.group_norm(torch.transpose(reference, 1, 3), 1))


class TestDist:

    def test_dist(self, device, float_dtype):
        a = NT(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0], device=device, dtype=float_dtype),
            ]
        )
        b = NT(
            [
                torch.tensor([2.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([1.0], device=device, dtype=float_dtype),
            ],
            **reference_options(a),
        )
        output = torch.dist(a, b, p=2)
        reference = torch.stack([torch.dist(x, y, p=2) for x, y in zip(a, b)])
        assert_close(output, reference)


class TestEinsum:

    @staticmethod
    def _ligand_mpnn_einsum_inputs(
        equation,
        lengths,
        *,
        device="cpu",
        dtype=None,
        requires_grad=False,
    ):
        if equation == "bli,bli->bl":
            lhs_template = NT(
                [torch.empty(length, 4, device=device, dtype=dtype) for length in lengths],
                ragged_dims=(0,),
            )
            rhs_template = NT(
                [torch.empty(length, 4, device=device, dtype=dtype) for length in lengths],
                ragged_dims=(0,),
            )
            lhs_values = torch.randn_like(lhs_template.concat, requires_grad=requires_grad)
            rhs_values = torch.randn_like(rhs_template.concat, requires_grad=requires_grad)
            reference = torch.einsum("xi,xi->x", lhs_values, rhs_values)
        elif equation == "blqp,blyq->blyp":
            lhs_template = NT(
                [torch.empty(length, 3, 4, device=device, dtype=dtype) for length in lengths],
                ragged_dims=(0,),
            )
            rhs_template = NT(
                [torch.empty(length, 5, 3, device=device, dtype=dtype) for length in lengths],
                ragged_dims=(0,),
            )
            lhs_values = torch.randn_like(lhs_template.concat, requires_grad=requires_grad)
            rhs_values = torch.randn_like(rhs_template.concat, requires_grad=requires_grad)
            reference = torch.einsum("xqp,xyq->xyp", lhs_values, rhs_values)
        else:
            raise AssertionError(f"unsupported test equation {equation}")
        return lhs_template.packed_like(lhs_values), rhs_template.packed_like(rhs_values), reference

    @pytest.mark.parametrize("equation", ["bli,bli->bl", "blqp,blyq->blyp"])
    def test_ligand_mpnn_einsum_values_and_vjp(self, equation, device, float_dtype):
        lhs, rhs, reference = self._ligand_mpnn_einsum_inputs(
            equation,
            (2, 5),
            device=device,
            dtype=float_dtype,
            requires_grad=True,
        )
        lhs_values = lhs.concat
        rhs_values = rhs.concat
        if equation == "bli,bli->bl":
            reference = torch.einsum("xi,xi->x", lhs_values, rhs_values)
            expected_tail = ()
        else:
            reference = torch.einsum("xqp,xyq->xyp", lhs_values, rhs_values)
            expected_tail = (5, 4)

        output = torch.einsum(equation, lhs, rhs)

        cotangent = torch.randn_like(reference)
        output_gradients = torch.autograd.grad(
            output.concat,
            (lhs_values, rhs_values),
            cotangent,
        )
        reference_gradients = torch.autograd.grad(reference, (lhs_values, rhs_values), cotangent)
        assert output.concat.shape == (sum((2, 5)), *expected_tail)
        assert output.ragged_dims == (0,)
        assert_close(output.concat, reference)
        for actual, expected in zip(output_gradients, reference_gradients):
            assert_close(actual, expected)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize("equation", ["bli,bli->bl", "blqp,blyq->blyp"])
    def test_ligand_mpnn_einsum_compiles_with_vjp(self, equation, device):
        compiled = torch.compile(
            lambda lhs_template, lhs_values, rhs_template, rhs_values: torch.einsum(
                equation,
                lhs_template.packed_like(lhs_values),
                rhs_template.packed_like(rhs_values),
            ).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        lhs, rhs, reference = self._ligand_mpnn_einsum_inputs(
            equation,
            (2, 3),
            device=device,
            requires_grad=True,
        )
        lhs_values = lhs.concat
        rhs_values = rhs.concat
        packed_equation = "xi,xi->x" if equation == "bli,bli->bl" else "xqp,xyq->xyp"
        reference = torch.einsum(packed_equation, lhs_values, rhs_values)
        lhs_template = lhs.packed_like(torch.empty_like(lhs_values))
        rhs_template = rhs.packed_like(torch.empty_like(rhs_values))
        output = compiled(lhs_template, lhs_values, rhs_template, rhs_values)
        cotangent = torch.randn_like(reference)
        output_gradients = torch.autograd.grad(output, (lhs_values, rhs_values), cotangent)
        reference_gradients = torch.autograd.grad(reference, (lhs_values, rhs_values), cotangent)
        assert_close(output, reference)
        for actual, expected in zip(output_gradients, reference_gradients):
            assert_close(actual, expected)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_einsum_matmul_equation_compile_fullgraph(self, device, float_dtype):
        a = torch.randn(2, 3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 3, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, 2, device=device, dtype=float_dtype)
        nt = NT([a, b])
        compiled = torch.compile(lambda x, y: torch.einsum("...ij,jk->...ik", x, y), backend="inductor", fullgraph=True)
        result = compiled(nt, w)
        reference = NT(
            [
                torch.einsum("...ij,jk->...ik", a, w),
                torch.einsum("...ij,jk->...ik", b, w),
            ],
            **reference_options(nt),
        )
        assert_close(result, reference)

    def test_einsum_matmul_equation_with_ellipsis(self, device, float_dtype):
        a = torch.randn(2, 3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 3, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, 2, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.einsum("...ij,jk->...ik", nt, w)
        reference = NT(
            [
                torch.einsum("...ij,jk->...ik", a, w),
                torch.einsum("...ij,jk->...ik", b, w),
            ],
            **reference_options(nt),
        )
        assert_close(result, reference)

    def test_einsum_matrix_vector_equation(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.einsum("ij,j->i", nt, w)
        reference = NT([torch.einsum("ij,j->i", a, w), torch.einsum("ij,j->i", b, w)], **reference_options(nt))
        assert_close(result, reference)

    def test_einsum_mismatched_batch_lengths_raises(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, device=device, dtype=float_dtype)
        nt1 = NT([a, a])
        nt2 = NT([b])
        with pytest.raises(ValueError, match="einsum: NestedTensor batch length mismatch"):
            torch.einsum("ij,ij->i", nt1, nt2)

    def test_einsum_nt_times_tensor(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, 2, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.einsum("ij,jk->ik", nt, w)
        assert_close(result[0], torch.einsum("ij,jk->ik", a, w))
        assert_close(result[1], torch.einsum("ij,jk->ik", b, w))

    def test_einsum_vector_matrix_equation(self, device, float_dtype):
        v = torch.randn(4, device=device, dtype=float_dtype)
        a = torch.randn(4, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 5, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.einsum("i,ij->j", v, nt)
        reference = NT([torch.einsum("i,ij->j", v, a), torch.einsum("i,ij->j", v, b)], **reference_options(nt))
        assert_close(result, reference)

    def test_einsum_pairwise_nested_operands(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(4, 3, device=device, dtype=float_dtype),
                torch.randn(6, 3, device=device, dtype=float_dtype),
                torch.randn(5, 3, device=device, dtype=float_dtype),
            ]
        )
        weights = torch.randn(3, 3, device=device, dtype=float_dtype)
        output = torch.einsum("bia,ac,bjc->bij", nt, weights, nt)
        reference = NT([torch.einsum("ia,ac,jc->ij", t, weights, t) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_einsum_broadcast_dense_batch(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(5, 4, device=device, dtype=float_dtype),
                torch.randn(7, 4, device=device, dtype=float_dtype),
            ]
        )
        dense = torch.randn(1, 7, 4, device=device, dtype=float_dtype)
        output = torch.einsum("blc,bmc->blm", nt, dense)
        reference = NT([torch.einsum("lc,mc->lm", t, dense[0]) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_einsum_global_query_compile(self, device, float_dtype):
        hidden_states = NT(
            torch.randn(5, 7, device=device, dtype=float_dtype),
            torch.randn(9, 7, device=device, dtype=float_dtype),
        )
        query_states = torch.randn(2, 3, 11, device=device, dtype=float_dtype)
        key_weight = torch.randn(3, 7, 11, device=device, dtype=float_dtype)
        value_weight = torch.randn(3, 7, 13, device=device, dtype=float_dtype)

        def einsums(hidden_states, query_states, key_weight, value_weight):
            key_states = torch.einsum("bls,hsk->bhlk", hidden_states, key_weight)
            value_states = torch.einsum("bls,hsv->bhlv", hidden_states, value_weight)
            attention_scores = torch.einsum("bhk,bhlk->bhl", query_states, key_states)
            context = torch.einsum("bhl,bhlv->bhv", attention_scores, value_states)
            return key_states, attention_scores, context

        compiled = torch.compile(einsums, backend="inductor", fullgraph=True)
        key_states, attention_scores, context = compiled(hidden_states, query_states, key_weight, value_weight)

        key_reference = NT(
            [torch.einsum("ls,hsk->hlk", hidden_state, key_weight) for hidden_state in hidden_states],
            **reference_options(hidden_states),
        )
        value_reference = NT(
            [torch.einsum("ls,hsv->hlv", hidden_state, value_weight) for hidden_state in hidden_states],
            **reference_options(hidden_states),
        )
        score_reference = NT(
            [
                torch.einsum("hk,hlk->hl", query_states[index], key_state)
                for index, key_state in enumerate(key_reference)
            ],
            **reference_options(hidden_states),
        )
        context_reference = NT(
            [torch.einsum("hl,hlv->hv", score, value) for score, value in zip(score_reference, value_reference)],
            **reference_options(hidden_states),
        )

        assert_close(key_states, key_reference)
        assert_close(attention_scores, score_reference)
        # Packed and per-sample contractions reduce in a different order. The
        # discrepancy compounds across four low-precision contractions, so use
        # a dtype-aware end-to-end tolerance rather than an FP32-like constant.
        atol, rtol = low_precision_cuda_tolerances(
            device,
            float_dtype,
            default=(1e-4, 1e-4),
            fp16=(2e-2, 7e-3),
            bf16=(8e-2, 5e-2),
        )
        assert_close(context, context_reference, atol=atol, rtol=rtol)

    def test_einsum_global_query_bad_layout(self, device, float_dtype):
        hidden_states = NT(
            torch.randn(4, 5, device=device, dtype=float_dtype),
            torch.randn(4, 9, device=device, dtype=float_dtype),
        )
        key_weight = torch.randn(3, 4, 11, device=device, dtype=float_dtype)

        with pytest.raises(RuntimeError, match="subscript s has size"):
            torch.einsum("bls,hsk->bhlk", hidden_states, key_weight)


class TestFlattenUnflatten:

    @pytest.mark.parametrize("backend", [None, "aot_eager"])
    @pytest.mark.parametrize("method,keyword", [(torch.Tensor.view, "size"), (torch.Tensor.reshape, "shape")])
    def test_native_signature(self, device, method, keyword, backend):
        torch.compiler.reset()
        parts = [torch.arange(6.0, device=device).requires_grad_() for _ in range(2)]
        source = NT(parts, ragged_dims=(0,))

        def operation(value):
            return method(value, **{keyword: (3, 2)})

        run = operation if backend is None else torch.compile(operation, backend=backend, fullgraph=True)
        output = run(source)
        torch.testing.assert_close(output.unbind(), [part.reshape(3, 2) for part in parts])
        torch.testing.assert_close(
            torch.autograd.grad(output.concat.square().sum(), parts), [2 * part for part in parts]
        )
        with pytest.raises(TypeError):
            method(source, (3, 2), **{keyword: (3, 2)})
        with pytest.raises(TypeError):
            method(source)

    @pytest.mark.parametrize("backend", [None, "aot_eager"])
    @pytest.mark.parametrize(
        "source_dtype,dtype",
        [
            (torch.float32, torch.int32),
            (torch.float32, torch.uint8),
            (torch.float32, torch.float64),
            (torch.complex64, torch.float64),
        ],
    )
    @pytest.mark.parametrize("ragged", [False, True])
    def test_dtype_view(self, device, source_dtype, dtype, ragged, backend):
        torch.compiler.reset()
        shapes = [(2, 4), (3, 6 if ragged else 4)]
        source = NT(
            [
                torch.arange(math.prod(shape), device=device, dtype=torch.float32)
                .to(source_dtype)
                .reshape(shape)
                .requires_grad_()
                for shape in shapes
            ],
            ragged_dims=(0, 1) if ragged else (0,),
        ).sin()
        expected = [part.view(dtype) for part in source.unbind()]
        loss = source.square().sum()
        native_loss = torch.ops.aten.square.default(source).sum()

        def operation(value):
            return torch.Tensor.view(value, dtype)

        run = operation if backend is None else torch.compile(operation, backend=backend, fullgraph=True)
        if backend is not None and (
            source_dtype.itemsize != dtype.itemsize or source_dtype.is_complex != dtype.is_complex
        ):
            with pytest.raises(torch._dynamo.exc.Unsupported, match="eager-only"):
                run(source)
            return
        output = run(source)
        torch.testing.assert_close(output.unbind(), expected)
        assert not output.requires_grad
        output.zero_()
        torch.testing.assert_close(source.concat, torch.zeros_like(source.concat))
        with pytest.raises(RuntimeError, match="modified"):
            loss.backward(torch.ones_like(loss))
        with pytest.raises(RuntimeError, match="modified"):
            native_loss.backward(torch.ones_like(native_loss))

    @pytest.mark.parametrize("backend", [None, "aot_eager"])
    @pytest.mark.parametrize("batch_first", [False, True])
    @pytest.mark.parametrize("batched", [False, True])
    def test_dense_template(self, device, batch_first, batched, backend):
        torch.compiler.reset()
        parts = [torch.arange(6.0, device=device, dtype=torch.float64).requires_grad_() for _ in range(2)]
        source = NT(parts, ragged_dims=(0,), batch_first=batch_first)
        shape = (2, 3, 2) if batch_first else (3, 2, 2)
        reference = torch.empty(shape if batched else (3, 2), device=device)

        def reshape(value, template):
            return value.reshape_as(template)

        operation = reshape if backend is None else torch.compile(reshape, backend=backend, fullgraph=True)
        output = operation(source, reference)
        torch.testing.assert_close(output.unbind(0 if batch_first else 1), [part.reshape(3, 2) for part in parts])
        torch.testing.assert_close(
            torch.autograd.grad(output.concat.square().sum(), parts), [2 * part for part in parts]
        )

    @pytest.mark.parametrize("transposed", [False, True])
    def test_view_as(self, device, transposed):
        parts = [torch.arange(n * n, device=device, dtype=torch.float64).requires_grad_() for n in (2, 3)]
        source = NT(parts, ragged_dims=(0,))
        reference = NT([torch.empty(n, n, device=device) for n in (2, 3)], ragged_dims=(0, 1))
        if transposed:
            reference = reference.transpose(1, 2)
        output = source.view_as(reference)
        expected = tuple(part.view(n, n) for part, n in zip(parts, (2, 3)))
        torch.testing.assert_close(output.unbind(0), expected)
        native = torch.ops.aten.view_as.default(self=source, other=reference)
        torch.testing.assert_close(native.unbind(0), expected)
        torch.testing.assert_close(torch.autograd.grad(output.square().sum(), parts), tuple(2 * part for part in parts))
        with torch.no_grad():
            source.concat.add_(7)
            torch.testing.assert_close(output.concat, source.concat)

    def test_view_strides(self, device):
        uniform = NT([torch.arange(6.0, device=device), torch.arange(6.0, device=device)], ragged_dims=(0,))
        dense_reference = torch.empty(3, 2, device=device)
        torch.testing.assert_close(
            uniform.view_as(dense_reference).unbind(0),
            tuple(part.view_as(dense_reference) for part in uniform.unbind(0)),
        )
        parts = [torch.arange(n * 3.0, device=device).reshape(n, 3) for n in (2, 4)]
        contiguous = NT(parts, ragged_dims=(0,))
        viewed = contiguous.view(1, -1)
        torch.testing.assert_close(viewed.unbind(0), tuple(part.view(1, -1) for part in parts))
        contiguous.concat.add_(7)
        torch.testing.assert_close(viewed.unbind(0), tuple((part + 7).view(1, -1) for part in parts))
        source = NT(parts, ragged_dims=(0,)).transpose(1, 2)
        with pytest.raises(RuntimeError, match="view size is not compatible"):
            source.view(-1)
        torch.testing.assert_close(source.reshape(-1).unbind(0), tuple(part.T.reshape(-1) for part in parts))

    @pytest.mark.parametrize("backend", [None, "aot_eager"])
    @pytest.mark.parametrize("native", [False, True])
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("transposed", [True, False])
    def test_reshape_as(self, device, batch_first, transposed, native, backend):
        torch.compiler.reset()
        parts = [torch.randn(n * n, device=device, dtype=torch.float64, requires_grad=True) for n in (2, 3)]
        source = NT(parts, ragged_dims=(0,), batch_first=batch_first, padding_value=-1, mask_value=False)
        reference = NT(
            [torch.empty(n, n, device=device) for n in (2, 3)],
            ragged_dims=(0, 1),
            batch_first=batch_first,
            padding_value=99,
            mask_value=True,
        )
        if transposed:
            reference = reference.transpose(1, 2) if batch_first else reference.transpose(0, 2)

        def reshape(value, template):
            if native:
                return torch.ops.aten.reshape_as.default(self=value, other=template)
            return value.reshape_as(template)

        operation = reshape if backend is None else torch.compile(reshape, backend=backend, fullgraph=True)
        output = operation(source, reference)
        expected = [part.reshape(n, n) for part, n in zip(parts, (2, 3))]
        torch.testing.assert_close(output.unbind(0 if batch_first else 1), expected)
        assert output.ragged_dims == reference.ragged_dims
        assert output.padding_value == source.padding_value
        assert output.mask_value == source.mask_value
        gradients = torch.autograd.grad(output.square().sum(), parts)
        torch.testing.assert_close(gradients, [2 * part for part in parts])
        with torch.inference_mode():
            torch.testing.assert_close(source.reshape_as(reference).unbind(0 if batch_first else 1), expected)
        with pytest.raises(RuntimeError, match="same number"):
            NT([torch.ones(3), torch.ones(9)], ragged_dims=(0,)).reshape_as(reference)
        wrong = NT(
            [torch.ones(length, device=device, dtype=torch.float64) for length in (5, 8)],
            ragged_dims=(0,),
            batch_first=batch_first,
        )
        with pytest.raises(RuntimeError, match="same number"):
            operation(wrong, reference)

    def test_flatten_and_unflatten_round_trip(self):
        nt = NT(
            [
                torch.arange(8, dtype=torch.float32).reshape(2, 2, 2),
                torch.arange(8, 16, dtype=torch.float32).reshape(2, 2, 2),
            ]
        )
        flattened = torch.flatten(nt, start_dim=1)
        reference = torch.flatten(nt.tensor, start_dim=1)
        assert_close(flattened, reference)
        unflattened = torch.unflatten(flattened, dim=1, sizes=(2, 2, 2))
        assert_close(unflattened, nt)

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize(
        "spatial_sizes",
        [((2, 4), (2, 4)), ((2, 4), (3, 4)), ((2, 6), (5, 2))],
        ids=["equal", "unequal", "different-axis-maxima"],
    )
    def test_flatten_ragged_spatial_axes_preserves_structure_and_vjp(self, device, batch_first, spatial_sizes):
        parts = [
            torch.randn(3, height, width, device=device, dtype=torch.float64, requires_grad=True)
            for height, width in spatial_sizes
        ]
        source = NT(parts, ragged_dims=(1, 2), batch_first=batch_first)
        flattened = source.flatten(2)
        max_tokens = max(height * width for height, width in spatial_sizes)
        expected_shape = (2, 3, max_tokens) if batch_first else (3, 2, max_tokens)
        assert flattened.shape == expected_shape
        assert flattened.ragged_dims == (1,)
        tokens = flattened.transpose(1, 2) if batch_first else flattened.transpose(0, 2)
        positions = [
            torch.randn(height * width, 3, device=device, dtype=torch.float64) for height, width in spatial_sizes
        ]
        position = NT(positions, ragged_dims=(0,), batch_first=batch_first)
        output = tokens + position
        expected = [part.flatten(1).T + position for part, position in zip(parts, positions)]
        assert output.ragged_dims == (0,)
        torch.testing.assert_close(output.unbind(0 if batch_first else 1), expected)
        gradients = torch.autograd.grad(output.concat.square().sum(), parts)
        expected_gradients = torch.autograd.grad(sum(value.square().sum() for value in expected), parts)
        torch.testing.assert_close(gradients, expected_gradients)

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("ragged_rank", [2, 3])
    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    def test_dynamic_flatten(self, device, batch_first, ragged_rank, backend):
        torch.compiler.reset()
        compiled = torch.compile(lambda value: value.flatten(2), backend=backend, fullgraph=True, dynamic=True)
        shapes = [((2, 5), (4, 3)), ((4, 5), (1, 2))]
        for spatial_shapes in shapes:
            parts = [
                torch.randn(3, *spatial, *((2,) if ragged_rank == 3 else ()), device=device, requires_grad=True)
                for spatial in spatial_shapes
            ]
            source = NT(parts, batch_first=batch_first, ragged_dims=tuple(range(1, ragged_rank + 1)))
            output = compiled(source)
            expected = [part.flatten(1) for part in parts]
            expected_shape = (2, 3, max(part.shape[-1] for part in expected))
            if not batch_first:
                expected_shape = (expected_shape[1], expected_shape[0], expected_shape[2])
            assert tuple(output.shape) == expected_shape
            torch.testing.assert_close(output.unbind(0 if batch_first else 1), tuple(expected))
            gradients = torch.autograd.grad(output.concat.square().sum(), parts)
            expected_gradients = torch.autograd.grad(sum(part.square().sum() for part in expected), parts)
            torch.testing.assert_close(gradients, expected_gradients)

    def test_flatten_static_dims_after_multi_ragged_prefix(self):
        nt = NT(
            [
                torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5),
                torch.arange(3 * 2 * 4 * 5, dtype=torch.float32).reshape(3, 2, 4, 5),
            ]
        )
        output = torch.flatten(nt, start_dim=3)
        reference = NT([torch.flatten(t, start_dim=2) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    @pytest.mark.parametrize("transposed", [False, True], ids=["canonical", "transposed"])
    def test_static_tail_flatten_unflatten_values_and_vjp(self, transposed):
        source = NestedTensor(
            [torch.empty(2, 2, 2, 4), torch.empty(3, 3, 2, 4)],
            ragged_dims=(0, 1),
        )
        if transposed:
            source = source.transpose(1, 2)
        leaf = torch.randn_like(source.concat, requires_grad=True)
        source = source.packed_like(leaf)

        flattened = source.flatten(-2)
        restored = flattened.unflatten(-1, (2, 4))

        assert flattened.ragged_dims == source.ragged_dims
        assert restored.shape == source.shape
        assert restored.ragged_dims == source.ragged_dims
        assert_close(flattened.concat, leaf.flatten(-2))
        assert_close(restored.concat, leaf)
        restored.concat.square().sum().backward()
        assert_close(leaf.grad, 2 * leaf)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_static_tail_flatten_unflatten_compiles(self):
        template = NestedTensor(
            [torch.empty(2, 2, 2, 4), torch.empty(3, 3, 2, 4)],
            ragged_dims=(0, 1),
        )
        values = torch.randn_like(template.concat, requires_grad=True)
        compiled = torch.compile(
            lambda structure, packed: structure.packed_like(packed)
            .flatten(-2)
            .unflatten(-1, (2, 4))
            .concat.square()
            .sum(),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        loss = compiled(template, values)
        gradient = torch.autograd.grad(loss, values)[0]

        assert_close(loss, values.square().sum())
        assert_close(gradient, 2 * values)

    def test_flatten_all(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([[1, 2], [3, 4]], device=device, dtype=float_dtype),
                torch.tensor([[5, 6], [7, 8]], device=device, dtype=float_dtype),
            ]
        )
        output = torch.flatten(nt, start_dim=0)
        assert isinstance(output, torch.Tensor)
        assert_close(output, torch.flatten(nt.tensor, start_dim=0))

    def test_flatten_ragged_2d_batch_and_first_dim_to_rows(self):
        # flatten(0, 1) on a ragged_rank>=2 NT collapses (batch + first ragged dim) into the batch
        # while a ragged dim remains -> an NT of sum(L_i) rows (each (M_i, ...)), the "rows of the grid"
        # view (e.g. for a packed row LSTM), instead of densifying to the padded max length.
        grid = NestedTensor([torch.randn(4, 4, 3), torch.randn(6, 6, 3)])
        rows = grid.flatten(0, 1)
        assert len(rows.ragged_dims) == 1
        assert [tuple(x.shape) for x in rows] == [(4, 3)] * 4 + [(6, 3)] * 6
        ref = [r for i in range(len(grid)) for r in grid[i].unbind(0)]
        assert all(torch.equal(rows[k], ref[k]) for k in range(len(ref)))

    def test_flatten_batch_first_false_rows(self):
        grid = NestedTensor([torch.randn(4, 4, 3), torch.randn(6, 6, 3)], batch_first=False)
        rows = grid.flatten(0, 1)
        assert rows.batch_first is True
        assert rows.shape == torch.Size((10, 6, 3))
        assert [tuple(x.shape) for x in rows] == [(4, 3), (6, 3)] * 4 + [(6, 3)] * 2
        ref = []
        for row in range(6):
            ref.extend(grid[batch][row] for batch in range(2) if row < grid[batch].shape[0])
        assert all(torch.equal(rows[index], ref[index]) for index in range(len(ref)))

    def test_flatten_batch_middle(self):
        parts = [torch.randn(n, n, 3, requires_grad=True) for n in (4, 6)]
        grid = NestedTensor(parts, batch_first=False)
        output = grid.flatten(1, 2)
        expected = [part[:, column] for part in parts for column in range(part.shape[1])]
        assert isinstance(output, NestedTensor) and not output.batch_first
        torch.testing.assert_close(output.unbind(1), expected)
        torch.testing.assert_close(torch.autograd.grad(output.square().sum(), parts), [2 * part for part in parts])

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_unflatten_batch(self, device, batch_first):
        parts = [torch.randn(n, n, device=device, dtype=torch.float64, requires_grad=True) for n in (2, 2, 3, 3)]
        source = NT(parts, ragged_dims=(0, 1), batch_first=batch_first)
        dim = 0 if batch_first else 1
        output = source.unflatten(dim, (2, -1))
        member_dim = 0 if batch_first else 1
        expected = [torch.stack(parts[start : start + 2], dim=member_dim) for start in (0, 2)]
        torch.testing.assert_close(output.unbind(dim), expected)
        native = torch.ops.aten.unflatten.int(self=source, dim=dim, sizes=[2, -1])
        torch.testing.assert_close(native.unbind(dim), expected)
        assert output.ragged_dims == ((1, 2) if batch_first else (0, 2))
        gradients = torch.autograd.grad(output.square().sum(), parts)
        torch.testing.assert_close(gradients, [2 * part for part in parts])
        with torch.inference_mode():
            torch.testing.assert_close(source.unflatten(dim, (2, -1)).unbind(dim), expected)
        with pytest.raises(ValueError, match="within each"):
            source.unflatten(dim, (1, 4))
        with pytest.raises(RuntimeError):
            source.unflatten(dim, (3, 2))

    @pytest.mark.parametrize("sizes", [(0, 2), (2, 0)])
    def test_unflatten_empty(self, sizes):
        output = NT([]).unflatten(0, sizes)
        assert output.shape == sizes
        assert output.numel() == 0


class TestExpand:

    def test_expand_as_uses_target_elements_with_mask_and_fullgraph_vjp(self, device):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")

        def masked_expand(template, values, target, mask):
            source = template.packed_like(values).expand_as(target)
            return torch.where(mask.expand_as(target), source, 0).concat

        compiled = torch.compile(masked_expand, backend="aot_eager", fullgraph=True, dynamic=True)
        for sizes in (((3, 2), (5, 4)), ((2, 4), (4, 3))):
            parts = [torch.randn(rows, 1, 1, device=device, dtype=torch.float64) for rows, _ in sizes]
            template = NT(parts, ragged_dims=(0,))
            values = template.concat.detach().requires_grad_()
            reference_parts = [part.detach().requires_grad_() for part in parts]
            target_parts = [torch.empty(rows, columns, 1, device=device) for rows, columns in sizes]
            target = NT(target_parts, ragged_dims=(0, 1))
            masks = [torch.rand(part.shape, device=device) > 0.4 for part in target_parts]
            mask = NT(masks, ragged_dims=(0, 1))
            dense_mask = torch.ones(len(target), 1, 1, 1, device=device, dtype=torch.bool)
            with fake_tensor_mod.FakeTensorMode(allow_non_fake_inputs=True):
                mask.expand_as(target)
                dense_mask.expand_as(target)
            reference = NT(
                [
                    torch.where(valid, part.expand_as(destination), 0)
                    for part, destination, valid in zip(reference_parts, target_parts, masks)
                ],
                ragged_dims=(0, 1),
            )
            for operation in (masked_expand, compiled):
                output = operation(template, values, target, mask)
                assert_close(output, reference.concat)
                weights = torch.randn_like(output)
                gradient = torch.autograd.grad(output, values, weights)[0]
                expected = torch.autograd.grad(reference.concat, reference_parts, weights, retain_graph=True)
                assert_close(gradient, NT(expected, ragged_dims=(0,)).concat)

    def test_expand_static_singleton_dim(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(4, 1, 3, device=device, dtype=float_dtype),
                torch.randn(6, 1, 3, device=device, dtype=float_dtype),
            ]
        )
        output = nt.expand(-1, -1, 4, -1)
        reference = NT([t.expand(-1, 4, -1) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_expand_sample_axis(self, device, float_dtype):
        parts = [torch.randn(length, 3, device=device, dtype=float_dtype, requires_grad=True) for length in (3, 5)]
        reference_parts = [part.detach().clone().requires_grad_() for part in parts]
        nt = NT(parts)

        output = nt.unsqueeze(-3).expand(-1, 4, nt.size(-2), -1)
        reference = [part.unsqueeze(0).expand(4, -1, -1) for part in reference_parts]
        for actual, expected in zip(output, reference):
            assert_close(actual, expected)

        weights = [torch.randn_like(element) for element in reference]
        loss = sum((element * weight).sum() for element, weight in zip(output, weights))
        reference_loss = sum((element * weight).sum() for element, weight in zip(reference, weights))
        gradients = torch.autograd.grad(loss, parts)
        reference_gradients = torch.autograd.grad(reference_loss, reference_parts)
        for actual, expected in zip(gradients, reference_gradients):
            assert_close(actual, expected)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_expand_sample_axis_compiles_with_vjp(self, device):
        def consume(template, values):
            reference_positions = template.packed_like(values)
            sampled = reference_positions.unsqueeze(-3).expand(-1, 4, reference_positions.size(-2), -1)
            return sampled.concat.square().sum()

        compiled = torch.compile(consume, backend="aot_eager", fullgraph=True, dynamic=True)
        template = NT([torch.empty(length, 3, device=device) for length in (2, 3)], ragged_dims=(0,))
        values = torch.randn_like(template.concat, requires_grad=True)
        loss = compiled(template, values)
        gradient = torch.autograd.grad(loss, values)[0]
        assert_close(loss, values.square().sum() * 4)
        assert_close(gradient, 8 * values)

    def test_expand_rejects_ragged_change(self):
        reference_positions = NT([torch.empty(1, 3), torch.empty(2, 3)], ragged_dims=(0,))
        sampled = reference_positions.unsqueeze(-3)
        with pytest.raises(RuntimeError, match="cannot change a ragged dimension"):
            sampled.expand(-1, 4, 7, -1)

    def test_expand_rejects_non_singleton_static_change(self):
        nt = NT([torch.empty(2, 2, 3), torch.empty(3, 2, 3)], ragged_dims=(0,))
        with pytest.raises(RuntimeError):
            nt.expand(-1, -1, 4, -1)

    @pytest.mark.parametrize("batch", [1, 3])
    def test_expand_rejects_batch_change(self, batch):
        nt = NT([torch.randn(4, 3), torch.randn(6, 3)])
        with pytest.raises(RuntimeError):
            nt.expand(batch, -1, -1)


class TestFlipAndRoll:

    def test_flip_translates_dims_and_rejects_batch(self, device):
        nt = NT([torch.tensor([[1, 2], [3, 4]], device=device), torch.tensor([[5, 6]], device=device)])
        output = torch.flip(nt, dims=(-1,))
        reference = NT([torch.flip(t, dims=(-1,)) for t in nt], **reference_options(nt))
        assert_close(output, reference)

        with pytest.raises(ValueError):
            torch.flip(nt, dims=(0,))
        with pytest.raises(ValueError):
            torch.flip(nt, dims=(-3,))

    def test_flip_ragged_last_dim(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(3, 4, device=device, dtype=float_dtype),
                torch.randn(3, 7, device=device, dtype=float_dtype),
            ]
        )
        output = torch.flip(nt, dims=(-1,))
        reference = NT([torch.flip(t, dims=(-1,)) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_flip_ragged_2d_dims(self, device, float_dtype):
        # ragged_rank == 2 (per-sample LxLxC grids): flipping a ragged dim must reverse that
        # per-element axis, not the packed channel axis. Regression for a silent no-op where the
        # element dim was used directly as the packed _values axis (only valid for ragged_rank==1).
        nt = NT(
            [
                torch.randn(3, 3, 2, device=device, dtype=float_dtype),
                torch.randn(5, 5, 2, device=device, dtype=float_dtype),
            ]
        )
        for nt_dim, elem_dim in ((1, 0), (2, 1), (3, 2)):
            output = torch.flip(nt, dims=(nt_dim,))
            reference = NT([torch.flip(t, dims=(elem_dim,)) for t in nt], **reference_options(nt))
            assert_close(output, reference)

    def test_roll_supports_dims_none(self, device):
        nt = NT([torch.tensor([1, 2, 3], device=device), torch.tensor([4, 5], device=device)])
        output = torch.roll(nt, shifts=1)
        reference = NT([torch.roll(t, shifts=1) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_roll_translates_dims_and_rejects_batch(self, device):
        nt = NT([torch.tensor([[1, 2], [3, 4]], device=device), torch.tensor([[5, 6]], device=device)])
        output = torch.roll(nt, shifts=1, dims=-1)
        reference = NT([torch.roll(t, shifts=1, dims=-1) for t in nt], **reference_options(nt))
        assert_close(output, reference)

        with pytest.raises(ValueError):
            torch.roll(nt, shifts=1, dims=0)

    def test_rot90_values_and_vjp(self, device, float_dtype):
        leaves = [
            torch.randn(2, 1, 3, device=device, dtype=float_dtype, requires_grad=True),
            torch.randn(3, 1, 3, device=device, dtype=float_dtype, requires_grad=True),
        ]
        references = [leaf.detach().clone().requires_grad_() for leaf in leaves]
        output = torch.rot90(NT(leaves), 1, (2, 3))
        expected = [torch.rot90(reference, 1, (1, 2)) for reference in references]
        weights = [torch.randn_like(element) for element in expected]
        actual_loss = sum((element * weight).sum() for element, weight in zip(output, weights))
        expected_loss = sum((element * weight).sum() for element, weight in zip(expected, weights))

        actual_gradients = torch.autograd.grad(actual_loss, leaves)
        expected_gradients = torch.autograd.grad(expected_loss, references)
        for actual, reference in zip(output, expected):
            assert_close(actual, reference)
        for actual, reference in zip(actual_gradients, expected_gradients):
            assert_close(actual, reference)


class TestIndexingReadOps:

    def test_index_select(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device, dtype=float_dtype),
                torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], device=device, dtype=float_dtype),
            ]
        )
        output = torch.index_select(nt, 0, torch.tensor([1, 0, 1], device=device))
        reference = NT([nt[1], nt[0], nt[1]], **reference_options(nt))
        assert_close(output, reference)

        output = torch.index_select(nt, 2, torch.tensor([2, 0], device=device))
        reference = NT(
            [torch.index_select(t, 1, torch.tensor([2, 0], device=device)) for t in nt], **reference_options(nt)
        )
        assert_close(output, reference)

        output = torch.index_select(nt, 1, torch.tensor([1, 0], device=device))
        reference = NT(
            [torch.index_select(t, 0, torch.tensor([1, 0], device=device)) for t in nt], **reference_options(nt)
        )
        assert_close(output, reference)

    def test_index_select_batch_first_false(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=float_dtype),
                torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device, dtype=float_dtype),
            ],
            batch_first=False,
        )
        output = torch.index_select(nt, 1, torch.tensor([1, 0, 1], device=device))
        reference = NT([nt[1], nt[0], nt[1]], **reference_options(nt))
        assert_close(output, reference)

    def test_masked_select(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        mask = nt > 2
        output = torch.masked_select(nt, mask)
        reference = NT([torch.masked_select(t, m) for t, m in zip(nt, mask)], **reference_options(nt))
        assert_close(output, reference)

        output = torch.masked_select(nt, mask.tensor)
        assert_close(output, reference)

        output = torch.masked_select(nt, torch.tensor(True, device=device))
        reference = NT([t.reshape(-1) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_masked_select_multi_ragged_exact_layout(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([[1.0, 0.0, 3.0], [0.0, 5.0, 6.0]], device=device, dtype=float_dtype),
                torch.tensor([[7.0, 0.0, 9.0, 10.0]], device=device, dtype=float_dtype),
            ]
        )
        mask = nt > 0
        output = torch.masked_select(nt, mask)
        reference = NT([torch.masked_select(t, m) for t, m in zip(nt, mask)], **reference_options(nt))
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "take_along_dim"), reason="requires torch.take_along_dim")
    def test_take_along_dim(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], device=device, dtype=float_dtype),
                torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=float_dtype),
            ]
        )
        indices = torch.tensor(
            [
                [[0, 2, 1], [2, 1, 0]],
                [[1, 1, 1], [0, 0, 0]],
            ],
            device=device,
            dtype=torch.long,
        )
        output = torch.take_along_dim(nt, indices, dim=2)
        indices_nt = nt.nested_like(indices, strict=False)
        reference = NT([torch.take_along_dim(t, i, dim=1) for t, i in zip(nt, indices_nt)], **reference_options(nt))
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "take_along_dim"), reason="requires torch.take_along_dim")
    def test_take_along_dim_with_nested_indices_ragged_dim(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], device=device, dtype=float_dtype),
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device=device, dtype=float_dtype),
            ]
        )
        indices = NT(
            [
                torch.tensor([[1, 1, 0]], device=device, dtype=torch.long),
                torch.tensor([[2, 2, 2], [0, 1, 0]], device=device, dtype=torch.long),
            ]
        )
        output = torch.take_along_dim(nt, indices, dim=1)
        reference = NT([torch.take_along_dim(t, idx, dim=0) for t, idx in zip(nt, indices)], **reference_options(nt))
        assert_close(output, reference)

    def test_multi_ragged_per_sample_row_selection(self, device, float_dtype):
        elements = [
            torch.randn(4, 3, 2, device=device, dtype=float_dtype),
            torch.randn(3, 2, 2, device=device, dtype=float_dtype),
        ]
        rows = [torch.tensor([3, 1], device=device), torch.tensor([2], device=device)]
        indices = [
            row[:, None, None].expand(-1, element.shape[1], element.shape[2]) for row, element in zip(rows, elements)
        ]
        nested = NT(elements, ragged_dims=(0, 1))
        nested_indices = NT(indices, ragged_dims=(0, 1))

        output = torch.gather(nested, 1, nested_indices)
        references = [torch.gather(element, 0, index) for element, index in zip(elements, indices)]

        assert output.ragged_dims == (0, 1)
        assert [element.shape for element in output] == [reference.shape for reference in references]
        for actual, expected in zip(output, references):
            assert_close(actual, expected)


class TestIndexingWriteOps:

    def test_index_add(self, device, float_dtype):
        base = NestedTensor(
            [torch.zeros(5, device=device, dtype=float_dtype), torch.zeros(4, device=device, dtype=float_dtype)]
        )
        index = torch.tensor([0, 2], device=device)
        src = NestedTensor(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0, 4.0], device=device, dtype=float_dtype),
            ],
            **reference_options(base),
        )
        output = torch.index_add(base, 1, index, src)
        reference = NT([torch.index_add(t, 0, index, s) for t, s in zip(base, src)], **reference_options(base))
        assert_close(output, reference)

    def test_index_add_static_dim(self, device, float_dtype):
        base = NestedTensor(
            [
                torch.zeros(2, 4, device=device, dtype=float_dtype),
                torch.zeros(3, 4, device=device, dtype=float_dtype),
            ]
        )
        index = torch.tensor([0, 2], device=device)
        src = NestedTensor(
            [
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=float_dtype),
                torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], device=device, dtype=float_dtype),
            ],
            **reference_options(base),
        )
        output = torch.index_add(base, 2, index, src)
        reference = NT([torch.index_add(t, 1, index, s) for t, s in zip(base, src)], **reference_options(base))
        assert_close(output, reference)

    def test_index_copy(self, device, float_dtype):
        base = NestedTensor(
            [torch.zeros(5, device=device, dtype=float_dtype), torch.zeros(4, device=device, dtype=float_dtype)]
        )
        index = torch.tensor([0, 2], device=device)
        src = NestedTensor(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0, 4.0], device=device, dtype=float_dtype),
            ],
            **reference_options(base),
        )
        output = torch.index_copy(base, 1, index, src)
        reference = NT([torch.index_copy(t, 0, index, s) for t, s in zip(base, src)], **reference_options(base))
        assert_close(output, reference)

    def test_index_copy_static_dim(self, device, float_dtype):
        base = NestedTensor(
            [
                torch.zeros(2, 4, device=device, dtype=float_dtype),
                torch.zeros(3, 4, device=device, dtype=float_dtype),
            ]
        )
        index = torch.tensor([0, 2], device=device)
        src = NestedTensor(
            [
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=float_dtype),
                torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], device=device, dtype=float_dtype),
            ],
            **reference_options(base),
        )
        output = torch.index_copy(base, 2, index, src)
        reference = NT([torch.index_copy(t, 1, index, s) for t, s in zip(base, src)], **reference_options(base))
        assert_close(output, reference)

    def test_index_put(self, device, float_dtype):
        base = NestedTensor(
            [torch.zeros(5, device=device, dtype=float_dtype), torch.zeros(4, device=device, dtype=float_dtype)]
        )
        index = torch.tensor([0, 2], device=device)
        values = NestedTensor(
            [
                torch.tensor([10.0, 20.0], device=device, dtype=float_dtype),
                torch.tensor([30.0, 40.0], device=device, dtype=float_dtype),
            ],
            **reference_options(base),
        )
        output = torch.index_put(base, (index,), values, accumulate=False)
        reference = NT([torch.index_put(t, (index,), v) for t, v in zip(base, values)], **reference_options(base))
        assert_close(output, reference)

    def test_index_put_multi_index(self, device, float_dtype):
        base = NestedTensor(
            [
                torch.zeros(4, 3, device=device, dtype=float_dtype),
                torch.zeros(5, 3, device=device, dtype=float_dtype),
            ]
        )
        rows = torch.tensor([[0, -1], [1, 2]], device=device)
        cols = torch.tensor([[1, 0], [2, 1]], device=device)
        values = NestedTensor(
            [
                torch.tensor([[13.0, 17.0], [19.0, 23.0]], device=device, dtype=float_dtype),
                torch.tensor([[29.0, 31.0], [37.0, 41.0]], device=device, dtype=float_dtype),
            ],
            **reference_options(base),
        )
        output = torch.index_put(base, (rows, cols), values, accumulate=False)
        reference = NT([torch.index_put(t, (rows, cols), v) for t, v in zip(base, values)], **reference_options(base))
        assert_close(output, reference)

    def test_index_put_matches_dense(self, device):
        seed = 0
        dtype = torch.float32
        shapes = ragged_shapes(seed, batch_size=3, min_len=3, max_len=6, trailing_shape=(4,))
        base = nested_rand(shapes, device, dtype)
        min_rows = min(shape[0] for shape in shapes)
        generator = torch.Generator()
        generator.manual_seed(seed)

        rows = torch.randint(0, min_rows, (2,), generator=generator)
        rows[-1] = rows[-1] - min_rows
        rows = rows.to(device=device, dtype=torch.long)

        shared_values = torch.randn(2, 4, device=device, dtype=dtype)
        row_output = torch.index_put(base, (rows,), shared_values, accumulate=False)
        row_reference = NT([torch.index_put(t, (rows,), shared_values) for t in base], **reference_options(base))
        assert_close(row_output, row_reference)

        dup_rows = rows.clone()
        dup_rows[0] = dup_rows[-1]
        dup_values = torch.randn(2, 4, device=device, dtype=dtype)
        dup_output = torch.index_put(base, (dup_rows,), dup_values, accumulate=True)
        dup_reference = NT(
            [torch.index_put(t, (dup_rows,), dup_values, accumulate=True) for t in base], **reference_options(base)
        )
        assert_close(dup_output, dup_reference)

        point_rows = torch.randint(0, min_rows, (2, 2), generator=generator)
        point_rows[0, 0] = point_rows[0, 0] - min_rows
        point_rows = point_rows.to(device=device, dtype=torch.long)
        point_cols = torch.randint(0, 4, (2, 2), generator=generator).to(device=device, dtype=torch.long)
        point_values = NestedTensor(
            [torch.randn(2, 2, device=device, dtype=dtype) for _ in range(len(base))],
            **reference_options(base),
        )
        point_output = torch.index_put(base, (point_rows, point_cols), point_values, accumulate=False)
        point_reference = NT(
            [torch.index_put(t, (point_rows, point_cols), v) for t, v in zip(base, point_values)],
            **reference_options(base),
        )
        assert_close(point_output, point_reference)

    def test_index_put_row_write(self, device, float_dtype):
        base = NestedTensor(
            [
                torch.zeros(4, 3, device=device, dtype=float_dtype),
                torch.zeros(5, 3, device=device, dtype=float_dtype),
            ]
        )
        index = torch.tensor([0, 2], device=device)
        values = NestedTensor(
            [
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device, dtype=float_dtype),
                torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], device=device, dtype=float_dtype),
            ],
            **reference_options(base),
        )
        output = torch.index_put(base, (index,), values, accumulate=False)
        reference = NT([torch.index_put(t, (index,), v) for t, v in zip(base, values)], **reference_options(base))
        assert_close(output, reference)

    def test_index_put_row_write_accumulate_duplicate_indices(self, device, float_dtype):
        base = NestedTensor(
            [
                torch.zeros(4, 3, device=device, dtype=float_dtype),
                torch.zeros(5, 3, device=device, dtype=float_dtype),
            ]
        )
        index = torch.tensor([0, 0, 2], device=device)
        values = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [10.0, 20.0, 30.0],
                [4.0, 5.0, 6.0],
            ],
            device=device,
            dtype=float_dtype,
        )
        output = torch.index_put(base, (index,), values, accumulate=True)
        reference = NT([torch.index_put(t, (index,), values, accumulate=True) for t in base], **reference_options(base))
        assert_close(output, reference)

    def test_index_put_row_write_dense_values(self, device, float_dtype):
        base = NestedTensor(
            [
                torch.zeros(4, 3, device=device, dtype=float_dtype),
                torch.zeros(5, 3, device=device, dtype=float_dtype),
            ]
        )
        index = torch.tensor([0, 2], device=device)
        values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device, dtype=float_dtype)
        output = torch.index_put(base, (index,), values, accumulate=False)
        reference = NT([torch.index_put(t, (index,), values) for t in base], **reference_options(base))
        assert_close(output, reference)

    def test_index_put_row_write_scalar_tensor(self, device, float_dtype):
        base = NestedTensor(
            [
                torch.zeros(4, 3, device=device, dtype=float_dtype),
                torch.zeros(5, 3, device=device, dtype=float_dtype),
            ]
        )
        index = torch.tensor([0, 2], device=device)
        value = torch.tensor(-3.0, device=device, dtype=float_dtype)
        output = torch.index_put(base, (index,), value, accumulate=False)
        reference = NT([torch.index_put(t, (index,), value) for t in base], **reference_options(base))
        assert_close(output, reference)

    def test_masked_fill(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, -2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([-4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        mask = nt > 0
        output = torch.masked_fill(nt, mask, 0.0)
        reference = NT([torch.masked_fill(t, m, 0.0) for t, m in zip(nt, mask)], **reference_options(nt))
        assert_close(output, reference)

        output = torch.masked_fill(nt, mask.tensor, 0.0)
        assert_close(output, reference)

        tensor_value = torch.tensor(0.0, device=device, dtype=float_dtype)
        output = torch.masked_fill(nt, mask, tensor_value)
        assert_close(output, reference)

    def test_masked_fill_broadcast_mask(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(3, 4, device=device, dtype=float_dtype),
                torch.randn(5, 4, device=device, dtype=float_dtype),
            ]
        )
        nested_mask = NT([t[:, :1] > 0 for t in nt], **reference_options(nt))
        output = torch.masked_fill(nt, nested_mask, -7.0)
        reference = NT([torch.masked_fill(t, t[:, :1] > 0, -7.0) for t in nt], **reference_options(nt))
        assert_close(output, reference)

        padded = nt.tensor
        lengths = [t.shape[0] for t in nt]
        for shape in ((2, 1, 4), (1, 5, 4), (5, 4), (2, 5, 1)):
            mask = torch.rand(*shape, device=device) > 0.5
            output = torch.masked_fill(nt, mask, -9.0)
            reference = padded.masked_fill(mask, -9.0)
            for index, length in enumerate(lengths):
                assert_close(output[index], reference[index, :length])

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_masked_fill_multi_ragged_singleton_mask_fullgraph(self, device, float_dtype):
        shapes = ((2, 3, 4), (3, 2, 4))
        template = NT([torch.empty(shape, device=device, dtype=float_dtype) for shape in shapes], ragged_dims=(0, 1))
        values = torch.randn_like(template.concat, requires_grad=True)
        nested = template.packed_like(values)
        masks = [torch.rand(*shape[:-1], 1, device=device) > 0.5 for shape in shapes]
        mask = NT(masks, ragged_dims=(0, 1))

        eager = torch.masked_fill(nested, mask, -7.0)
        compiled = torch.compile(
            lambda input_, condition: torch.masked_fill(input_, condition, -7.0).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        output = compiled(nested, mask)
        packed_lengths = tuple(shape[0] * shape[1] for shape in shapes)
        elements = [part.reshape(shape) for part, shape in zip(values.split(packed_lengths), shapes)]
        reference = torch.cat(
            [
                torch.masked_fill(element, condition, -7.0).reshape(-1, shape[-1])
                for element, condition, shape in zip(elements, masks, shapes)
            ]
        )
        cotangent = torch.randn_like(reference)
        actual_gradient = torch.autograd.grad(output, values, cotangent, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(reference, values, cotangent)[0]

        assert eager.ragged_dims == (0, 1)
        assert eager.element_sizes().tolist() == [list(shape) for shape in shapes]
        assert_close(eager.concat, reference)
        assert_close(output, reference)
        assert_close(actual_gradient, expected_gradient)

    def test_masked_fill_dense_source_ragged_mask(self, device, float_dtype):
        bias = torch.randn(2, 4, 3, 3, device=device, dtype=float_dtype)
        mask = NT(
            [
                torch.rand(2, 2, device=device) > 0.5,
                torch.rand(3, 3, device=device) > 0.5,
            ]
        )
        output = torch.masked_fill(bias, mask, float("-inf"))
        reference = NT(
            [
                bias[index, :, : element.shape[0], : element.shape[1]].masked_fill(element, float("-inf"))
                for index, element in enumerate(mask)
            ],
            **reference_options(mask),
        )
        assert_close(output, reference)

        with pytest.raises(ValueError, match="batch length mismatch"):
            torch.masked_fill(torch.randn(5, 4, 3, 3, device=device, dtype=float_dtype), mask, -1.0)

    def test_masked_scatter(self, device, float_dtype):
        base = NestedTensor(
            [torch.zeros(3, device=device, dtype=float_dtype), torch.zeros(2, device=device, dtype=float_dtype)]
        )
        mask = NestedTensor(
            [torch.tensor([True, False, True], device=device), torch.tensor([False, True], device=device)]
        )
        src = NestedTensor(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.masked_scatter(base, mask, src)
        reference = NT([torch.masked_scatter(t, m, s) for t, m, s in zip(base, mask, src)], **reference_options(base))
        assert_close(output, reference)

    def test_masked_scatter_same_shape_dense_source(self, device, float_dtype):
        base = NestedTensor(
            [torch.zeros(3, device=device, dtype=float_dtype), torch.zeros(2, device=device, dtype=float_dtype)]
        )
        mask = NestedTensor(
            [torch.tensor([True, False, True], device=device), torch.tensor([False, True], device=device)]
        )
        dense_source = torch.tensor(
            [[1.0, 2.0, 9.0], [3.0, 8.0, 7.0]],
            device=device,
            dtype=float_dtype,
        )

        output = torch.masked_scatter(base, mask, dense_source)
        nested_source = base.nested_like(dense_source, strict=False)
        reference = NT(
            [torch.masked_scatter(t, m, s) for t, m, s in zip(base, mask, nested_source)],
            **reference_options(base),
        )
        assert_close(output, reference)


class TestInnerProducts:

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_pairwise_outer_and_addr(self, batch_first):
        left = NT([torch.arange(2.0), torch.arange(3.0)], batch_first=batch_first)
        right = NT([torch.arange(4.0), torch.arange(2.0)], batch_first=batch_first)
        outer = torch.outer(left, right)
        biased = torch.addr(torch.tensor(2.0), left, right, beta=3, alpha=2)
        for actual, added, first, second in zip(outer, biased, left, right):
            assert_close(actual, torch.outer(first, second))
            assert_close(added, torch.addr(torch.tensor(2.0), first, second, beta=3, alpha=2))

    def test_dot_both_nested_ragged(self, device, float_dtype):
        lhs = NT(
            [
                torch.randn(4, device=device, dtype=float_dtype),
                torch.randn(3, device=device, dtype=float_dtype),
            ]
        )
        rhs = NT(
            [
                torch.randn(4, device=device, dtype=float_dtype),
                torch.randn(3, device=device, dtype=float_dtype),
            ]
        )
        result = torch.dot(lhs, rhs)
        reference = NT([torch.dot(a, b) for a, b in zip(lhs, rhs)], **reference_options(lhs))
        assert_close(result, reference)

    def test_inner_both_nested(self, device, float_dtype):
        a1 = torch.randn(3, 4, device=device, dtype=float_dtype)
        a2 = torch.randn(5, 4, device=device, dtype=float_dtype)
        b1 = torch.randn(3, 4, device=device, dtype=float_dtype)
        b2 = torch.randn(5, 4, device=device, dtype=float_dtype)
        nt1 = NT([a1, a2])
        nt2 = NT([b1, b2])
        result = torch.inner(nt1, nt2)
        reference = NT([torch.inner(a1, b1), torch.inner(a2, b2)], **reference_options(nt1))
        assert_close(result, reference)

    def test_inner_nested_tensor_with_vector(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.inner(nt, w)
        reference = NT([torch.inner(a, w), torch.inner(b, w)], **reference_options(nt))
        assert_close(result, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_inner_nested_tensor_with_vector_compile_fullgraph(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        compiled = torch.compile(lambda x, y: torch.inner(x, y), backend="inductor", fullgraph=True)
        result = compiled(nt, w)
        reference = NT([torch.inner(a, w), torch.inner(b, w)], **reference_options(nt))
        assert_close(result, reference)

    def test_inner_vector_with_nested_tensor(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.inner(w, nt)
        reference = NT([torch.inner(w, a), torch.inner(w, b)], **reference_options(nt))
        assert_close(result, reference)

    @pytest.mark.skipif(not hasattr(torch, "vdot"), reason="torch.vdot not available")
    def test_vdot_both_nested(self, device, float_dtype):
        lhs = NT(
            [
                torch.randn(4, device=device, dtype=float_dtype),
                torch.randn(4, device=device, dtype=float_dtype),
            ]
        )
        rhs = NT(
            [
                torch.randn(4, device=device, dtype=float_dtype),
                torch.randn(4, device=device, dtype=float_dtype),
            ]
        )
        result = torch.vdot(lhs, rhs)
        reference = NT([torch.vdot(a, b) for a, b in zip(lhs, rhs)], **reference_options(lhs))
        assert_close(result, reference)

    @pytest.mark.skipif(not hasattr(torch, "vdot"), reason="torch.vdot not available")
    def test_vdot_both_nested_ragged(self, device, float_dtype):
        lhs = NT(
            [
                torch.randn(4, device=device, dtype=float_dtype),
                torch.randn(3, device=device, dtype=float_dtype),
            ]
        )
        rhs = NT(
            [
                torch.randn(4, device=device, dtype=float_dtype),
                torch.randn(3, device=device, dtype=float_dtype),
            ]
        )
        result = torch.vdot(lhs, rhs)
        reference = NT([torch.vdot(a, b) for a, b in zip(lhs, rhs)], **reference_options(lhs))
        assert_close(result, reference)


class TestLikeCreators:

    def test_empty_like_preserves_shape(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(1, 3, device=device, dtype=float_dtype),
            ]
        )
        output = torch.empty_like(nt, dtype=torch.float64)
        assert output.dtype == torch.float64
        assert [t.shape for t in output] == [t.shape for t in nt]

    def test_full_like_dtype(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(1, 3, device=device, dtype=float_dtype),
            ]
        )
        output = torch.full_like(nt, 3.0, dtype=torch.float32)
        reference = NT([torch.full_like(t, 3.0, dtype=torch.float32) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_ones_like_dtype(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(1, 3, device=device, dtype=float_dtype),
            ]
        )
        output = torch.ones_like(nt, dtype=torch.float64)
        reference = NT([torch.ones_like(t, dtype=torch.float64) for t in nt], **reference_options(nt))
        assert_close(output, reference)
        assert output.dtype == torch.float64

    def test_randint_like(self, device):
        nt = NT([torch.zeros(3, device=device), torch.zeros(1, device=device)])
        torch.manual_seed(1016)
        output = torch.randint_like(nt, 10)
        torch.manual_seed(1016)
        reference = NT([torch.randint_like(t, 10) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_zeros_like(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(1, 3, device=device, dtype=float_dtype),
            ]
        )

        output = torch.zeros_like(nt)
        reference = NT([torch.zeros_like(t) for t in nt], **reference_options(nt))
        assert_close(output, reference)


class TestLinalgOps:
    """Tests for torch.linalg ops registered in torch_functions.py."""

    def test_cross_static_tail_values_and_vjp(self, device, float_dtype):
        template = NT(
            [torch.empty(2, 3, device=device), torch.empty(5, 3, device=device)],
            ragged_dims=(0,),
        )
        lhs_values = torch.randn_like(template.concat, dtype=float_dtype, requires_grad=True)
        rhs_values = torch.randn_like(template.concat, dtype=float_dtype, requires_grad=True)
        lhs = template.packed_like(lhs_values)
        rhs = template.packed_like(rhs_values)
        reference = torch.cross(lhs_values, rhs_values, dim=-1)

        output = torch.cross(lhs, rhs, dim=-1)
        linalg_output = torch.linalg.cross(lhs, rhs, dim=-1)

        cotangent = torch.randn_like(reference)
        actual_gradients = torch.autograd.grad(output.concat, (lhs_values, rhs_values), cotangent)
        expected_gradients = torch.autograd.grad(reference, (lhs_values, rhs_values), cotangent)
        assert output.ragged_dims == (0,)
        assert_close(output.concat, reference)
        assert_close(linalg_output.concat, reference)
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
            assert_close(actual, expected)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_cross_static_tail_compiles_with_vjp(self, device):
        compiled = torch.compile(
            lambda template, lhs, rhs: torch.cross(
                template.packed_like(lhs),
                template.packed_like(rhs),
                dim=-1,
            ).concat,
            backend="aot_eager",
            fullgraph=True,
        )
        template = NT([torch.empty(length, 3, device=device) for length in (2, 5)], ragged_dims=(0,))
        lhs = torch.randn_like(template.concat, requires_grad=True)
        rhs = torch.randn_like(template.concat, requires_grad=True)
        reference = torch.cross(lhs, rhs, dim=-1)
        output = compiled(template, lhs, rhs)
        cotangent = torch.randn_like(reference)
        actual_gradients = torch.autograd.grad(output, (lhs, rhs), cotangent)
        expected_gradients = torch.autograd.grad(reference, (lhs, rhs), cotangent)
        assert_close(output, reference)
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
            assert_close(actual, expected)

    def test_linalg_cholesky(self, device, float_dtype):
        # Create positive-definite matrices
        a_raw = torch.randn(3, 3, device=device, dtype=float_dtype)
        a = a_raw @ a_raw.T + 3 * torch.eye(3, device=device, dtype=float_dtype)
        b_raw = torch.randn(4, 4, device=device, dtype=float_dtype)
        b = b_raw @ b_raw.T + 3 * torch.eye(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        pair = _run_or_expect_unsupported(
            lambda: torch.linalg.cholesky(nt),
            lambda: NT([torch.linalg.cholesky(a), torch.linalg.cholesky(b)], **reference_options(nt)),
        )
        if pair is None:
            return
        result, reference = pair
        assert_close(result, reference)

    def test_linalg_det(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        pair = _run_or_expect_unsupported(
            lambda: torch.linalg.det(nt), lambda: (torch.linalg.det(a), torch.linalg.det(b))
        )
        if pair is None:
            return
        result, reference = pair
        ref_a, ref_b = reference
        assert_close(result[0].squeeze(), ref_a)
        assert_close(result[1].squeeze(), ref_b)

    def test_linalg_eigh(self, device, float_dtype):
        # Create symmetric matrices
        a_raw = torch.randn(3, 3, device=device, dtype=float_dtype)
        a = a_raw + a_raw.T
        b_raw = torch.randn(4, 4, device=device, dtype=float_dtype)
        b = b_raw + b_raw.T
        nt = NT([a, b])
        try:
            w_a, _ = torch.linalg.eigh(a)
            w_b, _ = torch.linalg.eigh(b)
        except RuntimeError as error:
            with pytest.raises(type(error)):
                torch.linalg.eigh(nt)
            return
        eigenvalues, eigenvectors = torch.linalg.eigh(nt)
        assert_close(eigenvalues[0], w_a)
        assert_close(eigenvalues[1], w_b)

    def test_linalg_inv(self, device, float_dtype):
        a = torch.eye(3, device=device, dtype=float_dtype) + 0.1 * torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.eye(4, device=device, dtype=float_dtype) + 0.1 * torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        pair = _run_or_expect_unsupported(
            lambda: torch.linalg.inv(nt),
            lambda: NT([torch.linalg.inv(a), torch.linalg.inv(b)], **reference_options(nt)),
        )
        if pair is None:
            return
        result, reference = pair
        assert_close(result, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_linalg_norm_matrix_dims_compile_fullgraph(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
                torch.randn(5, 3, 4, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(
            lambda x: torch.linalg.norm(x, ord="fro", dim=(2, 3)),
            backend="inductor",
            fullgraph=True,
        )
        result = compiled(nt)
        reference = NT([torch.linalg.norm(t, ord="fro", dim=(1, 2)) for t in nt], **reference_options(nt))
        assert_close(result, reference)

    def test_linalg_norm_negative_ord_ragged_dim(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])

        result = torch.linalg.norm(nt, ord=-1, dim=1)
        assert_close(result[0], torch.linalg.norm(a, ord=-1, dim=0))
        assert_close(result[1], torch.linalg.norm(b, ord=-1, dim=0))

    def test_linalg_norm_no_dim(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 2, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.linalg.norm(nt)
        assert_close(result[0].squeeze(), torch.linalg.norm(a))
        assert_close(result[1].squeeze(), torch.linalg.norm(b))

    def test_linalg_norm_with_dim(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.linalg.norm(nt, dim=1)
        assert_close(result[0], torch.linalg.norm(a, dim=0))
        assert_close(result[1], torch.linalg.norm(b, dim=0))

        global_result = torch.linalg.vector_norm(nt, ord=0)
        assert_close(global_result[0].squeeze(), torch.linalg.vector_norm(a, ord=0))
        assert_close(global_result[1].squeeze(), torch.linalg.vector_norm(b, ord=0))

    def test_linalg_qr(self, device, float_dtype):
        a = torch.randn(4, 3, device=device, dtype=float_dtype)
        b = torch.randn(5, 3, device=device, dtype=float_dtype)
        nt = NT([a, b])
        try:
            torch.linalg.qr(a)
            torch.linalg.qr(b)
        except RuntimeError as error:
            with pytest.raises(type(error)):
                torch.linalg.qr(nt)
            return
        Q, R = torch.linalg.qr(nt)
        # Verify Q @ R ≈ original
        atol, rtol = low_precision_cuda_tolerances(
            device,
            float_dtype,
            default=(1e-8, 1e-5),
            fp16=(5e-4, 5e-3),
            bf16=(5e-3, 3e-2),
        )
        assert_close(Q[0] @ R[0], a, atol=atol, rtol=rtol)
        assert_close(Q[1] @ R[1], b, atol=atol, rtol=rtol)

    def test_linalg_solve(self, device, float_dtype):
        a = torch.eye(3, device=device, dtype=float_dtype) + 0.1 * torch.randn(3, 3, device=device, dtype=float_dtype)
        b_vec = torch.randn(3, 1, device=device, dtype=float_dtype)
        a2 = torch.eye(4, device=device, dtype=float_dtype) + 0.1 * torch.randn(4, 4, device=device, dtype=float_dtype)
        b2_vec = torch.randn(4, 1, device=device, dtype=float_dtype)
        nt_a = NT([a, a2])
        nt_b = NT([b_vec, b2_vec])
        pair = _run_or_expect_unsupported(
            lambda: torch.linalg.solve(nt_a, nt_b),
            lambda: NT([torch.linalg.solve(a, b_vec), torch.linalg.solve(a2, b2_vec)], **reference_options(nt_a)),
        )
        if pair is None:
            return
        result, reference = pair
        assert_close(result, reference)

    def test_linalg_solve_mismatched_batch_lengths_raises(self, device, float_dtype):
        a = torch.eye(3, device=device, dtype=float_dtype)
        b = torch.randn(3, 1, device=device, dtype=float_dtype)
        nt_a = NT([a, a])
        nt_b = NT([b])
        with pytest.raises(ValueError, match="linalg.solve: NestedTensor batch length mismatch"):
            torch.linalg.solve(nt_a, nt_b)

    def test_linalg_svd(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 3, device=device, dtype=float_dtype)
        nt = NT([a, b])
        try:
            _, S_a, _ = torch.linalg.svd(a)
            _, S_b, _ = torch.linalg.svd(b)
        except RuntimeError as error:
            with pytest.raises(type(error)):
                torch.linalg.svd(nt)
            return
        U, S, Vh = torch.linalg.svd(nt)
        # SVD is unique up to sign, so compare singular values
        assert_close(S[0], S_a)
        assert_close(S[1], S_b)

    def test_matmul_and_linalg(self):
        a0 = torch.randn(2, 3, 3)
        a1 = torch.randn(3, 3, 3)
        b0 = torch.randn(2, 3, 4)
        b1 = torch.randn(3, 3, 4)
        nt_a = NT([a0, a1])
        nt_b = NT([b0, b1])

        matmul_out = torch.matmul(nt_a, nt_b)
        matmul_ref = NT([torch.matmul(x, y) for x, y in zip(nt_a, nt_b)], **reference_options(nt_a))
        assert_close(matmul_out, matmul_ref)

        sym_a0 = a0 + a0.transpose(-1, -2)
        sym_a1 = a1 + a1.transpose(-1, -2)
        nt_sym = NT([sym_a0, sym_a1])
        rhs0 = torch.randn(2, 3, 2)
        rhs1 = torch.randn(3, 3, 2)
        nt_rhs = NT([rhs0, rhs1])
        solve_out = torch.linalg.solve(nt_sym, nt_rhs)
        solve_ref = NT([torch.linalg.solve(x, y) for x, y in zip(nt_sym, nt_rhs)], **reference_options(nt_sym))
        assert_close(solve_out, solve_ref, rtol=1e-5, atol=1e-5)

        svd_u, svd_s, svd_vh = torch.linalg.svd(nt_a, full_matrices=True)
        for u_elem, s_elem, vh_elem, a_elem in zip(svd_u, svd_s, svd_vh, nt_a):
            ref_s = torch.linalg.svd(a_elem, full_matrices=True).S
            assert_close(s_elem, ref_s, rtol=1e-5, atol=1e-5)
            recon = u_elem @ torch.diag_embed(s_elem) @ vh_elem
            assert_close(recon, a_elem, rtol=1e-5, atol=1e-5)


class TestLogicOps:

    def test_bitwise_ops(self, device):
        nt = NestedTensor(
            [
                torch.tensor([1, 2, 3], device=device, dtype=torch.int64),
                torch.tensor([4, 5], device=device, dtype=torch.int64),
            ]
        )
        output = torch.bitwise_and(nt, 1)
        reference = NT([torch.bitwise_and(t, 1) for t in nt], **reference_options(nt))
        assert_close(output, reference)

        output = torch.bitwise_or(nt, 1)
        reference = NT([torch.bitwise_or(t, 1) for t in nt], **reference_options(nt))
        assert_close(output, reference)

        output = torch.bitwise_xor(nt, 1)
        reference = NT([torch.bitwise_xor(t, 1) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_logical_ops_accept_python_scalars(self, device):
        nt = NestedTensor([torch.tensor([True, False], device=device), torch.tensor([True], device=device)])
        true_tensor = torch.tensor(True, device=device)
        false_tensor = torch.tensor(False, device=device)
        output = torch.logical_and(nt, true_tensor)
        assert_close(output, nt)

        output = torch.logical_or(nt, false_tensor)
        assert_close(output, nt)

        output = torch.logical_xor(nt, true_tensor)
        reference = NT([torch.logical_xor(t, true_tensor) for t in nt], **reference_options(nt))
        assert_close(output, reference)

        output = torch.logical_or(nt.tensor, nt)
        assert_close(output, nt)


class TestMatrixMultiplication:

    @staticmethod
    def _complementary_matmul_inputs(lengths, device):
        left = [torch.randn(length, 3, 2, device=device, requires_grad=True) for length in lengths]
        right = [torch.randn(length, 2, 1, device=device, requires_grad=True) for length in lengths]
        return left, right

    @staticmethod
    def _complementary_matmul_reference(left, right):
        outputs = [torch.matmul(lhs.unsqueeze(1), rhs.unsqueeze(0)) for lhs, rhs in zip(left, right)]
        return torch.cat([output.flatten(0, 1) for output in outputs])

    def test_complementary_ragged_matmul_matches_per_element_and_preserves_gradients(self, device):
        lengths = (2, 4)
        left_parts, right_parts = self._complementary_matmul_inputs(lengths, device)
        reference_left = [part.detach().clone().requires_grad_() for part in left_parts]
        reference_right = [part.detach().clone().requires_grad_() for part in right_parts]
        left = NT(left_parts, ragged_dims=(0,)).unsqueeze(2)
        right = NT(right_parts, ragged_dims=(0,)).unsqueeze(2).transpose(1, 2)

        output = torch.matmul(left, right)

        reference = self._complementary_matmul_reference(reference_left, reference_right)
        cotangent = torch.randn_like(reference)
        actual_gradients = torch.autograd.grad(output.concat, (*left_parts, *right_parts), cotangent)
        expected_gradients = torch.autograd.grad(reference, (*reference_left, *reference_right), cotangent)

        assert output.ragged_dims == (0, 1)
        assert output.element_sizes().tolist() == [[length, length, 3, 1] for length in lengths]
        assert_close(output.concat, reference)
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
            assert_close(actual, expected)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_complementary_ragged_matmul_dynamic_fullgraph(self, device):
        def pairwise_matmul(left, right):
            return torch.matmul(left.unsqueeze(2), right.unsqueeze(2).transpose(1, 2))

        compiled = torch.compile(pairwise_matmul, backend="aot_eager", fullgraph=True, dynamic=True)

        for lengths in ((2, 3), (1, 4, 2)):
            left_parts, right_parts = self._complementary_matmul_inputs(lengths, device)
            left_parts = [part.detach() for part in left_parts]
            right_parts = [part.detach() for part in right_parts]

            output = compiled(NT(left_parts, ragged_dims=(0,)), NT(right_parts, ragged_dims=(0,)))
            reference = self._complementary_matmul_reference(left_parts, right_parts)

            assert output.ragged_dims == (0, 1)
            assert output.element_sizes().tolist() == [[length, length, 3, 1] for length in lengths]
            assert_close(output.concat, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_complementary_ragged_matmul_dynamic_fullgraph_wrapper_gradients(self, device):
        def pairwise_matmul(left, right):
            return torch.matmul(left.unsqueeze(2), right.unsqueeze(2).transpose(1, 2))

        compiled = torch.compile(pairwise_matmul, backend="aot_eager", fullgraph=True, dynamic=True)

        for lengths in ((2, 3), (1, 4, 2)):
            left_parts, right_parts = self._complementary_matmul_inputs(lengths, device)
            reference_left = [part.detach().clone().requires_grad_() for part in left_parts]
            reference_right = [part.detach().clone().requires_grad_() for part in right_parts]
            left = NT(left_parts, ragged_dims=(0,))
            right = NT(right_parts, ragged_dims=(0,))

            output = compiled(left, right)
            reference = self._complementary_matmul_reference(reference_left, reference_right)
            cotangent = torch.randn_like(reference)
            actual_gradients = torch.autograd.grad(output.concat, (*left_parts, *right_parts), cotangent)
            expected_gradients = torch.autograd.grad(reference, (*reference_left, *reference_right), cotangent)

            assert output.ragged_dims == (0, 1)
            assert output.element_sizes().tolist() == [[length, length, 3, 1] for length in lengths]
            assert_close(output.concat, reference)
            for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
                assert_close(actual, expected)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_addmm_compile_fullgraph_tensor_lhs(self, device, float_dtype):
        bias = torch.randn(5, 1, device=device, dtype=float_dtype)
        left = torch.randn(5, 4, device=device, dtype=float_dtype)
        right = NT(
            [
                torch.randn(4, 3, device=device, dtype=float_dtype),
                torch.randn(4, 2, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(lambda x, y, z: torch.addmm(x, y, z), backend="inductor", fullgraph=True)
        output = compiled(bias, left, right)
        reference = NT([torch.addmm(bias, left, b) for b in right], **reference_options(right))
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_addr_compile_fullgraph_dense_vector_with_nested_vector(self, device, float_dtype):
        bias = torch.randn(4, 1, device=device, dtype=float_dtype)
        left = torch.randn(4, device=device, dtype=float_dtype)
        right = NT(
            [
                torch.randn(3, device=device, dtype=float_dtype),
                torch.randn(2, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(lambda x, y, z: torch.addr(x, y, z), backend="inductor", fullgraph=True)
        output = compiled(bias, left, right)
        reference = NT([torch.addr(bias, left, b) for b in right], **reference_options(right))
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_addr_compile_fullgraph_nested_vector(self, device, float_dtype):
        bias = torch.randn(1, 4, device=device, dtype=float_dtype)
        left = NT(
            [
                torch.randn(3, device=device, dtype=float_dtype),
                torch.randn(2, device=device, dtype=float_dtype),
            ]
        )
        right = torch.randn(4, device=device, dtype=float_dtype)
        compiled = torch.compile(lambda x, y, z: torch.addr(x, y, z), backend="inductor", fullgraph=True)
        output = compiled(bias, left, right)
        reference = NT([torch.addr(bias, a, right) for a in left], **reference_options(left))
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_baddbmm_compile(self, device, float_dtype):
        bias = torch.randn(2, 3, 1, device=device, dtype=float_dtype)
        left = torch.randn(2, 3, 4, device=device, dtype=float_dtype)
        right_parts = [torch.randn(4, n, device=device, dtype=float_dtype) for n in (5, 3)]
        right = NT(right_parts, ragged_dims=(1,))
        compiled = torch.compile(lambda x, y, z: torch.baddbmm(x, y, z), backend="inductor", fullgraph=True)
        output = compiled(bias, left, right)
        expected = [
            torch.baddbmm(bias[index], left[index].unsqueeze(0), part.unsqueeze(0)).squeeze(0)
            for index, part in enumerate(right_parts)
        ]
        for actual, reference in zip(output.unbind(), expected):
            assert_close(actual, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize("operation", [torch.bmm, torch.matmul])
    def test_bmm_compile(self, device, float_dtype, operation):
        torch.compiler.reset()
        left = torch.randn(2, 3, 4, device=device, dtype=float_dtype)
        right_parts = [torch.randn(4, n, device=device, dtype=float_dtype) for n in (5, 3)]
        right = NT(right_parts, ragged_dims=(1,))
        compiled = torch.compile(lambda x, y: operation(x, y), backend="inductor", fullgraph=True)
        output = compiled(left, right)
        for actual, reference in zip(output.unbind(), [left[index] @ part for index, part in enumerate(right_parts)]):
            assert_close(actual, reference)

    @pytest.mark.parametrize("ordering", ["left", "right", "both"])
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("operation", [torch.bmm, torch.matmul])
    def test_bmm(self, device, ordering, batch_first, operation):
        dtype = torch.float64
        lengths = (2, 5)
        if ordering == "right":
            left = torch.randn(2, 4, 3, device=device, dtype=dtype, requires_grad=True)
            right_parts = [torch.randn(3, n, device=device, dtype=dtype, requires_grad=True) for n in lengths]
            right = NT(right_parts, ragged_dims=(1,), batch_first=batch_first)
            reference_left = left.detach().clone().requires_grad_()
            reference_right = [part.detach().clone().requires_grad_() for part in right_parts]
            actual_inputs, reference_inputs = (left, *right_parts), (reference_left, *reference_right)
            expected = [reference_left[index] @ part for index, part in enumerate(reference_right)]
        else:
            left_parts = [torch.randn(n, 3, device=device, dtype=dtype, requires_grad=True) for n in lengths]
            left = NT(left_parts, ragged_dims=(0,), batch_first=batch_first)
            reference_left = [part.detach().clone().requires_grad_() for part in left_parts]
            if ordering == "both":
                right_parts = [torch.randn(3, 4, device=device, dtype=dtype, requires_grad=True) for _ in lengths]
                right = NT(right_parts, ragged_dims=(0,), batch_first=batch_first)
                reference_right = [part.detach().clone().requires_grad_() for part in right_parts]
                actual_inputs, reference_inputs = (*left_parts, *right_parts), (*reference_left, *reference_right)
            else:
                right = torch.randn(2, 3, 4, device=device, dtype=dtype, requires_grad=True)
                reference_right = right.detach().clone().requires_grad_()
                actual_inputs, reference_inputs = (*left_parts, right), (*reference_left, reference_right)
            expected = [part @ reference_right[index] for index, part in enumerate(reference_left)]
        output = operation(left, right)
        assert output.batch_first is batch_first
        torch.testing.assert_close(output.unbind(0 if batch_first else 1), expected)
        probes = [torch.randn_like(part) for part in expected]
        gradients = torch.autograd.grad(sum((part * probe).sum() for part, probe in zip(output, probes)), actual_inputs)
        references = torch.autograd.grad(
            sum((part * probe).sum() for part, probe in zip(expected, probes)), reference_inputs
        )
        torch.testing.assert_close(gradients, references)

    @pytest.mark.parametrize("ordering", ["left", "right"])
    def test_matmul_hvp(self, device, ordering):
        parts = [torch.randn(3, 3, device=device, dtype=torch.float64, requires_grad=True) for _ in range(2)]
        dense = torch.randn(2, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        reference_parts = [part.detach().clone().requires_grad_() for part in parts]
        reference_dense = dense.detach().clone().requires_grad_()
        nested = NT(parts, ragged_dims=(0,) if ordering == "left" else (1,))
        output = torch.matmul(nested, dense) if ordering == "left" else torch.matmul(dense, nested)
        expected = [
            part @ reference_dense[index] if ordering == "left" else reference_dense[index] @ part
            for index, part in enumerate(reference_parts)
        ]
        inputs, reference_inputs = (*parts, dense), (*reference_parts, reference_dense)
        gradients = torch.autograd.grad(output.concat.square().sum(), inputs, create_graph=True)
        references = torch.autograd.grad(
            sum(part.square().sum() for part in expected), reference_inputs, create_graph=True
        )
        vectors = [torch.randn_like(part) for part in inputs]
        actual = torch.autograd.grad(sum((grad * vector).sum() for grad, vector in zip(gradients, vectors)), inputs)
        reference = torch.autograd.grad(
            sum((grad * vector).sum() for grad, vector in zip(references, vectors)), reference_inputs
        )
        torch.testing.assert_close(actual, reference)

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("beta", [0, 0.7])
    def test_baddbmm(self, device, batch_first, beta):
        parts = [torch.randn(n, 3, device=device, dtype=torch.float64, requires_grad=True) for n in (2, 5)]
        right = torch.randn(2, 3, 4, device=device, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(2, 1, 4, device=device, dtype=torch.float64, requires_grad=True)
        if beta == 0:
            with torch.no_grad():
                bias.fill_(float("nan"))
        reference_parts = [part.detach().clone().requires_grad_() for part in parts]
        reference_right = right.detach().clone().requires_grad_()
        reference_bias = bias.detach().clone().requires_grad_()
        output = torch.baddbmm(bias, NT(parts, ragged_dims=(0,), batch_first=batch_first), right, alpha=0.5, beta=beta)
        expected = [
            torch.baddbmm(
                reference_bias[index], part.unsqueeze(0), reference_right[index].unsqueeze(0), alpha=0.5, beta=beta
            ).squeeze(0)
            for index, part in enumerate(reference_parts)
        ]
        torch.testing.assert_close(output.unbind(0 if batch_first else 1), expected)
        gradients = torch.autograd.grad(output.concat.square().sum(), (*parts, right, bias))
        references = torch.autograd.grad(
            sum(part.square().sum() for part in expected), (*reference_parts, reference_right, reference_bias)
        )
        torch.testing.assert_close(gradients, references)

    @pytest.mark.parametrize("operation", [torch.bmm, torch.baddbmm])
    def test_bmm_errors(self, device, operation):
        left = NT([torch.randn(n, 3, device=device) for n in (2, 5)], ragged_dims=(0,))
        right = torch.randn(2, 3, 4, device=device)

        def call(a, b, **kwargs):
            return (
                operation(a, b, **kwargs)
                if operation is torch.bmm
                else operation(torch.zeros((), device=device), a, b, **kwargs)
            )

        for invalid in (NT([torch.randn(3, device=device)]), NT([torch.randn(2, 3, 4, device=device)])):
            with pytest.raises(RuntimeError, match="3-D"):
                call(invalid, right)
        with pytest.raises(RuntimeError, match="batch"):
            call(left, torch.randn(1, 3, 4, device=device))
        with pytest.raises(RuntimeError, match="contracted"):
            call(left, torch.randn(2, 2, 4, device=device))
        with pytest.raises(NotImplementedError, match="out"):
            call(left, right, out=torch.empty(2, 5, 4, device=device))

    def test_jagged_attention_matmul(self, device, float_dtype):
        query = NT(
            [
                torch.randn(2, 5, 4, device=device, dtype=float_dtype),
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
            ]
        )
        key = NT(
            [
                torch.randn(2, 4, 5, device=device, dtype=float_dtype),
                torch.randn(2, 4, 3, device=device, dtype=float_dtype),
            ]
        )
        value = NT(
            [
                torch.randn(2, 5, 6, device=device, dtype=float_dtype),
                torch.randn(2, 3, 6, device=device, dtype=float_dtype),
            ]
        )

        scores = torch.matmul(query, key)
        probs = torch.softmax(scores, dim=-1)
        log_probs = torch.log_softmax(scores, dim=-1)
        context = torch.matmul(probs, value)

        score_ref = NT([torch.matmul(q, k) for q, k in zip(query, key)], **reference_options(query))
        prob_ref = NT([torch.softmax(score, dim=-1) for score in score_ref], **reference_options(query))
        log_prob_ref = NT([torch.log_softmax(score, dim=-1) for score in score_ref], **reference_options(query))
        context_ref = NT([torch.matmul(prob, val) for prob, val in zip(prob_ref, value)], **reference_options(query))
        stage_atol, stage_rtol = low_precision_cuda_tolerances(
            device, float_dtype, default=(1e-5, 1e-5), fp16=(3e-3, 3e-3), bf16=(8e-3, 2e-2)
        )
        context_atol, context_rtol = low_precision_cuda_tolerances(
            device, float_dtype, default=(1e-5, 1e-5), fp16=(3e-3, 3e-3), bf16=(2e-2, 3e-2)
        )
        assert_close(scores, score_ref, atol=stage_atol, rtol=stage_rtol)
        assert_close(probs, prob_ref, atol=stage_atol, rtol=stage_rtol)
        assert_close(log_probs, log_prob_ref, atol=stage_atol, rtol=stage_rtol)
        assert_close(context, context_ref, atol=context_atol, rtol=context_rtol)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_jagged_attention_matmul_compile(self, device, float_dtype):
        query = NT(
            [
                torch.randn(2, 5, 4, device=device, dtype=float_dtype),
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
            ]
        )
        key = NT(
            [
                torch.randn(2, 4, 5, device=device, dtype=float_dtype),
                torch.randn(2, 4, 3, device=device, dtype=float_dtype),
            ]
        )
        value = NT(
            [
                torch.randn(2, 5, 6, device=device, dtype=float_dtype),
                torch.randn(2, 3, 6, device=device, dtype=float_dtype),
            ]
        )

        def attention(query, key, value):
            scores = torch.matmul(query, key)
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, value)

        compiled = torch.compile(attention, backend="inductor", fullgraph=True)
        output = compiled(query, key, value)

        reference = NT(
            [torch.matmul(torch.softmax(torch.matmul(q, k), dim=-1), v) for q, k, v in zip(query, key, value)],
            **reference_options(query),
        )
        context_atol, context_rtol = low_precision_cuda_tolerances(
            device, float_dtype, default=(1e-5, 1e-5), fp16=(3e-3, 3e-3), bf16=(2e-2, 3e-2)
        )
        assert_close(output, reference, atol=context_atol, rtol=context_rtol)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_matmul_compile_fullgraph_tensor_lhs(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 4, 5, device=device, dtype=float_dtype),
                torch.randn(1, 4, 5, device=device, dtype=float_dtype),
            ]
        )
        weight = torch.randn(3, 4, device=device, dtype=float_dtype)
        compiled = torch.compile(lambda x, y: torch.matmul(x, y), backend="inductor", fullgraph=True)
        output = compiled(weight, nt)
        reference = NT([torch.matmul(weight, t) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_matmul_compile_fullgraph_tensor_rhs(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
                torch.randn(1, 3, 4, device=device, dtype=float_dtype),
            ]
        )
        weight = torch.randn(4, 5, device=device, dtype=float_dtype)
        compiled = torch.compile(lambda x, y: torch.matmul(x, y), backend="inductor", fullgraph=True)
        output = compiled(nt, weight)
        reference = NT([torch.matmul(t, weight) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_matmul_supports_tensor_lhs(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(2, 4, device=device, dtype=float_dtype),
            ]
        )
        weight = torch.randn(5, 2, device=device, dtype=float_dtype)
        output = torch.matmul(weight, nt)
        reference = NT([torch.matmul(weight, t) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_matmul_supports_tensor_rhs(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(1, 3, device=device, dtype=float_dtype),
            ],
            padding_value=-1.0,
        )
        weight = torch.randn(3, 4, device=device, dtype=float_dtype)
        output = torch.matmul(nt, weight)
        reference = NT([torch.matmul(t, weight) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_mm_compile_fullgraph_tensor_lhs(self, device, float_dtype):
        left = torch.randn(5, 4, device=device, dtype=float_dtype)
        right = NT(
            [
                torch.randn(4, 3, device=device, dtype=float_dtype),
                torch.randn(4, 2, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(lambda x, y: torch.mm(x, y), backend="inductor", fullgraph=True)
        output = compiled(left, right)
        reference = NT([torch.mm(left, b) for b in right], **reference_options(right))
        assert_close(output, reference)

    def test_mm(self, device, float_dtype):
        left = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(1, 2, device=device, dtype=float_dtype),
            ]
        )
        right = NT(
            [
                torch.randn(3, 4, device=device, dtype=float_dtype),
                torch.randn(2, 5, device=device, dtype=float_dtype),
            ],
            **reference_options(left),
        )
        output = torch.mm(left, right)
        reference = NT([torch.mm(a, b) for a, b in zip(left, right)], **reference_options(left))
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_outer_compile_fullgraph_dense_vector_with_nested_vector(self, device, float_dtype):
        left = torch.randn(4, device=device, dtype=float_dtype)
        right = NT(
            [
                torch.randn(3, device=device, dtype=float_dtype),
                torch.randn(2, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(lambda x, y: torch.outer(x, y), backend="inductor", fullgraph=True)
        output = compiled(left, right)
        reference = NT([torch.outer(left, b) for b in right], **reference_options(right))
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_outer_compile_fullgraph_nested_vector(self, device, float_dtype):
        left = NT(
            [
                torch.randn(3, device=device, dtype=float_dtype),
                torch.randn(2, device=device, dtype=float_dtype),
            ]
        )
        right = torch.randn(4, device=device, dtype=float_dtype)
        compiled = torch.compile(lambda x, y: torch.outer(x, y), backend="inductor", fullgraph=True)
        output = compiled(left, right)
        reference = NT([torch.outer(a, right) for a in left], **reference_options(left))
        assert_close(output, reference)

    def test_mv_dense_matrix_with_nested_vector(self, device, float_dtype):
        lhs = torch.randn(6, 4, device=device, dtype=float_dtype)
        rhs = NT(
            [
                torch.randn(4, device=device, dtype=float_dtype),
                torch.randn(4, device=device, dtype=float_dtype),
            ]
        )
        result = torch.mv(lhs, rhs)
        reference = NT([torch.mv(lhs, vec) for vec in rhs], **reference_options(rhs))
        assert_close(result, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_mv_nested_tensor_with_vector_compile_fullgraph(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        compiled = torch.compile(lambda x, y: torch.mv(x, y), backend="inductor", fullgraph=True)
        result = compiled(nt, w)
        reference = NT([torch.mv(a, w), torch.mv(b, w)], **reference_options(nt))
        assert_close(result, reference)

    def test_mv_nested_tensor_with_vector(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.mv(nt, w)
        reference = NT([torch.mv(a, w), torch.mv(b, w)], **reference_options(nt))
        assert_close(result, reference)


class TestMatrixOps:
    """Tests for matrix/linalg ops registered in torch_functions.py."""

    def test_det(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        pair = _run_or_expect_unsupported(lambda: torch.det(nt), lambda: (torch.det(a), torch.det(b)))
        if pair is None:
            return
        result, reference = pair
        ref_a, ref_b = reference
        assert_close(result[0].squeeze(), ref_a)
        assert_close(result[1].squeeze(), ref_b)

    def test_diag_1d_to_2d(self, device, float_dtype):
        a = torch.randn(3, device=device, dtype=float_dtype)
        b = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.diag(nt)
        assert_close(result[0], torch.diag(a))
        assert_close(result[1], torch.diag(b))

    def test_diag_2d_to_1d(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.diag(nt)
        assert_close(result[0], torch.diag(a))
        assert_close(result[1], torch.diag(b))

    def test_diagflat(self, device, float_dtype):
        a = torch.randn(3, device=device, dtype=float_dtype)
        b = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.diagflat(nt)
        assert_close(result[0], torch.diagflat(a))
        assert_close(result[1], torch.diagflat(b))

    def test_diagonal(self, device, float_dtype):
        a = torch.randn(3, 4, 5, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, 5, device=device, dtype=float_dtype)
        nt = NT([a, b])
        # dim1=1, dim2=2 in NestedTensor → dim1=0, dim2=1 in element
        result = torch.diagonal(nt, dim1=1, dim2=2)
        assert_close(result[0], torch.diagonal(a, dim1=0, dim2=1))
        assert_close(result[1], torch.diagonal(b, dim1=0, dim2=1))

    def test_diagonal_static_dims(self, device, float_dtype):
        a = torch.randn(2, 3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 3, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.diagonal(nt, dim1=2, dim2=3)
        assert_close(result[0], torch.diagonal(a, dim1=1, dim2=2))
        assert_close(result[1], torch.diagonal(b, dim1=1, dim2=2))

    def test_inverse(self, device, float_dtype):
        # Use well-conditioned matrices
        a = torch.eye(3, device=device, dtype=float_dtype) + 0.1 * torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.eye(4, device=device, dtype=float_dtype) + 0.1 * torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        pair = _run_or_expect_unsupported(
            lambda: torch.inverse(nt),
            lambda: NT([torch.inverse(a), torch.inverse(b)], **reference_options(nt)),
        )
        if pair is None:
            return
        result, reference = pair
        assert_close(result, reference)

    def test_matrix_exp(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(2, 2, device=device, dtype=float_dtype)
        nt = NT([a, b])
        pair = _run_or_expect_unsupported(
            lambda: torch.matrix_exp(nt),
            lambda: NT([torch.matrix_exp(a), torch.matrix_exp(b)], **reference_options(nt)),
        )
        if pair is None:
            return
        result, reference = pair
        assert_close(result, reference, equal_nan=True)

    def test_matrix_power(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.matrix_power(nt, 3)
        assert_close(result[0], torch.matrix_power(a, 3))
        assert_close(result[1], torch.matrix_power(b, 3))

    def test_matrix_power_batched_ragged(self, device, float_dtype):
        a = torch.randn(2, 3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 3, 3, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.matrix_power(nt, 3)
        reference = NT([torch.matrix_power(a, 3), torch.matrix_power(b, 3)], **reference_options(nt))
        assert_close(result, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_matrix_power_compile_fullgraph_batched_ragged(self, device, float_dtype):
        a = torch.randn(2, 3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 3, 3, device=device, dtype=float_dtype)
        nt = NT([a, b])
        compiled = torch.compile(lambda x: torch.matrix_power(x, 3), backend="inductor", fullgraph=True)
        result = compiled(nt)
        reference = NT([torch.matrix_power(a, 3), torch.matrix_power(b, 3)], **reference_options(nt))
        assert_close(result, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_repeat_interleave_compile_fullgraph(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(4, 3, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(lambda x: torch.repeat_interleave(x, 2, dim=2), backend="inductor", fullgraph=True)
        result = compiled(nt)
        reference = NT([torch.repeat_interleave(t, 2, dim=1) for t in nt], **reference_options(nt))
        assert_close(result, reference)

    def test_repeat_interleave_no_dim(self, device, float_dtype):
        a = torch.randn(2, 3, device=device, dtype=float_dtype)
        b = torch.randn(3, 2, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.repeat_interleave(nt, 2)
        assert_close(result[0], torch.repeat_interleave(a, 2))
        assert_close(result[1], torch.repeat_interleave(b, 2))

    def test_repeat_interleave_with_dim(self, device, float_dtype):
        a = torch.randn(2, 3, device=device, dtype=float_dtype)
        b = torch.randn(2, 3, device=device, dtype=float_dtype)
        nt = NT([a, b])
        # dim=1 in NestedTensor → dim=0 in element
        result = torch.repeat_interleave(nt, 2, dim=1)
        assert_close(result[0], torch.repeat_interleave(a, 2, dim=0))
        assert_close(result[1], torch.repeat_interleave(b, 2, dim=0))

    def test_trace(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.trace(nt)
        # Scalar results get packed as 1-D elements in NestedTensor
        assert_close(result[0].squeeze(), torch.trace(a))
        assert_close(result[1].squeeze(), torch.trace(b))

    def test_tril(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.tril(nt)
        assert_close(result[0], torch.tril(a))
        assert_close(result[1], torch.tril(b))

    def test_triu(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.triu(nt)
        assert_close(result[0], torch.triu(a))
        assert_close(result[1], torch.triu(b))

    def test_triu_diagonal(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.triu(nt, diagonal=1)
        assert_close(result[0], torch.triu(a, diagonal=1))
        assert_close(result[1], torch.triu(b, diagonal=1))


class TestNonzeroAndTake:

    def test_nonzero_as_tuple_requires_matching_ndim(self, device):
        with pytest.raises(ValueError, match="same number of dimensions"):
            NT([torch.tensor([1, 0], device=device), torch.tensor([[1], [0]], device=device)])

    def test_nonzero_flattened(self, device):
        nt = NT(
            [
                torch.tensor([[1.0], [0.0]], device=device),
                torch.tensor([[0.0, 2.0]], device=device),
            ]
        )

        output = torch.nonzero(nt, as_tuple=False)
        reference = NT([torch.nonzero(t, as_tuple=False) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_nonzero_ignores_padding_value(self, device):
        nt = NT([torch.tensor([0, 1, 0], device=device), torch.tensor([2], device=device)], padding_value=9)
        output = torch.nonzero(nt, as_tuple=False)
        reference = NT([torch.nonzero(t, as_tuple=False) for t in nt], **reference_options(nt))
        assert_close(output, reference)

        output = torch.nonzero(nt, as_tuple=True)
        reference = (NT([torch.nonzero(t, as_tuple=True)[0] for t in nt], **reference_options(nt)),)
        assert isinstance(output, tuple)
        assert len(output) == 1
        assert_close(output[0], reference[0])

    def test_nonzero_matches_dense_empty_result_dtype(self, device):
        nt = NT([torch.zeros(3, device=device), torch.zeros(1, device=device)], padding_value=7)
        output = torch.nonzero(nt, as_tuple=False)
        reference = NT([torch.nonzero(t, as_tuple=False) for t in nt], **reference_options(nt))

        assert_close(output, reference)
        assert output.dtype == torch.long
        assert output[0].shape == torch.Size([0, 1])
        assert output[1].shape == torch.Size([0, 1])

    def test_nonzero_multi_ragged(self, device):
        nt = NT(
            [
                torch.tensor([[1, 0, 0], [0, 1, 1]], device=device),
                torch.tensor([[0, 2, 0, 3]], device=device),
            ]
        )
        output = torch.nonzero(nt, as_tuple=False)
        reference = NT([torch.nonzero(t, as_tuple=False) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_take_nested_index_is_per_sample(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0, 4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        index = NT([torch.tensor([1], device=device), torch.tensor([0, 2], device=device)], **reference_options(nt))
        output = torch.take(nt, index)
        reference = NT([torch.take(t.reshape(-1), i) for t, i in zip(nt, index)], **reference_options(nt))
        assert_close(output, reference)

    def test_take_tensor_index(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0], device=device, dtype=float_dtype),
            ],
            padding_value=99.0,
        )
        index = torch.tensor([0, 2], device=device)
        output = torch.take(nt, index)
        reference = torch.take(torch.cat([t.reshape(-1) for t in nt]), index)
        assert_close(output, reference)

    def test_count_nonzero(self, device):
        nt = NT([torch.tensor([1, 0, 2], device=device), torch.tensor([0], device=device)])
        output = torch.count_nonzero(nt)
        assert_close(output, torch.tensor(2, device=device))

        output = torch.count_nonzero(nt, dim=1)
        assert_close(output, torch.tensor([2, 0], device=device))

        with pytest.raises(ValueError):
            torch.count_nonzero(nt, dim=0)


class TestOrderStatistics:

    def test_kthvalue(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([3.0, 1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([4.0, 0.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.kthvalue(nt, k=2, dim=1)
        reference = tuple(torch.stack([torch.kthvalue(t, k=2, dim=0)[idx] for t in nt]) for idx in range(2))
        assert_close(output[0], reference[0])
        assert_close(output[1], reference[1])

    def test_median_dim_none_and_dim(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, 2.0, 10.0], device=device, dtype=float_dtype),
                torch.tensor([3.0], device=device, dtype=float_dtype),
            ],
            padding_value=-123.0,
        )
        output = torch.median(nt)
        reference = torch.median(torch.cat([t.reshape(-1) for t in nt]))
        assert_close(output, reference)

        output = torch.median(nt, dim=1, keepdim=True)
        reference = tuple(torch.stack([torch.median(t, dim=0, keepdim=True)[idx] for t in nt]) for idx in range(2))
        assert_close(output[0], reference[0])
        assert_close(output[1], reference[1])

    def test_mode(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, float("nan"), 3.0], device=device, dtype=float_dtype),
                torch.tensor([float("nan"), 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.mode(nt, dim=1)
        reference = tuple(torch.stack([torch.mode(t, dim=0)[idx] for t in nt]) for idx in range(2))
        assert_close(output[0], reference[0], equal_nan=True)
        assert_close(output[1], reference[1])

    def test_nanmedian(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, float("nan"), 3.0], device=device, dtype=float_dtype),
                torch.tensor([float("nan"), 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.nanmedian(nt)
        reference = torch.nanmedian(torch.cat([t.reshape(-1) for t in nt]))
        assert_close(output, reference, equal_nan=True)

        nt_static = NT(
            [
                torch.tensor([[1.0, float("nan"), 3.0], [2.0, 4.0, float("nan")]], device=device, dtype=float_dtype),
                torch.tensor(
                    [[float("nan"), 7.0, 8.0], [0.0, float("nan"), 1.0], [6.0, 5.0, 4.0]],
                    device=device,
                    dtype=float_dtype,
                ),
            ]
        )
        output = torch.nanmedian(nt_static, dim=2)
        reference = tuple(
            NT([torch.nanmedian(t, dim=1)[i] for t in nt_static], **reference_options(nt_static)) for i in range(2)
        )
        assert_close(output[0], reference[0], equal_nan=True)
        assert_close(output[1], reference[1])

    def test_nanquantile(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, float("nan"), 3.0], device=device, dtype=float_dtype),
                torch.tensor([float("nan"), 5.0], device=device, dtype=float_dtype),
            ]
        )
        pair = _run_or_expect_unsupported(
            lambda: torch.nanquantile(nt, 0.5),
            lambda: torch.nanquantile(torch.cat([t.reshape(-1) for t in nt]), 0.5),
        )
        if pair is None:
            return
        output, reference = pair
        assert_close(output, reference, equal_nan=True)

    def test_order_statistics_static_dim(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([[3.0, 1.0, 2.0], [5.0, 4.0, 6.0]], device=device, dtype=float_dtype),
                torch.tensor([[9.0, 7.0, 8.0], [0.0, 2.0, 1.0], [6.0, 5.0, 4.0]], device=device, dtype=float_dtype),
            ]
        )

        kthvalue_out = torch.kthvalue(nt, k=2, dim=2)
        kthvalue_ref = tuple(
            NT([torch.kthvalue(t, k=2, dim=1)[i] for t in nt], **reference_options(nt)) for i in range(2)
        )
        assert_close(kthvalue_out[0], kthvalue_ref[0])
        assert_close(kthvalue_out[1], kthvalue_ref[1])

        median_out = torch.median(nt, dim=2, keepdim=True)
        median_ref = tuple(
            NT([torch.median(t, dim=1, keepdim=True)[i] for t in nt], **reference_options(nt)) for i in range(2)
        )
        assert_close(median_out[0], median_ref[0])
        assert_close(median_out[1], median_ref[1])

        mode_out = torch.mode(nt, dim=2)
        mode_ref = tuple(NT([torch.mode(t, dim=1)[i] for t in nt], **reference_options(nt)) for i in range(2))
        assert_close(mode_out[0], mode_ref[0])
        assert_close(mode_out[1], mode_ref[1])

    def test_quantile(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, float("nan"), 3.0], device=device, dtype=float_dtype),
                torch.tensor([float("nan"), 5.0], device=device, dtype=float_dtype),
            ]
        )
        pair = _run_or_expect_unsupported(
            lambda: torch.quantile(nt, 0.5),
            lambda: torch.quantile(torch.cat([t.reshape(-1) for t in nt]), 0.5),
        )
        if pair is None:
            return
        output, reference = pair
        assert_close(output, reference, equal_nan=True)

        output = torch.quantile(nt, 0.5, keepdim=True)
        assert output.shape == (1, 1)


class TestRandomOps:

    def test_dropout_and_bernoulli(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.ones(4, device=device, dtype=float_dtype),
                torch.ones(2, device=device, dtype=float_dtype),
            ]
        )
        output = torch.dropout(nt, p=1.0, train=True)
        assert_close(output, torch.zeros_like(nt))

        output = torch.bernoulli(nt)
        assert_close(output, nt)

    def test_rand_like_and_randn_like_respect_kwargs(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.zeros(2, 3, device=device, dtype=float_dtype),
                torch.zeros(1, 3, device=device, dtype=float_dtype),
            ]
        )
        torch.manual_seed(1016)
        output = torch.rand_like(nt, dtype=torch.float64)
        assert output.dtype == torch.float64

        torch.manual_seed(1016)
        reference = NT([torch.rand_like(t, dtype=torch.float64) for t in nt], **reference_options(nt))
        assert_close(output, reference)

        torch.manual_seed(1016)
        output = torch.randn_like(nt)
        torch.manual_seed(1016)
        reference = NT([torch.randn_like(t) for t in nt], **reference_options(nt))
        assert_close(output, reference)


class TestReductionOps:

    @pytest.mark.parametrize(
        ("shapes", "ragged_dims", "dim", "element_dim", "operation", "keepdim"), _BOOLEAN_REDUCTION_CASES
    )
    def test_boolean_ragged_reduction_matches_per_sample(
        self, device, shapes, ragged_dims, dim, element_dim, operation, keepdim
    ):
        elements = boolean_reduction_elements(shapes, operation, device)
        nested = NT(elements, ragged_dims=ragged_dims)
        function = getattr(torch, operation)

        output = (
            nested.any(dim=dim, keepdim=keepdim) if operation == "any" else torch.all(nested, dim=dim, keepdim=keepdim)
        )
        reference_parts = [function(element, dim=element_dim, keepdim=keepdim) for element in elements]
        reference = ragged_reduction_reference(reference_parts, ragged_dims, element_dim, keepdim)

        if isinstance(reference, NT):
            assert output.ragged_dims == reference.ragged_dims
            assert_close(output.element_sizes(), reference.element_sizes())
        assert_close(output, reference)

        if len(ragged_dims) > 1 and keepdim:
            broadcast = torch.logical_and(nested, output)
            broadcast_reference = NT(
                [torch.logical_and(element, part) for element, part in zip(elements, reference_parts)],
                ragged_dims=ragged_dims,
            )
            assert_close(broadcast, broadcast_reference)

    @pytest.mark.parametrize(
        ("shapes", "ragged_dims", "dim", "element_dim", "operation", "keepdim"), _BOOLEAN_GRAPH_CASES
    )
    def test_boolean_ragged_reduction_supports_fake_tensor(
        self, shapes, ragged_dims, dim, element_dim, operation, keepdim
    ):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        symbolic_shapes = pytest.importorskip("torch.fx.experimental.symbolic_shapes")
        nested = NT([torch.empty(shape, dtype=torch.bool) for shape in shapes], ragged_dims=ragged_dims)
        function = getattr(torch, operation)

        shape_env = symbolic_shapes.ShapeEnv(allow_dynamic_output_shape_ops=True)
        with fake_tensor_mod.FakeTensorMode(shape_env=shape_env) as mode:
            output = function(mode.from_tensor(nested), dim=dim, keepdim=keepdim)

        assert fake_tensor_mod.is_fake(output.concat if isinstance(output, NT) else output)
        reference_parts = [
            function(torch.empty(shape, dtype=torch.bool), dim=element_dim, keepdim=keepdim) for shape in shapes
        ]
        reference = ragged_reduction_reference(reference_parts, ragged_dims, element_dim, keepdim)
        assert output.shape == reference.shape
        if isinstance(reference, NT):
            assert output.ragged_dims == reference.ragged_dims
            assert output.element_sizes().shape == reference.element_sizes().shape

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize(
        ("shapes", "ragged_dims", "dim", "element_dim", "operation", "keepdim"), _BOOLEAN_GRAPH_CASES
    )
    def test_boolean_ragged_reduction_compiles_fullgraph(
        self, device, shapes, ragged_dims, dim, element_dim, operation, keepdim
    ):
        elements = boolean_reduction_elements(shapes, operation, device)
        nested = NT(elements, ragged_dims=ragged_dims)
        function = getattr(torch, operation)
        reducer = partial(function, dim=dim, keepdim=keepdim)
        reference_parts = [function(element, dim=element_dim, keepdim=keepdim) for element in elements]
        reference = ragged_reduction_reference(reference_parts, ragged_dims, element_dim, keepdim)

        if len(ragged_dims) > 1 and keepdim:

            def consume(structure, values):
                source = structure.packed_like(values)
                return torch.logical_and(source, reducer(source)).concat

            expected = NT(
                [torch.logical_and(element, part) for element, part in zip(elements, reference_parts)],
                ragged_dims=ragged_dims,
            ).concat
        elif isinstance(reference, NT):

            def consume(structure, values):
                return reducer(structure.packed_like(values)).concat

            expected = reference.concat
        else:

            def consume(structure, values):
                return reducer(structure.packed_like(values))

            expected = reference

        compiled = torch.compile(consume, backend="aot_eager", fullgraph=True, dynamic=True)

        assert_close(compiled(nested, nested.concat), expected)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_boolean_multi_ragged_tuple_dim_compiles_fullgraph(self):
        elements = boolean_reduction_elements(((2, 4, 2), (3, 2, 2)), "any")
        nested = NT(elements, ragged_dims=(0, 1))
        compiled = torch.compile(
            lambda structure, values: torch.any(structure.packed_like(values), dim=(1, 2)),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        output = compiled(nested, nested.concat)
        reference = torch.stack([torch.any(element, dim=(0, 1)) for element in elements])
        assert_close(output, reference)

    def test_all_any_multi_dim_reduction_ignores_padding(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.ones(2, 2, device=device, dtype=float_dtype),
                torch.ones(1, 2, device=device, dtype=float_dtype),
            ]
        )
        output = torch.all(nt, dim=(0, 1))
        assert_close(output, torch.tensor([True, True], device=device))
        output = torch.any(nt, dim=(0, 1))
        assert_close(output, torch.tensor([True, True], device=device))

    def test_all_ignores_padding(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.ones(3, device=device, dtype=float_dtype),
                torch.ones(1, device=device, dtype=float_dtype),
            ]
        )
        output = torch.all(nt)
        assert_close(output, torch.tensor(True, device=device))

    def test_amax_amin_and_aminmax_ignore_padding(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0], device=device, dtype=float_dtype),
            ],
            padding_value=100.0,
        )
        output = torch.amax(nt)
        assert_close(output, torch.tensor(3.0, device=device, dtype=float_dtype))
        output = torch.amax(nt, keepdim=True)
        assert output.shape == (1, 1)
        assert_close(output, torch.tensor(3.0, device=device, dtype=float_dtype).reshape(1, 1))

        output = torch.amin(nt)
        assert_close(output, torch.tensor(1.0, device=device, dtype=float_dtype))

        output = torch.amax(nt, dim=0)
        assert_close(output, torch.tensor([2.0, 3.0], device=device, dtype=float_dtype))
        output = torch.amin(nt, dim=0)
        assert_close(output, torch.tensor([1.0, 3.0], device=device, dtype=float_dtype))

        output = torch.aminmax(nt)
        assert_close(output[0], torch.tensor(1.0, device=device, dtype=float_dtype))
        assert_close(output[1], torch.tensor(3.0, device=device, dtype=float_dtype))

        output = torch.aminmax(nt, dim=0)
        assert_close(output[0], torch.tensor([1.0, 3.0], device=device, dtype=float_dtype))
        assert_close(output[1], torch.tensor([2.0, 3.0], device=device, dtype=float_dtype))

    def test_reduce_ragged_last_dim(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(3, 4, device=device, dtype=float_dtype),
                torch.randn(3, 7, device=device, dtype=float_dtype),
            ]
        )
        mean_reference = torch.stack([torch.mean(t, dim=-1) for t in nt])
        sum_reference = torch.stack([torch.sum(t, dim=-1) for t in nt])
        amax_reference = torch.stack([torch.amax(t, dim=-1) for t in nt])
        amin_reference = torch.stack([torch.amin(t, dim=-1) for t in nt])

        assert_close(torch.mean(nt, dim=-1), mean_reference)
        assert_close(torch.sum(nt, dim=-1), sum_reference)
        assert_close(torch.amax(nt, dim=-1), amax_reference)
        assert_close(torch.amin(nt, dim=-1), amin_reference)
        assert_close(torch.sum(nt, dim=-1, keepdim=True), sum_reference.unsqueeze(-1))

        nt = NT(
            [
                torch.randn(4, 3, device=device, dtype=float_dtype),
                torch.randn(7, 3, device=device, dtype=float_dtype),
            ]
        )
        reference = torch.stack([torch.sum(t, dim=0) for t in nt])
        assert_close(torch.sum(nt, dim=1), reference)

    def test_sampled_projected_mask_ragged_sum_values_and_vjp(self, device, float_dtype):
        lengths = (3, 5)
        samples = 2
        channels = 3
        template = NT(
            [torch.empty(length, device=device, dtype=float_dtype) for length in lengths],
            ragged_dims=(0,),
        )
        values = torch.randn_like(template.concat, requires_grad=True)
        projected = template.packed_like(values).unsqueeze(-2).unsqueeze(-1).expand(-1, samples, -1, channels)

        output = projected.sum(dim=-2, keepdim=True)

        segment_sums = torch.stack([part.sum() for part in values.split(lengths)])
        expected = segment_sums[:, None, None, None].expand(-1, samples, 1, channels)
        cotangent = torch.randn_like(expected)
        actual_grad = torch.autograd.grad(output, values, cotangent)[0]
        expected_grad = torch.autograd.grad(expected, values, cotangent)[0]
        assert output.shape == (2, samples, 1, channels)
        assert_close(output, expected)
        assert_close(actual_grad, expected_grad)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_sampled_projected_mask_ragged_sum_compiles_with_vjp(self, device):
        def consume(template, values):
            projected = template.packed_like(values).unsqueeze(-2).unsqueeze(-1).expand(-1, 2, -1, 3)
            return projected.sum(dim=-2, keepdim=True)

        compiled = torch.compile(consume, backend="aot_eager", fullgraph=True)
        lengths = (3, 5)
        template = NT([torch.empty(length, device=device) for length in lengths], ragged_dims=(0,))
        values = torch.randn_like(template.concat, requires_grad=True)
        output = compiled(template, values)

        segment_sums = torch.stack([part.sum() for part in values.split(lengths)])
        expected = segment_sums[:, None, None, None].expand(-1, 2, 1, 3)
        cotangent = torch.randn_like(expected)
        actual_grad = torch.autograd.grad(output, values, cotangent)[0]
        expected_grad = torch.autograd.grad(expected, values, cotangent)[0]
        assert_close(output, expected)
        assert_close(actual_grad, expected_grad)

    def test_reduce_strided_ragged_last_dim(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(3, 8, device=device, dtype=float_dtype),
                torch.randn(3, 11, device=device, dtype=float_dtype),
            ]
        )
        mask = NT(
            [
                torch.ones(1, 8, device=device, dtype=float_dtype),
                torch.ones(1, 11, device=device, dtype=float_dtype),
            ]
        )

        def reduce_frames(features, frame_mask):
            features = torch.flip(features, dims=(-1,))
            frame_mask = torch.flip(frame_mask, dims=(-1,))
            pooled = []
            for offset in range(3):
                pooled.append(features[..., offset::3].amax(dim=-1))
            for offset in range(3):
                frame = features[..., offset::3]
                valid = frame_mask[..., offset::3]
                pooled.append(frame.sum(dim=-1) / valid.sum(dim=-1).clamp_min(torch.finfo(frame.dtype).eps))
            return torch.cat(pooled, dim=-1)

        reference = torch.stack([reduce_frames(t, m) for t, m in zip(nt, mask)])
        assert_close(reduce_frames(nt, mask), reference)

    def test_reduce_static_dims_preserves_ragged(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(4, 5, 6, device=device, dtype=float_dtype),
                torch.randn(4, 7, 6, device=device, dtype=float_dtype),
            ]
        )
        output = torch.sum(nt, dim=(1, 3))
        reference = NT([torch.sum(t, dim=(0, 2)) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    @pytest.mark.parametrize(
        ("dim", "element_dim"),
        [(1, 0), (-1, -1)],
        ids=("ragged", "static"),
    )
    def test_sum_values_and_vjp(self, device, float_dtype, dim, element_dim):
        leaves = [
            torch.randn(2, 3, device=device, dtype=float_dtype, requires_grad=True),
            torch.randn(4, 3, device=device, dtype=float_dtype, requires_grad=True),
        ]
        references = [leaf.detach().clone().requires_grad_() for leaf in leaves]
        nt = NT(leaves)
        output = torch.sum(nt, dim=dim)
        reference_parts = [torch.sum(tensor, dim=element_dim) for tensor in references]
        reference = NT(reference_parts, **reference_options(nt)) if dim == -1 else torch.stack(reference_parts)
        assert_close(output, reference)

        output_values = output.concat if isinstance(output, NestedTensor) else output
        grads = torch.autograd.grad(output_values.square().sum(), leaves)
        ref_grads = torch.autograd.grad(sum(tensor.square().sum() for tensor in reference_parts), references)
        for grad, ref_grad in zip(grads, ref_grads):
            assert_close(grad, ref_grad)

    def test_any_ignores_padding(self, device, float_dtype):
        nt = NestedTensor(
            [torch.zeros(3, device=device, dtype=float_dtype), torch.zeros(1, device=device, dtype=float_dtype)],
            padding_value=1.0,
        )
        output = torch.any(nt)
        assert_close(output, torch.tensor(False, device=device))

    def test_logsumexp(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.logsumexp(nt, dim=1)
        reference = torch.stack([torch.logsumexp(t, dim=0) for t in nt])
        assert_close(output, reference)

    def test_mean_multi_dim_integer_matches_dense_error(self, device):
        nt = NestedTensor(
            [
                torch.arange(24, device=device, dtype=torch.long).reshape(2, 3, 4),
                torch.arange(24, 48, device=device, dtype=torch.long).reshape(2, 3, 4),
            ]
        )
        with pytest.raises(RuntimeError, match="could not infer output dtype"):
            torch.mean(nt, dim=(1, 2))

    def test_nanmean_and_nansum(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, float("nan"), 3.0], device=device, dtype=float_dtype),
                torch.tensor([float("nan")], device=device, dtype=float_dtype),
            ]
        )
        output = torch.nansum(nt)
        assert_close(output, torch.tensor(4.0, device=device, dtype=float_dtype))

        output = torch.nanmean(nt)
        assert_close(output, torch.tensor(2.0, device=device, dtype=float_dtype), equal_nan=True)
        output = torch.nanmean(nt, keepdim=True)
        assert output.shape == (1, 1)
        assert_close(output, torch.tensor(2.0, device=device, dtype=float_dtype).reshape(1, 1), equal_nan=True)

        output = torch.nansum(nt, dim=0)
        reference = torch.stack([torch.nansum(t) for t in nt])
        assert_close(output, reference)

    @pytest.mark.parametrize("name", ["nanmean", "nansum"])
    def test_nan_reductions_multi_ragged_fullgraph_vjp(self, device, name):
        torch.compiler.reset()
        parts = [
            torch.tensor(
                [
                    [[float("nan"), 1, float("nan")], [1, 2, float("nan")]],
                    [[float("nan"), float("inf"), float("nan")], [3, float("nan"), float("nan")]],
                ],
                device=device,
                dtype=torch.float64,
                requires_grad=True,
            ),
            torch.tensor(
                [
                    [[float("nan"), 2, float("nan")]],
                    [[float("nan"), 4, float("nan")]],
                    [[float("nan"), 6, float("nan")]],
                ],
                device=device,
                dtype=torch.float64,
                requires_grad=True,
            ),
        ]
        references = [part.detach().clone().requires_grad_() for part in parts]
        source = NT(parts, ragged_dims=(0, 1))

        def consume(template, values):
            return getattr(template.packed_like(values), name)(dim=1).concat

        compiled = torch.compile(consume, backend="aot_eager", fullgraph=True, dynamic=True)
        actual = compiled(source, source.concat)
        gradients = torch.autograd.grad(actual.sum(), parts)
        expected = torch.cat([getattr(part, name)(dim=0) for part in references])
        torch.testing.assert_close(actual, expected, equal_nan=True)
        wanted = torch.autograd.grad(expected.sum(), references)
        for gradient, reference in zip(gradients, wanted):
            torch.testing.assert_close(gradient, reference, equal_nan=True)

    def test_prod_ignores_padding(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([2.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([2.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.prod(nt)
        assert_close(output, torch.tensor(8.0, device=device, dtype=float_dtype))
        torch.compiler.reset()
        values = torch.tensor([0, 2, 3, 0, 0], device=device, dtype=float_dtype, requires_grad=True)
        reference_values = values.detach().clone().requires_grad_()
        template = NT([values[:3].detach(), values[3:].detach()], ragged_dims=(0,), padding_value=100)
        compiled = torch.compile(
            lambda structure, packed: structure.packed_like(packed).prod(dim=1),
            backend="aot_eager",
            fullgraph=True,
        )
        actual = compiled(template, values)
        expected = torch.stack([reference_values[:3].prod(), reference_values[3:].prod()])
        assert_close(actual, expected)
        assert_close(
            torch.autograd.grad(actual.sum(), values)[0], torch.autograd.grad(expected.sum(), reference_values)[0]
        )

    def test_var_std_var_mean(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.var(nt, correction=1)
        assert_close(output, torch.tensor(2.5, device=device, dtype=float_dtype))
        output = torch.var(nt, correction=1, keepdim=True)
        assert output.shape == (1, 1)
        assert_close(output, torch.tensor(2.5, device=device, dtype=float_dtype).reshape(1, 1))

        output = torch.var(nt, dim=0, correction=1)
        reference = torch.stack([torch.var(t, correction=1) for t in nt])
        assert_close(output, reference)

        output = torch.std(nt, dim=0, correction=1)
        reference = torch.stack([torch.std(t, correction=1) for t in nt])
        assert_close(output, reference)

        output = torch.var_mean(nt, dim=0, correction=1)
        var_reference = torch.stack([torch.var(t, correction=1) for t in nt])
        mean_reference = torch.stack([torch.mean(t) for t in nt])
        assert_close(output[0], var_reference)
        assert_close(output[1], mean_reference)

        output = torch.var_mean(nt, correction=1)
        assert_close(output[0], torch.tensor(2.5, device=device, dtype=float_dtype))
        assert_close(output[1], torch.tensor(3.0, device=device, dtype=float_dtype))

    def test_multi_ragged_static_reductions_use_packed_axis(self, device, float_dtype):
        parts = [
            torch.randn(2, 2, 4, device=device, dtype=float_dtype),
            torch.randn(3, 3, 4, device=device, dtype=float_dtype),
        ]
        nt = NT(parts)

        argmax = torch.argmax(nt, dim=-1)
        count = torch.count_nonzero(nt, dim=-1)
        assert_close(argmax, NT([torch.argmax(t, dim=-1) for t in parts], **reference_options(nt)))
        assert_close(count, NT([torch.count_nonzero(t, dim=-1) for t in parts], **reference_options(nt)))

        norm = torch.linalg.vector_norm(nt, dim=-1, keepdim=True)
        linalg_norm = torch.linalg.norm(nt, dim=-1, keepdim=True)
        norm_reference = NT([torch.linalg.vector_norm(t, dim=-1, keepdim=True) for t in parts], **reference_options(nt))
        assert_close(norm, norm_reference)
        assert_close(linalg_norm, norm_reference)

        variance, mean = torch.var_mean(nt, dim=-1, correction=0, keepdim=True)
        variance_reference = NT(
            [torch.var(t, dim=-1, correction=0, keepdim=True) for t in parts],
            **reference_options(nt),
        )
        mean_reference = NT([torch.mean(t, dim=-1, keepdim=True) for t in parts], **reference_options(nt))
        assert_close(variance, variance_reference)
        assert_close(mean, mean_reference)

    @pytest.mark.parametrize("operation", [torch.sum, torch.mean, torch.amax, torch.amin])
    @pytest.mark.parametrize(("dim", "element_dim"), [(1, 0), (2, 1)])
    def test_multi_ragged_keepdim_reduction_broadcasts_back(self, device, float_dtype, operation, dim, element_dim):
        elements = [
            torch.randn(2, 3, 4, device=device, dtype=float_dtype),
            torch.randn(4, 2, 4, device=device, dtype=float_dtype),
        ]
        nested = NT(elements, ragged_dims=(0, 1))

        reduced = operation(nested, dim=dim, keepdim=True)
        output = nested + reduced
        reduced_references = [operation(element, dim=element_dim, keepdim=True) for element in elements]
        references = [element + reduced for element, reduced in zip(elements, reduced_references)]

        assert [element.shape for element in reduced] == [element.shape for element in reduced_references]
        assert output.ragged_dims == (0, 1)
        for actual, expected in zip(output, references):
            assert_close(actual, expected)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize(("dim", "element_dim"), [(1, 0), (2, 1)])
    @pytest.mark.parametrize("operation", [torch.sum, torch.amax, torch.amin, torch.mean])
    def test_multi_ragged_keepdim_reduction_and_broadcast_compile_dynamic_fullgraph(self, dim, element_dim, operation):
        if dim == 1:

            def consume(structure, values):
                source = structure.packed_like(values)
                return (source + operation(source, dim=1, keepdim=True)).concat

        else:

            def consume(structure, values):
                source = structure.packed_like(values)
                return (source + operation(source, dim=2, keepdim=True)).concat

        compiled = torch.compile(consume, backend="aot_eager", fullgraph=True, dynamic=True)
        for shapes in (((2, 3, 4), (4, 2, 4)), ((3, 2, 4), (1, 5, 4))):
            elements = [torch.randn(shape) for shape in shapes]
            nested = NT(elements, ragged_dims=(0, 1))
            expected = NT(
                [element + operation(element, dim=element_dim, keepdim=True) for element in elements],
                ragged_dims=(0, 1),
            )

            assert_close(compiled(nested, nested.concat), expected.concat)

    @pytest.mark.parametrize("dim", [(1, 2), (1, 3)])
    @pytest.mark.parametrize("keepdim", [False, True])
    @pytest.mark.parametrize("operation", [torch.amax, torch.amin, torch.mean, torch.var, torch.std])
    def test_joint_axes_fullgraph_values_and_vjp(self, device, dim, keepdim, operation):
        torch.compiler.reset()
        scale = -1 if operation is torch.amin else 1
        elements = [
            torch.tensor([[[5, 2], [5, 1]], [[5, 2], [0, 0]]], device=device, dtype=torch.float64)
            .mul_(scale)
            .requires_grad_(),
            torch.tensor([[[5, 3]], [[5, 1]], [[0, 3]]], device=device, dtype=torch.float64)
            .mul_(scale)
            .requires_grad_(),
        ]
        references = [element.detach().clone().requires_grad_() for element in elements]
        nested = NT(elements, ragged_dims=(0, 1), padding_value=100)

        def consume(template, values):
            return operation(template.packed_like(values), dim=dim, keepdim=keepdim)

        compiled = torch.compile(consume, backend="aot_eager", fullgraph=True, dynamic=True)
        output = compiled(nested, nested.concat)
        values = output.concat if isinstance(output, NestedTensor) else output
        gradients = torch.autograd.grad(values.sum(), elements)
        element_dims = tuple(axis - 1 for axis in dim)
        expected_parts = [operation(element, dim=element_dims, keepdim=keepdim) for element in references]
        if dim == (1, 2):
            expected = torch.stack(expected_parts)
        else:
            expected = NT(expected_parts, ragged_dims=(1,) if keepdim else (0,), padding_value=100)
        assert_close(output, expected)
        wanted = torch.autograd.grad(sum(part.sum() for part in expected_parts), references)
        for gradient, reference in zip(gradients, wanted):
            assert_close(gradient, reference)

    def test_multi_ragged_var_mean_and_vector_norm_preserve_grad(self, device, float_dtype):
        leaves = [
            torch.randn(2, 3, 4, device=device, dtype=float_dtype, requires_grad=True),
            torch.randn(4, 2, 4, device=device, dtype=float_dtype, requires_grad=True),
        ]
        references = [leaf.detach().clone().requires_grad_() for leaf in leaves]
        nt = NT(leaves)

        variance, mean = torch.var_mean(nt, dim=-1, correction=0)
        norm = torch.linalg.vector_norm(nt, dim=-1)
        ref_pairs = [torch.var_mean(t, dim=-1, correction=0) for t in references]
        ref_norms = [torch.linalg.vector_norm(t, dim=-1) for t in references]
        assert_close(variance, NT([pair[0] for pair in ref_pairs], **reference_options(nt)))
        assert_close(mean, NT([pair[1] for pair in ref_pairs], **reference_options(nt)))
        assert_close(norm, NT(ref_norms, **reference_options(nt)))

        weights = [torch.randn_like(t) for t in variance]
        loss = sum((value * weight).sum() for value, weight in zip(variance, weights))
        loss = loss + mean.concat.square().sum() + norm.concat.square().sum()
        ref_loss = sum((pair[0] * weight).sum() for pair, weight in zip(ref_pairs, weights))
        ref_loss = ref_loss + sum(pair[1].square().sum() for pair in ref_pairs)
        ref_loss = ref_loss + sum(value.square().sum() for value in ref_norms)
        grads = torch.autograd.grad(loss, leaves)
        ref_grads = torch.autograd.grad(ref_loss, references)
        for grad, ref_grad in zip(grads, ref_grads):
            assert_close(grad, ref_grad)

    def test_vector_norm_negative_ord_without_dim_falls_back_per_element(self, device, float_dtype):
        leaves = [
            (torch.rand(2, 3, device=device, dtype=float_dtype) + 0.5).requires_grad_(),
            (torch.rand(4, 2, device=device, dtype=float_dtype) + 0.5).requires_grad_(),
        ]
        references = [leaf.detach().clone().requires_grad_() for leaf in leaves]
        nt = NT(leaves)

        output = torch.linalg.vector_norm(nt, ord=-1)
        reference = [torch.linalg.vector_norm(tensor, ord=-1) for tensor in references]
        assert_close(output, NT(reference, **reference_options(nt)))

        grads = torch.autograd.grad(output.concat.square().sum(), leaves)
        ref_grads = torch.autograd.grad(sum(tensor.square() for tensor in reference), references)
        for grad, ref_grad in zip(grads, ref_grads):
            assert_close(grad, ref_grad)

    def test_multi_ragged_all_static_multi_dim_reductions(self, device, float_dtype):
        parts = [
            torch.randn(2, 2, 3, 4, device=device, dtype=float_dtype),
            torch.randn(3, 3, 3, 4, device=device, dtype=float_dtype),
        ]
        nt = NT(parts)
        count = torch.count_nonzero(nt, dim=(-2, -1))
        variance, mean = torch.var_mean(nt, dim=(-2, -1), correction=0)
        assert_close(count, NT([torch.count_nonzero(t, dim=(-2, -1)) for t in parts], **reference_options(nt)))
        assert_close(
            variance,
            NT([torch.var(t, dim=(-2, -1), correction=0) for t in parts], **reference_options(nt)),
        )
        assert_close(mean, NT([torch.mean(t, dim=(-2, -1)) for t in parts], **reference_options(nt)))

    def test_count_nonzero_mixed_dims_maps_padded_axes(self, device, float_dtype):
        parts = [
            torch.randn(2, 3, 4, device=device, dtype=float_dtype),
            torch.randn(5, 3, 4, device=device, dtype=float_dtype),
        ]
        nt = NT(parts)
        for dims in ((1, 2), (1, 3)):
            output = torch.count_nonzero(nt, dim=dims)
            element_dims = tuple(dim - 1 for dim in dims)
            reference = torch.stack([torch.count_nonzero(t, dim=element_dims) for t in parts])
            assert_close(output, reference)

        nt = NT(parts, batch_first=False)
        count = torch.count_nonzero(nt, dim=(0, 3))
        count_reference = torch.stack([torch.count_nonzero(t, dim=(0, 2)) for t in parts])
        assert_close(count, count_reference)

        summed = torch.sum(nt, dim=(0, 3))
        sum_reference = torch.stack([torch.sum(t, dim=(0, 2)) for t in parts])
        assert_close(summed, sum_reference)

    def test_linalg_matrix_norm_after_multi_ragged_dims(self, device, float_dtype):
        parts = [
            torch.randn(2, 2, 3, 4, device=device, dtype=float_dtype),
            torch.randn(3, 3, 3, 4, device=device, dtype=float_dtype),
        ]
        nt = NT(parts)
        for keepdim in (False, True):
            output = torch.linalg.norm(nt, dim=(-2, -1), keepdim=keepdim)
            reference = NT(
                [torch.linalg.norm(t, dim=(-2, -1), keepdim=keepdim) for t in parts],
                **reference_options(nt),
            )
            assert_close(output, reference)

    @pytest.mark.parametrize(
        ("shapes", "logical_dim", "element_dim"),
        [
            (((4, 3), (2, 3)), 0, 0),
            (((3, 4), (3, 2)), 2, 1),
        ],
        ids=("leading-ragged", "non-leading-ragged"),
    )
    def test_segment_reductions_respect_batch_first_false(
        self,
        device,
        float_dtype,
        shapes,
        logical_dim,
        element_dim,
    ):
        parts = [torch.randn(shape, device=device, dtype=float_dtype) for shape in shapes]
        nt = NT(parts, batch_first=False)

        for keepdim in (False, True):
            for op in (torch.argmax, torch.argmin):
                output = op(nt, dim=logical_dim, keepdim=keepdim)
                reference = torch.stack([op(t, dim=element_dim, keepdim=keepdim) for t in parts])
                if keepdim or element_dim != 0:
                    reference = reference.movedim(0, 1)
                assert_close(output, reference)

            norm = torch.linalg.vector_norm(nt, dim=logical_dim, keepdim=keepdim)
            norm_reference = NT(
                [torch.linalg.vector_norm(t, dim=element_dim, keepdim=keepdim) for t in parts],
                **reference_options(nt),
            )
            assert norm.shape == norm_reference.shape
            assert_close(norm, norm_reference)

        count = torch.count_nonzero(nt, dim=logical_dim)
        count_reference = torch.stack([torch.count_nonzero(t, dim=element_dim) for t in parts])
        if element_dim != 0:
            count_reference = count_reference.movedim(0, 1)
        assert_close(count, count_reference)


class TestRot90:

    def test_rot90_translates_dims_and_rejects_batch(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=float_dtype),
                torch.tensor([[5.0, 6.0]], device=device, dtype=float_dtype),
            ],
            padding_value=123.0,
        )

        output = torch.rot90(nt, k=1, dims=(1, 2))
        reference = NT([torch.rot90(t, k=1, dims=(0, 1)) for t in nt], **reference_options(nt))
        assert_close(output, reference)

        with pytest.raises(ValueError):
            torch.rot90(nt)

    @pytest.mark.parametrize(
        ("k", "expected_shapes"),
        [
            pytest.param(1, ((2, 5, 3), (4, 5, 3)), id="k1_swaps_plane"),
            pytest.param(2, ((2, 3, 5), (4, 3, 5)), id="k2_preserves_plane"),
        ],
    )
    def test_rot90_static_plane_metadata_respects_k(self, device, float_dtype, k, expected_shapes):
        nt = NT(
            [
                torch.randn(2, 3, 5, device=device, dtype=float_dtype),
                torch.randn(4, 3, 5, device=device, dtype=float_dtype),
            ]
        )

        output = torch.rot90(nt, k=k, dims=(2, 3))
        reference = NT([torch.rot90(t, k=k, dims=(1, 2)) for t in nt], **reference_options(nt))
        assert_close(output, reference)
        assert tuple(tuple(element.shape) for element in output) == expected_shapes


class TestPackedCoordinateTransforms:

    @pytest.mark.parametrize(
        ("operation", "shapes", "ragged_dims"),
        [
            ("roll", ((2, 4, 3), (3, 2, 3)), (1, 0)),
            ("roll_flat", ((2, 3), (2, 5)), (1,)),
            ("rot90", ((2, 3), (2, 5)), (1,)),
            ("diag_vector", ((0,), (4,)), (0,)),
            ("diag_matrix", ((2, 3), (4, 3)), (0,)),
            ("diagflat", ((2, 3), (2, 5)), (1,)),
        ],
    )
    def test_fullgraph_values_and_vjp(self, device, operation, shapes, ragged_dims):
        leaves = [torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
        references = [leaf.detach().clone().requires_grad_() for leaf in leaves]
        source = NT(leaves, ragged_dims=ragged_dims, batch_first=False)
        if operation == "roll":
            call = partial(torch.roll, shifts=(2, -1), dims=(0, 2))
            native = partial(torch.roll, shifts=(2, -1), dims=(0, 1))
        elif operation == "roll_flat":
            call = native = partial(torch.roll, shifts=3)
        elif operation == "rot90":
            call = partial(torch.rot90, k=1, dims=(0, 2))
            native = partial(torch.rot90, k=1, dims=(0, 1))
        elif operation == "diagflat":
            call = native = partial(torch.diagflat, offset=-1)
        else:
            call = native = partial(torch.diag, diagonal=-1)

        torch._dynamo.reset()
        compiled = torch.compile(call, backend="aot_eager", fullgraph=True, dynamic=True)
        output = compiled(source)
        expected = [native(value) for value in references]
        weights = [torch.randn_like(value) for value in expected]
        for actual, reference in zip(output, expected):
            torch.testing.assert_close(actual, reference)
        actual_loss = sum((value * weight).sum() for value, weight in zip(output, weights))
        expected_loss = sum((value * weight).sum() for value, weight in zip(expected, weights))
        gradients = torch.autograd.grad(actual_loss, leaves)
        reference_gradients = torch.autograd.grad(expected_loss, references)
        for actual, reference in zip(gradients, reference_gradients):
            torch.testing.assert_close(actual, reference)


class TestSearchsorted:

    def test_bucketize(self, device, float_dtype):
        boundaries = torch.tensor([1.0, 3.0, 5.0], device=device, dtype=float_dtype)
        a = torch.tensor([0.5, 2.0, 4.0, 6.0], device=device, dtype=float_dtype)
        b = torch.tensor([1.5, 3.5], device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.bucketize(nt, boundaries)
        assert_close(result[0], torch.bucketize(a, boundaries))
        assert_close(result[1], torch.bucketize(b, boundaries))

    def test_searchsorted_both_nested(self, device, float_dtype):
        sorted_a = torch.tensor([1.0, 3.0, 5.0], device=device, dtype=float_dtype)
        sorted_b = torch.tensor([2.0, 4.0, 6.0, 8.0], device=device, dtype=float_dtype)
        vals_a = torch.tensor([2.0, 4.0], device=device, dtype=float_dtype)
        vals_b = torch.tensor([3.0, 7.0], device=device, dtype=float_dtype)
        nt_sorted = NT([sorted_a, sorted_b])
        nt_vals = NT([vals_a, vals_b])
        result = torch.searchsorted(nt_sorted, nt_vals)
        assert_close(result[0], torch.searchsorted(sorted_a, vals_a))
        assert_close(result[1], torch.searchsorted(sorted_b, vals_b))

    def test_searchsorted_mismatched_batch_lengths_raises(self, device, float_dtype):
        sorted_a = torch.tensor([1.0, 3.0, 5.0], device=device, dtype=float_dtype)
        vals_a = torch.tensor([2.0, 4.0], device=device, dtype=float_dtype)
        nt_sorted = NT([sorted_a, sorted_a])
        nt_vals = NT([vals_a])
        with pytest.raises(ValueError, match="searchsorted: NestedTensor batch length mismatch"):
            torch.searchsorted(nt_sorted, nt_vals)

    def test_searchsorted_nested_sorter_requires_nested_sorted_sequence(self, device, float_dtype):
        boundaries = torch.tensor([1.0, 3.0, 5.0], device=device, dtype=float_dtype)
        vals_a = torch.tensor([2.0, 4.0], device=device, dtype=float_dtype)
        nt_vals = NT([vals_a])
        nt_sorter = NT([torch.tensor([0, 1, 2], device=device, dtype=torch.long)])
        with pytest.raises(TypeError, match="NestedTensor sorter requires sorted_sequence"):
            torch.searchsorted(boundaries, nt_vals, sorter=nt_sorter)

    def test_searchsorted_nested_sorter_with_nested_sorted_sequence(self, device, float_dtype):
        sorted_a = torch.tensor([1.0, 3.0, 5.0], device=device, dtype=float_dtype)
        sorted_b = torch.tensor([2.0, 4.0, 6.0], device=device, dtype=float_dtype)
        sorter_a = torch.tensor([0, 1, 2], device=device, dtype=torch.long)
        sorter_b = torch.tensor([0, 1, 2], device=device, dtype=torch.long)
        nt_sorted = NT([sorted_a, sorted_b])
        nt_sorter = NT([sorter_a, sorter_b])
        result = torch.searchsorted(nt_sorted, float(3.5), sorter=nt_sorter)
        reference = NT(
            [
                torch.searchsorted(sorted_a, float(3.5), sorter=sorter_a),
                torch.searchsorted(sorted_b, float(3.5), sorter=sorter_b),
            ],
            **reference_options(nt_sorted),
        )
        assert_close(result, reference)

    def test_searchsorted_shared_boundaries(self, device, float_dtype):
        boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0], device=device, dtype=float_dtype)
        vals_a = torch.tensor([2.0, 4.0, 6.0], device=device, dtype=float_dtype)
        vals_b = torch.tensor([0.5, 8.0], device=device, dtype=float_dtype)
        nt_vals = NT([vals_a, vals_b])
        result = torch.searchsorted(boundaries, nt_vals)
        assert_close(result[0], torch.searchsorted(boundaries, vals_a))
        assert_close(result[1], torch.searchsorted(boundaries, vals_b))


class TestSelectionOps:

    @pytest.mark.parametrize("name", ["min", "max"])
    @pytest.mark.parametrize("interface", ["function", "method"])
    @pytest.mark.parametrize("backend", ["eager", "aot_eager"])
    @pytest.mark.parametrize("reduction", ["global", "ragged", "static", "static_keepdim"])
    def test_extrema_dispatch_gradients(self, device, name, interface, backend, reduction):
        torch.compiler.reset()
        operation = getattr(torch if interface == "function" else torch.Tensor, name)
        dimension = 1 if reduction == "ragged" else -1

        def reduce(value):
            if reduction == "global":
                return operation(value)
            return operation(value, dim=dimension, keepdim=reduction == "static_keepdim")

        run = reduce if backend == "eager" else torch.compile(reduce, backend=backend, fullgraph=True, dynamic=True)
        for lengths in ((2, 4), (3, 5)):
            parts = [
                torch.arange(n * 3, device=device, dtype=torch.float64).reshape(n, 1, 3).remainder(2).requires_grad_()
                for n in lengths
            ]
            nested = NT(parts, ragged_dims=(0,))
            dense_operation = getattr(torch, name)
            keepdim = reduction == "static_keepdim"
            axis = 0 if reduction == "ragged" else -1
            if reduction == "global":
                expected = dense_operation(torch.cat([part.detach() for part in parts]))
            else:
                pairs = [dense_operation(part.detach(), dim=axis, keepdim=keepdim) for part in parts]
                combine = torch.stack if reduction == "ragged" else torch.cat
                expected = combine([pair.values for pair in pairs])
                expected_indices = combine([pair.indices for pair in pairs])

            result = run(nested)
            if reduction == "global":
                values = result
            else:
                values = result.values.concat if isinstance(result.values, NestedTensor) else result.values
                indices = result.indices.concat if isinstance(result.indices, NestedTensor) else result.indices
                torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
            torch.testing.assert_close(values, expected, rtol=0, atol=0)
            weights = torch.arange(1, 2 * values.numel() + 1, device=device, dtype=torch.float64)
            weights = weights.reshape(*values.shape, 2)[..., 0]
            gradients = torch.autograd.grad(values, parts, weights)

            if reduction == "global":
                count = sum((part == expected).sum() for part in parts)
                expected_gradients = [(part == expected).to(part.dtype) * weights / count for part in parts]
            else:
                element_weights = weights.unbind(0) if reduction == "ragged" else weights.split(lengths)
                expected_gradients = [
                    torch.zeros_like(part).scatter(
                        axis,
                        pair.indices if keepdim else pair.indices.unsqueeze(axis),
                        weight if keepdim else weight.unsqueeze(axis),
                    )
                    for part, pair, weight in zip(parts, pairs, element_weights)
                ]
            for actual, wanted in zip(gradients, expected_gradients):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

    def test_argmax_and_argmin_default_dim(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 5, 3], device=device, dtype=float_dtype),
                torch.tensor([4, 2], device=device, dtype=float_dtype),
            ]
        )
        torch.compiler.reset()
        output = torch.compile(torch.argmax, backend="aot_eager", fullgraph=True)(nt)
        reference = torch.tensor([1, 0], device=device)
        assert_close(output, reference)
        output = torch.compile(torch.argmin, backend="aot_eager", fullgraph=True)(nt)
        reference = torch.tensor([0, 1], device=device)
        assert_close(output, reference)

    def test_argmax_and_argmin_dim(self, device, float_dtype):
        a = torch.tensor([[1, 5, 3], [2, 0, 4]], device=device, dtype=float_dtype)
        b = torch.tensor([[9, 1, 2], [3, 7, 6], [4, 0, 8]], device=device, dtype=float_dtype)
        nt = NestedTensor([a, b])

        argmax_last = torch.argmax(nt, dim=2)
        argmax_last_ref = NestedTensor([torch.argmax(a, dim=1), torch.argmax(b, dim=1)], **reference_options(nt))
        assert_close(argmax_last, argmax_last_ref)

        argmin_ragged = torch.argmin(nt, dim=1)
        argmin_ragged_ref = torch.stack([torch.argmin(a, dim=0), torch.argmin(b, dim=0)])
        assert_close(argmin_ragged, argmin_ragged_ref)

    def test_argsort(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 3, 2], device=device, dtype=float_dtype),
                torch.tensor([4, 0], device=device, dtype=float_dtype),
            ]
        )
        reference = NT([torch.argsort(element) for element in nt], **reference_options(nt))
        assert_close(torch.argsort(nt), reference)

    def test_max_and_min_dim(self, device, float_dtype):
        a = torch.tensor([[1, 5, 3], [2, 0, 4]], device=device, dtype=float_dtype)
        b = torch.tensor([[9, 1, 2], [3, 7, 6], [4, 0, 8]], device=device, dtype=float_dtype)
        nt = NestedTensor([a, b])

        max_last = torch.max(nt, dim=2)
        max_last_ref_vals = NestedTensor(
            [torch.max(a, dim=1).values, torch.max(b, dim=1).values], **reference_options(nt)
        )
        max_last_ref_idxs = NestedTensor(
            [torch.max(a, dim=1).indices, torch.max(b, dim=1).indices], **reference_options(nt)
        )
        assert_close(max_last.values, max_last_ref_vals)
        assert_close(max_last.indices, max_last_ref_idxs)

        min_ragged = torch.min(nt, dim=1)
        min_ragged_ref_vals = torch.stack([torch.min(a, dim=0).values, torch.min(b, dim=0).values])
        min_ragged_ref_idxs = torch.stack([torch.min(a, dim=0).indices, torch.min(b, dim=0).indices])
        assert_close(min_ragged.values, min_ragged_ref_vals)
        assert_close(min_ragged.indices, min_ragged_ref_idxs)

    @pytest.mark.parametrize(
        ("nested_op", "element_op"),
        [
            (lambda value: torch.amax(value, dim=-1), lambda value: torch.amax(value, dim=-1)),
            (lambda value: torch.max(value, dim=-1).values, lambda value: torch.max(value, dim=-1).values),
            (lambda value: torch.topk(value, 2, dim=-1).values, lambda value: torch.topk(value, 2, dim=-1).values),
        ],
        ids=("amax", "max", "topk"),
    )
    def test_extrema_values_and_vjp(self, device, float_dtype, nested_op, element_op):
        leaves = [
            torch.randn(2, 4, device=device, dtype=float_dtype, requires_grad=True),
            torch.randn(3, 4, device=device, dtype=float_dtype, requires_grad=True),
        ]
        references = [leaf.detach().clone().requires_grad_() for leaf in leaves]
        output = nested_op(NT(leaves))
        expected = [element_op(reference) for reference in references]
        weights = [torch.randn_like(element) for element in expected]

        actual_loss = sum((element * weight).sum() for element, weight in zip(output, weights))
        expected_loss = sum((element * weight).sum() for element, weight in zip(expected, weights))
        actual_gradients = torch.autograd.grad(actual_loss, leaves)
        expected_gradients = torch.autograd.grad(expected_loss, references)

        for actual, reference in zip(output, expected):
            assert_close(actual, reference)
        for actual, reference in zip(actual_gradients, expected_gradients):
            assert_close(actual, reference)

    def test_non_leading_ragged_topk_and_max_min(self, device, float_dtype):
        leaves = [
            torch.randn(2, 3, device=device, dtype=float_dtype, requires_grad=True),
            torch.randn(2, 5, device=device, dtype=float_dtype, requires_grad=True),
        ]
        nt = NT(leaves)
        output = torch.topk(nt, 2, dim=2)
        references = [torch.topk(t, 2, dim=1) for t in leaves]
        assert_close(output.values, NT([reference.values for reference in references], **reference_options(nt)))
        assert_close(output.indices, NT([reference.indices for reference in references], **reference_options(nt)))
        torch.autograd.grad(output.values.concat.sum(), leaves, retain_graph=True)

        for op in (torch.max, torch.min):
            output = op(nt, dim=1)
            references = [op(t, dim=0) for t in leaves]
            assert_close(output.values, NT([reference.values for reference in references], **reference_options(nt)))
            assert_close(output.indices, NT([reference.indices for reference in references], **reference_options(nt)))
            assert torch.isfinite(output.values.concat).all()
            torch.autograd.grad(output.values.concat.sum(), leaves, retain_graph=True)

    @pytest.mark.parametrize("name", ["argmax", "argmin", "max", "min"])
    def test_empty_ragged_extrema_axis_raises(self, device, float_dtype, name):
        nt = NT(
            [
                torch.empty(0, 3, device=device, dtype=float_dtype),
                torch.randn(2, 3, device=device, dtype=float_dtype),
            ]
        )
        with pytest.raises(IndexError, match="non-zero size"):
            getattr(torch, name)(nt, dim=1)

    @pytest.mark.parametrize("name", ["max", "min"])
    def test_extrema_ties_route_gradient_to_returned_index(self, device, float_dtype, name):
        torch.compiler.reset()
        high = 2.0 if name == "max" else -2.0
        leaves = [
            torch.tensor([high, high, 0.0], device=device, dtype=float_dtype, requires_grad=True),
            torch.tensor([high, high], device=device, dtype=float_dtype, requires_grad=True),
        ]
        nt = NT(leaves)

        def consume(template, values):
            return getattr(template.packed_like(values), name)(dim=1)

        output = torch.compile(consume, backend="aot_eager", fullgraph=True)(nt, nt.concat)
        assert_close(output.indices, torch.zeros(2, device=device, dtype=torch.long))
        loss = output.values.concat.sum() if isinstance(output.values, NestedTensor) else output.values.sum()
        grads = torch.autograd.grad(loss, leaves)
        assert_close(grads[0], torch.tensor([1.0, 0.0, 0.0], device=device, dtype=float_dtype))
        assert_close(grads[1], torch.tensor([1.0, 0.0], device=device, dtype=float_dtype))

    def test_sort(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 3, 2], device=device, dtype=float_dtype),
                torch.tensor([4, 0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.sort(nt, dim=1, descending=False)
        reference = (
            torch.tensor([[1, 2, 3], [0, 4, 0]], device=device, dtype=float_dtype),
            output[1],
        )
        assert_close(output[0], reference[0])

    def test_topk(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 3, 2], device=device, dtype=float_dtype),
                torch.tensor([4, 0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.topk(nt, k=2, dim=1)
        reference = (
            NestedTensor(
                [
                    torch.tensor([3, 2], device=device, dtype=float_dtype),
                    torch.tensor([4, 0], device=device, dtype=float_dtype),
                ],
                **reference_options(nt),
            ),
            NestedTensor(
                [torch.tensor([1, 2], device=device), torch.tensor([0, 1], device=device)], **reference_options(nt)
            ),
        )
        assert_close(output[0], reference[0])
        assert_close(output[1], reference[1])


class TestSoftmaxVariants:

    def test_softmax_and_log_softmax_translate_dim(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(1, 3, device=device, dtype=float_dtype),
            ]
        )
        output = torch.softmax(nt, dim=2)
        reference = NT([torch.softmax(t, dim=1) for t in nt], **reference_options(nt))
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

        output = torch.log_softmax(nt, dim=2)
        reference = NT([torch.log_softmax(t, dim=1) for t in nt], **reference_options(nt))
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

        with pytest.raises(ValueError):
            torch.softmax(nt, dim=0)

    def test_softmax_ragged_last_dim(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(3, 4, device=device, dtype=float_dtype),
                torch.randn(3, 7, device=device, dtype=float_dtype),
            ]
        )
        output = torch.softmax(nt, dim=-1)
        reference = NT([torch.softmax(t, dim=-1) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    @pytest.mark.parametrize("name", ["softmax", "log_softmax"])
    def test_softmax_dtype_values_and_vjp(self, device, float_dtype, name):
        leaves = [
            torch.randn(2, 3, device=device, dtype=float_dtype, requires_grad=True),
            torch.randn(4, 3, device=device, dtype=float_dtype, requires_grad=True),
        ]
        references = [leaf.detach().clone().requires_grad_() for leaf in leaves]
        nt = NT(leaves)
        output = getattr(torch, name)(nt, dim=-1, dtype=torch.float64)

        reference = [getattr(torch, name)(tensor, dim=-1, dtype=torch.float64) for tensor in references]
        assert output.dtype == torch.float64
        assert_close(output, NT(reference, **reference_options(nt)), atol=1e-6, rtol=1e-6)

        weights = [torch.randn_like(tensor) for tensor in reference]
        loss = sum((tensor * weight).sum() for tensor, weight in zip(output, weights))
        ref_loss = sum((tensor * weight).sum() for tensor, weight in zip(reference, weights))
        grads = torch.autograd.grad(loss, leaves)
        ref_grads = torch.autograd.grad(ref_loss, references)
        for grad, ref_grad in zip(grads, ref_grads):
            assert_close(grad, ref_grad, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("name", ["softmax", "log_softmax"])
    def test_multi_ragged_softmax_all_varying_dims_preserves_grad(self, device, float_dtype, name):
        for dim in (1, 2):
            leaves = [
                torch.randn(2, 3, 4, device=device, dtype=float_dtype, requires_grad=True),
                torch.randn(4, 2, 4, device=device, dtype=float_dtype, requires_grad=True),
            ]
            references = [leaf.detach().clone().requires_grad_() for leaf in leaves]
            nt = NT(leaves)
            output = getattr(torch, name)(nt, dim=dim)
            reference = [getattr(torch, name)(tensor, dim=dim - 1) for tensor in references]
            value_atol, value_rtol = low_precision_cuda_tolerances(
                device,
                float_dtype,
                default=(1e-5, 1e-5),
                fp16=(1e-3, 1e-3),
                bf16=(1e-2, 1e-2),
            )
            assert_close(output, NT(reference, **reference_options(nt)), atol=value_atol, rtol=value_rtol)

            weights = [torch.randn_like(tensor) for tensor in reference]
            loss = sum((tensor * weight).sum() for tensor, weight in zip(output, weights))
            ref_loss = sum((tensor * weight).sum() for tensor, weight in zip(reference, weights))
            grads = torch.autograd.grad(loss, leaves)
            ref_grads = torch.autograd.grad(ref_loss, references)
            atol, rtol = low_precision_cuda_tolerances(
                device,
                float_dtype,
                default=(1e-4, 1e-4),
                fp16=(1e-3, 1e-3),
                bf16=(1e-2, 1e-2),
            )
            for grad, ref_grad in zip(grads, ref_grads):
                assert_close(grad, ref_grad, atol=atol, rtol=rtol)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_multi_ragged_softmax_compile_fullgraph(self):
        nt = NT([torch.randn(2, 3, 4), torch.randn(4, 2, 4)])
        compiled = torch.compile(lambda tensor: torch.softmax(tensor, dim=1), backend="inductor", fullgraph=True)
        assert_close(compiled(nt), torch.softmax(nt, dim=1))


class TestCreationFromExisting:
    r"""``new_*`` take a fully specified shape, so they cannot produce a ragged result."""

    def test_new_family_returns_plain_tensors(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(4, 3, device=device, dtype=float_dtype),
            ]
        )

        scalar = nt.new_zeros(())
        assert type(scalar) is torch.Tensor
        assert scalar.shape == torch.Size([])

        for output in (nt.new_zeros(2, 3), nt.new_ones([2, 3]), nt.new_empty((2, 3))):
            assert type(output) is torch.Tensor
            assert output.shape == torch.Size([2, 3])
            assert output.dtype == nt.dtype and output.device == nt.device

        full = nt.new_full((2, 3), 7.0)
        assert type(full) is torch.Tensor
        assert_close(full, torch.full((2, 3), 7.0, device=device, dtype=float_dtype))

        strided = nt.new_empty_strided((2, 3), (3, 1))
        assert type(strided) is torch.Tensor
        assert strided.shape == torch.Size([2, 3])
        assert strided.stride() == (3, 1)


class TestCdist:

    @staticmethod
    def _skip_unsupported_runtime_dtype(device, float_dtype):
        if device.type == "cuda" and float_dtype in (torch.float16, torch.bfloat16):
            pytest.skip("dense torch.cdist does not implement CUDA float16/bfloat16")

    @pytest.mark.parametrize(
        ("p", "compute_mode"),
        [
            (1.0, "donot_use_mm_for_euclid_dist"),
            (2.0, "use_mm_for_euclid_dist"),
            (float("inf"), "use_mm_for_euclid_dist_if_necessary"),
        ],
    )
    def test_values_shape_and_vjp_match_dense(self, device, p, compute_mode):
        left_values = torch.randn(7, 4, device=device, requires_grad=True)
        right_values = torch.randn(6, 4, device=device, requires_grad=True)
        left_parts = left_values.split((2, 5))
        right_parts = right_values.split((4, 2))
        left = NT(left_parts, ragged_dims=(0,))
        right = NT(right_parts, ragged_dims=(0,))

        output = left.cdist(right, p=p, compute_mode=compute_mode)
        reference = torch.cat(
            [
                torch.cdist(lhs, rhs, p=p, compute_mode=compute_mode).reshape(-1)
                for lhs, rhs in zip(left_parts, right_parts)
            ]
        )
        cotangent = torch.randn_like(reference)
        actual_gradients = torch.autograd.grad(
            output.concat,
            (left_values, right_values),
            cotangent,
            retain_graph=True,
        )
        expected_gradients = torch.autograd.grad(reference, (left_values, right_values), cotangent)

        assert output.ragged_dims == (0, 1)
        assert output.element_sizes().tolist() == [[2, 4], [5, 2]]
        torch.testing.assert_close(output.concat, reference)
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, expected)

    def test_function_and_method_forms_match(self, device, float_dtype):
        self._skip_unsupported_runtime_dtype(device, float_dtype)
        left = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(4, 3, device=device, dtype=float_dtype),
            ]
        )
        right = NT(
            [
                torch.randn(3, 3, device=device, dtype=float_dtype),
                torch.randn(1, 3, device=device, dtype=float_dtype),
            ]
        )
        assert_close(torch.cdist(left, right), left.cdist(right))

    def test_mixed_dense_operand_preserves_logical_topology(self, device):
        nested = NT(
            [
                torch.randn(2, 4, device=device),
                torch.randn(3, 4, device=device),
            ],
            ragged_dims=(0,),
        )
        dense = torch.randn(5, 4, device=device)

        output = torch.cdist(nested, dense)

        assert output.ragged_dims == (0,)
        assert [element.shape for element in output] == [torch.Size((2, 5)), torch.Size((3, 5))]
        assert_close(output, NT([torch.cdist(element, dense) for element in nested], ragged_dims=(0,)))

    def test_empty_points_keep_shape_and_autograd(self, device):
        left_values = torch.randn(2, 3, device=device, requires_grad=True)
        right_values = torch.randn(4, 3, device=device, requires_grad=True)
        left_parts = (left_values[:0], left_values)
        right_parts = (right_values[:3], right_values[3:])
        left = NT(left_parts, ragged_dims=(0,))
        right = NT(right_parts, ragged_dims=(0,))

        output = left.cdist(right)
        reference = torch.cat([torch.cdist(lhs, rhs).reshape(-1) for lhs, rhs in zip(left_parts, right_parts)])
        gradients = torch.autograd.grad(output.concat.sum(), (left_values, right_values))

        assert output.element_sizes().tolist() == [[0, 3], [2, 1]]
        torch.testing.assert_close(output.concat, reference)
        assert_close(gradients[0], torch.autograd.grad(reference.sum(), left_values, retain_graph=True)[0])
        assert_close(gradients[1], torch.autograd.grad(reference.sum(), right_values)[0])

    def test_batch_length_and_feature_mismatch_raise(self, device):
        left = NT([torch.randn(2, 3, device=device)], ragged_dims=(0,))
        with pytest.raises(ValueError, match="batch length mismatch"):
            torch.cdist(left, NT([torch.randn(2, 3, device=device), torch.randn(1, 3, device=device)]))
        with pytest.raises(RuntimeError, match="same number of columns"):
            torch.cdist(left, NT([torch.randn(4, 5, device=device)], ragged_dims=(0,)))

    def test_autocast_matches_dense(self, device):
        dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
        left_values = torch.randn(4, 3, device=device, dtype=dtype, requires_grad=True)
        right_values = torch.randn(5, 3, device=device, dtype=dtype, requires_grad=True)
        left = NT([left_values], ragged_dims=(0,))
        right = NT([right_values], ragged_dims=(0,))

        with torch.autocast(device.type, dtype=dtype):
            output = left.cdist(right).concat
            reference = torch.cdist(left_values, right_values).reshape(-1)

        assert output.dtype == reference.dtype
        torch.testing.assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize("public_function", [False, True], ids=["method", "torch"])
    def test_compile_fullgraph_preserves_values_and_vjp(self, device, public_function):
        left_values = torch.randn(7, 3, device=device, requires_grad=True)
        right_values = torch.randn(5, 3, device=device, requires_grad=True)
        left_parts = left_values.split((2, 5))
        right_parts = right_values.split((4, 1))
        left = NT(left_parts, ragged_dims=(0,), batch_first=False)
        right = NT(right_parts, ragged_dims=(0,), batch_first=False)
        compiled = torch.compile(
            lambda lhs, rhs: (torch.cdist(lhs, rhs) if public_function else lhs.cdist(rhs)).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        output = compiled(left, right)
        reference = torch.cat([torch.cdist(lhs, rhs).reshape(-1) for lhs, rhs in zip(left_parts, right_parts)])
        cotangent = torch.randn_like(reference)
        actual_gradients = torch.autograd.grad(
            output,
            (left_values, right_values),
            cotangent,
            retain_graph=True,
        )
        expected_gradients = torch.autograd.grad(reference, (left_values, right_values), cotangent)

        torch.testing.assert_close(output, reference)
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, expected)


class TestSegmentedSort:

    def test_sort_and_argsort_match_per_element(self, device, float_dtype):
        elements = [torch.randn(n, 3, device=device, dtype=float_dtype) for n in (3, 0, 5)]
        nested = NT(elements, ragged_dims=(0,))

        result = torch.sort(nested, dim=1)
        expected = [torch.sort(element, dim=0) for element in elements]

        assert_close(result.values, NT([item.values for item in expected], ragged_dims=(0,)))
        assert_close(result.indices, NT([item.indices for item in expected], ragged_dims=(0,)))
        assert_close(torch.argsort(nested, dim=1), result.indices)

    def test_topk_values_indices_and_errors_match_dense(self, device, float_dtype):
        elements = [torch.randn(n, 3, device=device, dtype=float_dtype) for n in (3, 5, 4)]
        nested = NT(elements, ragged_dims=(0,))

        result = torch.topk(nested, 2, dim=1, largest=False)
        expected = [torch.topk(element, 2, dim=0, largest=False) for element in elements]

        assert_close(result.values, NT([item.values for item in expected], ragged_dims=(0,)))
        assert_close(result.indices, NT([item.indices for item in expected], ragged_dims=(0,)))
        with pytest.raises(RuntimeError):
            torch.topk(nested, 4, dim=1)

    def test_sort_nonleading_declared_ragged_dim(self, device, float_dtype):
        elements = [torch.randn(2, n, 3, device=device, dtype=float_dtype) for n in (3, 5)]
        nested = NT(elements, ragged_dims=(1,))

        result = torch.sort(nested, dim=2)

        assert result.values.ragged_dims == (1,)
        assert_close(result.values, NT([torch.sort(element, dim=1).values for element in elements], ragged_dims=(1,)))
        assert_close(
            result.indices,
            NT([torch.sort(element, dim=1).indices for element in elements], ragged_dims=(1,)),
        )

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_sort_and_topk_compile_fullgraph(self, device, float_dtype):
        nested = NT(
            [torch.randn(n, device=device, dtype=float_dtype) for n in (3, 5, 2)],
            ragged_dims=(0,),
        )

        def run(value):
            sorted_values = torch.sort(value, dim=1).values
            top_values = torch.topk(value, 2, dim=1).values
            return sorted_values.concat, top_values.concat

        compiled = torch.compile(run, backend="aot_eager", fullgraph=True, dynamic=True)
        actual = compiled(nested)
        expected = run(nested)

        assert_close(actual[0], expected[0])
        assert_close(actual[1], expected[1])


class TestSegmentedCumulative:

    def test_cumprod_preserves_native_second_derivatives_at_zero(self):
        values = torch.tensor([0.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64, requires_grad=True)
        reference = values.detach().clone().requires_grad_()
        output = NT(values.split((3, 2))).cumprod(dim=1).concat
        expected = torch.cat([part.cumprod(0) for part in reference.split((3, 2))])
        gradient = torch.autograd.grad(output.sum(), values, create_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected.sum(), reference, create_graph=True)[0]
        second = torch.autograd.grad(gradient.sum(), values)[0]
        expected_second = torch.autograd.grad(expected_gradient.sum(), reference)[0]
        assert_close(gradient, expected_gradient)
        assert_close(second, expected_second)

    def test_cumprod_and_logcumsumexp_match_per_element(self, device, float_dtype):
        elements = [torch.randn(n, 3, device=device, dtype=float_dtype) for n in (3, 0, 5)]
        nested = NT(elements, ragged_dims=(0,))

        assert_close(
            torch.cumprod(nested, dim=1),
            NT([torch.cumprod(element, dim=0) for element in elements], ragged_dims=(0,)),
        )
        assert_close(
            torch.logcumsumexp(nested, dim=1),
            NT([torch.logcumsumexp(element, dim=0) for element in elements], ragged_dims=(0,)),
        )

    def test_cumprod_vjp_matches_per_element(self, device, float_dtype):
        values = torch.randn(8, 2, device=device, dtype=float_dtype, requires_grad=True)
        parts = values.split((3, 5))
        nested = NT(parts, ragged_dims=(0,))

        output = torch.cumprod(nested, dim=1).concat
        reference = torch.cat([torch.cumprod(part, dim=0) for part in parts])
        cotangent = torch.randn_like(reference)
        actual_gradient = torch.autograd.grad(output, values, cotangent, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(reference, values, cotangent)[0]

        assert_close(output, reference)
        assert_close(actual_gradient, expected_gradient)

    def test_cummax_and_cummin_values_and_indices(self, device, float_dtype):
        elements = [torch.randn(n, 3, device=device, dtype=float_dtype) for n in (3, 1, 5)]
        nested = NT(elements, ragged_dims=(0,))

        for op in (torch.cummax, torch.cummin):
            result = op(nested, dim=1)
            expected = [op(element, dim=0) for element in elements]
            assert_close(result.values, NT([item.values for item in expected], ragged_dims=(0,)))
            assert_close(result.indices, NT([item.indices for item in expected], ragged_dims=(0,)))

    @pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
    def test_cumprod_matches_dense_dtype_promotion(self, device, dtype):
        elements = [
            torch.tensor([1, 0, 1], device=device, dtype=dtype),
            torch.tensor([1, 1], device=device, dtype=dtype),
        ]
        nested = NT(elements, ragged_dims=(0,))

        output = torch.cumprod(nested, dim=1)

        assert output.dtype == torch.cumprod(elements[0], dim=0).dtype
        assert_close(output, NT([torch.cumprod(element, dim=0) for element in elements], ragged_dims=(0,)))

    def test_cumulative_ops_nonleading_ragged_dim(self, device, float_dtype):
        elements = [torch.randn(2, n, device=device, dtype=float_dtype) for n in (3, 5)]
        nested = NT(elements, ragged_dims=(1,))

        assert_close(
            torch.cumprod(nested, dim=2),
            NT([torch.cumprod(element, dim=1) for element in elements], ragged_dims=(1,)),
        )
        result = torch.cummax(nested, dim=2)
        assert_close(
            result.values,
            NT([torch.cummax(element, dim=1).values for element in elements], ragged_dims=(1,)),
        )
        assert_close(
            result.indices,
            NT([torch.cummax(element, dim=1).indices for element in elements], ragged_dims=(1,)),
        )

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize("public_function", [False, True], ids=["method", "torch"])
    def test_cumprod_compile_fullgraph_backward(self, device, public_function):
        values = torch.randn(8, 2, device=device, requires_grad=True)
        parts = values.split((3, 5))
        nested = NT(parts, ragged_dims=(0,))
        compiled = torch.compile(
            lambda value: (torch.cumprod(value, dim=1) if public_function else value.cumprod(dim=1)).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        output = compiled(nested)
        reference = torch.cat([torch.cumprod(part, dim=0) for part in parts])
        cotangent = torch.randn_like(reference)
        actual_gradient = torch.autograd.grad(output, values, cotangent, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(reference, values, cotangent)[0]

        assert_close(output, reference)
        assert_close(actual_gradient, expected_gradient)


class TestInversePermutation:

    def test_inverse_and_round_trip(self, device, float_dtype):
        from danling.tensors import inverse_permutation

        permutations = [torch.randperm(n, device=device) for n in (3, 0, 5, 2)]
        values = [torch.randn(n, device=device, dtype=float_dtype) for n in (3, 0, 5, 2)]
        nested = NT(permutations, ragged_dims=(0,))

        inverse = inverse_permutation(nested)
        expected = NT([torch.argsort(permutation) for permutation in permutations], ragged_dims=(0,))

        assert_close(inverse, expected)
        for element, permutation, undo in zip(values, permutations, inverse):
            assert_close(element[permutation][undo], element)

    def test_static_tail_and_nonleading_ragged_dim(self, device):
        from danling.tensors import inverse_permutation

        elements = [torch.stack([torch.randperm(n, device=device) for _ in range(2)]) for n in (3, 5)]
        nested = NT(elements, ragged_dims=(1,))

        output = inverse_permutation(nested, dim=2)

        assert output.ragged_dims == (1,)
        assert_close(output, NT([torch.argsort(element, dim=1) for element in elements], ragged_dims=(1,)))

    @pytest.mark.parametrize(
        "elements",
        [
            (torch.tensor([0, 0, 1]), torch.tensor([1, 0])),
            (torch.tensor([2.0, 0.0, 1.0]), torch.tensor([1.0, 0.0])),
        ],
        ids=["not-bijection", "floating-point"],
    )
    def test_invalid_permutation_raises(self, device, elements):
        from danling.tensors import inverse_permutation

        nested = NT([element.to(device) for element in elements], ragged_dims=(0,))
        with pytest.raises((RuntimeError, TypeError)):
            inverse_permutation(nested)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_inverse_permutation_compiles(self, device):
        from danling.tensors import inverse_permutation

        nested = NT([torch.randperm(n, device=device) for n in (3, 5, 2)], ragged_dims=(0,))
        compiled = torch.compile(
            lambda value: inverse_permutation(value).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        assert_close(compiled(nested), inverse_permutation(nested).concat)


class TestRank:

    @pytest.mark.parametrize("descending", [False, True])
    def test_rank_matches_double_argsort(self, device, float_dtype, descending):
        from danling.tensors import rank

        elements = [torch.randn(n, 3, device=device, dtype=float_dtype) for n in (3, 0, 5)]
        nested = NT(elements, ragged_dims=(0,))

        output = rank(nested, dim=1, descending=descending)
        expected = NT(
            [
                torch.argsort(
                    torch.argsort(element, dim=0, stable=True, descending=descending),
                    dim=0,
                )
                for element in elements
            ],
            ragged_dims=(0,),
        )

        assert_close(output, expected)

    def test_rank_rejects_multi_ragged_layout(self, device, float_dtype):
        from danling.tensors import rank

        nested = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(3, 2, device=device, dtype=float_dtype),
            ],
            ragged_dims=(0, 1),
        )
        with pytest.raises(ValueError, match="ragged dimension"):
            rank(nested)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize("requires_grad", [False, True])
    @pytest.mark.parametrize("composed", [False, True])
    def test_rank_compiles(self, device, float_dtype, requires_grad, composed):
        from danling.tensors import rank

        torch.compiler.reset()
        compiled = torch.compile(
            lambda value: rank(value * 2 if composed else value).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        for lengths in ((3, 5, 2), (4, 2, 3)):
            parts = [torch.randn(n, device=device, dtype=float_dtype, requires_grad=requires_grad) for n in lengths]
            nested = NT(parts, ragged_dims=(0,))
            expected = torch.cat([torch.argsort(torch.argsort(part, stable=True), stable=True) for part in parts])
            assert_close(compiled(nested), expected)


class TestCumcount:

    def test_counts_within_each_sample_and_static_tail(self, device, float_dtype):
        from danling.tensors import cumcount

        elements = [torch.randn(n, 3, device=device, dtype=float_dtype) for n in (3, 0, 5)]
        nested = NT(elements, ragged_dims=(0,))
        expected = [
            torch.arange(element.shape[0], device=device).view(-1, 1).expand_as(element) for element in elements
        ]

        assert_close(cumcount(nested), NT(expected, ragged_dims=(0,)))

    def test_nonleading_declared_ragged_dim(self, device, float_dtype):
        from danling.tensors import cumcount

        elements = [torch.randn(2, n, 3, device=device, dtype=float_dtype) for n in (3, 5)]
        nested = NT(elements, ragged_dims=(1,))
        expected = [
            torch.arange(element.shape[1], device=device).view(1, -1, 1).expand_as(element) for element in elements
        ]

        output = cumcount(nested, dim=2)

        assert output.ragged_dims == (1,)
        assert_close(output, NT(expected, ragged_dims=(1,)))


class TestSegmentedComposition:

    @pytest.mark.parametrize("tail", [(), (3,)])
    def test_rank_is_inverse_permutation_of_argsort(self, device, float_dtype, tail):
        from danling.tensors import inverse_permutation, rank

        nested = NT(
            [torch.randn(n, *tail, device=device, dtype=float_dtype) for n in (3, 5, 2)],
            ragged_dims=(0,),
        )

        assert_close(inverse_permutation(torch.argsort(nested, dim=1), dim=1), rank(nested, dim=1))


class TestUnaryBinaryMath:

    def test_unary_tensor_method_values_and_vjp(self):
        template = NestedTensor(
            [torch.empty(2, 2, 4), torch.empty(3, 3, 4)],
            ragged_dims=(0, 1),
        ).transpose(1, 2)
        packed_values = (torch.rand_like(template.concat) + 0.5).requires_grad_()
        expected_values = packed_values.detach().clone().requires_grad_()
        input_ = template.packed_like(packed_values)

        output = input_.sigmoid()
        expected = expected_values.sigmoid()
        grad_output = torch.randn_like(expected)
        actual_gradient = torch.autograd.grad(output.concat, packed_values, grad_output)[0]
        expected_gradient = torch.autograd.grad(expected, expected_values, grad_output)[0]

        assert_close(output.concat, expected)
        assert_close(actual_gradient, expected_gradient)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_sigmoid_head_split_merge_compiles_with_vjp(self):
        def run(template, values, weight):
            source = template.packed_like(values)
            projected = F.linear(source, weight).sigmoid().unflatten(-1, (2, 4))
            return projected.flatten(-2).concat.square().sum()

        compiled = torch.compile(run, backend="aot_eager", fullgraph=True)
        template = NestedTensor(
            [torch.empty(2, 2, 8), torch.empty(3, 3, 8)],
            ragged_dims=(0, 1),
        ).transpose(1, 2)
        packed_values = torch.randn_like(template.concat, requires_grad=True)
        expected_values = packed_values.detach().clone().requires_grad_()
        weight = torch.randn(8, 8, requires_grad=True)
        expected_weight = weight.detach().clone().requires_grad_()

        loss = compiled(template, packed_values, weight)
        expected_loss = (
            F.linear(expected_values, expected_weight).sigmoid().unflatten(-1, (2, 4)).flatten(-2).square().sum()
        )
        gradients = torch.autograd.grad(loss, (packed_values, weight))
        expected_gradients = torch.autograd.grad(expected_loss, (expected_values, expected_weight))

        assert_close(loss, expected_loss)
        for actual, expected in zip(gradients, expected_gradients, strict=True):
            assert_close(actual, expected)

    def test_addcdiv(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.addcdiv(nt, nt, nt + 1, value=2)
        reference = NT([torch.addcdiv(t, t, t + 1, value=2) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_addcdiv_tensor_input(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        reference = NT([torch.addcdiv(t, t, t + 1, value=2) for t in nt], **reference_options(nt))
        output = torch.addcdiv(nt.tensor, nt, nt + 1, value=2)
        assert_close(output, reference)

    def test_addcmul(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.addcmul(nt, nt, nt, value=2)
        reference = NT([torch.addcmul(t, t, t, value=2) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_addcmul_and_addcdiv_matches_dense(self, device):
        seed = 0
        dtype = torch.float32
        shapes = ragged_shapes(seed, batch_size=3, min_len=2, max_len=5, trailing_shape=(4,))
        nt = nested_rand(shapes, device, dtype)
        torch.manual_seed(seed)
        bias = torch.randn(4, device=device, dtype=dtype)
        scale = torch.randn(4, device=device, dtype=dtype)
        denom = torch.randn(4, device=device, dtype=dtype).abs() + 0.5

        addcmul_output = torch.addcmul(nt, bias, scale, value=0.25)
        addcmul_reference = NT([torch.addcmul(t, bias, scale, value=0.25) for t in nt], **reference_options(nt))
        assert_close(addcmul_output, addcmul_reference)

        addcdiv_output = torch.addcdiv(nt, scale, denom, value=-0.5)
        addcdiv_reference = NT([torch.addcdiv(t, scale, denom, value=-0.5) for t in nt], **reference_options(nt))
        assert_close(addcdiv_output, addcdiv_reference)

    def test_addcmul_tensor_input(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        reference = NT([torch.addcmul(t, t, t, value=2) for t in nt], **reference_options(nt))
        output = torch.addcmul(nt.tensor, nt, nt, value=2)
        assert_close(output, reference)

    def test_addcmul_tensor_input_requires_per_element_broadcast(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.ones(2, 3, device=device, dtype=float_dtype),
                torch.ones(4, 3, device=device, dtype=float_dtype),
            ]
        )
        bad = torch.arange(18, device=device, dtype=float_dtype).reshape(6, 3)

        with pytest.raises(RuntimeError, match="size of tensor"):
            torch.addcmul(nt, bad, bad, value=2)

    def test_clamp(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([[-1.0, 0.5], [2.0, 5.0]], device=device, dtype=float_dtype),
                torch.tensor([[10.0, -5.0], [1.0, 3.0]], device=device, dtype=float_dtype),
            ]
        )
        output = torch.clamp(nt, min=0.0, max=3.0)
        reference = torch.clamp(nt.tensor, min=0.0, max=3.0)
        assert_close(output, reference)

    def test_clamp_min_and_clamp_max(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, -2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([-4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.clamp_min(nt, 0.0)
        reference = NT([torch.clamp_min(t, 0.0) for t in nt], **reference_options(nt))
        assert_close(output, reference)

        output = torch.clamp_max(nt, 0.0)
        reference = NT([torch.clamp_max(t, 0.0) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_isfinite_isnan_nan_to_num(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([float("nan"), float("inf"), -float("inf")], device=device, dtype=float_dtype),
                torch.tensor([0.0, float("nan")], device=device, dtype=float_dtype),
            ]
        )
        output = torch.isnan(nt)
        reference = NT([torch.isnan(t) for t in nt], **reference_options(nt))
        assert_close(output, reference)

        output = torch.isfinite(nt)
        reference = NT([torch.isfinite(t) for t in nt], **reference_options(nt))
        assert_close(output, reference)

        output = torch.nan_to_num(nt, nan=0.0, posinf=1.0, neginf=-1.0)
        reference = NT([torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_maximum_and_minimum(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, -2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([-4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        scalar = torch.tensor(0.0, device=device, dtype=float_dtype)
        output = torch.maximum(nt, scalar)
        reference = NT([torch.maximum(t, scalar) for t in nt], **reference_options(nt))
        assert_close(output, reference)

        output = torch.minimum(nt, scalar)
        reference = NT([torch.minimum(t, scalar) for t in nt], **reference_options(nt))
        assert_close(output, reference)

        output = torch.maximum(nt, nt.tensor)
        assert_close(output, nt)

    def test_neg(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, -2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([-4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.neg(nt)
        reference = NT([torch.neg(t) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_hypot(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, device=device, dtype=float_dtype)
        nt_a = NT([a, a])
        nt_b = NT([b, b])
        result = torch.hypot(nt_a, nt_b)
        assert_close(result[0], torch.hypot(a, b))

    def test_logaddexp(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, device=device, dtype=float_dtype)
        nt_a = NT([a, a])
        nt_b = NT([b, b])
        result = torch.logaddexp(nt_a, nt_b)
        assert_close(result[0], torch.logaddexp(a, b))

    def test_logaddexp2(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, device=device, dtype=float_dtype)
        nt_a = NT([a, a])
        nt_b = NT([b, b])
        result = torch.logaddexp2(nt_a, nt_b)
        assert_close(result[0], torch.logaddexp2(a, b))

    def test_nextafter(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, device=device, dtype=float_dtype)
        nt_a = NT([a, a])
        nt_b = NT([b, b])
        result = torch.nextafter(nt_a, nt_b)
        assert_close(result[0], torch.nextafter(a, b))


class TestWhere:

    def test_where_broadcasts_scalar_condition(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 2, 3], device=device, dtype=float_dtype),
                torch.tensor([4, 5], device=device, dtype=float_dtype),
            ]
        )
        output = torch.where(nt > 2, nt, 0.0)
        reference = NT(
            [
                torch.tensor([0.0, 0.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([4.0, 5.0], device=device, dtype=float_dtype),
            ],
            **reference_options(nt),
        )
        assert_close(output, reference)

    def test_where_dense_condition(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 2, 3], device=device, dtype=float_dtype),
                torch.tensor([4, 5], device=device, dtype=float_dtype),
            ]
        )
        output = torch.where(nt.tensor > 2, nt, 0.0)
        reference = NT([torch.where(t > 2, t, 0.0) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_where_length_mismatch_raises(self):
        cond = NestedTensor([torch.tensor([True, False])])
        input_nt = NestedTensor([torch.tensor([1, 2]), torch.tensor([3])])
        with pytest.raises(ValueError, match="batch length mismatch"):
            _ = torch.where(cond, input_nt, 0)

    def test_where_matches_dense(self, device):
        seed = 0
        dtype = torch.float32
        shapes = ragged_shapes(seed, batch_size=3, min_len=2, max_len=5, trailing_shape=(4,))
        nt = nested_rand(shapes, device, dtype)
        torch.manual_seed(seed)
        condition = torch.randn(4, device=device) > 0
        other = torch.randn(4, device=device, dtype=dtype)
        output = torch.where(condition, nt, other)
        reference = NT([torch.where(condition, t, other) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_where_scalar_operands(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 2, 3], device=device, dtype=float_dtype),
                torch.tensor([4, 5], device=device, dtype=float_dtype),
            ]
        )
        scalar_self = torch.where(nt > 2, 1.0, nt)
        scalar_self_ref = NT([torch.where(t > 2, 1.0, t) for t in nt], **reference_options(nt))
        assert_close(scalar_self, scalar_self_ref)

        scalar_both = torch.where(nt > 2, 1.0, 0.0)
        scalar_both_ref = NT([torch.where(t > 2, 1.0, 0.0) for t in nt], **reference_options(nt))
        assert_close(scalar_both, scalar_both_ref)


def _public_ternary_result(op_name, reference, condition_values, packed_values, first, second):
    input_ = reference.packed_like(packed_values)
    if op_name == "where":
        condition = reference.packed_like(condition_values)
        return torch.where(condition, input_, first)
    if op_name == "addcmul":
        return torch.addcmul(input_, first, second, value=0.25)
    if op_name == "addcdiv":
        return torch.addcdiv(input_, first, second, value=-0.5)
    if op_name == "lerp":
        return torch.lerp(input_, first, second)
    if op_name == "lerp_scalar":
        return torch.lerp(input_, first, 0.25)
    raise AssertionError(f"unknown ternary op {op_name}")


def _dense_ternary_result(op_name, condition_values, packed_values, first, second):
    if op_name == "where":
        return torch.where(condition_values, packed_values, first)
    if op_name == "addcmul":
        return torch.addcmul(packed_values, first, second, value=0.25)
    if op_name == "addcdiv":
        return torch.addcdiv(packed_values, first, second, value=-0.5)
    if op_name == "lerp":
        return torch.lerp(packed_values, first, second)
    if op_name == "lerp_scalar":
        return torch.lerp(packed_values, first, 0.25)
    raise AssertionError(f"unknown ternary op {op_name}")


class TestTernaryAutograd:

    def test_ternary_retains_both_operands_ragged_axes(self):
        left = NT([torch.ones(2, 1, 3), torch.ones(4, 1, 3)], ragged_dims=(0,))
        right = NT([torch.full((1, 3, 3), 2.0), torch.full((1, 5, 3), 2.0)], ragged_dims=(1,))
        output = torch.where(left > 0, left, right)
        for actual, first, second in zip(output, left, right):
            assert_close(actual, torch.where(first > 0, first, second))

    @pytest.mark.parametrize("op_name", ["where", "addcmul", "addcdiv", "lerp", "lerp_scalar"])
    def test_public_ternary_preserves_autograd(self, op_name):
        reference = NT([torch.randn(2, 1), torch.randn(3, 1)])
        condition_values = torch.tensor([[True], [False], [True], [False], [True]])
        packed_values = torch.randn_like(reference.concat, requires_grad=True)
        expected_values = packed_values.detach().clone().requires_grad_()
        first = torch.randn(1, 4)
        second = torch.randn(1, 4)
        if op_name == "addcdiv":
            second = second.abs() + 0.5
        elif op_name == "lerp":
            second = second.sigmoid()

        output = _public_ternary_result(op_name, reference, condition_values, packed_values, first, second)
        loss = output.concat.square().sum()
        expected = _dense_ternary_result(op_name, condition_values, expected_values, first, second)
        expected_loss = expected.square().sum()
        (expected_grad,) = torch.autograd.grad(expected_loss, expected_values)
        loss.backward()

        assert [tuple(element.shape) for element in output] == [(2, 4), (3, 4)]
        assert_close(output.concat, expected)
        assert_close(loss, expected_loss)
        assert_close(packed_values.grad, expected_grad)

    def test_ternary_tensor_method_values_and_vjp(self):
        reference = NT([torch.randn(2, 1), torch.randn(3, 1)])
        packed_values = torch.randn_like(reference.concat, requires_grad=True)
        expected_values = packed_values.detach().clone().requires_grad_()
        input_ = reference.packed_like(packed_values)
        first = torch.randn(1, 4)
        weight = torch.rand(1, 4)

        output = input_.lerp(first, weight)
        expected = expected_values.lerp(first, weight)
        cotangent = torch.randn_like(expected)
        actual_gradient = torch.autograd.grad(output.concat, packed_values, cotangent)[0]
        expected_gradient = torch.autograd.grad(expected, expected_values, cotangent)[0]

        assert_close(output.concat, expected)
        assert_close(actual_gradient, expected_gradient)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_public_ternary_compiles_with_vjp(self):
        reference = NT([torch.randn(2, 1), torch.randn(3, 1)])
        condition_values = torch.tensor([[True], [False], [True], [False], [True]])
        packed_values = torch.randn_like(reference.concat, requires_grad=True)
        expected_values = packed_values.detach().clone().requires_grad_()
        first = torch.randn(1, 4)
        second = torch.randn(1, 4)

        compiled = torch.compile(
            lambda ref, cond, values, lhs, rhs: _public_ternary_result("where", ref, cond, values, lhs, rhs)
            .concat.square()
            .sum(),
            backend="aot_eager",
            fullgraph=True,
        )
        loss = compiled(reference, condition_values, packed_values, first, second)
        expected = _dense_ternary_result("where", condition_values, expected_values, first, second)
        expected_loss = expected.square().sum()
        (expected_grad,) = torch.autograd.grad(expected_loss, expected_values)
        loss.backward()

        assert_close(loss, expected_loss)
        assert_close(packed_values.grad, expected_grad)

    def test_public_ternary_supports_fake_tensor(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        FakeTensorMode = fake_tensor_mod.FakeTensorMode
        is_fake = fake_tensor_mod.is_fake

        with FakeTensorMode():
            reference = NT([torch.empty(2, 1), torch.empty(3, 1)])
            condition_values = torch.empty(5, 1, dtype=torch.bool)
            packed_values = torch.empty_like(reference.concat)
            first = torch.empty(1, 4)
            second = torch.ones(1, 4)
            output = _public_ternary_result("where", reference, condition_values, packed_values, first, second)

            assert is_fake(output.concat)
            assert output.concat.shape == torch.Size((5, 4))
            assert [tuple(element.shape) for element in output] == [(2, 4), (3, 4)]


class TestPackedStaticTailEinsum:

    @pytest.mark.parametrize(
        "samples,ragged,batch_first,parameter_prefix",
        [
            (2, (1,), True, "batch"),
            (3, (1,), True, "batch"),
            (2, (0, 1), True, "batch"),
            (3, (1,), False, "batch"),
            (3, (1,), True, "shared"),
            (3, (1,), False, "shared"),
            (3, (1,), True, "global"),
        ],
    )
    @pytest.mark.parametrize("equation", ["...ij,...nj->...ni", "...np,...qp->...nq", "...j,...nj->...n"])
    def test_named_ragged(self, device, samples, ragged, batch_first, parameter_prefix, equation):
        parts = [
            torch.randn(samples, length, 3, device=device, dtype=torch.float64, requires_grad=True) for length in (4, 6)
        ]
        source = NT(parts, ragged_dims=ragged, batch_first=batch_first)
        tail = (3,) if equation == "...j,...nj->...n" else ((5, 3) if equation == "...np,...qp->...nq" else (3, 3))
        prefix = (2, samples) if parameter_prefix == "batch" else ((samples,) if parameter_prefix == "shared" else ())
        weight = torch.randn(*prefix, *tail, device=device, dtype=torch.float64, requires_grad=True)
        if not batch_first and parameter_prefix == "batch":
            weight = weight.movedim(0, 1)
        reverse = equation == "...np,...qp->...nq"
        operands = (source, weight) if reverse else (weight, source)
        expected = []
        for index, part in enumerate(parts):
            local = weight.select(0 if batch_first else 1, index) if parameter_prefix == "batch" else weight
            expected.append(torch.einsum(equation, *((part, local) if reverse else (local, part))))
        output = torch.einsum(equation, *operands)
        torch.testing.assert_close(output.unbind(0 if batch_first else 1), expected)
        assert output.ragged_dims == ragged
        probes = [torch.randn_like(value) for value in expected]
        gradients = torch.autograd.grad(
            sum((value * probe).sum() for value, probe in zip(output.unbind(0 if batch_first else 1), probes)),
            (*parts, weight),
        )
        references = torch.autograd.grad(
            sum((value * probe).sum() for value, probe in zip(expected, probes)), (*parts, weight)
        )
        torch.testing.assert_close(gradients, references)

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_dense_shared_prefix_expansion_and_vjps(self, device, batch_first):
        parts = [torch.randn(n, 3, device=device, dtype=torch.float64, requires_grad=True) for n in (1, 4)]
        source = NestedTensor(parts, ragged_dims=(0,), batch_first=batch_first)
        weight = torch.randn(4, 3, 2, device=device, dtype=torch.float64, requires_grad=True)
        references = [torch.einsum("...c,...cd->...d", part, weight) for part in parts]
        expected = torch.cat(references)
        cotangent = torch.linspace(-0.8, 1.2, expected.numel(), device=device, dtype=torch.float64).reshape_as(expected)
        actual = torch.einsum("...c,...cd->...d", source, weight)
        gradients = torch.autograd.grad(actual.concat, (*parts, weight), cotangent, retain_graph=True)
        torch.testing.assert_close(actual.element_sizes().cpu(), torch.tensor([[4, 2], [4, 2]]))
        torch.testing.assert_close(actual.concat, expected)
        wanted = torch.autograd.grad(expected, (*parts, weight), cotangent)
        for gradient, reference in zip(gradients, wanted):
            torch.testing.assert_close(gradient, reference)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("layout", ["sequence", "samples", "pair", "static"])
    @pytest.mark.parametrize(
        "equation",
        [
            "...c,...cd->...d",
            "...ab,...bc->...ac",
            "...ab->...ba",
            "...aa->...",
            "...c,...cd,de->...e",
            "...c,cd->...d",
        ],
    )
    def test_static_tail_values_all_operand_vjps(self, device, dtype, batch_first, layout, equation):
        shapes, ragged = {
            "sequence": ([(0,), (2,), (5,)], (0,)),
            "samples": ([(0, 2), (2, 2), (5, 2)], (0,)),
            "pair": ([(0, 3), (2, 4), (5, 2)], (0, 1)),
            "static": ([(2,), (2,), (2,)], ()),
        }[layout]
        terms, output_term = equation.split("->")
        terms = terms.split(",")
        dimensions = {"a": 2, "b": 3, "c": 3, "d": 4, "e": 2}
        operands, element_operands, leaves = [], [], []
        for term in terms:
            labels = term[3:] if term.startswith("...") else term
            tail = tuple(dimensions[label] for label in labels)
            if term.startswith("..."):
                parts = [torch.randn(*shape, *tail, device=device, dtype=dtype, requires_grad=True) for shape in shapes]
                operands.append(
                    NestedTensor(parts, ragged_dims=ragged, batch_first=batch_first, padding_value=-7, mask_value=False)
                )
                element_operands.append(parts)
                leaves.extend(parts)
            else:
                values = torch.randn(*tail, device=device, dtype=dtype, requires_grad=True)
                operands.append(values)
                element_operands.append([values] * len(shapes))
                leaves.append(values)
        references = [
            torch.einsum(equation, *(parts[index] for parts in element_operands)) for index in range(len(shapes))
        ]
        cotangents = [torch.randn_like(value) for value in references]
        cotangent = NestedTensor(cotangents, ragged_dims=ragged, batch_first=batch_first)
        actual = torch.einsum(equation, *operands)
        gradients = torch.autograd.grad((actual * cotangent).sum(), leaves)
        expected = NestedTensor(references, ragged_dims=ragged, batch_first=batch_first)
        assert actual.batch_first is batch_first
        assert actual.ragged_dims == ragged
        assert actual.packed_dim_order == tuple(range(int(actual.element_sizes().size(1))))
        assert actual.padding_value == -7 and actual.mask_value is False
        torch.testing.assert_close(actual.element_sizes(), expected.element_sizes())
        torch.testing.assert_close(actual.concat, expected.concat)
        reference_gradients = torch.autograd.grad(
            sum((value * cot).sum() for value, cot in zip(references, cotangents)), leaves
        )
        for gradient, reference in zip(gradients, reference_gradients):
            torch.testing.assert_close(
                gradient,
                reference,
                atol=1e-5 if dtype == torch.float32 else 1e-12,
                rtol=1e-5 if dtype == torch.float32 else 1e-10,
            )

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_shared_dense_ellipsis_preserves_static_prefix(self, device, batch_first):
        shapes = [(0, 2, 3), (4, 2, 3)]
        parts = [torch.randn(*shape, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
        source = NestedTensor(parts, ragged_dims=(0,), batch_first=batch_first)
        weight = torch.randn(1, 1, 2, 3, 5, device=device, dtype=torch.float64, requires_grad=True)
        actual = torch.einsum("...c,...cd->...d", source, weight)
        gradients = torch.autograd.grad(actual.sum(), [*parts, weight])
        references = [torch.einsum("nsc,scd->nsd", part, weight[0, 0]) for part in parts]
        expected = NestedTensor(references, ragged_dims=(0,), batch_first=batch_first)
        torch.testing.assert_close(actual.element_sizes(), expected.element_sizes())
        torch.testing.assert_close(actual.concat, expected.concat)
        for gradient, reference in zip(
            gradients, torch.autograd.grad(sum(part.sum() for part in references), [*parts, weight])
        ):
            torch.testing.assert_close(gradient, reference)

    @pytest.mark.parametrize("layout", ["sequence", "pair"])
    def test_einsum_fullgraph_dynamic_values_and_vjps(self, device, layout):
        equation = "...c,...cd->...d"
        ragged = (0,) if layout == "sequence" else (0, 1)
        compiled = torch.compile(
            lambda a, av, b, bv: torch.einsum(equation, a.packed_like(av), b.packed_like(bv)).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        for lengths in ((2, 4), (0, 5)):
            prefix = [(n,) if layout == "sequence" else (n, 2) for n in lengths]
            a = NestedTensor([torch.empty(*shape, 3, device=device) for shape in prefix], ragged_dims=ragged)
            b = NestedTensor([torch.empty(*shape, 3, 4, device=device) for shape in prefix], ragged_dims=ragged)
            av, bv = torch.randn_like(a.concat, requires_grad=True), torch.randn_like(b.concat, requires_grad=True)
            output = compiled(a, av, b, bv)
            gradients = torch.autograd.grad(output.sum(), (av, bv))
            reference = torch.einsum("tc,tcd->td", av, bv)
            torch.testing.assert_close(output, reference)
            for actual, expected in zip(gradients, torch.autograd.grad(reference.sum(), (av, bv))):
                torch.testing.assert_close(actual, expected)

    def test_ragged_reduction(self, device):
        parts = [torch.randn(length, 3, device=device, requires_grad=True) for length in (2, 4)]
        actual = torch.einsum("...nc->...c", NestedTensor(parts, ragged_dims=(0,)))
        expected = [part.sum(0) for part in parts]
        torch.testing.assert_close(actual.unbind(), expected)
        probes = [torch.randn_like(part) for part in expected]
        gradients = torch.autograd.grad(sum((part * probe).sum() for part, probe in zip(actual, probes)), parts)
        torch.testing.assert_close(gradients, [probe.expand_as(part) for part, probe in zip(parts, probes)])

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("layout", ["sequence", "samples", "pair", "static"])
    @pytest.mark.parametrize("dense_coordinates", ["batch", "ragged", "batch_ragged"])
    def test_dense_ellipsis_reads_logical_prefix_and_parameter_vjp(
        self, device, batch_first, layout, dense_coordinates
    ):
        shapes, ragged = {
            "sequence": ([(0,), (2,), (5,)], (0,)),
            "samples": ([(0, 2), (2, 2), (5, 2)], (0,)),
            "pair": ([(0, 3), (2, 4), (5, 2)], (0, 1)),
            "static": ([(2,), (2,), (2,)], ()),
        }[layout]
        parts = [torch.randn(*shape, 3, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
        source = NestedTensor(parts, ragged_dims=ragged, batch_first=batch_first)
        prefix_rank = len(shapes[0]) + 1
        batch_axis = 0 if batch_first else 1
        prefix = list(source.shape[:prefix_rank])
        for index in range(prefix_rank):
            if index == batch_axis:
                if dense_coordinates == "ragged":
                    prefix[index] = 1
            else:
                element_axis = index if index < batch_axis else index - 1
                if element_axis in ragged and dense_coordinates == "batch":
                    prefix[index] = 1
        weight = torch.randn(*prefix, 3, 4, device=device, dtype=torch.float64, requires_grad=True)
        references = []
        for index, (part, shape) in enumerate(zip(parts, shapes)):
            local_weight = weight.select(batch_axis, index if prefix[batch_axis] != 1 else 0)
            selectors = tuple(
                slice(0, shape[axis]) if axis in ragged and local_weight.shape[axis] != 1 else slice(None)
                for axis in range(len(shape))
            )
            references.append(torch.einsum("...c,...cd->...d", part, local_weight[selectors]))
        cotangents = [torch.randn_like(value) for value in references]
        cotangent = NestedTensor(cotangents, ragged_dims=ragged, batch_first=batch_first)
        actual = torch.einsum("...c,...cd->...d", source, weight)
        gradients = torch.autograd.grad((actual * cotangent).sum(), [*parts, weight])
        expected = NestedTensor(references, ragged_dims=ragged, batch_first=batch_first)
        torch.testing.assert_close(actual.element_sizes(), expected.element_sizes())
        torch.testing.assert_close(actual.concat, expected.concat)
        reference_loss = sum((value * cot).sum() for value, cot in zip(references, cotangents))
        for actual_gradient, expected_gradient in zip(gradients, torch.autograd.grad(reference_loss, [*parts, weight])):
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-12, rtol=1e-10)

    def test_dense_ellipsis_invalid_batch_size_raises_ordinary_broadcast_error(self, device):
        source = NestedTensor([torch.randn(2, 3, device=device), torch.randn(4, 3, device=device)], ragged_dims=(0,))
        weights = torch.randn(3, 1, 3, 5, device=device)
        with pytest.raises(RuntimeError, match="broadcast|Shape mismatch"):
            torch.einsum("...c,...cd->...d", source, weights)

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_dense_batch_ellipsis_fullgraph_values_and_parameter_vjp(self, device, batch_first):
        compiled = torch.compile(
            lambda a, av, weight: torch.einsum("...c,...cd->...d", a.packed_like(av), weight).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        for lengths in ((2, 4), (0, 5)):
            source = NestedTensor(
                [torch.empty(n, 3, device=device) for n in lengths], ragged_dims=(0,), batch_first=batch_first
            )
            values = torch.randn_like(source.concat, requires_grad=True)
            weight_shape = (2, 1, 3, 4) if batch_first else (1, 2, 3, 4)
            weights = torch.randn(*weight_shape, device=device, requires_grad=True)
            actual = compiled(source, values, weights)
            actual_gradients = torch.autograd.grad(actual.sum(), (values, weights))
            parts = values.split(lengths)
            references = [
                torch.einsum("xc,cd->xd", part, weights[index, 0] if batch_first else weights[0, index])
                for index, part in enumerate(parts)
            ]
            reference = torch.cat(references)
            torch.testing.assert_close(actual, reference)
            for gradient, expected in zip(actual_gradients, torch.autograd.grad(reference.sum(), (values, weights))):
                torch.testing.assert_close(gradient, expected)

    def test_zero_static_contraction_and_empty_batch(self, device):
        parts = [torch.empty(n, 0, device=device, requires_grad=True) for n in (0, 4)]
        source = NestedTensor(parts, ragged_dims=(0,))
        weight = torch.empty(0, 3, device=device, requires_grad=True)
        output = torch.einsum("...c,cd->...d", source, weight)
        gradients = torch.autograd.grad(output.sum(), [*parts, weight])
        torch.testing.assert_close(output.concat, torch.zeros(4, 3, device=device))
        for gradient in gradients:
            assert gradient.numel() == 0
        empty = NestedTensor.from_packed(
            torch.empty(0, 3, device=device),
            torch.tensor([0]),
            torch.empty(0, 2, dtype=torch.long),
            permutation=(0, 1),
            ragged_dims=(0,),
            outer_size=torch.Size((0, 0, 3)),
            packed_sizes=(),
            element_shapes=(),
            validate=False,
        )
        output = torch.einsum("...c,cd->...d", empty, torch.empty(3, 4, device=device))
        assert output.shape == (0, 0, 4) and output.concat.shape == (0, 4)

    def test_fake_static_tail_preserves_metadata_without_data_reads(
        self,
    ):
        from torch._subclasses.fake_tensor import FakeTensorMode

        source = NestedTensor([torch.empty(2, 3), torch.empty(4, 3)], ragged_dims=(0,))
        with FakeTensorMode() as mode:
            fake = mode.from_tensor(source)
            weights = torch.empty(3, 5, device="cpu")
            output = torch.einsum("...c,cd->...d", fake, weights)
        assert output.shape == (2, 4, 5) and output.concat.shape == (6, 5)
        assert output.ragged_dims == (0,)

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("layout", ["sample_first", "permuted_static", "permuted_ragged"])
    @pytest.mark.parametrize("dense_weight", [False, True])
    def test_mapped_static_axes_keep_ragged_row_order_and_parameter_vjps(
        self, device, batch_first, layout, dense_weight
    ):
        shapes, ragged = {
            "sample_first": ([(2, 0), (2, 4)], (1,)),
            "permuted_static": ([(0, 2), (4, 2)], (0,)),
            "permuted_ragged": ([(2, 0), (2, 4)], (1, 0)),
        }[layout]
        parts = [torch.randn(*shape, 3, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
        if layout == "permuted_static":
            source = NestedTensor(
                [part.transpose(-1, -2) for part in parts], ragged_dims=ragged, batch_first=batch_first
            )
            batch_axis = 0 if batch_first else 1
            physical_order = (0, 2, 1)
            logical = [dim if dim < batch_axis else dim + 1 for dim in physical_order]
            logical.insert(batch_axis, batch_axis)
            source = source.permute(*logical)
            assert source.packed_dim_order != (0, 1, 2)
        else:
            source = NestedTensor(parts, ragged_dims=ragged, batch_first=batch_first)
        if dense_weight:
            batch_axis = 0 if batch_first else 1
            prefix = [1] * 3
            prefix[batch_axis] = len(parts)
            weights = torch.randn(*prefix, 3, 4, device=device, dtype=torch.float64, requires_grad=True)
            weight_parts = [weights.select(batch_axis, index).reshape(3, 4) for index in range(len(parts))]
            leaves = [*parts, weights]
        else:
            weight_parts = [
                torch.randn(*shape, 3, 4, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes
            ]
            weights = NestedTensor(weight_parts, ragged_dims=ragged, batch_first=batch_first)
            leaves = [*parts, *weight_parts]
        references = [torch.einsum("...c,...cd->...d", part, weight) for part, weight in zip(parts, weight_parts)]
        cotangents = [torch.randn_like(value) for value in references]
        cotangent = NestedTensor(cotangents, ragged_dims=ragged, batch_first=batch_first)
        output = torch.einsum("...c,...cd->...d", source, weights)
        gradients = torch.autograd.grad((output * cotangent).sum(), leaves)
        expected = NestedTensor(references, ragged_dims=ragged, batch_first=batch_first)
        assert output.ragged_dims == ragged and output.batch_first is batch_first
        torch.testing.assert_close(output.element_sizes(), expected.element_sizes())
        torch.testing.assert_close(output.concat, expected.concat)
        reference = sum((value * cot).sum() for value, cot in zip(references, cotangents))
        for gradient, expected_gradient in zip(gradients, torch.autograd.grad(reference, leaves)):
            torch.testing.assert_close(gradient, expected_gradient, atol=1e-12, rtol=1e-10)

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_nonleading_ragged_fullgraph_values_and_parameter_vjps(self, device, batch_first):
        compiled = torch.compile(
            lambda a, av, b, bv: torch.einsum("...c,...cd->...d", a.packed_like(av), b.packed_like(bv)).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        for lengths in ((2, 4), (0, 5)):
            a = NestedTensor(
                [torch.empty(2, n, 3, device=device) for n in lengths], ragged_dims=(1,), batch_first=batch_first
            )
            b = NestedTensor(
                [torch.empty(2, n, 3, 4, device=device) for n in lengths], ragged_dims=(1,), batch_first=batch_first
            )
            av, bv = torch.randn_like(a.concat, requires_grad=True), torch.randn_like(b.concat, requires_grad=True)
            output = compiled(a, av, b, bv)
            gradients = torch.autograd.grad(output.sum(), (av, bv))
            reference = torch.einsum("tsc,tscd->tsd", av, bv)
            torch.testing.assert_close(output, reference)
            for actual, expected in zip(gradients, torch.autograd.grad(reference.sum(), (av, bv))):
                torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_permuted_static_fullgraph_values_and_dense_parameter_vjps(self, device, batch_first):
        compiled = torch.compile(
            lambda a, av, w: torch.einsum("...c,...cd->...d", a.packed_like(av), w).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        for lengths in ((2, 4), (0, 5)):
            source = NestedTensor(
                [torch.empty(n, 3, 2, device=device) for n in lengths], ragged_dims=(0,), batch_first=batch_first
            )
            source = source.permute(0, 1, 3, 2)
            assert source.packed_dim_order == (0, 2, 1)
            values = torch.randn_like(source.concat, requires_grad=True)
            batch_axis = 0 if batch_first else 1
            prefix = [1, 1, 1]
            prefix[batch_axis] = 2
            weights = torch.randn(*prefix, 3, 4, device=device, requires_grad=True)
            output = compiled(source, values, weights)
            gradients = torch.autograd.grad(output.sum(), (values, weights))
            parts = values.split(lengths)
            references = [
                torch.einsum("nsc,cd->nsd", part.transpose(1, 2), weights.select(batch_axis, index).reshape(3, 4))
                for index, part in enumerate(parts)
            ]
            reference = torch.cat(references)
            torch.testing.assert_close(output, reference)
            for gradient, expected in zip(gradients, torch.autograd.grad(reference.sum(), (values, weights))):
                torch.testing.assert_close(gradient, expected)


class TestPackedMixedBroadcast:

    @pytest.mark.parametrize("tensor_backed", [False, True])
    @pytest.mark.parametrize("lengths", [(3, 5), (0, 5), (3, 3)])
    def test_sequence_tokens_and_batch_row_mask(self, device, tensor_backed, lengths):
        parts = [torch.randn(length, 7, device=device, dtype=torch.float64, requires_grad=True) for length in lengths]
        token_parts = [torch.arange(length, device=device) % 7 for length in lengths]
        input = NestedTensor(parts, ragged_dims=(0,))
        tokens = NestedTensor(token_parts, ragged_dims=(0,)).unsqueeze(-1)
        if tensor_backed:
            input = input.packed_with_lengths(input.concat, torch.tensor(lengths))
            tokens = tokens.packed_with_lengths(tokens.concat, torch.tensor(lengths))
        row_mask = torch.tensor([True, False], device=device).reshape(2, 1, 1)

        actual = torch.broadcast_tensors(input, tokens, row_mask)
        selected = -actual[0].gather(-1, actual[1][..., :1]).squeeze(-1)
        loss = selected.masked_fill(~actual[2][..., 0], 0).sum()
        gradients = torch.autograd.grad(loss, parts, retain_graph=True)

        references = [
            torch.broadcast_tensors(part, token[:, None], row_mask[index])
            for index, (part, token) in enumerate(zip(parts, token_parts))
        ]
        for operand_index, output in enumerate(actual):
            assert output.ragged_dims == (0,)
            assert output.element_sizes().tolist() == [[length, 7] for length in lengths]
            for index, reference in enumerate(references):
                torch.testing.assert_close(output[index], reference[operand_index])
        reference_loss = sum(
            -part.gather(-1, token[:, None]).sum() * row_mask[index, 0, 0]
            for index, (part, token) in enumerate(zip(parts, token_parts))
        )
        for gradient, reference in zip(gradients, torch.autograd.grad(reference_loss, parts)):
            torch.testing.assert_close(gradient, reference)

    @pytest.mark.parametrize("tensor_backed", [False, True])
    @pytest.mark.parametrize("batch_first", [False, True])
    def test_sample_axis_static_tail_and_dense_gradients(self, device, tensor_backed, batch_first):
        lengths = (3, 5)
        parts = [
            torch.randn(3, length, 1, device=device, dtype=torch.float64, requires_grad=True) for length in lengths
        ]
        input = NestedTensor(parts, ragged_dims=(1,), batch_first=batch_first, padding_value=-7, mask_value=False)
        row_bias = torch.randn(2, 1, 1, 1, device=device, dtype=torch.float64, requires_grad=True)
        head_bias = torch.randn(1, 3, 1, 7, device=device, dtype=torch.float64, requires_grad=True)
        scalar = torch.randn((), device=device, dtype=torch.float64, requires_grad=True)
        if not batch_first:
            row_bias = row_bias.transpose(0, 1)
            head_bias = head_bias.transpose(0, 1)
        if tensor_backed:
            # Keep the non-leading declared ragged layout but omit cached Python element shapes.
            input = NestedTensor.from_packed(
                input.concat,
                input._offsets,
                input._physical_shape,
                permutation=input._permutation,
                ragged_dims=input.ragged_dims,
                batch_first=input.batch_first,
                padding_value=input.padding_value,
                mask_value=input.mask_value,
                outer_size=input.shape,
                validate=False,
                materialize_python_metadata=False,
            )

        output = torch.broadcast_tensors(scalar, row_bias, input, head_bias)
        loss = sum((index + 1) * values.square().sum() for index, values in enumerate(output))
        gradients = torch.autograd.grad(loss, (*parts, scalar, row_bias, head_bias), retain_graph=True)

        references = [
            torch.broadcast_tensors(
                scalar,
                row_bias[index] if batch_first else row_bias[:, index],
                part,
                head_bias[0] if batch_first else head_bias[:, 0],
            )
            for index, part in enumerate(parts)
        ]
        for operand_index, values in enumerate(output):
            assert values.ragged_dims == (1,)
            assert values.padding_value == -7 and values.mask_value is False
            assert values.element_sizes().tolist() == [[3, length, 7] for length in lengths]
            for index, reference in enumerate(references):
                torch.testing.assert_close(values[index], reference[operand_index])
        reference_loss = sum(
            sum((index + 1) * value.square().sum() for index, value in enumerate(reference)) for reference in references
        )
        for gradient, reference in zip(
            gradients, torch.autograd.grad(reference_loss, (*parts, scalar, row_bias, head_bias))
        ):
            torch.testing.assert_close(gradient, reference)

    def test_nested_operands_keep_their_own_configuration(self, device):
        parts = [torch.randn(length, 4, device=device) for length in (3, 5)]
        first = NestedTensor(parts, ragged_dims=(0,), padding_value=-3, mask_value=False)
        second = NestedTensor([part[:, :1] for part in parts], ragged_dims=(0,), padding_value=5, mask_value=True)
        actual = torch.broadcast_tensors(first, second, torch.tensor(2.0, device=device))
        assert actual[0].padding_value == -3 and actual[0].mask_value is False
        assert actual[1].padding_value == 5 and actual[1].mask_value is True
        assert actual[2].padding_value == -3 and actual[2].mask_value is False
        for input, output in zip((first, second), actual[:2]):
            for index in range(2):
                torch.testing.assert_close(output[index], input[index].expand_as(first[index]))

    def test_ambiguous_dense_operand_retains_normal_error(self, device):
        input = NestedTensor([torch.randn(2, length, 5, device=device) for length in (3, 5)], ragged_dims=(1,))
        ambiguous = torch.randn(2, 1, 5, device=device)
        with pytest.raises(NotImplementedError, match="ambiguous"):
            torch.broadcast_tensors(input, ambiguous)

    def test_all_nested_outer_broadcast_still_builds_real_grids(self, device):
        parts = [torch.randn(length, 2, device=device, dtype=torch.float64, requires_grad=True) for length in (3, 5)]
        input = NestedTensor(parts, ragged_dims=(0,))
        left, right = torch.broadcast_tensors(input.unsqueeze(1), input.unsqueeze(2))
        gradients = torch.autograd.grad((left * right).sum(), parts, retain_graph=True)
        assert left.ragged_dims == (0, 1) and right.ragged_dims == (0, 1)
        assert left.element_sizes().tolist() == [[3, 3, 2], [5, 5, 2]]
        references = [torch.broadcast_tensors(part[None], part[:, None]) for part in parts]
        reference_loss = sum((lhs * rhs).sum() for lhs, rhs in references)
        for actual, reference in zip(gradients, torch.autograd.grad(reference_loss, parts)):
            torch.testing.assert_close(actual, reference)
        for index, (lhs, rhs) in enumerate(references):
            torch.testing.assert_close(left[index], lhs)
            torch.testing.assert_close(right[index], rhs)

    def test_tensor_backed_mixed_broadcast_fullgraph(self, device):
        parts = [torch.randn(length, 7, device=device) for length in (3, 5)]
        input = NestedTensor(parts, ragged_dims=(0,))
        input = input.packed_with_lengths(input.concat, torch.tensor([3, 5]))
        tokens = NestedTensor([torch.arange(length, device=device) % 7 for length in (3, 5)], ragged_dims=(0,))
        tokens = tokens.unsqueeze(-1).packed_with_lengths(tokens.concat[:, None], torch.tensor([3, 5]))
        mask = torch.tensor([True, False], device=device).reshape(2, 1, 1)

        def forward(input, tokens, mask):
            return tuple(value.concat for value in torch.broadcast_tensors(input, tokens, mask))

        compiled = torch.compile(forward, backend="eager", fullgraph=True)
        actual = compiled(input, tokens, mask)
        expected = forward(input, tokens, mask)
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output, reference)

    @pytest.mark.parametrize("tensor_backed", [False, True])
    def test_joint_static_shape_resolves_dense_reading_independently_of_operand_order(self, device, tensor_backed):
        lengths = (3, 4)
        first_parts = [
            torch.randn(1, length, 5, device=device, dtype=torch.float64, requires_grad=True) for length in lengths
        ]
        second_parts = [
            torch.randn(3, length, 5, device=device, dtype=torch.float64, requires_grad=True) for length in lengths
        ]
        first = NestedTensor(first_parts, ragged_dims=(1,))
        second = NestedTensor(second_parts, ragged_dims=(1,))
        dense = torch.randn(2, 1, 5, device=device, dtype=torch.float64, requires_grad=True)
        if tensor_backed:

            def tensor_layout(input):
                return NestedTensor.from_packed(
                    input.concat,
                    input._offsets,
                    input._physical_shape,
                    permutation=input._permutation,
                    ragged_dims=input.ragged_dims,
                    batch_first=input.batch_first,
                    outer_size=input.shape,
                    validate=False,
                    materialize_python_metadata=False,
                )

            first, second = tensor_layout(first), tensor_layout(second)

        inputs = (first, second, dense)
        leaves = (*first_parts, *second_parts, dense)
        references = [
            torch.broadcast_tensors(lhs, rhs, dense[index])
            for index, (lhs, rhs) in enumerate(zip(first_parts, second_parts))
        ]
        reference_loss = sum(
            sum((index + 1) * value.square().sum() for index, value in enumerate(values)) for values in references
        )
        reference_gradients = torch.autograd.grad(reference_loss, leaves, retain_graph=True)
        for order in itertools.permutations(range(3)):
            outputs = torch.broadcast_tensors(*(inputs[index] for index in order))
            canonical = tuple(outputs[order.index(index)] for index in range(3))
            loss = sum((index + 1) * values.square().sum() for index, values in enumerate(canonical))
            gradients = torch.autograd.grad(loss, leaves, retain_graph=True)
            for operand_index, output in enumerate(canonical):
                assert output.ragged_dims == (1,)
                assert output.element_sizes().tolist() == [[3, length, 5] for length in lengths]
                for element_index, values in enumerate(references):
                    torch.testing.assert_close(output[element_index], values[operand_index])
            for actual, expected in zip(gradients, reference_gradients):
                torch.testing.assert_close(actual, expected)

        if tensor_backed:

            def forward(first, second, dense):
                return tuple(output.concat for output in torch.broadcast_tensors(first, second, dense))

            compiled = torch.compile(forward, backend="eager", fullgraph=True)
            for lhs, rhs in ((first, second), (second, first)):
                outputs = compiled(lhs, rhs, dense)
                for actual, expected in zip(outputs, forward(lhs, rhs, dense)):
                    torch.testing.assert_close(actual, expected)


_NARROW_LAYOUTS = [
    pytest.param(((2, 2, 4), (3, 3, 4)), (0, 1), 0, id="square-outer"),
    pytest.param(((2, 2, 4), (3, 3, 4)), (0, 1), 1, id="square-inner"),
    pytest.param(((2, 4, 3), (3, 2, 3)), (0, 1), 0, id="rectangular-outer"),
    pytest.param(((2, 4, 3), (3, 2, 3)), (0, 1), 1, id="rectangular-inner"),
    pytest.param(((2, 2, 3, 4), (2, 3, 2, 4)), (1, 2), 1, id="nonleading-outer"),
    pytest.param(((2, 2, 3, 4), (2, 3, 2, 4)), (1, 2), 2, id="nonleading-inner"),
    pytest.param(((2, 3, 4), (3, 2, 4)), (1, 0), 0, id="reversed-inner"),
    pytest.param(((2, 3, 4), (3, 2, 4)), (1, 0), 1, id="reversed-outer"),
    pytest.param(((2, 3, 2, 4), (3, 2, 3, 4)), (0, 1, 2), 0, id="three-level-outer"),
    pytest.param(((2, 3, 2, 4), (3, 2, 3, 4)), (0, 1, 2), 1, id="three-level-middle"),
    pytest.param(((2, 3, 2, 4), (3, 2, 3, 4)), (0, 1, 2), 2, id="three-level-inner"),
]


def _narrow_elements(shapes, device):
    return tuple(
        (
            torch.arange(math.prod(shape), device=device, dtype=torch.float64).reshape(shape) + 1000 * index
        ).requires_grad_()
        for index, shape in enumerate(shapes)
    )


def _assert_narrow_structure(output, expected, ragged):
    shapes = [tuple(value.shape) for value in expected]
    assert output.element_sizes().tolist() == [list(shape) for shape in shapes]
    assert output.ragged_dims == ragged
    counts = [math.prod(shape[dim] for dim in ragged) for shape in shapes]
    assert output.packed_offsets().tolist() == [0, *accumulate(counts)]
    parents = [1] * len(shapes)
    for level, dim in enumerate(ragged):
        widths = [shape[dim] for shape, count in zip(shapes, parents) for _ in range(count)]
        assert output.ragged_level_offsets(level).tolist() == [0, *accumulate(widths)]
        parents = [count * shape[dim] for count, shape in zip(parents, shapes)]


def _check_narrow(shapes, ragged, element_dim, device, start, length, *, batch_first=True, method=False):
    inputs = _narrow_elements(shapes, device)
    references = tuple(value.detach().clone().requires_grad_() for value in inputs)
    source = NT(inputs, ragged_dims=ragged, batch_first=batch_first)
    dim = element_dim + 1 if batch_first or element_dim > 0 else element_dim
    output = source.narrow(dim, start, length) if method else torch.narrow(source, dim, start, length)
    expected = tuple(value.narrow(element_dim, start, length) for value in references)
    actual = output.unbind(0 if batch_first else 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _assert_narrow_structure(output, expected, ragged)
    assert output.packed_dim_order == source.packed_dim_order
    assert output.batch_first is batch_first
    weights = tuple(
        torch.arange(1, value.numel() + 1, device=device, dtype=value.dtype).reshape_as(value) for value in expected
    )
    loss = sum((value * weight).sum() for value, weight in zip(actual, weights))
    reference_loss = sum((value * weight).sum() for value, weight in zip(expected, weights))
    observed = torch.autograd.grad(loss, inputs)
    wanted = torch.autograd.grad(reference_loss, references)
    torch.testing.assert_close(observed, wanted, rtol=0, atol=0)


class _RepeatTensor(NestedTensor):
    pass


def _repeat_leaf(shapes, *, ragged_dims, batch_first=True):
    template = NestedTensor(
        [torch.empty(shape) for shape in shapes],
        ragged_dims=ragged_dims,
        batch_first=batch_first,
    )
    values = torch.randn_like(template.concat, requires_grad=True)
    return template.packed_like(values), values


def _repeat_elements(source: NestedTensor, repeats: int) -> list[torch.Tensor]:
    return [source[index] for index in range(len(source)) for _ in range(repeats)]


class TestBatchRepeat:

    def test_entries(self):
        source, values = _repeat_leaf(((2, 3), (0, 3), (5, 3)), ragged_dims=(0,))
        expected = NestedTensor(_repeat_elements(source, 3), ragged_dims=(0,))

        function_output = torch.repeat_interleave(source, 3, dim=0)
        method_output = source.repeat_interleave(3, dim=0)
        batch_output = source.repeat_batch(3)

        assert_close(function_output, expected)
        assert_close(method_output, expected)
        assert_close(batch_output, expected)
        assert function_output.ragged_dims == source.ragged_dims
        grad = torch.autograd.grad(function_output.concat.square().sum(), values)[0]
        assert_close(grad, 6 * values)

    def test_batch_dimension(self):
        source, values = _repeat_leaf(((2, 3), (4, 3)), ragged_dims=(0,), batch_first=False)
        expected = NestedTensor(_repeat_elements(source, 2), ragged_dims=(0,), batch_first=False)

        output = torch.repeat_interleave(source, 2, dim=1)
        negative_dim_output = torch.repeat_interleave(source, 2, dim=-2)

        assert_close(output, expected)
        assert_close(negative_dim_output, expected)
        assert output.shape == torch.Size((4, 4, 3))
        grad = torch.autograd.grad(output.concat.sum(), values)[0]
        assert_close(grad, torch.full_like(values, 2))

    def test_ragged(self):
        source, values = _repeat_leaf(((2, 3, 4), (1, 4, 4)), ragged_dims=(1, 0))
        expected = NestedTensor(_repeat_elements(source, 2), ragged_dims=(1, 0))

        output = torch.repeat_interleave(source, 2, dim=0)

        assert_close(output, expected)
        assert output.shape == expected.shape
        assert output.ragged_dims == (1, 0)
        grad = torch.autograd.grad(output.concat.square().sum(), values)[0]
        assert_close(grad, 4 * values)

    def test_counts(self):
        source = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)], ragged_dims=(0,))

        empty = torch.repeat_interleave(source, 0, dim=0)
        counted = torch.repeat_interleave(source, torch.tensor([0, 2]), dim=0)

        assert len(empty) == 0
        assert empty.concat.shape == (0, 3)
        assert_close(counted, NestedTensor([source[1], source[1]], ragged_dims=(0,)))

    def test_static(self):
        source = NestedTensor([torch.randn(2, 3), torch.randn(2, 3)])

        output = torch.repeat_interleave(source, 2, dim=0)

        assert_close(output, NestedTensor(_repeat_elements(source, 2)))

    def test_errors(self):
        source = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)], ragged_dims=(0,))

        with pytest.raises(RuntimeError, match="Repeats must be non-negative"):
            torch.repeat_interleave(source, -1, dim=0)
        with pytest.raises(NotImplementedError, match="dtype"):
            torch.repeat_interleave(source, torch.tensor([2.5, 1.5]), dim=0)
        with pytest.raises(RuntimeError, match="same size as input along dim 0"):
            torch.repeat_interleave(source, torch.tensor([1, 2, 3]), dim=0)
        with pytest.raises(RuntimeError, match="Invalid output_size"):
            torch.repeat_interleave(source, torch.tensor([2, 1]), dim=0, output_size=9)

    def test_fake(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        source = NestedTensor(
            [torch.empty(2, 3, 4), torch.empty(1, 4, 4)],
            ragged_dims=(1, 0),
        )

        with fake_tensor_mod.FakeTensorMode() as mode:
            output = torch.repeat_interleave(mode.from_tensor(source), 2, dim=0)

        assert fake_tensor_mod.is_fake(output.concat)
        assert output.shape == (4, 2, 4, 4)
        assert output.ragged_dims == (1, 0)

    def test_subclass(self):
        source = _RepeatTensor(
            [torch.randn(2, 3), torch.randn(4, 3)],
            ragged_dims=(0,),
        )
        eager_output = source.repeat_batch(2)

        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        with fake_tensor_mod.FakeTensorMode() as mode:
            fake_output = mode.from_tensor(source).repeat_batch(2)

        assert type(eager_output) is _RepeatTensor
        assert type(fake_output) is _RepeatTensor
        assert_close(eager_output, NestedTensor(_repeat_elements(source, 2), ragged_dims=(0,)))
        assert fake_tensor_mod.is_fake(fake_output.concat)
        assert fake_output.shape == eager_output.shape

    @pytest.mark.parametrize("entry", ["repeat-batch", "torch"])
    def test_compiled_subclass(self, entry):
        template = _RepeatTensor(
            [torch.empty(2, 3), torch.empty(4, 3)],
            ragged_dims=(0,),
        )
        values = torch.randn_like(template.concat)
        source = template.packed_like(values)
        expected = NestedTensor(_repeat_elements(source, 2), ragged_dims=(0,))

        def repeat(input_):
            if entry == "repeat-batch":
                return input_.repeat_batch(2)
            return torch.repeat_interleave(input_, 2, dim=0)

        output = torch.compile(repeat, backend="aot_eager", fullgraph=True, dynamic=True)(source)

        assert type(output) is _RepeatTensor
        assert_close(output, expected)

    @pytest.mark.parametrize("outside_loss", [False, True])
    def test_compiled_gradient(self, outside_loss):
        source, values = _repeat_leaf(((2, 3, 4), (1, 4, 4)), ragged_dims=(1, 0))

        def repeat(input_):
            output = input_.repeat_batch(2)
            return output if outside_loss else output.concat.square().sum()

        compiled = torch.compile(
            repeat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        output = compiled(source)
        loss = output.square().sum() if outside_loss else output
        loss.backward()

        assert_close(values.grad, 4 * values)


def _logical_dim(element_dim: int, batch_first: bool) -> int:
    if batch_first or element_dim > 0:
        return element_dim + 1
    return element_dim


def _gather_build_indices(shapes, element_dim, *, seed=0, index_shapes=None):
    generator = torch.Generator().manual_seed(seed)
    indices = []
    for shape, index_shape in zip(shapes, index_shapes or shapes):
        extent = shape[element_dim]
        if extent == 0:
            indices.append(torch.zeros(index_shape, dtype=torch.long))
        else:
            indices.append(torch.randint(0, extent, tuple(index_shape), generator=generator))
    return indices


def _gather_select_index(shapes, element_dim):
    extent = min(shape[element_dim] for shape in shapes)
    return torch.tensor(list(reversed(range(min(extent, 2)))), dtype=torch.long)


def _assert_elements(output, expected):
    assert len(output) == len(expected)
    for actual, reference in zip(output, expected):
        assert actual.shape == reference.shape
        torch.testing.assert_close(actual, reference)


def _gather_projected_case():
    sampled = NT(
        [torch.randn(2, 3, 4), torch.randn(2, 5, 4), torch.randn(2, 2, 4)],
        ragged_dims=(1,),
    )
    rows = sampled.concat.shape[0]
    tail = torch.arange(float(rows * 2 * 6)).reshape(rows, 2, 6)
    return sampled.packed_with_static_tail(tail)


def _gather_cross_topology_row_case(truth_lengths, selected_rows, *, prefix=2, requires_grad=True):
    elements = [torch.randn(prefix, length, 3, requires_grad=requires_grad) for length in truth_lengths]
    return elements, NT(elements, ragged_dims=(1,)), NT(selected_rows, ragged_dims=(0,))


_GATHER_CASES = [
    ("leading-ragged", [(3, 4), (5, 4), (2, 4)], 0, {}),
    ("trailing-ragged", [(4, 3), (4, 5), (4, 2)], 1, {"ragged_dims": (1,)}),
    ("multi-ragged-inner", [(3, 4, 2), (5, 2, 2), (2, 3, 2)], 1, {"ragged_dims": (0, 1)}),
    ("multi-ragged-outer", [(3, 4, 2), (5, 2, 2), (2, 3, 2)], 0, {"ragged_dims": (0, 1)}),
    ("static-tail", [(2, 3, 4), (2, 5, 4), (2, 2, 4)], 2, {"ragged_dims": (1,)}),
    ("batch-second", [(3, 4), (5, 4), (2, 4)], 0, {"batch_first": False}),
    ("empty-segment", [(1,), (0,), (4,)], 0, {"ragged_dims": (0,)}),
]


def _gather_cases_named(*names):
    return [case for case in _GATHER_CASES if case[0] in names]


def _gather_case_params(cases=None):
    cases = _GATHER_CASES if cases is None else cases
    return pytest.mark.parametrize(
        ("case_id", "shapes", "element_dim", "metadata"),
        cases,
        ids=[case[0] for case in cases],
    )


class TestGatherParity:

    @_gather_case_params(
        _gather_cases_named(
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
        elements = _element_values(shapes)
        indices = _gather_build_indices(shapes, element_dim)
        dim = _logical_dim(element_dim, metadata.get("batch_first", True))

        output = torch.gather(NT(elements, **metadata), dim, NT(indices, **metadata))

        _assert_elements(
            output,
            [torch.gather(element, element_dim, index) for element, index in zip(elements, indices)],
        )

    @_gather_case_params(_gather_cases_named("leading-ragged", "trailing-ragged", "batch-second"))
    def test_take_along_dim(self, case_id, shapes, element_dim, metadata):
        elements = _element_values(shapes)
        indices = _gather_build_indices(shapes, element_dim)
        dim = _logical_dim(element_dim, metadata.get("batch_first", True))

        output = torch.take_along_dim(NT(elements, **metadata), NT(indices, **metadata), dim=dim)

        _assert_elements(
            output,
            [torch.take_along_dim(element, index, dim=element_dim) for element, index in zip(elements, indices)],
        )

    @_gather_case_params(_gather_cases_named("leading-ragged", "multi-ragged-outer", "static-tail", "batch-second"))
    def test_index_select(self, case_id, shapes, element_dim, metadata):
        elements = _element_values(shapes)
        index = _gather_select_index(shapes, element_dim)
        dim = _logical_dim(element_dim, metadata.get("batch_first", True))

        output = torch.index_select(NT(elements, **metadata), dim, index)

        _assert_elements(
            output,
            [torch.index_select(element, element_dim, index) for element in elements],
        )

    def test_method_forms(self):
        shapes = [(3, 4), (5, 4), (2, 4)]
        elements = _element_values(shapes)
        indices = _gather_build_indices(shapes, 0)
        nested, index = NT(elements), NT(indices)

        _assert_elements(
            nested.gather(1, index),
            [torch.gather(element, 0, idx) for element, idx in zip(elements, indices)],
        )
        _assert_elements(
            torch.gather(nested, 1, index, sparse_grad=True),
            [torch.gather(element, 0, idx, sparse_grad=True) for element, idx in zip(elements, indices)],
        )
        _assert_elements(
            nested.take_along_dim(index, dim=1),
            [torch.take_along_dim(element, idx, dim=0) for element, idx in zip(elements, indices)],
        )
        selected = torch.tensor([1, 0])
        _assert_elements(
            nested.index_select(1, selected),
            [torch.index_select(element, 0, selected) for element in elements],
        )

    def test_projected_layout(self):
        projected = _gather_projected_case()
        elements = list(projected)
        shapes = [tuple(element.shape) for element in elements]
        indices = _gather_build_indices(shapes, 1)
        index = NT(indices, ragged_dims=(1,))
        selected = _gather_select_index(shapes, 1)

        _assert_elements(
            torch.gather(projected, 2, index),
            [torch.gather(element, 1, idx) for element, idx in zip(elements, indices)],
        )
        _assert_elements(
            torch.take_along_dim(projected, index, dim=2),
            [torch.take_along_dim(element, idx, dim=1) for element, idx in zip(elements, indices)],
        )
        _assert_elements(
            torch.index_select(projected, 2, selected),
            [torch.index_select(element, 1, selected) for element in elements],
        )

    def test_gather_index_may_have_a_different_output_shape(self):
        shapes = [(3, 4), (5, 4), (2, 4)]
        index_shapes = [(2, 2), (6, 2), (1, 2)]
        elements = _element_values(shapes)
        indices = _gather_build_indices(shapes, 0, index_shapes=index_shapes)

        output = torch.gather(NT(elements), 1, NT(indices))

        _assert_elements(
            output,
            [torch.gather(element, 0, index) for element, index in zip(elements, indices)],
        )


class TestNarrowIndex:

    def test_different_source_and_index_shapes(self):
        source_shapes = [(3, 4), (5, 4), (2, 4)]
        index_shapes = [(2, 2), (6, 2), (1, 2)]
        elements = _element_values(source_shapes)
        indices = _gather_build_indices(source_shapes, 0, index_shapes=index_shapes)
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
        elements = _element_values(source_shapes)
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
    def test_static_prefix(self, operation):
        elements, source, index = _gather_cross_topology_row_case(
            (4, 3),
            (torch.tensor([3, 1]), torch.tensor([2])),
        )

        output = self.select(operation, source, index)

        expected = self.dense_reference(operation, elements, list(index))
        assert output.ragged_dims == (1,)
        _assert_elements(output, expected)

    @pytest.mark.parametrize("operation", ["gather", "take_along_dim"])
    def test_fake(self, operation):
        fake_tensor = pytest.importorskip("torch._subclasses.fake_tensor")
        symbolic_shapes = pytest.importorskip("torch.fx.experimental.symbolic_shapes")
        _, source, index = _gather_cross_topology_row_case(
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

    @pytest.mark.parametrize("operation", ["gather", "take_along_dim"])
    def test_compiled_gradients(self, operation):
        compiled = torch.compile(
            lambda source, index: self.select(operation, source, index),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        cases = [
            ((4, 3), (torch.tensor([3, 1]), torch.tensor([2]))),
            ((5, 2), (torch.tensor([4, 2, 0]), torch.tensor([1]))),
        ]

        for truth_lengths, selected_rows in cases:
            inference_elements, inference_source, inference_index = _gather_cross_topology_row_case(
                truth_lengths, selected_rows, requires_grad=False
            )
            inference_output = compiled(inference_source, inference_index)
            _assert_elements(
                inference_output,
                self.dense_reference(operation, inference_elements, list(inference_index)),
            )

            elements, source, index = _gather_cross_topology_row_case(truth_lengths, selected_rows)
            output = compiled(source, index)
            reference_elements = [element.detach().clone().requires_grad_() for element in elements]
            expected = self.dense_reference(operation, reference_elements, list(index))

            assert output.ragged_dims == (1,)
            _assert_elements(output, expected)

            cotangent = torch.randn_like(output.concat)
            actual_gradients = torch.autograd.grad(output.concat, elements, cotangent)
            expected_concat = NT(expected, ragged_dims=(1,)).concat
            expected_gradients = torch.autograd.grad(expected_concat, reference_elements, cotangent)
            for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
                torch.testing.assert_close(actual, reference)


class TestDenseLookupNestedIndex:

    def test_values_shape_and_source_gradient(self):
        lookup = torch.randn(120, requires_grad=True)
        indices = NT(
            [torch.tensor([[0, 3], [7, -1]]), torch.tensor([[4, 9], [2, 6], [1, 8]])],
            ragged_dims=(0,),
        )
        expected = lookup[indices.concat]

        output = lookup[indices]

        cotangent = torch.randn_like(expected)
        actual_gradient = torch.autograd.grad(output.concat, lookup, cotangent, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, lookup, cotangent)[0]
        assert output.shape == indices.shape
        assert output.ragged_dims == indices.ragged_dims
        torch.testing.assert_close(output.concat, expected)
        torch.testing.assert_close(actual_gradient, expected_gradient)

    @pytest.mark.parametrize("values", [[0, 12], [0.0, 1.0]], ids=["bounds", "dtype"])
    def test_error_matches_dense_indexing(self, values):
        lookup = torch.arange(12.0)
        index = NT([torch.tensor(values)])

        with pytest.raises(IndexError):
            _ = lookup[index]

    def test_external_fake_index(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        template = NT(
            [torch.empty(2, 2, dtype=torch.long), torch.empty(3, 2, dtype=torch.long)],
            ragged_dims=(0,),
        )

        with fake_tensor_mod.FakeTensorMode():
            lookup = torch.empty(120)
            indices = template.packed_like(torch.empty(template.concat.shape, dtype=torch.long))
            output = lookup[indices]

        assert fake_tensor_mod.is_fake(output.concat)
        assert output.shape == (2, 3, 2)
        assert output.ragged_dims == (0,)

    def test_dynamic_fullgraph_with_source_gradient(self):
        compiled = torch.compile(
            lambda lookup, template, values: lookup[template.packed_like(values)].concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        for lengths in ((2, 3), (3, 5)):
            lookup = torch.randn(120, requires_grad=True)
            template = NT(
                [torch.empty(length, 2, dtype=torch.long) for length in lengths],
                ragged_dims=(0,),
            )
            values = torch.arange(template.concat.numel(), dtype=torch.long).reshape(template.concat.shape) % 120
            expected = lookup[values]

            output = compiled(lookup, template, values)

            cotangent = torch.randn_like(expected)
            actual_gradient = torch.autograd.grad(output, lookup, cotangent, retain_graph=True)[0]
            expected_gradient = torch.autograd.grad(expected, lookup, cotangent)[0]
            torch.testing.assert_close(output, expected)
            torch.testing.assert_close(actual_gradient, expected_gradient)


class TestDenseIndex:

    def test_batch_shaped_dense_index_rejects_padded_rows(self):
        shapes = [(3, 4), (5, 4), (2, 4)]
        nested = NT(_element_values(shapes))
        dense = torch.zeros(nested.shape, dtype=torch.long)

        # A complete logical dense index specifies every output row, including
        # rows beyond the true lengths of the shorter samples.
        with pytest.raises(RuntimeError, match="outside the selected dimension"):
            torch.gather(nested, 2, dense)

    @pytest.mark.parametrize(
        ("dim", "dense"),
        [
            (1, torch.zeros(2, 4, dtype=torch.long)),
            (2, torch.tensor([[3, 2, 1, 0]])),
        ],
        ids=["ragged-dim", "static-dim"],
    )
    def test_shared_dense_index(self, dim, dense):
        elements = _element_values([(3, 4), (5, 4), (2, 4)])

        output = torch.gather(NT(elements), dim, dense)

        _assert_elements(
            output,
            [torch.gather(element, dim - 1, dense) for element in elements],
        )

    def test_tall_index(self):
        elements = _element_values([(3, 4), (5, 4), (2, 4)])
        dense = torch.ones(6, 4, dtype=torch.long)

        output = torch.gather(NT(elements), 1, dense)

        _assert_elements(output, [torch.gather(element, 0, dense) for element in elements])


class TestGatherErrors:

    def test_segment_local_indices_do_not_cross_samples(self):
        elements = [torch.arange(12.0).reshape(3, 4), torch.arange(8.0).reshape(2, 4) + 100]
        indices = [torch.full((3, 4), 2, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long)]

        output = torch.gather(NT(elements), 1, NT(indices))

        _assert_elements(
            output,
            [torch.gather(element, 0, index) for element, index in zip(elements, indices)],
        )

    @pytest.mark.parametrize("bad_index", [-1, 2])
    def test_out_of_range_ragged_index_raises(self, bad_index):
        elements = [torch.zeros(3, 4), torch.zeros(2, 4)]
        indices = [torch.zeros(3, 4, dtype=torch.long), torch.full((2, 4), bad_index, dtype=torch.long)]

        with pytest.raises(IndexError, match="out of bounds"):
            torch.gather(NT(elements), 1, NT(indices))

    @pytest.mark.parametrize("backend,bad_index", [(None, -1), (None, 2), ("aot_eager", 2), ("inductor", 2)])
    @pytest.mark.parametrize(
        "operation", [torch.index_select, torch.Tensor.index_select, torch.ops.aten.index_select.default]
    )
    def test_index_select_bounds(self, backend, operation, bad_index):
        elements = [torch.zeros(2, 4), torch.ones(3, 4)]

        def call(input_, index):
            return operation(input_, 1, index)

        if backend is not None:
            call = torch.compile(call, backend=backend, fullgraph=True)

        with pytest.raises(IndexError if backend is None else RuntimeError, match="out of bounds"):
            call(NT(elements, ragged_dims=(0,)), torch.tensor([bad_index]))

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

    @pytest.mark.parametrize("dense_source,index_rows", [(True, 2), (False, 2), (False, 9)])
    @pytest.mark.parametrize("batch_first", [True, False])
    def test_mixed_take(self, device, dense_source, index_rows, batch_first):
        elements = [
            torch.randn(length, 3, device=device, requires_grad=True) for length in ((9, 9) if dense_source else (4, 9))
        ]
        source = torch.stack(elements) if dense_source else NT(elements, ragged_dims=(0,), batch_first=batch_first)
        indices = [torch.tensor([[0], [3]], device=device), torch.tensor([[1], [8]], device=device)]
        if index_rows == 9:
            indices = [idx.repeat(5, 1)[:index_rows].expand(-1, 3) for idx in indices]
        index = NT(indices, ragged_dims=(0,), batch_first=batch_first) if dense_source else torch.stack(indices)
        if not batch_first:
            if dense_source:
                source = source.movedim(0, 1)
            else:
                index = index.movedim(0, 1)
        dim = 1 if batch_first else 0
        expected = [torch.take_along_dim(part, idx, dim=0) for part, idx in zip(elements, indices)]
        for operation in (torch.take_along_dim, torch.Tensor.take_along_dim):
            output = operation(source, index, dim=dim)
            torch.testing.assert_close(output.unbind(0 if batch_first else 1), expected)
            actual_grad = torch.autograd.grad(output.square().sum(), elements, retain_graph=True)
            expected_grad = torch.autograd.grad(
                sum(part.square().sum() for part in expected), elements, retain_graph=True
            )
            torch.testing.assert_close(actual_grad, expected_grad)

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("lengths,index_batch", [((4, 4), 2), ((2, 4), 2), ((2, 4), 1)])
    @pytest.mark.parametrize("selected", [1, 3])
    def test_batched_take(self, device, batch_first, lengths, index_batch, selected):
        elements = [torch.randn(length, 2, 3, device=device, requires_grad=True) for length in lengths]
        source = NT(elements, ragged_dims=(0,), batch_first=batch_first)
        index = (
            torch.arange(4 * selected, device=device).reshape(1, 4, selected, 1)
            + torch.arange(index_batch, device=device).reshape(-1, 1, 1, 1)
        ) % 2
        reference_index = index.expand(2, -1, -1, -1)
        if not batch_first:
            index = index.movedim(0, 1)
        expected = [
            torch.take_along_dim(element, sample[:length], dim=-2)
            for element, sample, length in zip(elements, reference_index, lengths)
        ]

        for output in (torch.take_along_dim(source, index, dim=-2), source.take_along_dim(index, dim=-2)):
            _assert_elements(output, expected)
            actual_grad = torch.autograd.grad(output.concat.square().sum(), elements, retain_graph=True)
            reference_grad = torch.autograd.grad(
                sum(value.square().sum() for value in expected), elements, retain_graph=True
            )
            for actual, reference in zip(actual_grad, reference_grad, strict=True):
                torch.testing.assert_close(actual, reference)

    def test_flattened_take(self):
        elements = _element_values([(3, 4), (5, 4), (2, 4)])
        indices = [torch.tensor([0, 3, 5]), torch.tensor([1, 2]), torch.tensor([7])]

        output = torch.take_along_dim(NT(elements), NT(indices), dim=None)

        _assert_elements(
            output,
            [torch.take_along_dim(element, index, dim=None) for element, index in zip(elements, indices)],
        )

    def test_take_along_dim_broadcasts_a_dense_index(self):
        elements = _element_values([(3, 4), (5, 4), (2, 4)])
        dense = torch.zeros((1, 4), dtype=torch.long)

        for output in (
            torch.take_along_dim(NT(elements), dense, dim=1),
            NT(elements).take_along_dim(dense, dim=1),
        ):
            _assert_elements(
                output,
                [torch.take_along_dim(element, dense, dim=0) for element in elements],
            )

    def test_index_select_on_the_batch_dimension(self):
        elements = _element_values([(3, 4), (5, 4), (2, 4)])
        index = torch.tensor([2, 0, 2])

        output = torch.index_select(NT(elements), 0, index)

        _assert_elements(output, [elements[position] for position in index.tolist()])


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
        elements = [element.requires_grad_() for element in _element_values(shapes)]
        indices = _gather_build_indices(shapes, 0)

        torch.gather(NT(elements), 1, NT(indices)).concat.sum().backward()

        for element, index in zip(elements, indices):
            reference = element.detach().clone().requires_grad_()
            torch.gather(reference, 0, index).sum().backward()
            torch.testing.assert_close(element.grad, reference.grad)

    def test_take_along_dim_gradient_matches_dense_reference(self):
        shapes = [(3, 4), (5, 4), (2, 4)]
        elements = [element.requires_grad_() for element in _element_values(shapes)]
        indices = _gather_build_indices(shapes, 0)

        torch.take_along_dim(NT(elements), NT(indices), dim=1).concat.sum().backward()

        for element, index in zip(elements, indices):
            reference = element.detach().clone().requires_grad_()
            torch.take_along_dim(reference, index, dim=0).sum().backward()
            torch.testing.assert_close(element.grad, reference.grad)

    def test_index_select_gradient_matches_dense_reference(self):
        elements = [element.requires_grad_() for element in _element_values([(3, 4), (5, 4), (2, 4)])]
        index = torch.tensor([1, 0])

        torch.index_select(NT(elements), 1, index).concat.sum().backward()

        for element in elements:
            reference = element.detach().clone().requires_grad_()
            torch.index_select(reference, 0, index).sum().backward()
            torch.testing.assert_close(element.grad, reference.grad)

    def test_batch_index_select_accumulates_repeated_gradients(self):
        elements = [element.requires_grad_() for element in _element_values([(3, 4), (5, 4), (2, 4)])]

        torch.index_select(NT(elements), 0, torch.tensor([2, 0, 2])).concat.sum().backward()

        torch.testing.assert_close(elements[2].grad, torch.full((2, 4), 2.0))
        torch.testing.assert_close(elements[0].grad, torch.ones(3, 4))
        torch.testing.assert_close(elements[1].grad, torch.zeros(5, 4))


class TestGatherCompile:

    def test_gather_fullgraph(self):
        shapes = [(2, 3, 4), (2, 5, 4)]
        elements = _element_values(shapes)
        indices = _gather_build_indices(shapes, 1)
        compiled = torch.compile(
            lambda input_, index: torch.gather(input_, 2, index),
            backend="aot_eager",
            fullgraph=True,
        )

        output = compiled(
            NT(elements, ragged_dims=(1,)),
            NT(indices, ragged_dims=(1,)),
        )

        _assert_elements(
            output,
            [torch.gather(element, 1, index) for element, index in zip(elements, indices)],
        )

    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    @pytest.mark.parametrize(
        "operation", [torch.index_select, torch.Tensor.index_select, torch.ops.aten.index_select.default]
    )
    @pytest.mark.parametrize("dim", [1, 2])
    def test_index_select_fullgraph(self, device, backend, operation, dim):
        shapes = [(3, 4), (5, 4), (2, 4)]
        elements = [element.to(device).requires_grad_() for element in _element_values(shapes)]
        index = torch.tensor([1, 0], device=device)
        compiled = torch.compile(
            lambda input_, index_: operation(input_, dim, index_),
            backend=backend,
            fullgraph=True,
        )

        output = compiled(NT(elements, ragged_dims=(0,)), index)

        expected = [torch.index_select(element, dim - 1, index) for element in elements]
        _assert_elements(output, expected)
        torch.testing.assert_close(
            torch.autograd.grad(output.square().sum(), elements),
            torch.autograd.grad(sum(element.square().sum() for element in expected), elements),
        )

    def test_projected_gather_fullgraph(self):
        projected = _gather_projected_case()
        elements = list(projected)
        indices = _gather_build_indices([tuple(element.shape) for element in elements], 1)
        compiled = torch.compile(
            lambda input_, index: torch.gather(input_, 2, index),
            backend="aot_eager",
            fullgraph=True,
        )

        output = compiled(projected, NT(indices, ragged_dims=(1,)))

        _assert_elements(
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
            elements = _element_values(source_shapes)
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
            _assert_elements(actual, expected)

    def test_different_shapes_aot_eager_vjp_and_runtime_bounds(self):
        source_lengths = (3, 5, 2)
        index_lengths = (2, 6, 1)
        source_template = NT([torch.empty(length, 4) for length in source_lengths], ragged_dims=(0,))
        index_template = NT(
            [torch.empty(length, 2, dtype=torch.long) for length in index_lengths],
            ragged_dims=(0,),
        )
        indices = _gather_build_indices(
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


class TestLogicalDenseGather:

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize(
        "ragged_dims,axis,retained", [((0, 1), 1, 2), ((0, 1), 1, 1), ((1, 0), 0, 1), ((0, 1), 2, 1)]
    )
    def test_multi_take(self, device, batch_first, ragged_dims, axis, retained):
        shapes = ((2, 3, 4), (2 if retained == 2 else 3, 5, 4))
        parts = [torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
        shape = [1, retained, 1, 1]
        shape[axis + 1] = 2
        index = torch.arange(math.prod(shape), device=device).reshape(shape) % min(part.shape[axis] for part in parts)
        if not batch_first:
            index = index.movedim(0, 1)
        source = NT(parts, ragged_dims=ragged_dims, batch_first=batch_first)
        dim = axis + (batch_first or axis > 0)
        sample_index = index.select(0 if batch_first else 1, 0)
        expected = [torch.take_along_dim(part, sample_index, dim=axis) for part in parts]
        actual = torch.take_along_dim(source, index, dim=dim).unbind(0 if batch_first else 1)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(sum(p.square().sum() for p in actual), parts, retain_graph=True),
            torch.autograd.grad(sum(p.square().sum() for p in expected), parts),
        )

    @pytest.mark.parametrize("fullgraph", [False, True])
    @pytest.mark.parametrize("broadcast", [False, True])
    def test_multi_take_compile(self, device, fullgraph, broadcast):
        torch.compiler.reset()
        parts = [
            torch.randn(m, n, 4, device=device, requires_grad=True) for m, n in ((2, 3), (3 if broadcast else 2, 5))
        ]
        source = NT(parts, ragged_dims=(0, 1))
        index = torch.zeros((1, 1, 2, 1) if broadcast else (2, 2, 1, 4), device=device, dtype=torch.long)
        try:
            compiled = torch.compile(
                lambda source, index: torch.take_along_dim(source, index, dim=2),
                backend="aot_eager",
                fullgraph=fullgraph,
                dynamic=True,
            )
            actual = compiled(source, index)
            expected = [
                torch.take_along_dim(part, index[min(i, index.size(0) - 1)], dim=1) for i, part in enumerate(parts)
            ]
            if not broadcast:
                assert not isinstance(actual, NT)
            actual_parts = actual.unbind(0)
            torch.testing.assert_close(actual_parts, expected)
            torch.testing.assert_close(
                torch.autograd.grad(sum(part.square().sum() for part in actual_parts), parts),
                torch.autograd.grad(sum(part.square().sum() for part in expected), parts),
            )
        finally:
            torch.compiler.reset()

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_lengths(self, device, batch_first):
        parts = [torch.arange(n * 3, device=device, dtype=torch.float64).reshape(n, 3).requires_grad_() for n in (2, 4)]
        source = NestedTensor(parts, ragged_dims=(0,), batch_first=batch_first)
        index = torch.tensor([[[1, 0], [0, 1]], [[3, 2], [1, 0]]], device=device)
        if not batch_first:
            index = index.movedim(0, 1)
        actual = torch.gather(source, 1 if batch_first else 0, index)
        expected_parts = [
            torch.gather(part, 0, index.select(0 if batch_first else 1, i)) for i, part in enumerate(parts)
        ]
        expected = torch.stack(expected_parts, dim=0 if batch_first else 1)
        actual_values = actual.tensor if isinstance(actual, NestedTensor) else actual
        torch.testing.assert_close(actual_values, expected)
        cotangent = torch.arange(expected.numel(), device=device, dtype=expected.dtype).reshape_as(expected) / 7
        gradients = torch.autograd.grad(actual_values, parts, cotangent, retain_graph=True)
        wanted = torch.autograd.grad(expected, parts, cotangent)
        for gradient, reference in zip(gradients, wanted, strict=True):
            torch.testing.assert_close(gradient, reference)

    def test_maximal_shape(self, device):
        source = NestedTensor(
            [torch.tensor([1.0, 2.0], device=device), torch.tensor([3.0, 4.0, 5.0, 6.0], device=device)],
            ragged_dims=(0,),
        )
        index = torch.zeros((2, 4), dtype=torch.long, device=device)
        output = torch.gather(source, 1, index)
        values = output.concat.reshape(2, 4) if isinstance(output, NestedTensor) else output
        torch.testing.assert_close(values, torch.tensor([[1.0] * 4, [3.0] * 4], device=device))

    def test_bounds(self, device):
        source = NestedTensor([torch.arange(2, device=device), torch.arange(4, device=device)], ragged_dims=(0,))
        with pytest.raises((IndexError, RuntimeError), match="bounds"):
            torch.gather(source, 1, torch.tensor([[2], [0]], device=device))
        with pytest.raises(RuntimeError, match="int64|Long"):
            torch.gather(source, 1, torch.zeros((2, 1), device=device))

    @pytest.mark.parametrize("dim", [10, -10, 0.5])
    def test_dimensions(self, device, dim):
        source = NestedTensor([torch.zeros(n, 4, device=device) for n in (2, 4)], ragged_dims=(0,))
        index = torch.zeros((2, 1, 1), device=device, dtype=torch.long)
        with pytest.raises((IndexError, TypeError), match="range|int"):
            torch.gather(source, dim, index)

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_static_empty(self, device, batch_first):
        parts = [
            torch.arange(n * 3 * 4, device=device, dtype=torch.float64).reshape(n, 3, 4).requires_grad_()
            for n in (2, 4)
        ]
        source = NestedTensor(parts, ragged_dims=(0,), batch_first=batch_first).permute(0, 1, 3, 2)
        index = torch.tensor([[[[3], [0]]], [[[1], [2]]]], device=device).expand(-1, -1, -1, 3)
        empty_index = torch.empty((2, 5, 0, 3), device=device, dtype=torch.long)
        if not batch_first:
            index = index.movedim(0, 1)
            empty_index = empty_index.movedim(0, 1)
        actual = torch.gather(source, 2, index)
        empty = torch.gather(source, 2, empty_index)
        expected = torch.stack(
            [
                torch.gather(part.permute(0, 2, 1), 1, index.select(0 if batch_first else 1, i))
                for i, part in enumerate(parts)
            ],
            dim=0 if batch_first else 1,
        )
        torch.testing.assert_close(actual, expected)
        assert empty.shape == empty_index.shape
        actual_gradients = torch.autograd.grad(actual.sum(), parts, retain_graph=True)
        expected_gradients = torch.autograd.grad(expected.sum(), parts)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_cat(self, device):
        source = NestedTensor(
            [torch.arange(2, device=device, dtype=torch.float64), torch.arange(4, device=device, dtype=torch.float64)],
            ragged_dims=(0,),
        )
        dense = torch.tensor([[8.0], [9.0]], device=device, dtype=torch.float64)
        index = torch.tensor([[2], [4]], device=device)

        def operation(value, addition, indices):
            return torch.gather(torch.cat((value, addition), dim=-1), -1, indices)

        compiled = torch.compile(operation, backend="eager", fullgraph=True)
        actual = compiled(source, dense, index)
        torch.testing.assert_close(actual, dense)

    def test_sparse(self, device):
        source = NestedTensor(
            [torch.arange(2, device=device, dtype=torch.float64), torch.arange(4, device=device, dtype=torch.float64)],
            ragged_dims=(0,),
        )
        leaf = source.concat.detach().requires_grad_()
        source = source._packed_like_unchecked(leaf)
        output = torch.gather(source, 1, torch.tensor([[1], [3]], device=device), sparse_grad=True)
        gradient = torch.autograd.grad(output.sum(), leaf)[0]
        assert gradient.is_sparse
        torch.testing.assert_close(
            gradient.to_dense(), torch.tensor([0, 1, 0, 0, 0, 1], device=device, dtype=torch.float64)
        )

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("lengths, row_count", [((2, 2), 2), ((2, 4), 1)])
    @pytest.mark.parametrize("selected_axis, retained_extent", [(1, 3), (1, 1), (2, 4), (2, 2)])
    def test_static_sparse(self, device, batch_first, lengths, row_count, selected_axis, retained_extent):
        template = NestedTensor(
            [torch.zeros(n, 4, 3, device=device, dtype=torch.float64) for n in lengths],
            ragged_dims=(0,),
            batch_first=batch_first,
        )
        leaf = torch.arange(template.concat.numel(), device=device, dtype=torch.float64)
        leaf = leaf.reshape_as(template.concat).requires_grad_()
        source = template._packed_like_unchecked(leaf)
        shape = (2, row_count, 2, retained_extent) if selected_axis == 1 else (2, row_count, retained_extent, 2)
        source_extent = 4 if selected_axis == 1 else 3
        index = torch.arange(torch.Size(shape).numel(), device=device).reshape(shape) % source_extent
        logical_index = index if batch_first else index.movedim(0, 1)
        actual = torch.gather(source, selected_axis + 1, logical_index, sparse_grad=True)

        reference_leaf = leaf.detach().clone().requires_grad_()
        start = 0
        references = []
        for batch, length in enumerate(lengths):
            references.append(torch.gather(reference_leaf[start : start + length], selected_axis, index[batch]))
            start += length
        reference = torch.stack(references)
        if not batch_first:
            reference = reference.movedim(0, 1)
        torch.testing.assert_close(actual, reference)
        cotangent = torch.linspace(-0.8, 1.2, actual.numel(), device=device, dtype=torch.float64).reshape_as(actual)
        gradient = torch.autograd.grad(actual, leaf, cotangent)[0]
        reference_gradient = torch.autograd.grad(reference, reference_leaf, cotangent)[0]
        assert gradient.is_sparse
        torch.testing.assert_close(gradient.to_dense(), reference_gradient)


def _scatter_build_case(shapes, element_dim, *, seed=0, duplicates=False, **metadata):
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


_SCATTER_CASES = [
    ("leading-ragged", [(3, 4), (2, 4)], 0, {}),
    ("trailing-ragged", [(4, 3), (4, 2)], 1, {"ragged_dims": (1,)}),
    ("multi-ragged-inner", [(3, 4, 2), (2, 3, 2)], 1, {"ragged_dims": (0, 1)}),
    ("multi-ragged-outer", [(3, 4, 2), (2, 3, 2)], 0, {"ragged_dims": (0, 1)}),
    ("static-tail", [(2, 3, 4), (2, 2, 4)], 2, {"ragged_dims": (1,)}),
    ("batch-second", [(3, 4), (2, 4)], 0, {"batch_first": False}),
]


def _scatter_cases_named(*names):
    return [case for case in _SCATTER_CASES if case[0] in names]


def _scatter_case_params(cases=None):
    cases = _SCATTER_CASES if cases is None else cases
    return pytest.mark.parametrize(
        ("case_id", "shapes", "element_dim", "metadata"),
        cases,
        ids=[case[0] for case in cases],
    )


class TestScatterParity:

    @_scatter_case_params(
        _scatter_cases_named(
            "leading-ragged",
            "trailing-ragged",
            "multi-ragged-inner",
            "multi-ragged-outer",
            "static-tail",
            "batch-second",
        )
    )
    def test_scatter_tensor_source(self, case_id, shapes, element_dim, metadata):
        nested, index, source, bases, indices, sources = _scatter_build_case(shapes, element_dim, **metadata)
        dim = _logical_dim(element_dim, metadata.get("batch_first", True))

        output = torch.scatter(nested, dim, index, source)

        expected = [torch.scatter(base, element_dim, idx, src) for base, idx, src in zip(bases, indices, sources)]
        _assert_elements(output, expected)

    def test_scatter_scalar_source(self):
        shapes = [(3, 4), (2, 4)]
        nested, index, _, bases, indices, _ = _scatter_build_case(shapes, 0)

        output = torch.scatter(nested, 1, index, -7.5)

        _assert_elements(output, [torch.scatter(base, 0, idx, -7.5) for base, idx in zip(bases, indices)])

    @_scatter_case_params(_scatter_cases_named("leading-ragged", "trailing-ragged", "batch-second"))
    def test_scatter_add(self, case_id, shapes, element_dim, metadata):
        nested, index, source, bases, indices, sources = _scatter_build_case(
            shapes,
            element_dim,
            duplicates=True,
            **metadata,
        )
        dim = _logical_dim(element_dim, metadata.get("batch_first", True))

        output = torch.scatter_add(nested, dim, index, source)

        expected = [torch.scatter_add(base, element_dim, idx, src) for base, idx, src in zip(bases, indices, sources)]
        _assert_elements(output, expected)

    @pytest.mark.parametrize("reduce", ["sum", "prod", "amax", "amin"])
    @pytest.mark.parametrize("include_self", [False, True])
    def test_scatter_reduce(self, reduce, include_self):
        shapes = [(3, 4), (2, 4)]
        nested, index, source, bases, indices, sources = _scatter_build_case(shapes, 0, duplicates=True)

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
        _assert_elements(output, expected)


class TestScatterErrors:

    def test_segment_local_indices_do_not_cross_samples(self):
        bases = [torch.arange(24.0).reshape(2, 3, 4), torch.arange(16.0).reshape(2, 2, 4)]
        indices = [torch.full((2, 3, 4), 2, dtype=torch.long), torch.zeros(2, 2, 4, dtype=torch.long)]
        sources = [torch.full((2, 3, 4), 7.0), torch.full((2, 2, 4), 8.0)]

        output = torch.scatter(NT(bases), 2, NT(indices), NT(sources))

        _assert_elements(
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

    @pytest.mark.parametrize("operation", ["scatter", "scatter_add", "scatter_reduce"])
    def test_scatter_fullgraph(self, operation):
        shapes = [(2, 3, 4), (2, 2, 4)]
        nested, index, source, bases, indices, sources = _scatter_build_case(
            shapes,
            1,
            ragged_dims=(1,),
        )
        kwargs = {"reduce": "sum"} if operation == "scatter_reduce" else {}
        compiled = torch.compile(
            lambda input_, indices_, source_: getattr(torch, operation)(input_, 2, indices_, source_, **kwargs),
            backend="aot_eager",
            fullgraph=True,
        )

        output = compiled(nested, index, source)

        _assert_elements(
            output,
            [getattr(torch, operation)(base, 1, idx, src, **kwargs) for base, idx, src in zip(bases, indices, sources)],
        )


MASKED_CASES = [
    ("leading-ragged", [(3, 4), (2, 4)], {}),
    ("trailing-ragged", [(2, 3), (2, 2)], {"ragged_dims": (1,)}),
    ("empty-segment", [(1, 4), (0, 4)], {}),
    ("batch-second", [(3, 4), (2, 4)], {"batch_first": False}),
]


def _scatter_build_masked_case(shapes, *, exact_source):
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
        bases, masks, sources = _scatter_build_masked_case(shapes, exact_source=True)
        source_metadata = {key: value for key, value in metadata.items() if key != "ragged_dims"}

        output = torch.masked_scatter(
            NT(bases, **metadata),
            NT(masks, **metadata),
            NT(sources, **source_metadata),
        )

        _assert_elements(
            output,
            [torch.masked_scatter(base, mask, source) for base, mask, source in zip(bases, masks, sources)],
        )

    def test_oversized_source_matches_dense_reference(self):
        shapes = [(3, 4), (2, 4)]
        bases, masks, sources = _scatter_build_masked_case(shapes, exact_source=False)

        output = torch.masked_scatter(NT(bases), NT(masks), NT(sources))

        _assert_elements(
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

        _assert_elements(
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


def _split_batch_dim_of(batch_first: bool) -> int:
    return 0 if batch_first else 1


def _element_values(shapes, offset: float = 0.0):
    return [
        torch.arange(float(math.prod(shape))).reshape(shape) + 1000.0 * position + offset
        for position, shape in enumerate(shapes)
    ]


def _split_assert_parts_match(parts, expected_parts):
    assert len(parts) == len(expected_parts)
    for part, expected in zip(parts, expected_parts):
        _assert_elements(part, expected)


def _split_projected_case():
    return NT([torch.randn(2, 3, 4), torch.randn(2, 5, 4), torch.randn(2, 2, 4)]).permute(0, 3, 1, 2)


_SPLIT_CASES = [
    ("leading-ragged", [(3, 4), (5, 4), (2, 4)], {"ragged_dims": (0,)}),
    ("trailing-ragged", [(4, 3), (4, 5), (4, 2)], {"ragged_dims": (1,)}),
    ("multi-ragged", [(3, 4, 2), (5, 2, 2), (2, 3, 2)], {"ragged_dims": (0, 1)}),
    ("batch-second", [(3, 4), (5, 4), (2, 4)], {"batch_first": False}),
    ("empty-segment", [(0, 4), (5, 4)], {"ragged_dims": (0,)}),
]


def _split_cases_named(*names):
    return [case for case in _SPLIT_CASES if case[0] in names]


def _split_case_params(cases=None):
    cases = _SPLIT_CASES if cases is None else cases
    return pytest.mark.parametrize(
        ("case_id", "shapes", "metadata"),
        cases,
        ids=[case[0] for case in cases],
    )


class TestNarrow:

    @_split_case_params(_split_cases_named("leading-ragged", "batch-second", "empty-segment"))
    def test_batch_dimension(self, case_id, shapes, metadata):
        elements = _element_values(shapes)
        batch_dim = _split_batch_dim_of(metadata.get("batch_first", True))

        output = torch.narrow(NT(elements, **metadata), batch_dim, 1, 1)

        _assert_elements(output, elements[1:2])

    @_split_case_params(_split_cases_named("leading-ragged", "trailing-ragged", "multi-ragged", "batch-second"))
    def test_element_dimensions(self, case_id, shapes, metadata):
        elements = _element_values(shapes)
        batch_first = metadata.get("batch_first", True)
        for element_dim in range(len(shapes[0])):
            length = min(2, min(shape[element_dim] for shape in shapes))

            output = torch.narrow(
                NT(elements, **metadata),
                _logical_dim(element_dim, batch_first),
                0,
                length,
            )

            _assert_elements(
                output,
                [torch.narrow(element, element_dim, 0, length) for element in elements],
            )

    def test_method_and_negative_indices(self):
        elements = _element_values([(3, 4), (5, 4), (2, 4)])

        output = NT(elements).narrow(-1, -2, 2)

        _assert_elements(output, [torch.narrow(element, 1, -2, 2) for element in elements])

    def test_projected_layout(self):
        projected = _split_projected_case()
        elements = list(projected)

        output = torch.narrow(projected, 2, 0, 2)

        _assert_elements(output, [torch.narrow(element, 1, 0, 2) for element in elements])

    def test_invalid_span_or_dimension_raises(self):
        nested = NT([torch.randn(2, 4), torch.randn(5, 4)])
        with pytest.raises(RuntimeError):
            torch.narrow(nested, 1, 0, 4)
        with pytest.raises(IndexError):
            torch.narrow(nested, 3, 0, 1)

    @pytest.mark.parametrize(("shapes", "ragged", "element_dim"), _NARROW_LAYOUTS)
    @pytest.mark.parametrize(("start", "length"), ((0, 1), (-1, 1), (-2, 2), (0, 0)))
    def test_values(self, device, shapes, ragged, element_dim, start, length):
        _check_narrow(shapes, ragged, element_dim, device, start, length)

    @pytest.mark.parametrize("element_dim", (0, 1))
    @pytest.mark.parametrize("batch_first", (True, False))
    def test_batch_placement(self, device, element_dim, batch_first):
        _check_narrow(((2, 2, 4), (3, 3, 4)), (0, 1), element_dim, device, -1, 1, batch_first=batch_first, method=True)

    @pytest.mark.parametrize(
        ("shapes", "element_dim", "length"),
        (
            (((2, 0, 4), (3, 2, 4)), 0, 1),
            (((0, 2, 4), (3, 2, 4)), 0, 0),
            (((2, 2, 0), (3, 3, 0)), 1, 1),
        ),
    )
    def test_empty_elements(self, device, shapes, element_dim, length):
        _check_narrow(shapes, (0, 1), element_dim, device, 0, length)

    @pytest.mark.parametrize("dim", (1, 2))
    @pytest.mark.parametrize(("start", "length"), ((2, 1), (-3, 1), (3, 0)))
    def test_bounds(self, device, dim, start, length):
        source = NT(_narrow_elements(((2, 2, 4), (3, 3, 4)), device), ragged_dims=(0, 1))
        with pytest.raises(RuntimeError, match="exceeds dimension size.*element 0"):
            torch.narrow(source, dim, start, length)

    @pytest.mark.parametrize("dim", (1, 2))
    def test_empty_batch(self, device, dim):
        inputs = _narrow_elements(((2, 2, 4), (3, 3, 4)), device)
        source = torch.narrow(NT(inputs, ragged_dims=(0, 1)), 0, 0, 0)
        output = torch.narrow(source, dim, 0, 1)
        shape = [0, 3, 3, 4]
        shape[dim] = 1
        assert tuple(output.shape) == tuple(shape)
        assert output.element_sizes().shape == (0, 3)
        assert output.ragged_dims == (0, 1)
        assert output.packed_offsets().tolist() == [0]
        assert all(output.ragged_level_offsets(level).tolist() == [0] for level in (0, 1))
        torch.testing.assert_close(
            torch.autograd.grad(output.concat.sum(), inputs), tuple(torch.zeros_like(x) for x in inputs)
        )
        with pytest.raises(RuntimeError, match="exceeds dimension size"):
            torch.narrow(source, dim, 3, 1)

    @pytest.mark.parametrize("dim", (0, 3))
    def test_alias(self, device, dim):
        source = NT(_narrow_elements(((2, 2, 4), (3, 3, 4)), device), ragged_dims=(0, 1))
        expected = [part.detach().clone() for part in source.unbind()]
        output = torch.narrow(source, dim, 0, 1)
        with torch.no_grad():
            output.add_(1)
        if dim == 0:
            expected[0].add_(1)
        else:
            for part in expected:
                part[..., :1].add_(1)
        torch.testing.assert_close(source.unbind(), expected)

    def test_noncontiguous(self, device):
        template = NT(_narrow_elements(((2, 2, 4), (3, 3, 4)), device), ragged_dims=(0, 1))
        base = torch.arange(52, device=device, dtype=torch.float64).reshape(4, 13).requires_grad_()
        reference = base.detach().clone().requires_grad_()
        source = template.packed_like(base.t())
        output = torch.narrow(source, 2, torch.tensor(-1, device=device), 1)
        parts = reference.t().split((4, 9))
        expected = tuple(part.reshape(n, n, 4).narrow(1, -1, 1) for part, n in zip(parts, (2, 3)))
        actual = output.unbind()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        observed = torch.autograd.grad(sum(value.square().sum() for value in actual), base)[0]
        wanted = torch.autograd.grad(sum(value.square().sum() for value in expected), reference)[0]
        torch.testing.assert_close(observed, wanted, rtol=0, atol=0)

    def test_compile_limit(self):
        source = NT(_narrow_elements(((2, 2, 4), (3, 3, 4)), "cpu"), ragged_dims=(0, 1))
        compiled = torch.compile(
            lambda value: torch.narrow(value, 2, 0, 1), backend="aot_eager", fullgraph=True, dynamic=True
        )
        with pytest.raises(Exception, match="multi-ragged slicing requires concrete eager extents"):
            compiled(source)

    @pytest.mark.parametrize("dim", (1, 2))
    @pytest.mark.parametrize("factory", (torch.empty_like, torch.ones_like))
    def test_mask(self, device, dim, factory):
        """A public-API composition, not a test of the unavailable external class."""
        inputs = _narrow_elements(((2, 2, 4), (3, 3, 4)), device)
        references = tuple(value.detach().clone().requires_grad_() for value in inputs)
        source = NT(inputs, ragged_dims=(0, 1))
        narrowed = torch.narrow(source, dim, 0, 1)
        mask = factory(narrowed)
        mask.bernoulli_(0.5, generator=torch.Generator(device=device).manual_seed(941)).div_(0.5)
        output = source * mask
        assert mask.requires_grad is False
        assert mask.ragged_dims == (0, 1)
        sampled_masks = tuple(value.detach() for value in mask.unbind())
        expected = tuple(value * sampled for value, sampled in zip(references, sampled_masks))
        actual = output.unbind()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _assert_narrow_structure(output, expected, (0, 1))
        weights = tuple(
            torch.arange(1, value.numel() + 1, device=device, dtype=value.dtype).reshape_as(value) for value in expected
        )
        observed = torch.autograd.grad(sum((x * w).sum() for x, w in zip(actual, weights)), inputs)
        wanted = torch.autograd.grad(sum((x * w).sum() for x, w in zip(expected, weights)), references)
        torch.testing.assert_close(observed, wanted, rtol=0, atol=0)


class TestSplitFamily:

    @_split_case_params(_split_cases_named("leading-ragged", "batch-second"))
    def test_split_batch_dimension(self, case_id, shapes, metadata):
        elements = _element_values(shapes)
        nested = NT(elements, **metadata)
        batch_dim = _split_batch_dim_of(metadata.get("batch_first", True))

        parts = torch.split(nested, 2, dim=batch_dim)
        index_parts = torch.split(torch.arange(len(elements)), 2)
        expected = [[elements[int(index)] for index in indices] for indices in index_parts]

        _split_assert_parts_match(parts, expected)

    @pytest.mark.parametrize("operation", ["chunk", "tensor_split"])
    def test_chunk_and_tensor_split_batch_dimension(self, operation):
        elements = _element_values([(3, 4), (5, 4), (2, 4)])
        function = getattr(torch, operation)

        parts = function(NT(elements), 2, dim=0)
        index_parts = function(torch.arange(len(elements)), 2)

        _split_assert_parts_match(parts, [[elements[int(index)] for index in indices] for indices in index_parts])

    @_split_case_params(_split_cases_named("leading-ragged", "trailing-ragged", "multi-ragged", "batch-second"))
    def test_split_element_dimensions(self, case_id, shapes, metadata):
        elements = _element_values(shapes)
        nested = NT(elements, **metadata)
        batch_first = metadata.get("batch_first", True)
        for element_dim in range(len(shapes[0])):
            references = [torch.split(element, 2, dim=element_dim) for element in elements]
            dim = _logical_dim(element_dim, batch_first)
            if len({len(reference) for reference in references}) > 1:
                with pytest.raises(ValueError, match="uniform per-element split counts"):
                    torch.split(nested, 2, dim=dim)
                continue

            parts = torch.split(nested, 2, dim=dim)

            _split_assert_parts_match(parts, [list(part) for part in zip(*references)])

    def test_chunk_and_tensor_split_element_dimensions(self):
        elements = _element_values([(3, 4), (5, 4), (2, 4)])
        nested = NT(elements)

        chunked = torch.chunk(nested, 2, dim=2)
        chunk_references = [torch.chunk(element, 2, dim=1) for element in elements]
        _split_assert_parts_match(chunked, [list(part) for part in zip(*chunk_references)])

        split = torch.tensor_split(nested, 3, dim=1)
        split_references = [torch.tensor_split(element, 3, dim=0) for element in elements]
        _split_assert_parts_match(split, [list(part) for part in zip(*split_references)])

    def test_tensor_split_preserves_zero_width_and_source_derived_empty_shapes(self):
        elements = [torch.arange(2.0), torch.arange(10.0, 13.0)]
        nested = NT(elements, ragged_dims=(0,))

        parts = torch.tensor_split(nested, [0], dim=1)

        references = [torch.tensor_split(element, [0]) for element in elements]
        _split_assert_parts_match(parts, [list(part) for part in zip(*references)])
        empty_parts = torch.tensor_split(nested[:0], [0], dim=1)
        assert [part.shape for part in empty_parts] == [part[:0].shape for part in parts]

    def test_split_sections_and_errors(self):
        elements = _element_values([(3, 4), (5, 4), (2, 4)])
        parts = torch.split(NT(elements), [1, 2], dim=0)
        _split_assert_parts_match(parts, [elements[:1], elements[1:]])

        with pytest.raises(ValueError):
            torch.split(NT(elements), [1], dim=0)

    def test_vsplit_hsplit_and_dsplit_axes(self):
        elements = _element_values([(4, 6), (4, 6)])
        nested = NT(elements, ragged_dims=(1,))

        _split_assert_parts_match(torch.vsplit(nested, 2), [elements[:1], elements[1:]])
        _split_assert_parts_match(
            torch.hsplit(nested, 2),
            [[element[:2] for element in elements], [element[2:] for element in elements]],
        )
        _split_assert_parts_match(
            torch.dsplit(nested, 2),
            [[element[:, :3] for element in elements], [element[:, 3:] for element in elements]],
        )


class TestUnbind:

    @pytest.mark.parametrize(
        "entry,ragged,batch_first,patterns",
        [
            ("torch", (1,), True, (((1, 3), (1, 5)), ((1, 0), (1, 1)))),
            ("method", (0, 1), False, (((2, 3, 2), (4, 1, 2)), ((0, 2, 2), (1, 0, 2)))),
            ("native", (1, 0), True, (((3, 2, 2), (1, 4, 2)), ((2, 0, 2), (1, 1, 2)))),
        ],
        ids=["torch-trailing", "method-multi", "native-permuted"],
    )
    def test_batch(self, device, entry, ragged, batch_first, patterns):
        dim = 0 if batch_first else 1

        def unbind(source):
            if entry == "torch":
                return torch.unbind(source, dim)
            if entry == "method":
                return source.unbind(dim)
            return torch.ops.aten.unbind.int(self=source, dim=dim)

        torch.compiler.reset()
        compiled = torch.compile(unbind, backend="aot_eager", fullgraph=True, dynamic=True)
        try:
            for shapes in patterns:
                parts = [torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
                references = [part.detach().clone().requires_grad_() for part in parts]
                source = NT(parts, ragged_dims=ragged, batch_first=batch_first)
                actual = compiled(source)
                assert len(actual) == len(parts)
                assert all(isinstance(part, torch.Tensor) and not isinstance(part, NT) for part in actual)
                torch.testing.assert_close(actual, tuple(references))
                gradients = torch.autograd.grad(sum(part.square().sum() for part in actual), parts)
                reference_gradients = torch.autograd.grad(sum(part.square().sum() for part in references), references)
                torch.testing.assert_close(gradients, reference_gradients)
                with torch.inference_mode():
                    torch.testing.assert_close(compiled(source), tuple(references))
        finally:
            torch.compiler.reset()

    def test_batch_and_static_dimensions(self):
        elements = _element_values([(3, 4), (5, 4), (2, 4)])
        nested = NT(elements, ragged_dims=(0,))

        batch_parts = torch.unbind(nested, dim=0)
        for actual, reference in zip(batch_parts, elements):
            torch.testing.assert_close(actual, reference)

        static_parts = torch.unbind(nested, dim=2)
        references = [torch.unbind(element, dim=1) for element in elements]
        _split_assert_parts_match(static_parts, [list(part) for part in zip(*references)])

    def test_ragged_dimension_is_refused(self):
        nested = NT(_element_values([(3, 4), (5, 4)]), ragged_dims=(0,))

        with pytest.raises(NotImplementedError, match="ragged"):
            torch.unbind(nested, dim=1)


class TestCat:

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("dense", [True, False])
    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    def test_dynamic_cat(self, device, batch_first, dense, backend):
        torch.compiler.reset()
        dim = 1 if batch_first else 0
        compiled = torch.compile(lambda a, b: torch.cat((a, b), dim), backend=backend, fullgraph=True, dynamic=True)
        for lengths, right_lengths in (((2, 4), (1, 1)), ((3, 5), (1, 1)), ((0, 0), (0, 0))):
            parts = [torch.randn(length, 3, device=device, requires_grad=True) for length in lengths]
            right = [torch.randn(length, 3, device=device, requires_grad=True) for length in right_lengths]
            source = NT(parts, ragged_dims=(0,), batch_first=batch_first)
            if dense:
                other = torch.stack(right, dim=0 if batch_first else 1).detach().requires_grad_()
                operands = (*parts, other)
                reference = [
                    torch.cat((part, other.select(0 if batch_first else 1, i))) for i, part in enumerate(parts)
                ]
            else:
                other = NT(right, ragged_dims=(0,), batch_first=batch_first)
                operands = (*parts, *right)
                reference = [torch.cat((left, value)) for left, value in zip(parts, right)]
            output = compiled(source, other)
            torch.testing.assert_close(output.unbind(0 if batch_first else 1), tuple(reference))
            gradients = torch.autograd.grad(output.concat.square().sum(), operands)
            expected = torch.autograd.grad(sum(value.square().sum() for value in reference), operands)
            torch.testing.assert_close(gradients, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="spatial NestedTensor conv2d requires CUDA")
    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    def test_spatial_cat(self, backend):
        torch.compiler.reset()

        def model(images, cls, weight):
            tokens = F.conv2d(images, weight, stride=2).flatten(2).transpose(1, 2)
            return torch.cat((cls.expand(2, -1, -1), tokens), dim=1)

        compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=True)
        for shapes in [((4, 10), (8, 6)), ((8, 10), (2, 4))]:
            parts = [torch.randn(3, *shape, device="cuda", requires_grad=True) for shape in shapes]
            cls = torch.randn(1, 1, 3, device="cuda", requires_grad=True)
            weight = torch.randn(3, 3, 2, 2, device="cuda", requires_grad=True)
            output = compiled(NT(parts, ragged_dims=(1, 2)), cls, weight)
            reference = [
                torch.cat((cls[0], F.conv2d(part.unsqueeze(0), weight, stride=2).squeeze(0).flatten(1).T))
                for part in parts
            ]
            torch.testing.assert_close(output.unbind(), tuple(reference), rtol=1e-4, atol=1e-4)
            operands = (*parts, cls, weight)
            gradients = torch.autograd.grad(output.concat.square().sum(), operands)
            expected = torch.autograd.grad(sum(value.square().sum() for value in reference), operands)
            torch.testing.assert_close(gradients, expected, rtol=1e-4, atol=1e-4)

    @_split_case_params(
        _split_cases_named(
            "leading-ragged",
            "trailing-ragged",
            "multi-ragged",
            "batch-second",
            "empty-segment",
        )
    )
    def test_batch_dimension(self, case_id, shapes, metadata):
        left = _element_values(shapes)
        right = _element_values(shapes, offset=7.0)
        batch_dim = _split_batch_dim_of(metadata.get("batch_first", True))

        output = torch.cat((NT(left, **metadata), NT(right, **metadata)), dim=batch_dim)

        _assert_elements(output, [*left, *right])

    @_split_case_params(_split_cases_named("leading-ragged", "trailing-ragged", "multi-ragged", "batch-second"))
    def test_element_dimensions(self, case_id, shapes, metadata):
        left = _element_values(shapes)
        right = _element_values(shapes, offset=7.0)
        batch_first = metadata.get("batch_first", True)
        for element_dim in range(len(shapes[0])):

            output = torch.cat(
                (NT(left, **metadata), NT(right, **metadata)),
                dim=_logical_dim(element_dim, batch_first),
            )

            _assert_elements(
                output,
                [torch.cat((a, b), dim=element_dim) for a, b in zip(left, right)],
            )

    def test_projected_layout(self):
        projected = _split_projected_case()
        elements = list(projected)

        output = torch.cat((projected, projected), dim=2)

        _assert_elements(output, [torch.cat((element, element), dim=1) for element in elements])

    def test_aligns_static_orders_and_gradients(self):
        lengths = (3, 5)
        left_template = NT([torch.empty(length, 10, 2) for length in lengths], ragged_dims=(0,))
        middle_template = NT([torch.empty(length, 2, 64) for length in lengths], ragged_dims=(0,))
        right_template = NT([torch.empty(length, 2, 4) for length in lengths], ragged_dims=(0,))
        left_values = torch.randn_like(left_template.concat, requires_grad=True)
        middle_values = torch.randn_like(middle_template.concat, requires_grad=True)
        right_values = torch.randn_like(right_template.concat, requires_grad=True)
        left = left_template.packed_like(left_values).movedim(2, 3)
        middle = middle_template.packed_like(middle_values)
        right = right_template.packed_like(right_values)
        expected = torch.cat(
            (left_values, middle_values.movedim(1, 2), right_values.movedim(1, 2)),
            dim=1,
        )

        output = torch.cat((left, middle, right), dim=-1)

        leaves = (left_values, middle_values, right_values)
        cotangent = torch.randn_like(expected)
        actual_gradients = torch.autograd.grad(output.concat, leaves, cotangent)
        expected_gradients = torch.autograd.grad(expected, leaves, cotangent)
        assert output.shape == torch.Size((2, 5, 2, 78))
        torch.testing.assert_close(output.concat, expected)
        for actual, reference in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, reference)

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
            group = _element_values(shapes, offset=7.0 * position)
            elements.extend(group)
            operands.append(NT(group))

        _assert_elements(torch.cat(operands, dim=0), elements)

    def test_accepts_a_dense_batch_operand(self):
        elements = _element_values([(4, 3), (4, 5)])
        dense = _element_values([(4, 2)], offset=7.0)[0]

        _assert_elements(torch.cat((NT(elements), dense), dim=0), [*elements, dense])

    def test_refuses_to_invent_a_ragged_dimension(self):
        left = _element_values([(2, 3), (1, 3)])
        right = _element_values([(2, 4), (1, 4)], offset=7.0)

        with pytest.raises(ValueError, match="no operand varies"):
            torch.cat((NT(left), NT(right)), dim=0)


class TestBatchTransforms:

    @pytest.mark.parametrize("operation", ["stack", "movedim", "permute"])
    def test_values(self, device, operation):
        parts = [torch.randn(length, 3, device=device, dtype=torch.float64, requires_grad=True) for length in (2, 5)]
        references = [part.detach().clone().requires_grad_() for part in parts]
        source = NT(parts, ragged_dims=(0,))
        if operation == "stack":
            output = torch.stack((source, source.sin()), dim=0)
            expected = [torch.stack((part, part.sin())) for part in references]
            shape = (2, 2, 5, 3)
        else:
            output = torch.movedim(source, 0, 1) if operation == "movedim" else torch.permute(source, (1, 0, 2))
            expected = references
            shape = (5, 2, 3)
        assert output.batch_first is False
        assert output.shape == shape
        torch.testing.assert_close(output.unbind(1), expected)
        restored = torch.permute(output, (1, 0, *range(2, output.dim())))
        assert restored.batch_first is True
        torch.testing.assert_close(restored.unbind(0), expected)
        probes = [torch.randn_like(part) for part in expected]
        actual_gradients = torch.autograd.grad(
            sum((part * probe).sum() for part, probe in zip(restored, probes)), parts
        )
        reference_gradients = torch.autograd.grad(
            sum((part * probe).sum() for part, probe in zip(expected, probes)), references
        )
        torch.testing.assert_close(actual_gradients, reference_gradients)

    def test_limit(self):
        source = NT([torch.zeros(2, 3), torch.zeros(5, 3)], ragged_dims=(0,))
        with pytest.raises(ValueError, match="position zero or one"):
            torch.permute(source, (1, 2, 0))
        with pytest.raises(ValueError, match="position zero or one"):
            torch.movedim(source, 0, 2)


class TestStack:

    @pytest.mark.parametrize("batch_first", [False, True])
    def test_element_dimension(self, batch_first):
        shapes = [(3, 4), (5, 4), (2, 4)]
        left = _element_values(shapes)
        right = _element_values(shapes, offset=7.0)
        dim = 2

        output = torch.stack(
            (NT(left, batch_first=batch_first), NT(right, batch_first=batch_first)),
            dim=dim,
        )

        _assert_elements(output, [torch.stack((a, b), dim=1) for a, b in zip(left, right)])

    @pytest.mark.parametrize("dim", [0, 1])
    def test_batch_limit(self, dim):
        nested = NT(_element_values([(3, 4), (5, 4)]), batch_first=False)

        with pytest.raises(NotImplementedError, match="batch dimension"):
            torch.stack((nested, nested), dim=dim)

    def test_common_stack_aliases(self):
        left = _element_values([(3, 4), (5, 4)])
        right = _element_values([(3, 4), (5, 4)], offset=7.0)
        nested_left, nested_right = NT(left), NT(right)

        _assert_elements(torch.vstack((nested_left, nested_right)), [*left, *right])
        _assert_elements(
            torch.hstack((nested_left, nested_right)),
            [torch.cat((a, b), dim=0) for a, b in zip(left, right)],
        )
        _assert_elements(
            torch.dstack((nested_left, nested_right)),
            [torch.cat((a, b), dim=1) for a, b in zip(left, right)],
        )


class TestMethodAndAliasSpellings:

    def test_method_forms(self):
        elements = _element_values([(3, 4), (5, 4)])
        nested = NT(elements)

        _assert_elements(nested.narrow(0, 0, 1), elements[:1])
        _split_assert_parts_match(nested.split(1, dim=0), [[elements[0]], [elements[1]]])
        _split_assert_parts_match(nested.chunk(2, dim=0), [[elements[0]], [elements[1]]])
        _split_assert_parts_match(nested.tensor_split(2, dim=0), [[elements[0]], [elements[1]]])
        for actual, reference in zip(nested.unbind(0), elements):
            torch.testing.assert_close(actual, reference)

    def test_tensor_split_accepts_tensor_indices(self):
        elements = _element_values([(3, 4), (5, 4)])

        parts = torch.tensor_split(NT(elements), torch.tensor([1]), dim=0)

        _split_assert_parts_match(parts, [elements[:1], elements[1:]])

    def test_concat_aliases(self):
        left = _element_values([(3, 4), (5, 4)])
        right = _element_values([(3, 4), (5, 4)], offset=7.0)

        for alias in (torch.concat, torch.concatenate):
            _assert_elements(alias((NT(left), NT(right)), dim=0), [*left, *right])


class TestSplitCompile:

    def test_narrow_and_split_fullgraph(self):
        elements = _element_values([(3, 4), (5, 4), (2, 4)])
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

        _assert_elements(output, [torch.narrow(element, 0, 0, 2) for element in elements])
        references = [torch.split(element, 2, dim=1) for element in elements]
        _split_assert_parts_match(parts, [list(part) for part in zip(*references)])

    def test_cat_and_stack_fullgraph(self):
        shapes = [(3, 4), (5, 4), (2, 4)]
        left = _element_values(shapes)
        right = _element_values(shapes, offset=7.0)
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

        _assert_elements(cat_output, [*left, *right])
        _assert_elements(
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

            _assert_elements(output, [*left, *right])

    def test_static_order_aligned_cat_dynamic_fullgraph_with_gradient(self):
        def run(left_template, left, middle_template, middle, right_template, right):
            left_nested = left_template.packed_like(left).movedim(2, 3)
            middle_nested = middle_template.packed_like(middle)
            right_nested = right_template.packed_like(right)
            return torch.cat((left_nested, middle_nested, right_nested), dim=-1).concat

        compiled = torch.compile(run, backend="aot_eager", fullgraph=True, dynamic=True)
        for lengths in ((2, 3), (3, 5)):
            left_template = NT([torch.empty(length, 10, 2) for length in lengths], ragged_dims=(0,))
            middle_template = NT([torch.empty(length, 2, 64) for length in lengths], ragged_dims=(0,))
            right_template = NT([torch.empty(length, 2, 4) for length in lengths], ragged_dims=(0,))
            left = torch.randn_like(left_template.concat, requires_grad=True)
            middle = torch.randn_like(middle_template.concat, requires_grad=True)
            right = torch.randn_like(right_template.concat, requires_grad=True)
            expected = torch.cat(
                (left, middle.movedim(1, 2), right.movedim(1, 2)),
                dim=1,
            )

            output = compiled(left_template, left, middle_template, middle, right_template, right)

            leaves = (left, middle, right)
            cotangent = torch.randn_like(expected)
            actual_gradients = torch.autograd.grad(output, leaves, cotangent, retain_graph=True)
            expected_gradients = torch.autograd.grad(expected, leaves, cotangent)
            torch.testing.assert_close(output, expected)
            for actual, reference in zip(actual_gradients, expected_gradients):
                torch.testing.assert_close(actual, reference)


class TestSplitAutograd:

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

    @pytest.mark.parametrize("dim", [0, 1], ids=["batch", "ragged"])
    def test_cat_fullgraph_preserves_repeated_input_gradient(self, device, dim):
        values = torch.randn(5, 3, device=device, requires_grad=True)
        source = NestedTensor(values.split((2, 3)), ragged_dims=(0,))
        compiled = torch.compile(
            lambda value: torch.cat((value, value), dim=dim),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        output = compiled(source)
        expected = list(source) * 2 if dim == 0 else [torch.cat((part, part)) for part in source]
        for actual, reference in zip(output, expected):
            torch.testing.assert_close(actual, reference)
        gradient = torch.autograd.grad(output.concat.sum(), values)[0]
        torch.testing.assert_close(gradient, torch.full_like(values, 2))


class TestDenseMask:

    @pytest.mark.parametrize("layout", ["vector", "matrix", "static"])
    def test_values(self, device, layout):
        shapes = {"vector": ((3,), (5,)), "matrix": ((2, 3), (3, 2)), "static": ((2, 3), (2, 5))}[layout]
        masks = [torch.arange(math.prod(shape), device=device).reshape(shape) % 2 == 0 for shape in shapes]
        mask = NT(masks, ragged_dims=(1,) if layout == "static" else tuple(range(len(shapes[0]))))
        logical_shape = tuple(max(shape[axis] for shape in shapes) for axis in range(len(shapes[0])))
        reference_mask = torch.zeros((2, *logical_shape), device=device, dtype=torch.bool)
        for i, part in enumerate(masks):
            reference_mask[(i, *(slice(0, size) for size in part.shape))] = part
        tail = (2,) if layout == "vector" else ()
        values = torch.randn(2, *logical_shape, *tail, device=device, dtype=torch.float64, requires_grad=True)
        reference_values = values.detach().clone().requires_grad_()
        output = values[mask]
        expected = reference_values[reference_mask]
        torch.testing.assert_close(output, expected)
        probe = torch.randn_like(expected)
        torch.testing.assert_close(
            torch.autograd.grad(output, values, probe), torch.autograd.grad(expected, reference_values, probe)
        )

        for method in (False, True):
            values = values.detach().requires_grad_()
            reference_values = values.detach().clone().requires_grad_()
            updates = [
                torch.randn(int(part.sum()), *tail, device=device, dtype=torch.float64, requires_grad=True)
                for part in masks
            ]
            references = [part.detach().clone().requires_grad_() for part in updates]
            source = values * 2
            expected = reference_values * 2
            alias = source.detach()
            if method:
                source.index_put_((mask,), NT(updates, ragged_dims=(0,)))
                expected.index_put_((reference_mask,), torch.cat(references))
            else:
                source[mask] = NT(updates, ragged_dims=(0,))
                expected[reference_mask] = torch.cat(references)
            torch.testing.assert_close(source, expected)
            torch.testing.assert_close(alias, expected)
            probe = torch.randn_like(expected)
            torch.testing.assert_close(
                torch.autograd.grad(source, (values, *updates), probe),
                torch.autograd.grad(expected, (reference_values, *references), probe),
            )

    def test_errors(self):
        mask = NT([torch.tensor([True, False]), torch.tensor([False, True])], ragged_dims=(0,))
        with pytest.raises(IndexError, match="mask shape"):
            torch.ones(2, 3)[mask]
        source = torch.ones(2, 2, requires_grad=True)
        with pytest.raises(RuntimeError, match="leaf Variable"):
            source.index_put_((mask,), torch.ones(2))


class TestMixedCat:

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("static_axis", [False, True])
    @pytest.mark.parametrize("reverse", [False, True])
    def test_axis_gradients(self, device, batch_first, static_axis, reverse):
        parts = [torch.randn(n, 3, device=device, dtype=torch.float64, requires_grad=True) for n in (0, 2, 4)]
        source = NestedTensor(parts, ragged_dims=(0,), batch_first=batch_first)
        dense = torch.randn(
            (3, 4, 2) if static_axis else (3, 1, 3), device=device, dtype=torch.float64, requires_grad=True
        )
        operand = dense if batch_first else dense.movedim(0, 1)
        dim = 2 if static_axis else (1 if batch_first else 0)
        output = torch.cat((operand, source) if reverse else (source, operand), dim=dim)
        additions = [dense[i, : part.shape[0]] if static_axis else dense[i] for i, part in enumerate(parts)]
        wanted = [
            torch.cat((addition, part) if reverse else (part, addition), dim=1 if static_axis else 0)
            for part, addition in zip(parts, additions)
        ]
        torch.testing.assert_close(output.concat, torch.cat(wanted))
        expected_shapes = torch.tensor([list(value.shape) for value in wanted])
        torch.testing.assert_close(output.element_sizes().cpu(), expected_shapes)
        cotangent = torch.linspace(-0.8, 1.2, output.concat.numel(), device=device, dtype=torch.float64).reshape_as(
            output.concat
        )
        gradients = torch.autograd.grad(output.concat, (*parts, dense), cotangent, retain_graph=True)
        references = torch.autograd.grad(torch.cat(wanted), (*parts, dense), cotangent)
        for gradient, reference in zip(gradients, references, strict=True):
            torch.testing.assert_close(gradient, reference)


class TestRaggedCat:

    @pytest.mark.parametrize("joining_axis", [0, 1, 2])
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("static_prefix", [True, False])
    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize("both_ragged", [True, False])
    def test_shapes_gradients(self, device, joining_axis, batch_first, static_prefix, reverse, both_ragged):
        left_shapes = [[0, 2, 3], [2, 3, 4], [3, 4, 2]]
        right_shapes = [list(shape) for shape in left_shapes]
        for index, shape in enumerate(right_shapes):
            shape[joining_axis] = [2, 0, 1][index] if both_ragged else 1
        left = [torch.randn(*shape, 2, device=device, dtype=torch.float64, requires_grad=True) for shape in left_shapes]
        right = [
            torch.randn(*shape, 2, device=device, dtype=torch.float64, requires_grad=True) for shape in right_shapes
        ]
        left_values = [value.movedim(-1, 0) for value in left] if static_prefix else left
        right_values = [value.movedim(-1, 0) for value in right] if static_prefix else right
        ragged_dims = (1, 2, 3) if static_prefix else (0, 1, 2)
        element_join = joining_axis + int(static_prefix)
        right_ragged = ragged_dims if both_ragged else tuple(axis for axis in ragged_dims if axis != element_join)
        first = NestedTensor(left_values, ragged_dims=ragged_dims, batch_first=batch_first)
        second = NestedTensor(right_values, ragged_dims=right_ragged, batch_first=batch_first)
        operands = (second, first) if reverse else (first, second)
        logical_join = element_join + (1 if batch_first or element_join else 0)
        output = torch.cat(operands, dim=logical_join)
        expected_parts = [
            torch.cat((b, a) if reverse else (a, b), dim=element_join)
            for a, b in zip(left_values, right_values, strict=True)
        ]
        expected = torch.cat(
            [value.movedim(0, -1).flatten(0, 2) if static_prefix else value.flatten(0, 2) for value in expected_parts]
        )
        torch.testing.assert_close(output.concat, expected)
        expected_sizes = torch.tensor([list(value.shape) for value in expected_parts])
        torch.testing.assert_close(output.element_sizes().cpu(), expected_sizes)
        assert output.ragged_dims == ragged_dims
        expected_outer = [max(shape[axis] for shape in expected_sizes.tolist()) for axis in range(4)]
        expected_outer.insert(0 if batch_first else 1, 3)
        assert tuple(output.shape) == tuple(expected_outer)
        cotangent = torch.linspace(-0.8, 1.2, output.concat.numel(), device=device, dtype=torch.float64).reshape_as(
            output.concat
        )
        gradients = torch.autograd.grad(output.concat, (*left, *right), cotangent, retain_graph=True)
        references = torch.autograd.grad(expected, (*left, *right), cotangent)
        for observed, wanted in zip(gradients, references, strict=True):
            torch.testing.assert_close(observed, wanted)

    def test_reduction_identity(self, device):
        parts = [torch.randn(2, 2, device=device), torch.randn(3, 3, device=device)]
        source = NestedTensor(parts, ragged_dims=(0, 1))
        identity = torch.zeros_like(source.mean(dim=-1, keepdim=True))
        output = torch.cat((source, identity), dim=-1)
        expected = torch.cat([torch.cat((part, torch.zeros_like(part[:, :1])), dim=-1).flatten() for part in parts])
        torch.testing.assert_close(output.concat, expected)
        torch.testing.assert_close(output.element_sizes().cpu(), torch.tensor([[2, 3], [3, 4]]))

    def test_nonjoining_mismatch(self, device):
        left = NestedTensor([torch.ones(2, 3, device=device), torch.ones(4, 5, device=device)], ragged_dims=(0, 1))
        right = NestedTensor([torch.ones(3, 1, device=device), torch.ones(4, 1, device=device)], ragged_dims=(0,))
        with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
            torch.cat((left, right), dim=-1)
