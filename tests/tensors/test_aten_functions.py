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

r"""Tests for ``danling.tensors.aten_functions`` — aten-level dispatch handlers."""

import math
import operator

import pytest
import torch

from danling.tensors import NestedTensor
from tests.tensors.utils import assert_close, nested_rand, ragged_shapes

NT = NestedTensor


def reference_options(source: NestedTensor) -> dict:
    r"""Return public construction options for an elementwise reference."""
    return {
        "batch_first": source.batch_first,
        "padding_value": source.padding_value,
        "mask_value": source.mask_value,
    }


class TestClone:

    @pytest.mark.parametrize("backend", [None, "aot_eager", "inductor"])
    @pytest.mark.parametrize("batch_first,transposed", [(True, False), (False, True)])
    def test_copy(self, device, backend, batch_first, transposed):
        torch.compiler.reset()
        operation = (
            torch.clone
            if backend is None
            else torch.compile(torch.clone, backend=backend, fullgraph=True, dynamic=True)
        )
        for lengths in ((2, 4), (3, 5)):
            parts = [
                torch.randn(length, 3, 4, device=device, dtype=torch.float64, requires_grad=True) for length in lengths
            ]
            source = NT(parts, ragged_dims=(0,), batch_first=batch_first, padding_value=-1, mask_value=True)
            expected = parts
            if transposed:
                source = source.transpose(-1, -2)
                expected = [part.transpose(-1, -2) for part in parts]

            output = operation(source)

            assert output.shape == source.shape
            assert output.stride() == source.stride()
            assert output.ragged_dims == source.ragged_dims
            assert output.batch_first is batch_first
            assert output.padding_value == -1
            assert output.mask_value is True
            torch.testing.assert_close(output.element_sizes(), source.element_sizes())
            torch.testing.assert_close(output.unbind(0 if batch_first else 1), expected)
            torch.testing.assert_close(
                torch.autograd.grad(output.concat.square().sum(), parts), [2 * part for part in parts]
            )
            before = source.concat.detach().clone()
            with torch.no_grad():
                output.add_(1)
            torch.testing.assert_close(source.concat, before)


class TestElementwiseOps:

    @pytest.mark.parametrize("backend", [None, "aot_eager"])
    @pytest.mark.parametrize("derived", [False, True])
    @pytest.mark.parametrize(
        "shape,expected_shape",
        [
            ((0, 2), (0, 2)),
            ((-1,), (0, 0)),
            ((-1, 2), (0, 0, 2)),
            ((2, -1), (0, 2, 0)),
            ((0, -1), (0, 0)),
            ((-2,), None),
            ((-1, -1), None),
            ((-1, 3, -1), None),
            ((-1, 0), None),
            ((0, 0, -1), None),
        ],
    )
    def test_empty_reshape(self, device, backend, derived, shape, expected_shape):
        torch.compiler.reset()
        leaf = torch.randn(2, 3, device=device, dtype=torch.float64, requires_grad=True)
        source = NT([leaf])[:0] if derived else NT([], device=device, dtype=leaf.dtype)

        def reshape(value):
            return torch.ops.aten.reshape.default(value, shape)

        operation = reshape if backend is None else torch.compile(reshape, backend=backend, fullgraph=True)
        if expected_shape is None:
            with pytest.raises(RuntimeError):
                torch.empty(0, device=device).reshape(shape)
            with pytest.raises(RuntimeError):
                operation(source)
            return
        output = operation(source)
        assert isinstance(output, NT)
        assert output.shape == expected_shape
        torch.testing.assert_close(output.tensor, torch.empty(expected_shape, device=device, dtype=source.dtype))
        assert output.numel() == 0
        if derived:
            gradient = torch.autograd.grad(output.concat.sum(), leaf)[0]
            torch.testing.assert_close(gradient, torch.zeros_like(leaf))

    @pytest.mark.parametrize("ragged", [False, True])
    @pytest.mark.parametrize("operation", [torch.ops.aten.frexp.Tensor, torch.ops.aten.diag_embed.default])
    def test_empty_native_outputs(self, device, ragged, operation):
        values = torch.empty(0, 4, device=device)
        source = NT.from_packed(
            values,
            torch.tensor([0]),
            torch.empty(0, 2 if ragged else 1, dtype=torch.long),
            ragged_dims=(0,) if ragged else (),
            outer_size=(0, 5, 4) if ragged else (0, 4),
        )
        actual = operation(source)
        expected = operation(values)
        if isinstance(actual, torch.Tensor):
            actual, expected = (actual,), (expected,)
        for result, reference in zip(actual, expected):
            assert result.shape == (0, *((5,) if ragged else ()), *reference.shape[1:])
            assert result.ragged_dims == source.ragged_dims
            assert result.dtype == reference.dtype
            torch.testing.assert_close(result.concat, reference)

    def test_empty_native_error(self, device):
        values = torch.empty(0, 4, device=device, dtype=torch.int64)
        source = NT.from_packed(
            values, torch.tensor([0]), torch.empty(0, 1, dtype=torch.long), ragged_dims=(), outer_size=(0, 4)
        )
        operation = torch.ops.aten._fft_r2c.default
        with pytest.raises((RuntimeError, TypeError)) as error:
            operation(values, [1], 0, True)
        with pytest.raises(type(error.value)):
            operation(source, [1], 0, True)

    @pytest.mark.parametrize("inference", [False, True])
    @pytest.mark.parametrize(
        "operation,reference",
        [
            (lambda x: torch.ops.aten.squeeze.default(self=x), lambda x: x.squeeze()),
            (lambda x: torch.ops.aten.squeeze.dim(self=x, dim=2), lambda x: x.squeeze(1)),
            (lambda x: torch.ops.aten.unsqueeze.default(self=x, dim=2), lambda x: x.unsqueeze(1)),
            (lambda x: torch.ops.aten.transpose.int(self=x, dim0=1, dim1=3), lambda x: x.transpose(0, 2)),
            (lambda x: torch.ops.aten.permute.default(self=x, dims=[0, 3, 2, 1]), lambda x: x.permute(2, 1, 0)),
            (lambda x: torch.ops.aten.select.int(self=x, dim=1, index=-1), lambda x: x.select(0, -1)),
            (
                lambda x: torch.ops.aten.as_strided.default(x, x.shape, x.stride()),
                lambda x: x.as_strided(x.shape, x.stride()),
            ),
        ],
        ids=["default-squeeze", "squeeze", "unsqueeze", "transpose", "permute", "select", "as-strided"],
    )
    def test_native_views(self, device, inference, operation, reference):
        parts = [torch.randn(n, 1, 3, device=device, dtype=torch.float64, requires_grad=True) for n in (2, 4)]
        references = [part.detach().clone().requires_grad_() for part in parts]
        source = NT(parts, ragged_dims=(0,))
        with torch.inference_mode(inference):
            output = operation(source)
            expected = [reference(part) for part in references]
            torch.testing.assert_close(output.unbind(), expected)
        if not inference:
            actual_gradients = torch.autograd.grad(output.square().sum(), parts)
            expected_gradients = torch.autograd.grad(sum(part.square().sum() for part in expected), references)
            torch.testing.assert_close(actual_gradients, expected_gradients)

    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize("batch_first", [False, True])
    @pytest.mark.parametrize("mode", ["eager", "inference"])
    def test_scalar_unsqueeze(self, device, dim, batch_first, mode):
        torch.compiler.reset()
        parts = [torch.randn((), device=device, dtype=torch.float64, requires_grad=True) for _ in range(2)]
        references = [part.detach().clone().requires_grad_() for part in parts]
        source = NT(parts, batch_first=batch_first)
        dense = torch.stack(references)

        def unsqueeze(value):
            return torch.ops.aten.unsqueeze.default(value, dim)

        operation = unsqueeze if mode != "compiled" else torch.compile(unsqueeze, backend="aot_eager", fullgraph=True)
        with torch.inference_mode(mode == "inference"):
            actual = operation(source)
            expected = dense.unsqueeze(dim)
            assert actual.shape == expected.shape
            assert actual.batch_first is (dim == 1)
            assert actual.element_sizes().tolist() == [[1], [1]]
            torch.testing.assert_close(actual.tensor, expected)
        if mode != "inference":
            torch.testing.assert_close(
                torch.autograd.grad(actual.square().sum(), parts),
                torch.autograd.grad(expected.square().sum(), references),
            )

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("lengths", [(2, 3), (2, 4), (17, 34)])
    def test_reshape(self, device, batch_first, lengths):
        parts = [torch.randn(length, 4, device=device, requires_grad=True) for length in lengths]
        source = NT([part.reshape(1, -1) for part in parts], ragged_dims=(1,), batch_first=batch_first)
        shape = (-1, 4)
        mixed = list(shape)
        for output in (
            source.reshape(shape),
            source.reshape(mixed),
            torch.ops.aten.reshape.default(self=source, shape=shape),
            torch.ops.aten.view.default(self=source, size=shape),
        ):
            torch.testing.assert_close(output.unbind(0 if batch_first else 1), tuple(parts))
            actual = torch.autograd.grad(output.square().sum(), parts, retain_graph=True)
            expected = torch.autograd.grad(sum(part.square().sum() for part in parts), parts)
            torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("strided", [False, True])
    def test_static_view(self, device, strided):
        parts = [
            torch.randn(length, 12 if strided else 6, device=device, dtype=torch.float64, requires_grad=True)
            for length in (2, 5)
        ]
        references = [part.detach().clone().requires_grad_() for part in parts]
        source = NT(parts, ragged_dims=(0,))
        if strided:
            source = source.packed_with_static_tail(source.concat[:, 1::2])
            elements = [part[:, 1::2] for part in references]
        else:
            elements = references
        expected = tuple(part.view(part.shape[0], 2, 3) for part in elements)
        target = (*source.shape[:-1], 2, 3)
        output = torch.ops.aten.view.default(source, target)
        reshaped = torch.ops.aten.reshape.default(source, target)
        collapsed = torch.ops.aten.view.default(output, source.shape)
        torch.testing.assert_close(output.unbind(), expected)
        torch.testing.assert_close(reshaped.unbind(), expected)
        torch.testing.assert_close(collapsed.unbind(), tuple(elements))
        actual = torch.autograd.grad(output.concat.sin().sum(), parts, create_graph=True)
        reference = torch.autograd.grad(sum(part.sin().sum() for part in expected), references, create_graph=True)
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(
            torch.autograd.grad(sum(part.sum() for part in actual), parts),
            torch.autograd.grad(sum(part.sum() for part in reference), references),
        )
        before = source.concat.detach().clone()
        with torch.no_grad():
            output.concat.add_(1)
            torch.testing.assert_close(source.concat, before + 1)
            source.concat.mul_(2)
            torch.testing.assert_close(output.concat, (2 * (before + 1)).view(-1, 2, 3))

    def test_batch_view(self, device):
        parts = [torch.randn(length, 3, device=device, dtype=torch.float64, requires_grad=True) for length in (2, 5)]
        references = [part.detach().clone().requires_grad_() for part in parts]
        output = torch.ops.aten.transpose.int(NT(parts, ragged_dims=(0,)).sin(), 0, 1)
        expected = [part.sin() for part in references]
        assert output.batch_first is False
        assert output.shape == (5, 2, 3)
        torch.testing.assert_close(output.unbind(1), expected)
        probes = [torch.randn_like(part) for part in expected]
        actual_gradients = torch.autograd.grad(
            sum((part * probe).sum() for part, probe in zip(output.unbind(1), probes)), parts
        )
        reference_gradients = torch.autograd.grad(
            sum((part * probe).sum() for part, probe in zip(expected, probes)), references
        )
        torch.testing.assert_close(actual_gradients, reference_gradients)

    @pytest.mark.parametrize("width", [1, 4], ids=["identity", "static-expansion"])
    def test_native_expand_preserves_logical_ragged_axes(self, device, width):
        leaves = [torch.randn(1, length, device=device, dtype=torch.float64, requires_grad=True) for length in (2, 3)]
        source = NT(leaves, ragged_dims=(1,))

        output = torch.ops.aten.expand.default(source, [2, width, 3])

        assert output.ragged_dims == (1,)
        for actual, reference in zip(output, leaves):
            assert_close(actual, reference.expand(width, -1))
        gradients = torch.autograd.grad(output.concat.sum(), leaves)
        for actual, reference in zip(gradients, leaves):
            assert_close(actual, torch.full_like(reference, width))
        direction = NT([torch.randn_like(leaf) for leaf in leaves], ragged_dims=(1,))
        for primal in (source, source.detach()):
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(primal, direction)
                _, tangent = torch.autograd.forward_ad.unpack_dual(torch.ops.aten.expand.default(dual, [2, width, 3]))
                assert tangent is not None
                for actual, reference in zip(tangent, direction):
                    assert_close(actual, reference.expand(width, -1))
        with torch.no_grad():
            source.concat.add_(1)
            for actual, reference in zip(output, source):
                assert_close(actual, reference.expand(width, -1))

    @pytest.mark.parametrize(
        "shape,batch_first", [((), True), ((), False), ((3,), False), ((2, 3), True), ((2, 3), False)]
    )
    def test_native_stack_mixes_dense_and_nested_rectangular_batches(self, device, shape, batch_first):
        leaves = [torch.randn(shape, dtype=torch.float64, device=device, requires_grad=True) for _ in range(2)]
        source = NT(leaves, ragged_dims=(), batch_first=batch_first)
        elements = leaves
        if len(shape) == 2:
            source = source.transpose(1 if batch_first else 0, 2)
            elements = [leaf.t() for leaf in leaves]
        reference = torch.stack(elements, dim=0 if batch_first or not shape else 1)
        dense = torch.randn_like(reference, requires_grad=True)

        output = torch.ops.aten.stack.default([dense, source, source], 1)
        expected = torch.stack([dense, reference, reference], 1)

        assert not isinstance(output, NT)
        assert_close(output, expected)
        cotangent = torch.randn_like(expected)
        actual_gradients = torch.autograd.grad(output, (dense, *leaves), cotangent)
        expected_gradients = torch.autograd.grad(expected, (dense, *leaves), cotangent)
        for actual, reference in zip(actual_gradients, expected_gradients):
            assert_close(actual, reference)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
    @pytest.mark.parametrize("pin_memory", [False, True])
    def test_zeros_like_follows_requested_pinning(self, pin_memory):
        source = NT([torch.randn(2, 3), torch.randn(4, 3)], pin_memory=True)
        expected = torch.zeros_like(source.concat, pin_memory=pin_memory)

        output = torch.zeros_like(source, pin_memory=pin_memory)

        assert output.concat.is_pinned() is expected.is_pinned()

    def test_creation_like_rejects_sparse_result(self):
        source = NT([torch.randn(2, 3), torch.randn(4, 3)])

        with pytest.raises(TypeError, match="dense Tensor with torch.strided layout"):
            torch.zeros_like(source, layout=torch.sparse_coo)

    def test_binary_dense_same_shape(self):
        nt = NT([torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])])
        dense = nt.tensor + 0.5
        nested_dense = nt.nested_like(dense, strict=False)

        pow_out = torch.ops.aten.pow.Tensor_Tensor(nt, dense)
        pow_rev_out = torch.ops.aten.pow.Tensor_Tensor(dense, nt)
        and_out = torch.ops.aten.bitwise_and.Tensor(nt.int(), dense.int())

        pow_ref = NT([torch.pow(a, b) for a, b in zip(nt, nested_dense)], **reference_options(nt))
        pow_rev_ref = NT([torch.pow(b, a) for a, b in zip(nt, nested_dense)], **reference_options(nt))
        and_ref = NT([torch.bitwise_and(a.int(), b.int()) for a, b in zip(nt, nested_dense)], **reference_options(nt))
        assert_close(pow_out, pow_ref)
        assert_close(pow_rev_out, pow_rev_ref)
        assert_close(and_out, and_ref)

    @pytest.mark.parametrize("shape,packed", [((2, 3), False), ((3,), True)])
    def test_rectangular_dense_operands_use_logical_then_packed_axes(self, device, shape, packed):
        leaves = [torch.randn(shape, dtype=torch.float64, device=device, requires_grad=True) for _ in range(2)]
        source = NT(leaves, ragged_dims=(), batch_first=False)
        reference = torch.stack(leaves, dim=1)
        dense = torch.randn_like(source.concat if packed else reference, requires_grad=True)

        output = source * dense
        expected = reference * (dense.movedim(0, 1) if packed else dense)

        assert_close(output.tensor, expected)
        cotangent = torch.randn_like(expected)
        actual_gradients = torch.autograd.grad(output.tensor, (*leaves, dense), cotangent)
        expected_gradients = torch.autograd.grad(expected, (*leaves, dense), cotangent)
        for actual, reference in zip(actual_gradients, expected_gradients):
            assert_close(actual, reference)

    @pytest.mark.parametrize("op", [torch.mul, torch.add])
    def test_binary_op_roundtrip(self, op):
        nt = NT([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])])
        scalar = 2.0
        result = op(nt, scalar)
        for r, t in zip(result, nt):
            assert_close(r, op(t, scalar))

    @pytest.mark.parametrize("operand", [torch.tensor([[2.0], [4.0]]), torch.tensor([[[2.0]], [[4.0]]])])
    def test_binary_op_batchwise_dense_operand(self, operand):
        nt = NT(
            [
                torch.tensor([[2.0, 4.0], [6.0, 8.0]]),
                torch.tensor([[12.0, 16.0], [20.0, 24.0], [28.0, 32.0]]),
            ]
        )
        result = nt / operand
        reference = NT([tensor / operand[index] for index, tensor in enumerate(nt)], **reference_options(nt))
        assert_close(result, reference)

    def test_binary_op_batchwise_dense_operand_with_static_head_dim(self):
        nt = NT(
            [
                torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3),
                torch.arange(3 * 2 * 3, dtype=torch.float32).reshape(3, 2, 3),
            ],
            batch_first=True,
        )
        operand = torch.randn(2, 2, 3)
        result = nt * operand
        reference = NT([tensor * operand[index] for index, tensor in enumerate(nt)], **reference_options(nt))
        assert_close(result, reference)

    def test_dense_operand_can_target_static_tail_or_batch(self):
        nt = NT(
            [
                torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3),
                torch.arange(3 * 2 * 3, dtype=torch.float32).reshape(3, 2, 3),
            ],
            batch_first=True,
        )
        operand = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])

        tail_aligned = nt * operand
        batchwise = operand.reshape(2, 1, 1, operand.shape[-1])
        batch_aligned = nt * batchwise

        for index, element in enumerate(nt):
            assert_close(tail_aligned[index], element * operand)
            assert_close(batch_aligned[index], element * operand[index])

    @pytest.mark.parametrize("reverse", [False, True])
    def test_binary_op_dense_padded_shape_trailing_broadcast(self, reverse):
        # A mask-like NT with a singleton channel dim against a dense operand carrying the
        # full channel dim: padded shapes are broadcast-compatible ([B, S, 1] vs [B, S, H])
        # and must align on the ragged sequence dim while broadcasting the trailing dim.
        nt = NT([torch.tensor([[1.0], [0.0]]), torch.tensor([[1.0], [1.0], [0.0]])], batch_first=True)  # [2, S, 1]
        dense = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)  # [2, Smax=3, H=4]
        result = torch.ops.aten.mul.Tensor(dense, nt) if reverse else torch.ops.aten.mul.Tensor(nt, dense)
        reference = NT([dense[i][: len(elem)] * elem for i, elem in enumerate(nt)], **reference_options(nt))
        assert tuple(result[0].shape) == (2, 4)
        assert_close(result, reference)

    @pytest.mark.parametrize("op", [torch.abs, torch.neg, torch.exp, torch.sign])
    def test_unary_op_roundtrip(self, op):
        nt = NT([torch.tensor([1.0, -2.0, 3.0]), torch.tensor([-4.0, 5.0])])
        result = op(nt)
        for r, t in zip(result, nt):
            assert_close(r, op(t))


class TestBatchFlatten:

    @pytest.mark.parametrize("pair", [False, True])
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize(
        "operation",
        [
            lambda x: x.flatten(0, 1),
            lambda x: torch.flatten(x, 0, 1),
            lambda x: torch.ops.aten.flatten.using_ints(self=x, start_dim=0, end_dim=1),
        ],
        ids=["method", "torch", "native"],
    )
    def test_values(self, device, pair, batch_first, operation):
        parts = [
            torch.randn(2, n, *((n,) if pair else ()), 3, device=device, dtype=torch.float64, requires_grad=True)
            for n in (2, 5)
        ]
        source = NT(parts, ragged_dims=(1, 2) if pair else (1,), batch_first=batch_first)
        expected = (
            [part[index] for part in parts for index in range(2)]
            if batch_first
            else [part[index] for index in range(2) for part in parts]
        )
        output = operation(source)
        assert isinstance(output, NT) and output.batch_first
        assert output.ragged_dims == ((0, 1) if pair else (0,))
        torch.testing.assert_close(output.unbind(), expected)
        torch.testing.assert_close(torch.autograd.grad(output.square().sum(), parts), [2 * part for part in parts])
        with torch.inference_mode():
            torch.testing.assert_close(operation(source).unbind(), expected)
        dim = 0 if batch_first else 1
        assert torch.flatten(source, dim, dim) is source
        assert torch.ops.aten.flatten.using_ints(source, dim, dim) is source

    @pytest.mark.parametrize("pair", [False, True])
    @pytest.mark.parametrize("static", [False, True])
    def test_roundtrip(self, device, pair, static):
        fake = pytest.importorskip("torch._subclasses.fake_tensor")

        def fold(source):
            return source.flatten(0, 1).unflatten(0, (2, 2))

        compiled = torch.compile(fold, backend="aot_eager", fullgraph=True, dynamic=True)
        ragged = (1, 2) if pair else (1,)
        if not static:
            ragged = (0, *ragged)
        for lengths in ((2, 5), (4, 1)):
            parts = [
                torch.randn(2, n, *((n,) if pair else ()), 3, device=device, dtype=torch.float64, requires_grad=True)
                for n in lengths
            ]
            source = NT(parts, ragged_dims=ragged)
            actual = compiled(source)
            torch.testing.assert_close(actual.unbind(), parts)
            torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), parts), [2 * part for part in parts])
        with fake.FakeTensorMode() as mode:
            actual = fold(mode.from_tensor(source))
        assert isinstance(actual, NT) and fake.is_fake(actual.concat)
        assert actual.shape == source.shape


class TestTo:

    @pytest.mark.parametrize(
        "mode", [torch.enable_grad, torch.no_grad, torch.inference_mode], ids=["grad", "no-grad", "inference"]
    )
    @pytest.mark.parametrize(
        ("method", "dtype", "target"),
        [
            pytest.param("bool", torch.bool, torch.bool, id="bool-identity"),
            pytest.param("to", torch.bool, torch.bool, id="to-identity"),
            pytest.param("bool", torch.bfloat16, torch.bool, id="bool"),
            pytest.param("float", torch.bfloat16, torch.float32, id="float"),
            pytest.param("to", torch.bfloat16, torch.float32, id="to"),
            pytest.param("aten", torch.bfloat16, torch.float32, id="aten"),
        ],
    )
    def test_dtype(self, device, mode, method, dtype, target):
        with mode():
            parts = [
                torch.arange(length * length, device=device).reshape(length, length).to(dtype) for length in (2, 3)
            ]
            if dtype.is_floating_point and torch.is_grad_enabled():
                parts = [part.requires_grad_() for part in parts]
            source = NT(parts, ragged_dims=(0, 1)).transpose(1, 2)
            if method == "aten":
                result = torch.ops.aten.to.dtype(source, target)
            elif method == "to":
                result = source.to(target)
            else:
                result = getattr(source, method)()
            expected = [part.t().to(target) for part in parts]
            torch.testing.assert_close(result.unbind(), expected)
            assert result.ragged_dims == source.ragged_dims
            assert result.packed_dim_order == source.packed_dim_order
            torch.testing.assert_close(result.element_sizes(), source.element_sizes())
            if dtype == target:
                assert result is source
            if target.is_floating_point and torch.is_grad_enabled():
                probes = [torch.randn_like(value) for value in expected]
                actual_gradients = torch.autograd.grad(
                    sum((value * probe).sum() for value, probe in zip(result, probes)), parts
                )
                expected_gradients = torch.autograd.grad(
                    sum((value * probe).sum() for value, probe in zip(expected, probes)), parts
                )
                torch.testing.assert_close(actual_gradients, expected_gradients)


class TestBitwiseAliases:

    @pytest.mark.parametrize(
        "mode", [torch.enable_grad, torch.no_grad, torch.inference_mode], ids=["grad", "no-grad", "inference"]
    )
    @pytest.mark.parametrize(
        ("operation", "alias", "dtype"),
        [
            pytest.param(operator.and_, torch.ops.aten.__and__, torch.bool, id="and"),
            pytest.param(operator.or_, torch.ops.aten.__or__, torch.bool, id="or"),
            pytest.param(operator.xor, torch.ops.aten.__xor__, torch.bool, id="xor"),
            pytest.param(operator.lshift, torch.ops.aten.__lshift__, torch.int64, id="left-shift"),
            pytest.param(operator.rshift, torch.ops.aten.__rshift__, torch.int64, id="right-shift"),
        ],
    )
    def test_native(self, device, mode, operation, alias, dtype):
        with mode():
            lengths = (3, 5)
            parts = [(torch.arange(length, device=device) % 3).to(dtype).unsqueeze(0) for length in lengths]
            source = NT(parts, ragged_dims=(1,))
            dense = torch.tensor([0, 2], device=device, dtype=dtype).reshape(2, 1, 1)
            sizes = torch.tensor([[1, length] for length in lengths])

            def check(actual, expected):
                torch.testing.assert_close(actual.unbind(), expected)
                assert actual.shape == (2, 1, max(lengths))
                assert actual.ragged_dims == (1,)
                torch.testing.assert_close(actual.element_sizes(), sizes)

            for function in (operation, alias.Tensor):
                check(function(dense, source), [operation(dense[i], part) for i, part in enumerate(parts)])
                check(function(source, dense), [operation(part, dense[i]) for i, part in enumerate(parts)])
            scalar = True if dtype == torch.bool else 1
            check(operation(source, scalar), [operation(part, scalar) for part in parts])
            check(operation(scalar, source), [operation(scalar, part) for part in parts])
            check(alias.Scalar(source, scalar), [operation(part, scalar) for part in parts])
            torch.testing.assert_close(source.unbind(), parts)
            torch.testing.assert_close(source.element_sizes(), sizes)
            assert source.shape == (2, 1, max(lengths))
            assert source.ragged_dims == (1,)


class TestInPlace:

    @pytest.mark.parametrize("axis", [0, 1])
    def test_broadcast(self, device, axis):
        operation = torch.ops.aten.add_.Tensor
        shapes = ((2, 3), (4, 5))
        parts = [torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
        operands = [
            torch.randn(
                *(1 if dim == axis else size for dim, size in enumerate(shape)),
                device=device,
                dtype=torch.float64,
                requires_grad=True,
            )
            for shape in shapes
        ]
        references = [part.detach().clone().requires_grad_() for part in parts]
        reference_operands = [part.detach().clone().requires_grad_() for part in operands]
        output = NT(parts, ragged_dims=(0, 1)).sin()
        other = NT(operands, ragged_dims=(1 - axis,)).cos()
        alias = output.detach()
        _ = alias.tensor

        assert operation(output, other) is output

        expected = [operation(part.sin(), operand.cos()) for part, operand in zip(references, reference_operands)]
        torch.testing.assert_close(output.unbind(), expected)
        torch.testing.assert_close(alias.tensor, output.tensor)
        probes = [torch.randn_like(part) for part in expected]
        torch.testing.assert_close(
            torch.autograd.grad(sum((part * probe).sum() for part, probe in zip(output, probes)), (*parts, *operands)),
            torch.autograd.grad(
                sum((part * probe).sum() for part, probe in zip(expected, probes)), (*references, *reference_operands)
            ),
        )

    def test_shape(self):
        source = NT([torch.randn(length, 1) for length in (2, 4)], ragged_dims=(0,))
        other = NT([torch.randn(length, 3) for length in (2, 4)], ragged_dims=(0,))
        expected = [part.clone() for part in source]
        with pytest.raises(RuntimeError, match="shape|broadcast"):
            torch.ops.aten.add_.Tensor(source, other)
        torch.testing.assert_close(source.unbind(), expected)


class TestIndexingOps:

    @pytest.mark.parametrize("name", ["gather", "scatter"])
    @pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64, torch.float32])
    def test_index_dtype(self, device, name, index_dtype):
        parts = [torch.randn(shape, device=device, requires_grad=True) for shape in ((2, 3, 4), (3, 2, 4))]
        indices = [torch.zeros(shape, device=device, dtype=index_dtype) for shape in ((1, 2, 3), (2, 1, 3))]
        source = NT(parts, ragged_dims=(0, 1))
        index = NT(indices, ragged_dims=(0, 1))
        operation = getattr(torch, name)
        rest = () if name == "gather" else (1.0,)
        if index_dtype.is_floating_point:
            with pytest.raises(RuntimeError):
                operation(parts[0], 2, indices[0], *rest)
            with pytest.raises(RuntimeError):
                operation(source, 3, index, *rest)
            return
        output = operation(source, 3, index, *rest)
        expected = [operation(part, 2, idx, *rest) for part, idx in zip(parts, indices)]
        torch.testing.assert_close(output.unbind(), expected)
        torch.testing.assert_close(
            torch.autograd.grad(output.concat.square().sum(), parts),
            torch.autograd.grad(sum(part.square().sum() for part in expected), parts),
        )

    @pytest.mark.parametrize("operation", ["slice", "select"])
    def test_view_backward(self, device, operation):
        parts = [torch.randn(n, 2, device=device, dtype=torch.float64) for n in (2, 4)]
        if operation == "slice":
            source = NT(parts, ragged_dims=(0,))
            actual = torch.ops.aten.slice_backward.default(source, (2, 4, 5), 2, 1, 5, 2)
            expected = [torch.ops.aten.slice_backward.default(part, (part.size(0), 5), 1, 1, 5, 2) for part in parts]
        else:
            parts = [part[:, 0] for part in parts]
            source = NT(parts, ragged_dims=(0,))
            actual = torch.ops.aten.select_backward.default(source, (2, 4, 5), 2, 3)
            expected = [torch.ops.aten.select_backward.default(part, (part.size(0), 5), 1, 3) for part in parts]
        for result, reference in zip(actual, expected):
            assert_close(result, reference)

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("ragged_dims,axis", [((0,), 0), ((0, 1), 1), ((1, 0), 0), ((0, 1), 2)])
    @pytest.mark.parametrize(
        "name,inplace",
        [("gather", False), ("scatter", False), ("scatter", True), ("scatter_add", False), ("scatter_reduce", False)],
    )
    def test_batched_index(self, device, batch_first, ragged_dims, axis, name, inplace):
        shapes = ((2, 3, 4), (3, 3 if len(ragged_dims) == 1 else 5, 4))
        parts = [torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
        references = [part.detach().clone().requires_grad_() for part in parts]
        shape = [2, 2, 2, 4]
        shape[axis + 1] = 1
        index = torch.arange(math.prod(shape), device=device).reshape(shape) % min(part.shape[axis] for part in parts)
        values = torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True)
        ref_values = values.detach().clone().requires_grad_()
        if not batch_first:
            index = index.movedim(0, 1)
            values = values.movedim(0, 1)
            ref_values = ref_values.movedim(0, 1)
        source = NT(parts, ragged_dims=ragged_dims, batch_first=batch_first).sin()
        dim = axis + (batch_first or axis > 0)
        kwargs = {"reduce": "sum", "include_self": False} if name == "scatter_reduce" else {}
        native = {
            "gather": torch.ops.aten.gather.default,
            "scatter": torch.ops.aten.scatter.src,
            "scatter_add": torch.ops.aten.scatter_add.default,
            "scatter_reduce": torch.ops.aten.scatter_reduce.two,
        }[name]
        if inplace:
            output = source.scatter_(dim, index, values)
            assert output is source
        else:
            output = native(source, dim, index, *(() if name == "gather" else (values,)), **kwargs)
        expected = [
            getattr(torch, name)(
                part.sin(),
                axis,
                index.select(0 if batch_first else 1, i),
                *(() if name == "gather" else (ref_values.select(0 if batch_first else 1, i),)),
                **kwargs,
            )
            for i, part in enumerate(references)
        ]
        actual = output.unbind(0 if batch_first else 1)
        torch.testing.assert_close(actual, expected)
        leaves = parts if name == "gather" else [*parts, values]
        ref_leaves = references if name == "gather" else [*references, ref_values]
        torch.testing.assert_close(
            torch.autograd.grad(sum(p.square().sum() for p in actual), leaves),
            torch.autograd.grad(sum(p.square().sum() for p in expected), ref_leaves),
        )

    @pytest.mark.parametrize("name", ["gather", "scatter", "scatter_add", "scatter_reduce"])
    @pytest.mark.parametrize("fullgraph", [False, True])
    def test_batched_compile(self, device, name, fullgraph):
        torch.compiler.reset()
        parts = [torch.randn(m, n, 3, device=device, requires_grad=True) for m, n in ((2, 3), (3, 5))]
        index = torch.tensor([0, 1], device=device).reshape(2, 1, 1, 1).expand(2, 2, 1, 3)
        values = torch.randn(index.shape, device=device, requires_grad=True)
        kwargs = {"reduce": "sum"} if name == "scatter_reduce" else {}

        def operation(source, index, values):
            return getattr(torch, name)(source, 2, index, *(() if name == "gather" else (values,)), **kwargs)

        try:
            compiled = torch.compile(operation, backend="aot_eager", fullgraph=fullgraph, dynamic=True)
            output = compiled(NT(parts, ragged_dims=(0, 1)), index, values)
            expected = [
                getattr(torch, name)(part, 1, index[i], *(() if name == "gather" else (values[i],)), **kwargs)
                for i, part in enumerate(parts)
            ]
            actual = output.unbind(0)
            torch.testing.assert_close(actual, expected)
            leaves = parts if name == "gather" else [*parts, values]
            torch.testing.assert_close(
                torch.autograd.grad(sum(p.square().sum() for p in actual), leaves),
                torch.autograd.grad(sum(p.square().sum() for p in expected), leaves),
            )
        finally:
            torch.compiler.reset()

    @pytest.mark.parametrize("nested,dense", [(False, False), (True, False), (True, True)])
    @pytest.mark.parametrize("source_batch", [1, 2])
    def test_batched_prefix(self, device, nested, dense, source_batch):
        parts = [torch.randn(2, 3, 4, device=device, requires_grad=True) for _ in range(2)]
        values = torch.randn(source_batch, 2, 1, 4, device=device, requires_grad=True)
        source = torch.stack(parts) if dense else NT(parts, ragged_dims=(0, 1))
        src = NT(values.unbind(), ragged_dims=(0, 1)) if nested else values
        index = torch.zeros(1, 2, 1, 4, device=device, dtype=torch.long)
        actual = torch.scatter(source, 2, index, src)
        expected = torch.scatter(torch.stack(parts), 2, index, values)
        torch.testing.assert_close(actual.unbind(), expected.unbind())
        reference = [parts[0].scatter(1, index[0], values[0]), parts[1]]
        torch.testing.assert_close(
            torch.autograd.grad(actual.square().sum(), [*parts, values]),
            torch.autograd.grad(sum(part.square().sum() for part in reference), [*parts, values]),
        )

    @pytest.mark.parametrize("shape", [(1, 2, 1, 4), (2, 1, 1, 4)])
    def test_batched_source(self, device, shape):
        dense = torch.zeros(2, 2, 3, 4, device=device)
        source = NT(dense.unbind(), ragged_dims=(0, 1))
        index = torch.zeros(2, 2, 1, 4, device=device, dtype=torch.long)
        values = torch.zeros(shape, device=device)
        for target in (dense, source):
            with pytest.raises((ValueError, RuntimeError)):
                torch.scatter(target, 2, index, values)

    @pytest.mark.parametrize("name", ["gather", "scatter"])
    @pytest.mark.parametrize("bad", [-1, 3])
    def test_batched_bounds(self, device, name, bad):
        source = NT([torch.zeros(m, n, 4, device=device) for m, n in ((2, 3), (3, 5))], ragged_dims=(0, 1))
        index = torch.full((2, 2, 1, 4), bad, device=device, dtype=torch.long)
        with pytest.raises((IndexError, RuntimeError), match="bounds"):
            getattr(torch, name)(source, 2, index, *(() if name == "gather" else (1.0,)))

    @pytest.mark.parametrize("operation", [torch.gather, torch.Tensor.gather, torch.ops.aten.gather.default])
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("element_dim", [0, 1])
    def test_sparse_gather(self, device, operation, batch_first, element_dim):
        base = torch.randn(2, 4, 3, device=device)
        source = (base if batch_first else base.movedim(0, 1)).detach().requires_grad_()
        references = [part.clone().requires_grad_() for part in base.unbind()]
        shapes = [(2, 3), (3, 3)] if element_dim == 0 else [(2, 2), (3, 2)]
        indices = [torch.arange(math.prod(shape), device=device).reshape(shape) % (4 - element_dim) for shape in shapes]
        index = NT(indices, ragged_dims=(0,), batch_first=batch_first)
        dim = element_dim + (batch_first or element_dim > 0)
        output = operation(source, dim, index, sparse_grad=True)
        expected = [torch.gather(part, element_dim, idx, sparse_grad=True) for part, idx in zip(references, indices)]
        torch.testing.assert_close(output.unbind(0 if batch_first else 1), expected)
        gradient = torch.autograd.grad(output.square().sum(), source)[0]
        expected_gradients = torch.autograd.grad(sum(part.square().sum() for part in expected), references)
        assert gradient.layout == torch.sparse_coo
        expected_gradient = torch.stack([grad.to_dense() for grad in expected_gradients])
        torch.testing.assert_close(
            gradient.to_dense(), expected_gradient if batch_first else expected_gradient.movedim(0, 1)
        )

    @pytest.mark.parametrize("dense", [True, False])
    @pytest.mark.parametrize("overlap", [True, False])
    def test_index_overlap(self, device, dense, overlap):
        base = torch.tensor([0, 1, 0, 1, 2], device=device)
        before = base.clone()
        offset = 2 if overlap else 3
        source = (
            base[:3].reshape(1, 3)
            if dense
            else NT.from_packed(base[:3], torch.tensor([0, 3]), torch.tensor([[3]]), ragged_dims=(0,))
        )
        index = NT.from_packed(base[offset : offset + 2], torch.tensor([0, 2]), torch.tensor([[2]]), ragged_dims=(0,))
        if overlap:
            with pytest.raises(RuntimeError, match="overlapping memory|single memory location"):
                source.scatter_(1, index, 7)
            torch.testing.assert_close(base, before)
        else:
            source.scatter_(1, index, 7)
            before[:3].scatter_(0, before[3:], 7)
            torch.testing.assert_close(base, before)

    def test_native_keywords(self, device):
        parts = [torch.arange(length, dtype=torch.float32, device=device) for length in (3, 5)]
        indices = [torch.tensor([2, 0], device=device), torch.tensor([4], device=device)]
        values = [torch.full(index.shape, -2.0, device=device) for index in indices]
        source, index, src = NT(parts), NT(indices), NT(values)
        for operation, kwargs, expected in (
            (torch.ops.aten.gather.default, {}, [part.gather(0, idx) for part, idx in zip(parts, indices)]),
            (
                torch.ops.aten.scatter.value,
                {"value": -2.0},
                [part.scatter(0, idx, -2.0) for part, idx in zip(parts, indices)],
            ),
            (
                torch.ops.aten.scatter.src,
                {"src": src},
                [part.scatter(0, idx, value) for part, idx, value in zip(parts, indices, values)],
            ),
        ):
            output = operation(self=source, dim=1, index=index, **kwargs)
            torch.testing.assert_close(output.unbind(), expected)

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("element_dim", [0, 1])
    @pytest.mark.parametrize("operation", [torch.gather, torch.ops.aten.gather.default])
    def test_dense_gather(self, device, batch_first, element_dim, operation):
        source = torch.randn(2, 4, 3, device=device, requires_grad=True)
        shapes = [(2, 3), (3, 3)] if element_dim == 0 else [(2, 2), (3, 2)]
        indices = [torch.arange(math.prod(shape), device=device).reshape(shape) % (4 - element_dim) for shape in shapes]
        index = NT(indices, ragged_dims=(0,), batch_first=batch_first)
        logical_source = source if batch_first else source.movedim(0, 1)
        dim = element_dim + (batch_first or element_dim > 0)

        output = operation(logical_source, dim, index)
        expected = [torch.gather(part, element_dim, idx) for part, idx in zip(source, indices)]
        torch.testing.assert_close(output.unbind(0 if batch_first else 1), expected)
        actual_grad = torch.autograd.grad(output.square().sum(), source, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(sum(part.square().sum() for part in expected), source)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

    @pytest.mark.parametrize("dense", [True, False])
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize(
        "name,inplace",
        [
            ("scatter", False),
            ("scatter", True),
            ("scatter_add", False),
            ("scatter_add", True),
            ("scatter_reduce", False),
            ("scatter_reduce", True),
        ],
    )
    def test_scatter_topology(self, device, dense, batch_first, name, inplace):
        lengths = (9, 9) if dense else (4, 9)
        parts = [torch.randn(length, 3, device=device, requires_grad=True) for length in lengths]
        references = [part.detach().clone().requires_grad_() for part in parts]
        indices = [
            torch.tensor([0], device=device)[:, None].expand(-1, 3),
            torch.tensor([1, 8], device=device)[:, None].expand(-1, 3),
        ]
        values = [torch.randn(index.shape, device=device, requires_grad=True) for index in indices]
        ref_values = [value.detach().clone().requires_grad_() for value in values]
        source = torch.stack(parts) if dense else NT(parts, ragged_dims=(0,), batch_first=batch_first)
        if dense and not batch_first:
            source = source.movedim(0, 1)
        source = source.sin()
        index = NT(indices, ragged_dims=(0,), batch_first=batch_first)
        src = NT(values, ragged_dims=(0,), batch_first=batch_first)
        dim = 1 if batch_first else 0
        kwargs = {"reduce": "sum", "include_self": False} if name == "scatter_reduce" else {}
        operation = getattr(torch.Tensor, name + "_") if inplace else getattr(torch, name)

        output = operation(source, dim, index, src, **kwargs)
        expected = [
            getattr(torch, name)(part.sin(), 0, idx, value, **kwargs)
            for part, idx, value in zip(references, indices, ref_values)
        ]
        torch.testing.assert_close(output.unbind(0 if batch_first else 1), expected)
        if inplace:
            assert output is source
        actual_grad = torch.autograd.grad(output.square().sum(), [*parts, *values])
        expected_grad = torch.autograd.grad(sum(part.square().sum() for part in expected), [*references, *ref_values])
        torch.testing.assert_close(actual_grad, expected_grad)

    @pytest.mark.parametrize("dense", [True, False])
    def test_scalar_scatter(self, device, dense):
        parts = [torch.zeros(length, dtype=torch.long, device=device) for length in ((9, 9) if dense else (4, 9))]
        source = torch.stack(parts) if dense else NT(parts)
        indices = [torch.tensor([0], device=device), torch.tensor([1, 8], device=device)]
        output = source.scatter_(1, NT(indices), 1)
        assert output is source
        torch.testing.assert_close(output.unbind(), [part.scatter(0, idx, 1) for part, idx in zip(parts, indices)])

    @pytest.mark.parametrize("dense", [True, False])
    def test_scatter_mutation(self, device, dense):
        parts = [torch.randn(length, device=device, requires_grad=True) for length in ((9, 9) if dense else (4, 9))]
        source = torch.stack(parts) if dense else NT(parts)
        index = NT([torch.tensor([0], device=device), torch.tensor([1, 8], device=device)])
        saved = source.square().sum()
        source.scatter_(1, index, 1)
        with pytest.raises(RuntimeError, match="modified"):
            torch.autograd.grad(saved, parts)
        leaf = (torch.zeros(2, 9, device=device) if dense else NT([part.detach() for part in parts])).requires_grad_()
        with pytest.raises(RuntimeError, match="leaf Variable"):
            leaf.scatter_(1, index, 1)
        with pytest.raises(RuntimeError, match="overlapping memory|single memory location"):
            source.scatter_(1, index, source)

    @pytest.mark.parametrize("operation", [torch.select, torch.Tensor.select, torch.ops.aten.select.int])
    @pytest.mark.parametrize("mode", ["grad", "inference", "leaf"])
    @pytest.mark.parametrize("dim,index", [(-3, 0), (-2, -1), (-1, 1)])
    def test_select(self, device, operation, mode, dim, index):
        parts = [torch.randn(m, n, 3, dtype=torch.float64, device=device) for m, n in ((1, 2), (2, 4))]
        references = [part.clone().requires_grad_() for part in parts]
        if mode == "grad":
            parts = [part.requires_grad_() for part in parts]
        source = NT(parts, ragged_dims=(0, 1))
        if mode == "leaf":
            source.requires_grad_()
        with torch.inference_mode(mode == "inference"):
            output = operation(source, dim, index)
            expected = [part.select(dim, index) for part in references]
            torch.testing.assert_close(output.unbind(), expected)
        if mode != "inference":
            actual_gradients = torch.autograd.grad(output.square().sum(), source if mode == "leaf" else parts)
            expected_gradients = torch.autograd.grad(sum(part.square().sum() for part in expected), references)
            if mode == "leaf":
                actual_gradients = actual_gradients[0].unbind()
            torch.testing.assert_close(actual_gradients, expected_gradients)

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_select_layout(self, device, batch_first):
        parts = [torch.randn(m, 3, n, device=device, requires_grad=True) for m, n in ((1, 2), (2, 4))]
        source = NT(parts, ragged_dims=(0, 2), batch_first=batch_first).transpose(-1, -2)
        elements = [part.transpose(-1, -2) for part in parts]
        for dim in range(source.dim()):
            for index in (0, -1):
                output = source.select(dim, index)
                batch_dim = 0 if batch_first else 1
                if dim == batch_dim:
                    torch.testing.assert_close(output, elements[index])
                else:
                    physical_dim = dim - (dim > batch_dim)
                    expected = [part.select(physical_dim, index) for part in elements]
                    output_batch = batch_dim - (dim < batch_dim)
                    torch.testing.assert_close(output.unbind(output_batch), expected)
                    actual = torch.autograd.grad(output.sum(), parts, retain_graph=True)
                    reference = torch.autograd.grad(sum(part.sum() for part in expected), parts, retain_graph=True)
                    torch.testing.assert_close(actual, reference)
        with pytest.raises(IndexError):
            source.select(1 if batch_first else 0, 1)

    @pytest.mark.parametrize("dim,start,stop,step", [(-1, -3, -1, 1), (1, 0, 3, 2)])
    def test_slice_noncanonical_axes_and_gradients(self, device, dim, start, stop, step):
        elements = [
            torch.randn(3, 2, dtype=torch.float64, device=device, requires_grad=True),
            torch.randn(3, 4, dtype=torch.float64, device=device, requires_grad=True),
        ]
        source = NT(elements, ragged_dims=(1,))
        actual = torch.ops.aten.slice.Tensor(source, dim, start, stop, step)
        reference = [
            torch.ops.aten.slice.Tensor(element, dim - 1 if dim > 0 else dim, start, stop, step) for element in elements
        ]
        for result, expected in zip(actual, reference):
            assert_close(result, expected)
        actual_grads = torch.autograd.grad(actual.sum(), elements, retain_graph=True)
        expected_grads = torch.autograd.grad(sum(result.sum() for result in reference), elements)
        for result, expected in zip(actual_grads, expected_grads):
            assert_close(result, expected)

    @pytest.mark.parametrize("operation", [torch.index_add, torch.index_copy])
    def test_index_write_maps_multi_ragged_static_axis(self, operation):
        parts = [torch.arange(r * s * 9.0).reshape(r, s, 3, 3) for r, s in ((2, 3), (3, 4))]
        indices = torch.tensor([1, 0, 2])
        source = NT([part + 1000 for part in parts])
        output = operation(NT(parts), 3, indices, source)
        for actual, part, replacement in zip(output, parts, source):
            assert_close(actual, operation(part, 2, indices, replacement))

    def test_index_put_preserves_nonleading_ragged_coordinates(self):
        parts = [torch.arange(12.0).reshape(2, 2, 3) + 100 * i for i in range(2)]
        indices, values = (torch.tensor([0]),), torch.full((1, 2, 3), -1.0)
        output = torch.index_put(NT(parts, ragged_dims=(1,)), indices, values)
        for actual, part in zip(output, parts):
            assert_close(actual, torch.index_put(part, indices, values))

    def test_index_select_retains_all_outer_ragged_levels(self):
        parts = [torch.arange(r * s * t).reshape(r, s, t) for r, s, t in ((2, 3, 4), (3, 4, 5))]
        indices = torch.tensor([0, 1])
        output = torch.index_select(NT(parts, ragged_dims=(0, 1, 2)), 3, indices)
        for actual, part in zip(output, parts):
            assert_close(actual, part.index_select(2, indices))

    def test_index_put_matches_dense(self, device):
        seed = 0
        dtype = torch.float32
        shapes = ragged_shapes(seed, batch_size=3, min_len=3, max_len=6, trailing_shape=(4,))
        base = nested_rand(shapes, device, dtype)
        min_rows = min(shape[0] for shape in shapes)
        generator = torch.Generator()
        generator.manual_seed(seed)

        row_count = 2
        rows = torch.randint(0, min_rows, (row_count,), generator=generator)
        if row_count > 1:
            rows[-1] = rows[-1] - min_rows
        rows = rows.to(device=device, dtype=torch.long)

        shared_values = torch.randn(row_count, 4, device=device, dtype=dtype)
        row_out = torch.ops.aten.index_put.default(base, [rows], shared_values, False)
        row_ref = NT([torch.index_put(t, (rows,), shared_values) for t in base], **reference_options(base))
        assert_close(row_out, row_ref)

        dup_rows = rows.clone()
        dup_rows[0] = dup_rows[-1]
        dup_values = torch.randn(row_count, 4, device=device, dtype=dtype)
        dup_out = torch.ops.aten.index_put.default(base, [dup_rows], dup_values, True)
        dup_ref = NT(
            [torch.index_put(t, (dup_rows,), dup_values, accumulate=True) for t in base], **reference_options(base)
        )
        assert_close(dup_out, dup_ref)

        point_rows = torch.randint(0, min_rows, (2, 2), generator=generator)
        point_rows[0, 0] = point_rows[0, 0] - min_rows
        point_rows = point_rows.to(device=device, dtype=torch.long)
        point_cols = torch.randint(0, 4, (2, 2), generator=generator).to(device=device, dtype=torch.long)
        point_values = NT(
            [torch.randn(2, 2, device=device, dtype=dtype) for _ in range(len(base))],
            **reference_options(base),
        )
        point_out = torch.ops.aten.index_put.default(base, [point_rows, point_cols], point_values, False)
        point_ref = NT(
            [torch.index_put(t, (point_rows, point_cols), v) for t, v in zip(base, point_values)],
            **reference_options(base),
        )
        assert_close(point_out, point_ref)

    def test_index_write(self):
        nt = NT(
            [
                torch.zeros(2, 4),
                torch.zeros(3, 4),
            ]
        )
        index = torch.tensor([0, 2], dtype=torch.long)
        src = NT(
            [
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
            ]
        )

        index_add_out = torch.ops.aten.index_add.default(nt, 2, index, src)
        index_add_ref = NT([torch.index_add(t, 1, index, s) for t, s in zip(nt, src)], **reference_options(nt))
        assert_close(index_add_out, index_add_ref)

        index_copy_out = torch.ops.aten.index_copy.default(nt, 2, index, src)
        index_copy_ref = NT([torch.index_copy(t, 1, index, s) for t, s in zip(nt, src)], **reference_options(nt))
        assert_close(index_copy_out, index_copy_ref)

        row_nt = NT(
            [
                torch.zeros(4, 3),
                torch.zeros(5, 3),
            ]
        )
        row_index = torch.tensor([0, 2], dtype=torch.long)
        row_values = NT(
            [
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
            ]
        )

        index_put_out = torch.ops.aten.index_put.default(row_nt, [row_index], row_values, False)
        index_put_ref = NT(
            [torch.index_put(t, (row_index,), v) for t, v in zip(row_nt, row_values)], **reference_options(row_nt)
        )
        assert_close(index_put_out, index_put_ref)

        shared_values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        shared_out = torch.ops.aten.index_put.default(row_nt, [row_index], shared_values, False)
        shared_ref = NT([torch.index_put(t, (row_index,), shared_values) for t in row_nt], **reference_options(row_nt))
        assert_close(shared_out, shared_ref)

        scalar_value = torch.tensor(-3.0)
        scalar_out = torch.ops.aten.index_put.default(row_nt, [row_index], scalar_value, False)
        scalar_ref = NT([torch.index_put(t, (row_index,), scalar_value) for t in row_nt], **reference_options(row_nt))
        assert_close(scalar_out, scalar_ref)

        duplicate_index = torch.tensor([0, 0, 2], dtype=torch.long)
        duplicate_values = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [10.0, 20.0, 30.0],
                [4.0, 5.0, 6.0],
            ]
        )
        duplicate_out = torch.ops.aten.index_put.default(row_nt, [duplicate_index], duplicate_values, True)
        duplicate_ref = NT(
            [torch.index_put(t, (duplicate_index,), duplicate_values, accumulate=True) for t in row_nt],
            **reference_options(row_nt),
        )
        assert_close(duplicate_out, duplicate_ref)

        point_rows = torch.tensor([[0, -1], [1, 2]], dtype=torch.long)
        point_cols = torch.tensor([[1, 0], [2, 1]], dtype=torch.long)
        point_values = NT(
            [
                torch.tensor([[13.0, 17.0], [19.0, 23.0]]),
                torch.tensor([[29.0, 31.0], [37.0, 41.0]]),
            ]
        )
        point_out = torch.ops.aten.index_put.default(row_nt, [point_rows, point_cols], point_values, False)
        point_ref = NT(
            [torch.index_put(t, (point_rows, point_cols), v) for t, v in zip(row_nt, point_values)],
            **reference_options(row_nt),
        )
        assert_close(point_out, point_ref)

    def test_indexing(self):
        nt = NT(
            [
                torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            ]
        )

        batch_index = torch.tensor([1, 0, 1], dtype=torch.long)
        batch_out = torch.ops.aten.index_select.default(nt, 0, batch_index)
        batch_ref = NT([nt[1], nt[0], nt[1]], **reference_options(nt))
        assert_close(batch_out, batch_ref)

        gather_dense = torch.tensor(
            [
                [[0, 2, 1], [1, 0, 2], [0, 0, 0]],
                [[2, 1, 0], [0, 2, 1], [1, 1, 1]],
            ],
            dtype=torch.long,
        )
        with pytest.raises(RuntimeError, match="index size exceeds"):
            torch.ops.aten.gather.default(nt, 2, gather_dense, sparse_grad=False)
        gather_valid = gather_dense[:, :2]
        gather_out = torch.ops.aten.gather.default(nt, 2, gather_valid, sparse_grad=False)
        gather_ref_dense = torch.stack([torch.gather(t, 1, idx) for t, idx in zip(nt, gather_valid)])
        assert_close(gather_out, gather_ref_dense)
        gather_nt = nt.nested_like(gather_dense, strict=False)
        gather_out = torch.ops.aten.gather.default(nt, 2, gather_nt, sparse_grad=False)
        gather_ref = NT([torch.gather(t, 1, idx) for t, idx in zip(nt, gather_nt)], **reference_options(nt))
        assert_close(gather_out, gather_ref)

        ragged_index = NT(
            [
                torch.tensor([[1, 0, 1]], dtype=torch.long),
                torch.tensor([[2, 1, 0], [0, 2, 1]], dtype=torch.long),
            ]
        )
        gather_ragged = torch.ops.aten.gather.default(nt, 1, ragged_index, sparse_grad=False)
        gather_ragged_ref = NT([torch.gather(t, 0, idx) for t, idx in zip(nt, ragged_index)], **reference_options(nt))
        assert_close(gather_ragged, gather_ragged_ref)

        index = torch.tensor([2, 0], dtype=torch.long)
        index_select_out = torch.ops.aten.index_select.default(nt, 2, index)
        index_select_ref = NT([torch.index_select(t, 1, index) for t in nt], **reference_options(nt))
        assert_close(index_select_out, index_select_ref)

    def test_scatter(self):
        nt = NT(
            [
                torch.zeros(2, 3),
                torch.zeros(3, 3),
            ]
        )
        index = NT(
            [
                torch.tensor([[0, 2, 1], [1, 0, 2]], dtype=torch.long),
                torch.tensor([[2, 1, 0], [0, 2, 1], [1, 1, 1]], dtype=torch.long),
            ]
        )
        src = NT(
            [
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]),
            ]
        )

        scatter_src_out = torch.ops.aten.scatter.src(nt, 2, index, src)
        scatter_src_ref = NT(
            [torch.scatter(t, 1, idx, s) for t, idx, s in zip(nt, index, src)], **reference_options(nt)
        )
        assert_close(scatter_src_out, scatter_src_ref)

        scatter_value_out = torch.ops.aten.scatter.value(nt, 2, index, -1.0)
        scatter_value_ref = NT([torch.scatter(t, 1, idx, -1.0) for t, idx in zip(nt, index)], **reference_options(nt))
        assert_close(scatter_value_out, scatter_value_ref)

        scatter_add_out = torch.ops.aten.scatter_add.default(nt, 2, index, src)
        scatter_add_ref = NT(
            [torch.scatter_add(t, 1, idx, s) for t, idx, s in zip(nt, index, src)], **reference_options(nt)
        )
        assert_close(scatter_add_out, scatter_add_ref)

        if hasattr(torch.ops.aten, "scatter_reduce"):
            scatter_reduce_out = torch.ops.aten.scatter_reduce.two(nt, 2, index, src, "sum", include_self=True)
            scatter_reduce_ref = NT(
                [
                    torch.scatter_reduce(t, 1, idx, s, reduce="sum", include_self=True)
                    for t, idx, s in zip(nt, index, src)
                ],
                **reference_options(nt),
            )
            assert_close(scatter_reduce_out, scatter_reduce_ref)

    def test_searchsorted(self):
        sorted_nt = NT(
            [
                torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]),
                torch.tensor([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0], [2.0, 6.0, 8.0]]),
            ]
        )
        values_nt = NT(
            [
                torch.tensor([[0.5, 3.0, 6.5], [1.0, 5.0, 6.0]]),
                torch.tensor([[1.0, 2.0, 4.0], [0.0, 3.0, 6.0], [1.5, 6.5, 9.0]]),
            ]
        )
        boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0])

        nt_nt = torch.ops.aten.searchsorted.Tensor(sorted_nt, values_nt)
        tensor_nt = torch.ops.aten.searchsorted.Tensor(boundaries, values_nt)
        ref_nt_nt = NT([torch.searchsorted(s, v) for s, v in zip(sorted_nt, values_nt)], **reference_options(values_nt))
        ref_tensor_nt = NT([torch.searchsorted(boundaries, v) for v in values_nt], **reference_options(values_nt))

        assert_close(nt_nt, ref_nt_nt)
        assert_close(tensor_nt, ref_tensor_nt)

    def test_searchsorted_rejects_nested_sorter_without_sorted_sequence(self):
        boundaries = torch.tensor([1.0, 3.0, 5.0])
        values = torch.tensor([2.0, 4.0])
        sorter = NT([torch.tensor([0, 1, 2], dtype=torch.long)])
        with pytest.raises(TypeError, match="NestedTensor sorter requires sorted_sequence"):
            torch.ops.aten.searchsorted.Tensor(boundaries, values, sorter=sorter)


class TestLinearAlgebraOps:

    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    @pytest.mark.parametrize("ordering", ["left", "right"])
    def test_sliced_matrix(self, device, backend, ordering):
        torch.compiler.reset()

        def multiply(value, weight):
            if ordering == "left":
                return torch.bmm(value[:, 1:], weight)
            return torch.bmm(weight, value[:, :, 1:])

        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            compiled = torch.compile(multiply, backend=backend, fullgraph=True, dynamic=True)
            for lengths in ((3, 6), (5, 9), (1, 4), (1, 1)):
                shapes = [(length, 3) if ordering == "left" else (3, length) for length in lengths]
                parts = [torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
                weight = torch.randn(
                    (2, 3, 4) if ordering == "left" else (2, 4, 3),
                    device=device,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                references = [part.detach().clone().requires_grad_() for part in (*parts, weight)]
                nested = NT(parts, ragged_dims=(0,) if ordering == "left" else (1,))
                output = compiled(nested, weight)
                expected = [
                    part[1:] @ references[-1][index] if ordering == "left" else references[-1][index] @ part[:, 1:]
                    for index, part in enumerate(references[:-1])
                ]
                torch.testing.assert_close(output.unbind(), expected)
                torch.testing.assert_close(
                    torch.autograd.grad(output.concat.square().sum(), (*parts, weight)),
                    torch.autograd.grad(sum(part.square().sum() for part in expected), references),
                )

    @pytest.mark.parametrize("operation", [torch.bmm, torch.matmul])
    @pytest.mark.parametrize("ordering", ["left", "right"])
    @pytest.mark.parametrize("dtype", [torch.int64, torch.complex64, torch.float64])
    @pytest.mark.parametrize("dimensions", [((2, 67), 3, 4), ((0, 1, 2), 5, 7), ((1, 137), 33, 35)])
    def test_batched_matrix(self, device, operation, ordering, dtype, dimensions):
        lengths, contracted, output_size = dimensions

        def random(shape):
            if dtype == torch.int64:
                return torch.randint(-3, 4, shape, device=device)
            return torch.randn(shape, device=device, dtype=dtype) / contracted**0.5

        parts = [random((length, contracted) if ordering == "left" else (contracted, length)) for length in lengths]
        dense = random(
            (len(lengths), contracted, output_size) if ordering == "left" else (len(lengths), output_size, contracted)
        )
        if dtype != torch.int64:
            parts = [part.requires_grad_() for part in parts]
            dense.requires_grad_()
        nested = NT(parts, ragged_dims=(0,) if ordering == "left" else (1,))
        left, right = (nested, dense) if ordering == "left" else (dense, nested)
        try:
            expected = [
                (
                    torch.bmm(part.unsqueeze(0), dense[i : i + 1]).squeeze(0)
                    if ordering == "left"
                    else torch.bmm(dense[i : i + 1], part.unsqueeze(0)).squeeze(0)
                )
                for i, part in enumerate(parts)
            ]
        except RuntimeError:
            with pytest.raises(RuntimeError):
                operation(left, right)
            return
        output = operation(left, right)
        assert output.dtype == dtype
        torch.testing.assert_close(output.unbind(), expected)
        if dtype == torch.int64:
            return
        inputs = (*parts, dense)
        gradients = torch.autograd.grad(output.concat.abs().square().sum(), inputs, create_graph=True)
        references = torch.autograd.grad(sum(part.abs().square().sum() for part in expected), inputs, create_graph=True)
        torch.testing.assert_close(gradients, references)
        torch.testing.assert_close(
            torch.autograd.grad(sum(grad.real.sum() for grad in gradients), inputs),
            torch.autograd.grad(sum(grad.real.sum() for grad in references), inputs),
        )

    def test_matrix_kernels_follow_permuted_logical_axes(self):
        parts = [torch.randn(rows, 3, 3, dtype=torch.float64) for rows in (2, 4)]
        source = torch.permute(NT(parts, ragged_dims=(0,)), (0, 1, 3, 2))
        weight = torch.randn(3, 3, dtype=torch.float64)
        product = torch.matmul(source, weight)
        triangular = torch.tril(source)
        q, r = torch.linalg.qr(source)
        u, singular, vh = torch.linalg.svd(source, full_matrices=False)
        for i, part in enumerate(parts):
            expected = part.mT
            assert_close(product[i], expected @ weight)
            assert_close(triangular[i], torch.tril(expected))
            assert_close(q[i] @ r[i], expected)
            assert_close(u[i] @ torch.diag_embed(singular[i]) @ vh[i], expected)

    def test_matrix_vector_keeps_interior_ragged_axes(self):
        parts = [torch.randn(2, length, 3) for length in (2, 4)]
        vector = torch.randn(3)
        output = torch.matmul(NT(parts, ragged_dims=(1,)), vector)
        for actual, part in zip(output, parts):
            assert_close(actual, part @ vector)

    def test_determinant_rejects_nonsquare_samples(self):
        source = NT([torch.ones(2, 3), torch.ones(3, 2)], ragged_dims=(0, 1))
        with pytest.raises(RuntimeError, match="square"):
            torch.linalg.det(source)

    def test_diagonal_and_trace(self):
        nt_diagonal = NT(
            [
                torch.randn(2, 3, 4),
                torch.randn(3, 3, 4),
            ]
        )
        diagonal_out = torch.ops.aten.diagonal.default(nt_diagonal, 0, 2, 3)
        diagonal_ref = NT(
            [torch.diagonal(t, offset=0, dim1=1, dim2=2) for t in nt_diagonal], **reference_options(nt_diagonal)
        )
        assert_close(diagonal_out, diagonal_ref)

        nt_trace = NT(
            [
                torch.randn(3, 3),
                torch.randn(4, 4),
            ]
        )
        trace_out = torch.ops.aten.trace.default(nt_trace)
        trace_ref = NT([torch.trace(ti) for ti in nt_trace], **reference_options(nt_trace))
        assert_close(trace_out, trace_ref)

    def test_linalg_qr_eigh_svd_solve(self):
        qr_a0 = torch.randn(3, 3)
        qr_a1 = torch.randn(5, 3)
        nt_qr = NT([qr_a0, qr_a1])

        qr_q, qr_r = torch.ops.aten.linalg_qr.default(nt_qr, mode="reduced")
        for q_elem, r_elem, a_elem in zip(qr_q, qr_r, nt_qr):
            assert_close(q_elem @ r_elem, a_elem, rtol=1e-5, atol=1e-5)

        sym0 = torch.randn(3, 3)
        sym1 = torch.randn(4, 4)
        sym0 = sym0 + sym0.transpose(-1, -2)
        sym1 = sym1 + sym1.transpose(-1, -2)
        nt_sym = NT([sym0, sym1])
        eig_vals, eig_vecs = torch.ops.aten.linalg_eigh.default(nt_sym, UPLO="L")
        for vals_elem, vecs_elem, sym_elem in zip(eig_vals, eig_vecs, nt_sym):
            ref_vals, _ = torch.linalg.eigh(sym_elem, UPLO="L")
            assert_close(vals_elem, ref_vals, rtol=1e-5, atol=1e-5)
            recon = vecs_elem @ torch.diag_embed(vals_elem) @ vecs_elem.transpose(-1, -2)
            assert_close(recon, sym_elem, rtol=1e-5, atol=1e-5)

        svd_a0 = torch.randn(3, 2)
        svd_a1 = torch.randn(4, 2)
        nt_svd = NT([svd_a0, svd_a1])
        svd_u, svd_s, svd_vh = torch.ops.aten.linalg_svd.default(nt_svd, full_matrices=False)
        for u_elem, s_elem, vh_elem, a_elem in zip(svd_u, svd_s, svd_vh, nt_svd):
            ref_s = torch.linalg.svd(a_elem, full_matrices=False).S
            assert_close(s_elem, ref_s, rtol=1e-5, atol=1e-5)
            recon = u_elem @ torch.diag_embed(s_elem) @ vh_elem
            assert_close(recon, a_elem, rtol=1e-5, atol=1e-5)

        spd0 = torch.randn(3, 3)
        spd1 = torch.randn(4, 4)
        spd0 = spd0 @ spd0.transpose(-1, -2) + 1e-1 * torch.eye(3)
        spd1 = spd1 @ spd1.transpose(-1, -2) + 1e-1 * torch.eye(4)
        nt_spd = NT([spd0, spd1])
        rhs0 = torch.randn(3, 2)
        rhs1 = torch.randn(4, 2)
        nt_rhs = NT([rhs0, rhs1])

        solve_out = torch.ops.aten.linalg_solve.default(nt_spd, nt_rhs)
        solve_ref = NT([torch.linalg.solve(a, b) for a, b in zip(nt_spd, nt_rhs)], **reference_options(nt_spd))
        for out_elem, ref_elem in zip(solve_out, solve_ref):
            assert_close(out_elem, ref_elem, rtol=1e-5, atol=1e-5)

    def test_linalg_qr_and_eigh_batched_matrices(self):
        """QR and eigendecomposition on nested tensors whose elements are batches of matrices."""
        a0 = torch.randn(2, 3, 3)
        a1 = torch.randn(3, 3, 3)
        nt_a = NT([a0, a1])

        qr_q, qr_r = torch.ops.aten.linalg_qr.default(nt_a, mode="reduced")
        for q_elem, r_elem, a_elem in zip(qr_q, qr_r, nt_a):
            assert_close(q_elem @ r_elem, a_elem, rtol=1e-5, atol=1e-5)

        sym_a0 = a0 + a0.transpose(-1, -2)
        sym_a1 = a1 + a1.transpose(-1, -2)
        nt_sym = NT([sym_a0, sym_a1])
        eigh_vals, eigh_vecs = torch.ops.aten.linalg_eigh.default(nt_sym, UPLO="L")
        for vals_elem, vecs_elem, sym_elem in zip(eigh_vals, eigh_vecs, nt_sym):
            ref_vals, _ = torch.linalg.eigh(sym_elem, UPLO="L")
            assert_close(vals_elem, ref_vals, rtol=1e-5, atol=1e-5)
            recon = vecs_elem @ torch.diag_embed(vals_elem) @ vecs_elem.transpose(-1, -2)
            assert_close(recon, sym_elem, rtol=1e-5, atol=1e-5)

    def test_linalg_vector_norm(self):
        base = NT(
            [
                torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
                torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
            ]
        )

        global_out = torch.ops.aten.linalg_vector_norm.default(base, 2, None, False)
        global_ref = NT([torch.linalg.vector_norm(t, ord=2) for t in base], **reference_options(base))
        assert_close(global_out, global_ref)

        dim_out = torch.ops.aten.linalg_vector_norm.default(base, 2, [1], False)
        dim_ref = NT([torch.linalg.vector_norm(t, ord=2, dim=0) for t in base], **reference_options(base))
        assert_close(dim_out, dim_ref)


class TestMaskingOps:

    def test_masked_fill(self):
        base = NT([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
        mask = NT([torch.tensor([True, False, True]), torch.tensor([False, True])])

        scalar_out = torch.ops.aten.masked_fill.Scalar(base, mask, 0.0)
        scalar_ref = NT([torch.masked_fill(t, m, 0.0) for t, m in zip(base, mask)], **reference_options(base))
        assert_close(scalar_out, scalar_ref)

        value = torch.tensor(-1.5, dtype=base.dtype, device=base.device)
        tensor_out = torch.ops.aten.masked_fill.Tensor(base, mask, value)
        tensor_ref = NT([torch.masked_fill(t, m, value) for t, m in zip(base, mask)], **reference_options(base))
        assert_close(tensor_out, tensor_ref)

    def test_masked_scatter_nested_source(self):
        base = NT([torch.zeros(3), torch.zeros(2)])
        mask = NT([torch.tensor([True, False, True]), torch.tensor([False, True])])
        source = NT([torch.tensor([1.0, 2.0]), torch.tensor([3.0])])

        output = torch.ops.aten.masked_scatter.default(base, mask, source)
        reference = NT(
            [torch.masked_scatter(t, m, s) for t, m, s in zip(base, mask, source)], **reference_options(base)
        )

        assert_close(output, reference)

    def test_masked_select(self):
        base = NT([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
        mask = NT([torch.tensor([True, False, True]), torch.tensor([False, True])])

        output = torch.ops.aten.masked_select.default(base, mask)
        reference = NT([torch.masked_select(t, m) for t, m in zip(base, mask)], **reference_options(base))
        assert_close(output, reference)

    def test_nonzero(self):
        base = NT([torch.tensor([[0.0, 1.0], [2.0, 0.0]]), torch.tensor([[3.0, 0.0]])])

        output = torch.ops.aten.nonzero.default(base)
        reference = NT([torch.nonzero(t, as_tuple=False) for t in base], **reference_options(base))

        assert_close(output, reference)
        assert output.dtype == torch.long


class TestReductionOps:

    @pytest.mark.parametrize("case", ["true", "false", "empty"])
    def test_all_true(self, device, case):
        lengths = (0, 0) if case == "empty" else (2, 3)
        parts = [torch.ones(length, 4, dtype=torch.bool, device=device) for length in lengths]
        if case == "false":
            parts[0][0, 0] = False
        source = NT(parts, ragged_dims=(0,))
        expected = torch.all(torch.cat(parts))
        for actual in (torch._is_all_true(source), torch.ops.aten._is_all_true.default(source)):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.ops.aten._is_any_true.default(source), torch.any(torch.cat(parts)))

    @pytest.mark.parametrize("lengths", [(2, 3), (0, 0)])
    def test_native_count(self, device, lengths):
        parts = [torch.arange(length * 3, device=device).reshape(length, 3).remainder(2) for length in lengths]
        source = NT(parts, ragged_dims=(0,))
        operation = torch.ops.aten.count_nonzero.default
        torch.testing.assert_close(operation(source), torch.count_nonzero(torch.cat(parts)))
        for dim in (1, -1):
            expected = [torch.count_nonzero(part, dim=dim - 1 if dim > 0 else dim) for part in parts]
            actual = operation(source, dim=dim)
            if isinstance(actual, NT):
                torch.testing.assert_close(actual.unbind(), expected)
            else:
                torch.testing.assert_close(actual, torch.stack(expected))

    def test_multinomial(self, device):
        source = NT([torch.ones(length, 3, device=device) for length in (2, 3)], ragged_dims=(0,))
        generator = torch.Generator(device=device).manual_seed(23)
        reference_generator = torch.Generator(device=device).manual_seed(23)
        actual = torch.ops.aten.multinomial.default(source, 2, True, generator=generator)
        expected = torch.multinomial(source.concat, 2, True, generator=reference_generator)
        torch.testing.assert_close(actual.concat, expected)
        assert tuple(part.shape for part in actual.unbind(0)) == ((2, 2), (3, 2))
        torch.testing.assert_close(generator.get_state(), reference_generator.get_state())

    def test_categorical(self, device):
        parts = [torch.randn(length, 4, 3, dtype=torch.float64, device=device, requires_grad=True) for length in (2, 3)]
        distribution = torch.distributions.Categorical(logits=NT(parts, ragged_dims=(0,)), validate_args=True)
        references = [torch.distributions.Categorical(logits=part, validate_args=True) for part in parts]
        torch.testing.assert_close(distribution.probs.unbind(0), tuple(reference.probs for reference in references))
        labels = NT([torch.zeros(part.shape[:-1], dtype=torch.long, device=device) for part in parts], ragged_dims=(0,))
        output = distribution.log_prob(labels)
        expected = [reference.log_prob(label) for reference, label in zip(references, labels)]
        torch.testing.assert_close(output.unbind(0), tuple(expected))
        gradients = torch.autograd.grad(output.concat.sum(), parts, retain_graph=True)
        reference_gradients = torch.autograd.grad(sum(value.sum() for value in expected), parts)
        torch.testing.assert_close(gradients, reference_gradients)
        invalid = [torch.full_like(part, torch.nan) for part in parts]
        with pytest.raises(ValueError, match="invalid values"):
            torch.distributions.Categorical(logits=NT(invalid, ragged_dims=(0,)), validate_args=True)

    @pytest.mark.parametrize("operation,data", [(torch.std, [1.0, 2.0, 3.0]), (torch.prod, [0.0, 2.0, 3.0])])
    def test_reductions_preserve_second_derivatives(self, operation, data):
        values = torch.tensor(data, dtype=torch.float64, requires_grad=True)
        reference = values.detach().clone().requires_grad_()
        options = {"correction": 0} if operation is torch.std else {}
        output = operation(NT(values), dim=1, **options)
        expected = operation(reference, dim=0, **options)
        gradient = torch.autograd.grad(output.sum(), values, create_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, reference, create_graph=True)[0]
        second = torch.autograd.grad(gradient.sum(), values)[0]
        expected_second = torch.autograd.grad(expected_gradient.sum(), reference)[0]
        assert_close(gradient, expected_gradient)
        assert_close(second, expected_second)

    def test_vector_norm_zero_coordinates_and_dtype(self):
        values = torch.tensor([0.0, 2.0], dtype=torch.float64, requires_grad=True)
        output = torch.linalg.vector_norm(NT(values), ord=0.5, dim=1)
        gradient = torch.autograd.grad(output.sum(), values)[0]
        assert_close(gradient, torch.tensor([0.0, 1.0], dtype=torch.float64))
        with pytest.raises(RuntimeError, match="dtype"):
            torch.linalg.vector_norm(NT(values), dim=1, dtype=torch.float32)

    def test_order_statistics_map_static_axis_after_ragged_collapse(self):
        parts = [torch.arange(24.0).reshape(2, 3, 4), torch.arange(24.0).reshape(3, 2, 4)]
        output = torch.median(NT(parts, ragged_dims=(0, 1)), dim=3)
        for actual, part in zip(output.values, parts):
            assert_close(actual, torch.median(part, dim=2).values)

    @pytest.mark.parametrize("operation", [torch.sum, torch.mean])
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize(
        "input_dtype,result_dtype",
        [
            (torch.float16, None),
            (torch.bfloat16, None),
            (torch.float32, torch.float16),
            (torch.float32, torch.bfloat16),
            (torch.float16, torch.float32),
        ],
    )
    def test_reduction_dtype_gradients(self, device, operation, axis, input_dtype, result_dtype):
        elements = [
            torch.randn(shape, device=device, dtype=input_dtype, requires_grad=True) for shape in ((2, 3, 4), (4, 2, 4))
        ]
        nested = NT(elements, ragged_dims=(0, 1))

        output = operation(nested, dim=axis + 1, keepdim=True, dtype=result_dtype)
        references = [operation(element, dim=axis, keepdim=True, dtype=result_dtype) for element in elements]
        gradients = torch.autograd.grad(output.concat.sum(), elements)

        assert output.dtype == (result_dtype or input_dtype)
        for actual, expected in zip(output, references):
            torch.testing.assert_close(actual, expected)
        for actual, element in zip(gradients, elements):
            derivative = 1.0 if operation is torch.sum else 1.0 / element.shape[axis]
            expected = torch.full_like(element, derivative, dtype=torch.float64)
            if not (
                operation is torch.mean
                and element.device.type == "cpu"
                and output.dtype in (torch.float16, torch.bfloat16)
            ):
                expected = expected.to(dtype=output.dtype)
            expected = expected.to(dtype=input_dtype)
            torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("result_dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("layout", ["single", "multiple", "static"])
    def test_mean_preserves_precision(self, device, result_dtype, layout):
        values = torch.tensor([1.0003, -1.0, 0.0], device=device, dtype=torch.float32)
        if layout == "single":
            nested = NT([values, torch.ones(2, device=device)], ragged_dims=(0,))
            axis = 1
        elif layout == "multiple":
            nested = NT([values.reshape(3, 1), torch.ones(2, 2, device=device)], ragged_dims=(0, 1))
            axis = 1
        else:
            nested = NT([values.reshape(1, 3), torch.ones(2, 3, device=device)], ragged_dims=(0,))
            axis = 2

        output = torch.mean(nested, dim=axis, dtype=result_dtype)

        reference_values = values if values.device.type == "cpu" else values.to(dtype=result_dtype)
        expected = math.fsum(reference_values.tolist()) / values.numel()
        expected = torch.tensor(expected, device=device, dtype=result_dtype)
        torch.testing.assert_close(output[0].reshape(-1)[0], expected, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "aten_op,torch_op",
        [
            pytest.param(torch.ops.aten.max, torch.max, id="max"),
            pytest.param(torch.ops.aten.min, torch.min, id="min"),
        ],
    )
    def test_max_min(self, aten_op, torch_op):
        nt = NT(
            [
                torch.tensor([[1.0, 5.0, 3.0], [2.0, 0.0, 4.0]]),
                torch.tensor([[9.0, 1.0, 2.0], [3.0, 7.0, 6.0], [4.0, 0.0, 8.0]]),
            ]
        )
        # Global reduction
        assert_close(aten_op.default(nt), torch_op(nt.concat))

        # Static dim
        vals_last, idxs_last = aten_op.dim(nt, 2, False)
        vals_last_ref = NT([torch_op(t, dim=1).values for t in nt], **reference_options(nt))
        idxs_last_ref = NT([torch_op(t, dim=1).indices for t in nt], **reference_options(nt))
        assert_close(vals_last, vals_last_ref)
        assert_close(idxs_last, idxs_last_ref)

        # Ragged dim
        vals_ragged, idxs_ragged = aten_op.dim(nt, 1, False)
        vals_ragged_ref = torch.stack([torch_op(t, dim=0).values for t in nt])
        idxs_ragged_ref = torch.stack([torch_op(t, dim=0).indices for t in nt])
        assert_close(vals_ragged, vals_ragged_ref)
        assert_close(idxs_ragged, idxs_ragged_ref)

    @pytest.mark.parametrize(
        "aten_op,torch_op,global_ref",
        [
            pytest.param(torch.ops.aten.argmax, torch.argmax, torch.tensor([1, 0], dtype=torch.long), id="argmax"),
            pytest.param(torch.ops.aten.argmin, torch.argmin, torch.tensor([4, 7], dtype=torch.long), id="argmin"),
        ],
    )
    def test_argmax_argmin(self, aten_op, torch_op, global_ref):
        nt = NT(
            [
                torch.tensor([[1.0, 5.0, 3.0], [2.0, 0.0, 4.0]]),
                torch.tensor([[9.0, 1.0, 2.0], [3.0, 7.0, 6.0], [4.0, 0.0, 8.0]]),
            ]
        )
        # Global
        assert_close(aten_op.default(nt), global_ref)

        # Static dim
        out_last = aten_op.default(nt, 2, False)
        ref_last = NT([torch_op(t, dim=1) for t in nt], **reference_options(nt))
        assert_close(out_last, ref_last)

        # Ragged dim
        out_ragged = aten_op.default(nt, 1, False)
        ref_ragged = torch.stack([torch_op(t, dim=0) for t in nt])
        assert_close(out_ragged, ref_ragged)

    def test_order_stat(self):
        nt = NT(
            [
                torch.tensor([[3.0, 1.0, 2.0], [5.0, 4.0, 6.0]]),
                torch.tensor([[9.0, 7.0, 8.0], [0.0, 2.0, 1.0], [6.0, 5.0, 4.0]]),
            ]
        )

        kth_vals, kth_idxs = torch.ops.aten.kthvalue.default(nt, 2, 2, False)
        kth_ref = tuple(NT([torch.kthvalue(t, 2, dim=1)[i] for t in nt], **reference_options(nt)) for i in range(2))
        assert_close(kth_vals, kth_ref[0])
        assert_close(kth_idxs, kth_ref[1])

        median_global = torch.ops.aten.median.default(nt)
        assert_close(median_global, torch.median(nt.concat))

        median_vals, median_idxs = torch.ops.aten.median.dim(nt, 2, True)
        median_ref = tuple(
            NT([torch.median(t, dim=1, keepdim=True)[i] for t in nt], **reference_options(nt)) for i in range(2)
        )
        assert_close(median_vals, median_ref[0])
        assert_close(median_idxs, median_ref[1])

        mode_vals, mode_idxs = torch.ops.aten.mode.default(nt, 2, False)
        mode_ref = tuple(NT([torch.mode(t, dim=1)[i] for t in nt], **reference_options(nt)) for i in range(2))
        assert_close(mode_vals, mode_ref[0])
        assert_close(mode_idxs, mode_ref[1])

        nan_nt = NT(
            [
                torch.tensor([[1.0, float("nan"), 3.0], [2.0, 4.0, float("nan")]]),
                torch.tensor([[float("nan"), 7.0, 8.0], [0.0, float("nan"), 1.0], [6.0, 5.0, 4.0]]),
            ]
        )
        nanmedian_global = torch.ops.aten.nanmedian.default(nan_nt)
        assert_close(nanmedian_global, torch.nanmedian(nan_nt.concat), equal_nan=True)

        nanmedian_vals, nanmedian_idxs = torch.ops.aten.nanmedian.dim(nan_nt, 2, False)
        nanmedian_ref = tuple(
            NT([torch.nanmedian(t, dim=1)[i] for t in nan_nt], **reference_options(nan_nt)) for i in range(2)
        )
        assert_close(nanmedian_vals, nanmedian_ref[0], equal_nan=True)
        assert_close(nanmedian_idxs, nanmedian_ref[1])


class TestSortingOps:

    def test_ragged_topk_k_exceeds_min_length(self):
        nt = NT(
            [
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
            ]
        )
        with pytest.raises(RuntimeError, match="k out of range"):
            torch.ops.aten.topk.default(nt, 3, 1, True, True)

    @pytest.mark.parametrize(
        "aten_fn,ref_fn",
        [
            pytest.param(
                lambda x: torch.ops.aten.sort.default(x, 2, False),
                lambda t: torch.sort(t, dim=1, descending=False).values,
                id="sort",
            ),
            pytest.param(
                lambda x: torch.ops.aten.argsort.default(x, 2, False),
                lambda t: torch.argsort(t, dim=1, descending=False),
                id="argsort",
            ),
            pytest.param(
                lambda x: torch.ops.aten.topk.default(x, 2, 2, True, True),
                lambda t: torch.topk(t, 2, dim=1, largest=True, sorted=True).values,
                id="topk",
            ),
            pytest.param(
                lambda x: torch.ops.aten.cumsum.default(x, 2),
                lambda t: torch.cumsum(t, dim=1),
                id="cumsum",
            ),
            pytest.param(
                lambda x: torch.ops.aten.cumprod.default(x, 2),
                lambda t: torch.cumprod(t, dim=1),
                id="cumprod",
            ),
            pytest.param(
                lambda x: torch.ops.aten.logcumsumexp.default(x, 2),
                lambda t: torch.logcumsumexp(t, dim=1),
                id="logcumsumexp",
            ),
            pytest.param(
                lambda x: torch.ops.aten.cummax.default(x, 2),
                lambda t: torch.cummax(t, dim=1).values,
                id="cummax",
            ),
            pytest.param(
                lambda x: torch.ops.aten.cummin.default(x, 2),
                lambda t: torch.cummin(t, dim=1).values,
                id="cummin",
            ),
            pytest.param(
                lambda x: torch.ops.aten.flip.default(x, [2]),
                lambda t: torch.flip(t, dims=[1]),
                id="flip",
            ),
        ],
    )
    def test_sorting_and_cumulative_op(self, aten_fn, ref_fn):
        nt = NT(
            [
                torch.tensor([[3.0, 1.0, 2.0], [4.0, 0.0, 5.0]]),
                torch.tensor([[2.0, 5.0, 1.0], [7.0, 6.0, 8.0], [9.0, 4.0, 3.0]]),
            ]
        )
        result = aten_fn(nt)
        if isinstance(result, tuple):
            result = result[0]
        reference = NT([ref_fn(t) for t in nt], **reference_options(nt))
        assert_close(result, reference)

    def test_cumsum_does_not_mix_sample_prefixes(self):
        parts = [torch.tensor([float("inf")]), torch.tensor([1.0, 2.0]), torch.tensor([1e20, 1.0, -1e20, 3.0])]
        output = torch.cumsum(NT(parts), dim=1)
        for actual, part in zip(output, parts):
            assert_close(actual, part.cumsum(0))

    def test_cumsum_ragged_dim(self):
        nt = NT(
            [
                torch.tensor([[3.0, 1.0, 2.0], [4.0, 0.0, 5.0]]),
                torch.tensor([[2.0, 5.0, 1.0], [7.0, 6.0, 8.0], [9.0, 4.0, 3.0]]),
            ]
        )
        cumsum_ragged = torch.ops.aten.cumsum.default(nt, 1)
        for got, t in zip(cumsum_ragged, nt):
            assert_close(got, torch.cumsum(t, dim=0))


class TestTernaryOps:

    def test_ternary_and_take(self):
        nt = NT(
            [
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([4.0, 5.0]),
            ]
        )

        dense = nt.tensor
        where_out = torch.ops.aten.where.ScalarOther(dense > 2, nt, 0.0)
        where_ref = NT([torch.where(t > 2, t, 0.0) for t in nt], **reference_options(nt))
        assert_close(where_out, where_ref)

        where_scalar_out = torch.ops.aten.where.Scalar(nt > 2, 1.0, 0.0)
        where_scalar_ref = NT([torch.where(t > 2, 1.0, 0.0) for t in nt], **reference_options(nt))
        assert_close(where_scalar_out, where_scalar_ref)

        addcmul_out = torch.ops.aten.addcmul.default(dense, nt, nt, value=2)
        addcmul_ref = NT([torch.addcmul(t, t, t, value=2) for t in nt], **reference_options(nt))
        assert_close(addcmul_out, addcmul_ref)

        addcdiv_out = torch.ops.aten.addcdiv.default(dense, nt, nt + 1, value=2)
        addcdiv_ref = NT([torch.addcdiv(t, t, t + 1, value=2) for t in nt], **reference_options(nt))
        assert_close(addcdiv_out, addcdiv_ref)

        take_index = torch.tensor([0, 2, 4], dtype=torch.long)
        take_out = torch.ops.aten.take.default(nt, take_index)
        take_ref = torch.take(torch.cat([t.reshape(-1) for t in nt]), take_index)
        assert_close(take_out, take_ref)

    def test_ternary_accepts_broadcast(self):
        nt = NT(
            [
                torch.ones(2, 3),
                torch.ones(4, 3),
            ]
        )
        bias = torch.arange(3.0)

        output = torch.ops.aten.addcmul.default(nt, bias, bias, value=2)
        reference = NT([torch.addcmul(t, bias, bias, value=2) for t in nt], **reference_options(nt))

        assert_close(output, reference)

    def test_ternary_rejects_packed_only_broadcast(self):
        nt = NT(
            [
                torch.ones(2, 3),
                torch.ones(4, 3),
            ]
        )
        bad = torch.arange(18.0).reshape(6, 3)

        with pytest.raises(RuntimeError, match="size of tensor"):
            torch.ops.aten.addcmul.default(nt, bad, bad, value=2)

    def test_ternary_matches_dense(self, device):
        seed = 0
        dtype = torch.float32
        shapes = ragged_shapes(seed, batch_size=3, min_len=2, max_len=5, trailing_shape=(4,))
        nt = nested_rand(shapes, device, dtype)

        torch.manual_seed(seed)
        bias = torch.randn(4, device=device, dtype=dtype)
        scale = torch.randn(4, device=device, dtype=dtype)
        denom = torch.randn(4, device=device, dtype=dtype).abs() + 0.5
        condition = torch.randn(4, device=device) > 0

        addcmul_out = torch.ops.aten.addcmul.default(nt, bias, scale, value=0.25)
        addcmul_ref = NT([torch.addcmul(t, bias, scale, value=0.25) for t in nt], **reference_options(nt))
        assert_close(addcmul_out, addcmul_ref)

        addcdiv_out = torch.ops.aten.addcdiv.default(nt, scale, denom, value=-0.5)
        addcdiv_ref = NT([torch.addcdiv(t, scale, denom, value=-0.5) for t in nt], **reference_options(nt))
        assert_close(addcdiv_out, addcdiv_ref)

        where_out = torch.ops.aten.where.self(condition, nt, bias)
        where_ref = NT([torch.where(condition, t, bias) for t in nt], **reference_options(nt))
        assert_close(where_out, where_ref)
