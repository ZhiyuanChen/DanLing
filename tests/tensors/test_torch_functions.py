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

import pytest
import torch
from torch.nn import functional as F

from danling.tensors import NestedTensor, nested_execution_guard
from tests.tensors.utils import assert_close, low_precision_cuda_tolerances, nested_rand, ragged_shapes

NT = NestedTensor


def reference_options(source: NestedTensor) -> dict:
    r"""Return public construction options for an elementwise reference."""
    return {
        "batch_first": source.batch_first,
        "padding_value": source.padding_value,
        "mask_value": source.mask_value,
    }


def _run_or_expect_unsupported(nested_call, tensor_call):
    try:
        reference = tensor_call()
    except RuntimeError as error:
        with pytest.raises(type(error)):
            nested_call()
        return None
    return nested_call(), reference


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
        assert isinstance(output, NestedTensor)
        assert_close(output, nt.tensor + tensor)

    def test_div_converts_other_to_nested(self, device, float_dtype):
        nt = NestedTensor([torch.ones(2, device=device, dtype=float_dtype) * 4])
        other = torch.full_like(nt.tensor, 2, device=device, dtype=float_dtype)
        output = torch.div(other, nt)
        assert isinstance(output, NestedTensor)
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
        assert isinstance(output, NestedTensor)
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

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compile_det(self):
        torch.manual_seed(1016)
        nt = NT([torch.randn(size, size) for size in (3, 4)])
        compiled = torch.compile(torch.linalg.det, backend="inductor", fullgraph=True)
        output = compiled(nt)
        reference = NT([torch.linalg.det(element) for element in nt], **reference_options(nt))
        assert isinstance(output, NestedTensor)
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
        assert isinstance(dropout_comp, NestedTensor)
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
        assert isinstance(result, NestedTensor)
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

    def test_squeeze_default(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 1, 3, device=device, dtype=float_dtype),
                torch.randn(3, 1, 3, device=device, dtype=float_dtype),
            ]
        )
        output = torch.squeeze(nt)
        reference = NT([torch.squeeze(t) for t in nt], **reference_options(nt))
        assert_close(output, reference)

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

    def test_unsqueeze_batch_dim_raises(self):
        nt = NestedTensor([torch.tensor([1, 2])])
        with pytest.raises(ValueError):
            torch.unsqueeze(nt, dim=0)

    def test_unsqueeze_batch_first_false_respects_batch_dim(self):
        nt = NestedTensor([torch.tensor([1, 2])], batch_first=False)
        with pytest.raises(ValueError):
            torch.unsqueeze(nt, dim=0)
        with pytest.raises(ValueError):
            torch.unsqueeze(nt, dim=1)
        output = torch.unsqueeze(nt, dim=2)
        assert isinstance(output, NestedTensor)
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
        assert isinstance(result, NestedTensor)
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
        assert isinstance(result, NestedTensor)
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

    def test_flatten_start_dim_zero_returns_tensor(self, device, float_dtype):
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
        assert isinstance(rows, NestedTensor)
        assert len(rows.ragged_dims) == 1
        assert [tuple(x.shape) for x in rows] == [(4, 3)] * 4 + [(6, 3)] * 6
        ref = [r for i in range(len(grid)) for r in grid[i].unbind(0)]
        assert all(torch.equal(rows[k], ref[k]) for k in range(len(ref)))

    def test_flatten_batch_first_false_rows(self):
        grid = NestedTensor([torch.randn(4, 4, 3), torch.randn(6, 6, 3)], batch_first=False)
        rows = grid.flatten(0, 1)
        assert isinstance(rows, NestedTensor)
        assert rows.batch_first is False
        assert [tuple(x.shape) for x in rows] == [(4, 3), (6, 3)] * 4 + [(6, 3)] * 2
        ref = []
        for row in range(6):
            ref.extend(grid[batch][row] for batch in range(2) if row < grid[batch].shape[0])
        assert all(torch.equal(rows[index], ref[index]) for index in range(len(ref)))

    def test_flatten_batch_first_false_non_row_dense(self):
        grid = NestedTensor([torch.randn(4, 4, 3), torch.randn(6, 6, 3)], batch_first=False)
        dense = grid.flatten(1, 2)
        assert isinstance(dense, torch.Tensor)
        assert not isinstance(dense, NestedTensor)
        assert_close(dense, torch.flatten(grid.tensor, 1, 2))

    def test_unflatten_batch_dim_not_supported(self):
        nt = NestedTensor([torch.tensor([[1, 2]])])
        with pytest.raises(ValueError):
            torch.unflatten(nt, dim=0, sizes=(1, 2))


class TestExpand:

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

    def test_expand_sample_axis_values_and_vjp(self, device, float_dtype):
        template = NT(
            [
                torch.empty(3, 3, device=device, dtype=float_dtype),
                torch.empty(5, 3, device=device, dtype=float_dtype),
            ]
        )
        values = torch.randn_like(template.concat, requires_grad=True)
        reference_positions = template.packed_like(values)

        sampled = reference_positions.unsqueeze(-3).expand(
            len(reference_positions), 4, reference_positions.shape[-2], 3
        )

        expected = values.unsqueeze(1).expand(-1, 4, -1)
        assert sampled.concat.shape == (8, 4, 3)
        assert sampled.ragged_dims == (1,)
        assert_close(sampled.concat, expected)

        cotangent = torch.randn_like(expected)
        actual_grad = torch.autograd.grad(sampled.concat, values, cotangent)[0]
        expected_grad = torch.autograd.grad(expected, values, cotangent)[0]
        assert_close(actual_grad, expected_grad)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_expand_sample_axis_compiles_with_vjp(self, device):
        def consume(template, values):
            reference_positions = template.packed_like(values)
            sampled = reference_positions.unsqueeze(-3).expand(
                reference_positions.shape[0], 4, reference_positions.shape[-2], reference_positions.shape[-1]
            )
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
        assert isinstance(output, NestedTensor)
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
        assert isinstance(result, NestedTensor)
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
        assert isinstance(result, NestedTensor)
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
        assert isinstance(eigenvalues, NestedTensor)
        assert isinstance(eigenvectors, NestedTensor)
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
        assert isinstance(result, NestedTensor)
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
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.linalg.norm(a, ord=-1, dim=0))
        assert_close(result[1], torch.linalg.norm(b, ord=-1, dim=0))

    def test_linalg_norm_no_dim(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 2, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.linalg.norm(nt)
        assert isinstance(result, NestedTensor)
        assert_close(result[0].squeeze(), torch.linalg.norm(a))
        assert_close(result[1].squeeze(), torch.linalg.norm(b))

    def test_linalg_norm_with_dim(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.linalg.norm(nt, dim=1)
        assert isinstance(result, NestedTensor)
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
        assert isinstance(Q, NestedTensor)
        assert isinstance(R, NestedTensor)
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
        assert isinstance(result, NestedTensor)
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
        assert isinstance(U, NestedTensor)
        assert isinstance(S, NestedTensor)
        assert isinstance(Vh, NestedTensor)
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

        with nested_execution_guard(
            forbid_iteration=True,
            forbid_storage_map=True,
            forbid_eager_fallback=True,
            forbid_padded_materialization=True,
            forbid_dense_repack=True,
        ):
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

            assert isinstance(output, NestedTensor)
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
    def test_baddbmm_compile_fullgraph_tensor_lhs(self, device, float_dtype):
        bias = torch.randn(2, 3, 1, device=device, dtype=float_dtype)
        left = torch.randn(2, 3, 4, device=device, dtype=float_dtype)
        right = NT(
            [
                torch.randn(2, 4, 5, device=device, dtype=float_dtype),
                torch.randn(2, 4, 3, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(lambda x, y, z: torch.baddbmm(x, y, z), backend="inductor", fullgraph=True)
        output = compiled(bias, left, right)
        reference = NT([torch.baddbmm(bias, left, b) for b in right], **reference_options(right))
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_bmm_compile_fullgraph_tensor_lhs(self, device, float_dtype):
        left = torch.randn(2, 3, 4, device=device, dtype=float_dtype)
        right = NT(
            [
                torch.randn(2, 4, 5, device=device, dtype=float_dtype),
                torch.randn(2, 4, 3, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(lambda x, y: torch.bmm(x, y), backend="inductor", fullgraph=True)
        output = compiled(left, right)
        reference = NT([torch.bmm(left, b) for b in right], **reference_options(right))
        assert_close(output, reference)

    def test_bmm(self, device, float_dtype):
        left = NT(
            [
                torch.randn(2, 1, 3, device=device, dtype=float_dtype),
                torch.randn(1, 2, 3, device=device, dtype=float_dtype),
            ]
        )
        right = NT(
            [
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
                torch.randn(1, 3, 5, device=device, dtype=float_dtype),
            ],
            **reference_options(left),
        )
        output = torch.bmm(left, right)
        reference = NT([torch.bmm(a, b) for a, b in zip(left, right)], **reference_options(left))
        assert_close(output, reference)

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
        assert isinstance(result, NestedTensor)
        assert_close(result[0].squeeze(), ref_a)
        assert_close(result[1].squeeze(), ref_b)

    def test_diag_1d_to_2d(self, device, float_dtype):
        a = torch.randn(3, device=device, dtype=float_dtype)
        b = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.diag(nt)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.diag(a))
        assert_close(result[1], torch.diag(b))

    def test_diag_2d_to_1d(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.diag(nt)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.diag(a))
        assert_close(result[1], torch.diag(b))

    def test_diagflat(self, device, float_dtype):
        a = torch.randn(3, device=device, dtype=float_dtype)
        b = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.diagflat(nt)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.diagflat(a))
        assert_close(result[1], torch.diagflat(b))

    def test_diagonal(self, device, float_dtype):
        a = torch.randn(3, 4, 5, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, 5, device=device, dtype=float_dtype)
        nt = NT([a, b])
        # dim1=1, dim2=2 in NestedTensor → dim1=0, dim2=1 in element
        result = torch.diagonal(nt, dim1=1, dim2=2)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.diagonal(a, dim1=0, dim2=1))
        assert_close(result[1], torch.diagonal(b, dim1=0, dim2=1))

    def test_diagonal_static_dims(self, device, float_dtype):
        a = torch.randn(2, 3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 3, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.diagonal(nt, dim1=2, dim2=3)
        assert isinstance(result, NestedTensor)
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
        assert isinstance(result, NestedTensor)
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
        assert isinstance(result, NestedTensor)
        assert_close(result, reference, equal_nan=True)

    def test_matrix_power(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.matrix_power(nt, 3)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.matrix_power(a, 3))
        assert_close(result[1], torch.matrix_power(b, 3))

    def test_matrix_power_batched_ragged(self, device, float_dtype):
        a = torch.randn(2, 3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 3, 3, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.matrix_power(nt, 3)
        reference = NT([torch.matrix_power(a, 3), torch.matrix_power(b, 3)], **reference_options(nt))
        assert isinstance(result, NestedTensor)
        assert_close(result, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_matrix_power_compile_fullgraph_batched_ragged(self, device, float_dtype):
        a = torch.randn(2, 3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 3, 3, device=device, dtype=float_dtype)
        nt = NT([a, b])
        compiled = torch.compile(lambda x: torch.matrix_power(x, 3), backend="inductor", fullgraph=True)
        result = compiled(nt)
        reference = NT([torch.matrix_power(a, 3), torch.matrix_power(b, 3)], **reference_options(nt))
        assert isinstance(result, NestedTensor)
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
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.repeat_interleave(a, 2))
        assert_close(result[1], torch.repeat_interleave(b, 2))

    def test_repeat_interleave_with_dim(self, device, float_dtype):
        a = torch.randn(2, 3, device=device, dtype=float_dtype)
        b = torch.randn(2, 3, device=device, dtype=float_dtype)
        nt = NT([a, b])
        # dim=1 in NestedTensor → dim=0 in element
        result = torch.repeat_interleave(nt, 2, dim=1)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.repeat_interleave(a, 2, dim=0))
        assert_close(result[1], torch.repeat_interleave(b, 2, dim=0))

    def test_trace(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.trace(nt)
        assert isinstance(result, NestedTensor)
        # Scalar results get packed as 1-D elements in NestedTensor
        assert_close(result[0].squeeze(), torch.trace(a))
        assert_close(result[1].squeeze(), torch.trace(b))

    def test_tril(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.tril(nt)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.tril(a))
        assert_close(result[1], torch.tril(b))

    def test_triu(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.triu(nt)
        assert isinstance(result, NestedTensor)
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

    def test_prod_ignores_padding(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([2.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([2.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.prod(nt)
        assert_close(output, torch.tensor(8.0, device=device, dtype=float_dtype))

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

    @pytest.mark.parametrize(("dim", "element_dim"), [(1, 0), (2, 1)])
    def test_multi_ragged_keepdim_reduction_broadcasts_back(self, device, float_dtype, dim, element_dim):
        elements = [
            torch.randn(2, 3, 4, device=device, dtype=float_dtype),
            torch.randn(4, 2, 4, device=device, dtype=float_dtype),
        ]
        nested = NT(elements, ragged_dims=(0, 1))

        reduced = nested.sum(dim=dim, keepdim=True)
        output = nested + reduced
        references = [element + element.sum(dim=element_dim, keepdim=True) for element in elements]

        assert [element.shape for element in reduced] == [
            element.sum(dim=element_dim, keepdim=True).shape for element in elements
        ]
        assert output.ragged_dims == (0, 1)
        for actual, expected in zip(output, references):
            assert_close(actual, expected)

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
        count_reference = torch.stack([torch.count_nonzero(t, dim=(0, 2)) for t in parts]).movedim(0, 1)
        assert_close(count, count_reference)

        summed = torch.sum(nt, dim=(0, 3))
        sum_reference = torch.stack([torch.sum(t, dim=(0, 2)) for t in parts]).movedim(0, 1)
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
                if parts[0].dim() - (0 if keepdim else 1) > 0:
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
        if parts[0].dim() - 1 > 0:
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


class TestSearchsorted:

    def test_bucketize(self, device, float_dtype):
        boundaries = torch.tensor([1.0, 3.0, 5.0], device=device, dtype=float_dtype)
        a = torch.tensor([0.5, 2.0, 4.0, 6.0], device=device, dtype=float_dtype)
        b = torch.tensor([1.5, 3.5], device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.bucketize(nt, boundaries)
        assert isinstance(result, NestedTensor)
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
        assert isinstance(result, NestedTensor)
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
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.searchsorted(boundaries, vals_a))
        assert_close(result[1], torch.searchsorted(boundaries, vals_b))


class TestSelectionOps:

    def test_argmax_and_argmin_default_dim(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 5, 3], device=device, dtype=float_dtype),
                torch.tensor([4, 2], device=device, dtype=float_dtype),
            ]
        )
        output = torch.argmax(nt)
        reference = torch.tensor([1, 0], device=device)
        assert_close(output, reference)
        output = torch.argmin(nt)
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

    def test_max_ties_route_gradient_to_returned_index(self, device, float_dtype):
        high = 2.0
        leaves = [
            torch.tensor([high, high, 0.0], device=device, dtype=float_dtype, requires_grad=True),
            torch.tensor([high, high], device=device, dtype=float_dtype, requires_grad=True),
        ]
        nt = NT(leaves)
        output = torch.max(nt, dim=1)
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
        assert isinstance(output[0], NestedTensor)
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
    def test_compile_fullgraph_preserves_values_and_vjp(self, device):
        left_values = torch.randn(7, 3, device=device, requires_grad=True)
        right_values = torch.randn(5, 3, device=device, requires_grad=True)
        left_parts = left_values.split((2, 5))
        right_parts = right_values.split((4, 1))
        left = NT(left_parts, ragged_dims=(0,), batch_first=False)
        right = NT(right_parts, ragged_dims=(0,), batch_first=False)
        compiled = torch.compile(
            lambda lhs, rhs: lhs.cdist(rhs).concat,
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
    def test_cumprod_compile_fullgraph_backward(self, device):
        values = torch.randn(8, 2, device=device, requires_grad=True)
        parts = values.split((3, 5))
        nested = NT(parts, ragged_dims=(0,))
        compiled = torch.compile(
            lambda value: value.cumprod(dim=1).concat,
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
    def test_rank_compiles(self, device, float_dtype):
        from danling.tensors import rank

        nested = NT(
            [torch.randn(n, device=device, dtype=float_dtype) for n in (3, 5, 2)],
            ragged_dims=(0,),
        )
        compiled = torch.compile(
            lambda value: rank(value).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        assert_close(compiled(nested), rank(nested).concat)


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
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.hypot(a, b))

    def test_logaddexp(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, device=device, dtype=float_dtype)
        nt_a = NT([a, a])
        nt_b = NT([b, b])
        result = torch.logaddexp(nt_a, nt_b)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.logaddexp(a, b))

    def test_logaddexp2(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, device=device, dtype=float_dtype)
        nt_a = NT([a, a])
        nt_b = NT([b, b])
        result = torch.logaddexp2(nt_a, nt_b)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.logaddexp2(a, b))

    def test_nextafter(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, device=device, dtype=float_dtype)
        nt_a = NT([a, a])
        nt_b = NT([b, b])
        result = torch.nextafter(nt_a, nt_b)
        assert isinstance(result, NestedTensor)
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
        assert isinstance(output, NestedTensor)
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
