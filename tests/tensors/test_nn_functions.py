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

import functools

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from danling.tensors import NestedTensor, create_flex_block_mask
from danling.tensors.nn_functions import (
    _concat_tensors,
    _nested_from_padded_tensor,
    _restore_flex_dense_tensor,
    _sdpa_pack_native,
    _sdpa_restore_native,
)
from tests.tensors.utils import (
    assert_close,
    assert_nested_function_matches,
    low_precision_cuda_tolerances,
    nested_rand,
    packed_result,
)

NT = NestedTensor


try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except Exception:
    create_block_mask = None
    flex_attention = None


def _maybe_xfail_upstream_flex_error(exc: Exception) -> None:
    message = str(exc)
    if "Could not guard on data-dependent expression" in message:
        pytest.xfail("Upstream PyTorch FlexAttention nested compile limitation")
    if "block_mask was created for block_mask.shape" in message:
        pytest.xfail("Upstream PyTorch FlexAttention treats DanLing nested inputs as dense for block-mask validation")
    if "Please convert all Tensors to FakeTensors first" in message:
        pytest.xfail("Upstream PyTorch FlexAttention fake-tensor limitation under outer fullgraph compile")
    if "Logger not supported for non-export cases" in message or "logging.Logger method not supported" in message:
        pytest.xfail("Upstream PyTorch FlexAttention logging limitation under outer fullgraph compile")
    if "aten._local_scalar_dense.default" in message:
        pytest.xfail("Upstream PyTorch FlexAttention scalar-output limitation under outer fullgraph compile")


def _make_test_flex_block_mask(lengths: list[int], max_len: int, device, *, is_causal: bool):
    if create_block_mask is None:
        raise RuntimeError("FlexAttention unavailable")
    lengths_tensor = torch.tensor(lengths, device=device, dtype=torch.int32)

    def mask_mod(b, h, q_idx, kv_idx):
        valid = (q_idx < lengths_tensor[b]) & (kv_idx < lengths_tensor[b])
        if is_causal:
            valid = valid & (q_idx >= kv_idx)
        return valid

    return create_block_mask(mask_mod, len(lengths), None, max_len, max_len, device=device, _compile=False)


@functools.lru_cache(maxsize=1)
def _compiled_test_flex_attention():
    if flex_attention is None:
        raise RuntimeError("FlexAttention unavailable")
    return torch.compile(flex_attention, backend="inductor", fullgraph=True)


def _compile_fullgraph(fn):
    return torch.compile(fn, backend="inductor", fullgraph=True)


class TestActivations:

    @pytest.mark.parametrize(
        ("activation", "shape", "kwargs"),
        [
            pytest.param(F.softsign, [(2, 3), (1, 3)], {}, id="softsign"),
            pytest.param(F.tanhshrink, [(2, 3), (1, 3)], {}, id="tanhshrink"),
            pytest.param(F.sigmoid, [(2, 3), (1, 3)], {}, id="sigmoid"),
            pytest.param(F.tanh, [(2, 3), (1, 3)], {}, id="tanh"),
            pytest.param(F.relu6, [(2, 3), (1, 3)], {}, id="relu6"),
            pytest.param(F.elu, [(2, 4), (1, 4)], {}, id="elu"),
            pytest.param(F.celu, [(2, 4), (1, 4)], {}, id="celu"),
            pytest.param(F.selu, [(2, 4), (1, 4)], {}, id="selu"),
            pytest.param(F.relu, [(2, 4), (1, 4)], {"inplace": False}, id="relu"),
            pytest.param(F.leaky_relu, [(2, 4), (1, 4)], {"negative_slope": 0.1}, id="leaky_relu"),
            pytest.param(F.rrelu, [(2, 4), (1, 4)], {"training": False}, id="rrelu"),
            pytest.param(F.glu, [(2, 4), (1, 4)], {"dim": -1}, id="glu"),
            pytest.param(F.gelu, [(2, 3), (1, 3)], {}, id="gelu"),
            pytest.param(F.softplus, [(2, 3), (1, 3)], {}, id="softplus"),
            pytest.param(F.hardsigmoid, [(2, 3), (1, 3)], {}, id="hardsigmoid"),
            pytest.param(F.hardswish, [(2, 3), (1, 3)], {}, id="hardswish"),
            pytest.param(F.hardtanh, [(2, 3), (1, 3)], {}, id="hardtanh"),
            pytest.param(F.softshrink, [(2, 3), (1, 3)], {"lambd": 0.5}, id="softshrink"),
            pytest.param(F.hardshrink, [(2, 3), (1, 3)], {"lambd": 0.5}, id="hardshrink"),
            pytest.param(F.threshold, [(2, 3), (1, 3)], {"threshold": 0.0, "value": -0.1}, id="threshold"),
            pytest.param(F.silu, [(2, 3), (1, 3)], {}, id="silu"),
            pytest.param(F.mish, [(2, 3), (1, 3)], {}, id="mish"),
            pytest.param(F.logsigmoid, [(2, 3), (1, 3)], {}, id="logsigmoid"),
        ],
    )
    def test_matches_tensor(self, activation, shape, kwargs, device, float_dtype):
        nt = nested_rand(shape, device, float_dtype)
        assert_nested_function_matches(activation, nt, **kwargs)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize("activation", [F.relu, F.gelu, F.silu])
    def test_unregistered_activation_compile_fullgraph(self, activation, device, float_dtype):
        nt = nested_rand([(2, 4), (1, 4)], device, float_dtype)

        torch._dynamo.reset()

        def apply(x):
            return activation(x)

        compiled = _compile_fullgraph(apply)
        output = compiled(nt)
        reference = activation(nt.tensor)
        assert_close(output, reference)


class TestAdaptiveAvgPool:

    def test_adaptive_avg_pool1d(self, device, float_dtype):
        input = NT(
            [
                torch.arange(6.0, device=device, dtype=float_dtype).view(1, 1, 6),
                torch.ones(1, 1, 6, device=device, dtype=float_dtype),
            ]
        )
        output = F.adaptive_avg_pool1d(input, output_size=3)
        reference = torch.stack([F.adaptive_avg_pool1d(t, output_size=3) for t in input])
        assert_close(output, reference)

    def test_adaptive_avg_pool2d(self, device, float_dtype):
        input = nested_rand([(1, 3, 3), (1, 3, 3)], device, float_dtype)
        output = F.adaptive_avg_pool2d(input, output_size=(1, 1))
        reference = torch.stack([F.adaptive_avg_pool2d(t, output_size=(1, 1)) for t in input])
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_adaptive_avg_pool3d(self, device, float_dtype):
        input = NT(
            [
                torch.randn(1, 1, 2, 2, 2, device=device, dtype=float_dtype),
                torch.randn(1, 1, 3, 3, 3, device=device, dtype=float_dtype),
            ]
        )
        output = F.adaptive_avg_pool3d(input, output_size=(1, 1, 1))
        reference = torch.stack([F.adaptive_avg_pool3d(t, output_size=(1, 1, 1)) for t in input])
        assert_close(output, reference, atol=1e-6, rtol=1e-6)


class TestAdaptiveMaxPool:

    def test_adaptive_max_pool1d(self, device, float_dtype):
        input = NT(
            [
                torch.arange(1, 7, device=device, dtype=float_dtype).view(1, 1, 6),
                torch.ones(1, 1, 6, device=device, dtype=float_dtype),
            ]
        )
        output = F.adaptive_max_pool1d(input, output_size=3)
        reference = torch.stack([F.adaptive_max_pool1d(t, output_size=3) for t in input])
        assert_close(output, reference)

    def test_adaptive_max_pool2d(self, device, float_dtype):
        input = NT(
            [
                torch.arange(9, device=device, dtype=float_dtype).view(1, 1, 3, 3),
                torch.ones(1, 1, 3, 3, device=device, dtype=float_dtype),
            ]
        )
        output = F.adaptive_max_pool2d(input, output_size=(1, 1))
        reference = torch.stack([F.adaptive_max_pool2d(t, output_size=(1, 1)) for t in input])
        assert_close(output, reference)


class TestAdaptiveMaxPoolWithIndices:

    def test_adaptive_max_pool1d_with_indices(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(1, 7, dtype=float_dtype, device=device).view(1, 1, 6),
                torch.ones(1, 1, 6, device=device, dtype=float_dtype),
            ]
        )
        output, idx = F.adaptive_max_pool1d_with_indices(nt, output_size=3)
        reference_output, reference_idx = zip(
            *[F.adaptive_max_pool1d(t, output_size=3, return_indices=True) for t in nt]
        )
        assert_close(output, torch.stack(reference_output))
        assert_close(idx, torch.stack(reference_idx))

    def test_adaptive_max_pool2d_with_indices(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(1, 1, 3, 3, device=device, dtype=float_dtype),
                torch.randn(1, 1, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output, idx = F.adaptive_max_pool2d_with_indices(nt, output_size=(2, 2))
        reference_output, reference_idx = zip(
            *[F.adaptive_max_pool2d(t, output_size=(2, 2), return_indices=True) for t in nt]
        )
        assert_close(output, torch.stack(reference_output), atol=1e-6, rtol=1e-6)
        assert_close(idx, torch.stack(reference_idx))

    def test_adaptive_max_pool3d_with_indices(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(1, 2, 2, 2, 2, device=device, dtype=float_dtype),
                torch.randn(1, 2, 2, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output, idx = F.adaptive_max_pool3d_with_indices(nt, output_size=(1, 1, 1))
        reference_output, reference_idx = zip(
            *[F.adaptive_max_pool3d(t, output_size=(1, 1, 1), return_indices=True) for t in nt]
        )
        assert_close(output, torch.stack(reference_output))
        assert_close(idx, torch.stack(reference_idx))


class TestAvgPool:

    def test_avg_pool1d(self, device, float_dtype):
        input = NT(
            [
                torch.arange(8, device=device, dtype=float_dtype).reshape(1, 1, 8),
                torch.ones(1, 1, 8, device=device, dtype=float_dtype),
            ]
        )
        output = F.avg_pool1d(input, kernel_size=2, stride=2)
        reference = torch.stack([F.avg_pool1d(t, kernel_size=2, stride=2) for t in input])
        assert_close(output, reference)

    def test_avg_pool2d(self, device, float_dtype):
        input = NT(
            [
                torch.arange(16.0, device=device, dtype=float_dtype).view(1, 1, 4, 4),
                torch.ones(1, 1, 4, 4, device=device, dtype=float_dtype),
            ]
        )
        output = F.avg_pool2d(input, kernel_size=2, stride=2)
        reference = torch.stack([F.avg_pool2d(t, kernel_size=2, stride=2) for t in input])
        assert_close(output, reference)

    def test_avg_pool3d(self, device, float_dtype):
        input = NT(
            [
                torch.arange(8.0, device=device, dtype=float_dtype).view(1, 1, 2, 2, 2),
                torch.ones(1, 1, 2, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.avg_pool3d(input, kernel_size=2)
        reference = torch.stack([F.avg_pool3d(t, kernel_size=2) for t in input])
        assert_close(output, reference)


class TestBilinear:

    def test_bilinear(self, device, float_dtype):
        x1 = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        x2 = nested_rand([(2, 4), (1, 4)], device, float_dtype)
        weight = torch.randn(5, 3, 4, device=device, dtype=float_dtype)
        bias = torch.randn(5, device=device, dtype=float_dtype)
        output = F.bilinear(x1, x2, weight, bias)
        reference = NT([F.bilinear(a, b, weight, bias) for a, b in zip(x1, x2)], **x1._meta())
        assert_close(output, reference, atol=1e-5, rtol=1e-5)


class TestChannelShuffle:

    def test_channel_shuffle(self, device, float_dtype):
        x = NT(
            [
                torch.randn(4, 2, 2, device=device, dtype=float_dtype),
                torch.randn(4, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.channel_shuffle(x, groups=2)
        reference = NT([F.channel_shuffle(t, groups=2) for t in x], **x._meta())
        assert_close(output, reference)


class TestClassificationLosses:

    def test_binary_cross_entropy(self, device, float_dtype):
        logits = NT(
            [
                torch.rand(2, 3, device=device, dtype=float_dtype),
                torch.rand(1, 3, device=device, dtype=float_dtype),
            ]
        )
        targets = NT(
            [
                torch.rand(2, 3, device=device, dtype=float_dtype),
                torch.rand(1, 3, device=device, dtype=float_dtype),
            ]
        )
        output = F.binary_cross_entropy(logits, targets, reduction="sum")
        reference = F.binary_cross_entropy(
            torch.cat(tuple(logits), dim=0),
            torch.cat(tuple(targets), dim=0),
            reduction="sum",
        )
        assert_close(output, reference)

    def test_binary_cross_entropy_with_logits(self, device, float_dtype):
        logits = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        targets = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        output = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
        reference = F.binary_cross_entropy_with_logits(
            torch.cat(tuple(logits), dim=0),
            torch.cat(tuple(targets), dim=0),
            reduction="sum",
        )
        assert_close(output, reference)

    def test_binary_cross_entropy_with_logits_after_method_squeeze_preserves_grad(self, device, float_dtype):
        logits = NT(
            [
                torch.randn(2, 3, 1, device=device, dtype=float_dtype, requires_grad=True),
                torch.randn(1, 3, 1, device=device, dtype=float_dtype, requires_grad=True),
            ]
        )
        targets = NT(
            [
                torch.rand(2, 3, 1, device=device, dtype=float_dtype),
                torch.rand(1, 3, 1, device=device, dtype=float_dtype),
            ]
        )
        output = F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets.squeeze(-1), reduction="mean")
        assert output.requires_grad
        assert output.grad_fn is not None

    def test_cross_entropy_loss(self, device, float_dtype):
        logits = NT(
            [
                torch.tensor([[2.0, 0.5], [0.1, 1.0]], device=device, dtype=float_dtype),
                torch.tensor([[1.0, 0.0]], device=device, dtype=float_dtype),
            ]
        )
        targets = NT(
            [torch.tensor([0, 1], device=device, dtype=torch.long), torch.tensor([1], device=device, dtype=torch.long)]
        )
        output = F.cross_entropy(logits, targets, reduction="sum")
        reference_input = torch.cat(tuple(logits), dim=0)
        reference_target = torch.cat(tuple(targets), dim=0)
        reference = F.cross_entropy(reference_input, reference_target, reduction="sum")
        assert_close(output, reference)

    def test_kl_div(self, device, float_dtype):
        p = NT(
            [
                torch.log_softmax(torch.tensor([[0.2, 0.8]], device=device, dtype=float_dtype), dim=-1),
                torch.log_softmax(torch.tensor([[0.5, 0.5]], device=device, dtype=float_dtype), dim=-1),
            ]
        )
        q = NT(
            [
                torch.tensor([[0.3, 0.7]], device=device, dtype=float_dtype),
                torch.tensor([[0.4, 0.6]], device=device, dtype=float_dtype),
            ]
        )
        output = F.kl_div(p, q, reduction="sum", log_target=False)
        reference = F.kl_div(
            torch.cat(tuple(p), dim=0),
            torch.cat(tuple(q), dim=0),
            reduction="sum",
            log_target=False,
        )
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_multi_margin_loss(self, device, float_dtype):
        inp = NT(
            [
                torch.tensor([[0.2, 0.8, 0.1]], device=device, dtype=float_dtype),
                torch.tensor([[0.5, 0.3, 0.2]], device=device, dtype=float_dtype),
            ]
        )
        tgt = NT(
            [torch.tensor([1], device=device, dtype=torch.long), torch.tensor([0], device=device, dtype=torch.long)]
        )
        output = F.multi_margin_loss(inp, tgt, reduction="sum")
        reference = F.multi_margin_loss(torch.cat(tuple(inp), dim=0), torch.cat(tuple(tgt), dim=0), reduction="sum")
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_multilabel_margin_loss(self, device, float_dtype):
        inp = NT(
            [
                torch.tensor([[0.2, 0.5, 0.1]], device=device, dtype=float_dtype),
                torch.tensor([[0.3, 0.4, 0.2]], device=device, dtype=float_dtype),
            ]
        )
        tgt = NT(
            [
                torch.tensor([[1, 0, -1]], device=device, dtype=torch.long),
                torch.tensor([[0, 2, -1]], device=device, dtype=torch.long),
            ]
        )
        output = F.multilabel_margin_loss(inp, tgt, reduction="sum")
        reference = F.multilabel_margin_loss(
            torch.cat(tuple(inp), dim=0),
            torch.cat(tuple(tgt), dim=0),
            reduction="sum",
        )
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_multilabel_soft_margin_loss(self, device, float_dtype):
        inp = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        tgt = NT(
            [
                torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=float_dtype),
                torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=float_dtype),
            ]
        )
        weight = torch.tensor([1.0, 0.5, 2.0], device=device, dtype=float_dtype)
        output = F.multilabel_soft_margin_loss(inp, tgt, weight=weight, reduction="sum")
        reference = F.multilabel_soft_margin_loss(
            torch.cat(tuple(inp), dim=0),
            torch.cat(tuple(tgt), dim=0),
            weight=weight,
            reduction="sum",
        )
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_nll_loss(self, device, float_dtype):
        logits = NT(
            [
                torch.tensor([[2.0, 0.5], [0.1, 1.0]], device=device, dtype=float_dtype),
                torch.tensor([[1.0, 0.0]], device=device, dtype=float_dtype),
            ]
        )
        log_probs = NT([torch.log_softmax(t, dim=-1) for t in logits], **logits._meta())
        targets = NT(
            [torch.tensor([0, 1], device=device, dtype=torch.long), torch.tensor([1], device=device, dtype=torch.long)]
        )
        output = F.nll_loss(log_probs, targets, reduction="sum")
        reference_input = torch.cat(tuple(logits), dim=0)
        reference_target = torch.cat(tuple(targets), dim=0)
        reference = F.nll_loss(reference_input.log_softmax(dim=-1), reference_target, reduction="sum")
        assert_close(output, reference)

    def test_soft_margin_loss(self, device, float_dtype):
        inp = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        tgt = NT(
            [
                torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device, dtype=float_dtype),
                torch.tensor([[1.0, 1.0, -1.0]], device=device, dtype=float_dtype),
            ]
        )
        output = F.soft_margin_loss(inp, tgt, reduction="sum")
        reference = F.soft_margin_loss(torch.cat(tuple(inp), dim=0), torch.cat(tuple(tgt), dim=0), reduction="sum")
        assert_close(output, reference, atol=1e-6, rtol=1e-6)


class TestCompile:

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_nn_functional_compile_matches_reference(self):
        nt = NT(
            [
                torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
                torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
            ]
        )
        weight = torch.tensor([[0.2, -0.5], [1.1, 0.3]])
        bias = torch.tensor([0.4, -0.2])

        def _compile(fn):
            return torch.compile(fn, backend="inductor", fullgraph=True)

        linear_fn = _compile(lambda x: F.linear(x, weight, bias))
        softmax_fn = _compile(lambda x: F.softmax(x, dim=1))
        log_softmax_fn = _compile(lambda x: F.log_softmax(x, dim=1))
        layer_norm_fn = _compile(lambda x: F.layer_norm(x, (2,)))
        rms_norm_fn = _compile(lambda x: F.rms_norm(x, (2,)))
        linear_comp = linear_fn(nt)
        softmax_comp = softmax_fn(nt)
        log_softmax_comp = log_softmax_fn(nt)
        layer_norm_comp = layer_norm_fn(nt)
        rms_norm_comp = rms_norm_fn(nt)

        ref_linear = NT([F.linear(t, weight, bias) for t in nt], **nt._meta())
        ref_softmax = NT([F.softmax(t, dim=0) for t in nt], **nt._meta())
        ref_log_softmax = NT([F.log_softmax(t, dim=0) for t in nt], **nt._meta())
        ref_layer_norm = NT([F.layer_norm(t, (2,)) for t in nt], **nt._meta())
        ref_rms_norm = NT([F.rms_norm(t, (2,)) for t in nt], **nt._meta())
        assert isinstance(linear_comp, NestedTensor)
        assert isinstance(softmax_comp, NestedTensor)
        assert isinstance(log_softmax_comp, NestedTensor)
        assert isinstance(layer_norm_comp, NestedTensor)
        assert isinstance(rms_norm_comp, NestedTensor)
        assert linear_comp._has_same_layout(ref_linear)
        assert softmax_comp._has_same_layout(ref_softmax)
        assert log_softmax_comp._has_same_layout(ref_log_softmax)
        assert layer_norm_comp._has_same_layout(ref_layer_norm)
        assert rms_norm_comp._has_same_layout(ref_rms_norm)
        assert_close(linear_comp, ref_linear)
        assert_close(softmax_comp, ref_softmax)
        assert_close(log_softmax_comp, ref_log_softmax)
        assert_close(layer_norm_comp, ref_layer_norm)
        assert_close(rms_norm_comp, ref_rms_norm)


class TestConcatTensors:

    def test__concat_tensors_with_plain_tensors(self, device, float_dtype):
        first = torch.arange(4, device=device, dtype=float_dtype).reshape(2, 2)
        second = torch.arange(4, 8, device=device, dtype=float_dtype).reshape(2, 2)
        out_first, out_second = _concat_tensors(first, second)
        assert_close(out_first, first)
        assert_close(out_second, second)


class TestConv:

    @pytest.mark.parametrize("shape", [[(5, 8), (7, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("groups", [1, 2])
    def test_conv1d(self, shape, kernel_size, stride, padding, dilation, groups, device, float_dtype):
        base = nested_rand(shape, device, float_dtype)
        weight = torch.randn(4, base.shape[-1] // groups, kernel_size, device=device, dtype=float_dtype)
        bias = torch.randn(4, device=device, dtype=float_dtype)
        input = base.transpose(-1, -2)
        output = F.conv1d(input, weight, bias, stride, padding, dilation, groups)
        reference = NT([F.conv1d(t, weight, bias, stride, padding, dilation, groups) for t in input], **input._meta())
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_conv1d_batch_first_false(self, device, float_dtype):
        input = NT(
            [
                torch.randn(2, 5, device=device, dtype=float_dtype),
                torch.randn(2, 5, device=device, dtype=float_dtype),
            ],
            batch_first=False,
        )
        weight = torch.randn(4, 2, 3, device=device, dtype=float_dtype)
        bias = torch.randn(4, device=device, dtype=float_dtype)
        output = F.conv1d(input, weight, bias, stride=1, padding=1)
        reference = NT([F.conv1d(t, weight, bias, stride=1, padding=1) for t in input], **input._meta())
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_conv1d_ragged_nonzero_padding_value(self, device, float_dtype):
        input = NT(
            [
                torch.randn(2, 5, device=device, dtype=float_dtype),
                torch.randn(2, 3, device=device, dtype=float_dtype),
            ],
            padding_value=7.0,
        )
        weight = torch.randn(4, 2, 3, device=device, dtype=float_dtype)
        bias = torch.randn(4, device=device, dtype=float_dtype)
        output = F.conv1d(input, weight, bias, stride=1, padding=1)
        reference = NT([F.conv1d(t, weight, bias, stride=1, padding=1) for t in input], **input._meta())
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("shape", [[(5, 7, 8), (11, 13, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("groups", [1, 2])
    def test_conv2d(self, shape, kernel_size, stride, padding, dilation, groups, device, float_dtype):
        base = nested_rand(shape, device, float_dtype)
        weight = torch.randn(4, base.shape[-1] // groups, kernel_size, kernel_size, device=device, dtype=float_dtype)
        bias = torch.randn(4, device=device, dtype=float_dtype)
        input = base.transpose(1, -1)
        output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        reference = NT([F.conv2d(t, weight, bias, stride, padding, dilation, groups) for t in input], **input._meta())
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("shape", [[(5, 7, 9, 8), (11, 13, 15, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("groups", [1, 2])
    def test_conv3d(self, shape, kernel_size, stride, padding, dilation, groups, device, float_dtype):
        base = nested_rand(shape, device, float_dtype)
        weight = torch.randn(
            4, base.shape[-1] // groups, kernel_size, kernel_size, kernel_size, device=device, dtype=float_dtype
        )
        bias = torch.randn(4, device=device, dtype=float_dtype)
        input = base.permute(0, 4, 1, 2, 3)
        output = F.conv3d(input, weight, bias, stride, padding, dilation, groups)
        reference = NT([F.conv3d(t, weight, bias, stride, padding, dilation, groups) for t in input], **input._meta())
        assert_close(output, reference, atol=1e-5, rtol=1e-5)


class TestConvTranspose:

    @pytest.mark.parametrize("shape", [[(5, 8), (7, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("output_padding", [0, 1])
    @pytest.mark.parametrize("groups", [1, 2])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_conv_transpose1d_functional(
        self, shape, kernel_size, stride, padding, output_padding, groups, dilation, device, float_dtype
    ):
        if stride == 1 and output_padding > 0:
            pytest.skip("output_padding > 0 only valid when stride > 1")
        input = nested_rand(shape, device, float_dtype)
        weight = torch.randn(input.shape[-1], 4 // groups, kernel_size, device=device, dtype=float_dtype)
        bias = torch.randn(4, device=device, dtype=float_dtype)
        input = input.transpose(1, -1)
        output = F.conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation)
        reference = NT(
            [F.conv_transpose1d(t, weight, bias, stride, padding, output_padding, groups, dilation) for t in input],
            **input._meta(),
        )
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("shape", [[(5, 7, 8), (11, 13, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("output_padding", [0, 1])
    @pytest.mark.parametrize("groups", [1, 2])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_conv_transpose2d_functional(
        self, shape, kernel_size, stride, padding, output_padding, dilation, groups, device, float_dtype
    ):
        if stride == 1 and output_padding > 0:
            pytest.skip("output_padding > 0 only valid when stride > 1")
        input = nested_rand(shape, device, float_dtype)
        weight = torch.randn(input.shape[-1], 4 // groups, kernel_size, kernel_size, device=device, dtype=float_dtype)
        bias = torch.randn(4, device=device, dtype=float_dtype)
        input = input.permute(0, 3, 1, 2)
        output = F.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)
        reference = NT(
            [F.conv_transpose2d(t, weight, bias, stride, padding, output_padding, groups, dilation) for t in input],
            **input._meta(),
        )
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("shape", [[(5, 7, 9, 8), (11, 13, 15, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("output_padding", [0, 1])
    @pytest.mark.parametrize("groups", [1, 2])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_conv_transpose3d_functional(
        self, shape, kernel_size, stride, padding, output_padding, dilation, groups, device, float_dtype
    ):
        if stride == 1 and output_padding > 0:
            pytest.skip("output_padding > 0 only valid when stride > 1")
        input = nested_rand(shape, device, float_dtype)
        weight = torch.randn(
            input.shape[-1], 4 // groups, kernel_size, kernel_size, kernel_size, device=device, dtype=float_dtype
        )
        bias = torch.randn(4, device=device, dtype=float_dtype)
        input = input.transpose(1, -1)
        output = F.conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation)
        reference = NT(
            [F.conv_transpose3d(t, weight, bias, stride, padding, output_padding, groups, dilation) for t in input],
            **input._meta(),
        )
        assert_close(output, reference, atol=1e-5, rtol=1e-5)


class TestDropout:

    def test_dropout_eval_is_identity(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        output = F.dropout(nt, p=0.7, training=False)
        assert_close(output, nt)

    def test_dropout_training(self, device, float_dtype):
        nt = NT(
            [
                torch.ones(10, 20, device=device, dtype=float_dtype),
                torch.ones(8, 20, device=device, dtype=float_dtype),
            ]
        )
        torch.manual_seed(1016)
        output = F.dropout(nt, p=0.5, training=True)
        torch.manual_seed(1016)
        reference = packed_result(nt, F.dropout(nt._values, p=0.5, training=True))
        assert_close(output, reference)

    def test_dropout_variants_eval_is_identity(self, device, float_dtype):
        """All dropout variants are identity in eval mode."""
        nt_3d = NT(
            [
                torch.randn(1, 3, 4, device=device, dtype=float_dtype),
                torch.randn(1, 3, 4, device=device, dtype=float_dtype),
            ]
        )
        assert_close(F.dropout1d(nt_3d, p=0.2, training=False), nt_3d)

        nt_4d = NT(
            [
                torch.randn(1, 2, 2, 2, device=device, dtype=float_dtype),
                torch.randn(1, 2, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        assert_close(F.dropout2d(nt_4d, p=0.3, training=False), nt_4d)

        nt_5d = NT(
            [
                torch.randn(1, 2, 2, 2, 2, device=device, dtype=float_dtype),
                torch.randn(1, 2, 2, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        assert_close(F.dropout3d(nt_5d, p=0.4, training=False), nt_5d)

        nt_2d = nested_rand([(2, 3), (2, 3)], device, float_dtype)
        assert_close(F.alpha_dropout(nt_2d, p=0.1, training=False), nt_2d)
        assert_close(F.feature_alpha_dropout(nt_4d, p=0.25, training=False), nt_4d)


class TestEmbeddingOps:

    def test_embedding(self, device, float_dtype):
        weight = torch.randn(10, 4, device=device, dtype=float_dtype)
        nt_idx = NT(
            [
                torch.tensor([1, 3, 5], dtype=torch.long, device=device),
                torch.tensor([0, 2], dtype=torch.long, device=device),
            ]
        )
        output = F.embedding(nt_idx, weight)
        reference = NT([F.embedding(t, weight) for t in nt_idx], **nt_idx._meta())
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_embedding_bag(self, device, float_dtype):
        weight = torch.randn(10, 4, device=device, dtype=float_dtype)
        nt_idx = NT(
            [
                torch.tensor([1, 3, 5], dtype=torch.long, device=device),
                torch.tensor([0, 2], dtype=torch.long, device=device),
            ]
        )
        offsets = torch.tensor([0], dtype=torch.long, device=device)
        output = F.embedding_bag(nt_idx, weight, offsets=offsets, mode="mean")
        reference = NT([F.embedding_bag(t, weight, offsets=offsets, mode="mean") for t in nt_idx], **nt_idx._meta())
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_embedding_bag_compile_fullgraph_default_offsets(self, device, float_dtype):
        weight = torch.randn(16, 8, device=device, dtype=float_dtype)
        nt_idx = NT(
            [
                torch.tensor([1, 3, 5], dtype=torch.long, device=device),
                torch.tensor([0, 2, 4, 6], dtype=torch.long, device=device),
            ]
        )
        compiled = torch.compile(lambda x, w: F.embedding_bag(x, w, mode="mean"), backend="inductor", fullgraph=True)
        output = compiled(nt_idx, weight)
        reference = NT(
            [F.embedding_bag(t, weight, offsets=torch.tensor([0], device=device), mode="mean") for t in nt_idx],
            **nt_idx._meta(),
        )
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_embedding_bag_shared_offsets(self, device, float_dtype):
        weight = torch.randn(16, 8, device=device, dtype=float_dtype)
        nt_idx = NT(
            [
                torch.tensor([1, 3, 5, 7], dtype=torch.long, device=device),
                torch.tensor([0, 2, 4, 6], dtype=torch.long, device=device),
            ]
        )
        offsets = torch.tensor([0, 2], dtype=torch.long, device=device)
        output = F.embedding_bag(nt_idx, weight, offsets=offsets, mode="sum")
        reference = NT([F.embedding_bag(t, weight, offsets=offsets, mode="sum") for t in nt_idx], **nt_idx._meta())
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_embedding_bag_packed(self, device, float_dtype):
        weight = torch.randn(16, 8, device=device, dtype=float_dtype)
        nt_idx = NT(
            [
                torch.tensor([1, 3, 5], dtype=torch.long, device=device),
                torch.tensor([0, 2, 4, 6], dtype=torch.long, device=device),
            ]
        )
        output = F.embedding_bag(nt_idx, weight, mode="mean")
        reference = NT(
            [F.embedding_bag(t, weight, offsets=torch.tensor([0], device=device), mode="mean") for t in nt_idx],
            **nt_idx._meta(),
        )
        assert_close(output, reference, atol=1e-6, rtol=1e-6)


class TestFractionalMaxPool:

    def test_fractional_max_pool2d(self, device, float_dtype):
        x = NT(
            [
                torch.randn(1, 4, 4, device=device, dtype=float_dtype),
                torch.randn(1, 4, 4, device=device, dtype=float_dtype),
            ]
        )
        random_samples = torch.tensor([[[0.5, 0.5]]], dtype=float_dtype, device=device)
        output = F.fractional_max_pool2d(x, kernel_size=2, output_size=2, _random_samples=random_samples)
        reference = NT(
            [F.fractional_max_pool2d(t, kernel_size=2, output_size=2, _random_samples=random_samples) for t in x],
            **x._meta(),
        )
        assert_close(output, reference)

    def test_fractional_max_pool3d(self, device, float_dtype):
        x = NT(
            [
                torch.randn(1, 4, 4, 4, device=device, dtype=float_dtype),
                torch.randn(1, 4, 4, 4, device=device, dtype=float_dtype),
            ]
        )
        random_samples = torch.tensor([[[0.3, 0.3, 0.3]]], dtype=float_dtype, device=device)
        output = F.fractional_max_pool3d(x, kernel_size=2, output_size=2, _random_samples=random_samples)
        reference = NT(
            [F.fractional_max_pool3d(t, kernel_size=2, output_size=2, _random_samples=random_samples) for t in x],
            **x._meta(),
        )
        assert_close(output, reference)

    def test_fractional_max_pool2d_with_indices(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(16, dtype=float_dtype, device=device).view(1, 1, 4, 4),
                torch.ones(1, 1, 4, 4, device=device, dtype=float_dtype),
            ]
        )
        random_samples = torch.tensor([[[0.5, 0.5]]], dtype=float_dtype, device=device)
        output, idx = F.fractional_max_pool2d_with_indices(
            nt, kernel_size=2, output_size=2, _random_samples=random_samples
        )
        reference_output, reference_idx = zip(
            *[
                F.fractional_max_pool2d(
                    t, kernel_size=2, output_size=2, _random_samples=random_samples, return_indices=True
                )
                for t in nt
            ]
        )
        assert_close(output, torch.stack(reference_output), atol=1e-6, rtol=1e-6)
        assert_close(idx, torch.stack(reference_idx))

    def test_fractional_max_pool3d_with_indices(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(64, dtype=float_dtype, device=device).view(1, 1, 4, 4, 4),
                torch.ones(1, 1, 4, 4, 4, device=device, dtype=float_dtype),
            ]
        )
        random_samples = torch.tensor([[[0.3, 0.3, 0.3]]], dtype=float_dtype, device=device)
        output, idx = F.fractional_max_pool3d_with_indices(
            nt, kernel_size=2, output_size=2, _random_samples=random_samples
        )
        reference_output, reference_idx = zip(
            *[
                F.fractional_max_pool3d(
                    t, kernel_size=2, output_size=2, _random_samples=random_samples, return_indices=True
                )
                for t in nt
            ]
        )
        reference_output = NT(reference_output, **nt._meta())
        reference_idx = NT(reference_idx, **nt._meta())
        assert_close(output, reference_output)
        assert_close(idx, reference_idx)


class TestGridOps:

    def test_affine_grid(self, device, float_dtype):
        imgs = [
            torch.arange(4.0, device=device, dtype=float_dtype).view(1, 1, 2, 2),
            torch.arange(4.0, 8.0, device=device, dtype=float_dtype).view(1, 1, 2, 2),
        ]
        thetas = [torch.eye(2, 3, device=device, dtype=float_dtype).unsqueeze(0) for _ in imgs]
        nt_theta = NT(thetas)
        grids = F.affine_grid(nt_theta, size=imgs[0].shape, align_corners=False)
        nt_imgs = NT(imgs)
        output = F.grid_sample(nt_imgs, grids, align_corners=False)
        reference = NT(
            [F.grid_sample(img, grid, align_corners=False) for img, grid in zip(nt_imgs, grids)],
            **nt_imgs._meta(),
        )
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_grid_sample_tensor_grid(self, device, float_dtype):
        imgs = [
            torch.arange(4.0, device=device, dtype=float_dtype).view(1, 1, 2, 2),
            torch.arange(4.0, 8.0, device=device, dtype=float_dtype).view(1, 1, 2, 2),
        ]
        nt_imgs = NT(imgs)
        grid = torch.zeros(1, 2, 2, 2, device=device, dtype=float_dtype)
        output = F.grid_sample(nt_imgs, grid, align_corners=False)
        reference = torch.stack([F.grid_sample(img, grid, align_corners=False) for img in nt_imgs])
        assert_close(output, reference, atol=1e-6, rtol=1e-6)


class TestInterpolate:

    def test_interpolate_bilinear(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(4.0, device=device, dtype=float_dtype).view(1, 1, 2, 2),
                torch.ones(1, 1, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.interpolate(nt, scale_factor=2, mode="bilinear", align_corners=False)
        reference = NT(
            [F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False) for t in nt],
            **nt._meta(),
        )
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_interpolate_nearest(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(4.0, device=device, dtype=float_dtype).view(1, 1, 2, 2),
                torch.ones(1, 1, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.interpolate(nt, scale_factor=2, mode="nearest")
        reference = NT([F.interpolate(t, scale_factor=2, mode="nearest") for t in nt], **nt._meta())
        assert_close(output, reference)


class TestLinear:

    @pytest.mark.parametrize("shape", [[(3, 5), (3, 5)], [(3, 5), (2, 5)], [(2, 3, 5), (3, 2, 5)]])
    def test_linear(self, shape):
        input = NT([torch.randn(*i) for i in shape])
        weight = torch.randn(3, input.shape[-1])
        bias = torch.randn(3)
        output = F.linear(input, weight, bias)
        reference = F.linear(input.tensor, weight, bias)
        assert_close(output, reference)

    def test_linear_1d(self):
        input = NT([torch.randn(5), torch.randn(5)])
        weight = torch.randn(3, 5)
        bias = torch.randn(3)
        output = F.linear(input, weight, bias)
        reference = NT([F.linear(t, weight, bias) for t in input], **input._meta())
        assert_close(output, reference)


class TestLpPool:

    def test_lp_pool1d(self, device, float_dtype):
        input = NT(
            [
                torch.arange(1, 5, device=device, dtype=float_dtype).view(1, 1, 4),
                torch.ones(1, 1, 4, device=device, dtype=float_dtype),
            ]
        )
        output = F.lp_pool1d(input, 2, kernel_size=2, stride=2)
        reference = torch.stack([F.lp_pool1d(t, 2, kernel_size=2, stride=2) for t in input])
        assert_close(output, reference)

    def test_lp_pool2d(self, device, float_dtype):
        input = NT(
            [
                torch.arange(16, device=device, dtype=float_dtype).view(1, 1, 4, 4),
                torch.ones(1, 1, 4, 4, device=device, dtype=float_dtype),
            ]
        )
        output = F.lp_pool2d(input, 2, kernel_size=2, stride=2)
        reference = torch.stack([F.lp_pool2d(t, 2, kernel_size=2, stride=2) for t in input])
        assert_close(output, reference)

    def test_lp_pool3d(self, device, float_dtype):
        input = NT(
            [
                torch.arange(8, device=device, dtype=float_dtype).view(1, 1, 2, 2, 2),
                torch.ones(1, 1, 2, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.lp_pool3d(input, 2, kernel_size=2)
        reference = torch.stack([F.lp_pool3d(t, 2, kernel_size=2) for t in input])
        assert_close(output, reference)


class TestMaxPool:

    def test_max_pool1d(self, device, float_dtype):
        input = NT(
            [
                torch.arange(1, 7, device=device, dtype=float_dtype).view(1, 1, 6),
                torch.ones(1, 1, 6, device=device, dtype=float_dtype),
            ]
        )
        output = F.max_pool1d(input, kernel_size=2, stride=2)
        reference = torch.stack([F.max_pool1d(t, kernel_size=2, stride=2) for t in input])
        assert_close(output, reference)

    def test_max_pool2d(self, device, float_dtype):
        input = NT(
            [
                torch.arange(16, device=device, dtype=float_dtype).reshape(1, 4, 4),
                torch.arange(16, 32, device=device, dtype=float_dtype).reshape(1, 4, 4),
            ]
        )
        output = F.max_pool2d(input, kernel_size=2)
        reference = torch.stack([F.max_pool2d(t, kernel_size=2) for t in input])
        assert_close(output, reference)

    def test_max_pool3d(self, device, float_dtype):
        input = NT(
            [
                torch.arange(8, device=device, dtype=float_dtype).view(1, 1, 2, 2, 2),
                torch.ones(1, 1, 2, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.max_pool3d(input, kernel_size=2)
        reference = torch.stack([F.max_pool3d(t, kernel_size=2) for t in input])
        assert_close(output, reference)


class TestMaxPoolWithIndices:

    def test_max_pool1d_with_indices(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(1, 7, dtype=float_dtype, device=device).view(1, 1, 6),
                torch.ones(1, 1, 6, device=device, dtype=float_dtype),
            ]
        )
        output, idx = F.max_pool1d_with_indices(nt, kernel_size=2, stride=2)
        reference_output, reference_idx = zip(
            *[F.max_pool1d(t, kernel_size=2, stride=2, return_indices=True) for t in nt]
        )
        assert_close(output, torch.stack(reference_output))
        assert_close(idx, torch.stack(reference_idx))

    def test_max_pool2d_with_indices(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(16, dtype=float_dtype, device=device).view(1, 1, 4, 4),
                torch.arange(16, 32, dtype=float_dtype, device=device).view(1, 1, 4, 4),
            ]
        )
        output, idx = F.max_pool2d_with_indices(nt, kernel_size=2, stride=2)
        reference_output, reference_idx = zip(
            *[F.max_pool2d(t, kernel_size=2, stride=2, return_indices=True) for t in nt]
        )
        assert_close(output, torch.stack(reference_output), atol=1e-6, rtol=1e-6)
        assert_close(idx, torch.stack(reference_idx))

    def test_max_pool3d_with_indices(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(8, dtype=float_dtype, device=device).view(1, 1, 2, 2, 2),
                torch.ones(1, 1, 2, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output, idx = F.max_pool3d_with_indices(nt, kernel_size=2)
        reference_output, reference_idx = zip(*[F.max_pool3d(t, kernel_size=2, return_indices=True) for t in nt])
        assert_close(output, torch.stack(reference_output))
        assert_close(idx, torch.stack(reference_idx))


class TestMaxUnpool:

    def test_max_unpool1d_nested_indices(self):
        orig = [
            torch.arange(1, 5, dtype=torch.float32).reshape(1, 1, 4),
            torch.arange(5, 9, dtype=torch.float32).reshape(1, 1, 4),
        ]
        pooled_indices = [F.max_pool1d(t, kernel_size=2, stride=2, return_indices=True) for t in orig]
        pooled = [p[0] for p in pooled_indices]
        indices = [p[1] for p in pooled_indices]
        pooled_nt = NT(pooled)
        indices_nt = NT(indices)
        unpooled = F.max_unpool1d(pooled_nt, indices_nt, kernel_size=2, stride=2, output_size=orig[0].shape)
        reference = torch.stack(
            [
                F.max_unpool1d(pooled[i], indices[i], kernel_size=2, stride=2, output_size=orig[i].shape)
                for i in range(2)
            ]
        )
        assert_close(unpooled, reference)

    def test_max_unpool1d_tensor_indices(self):
        orig = torch.arange(1, 5, dtype=torch.float32).reshape(1, 1, 4)
        pooled, idx = F.max_pool1d(orig, kernel_size=2, stride=2, return_indices=True)
        pooled_nt = NT([pooled])
        unpooled = F.max_unpool1d(pooled_nt, idx, kernel_size=2, stride=2, output_size=orig.shape)
        reference = F.max_unpool1d(pooled, idx, kernel_size=2, stride=2, output_size=orig.shape)
        reference = NT([reference], **unpooled._meta())
        assert_close(unpooled, reference)

    def test_max_unpool2d_nested_indices(self):
        orig = [
            torch.arange(1, 10, dtype=torch.float32).view(1, 1, 3, 3),
            torch.arange(10, 19, dtype=torch.float32).view(1, 1, 3, 3),
        ]
        pooled_indices = [F.max_pool2d(t, kernel_size=2, stride=1, return_indices=True) for t in orig]
        pooled = [p[0] for p in pooled_indices]
        indices = [p[1] for p in pooled_indices]
        pooled_nt = NT(pooled)
        indices_nt = NT(indices)
        unpooled = F.max_unpool2d(pooled_nt, indices_nt, kernel_size=2, stride=1, output_size=orig[0].shape)
        reference = torch.stack(
            [
                F.max_unpool2d(pooled[i], indices[i], kernel_size=2, stride=1, output_size=orig[i].shape)
                for i in range(2)
            ]
        )
        assert_close(unpooled, reference)

    def test_max_unpool2d_tensor_indices(self):
        orig = torch.arange(1, 10, dtype=torch.float32).view(1, 1, 3, 3)
        pooled, idx = F.max_pool2d(orig, kernel_size=2, stride=1, return_indices=True)
        pooled_nt = NT([pooled])
        unpooled = F.max_unpool2d(pooled_nt, idx, kernel_size=2, stride=1, output_size=orig.shape)
        reference = F.max_unpool2d(pooled, idx, kernel_size=2, stride=1, output_size=orig.shape)
        reference = NT([reference], **unpooled._meta())
        assert_close(unpooled, reference)

    def test_max_unpool3d_nested_indices(self):
        orig = [
            torch.arange(1, 9, dtype=torch.float32).view(1, 1, 2, 2, 2),
            torch.arange(9, 17, dtype=torch.float32).view(1, 1, 2, 2, 2),
        ]
        pooled_indices = [F.max_pool3d(t, kernel_size=2, return_indices=True) for t in orig]
        pooled = [p[0] for p in pooled_indices]
        indices = [p[1] for p in pooled_indices]
        pooled_nt = NT(pooled)
        indices_nt = NT(indices)
        unpooled = F.max_unpool3d(pooled_nt, indices_nt, kernel_size=2, output_size=orig[0].shape)
        reference = torch.stack(
            [F.max_unpool3d(pooled[i], indices[i], kernel_size=2, output_size=orig[i].shape) for i in range(2)]
        )
        assert_close(unpooled, reference)

    def test_max_unpool3d_tensor_indices(self):
        orig = torch.arange(1, 9, dtype=torch.float32).view(1, 1, 2, 2, 2)
        pooled, idx = F.max_pool3d(orig, kernel_size=2, return_indices=True)
        pooled_nt = NT([pooled])
        unpooled = F.max_unpool3d(pooled_nt, idx, kernel_size=2, output_size=orig.shape)
        reference = F.max_unpool3d(pooled, idx, kernel_size=2, output_size=orig.shape)
        reference = NT([reference], **unpooled._meta())
        assert_close(unpooled, reference)


class TestModuleIntegration:

    def test_conv2d_module(self, device, float_dtype):
        input = nested_rand([(5, 7, 8), (11, 13, 8)], device, float_dtype).permute(0, 3, 1, 2)
        layer = nn.Conv2d(input.shape[1], 4, kernel_size=2, padding=1).to(device=device, dtype=float_dtype)
        reference_layer = nn.Conv2d(input.shape[1], 4, kernel_size=2, padding=1).to(device=device, dtype=float_dtype)
        reference_layer.load_state_dict(layer.state_dict())

        output = layer(input)
        reference_storage = [reference_layer(t.unsqueeze(0)).squeeze(0) for t in input]
        reference = NT(reference_storage, **input._meta())
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

        output.sum().backward()
        sum(part.sum() for part in reference_storage).backward()
        assert_close(layer.weight.grad, reference_layer.weight.grad, atol=1e-5, rtol=1e-2)
        assert_close(layer.bias.grad, reference_layer.bias.grad, atol=1e-5, rtol=1e-2)

    def test_linear_module(self, device, float_dtype):
        input = nested_rand([(3, 5), (2, 5)], device, float_dtype)
        layer = nn.Linear(input.shape[-1], 3).to(device=device, dtype=float_dtype)
        reference_layer = nn.Linear(input.shape[-1], 3).to(device=device, dtype=float_dtype)
        reference_layer.load_state_dict(layer.state_dict())

        output = layer(input)
        reference = reference_layer(input.tensor)
        reference = reference.masked_fill(~output.mask.unsqueeze(-1), 0)
        assert_close(output, reference)

        output.sum().backward()
        reference.sum().backward()
        assert_close(layer.weight.grad, reference_layer.weight.grad)
        assert_close(layer.bias.grad, reference_layer.bias.grad)


class TestMultiHeadAttentionForward:

    def test_mha_batch_first_mismatch_raises_clear_error(self):
        embed_dim = 4
        num_heads = 2
        query = NestedTensor([torch.randn(3, embed_dim), torch.randn(2, embed_dim)], batch_first=True)
        key = NestedTensor([torch.randn(3, embed_dim), torch.randn(2, embed_dim)], batch_first=False)
        weight = torch.randn(3 * embed_dim, embed_dim)
        bias = torch.randn(3 * embed_dim)
        out_weight = torch.randn(embed_dim, embed_dim)
        out_bias = torch.randn(embed_dim)

        with pytest.raises(ValueError, match="batch_first mismatch between query and key"):
            F.multi_head_attention_forward(
                query,
                key,
                key,
                embed_dim,
                num_heads,
                weight,
                bias,
                None,
                None,
                False,
                0.0,
                out_weight,
                out_bias,
                training=False,
                need_weights=False,
            )

    def test_mha_requires_nested_query(self):
        tensor_query = torch.randn(2, 3, 4)
        nested_key = NestedTensor([torch.randn(3, 4)])
        with pytest.raises(TypeError):
            F.multi_head_attention_forward(
                tensor_query,
                nested_key,
                nested_key,
                4,
                1,
                torch.randn(12, 4),
                torch.randn(12),
                None,
                None,
                False,
                0.0,
                torch.randn(4, 4),
                torch.randn(4),
            )

    def test_multi_head_attention_batch_first_false(self, device, float_dtype):
        embed_dim = 4
        num_heads = 2
        lengths = [2, 3]
        data = [torch.randn(length, embed_dim, device=device, dtype=float_dtype) for length in lengths]
        query = NT(data, batch_first=False)

        module = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False, dropout=0.0).to(
            device=device, dtype=float_dtype
        )

        key_padding_mask = (~query.mask).transpose(0, 1)
        reference, _ = module(query.tensor, query.tensor, query.tensor, key_padding_mask=key_padding_mask)

        output, weights = F.multi_head_attention_forward(
            query,
            query,
            query,
            embed_dim,
            num_heads,
            module.in_proj_weight,
            module.in_proj_bias,
            module.bias_k,
            module.bias_v,
            False,
            0.0,
            module.out_proj.weight,
            module.out_proj.bias,
            training=False,
            need_weights=False,
        )

        assert isinstance(output, NestedTensor)
        assert weights is None
        atol, rtol = low_precision_cuda_tolerances(
            device,
            float_dtype,
            default=(1e-6, 1e-6),
            fp16=(1e-3, 1e-3),
            bf16=(5e-3, 5e-3),
        )
        assert_close(output, reference, atol=atol, rtol=rtol)

    def test_multi_head_attention_cross_attention(self):
        torch.manual_seed(1016)
        embed_dim = 4
        num_heads = 2
        mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0)
        query = NestedTensor([torch.randn(2, embed_dim), torch.randn(1, embed_dim)])
        key = NestedTensor([torch.randn(3, embed_dim), torch.randn(2, embed_dim)])
        output, weights = F.multi_head_attention_forward(
            query,
            key,
            key,
            embed_dim,
            num_heads,
            mha.in_proj_weight,
            mha.in_proj_bias,
            mha.bias_k,
            mha.bias_v,
            mha.add_zero_attn,
            mha.dropout,
            mha.out_proj.weight,
            mha.out_proj.bias,
            training=mha.training,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=True,
            is_causal=False,
        )
        reference, _ = mha(query.tensor, key.tensor, key.tensor, key_padding_mask=~key.mask, need_weights=False)
        reference = reference.masked_fill(~query.mask.unsqueeze(-1), 0)
        assert weights is None
        assert_close(output, reference, atol=1e-5)

    def test_multi_head_attention_cross_attention_with_dense_key_value(self):
        torch.manual_seed(1016)
        embed_dim = 4
        num_heads = 2
        mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0)
        query = NestedTensor([torch.randn(2, embed_dim), torch.randn(1, embed_dim)])
        key = NestedTensor([torch.randn(3, embed_dim), torch.randn(2, embed_dim)])
        key_dense = key.tensor

        output, weights = F.multi_head_attention_forward(
            query,
            key_dense,
            key_dense,
            embed_dim,
            num_heads,
            mha.in_proj_weight,
            mha.in_proj_bias,
            mha.bias_k,
            mha.bias_v,
            mha.add_zero_attn,
            mha.dropout,
            mha.out_proj.weight,
            mha.out_proj.bias,
            training=mha.training,
            key_padding_mask=~key.mask,
            need_weights=False,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=True,
            is_causal=False,
        )
        reference, _ = mha(query.tensor, key_dense, key_dense, key_padding_mask=~key.mask, need_weights=False)
        reference = reference.masked_fill(~query.mask.unsqueeze(-1), 0)
        assert weights is None
        assert_close(output, reference, atol=1e-5)

    def test_multi_head_attention_custom_mask_value(self):
        torch.manual_seed(1016)
        embed_dim = 4
        num_heads = 2
        mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0)
        query = NestedTensor(
            [torch.randn(2, embed_dim), torch.randn(1, embed_dim)],
            mask_value=True,
        )
        key = NestedTensor(
            [torch.randn(3, embed_dim), torch.randn(1, embed_dim)],
            mask_value=True,
        )
        output, weights = F.multi_head_attention_forward(
            query,
            key,
            key,
            embed_dim,
            num_heads,
            mha.in_proj_weight,
            mha.in_proj_bias,
            mha.bias_k,
            mha.bias_v,
            mha.add_zero_attn,
            mha.dropout,
            mha.out_proj.weight,
            mha.out_proj.bias,
            training=mha.training,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=True,
            is_causal=False,
        )
        reference, _ = mha(query.tensor, key.tensor, key.tensor, key_padding_mask=key.mask, need_weights=False)
        reference = reference.masked_fill(query.mask.unsqueeze(-1), 0)
        assert weights is None
        assert_close(output, reference, atol=1e-5)

    def test_multi_head_attention_forward(self):
        torch.manual_seed(1016)
        embed_dim = 4
        num_heads = 2
        mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0)
        input = NestedTensor([torch.randn(3, embed_dim), torch.randn(2, embed_dim)])
        attn_output, attn_weights = F.multi_head_attention_forward(
            input,
            input,
            input,
            embed_dim,
            num_heads,
            mha.in_proj_weight,
            mha.in_proj_bias,
            mha.bias_k,
            mha.bias_v,
            mha.add_zero_attn,
            mha.dropout,
            mha.out_proj.weight,
            mha.out_proj.bias,
            training=mha.training,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=True,
            is_causal=False,
        )
        reference, _ = mha(input.tensor, input.tensor, input.tensor, key_padding_mask=~input.mask, need_weights=False)
        reference = reference.masked_fill(~input.mask.unsqueeze(-1), 0)
        assert attn_weights is None
        assert_close(attn_output, reference, atol=1e-5)

    def test_multi_head_attention_masks_padding_tokens(self, device, float_dtype):
        embed_dim = 1
        num_heads = 1
        mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False, dropout=0.0).to(
            device=device, dtype=float_dtype
        )

        seq1 = torch.tensor([[1.0], [0.5]], device=device, dtype=float_dtype)
        seq2 = torch.tensor([[2.0]], device=device, dtype=float_dtype)
        nested = NestedTensor([seq1, seq2], padding_value=10.0, dtype=float_dtype, device=device)

        qkv = nested.tensor.transpose(0, 1)
        key_padding_mask = ~nested.mask

        reference, _ = F.multi_head_attention_forward(
            qkv,
            qkv,
            qkv,
            embed_dim,
            num_heads,
            mha.in_proj_weight,
            mha.in_proj_bias,
            mha.bias_k,
            mha.bias_v,
            mha.add_zero_attn,
            0.0,
            mha.out_proj.weight,
            mha.out_proj.bias,
            training=mha.training,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=True,
            is_causal=False,
        )
        reference = reference.transpose(0, 1)
        reference = reference.masked_fill(~nested.mask.unsqueeze(-1), nested.padding_value)

        output, weights = F.multi_head_attention_forward(
            nested,
            nested,
            nested,
            embed_dim,
            num_heads,
            mha.in_proj_weight,
            mha.in_proj_bias,
            mha.bias_k,
            mha.bias_v,
            mha.add_zero_attn,
            0.0,
            mha.out_proj.weight,
            mha.out_proj.bias,
            training=mha.training,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=True,
            is_causal=False,
        )

        assert weights is None
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_multi_head_attention_with_weights(self):
        torch.manual_seed(1016)
        embed_dim = 6
        num_heads = 3
        mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0)
        query = NestedTensor([torch.randn(2, embed_dim), torch.randn(1, embed_dim)])
        output, weights = F.multi_head_attention_forward(
            query,
            query,
            query,
            embed_dim,
            num_heads,
            mha.in_proj_weight,
            mha.in_proj_bias,
            mha.bias_k,
            mha.bias_v,
            mha.add_zero_attn,
            mha.dropout,
            mha.out_proj.weight,
            mha.out_proj.bias,
            training=mha.training,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=True,
            is_causal=False,
        )
        reference, reference_weights = mha(
            query.tensor, query.tensor, query.tensor, key_padding_mask=~query.mask, need_weights=True
        )
        reference = reference.masked_fill(~query.mask.unsqueeze(-1), 0)
        assert_close(output, reference, atol=1e-5)
        assert weights.shape[0] == query.tensor.shape[1]


class TestNormalizationOps:

    def test_batch_norm(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        running_mean = torch.zeros(3, device=device, dtype=float_dtype)
        running_var = torch.ones(3, device=device, dtype=float_dtype)
        output = F.batch_norm(nt, running_mean=running_mean, running_var=running_var, training=True)
        concat, shapes = nt.concatenate()
        reference = NestedTensor.from_concatenated(
            F.batch_norm(concat, running_mean=running_mean, running_var=running_var, training=True),
            shapes,
            **nt._meta(),
        )
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_group_norm(self, device, float_dtype):
        nt = nested_rand([(3, 4), (2, 4)], device, float_dtype)
        output = F.group_norm(nt, num_groups=1)
        reference = NT([F.group_norm(t.unsqueeze(0), num_groups=1).squeeze(0) for t in nt], **nt._meta())
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_instance_norm(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(3, 2, 2, device=device, dtype=float_dtype),
                torch.randn(3, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.instance_norm(nt, use_input_stats=True)
        reference = NT([F.instance_norm(t.unsqueeze(0), use_input_stats=True).squeeze(0) for t in nt], **nt._meta())
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_layer_norm(self, device, float_dtype):
        nt = nested_rand([(3, 4), (2, 4)], device, float_dtype)
        output = F.layer_norm(nt, normalized_shape=(4,))
        reference = NT([F.layer_norm(t, (4,)) for t in nt], **nt._meta())
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_local_response_norm(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(3, 3, 3, device=device, dtype=float_dtype),
                torch.randn(3, 3, 3, device=device, dtype=float_dtype),
            ]
        )
        output = F.local_response_norm(nt, size=2)
        reference = NT([F.local_response_norm(t.unsqueeze(0), size=2).squeeze(0) for t in nt], **nt._meta())
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_rms_norm(self, device, float_dtype):
        nt = nested_rand([(1, 4), (1, 4)], device, float_dtype)
        output = F.rms_norm(nt, normalized_shape=(4,))
        reference = NT([F.rms_norm(t, (4,)) for t in nt], **nt._meta())
        assert_close(output, reference, atol=1e-5, rtol=1e-5)


class TestNormalizeFunction:

    def test_normalize_batch_dim_raises(self, device, float_dtype):
        input = nested_rand([(3, 4), (2, 4)], device, float_dtype)
        with pytest.raises(ValueError):
            F.normalize(input, dim=0)

    def test_normalize(self, device, float_dtype):
        input = nested_rand([(3, 4), (3, 4)], device, float_dtype)
        output = F.normalize(input, dim=2)
        reference = F.normalize(input.tensor, dim=2)
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_normalize_ragged_axis(self, device, float_dtype):
        input = NT(
            [
                torch.tensor([3.0, 4.0], device=device, dtype=float_dtype),
                torch.tensor([1.0, 2.0, 2.0], device=device, dtype=float_dtype),
            ]
        )
        output = F.normalize(input, dim=1)
        reference = NT([F.normalize(t, dim=0) for t in input], **input._meta())
        assert_close(output, reference, atol=1e-6, rtol=1e-6)


class TestOneHot:

    def test_one_hot(self):
        x = NT([torch.tensor([0, 1, 2], dtype=torch.long), torch.tensor([1, 0], dtype=torch.long)])
        output = F.one_hot(x, num_classes=3)
        reference = NT([F.one_hot(t, num_classes=3) for t in x], **x._meta())
        assert_close(output, reference)


class TestPad:

    def test_pad(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(4.0, device=device, dtype=float_dtype).view(1, 1, 2, 2),
                torch.ones(1, 1, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.pad(nt, (1, 1, 1, 1), value=0.5)
        reference = NT([F.pad(t, (1, 1, 1, 1), value=0.5) for t in nt], **nt._meta())
        assert_close(output, reference)

    def test_pad_ragged_leading_dim(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(4.0, device=device, dtype=float_dtype).view(1, 1, 2, 2),
                torch.arange(8.0, device=device, dtype=float_dtype).view(2, 1, 2, 2),
            ]
        )
        output = F.pad(nt, (1, 1, 1, 1), value=0.25)
        reference = NT([F.pad(t, (1, 1, 1, 1), value=0.25) for t in nt], **nt._meta())
        assert_close(output, reference)


class TestPairwiseDistance:

    def test_pairwise_distance(self, device, float_dtype):
        x1 = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        x2 = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        output = F.pairwise_distance(x1, x2)
        reference = NT([F.pairwise_distance(a, b) for a, b in zip(x1, x2)], **x1._meta())
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_pairwise_distance_p1(self, device, float_dtype):
        x1 = nested_rand([(2, 4), (3, 4)], device, float_dtype)
        x2 = nested_rand([(2, 4), (3, 4)], device, float_dtype)
        output = F.pairwise_distance(x1, x2, p=1)
        reference = NT([F.pairwise_distance(a, b, p=1) for a, b in zip(x1, x2)], **x1._meta())
        assert_close(output, reference, atol=1e-6, rtol=1e-6)


class TestPdist:

    def test_pdist(self, device, float_dtype):
        x = NT(
            [
                torch.randn(4, 3, device=device, dtype=float_dtype),
                torch.randn(3, 3, device=device, dtype=float_dtype),
            ]
        )
        try:
            reference = NT([F.pdist(t) for t in x], **x._meta())
        except RuntimeError as error:
            with pytest.raises(type(error)):
                F.pdist(x)
            return
        output = F.pdist(x)
        assert_close(output, reference, atol=1e-6, rtol=1e-6)


class TestPixelShuffle:

    def test_pixel_shuffle(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(4, device=device, dtype=float_dtype).view(1, 4, 1, 1),
                torch.arange(4, 8, device=device, dtype=float_dtype).view(1, 4, 1, 1),
            ]
        )
        output = F.pixel_shuffle(nt, upscale_factor=2)
        reference = NT([F.pixel_shuffle(t, upscale_factor=2) for t in nt], **nt._meta())
        assert_close(output, reference)

    def test_pixel_unshuffle(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(4, device=device, dtype=float_dtype).view(1, 4, 1, 1),
                torch.arange(4, 8, device=device, dtype=float_dtype).view(1, 4, 1, 1),
            ]
        )
        output = F.pixel_shuffle(nt, upscale_factor=2)
        output = F.pixel_unshuffle(output, downscale_factor=2)
        assert_close(output, nt)


class TestRankingLosses:

    def test_cosine_embedding_loss(self, device, float_dtype):
        x1 = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        x2 = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        target = NT(
            [
                torch.tensor([1.0, -1.0], device=device, dtype=float_dtype),
                torch.tensor([1.0], device=device, dtype=float_dtype),
            ]
        )
        output = F.cosine_embedding_loss(x1, x2, target, reduction="sum")
        reference = F.cosine_embedding_loss(
            torch.cat(tuple(x1), dim=0),
            torch.cat(tuple(x2), dim=0),
            torch.cat(tuple(target), dim=0),
            reduction="sum",
        )
        assert_close(output, reference)

    def test_hinge_embedding_loss(self, device, float_dtype):
        x = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        target = NT(
            [
                torch.tensor([1.0, -1.0], device=device, dtype=float_dtype),
                torch.tensor([1.0], device=device, dtype=float_dtype),
            ]
        )
        output = F.hinge_embedding_loss(x, target, reduction="sum")
        reference = F.hinge_embedding_loss(
            torch.cat(tuple(x), dim=0),
            torch.cat(tuple(target), dim=0),
            reduction="sum",
        )
        assert_close(output, reference)

    def test_margin_ranking_loss(self, device, float_dtype):
        input1 = nested_rand([(2,), (1,)], device, float_dtype)
        input2 = nested_rand([(2,), (1,)], device, float_dtype)
        target = NT(
            [
                torch.tensor([1.0, -1.0], device=device, dtype=float_dtype),
                torch.tensor([1.0], device=device, dtype=float_dtype),
            ]
        )
        output = F.margin_ranking_loss(input1, input2, target, reduction="sum")
        reference = F.margin_ranking_loss(
            torch.cat(tuple(input1), dim=0),
            torch.cat(tuple(input2), dim=0),
            torch.cat(tuple(target), dim=0),
            reduction="sum",
        )
        assert_close(output, reference)

    def test_triplet_margin_loss(self, device, float_dtype):
        anchor = nested_rand([(2, 4), (1, 4)], device, float_dtype)
        positive = nested_rand([(2, 4), (1, 4)], device, float_dtype)
        negative = nested_rand([(2, 4), (1, 4)], device, float_dtype)
        output = F.triplet_margin_loss(anchor, positive, negative, reduction="sum")
        reference = F.triplet_margin_loss(
            torch.cat(tuple(anchor), dim=0),
            torch.cat(tuple(positive), dim=0),
            torch.cat(tuple(negative), dim=0),
            reduction="sum",
        )
        assert_close(output, reference)

    def test_triplet_margin_with_distance_loss(self, device, float_dtype):
        anchor = nested_rand([(2, 4), (1, 4)], device, float_dtype)
        positive = nested_rand([(2, 4), (1, 4)], device, float_dtype)
        negative = nested_rand([(2, 4), (1, 4)], device, float_dtype)
        output = F.triplet_margin_with_distance_loss(anchor, positive, negative, reduction="sum")
        reference = F.triplet_margin_with_distance_loss(
            anchor.concat, positive.concat, negative.concat, reduction="sum"
        )
        assert_close(output, reference, atol=1e-6, rtol=1e-6)


class TestRegressionLosses:

    def test_gaussian_nll_loss(self, device, float_dtype):
        pred = nested_rand([(2, 2), (1, 2)], device, float_dtype)
        target = nested_rand([(2, 2), (1, 2)], device, float_dtype)
        var = torch.ones_like(torch.cat(tuple(pred), dim=0))
        output = F.gaussian_nll_loss(pred, target, var=var, reduction="sum")
        reference = F.gaussian_nll_loss(
            torch.cat(tuple(pred), dim=0),
            torch.cat(tuple(target), dim=0),
            var=var,
            reduction="sum",
        )
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_huber_loss(self, device, float_dtype):
        pred = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        target = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        output = F.huber_loss(pred, target, reduction="sum")
        reference = F.huber_loss(torch.cat(tuple(pred), dim=0), torch.cat(tuple(target), dim=0), reduction="sum")
        assert_close(output, reference)

    def test_l1_loss(self, device, float_dtype):
        pred = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        target = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        output = F.l1_loss(pred, target, reduction="sum")
        reference = F.l1_loss(torch.cat(tuple(pred), dim=0), torch.cat(tuple(target), dim=0), reduction="sum")
        assert_close(output, reference)

    def test_mse_loss(self, device, float_dtype):
        pred = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        target = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        output = F.mse_loss(pred, target, reduction="sum")
        reference = F.mse_loss(torch.cat(tuple(pred), dim=0), torch.cat(tuple(target), dim=0), reduction="sum")
        assert_close(output, reference)

    def test_mse_loss_mask_value_true(self, device, float_dtype):
        pred = NT(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0], device=device, dtype=float_dtype),
            ],
            mask_value=True,
        )
        target = torch.tensor([[0.0, 0.0], [10.0, 100.0]], device=device, dtype=float_dtype)
        output = F.mse_loss(pred, target, reduction="mean")
        reference = torch.tensor([1.0, 4.0, 49.0], device=device, dtype=float_dtype).mean()
        assert_close(output, reference)

    def test_poisson_nll_loss(self, device, float_dtype):
        pred = NT(
            [torch.rand(2, 2, device=device, dtype=float_dtype), torch.rand(1, 2, device=device, dtype=float_dtype)]
        )
        target = NT(
            [torch.rand(2, 2, device=device, dtype=float_dtype), torch.rand(1, 2, device=device, dtype=float_dtype)]
        )
        output = F.poisson_nll_loss(pred, target, reduction="sum")
        reference = F.poisson_nll_loss(
            torch.cat(tuple(pred), dim=0),
            torch.cat(tuple(target), dim=0),
            reduction="sum",
        )
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_smooth_l1_loss(self, device, float_dtype):
        pred = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        target = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        output = F.smooth_l1_loss(pred, target, reduction="sum")
        reference = F.smooth_l1_loss(
            torch.cat(tuple(pred), dim=0),
            torch.cat(tuple(target), dim=0),
            reduction="sum",
        )
        assert_close(output, reference)


@pytest.mark.skipif(not hasattr(F, "scaled_dot_product_attention"), reason="scaled_dot_product_attention not available")
class TestScaledDotProductAttention:

    def test_sdpa_native_sequence_layout(self, device, float_dtype):
        first = torch.randn(4, 11, 32, device=device, dtype=float_dtype)
        second = torch.randn(4, 7, 32, device=device, dtype=float_dtype)
        query = NT([first, second])

        assert query._values.shape == (18, 4, 32)
        assert query._offsets.tolist() == [0, 11, 18]
        assert_close(query[0], first)
        assert_close(query[1], second)

    @pytest.mark.skipif(create_block_mask is None or flex_attention is None, reason="FlexAttention not available")
    @pytest.mark.parametrize("is_causal", [False, True])
    def test_flex_attention_matches_dense_sdpa(self, device, is_causal):
        if device.type != "cuda":
            pytest.skip("compiled FlexAttention benchmark path is CUDA-only")

        batch_size, num_heads, max_len, head_dim = 2, 4, 32, 16
        lengths = [32, 19]
        dtype = torch.float32

        query = torch.randn(batch_size, num_heads, max_len, head_dim, device=device, dtype=dtype)
        key = torch.randn(batch_size, num_heads, max_len, head_dim, device=device, dtype=dtype)
        value = torch.randn(batch_size, num_heads, max_len, head_dim, device=device, dtype=dtype)

        block_mask = _make_test_flex_block_mask(lengths, max_len, device, is_causal=is_causal)
        key_padding_mask = torch.zeros(batch_size, 1, 1, max_len, dtype=torch.bool, device=device)
        query_valid_mask = torch.zeros(batch_size, 1, max_len, 1, dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            key_padding_mask[i, 0, 0, :length] = True
            query_valid_mask[i, 0, :length, 0] = True

        flex_out = _compiled_test_flex_attention()(query, key, value, block_mask=block_mask)
        sdpa_out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=key_padding_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        assert_close(
            flex_out.masked_fill(~query_valid_mask, 0),
            sdpa_out.masked_fill(~query_valid_mask, 0),
            atol=1e-4,
            rtol=1e-4,
        )

    @pytest.mark.skipif(create_block_mask is None or flex_attention is None, reason="FlexAttention not available")
    def test_flex_attention_wrapper(self, device):
        if device.type != "cuda":
            pytest.skip("DanLing FlexAttention wrapper is currently CUDA-focused")

        query = NT(
            [
                torch.randn(4, 23, 16, device=device, dtype=torch.float32),
                torch.randn(4, 11, 16, device=device, dtype=torch.float32),
            ]
        )
        key = NT(
            [
                torch.randn(4, 23, 16, device=device, dtype=torch.float32),
                torch.randn(4, 11, 16, device=device, dtype=torch.float32),
            ]
        )
        value = NT(
            [
                torch.randn(4, 23, 16, device=device, dtype=torch.float32),
                torch.randn(4, 11, 16, device=device, dtype=torch.float32),
            ]
        )

        try:
            output = flex_attention(query, key, value)
        except Exception as exc:
            _maybe_xfail_upstream_flex_error(exc)
            raise
        reference = NT(
            [F.scaled_dot_product_attention(q, k, v, dropout_p=0.0) for q, k, v in zip(query, key, value)],
            **query._meta(),
        )
        assert isinstance(output, NT)
        assert_close(output, reference, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(create_block_mask is None or flex_attention is None, reason="FlexAttention not available")
    def test_flex_attention_wrapper_supports_danling_block_mask(self, device):
        if device.type != "cuda":
            pytest.skip("DanLing FlexAttention wrapper is currently CUDA-focused")

        query = NT(
            [
                torch.randn(4, 19, 16, device=device, dtype=torch.float32),
                torch.randn(4, 7, 16, device=device, dtype=torch.float32),
            ]
        )
        key = NT(
            [
                torch.randn(4, 19, 16, device=device, dtype=torch.float32),
                torch.randn(4, 7, 16, device=device, dtype=torch.float32),
            ]
        )
        value = NT(
            [
                torch.randn(4, 19, 16, device=device, dtype=torch.float32),
                torch.randn(4, 7, 16, device=device, dtype=torch.float32),
            ]
        )

        block_mask = create_flex_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            query,
            key,
        )
        try:
            output, lse = flex_attention(query, key, value, block_mask=block_mask, return_lse=True)
        except Exception as exc:
            _maybe_xfail_upstream_flex_error(exc)
            raise
        reference = NT(
            [
                F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
                for q, k, v in zip(query, key, value)
            ],
            **query._meta(),
        )
        assert isinstance(output, NT)
        assert isinstance(lse, NT)
        assert_close(output, reference, atol=1e-4, rtol=1e-4)

    def test_nested_from_padded_tensor_roundtrip_for_attention_like_layout(self, device, float_dtype):
        source = NT(
            [
                torch.randn(4, 11, 32, device=device, dtype=float_dtype),
                torch.randn(4, 7, 32, device=device, dtype=float_dtype),
            ]
        )
        padded = source.tensor[..., :16]

        restored = _nested_from_padded_tensor(source, padded)
        reference = NT([tensor[..., :16] for tensor in source], **source._meta())

        assert restored._varying_dims == source._varying_dims
        assert restored._static_dims == source._static_dims
        assert_close(restored, reference)

    def test_nested_from_padded_tensor_roundtrip_for_conv_like_layout(self, device, float_dtype):
        source = NT(
            [
                torch.randn(3, 5, 7, device=device, dtype=float_dtype),
                torch.randn(3, 4, 6, device=device, dtype=float_dtype),
            ]
        )
        padded = source.tensor[:, :2]

        restored = _nested_from_padded_tensor(source, padded)
        reference = NT([tensor[:2] for tensor in source], **source._meta())

        assert restored._varying_dims == source._varying_dims
        assert restored._static_dims == source._static_dims
        assert_close(restored, reference)

    def test_restore_flex_dense_tensor_updates_suffix_metadata(self, device, float_dtype):
        query = NT(
            [
                torch.randn(4, 11, 32, device=device, dtype=float_dtype),
                torch.randn(4, 7, 32, device=device, dtype=float_dtype),
            ]
        )
        output = torch.randn(1, 4, query._values.size(0), 6, device=device, dtype=float_dtype)

        restored = _restore_flex_dense_tensor(output, query)

        assert tuple(tuple(int(size) for size in row) for row in restored._physical_shape.tolist()) == (
            (4, 11, 6),
            (4, 7, 6),
        )
        assert restored._element_shapes == ((4, 11, 6), (4, 7, 6))
        assert restored._packed_sizes == query._packed_sizes
        assert restored.shape[-1] == 6

    def test_sdpa_batch_first_false_matches_reference(self):
        device = torch.device("cpu")
        dtype = torch.float32
        query_elems = [
            torch.randn(2, 5, 8, device=device, dtype=dtype),
            torch.randn(2, 3, 8, device=device, dtype=dtype),
        ]
        key_elems = [
            torch.randn(2, 6, 8, device=device, dtype=dtype),
            torch.randn(2, 4, 8, device=device, dtype=dtype),
        ]
        value_elems = [
            torch.randn(2, 6, 8, device=device, dtype=dtype),
            torch.randn(2, 4, 8, device=device, dtype=dtype),
        ]
        query = NT(query_elems, batch_first=False)
        key = NT(key_elems, batch_first=False)
        value = NT(value_elems, batch_first=False)

        output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        reference = NT(
            [F.scaled_dot_product_attention(q, k, v, dropout_p=0.0) for q, k, v in zip(query, key, value)],
            **query._meta(),
        )
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_sdpa_batch_first_false_with_nested_mask_matches_reference(self):
        device = torch.device("cpu")
        dtype = torch.float32
        query_elems = [
            torch.randn(2, 4, 8, device=device, dtype=dtype),
            torch.randn(2, 3, 8, device=device, dtype=dtype),
        ]
        key_elems = [
            torch.randn(2, 5, 8, device=device, dtype=dtype),
            torch.randn(2, 4, 8, device=device, dtype=dtype),
        ]
        value_elems = [
            torch.randn(2, 5, 8, device=device, dtype=dtype),
            torch.randn(2, 4, 8, device=device, dtype=dtype),
        ]
        masks = [
            torch.ones(2, 4, 5, device=device, dtype=torch.bool),
            torch.ones(2, 3, 4, device=device, dtype=torch.bool),
        ]
        masks[0][:, :, -1] = False
        masks[1][:, -1, :] = False

        query = NT(query_elems, batch_first=False)
        key = NT(key_elems, batch_first=False)
        value = NT(value_elems, batch_first=False)
        attn_mask = NT(masks, batch_first=False)

        output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0)
        reference = NT(
            [
                F.scaled_dot_product_attention(q, k, v, attn_mask=m, dropout_p=0.0)
                for q, k, v, m in zip(query, key, value, masks)
            ],
            **query._meta(),
        )
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_sdpa_batched_mask(self, device, float_dtype):
        query = NT(
            [
                torch.randn(2, 6, 16, device=device, dtype=float_dtype),
                torch.randn(2, 6, 16, device=device, dtype=float_dtype),
            ]
        )
        mask = torch.ones(2, 2, 6, 6, dtype=torch.bool, device=device)
        mask[0, :, :, -1] = False
        mask[1, :, -1, :] = False
        output = F.scaled_dot_product_attention(query, query, query, attn_mask=mask, dropout_p=0.0)
        reference = NT(
            [F.scaled_dot_product_attention(q, q, q, attn_mask=mask[i], dropout_p=0.0) for i, q in enumerate(query)],
            **query._meta(),
        )
        atol, rtol = low_precision_cuda_tolerances(
            device,
            float_dtype,
            default=(1e-5, 1e-5),
            fp16=(1e-3, 1e-3),
            bf16=(5e-3, 5e-3),
        )
        assert_close(output, reference, atol=atol, rtol=rtol)

    def test_sdpa_matches_reference(self, device, float_dtype):
        query = NT(
            [
                torch.randn(4, 64, 32, device=device, dtype=float_dtype),
                torch.randn(4, 16, 32, device=device, dtype=float_dtype),
            ]
        )
        key = NT(
            [
                torch.randn(4, 64, 32, device=device, dtype=float_dtype),
                torch.randn(4, 20, 32, device=device, dtype=float_dtype),
            ]
        )
        value = NT(
            [
                torch.randn(4, 64, 32, device=device, dtype=float_dtype),
                torch.randn(4, 20, 32, device=device, dtype=float_dtype),
            ]
        )
        output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        reference = NT(
            [F.scaled_dot_product_attention(q, k, v, dropout_p=0.0) for q, k, v in zip(query, key, value)],
            **query._meta(),
        )
        atol, rtol = low_precision_cuda_tolerances(
            device,
            float_dtype,
            default=(1e-5, 1e-5),
            fp16=(1e-3, 1e-3),
            bf16=(5e-3, 5e-3),
        )
        assert_close(output, reference, atol=atol, rtol=rtol)

    def test_sdpa_matches_reference_dense_batch(self, device, float_dtype):
        query = NT(
            [
                torch.randn(4, 64, 32, device=device, dtype=float_dtype),
                torch.randn(4, 56, 32, device=device, dtype=float_dtype),
            ]
        )
        output = F.scaled_dot_product_attention(query, query, query, dropout_p=0.0)
        reference = NT([F.scaled_dot_product_attention(q, q, q, dropout_p=0.0) for q in query], **query._meta())
        atol, rtol = low_precision_cuda_tolerances(
            device,
            float_dtype,
            default=(1e-5, 1e-5),
            fp16=(1e-3, 1e-3),
            bf16=(5e-3, 5e-3),
        )
        assert_close(output, reference, atol=atol, rtol=rtol)

    def test_sdpa_mismatched_batch_lengths_raises(self):
        query = NT([torch.randn(2, 4, 8), torch.randn(2, 3, 8)])
        key = NT([torch.randn(2, 4, 8)])
        with pytest.raises(ValueError, match="NestedTensor batch length mismatch"):
            F.scaled_dot_product_attention(query, key, key, dropout_p=0.0)

    def test_sdpa_native_bridge_roundtrip(self, device, float_dtype):
        query = NT(
            [
                torch.randn(4, 11, 32, device=device, dtype=float_dtype),
                torch.randn(4, 7, 32, device=device, dtype=float_dtype),
            ]
        )

        packed, _lengths, _cumulative, _max_seqlen = _sdpa_pack_native(query)
        restored = _sdpa_restore_native(packed, query)

        assert_close(restored, query)

    def test_sdpa_requires_nested_query(self):
        tensor_query = torch.randn(2, 2, 4, 8)
        key = NT([torch.randn(2, 4, 8)])
        with pytest.raises(TypeError):
            F.scaled_dot_product_attention(tensor_query, key, key, dropout_p=0.0)

    def test_sdpa_restore_native_updates_last_dim_metadata(self, device, float_dtype):
        query = NT(
            [
                torch.randn(4, 11, 32, device=device, dtype=float_dtype),
                torch.randn(4, 7, 32, device=device, dtype=float_dtype),
            ]
        )
        packed = torch.randn(query._values.size(0), query._values.size(1), 16, device=device, dtype=float_dtype)

        restored = _sdpa_restore_native(packed, query)

        assert tuple(tuple(int(size) for size in row) for row in restored._physical_shape.tolist()) == (
            (4, 11, 16),
            (4, 7, 16),
        )
        assert restored._element_shapes == ((4, 11, 16), (4, 7, 16))
        assert restored._packed_sizes == query._packed_sizes
        assert restored.shape[-1] == 16

    def test_sdpa_tensor_key_value(self, device, float_dtype):
        query = NT(
            [
                torch.randn(2, 6, 16, device=device, dtype=float_dtype),
                torch.randn(2, 4, 16, device=device, dtype=float_dtype),
            ]
        )
        output = F.scaled_dot_product_attention(query, query.tensor, query.tensor, dropout_p=0.0)
        reference = NT([F.scaled_dot_product_attention(q, q, q, dropout_p=0.0) for q in query], **query._meta())
        atol, rtol = low_precision_cuda_tolerances(
            device,
            float_dtype,
            default=(1e-5, 1e-5),
            fp16=(1e-3, 1e-3),
            bf16=(5e-3, 5e-3),
        )
        assert_close(output, reference, atol=atol, rtol=rtol)


class TestSequenceLosses:

    def test_ctc_loss(self):
        torch.manual_seed(1016)
        vocab = 5
        logits = [
            torch.log_softmax(torch.randn(3, vocab), dim=-1),
            torch.log_softmax(torch.randn(2, vocab), dim=-1),
        ]
        targets = [torch.tensor([1, 2], dtype=torch.long), torch.tensor([2], dtype=torch.long)]
        nt_logits = NT(logits)
        nt_targets = NT(targets)
        input_lengths = torch.tensor([len(l) for l in logits], dtype=torch.long)  # noqa: E741
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)  # noqa: E741

        output = F.ctc_loss(nt_logits, nt_targets, input_lengths, target_lengths, reduction="sum")

        padded_logits = nt_logits.tensor.transpose(0, 1)
        targets_concat = torch.cat(targets, dim=0)
        reference = F.ctc_loss(padded_logits, targets_concat, input_lengths, target_lengths, reduction="sum")

        assert_close(output, reference, atol=1e-6, rtol=1e-6)


class TestSoftmaxFamily:

    def test_gumbel_softmax(self, device, float_dtype):
        nt = nested_rand([(2, 4), (3, 4)], device, float_dtype)
        torch.manual_seed(1016)
        reference = packed_result(nt, F.gumbel_softmax(nt._values, dim=-1))
        torch.manual_seed(1016)
        output = F.gumbel_softmax(nt, dim=-1)
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_log_softmax(self, device, float_dtype):
        nt = nested_rand([(2, 4), (3, 4)], device, float_dtype)
        assert_nested_function_matches(F.log_softmax, nt, dim=-1)

    def test_softmax(self, device, float_dtype):
        nt = nested_rand([(2, 4), (3, 4)], device, float_dtype)
        assert_nested_function_matches(F.softmax, nt, dim=-1)

    def test_softmax_accepts_positive_dim(self, device, float_dtype):
        nt = nested_rand([(2, 4), (3, 4)], device, float_dtype)
        output = F.softmax(nt, dim=2)
        reference = F.softmax(nt.tensor, dim=2)
        assert_close(output, reference)

    @pytest.mark.parametrize("op", [F.softmax, F.log_softmax, F.softmin])
    def test_softmax_family_ragged_axis(self, op, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0, 4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = op(nt, dim=1)
        reference = NT([op(t, dim=0) for t in nt], **nt._meta())
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_softmin(self, device, float_dtype):
        nt = nested_rand([(2, 4), (3, 4)], device, float_dtype)
        assert_nested_function_matches(F.softmin, nt, dim=-1)


class TestUnfoldFold:

    def test_unfold_and_fold_round_trip(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(1.0, 10.0, device=device, dtype=float_dtype).view(1, 1, 3, 3),
                torch.ones(1, 1, 3, 3, device=device, dtype=float_dtype),
            ]
        )
        output = F.unfold(nt, kernel_size=2, stride=1)
        reference = NT([F.unfold(t, kernel_size=2, stride=1) for t in nt], **nt._meta())
        assert_close(output, reference)

        unfolded = output
        output = F.fold(unfolded, output_size=(3, 3), kernel_size=2, stride=1)
        reference = NT([F.fold(t, output_size=(3, 3), kernel_size=2, stride=1) for t in unfolded], **unfolded._meta())
        assert_close(output, reference, atol=1e-6, rtol=1e-6)
