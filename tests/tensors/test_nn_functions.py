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

import math
import os
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from danling.tensors import NestedTensor, create_flex_block_mask
from tests.tensors.utils import (
    assert_close,
    assert_nested_function_matches,
    low_precision_cuda_tolerances,
    nested_rand,
)

NT = NestedTensor


def reference_options(source: NestedTensor) -> dict:
    r"""Return public construction options for an elementwise reference."""
    return {
        "batch_first": source.batch_first,
        "padding_value": source.padding_value,
        "mask_value": source.mask_value,
    }


try:
    from torch.nn.attention.flex_attention import flex_attention
except Exception:
    flex_attention = None


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
    def test_activation_compile_fullgraph(self, device):
        nt = nested_rand([(2, 4), (1, 4)], device, torch.float32)
        compiled = _compile_fullgraph(F.gelu)
        output = compiled(nt)
        reference = F.gelu(nt.tensor)
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
        reference = NT([F.bilinear(a, b, weight, bias) for a, b in zip(x1, x2)], **reference_options(x1))
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
        reference = NT([F.channel_shuffle(t, groups=2) for t in x], **reference_options(x))
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
        values = torch.randn(3, 3, 1, device=device, dtype=float_dtype, requires_grad=True)
        reference_values = values.detach().clone().requires_grad_()
        logits = NT(values.split((2, 1)))
        targets = NT(
            [
                torch.rand(2, 3, 1, device=device, dtype=float_dtype),
                torch.rand(1, 3, 1, device=device, dtype=float_dtype),
            ]
        )
        output = F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets.squeeze(-1), reduction="mean")
        reference = F.binary_cross_entropy_with_logits(
            reference_values.squeeze(-1),
            targets.concat.squeeze(-1),
            reduction="mean",
        )

        actual_gradient = torch.autograd.grad(output, values)[0]
        expected_gradient = torch.autograd.grad(reference, reference_values)[0]
        assert_close(output, reference)
        assert_close(actual_gradient, expected_gradient)

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
        output = F.cross_entropy(logits.movedim(-1, 1), targets, reduction="sum")
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
        log_probs = NT([torch.log_softmax(t, dim=-1) for t in logits], **reference_options(logits))
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

        ref_linear = NT([F.linear(t, weight, bias) for t in nt], **reference_options(nt))
        ref_softmax = NT([F.softmax(t, dim=0) for t in nt], **reference_options(nt))
        ref_log_softmax = NT([F.log_softmax(t, dim=0) for t in nt], **reference_options(nt))
        ref_layer_norm = NT([F.layer_norm(t, (2,)) for t in nt], **reference_options(nt))
        ref_rms_norm = NT([F.rms_norm(t, (2,)) for t in nt], **reference_options(nt))
        assert isinstance(linear_comp, NestedTensor)
        assert isinstance(softmax_comp, NestedTensor)
        assert isinstance(log_softmax_comp, NestedTensor)
        assert isinstance(layer_norm_comp, NestedTensor)
        assert isinstance(rms_norm_comp, NestedTensor)
        assert_close(linear_comp, ref_linear)
        assert_close(softmax_comp, ref_softmax)
        assert_close(log_softmax_comp, ref_log_softmax)
        assert_close(layer_norm_comp, ref_layer_norm)
        assert_close(rms_norm_comp, ref_rms_norm)


class TestConv:

    @staticmethod
    def _tolerances(device, dtype):
        if dtype == torch.float64:
            return 1e-8, 1e-8
        if device.type != "cuda":
            return 1e-5, 1e-5
        return low_precision_cuda_tolerances(
            device,
            dtype,
            default=(5e-3, 5e-3),
            fp16=(5e-3, 5e-3),
            bf16=(1e-1, 1e-1),
        )

    @pytest.mark.parametrize(
        ("kernel_size", "stride", "padding", "dilation", "groups"),
        [(1, 1, 0, 1, 1), (2, 2, 1, 2, 2)],
        ids=("default", "combined-options"),
    )
    def test_conv1d(self, kernel_size, stride, padding, dilation, groups, device, float_dtype):
        shape = [(5, 8), (7, 8)]
        base = nested_rand(shape, device, float_dtype)
        weight = torch.randn(4, base.shape[-1] // groups, kernel_size, device=device, dtype=float_dtype)
        bias = torch.randn(4, device=device, dtype=float_dtype)
        input = base.transpose(-1, -2)
        output = F.conv1d(input, weight, bias, stride, padding, dilation, groups)
        reference = NT(
            [F.conv1d(t, weight, bias, stride, padding, dilation, groups) for t in input], **reference_options(input)
        )
        atol, rtol = self._tolerances(device, float_dtype)
        assert_close(output, reference, atol=atol, rtol=rtol)

    def test_conv1d_module_pointwise(self, device, float_dtype):
        input = NT(
            [
                torch.randn(2, 5, device=device, dtype=float_dtype),
                torch.randn(2, 7, device=device, dtype=float_dtype),
            ]
        )
        module = nn.Conv1d(2, 4, kernel_size=1).to(device=device, dtype=float_dtype).eval()

        output = module(input)

        reference = NT([module(t) for t in input], **reference_options(input))
        atol, rtol = self._tolerances(device, float_dtype)
        assert_close(output, reference, atol=atol, rtol=rtol)

    def test_conv1d_padding_same(self, device, float_dtype):
        input = NT(
            [
                torch.randn(2, 5, device=device, dtype=float_dtype),
                torch.randn(2, 7, device=device, dtype=float_dtype),
            ]
        )
        weight = torch.randn(4, 2, 3, device=device, dtype=float_dtype)
        bias = torch.randn(4, device=device, dtype=float_dtype)
        output = F.conv1d(input, weight, bias, padding="same")
        reference = NT([F.conv1d(t, weight, bias, padding="same") for t in input], **reference_options(input))
        atol, rtol = self._tolerances(device, float_dtype)
        assert_close(output, reference, atol=atol, rtol=rtol)

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
        reference = NT([F.conv1d(t, weight, bias, stride=1, padding=1) for t in input], **reference_options(input))
        atol, rtol = self._tolerances(device, float_dtype)
        assert_close(output, reference, atol=atol, rtol=rtol)

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
        reference = NT([F.conv1d(t, weight, bias, stride=1, padding=1) for t in input], **reference_options(input))
        atol, rtol = self._tolerances(device, float_dtype)
        assert_close(output, reference, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        ("kernel_size", "stride", "padding", "dilation", "groups"),
        [(1, 1, 0, 1, 1), (2, 2, 1, 2, 2)],
        ids=("default", "combined-options"),
    )
    def test_conv2d(self, kernel_size, stride, padding, dilation, groups, device, float_dtype):
        shape = [(5, 7, 8), (11, 13, 8)]
        base = nested_rand(shape, device, float_dtype)
        weight = torch.randn(4, base.shape[-1] // groups, kernel_size, kernel_size, device=device, dtype=float_dtype)
        bias = torch.randn(4, device=device, dtype=float_dtype)
        input = base.transpose(1, -1)
        output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        reference = NT(
            [F.conv2d(t, weight, bias, stride, padding, dilation, groups) for t in input], **reference_options(input)
        )
        atol, rtol = self._tolerances(device, float_dtype)
        assert_close(output, reference, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        ("kernel_size", "stride", "padding", "dilation", "groups"),
        [(1, 1, 0, 1, 1), (2, 2, 1, 2, 2)],
        ids=("default", "combined-options"),
    )
    def test_conv3d(self, kernel_size, stride, padding, dilation, groups, device, float_dtype):
        shape = [(5, 7, 9, 8), (11, 13, 15, 8)]
        base = nested_rand(shape, device, float_dtype)
        weight = torch.randn(
            4, base.shape[-1] // groups, kernel_size, kernel_size, kernel_size, device=device, dtype=float_dtype
        )
        bias = torch.randn(4, device=device, dtype=float_dtype)
        input = base.permute(0, 4, 1, 2, 3)
        output = F.conv3d(input, weight, bias, stride, padding, dilation, groups)
        reference = NT(
            [F.conv3d(t, weight, bias, stride, padding, dilation, groups) for t in input], **reference_options(input)
        )
        atol, rtol = self._tolerances(device, float_dtype)
        assert_close(output, reference, atol=atol, rtol=rtol)


class TestConvTranspose:

    @staticmethod
    def _tolerances(device, dtype):
        if dtype == torch.float64:
            return 1e-8, 1e-8
        if device.type != "cuda":
            return 1e-5, 1e-5
        return low_precision_cuda_tolerances(
            device,
            dtype,
            default=(5e-3, 5e-3),
            fp16=(5e-3, 5e-3),
            bf16=(1e-1, 1e-1),
        )

    @pytest.mark.parametrize(
        ("kernel_size", "stride", "padding", "output_padding", "groups", "dilation"),
        [(1, 1, 0, 0, 1, 1), (2, 2, 1, 1, 2, 2)],
        ids=("default", "combined-options"),
    )
    def test_conv_transpose1d_functional(
        self, kernel_size, stride, padding, output_padding, groups, dilation, device, float_dtype
    ):
        shape = [(5, 8), (7, 8)]
        input = nested_rand(shape, device, float_dtype)
        weight = torch.randn(input.shape[-1], 4 // groups, kernel_size, device=device, dtype=float_dtype)
        bias = torch.randn(4, device=device, dtype=float_dtype)
        input = input.transpose(1, -1)
        output = F.conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation)
        reference = NT(
            [F.conv_transpose1d(t, weight, bias, stride, padding, output_padding, groups, dilation) for t in input],
            **reference_options(input),
        )
        atol, rtol = self._tolerances(device, float_dtype)
        assert_close(output, reference, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        ("kernel_size", "stride", "padding", "output_padding", "groups", "dilation"),
        [(1, 1, 0, 0, 1, 1), (2, 2, 1, 1, 2, 2)],
        ids=("default", "combined-options"),
    )
    def test_conv_transpose2d_functional(
        self, kernel_size, stride, padding, output_padding, groups, dilation, device, float_dtype
    ):
        shape = [(5, 7, 8), (11, 13, 8)]
        input = nested_rand(shape, device, float_dtype)
        weight = torch.randn(input.shape[-1], 4 // groups, kernel_size, kernel_size, device=device, dtype=float_dtype)
        bias = torch.randn(4, device=device, dtype=float_dtype)
        input = input.permute(0, 3, 1, 2)
        output = F.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)
        reference = NT(
            [F.conv_transpose2d(t, weight, bias, stride, padding, output_padding, groups, dilation) for t in input],
            **reference_options(input),
        )
        atol, rtol = self._tolerances(device, float_dtype)
        assert_close(output, reference, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        ("kernel_size", "stride", "padding", "output_padding", "groups", "dilation"),
        [(1, 1, 0, 0, 1, 1), (2, 2, 1, 1, 2, 2)],
        ids=("default", "combined-options"),
    )
    def test_conv_transpose3d_functional(
        self, kernel_size, stride, padding, output_padding, groups, dilation, device, float_dtype
    ):
        shape = [(5, 7, 9, 8), (11, 13, 15, 8)]
        input = nested_rand(shape, device, float_dtype)
        weight = torch.randn(
            input.shape[-1], 4 // groups, kernel_size, kernel_size, kernel_size, device=device, dtype=float_dtype
        )
        bias = torch.randn(4, device=device, dtype=float_dtype)
        input = input.transpose(1, -1)
        output = F.conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation)
        reference = NT(
            [F.conv_transpose3d(t, weight, bias, stride, padding, output_padding, groups, dilation) for t in input],
            **reference_options(input),
        )
        atol, rtol = self._tolerances(device, float_dtype)
        assert_close(output, reference, atol=atol, rtol=rtol)


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
        output = F.dropout(nt, p=1.0, training=True)
        assert_close(output, torch.zeros_like(nt))

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
        reference = NT([F.embedding(t, weight) for t in nt_idx], **reference_options(nt_idx))
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("ragged_dims", [(0, 1), (1, 0)])
    def test_embedding_multi_ragged_values_and_vjp(
        self,
        device,
        float_dtype,
        ragged_dims,
    ):
        indices = NT(
            [
                torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.long, device=device),
                torch.tensor([[0, 7], [8, 2], [6, 1]], dtype=torch.long, device=device),
            ],
            ragged_dims=ragged_dims,
        )
        weight = torch.randn(10, 4, device=device, dtype=float_dtype, requires_grad=True)
        reference_weight = weight.detach().clone().requires_grad_()

        output = F.embedding(indices, weight)

        reference = F.embedding(indices.concat, reference_weight)
        cotangent = torch.randn_like(reference)
        output_gradient = torch.autograd.grad(output.concat, weight, cotangent)[0]
        reference_gradient = torch.autograd.grad(reference, reference_weight, cotangent)[0]
        assert output.ragged_dims == ragged_dims
        assert [element.shape for element in output] == [torch.Size((2, 3, 4)), torch.Size((3, 2, 4))]
        assert_close(output.concat, reference)
        assert_close(output_gradient, reference_gradient)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_embedding_multi_ragged_compiles_with_vjp(self, device):
        compiled = torch.compile(
            lambda indices, weight: F.embedding(indices, weight).square().concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        indices = NT(
            [torch.arange(6, device=device).reshape(2, 3), torch.arange(4, device=device).reshape(1, 4)],
            ragged_dims=(0, 1),
        )
        weight = torch.randn(13, 4, device=device, requires_grad=True)
        reference_weight = weight.detach().clone().requires_grad_()
        output = compiled(indices, weight)
        reference = F.embedding(indices.concat, reference_weight).square()
        cotangent = torch.randn_like(reference)
        output_gradient = torch.autograd.grad(output, weight, cotangent)[0]
        reference_gradient = torch.autograd.grad(reference, reference_weight, cotangent)[0]
        assert_close(output, reference)
        assert_close(output_gradient, reference_gradient)

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
        reference = NT(
            [F.embedding_bag(t, weight, offsets=offsets, mode="mean") for t in nt_idx], **reference_options(nt_idx)
        )
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
            **reference_options(nt_idx),
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
        reference = NT(
            [F.embedding_bag(t, weight, offsets=offsets, mode="sum") for t in nt_idx], **reference_options(nt_idx)
        )
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
            **reference_options(nt_idx),
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
            **reference_options(x),
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
            **reference_options(x),
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
        reference_output = NT(reference_output, **reference_options(nt))
        reference_idx = NT(reference_idx, **reference_options(nt))
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
            **reference_options(nt_imgs),
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
                torch.arange(8.0, device=device, dtype=float_dtype).view(2, 2, 2),
                torch.ones(2, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.interpolate(nt, scale_factor=2, mode="bilinear", align_corners=False)
        reference = NT(
            [
                F.interpolate(t.unsqueeze(0), scale_factor=2, mode="bilinear", align_corners=False).squeeze(0)
                for t in nt
            ],
            **reference_options(nt),
        )
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_interpolate_nearest(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(8.0, device=device, dtype=float_dtype).view(2, 2, 2),
                torch.ones(2, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.interpolate(nt, scale_factor=2, mode="nearest")
        reference = NT(
            [F.interpolate(t.unsqueeze(0), scale_factor=2, mode="nearest").squeeze(0) for t in nt],
            **reference_options(nt),
        )
        assert_close(output, reference)

    def test_interpolate_ragged_spatial(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(8.0, device=device, dtype=float_dtype).view(2, 2, 2),
                torch.ones(2, 3, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.interpolate(nt, scale_factor=2, mode="nearest")
        reference = NT(
            [F.interpolate(t.unsqueeze(0), scale_factor=2, mode="nearest").squeeze(0) for t in nt],
            **reference_options(nt),
        )
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
        reference = NT([F.linear(t, weight, bias) for t in input], **reference_options(input))
        assert_close(output, reference)

    def test_permuted_static_features_values_and_vjp(self, device, float_dtype):
        template = NT(
            [torch.empty(2, 78, 2, device=device), torch.empty(5, 78, 2, device=device)],
            ragged_dims=(0,),
        )
        values = torch.randn_like(template.concat, dtype=float_dtype, requires_grad=True)
        weight = torch.randn(8, 78, device=device, dtype=float_dtype, requires_grad=True)
        bias = torch.randn(8, device=device, dtype=float_dtype, requires_grad=True)
        input = template.packed_like(values).movedim(2, 3)
        expected = F.linear(values.movedim(1, 2), weight, bias)

        output = F.linear(input, weight, bias)

        leaves = (values, weight, bias)
        cotangent = torch.randn_like(expected)
        actual_gradients = torch.autograd.grad(output.concat, leaves, cotangent, retain_graph=True)
        expected_gradients = torch.autograd.grad(expected, leaves, cotangent)
        assert output.shape == torch.Size((2, 5, 2, 8))
        assert output.ragged_dims == (0,)
        assert_close(output.concat, expected)
        for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
            assert_close(actual, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_permuted_static_features_compile_with_vjp(self, device):
        def run(template, values, weight, bias):
            input = template.packed_like(values).movedim(2, 3)
            return F.linear(input, weight, bias).concat

        compiled = torch.compile(run, backend="aot_eager", fullgraph=True, dynamic=True)
        template = NT([torch.empty(2, 78, 2), torch.empty(3, 78, 2)], ragged_dims=(0,))
        values = torch.randn_like(template.concat, device=device, requires_grad=True)
        weight = torch.randn(8, 78, device=device, requires_grad=True)
        bias = torch.randn(8, device=device, requires_grad=True)
        expected = F.linear(values.movedim(1, 2), weight, bias)
        output = compiled(template, values, weight, bias)
        leaves = (values, weight, bias)
        cotangent = torch.randn_like(expected)
        actual_gradients = torch.autograd.grad(output, leaves, cotangent)
        expected_gradients = torch.autograd.grad(expected, leaves, cotangent)
        assert_close(output, expected)
        for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
            assert_close(actual, reference)


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
        reference = NT([reference], **reference_options(unpooled))
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
        reference = NT([reference], **reference_options(unpooled))
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
        reference = NT([reference], **reference_options(unpooled))
        assert_close(unpooled, reference)


class TestModuleIntegration:

    def test_conv2d_module(self, device, float_dtype):
        input = nested_rand([(5, 7, 8), (11, 13, 8)], device, float_dtype).permute(0, 3, 1, 2)
        layer = nn.Conv2d(input.shape[1], 4, kernel_size=2, padding=1).to(device=device, dtype=float_dtype)
        reference_layer = nn.Conv2d(input.shape[1], 4, kernel_size=2, padding=1).to(device=device, dtype=float_dtype)
        reference_layer.load_state_dict(layer.state_dict())

        output = layer(input)
        reference_storage = [reference_layer(t.unsqueeze(0)).squeeze(0) for t in input]
        reference = NT(reference_storage, **reference_options(input))
        atol, rtol = TestConv._tolerances(device, float_dtype)
        assert_close(output, reference, atol=atol, rtol=rtol)

        output.sum().backward()
        sum(part.sum() for part in reference_storage).backward()
        assert_close(layer.weight.grad, reference_layer.weight.grad, atol=atol, rtol=max(rtol, 1e-2))
        assert_close(layer.bias.grad, reference_layer.bias.grad, atol=atol, rtol=max(rtol, 1e-2))

    def test_linear_module(self, device, float_dtype):
        input = nested_rand([(3, 5), (2, 5)], device, float_dtype)
        layer = nn.Linear(input.shape[-1], 3).to(device=device, dtype=float_dtype)
        reference_layer = nn.Linear(input.shape[-1], 3).to(device=device, dtype=float_dtype)
        reference_layer.load_state_dict(layer.state_dict())

        output = layer(input)
        reference = [reference_layer(element) for element in input]
        assert [element.shape for element in output] == [element.shape for element in reference]
        for actual, expected in zip(output, reference):
            assert_close(actual, expected)

        sum(element.sum() for element in output).backward()
        # d(sum(y))/dW is the input column sum. Per-element half-precision
        # gradients round before accumulation and are not an exact oracle.
        columns = torch.cat([element.detach().cpu().double() for element in input]).T.tolist()
        expected_weight = layer.weight.new_tensor([math.fsum(column) for column in columns])
        assert_close(layer.weight.grad, expected_weight.expand_as(layer.weight))
        assert_close(layer.bias.grad, torch.full_like(layer.bias, sum(len(element) for element in input)))

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compiled_linear_module_dynamic_fullgraph_gradients(self, device):
        layer = nn.Linear(3, 4).to(device)
        reference_layer = nn.Linear(3, 4).to(device)
        reference_layer.load_state_dict(layer.state_dict())
        compiled = torch.compile(layer, backend="aot_eager", fullgraph=True, dynamic=True)

        for lengths in ((2, 4), (3, 1, 5)):
            elements = [torch.randn(length, 3, device=device, requires_grad=True) for length in lengths]
            reference_elements = [element.detach().clone().requires_grad_() for element in elements]

            output = compiled(NT(elements, ragged_dims=(0,))).concat
            reference = torch.cat([reference_layer(element) for element in reference_elements])
            assert_close(output, reference)

            output.square().sum().backward()
            reference.square().sum().backward()

            for actual, expected in zip(elements, reference_elements, strict=True):
                assert_close(actual.grad, expected.grad)
            assert_close(layer.weight.grad, reference_layer.weight.grad)
            assert_close(layer.bias.grad, reference_layer.bias.grad)

            layer.zero_grad(set_to_none=True)
            reference_layer.zero_grad(set_to_none=True)

    @pytest.mark.parametrize(
        "module_factory",
        [
            pytest.param(lambda: nn.RNN(8, 16, batch_first=True), id="rnn"),
            pytest.param(lambda: nn.GRU(8, 16, batch_first=True), id="gru"),
            pytest.param(lambda: nn.LSTM(8, 16, num_layers=2, bidirectional=True, batch_first=True), id="lstm"),
        ],
    )
    def test_recurrent_module(self, device, module_factory, monkeypatch):
        # This comparison asks for float32 accuracy, including inside cuDNN.
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
        dtype = torch.float32
        elements = [torch.randn(length, 8, device=device, dtype=dtype) for length in (4, 7, 5)]
        nested = NT([t.clone() for t in elements])
        module = module_factory().to(device=device, dtype=dtype).eval()

        with torch.no_grad():
            output = module(nested)[0]
            reference = NT([module(t.unsqueeze(0))[0].squeeze(0) for t in elements], **reference_options(nested))

        assert isinstance(output, NestedTensor)
        assert_close(output, reference, atol=1e-4, rtol=1e-4)
        if isinstance(module, nn.RNN):
            weight_ih = module.weight_ih_l0.detach().cpu().double()
            weight_hh = module.weight_hh_l0.detach().cpu().double()
            bias_ih = module.bias_ih_l0.detach().cpu().double()
            bias_hh = module.bias_hh_l0.detach().cpu().double()
            for actual, element in zip(output, elements, strict=True):
                hidden = torch.zeros(module.hidden_size, dtype=torch.float64)
                expected = []
                for step in element.cpu().double():
                    hidden = torch.tanh(weight_ih @ step + bias_ih + weight_hh @ hidden + bias_hh)
                    expected.append(hidden)
                assert_close(actual.cpu().double(), torch.stack(expected), atol=1e-4, rtol=1e-4)

    def test_lstm_state(self, device):
        dtype = torch.float32
        elements = [torch.randn(length, 8, device=device, dtype=dtype) for length in (4, 7, 5)]
        nested = NT([t.clone() for t in elements])
        module = nn.LSTM(8, 16, batch_first=True).to(device=device, dtype=dtype).eval()

        with torch.no_grad():
            _, (h_n, c_n) = module(nested)
            states = [module(t.unsqueeze(0))[1] for t in elements]

        assert_close(h_n, torch.cat([state[0] for state in states], dim=1), atol=1e-4, rtol=1e-4)
        assert_close(c_n, torch.cat([state[1] for state in states], dim=1), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_recurrent_empty(self, device, batch_first):
        dtype = torch.float32
        elements = [torch.randn(0, 8, device=device, dtype=dtype), torch.randn(3, 8, device=device, dtype=dtype)]
        nested = NT([t.clone() for t in elements], batch_first=batch_first)
        module = nn.GRU(8, 16, batch_first=batch_first).to(device=device, dtype=dtype).eval()
        batch_axis = 0 if batch_first else 1

        with torch.no_grad():
            output, hidden = module(nested)
            reference = module(elements[1].unsqueeze(batch_axis))[0].squeeze(batch_axis)

        assert [tuple(t.shape) for t in output] == [(0, 16), (3, 16)]
        assert_close(output[1], reference, atol=1e-4, rtol=1e-4)
        assert hidden.shape == (1, 2, 16)

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_recurrent_all_empty(self, device, batch_first):
        dtype = torch.float32
        nested = NT([torch.randn(0, 8, device=device, dtype=dtype) for _ in range(2)], batch_first=batch_first)
        module = nn.GRU(8, 16, batch_first=batch_first).to(device=device, dtype=dtype).eval()

        with torch.no_grad():
            output, hidden = module(nested)

        assert [tuple(t.shape) for t in output] == [(0, 16), (0, 16)]
        assert hidden.shape == (1, 2, 16)


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
            **reference_options(nt),
        )
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_batch_norm_eval_channel_first_stays_packed(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(4, 17, device=device, dtype=float_dtype),
                torch.randn(4, 29, device=device, dtype=float_dtype),
            ]
        )
        running_mean = torch.randn(4, device=device, dtype=float_dtype)
        running_var = torch.rand(4, device=device, dtype=float_dtype) + 0.5
        weight = torch.randn(4, device=device, dtype=float_dtype)
        bias = torch.randn(4, device=device, dtype=float_dtype)

        output = F.batch_norm(nt, running_mean, running_var, weight, bias, training=False)
        reference = NT(
            [
                F.batch_norm(t.unsqueeze(0), running_mean, running_var, weight, bias, training=False).squeeze(0)
                for t in nt
            ],
            **reference_options(nt),
        )

        assert isinstance(output, NestedTensor)
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_group_norm(self, device, float_dtype):
        nt = nested_rand([(3, 4), (2, 4)], device, float_dtype)
        output = F.group_norm(nt, num_groups=1)
        reference = NT([F.group_norm(t.unsqueeze(0), num_groups=1).squeeze(0) for t in nt], **reference_options(nt))
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_instance_norm(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(3, 2, 2, device=device, dtype=float_dtype),
                torch.randn(3, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.instance_norm(nt, use_input_stats=True)
        reference = NT(
            [F.instance_norm(t.unsqueeze(0), use_input_stats=True).squeeze(0) for t in nt], **reference_options(nt)
        )
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_layer_norm(self, device, float_dtype):
        nt = nested_rand([(3, 4), (2, 4)], device, float_dtype)
        output = F.layer_norm(nt, normalized_shape=(4,))
        reference = NT([F.layer_norm(t, (4,)) for t in nt], **reference_options(nt))
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_local_response_norm(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(3, 3, 3, device=device, dtype=float_dtype),
                torch.randn(3, 3, 3, device=device, dtype=float_dtype),
            ]
        )
        output = F.local_response_norm(nt, size=2)
        reference = NT([F.local_response_norm(t.unsqueeze(0), size=2).squeeze(0) for t in nt], **reference_options(nt))
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_rms_norm(self, device, float_dtype):
        nt = nested_rand([(1, 4), (1, 4)], device, float_dtype)
        output = F.rms_norm(nt, normalized_shape=(4,))
        reference = NT([F.rms_norm(t, (4,)) for t in nt], **reference_options(nt))
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
        reference = NT([F.normalize(t, dim=0) for t in input], **reference_options(input))
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_normalize_static_tail_values_and_vjp(self, device, float_dtype):
        template = NT(
            [
                torch.empty(2, 4, device=device, dtype=float_dtype),
                torch.empty(5, 4, device=device, dtype=float_dtype),
            ],
            ragged_dims=(0,),
        )
        values = torch.randn_like(template.concat, requires_grad=True)
        input = template.packed_like(values)
        reference = F.normalize(values, p=1.5, dim=-1, eps=0.25)
        output = F.normalize(input, p=1.5, dim=-1, eps=0.25)

        cotangent = torch.randn_like(reference)
        output_gradient = torch.autograd.grad(output.concat, values, cotangent)[0]
        reference_gradient = torch.autograd.grad(reference, values, cotangent)[0]
        assert output.shape == input.shape
        assert output.ragged_dims == input.ragged_dims
        assert output.concat.shape == values.shape
        assert_close(output.concat, reference, atol=1e-6, rtol=1e-6)
        assert_close(output_gradient, reference_gradient, atol=1e-6, rtol=1e-6)

    def test_normalize_static_tail_nonleading_ragged_layout_with_vjp(self, device, float_dtype):
        template = NT(
            [
                torch.empty(2, 3, 4, device=device, dtype=float_dtype),
                torch.empty(2, 5, 4, device=device, dtype=float_dtype),
            ],
            ragged_dims=(1,),
        )
        values = torch.randn_like(template.concat, requires_grad=True)
        input = template.packed_like(values)
        reference = F.normalize(values, dim=-1)

        output = F.normalize(input, dim=-1)

        cotangent = torch.randn_like(reference)
        output_gradient = torch.autograd.grad(output.concat, values, cotangent)[0]
        reference_gradient = torch.autograd.grad(reference, values, cotangent)[0]
        assert output.ragged_dims == (1,)
        assert output.shape == input.shape
        assert_close(output.concat, reference)
        assert_close(output_gradient, reference_gradient)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_normalize_static_tail_compiles_with_vjp(self, device):
        compiled = torch.compile(
            lambda template, values: F.normalize(
                template.packed_like(values),
                p=1.5,
                dim=-1,
                eps=0.25,
            ).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        template = NT([torch.empty(2, 4), torch.empty(3, 4)], ragged_dims=(0,))
        values = torch.randn_like(template.concat, device=device, requires_grad=True)
        reference = F.normalize(values, p=1.5, dim=-1, eps=0.25)
        output = compiled(template, values)
        cotangent = torch.randn_like(reference)
        output_gradient = torch.autograd.grad(output, values, cotangent)[0]
        reference_gradient = torch.autograd.grad(reference, values, cotangent)[0]
        assert_close(output, reference)
        assert_close(output_gradient, reference_gradient)


class TestOneHot:

    def test_one_hot(self):
        x = NT([torch.tensor([0, 1, 2], dtype=torch.long), torch.tensor([1, 0], dtype=torch.long)])
        output = F.one_hot(x, num_classes=3)
        reference = NT([F.one_hot(t, num_classes=3) for t in x], **reference_options(x))
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
        reference = NT([F.pad(t, (1, 1, 1, 1), value=0.5) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_pad_ragged_leading_dim(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(4.0, device=device, dtype=float_dtype).view(1, 1, 2, 2),
                torch.arange(8.0, device=device, dtype=float_dtype).view(2, 1, 2, 2),
            ]
        )
        output = F.pad(nt, (1, 1, 1, 1), value=0.25)
        reference = NT([F.pad(t, (1, 1, 1, 1), value=0.25) for t in nt], **reference_options(nt))
        assert_close(output, reference)

    def test_pad_variable_last_dim(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(4, 17, device=device, dtype=float_dtype),
                torch.randn(4, 29, device=device, dtype=float_dtype),
            ]
        )
        output = F.pad(nt, (3, 5), value=0.25)
        reference = NT([F.pad(t, (3, 5), value=0.25) for t in nt], **reference_options(nt))
        assert_close(output, reference)


class TestPairwiseDistance:

    def test_pairwise_distance(self, device, float_dtype):
        x1 = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        x2 = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        output = F.pairwise_distance(x1, x2)
        reference = NT([F.pairwise_distance(a, b) for a, b in zip(x1, x2)], **reference_options(x1))
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_pairwise_distance_p1(self, device, float_dtype):
        x1 = nested_rand([(2, 4), (3, 4)], device, float_dtype)
        x2 = nested_rand([(2, 4), (3, 4)], device, float_dtype)
        output = F.pairwise_distance(x1, x2, p=1)
        reference = NT([F.pairwise_distance(a, b, p=1) for a, b in zip(x1, x2)], **reference_options(x1))
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
            reference = NT([F.pdist(t) for t in x], **reference_options(x))
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
        reference = NT([F.pixel_shuffle(t, upscale_factor=2) for t in nt], **reference_options(nt))
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

    @pytest.mark.skipif(flex_attention is None, reason="FlexAttention not available")
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

        output = flex_attention(query, key, value)
        reference = NT(
            [F.scaled_dot_product_attention(q, k, v, dropout_p=0.0) for q, k, v in zip(query, key, value)],
            **reference_options(query),
        )
        assert isinstance(output, NT)
        assert_close(output, reference, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(flex_attention is None, reason="FlexAttention not available")
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
        output, lse = flex_attention(query, key, value, block_mask=block_mask, return_lse=True)
        reference = NT(
            [
                F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
                for q, k, v in zip(query, key, value)
            ],
            **reference_options(query),
        )
        assert isinstance(output, NT)
        assert isinstance(lse, NT)
        assert_close(output, reference, atol=1e-4, rtol=1e-4)

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
            **reference_options(query),
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
            **reference_options(query),
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
            **reference_options(query),
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
        query_parts = [
            torch.randn(2, length, 8, device=device, dtype=float_dtype, requires_grad=True) for length in (6, 4)
        ]
        key_parts = [
            torch.randn(2, length, 8, device=device, dtype=float_dtype, requires_grad=True) for length in (7, 3)
        ]
        value_parts = [
            torch.randn(2, length, 8, device=device, dtype=float_dtype, requires_grad=True) for length in (7, 3)
        ]
        reference_query = [part.detach().clone().requires_grad_() for part in query_parts]
        reference_key = [part.detach().clone().requires_grad_() for part in key_parts]
        reference_value = [part.detach().clone().requires_grad_() for part in value_parts]
        query = NT(query_parts)
        key = NT(key_parts)
        value = NT(value_parts)
        output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        reference = [
            F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            for q, k, v in zip(reference_query, reference_key, reference_value)
        ]
        atol, rtol = low_precision_cuda_tolerances(
            device,
            float_dtype,
            default=(1e-5, 1e-5),
            fp16=(1e-3, 1e-3),
            bf16=(5e-3, 5e-3),
        )
        for actual, expected in zip(output, reference):
            assert_close(actual, expected, atol=atol, rtol=rtol)

        weights = [torch.randn_like(element) for element in reference]
        loss = sum((element * weight).sum() for element, weight in zip(output, weights))
        reference_loss = sum((element * weight).sum() for element, weight in zip(reference, weights))
        gradients = torch.autograd.grad(loss, (*query_parts, *key_parts, *value_parts))
        reference_gradients = torch.autograd.grad(
            reference_loss,
            (*reference_query, *reference_key, *reference_value),
        )
        for actual, expected in zip(gradients, reference_gradients):
            assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_sdpa_is_causal(self, device, float_dtype):
        query, key, value = (
            NT([torch.randn(2, length, 8, device=device, dtype=float_dtype) for length in (6, 4)]) for _ in range(3)
        )
        output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        reference = [F.scaled_dot_product_attention(q, k, v, is_causal=True) for q, k, v in zip(query, key, value)]
        atol, rtol = low_precision_cuda_tolerances(
            device,
            float_dtype,
            default=(1e-5, 1e-5),
            fp16=(1e-3, 1e-3),
            bf16=(5e-3, 5e-3),
        )
        for actual, expected in zip(output, reference):
            assert_close(actual, expected, atol=atol, rtol=rtol)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Native FlashAttention requires CUDA")
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("kind", ["bidirectional", "causal", "cross"])
    def test_sdpa_inductor_dynamic_training_preserves_structure_and_gradients(self, batch_first, kind):
        if not torch.cuda.is_bf16_supported() or torch.cuda.get_device_capability()[0] < 8:
            pytest.skip("Native BF16 FlashAttention requires an Ampere-or-newer GPU")
        from itertools import accumulate

        from torch.nn.attention import SDPBackend, sdpa_kernel

        class ProjectedAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(32, 32)
                self.key = nn.Linear(32, 32)
                self.value = nn.Linear(32, 32)
                self.output = nn.Linear(32, 32)

            def forward(self, query, key, value):
                sequence_axis = 1 if batch_first else 0
                query = self.query(query).unflatten(-1, (4, 8)).transpose(sequence_axis, 2)
                key = self.key(key).unflatten(-1, (4, 8)).transpose(sequence_axis, 2)
                value = self.value(value).unflatten(-1, (4, 8)).transpose(sequence_axis, 2)
                attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=kind == "causal")
                return self.output(attended.transpose(sequence_axis, 2).flatten(-2))

            def dense(self, query, key, value):
                query = self.query(query).unflatten(-1, (4, 8)).transpose(0, 1)
                key = self.key(key).unflatten(-1, (4, 8)).transpose(0, 1)
                value = self.value(value).unflatten(-1, (4, 8)).transpose(0, 1)
                attended = F.scaled_dot_product_attention(
                    query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), dropout_p=0.0, is_causal=kind == "causal"
                ).squeeze(0)
                return self.output(attended.transpose(0, 1).flatten(-2))

        model = ProjectedAttention().cuda().bfloat16()
        reference_model = ProjectedAttention().cuda().bfloat16()
        reference_model.load_state_dict(model.state_dict())
        query_patterns = ((2, 5, 5, 7), (3, 3, 6, 7), (1, 3, 5, 10), (4, 4, 6, 9))
        key_patterns = ((5, 2, 7, 5), (3, 7, 3, 6), (10, 1, 3, 5), (3, 8, 4, 7))
        torch.compiler.reset()
        compiled = torch.compile(model, backend="inductor", fullgraph=True, dynamic=True)
        try:
            # Flash is a declared numerical backend here; unsupported configurations
            # must fail rather than silently changing to a padded or per-element path.
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                for index, query_lengths in enumerate(query_patterns):
                    key_lengths = key_patterns[index] if kind == "cross" else query_lengths
                    parts = tuple(
                        [
                            torch.randn(length, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                            for length in lengths
                        ]
                        for lengths in (query_lengths, key_lengths, key_lengths)
                    )
                    reference_parts = tuple([part.detach().clone().requires_grad_() for part in side] for side in parts)
                    nested = tuple(NT(side, ragged_dims=(0,), batch_first=batch_first) for side in parts)
                    model.zero_grad(set_to_none=True)
                    reference_model.zero_grad(set_to_none=True)
                    output = compiled(*nested)
                    expected = [reference_model.dense(q, k, v) for q, k, v in zip(*reference_parts)]
                    assert isinstance(output, NT)
                    assert output.batch_first is batch_first
                    assert output.ragged_dims == (0,)
                    assert output.element_sizes().tolist() == [[length, 32] for length in query_lengths]
                    assert output.packed_offsets(device="cpu").tolist() == [0, *accumulate(query_lengths)]
                    assert output.ragged_level_offsets(0, device="cpu").tolist() == [0, *accumulate(query_lengths)]
                    for actual, reference in zip(output, expected):
                        assert_close(actual, reference, atol=5e-3, rtol=2e-2)

                    # Consume the returned wrapper in eager code, exercising the
                    # forward/backward graph boundary as well as all Q/K/V inputs.
                    weights = [torch.randn_like(part) for part in expected]
                    loss = (output.concat.float() * torch.cat(weights).float()).sum()
                    reference_loss = sum(
                        (part.float() * weight.float()).sum() for part, weight in zip(expected, weights)
                    )
                    loss.backward()
                    reference_loss.backward()
                    for side, reference_side in zip(parts, reference_parts):
                        for part, reference in zip(side, reference_side):
                            assert part.grad is not None and reference.grad is not None
                            assert torch.isfinite(part.grad).all()
                            assert_close(part.grad, reference.grad, atol=5e-3, rtol=2e-2)
                    for parameter, reference in zip(model.parameters(), reference_model.parameters()):
                        assert parameter.grad is not None and reference.grad is not None
                        assert torch.isfinite(parameter.grad).all()
                        assert_close(parameter.grad, reference.grad, atol=5e-2, rtol=2e-2)
        finally:
            torch.compiler.reset()

    def test_sdpa_mismatched_batch_lengths_raises(self):
        query = NT([torch.randn(2, 4, 8), torch.randn(2, 3, 8)])
        key = NT([torch.randn(2, 4, 8)])
        with pytest.raises(ValueError, match="NestedTensor batch length mismatch"):
            F.scaled_dot_product_attention(query, key, key, dropout_p=0.0)

    def test_sdpa_requires_nested_query(self):
        tensor_query = torch.randn(2, 2, 4, 8)
        key = NT([torch.randn(2, 4, 8)])
        with pytest.raises(TypeError):
            F.scaled_dot_product_attention(tensor_query, key, key, dropout_p=0.0)

    def test_sdpa_tensor_key_value(self, device, float_dtype):
        query = NT(
            [
                torch.randn(2, 6, 16, device=device, dtype=float_dtype),
                torch.randn(2, 4, 16, device=device, dtype=float_dtype),
            ]
        )
        output = F.scaled_dot_product_attention(query, query.tensor, query.tensor, dropout_p=0.0)
        reference = NT(
            [F.scaled_dot_product_attention(q, q, q, dropout_p=0.0) for q in query], **reference_options(query)
        )
        atol, rtol = low_precision_cuda_tolerances(
            device,
            float_dtype,
            default=(1e-5, 1e-5),
            fp16=(1e-3, 1e-3),
            bf16=(5e-3, 5e-3),
        )
        assert_close(output, reference, atol=atol, rtol=rtol)

    @staticmethod
    def _ragged_attention_reference(query, key, value, score_mod=None, scale=None):
        scale = scale if scale is not None else query[0].shape[-1] ** -0.5
        outputs = []
        for index, (q, k, v) in enumerate(zip(query, key, value)):
            heads, q_len, _ = q.shape
            kv_len = k.shape[1]
            scores = (q @ k.transpose(-1, -2)) * scale
            if score_mod is not None:
                head = torch.arange(heads, device=q.device).view(heads, 1, 1)
                q_idx = torch.arange(q_len, device=q.device).view(1, q_len, 1)
                kv_idx = torch.arange(kv_len, device=q.device).view(1, 1, kv_len)
                batch = torch.tensor(index, device=q.device)
                scores = score_mod(scores, batch, head, q_idx, kv_idx).broadcast_to(scores.shape)
            outputs.append(torch.softmax(scores, dim=-1) @ v)
        return NT(outputs, **reference_options(query))

    @pytest.mark.skipif(flex_attention is None, reason="FlexAttention not available")
    def test_flex_eager_default_matches_sdpa(self, device):
        query = NT([torch.randn(4, 3, 16, device=device), torch.randn(4, 5, 16, device=device)])
        key = NT([torch.randn(4, 7, 16, device=device), torch.randn(4, 2, 16, device=device)])
        value = NT([torch.randn(4, 7, 8, device=device), torch.randn(4, 2, 8, device=device)])

        output = flex_attention(query, key, value)
        reference = NT(
            [F.scaled_dot_product_attention(q, k, v, dropout_p=0.0) for q, k, v in zip(query, key, value)],
            **reference_options(query),
        )
        assert isinstance(output, NT)
        assert_close(output, reference, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(flex_attention is None, reason="FlexAttention not available")
    def test_flex_eager_cross_attention(self, device):
        # Eager ragged Flex with per-sequence q_len != kv_len must not shape-error.
        query = NT([torch.randn(4, 3, 16, device=device), torch.randn(4, 5, 16, device=device)])
        key = NT([torch.randn(4, 7, 16, device=device), torch.randn(4, 2, 16, device=device)])
        value = NT([torch.randn(4, 7, 16, device=device), torch.randn(4, 2, 16, device=device)])
        slopes = torch.arange(1, 5, device=device, dtype=torch.float32) * 0.1

        def score_mod(score, b, h, q_idx, kv_idx):
            return score - slopes[h] * (q_idx - kv_idx).abs()

        output = flex_attention(query, key, value, score_mod=score_mod)
        reference = self._ragged_attention_reference(query, key, value, score_mod)
        assert isinstance(output, NT)
        assert_close(output, reference, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(flex_attention is None, reason="FlexAttention not available")
    def test_flex_eager_non_additive_score_mod(self, device):
        # score_mod is an arbitrary transform, not necessarily an additive bias.
        query = NT([torch.randn(4, 3, 16, device=device), torch.randn(4, 7, 16, device=device)])
        key = NT([torch.randn(4, 3, 16, device=device), torch.randn(4, 7, 16, device=device)])
        value = NT([torch.randn(4, 3, 16, device=device), torch.randn(4, 7, 16, device=device)])

        def score_mod(score, b, h, q_idx, kv_idx):
            return score * 2

        output = flex_attention(query, key, value, score_mod=score_mod)
        reference = self._ragged_attention_reference(query, key, value, score_mod)
        assert_close(output, reference, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(flex_attention is None, reason="FlexAttention not available")
    def test_flex_eager_broadcast_valued_score_mod(self, device):
        # score_mod may return a broadcastable result rather than a fully expanded (heads, q, kv).
        query = NT([torch.randn(4, 5, 16, device=device), torch.randn(4, 2, 16, device=device)])
        key = NT([torch.randn(4, 5, 16, device=device), torch.randn(4, 2, 16, device=device)])
        value = NT([torch.randn(4, 5, 16, device=device), torch.randn(4, 2, 16, device=device)])

        def score_mod(score, b, h, q_idx, kv_idx):
            return q_idx.to(score.dtype)

        output = flex_attention(query, key, value, score_mod=score_mod)
        reference = self._ragged_attention_reference(query, key, value, score_mod)
        assert isinstance(output, NT)
        assert_close(output, reference, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(flex_attention is None, reason="FlexAttention not available")
    def test_flex_eager_preserves_batch_first(self, device):
        query = NT([torch.randn(4, 3, 16, device=device), torch.randn(4, 5, 16, device=device)], batch_first=False)
        key = NT([torch.randn(4, 3, 16, device=device), torch.randn(4, 5, 16, device=device)], batch_first=False)
        value = NT([torch.randn(4, 3, 16, device=device), torch.randn(4, 5, 16, device=device)], batch_first=False)

        def score_mod(score, b, h, q_idx, kv_idx):
            return score * 2

        output = flex_attention(query, key, value, score_mod=score_mod)
        assert output.batch_first is False
        reference = self._ragged_attention_reference(query, key, value, score_mod)
        assert_close(output, reference, atol=1e-4, rtol=1e-4)


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
        reference = nt.packed_like(F.gumbel_softmax(nt.concat, dim=-1))
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
    def test_softmax_family_nonleading_ragged_axis(self, op, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, 3, device=device, dtype=float_dtype),
                torch.randn(2, 5, 5, device=device, dtype=float_dtype),
            ]
        )
        output = op(nt, dim=-1)
        reference = NT([op(t, dim=-1) for t in nt], **reference_options(nt))
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("op", [F.softmax, F.log_softmax, F.softmin])
    def test_softmax_family_ragged_axis(self, op, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0, 4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = op(nt, dim=1)
        reference = NT([op(t, dim=0) for t in nt], **reference_options(nt))
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
        reference = NT([F.unfold(t, kernel_size=2, stride=1) for t in nt], **reference_options(nt))
        assert_close(output, reference)

        unfolded = output
        output = F.fold(unfolded, output_size=(3, 3), kernel_size=2, stride=1)
        reference = NT(
            [F.fold(t, output_size=(3, 3), kernel_size=2, stride=1) for t in unfolded], **reference_options(unfolded)
        )
        assert_close(output, reference, atol=1e-6, rtol=1e-6)


class TestCrossEntropy:
    r"""PyTorch class-axis cross entropy retaining native target positions."""

    @staticmethod
    def _logits_and_targets(device, float_dtype):
        logits = NestedTensor(
            [
                torch.randn(2, 5, device=device, dtype=float_dtype),
                torch.randn(3, 5, device=device, dtype=float_dtype),
            ]
        )
        targets = NestedTensor(
            [
                torch.randint(0, 5, (2,), device=device),
                torch.randint(0, 5, (3,), device=device),
            ]
        )
        return logits.movedim(-1, 1), targets

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_matches_dense_on_packed_rows(self, device, float_dtype, reduction):
        # Logical input is (B, C, L); its packed rows are (sum(L), C).
        logits, targets = self._logits_and_targets(device, float_dtype)
        output = F.cross_entropy(logits, targets, reduction=reduction)
        reference = F.cross_entropy(logits.concat, targets.concat, reduction=reduction)
        torch.testing.assert_close(output, reference)

    def test_weight_and_ignore_index_match_dense(self, device, float_dtype):
        logits, targets = self._logits_and_targets(device, float_dtype)
        weight = torch.rand(5, device=device, dtype=float_dtype)
        ignore_index = int(targets[0][0])

        kwargs = {"weight": weight, "ignore_index": ignore_index}
        torch.testing.assert_close(
            F.cross_entropy(logits, targets, **kwargs),
            F.cross_entropy(logits.concat, targets.concat, **kwargs),
        )

    def test_reduction_none_keeps_multi_ragged_structure(self, device, float_dtype):
        # The generic path concatenates into one (rows, C) matrix, which flattens this away.
        logits = NestedTensor(
            [
                torch.randn(2, 2, 5, device=device, dtype=float_dtype),
                torch.randn(3, 3, 5, device=device, dtype=float_dtype),
            ]
        )
        targets = NestedTensor(
            [
                torch.randint(0, 5, (2, 2), device=device),
                torch.randint(0, 5, (3, 3), device=device),
            ]
        )
        output = F.cross_entropy(logits.movedim(-1, 1), targets, reduction="none")
        assert isinstance(output, NestedTensor)
        assert [tuple(element.shape) for element in output] == [(2, 2), (3, 3)]

        reference = NestedTensor(
            [
                -torch.log_softmax(element, -1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
                for element, target in zip(logits, targets)
            ],
            ragged_dims=targets.ragged_dims,
        )
        torch.testing.assert_close(output.concat, reference.concat)

    def test_preserves_autograd(self, device, float_dtype):
        values = torch.randn(5, 5, device=device, dtype=float_dtype, requires_grad=True)
        logits = NestedTensor([values[:2], values[2:]])
        targets = NestedTensor([torch.randint(0, 5, (2,), device=device), torch.randint(0, 5, (3,), device=device)])
        loss = F.cross_entropy(logits.movedim(-1, 1), targets)
        assert loss.requires_grad
        loss.backward()
        assert values.grad is not None

    def test_ignore_index_matches_dense(self, device, float_dtype):
        logits, targets = self._logits_and_targets(device, float_dtype)

        # An ignored sentinel is allowed to sit outside the class range.
        sentinel = NestedTensor(
            [
                torch.tensor([999, 1], device=device),
                torch.tensor([0, 1, 999], device=device),
            ]
        )
        torch.testing.assert_close(
            F.cross_entropy(logits, sentinel, ignore_index=999),
            F.cross_entropy(logits.concat, sentinel.concat, ignore_index=999),
        )

    def test_rejects_out_of_range_label_that_is_not_ignored(self, device, float_dtype, request):
        if device.type == "cuda" and os.environ.get("DANLING_TEST_CUDA_ASSERT_CHILD") != "1":
            # A device assertion poisons the CUDA context even when caught. Execute
            # this same parametrized case in a fresh process, synchronizing below.
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-o",
                    "addopts=",
                    "-p",
                    "no:cacheprovider",
                    "-q",
                    request.node.nodeid,
                ],
                cwd=request.config.rootpath,
                env={**os.environ, "DANLING_TEST_CUDA_ASSERT_CHILD": "1"},
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            torch.testing.assert_close(torch.ones(1, device=device).sum(), torch.ones((), device=device))
            return
        # Clamping an invalid label into a valid class would score it silently; the dense
        # operator raises, so this must raise too.
        logits, _ = self._logits_and_targets(device, float_dtype)
        bad = NestedTensor(
            [
                torch.tensor([-5, 1], device=device),
                torch.tensor([0, 1, 2], device=device),
            ]
        )
        with pytest.raises((RuntimeError, IndexError), match="out of bounds|out of range|device-side assert"):
            F.cross_entropy(logits, bad, ignore_index=-100)
            if device.type == "cuda":
                torch.cuda.synchronize()

    def test_label_smoothing_matches_dense(self, device, float_dtype):
        logits, targets = self._logits_and_targets(device, float_dtype)
        torch.testing.assert_close(
            F.cross_entropy(logits, targets, label_smoothing=0.1),
            F.cross_entropy(logits.concat, targets.concat, label_smoothing=0.1),
        )

    @pytest.mark.parametrize("layout", ["sequence", "samples", "pair", "classification"])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("smoothing", [0.0, 0.1])
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("probabilities", [False, True])
    def test_native_value_and_vjp(self, device, float_dtype, layout, reduction, smoothing, weighted, probabilities):
        from danling.tensors.ops import nested_execution_guard

        classes = 5
        position_shapes = {
            "sequence": [(0,), (2,), (5,)],
            "samples": [(2, 0), (2, 2), (2, 5)],
            "pair": [(0, 3), (2, 4), (5, 2)],
            "classification": [(), (), ()],
        }[layout]
        ragged = {"sequence": (0,), "samples": (1,), "pair": (0, 1), "classification": ()}[layout]
        elements = [
            torch.randn(*shape, classes, device=device, dtype=float_dtype, requires_grad=True)
            for shape in position_shapes
        ]
        logits = NestedTensor(elements, ragged_dims=ragged).movedim(-1, 1)
        if probabilities:
            targets = [torch.randn_like(element).softmax(-1).detach().requires_grad_() for element in elements]
            native_target = NestedTensor(targets, ragged_dims=ragged).movedim(-1, 1)
        else:
            targets = [torch.randint(classes, shape, device=device) for shape in position_shapes]
            # Give ignored positions a sentinel outside the vocabulary, including an entire element.
            targets[1].fill_(999)
            native_target = NestedTensor(targets, ragged_dims=ragged)
            if layout == "pair":
                # Identical logical targets with a different packed ragged-axis order.
                native_target = NestedTensor([target.T for target in targets], ragged_dims=(0, 1)).transpose(-1, -2)
        weight = torch.linspace(0.2, 1.1, classes, dtype=float_dtype, device=device) if weighted else None
        kwargs = {
            "weight": weight,
            "label_smoothing": smoothing,
            "reduction": reduction,
            "ignore_index": -100 if probabilities else 999,
        }
        with nested_execution_guard(
            forbid_iteration=True,
            forbid_storage_map=True,
            forbid_eager_fallback=True,
            forbid_padded_materialization=True,
            forbid_dense_repack=True,
        ):
            actual = F.cross_entropy(logits, native_target, **kwargs)
            objective = actual.sum() if reduction == "none" else actual
            leaves = elements + targets if probabilities else elements
            actual_gradient = torch.autograd.grad(objective, leaves)

        # Evaluate the complete equation independently in FP64 on the actual quantized data.
        # Grouping half-precision element losses and weight sums before a global mean introduces
        # an extra rounded denominator and can scale every reference gradient incorrectly.
        reference_elements = [element.detach().double().requires_grad_() for element in elements]
        reference_targets = (
            [target.detach().double().requires_grad_() for target in targets] if probabilities else targets
        )
        reference_weight = weight.detach().double() if weighted else None
        values, position_weights, gradient_scales = [], [], []
        target_gradient_scales = []
        for element, target in zip(reference_elements, reference_targets):
            log_probabilities = element.log_softmax(-1)
            if probabilities:
                distribution = (1.0 - smoothing) * target + smoothing / classes
                position_weights.append(torch.ones_like(target[..., 0]).reshape(-1))
            else:
                valid = target != 999
                safe_target = torch.where(valid, target, 0)
                distribution = (1.0 - smoothing) * F.one_hot(safe_target, classes).double() + smoothing / classes
                distribution = torch.where(valid.unsqueeze(-1), distribution, 0.0)
                selected_weight = (
                    reference_weight[safe_target] if weighted else torch.ones_like(valid, dtype=torch.float64)
                )
                position_weights.append(torch.where(valid, selected_weight, 0.0).reshape(-1))
            coefficient = distribution * reference_weight if weighted else distribution
            values.append(-(coefficient * log_probabilities).sum(-1))
            # Before cancellation, both terms of dL/dz = softmax(z) * sum(coefficient)
            # - coefficient are bounded by this row's total coefficient mass.
            gradient_scales.append(coefficient.sum(-1, keepdim=True).detach().expand_as(element))
            if probabilities:
                target_scale = (1.0 - smoothing) * log_probabilities.detach().abs()
                target_gradient_scales.append(target_scale * reference_weight if weighted else target_scale)
        reference = torch.cat([value.reshape(-1) for value in values]).sum()
        denominator = torch.cat(position_weights).sum()
        if reduction == "mean":
            reference = reference / denominator
            gradient_scales = [scale / denominator for scale in gradient_scales]
            target_gradient_scales = [scale / denominator for scale in target_gradient_scales]
        expected = NestedTensor(values, ragged_dims=ragged)
        if reduction == "none":
            assert isinstance(actual, NestedTensor)
            torch.testing.assert_close(actual.element_sizes(), expected.element_sizes())
            assert actual.ragged_dims == expected.ragged_dims
            actual_values, expected_values = actual.concat, expected.concat
        else:
            actual_values, expected_values = actual, reference
        reference_leaves = reference_elements + reference_targets if probabilities else reference_elements
        expected_gradient = torch.autograd.grad(reference, reference_leaves)
        if float_dtype in (torch.float16, torch.bfloat16):
            # Native CE stores log probabilities, loss/weight totals and backward products in
            # the requested dtype. Eight elementary roundings bound the composed coefficient,
            # normalization and subtractive backward arithmetic: gamma_8 = 8u / (1 - 8u).
            # Use the non-cancelled derivative scale rather than demanding a relative match
            # to a tiny gradient; ignored rows must still be exactly zero.
            info = torch.finfo(float_dtype)
            unit_roundoff = info.eps / 2
            gamma = 8 * unit_roundoff / (1 - 8 * unit_roundoff)
            subnormal_step = info.tiny * info.eps
            value_error = (actual_values.double() - expected_values).abs()
            assert torch.all(value_error <= gamma * expected_values.abs() + subnormal_step)
            for actual_grad, expected_grad, scale in zip(
                actual_gradient, expected_gradient, gradient_scales + target_gradient_scales
            ):
                assert torch.all((actual_grad.double() - expected_grad).abs() <= gamma * scale + subnormal_step)
                torch.testing.assert_close(
                    actual_grad[scale == 0], torch.zeros_like(actual_grad[scale == 0]), atol=0.0, rtol=0.0
                )
        else:
            # Retain the previous float32/float64 tolerances after rounding the FP64 equation
            # to the result dtype. This checks the intended equation, not another grouped CE.
            torch.testing.assert_close(actual_values, expected_values.to(float_dtype))
            for actual_grad, expected_grad in zip(actual_gradient, expected_gradient):
                torch.testing.assert_close(
                    actual_grad,
                    expected_grad.to(float_dtype),
                    atol=1e-5 if float_dtype != torch.float64 else 1e-12,
                    rtol=1e-5,
                )

    def test_all_ignored_returns_nan_and_zero_gradient(self, device, float_dtype):
        elements = [torch.randn(n, 5, device=device, dtype=float_dtype, requires_grad=True) for n in (2, 4)]
        logits = NestedTensor(elements).movedim(-1, 1)
        targets = NestedTensor([torch.full((n,), 999, device=device) for n in (2, 4)])
        loss = F.cross_entropy(logits, targets, ignore_index=999, label_smoothing=0.1)
        assert loss.isnan()
        for gradient in torch.autograd.grad(loss, elements):
            torch.testing.assert_close(gradient, torch.zeros_like(gradient))

    def test_class_last_is_not_silently_detected(self, device, float_dtype):
        logits = NestedTensor(
            [torch.randn(2, 5, device=device, dtype=float_dtype), torch.randn(3, 5, device=device, dtype=float_dtype)]
        )
        targets = NestedTensor(
            [torch.zeros(2, dtype=torch.long, device=device), torch.zeros(3, dtype=torch.long, device=device)]
        )
        with pytest.raises(ValueError, match="static class axis"):
            F.cross_entropy(logits, targets)

    @pytest.mark.parametrize("probabilities", [False, True])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_dense_input_reads_only_native_target_positions(self, device, float_dtype, probabilities, reduction):
        from danling.tensors.ops import nested_execution_guard

        lengths, classes = (0, 2, 5), 5
        values = torch.randn(3, classes, 5, device=device, dtype=float_dtype, requires_grad=True)
        if probabilities:
            target_elements = [
                torch.randn(classes, n, device=device, dtype=float_dtype).softmax(0).requires_grad_() for n in lengths
            ]
            targets = NestedTensor(target_elements, ragged_dims=(1,))
        else:
            target_elements = [torch.randint(classes, (n,), device=device) for n in lengths]
            targets = NestedTensor(target_elements, ragged_dims=(0,))
        with nested_execution_guard(
            forbid_iteration=True,
            forbid_storage_map=True,
            forbid_eager_fallback=True,
            forbid_padded_materialization=True,
            forbid_dense_repack=True,
        ):
            result = F.cross_entropy(values, targets, label_smoothing=0.1, reduction=reduction)
            actual = result.sum() if reduction == "none" else result
            gradient = torch.autograd.grad(actual, values)[0]
        reference_parts = [
            F.cross_entropy(
                values[i : i + 1, :, :n], target.unsqueeze(0), label_smoothing=0.1, reduction="none"
            ).squeeze(0)
            for i, (n, target) in enumerate(zip(lengths, target_elements))
        ]
        reference = sum(part.sum() for part in reference_parts)
        if reduction == "mean":
            reference = reference / sum(lengths)
        if reduction == "none":
            torch.testing.assert_close(result.concat, torch.cat(reference_parts))
        else:
            torch.testing.assert_close(result, reference)
        torch.testing.assert_close(gradient, torch.autograd.grad(reference, values)[0])

    @pytest.mark.parametrize("probabilities", [False, True])
    def test_dense_targets_follow_logical_shape(self, device, float_dtype, probabilities):
        from danling.tensors.ops import nested_execution_guard

        elements = [torch.randn(n, 5, device=device, dtype=float_dtype, requires_grad=True) for n in (2, 5)]
        logits = NestedTensor(elements, ragged_dims=(0,)).movedim(-1, 1)
        targets = (
            torch.randn(2, 5, 5, device=device, dtype=float_dtype).softmax(1)
            if probabilities
            else torch.randint(5, (2, 5), device=device)
        )
        with nested_execution_guard(
            forbid_iteration=True,
            forbid_storage_map=True,
            forbid_eager_fallback=True,
            forbid_padded_materialization=True,
            forbid_dense_repack=True,
        ):
            result = F.cross_entropy(logits, targets, label_smoothing=0.1, reduction="sum")
            gradient = torch.autograd.grad(result, elements)
        reference = sum(
            F.cross_entropy(
                element.T.unsqueeze(0),
                targets[i : i + 1, :, :n] if probabilities else targets[i : i + 1, :n],
                label_smoothing=0.1,
                reduction="sum",
            )
            for i, (n, element) in enumerate(zip((2, 5), elements))
        )
        torch.testing.assert_close(result, reference)
        for actual, expected in zip(gradient, torch.autograd.grad(reference, elements)):
            torch.testing.assert_close(actual, expected)
        with pytest.raises(ValueError, match="shapes must match"):
            F.cross_entropy(logits, targets[0], label_smoothing=0.1)

    @pytest.mark.parametrize("classes", [3, 5])
    @pytest.mark.parametrize("native_input", [False, True])
    def test_mixed_probability_targets_preserve_static_axis_order(self, device, classes, native_input):
        from danling.tensors.ops import nested_execution_guard

        # Classes and the static position axis coincide in one case: a wrong permutation
        # then has a valid shape and must still be detected by values and both VJPs.
        elements = [
            torch.randn(*shape, classes, dtype=torch.float64, device=device).requires_grad_()
            for shape in [(2, 3, 3), (4, 2, 3)]
        ]
        if not native_input:
            elements = [element.detach().softmax(-1).requires_grad_() for element in elements]
        native = NestedTensor(elements, ragged_dims=(0, 1)).movedim(-1, 1).transpose(2, 3)
        logical_elements = [element.movedim(-1, 0).transpose(1, 2) for element in elements]
        dense = torch.randn(tuple(native.shape), dtype=torch.float64, device=device)
        if native_input:
            dense = dense.softmax(1)
        dense.requires_grad_()
        logits, targets = (native, dense) if native_input else (dense, native)
        weights = torch.linspace(0.2, 1.1, classes, dtype=torch.float64, device=device)
        kwargs = {"weight": weights, "label_smoothing": 0.17, "reduction": "none"}
        leaves = elements + [dense]
        with nested_execution_guard(
            forbid_iteration=True,
            forbid_storage_map=True,
            forbid_eager_fallback=True,
            forbid_padded_materialization=True,
            forbid_dense_repack=True,
        ):
            result = F.cross_entropy(logits, targets, **kwargs)
            actual_gradients = torch.autograd.grad(result.sum(), leaves)

        # Form rows directly from each element's logical class axis and slice the supplied
        # dense storage. This reference is independent of the NestedTensor packed permutation.
        input_rows, target_rows, logical_shapes = [], [], []
        for index, element in enumerate(logical_elements):
            logical_shape = tuple(element.shape[1:])
            slices = (index, slice(None), *(slice(0, extent) for extent in logical_shape))
            dense_element = dense[slices]
            input_element, target_element = (element, dense_element) if native_input else (dense_element, element)
            input_rows.append(input_element.movedim(0, -1).reshape(-1, classes))
            target_rows.append(target_element.movedim(0, -1).reshape(-1, classes))
            logical_shapes.append(logical_shape)
        expected = F.cross_entropy(torch.cat(input_rows), torch.cat(target_rows), **kwargs)
        assert isinstance(result, NestedTensor)
        assert result.ragged_dims == (1, 0)
        assert [tuple(element.shape) for element in result] == logical_shapes
        offset = 0
        for actual_element, shape in zip(result, logical_shapes):
            positions = math.prod(shape)
            torch.testing.assert_close(actual_element, expected[offset : offset + positions].reshape(shape))
            offset += positions
        for actual, expected_gradient in zip(actual_gradients, torch.autograd.grad(expected.sum(), leaves)):
            torch.testing.assert_close(actual, expected_gradient)

    @pytest.mark.parametrize("probabilities", [False, True])
    def test_one_dimensional_input_uses_class_axis_zero(self, device, float_dtype, probabilities):
        values = torch.randn(5, device=device, dtype=float_dtype, requires_grad=True)
        logits = NestedTensor(list(values.unbind()), ragged_dims=())
        target = torch.randn_like(values).softmax(0) if probabilities else torch.tensor(2, device=device)
        native_target = NestedTensor(list(target.unbind()), ragged_dims=()) if probabilities else target
        output = F.cross_entropy(logits, native_target, label_smoothing=0.1)
        reference = F.cross_entropy(values, target, label_smoothing=0.1)
        torch.testing.assert_close(output, reference)
        torch.testing.assert_close(torch.autograd.grad(output, values)[0], torch.autograd.grad(reference, values)[0])

    @pytest.mark.parametrize("size_average,reduce", [(False, False), (False, True), (True, True)])
    def test_legacy_reduction_aliases(self, device, float_dtype, size_average, reduce):
        logits, targets = self._logits_and_targets(device, float_dtype)
        with pytest.warns(UserWarning, match="size_average and reduce"):
            output = F.cross_entropy(logits, targets, size_average=size_average, reduce=reduce, label_smoothing=0.1)
        with pytest.warns(UserWarning, match="size_average and reduce"):
            expected = F.cross_entropy(
                logits.concat, targets.concat, size_average=size_average, reduce=reduce, label_smoothing=0.1
            )
        torch.testing.assert_close(output.concat if isinstance(output, NestedTensor) else output, expected)

    def test_target_type_and_probability_ignore_errors_are_preserved(self, device, float_dtype):
        logits, targets = self._logits_and_targets(device, float_dtype)
        with pytest.raises(RuntimeError, match="Long"):
            F.cross_entropy(logits, targets.short())
        probabilities = logits.softmax(1)
        with pytest.raises(RuntimeError, match="ignore_index"):
            F.cross_entropy(logits, probabilities, ignore_index=1)

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_zero_classes_and_ignored_positions_follow_dense(self, device, float_dtype, reduction):
        from danling.tensors.ops import nested_execution_guard

        values = torch.empty(3, 0, device=device, dtype=float_dtype, requires_grad=True)
        logits = NestedTensor(list(values.unbind()), ragged_dims=())
        target_values = torch.full((3,), -100, device=device)
        targets = NestedTensor(list(target_values.unbind()), ragged_dims=())
        with nested_execution_guard(
            forbid_iteration=True,
            forbid_storage_map=True,
            forbid_eager_fallback=True,
            forbid_padded_materialization=True,
            forbid_dense_repack=True,
        ):
            output = F.cross_entropy(logits, targets, reduction=reduction)
            actual = output.sum() if reduction == "none" else output
            gradient = torch.autograd.grad(actual, values)[0]
        reference = F.cross_entropy(values, target_values, reduction=reduction)
        torch.testing.assert_close(
            output.concat if isinstance(output, NestedTensor) else output, reference, equal_nan=True
        )
        torch.testing.assert_close(gradient, torch.autograd.grad(reference.sum(), values)[0])

    @pytest.mark.parametrize("native", [False, True])
    @pytest.mark.parametrize("unbatched", [False, True])
    @pytest.mark.parametrize("probabilities", [False, True])
    @pytest.mark.parametrize("smoothing", [0.0, 0.1])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_native_class_weights_value_and_input_vjp(
        self, device, float_dtype, native, unbatched, probabilities, smoothing, reduction
    ):
        from danling.tensors.ops import nested_execution_guard

        values = torch.randn(2, 3, device=device, dtype=float_dtype, requires_grad=True)
        target_values = values.detach().softmax(-1) if probabilities else torch.tensor([0, 2], device=device)
        weights = torch.tensor([0.4, 0.7, 1.2], device=device, dtype=float_dtype)
        native_weights = NestedTensor(list(weights.unbind()), ragged_dims=())
        dense_input = values[0] if unbatched else values
        dense_target = target_values[0] if unbatched else target_values
        input = NestedTensor(list(dense_input.unbind()), ragged_dims=()) if native else dense_input
        target = (
            NestedTensor(list(dense_target.unbind()), ragged_dims=()) if native and dense_target.dim() else dense_target
        )
        with nested_execution_guard(
            forbid_iteration=True,
            forbid_storage_map=True,
            forbid_eager_fallback=True,
            forbid_padded_materialization=True,
            forbid_dense_repack=True,
        ):
            output = F.cross_entropy(
                input, target, weight=native_weights, label_smoothing=smoothing, reduction=reduction
            )
            gradient = torch.autograd.grad(output.sum(), values)[0]
        reference = F.cross_entropy(
            dense_input, dense_target, weight=weights, label_smoothing=smoothing, reduction=reduction
        )
        torch.testing.assert_close(output.concat if isinstance(output, NestedTensor) else output, reference)
        torch.testing.assert_close(gradient, torch.autograd.grad(reference.sum(), values)[0])

    @pytest.mark.parametrize("native", [False, True])
    @pytest.mark.parametrize("probabilities", [False, True])
    def test_native_class_weight_grad_follows_torch(self, device, float_dtype, native, probabilities):
        values = torch.randn(2, 3, device=device, dtype=float_dtype, requires_grad=True)
        target_values = values.detach().softmax(-1) if probabilities else torch.tensor([0, 2], device=device)
        weights = torch.tensor([0.4, 0.7, 1.2], device=device, dtype=float_dtype, requires_grad=True)
        native_weights = NestedTensor(list(weights.unbind()), ragged_dims=())
        input = NestedTensor(list(values.unbind()), ragged_dims=()) if native else values
        target = NestedTensor(list(target_values.unbind()), ragged_dims=()) if native else target_values
        if not probabilities:
            with pytest.raises(RuntimeError, match="not differentiable.*weight|weight.*requires_grad"):
                F.cross_entropy(values, target_values, weight=weights, label_smoothing=0.1)
            with pytest.raises(RuntimeError, match="not differentiable.*weight|weight.*requires_grad"):
                F.cross_entropy(input, target, weight=native_weights, label_smoothing=0.1)
            return
        output = F.cross_entropy(input, target, weight=native_weights, label_smoothing=0.1)
        reference = F.cross_entropy(values, target_values, weight=weights, label_smoothing=0.1)
        torch.testing.assert_close(output, reference)
        actual_gradients = torch.autograd.grad(output, (values, weights))
        expected_gradients = torch.autograd.grad(reference, (values, weights))
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("probabilities", [False, True])
    def test_native_class_weight_dtype_is_not_cast(self, device, float_dtype, probabilities):
        values = torch.randn(2, 3, device=device, dtype=float_dtype)
        target = values.softmax(-1) if probabilities else torch.tensor([0, 2], device=device)
        weight_dtype = torch.float32 if float_dtype == torch.float64 else torch.float64
        weights = torch.tensor([0.4, 0.7, 1.2], device=device, dtype=weight_dtype)
        native_weights = NestedTensor(list(weights.unbind()), ragged_dims=())
        if probabilities:
            output = F.cross_entropy(values, target, weight=native_weights)
            reference = F.cross_entropy(values, target, weight=weights)
            assert output.dtype == reference.dtype
            torch.testing.assert_close(output, reference)
        else:
            with pytest.raises(RuntimeError, match="scalar type|same dtype|expected.*type"):
                F.cross_entropy(values, target, weight=weights)
            with pytest.raises(RuntimeError, match="scalar type|same dtype|expected.*type"):
                F.cross_entropy(values, target, weight=native_weights)

    def test_native_class_weights_require_logical_vector(self, device, float_dtype):
        values = torch.randn(2, 3, device=device, dtype=float_dtype)
        target = torch.tensor([0, 2], device=device)
        weights = NestedTensor([torch.ones(3, device=device, dtype=float_dtype)] * 2, ragged_dims=())
        with pytest.raises(ValueError, match="class weights must be one-dimensional"):
            F.cross_entropy(values, target, weight=weights)
