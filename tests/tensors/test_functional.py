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
from torch import nn
from torch.nn import functional as F

from danling.tensors import NestedTensor
from danling.tensors.functional import concat_tensors
from tests.tensors.utils import assert_close, assert_nested_function_matches, nested_rand

NT = NestedTensor


class TestConcatHelpers:

    def test_concat_tensors_with_plain_tensors(self, device, float_dtype):
        first = torch.arange(4, device=device, dtype=float_dtype).reshape(2, 2)
        second = torch.arange(4, 8, device=device, dtype=float_dtype).reshape(2, 2)
        out_first, out_second = concat_tensors(first, second)
        assert_close(out_first, first)
        assert_close(out_second, second)


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
        reference = NT([F.linear(t, weight, bias) for t in input], **input._state)
        assert_close(output, reference)


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
        reference = F.conv1d(input.tensor, weight, bias, stride, padding, dilation, groups)
        assert_close(output, reference, atol=1e-5)

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
        reference = F.conv2d(input.tensor, weight, bias, stride, padding, dilation, groups)
        assert_close(output, reference, atol=1e-5)

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
        reference = F.conv3d(input.tensor, weight, bias, stride, padding, dilation, groups)
        assert_close(output, reference, atol=1e-5)


class TestNormalize:

    def test_normalize_matches_tensor(self, device, float_dtype):
        input = nested_rand([(3, 4), (3, 4)], device, float_dtype)
        output = F.normalize(input, dim=2)
        reference = F.normalize(input.tensor, dim=2)
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_normalize_batch_dim_raises(self, device, float_dtype):
        input = nested_rand([(3, 4), (2, 4)], device, float_dtype)
        with pytest.raises(NotImplementedError):
            F.normalize(input, dim=0)


class TestMultiHeadAttention:

    def test_multi_head_attention_batch_first_false(self, device, float_dtype):
        embed_dim = 4
        num_heads = 2
        lengths = [2, 3]
        data = [torch.randn(length, embed_dim, device=device, dtype=float_dtype) for length in lengths]
        query = NT(data, batch_first=False)

        module = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False, dropout=0.0)
        module = module.to(device=device, dtype=float_dtype)

        key_padding_mask = (~query.mask).transpose(0, 1)
        reference, _ = module(query.tensor, query.tensor, query.tensor, key_padding_mask=key_padding_mask)

        output = F.multi_head_attention_forward(
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
        assert_close(output, reference, atol=1e-6, rtol=1e-6)


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


class TestSoftmax:

    def test_softmax(self, device, float_dtype):
        nt = nested_rand([(2, 4), (3, 4)], device, float_dtype)
        assert_nested_function_matches(F.softmax, nt, dim=-1)

    def test_softmax_accepts_positive_dim(self, device, float_dtype):
        nt = nested_rand([(2, 4), (3, 4)], device, float_dtype)
        output = F.softmax(nt, dim=2)
        reference = F.softmax(nt.tensor, dim=2)
        assert_close(output, reference)


class TestLogSoftmax:

    def test_log_softmax(self, device, float_dtype):
        nt = nested_rand([(2, 4), (3, 4)], device, float_dtype)
        assert_nested_function_matches(F.log_softmax, nt, dim=-1)


class TestSoftmin:

    def test_softmin(self, device, float_dtype):
        nt = nested_rand([(2, 4), (3, 4)], device, float_dtype)
        assert_nested_function_matches(F.softmin, nt, dim=-1)


class TestGumbelSoftmax:

    def test_gumbel_softmax(self, device, float_dtype):
        nt = nested_rand([(2, 4), (3, 4)], device, float_dtype)
        torch.manual_seed(1016)
        reference = NT([F.gumbel_softmax(t, dim=-1) for t in nt], **nt._state)
        torch.manual_seed(1016)
        output = F.gumbel_softmax(nt, dim=-1)
        assert_close(output, reference, atol=1e-6, rtol=1e-6)


class TestSoftsign:

    def test_softsign(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        assert_nested_function_matches(F.softsign, nt)


class TestTanhshrink:

    def test_tanhshrink(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        assert_nested_function_matches(F.tanhshrink, nt)


class TestSigmoid:

    def test_sigmoid(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        assert_nested_function_matches(F.sigmoid, nt)


class TestTanh:

    def test_tanh(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        assert_nested_function_matches(F.tanh, nt)


class TestRelu6:

    def test_relu6(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        assert_nested_function_matches(F.relu6, nt)


class TestElu:

    def test_elu(self, device, float_dtype):
        nt = nested_rand([(2, 4), (1, 4)], device, float_dtype)
        assert_nested_function_matches(F.elu, nt)


class TestCelu:

    def test_celu(self, device, float_dtype):
        nt = nested_rand([(2, 4), (1, 4)], device, float_dtype)
        assert_nested_function_matches(F.celu, nt)


class TestSelu:

    def test_selu(self, device, float_dtype):
        nt = nested_rand([(2, 4), (1, 4)], device, float_dtype)
        assert_nested_function_matches(F.selu, nt)


class TestRelu:

    def test_relu(self, device, float_dtype):
        nt = nested_rand([(2, 4), (1, 4)], device, float_dtype)
        assert_nested_function_matches(F.relu, nt, inplace=False)


class TestLeakyRelu:

    def test_leaky_relu(self, device, float_dtype):
        nt = nested_rand([(2, 4), (1, 4)], device, float_dtype)
        assert_nested_function_matches(F.leaky_relu, nt, negative_slope=0.1)


class TestRrelu:

    def test_rrelu(self, device, float_dtype):
        nt = nested_rand([(2, 4), (1, 4)], device, float_dtype)
        assert_nested_function_matches(F.rrelu, nt, training=False)


class TestGlu:

    def test_glu(self, device, float_dtype):
        nt = nested_rand([(2, 4), (1, 4)], device, float_dtype)
        assert_nested_function_matches(F.glu, nt, dim=-1)


class TestGelu:

    def test_gelu(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        assert_nested_function_matches(F.gelu, nt)


class TestSoftplus:

    def test_softplus(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        assert_nested_function_matches(F.softplus, nt)


class TestHardsigmoid:

    def test_hardsigmoid(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        assert_nested_function_matches(F.hardsigmoid, nt)


class TestHardswish:

    def test_hardswish(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        assert_nested_function_matches(F.hardswish, nt)


class TestHardtanh:

    def test_hardtanh(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        assert_nested_function_matches(F.hardtanh, nt)


class TestSoftshrink:

    def test_softshrink(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        assert_nested_function_matches(F.softshrink, nt, lambd=0.5)


class TestHardshrink:

    def test_hardshrink(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        assert_nested_function_matches(F.hardshrink, nt, lambd=0.5)


class TestThreshold:

    def test_threshold(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        assert_nested_function_matches(F.threshold, nt, threshold=0.0, value=-0.1)


class TestSilu:

    def test_silu(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        assert_nested_function_matches(F.silu, nt)


class TestMish:

    def test_mish(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        assert_nested_function_matches(F.mish, nt)


class TestLogsigmoid:

    def test_logsigmoid(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        assert_nested_function_matches(F.logsigmoid, nt)


class TestAffineGrid:

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
            **nt_imgs._state,
        )
        assert_close(output, reference, atol=1e-6, rtol=1e-6)


class TestGridSample:

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


class TestEmbedding:

    def test_embedding(self, device, float_dtype):
        weight = torch.randn(10, 4, device=device, dtype=float_dtype)
        nt_idx = NT(
            [
                torch.tensor([1, 3, 5], dtype=torch.long, device=device),
                torch.tensor([0, 2], dtype=torch.long, device=device),
            ]
        )
        output = F.embedding(nt_idx, weight)
        reference = NT([F.embedding(t, weight) for t in nt_idx], **nt_idx._state)
        assert_close(output, reference, atol=1e-6, rtol=1e-6)


class TestEmbeddingBag:

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
        reference = NT([F.embedding_bag(t, weight, offsets=offsets, mode="mean") for t in nt_idx], **nt_idx._state)
        assert_close(output, reference, atol=1e-6, rtol=1e-6)


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


class TestFractionalMaxPoolWithIndices:

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
        reference_output = NT(reference_output, **nt._state)
        reference_idx = NT(reference_idx, **nt._state)
        assert_close(output, reference_output)
        assert_close(idx, reference_idx)


class TestDropout:

    def test_dropout_eval_deterministic(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        output = F.dropout(nt, p=0.7, training=False)
        assert_close(output, nt)

    def test_dropout(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(1, 3, 4, device=device, dtype=float_dtype),
                torch.randn(1, 3, 4, device=device, dtype=float_dtype),
            ]
        )
        output = F.dropout(nt, p=0.2, training=False)
        assert_close(output, nt)

    def test_dropout1d(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(1, 3, 4, device=device, dtype=float_dtype),
                torch.randn(1, 3, 4, device=device, dtype=float_dtype),
            ]
        )
        output = F.dropout1d(nt, p=0.2, training=False)
        assert_close(output, nt)

    def test_dropout2d(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(1, 2, 2, 2, device=device, dtype=float_dtype),
                torch.randn(1, 2, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.dropout2d(nt, p=0.3, training=False)
        assert_close(output, nt)

    def test_dropout3d(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(1, 2, 2, 2, 2, device=device, dtype=float_dtype),
                torch.randn(1, 2, 2, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.dropout3d(nt, p=0.4, training=False)
        assert_close(output, nt)

    def test_alpha_dropout(self, device, float_dtype):
        nt = nested_rand([(2, 3), (2, 3)], device, float_dtype)
        output = F.alpha_dropout(nt, p=0.1, training=False)
        assert_close(output, nt)

    def test_feature_alpha_dropout(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(1, 3, 2, 2, device=device, dtype=float_dtype),
                torch.randn(1, 3, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.feature_alpha_dropout(nt, p=0.25, training=False)
        assert_close(output, nt)


class TestPad:

    def test_pad(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(4.0, device=device, dtype=float_dtype).view(1, 1, 2, 2),
                torch.ones(1, 1, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.pad(nt, (1, 1, 1, 1), value=0.5)
        reference = NT([F.pad(t, (1, 1, 1, 1), value=0.5) for t in nt], **nt._state)
        assert_close(output, reference)


class TestCrossEntropyLoss:

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


class TestNllLoss:

    def test_nll_loss(self, device, float_dtype):
        logits = NT(
            [
                torch.tensor([[2.0, 0.5], [0.1, 1.0]], device=device, dtype=float_dtype),
                torch.tensor([[1.0, 0.0]], device=device, dtype=float_dtype),
            ]
        )
        log_probs = NT([torch.log_softmax(t, dim=-1) for t in logits], **logits._state)
        targets = NT(
            [torch.tensor([0, 1], device=device, dtype=torch.long), torch.tensor([1], device=device, dtype=torch.long)]
        )
        output = F.nll_loss(log_probs, targets, reduction="sum")
        reference_input = torch.cat(tuple(logits), dim=0)
        reference_target = torch.cat(tuple(targets), dim=0)
        reference = F.nll_loss(reference_input.log_softmax(dim=-1), reference_target, reduction="sum")
        assert_close(output, reference)


class TestMseLoss:

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


class TestL1Loss:

    def test_l1_loss(self, device, float_dtype):
        pred = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        target = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        output = F.l1_loss(pred, target, reduction="sum")
        reference = F.l1_loss(torch.cat(tuple(pred), dim=0), torch.cat(tuple(target), dim=0), reduction="sum")
        assert_close(output, reference)


class TestSmoothL1Loss:

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


class TestHuberLoss:

    def test_huber_loss(self, device, float_dtype):
        pred = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        target = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        output = F.huber_loss(pred, target, reduction="sum")
        reference = F.huber_loss(torch.cat(tuple(pred), dim=0), torch.cat(tuple(target), dim=0), reduction="sum")
        assert_close(output, reference)


class TestMarginRankingLoss:

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


class TestTripletMarginLoss:

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


class TestBinaryCrossEntropyLoss:

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


class TestBinaryCrossEntropyWithLogitsLoss:

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


class TestCosineEmbeddingLoss:

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


class TestHingeEmbeddingLoss:

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


class TestGaussianNllLoss:

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


class TestCtcLoss:

    def test_ctc_loss_nested_matches_tensor(self):
        torch.manual_seed(0)
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


class TestPoissonNllLoss:

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


class TestSoftMarginLoss:

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


class TestMultiMarginLoss:

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


class TestMultilabelMarginLoss:

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


class TestMultilabelSoftMarginLoss:

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


class TestTripletMarginWithDistanceLoss:

    def test_triplet_margin_with_distance_loss(self, device, float_dtype):
        anchor = nested_rand([(2, 4), (1, 4)], device, float_dtype)
        positive = nested_rand([(2, 4), (1, 4)], device, float_dtype)
        negative = nested_rand([(2, 4), (1, 4)], device, float_dtype)
        output = F.triplet_margin_with_distance_loss(anchor, positive, negative, reduction="sum")
        reference = F.triplet_margin_with_distance_loss(
            anchor.concat, positive.concat, negative.concat, reduction="sum"
        )
        assert_close(output, reference, atol=1e-6, rtol=1e-6)


class TestUpsampleAliases:

    def test_upsample_aliases(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(4.0, device=device, dtype=float_dtype).view(1, 1, 2, 2),
                torch.ones(1, 1, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.upsample(nt, scale_factor=2, mode="nearest")
        reference = NT([F.upsample(t, scale_factor=2, mode="nearest") for t in nt], **nt._state)
        assert_close(output, reference)

        output = F.upsample_nearest(nt, scale_factor=2)
        reference = NT([F.upsample_nearest(t, scale_factor=2) for t in nt], **nt._state)
        assert_close(output, reference)

        output = F.upsample_bilinear(nt, scale_factor=2)
        reference = NT([F.upsample_bilinear(t, scale_factor=2) for t in nt], **nt._state)
        assert_close(output, reference, atol=1e-6, rtol=1e-6)


class TestGroupNorm:

    def test_group_norm(self, device, float_dtype):
        nt = nested_rand([(3, 4), (2, 4)], device, float_dtype)
        output = F.group_norm(nt, num_groups=1)
        reference = NT([F.group_norm(t.unsqueeze(0), num_groups=1).squeeze(0) for t in nt], **nt._state)
        assert_close(output, reference, atol=1e-5, rtol=1e-5)


class TestLayerNorm:

    def test_layer_norm(self, device, float_dtype):
        nt = nested_rand([(3, 4), (2, 4)], device, float_dtype)
        output = F.layer_norm(nt, normalized_shape=(4,))
        reference = NT([F.layer_norm(t, (4,)) for t in nt], **nt._state)
        assert_close(output, reference, atol=1e-5, rtol=1e-5)


class TestBatchNorm:

    def test_batch_norm(self, device, float_dtype):
        nt = nested_rand([(2, 3), (1, 3)], device, float_dtype)
        running_mean = torch.zeros(3, device=device, dtype=float_dtype)
        running_var = torch.ones(3, device=device, dtype=float_dtype)
        output = F.batch_norm(nt, running_mean=running_mean, running_var=running_var, training=True)
        concat, shapes = nt.concatenate()
        reference = NestedTensor.from_concatenated(
            F.batch_norm(concat, running_mean=running_mean, running_var=running_var, training=True), shapes, **nt._state
        )
        assert_close(output, reference, atol=1e-5, rtol=1e-5)


class TestInstanceNorm:

    def test_instance_norm(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(3, 2, 2, device=device, dtype=float_dtype),
                torch.randn(3, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.instance_norm(nt, use_input_stats=True)
        reference = NT([F.instance_norm(t.unsqueeze(0), use_input_stats=True).squeeze(0) for t in nt], **nt._state)
        assert_close(output, reference, atol=1e-5, rtol=1e-5)


class TestRmsNorm:

    def test_rms_norm(self, device, float_dtype):
        nt = nested_rand([(1, 4), (1, 4)], device, float_dtype)
        output = F.rms_norm(nt, normalized_shape=(4,))
        reference = NT([F.rms_norm(t, (4,)) for t in nt], **nt._state)
        assert_close(output, reference, atol=1e-5, rtol=1e-5)


class TestLocalResponseNorm:

    def test_local_response_norm(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(3, 3, 3, device=device, dtype=float_dtype),
                torch.randn(3, 3, 3, device=device, dtype=float_dtype),
            ]
        )
        output = F.local_response_norm(nt, size=2)
        reference = NT([F.local_response_norm(t.unsqueeze(0), size=2).squeeze(0) for t in nt], **nt._state)
        assert_close(output, reference, atol=1e-5, rtol=1e-5)


class TestInterpolate:

    def test_interpolate_nearest(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(4.0, device=device, dtype=float_dtype).view(1, 1, 2, 2),
                torch.ones(1, 1, 2, 2, device=device, dtype=float_dtype),
            ]
        )
        output = F.interpolate(nt, scale_factor=2, mode="nearest")
        reference = NT([F.interpolate(t, scale_factor=2, mode="nearest") for t in nt], **nt._state)
        assert_close(output, reference)

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
            **nt._state,
        )
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
        reference = NT([F.pixel_shuffle(t, upscale_factor=2) for t in nt], **nt._state)
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


class TestKLDivergence:

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


class TestStackNestedTensor:

    def test_stack_nested_tensor_dim0(self):
        a = NT([torch.tensor([1, 2]), torch.tensor([3, 4])])
        b = NT([torch.tensor([5, 6]), torch.tensor([7, 8])])
        stacked = torch.stack([a, b], dim=0)
        assert isinstance(stacked, NestedTensor)
        assert len(stacked) == len(a) == len(b) == 2
        assert stacked.shape == torch.Size([2, 2, 2])
        reference = NT([torch.stack([a[i], b[i]], dim=0) for i in range(len(a))], **a._state)
        assert_close(stacked, reference)


class TestFlattenUnflatten:

    def test_flatten_and_unflatten(self):
        nt = NT(
            [
                torch.arange(8, dtype=torch.float32).reshape(2, 2, 2),
                torch.arange(8, 16, dtype=torch.float32).reshape(2, 2, 2),
            ]
        )
        output = torch.flatten(nt, start_dim=1)
        reference = torch.flatten(nt.tensor, start_dim=1)
        assert_close(output, reference)
        output = torch.unflatten(output, dim=1, sizes=(2, 2, 2))
        assert_close(output, nt)


class TestClamp:

    def test_clamp(self):
        nt = NestedTensor([torch.tensor([[-1.0, 0.5], [2.0, 5.0]]), torch.tensor([[10.0, -5.0], [1.0, 3.0]])])
        output = torch.clamp(nt, min=0.0, max=3.0)
        reference = torch.clamp(nt.tensor, min=0.0, max=3.0)
        assert_close(output, reference)


class TestSqueezeUnsqueeze:

    def test_squeeze_unsqueeze(self):
        input = NestedTensor([torch.randn(1, 3), torch.randn(1, 3)])
        squeezed = torch.squeeze(input, dim=1)
        assert squeezed.shape == torch.Size([2, 3])
        unsqueezed = torch.unsqueeze(squeezed, dim=2)
        assert unsqueezed.shape == torch.Size([2, 3, 1])


class TestMoveaxis:

    def test_moveaxis(self):
        input = NestedTensor([torch.randn(2, 3, 4), torch.randn(2, 3, 4)])
        moved = torch.moveaxis(input, 1, 2)
        reference = torch.moveaxis(input.tensor, 1, 2)
        assert_close(moved, reference)


class TestSwapaxes:

    def test_swapaxes(self):
        input = NestedTensor([torch.randn(2, 3, 4), torch.randn(2, 3, 4)])
        swapped = torch.swapaxes(input, 1, 2)
        reference = torch.swapaxes(input.tensor, 1, 2)
        assert_close(swapped, reference)


class TestCat:

    def test_cat_inner_dim(self):
        a = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        b = NestedTensor([torch.tensor([10, 20, 30]), torch.tensor([40, 50])])
        output = torch.cat((a, b), dim=1)
        assert output.tolist() == [[1, 2, 3, 10, 20, 30], [4, 5, 40, 50]]


class TestGather:

    def test_gather(self):
        data = NestedTensor([torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])])
        index = NestedTensor([torch.tensor([[0, 2]]), torch.tensor([[1, 0]])])
        gathered = torch.gather(data, dim=2, index=index)
        reference = torch.gather(data.tensor, dim=2, index=index.tensor)
        assert_close(gathered, reference)


class TestScatter:

    def test_scatter(self):
        data = NestedTensor([torch.zeros(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32)])
        index = NestedTensor([torch.tensor([0, 1, 2]), torch.tensor([0, 1, 0])])
        src = NestedTensor([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])])
        output = torch.scatter(data, dim=1, index=index, src=src)
        reference = torch.scatter(data.tensor, dim=1, index=index.tensor, src=src.tensor)
        assert_close(output, reference)


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
        reference = NT([reference], **unpooled._state)
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
        reference = NT([reference], **unpooled._state)
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
        reference = NT([reference], **unpooled._state)
        assert_close(unpooled, reference)


class TestUnfoldFold:

    def test_unfold_and_fold_round_trip(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(1.0, 10.0, device=device, dtype=float_dtype).view(1, 1, 3, 3),
                torch.ones(1, 1, 3, 3, device=device, dtype=float_dtype),
            ]
        )
        output = F.unfold(nt, kernel_size=2, stride=1)
        reference = NT([F.unfold(t, kernel_size=2, stride=1) for t in nt], **nt._state)
        assert_close(output, reference)

        unfolded = output
        output = F.fold(unfolded, output_size=(3, 3), kernel_size=2, stride=1)
        reference = NT([F.fold(t, output_size=(3, 3), kernel_size=2, stride=1) for t in unfolded], **unfolded._state)
        assert_close(output, reference, atol=1e-6, rtol=1e-6)


class TestMultiheadAttention:

    def test_multi_head_attention_forward(self):
        torch.manual_seed(1016)
        embed_dim = 4
        num_heads = 2
        mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0)
        input = NestedTensor([torch.randn(3, embed_dim), torch.randn(2, embed_dim)])
        attn_output = F.multi_head_attention_forward(
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
        assert_close(attn_output, reference, atol=1e-5)

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

    def test_multi_head_attention_cross_attention_mask_matches_key(self):
        torch.manual_seed(1016)
        embed_dim = 4
        num_heads = 2
        mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0)
        query = NestedTensor([torch.randn(2, embed_dim), torch.randn(1, embed_dim)])
        key = NestedTensor([torch.randn(3, embed_dim), torch.randn(2, embed_dim)])
        output = F.multi_head_attention_forward(
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
        assert_close(output, reference, atol=1e-5)

    def test_multi_head_attention_respects_custom_mask_value(self):
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
        output = F.multi_head_attention_forward(
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
        assert_close(output, reference, atol=1e-5)

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

        output = F.multi_head_attention_forward(
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

        assert_close(output, reference, atol=1e-6, rtol=1e-6)

    def test_multi_head_attention_requires_nested_query(self):
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


@pytest.mark.skipif(not hasattr(F, "scaled_dot_product_attention"), reason="scaled_dot_product_attention not available")
class TestScaledDotProductAttention:

    def test_sdpa_matches_reference_in_sequential_path(self, device, float_dtype):
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
            [
                F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
                for q, k, v in zip(query._storage, key._storage, value._storage)
            ],
            **query._state,
        )
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_sdpa_matches_reference_high_occupancy(self, device, float_dtype):
        query = NT(
            [
                torch.randn(4, 64, 32, device=device, dtype=float_dtype),
                torch.randn(4, 56, 32, device=device, dtype=float_dtype),
            ]
        )
        output = F.scaled_dot_product_attention(query, query, query, dropout_p=0.0)
        reference = NT(
            [F.scaled_dot_product_attention(q, q, q, dropout_p=0.0) for q in query._storage],
            **query._state,
        )
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_sdpa_tensor_key_value_matching_shape(self, device, float_dtype):
        query = NT(
            [
                torch.randn(2, 6, 16, device=device, dtype=float_dtype),
                torch.randn(2, 4, 16, device=device, dtype=float_dtype),
            ]
        )
        output = F.scaled_dot_product_attention(query, query.tensor, query.tensor, dropout_p=0.0)
        reference = NT(
            [F.scaled_dot_product_attention(q, q, q, dropout_p=0.0) for q in query._storage],
            **query._state,
        )
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_sdpa_batched_attn_mask_tensor(self, device, float_dtype):
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
            [
                F.scaled_dot_product_attention(q, q, q, attn_mask=mask[i], dropout_p=0.0)
                for i, q in enumerate(query._storage)
            ],
            **query._state,
        )
        assert_close(output, reference, atol=1e-5, rtol=1e-5)

    def test_sdpa_requires_nested_query(self):
        tensor_query = torch.randn(2, 2, 4, 8)
        key = NT([torch.randn(2, 4, 8)])
        with pytest.raises(TypeError):
            F.scaled_dot_product_attention(tensor_query, key, key, dropout_p=0.0)

    def test_sdpa_mismatched_batch_lengths_raises(self):
        query = NT([torch.randn(2, 4, 8), torch.randn(2, 3, 8)])
        key = NT([torch.randn(2, 4, 8)])
        with pytest.raises(ValueError, match="NestedTensor batch length mismatch"):
            F.scaled_dot_product_attention(query, key, key, dropout_p=0.0)


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
        reference = F.conv_transpose1d(input.tensor, weight, bias, stride, padding, output_padding, groups, dilation)
        assert_close(output, reference, atol=1e-5)

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
        reference = F.conv_transpose2d(input.tensor, weight, bias, stride, padding, output_padding, groups, dilation)
        assert_close(output, reference, atol=1e-5)

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
        reference = F.conv_transpose3d(input.tensor, weight, bias, stride, padding, output_padding, groups, dilation)
        assert_close(output, reference, atol=1e-5)
