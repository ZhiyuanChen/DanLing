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
from torch import nn

from tests.tensors.utils import assert_close, nested_rand


class TestLinearModule:

    @pytest.mark.parametrize("shape", [[(3, 5), (2, 5)], [(2, 3, 5), (3, 2, 5)]])
    def test_linear_module_matches_tensor(self, shape, device, float_dtype):
        input = nested_rand(shape, device, float_dtype)
        layer = nn.Linear(input.shape[-1], 3).to(device=device, dtype=float_dtype)
        reference_layer = nn.Linear(input.shape[-1], 3).to(device=device, dtype=float_dtype)
        reference_layer.load_state_dict(layer.state_dict())
        output = layer(input)
        reference = reference_layer(input.tensor)
        reference.masked_fill_(~output.mask.unsqueeze(-1), 0)
        assert_close(output, reference)
        output.sum().backward()
        reference.sum().backward()
        assert_close(layer.weight.grad, reference_layer.weight.grad)
        assert_close(layer.bias.grad, reference_layer.bias.grad)


class TestConv1dModule:
    @pytest.mark.parametrize("shape", [[(5, 8), (7, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("groups", [1, 2])
    def test_conv1d_module(self, shape, kernel_size, stride, padding, dilation, groups, device, float_dtype):
        input = nested_rand(shape, device, float_dtype)
        layer = nn.Conv1d(input.shape[-1], 4, kernel_size).to(device=device, dtype=float_dtype)
        reference_layer = nn.Conv1d(input.shape[-1], 4, kernel_size).to(device=device, dtype=float_dtype)
        reference_layer.load_state_dict(layer.state_dict())
        input = input.transpose(1, -1)
        output = layer(input)
        reference = reference_layer(input.tensor)
        reference.masked_fill_(~output.mask, 0)
        assert_close(output, reference, atol=1e-6)
        output.sum().backward()
        reference.sum().backward()
        assert_close(layer.weight.grad, reference_layer.weight.grad)
        assert_close(layer.bias.grad, reference_layer.bias.grad)


class TestConv2dModule:
    @pytest.mark.parametrize("shape", [[(5, 7, 8), (11, 13, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("groups", [1, 2])
    def test_conv2d_module(self, shape, kernel_size, stride, padding, dilation, groups, device, float_dtype):
        input = nested_rand(shape, device, float_dtype)
        layer = nn.Conv2d(input.shape[-1], 4, kernel_size, stride, padding, dilation, groups).to(
            device=device, dtype=float_dtype
        )
        reference_layer = nn.Conv2d(input.shape[-1], 4, kernel_size, stride, padding, dilation, groups).to(
            device=device, dtype=float_dtype
        )
        reference_layer.load_state_dict(layer.state_dict())
        input = input.permute(0, 3, 1, 2)
        output = layer(input)
        reference = reference_layer(input.tensor)
        reference.masked_fill_(~output.mask, 0)
        assert_close(output, reference, atol=1e-6)
        output.sum().backward()
        reference.sum().backward()
        assert_close(layer.weight.grad, reference_layer.weight.grad)
        assert_close(layer.bias.grad, reference_layer.bias.grad)


class TestConv3dModule:
    @pytest.mark.parametrize("shape", [[(5, 7, 9, 8), (11, 13, 15, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("groups", [1, 2])
    def test_conv3d_module(self, shape, kernel_size, stride, padding, dilation, groups, device, float_dtype):
        input = nested_rand(shape, device, float_dtype)
        layer = nn.Conv3d(input.shape[-1], 4, kernel_size, stride, padding, dilation, groups).to(
            device=device, dtype=float_dtype
        )
        reference_layer = nn.Conv3d(input.shape[-1], 4, kernel_size, stride, padding, dilation, groups).to(
            device=device, dtype=float_dtype
        )
        reference_layer.load_state_dict(layer.state_dict())
        input = input.transpose(1, -1)
        output = layer(input)
        reference = reference_layer(input.tensor)
        reference.masked_fill_(~output.mask, 0)
        assert_close(output, reference, atol=1e-6)
        output.sum().backward()
        reference.sum().backward()
        assert_close(layer.weight.grad, reference_layer.weight.grad, atol=1e-5, rtol=1e-2)
        assert_close(layer.bias.grad, reference_layer.bias.grad, atol=1e-5, rtol=1e-2)


class TestConvTransposeModule:
    @pytest.mark.parametrize("shape", [[(5, 8), (7, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("output_padding", [0, 1])
    @pytest.mark.parametrize("groups", [1, 2])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_conv_transpose1d_module(
        self, shape, kernel_size, stride, padding, output_padding, groups, dilation, device, float_dtype
    ):
        if stride == 1 and output_padding > 0:
            pytest.skip("output_padding > 0 only valid when stride > 1")
        input = nested_rand(shape, device, float_dtype)
        layer = nn.ConvTranspose1d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        ).to(device=device, dtype=float_dtype)
        reference_layer = nn.ConvTranspose1d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        ).to(device=device, dtype=float_dtype)
        reference_layer.load_state_dict(layer.state_dict())
        input = input.transpose(-1, -2)
        output = layer(input)
        reference = reference_layer(input.tensor)
        reference.masked_fill_(~output.mask, 0)
        assert_close(output, reference, atol=1e-6)
        output.sum().backward()
        reference.sum().backward()
        assert_close(layer.weight.grad, reference_layer.weight.grad)
        assert_close(layer.bias.grad, reference_layer.bias.grad)

    @pytest.mark.parametrize("shape", [[(5, 7, 8), (11, 13, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("output_padding", [0, 1])
    @pytest.mark.parametrize("groups", [1, 2])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_conv_transpose2d_module(
        self, shape, kernel_size, stride, padding, output_padding, dilation, groups, device, float_dtype
    ):
        if stride == 1 and output_padding > 0:
            pytest.skip("output_padding > 0 only valid when stride > 1")
        input = nested_rand(shape, device, float_dtype)
        layer = nn.ConvTranspose2d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        ).to(device=device, dtype=float_dtype)
        reference_layer = nn.ConvTranspose2d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        ).to(device=device, dtype=float_dtype)
        reference_layer.load_state_dict(layer.state_dict())
        input = input.transpose(1, -1)
        output = layer(input)
        reference = reference_layer(input.tensor)
        reference.masked_fill_(~output.mask, 0)
        assert_close(output, reference, atol=1e-6)
        output.sum().backward()
        reference.sum().backward()
        assert_close(layer.weight.grad, reference_layer.weight.grad)
        assert_close(layer.bias.grad, reference_layer.bias.grad)

    @pytest.mark.parametrize("shape", [[(5, 7, 9, 8), (11, 13, 15, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("output_padding", [0, 1])
    @pytest.mark.parametrize("groups", [1, 2])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_conv_transpose3d_module(
        self, shape, kernel_size, stride, padding, output_padding, dilation, groups, device, float_dtype
    ):
        if stride == 1 and output_padding > 0:
            pytest.skip("output_padding > 0 only valid when stride > 1")
        input = nested_rand(shape, device, float_dtype)
        layer = nn.ConvTranspose3d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        ).to(device=device, dtype=float_dtype)
        reference_layer = nn.ConvTranspose3d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        ).to(device=device, dtype=float_dtype)
        reference_layer.load_state_dict(layer.state_dict())
        input = input.permute(0, 4, 1, 2, 3)
        output = layer(input)
        reference = reference_layer(input.tensor)
        reference.masked_fill_(~output.mask, 0)
        assert_close(output, reference, atol=1e-6)
        output.sum().backward()
        reference.sum().backward()
        assert_close(layer.weight.grad, reference_layer.weight.grad, atol=1e-5, rtol=1e-2)
        assert_close(layer.bias.grad, reference_layer.bias.grad, atol=1e-5, rtol=1e-2)
