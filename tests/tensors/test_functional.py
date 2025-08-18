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
from torch.testing import assert_close

from danling.tensors import NestedTensor


class TestLinear:

    @pytest.mark.parametrize("shape", [[(3, 5), (2, 5)], [(2, 3, 5), (3, 2, 5)]])
    def test_functional(self, shape):
        input = NestedTensor([torch.randn(*i) for i in shape])
        weight = torch.randn(3, input.shape[-1])
        bias = torch.randn(3)
        output = F.linear(input, weight, bias)
        reference = F.linear(input.tensor, weight, bias)
        assert torch.allclose(output, reference)

    @pytest.mark.parametrize("shape", [[(3, 5), (2, 5)], [(2, 3, 5), (3, 2, 5)]])
    def test_module(self, shape):
        input = NestedTensor([torch.randn(*i) for i in shape])
        layer1 = nn.Linear(input.shape[-1], 3)
        layer2 = nn.Linear(input.shape[-1], 3)
        layer2.load_state_dict(layer1.state_dict())
        output1 = layer1(input)
        output2 = layer2(input.tensor)
        output2.masked_fill_(~output1.mask.unsqueeze(-1), 0)
        assert torch.allclose(output1, output2)
        output1.sum().backward()
        output2.sum().backward()
        assert_close(layer1.weight.grad, layer2.weight.grad)
        assert_close(layer1.bias.grad, layer2.bias.grad)


class TestConv1d:
    @pytest.mark.parametrize("shape", [[(5, 8), (7, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("groups", [1, 2])
    def test_functional(self, shape, kernel_size, stride, padding, dilation, groups):
        input = NestedTensor([torch.randn(*i) for i in shape])
        weight = torch.randn(4, input.shape[-1] // groups, kernel_size)
        bias = torch.randn(4)
        input = input.transpose(-1, -2)
        output = F.conv1d(input, weight, bias, stride, padding, dilation, groups)
        reference = F.conv1d(input.tensor, weight, bias, stride, padding, dilation, groups)
        assert torch.allclose(output, reference, atol=1e-6)

    @pytest.mark.parametrize("shape", [[(5, 8), (7, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("groups", [1, 2])
    def test_module(self, shape, kernel_size, stride, padding, dilation, groups):
        input = NestedTensor([torch.randn(*i) for i in shape])
        layer1 = nn.Conv1d(input.shape[-1], 4, kernel_size)
        layer2 = nn.Conv1d(input.shape[-1], 4, kernel_size)
        layer2.load_state_dict(layer1.state_dict())
        input = input.transpose(1, -1)
        output1 = layer1(input)
        output2 = layer2(input.tensor)
        output2.masked_fill_(~output1.mask, 0)
        assert torch.allclose(output1, output2, atol=1e-6)
        output1.sum().backward()
        output2.sum().backward()
        assert_close(layer1.weight.grad, layer2.weight.grad)
        assert_close(layer1.bias.grad, layer2.bias.grad)


class TestConv2d:
    @pytest.mark.parametrize("shape", [[(5, 7, 8), (11, 13, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("groups", [1, 2])
    def test_functional(self, shape, kernel_size, stride, padding, dilation, groups):
        input = NestedTensor([torch.randn(*i) for i in shape])
        weight = torch.randn(4, input.shape[-1] // groups, kernel_size, kernel_size)
        bias = torch.randn(4)
        input = input.transpose(1, -1)
        output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        reference = F.conv2d(input.tensor, weight, bias, stride, padding, dilation, groups)
        assert torch.allclose(output, reference, atol=1e-6)

    @pytest.mark.parametrize("shape", [[(5, 7, 8), (11, 13, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("groups", [1, 2])
    def test_module(self, shape, kernel_size, stride, padding, dilation, groups):
        input = NestedTensor([torch.randn(*i) for i in shape])
        layer1 = nn.Conv2d(input.shape[-1], 4, kernel_size, stride, padding, dilation, groups)
        layer2 = nn.Conv2d(input.shape[-1], 4, kernel_size, stride, padding, dilation, groups)
        layer2.load_state_dict(layer1.state_dict())
        input = input.permute(0, 3, 1, 2)
        output1 = layer1(input)
        output2 = layer2(input.tensor)
        output2.masked_fill_(~output1.mask, 0)
        output1.sum().backward()
        output2.sum().backward()
        assert_close(layer1.weight.grad, layer2.weight.grad)
        assert_close(layer1.bias.grad, layer2.bias.grad)


class TestConv3d:
    @pytest.mark.parametrize("shape", [[(5, 7, 9, 8), (11, 13, 15, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("groups", [1, 2])
    def test_functional(self, shape, kernel_size, stride, padding, dilation, groups):
        input = NestedTensor([torch.randn(*i) for i in shape])
        weight = torch.randn(4, input.shape[-1] // groups, kernel_size, kernel_size, kernel_size)
        bias = torch.randn(4)
        input = input.permute(0, 4, 1, 2, 3)
        output = F.conv3d(input, weight, bias, stride, padding, dilation, groups)
        reference = F.conv3d(input.tensor, weight, bias, stride, padding, dilation, groups)
        assert torch.allclose(output, reference, atol=1e-6)

    @pytest.mark.parametrize("shape", [[(5, 7, 9, 8), (11, 13, 15, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("groups", [1, 2])
    def test_module(self, shape, kernel_size, stride, padding, dilation, groups):
        input = NestedTensor([torch.randn(*i) for i in shape])
        layer1 = nn.Conv3d(input.shape[-1], 4, kernel_size, stride, padding, dilation, groups)
        layer2 = nn.Conv3d(input.shape[-1], 4, kernel_size, stride, padding, dilation, groups)
        layer2.load_state_dict(layer1.state_dict())
        input = input.transpose(1, -1)
        output1 = layer1(input)
        output2 = layer2(input.tensor)
        output2.masked_fill_(~output1.mask, 0)
        assert torch.allclose(output1, output2, atol=1e-6)
        output1.sum().backward()
        output2.sum().backward()
        assert_close(layer1.weight.grad, layer2.weight.grad, atol=1e-5, rtol=1e-2)
        assert_close(layer1.bias.grad, layer2.bias.grad, atol=1e-5, rtol=1e-2)


class TestConvTranspose1d:
    @pytest.mark.parametrize("shape", [[(5, 8), (7, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("output_padding", [0, 1])
    @pytest.mark.parametrize("groups", [1, 2])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_functional(self, shape, kernel_size, stride, padding, output_padding, groups, dilation):
        if stride == 1 and output_padding > 0:
            pytest.skip("output_padding > 0 only valid when stride > 1")
        input = NestedTensor([torch.randn(*i) for i in shape])
        weight = torch.randn(input.shape[-1], 4 // groups, kernel_size)
        bias = torch.randn(4)
        input = input.transpose(1, -1)
        output = F.conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation)
        reference = F.conv_transpose1d(input.tensor, weight, bias, stride, padding, output_padding, groups, dilation)
        assert torch.allclose(output, reference, atol=1e-6)

    @pytest.mark.parametrize("shape", [[(5, 8), (7, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("output_padding", [0, 1])
    @pytest.mark.parametrize("groups", [1, 2])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_module(self, shape, kernel_size, stride, padding, output_padding, groups, dilation):
        if stride == 1 and output_padding > 0:
            pytest.skip("output_padding > 0 only valid when stride > 1")
        input = NestedTensor([torch.randn(*i) for i in shape])
        layer1 = nn.ConvTranspose1d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        )
        layer2 = nn.ConvTranspose1d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        )
        layer2.load_state_dict(layer1.state_dict())
        input = input.transpose(-1, -2)
        output1 = layer1(input)
        output2 = layer2(input.tensor)
        output2.masked_fill_(~output1.mask, 0)
        assert torch.allclose(output1, output2, atol=1e-6)
        output1.sum().backward()
        output2.sum().backward()
        assert_close(layer1.weight.grad, layer2.weight.grad)
        assert_close(layer1.bias.grad, layer2.bias.grad)


class TestConvTranspose2d:
    @pytest.mark.parametrize("shape", [[(5, 7, 8), (11, 13, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("output_padding", [0, 1])
    @pytest.mark.parametrize("groups", [1, 2])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_functional(self, shape, kernel_size, stride, padding, output_padding, dilation, groups):
        if stride == 1 and output_padding > 0:
            pytest.skip("output_padding > 0 only valid when stride > 1")
        input = NestedTensor([torch.randn(*i) for i in shape])
        weight = torch.randn(input.shape[-1], 4 // groups, kernel_size, kernel_size)
        bias = torch.randn(4)
        input = input.permute(0, 3, 1, 2)
        output = F.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)
        reference = F.conv_transpose2d(input.tensor, weight, bias, stride, padding, output_padding, groups, dilation)
        assert torch.allclose(output, reference, atol=1e-6)

    @pytest.mark.parametrize("shape", [[(5, 7, 8), (11, 13, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("output_padding", [0, 1])
    @pytest.mark.parametrize("groups", [1, 2])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_module(self, shape, kernel_size, stride, padding, output_padding, dilation, groups):
        if stride == 1 and output_padding > 0:
            pytest.skip("output_padding > 0 only valid when stride > 1")
        input = NestedTensor([torch.randn(*i) for i in shape])
        layer1 = nn.ConvTranspose2d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        )
        layer2 = nn.ConvTranspose2d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        )
        layer2.load_state_dict(layer1.state_dict())
        input = input.transpose(1, -1)
        output1 = layer1(input)
        output2 = layer2(input.tensor)
        output2.masked_fill_(~output1.mask, 0)
        assert torch.allclose(output1, output2, atol=1e-6)
        output1.sum().backward()
        output2.sum().backward()
        assert_close(layer1.weight.grad, layer2.weight.grad)
        assert_close(layer1.bias.grad, layer2.bias.grad)


class TestConvTranspose3d:
    @pytest.mark.parametrize("shape", [[(5, 7, 9, 8), (11, 13, 15, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("output_padding", [0, 1])
    @pytest.mark.parametrize("groups", [1, 2])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_functional(self, shape, kernel_size, stride, padding, output_padding, dilation, groups):
        if stride == 1 and output_padding > 0:
            pytest.skip("output_padding > 0 only valid when stride > 1")
        input = NestedTensor([torch.randn(*i) for i in shape])
        weight = torch.randn(input.shape[-1], 4 // groups, kernel_size, kernel_size, kernel_size)
        bias = torch.randn(4)
        input = input.transpose(1, -1)
        output = F.conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation)
        reference = F.conv_transpose3d(input.tensor, weight, bias, stride, padding, output_padding, groups, dilation)
        assert torch.allclose(output, reference, atol=1e-6)

    @pytest.mark.parametrize("shape", [[(5, 7, 9, 8), (11, 13, 15, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("output_padding", [0, 1])
    @pytest.mark.parametrize("groups", [1, 2])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_module(self, shape, kernel_size, stride, padding, output_padding, dilation, groups):
        if stride == 1 and output_padding > 0:
            pytest.skip("output_padding > 0 only valid when stride > 1")
        input = NestedTensor([torch.randn(*i) for i in shape])
        layer1 = nn.ConvTranspose3d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        )
        layer2 = nn.ConvTranspose3d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        )
        layer2.load_state_dict(layer1.state_dict())
        input = input.permute(0, 4, 1, 2, 3)
        output1 = layer1(input)
        output2 = layer2(input.tensor)
        output2.masked_fill_(~output1.mask, 0)
        assert torch.allclose(output1, output2, atol=1e-6)
        output1.sum().backward()
        output2.sum().backward()
        assert_close(layer1.weight.grad, layer2.weight.grad, atol=1e-5, rtol=1e-2)
        assert_close(layer1.bias.grad, layer2.bias.grad, atol=1e-5, rtol=1e-2)
