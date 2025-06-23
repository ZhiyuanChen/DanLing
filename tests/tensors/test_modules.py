import pytest
import torch
from torch import nn
from torch.testing import assert_close

from tests.tensors.utils import nested_rand


class TestLinearModule:

    @pytest.mark.parametrize("shape", [[(3, 5), (2, 5)], [(2, 3, 5), (3, 2, 5)]])
    def test_linear_module_matches_tensor(self, shape, device, float_dtype):
        input = nested_rand(shape, device, float_dtype)
        layer1 = nn.Linear(input.shape[-1], 3).to(device=device, dtype=float_dtype)
        layer2 = nn.Linear(input.shape[-1], 3).to(device=device, dtype=float_dtype)
        layer2.load_state_dict(layer1.state_dict())
        output1 = layer1(input)
        output2 = layer2(input.tensor)
        output2.masked_fill_(~output1.mask.unsqueeze(-1), 0)
        assert torch.allclose(output1, output2)
        output1.sum().backward()
        output2.sum().backward()
        assert_close(layer1.weight.grad, layer2.weight.grad)
        assert_close(layer1.bias.grad, layer2.bias.grad)


class TestConv1dModule:
    @pytest.mark.parametrize("shape", [[(5, 8), (7, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("groups", [1, 2])
    def test_conv1d_module(self, shape, kernel_size, stride, padding, dilation, groups, device, float_dtype):
        input = nested_rand(shape, device, float_dtype)
        layer1 = nn.Conv1d(input.shape[-1], 4, kernel_size).to(device=device, dtype=float_dtype)
        layer2 = nn.Conv1d(input.shape[-1], 4, kernel_size).to(device=device, dtype=float_dtype)
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


class TestConv2dModule:
    @pytest.mark.parametrize("shape", [[(5, 7, 8), (11, 13, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("groups", [1, 2])
    def test_conv2d_module(self, shape, kernel_size, stride, padding, dilation, groups, device, float_dtype):
        input = nested_rand(shape, device, float_dtype)
        layer1 = nn.Conv2d(input.shape[-1], 4, kernel_size, stride, padding, dilation, groups).to(
            device=device, dtype=float_dtype
        )
        layer2 = nn.Conv2d(input.shape[-1], 4, kernel_size, stride, padding, dilation, groups).to(
            device=device, dtype=float_dtype
        )
        layer2.load_state_dict(layer1.state_dict())
        input = input.permute(0, 3, 1, 2)
        output1 = layer1(input)
        output2 = layer2(input.tensor)
        output2.masked_fill_(~output1.mask, 0)
        output1.sum().backward()
        output2.sum().backward()
        assert_close(layer1.weight.grad, layer2.weight.grad)
        assert_close(layer1.bias.grad, layer2.bias.grad)


class TestConv3dModule:
    @pytest.mark.parametrize("shape", [[(5, 7, 9, 8), (11, 13, 15, 8)]])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("groups", [1, 2])
    def test_conv3d_module(self, shape, kernel_size, stride, padding, dilation, groups, device, float_dtype):
        input = nested_rand(shape, device, float_dtype)
        layer1 = nn.Conv3d(input.shape[-1], 4, kernel_size, stride, padding, dilation, groups).to(
            device=device, dtype=float_dtype
        )
        layer2 = nn.Conv3d(input.shape[-1], 4, kernel_size, stride, padding, dilation, groups).to(
            device=device, dtype=float_dtype
        )
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
        layer1 = nn.ConvTranspose1d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        ).to(device=device, dtype=float_dtype)
        layer2 = nn.ConvTranspose1d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        ).to(device=device, dtype=float_dtype)
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
        layer1 = nn.ConvTranspose2d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        ).to(device=device, dtype=float_dtype)
        layer2 = nn.ConvTranspose2d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        ).to(device=device, dtype=float_dtype)
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
        layer1 = nn.ConvTranspose3d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        ).to(device=device, dtype=float_dtype)
        layer2 = nn.ConvTranspose3d(
            input.shape[-1], 4, kernel_size, stride, padding, output_padding, groups, bias=True, dilation=dilation
        ).to(device=device, dtype=float_dtype)
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
