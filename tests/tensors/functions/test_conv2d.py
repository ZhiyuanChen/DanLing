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

from danling.tensors import NestedTensor as NT


def _compare_conv2d(device, tolerances, *, kernel_size: int, padding: int, stride: int = 1, dilation: int = 1):
    parts = [
        torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True)
        for shape in [(3, 65, 67), (3, 17, 19)]
    ]
    references = [part.detach().clone().requires_grad_() for part in parts]
    weight = torch.randn(5, 3, kernel_size, kernel_size, device=device, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(5, device=device, dtype=torch.float64, requires_grad=True)
    reference_weight = weight.detach().clone().requires_grad_()
    reference_bias = bias.detach().clone().requires_grad_()
    kwargs = {"padding": padding, "stride": stride, "dilation": dilation}
    actual = F.conv2d(NT(parts), weight, bias, **kwargs)
    expected = [F.conv2d(part, reference_weight, reference_bias, **kwargs) for part in references]
    atol, rtol = tolerances(torch.float64)
    torch.testing.assert_close(actual.unbind(), expected, atol=atol, rtol=rtol)
    probes = [torch.randn_like(part) for part in expected]
    actual_gradients = torch.autograd.grad(
        sum((part * probe).sum() for (part, probe) in zip(actual, probes)), (*parts, weight, bias)
    )
    reference_gradients = torch.autograd.grad(
        sum((part * probe).sum() for (part, probe) in zip(expected, probes)),
        (*references, reference_weight, reference_bias),
    )
    torch.testing.assert_close(actual_gradients, reference_gradients, atol=atol, rtol=rtol)


class TestConvolution:

    def test_pointwise(self, device, tolerances):
        _compare_conv2d(device, kernel_size=1, padding=0, tolerances=tolerances)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_spatial_backward(self, device, dtype):
        if device.type != "cuda":
            pytest.skip("CUDA spatial convolution")
        parts = [torch.randn(128, h, w, device=device, dtype=dtype, requires_grad=True) for h, w in ((4, 5), (5, 4))]
        references = [part.detach().clone().requires_grad_() for part in parts]
        weight = torch.randn(128, 128, 3, 3, device=device, dtype=dtype, requires_grad=True)
        bias = torch.randn(128, device=device, dtype=dtype, requires_grad=True)
        reference_weight = weight.detach().clone().requires_grad_()
        reference_bias = bias.detach().clone().requires_grad_()
        with torch.backends.cudnn.flags(allow_tf32=False):
            actual = F.conv2d(NT(parts, ragged_dims=(1, 2)), weight, bias, padding=1).unbind()
            expected = [F.conv2d(part, reference_weight, reference_bias, padding=1) for part in references]
            torch.testing.assert_close(actual, expected)
            probes = [torch.randn_like(part) for part in expected]
            for _ in range(2):
                gradients = torch.autograd.grad(actual, (*parts, weight, bias), probes, retain_graph=True)
                reference_gradients = torch.autograd.grad(
                    expected, (*references, reference_weight, reference_bias), probes, retain_graph=True
                )
                torch.testing.assert_close(gradients, reference_gradients)

    @pytest.mark.parametrize(("stride", "dilation"), [(1, 1), (2, 1), (1, 2)])
    def test_spatial(self, device, tolerances, stride, dilation):
        _compare_conv2d(
            device, kernel_size=3, padding=dilation, stride=stride, dilation=dilation, tolerances=tolerances
        )

    def test_unsampled_values(self, device):
        parts = [torch.ones(1, size, size, device=device, dtype=torch.float64, requires_grad=True) for size in (6, 12)]
        with torch.no_grad():
            parts[0][0, 4, 4] = float("nan")
        references = [part.detach().clone().requires_grad_() for part in parts]
        weight = torch.arange(1.0, 5.0, device=device, dtype=torch.float64).reshape(1, 1, 2, 2).requires_grad_()
        reference_weight = weight.detach().clone().requires_grad_()
        kwargs = {"stride": 4, "dilation": 2}
        actual = F.conv2d(NT(parts, ragged_dims=(1, 2)), weight, **kwargs)
        expected = [F.conv2d(part, reference_weight, **kwargs) for part in references]
        torch.testing.assert_close(actual.unbind(), expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.concat.sum(), (*parts, weight)),
            torch.autograd.grad(sum(part.sum() for part in expected), (*references, reference_weight)),
        )

    def test_bias(self, device):
        input = NT([torch.randn(3, 11, 13, device=device), torch.randn(3, 7, 9, device=device)])
        weight = torch.randn(5, 3, 3, 3, device=device)
        bias = torch.randn(4, device=device)
        with pytest.raises(RuntimeError):
            F.conv2d(input, weight, bias, padding=1)

    def test_autocast(self, device):
        if device.type != "cuda":
            pytest.skip("CUDA autocast")
        shapes = [(2, 6, 7), (2, 5, 9)]
        parts = [torch.randn(shape, device=device, requires_grad=True) for shape in shapes]
        references = [part.detach().clone().requires_grad_() for part in parts]
        weight = torch.randn(3, 2, 3, 3, device=device, requires_grad=True)
        bias = torch.randn(3, device=device, requires_grad=True)
        reference_weight = weight.detach().clone().requires_grad_()
        reference_bias = bias.detach().clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = F.conv2d(NT(parts), weight, bias, **{"stride": 2, "padding": 1})
            expected = [
                F.conv2d(part, reference_weight, reference_bias, **{"stride": 2, "padding": 1}) for part in references
            ]
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual.unbind(), expected, atol=0.02, rtol=0.02)
        probes = [torch.randn_like(part) for part in expected]
        actual_gradients = torch.autograd.grad(
            sum((part * probe).sum() for (part, probe) in zip(actual, probes)), (*parts, weight, bias)
        )
        reference_gradients = torch.autograd.grad(
            sum((part * probe).sum() for (part, probe) in zip(expected, probes)),
            (*references, reference_weight, reference_bias),
        )
        torch.testing.assert_close(actual_gradients, reference_gradients, atol=0.03, rtol=0.03)

    @pytest.mark.parametrize("backend", [None, "aot_eager", "inductor"])
    @pytest.mark.parametrize(("stride", "padding", "dilation", "groups"), [(1, 1, 1, 1), (2, 2, 2, 2)])
    def test_dynamic_spatial(self, device, tolerances, backend, stride, padding, dilation, groups):
        torch.compiler.reset()

        def convolve(source, weight, bias):
            return F.conv2d(source, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

        operation = (
            convolve if backend is None else torch.compile(convolve, backend=backend, fullgraph=True, dynamic=True)
        )
        for shapes in (((5, 7), (17, 19)), ((9, 13), (18, 21))):
            parts = [torch.randn(4, *shape, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
            references = [part.detach().clone().requires_grad_() for part in parts]
            weight = torch.randn(6, 4 // groups, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
            bias = torch.randn(6, device=device, dtype=torch.float64, requires_grad=True)
            reference_weight = weight.detach().clone().requires_grad_()
            reference_bias = bias.detach().clone().requires_grad_()
            source = NT(parts, ragged_dims=(1, 2))
            actual = operation(source, weight, bias)
            expected = [convolve(part, reference_weight, reference_bias) for part in references]
            atol, rtol = tolerances(torch.float64)
            torch.testing.assert_close(actual.unbind(), expected, atol=atol, rtol=rtol)
            torch.testing.assert_close(actual.element_sizes(), torch.tensor([part.shape for part in expected]))
            probes = [torch.randn_like(part) for part in expected]
            actual_gradients = torch.autograd.grad(
                sum((part * probe).sum() for part, probe in zip(actual, probes)), (*parts, weight, bias)
            )
            reference_gradients = torch.autograd.grad(
                sum((part * probe).sum() for part, probe in zip(expected, probes)),
                (*references, reference_weight, reference_bias),
            )
            torch.testing.assert_close(actual_gradients, reference_gradients, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("backend", [None, "aot_eager", "inductor"])
    def test_composed(self, device, tolerances, backend):
        torch.compiler.reset()

        def convolve(source, first, second):
            return F.conv2d(F.conv2d(source, first, stride=2, padding=1), second, padding=1)

        operation = (
            convolve if backend is None else torch.compile(convolve, backend=backend, fullgraph=True, dynamic=True)
        )
        for shapes in (((5, 7), (17, 19)), ((9, 13), (18, 21))):
            parts = [torch.randn(4, *shape, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
            references = [part.detach().clone().requires_grad_() for part in parts]
            first = torch.randn(6, 4, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
            second = torch.randn(3, 6, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
            reference_first = first.detach().clone().requires_grad_()
            reference_second = second.detach().clone().requires_grad_()
            actual = operation(NT(parts, ragged_dims=(1, 2)), first, second)
            expected = [convolve(part, reference_first, reference_second) for part in references]
            atol, rtol = tolerances(torch.float64)
            torch.testing.assert_close(actual.unbind(), expected, atol=atol, rtol=rtol)
            torch.testing.assert_close(actual.element_sizes(), torch.tensor([part.shape for part in expected]))
            probes = [torch.randn_like(part) for part in expected]
            torch.testing.assert_close(
                torch.autograd.grad(
                    sum((part * probe).sum() for part, probe in zip(actual, probes)), (*parts, first, second)
                ),
                torch.autograd.grad(
                    sum((part * probe).sum() for part, probe in zip(expected, probes)),
                    (*references, reference_first, reference_second),
                ),
                atol=atol,
                rtol=rtol,
            )

    @pytest.mark.parametrize("backend", ["aot_eager"])
    def test_fullgraph(self, backend):
        parts = [torch.randn(shape, dtype=torch.float64, requires_grad=True) for shape in [(2, 3, 4), (2, 4, 6)]]
        weight = torch.randn(3, 2, 1, 1, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(3, dtype=torch.float64, requires_grad=True)
        expected = [F.conv2d(part, weight, bias) for part in parts]
        expected_gradients = torch.autograd.grad(sum(part.square().sum() for part in expected), (*parts, weight, bias))
        actual = torch.compile(F.conv2d, backend=backend, fullgraph=True)(NT(parts), weight, bias)
        torch.testing.assert_close(actual.unbind(), expected)
        actual_gradients = torch.autograd.grad(actual.concat.square().sum(), (*parts, weight, bias))
        torch.testing.assert_close(actual_gradients, expected_gradients)
