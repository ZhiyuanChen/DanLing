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


def _compare_conv1d(device, tolerances, *, kernel_size: int, padding: int):
    parts = [torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True) for shape in [(3, 17), (3, 29)]]
    references = [part.detach().clone().requires_grad_() for part in parts]
    weight = torch.randn(5, 3, kernel_size, device=device, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(5, device=device, dtype=torch.float32, requires_grad=True)
    reference_weight = weight.detach().clone().requires_grad_()
    reference_bias = bias.detach().clone().requires_grad_()
    kwargs = {"padding": padding}
    actual = F.conv1d(NT(parts), weight, bias, **kwargs)
    expected = [F.conv1d(part, reference_weight, reference_bias, **kwargs) for part in references]
    atol, rtol = tolerances()
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
        _compare_conv1d(device, kernel_size=1, padding=0, tolerances=tolerances)

    def test_spatial(self, device, tolerances):
        _compare_conv1d(device, kernel_size=3, padding=1, tolerances=tolerances)

    def test_bias(self, device):
        input = NT([torch.randn(3, 17, device=device), torch.randn(3, 29, device=device)])
        weight = torch.randn(5, 3, 3, device=device)
        bias = torch.randn(4, device=device)
        with pytest.raises(RuntimeError):
            F.conv1d(input, weight, bias, padding=1)

    def test_autocast(self, device):
        if device.type != "cuda":
            pytest.skip("CUDA autocast")
        shapes = [(2, 9), (2, 7)]
        parts = [torch.randn(shape, device=device, requires_grad=True) for shape in shapes]
        references = [part.detach().clone().requires_grad_() for part in parts]
        weight = torch.randn(3, 2, 3, device=device, requires_grad=True)
        bias = torch.randn(3, device=device, requires_grad=True)
        reference_weight = weight.detach().clone().requires_grad_()
        reference_bias = bias.detach().clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = F.conv1d(NT(parts), weight, bias, **{"stride": 2, "padding": 1})
            expected = [
                F.conv1d(part, reference_weight, reference_bias, **{"stride": 2, "padding": 1}) for part in references
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

    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    def test_fullgraph(self, backend):
        parts = [torch.randn(shape, dtype=torch.float64, requires_grad=True) for shape in [(2, 3), (2, 5)]]
        weight = torch.randn(3, 2, 1, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(3, dtype=torch.float64, requires_grad=True)
        expected = [F.conv1d(part, weight, bias) for part in parts]
        expected_gradients = torch.autograd.grad(sum(part.square().sum() for part in expected), (*parts, weight, bias))
        actual = torch.compile(F.conv1d, backend=backend, fullgraph=True)(NT(parts), weight, bias)
        torch.testing.assert_close(actual.unbind(), expected)
        actual_gradients = torch.autograd.grad(actual.concat.square().sum(), (*parts, weight, bias))
        torch.testing.assert_close(actual_gradients, expected_gradients)
