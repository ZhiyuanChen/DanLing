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


def _compare_conv1d(device, tolerances, *, kernel_size: int, padding: int, stride: int = 1):
    parts = [
        torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True) for shape in [(3, 4097), (3, 8195)]
    ]
    references = [part.detach().clone().requires_grad_() for part in parts]
    weight = torch.randn(5, 3, kernel_size, device=device, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(5, device=device, dtype=torch.float64, requires_grad=True)
    reference_weight = weight.detach().clone().requires_grad_()
    reference_bias = bias.detach().clone().requires_grad_()
    kwargs = {"padding": padding, "stride": stride}
    actual = F.conv1d(NT(parts), weight, bias, **kwargs)
    expected = [F.conv1d(part, reference_weight, reference_bias, **kwargs) for part in references]
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


class TestDepthwise:

    @pytest.mark.parametrize("backend", [None, "aot_eager", "inductor"])
    def test_bias(self, device, backend):
        torch.compiler.reset()

        def convolve(source, weight, bias):
            return F.conv1d(source, weight, bias)

        operation = (
            convolve if backend is None else torch.compile(convolve, backend=backend, fullgraph=True, dynamic=True)
        )
        for lengths in ((2, 2), (2, 2, 2)):
            parts = [torch.zeros(1, length, device=device) for length in lengths]
            for part in parts:
                part[0, :2] = torch.tensor([1e8, -1e8], device=device)
                part.requires_grad_()
            weight = torch.ones(1, 1, 2, device=device, requires_grad=True)
            bias = torch.ones(1, device=device, requires_grad=True)
            output = operation(NT(parts, ragged_dims=(1,)), weight, bias)
            expected = [convolve(part, weight, bias) for part in parts]
            torch.testing.assert_close(output.unbind(), expected)
            torch.testing.assert_close(
                torch.autograd.grad(output.values().sum(), (*parts, weight, bias)),
                torch.autograd.grad(sum(value.sum() for value in expected), (*parts, weight, bias)),
            )

    @pytest.mark.parametrize("backend", [None, "aot_eager", "inductor"])
    @pytest.mark.parametrize("operand", ["input", "weight"])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_nonfinite(self, device, backend, operand, padding, dtype):
        torch.compiler.reset()

        def convolve(source, weight):
            return F.conv1d(source, weight, padding=padding, groups=1)

        operation = (
            convolve if backend is None else torch.compile(convolve, backend=backend, fullgraph=True, dynamic=True)
        )
        for lengths in ((3, 3), (5, 6)):
            parts = [torch.ones(1, length, device=device, dtype=dtype) for length in lengths]
            weight = torch.ones(1, 1, 3, device=device, dtype=dtype)
            if operand == "input":
                parts[0][0, -1] = float("nan")
            else:
                weight[0, 0, 0] = float("nan")
            parts = [part.requires_grad_() for part in parts]
            weight.requires_grad_()
            references = [part.detach().clone().requires_grad_() for part in parts]
            reference_weight = weight.detach().clone().requires_grad_()
            output = operation(NT(parts, ragged_dims=(1,)), weight)
            expected = [F.conv1d(F.pad(part, (padding, padding)), reference_weight) for part in references]
            torch.testing.assert_close(output.unbind(), expected, equal_nan=True)
            torch.testing.assert_close(
                torch.autograd.grad(output.values().sum(), (*parts, weight)),
                torch.autograd.grad(sum(value.sum() for value in expected), (*references, reference_weight)),
                equal_nan=True,
            )

    @pytest.mark.parametrize("backend", [None, "aot_eager", "inductor"])
    def test_causal(self, device, backend):
        torch._dynamo.reset()
        channels = 8192 if device.type == "cuda" else 3
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float64

        def causal(template, values, weight, bias):
            source = template.packed_like(values)
            return F.silu(F.conv1d(source, weight, bias, padding=3, groups=channels)[..., :-3])

        operation = causal if backend is None else torch.compile(causal, backend=backend, fullgraph=True, dynamic=True)
        for lengths in ((1, 3, 7, 13), (2, 5, 1)):
            template = NT(
                [torch.empty(channels, length, device=device, dtype=dtype) for length in lengths], ragged_dims=(1,)
            )
            values = (
                torch.randn(sum(lengths), channels * 2, device=device, dtype=dtype)[:, ::2].detach().requires_grad_()
            )
            weight = (torch.randn(channels, 1, 4, device=device, dtype=dtype) / 4).requires_grad_()
            bias = (torch.randn(channels, device=device, dtype=dtype) / 4).requires_grad_()
            reference_values = values.detach().clone().to(torch.float64).requires_grad_()
            reference_weight = weight.detach().clone().to(torch.float64).requires_grad_()
            reference_bias = bias.detach().clone().to(torch.float64).requires_grad_()
            with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
                actual = operation(template, values, weight, bias)
            expected = [
                F.silu(F.conv1d(part.T, reference_weight, reference_bias, padding=3, groups=channels)[..., :-3])
                for part in reference_values.split(lengths)
            ]
            atol, rtol = (0.03, 0.03) if dtype == torch.bfloat16 else (1e-10, 1e-10)
            assert actual.dtype == dtype
            torch.testing.assert_close(actual.element_sizes(), torch.tensor([(channels, length) for length in lengths]))
            torch.testing.assert_close(actual.unbind(), expected, atol=atol, rtol=rtol, check_dtype=False)
            probes = [torch.randn_like(part, dtype=dtype) for part in expected]
            gradients = torch.autograd.grad(
                sum((part * probe).sum() for (part, probe) in zip(actual, probes)), (values, weight, bias)
            )
            reference_gradients = torch.autograd.grad(
                sum((part * probe).sum() for (part, probe) in zip(expected, probes)),
                (reference_values, reference_weight, reference_bias),
            )
            torch.testing.assert_close(gradients, reference_gradients, atol=atol, rtol=rtol, check_dtype=False)

    @pytest.mark.parametrize("backend", [None, "aot_eager"])
    @pytest.mark.parametrize(("kernel", "padding", "dilation", "multiplier"), [(3, 1, 2, 2), (4, 3, 1, 1)])
    def test_convolution(self, device, backend, kernel, padding, dilation, multiplier):
        torch._dynamo.reset()
        channels = 3
        parts = [
            torch.randn(channels, length, device=device, dtype=torch.float64, requires_grad=True) for length in (3, 7)
        ]
        references = [part.detach().clone().requires_grad_() for part in parts]
        weight = torch.randn(channels * multiplier, 1, kernel, device=device, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(channels * multiplier, device=device, dtype=torch.float64, requires_grad=True)
        reference_weight = weight.detach().clone().requires_grad_()
        reference_bias = bias.detach().clone().requires_grad_()

        def convolve(source, weight, bias):
            return F.conv1d(source, weight, bias, padding=padding, dilation=dilation, groups=channels)

        operation = convolve if backend is None else torch.compile(convolve, backend=backend, fullgraph=True)
        actual = operation(NT(parts, ragged_dims=(1,)), weight, bias)
        expected = [convolve(part, reference_weight, reference_bias) for part in references]
        torch.testing.assert_close(actual.unbind(), expected)
        probes = [torch.randn_like(part) for part in expected]
        torch.testing.assert_close(
            torch.autograd.grad(
                sum((part * probe).sum() for (part, probe) in zip(actual, probes)), (*parts, weight, bias)
            ),
            torch.autograd.grad(
                sum((part * probe).sum() for (part, probe) in zip(expected, probes)),
                (*references, reference_weight, reference_bias),
            ),
        )

    def test_boundary(self, device):
        parts = [torch.full((2, 2), float("nan"), device=device), torch.tensor([[2.0], [-3.0]], device=device)]
        weight = torch.ones(2, 1, 4, device=device)
        actual = F.conv1d(NT(parts, ragged_dims=(1,)), weight, padding=3, groups=2)[..., :-3]
        expected = [F.conv1d(part, weight, padding=3, groups=2)[..., :-3] for part in parts]
        torch.testing.assert_close(actual.unbind(), expected, equal_nan=True)


def _compare_conv1d_boundary(device, *, configuration, operand, nonfinite, grad_enabled, operation=None, shapes=None):
    channels, outputs, groups, stride, dilation = configuration
    if shapes is None:
        shapes = ((5,), (19,))
    parts = [torch.ones(channels, *shape, device=device, dtype=torch.float64) for shape in shapes]
    weight = torch.ones(outputs, channels // groups, 3, device=device, dtype=torch.float64)
    bias = -torch.ones(outputs, device=device, dtype=torch.float64)
    if operand == "input":
        parts[0][0, -1] = nonfinite
    elif operand == "weight":
        weight[:, :, 0] = nonfinite
    parts = [part.requires_grad_(grad_enabled) for part in parts]
    weight.requires_grad_(grad_enabled)
    bias.requires_grad_(grad_enabled)
    references = [part.detach().clone().requires_grad_(grad_enabled) for part in parts]
    reference_weight = weight.detach().clone().requires_grad_(grad_enabled)
    reference_bias = bias.detach().clone().requires_grad_(grad_enabled)
    kwargs = {"stride": stride, "dilation": dilation, "groups": groups}
    source = NT(parts, ragged_dims=(1,))
    with torch.set_grad_enabled(grad_enabled):
        actual = (
            F.conv1d(source, weight, bias, padding=dilation, **kwargs)
            if operation is None
            else operation(source, weight, bias)
        ).unbind()
        # Padding is part of the packed contract. Only nonexistent tile outputs
        # are omitted; true padded windows keep their nonfinite arithmetic.
        expected = [
            F.conv1d(F.pad(part, (dilation, dilation)), reference_weight, reference_bias, **kwargs)
            for part in references
        ]
        torch.testing.assert_close(actual, expected, equal_nan=True)
        if grad_enabled:
            probes = [torch.ones_like(part) for part in expected]
            if operand == "gradient":
                probes[0][0, 0] = nonfinite
            torch.testing.assert_close(
                torch.autograd.grad(actual, (*parts, weight, bias), probes),
                torch.autograd.grad(expected, (*references, reference_weight, reference_bias), probes),
                equal_nan=True,
            )


class TestConvolution:

    @pytest.mark.parametrize("configuration", [(2, 3, 1, 1, 1), (4, 6, 2, 1, 1), (2, 4, 2, 2, 2)])
    @pytest.mark.parametrize("operand", ["input", "weight"])
    @pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
    @pytest.mark.parametrize("grad_enabled", [False, True])
    def test_boundary_nonfinite(self, device, configuration, operand, nonfinite, grad_enabled):
        _compare_conv1d_boundary(
            device, configuration=configuration, operand=operand, nonfinite=nonfinite, grad_enabled=grad_enabled
        )

    @pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
    def test_boundary_nonfinite_cotangent(self, device, nonfinite):
        _compare_conv1d_boundary(
            device, configuration=(4, 6, 2, 2, 2), operand="gradient", nonfinite=nonfinite, grad_enabled=True
        )

    @pytest.mark.parametrize("operand", ["input", "weight"])
    def test_boundary_multiple_tiles(self, device, operand):
        _compare_conv1d_boundary(
            device,
            configuration=(2, 3, 1, 1, 1),
            operand=operand,
            nonfinite=float("nan"),
            grad_enabled=True,
            shapes=((5,), (1057,)),
        )

    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    @pytest.mark.parametrize("operand", ["input", "weight"])
    def test_boundary_dynamic(self, device, backend, operand):
        torch.compiler.reset()

        def convolve(source, weight, bias):
            return F.conv1d(source, weight, bias, stride=2, padding=2, dilation=2, groups=2)

        operation = torch.compile(convolve, backend=backend, fullgraph=True, dynamic=True)
        for shapes in (((5,), (17,)), ((9,), (13,))):
            _compare_conv1d_boundary(
                device,
                configuration=(4, 6, 2, 2, 2),
                operand=operand,
                nonfinite=float("nan"),
                grad_enabled=True,
                operation=operation,
                shapes=shapes,
            )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_boundary_autocast(self, device, dtype):
        if device.type == "cpu" and dtype == torch.float16:
            pytest.skip("CPU autocast uses bfloat16")
        parts = [torch.randn(4, *shape, device=device, requires_grad=True) for shape in ((5,), (19,))]
        references = [part.detach().clone().requires_grad_() for part in parts]
        weight = torch.randn(6, 2, 3, device=device, requires_grad=True)
        bias = torch.randn(6, device=device, requires_grad=True)
        reference_weight = weight.detach().clone().requires_grad_()
        reference_bias = bias.detach().clone().requires_grad_()
        kwargs = {"stride": 2, "dilation": 2, "groups": 2}
        with torch.autocast(device.type, dtype=dtype):
            actual = F.conv1d(NT(parts, ragged_dims=(1,)), weight, bias, padding=2, **kwargs).unbind()
            expected = [
                F.conv1d(F.pad(part, (2, 2)), reference_weight, reference_bias, **kwargs) for part in references
            ]
        assert actual[0].dtype == dtype
        tolerance = 0.04 if dtype == torch.bfloat16 else 0.005
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        probes = [torch.randn_like(part) for part in expected]
        torch.testing.assert_close(
            torch.autograd.grad(actual, (*parts, weight, bias), probes),
            torch.autograd.grad(expected, (*references, reference_weight, reference_bias), probes),
            atol=tolerance,
            rtol=tolerance,
        )

    def test_pointwise(self, device, tolerances):
        _compare_conv1d(device, kernel_size=1, padding=0, tolerances=tolerances)

    @pytest.mark.parametrize("stride", [1, 2, 4])
    def test_spatial(self, device, tolerances, stride):
        _compare_conv1d(device, kernel_size=3, padding=1, stride=stride, tolerances=tolerances)

    @pytest.mark.parametrize("backend", [None, "aot_eager", "inductor"])
    @pytest.mark.parametrize(("stride", "dilation", "kernel"), [(2, 1, 1), (1, 2, 2)])
    def test_unsampled_values(self, device, stride, dilation, kernel, backend):
        torch.compiler.reset()
        data = [1.0, float("nan"), 2.0, float("nan"), 3.0] if stride == 2 else [1.0, float("nan"), 3.0]
        parts = [torch.tensor([data], device=device, dtype=torch.float64, requires_grad=True) for _ in range(2)]
        references = [part.detach().clone().requires_grad_() for part in parts]
        weight = torch.ones(1, 1, kernel, device=device, dtype=torch.float64, requires_grad=True)
        reference_weight = weight.detach().clone().requires_grad_()
        kwargs = {"stride": stride, "dilation": dilation}

        def convolve(source, weight):
            return F.conv1d(source, weight, **kwargs)

        operation = (
            convolve if backend is None else torch.compile(convolve, backend=backend, fullgraph=True, dynamic=True)
        )
        actual = operation(NT(parts, ragged_dims=(1,)), weight)
        expected = [convolve(part, reference_weight) for part in references]
        torch.testing.assert_close(actual.unbind(), expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.values().sum(), (*parts, weight)),
            torch.autograd.grad(sum(part.sum() for part in expected), (*references, reference_weight)),
        )

    def test_bias(self, device):
        input = NT([torch.randn(3, 17, device=device), torch.randn(3, 29, device=device)])
        weight = torch.randn(5, 3, 3, device=device)
        bias = torch.randn(4, device=device)
        with pytest.raises(RuntimeError):
            F.conv1d(input, weight, bias, padding=1)

    @pytest.mark.parametrize("groups,stride", [(1, 2), (2, 1)])
    def test_autocast(self, device, groups, stride):
        if device.type != "cuda":
            pytest.skip("CUDA autocast")
        shapes = [(2, 9), (2, 7)]
        parts = [torch.randn(shape, device=device, requires_grad=True) for shape in shapes]
        references = [part.detach().clone().requires_grad_() for part in parts]
        channels = 3 if groups == 1 else 4
        weight = torch.randn(channels, 2 // groups, 3, device=device, requires_grad=True)
        bias = torch.randn(channels, device=device, requires_grad=True)
        reference_weight = weight.detach().clone().requires_grad_()
        reference_bias = bias.detach().clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = F.conv1d(NT(parts), weight, bias, stride=stride, padding=1, groups=groups)
            expected = [
                F.conv1d(part, reference_weight, reference_bias, stride=stride, padding=1, groups=groups)
                for part in references
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
            return F.conv1d(source, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

        operation = (
            convolve if backend is None else torch.compile(convolve, backend=backend, fullgraph=True, dynamic=True)
        )
        for shapes in (((5,), (19,)), ((9,), (34,))):
            parts = [torch.randn(4, *shape, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
            references = [part.detach().clone().requires_grad_() for part in parts]
            weight = torch.randn(6, 4 // groups, 3, device=device, dtype=torch.float64, requires_grad=True)
            bias = torch.randn(6, device=device, dtype=torch.float64, requires_grad=True)
            reference_weight = weight.detach().clone().requires_grad_()
            reference_bias = bias.detach().clone().requires_grad_()
            source = NT(parts, ragged_dims=(1,))
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

    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    def test_fullgraph(self, backend):
        parts = [torch.randn(shape, dtype=torch.float64, requires_grad=True) for shape in [(2, 3), (2, 5)]]
        weight = torch.randn(3, 2, 1, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(3, dtype=torch.float64, requires_grad=True)
        expected = [F.conv1d(part, weight, bias) for part in parts]
        expected_gradients = torch.autograd.grad(sum(part.square().sum() for part in expected), (*parts, weight, bias))
        actual = torch.compile(F.conv1d, backend=backend, fullgraph=True)(NT(parts), weight, bias)
        torch.testing.assert_close(actual.unbind(), expected)
        actual_gradients = torch.autograd.grad(actual.values().square().sum(), (*parts, weight, bias))
        torch.testing.assert_close(actual_gradients, expected_gradients)
