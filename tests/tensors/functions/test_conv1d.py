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

from contextlib import nullcontext

import pytest
import torch
from torch.nn import functional as F

from danling.tensors import NestedTensor as NT
from danling.tensors.ops import nested_execution_guard
from tests.tensors.utils import assert_close, low_precision_cuda_tolerances


def _tolerances(device):
    if device.type != "cuda":
        return (1e-5, 1e-5)
    return low_precision_cuda_tolerances(
        device,
        torch.float32,
        default=(5e-3, 5e-3),
        fp16=(5e-3, 5e-3),
        bf16=(5e-2, 5e-2),
    )


def _assert_input_grads_match(inputs, reference_inputs, *, atol, rtol):
    for input_tensor, reference_tensor in zip(inputs, reference_inputs):
        assert_close(input_tensor.grad, reference_tensor.grad, atol=atol, rtol=rtol)


def _spatial_tile_context(device):
    if device.type != "cuda":
        return nullcontext()
    pytest.importorskip("triton")
    return nested_execution_guard(forbid_storage_map=True)


def _compare_conv1d(device, *, kernel_size: int, padding: int, require_spatial_tile=False):
    inputs = [
        torch.randn(3, 17, device=device, requires_grad=True),
        torch.randn(3, 29, device=device, requires_grad=True),
    ]
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    input = NT(inputs)
    weight = torch.randn(5, 3, kernel_size, device=device, requires_grad=True)
    bias = torch.randn(5, device=device, requires_grad=True)
    reference_weight = weight.detach().clone().requires_grad_()
    reference_bias = bias.detach().clone().requires_grad_()

    with _spatial_tile_context(device) if require_spatial_tile else nullcontext():
        output = F.conv1d(input, weight, bias, padding=padding)
    reference_input = NT(reference_inputs)
    reference_dense = F.conv1d(reference_input.tensor, reference_weight, reference_bias, padding=padding)
    reference = output.nested_like(reference_dense)

    atol, rtol = _tolerances(device)
    assert_close(output, reference_dense, atol=atol, rtol=rtol)
    with _spatial_tile_context(device) if require_spatial_tile else nullcontext():
        output.sum().backward()
    reference.sum().backward()
    _assert_input_grads_match(inputs, reference_inputs, atol=atol, rtol=rtol)
    assert_close(weight.grad, reference_weight.grad, atol=atol, rtol=rtol)
    assert_close(bias.grad, reference_bias.grad, atol=atol, rtol=rtol)


def test_pointwise_conv1d(device):
    _compare_conv1d(device, kernel_size=1, padding=0)


def test_spatial_conv1d(device):
    _compare_conv1d(device, kernel_size=3, padding=1, require_spatial_tile=device.type == "cuda")


def test_conv1d_invalid_bias_shape_matches_torch(device):
    input = NT(
        [
            torch.randn(3, 17, device=device),
            torch.randn(3, 29, device=device),
        ]
    )
    weight = torch.randn(5, 3, 3, device=device)
    bias = torch.randn(4, device=device)

    with pytest.raises(RuntimeError):
        F.conv1d(input, weight, bias, padding=1)


@pytest.mark.parametrize(("rank", "backend"), [(1, "aot_eager"), (2, "aot_eager"), (1, "inductor")])
def test_compiled_pointwise_output_preserves_parameter_gradients(rank, backend):
    shapes = [(2, 3), (2, 5)] if rank == 1 else [(2, 3, 4), (2, 4, 6)]
    parts = [torch.randn(shape, dtype=torch.float64, requires_grad=True) for shape in shapes]
    weight = torch.randn(3, 2, *([1] * rank), dtype=torch.float64, requires_grad=True)
    bias = torch.randn(3, dtype=torch.float64, requires_grad=True)
    fn = F.conv1d if rank == 1 else F.conv2d
    expected = [fn(part, weight, bias) for part in parts]
    expected_grads = torch.autograd.grad(sum(part.square().sum() for part in expected), (*parts, weight, bias))
    compiled = torch.compile(fn, backend=backend, fullgraph=True)
    output = compiled(NT(parts), weight, bias)
    actual_grads = torch.autograd.grad(output.concat.square().sum(), (*parts, weight, bias))
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize(("rank", "transpose"), [(1, False), (2, False), (1, True), (2, True)])
def test_spatial_convolution_autocast_matches_native_vjps(device, rank, transpose):
    if device.type != "cuda":
        pytest.skip("spatial tile autocast requires CUDA")
    shapes = [(2, 9), (2, 7)] if rank == 1 else [(2, 6, 7), (2, 5, 9)]
    parts = [torch.randn(shape, device=device, requires_grad=True) for shape in shapes]
    references = [part.detach().clone().requires_grad_() for part in parts]
    channels = (2, 3) if transpose else (3, 2)
    weight = torch.randn(*channels, *([3] * rank), device=device, requires_grad=True)
    bias = torch.randn(3, device=device, requires_grad=True)
    reference_weight, reference_bias = weight.detach().clone().requires_grad_(), bias.detach().clone().requires_grad_()
    fn = getattr(F, f"conv_transpose{rank}d" if transpose else f"conv{rank}d")
    kwargs = {"stride": 2, "padding": 1}
    if transpose:
        kwargs["output_padding"] = 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = fn(NT(parts), weight, bias, **kwargs)
        expected = [fn(part, reference_weight, reference_bias, **kwargs) for part in references]
    assert actual.dtype == torch.bfloat16
    probes = [torch.randn_like(part) for part in expected]
    actual_grads = torch.autograd.grad(
        sum((part * probe).sum() for part, probe in zip(actual, probes)), (*parts, weight, bias)
    )
    expected_grads = torch.autograd.grad(
        sum((part * probe).sum() for part, probe in zip(expected, probes)),
        (*references, reference_weight, reference_bias),
    )
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, atol=0.02, rtol=0.02)
    for result, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(result, reference, atol=0.03, rtol=0.03)
