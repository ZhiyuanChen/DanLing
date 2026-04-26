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


def _tolerances(device, dtype=torch.float32):
    if dtype == torch.float64:
        return (1e-8, 1e-8)
    if device.type != "cuda":
        return (1e-5, 1e-5)
    return low_precision_cuda_tolerances(
        device,
        dtype,
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


def _compare_conv_transpose2d(
    device,
    dtype,
    *,
    kernel_size,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    require_spatial_tile=False,
):
    inputs = [
        torch.randn(3, 11, 13, device=device, dtype=dtype, requires_grad=True),
        torch.randn(3, 7, 9, device=device, dtype=dtype, requires_grad=True),
    ]
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    input = NT(inputs)
    weight = torch.randn(3, 5, *kernel_size, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(5, device=device, dtype=dtype, requires_grad=True)
    reference_weight = weight.detach().clone().requires_grad_()
    reference_bias = bias.detach().clone().requires_grad_()

    kwargs = {
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "dilation": dilation,
    }
    with _spatial_tile_context(device) if require_spatial_tile else nullcontext():
        output = F.conv_transpose2d(input, weight, bias, **kwargs)
    reference = NT([F.conv_transpose2d(t, reference_weight, reference_bias, **kwargs) for t in reference_inputs])
    dense_reference = F.conv_transpose2d(
        NT([tensor.detach() for tensor in inputs]).tensor,
        weight.detach(),
        bias.detach(),
        **kwargs,
    )

    atol, rtol = _tolerances(device, dtype)
    assert_close(output, dense_reference, atol=atol, rtol=rtol)
    assert_close(output, reference, atol=atol, rtol=rtol)
    grad_output = NT([torch.randn_like(t) for t in reference])
    with _spatial_tile_context(device) if require_spatial_tile else nullcontext():
        (output * grad_output).sum().backward()
    (reference * grad_output).sum().backward()
    _assert_input_grads_match(inputs, reference_inputs, atol=atol, rtol=rtol)
    assert_close(weight.grad, reference_weight.grad, atol=atol, rtol=rtol)
    assert_close(bias.grad, reference_bias.grad, atol=atol, rtol=rtol)


def test_pointwise_conv_transpose2d(device):
    _compare_conv_transpose2d(device, torch.float32, kernel_size=(1, 1))


def test_spatial_conv_transpose2d(device, float_dtype):
    _compare_conv_transpose2d(
        device,
        float_dtype,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        output_padding=(1, 1),
        require_spatial_tile=device.type == "cuda",
    )


def test_spatial_conv_transpose2d_output_padding_extends_tail(device):
    _compare_conv_transpose2d(
        device,
        torch.float32,
        kernel_size=(2, 3),
        stride=(3, 2),
        padding=(0, 0),
        output_padding=(2, 1),
        dilation=(2, 1),
        require_spatial_tile=device.type == "cuda",
    )
