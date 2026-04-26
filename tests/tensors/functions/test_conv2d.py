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


def _compare_conv2d(device, *, kernel_size: int, padding: int, require_spatial_tile=False):
    inputs = [
        torch.randn(3, 11, 13, device=device, requires_grad=True),
        torch.randn(3, 7, 9, device=device, requires_grad=True),
    ]
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    input = NT(inputs)
    weight = torch.randn(5, 3, kernel_size, kernel_size, device=device, requires_grad=True)
    bias = torch.randn(5, device=device, requires_grad=True)
    reference_weight = weight.detach().clone().requires_grad_()
    reference_bias = bias.detach().clone().requires_grad_()

    with _spatial_tile_context(device) if require_spatial_tile else nullcontext():
        output = F.conv2d(input, weight, bias, padding=padding)
    reference_input = NT(reference_inputs)
    reference_dense = F.conv2d(reference_input.tensor, reference_weight, reference_bias, padding=padding)
    reference = output.nested_like(reference_dense)

    atol, rtol = _tolerances(device)
    assert_close(output, reference_dense, atol=atol, rtol=rtol)
    with _spatial_tile_context(device) if require_spatial_tile else nullcontext():
        output.sum().backward()
    reference.sum().backward()
    _assert_input_grads_match(inputs, reference_inputs, atol=atol, rtol=rtol)
    assert_close(weight.grad, reference_weight.grad, atol=atol, rtol=rtol)
    assert_close(bias.grad, reference_bias.grad, atol=atol, rtol=rtol)


def test_pointwise_conv2d(device):
    _compare_conv2d(device, kernel_size=1, padding=0)


def test_spatial_conv2d(device):
    _compare_conv2d(device, kernel_size=3, padding=1, require_spatial_tile=device.type == "cuda")


def test_conv2d_invalid_bias_shape_matches_torch(device):
    input = NT(
        [
            torch.randn(3, 11, 13, device=device),
            torch.randn(3, 7, 9, device=device),
        ]
    )
    weight = torch.randn(5, 3, 3, 3, device=device)
    bias = torch.randn(4, device=device)

    with pytest.raises(RuntimeError):
        F.conv2d(input, weight, bias, padding=1)
