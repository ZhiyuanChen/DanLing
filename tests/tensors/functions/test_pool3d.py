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
from tests.tensors.utils import assert_close


def _assert_input_grads_match(inputs, references):
    for input_tensor, reference_tensor in zip(inputs, references):
        assert_close(input_tensor.grad, reference_tensor.grad)


def _reference_pool(inputs, pool_fn, **kwargs):
    outputs = [pool_fn(t, **kwargs) for t in inputs]
    if kwargs.get("return_indices", False):
        values, indices = zip(*outputs)
        return NT(values), NT(indices)
    return NT(outputs)


def _packed_pool_context(device, return_indices):
    if device.type != "cuda" or return_indices:
        return nullcontext()
    pytest.importorskip("triton")
    return nested_execution_guard(forbid_storage_map=True)


def _compare_pool(inputs, pool_fn, *, require_packed=False, **kwargs):
    references = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    input = NT(inputs)

    return_indices = kwargs.get("return_indices", False)
    with _packed_pool_context(input.device, return_indices) if require_packed else nullcontext():
        output = pool_fn(input, **kwargs)
    reference = _reference_pool(references, pool_fn, **kwargs)

    if kwargs.get("return_indices", False):
        output_values, output_indices = output
        reference_values, reference_indices = reference
        assert_close(output_values, reference_values)
        assert_close(output_indices, reference_indices)
        output_values.sum().backward()
        reference_values.sum().backward()
    else:
        assert_close(output, reference)
        with _packed_pool_context(input.device, return_indices) if require_packed else nullcontext():
            output.sum().backward()
        reference.sum().backward()
    _assert_input_grads_match(inputs, references)


def test_avg_pool3d(device):
    inputs = [
        torch.randn(3, 5, 7, 9, device=device, requires_grad=True),
        torch.randn(3, 3, 5, 7, device=device, requires_grad=True),
    ]

    _compare_pool(
        inputs,
        F.avg_pool3d,
        kernel_size=3,
        stride=2,
        padding=1,
        count_include_pad=False,
        require_packed=device.type == "cuda",
    )


def test_max_pool3d(device):
    inputs = [
        torch.randn(3, 5, 7, 9, device=device, requires_grad=True),
        torch.randn(3, 3, 5, 7, device=device, requires_grad=True),
    ]

    _compare_pool(inputs, F.max_pool3d, kernel_size=3, stride=2, padding=1, require_packed=device.type == "cuda")


def test_avg_pool3d_divisor_override_zero_matches_torch(device):
    input = NT(
        [
            torch.randn(3, 5, 7, 9, device=device),
            torch.randn(3, 3, 5, 7, device=device),
        ]
    )

    with pytest.raises(RuntimeError, match="divisor"):
        F.avg_pool3d(input, kernel_size=3, divisor_override=0)


def test_max_pool3d_return_indices(device):
    inputs = [
        torch.randn(3, 5, 7, 9, device=device, requires_grad=True),
        torch.randn(3, 3, 5, 7, device=device, requires_grad=True),
    ]

    _compare_pool(inputs, F.max_pool3d, kernel_size=2, stride=2, return_indices=True)
