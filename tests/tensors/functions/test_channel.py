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
from danling.tensors.ops import nested_execution_guard
from tests.tensors.utils import assert_close


def test_channel_shuffle_uses_unbatched_sample_semantics(device):
    tensor = torch.arange(4 * 4, device=device, dtype=torch.float32).reshape(4, 4, 1)
    input = NT([tensor])

    output = F.channel_shuffle(input, groups=2)
    per_sample = F.channel_shuffle(tensor, groups=2)
    batched_channel = F.channel_shuffle(tensor.unsqueeze(0), groups=2).squeeze(0)

    assert_close(output[0], per_sample)
    assert not torch.equal(output[0], batched_channel)


def test_channel_shuffle(device):
    tensors = [
        torch.randn(4, 2, 3, device=device),
        torch.randn(5, 2, 7, device=device),
    ]
    input = NT(tensors)

    output = F.channel_shuffle(input, groups=2)
    reference = NT([F.channel_shuffle(t, groups=2) for t in tensors])

    assert_close(output, reference)


def test_channel_shuffle_packed_chw_fast_path(device):
    if torch.device(device).type != "cuda":
        pytest.skip("packed CHW channel_shuffle fast path is CUDA-only")
    tensors = [
        torch.randn(4, 6, 5, device=device),
        torch.randn(4, 4, 3, device=device),
    ]
    input = NT(tensors)

    assert input._permutation == (1, 2, 0)
    with nested_execution_guard(forbid_storage_map=True):
        output = F.channel_shuffle(input, groups=2)
    reference = NT([F.channel_shuffle(t, groups=2) for t in tensors], **input._meta())

    assert_close(output, reference)


def test_channel_shuffle_packed_chw_backward(device):
    if torch.device(device).type != "cuda":
        pytest.skip("packed CHW channel_shuffle fast path is CUDA-only")
    tensors = [
        torch.randn(4, 6, 5, device=device, requires_grad=True),
        torch.randn(4, 4, 3, device=device, requires_grad=True),
    ]
    reference_tensors = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    input = NT(tensors)
    reference_input = NT(reference_tensors)

    output = F.channel_shuffle(input, groups=2)
    reference = NT([F.channel_shuffle(t, groups=2) for t in reference_input], **input._meta())

    assert_close(output, reference)
    output._values.square().sum().backward()
    reference._values.square().sum().backward()
    for tensor, reference_tensor in zip(tensors, reference_tensors):
        assert_close(tensor.grad, reference_tensor.grad)


def test_pixel_shuffle(device):
    tensors = [
        torch.randn(4, 2, 3, device=device),
        torch.randn(4, 5, 7, device=device),
    ]
    input = NT(tensors)

    output = F.pixel_shuffle(input, upscale_factor=2)
    reference = F.pixel_shuffle(input.tensor, upscale_factor=2)

    assert_close(output, reference)


def test_pixel_shuffle_backward(device):
    tensors = [
        torch.randn(4, 2, 3, device=device, requires_grad=True),
        torch.randn(4, 5, 7, device=device, requires_grad=True),
    ]
    reference_tensors = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    input = NT(tensors)
    reference_input = NT(reference_tensors)

    output = F.pixel_shuffle(input, upscale_factor=2)
    reference = NT([F.pixel_shuffle(t, upscale_factor=2) for t in reference_input], **input._meta())

    assert_close(output, reference)
    output._values.square().sum().backward()
    reference._values.square().sum().backward()
    for tensor, reference_tensor in zip(tensors, reference_tensors):
        assert_close(tensor.grad, reference_tensor.grad)


def test_pixel_shuffle_round_trip(device):
    input = NT(
        [
            torch.randn(4, 2, 3, device=device),
            torch.randn(4, 5, 7, device=device),
        ]
    )

    output = F.pixel_unshuffle(F.pixel_shuffle(input, upscale_factor=2), downscale_factor=2)

    assert_close(output, input)


def test_pixel_unshuffle_backward(device):
    tensors = [
        torch.randn(2, 4, 6, device=device, requires_grad=True),
        torch.randn(2, 8, 10, device=device, requires_grad=True),
    ]
    reference_tensors = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    input = NT(tensors)
    reference_input = NT(reference_tensors)

    output = F.pixel_unshuffle(input, downscale_factor=2)
    reference = NT([F.pixel_unshuffle(t, downscale_factor=2) for t in reference_input], **input._meta())

    assert_close(output, reference)
    output._values.square().sum().backward()
    reference._values.square().sum().backward()
    for tensor, reference_tensor in zip(tensors, reference_tensors):
        assert_close(tensor.grad, reference_tensor.grad)


@pytest.mark.parametrize("op", ["channel_shuffle", "pixel_shuffle", "pixel_unshuffle"])
def test_packed_channel_op_sum_backward(device, op):
    if torch.device(device).type != "cuda":
        pytest.skip("packed channel op fast paths are CUDA-only")
    if op == "channel_shuffle":
        tensors = [
            torch.randn(4, 8, 5, device=device, requires_grad=True),
            torch.randn(4, 4, 3, device=device, requires_grad=True),
        ]
    else:
        tensors = [
            torch.randn(4, 8, 6, device=device, requires_grad=True),
            torch.randn(4, 4, 4, device=device, requires_grad=True),
        ]
    reference_tensors = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    input = NT(tensors)
    reference_input = NT(reference_tensors)

    if op == "channel_shuffle":
        with nested_execution_guard(forbid_storage_map=True):
            output = F.channel_shuffle(input, groups=2)
        reference = NT([F.channel_shuffle(t, groups=2) for t in reference_input], **input._meta())
    elif op == "pixel_shuffle":
        with nested_execution_guard(forbid_storage_map=True):
            output = F.pixel_shuffle(input, upscale_factor=2)
        reference = NT([F.pixel_shuffle(t, upscale_factor=2) for t in reference_input], **input._meta())
    else:
        with nested_execution_guard(forbid_storage_map=True):
            output = F.pixel_unshuffle(input, downscale_factor=2)
        reference = NT([F.pixel_unshuffle(t, downscale_factor=2) for t in reference_input], **input._meta())

    assert_close(output, reference)
    output.sum().backward()
    reference.sum().backward()
    for tensor, reference_tensor in zip(tensors, reference_tensors):
        assert_close(tensor.grad, reference_tensor.grad)
