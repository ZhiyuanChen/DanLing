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


class TestPooling:

    def test_average(self, device, check_pool):
        inputs = [
            torch.randn(3, 5, 7, 9, device=device, requires_grad=True),
            torch.randn(3, 3, 5, 7, device=device, requires_grad=True),
        ]
        check_pool(inputs, F.avg_pool3d, kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def test_maximum(self, device, check_pool):
        inputs = [
            torch.randn(3, 5, 7, 9, device=device, requires_grad=True),
            torch.randn(3, 3, 5, 7, device=device, requires_grad=True),
        ]
        check_pool(inputs, F.max_pool3d, kernel_size=3, stride=2, padding=1)

    @pytest.mark.parametrize("value", [float("-inf"), float("nan")])
    def test_maximum_empty_window_gradient(self, device, value):
        if device.type != "cuda":
            pytest.skip("packed max pool kernels require CUDA")
        first = torch.arange(64, device=device, dtype=torch.float32).reshape(1, 4, 4, 4).requires_grad_()
        empty_window = torch.ones(1, 2, 2, 2, device=device, requires_grad=True)
        nonfinite = torch.full((1, 3, 3, 3), value, device=device, requires_grad=True)
        reference = first.detach().clone().requires_grad_()
        kwargs = {"kernel_size": 2, "stride": 1, "padding": 1, "dilation": 3}
        output = F.max_pool3d(NT([first, empty_window, nonfinite], ragged_dims=(1, 2, 3)), **kwargs)
        parts = output.unbind()
        torch.testing.assert_close(parts[0], F.max_pool3d(reference, **kwargs))
        assert torch.isneginf(parts[1]).all()
        torch.testing.assert_close(parts[2], torch.full_like(parts[2], value), equal_nan=True)
        actual = torch.autograd.grad(output.concat, (first, empty_window, nonfinite), torch.ones_like(output.concat))
        expected_first = torch.autograd.grad(F.max_pool3d(reference, **kwargs).sum(), reference)[0]
        expected_nonfinite = torch.zeros_like(nonfinite)
        expected_nonfinite[0, ::2, ::2, ::2] = 1
        torch.testing.assert_close(actual, (expected_first, torch.zeros_like(empty_window), expected_nonfinite))

    def test_divisor(self, device):
        input = NT([torch.randn(3, 5, 7, 9, device=device), torch.randn(3, 3, 5, 7, device=device)])
        with pytest.raises(RuntimeError, match="divisor"):
            F.avg_pool3d(input, kernel_size=3, divisor_override=0)

    def test_indices(self, device, check_pool):
        inputs = [
            torch.randn(3, 5, 7, 9, device=device, requires_grad=True),
            torch.randn(3, 3, 5, 7, device=device, requires_grad=True),
        ]
        check_pool(inputs, F.max_pool3d, kernel_size=2, stride=2, return_indices=True)

    @pytest.mark.parametrize("value", [float("-inf"), float("nan")])
    def test_nonfinite(self, device, value):
        if device.type != "cuda":
            pytest.skip("packed max pool kernels require CUDA")
        shapes = [(1, *[2] * (3 - 1), length) for length in (4, 6)]
        parts = [torch.full(shape, value, device=device, requires_grad=True) for shape in shapes]
        references = [part.detach().clone().requires_grad_() for part in parts]
        fn = getattr(F, f"max_pool{3}d")
        actual = fn(NT(parts), 2)
        expected = [fn(part, 2) for part in references]
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, equal_nan=True)
        actual_grads = torch.autograd.grad(actual.concat.sum(), parts)
        expected_grads = torch.autograd.grad(sum(part.sum() for part in expected), references)
        for result, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(result, reference)

    def test_float64(self, device):
        if device.type != "cuda":
            pytest.skip("CUDA kernel dtype routing")
        shapes = [(2, *[3] * (3 - 1), length) for length in (4, 6)]
        parts = [torch.randn(shape, dtype=torch.float64, device=device, requires_grad=True) for shape in shapes]
        fn = getattr(F, f"avg_pool{3}d")
        actual = fn(NT(parts), 3, stride=1, padding=1, count_include_pad=False)
        expected = [fn(part, 3, stride=1, padding=1, count_include_pad=False) for part in parts]
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference)
        actual_grads = torch.autograd.grad(actual.concat.square().sum(), parts, retain_graph=True)
        expected_grads = torch.autograd.grad(sum(part.square().sum() for part in expected), parts)
        for result, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(result, reference)
