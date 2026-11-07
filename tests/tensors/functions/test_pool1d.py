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
            torch.randn(3, 17, device=device, requires_grad=True),
            torch.randn(3, 29, device=device, requires_grad=True),
        ]
        check_pool(inputs, F.avg_pool1d, kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def test_maximum(self, device, check_pool):
        inputs = [
            torch.randn(3, 17, device=device, requires_grad=True),
            torch.randn(3, 29, device=device, requires_grad=True),
        ]
        check_pool(inputs, F.max_pool1d, kernel_size=3, stride=2, padding=1, dilation=2)

    def test_indices(self, device, check_pool):
        inputs = [
            torch.randn(3, 17, device=device, requires_grad=True),
            torch.randn(3, 29, device=device, requires_grad=True),
        ]
        check_pool(inputs, F.max_pool1d, kernel_size=2, stride=2, return_indices=True)

    @pytest.mark.parametrize("value", [float("-inf"), float("nan")])
    def test_nonfinite(self, device, value):
        if device.type != "cuda":
            pytest.skip("packed max pool kernels require CUDA")
        shapes = [(1, *[2] * (1 - 1), length) for length in (4, 6)]
        parts = [torch.full(shape, value, device=device, requires_grad=True) for shape in shapes]
        references = [part.detach().clone().requires_grad_() for part in parts]
        fn = getattr(F, f"max_pool{1}d")
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
        shapes = [(2, *[3] * (1 - 1), length) for length in (4, 6)]
        parts = [torch.randn(shape, dtype=torch.float64, device=device, requires_grad=True) for shape in shapes]
        fn = getattr(F, f"avg_pool{1}d")
        actual = fn(NT(parts), 3, stride=1, padding=1, count_include_pad=False)
        expected = [fn(part, 3, stride=1, padding=1, count_include_pad=False) for part in parts]
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference)
        actual_grads = torch.autograd.grad(actual.concat.square().sum(), parts, retain_graph=True)
        expected_grads = torch.autograd.grad(sum(part.square().sum() for part in expected), parts)
        for result, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(result, reference)
