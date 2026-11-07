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


def _assert_vjp(actual, expected, parts, references):
    torch.testing.assert_close(actual.unbind(), expected)
    probes = [torch.randn_like(part) for part in expected]
    actual_gradients = torch.autograd.grad(sum((part * probe).sum() for part, probe in zip(actual, probes)), parts)
    expected_gradients = torch.autograd.grad(
        sum((part * probe).sum() for part, probe in zip(expected, probes)), references
    )
    torch.testing.assert_close(actual_gradients, expected_gradients)


class TestChannel:
    def test_channels(self, device):
        value = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4, 1)
        actual = F.channel_shuffle(NT([value]), groups=2)
        expected = F.channel_shuffle(value.unsqueeze(0), groups=2).squeeze(0)
        torch.testing.assert_close(actual[0], expected)

    @pytest.mark.parametrize("shapes", [[(4, 2, 3), (6, 2, 7)], [(4, 6, 5), (4, 4, 3)]])
    def test_shuffle(self, device, shapes):
        parts = [torch.randn(shape, device=device, requires_grad=True) for shape in shapes]
        references = [part.detach().clone().requires_grad_() for part in parts]
        actual = F.channel_shuffle(NT(parts), groups=2)
        expected = [F.channel_shuffle(part.unsqueeze(0), groups=2).squeeze(0) for part in references]
        _assert_vjp(actual, expected, parts, references)

    def test_strided_gradient(self, device):
        parts = [torch.randn(4, 2, 3, device=device, requires_grad=True) for _ in range(2)]
        references = [part.detach().clone().requires_grad_() for part in parts]
        cotangent = torch.randn(8, 3, 2, device=device).transpose(1, 2)
        actual = F.channel_shuffle(NT(parts), groups=2)
        gradients = torch.autograd.grad(actual.concat, parts, cotangent)
        expected = [F.channel_shuffle(part.unsqueeze(0), groups=2).squeeze(0) for part in references]
        reference_gradients = [
            torch.autograd.grad(part, reference, probe.contiguous())[0]
            for part, reference, probe in zip(expected, references, cotangent.split(4))
        ]
        torch.testing.assert_close(actual.unbind(), expected)
        torch.testing.assert_close(gradients, reference_gradients)

    @pytest.mark.parametrize("operation", ["shuffle", "unshuffle"])
    def test_pixels(self, device, operation):
        shapes = [(4, 2, 3), (4, 5, 7)] if operation == "shuffle" else [(4, 4, 6), (4, 6, 8)]
        parts = [torch.randn(shape, device=device, requires_grad=True) for shape in shapes]
        references = [part.detach().clone().requires_grad_() for part in parts]
        if operation == "shuffle":
            actual = F.pixel_shuffle(NT(parts), upscale_factor=2)
            expected = [F.pixel_shuffle(part, upscale_factor=2) for part in references]
        else:
            actual = F.pixel_unshuffle(NT(parts), downscale_factor=2)
            expected = [F.pixel_unshuffle(part, downscale_factor=2) for part in references]
        _assert_vjp(actual, expected, parts, references)

    def test_roundtrip(self, device):
        parts = [torch.randn(4, 2, 3, device=device), torch.randn(4, 5, 7, device=device)]
        restored = F.pixel_unshuffle(F.pixel_shuffle(NT(parts), upscale_factor=2), downscale_factor=2)
        torch.testing.assert_close(restored.unbind(), parts)
