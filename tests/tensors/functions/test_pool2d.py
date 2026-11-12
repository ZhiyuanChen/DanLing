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

from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from danling.tensors import NestedTensor as NT


class TestPooling:

    def test_average(self, device, check_pool):
        inputs = [
            torch.randn(3, 11, 13, device=device, requires_grad=True),
            torch.randn(3, 7, 9, device=device, requires_grad=True),
        ]
        check_pool(
            inputs, F.avg_pool2d, kernel_size=3, stride=2, padding=1, count_include_pad=False, divisor_override=9
        )

    def test_maximum(self, device, check_pool):
        inputs = [
            torch.randn(3, 11, 13, device=device, requires_grad=True),
            torch.randn(3, 7, 9, device=device, requires_grad=True),
        ]
        check_pool(inputs, F.max_pool2d, kernel_size=3, stride=2, padding=1, dilation=2)

    @pytest.mark.parametrize("backend", [None, "aot_eager", "inductor"])
    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_empty_input(self, device, backend, axis):
        if backend is not None and device.type != "cuda":
            pytest.skip("compiled packed max pool requires CUDA")
        shapes = [[1, 3, 3], [1, 5, 4]]
        shapes[0][axis] = 0
        if axis == 0:
            shapes[1][axis] = 0
        parts = [torch.empty(shape, device=device) for shape in shapes]

        def pool(input):
            return F.max_pool2d(input, 2, stride=1, padding=1)

        with pytest.raises(RuntimeError):
            pool(parts[0])
        operation = pool if backend is None else torch.compile(pool, backend=backend, fullgraph=True, dynamic=True)
        with pytest.raises(RuntimeError):
            operation(NT(parts, ragged_dims=(1, 2)))

    @pytest.mark.parametrize("value", [float("-inf"), float("nan")])
    @pytest.mark.parametrize("compiled", [False, True])
    def test_maximum_empty_window_gradient(self, device, value, compiled):
        if device.type != "cuda":
            pytest.skip("packed max pool kernels require CUDA")
        first = torch.arange(16, device=device, dtype=torch.float32).reshape(1, 4, 4).requires_grad_()
        empty_window = torch.ones(1, 2, 2, device=device, requires_grad=True)
        nonfinite = torch.full((1, 3, 3), value, device=device, requires_grad=True)
        reference = first.detach().clone().requires_grad_()
        kwargs = {"kernel_size": 2, "stride": 1, "padding": 1, "dilation": 3}

        def operation(input):
            return F.max_pool2d(input, **kwargs)

        if compiled:
            operation = torch.compile(operation, fullgraph=True, dynamic=True)
        output = operation(NT([first, empty_window, nonfinite], ragged_dims=(1, 2)))
        parts = output.unbind()
        torch.testing.assert_close(parts[0], F.max_pool2d(reference, **kwargs))
        assert torch.isneginf(parts[1]).all()
        torch.testing.assert_close(parts[2], torch.full_like(parts[2], value), equal_nan=True)
        actual = torch.autograd.grad(output.concat, (first, empty_window, nonfinite), torch.ones_like(output.concat))
        expected_first = torch.autograd.grad(F.max_pool2d(reference, **kwargs).sum(), reference)[0]
        expected_nonfinite = torch.zeros_like(nonfinite)
        expected_nonfinite[0, ::2, ::2] = 1
        torch.testing.assert_close(actual, (expected_first, torch.zeros_like(empty_window), expected_nonfinite))

    def test_divisor(self, device):
        input = NT([torch.randn(3, 11, 13, device=device), torch.randn(3, 7, 9, device=device)])
        with pytest.raises(RuntimeError, match="divisor"):
            F.avg_pool2d(input, kernel_size=3, divisor_override=0)

    def test_indices(self, device, check_pool):
        inputs = [
            torch.randn(3, 11, 13, device=device, requires_grad=True),
            torch.randn(3, 7, 9, device=device, requires_grad=True),
        ]
        check_pool(inputs, F.max_pool2d, kernel_size=2, stride=2, return_indices=True)

    @pytest.mark.parametrize("pool", ("avg", "max"))
    @pytest.mark.parametrize("layout", ("transposed", "sliced", "expanded_rows", "expanded_channels"))
    def test_layout(self, device, float_dtype, pool, layout):
        shapes = ((3, 5, 7), (3, 6, 4))
        counts = tuple(height * width for (_, height, width) in shapes)
        rows, channels = (sum(counts), shapes[0][0])
        base_shapes = {
            "transposed": (channels, rows),
            "sliced": (2 * rows + 1, 2 * channels + 1),
            "expanded_rows": (1, channels),
            "expanded_channels": (rows, 1),
        }
        base = torch.randn(base_shapes[layout], device=device, dtype=float_dtype, requires_grad=True)
        reference_base = base.detach().clone().requires_grad_()

        def packed_view(value):
            if layout == "transposed":
                return value.t()
            if layout == "sliced":
                return value[1::2, 1::2]
            return value.expand(rows, channels)

        template = NT([torch.empty(shape, device=device, dtype=float_dtype) for shape in shapes], ragged_dims=(1, 2))
        nested = template.packed_like(packed_view(base))
        references = tuple(
            chunk.reshape(height, width, channels).permute(2, 0, 1)
            for (chunk, (_, height, width)) in zip(packed_view(reference_base).split(counts), shapes)
        )
        function = F.avg_pool2d if pool == "avg" else F.max_pool2d
        kwargs = {"kernel_size": 3, "stride": 2, "padding": 1}
        if pool == "avg":
            kwargs["count_include_pad"] = False
        output = function(nested, **kwargs)
        expected = tuple(function(value, **kwargs) for value in references)
        assert output.element_sizes().tolist() == [list(value.shape) for value in expected]
        assert output.ragged_dims == (1, 2)
        torch.testing.assert_close(tuple(output.unbind()), expected)
        weights = tuple(
            torch.linspace(0.25, 1.25, value.numel(), device=device, dtype=float_dtype).reshape_as(value)
            for value in expected
        )
        loss = sum((value * weight).sum() for (value, weight) in zip(output.unbind(), weights))
        expected_loss = sum((value * weight).sum() for (value, weight) in zip(expected, weights))
        gradient = torch.autograd.grad(loss, base)[0]
        expected_gradient = torch.autograd.grad(expected_loss, reference_base)[0]
        torch.testing.assert_close(gradient, expected_gradient)

    def test_composition(self, device, float_dtype):
        model = (
            torch.nn.Sequential(
                torch.nn.BatchNorm2d(3), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            .to(device=device, dtype=float_dtype)
            .eval()
        )
        reference_model = deepcopy(model)
        inputs = tuple(
            torch.randn(shape, device=device, dtype=float_dtype, requires_grad=True) for shape in ((3, 5, 7), (3, 6, 4))
        )
        references = tuple(value.detach().clone().requires_grad_() for value in inputs)
        output = model(NT(inputs, ragged_dims=(1, 2)))
        expected = tuple(reference_model(value.unsqueeze(0)).squeeze(0) for value in references)
        torch.testing.assert_close(tuple(output.unbind()), expected)
        weights = tuple(
            torch.linspace(0.25, 1.25, value.numel(), device=device, dtype=float_dtype).reshape_as(value)
            for value in expected
        )
        loss = sum((value * weight).sum() for (value, weight) in zip(output.unbind(), weights))
        expected_loss = sum((value * weight).sum() for (value, weight) in zip(expected, weights))
        gradients = torch.autograd.grad(loss, (*inputs, *model.parameters()))
        expected_gradients = torch.autograd.grad(expected_loss, (*references, *reference_model.parameters()))
        torch.testing.assert_close(gradients, expected_gradients)

    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kernel_size": 3, "stride": 2, "padding": 1},
            {"kernel_size": (3, 2), "stride": (1, 2), "padding": (1, 0), "dilation": (2, 1)},
        ],
    )
    def test_compiled_dynamic(self, device, backend, kwargs, batch_first):
        if device.type != "cuda":
            pytest.skip("compiled packed max pool requires CUDA")
        torch.compiler.reset()

        def operation(input):
            return F.max_pool2d(input, **kwargs)

        compiled = torch.compile(operation, backend=backend, fullgraph=True, dynamic=True)
        for sizes in (((3, 9, 11), (3, 7, 13)), ((3, 12, 15), (3, 11, 9)), ((3, 5, 7), (3, 44, 43))):
            parts = [torch.randn(shape, device=device, requires_grad=True) for shape in sizes]
            references = [part.detach().clone().requires_grad_() for part in parts]
            actual = compiled(NT(parts, ragged_dims=(1, 2), batch_first=batch_first))
            expected = [operation(part) for part in references]
            result = actual.unbind(0 if batch_first else 1)
            torch.testing.assert_close(result, expected)
            probes = [torch.randn_like(part) for part in expected]
            actual_gradients = torch.autograd.grad(result, parts, grad_outputs=probes)
            expected_gradients = torch.autograd.grad(expected, references, grad_outputs=probes)
            torch.testing.assert_close(actual_gradients, expected_gradients)

    def test_compiled_checkpoint(self, device):
        if device.type != "cuda":
            pytest.skip("compiled packed max pool requires CUDA")

        def pooled(input):
            return F.max_pool2d(input, 3, 2, 1)

        def nested_checkpoint(input):
            return checkpoint(lambda value: checkpoint(pooled, value, use_reentrant=False), input, use_reentrant=False)

        compiled = torch.compile(nested_checkpoint, fullgraph=True, dynamic=True)
        for sizes in (((3, 9, 11), (3, 7, 13)), ((3, 12, 15), (3, 11, 9))):
            parts = [torch.randn(shape, device=device, requires_grad=True) for shape in sizes]
            references = [part.detach().clone().requires_grad_() for part in parts]
            actual = compiled(NT(parts, ragged_dims=(1, 2)))
            expected = [pooled(part) for part in references]
            torch.testing.assert_close(actual.unbind(), expected)
            probes = [torch.randn_like(part) for part in expected]
            actual_gradients = torch.autograd.grad(actual.unbind(), parts, grad_outputs=probes)
            expected_gradients = torch.autograd.grad(expected, references, grad_outputs=probes)
            torch.testing.assert_close(actual_gradients, expected_gradients)

    @pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
    def test_compiled_inference(self, device, context):
        if device.type != "cuda":
            pytest.skip("compiled packed max pool requires CUDA")
        parts = [torch.randn(3, h, w, device=device, requires_grad=True) for h, w in ((9, 11), (7, 13))]
        source = NT(parts, ragged_dims=(1, 2))
        compiled = torch.compile(lambda input: F.max_pool2d(input, 3, 2, 1), fullgraph=True, dynamic=True)
        with context():
            actual = compiled(source)
            expected = [F.max_pool2d(part, 3, 2, 1) for part in parts]
        assert not actual.requires_grad
        torch.testing.assert_close(actual.unbind(), expected)

    @pytest.mark.parametrize("value", [float("-inf"), float("nan")])
    def test_nonfinite(self, device, value):
        if device.type != "cuda":
            pytest.skip("packed max pool kernels require CUDA")
        shapes = [(1, *[2] * (2 - 1), length) for length in (4, 6)]
        parts = [torch.full(shape, value, device=device, requires_grad=True) for shape in shapes]
        references = [part.detach().clone().requires_grad_() for part in parts]
        fn = getattr(F, f"max_pool{2}d")
        actual = fn(NT(parts), 2)
        expected = [fn(part, 2) for part in references]
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, equal_nan=True)
        actual_grads = torch.autograd.grad(actual.concat.sum(), parts)
        expected_grads = torch.autograd.grad(sum(part.sum() for part in expected), references)
        for result, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(result, reference)
