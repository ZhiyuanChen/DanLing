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
from copy import deepcopy

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


def test_avg_pool2d(device):
    inputs = [
        torch.randn(3, 11, 13, device=device, requires_grad=True),
        torch.randn(3, 7, 9, device=device, requires_grad=True),
    ]

    _compare_pool(
        inputs,
        F.avg_pool2d,
        kernel_size=3,
        stride=2,
        padding=1,
        count_include_pad=False,
        divisor_override=9,
        require_packed=device.type == "cuda",
    )


def test_max_pool2d(device):
    inputs = [
        torch.randn(3, 11, 13, device=device, requires_grad=True),
        torch.randn(3, 7, 9, device=device, requires_grad=True),
    ]

    _compare_pool(
        inputs,
        F.max_pool2d,
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=2,
        require_packed=device.type == "cuda",
    )


def test_avg_pool2d_divisor_override_zero_matches_torch(device):
    input = NT(
        [
            torch.randn(3, 11, 13, device=device),
            torch.randn(3, 7, 9, device=device),
        ]
    )

    with pytest.raises(RuntimeError, match="divisor"):
        F.avg_pool2d(input, kernel_size=3, divisor_override=0)


def test_max_pool2d_return_indices(device):
    inputs = [
        torch.randn(3, 11, 13, device=device, requires_grad=True),
        torch.randn(3, 7, 9, device=device, requires_grad=True),
    ]

    _compare_pool(inputs, F.max_pool2d, kernel_size=2, stride=2, return_indices=True)


@pytest.mark.parametrize("pool", ("avg", "max"))
@pytest.mark.parametrize("layout", ("transposed", "sliced", "expanded_rows", "expanded_channels"))
def test_pool2d_strided_packed_values_and_gradients(device, float_dtype, pool, layout):
    shapes = ((3, 5, 7), (3, 6, 4))
    counts = tuple(height * width for _, height, width in shapes)
    rows, channels = sum(counts), shapes[0][0]
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
            # Exercise nonzero storage offset and non-unit strides on both axes.
            return value[1::2, 1::2]
        return value.expand(rows, channels)

    template = NT([torch.empty(shape, device=device, dtype=float_dtype) for shape in shapes], ragged_dims=(1, 2))
    nested = template.packed_like(packed_view(base))
    references = tuple(
        chunk.reshape(height, width, channels).permute(2, 0, 1)
        for chunk, (_, height, width) in zip(packed_view(reference_base).split(counts), shapes)
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

    # Unequal cotangents test routing of every output into its actual input
    # view, including accumulation through expanded (zero-stride) dimensions.
    weights = tuple(
        torch.linspace(0.25, 1.25, value.numel(), device=device, dtype=float_dtype).reshape_as(value)
        for value in expected
    )
    loss = sum((value * weight).sum() for value, weight in zip(output.unbind(), weights))
    expected_loss = sum((value * weight).sum() for value, weight in zip(expected, weights))
    gradient = torch.autograd.grad(loss, base)[0]
    expected_gradient = torch.autograd.grad(expected_loss, reference_base)[0]
    torch.testing.assert_close(gradient, expected_gradient)


def test_frozen_batch_norm_relu_pool_composition(device, float_dtype):
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
    loss = sum((value * weight).sum() for value, weight in zip(output.unbind(), weights))
    expected_loss = sum((value * weight).sum() for value, weight in zip(expected, weights))
    gradients = torch.autograd.grad(loss, (*inputs, *model.parameters()))
    expected_gradients = torch.autograd.grad(expected_loss, (*references, *reference_model.parameters()))
    torch.testing.assert_close(gradients, expected_gradients)


def test_compiled_max_pool2d_reports_unsupported(device):
    # F.max_pool2d is registered eager-only: under compile the handler must fail explicitly
    # instead of comparing fake metadata or replaying per sample.
    torch.compiler.reset()
    compiled = torch.compile(
        lambda template, values: F.max_pool2d(template.packed_like(values), 2),
        backend="aot_eager",
        fullgraph=True,
        dynamic=True,
    )
    try:
        template = NT([torch.randn(3, 8, 10, device=device), torch.randn(3, 6, 4, device=device)], ragged_dims=(1, 2))
        values = template.concat.detach().requires_grad_()
        with pytest.raises(Exception, match="compile-safe path not implemented"):
            compiled(template, values)
    finally:
        torch.compiler.reset()
