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

from __future__ import annotations

import pytest
import torch

from danling.tensors import NestedTensor
from tests.tensors.utils import assert_close


class _DerivedNestedTensor(NestedTensor):
    pass


def _packed_leaf(shapes, *, ragged_dims, batch_first=True):
    template = NestedTensor(
        [torch.empty(shape) for shape in shapes],
        ragged_dims=ragged_dims,
        batch_first=batch_first,
    )
    values = torch.randn_like(template.concat, requires_grad=True)
    return template.packed_like(values), values


def _expected_elements(source: NestedTensor, repeats: int) -> list[torch.Tensor]:
    return [source[index] for index in range(len(source)) for _ in range(repeats)]


def test_batch_repeat_interleave_preserves_order_entry_points_and_gradients():
    source, values = _packed_leaf(((2, 3), (0, 3), (5, 3)), ragged_dims=(0,))
    expected = NestedTensor(_expected_elements(source, 3), ragged_dims=(0,))

    function_output = torch.repeat_interleave(source, 3, dim=0)
    method_output = source.repeat_interleave(3, dim=0)
    batch_output = source.repeat_batch(3)

    assert_close(function_output, expected)
    assert_close(method_output, expected)
    assert_close(batch_output, expected)
    assert function_output.ragged_dims == source.ragged_dims
    grad = torch.autograd.grad(function_output.concat.square().sum(), values)[0]
    assert_close(grad, 6 * values)


def test_batch_repeat_interleave_respects_nonleading_batch_dimension():
    source, values = _packed_leaf(((2, 3), (4, 3)), ragged_dims=(0,), batch_first=False)
    expected = NestedTensor(_expected_elements(source, 2), ragged_dims=(0,), batch_first=False)

    output = torch.repeat_interleave(source, 2, dim=1)
    negative_dim_output = torch.repeat_interleave(source, 2, dim=-2)

    assert_close(output, expected)
    assert_close(negative_dim_output, expected)
    assert output.shape == torch.Size((4, 4, 3))
    grad = torch.autograd.grad(output.concat.sum(), values)[0]
    assert_close(grad, torch.full_like(values, 2))


def test_batch_repeat_interleave_preserves_multi_ragged_layout():
    source, values = _packed_leaf(((2, 3, 4), (1, 4, 4)), ragged_dims=(1, 0))
    expected = NestedTensor(_expected_elements(source, 2), ragged_dims=(1, 0))

    output = torch.repeat_interleave(source, 2, dim=0)

    assert_close(output, expected)
    assert output.shape == expected.shape
    assert output.ragged_dims == (1, 0)
    grad = torch.autograd.grad(output.concat.square().sum(), values)[0]
    assert_close(grad, 4 * values)




def test_batch_repeat_interleave_supports_static_layout():
    source = NestedTensor([torch.randn(2, 3), torch.randn(2, 3)])

    output = torch.repeat_interleave(source, 2, dim=0)

    assert_close(output, NestedTensor(_expected_elements(source, 2)))




def test_batch_repeat_interleave_supports_fake_tensor():
    fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
    source = NestedTensor(
        [torch.empty(2, 3, 4), torch.empty(1, 4, 4)],
        ragged_dims=(1, 0),
    )

    with fake_tensor_mod.FakeTensorMode() as mode:
        output = torch.repeat_interleave(mode.from_tensor(source), 2, dim=0)

    assert fake_tensor_mod.is_fake(output.concat)
    assert output.shape == (4, 2, 4, 4)
    assert output.ragged_dims == (1, 0)


def test_repeat_batch_preserves_nested_tensor_subclass_eager_and_fake():
    source = _DerivedNestedTensor(
        [torch.randn(2, 3), torch.randn(4, 3)],
        ragged_dims=(0,),
    )
    eager_output = source.repeat_batch(2)

    fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
    with fake_tensor_mod.FakeTensorMode() as mode:
        fake_output = mode.from_tensor(source).repeat_batch(2)

    assert type(eager_output) is _DerivedNestedTensor
    assert type(fake_output) is _DerivedNestedTensor
    assert_close(eager_output, NestedTensor(_expected_elements(source, 2), ragged_dims=(0,)))
    assert fake_tensor_mod.is_fake(fake_output.concat)
    assert fake_output.shape == eager_output.shape


@pytest.mark.parametrize("entry", ["repeat-batch", "torch"])
def test_batch_repeat_preserves_nested_tensor_subclass_when_compiled(entry):
    template = _DerivedNestedTensor(
        [torch.empty(2, 3), torch.empty(4, 3)],
        ragged_dims=(0,),
    )
    values = torch.randn_like(template.concat)
    source = template.packed_like(values)
    expected = NestedTensor(_expected_elements(source, 2), ragged_dims=(0,))

    def repeat(input_):
        if entry == "repeat-batch":
            return input_.repeat_batch(2)
        return torch.repeat_interleave(input_, 2, dim=0)

    output = torch.compile(repeat, backend="aot_eager", fullgraph=True, dynamic=True)(source)

    assert type(output) is _DerivedNestedTensor
    assert_close(output, expected)


def test_batch_repeat_interleave_aot_fullgraph_backward():
    source, values = _packed_leaf(((2, 3, 4), (1, 4, 4)), ragged_dims=(1, 0))
    compiled = torch.compile(
        lambda input_: input_.repeat_batch(2).concat.square().sum(),
        backend="aot_eager",
        fullgraph=True,
        dynamic=True,
    )

    loss = compiled(source)
    loss.backward()

    assert_close(values.grad, 4 * values)
