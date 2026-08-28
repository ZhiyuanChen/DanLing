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

from danling.tensors import NestedTensor, nested_execution_guard
from tests.tensors.utils import assert_close


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


def test_batch_repeat_interleave_preserves_order_topology_and_gradients():
    source, values = _packed_leaf(((2, 3), (0, 3), (5, 3)), ragged_dims=(0,))

    with nested_execution_guard(
        forbid_iteration=True,
        forbid_storage_map=True,
        forbid_eager_fallback=True,
        forbid_padded_materialization=True,
        forbid_dense_repack=True,
    ):
        output = torch.repeat_interleave(source, 3, dim=0)
        method_output = source.repeat_interleave(3, dim=0)
        batch_output = source.repeat_batch(3)

    assert_close(output, NestedTensor(_expected_elements(source, 3), ragged_dims=(0,)))
    assert_close(method_output, output)
    assert_close(batch_output, output)
    assert_close(output.packed_offsets(), torch.tensor([0, 2, 4, 6, 6, 6, 6, 11, 16, 21]))
    assert output.ragged_dims == source.ragged_dims
    assert output.packed_dim_order == source.packed_dim_order
    output._validate_metadata()

    grad = torch.autograd.grad(output.concat.square().sum(), values)[0]
    assert_close(grad, 6 * values)


def test_batch_repeat_interleave_respects_nonleading_batch_dimension():
    source, values = _packed_leaf(((2, 3), (4, 3)), ragged_dims=(0,), batch_first=False)

    output = torch.repeat_interleave(source, 2, dim=1)
    negative_dim_output = torch.repeat_interleave(source, 2, dim=-2)

    expected = NestedTensor(_expected_elements(source, 2), ragged_dims=(0,), batch_first=False)
    assert_close(output, expected)
    assert_close(negative_dim_output, expected)
    assert output.batch_first is False
    assert output.shape == torch.Size((4, 4, 3))
    grad = torch.autograd.grad(output.concat.sum(), values)[0]
    assert_close(grad, torch.full_like(values, 2))


def test_batch_repeat_interleave_preserves_permuted_multi_ragged_hierarchy():
    source, values = _packed_leaf(((2, 3, 4), (1, 4, 4)), ragged_dims=(1, 0))

    with nested_execution_guard(
        forbid_iteration=True,
        forbid_storage_map=True,
        forbid_eager_fallback=True,
        forbid_padded_materialization=True,
        forbid_dense_repack=True,
    ):
        output = torch.repeat_interleave(source, 2, dim=0)

    expected = NestedTensor(_expected_elements(source, 2), ragged_dims=(1, 0))
    assert_close(output, expected)
    assert output.ragged_dims == (1, 0)
    assert output.packed_dim_order == (1, 0, 2)
    assert_close(output.ragged_level_offsets(0), expected.ragged_level_offsets(0))
    assert_close(output.ragged_level_offsets(1), expected.ragged_level_offsets(1))
    output._validate_metadata()
    grad = torch.autograd.grad(output.concat.square().sum(), values)[0]
    assert_close(grad, 4 * values)


@pytest.mark.parametrize("repeats", [0, 1])
def test_batch_repeat_interleave_handles_zero_and_identity_counts(repeats):
    source, values = _packed_leaf(((0, 3), (2, 3)), ragged_dims=(0,))

    output = torch.repeat_interleave(source, repeats, dim=0, output_size=len(source) * repeats)

    assert len(output) == len(source) * repeats
    assert output.concat.shape == (values.shape[0] * repeats, values.shape[1])
    if repeats:
        assert_close(output, NestedTensor(_expected_elements(source, repeats), ragged_dims=(0,)))
    output._validate_metadata()
    grad = torch.autograd.grad(output.concat.sum(), values, allow_unused=True)[0]
    if repeats == 0:
        assert_close(grad, torch.zeros_like(values))
    else:
        assert_close(grad, torch.ones_like(values))


def test_batch_repeat_interleave_supports_inferred_and_static_eager_layouts():
    ragged = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
    static = NestedTensor([torch.randn(2, 3), torch.randn(2, 3)])

    ragged_output = torch.repeat_interleave(ragged, 2, dim=0)
    static_output = torch.repeat_interleave(static, 2, dim=0)

    assert_close(ragged_output, NestedTensor(_expected_elements(ragged, 2)))
    assert_close(static_output, NestedTensor(_expected_elements(static, 2)))
    ragged_output._validate_metadata()
    static_output._validate_metadata()


def test_batch_repeat_interleave_validates_scalar_count_and_output_size():
    source = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)], ragged_dims=(0,))

    with pytest.raises(RuntimeError, match="Repeats must be non-negative"):
        torch.repeat_interleave(source, -1, dim=0)
    with pytest.raises(TypeError, match="invalid combination of arguments"):
        torch.repeat_interleave(source, True, dim=0)
    with pytest.raises(TypeError, match="requires repeats to be an int"):
        torch.repeat_interleave(source, torch.tensor(2), dim=0)
    with pytest.raises(RuntimeError, match="Invalid output_size, expected 4 but got 3"):
        torch.repeat_interleave(source, 2, dim=0, output_size=3)


def test_batch_repeat_interleave_supports_external_fake_tensor():
    fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
    source = NestedTensor(
        [torch.empty(2, 3, 4), torch.empty(1, 4, 4)],
        ragged_dims=(1, 0),
    )

    with fake_tensor_mod.FakeTensorMode() as mode:
        fake_source = mode.from_tensor(source)
        output = torch.repeat_interleave(fake_source, 2, dim=0)

    assert fake_tensor_mod.is_fake(output.concat)
    assert fake_tensor_mod.is_fake(output.packed_offsets())
    assert fake_tensor_mod.is_fake(output.ragged_level_offsets(0))
    assert fake_tensor_mod.is_fake(output.ragged_level_offsets(1))
    assert output.shape == (4, 2, 4, 4)
    assert output.ragged_dims == (1, 0)


@pytest.mark.parametrize("entry", ["torch", "method"])
def test_batch_repeat_interleave_reuses_one_dynamic_fullgraph_with_backward(entry):
    from torch._dynamo.testing import CompileCounter

    counter = CompileCounter()

    def consume(source):
        output = torch.repeat_interleave(source, 3, dim=0) if entry == "torch" else source.repeat_interleave(3, dim=0)
        return output.concat.square().sum(), output.packed_offsets(), output.element_sizes()

    compiled = torch.compile(consume, backend=counter, fullgraph=True, dynamic=True)
    for lengths in ((2, 5), (4, 1), (3, 6)):
        source, values = _packed_leaf(tuple((length, 4) for length in lengths), ragged_dims=(0,))
        with nested_execution_guard(
            forbid_iteration=True,
            forbid_storage_map=True,
            forbid_eager_fallback=True,
            forbid_padded_materialization=True,
            forbid_dense_repack=True,
        ):
            loss, offsets, sizes = compiled(source)
        loss.backward()
        assert_close(values.grad, 6 * values)
        expected_lengths = torch.tensor(lengths).repeat_interleave(3)
        assert_close(offsets, torch.nn.functional.pad(expected_lengths.cumsum(0), (1, 0)))
        assert_close(sizes[:, 0], expected_lengths)

    assert counter.frame_count == 1


def test_batch_repeat_interleave_multi_ragged_aot_fullgraph_backward():
    def consume(source):
        return source.repeat_batch(2).concat.square().sum()

    compiled = torch.compile(consume, backend="aot_eager", fullgraph=True, dynamic=True)
    for shapes in (((2, 3, 4), (1, 4, 4)), ((3, 2, 4), (2, 5, 4))):
        source, values = _packed_leaf(shapes, ragged_dims=(1, 0))
        loss = compiled(source)
        loss.backward()
        assert_close(values.grad, 4 * values)
