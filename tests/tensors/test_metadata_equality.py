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


def _square_pair(lengths: tuple[int, ...], channels: int = 2) -> NestedTensor:
    return NestedTensor(
        [torch.randn(length, length, channels) for length in lengths],
        ragged_dims=(0, 1),
    )


def test_metadata_tensor_equal_preserves_eager_mismatch_contracts():
    assert NestedTensor._meta_tensor_equal(torch.tensor([0, 2]), torch.tensor([0, 2]))
    assert not NestedTensor._meta_tensor_equal(torch.tensor([0, 2]), torch.tensor([0, 3]))
    assert not NestedTensor._meta_tensor_equal(
        torch.tensor([0, 2]),
        torch.tensor([0, 2, 4]),
        runtime_assert=True,
    )
    with pytest.raises(RuntimeError, match="metadata values differ"):
        NestedTensor._meta_tensor_equal(
            torch.tensor([0, 2]),
            torch.tensor([0, 3]),
            "metadata values differ",
            runtime_assert=True,
        )

    lhs = _square_pair((2, 4))
    rhs = _square_pair((4, 2))
    assert not lhs._has_same_structure(rhs)
    with pytest.raises(RuntimeError, match="must match the size"):
        lhs + rhs


def test_metadata_tensor_equal_accepts_known_and_unbacked_fake_shapes():
    fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    pair = _square_pair((2, 3))
    reference = NestedTensor([torch.empty(1), torch.empty(1)])
    mode = fake_tensor_mod.FakeTensorMode(shape_env=ShapeEnv(allow_dynamic_output_shape_ops=True))
    with mode:
        fake_pair = mode.from_tensor(pair)
        fake_reference = mode.from_tensor(reference)
        fake_values = mode.from_tensor(torch.empty(13, 2))
        fake_lengths = mode.from_tensor(torch.tensor([2, 3]))
        rebuilt = fake_reference.packed_with_square_lengths(fake_values, fake_lengths)
        same_structure = fake_pair._has_same_structure(rebuilt)

    assert same_structure
    assert fake_tensor_mod.is_fake(fake_pair.ragged_level_offsets(1))
    assert fake_tensor_mod.is_fake(rebuilt.ragged_level_offsets(1))


def test_metadata_tensor_equal_rejects_statically_known_fake_shape_mismatch():
    fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")

    with fake_tensor_mod.FakeTensorMode() as mode:
        lhs = mode.from_tensor(torch.empty(2, dtype=torch.long))
        different_size = mode.from_tensor(torch.empty(3, dtype=torch.long))
        different_rank = mode.from_tensor(torch.empty(2, 1, dtype=torch.long))
        assert not NestedTensor._meta_tensor_equal(lhs, different_size)
        assert not NestedTensor._meta_tensor_equal(lhs, different_rank)


def test_known_and_unbacked_metadata_reuse_one_dynamic_fullgraph():
    from torch._dynamo.testing import CompileCounter

    reference = NestedTensor([torch.empty(1), torch.empty(1)])
    counter = CompileCounter()

    def consume(pair, ref, values, lengths):
        rebuilt = ref.packed_with_square_lengths(values, lengths)
        return (pair + rebuilt).concat

    compiled = torch.compile(consume, backend=counter, fullgraph=True, dynamic=True)
    for lengths in ((2, 3), (1, 4)):
        pair = _square_pair(lengths)
        values = torch.randn(sum(length * length for length in lengths), 2)
        output = compiled(pair, reference, values, torch.tensor(lengths))
        assert_close(output, pair.concat + values)

    assert counter.frame_count == 1


@pytest.mark.parametrize(
    ("known_lengths", "mismatched_lengths"),
    [
        ((2, 4), (4, 2)),
        ((1, 7), (5, 5)),
    ],
    ids=("offset-values", "offset-shape"),
)
def test_compiled_metadata_equality_rejects_runtime_mismatches(known_lengths, mismatched_lengths):
    from torch._dynamo.testing import CompileCounter

    pair = _square_pair(known_lengths)
    reference = NestedTensor([torch.empty(1), torch.empty(1)])
    packed_size = sum(length * length for length in known_lengths)
    values = torch.randn(packed_size, 2)
    counter = CompileCounter()

    def consume(known, ref, packed_values, lengths):
        rebuilt = ref.packed_with_square_lengths(packed_values, lengths)
        return (known + rebuilt).concat

    compiled = torch.compile(consume, backend=counter, fullgraph=True, dynamic=True)
    output = compiled(pair, reference, values, torch.tensor(known_lengths))
    assert_close(output, pair.concat + values)
    with pytest.raises(RuntimeError, match="NestedTensor ragged offsets must match"):
        compiled(pair, reference, values, torch.tensor(mismatched_lengths))

    assert counter.frame_count == 1
