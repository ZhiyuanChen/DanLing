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


def test_ragged_metadata_mismatch_is_rejected_eagerly():
    lhs = _square_pair((2, 4))
    rhs = _square_pair((4, 2))

    with pytest.raises(RuntimeError, match="must match"):
        lhs + rhs


def test_ragged_metadata_supports_fake_tensor_operations():
    fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    pair = _square_pair((2, 3))
    reference = NestedTensor([torch.empty(1), torch.empty(1)])
    mode = fake_tensor_mod.FakeTensorMode(shape_env=ShapeEnv(allow_dynamic_output_shape_ops=True))
    with mode:
        fake_pair = mode.from_tensor(pair)
        fake_reference = mode.from_tensor(reference)
        rebuilt = fake_reference.packed_with_square_lengths(
            mode.from_tensor(torch.empty(13, 2)),
            mode.from_tensor(torch.tensor([2, 3])),
        )
        output = fake_pair + rebuilt

    assert fake_tensor_mod.is_fake(output.concat)
    assert output.shape == pair.shape


def test_compiled_ragged_metadata_checks_runtime_values():
    pair = _square_pair((2, 4))
    reference = NestedTensor([torch.empty(1), torch.empty(1)])
    values = torch.randn(20, 2)

    def consume(known, ref, packed_values, lengths):
        rebuilt = ref.packed_with_square_lengths(packed_values, lengths)
        return (known + rebuilt).concat

    compiled = torch.compile(consume, backend="aot_eager", fullgraph=True, dynamic=True)

    assert_close(compiled(pair, reference, values, torch.tensor([2, 4])), pair.concat + values)
    with pytest.raises(RuntimeError, match="NestedTensor ragged offsets must match"):
        compiled(pair, reference, values, torch.tensor([4, 2]))
