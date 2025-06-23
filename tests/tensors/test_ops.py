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

from danling.tensors import NestedTensor
from danling.tensors.aten_functions import NestedTensorAtenRegistry
from danling.tensors.ops import (
    ATEN_BINARY_ELEMENTWISE_OPS,
    ATEN_UNARY_ELEMENTWISE_OPS,
    TORCH_BINARY_ELEMENTWISE_OPS,
    TORCH_UNARY_ELEMENTWISE_OPS,
    _concat_dim_for_tensor_dim,
    _translate_dim,
    _translate_dims,
)
from danling.tensors.torch_functions import NestedTensorFuncRegistry

NT = NestedTensor


def test_translate_dim_batch_first():
    nt = NT([torch.randn(2, 3), torch.randn(1, 3)])
    assert _translate_dim(nt, 1) == 0
    assert _translate_dim(nt, 2) == 1
    with pytest.raises(ValueError, match="batch dimension"):
        _translate_dim(nt, 0)


def test_translate_dims_batch_first_false():
    nt = NT([torch.randn(2, 3), torch.randn(1, 3)], batch_first=False)
    assert _translate_dims(nt, (0, 2)) == (0, 1)
    with pytest.raises(ValueError, match="batch dimension"):
        _translate_dims(nt, (1,))


def test_concat_dim_for_tensor_dim():
    nt = NT([torch.randn(2, 3), torch.randn(1, 3)])
    assert _concat_dim_for_tensor_dim(nt, 0) is None
    assert _concat_dim_for_tensor_dim(nt, 1) == 1
    assert _concat_dim_for_tensor_dim(nt, -1) == 1


def test_concat_dim_for_tensor_dim_out_of_range():
    nt = NT([torch.randn(2, 3), torch.randn(1, 3)])
    with pytest.raises(IndexError):
        _concat_dim_for_tensor_dim(nt, 2)


def test_dispatch_table_covers_elementwise_ops():
    """Ensure all registered elementwise ops have dispatch table entries."""
    for op in ATEN_UNARY_ELEMENTWISE_OPS:
        assert op in NestedTensorAtenRegistry, f"Unary op {op} registered but missing from NestedTensorAtenRegistry"
    for op in ATEN_BINARY_ELEMENTWISE_OPS:
        assert op in NestedTensorAtenRegistry, f"Binary op {op} registered but missing from NestedTensorAtenRegistry"


def test_function_registry_populated():
    """Ensure torch.* function overrides are registered."""
    registry = NestedTensorFuncRegistry
    for op in TORCH_UNARY_ELEMENTWISE_OPS:
        assert op in registry, f"Unary op {op} not in NestedTensorFuncRegistry"
    for op in TORCH_BINARY_ELEMENTWISE_OPS:
        assert op in registry, f"Binary op {op} not in NestedTensorFuncRegistry"


def test_unary_ops_roundtrip():
    """Verify a sample of unary ops produce correct per-element values."""
    nt = NT([torch.tensor([1.0, -2.0, 3.0]), torch.tensor([-4.0, 5.0])])
    for op in [torch.abs, torch.neg, torch.exp, torch.sign]:
        result = op(nt)
        assert isinstance(result, NestedTensor), f"{op} did not return NestedTensor"
        for r, t in zip(result, nt):
            torch.testing.assert_close(r, op(t))


def test_binary_ops_roundtrip():
    """Verify a sample of binary ops produce correct per-element values."""
    nt = NT([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])])
    scalar = 2.0
    for op in [torch.mul, torch.add]:
        result = op(nt, scalar)
        assert isinstance(result, NestedTensor), f"{op} did not return NestedTensor"
        for r, t in zip(result, nt):
            torch.testing.assert_close(r, op(t, scalar))
