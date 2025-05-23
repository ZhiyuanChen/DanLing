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

import random

import pytest
import torch

from danling.tensors import NestedTensor
from danling.tensors.nested_tensor import NestedTensorFuncWrapper

random.seed(0)
torch.manual_seed(0)

EPSILON = 1e-5


def test_compare():
    value = 999999
    small = NestedTensor([[-value, -value, -value], [-value, -value]])
    big = abs(small)
    zero = 0
    assert (big > small).all()
    assert (big > small.tensor).all()
    assert (big > zero).all()
    assert (big > torch.tensor(zero)).all()
    assert (big >= small).all()
    assert (big >= small.tensor).all()
    assert (big >= zero).all()
    assert (big >= torch.tensor(zero)).all()
    assert (big == value).all()
    assert (big == big.tensor).all()
    assert (small < big).all()
    assert (small < big.tensor).all()
    assert (small < zero).all()
    assert (small < torch.tensor(zero)).all()
    assert (small <= big).all()
    assert (small <= big.tensor).all()
    assert (small <= zero).all()
    assert (small <= torch.tensor(zero)).all()
    with pytest.raises(TypeError):
        assert small < "small"
    with pytest.raises(TypeError):
        assert small > "small"
    with pytest.raises(TypeError):
        assert small <= "small"
    with pytest.raises(TypeError):
        assert small >= "small"
    assert small != "small"


@pytest.mark.parametrize(
    "i",
    [
        NestedTensor([[6, 5, 4], [3, 2]]),
        torch.tensor([[6, 5, 4], [3, 2, 1]]),
        torch.randn(2, 3),
        1,
        0,
        -1,
        random.random(),
    ],
)
def test_add_sub(i):
    a = NestedTensor([[2, 3, 4], [5, 6]]).float()
    b = a.clone()
    assert torch.sum((a + i) - (i + a)) < EPSILON
    assert torch.sum((a - i) + (i - a)) < EPSILON
    a += i
    b -= -i
    assert torch.sum(a - b) < EPSILON


@pytest.mark.parametrize(
    "i",
    [
        NestedTensor([[6, 5, 4], [3, 2]]),
        torch.tensor([[6, 5, 4], [3, 2, 1]]),
        torch.randn(2, 3),
        1,
        -1,
        random.random(),
    ],
)
def test_mul_truediv(i):
    a = NestedTensor([[2, 3, 4], [5, 6]]).float()
    b = a.clone()
    assert torch.sum(a * i - i / (1 / a)) < EPSILON
    assert torch.sum(a / (1 / i) - i * a) < EPSILON
    assert torch.sum(a * i - i * a) < EPSILON
    a *= i
    b /= 1 / i
    assert torch.sum(a - b) < EPSILON


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires PyTorch CUDA")
@pytest.mark.skipif(torch.__version__ < "1.12", reason="requires PyTorch 1.12 or higher")
@pytest.mark.parametrize(
    "i",
    [
        NestedTensor([[6, 5, 4], [3, 2]]),
        torch.tensor([[6, 5, 4], [3, 2, 1]]),
        torch.randn(2, 3),
        1,
        -1,
        random.random(),
    ],
)
def test_pow_log(i):
    a = NestedTensor([[2, 3, 4], [5, 6]]).float()
    b = a.clone()
    assert torch.sum(torch.log(a**i) / torch.log(a) - i) < EPSILON
    a **= i
    assert torch.sum(torch.log(a) / torch.log(b) - i) < EPSILON


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires PyTorch CUDA")
@pytest.mark.skipif(torch.__version__ < "2.1", reason="requires PyTorch 2.1 or higher")
@pytest.mark.parametrize(
    "i",
    [
        NestedTensor([[6, 5, 4], [3, 2]]),
        torch.tensor([[6, 5, 4], [3, 2, 1]]),
        1,
    ],
)
def test_shift(i):
    a = NestedTensor([[2, 3, 4], [5, 6]])
    # assert torch.sum(a << i >> i - a) < EPSILON
    assert torch.sum(a >> i << i - a) < EPSILON
    b = a.clone()
    b <<= i
    assert torch.sum(a << i - b) < EPSILON
    b >>= i
    assert torch.sum(a - b) < EPSILON


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires PyTorch CUDA")
@pytest.mark.parametrize(
    "i",
    [
        NestedTensor([[6, 5, 4], [3, 2]]),
        torch.tensor([[6, 5, 4], [3, 2, 1]]),
        torch.randint(0, 9, (2, 3)),
        1,
        0,
        -1,
        random.randint(0, 9),
    ],
)
def test_logic(i):
    a = NestedTensor([[2, 3, 4], [5, 6]]).int()
    assert torch.sum(a & i - i & a) < EPSILON
    assert torch.sum((a | i) - (i | a)) < EPSILON
    assert torch.sum(a ^ i - i ^ a) < EPSILON
    assert torch.sum(~a & ~i - ~(+a | +i)) < EPSILON
    assert torch.sum(~(+i | +a) - ~i & ~a) < EPSILON
    b = a.clone() + 1
    assert torch.sum(((a & i) | (i & b)) - (i & (a | b))) < EPSILON
    assert torch.sum(((i | a) & (b | i)) - (i | (a & b))) < EPSILON
    assert torch.sum(((a ^ i) ^ b) - (a ^ (i ^ b))) < EPSILON
    b = a.clone()
    b &= i
    assert torch.sum(a & i - b) < EPSILON
    b = a.clone()
    b |= i
    assert torch.sum((a | i) - b) < EPSILON
    b = a.clone()
    b ^= i
    assert torch.sum((a ^ i) - b) < EPSILON


@pytest.mark.parametrize(
    "i",
    [
        NestedTensor([[6, 5, 4], [3, 2]], padding_value=-1),
        torch.tensor([[6, 5, 4], [3, 2, 1]]),
        torch.randint(1, 9, (2, 3)),
        2,
        -2,
        random.randint(1, 9),
    ],
)
def test_floordiv(i):
    a = NestedTensor([[2, 3, 4], [5, 6]])
    a.padding_value = -1
    assert (a // i == a.tensor // i).all()
    assert (i // a == i // a.tensor).all()


def test_ifloordiv():
    a = NestedTensor([[2, 3, 4], [5, 6]], dtype=torch.float32)
    b = a.clone()
    a.padding_value = -1
    a //= 1
    assert (a == b).all()
    a //= b
    assert (a == torch.ones(2, 3)).all()
    a //= torch.ones(2, 3)
    assert (a == torch.ones(2, 3)).all()


@pytest.mark.parametrize(
    "i",
    [
        NestedTensor([[6, 5, 4], [3, 2]], padding_value=-1),
        torch.tensor([[6, 5, 4], [3, 2, 1]]),
        torch.randint(1, 9, (2, 3)),
        2,
        -2,
        random.randint(1, 9),
    ],
)
def test_mod(i):
    a = NestedTensor([[2, 3, 4], [5, 6]])
    a.padding_value = -1
    assert (a % i == a.tensor % i).all()
    assert (i % a == i % a.tensor).all()


def test_imod():
    a = NestedTensor([[2, 3, 4], [5, 6]])
    a %= NestedTensor([[6, 5, 4], [3, 2]])
    assert (a == NestedTensor([[2, 3, 0], [2, 0]])).all()
    a = NestedTensor([[2, 3, 4], [5, 6]])
    a %= 2
    assert (a == NestedTensor([[0, 1, 0], [1, 0]])).all()
    a %= torch.ones_like(a.tensor)
    assert (a == torch.zeros(2, 3)).all()


def test_arise():
    a = NestedTensor([[2, 3, 4], [5, 6]])
    with pytest.raises(ValueError):
        _ = a[""]
    with pytest.raises(ValueError):
        _ = NestedTensor(False)
    with pytest.raises(ValueError):
        _ = NestedTensorFuncWrapper(False)
    with pytest.raises(ValueError):
        _ = NestedTensorFuncWrapper([])
    with pytest.raises(ValueError):
        _ = NestedTensorFuncWrapper(False, False)
    with pytest.raises(ValueError):
        temp = a.clone()
        temp._storage = ()
        _ = temp.int()


def test_0dim():
    length = 5
    nested_tensor = NestedTensor(range(length))
    tensor, mask = nested_tensor.tensor_mask
    assert len(nested_tensor) == length
    assert nested_tensor == torch.arange(length)
    assert (tensor == torch.arange(length)).all()
    assert (nested_tensor.tensor == torch.arange(length)).all()
    assert (mask == torch.ones(length)).all()
    assert (nested_tensor.mask == torch.ones(length)).all()


def test_1dim():
    lengths = [2, 3, 5, 7]
    additional_length = 11
    channels = 8
    nested_tensor = NestedTensor(torch.randn(length, channels) for length in lengths)
    tensor, mask = nested_tensor.tensor_mask
    assert tensor.shape == nested_tensor.tensor.shape == torch.Size((len(lengths), max(lengths), channels))
    assert mask.shape == nested_tensor.mask.shape == torch.Size((len(lengths), max(lengths)))
    assert torch.sum(tensor @ nested_tensor.T - nested_tensor.tensor @ nested_tensor.T) < EPSILON
    lengths.append(additional_length)
    nested_tensor = torch.cat([nested_tensor, torch.randn(additional_length, channels)])
    tensor, mask = nested_tensor.tensor_mask
    assert nested_tensor.tensor.shape == torch.Size((len(lengths), max(lengths), channels))
    assert nested_tensor.mask.shape == torch.Size((len(lengths), max(lengths)))
    assert torch.sum(nested_tensor.tensor @ nested_tensor.T - tensor @ nested_tensor.T) < EPSILON


def test_torch_func():
    a = NestedTensor([[2, 3, 4], [5, 6]])
    assert (torch.isin(a, a.tensor[0, 1]) == torch.isin(a.tensor, a.tensor[0, 1])).all()
    assert (torch.isin(a, a.tensor[0, 1]) == torch.isin(a.tensor, a.tensor[0, 1])).all()
    with pytest.raises(NotImplementedError):
        torch.stack([a])
    a = NestedTensor([[2, 3, 4], [5, 6]]).float()
    assert torch.sum(torch.mean(a) - a.mean()) < EPSILON
    assert torch.sum(torch.sqrt(a) - a.sqrt()) < EPSILON
    assert torch.sum(torch.log(a) - a.log()) < EPSILON


@pytest.mark.skipif(torch.__version__ < "2.1", reason="requires PyTorch 2.1 or higher")
def test_where():
    a = NestedTensor([[2, 3, 4], [5, 6]])
    assert (a.where(a > 3, -1.0) == NestedTensor([[-1.0, -1.0, 4.0], [5.0, 6.0]])).all()
    assert (a.where(a.tensor > 3, -1.0) == NestedTensor([[-1.0, -1.0, 4.0], [5.0, 6.0]])).all()
    assert (a.where(torch.tensor(False), 1) == NestedTensor([[1, 1, 1], [1, 1]])).all()
