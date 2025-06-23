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

from danling.tensors import NestedTensor, PNTensor
from danling.tensors import tensor as pn_tensor
from danling.tensors.nested_tensor import NestedTensorFuncWrapper

random.seed(0)
torch.manual_seed(0)


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
    assert torch.equal((a + i), (i + a))
    assert torch.equal((a - i), -(i - a))
    a += i
    b -= -i
    assert torch.equal(a, b)


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
    assert torch.allclose(a * i, i / (1 / a))
    assert torch.allclose(a / (1 / i), i * a)
    assert torch.allclose(a * i, i * a)
    a *= i
    b /= 1 / i
    assert torch.allclose(a, b)


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
    assert torch.equal(torch.log(a**i) / torch.log(a), i)
    a **= i
    assert torch.equal(torch.log(a) / torch.log(b), i)


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
    assert torch.equal(a << i >> i, a)
    assert torch.equal(a >> i << i, a)
    b = a.clone()
    b <<= i
    assert torch.equal(a << i, b)
    b >>= i
    assert torch.equal(a, b)


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
    assert torch.equal(a & i, i & a)
    assert torch.equal((a | i), (i | a))
    assert torch.equal(a ^ i, i ^ a)
    assert torch.equal(~a & ~i, ~(+a | +i))
    assert torch.equal(~(+i | +a), ~i & ~a)
    b = a.clone() + 1
    assert torch.equal(((a & i) | (i & b)), (i & (a | b)))
    assert torch.equal(((i | a) & (b | i)), (i | (a & b)))
    assert torch.equal(((a ^ i) ^ b), (a ^ (i ^ b)))
    b = a.clone()
    b &= i
    assert torch.equal(a & i, b)
    b = a.clone()
    b |= i
    assert torch.equal(a | i, b)
    b = a.clone()
    b ^= i
    assert torch.equal(a ^ i, b)


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
    assert torch.equal(a // i, a.tensor // i)
    assert torch.equal(i // a, i // a.tensor)


def test_ifloordiv():
    a = NestedTensor([[2, 3, 4], [5, 6]], dtype=torch.float32)
    b = a.clone()
    a.padding_value = -1
    a //= 1
    assert torch.equal(a, b)
    a //= b
    assert torch.equal(a, torch.ones(2, 3))
    a //= torch.ones(2, 3)
    assert torch.equal(a, torch.ones(2, 3))


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
    assert torch.equal(a % i, a.tensor % i)
    assert torch.equal(i % a, i % a.tensor)


def test_imod():
    a = NestedTensor([[2, 3, 4], [5, 6]])
    a %= NestedTensor([[6, 5, 4], [3, 2]])
    assert torch.equal(a, NestedTensor([[2, 3, 0], [2, 0]]))
    a = NestedTensor([[2, 3, 4], [5, 6]])
    a %= 2
    assert torch.equal(a, NestedTensor([[0, 1, 0], [1, 0]]))
    a %= torch.ones_like(a.tensor)
    assert torch.equal(a, torch.zeros(2, 3))


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
    assert torch.equal(tensor @ nested_tensor.T, nested_tensor.tensor @ nested_tensor.T)
    lengths.append(additional_length)
    nested_tensor = torch.cat([nested_tensor, torch.randn(additional_length, channels)])
    tensor, mask = nested_tensor.tensor_mask
    assert nested_tensor.tensor.shape == torch.Size((len(lengths), max(lengths), channels))
    assert nested_tensor.mask.shape == torch.Size((len(lengths), max(lengths)))
    assert torch.equal(nested_tensor.tensor @ nested_tensor.T, tensor @ nested_tensor.T)


def test_torch_func():
    a = NestedTensor([[2, 3, 4], [5, 6]])
    assert torch.equal(torch.isin(a, a.tensor[0, 1]), torch.isin(a.tensor, a.tensor[0, 1]))
    assert torch.equal(torch.isin(a, a.tensor[0, 1]), torch.isin(a.tensor, a.tensor[0, 1]))
    a = NestedTensor([[2, 3, 4], [5, 6]]).float()
    assert torch.equal(torch.mean(a), a.mean())
    assert torch.equal(torch.sqrt(a), a.sqrt())
    assert torch.equal(torch.log(a), a.log())


@pytest.mark.skipif(torch.__version__ < "2.1", reason="requires PyTorch 2.1 or higher")
def test_where():
    a = NestedTensor([[2, 3, 4], [5, 6]])
    assert torch.equal(a.where(a > 3, -1.0), NestedTensor([[-1.0, -1.0, 4.0], [5.0, 6.0]]))
    assert torch.equal(a.where(a.tensor > 3, -1.0), NestedTensor([[-1.0, -1.0, 4.0], [5.0, 6.0]]))
    assert torch.equal(a.where(torch.tensor(False), 1), NestedTensor([[1, 1, 1], [1, 1]]))


def test_permute():
    nested_tensor = NestedTensor([torch.randn(3, 4, 5), torch.randn(2, 4, 5)])
    original_shape = nested_tensor.shape
    assert original_shape == torch.Size([2, 3, 4, 5])

    permuted = nested_tensor.permute(0, 3, 1, 2)
    assert permuted.shape == torch.Size([2, 5, 3, 4])
    assert permuted is not nested_tensor

    assert permuted._storage[0].shape == torch.Size([5, 3, 4])
    assert permuted._storage[1].shape == torch.Size([5, 2, 4])
    assert nested_tensor.shape == torch.Size([2, 3, 4, 5])

    nested_tensor2 = NestedTensor([torch.randn(3, 4), torch.randn(2, 4)])
    permuted2 = nested_tensor2.permute(0, -1, -2)
    assert permuted2.shape == torch.Size([2, 4, 3])

    with pytest.raises(ValueError, match="Expected 3 dimensions"):
        nested_tensor2.permute(0, 1)

    nested_tensor3 = NestedTensor([torch.randn(3, 4), torch.randn(2, 4)])
    with pytest.raises(ValueError, match="Batch dimension.*must be included"):
        nested_tensor3.permute(1, 2, -1)


def test_transpose():
    nested_tensor = NestedTensor([torch.randn(3, 4), torch.randn(2, 4)])
    original_shape = nested_tensor.shape
    assert original_shape == torch.Size([2, 3, 4])

    transposed = nested_tensor.transpose(1, 2)
    assert transposed.shape == torch.Size([2, 4, 3])
    assert transposed is not nested_tensor

    assert transposed._storage[0].shape == torch.Size([4, 3])
    assert transposed._storage[1].shape == torch.Size([4, 2])
    assert nested_tensor.shape == torch.Size([2, 3, 4])

    nested_tensor2 = NestedTensor([torch.randn(3, 4, 5), torch.randn(2, 4, 5)])
    transposed2 = nested_tensor2.transpose(-2, -1)
    assert transposed2.shape == torch.Size([2, 3, 5, 4])

    with pytest.raises(ValueError, match="Cannot transpose the batch dimension"):
        nested_tensor2.transpose(0, 1)

    with pytest.raises(ValueError, match="Cannot transpose the batch dimension"):
        nested_tensor2.transpose(1, 0)


def test_reshape():
    nested_tensor = NestedTensor([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])])
    original_shape = nested_tensor.shape
    assert original_shape == torch.Size([2, 2, 2])

    reshaped = nested_tensor.reshape(4)
    assert reshaped.shape == torch.Size([2, 4])
    assert reshaped is not nested_tensor

    assert reshaped._storage[0].shape == torch.Size([4])
    assert reshaped._storage[1].shape == torch.Size([4])
    assert nested_tensor.shape == torch.Size([2, 2, 2])

    nested_tensor2 = NestedTensor([torch.randn(2, 3, 4), torch.randn(2, 3, 4)])
    reshaped2 = nested_tensor2.reshape(-1, 4)
    assert reshaped2.shape == torch.Size([2, 6, 4])

    empty_nested = NestedTensor([])
    result = empty_nested.reshape(5)
    assert result is not empty_nested
    assert len(result._storage) == 0


def test_view():
    nested_tensor = NestedTensor([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])])
    original_shape = nested_tensor.shape
    assert original_shape == torch.Size([2, 2, 2])

    viewed = nested_tensor.view(4)
    assert viewed.shape == torch.Size([2, 4])
    assert viewed is not nested_tensor

    assert viewed._storage[0].shape == torch.Size([4])
    assert viewed._storage[1].shape == torch.Size([4])
    assert nested_tensor.shape == torch.Size([2, 2, 2])

    nested_tensor2 = NestedTensor([torch.randn(2, 6), torch.randn(2, 6)])
    viewed2 = nested_tensor2.view(-1, 3)
    assert viewed2.shape == torch.Size([2, 4, 3])

    nested_tensor3 = NestedTensor([torch.randn(4), torch.randn(4)])
    viewed3 = nested_tensor3.view(2, 2)
    viewed3._storage[0][0, 0] = 999
    assert nested_tensor3._storage[0][0] == 999

    empty_nested = NestedTensor([])
    result = empty_nested.view(5)
    assert result is not empty_nested
    assert len(result._storage) == 0


def test_view_with_different_shapes():
    nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])

    viewed = nested_tensor.view(-1)
    assert viewed is not nested_tensor
    assert viewed._storage[0].shape == torch.Size([3])
    assert viewed._storage[1].shape == torch.Size([2])


def test_method_chaining():
    nested_tensor = NestedTensor([torch.randn(2, 3, 4), torch.randn(2, 3, 4)])

    result = nested_tensor.transpose(1, 2).reshape(-1, 6).view(24, 1)
    assert result is not nested_tensor
    assert result.shape == torch.Size([2, 24, 1])
    assert nested_tensor.shape == torch.Size([2, 2, 3, 4])


def test_concat_equal_lengths():
    nested_tensor = NestedTensor([torch.tensor([1, 2]), torch.tensor([3, 4])])
    assert torch.equal(nested_tensor.concat, torch.tensor([1, 2, 3, 4]))


def test_concat_batch_first_false():
    nested_tensor = NestedTensor(
        [torch.arange(3).unsqueeze(1), torch.arange(3, 7).unsqueeze(1)],
        batch_first=False,
    )
    concat = nested_tensor.concat
    assert concat.shape == torch.Size([7, 1])
    assert torch.equal(concat.squeeze(1), torch.arange(7))


def test_bool_empty_nested_tensor():
    assert bool(NestedTensor([])) is False


def test_pn_tensor_concat():
    pn = pn_tensor([1, 2, 3])
    assert isinstance(pn, PNTensor)
    assert torch.equal(pn.concat, pn)
