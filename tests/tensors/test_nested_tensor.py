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
from tests.tensors.utils import assert_close

random.seed(0)
torch.manual_seed(0)

NT = NestedTensor


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


def test_compare_preserves_state():
    state = {"batch_first": False, "padding_value": -1, "mask_value": True, "pin_memory": True}
    left = NestedTensor([torch.tensor([[2, 0], [1, 0]])], **state)
    right = NestedTensor([torch.tensor([[1, 0], [1, 0]])], **state)
    output = left > right
    assert isinstance(output, NestedTensor)
    assert output.batch_first is False
    assert output.padding_value == -1
    assert output.mask_value is True
    assert output._pin_memory is True


def test_length_mismatch_equality_and_ops():
    shorter = NestedTensor([[1, 2]])
    longer = NestedTensor([[1, 2], [3]])

    assert torch.equal(shorter, longer) is False
    assert torch.equal(longer, shorter) is False
    assert torch.allclose(shorter, longer) is False
    with pytest.raises(ValueError):
        _ = shorter == longer
    with pytest.raises(ValueError):
        _ = torch.eq(shorter, longer)
    with pytest.raises(ValueError):
        _ = shorter + longer
    with pytest.raises(ValueError):
        _ = torch.add(shorter, longer)


def test_from_tensor_mask_high_dimensional():
    padded = torch.tensor(
        [
            [[1, 2, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]],
            [[9, 8, 7, 0], [6, 5, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=torch.bool,
    )
    nested = NestedTensor.from_tensor_mask(padded, mask)
    reference = NT(
        [
            torch.tensor([[1.0, 2.0], [3.0, 0.0]], dtype=torch.float32),
            torch.tensor([[9.0, 8.0, 7.0], [6.0, 5.0, 0.0]], dtype=torch.float32),
        ]
    )
    assert_close(nested, reference)

    padded_mask = ~mask
    nested_inverted = NestedTensor.from_tensor_mask(padded, padded_mask, mask_value=True)
    assert_close(nested_inverted, reference)


def test_from_tensor_mask_1d_trims_with_mask():
    padded = torch.tensor([1, 2, 0, 0])
    mask = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    nested = NestedTensor.from_tensor_mask(padded, mask)
    assert_close(nested, NT([torch.tensor([1, 2])]))
    padded_mask = ~mask
    nested_inverted = NestedTensor.from_tensor_mask(padded, padded_mask, mask_value=True)
    assert_close(nested_inverted, NT([torch.tensor([1, 2])]))


def test_from_tensor_mask_2d_batch_mismatch_raises():
    padded = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    with pytest.raises(ValueError, match="Tensor/mask batch dimension mismatch"):
        NestedTensor.from_tensor_mask(padded, mask)


def test_from_tensor_mask_batched_scalar():
    padded = torch.tensor([1.0, 2.0, 0.0])
    mask = torch.tensor([True, True, False])
    output = NestedTensor.from_tensor_mask(padded, mask, batched=True)
    reference = NT([torch.tensor(1.0), torch.tensor(2.0)])
    assert_close(output, reference)

    padded_mask = ~mask
    output = NestedTensor.from_tensor_mask(padded, padded_mask, mask_value=True, batched=True)
    assert_close(output, reference)


def test_torch_function_fallback_preserves_scalar_batch():
    nested_tensor = NestedTensor([torch.tensor(1.2), torch.tensor(2.3)])
    output = torch.ceil(nested_tensor)
    reference = NT([torch.tensor(2.0), torch.tensor(3.0)])
    assert_close(output, reference)


def test_concat_scalar_round_trip():
    nested_tensor = NestedTensor([torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)])
    assert_close(nested_tensor.concat, torch.tensor([1.0, 2.0, 3.0]))

    concat, shapes = nested_tensor.concatenate()
    output = NestedTensor.from_concatenated(concat, shapes, **nested_tensor._state)
    assert_close(output, nested_tensor)

    nested_tensor = NestedTensor([torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)], batch_first=False)
    assert_close(nested_tensor.concat, torch.tensor([1.0, 2.0, 3.0]))

    concat, shapes = nested_tensor.concatenate()
    output = NestedTensor.from_concatenated(concat, shapes, **nested_tensor._state)
    assert_close(output, nested_tensor)


def test_from_concatenated_raises_on_extra_elements():
    concat = torch.arange(4, dtype=torch.float32)
    shapes = (torch.Size([1, 1]), torch.Size([1, 1]))
    with pytest.raises(ValueError):
        NestedTensor.from_concatenated(concat, shapes)


def test_from_concatenated_identical_shapes_round_trip():
    nested_tensor = NestedTensor([torch.randn(3, 5), torch.randn(3, 5)])
    concat, shapes = nested_tensor.concatenate()
    reconstructed = NestedTensor.from_concatenated(concat, shapes, **nested_tensor._state)
    assert_close(reconstructed, nested_tensor)


def test_torch_all_consistency():
    nested_tensor = NestedTensor(torch.ones(2), torch.ones(3))
    assert nested_tensor.all()
    assert torch.all(nested_tensor)


def test_torch_any_consistency():
    nested_tensor = NestedTensor(torch.zeros(2), torch.ones(3))
    assert nested_tensor.any()
    assert torch.any(nested_tensor)
    all_zero = NestedTensor(torch.zeros(2), torch.zeros(3))
    assert not all_zero.any()
    assert not torch.any(all_zero)


def test_getitem_tuple_tensor_batch_index():
    nested_tensor = NestedTensor([torch.arange(3), torch.arange(5) + 10])
    batch_idx = torch.tensor([0, 1, 0])
    kv_idx = torch.tensor([0, 2, 1])
    assert torch.equal(nested_tensor[batch_idx, kv_idx], nested_tensor.tensor[batch_idx, kv_idx])


def test_getitem_tuple_slice_rest():
    nested_tensor = NestedTensor([torch.arange(3), torch.arange(5)])
    sliced = nested_tensor[:, 1:]
    assert isinstance(sliced, NestedTensor)
    assert torch.equal(sliced[0], torch.tensor([1, 2]))
    assert torch.equal(sliced[1], torch.tensor([1, 2, 3, 4]))


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
    assert_close((a + i), (i + a))
    assert_close((a - i), -(i - a))
    a += i
    b -= -i
    assert_close(a, b)


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
    assert_close(a * i, i / (1 / a))
    assert_close(a / (1 / i), i * a)
    assert_close(a * i, i * a)
    a *= i
    b /= 1 / i
    assert_close(a, b)


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
    assert_close(torch.log(a**i) / torch.log(a), i)
    a **= i
    assert_close(torch.log(a) / torch.log(b), i)


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
    assert_close(a << i >> i, a)
    assert_close(a >> i << i, a)
    b = a.clone()
    b <<= i
    assert_close(a << i, b)
    b >>= i
    assert_close(a, b)


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
    assert_close(a & i, i & a)
    assert_close((a | i), (i | a))
    assert_close(a ^ i, i ^ a)
    assert_close(~a & ~i, ~(+a | +i))
    assert_close(~(+i | +a), ~i & ~a)
    b = a.clone() + 1
    assert_close(((a & i) | (i & b)), (i & (a | b)))
    assert_close(((i | a) & (b | i)), (i | (a & b)))
    assert_close(((a ^ i) ^ b), (a ^ (i ^ b)))
    b = a.clone()
    b &= i
    assert_close(a & i, b)
    b = a.clone()
    b |= i
    assert_close(a | i, b)
    b = a.clone()
    b ^= i
    assert_close(a ^ i, b)


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
    assert_close(a // i, a.tensor // i)
    assert_close(i // a, i // a.tensor)


def test_ifloordiv():
    a = NestedTensor([[2, 3, 4], [5, 6]], dtype=torch.float32)
    b = a.clone()
    a.padding_value = -1
    a //= 1
    assert_close(a, b)
    a //= b
    assert_close(a, torch.ones(2, 3))
    a //= torch.ones(2, 3)
    assert_close(a, torch.ones(2, 3))


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
    assert_close(a % i, a.tensor % i)
    assert_close(i % a, i % a.tensor)


def test_imod():
    a = NestedTensor([[2, 3, 4], [5, 6]])
    a %= NestedTensor([[6, 5, 4], [3, 2]])
    assert_close(a, NestedTensor([[2, 3, 0], [2, 0]]))
    a = NestedTensor([[2, 3, 4], [5, 6]])
    a %= 2
    assert_close(a, NestedTensor([[0, 1, 0], [1, 0]]))
    a %= torch.ones_like(a.tensor)
    assert_close(a, torch.zeros_like(a.tensor))


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


def test_empty_nested_tensor_accessors():
    nested_tensor = NestedTensor([], dtype=torch.float32)
    assert nested_tensor.size() == torch.Size([0])
    assert nested_tensor.dim() == 1
    tensor, mask = nested_tensor.tensor_mask
    assert tensor.shape == torch.Size([0])
    assert mask.shape == torch.Size([0])
    assert nested_tensor.tensor.shape == torch.Size([0])
    assert nested_tensor.mask.shape == torch.Size([0])
    assert nested_tensor.occupancy == 0.0


def test_tensor_mask_does_not_squeeze_last_dim():
    nt = NestedTensor([torch.tensor([1]), torch.tensor([2])])
    tensor, mask = nt.tensor_mask
    assert tensor.shape == nt.tensor.shape
    assert mask.shape == nt.mask.shape == torch.Size([2, 1])

    nt_bf_false = NestedTensor([torch.tensor([1, 2, 3])], batch_first=False)
    tensor2, mask2 = nt_bf_false.tensor_mask
    assert tensor2.shape == nt_bf_false.tensor.shape
    assert mask2.shape == nt_bf_false.mask.shape == torch.Size([3, 1])


def test_1dim():
    lengths = [2, 3, 5, 7]
    additional_length = 11
    channels = 8
    nested_tensor = NestedTensor(torch.randn(length, channels) for length in lengths)
    tensor, mask = nested_tensor.tensor_mask
    assert tensor.shape == nested_tensor.tensor.shape == torch.Size((len(lengths), max(lengths), channels))
    assert mask.shape == nested_tensor.mask.shape == torch.Size((len(lengths), max(lengths)))
    assert_close(tensor @ nested_tensor.T, nested_tensor.tensor @ nested_tensor.T)
    lengths.append(additional_length)
    nested_tensor = torch.cat([nested_tensor, torch.randn(additional_length, channels)])
    tensor, mask = nested_tensor.tensor_mask
    assert nested_tensor.tensor.shape == torch.Size((len(lengths), max(lengths), channels))
    assert nested_tensor.mask.shape == torch.Size((len(lengths), max(lengths)))
    assert_close(nested_tensor.tensor @ nested_tensor.T, tensor @ nested_tensor.T)


def test_torch_func():
    a = NestedTensor([[2, 3, 4], [5, 6]])
    assert_close(torch.isin(a, a.tensor[0, 1]), torch.isin(a.tensor, a.tensor[0, 1]))
    assert_close(torch.isin(a, a.tensor[0, 1]), torch.isin(a.tensor, a.tensor[0, 1]))
    a = NestedTensor([[2, 3, 4], [5, 6]]).float()
    assert_close(torch.mean(a), a.mean())
    assert_close(torch.sqrt(a), a.sqrt())
    assert_close(torch.log(a), a.log())
    assert torch.numel(a) == a.numel()


@pytest.mark.skipif(torch.__version__ < "2.1", reason="requires PyTorch 2.1 or higher")
def test_where():
    a = NestedTensor([[2, 3, 4], [5, 6]])
    assert_close(a.where(a > 3, -1.0), NT([[-1.0, -1.0, 4.0], [5.0, 6.0]]))
    assert_close(a.where(a.tensor > 3, -1.0), NT([[-1.0, -1.0, 4.0], [5.0, 6.0]]))
    assert_close(a.where(torch.tensor(False), 1), NT([[1, 1, 1], [1, 1]]))


def test_permute():
    nested_tensor = NestedTensor([torch.randn(3, 4, 5), torch.randn(2, 4, 5)])
    original_shape = nested_tensor.shape
    assert original_shape == torch.Size([2, 3, 4, 5])

    permuted = nested_tensor.permute(0, 3, 1, 2)
    assert permuted.shape == torch.Size([2, 5, 3, 4])
    assert permuted is not nested_tensor

    assert permuted[0].shape == torch.Size([5, 3, 4])
    assert permuted[1].shape == torch.Size([5, 2, 4])
    assert nested_tensor.shape == torch.Size([2, 3, 4, 5])

    nested_tensor2 = NestedTensor([torch.randn(3, 4), torch.randn(2, 4)])
    permuted2 = nested_tensor2.permute(0, -1, -2)
    assert permuted2.shape == torch.Size([2, 4, 3])

    with pytest.raises(ValueError, match="Expected 3 dimensions"):
        nested_tensor2.permute(0, 1)

    nested_tensor3 = NestedTensor([torch.randn(3, 4), torch.randn(2, 4)])
    with pytest.raises(ValueError, match="Invalid permutation dims .* for shape with . dims"):
        nested_tensor3.permute(1, 2, -1)

    with pytest.raises(ValueError, match="batch dimension"):
        nested_tensor.permute(1, 0, 2, 3)


def test_transpose():
    nested_tensor = NestedTensor([torch.randn(3, 4), torch.randn(2, 4)])
    original_shape = nested_tensor.shape
    assert original_shape == torch.Size([2, 3, 4])

    transposed = nested_tensor.transpose(1, 2)
    assert transposed.shape == torch.Size([2, 4, 3])
    assert transposed is not nested_tensor

    assert transposed[0].shape == torch.Size([4, 3])
    assert transposed[1].shape == torch.Size([4, 2])
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

    assert reshaped[0].shape == torch.Size([4])
    assert reshaped[1].shape == torch.Size([4])
    assert nested_tensor.shape == torch.Size([2, 2, 2])

    nested_tensor2 = NestedTensor([torch.randn(2, 3, 4), torch.randn(2, 3, 4)])
    reshaped2 = nested_tensor2.reshape(len(nested_tensor2), -1, 4)
    assert reshaped2.shape == torch.Size([2, 6, 4])

    empty_nested = NestedTensor([])
    output = empty_nested.reshape(5)
    assert output is not empty_nested
    assert len(output) == 0


def test_view():
    nested_tensor = NestedTensor([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])])
    original_shape = nested_tensor.shape
    assert original_shape == torch.Size([2, 2, 2])

    viewed = nested_tensor.view(4)
    assert viewed.shape == torch.Size([2, 4])
    assert viewed is not nested_tensor

    assert viewed[0].shape == torch.Size([4])
    assert viewed[1].shape == torch.Size([4])
    assert nested_tensor.shape == torch.Size([2, 2, 2])

    nested_tensor2 = NestedTensor([torch.randn(2, 6), torch.randn(2, 6)])
    viewed2 = nested_tensor2.view(len(nested_tensor2), -1, 3)
    assert viewed2.shape == torch.Size([2, 4, 3])

    nested_tensor3 = NestedTensor([torch.randn(4), torch.randn(4)])
    viewed3 = nested_tensor3.view(2, 2)
    viewed3[0][0, 0] = 999
    assert nested_tensor3[0][0] == 999

    empty_nested = NestedTensor([])
    output = empty_nested.view(5)
    assert output is not empty_nested
    assert len(output) == 0


def test_view_with_different_shapes():
    nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])

    viewed = nested_tensor.view(len(nested_tensor), -1)
    assert viewed is not nested_tensor
    assert viewed[0].shape == torch.Size([3])
    assert viewed[1].shape == torch.Size([2])


def test_view_with_batch_dim_and_dynamic_lengths():
    nested_tensor = NestedTensor([torch.randn(3, 640), torch.randn(5, 640)])
    target_shape = nested_tensor.size()[:-1] + (20, 32)
    viewed = nested_tensor.view(*target_shape)
    padded = nested_tensor.tensor
    assert viewed.shape == torch.Size([2, 5, 20, 32])
    assert viewed._storage[0].shape == torch.Size([3, 20, 32])
    assert viewed._storage[1].shape == torch.Size([5, 20, 32])
    assert torch.equal(viewed, padded.view(*target_shape))


def test_view_with_explicit_batch_and_reduced_rank():
    nested_tensor = NestedTensor([torch.randn(3, 4), torch.randn(3, 4)])
    viewed = nested_tensor.view(len(nested_tensor), -1)
    assert viewed.shape == torch.Size([2, 12])
    assert viewed._storage[0].shape == torch.Size([12])
    assert viewed._storage[1].shape == torch.Size([12])


def test_view_with_inserted_dim_before_dynamic_length():
    nested_tensor = NestedTensor([torch.randn(3, 8), torch.randn(5, 8)])
    target_shape = (len(nested_tensor), 1, nested_tensor.size(1), nested_tensor.size(2))
    viewed = nested_tensor.view(*target_shape)
    assert viewed.shape == torch.Size([2, 1, 5, 8])
    assert viewed._storage[0].shape == torch.Size([1, 3, 8])
    assert viewed._storage[1].shape == torch.Size([1, 5, 8])


def test_method_chaining():
    nested_tensor = NestedTensor([torch.randn(2, 3, 4), torch.randn(2, 3, 4)])

    output = nested_tensor.transpose(1, 2).reshape(len(nested_tensor), -1, 6).view(24, 1)
    assert output is not nested_tensor
    assert output.shape == torch.Size([2, 24, 1])
    assert nested_tensor.shape == torch.Size([2, 2, 3, 4])


def test_concat_equal_lengths():
    nested_tensor = NestedTensor([torch.tensor([1, 2]), torch.tensor([3, 4])])
    assert_close(nested_tensor.concat, torch.tensor([1, 2, 3, 4]))


def test_sum_with_list_dim_matches_int():
    nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
    reference = nested_tensor.sum(dim=0)
    assert_close(nested_tensor.sum(dim=[0]), reference)


def test_from_concatenated_round_trip_multidim():
    nested_tensor = NestedTensor([torch.randn(2, 3, 4), torch.randn(2, 3, 4)])
    concat, shapes = nested_tensor.concatenate()
    output = NestedTensor.from_concatenated(concat, shapes)
    assert_close(output, nested_tensor)


def test_from_concatenated_round_trip_mixed_shapes():
    nested_tensor = NestedTensor([torch.randn(2, 3, 4), torch.randn(1, 3, 4)])
    concat, shapes = nested_tensor.concatenate()
    output = NestedTensor.from_concatenated(concat, shapes)
    assert_close(output, nested_tensor)


def test_from_tensor_mask_ndim3():
    padded = torch.tensor([[[1, 2, 0], [3, 0, 0]]])
    mask = torch.tensor([[[1, 1, 0], [1, 0, 0]]], dtype=torch.bool)
    output = NestedTensor.from_tensor_mask(padded, mask)
    assert output.tensor.shape == torch.Size([1, 2, 2])
    reference = NT([torch.tensor([[1, 2], [3, 0]])])
    assert_close(output, reference)


def test_from_tensor_mask_channel_preserved():
    padded = torch.tensor([[[[1], [0]], [[2], [0]]]])
    mask = torch.tensor([[[1, 0], [1, 0]]], dtype=torch.bool)
    output = NestedTensor.from_tensor_mask(padded, mask)
    assert output.tensor.shape == torch.Size([1, 2, 1, 1])
    reference = NT([torch.tensor([[[1]], [[2]]])])
    assert_close(output, reference)


def test_concat_batch_first_false():
    nested_tensor = NestedTensor(
        [torch.arange(3).unsqueeze(1), torch.arange(3, 7).unsqueeze(1)],
        batch_first=False,
    )
    concat = nested_tensor.concat
    assert concat.shape == torch.Size([7, 1])
    assert_close(concat.squeeze(1), torch.arange(7))


def test_size_and_nested_like_batch_first_false():
    tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8], [9, 10]])]
    nested_tensor = NestedTensor(tensors, batch_first=False)

    assert nested_tensor.size() == torch.Size([3, 2, 2])
    assert nested_tensor.size(0) == 3  # max length dimension
    assert nested_tensor.size(1) == len(tensors)  # batch dimension when batch_first is False

    cloned = nested_tensor.nested_like(nested_tensor.tensor)
    assert_close(cloned, nested_tensor)


def test_bool_empty_nested_tensor():
    assert bool(NestedTensor([])) is False


def test_pn_tensor_concat():
    pn = pn_tensor([1, 2, 3])
    assert isinstance(pn, PNTensor)
    assert_close(pn.concat, pn)


def test_method_dtype_change_preserves_state():
    nested_tensor = NestedTensor(
        [torch.tensor([1, 2], dtype=torch.int64), torch.tensor([3], dtype=torch.int64)],
        batch_first=False,
        padding_value=-1,
        mask_value=True,
    )
    floated = nested_tensor.float()

    assert isinstance(floated, NestedTensor)
    assert floated.dtype == torch.float32
    assert all(t.dtype == torch.float32 for t in floated)
    assert floated.batch_first is False
    assert floated.padding_value == -1
    assert floated.mask_value is True


def test_to_accepts_nested_tensor():
    nested_tensor = NestedTensor(
        [torch.tensor([1, 2], dtype=torch.float32), torch.tensor([3], dtype=torch.float32)],
        batch_first=False,
        padding_value=-1,
        mask_value=True,
    )
    other = NestedTensor([torch.tensor([1], dtype=torch.float64)])
    output = nested_tensor.to(other, non_blocking=True)

    assert output.dtype == other.dtype
    assert all(t.dtype == other.dtype for t in output)
    assert output.device == other.device
    assert output.batch_first is False
    assert output.padding_value == -1
    assert output.mask_value is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to test device movement")
def test_method_device_change_moves_nested_tensor_to_cuda():
    nested_tensor = NestedTensor(
        [torch.tensor([1, 2]), torch.tensor([3])],
        batch_first=False,
        padding_value=-1,
        mask_value=True,
    )
    moved = nested_tensor.cuda()

    assert isinstance(moved, NestedTensor)
    assert moved.device.type == "cuda"
    assert all(t.device.type == "cuda" for t in moved)
    assert moved.batch_first is False
    assert moved.padding_value == -1
    assert moved.mask_value is True


def test_getitem_preserves_state():
    nested_tensor = NestedTensor(
        [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])],
        batch_first=False,
        padding_value=-1,
        mask_value=True,
    )

    tuple_indexed = nested_tensor[:, 0]
    assert isinstance(tuple_indexed, NestedTensor)
    assert tuple_indexed.batch_first is False
    assert tuple_indexed.padding_value == -1
    assert tuple_indexed.mask_value is True

    boolean_indexed = nested_tensor[nested_tensor.tensor > 0]
    assert boolean_indexed.batch_first is False
    assert boolean_indexed.padding_value == -1
    assert boolean_indexed.mask_value is True


def test_tuple_getitem_with_leading_int():
    nested_tensor = NestedTensor([torch.tensor([1, 2]), torch.tensor([3])])
    assert nested_tensor[0, 0].item() == 1
    assert_close(nested_tensor[1, ...], torch.tensor([3]))


def test_getitem_nested_index_length_mismatch_raises():
    nested_tensor = NestedTensor([torch.tensor([1, 2]), torch.tensor([3])])
    nested_index = NestedTensor([torch.tensor([0])])
    with pytest.raises(ValueError, match="batch length mismatch"):
        _ = nested_tensor[nested_index]


def test_mixed_dtype_inputs_promote_to_common_dtype():
    nested_tensor = NestedTensor([torch.tensor([1], dtype=torch.int64), torch.tensor([1.5], dtype=torch.float32)])
    assert nested_tensor.dtype == torch.float32
    assert all(t.dtype == torch.float32 for t in nested_tensor)
    assert nested_tensor.tensor.dtype == torch.float32
    assert_close(nested_tensor.tensor, torch.tensor([[1.0], [1.5]], dtype=torch.float32))


def test_concatenate_empty_preserves_dtype():
    nested_tensor = NestedTensor([], dtype=torch.float64)
    concat, shapes = nested_tensor.concatenate()
    assert concat.dtype == torch.float64
    assert concat.device == torch.device("cpu")
    assert shapes == ()


def test_unary_ops_preserve_state():
    nested_tensor = NestedTensor(
        [torch.tensor([[1, -2], [3, -4]]), torch.tensor([[5]])],
        batch_first=False,
        padding_value=-1,
        mask_value=True,
        pin_memory=True,
    )

    for op in (lambda x: +x, lambda x: -x, lambda x: ~x):
        output = op(nested_tensor)
        assert output.batch_first is False
        assert output.padding_value == -1
        assert output.mask_value is True
        assert output._pin_memory is True  # state should be carried over


def test_where_preserves_state_with_scalar_inputs():
    nested_tensor = NestedTensor(
        [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5]])],
        batch_first=False,
        padding_value=-1,
        mask_value=True,
    )

    output = nested_tensor.where(torch.tensor(True), 0)
    assert output.batch_first is False
    assert output.padding_value == -1
    assert output.mask_value is True
    assert_close(output, nested_tensor)

    output = nested_tensor.where(torch.tensor(False), -1)
    assert output.batch_first is False
    assert output.padding_value == -1
    assert output.mask_value is True
    assert_close(output, torch.full_like(nested_tensor.tensor, -1))


def test_single_tensor_not_unbound():
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    nested_tensor = NestedTensor(tensor)

    assert len(nested_tensor) == 1
    assert_close(nested_tensor[0], tensor)
    assert nested_tensor.shape == torch.Size([1, 2, 3])
    assert_close(nested_tensor.tensor, tensor.unsqueeze(0))


def test_torch_dispatch_allows_none_kwargs():
    nested_tensor = NestedTensor([[1, 2]])
    output = NestedTensor.__torch_dispatch__(torch.add, (NestedTensor,), (nested_tensor, nested_tensor), None)
    assert_close(output, nested_tensor.tensor + nested_tensor.tensor)


def test_sum_multi_dim_batch_first_false_handles_padding():
    nested_tensor = NestedTensor(
        [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[7.0, 8.0, 9.0]])],
        batch_first=False,
        padding_value=5.0,
    )
    output = nested_tensor.sum(dim=[0, 2])
    reference = torch.tensor([21.0, 24.0])
    assert_close(output, reference)
