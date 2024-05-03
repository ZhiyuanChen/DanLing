import random

import pytest
import torch

from danling.tensors import NestedTensor
from danling.tensors.nested_tensor import NestedTensorFuncWrapper


class Test:
    seed = 0
    epsilon = 1e-5
    nested_tensor = NestedTensor([[2, 3, 4], [5, 6]])
    random.seed(seed)
    torch.manual_seed(seed)

    def test_compare(self):
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
    def test_add_sub(self, i):
        a = self.nested_tensor.clone().float()
        b = a.clone()
        assert torch.sum((a + i) - (i + a)) < self.epsilon
        assert torch.sum((a - i) + (i - a)) < self.epsilon
        a += i
        b -= -i
        assert torch.sum(a - b) < self.epsilon

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
    def test_mul_truediv(self, i):
        a = self.nested_tensor.clone().float()
        b = a.clone()
        assert torch.sum(a * i - i / (1 / a)) < self.epsilon
        assert torch.sum(a / (1 / i) - i * a) < self.epsilon
        assert torch.sum(a * i - i * a) < self.epsilon
        a *= i
        b /= 1 / i
        assert torch.sum(a - b) < self.epsilon

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
    def test_pow_log(self, i):
        a = self.nested_tensor.clone().float()
        b = a.clone()
        assert torch.sum(torch.log(a**i) / torch.log(a) - i) < self.epsilon
        a **= i
        assert torch.sum(torch.log(a) / torch.log(b) - i) < self.epsilon

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
    def test_shift(self, i):
        a = self.nested_tensor.clone()
        # assert torch.sum(a << i >> i - a) < self.epsilon
        assert torch.sum(a >> i << i - a) < self.epsilon
        b = a.clone()
        b <<= i
        assert torch.sum(a << i - b) < self.epsilon
        b >>= i
        assert torch.sum(a - b) < self.epsilon

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
    def test_logic(self, i):
        a = self.nested_tensor.clone().int()
        assert torch.sum(a & i - i & a) < self.epsilon
        assert torch.sum((a | i) - (i | a)) < self.epsilon
        assert torch.sum(a ^ i - i ^ a) < self.epsilon
        assert torch.sum(~a & ~i - ~(+a | +i)) < self.epsilon
        assert torch.sum(~(+i | +a) - ~i & ~a) < self.epsilon
        b = a.clone() + 1
        assert torch.sum(((a & i) | (i & b)) - (i & (a | b))) < self.epsilon
        assert torch.sum(((i | a) & (b | i)) - (i | (a & b))) < self.epsilon
        assert torch.sum(((a ^ i) ^ b) - (a ^ (i ^ b))) < self.epsilon
        b = a.clone()
        b &= i
        assert torch.sum(a & i - b) < self.epsilon
        b = a.clone()
        b |= i
        assert torch.sum((a | i) - b) < self.epsilon
        b = a.clone()
        b ^= i
        assert torch.sum((a ^ i) - b) < self.epsilon

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
    def test_floordiv(self, i):
        a = self.nested_tensor.clone()
        a.padding_value = -1
        assert (a // i == a.tensor // i).all()
        assert (i // a == i // a.tensor).all()

    def test_ifloordiv(self):
        a = self.nested_tensor.clone()
        a.padding_value = -1
        a //= 1
        assert (a == self.nested_tensor.clone()).all()
        a //= self.nested_tensor.clone()
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
    def test_mod(self, i):
        a = self.nested_tensor.clone()
        a.padding_value = -1
        assert (a % i == a.tensor % i).all()
        assert (i % a == i % a.tensor).all()

    def test_imod(self):
        a = self.nested_tensor.clone()
        a %= NestedTensor([[6, 5, 4], [3, 2]])
        assert (a == NestedTensor([[2, 3, 0], [2, 0]])).all()
        a = self.nested_tensor.clone()
        a %= 2
        assert (a == NestedTensor([[0, 1, 0], [1, 0]])).all()
        a %= torch.ones_like(a.tensor)
        assert (a == torch.zeros(2, 3)).all()

    def test_arise(self):
        a = self.nested_tensor.clone()
        with pytest.raises(ValueError):
            _ = a[""]
        with pytest.raises(ValueError):
            _ = NestedTensor()
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
            temp.storage().clear()
            _ = temp.int()

    def test_0dim(self):
        length = 5
        nested_tensor = NestedTensor(range(length))
        assert len(nested_tensor) == length
        assert nested_tensor == torch.arange(length)
        assert (nested_tensor.tensor == torch.arange(length)).all()
        assert (nested_tensor.mask == torch.ones(length)).all()

    def test_1dim(self):
        lengths = [2, 3, 5, 7]
        additional_length = 11
        channels = 8
        nested_tensor = NestedTensor(torch.randn(length, channels) for length in lengths)
        assert nested_tensor.tensor.shape == torch.Size((len(lengths), max(lengths), channels))
        assert nested_tensor.mask.shape == torch.Size((len(lengths), max(lengths)))
        assert torch.sum(nested_tensor @ nested_tensor.T - nested_tensor.tensor @ nested_tensor.T) < self.epsilon
        lengths.append(additional_length)
        nested_tensor.storage().append(torch.randn(additional_length, channels))
        assert nested_tensor.tensor.shape == torch.Size((len(lengths), max(lengths), channels))
        assert nested_tensor.mask.shape == torch.Size((len(lengths), max(lengths)))
        assert torch.sum(nested_tensor.tensor @ nested_tensor.T - nested_tensor @ nested_tensor.T) < self.epsilon

    def test_torch_func(self):
        a = self.nested_tensor.clone()
        assert (torch.isin(a, a.tensor[0, 1]) == torch.isin(a.tensor, a.tensor[0, 1])).all()
        assert (torch.isin(a, a.tensor[0, 1]) == torch.isin(a.tensor, a.tensor[0, 1])).all()
        with pytest.raises(NotImplementedError):
            torch.stack([a])
        a = self.nested_tensor.clone().float()
        assert torch.sum(torch.mean(a) - a.mean()) < self.epsilon
        assert torch.sum(torch.sqrt(a) - a.sqrt()) < self.epsilon
        assert torch.sum(torch.log(a) - a.log()) < self.epsilon

    @pytest.mark.skipif(torch.__version__ < "2.1", reason="requires PyTorch 2.1 or higher")
    def test_where(self):
        a = self.nested_tensor.clone()
        assert (a.where(a > 3, -1.0) == NestedTensor([[-1.0, -1.0, 4.0], [5.0, 6.0]])).all()
        assert (a.where(a.tensor > 3, -1.0) == NestedTensor([[-1.0, -1.0, 4.0], [5.0, 6.0]])).all()
        assert (a.where(torch.tensor(False), 1) == NestedTensor([[1, 1, 1], [1, 1]])).all()
