import pytest
import torch

from danling.tensors import NestedTensor
from danling.tensors.functions import (
    NestedTensorFuncWrapper,
    TorchFuncRegistry,
    add,
    cat,
    div,
    flatten,
    gather,
    pow,
    scatter,
    stack,
    sub,
    unflatten,
    unsqueeze,
)


class TestTorchFuncRegistry:

    def test_duplicate_registration_raises(self):
        registry = TorchFuncRegistry()

        @registry.implement(torch.neg)
        def _neg(x):
            return -x

        with pytest.raises(ValueError):
            registry.implement(torch.neg)(_neg)


class TestNestedTensorFuncWrapper:

    def test_stack_returns_tensor_when_shapes_match(self, device, float_dtype):
        wrapper = NestedTensorFuncWrapper(lambda x: x + 1, lambda x: x + 2)
        data = torch.arange(2, device=device, dtype=float_dtype)
        out = wrapper(data)
        assert isinstance(out, torch.Tensor)
        assert torch.equal(out, torch.stack([data + 1, data + 2]))

    def test_stack_failure_returns_nested_tensor_with_state(self, device, float_dtype):
        wrapper = NestedTensorFuncWrapper(
            lambda: torch.ones(1, device=device, dtype=float_dtype),
            lambda: torch.ones(2, device=device, dtype=float_dtype),
            state={"device": device},
        )
        out = wrapper()
        assert isinstance(out, NestedTensor)
        assert out.device == device
        assert [t.numel() for t in out._storage] == [1, 2]

    def test_hashable_single_value_returns_element(self):
        wrapper = NestedTensorFuncWrapper(lambda: "a", lambda: "a")
        assert wrapper() == "a"


class TestArithmeticFunctions:

    def test_add_converts_tensor_to_nested(self, device, float_dtype):
        nt = NestedTensor([torch.ones(2, device=device, dtype=float_dtype)])
        tensor = torch.ones_like(nt.tensor)
        out = add(nt, tensor)
        assert isinstance(out, NestedTensor)
        assert torch.equal(out.tensor, nt.tensor + tensor)

    def test_div_converts_other_to_nested(self, device, float_dtype):
        nt = NestedTensor([torch.ones(2, device=device, dtype=float_dtype) * 4])
        other = torch.full_like(nt.tensor, 2, device=device, dtype=float_dtype)
        out = div(other, nt)
        assert isinstance(out, NestedTensor)
        assert torch.equal(out.tensor, other / nt.tensor)

    def test_pow_with_scalar_and_nested_exponent(self, device, float_dtype):
        base = NestedTensor([torch.arange(1.0, 3.0, device=device, dtype=float_dtype)])
        scalar_out = pow(base, 2)
        assert torch.equal(scalar_out.tensor, base.tensor.pow(2))
        exponent = torch.full_like(base.tensor, 2, device=device, dtype=float_dtype)
        nested_out = pow(base, exponent)
        assert torch.equal(nested_out.tensor, base.tensor.pow(exponent))

    def test_sub_converts_tensor_argument(self, device, float_dtype):
        nt = NestedTensor([torch.ones(2, device=device, dtype=float_dtype)])
        tensor = torch.zeros_like(nt.tensor)
        out = sub(tensor, nt)
        assert isinstance(out, NestedTensor)
        assert torch.equal(out.tensor, tensor - nt.tensor)


class TestCatFunction:

    def test_cat_dim0_preserves_state(self, device, float_dtype):
        nt = NestedTensor([torch.ones(2, device=device, dtype=float_dtype)])
        mixed = torch.zeros(2, device=device, dtype=float_dtype)
        out = cat((nt, mixed), dim=0)
        assert isinstance(out, NestedTensor)
        assert out.device == nt.device
        assert torch.equal(out._storage[0], nt._storage[0])
        assert torch.equal(out._storage[1], mixed)

    def test_cat_non_zero_dim_requires_nested_tensors(self):
        nt = NestedTensor([torch.ones(2)])
        with pytest.raises(NotImplementedError):
            cat((nt, torch.ones(2)), dim=1)

    def test_cat_non_zero_dim_length_mismatch(self):
        nt1 = NestedTensor([torch.ones(2)])
        nt2 = NestedTensor([torch.ones(2), torch.ones(3)])
        with pytest.raises(ValueError):
            cat((nt1, nt2), dim=1)


class TestStackFunction:

    def test_stack_requires_dim_zero(self):
        nt = NestedTensor([torch.ones(1)])
        with pytest.raises(NotImplementedError):
            stack([nt], dim=1)

    def test_stack_requires_nested_tensor_inputs(self):
        with pytest.raises(NotImplementedError):
            stack([torch.ones(1)])

    def test_stack_empty_sequence(self):
        with pytest.raises(ValueError):
            stack([])


class TestFlattenUnflattenFunctions:

    def test_flatten_start_dim_zero_returns_tensor(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([[1, 2], [3, 4]], device=device, dtype=float_dtype),
                torch.tensor([[5, 6], [7, 8]], device=device, dtype=float_dtype),
            ]
        )
        out = flatten(nt, start_dim=0)
        assert isinstance(out, torch.Tensor)
        assert torch.equal(out, torch.flatten(nt.tensor, start_dim=0))

    def test_unflatten_batch_dim_not_supported(self):
        nt = NestedTensor([torch.tensor([[1, 2]])])
        with pytest.raises(NotImplementedError):
            unflatten(nt, dim=0, sizes=(1, 2))


class TestDimensionTransformFunctions:

    def test_unsqueeze_batch_dim_raises(self):
        nt = NestedTensor([torch.tensor([1, 2])])
        with pytest.raises(ValueError):
            unsqueeze(nt, dim=0)

    def test_swapaxes_batch_dim_raises(self):
        nt = NestedTensor([torch.ones(2, 2)])
        with pytest.raises(ValueError):
            torch.swapaxes(nt, 0, 1)


class TestGatherScatterFunctions:

    def test_gather_on_batch_dim_not_supported(self):
        nt = NestedTensor([torch.tensor([[1, 2]])])
        idx = torch.zeros_like(nt.tensor, dtype=torch.long)
        with pytest.raises(NotImplementedError):
            gather(nt, 0, idx)

    def test_gather_with_tensor_index_converts(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.arange(6, device=device, dtype=float_dtype).view(1, 6),
                torch.arange(6, 12, device=device, dtype=float_dtype).view(1, 6),
            ]
        )
        index = torch.zeros_like(nt.tensor, dtype=torch.long)
        out = gather(nt, 1, index)
        refs = [torch.gather(t, 0, torch.zeros_like(t, dtype=torch.long)) for t in nt._storage]
        for o, r in zip(out._storage, refs):
            assert torch.equal(o, r)

    def test_scatter_with_tensor_index_and_src(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.zeros(2, 2, device=device, dtype=float_dtype),
                torch.zeros(2, 2, device=device, dtype=float_dtype),
            ]
        )
        index = torch.tensor(
            [[[0, 1], [0, 1]], [[1, 0], [1, 0]]],
            dtype=torch.long,
            device=device,
        )
        src = torch.ones_like(index, dtype=float_dtype, device=device)
        out = scatter(nt, 1, index, src)
        refs = [torch.scatter(t, 0, idx, src_slice) for t, idx, src_slice in zip(nt._storage, index, src)]
        for o, r in zip(out._storage, refs):
            assert torch.equal(o, r)
