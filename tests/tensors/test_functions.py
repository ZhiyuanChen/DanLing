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
from tests.tensors.utils import assert_close

NT = NestedTensor


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
        output = wrapper(data)
        assert isinstance(output, torch.Tensor)
        assert_close(output, torch.stack([data + 1, data + 2]))

    def test_stack_failure_returns_nested_tensor_with_state(self, device, float_dtype):
        wrapper = NestedTensorFuncWrapper(
            lambda: torch.ones(1, device=device, dtype=float_dtype),
            lambda: torch.ones(2, device=device, dtype=float_dtype),
            state={"device": device},
        )
        output = wrapper()
        assert isinstance(output, NestedTensor)
        assert output.device == device
        assert [t.numel() for t in output] == [1, 2]

    def test_hashable_single_value_returns_element(self):
        wrapper = NestedTensorFuncWrapper(lambda: "a", lambda: "a")
        assert wrapper() == "a"

    def test_single_callable_is_supported(self):
        wrapper = NestedTensorFuncWrapper(lambda x: x + 1)
        assert wrapper(1) == 2


class TestArithmeticFunctions:

    def test_add_converts_tensor_to_nested(self, device, float_dtype):
        nt = NestedTensor([torch.ones(2, device=device, dtype=float_dtype)])
        tensor = torch.ones_like(nt.tensor)
        output = add(nt, tensor)
        assert isinstance(output, NestedTensor)
        assert_close(output, nt.tensor + tensor)

    def test_div_converts_other_to_nested(self, device, float_dtype):
        nt = NestedTensor([torch.ones(2, device=device, dtype=float_dtype) * 4])
        other = torch.full_like(nt.tensor, 2, device=device, dtype=float_dtype)
        output = div(other, nt)
        assert isinstance(output, NestedTensor)
        assert_close(output, other / nt.tensor)

    def test_pow_with_scalar_and_nested_exponent(self, device, float_dtype):
        base = NestedTensor([torch.arange(1.0, 3.0, device=device, dtype=float_dtype)])
        output = pow(base, 2)
        reference = base.tensor.pow(2)
        assert_close(output, reference)
        exponent = torch.full_like(base.tensor, 2, device=device, dtype=float_dtype)
        output = pow(base, exponent)
        reference = base.tensor.pow(exponent)
        assert_close(output, reference)

    def test_sub_converts_tensor_argument(self, device, float_dtype):
        nt = NestedTensor([torch.ones(2, device=device, dtype=float_dtype)])
        tensor = torch.zeros_like(nt.tensor)
        output = sub(tensor, nt)
        assert isinstance(output, NestedTensor)
        assert_close(output, tensor - nt.tensor)

    def test_wrapped_ops_preserve_state(self):
        nt = NestedTensor(
            [torch.tensor([1, 2]), torch.tensor([3])],
            batch_first=False,
            padding_value=-5,
            mask_value=True,
        )
        output = torch.add(nt, 1)
        assert output.batch_first is False
        assert output.padding_value == -5
        assert output.mask_value is True
        reference = NT([t + 1 for t in nt], **nt._state)
        assert_close(output, reference)


class TestCatFunction:

    def test_cat_dim0_preserves_state(self, device, float_dtype):
        nt = NestedTensor([torch.ones(2, device=device, dtype=float_dtype)])
        mixed = torch.zeros(2, device=device, dtype=float_dtype)
        output = cat((nt, mixed), dim=0)
        assert isinstance(output, NestedTensor)
        assert output.device == nt.device
        reference = NT([nt[0], mixed], **nt._state)
        assert_close(output, reference)

    def test_cat_non_zero_dim_requires_nested_tensors(self):
        nt = NestedTensor([torch.ones(2)])
        with pytest.raises(NotImplementedError):
            cat((nt, torch.ones(2)), dim=1)

    def test_cat_non_zero_dim_length_mismatch(self):
        nt1 = NestedTensor([torch.ones(2)])
        nt2 = NestedTensor([torch.ones(2), torch.ones(3)])
        with pytest.raises(ValueError):
            cat((nt1, nt2), dim=1)

    def test_cat_batch_first_false_batch_dim(self):
        nt = NestedTensor([torch.arange(3).unsqueeze(1), torch.arange(3, 5).unsqueeze(1)], batch_first=False)
        extra = torch.arange(5, 8).unsqueeze(1)
        output = torch.cat((nt, extra), dim=1)
        reference = NT([*nt, extra], **nt._state)
        assert_close(output, reference)
        assert output.batch_first is False

    def test_cat_batch_first_false_non_batch_dims(self):
        nt1 = NestedTensor([torch.ones(2, 2), torch.ones(3, 2)], batch_first=False)
        nt2_seq = NestedTensor([torch.zeros(1, 2), torch.zeros(2, 2)], batch_first=False)
        nt2_feat = NestedTensor([torch.zeros(2, 1), torch.zeros(3, 1)], batch_first=False)

        output_seq = torch.cat((nt1, nt2_seq), dim=0)
        reference_seq = NT([torch.cat([a, b], dim=0) for a, b in zip(nt1, nt2_seq)], **nt1._state)
        assert_close(output_seq, reference_seq)

        output_feat = torch.cat((nt1, nt2_feat), dim=2)
        reference_feat = NT([torch.cat([a, b], dim=1) for a, b in zip(nt1, nt2_feat)], **nt1._state)
        assert_close(output_feat, reference_feat)


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
        output = flatten(nt, start_dim=0)
        assert isinstance(output, torch.Tensor)
        assert_close(output, torch.flatten(nt.tensor, start_dim=0))

    def test_unflatten_batch_dim_not_supported(self):
        nt = NestedTensor([torch.tensor([[1, 2]])])
        with pytest.raises(NotImplementedError):
            unflatten(nt, dim=0, sizes=(1, 2))


class TestDimensionTransformFunctions:

    def test_unsqueeze_batch_dim_raises(self):
        nt = NestedTensor([torch.tensor([1, 2])])
        with pytest.raises(ValueError):
            unsqueeze(nt, dim=0)

    def test_unsqueeze_batch_first_false_respects_batch_dim(self):
        nt = NestedTensor([torch.tensor([1, 2])], batch_first=False)
        with pytest.raises(ValueError):
            torch.unsqueeze(nt, dim=0)
        with pytest.raises(ValueError):
            torch.unsqueeze(nt, dim=1)
        output = torch.unsqueeze(nt, dim=2)
        assert isinstance(output, NestedTensor)
        assert output.batch_first is False
        assert output.tensor.shape == torch.Size([2, 1, 1])

    def test_swapaxes_batch_dim_raises(self):
        nt = NestedTensor([torch.ones(2, 2)])
        with pytest.raises(ValueError):
            torch.swapaxes(nt, 0, 1)

    def test_torch_permute_accepts_dims_sequence(self, device, float_dtype):
        nt = NestedTensor([torch.randn(2, 3, 4, device=device, dtype=float_dtype)])
        dims = (0, 3, 2, 1)
        reference = nt.permute(*dims)
        output = torch.permute(nt, dims)
        assert_close(output, reference)
        assert output.shape == reference.shape


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
        output = gather(nt, 1, index)
        reference = NT([torch.gather(t, 0, torch.zeros_like(t, dtype=torch.long)) for t in nt], **nt._state)
        assert_close(output, reference)

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
        output = scatter(nt, 1, index, src)
        reference = NT([torch.scatter(t, 0, idx, src_slice) for t, idx, src_slice in zip(nt, index, src)], **nt._state)
        assert_close(output, reference)

    def test_gather_batch_first_false_respects_batch_dim(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.arange(6, device=device, dtype=float_dtype).view(2, 3),
                torch.arange(6, 12, device=device, dtype=float_dtype).view(3, 2),
            ],
            batch_first=False,
        )
        index = torch.zeros_like(nt.tensor, dtype=torch.long)
        output = gather(nt, 0, index)
        reference = NT([torch.gather(t, 0, torch.zeros_like(t, dtype=torch.long)) for t in nt], **nt._state)
        assert_close(output, reference)

    def test_scatter_batch_first_false_respects_batch_dim(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.zeros(2, 2, device=device, dtype=float_dtype),
                torch.zeros(3, 2, device=device, dtype=float_dtype),
            ],
            batch_first=False,
        )
        index = torch.zeros_like(nt.tensor, dtype=torch.long)
        src = torch.ones_like(nt.tensor, dtype=float_dtype)
        output = scatter(nt, 2, index, src)
        reference = NT(
            [
                torch.scatter(
                    t,
                    1,
                    torch.zeros_like(t, dtype=torch.long),
                    torch.ones_like(t, dtype=float_dtype),
                )
                for t in nt
            ],
            **nt._state,
        )
        assert_close(output, reference)


class TestAdditionalFunctions:

    def test_neg(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, -2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([-4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.neg(nt)
        reference = NT([torch.neg(t) for t in nt], **nt._state)
        assert_close(output, reference)

    def test_maximum_and_minimum(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, -2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([-4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        scalar = torch.tensor(0.0, device=device, dtype=float_dtype)
        output = torch.maximum(nt, scalar)
        reference = NT([torch.maximum(t, scalar) for t in nt], **nt._state)
        assert_close(output, reference)

        output = torch.minimum(nt, scalar)
        reference = NT([torch.minimum(t, scalar) for t in nt], **nt._state)
        assert_close(output, reference)

        output = torch.maximum(nt, nt.tensor)
        assert_close(output, nt)

    def test_where_broadcasts_scalar_condition(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 2, 3], device=device, dtype=float_dtype),
                torch.tensor([4, 5], device=device, dtype=float_dtype),
            ]
        )
        output = torch.where(nt > 2, nt, 0.0)
        assert isinstance(output, NestedTensor)
        reference = NT(
            [
                torch.tensor([0.0, 0.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([4.0, 5.0], device=device, dtype=float_dtype),
            ],
            **nt._state,
        )
        assert_close(output, reference)

    def test_where_length_mismatch_raises(self):
        cond = NestedTensor([torch.tensor([True, False])])
        input_nt = NestedTensor([torch.tensor([1, 2]), torch.tensor([3])])
        with pytest.raises(ValueError, match="batch length mismatch"):
            _ = torch.where(cond, input_nt, 0)

    def test_argmax_and_argmin_default_dim(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 5, 3], device=device, dtype=float_dtype),
                torch.tensor([4, 2], device=device, dtype=float_dtype),
            ]
        )
        output = torch.argmax(nt)
        reference = torch.tensor([1, 0], device=device)
        assert_close(output, reference)
        output = torch.argmin(nt)
        reference = torch.tensor([0, 1], device=device)
        assert_close(output, reference)

    def test_topk_and_sort(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 3, 2], device=device, dtype=float_dtype),
                torch.tensor([4, 0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.topk(nt, k=2, dim=1)
        assert isinstance(output[0], NestedTensor)
        reference = (
            NestedTensor(
                [
                    torch.tensor([3, 2], device=device, dtype=float_dtype),
                    torch.tensor([4, 0], device=device, dtype=float_dtype),
                ],
                **nt._state,
            ),
            NestedTensor([torch.tensor([1, 2], device=device), torch.tensor([0, 1], device=device)], **nt._state),
        )
        assert_close(output[0], reference[0])
        assert_close(output[1], reference[1])
        output = torch.sort(nt, dim=1, descending=False)
        reference = (
            torch.tensor([[1, 2, 3], [0, 4, 0]], device=device, dtype=float_dtype),
            output[1],
        )
        assert_close(output[0], reference[0])
        assert_close(torch.argsort(nt), reference[1])

    def test_cumulative_functions(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 2, 3], device=device, dtype=float_dtype),
                torch.tensor([4, 5], device=device, dtype=float_dtype),
            ]
        )
        output = torch.cumsum(nt, dim=1)
        reference = torch.tensor([[1, 3, 6], [4, 9, 0]], device=device, dtype=float_dtype)
        assert_close(output, reference)
        output = torch.cumprod(nt, dim=1)
        reference = torch.tensor([[1, 2, 6], [4, 20, 0]], device=device, dtype=float_dtype)
        assert_close(output, reference)

    def test_cumulative_functions_batch_dim_not_supported(self, device, float_dtype):
        nt = NestedTensor([torch.tensor([1.0, 2.0], device=device, dtype=float_dtype)])
        with pytest.raises(NotImplementedError):
            torch.cumsum(nt, dim=0)
        with pytest.raises(NotImplementedError):
            torch.cumprod(nt, dim=0)

        nt_bf_false = NestedTensor([torch.tensor([1.0, 2.0], device=device, dtype=float_dtype)], batch_first=False)
        with pytest.raises(NotImplementedError):
            torch.cumsum(nt_bf_false, dim=1)
        with pytest.raises(NotImplementedError):
            torch.cumprod(nt_bf_false, dim=1)

    def test_addcmul_and_addcdiv(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.addcmul(nt, nt, nt, value=2)
        reference = NT([torch.addcmul(t, t, t, value=2) for t in nt], **nt._state)
        assert_close(output, reference)

        output = torch.addcmul(nt.tensor, nt, nt, value=2)
        assert_close(output, reference)

        output = torch.addcdiv(nt, nt, nt + 1, value=2)
        reference = NT([torch.addcdiv(t, t, t + 1, value=2) for t in nt], **nt._state)
        assert_close(output, reference)

    def test_clamp_min_and_clamp_max(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, -2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([-4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.clamp_min(nt, 0.0)
        reference = NT([torch.clamp_min(t, 0.0) for t in nt], **nt._state)
        assert_close(output, reference)

        output = torch.clamp_max(nt, 0.0)
        reference = NT([torch.clamp_max(t, 0.0) for t in nt], **nt._state)
        assert_close(output, reference)

    def test_logical_ops_accept_python_scalars(self, device):
        nt = NestedTensor([torch.tensor([True, False], device=device), torch.tensor([True], device=device)])
        true_tensor = torch.tensor(True, device=device)
        false_tensor = torch.tensor(False, device=device)
        output = torch.logical_and(nt, true_tensor)
        assert_close(output, nt)

        output = torch.logical_or(nt, false_tensor)
        assert_close(output, nt)

        output = torch.logical_xor(nt, true_tensor)
        reference = NT([torch.logical_xor(t, true_tensor) for t in nt], **nt._state)
        assert_close(output, reference)

        output = torch.logical_or(nt.tensor, nt)
        assert_close(output, nt)

    def test_bitwise_ops(self, device):
        nt = NestedTensor(
            [
                torch.tensor([1, 2, 3], device=device, dtype=torch.int64),
                torch.tensor([4, 5], device=device, dtype=torch.int64),
            ]
        )
        output = torch.bitwise_and(nt, 1)
        reference = NT([torch.bitwise_and(t, 1) for t in nt], **nt._state)
        assert_close(output, reference)

        output = torch.bitwise_or(nt, 1)
        reference = NT([torch.bitwise_or(t, 1) for t in nt], **nt._state)
        assert_close(output, reference)

        output = torch.bitwise_xor(nt, 1)
        reference = NT([torch.bitwise_xor(t, 1) for t in nt], **nt._state)
        assert_close(output, reference)

    def test_isfinite_isnan_nan_to_num(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([float("nan"), float("inf"), -float("inf")], device=device, dtype=float_dtype),
                torch.tensor([0.0, float("nan")], device=device, dtype=float_dtype),
            ]
        )
        output = torch.isnan(nt)
        reference = NT([torch.isnan(t) for t in nt], **nt._state)
        assert_close(output, reference)

        output = torch.isfinite(nt)
        reference = NT([torch.isfinite(t) for t in nt], **nt._state)
        assert_close(output, reference)

        output = torch.nan_to_num(nt, nan=0.0, posinf=1.0, neginf=-1.0)
        reference = NT([torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0) for t in nt], **nt._state)
        assert_close(output, reference)

    def test_scatter_add(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.zeros(2, 3, device=device, dtype=float_dtype),
                torch.zeros(1, 3, device=device, dtype=float_dtype),
            ]
        )
        index = torch.zeros_like(nt.tensor, dtype=torch.long)
        src = torch.ones_like(nt.tensor, dtype=float_dtype)
        output = torch.scatter_add(nt, 1, index, src)
        reference = NT(
            [
                torch.scatter_add(t, 0, torch.zeros_like(t, dtype=torch.long), torch.ones_like(t, dtype=float_dtype))
                for t in nt
            ],
            **nt._state,
        )
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "scatter_reduce"), reason="requires torch.scatter_reduce")
    def test_scatter_reduce_sum(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.zeros(2, 3, device=device, dtype=float_dtype),
                torch.zeros(1, 3, device=device, dtype=float_dtype),
            ]
        )
        index = torch.zeros_like(nt.tensor, dtype=torch.long)
        src = torch.ones_like(nt.tensor, dtype=float_dtype)
        output = torch.scatter_reduce(nt, 1, index, src, reduce="sum", include_self=True)
        reference = NT(
            [
                torch.scatter_reduce(
                    t,
                    0,
                    torch.zeros_like(t, dtype=torch.long),
                    torch.ones_like(t, dtype=float_dtype),
                    reduce="sum",
                    include_self=True,
                )
                for t in nt
            ],
            **nt._state,
        )
        assert_close(output, reference)

    def test_index_select(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device, dtype=float_dtype),
                torch.tensor([[7.0, 8.0, 9.0]], device=device, dtype=float_dtype),
            ]
        )
        output = torch.index_select(nt, 0, torch.tensor([1, 0], device=device))
        reference = NT([nt[1], nt[0]], **nt._state)
        assert_close(output, reference)

        output = torch.index_select(nt, 2, torch.tensor([2, 0], device=device))
        reference = NT([torch.index_select(t, 1, torch.tensor([2, 0], device=device)) for t in nt], **nt._state)
        assert_close(output, reference)

    def test_masked_select(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        mask = nt > 2
        output = torch.masked_select(nt, mask)
        reference = NT([torch.masked_select(t, m) for t, m in zip(nt, mask)], **nt._state)
        assert_close(output, reference)

        output = torch.masked_select(nt, mask.tensor)
        assert_close(output, reference)

        output = torch.masked_select(nt, torch.tensor(True, device=device))
        reference = NT([t.reshape(-1) for t in nt], **nt._state)
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "take_along_dim"), reason="requires torch.take_along_dim")
    def test_take_along_dim(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], device=device, dtype=float_dtype),
                torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=float_dtype),
            ]
        )
        indices = torch.tensor(
            [
                [[0, 2, 1], [2, 1, 0]],
                [[1, 1, 1], [0, 0, 0]],
            ],
            device=device,
            dtype=torch.long,
        )
        output = torch.take_along_dim(nt, indices, dim=2)
        indices_nt = nt.nested_like(indices, strict=False)
        reference = NT([torch.take_along_dim(t, i, dim=1) for t, i in zip(nt, indices_nt)], **nt._state)
        assert_close(output, reference)


class TestConcatAliases:

    def test_concat_and_concatenate_dim0(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([4.0], device=device, dtype=float_dtype),
            ]
        )
        extra = torch.tensor([5.0, 6.0], device=device, dtype=float_dtype)
        output = torch.concat((nt, extra), dim=0)
        reference = NT([*nt, extra], **nt._state)
        assert_close(output, reference)

        output = torch.concatenate((nt, extra), dim=0)
        assert_close(output, reference)

    def test_concat_non_zero_dim(self, device, float_dtype):
        nt1 = NestedTensor(
            [
                torch.zeros(2, 2, device=device, dtype=float_dtype),
                torch.zeros(1, 2, device=device, dtype=float_dtype),
            ]
        )
        nt2 = NestedTensor(
            [
                torch.ones(2, 1, device=device, dtype=float_dtype),
                torch.ones(1, 1, device=device, dtype=float_dtype),
            ],
            **nt1._state,
        )
        output = torch.concat((nt1, nt2), dim=2)
        reference = NT([torch.cat([a, b], dim=1) for a, b in zip(nt1, nt2)], **nt1._state)
        assert_close(output, reference)


class TestSplitChunkUnbind:

    def test_split_and_chunk_batch_dim(self, device, float_dtype):
        nt = NestedTensor([torch.tensor([i], device=device, dtype=float_dtype) for i in range(5)])
        output = torch.split(nt, 2, dim=0)
        assert isinstance(output, tuple)
        assert [len(x) for x in output] == [2, 2, 1]
        storage = tuple(nt)
        reference = (
            NT(storage[:2], **nt._state),
            NT(storage[2:4], **nt._state),
            NT(storage[4:], **nt._state),
        )
        assert_close(output[0], reference[0])
        assert_close(output[1], reference[1])
        assert_close(output[2], reference[2])

        output = torch.split(nt, [1, 3, 1], dim=0)
        assert [len(x) for x in output] == [1, 3, 1]

        with pytest.raises(ValueError):
            torch.split(nt, [1, 1], dim=0)
        with pytest.raises(NotImplementedError):
            torch.split(nt, 2, dim=1)

        output = torch.chunk(nt, 2, dim=0)
        assert [len(x) for x in output] == [3, 2]
        output = torch.chunk(nt, 10, dim=0)
        assert [len(x) for x in output] == [1, 1, 1, 1, 1]
        with pytest.raises(NotImplementedError):
            torch.chunk(nt, 2, dim=1)

    def test_unbind_batch_dim_respects_batch_first(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0], device=device, dtype=float_dtype),
                torch.tensor([2.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.unbind(nt, dim=0)
        assert isinstance(output, tuple)
        assert_close(output[0], nt[0])
        assert_close(output[1], nt[1])

        nt = NestedTensor(
            [
                torch.tensor([1.0], device=device, dtype=float_dtype),
                torch.tensor([2.0], device=device, dtype=float_dtype),
            ],
            batch_first=False,
        )
        output = torch.unbind(nt, dim=1)
        assert_close(output[0], nt[0])
        assert_close(output[1], nt[1])
        with pytest.raises(NotImplementedError):
            torch.unbind(nt, dim=0)


class TestReductionOps:

    def test_all_any_and_prod_ignore_padding(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.ones(3, device=device, dtype=float_dtype),
                torch.ones(1, device=device, dtype=float_dtype),
            ]
        )
        output = torch.all(nt)
        assert_close(output, torch.tensor(True, device=device))

        nt = NestedTensor(
            [torch.zeros(3, device=device, dtype=float_dtype), torch.zeros(1, device=device, dtype=float_dtype)],
            padding_value=1.0,
        )
        output = torch.any(nt)
        assert_close(output, torch.tensor(False, device=device))

        nt = NestedTensor(
            [
                torch.tensor([2.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([2.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.prod(nt)
        assert_close(output, torch.tensor(8.0, device=device, dtype=float_dtype))

    def test_all_any_multi_dim_reduction_ignores_padding(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.ones(2, 2, device=device, dtype=float_dtype),
                torch.ones(1, 2, device=device, dtype=float_dtype),
            ]
        )
        output = torch.all(nt, dim=(0, 1))
        assert_close(output, torch.tensor([True, True], device=device))
        output = torch.any(nt, dim=(0, 1))
        assert_close(output, torch.tensor([True, True], device=device))

    def test_amax_amin_aminmax_and_logsumexp(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0], device=device, dtype=float_dtype),
            ],
            padding_value=100.0,
        )
        output = torch.amax(nt)
        assert_close(output, torch.tensor(3.0, device=device, dtype=float_dtype))
        output = torch.amax(nt, keepdim=True)
        assert output.shape == (1, 1)
        assert_close(output, torch.tensor(3.0, device=device, dtype=float_dtype).reshape(1, 1))

        output = torch.amin(nt)
        assert_close(output, torch.tensor(1.0, device=device, dtype=float_dtype))

        output = torch.amax(nt, dim=0)
        assert_close(output, torch.tensor([2.0, 3.0], device=device, dtype=float_dtype))
        output = torch.amin(nt, dim=0)
        assert_close(output, torch.tensor([1.0, 3.0], device=device, dtype=float_dtype))

        output = torch.aminmax(nt)
        assert_close(output[0], torch.tensor(1.0, device=device, dtype=float_dtype))
        assert_close(output[1], torch.tensor(3.0, device=device, dtype=float_dtype))

        output = torch.aminmax(nt, dim=0)
        assert_close(output[0], torch.tensor([1.0, 3.0], device=device, dtype=float_dtype))
        assert_close(output[1], torch.tensor([2.0, 3.0], device=device, dtype=float_dtype))

        output = torch.logsumexp(nt, dim=1)
        reference = torch.stack([torch.logsumexp(t, dim=0) for t in nt])
        assert_close(output, reference)

    def test_var_std_var_mean(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.var(nt, correction=1)
        assert_close(output, torch.tensor(2.5, device=device, dtype=float_dtype))
        output = torch.var(nt, correction=1, keepdim=True)
        assert output.shape == (1, 1)
        assert_close(output, torch.tensor(2.5, device=device, dtype=float_dtype).reshape(1, 1))

        output = torch.var(nt, dim=0, correction=1)
        reference = torch.stack([torch.var(t, correction=1) for t in nt])
        assert_close(output, reference)

        output = torch.std(nt, dim=0, correction=1)
        reference = torch.stack([torch.std(t, correction=1) for t in nt])
        assert_close(output, reference)

        output = torch.var_mean(nt, dim=0, correction=1)
        var_reference = torch.stack([torch.var(t, correction=1) for t in nt])
        mean_reference = torch.stack([torch.mean(t) for t in nt])
        assert_close(output[0], var_reference)
        assert_close(output[1], mean_reference)

        output = torch.var_mean(nt, correction=1)
        assert_close(output[0], torch.tensor(2.5, device=device, dtype=float_dtype))
        assert_close(output[1], torch.tensor(3.0, device=device, dtype=float_dtype))

    def test_nanmean_and_nansum(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, float("nan"), 3.0], device=device, dtype=float_dtype),
                torch.tensor([float("nan")], device=device, dtype=float_dtype),
            ]
        )
        output = torch.nansum(nt)
        assert_close(output, torch.tensor(4.0, device=device, dtype=float_dtype))

        output = torch.nanmean(nt)
        assert_close(output, torch.tensor(2.0, device=device, dtype=float_dtype), equal_nan=True)
        output = torch.nanmean(nt, keepdim=True)
        assert output.shape == (1, 1)
        assert_close(output, torch.tensor(2.0, device=device, dtype=float_dtype).reshape(1, 1), equal_nan=True)

        output = torch.nansum(nt, dim=0)
        reference = torch.stack([torch.nansum(t) for t in nt])
        assert_close(output, reference)

        output = torch.nanmean(nt, dim=0)
        reference = torch.stack([torch.nanmean(t) for t in nt])
        assert_close(output, reference, equal_nan=True)


class TestMaskedAndIndexingOps:

    def test_masked_fill_and_masked_scatter(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, -2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([-4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        mask = nt > 0
        output = torch.masked_fill(nt, mask, 0.0)
        reference = NT([torch.masked_fill(t, m, 0.0) for t, m in zip(nt, mask)], **nt._state)
        assert_close(output, reference)

        output = torch.masked_fill(nt, mask.tensor, 0.0)
        assert_close(output, reference)

        base = NestedTensor(
            [torch.zeros(3, device=device, dtype=float_dtype), torch.zeros(2, device=device, dtype=float_dtype)]
        )
        mask = NestedTensor(
            [torch.tensor([True, False, True], device=device), torch.tensor([False, True], device=device)]
        )
        src = NestedTensor(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.masked_scatter(base, mask, src)
        reference = NT([torch.masked_scatter(t, m, s) for t, m, s in zip(base, mask, src)], **base._state)
        assert_close(output, reference)

    def test_index_add_copy_and_put(self, device, float_dtype):
        base = NestedTensor(
            [torch.zeros(5, device=device, dtype=float_dtype), torch.zeros(4, device=device, dtype=float_dtype)]
        )
        index = torch.tensor([0, 2], device=device)
        src = NestedTensor(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0, 4.0], device=device, dtype=float_dtype),
            ],
            **base._state,
        )
        output = torch.index_add(base, 1, index, src)
        reference = NT([torch.index_add(t, 0, index, s) for t, s in zip(base, src)], **base._state)
        assert_close(output, reference)

        output = torch.index_copy(base, 1, index, src)
        reference = NT([torch.index_copy(t, 0, index, s) for t, s in zip(base, src)], **base._state)
        assert_close(output, reference)

        values = NestedTensor(
            [
                torch.tensor([10.0, 20.0], device=device, dtype=float_dtype),
                torch.tensor([30.0, 40.0], device=device, dtype=float_dtype),
            ],
            **base._state,
        )
        output = torch.index_put(base, (index,), values, accumulate=False)
        reference = NT([torch.index_put(t, (index,), v) for t, v in zip(base, values)], **base._state)
        assert_close(output, reference)


class TestRandomOps:

    def test_rand_like_and_randn_like_respect_kwargs(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.zeros(2, 3, device=device, dtype=float_dtype),
                torch.zeros(1, 3, device=device, dtype=float_dtype),
            ]
        )
        torch.manual_seed(123)
        output = torch.rand_like(nt, dtype=torch.float64)
        assert output.dtype == torch.float64

        torch.manual_seed(123)
        reference = NT([torch.rand_like(t, dtype=torch.float64) for t in nt], **nt._state)
        assert_close(output, reference)

        torch.manual_seed(456)
        output = torch.randn_like(nt)
        torch.manual_seed(456)
        reference = NT([torch.randn_like(t) for t in nt], **nt._state)
        assert_close(output, reference)

    def test_dropout_and_bernoulli_match_per_tensor(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.ones(4, device=device, dtype=float_dtype),
                torch.ones(2, device=device, dtype=float_dtype),
            ]
        )
        torch.manual_seed(0)
        output = torch.dropout(nt, p=0.5, train=True)
        torch.manual_seed(0)
        reference = NT([torch.dropout(t, p=0.5, train=True) for t in nt], **nt._state)
        assert_close(output, reference)

        torch.manual_seed(0)
        output = torch.bernoulli(nt)
        torch.manual_seed(0)
        reference = NT([torch.bernoulli(t) for t in nt], **nt._state)
        assert_close(output, reference)


class TestLikeCreators:

    def test_zeros_ones_and_full_like(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(1, 3, device=device, dtype=float_dtype),
            ]
        )

        output = torch.zeros_like(nt)
        reference = NT([torch.zeros_like(t) for t in nt], **nt._state)
        assert_close(output, reference)

        output = torch.ones_like(nt, dtype=torch.float64)
        reference = NT([torch.ones_like(t, dtype=torch.float64) for t in nt], **nt._state)
        assert_close(output, reference)
        assert output.dtype == torch.float64

        output = torch.full_like(nt, 3.0, dtype=torch.float32)
        reference = NT([torch.full_like(t, 3.0, dtype=torch.float32) for t in nt], **nt._state)
        assert_close(output, reference)

    def test_empty_like_preserves_shape(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(1, 3, device=device, dtype=float_dtype),
            ]
        )
        output = torch.empty_like(nt, dtype=torch.float64)
        assert isinstance(output, NestedTensor)
        assert output.dtype == torch.float64
        assert [t.shape for t in output] == [t.shape for t in nt]

    def test_randint_like_matches_per_tensor(self, device):
        nt = NT([torch.zeros(3, device=device), torch.zeros(1, device=device)])
        torch.manual_seed(123)
        output = torch.randint_like(nt, 10)
        torch.manual_seed(123)
        reference = NT([torch.randint_like(t, 10) for t in nt], **nt._state)
        assert_close(output, reference)


class TestFlipAndRoll:

    def test_flip_translates_dims_and_rejects_batch(self, device):
        nt = NT([torch.tensor([[1, 2], [3, 4]], device=device), torch.tensor([[5, 6]], device=device)])
        output = torch.flip(nt, dims=(-1,))
        reference = NT([torch.flip(t, dims=(-1,)) for t in nt], **nt._state)
        assert_close(output, reference)

        with pytest.raises(ValueError):
            torch.flip(nt, dims=(0,))
        with pytest.raises(ValueError):
            torch.flip(nt, dims=(-3,))

    def test_roll_supports_dims_none(self, device):
        nt = NT([torch.tensor([1, 2, 3], device=device), torch.tensor([4, 5], device=device)])
        output = torch.roll(nt, shifts=1)
        reference = NT([torch.roll(t, shifts=1) for t in nt], **nt._state)
        assert_close(output, reference)

    def test_roll_translates_dims_and_rejects_batch(self, device):
        nt = NT([torch.tensor([[1, 2], [3, 4]], device=device), torch.tensor([[5, 6]], device=device)])
        output = torch.roll(nt, shifts=1, dims=-1)
        reference = NT([torch.roll(t, shifts=1, dims=-1) for t in nt], **nt._state)
        assert_close(output, reference)

        with pytest.raises(ValueError):
            torch.roll(nt, shifts=1, dims=0)


class TestSoftmaxVariants:

    def test_softmax_and_log_softmax_translate_dim(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(1, 3, device=device, dtype=float_dtype),
            ]
        )
        output = torch.softmax(nt, dim=2)
        reference = NT([torch.softmax(t, dim=1) for t in nt], **nt._state)
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

        output = torch.log_softmax(nt, dim=2)
        reference = NT([torch.log_softmax(t, dim=1) for t in nt], **nt._state)
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

        with pytest.raises(NotImplementedError):
            torch.softmax(nt, dim=0)


class TestAdditionalCumulativeOps:

    def test_logcumsumexp_matches_per_tensor(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(3, device=device, dtype=float_dtype),
                torch.randn(2, device=device, dtype=float_dtype),
            ]
        )
        output = torch.logcumsumexp(nt, dim=1)
        reference = NT([torch.logcumsumexp(t, dim=0) for t in nt], **nt._state)
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

        with pytest.raises(NotImplementedError):
            torch.logcumsumexp(nt, dim=0)

    def test_cummax_and_cummin_return_values_and_indices(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, 0.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0, 1.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.cummax(nt, dim=1)
        reference = tuple(NT([torch.cummax(t, dim=0)[idx] for t in nt], **nt._state) for idx in range(2))
        assert isinstance(output, tuple)
        assert_close(output[0], reference[0])
        assert_close(output[1], reference[1])

        output = torch.cummin(nt, dim=1)
        reference = tuple(NT([torch.cummin(t, dim=0)[idx] for t in nt], **nt._state) for idx in range(2))
        assert_close(output[0], reference[0])
        assert_close(output[1], reference[1])

        with pytest.raises(NotImplementedError):
            torch.cummax(nt, dim=0)


class TestMatmulMmBmm:

    def test_matmul_supports_tensor_rhs(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(1, 3, device=device, dtype=float_dtype),
            ],
            padding_value=-1.0,
        )
        weight = torch.randn(3, 4, device=device, dtype=float_dtype)
        output = torch.matmul(nt, weight)
        reference = NT([torch.matmul(t, weight) for t in nt], **nt._state)
        assert_close(output, reference)

    def test_matmul_supports_tensor_lhs(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(2, 4, device=device, dtype=float_dtype),
            ]
        )
        weight = torch.randn(5, 2, device=device, dtype=float_dtype)
        output = torch.matmul(weight, nt)
        reference = NT([torch.matmul(weight, t) for t in nt], **nt._state)
        assert_close(output, reference)

    def test_mm_and_bmm_match_per_sample(self, device, float_dtype):
        left = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(1, 2, device=device, dtype=float_dtype),
            ]
        )
        right = NT(
            [
                torch.randn(3, 4, device=device, dtype=float_dtype),
                torch.randn(2, 5, device=device, dtype=float_dtype),
            ],
            **left._state,
        )
        output = torch.mm(left, right)
        reference = NT([torch.mm(a, b) for a, b in zip(left, right)], **left._state)
        assert_close(output, reference)

        left = NT(
            [
                torch.randn(2, 1, 3, device=device, dtype=float_dtype),
                torch.randn(1, 2, 3, device=device, dtype=float_dtype),
            ]
        )
        right = NT(
            [
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
                torch.randn(1, 3, 5, device=device, dtype=float_dtype),
            ],
            **left._state,
        )
        output = torch.bmm(left, right)
        reference = NT([torch.bmm(a, b) for a, b in zip(left, right)], **left._state)
        assert_close(output, reference)


class TestCountNonzeroAndDist:

    def test_count_nonzero(self, device):
        nt = NT([torch.tensor([1, 0, 2], device=device), torch.tensor([0], device=device)])
        output = torch.count_nonzero(nt)
        assert_close(output, torch.tensor(2, device=device))

        output = torch.count_nonzero(nt, dim=1)
        assert_close(output, torch.tensor([2, 0], device=device))

        with pytest.raises(NotImplementedError):
            torch.count_nonzero(nt, dim=0)

    def test_dist_stacks_per_sample_results(self, device, float_dtype):
        a = NT(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0], device=device, dtype=float_dtype),
            ]
        )
        b = NT(
            [
                torch.tensor([2.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([1.0], device=device, dtype=float_dtype),
            ],
            **a._state,
        )
        output = torch.dist(a, b, p=2)
        reference = torch.stack([torch.dist(x, y, p=2) for x, y in zip(a, b)])
        assert_close(output, reference)


class TestRot90:

    def test_rot90_translates_dims_and_rejects_batch(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=float_dtype),
                torch.tensor([[5.0, 6.0]], device=device, dtype=float_dtype),
            ],
            padding_value=123.0,
        )

        output = torch.rot90(nt, k=1, dims=(1, 2))
        reference = NT([torch.rot90(t, k=1, dims=(0, 1)) for t in nt], **nt._state)
        assert_close(output, reference)

        with pytest.raises(ValueError):
            torch.rot90(nt)


class TestNonzeroAndTake:

    def test_nonzero_ignores_padding_value(self, device):
        nt = NT([torch.tensor([0, 1, 0], device=device), torch.tensor([2], device=device)], padding_value=9)
        output = torch.nonzero(nt, as_tuple=False)
        reference = NT([torch.nonzero(t, as_tuple=False) for t in nt], **nt._state)
        assert_close(output, reference)

        output = torch.nonzero(nt, as_tuple=True)
        reference = (NT([torch.nonzero(t, as_tuple=True)[0] for t in nt], **nt._state),)
        assert isinstance(output, tuple)
        assert len(output) == 1
        assert_close(output[0], reference[0])

    def test_nonzero_as_tuple_requires_matching_ndim(self, device):
        nt = NT([torch.tensor([1, 0], device=device), torch.tensor([[1], [0]], device=device)])
        with pytest.raises(NotImplementedError):
            torch.nonzero(nt, as_tuple=True)

    def test_take_tensor_index_matches_flat_storage(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0], device=device, dtype=float_dtype),
            ],
            padding_value=99.0,
        )
        index = torch.tensor([0, 2], device=device)
        output = torch.take(nt, index)
        reference = torch.take(torch.cat([t.reshape(-1) for t in nt]), index)
        assert_close(output, reference)

    def test_take_nested_index_is_per_sample(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0, 4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        index = NT([torch.tensor([1], device=device), torch.tensor([0, 2], device=device)], **nt._state)
        output = torch.take(nt, index)
        reference = NT([torch.take(t.reshape(-1), i) for t, i in zip(nt, index)], **nt._state)
        assert_close(output, reference)


class TestOrderStatistics:

    def test_kthvalue_matches_per_sample(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([3.0, 1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([4.0, 0.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.kthvalue(nt, k=2, dim=1)
        reference = tuple(torch.stack([torch.kthvalue(t, k=2, dim=0)[idx] for t in nt]) for idx in range(2))
        assert_close(output[0], reference[0])
        assert_close(output[1], reference[1])

    def test_median_dim_none_and_dim(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, 2.0, 10.0], device=device, dtype=float_dtype),
                torch.tensor([3.0], device=device, dtype=float_dtype),
            ],
            padding_value=-123.0,
        )
        output = torch.median(nt)
        reference = torch.median(torch.cat([t.reshape(-1) for t in nt]))
        assert_close(output, reference)

        output = torch.median(nt, dim=1, keepdim=True)
        reference = tuple(torch.stack([torch.median(t, dim=0, keepdim=True)[idx] for t in nt]) for idx in range(2))
        assert_close(output[0], reference[0])
        assert_close(output[1], reference[1])

    def test_nanmedian_mode_quantile_and_nanquantile(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, float("nan"), 3.0], device=device, dtype=float_dtype),
                torch.tensor([float("nan"), 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.nanmedian(nt)
        reference = torch.nanmedian(torch.cat([t.reshape(-1) for t in nt]))
        assert_close(output, reference, equal_nan=True)

        output = torch.mode(nt, dim=1)
        reference = tuple(torch.stack([torch.mode(t, dim=0)[idx] for t in nt]) for idx in range(2))
        assert_close(output[0], reference[0], equal_nan=True)
        assert_close(output[1], reference[1])

        output = torch.quantile(nt, 0.5)
        reference = torch.quantile(torch.cat([t.reshape(-1) for t in nt]), 0.5)
        assert_close(output, reference, equal_nan=True)

        output = torch.quantile(nt, 0.5, keepdim=True)
        assert output.shape == (1, 1)

        output = torch.nanquantile(nt, 0.5)
        reference = torch.nanquantile(torch.cat([t.reshape(-1) for t in nt]), 0.5)
        assert_close(output, reference, equal_nan=True)
