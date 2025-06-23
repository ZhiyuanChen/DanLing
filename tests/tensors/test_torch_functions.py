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
from danling.tensors.torch_functions import (
    cat,
    flatten,
    gather,
    scatter,
    stack,
    unflatten,
    unsqueeze,
)
from tests.tensors.utils import assert_close, low_precision_cuda_tolerances, nested_rand, ragged_shapes

NT = NestedTensor


def _packed_result(ref: NT, values: torch.Tensor) -> NT:
    return NT._from_packed(
        values,
        ref._offsets,
        ref._physical_shape,
        batch_first=ref.batch_first,
        padding_value=ref.padding_value,
        mask_value=ref.mask_value,
        pin_memory=ref._pin_memory,
        outer_size=ref._logical_shape,
        packed_sizes=ref._packed_sizes,
        element_shapes=ref._element_shapes,
    )


def _run_or_expect_unsupported(nested_call, tensor_call):
    try:
        reference = tensor_call()
    except (RuntimeError, NotImplementedError) as error:
        with pytest.raises(type(error)):
            nested_call()
        return None
    return nested_call(), reference


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_det_compile_smoke():
    torch.manual_seed(1019)
    for op in (torch.det, torch.linalg.det):
        for nt in (
            NT([torch.randn(2, 2, dtype=torch.float32), torch.randn(2, 2, dtype=torch.float32)]),
            NT([torch.randn(3, 3, dtype=torch.float32), torch.randn(4, 4, dtype=torch.float32)]),
        ):
            compiled = torch.compile(lambda x, operator=op: operator(x), backend="eager", fullgraph=True)
            output = compiled(nt)
            reference = NT([op(t) for t in nt], **nt._meta())
            assert isinstance(output, NT)
            assert_close(output, reference)


class TestArithmeticFunctions:

    def test_add_converts_tensor_to_nested(self, device, float_dtype):
        nt = NestedTensor([torch.ones(2, device=device, dtype=float_dtype)])
        tensor = torch.ones_like(nt.tensor)
        output = torch.add(nt, tensor)
        assert isinstance(output, NestedTensor)
        assert_close(output, nt.tensor + tensor)

    def test_div_converts_other_to_nested(self, device, float_dtype):
        nt = NestedTensor([torch.ones(2, device=device, dtype=float_dtype) * 4])
        other = torch.full_like(nt.tensor, 2, device=device, dtype=float_dtype)
        output = torch.div(other, nt)
        assert isinstance(output, NestedTensor)
        assert_close(output, other / nt.tensor)

    def test_pow_with_scalar_and_nested_exponent(self, device, float_dtype):
        base = NestedTensor([torch.arange(1.0, 3.0, device=device, dtype=float_dtype)])
        output = torch.pow(base, 2)
        reference = base.tensor.pow(2)
        assert_close(output, reference)
        exponent = torch.full_like(base.tensor, 2, device=device, dtype=float_dtype)
        output = torch.pow(base, exponent)
        reference = base.tensor.pow(exponent)
        assert_close(output, reference)

    def test_sub_converts_tensor_argument(self, device, float_dtype):
        nt = NestedTensor([torch.ones(2, device=device, dtype=float_dtype)])
        tensor = torch.zeros_like(nt.tensor)
        output = torch.sub(tensor, nt)
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
        reference = NT([t + 1 for t in nt], **nt._meta())
        assert_close(output, reference)


class TestCatFunction:

    def test_cat_dim0_preserves_state(self, device, float_dtype):
        nt = NestedTensor([torch.ones(2, device=device, dtype=float_dtype)])
        mixed = torch.zeros(2, device=device, dtype=float_dtype)
        output = cat((nt, mixed), dim=0)
        assert isinstance(output, NestedTensor)
        assert output.device == nt.device
        reference = NT([nt[0], mixed], **nt._meta())
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
        reference = NT([*nt, extra], **nt._meta())
        assert_close(output, reference)
        assert output.batch_first is False

    def test_cat_batch_first_false_non_batch_dims(self):
        nt1 = NestedTensor([torch.ones(2, 2), torch.ones(3, 2)], batch_first=False)
        nt2_seq = NestedTensor([torch.zeros(1, 2), torch.zeros(2, 2)], batch_first=False)
        nt2_feat = NestedTensor([torch.zeros(2, 1), torch.zeros(3, 1)], batch_first=False)

        output_seq = torch.cat((nt1, nt2_seq), dim=0)
        reference_seq = NT([torch.cat([a, b], dim=0) for a, b in zip(nt1, nt2_seq)], **nt1._meta())
        assert_close(output_seq, reference_seq)

        output_feat = torch.cat((nt1, nt2_feat), dim=2)
        reference_feat = NT([torch.cat([a, b], dim=1) for a, b in zip(nt1, nt2_feat)], **nt1._meta())
        assert_close(output_feat, reference_feat)

    def test_cat_dim0_incompatible_packed_layout_falls_back(self):
        nt1 = NT([torch.arange(6.0).reshape(2, 3), torch.arange(3.0).reshape(1, 3)])
        nt2 = NT([torch.arange(8.0).reshape(2, 4), torch.arange(4.0).reshape(1, 4)])
        output = torch.cat((nt1, nt2), dim=0)
        reference = NT([*nt1, *nt2], **nt1._meta())
        assert_close(output, reference)

    def test_cat_dim0_validates_nested_metadata(self):
        nt_a = NT([torch.tensor([1.0, 2.0])], batch_first=True, padding_value=0.0, mask_value=False)
        nt_b = NT([torch.tensor([3.0])], batch_first=False, padding_value=0.0, mask_value=False)
        with pytest.raises(ValueError, match="share batch_first, padding_value, and mask_value"):
            torch.cat((nt_a, nt_b), dim=0)

    def test_cat_dim0_validates_nested_metadata_in_mixed_inputs(self):
        nt_a = NT([torch.tensor([1.0, 2.0])], batch_first=True, padding_value=0.0, mask_value=False)
        nt_b = NT([torch.tensor([3.0])], batch_first=True, padding_value=-1.0, mask_value=False)
        with pytest.raises(ValueError, match="share batch_first, padding_value, and mask_value"):
            torch.cat((nt_a, torch.tensor([9.0]), nt_b), dim=0)

    def test_cat_dim0_with_empty_nested_inputs(self):
        nt_empty = NT([], dtype=torch.float32)
        nt = NT([torch.tensor([1.0, 2.0])])

        out_left_empty = torch.cat((nt_empty, nt), dim=0)
        assert len(out_left_empty) == 1
        assert_close(out_left_empty[0], nt[0])

        out_right_empty = torch.cat((nt, nt_empty), dim=0)
        assert len(out_right_empty) == 1
        assert_close(out_right_empty[0], nt[0])

        nt2d = NT([torch.arange(6.0).reshape(2, 3), torch.arange(3.0).reshape(1, 3)])
        out_left_empty_2d = torch.cat((nt_empty, nt2d), dim=0)
        assert len(out_left_empty_2d) == len(nt2d)
        assert_close(out_left_empty_2d, nt2d)

        out_right_empty_2d = torch.cat((nt2d, nt_empty), dim=0)
        assert len(out_right_empty_2d) == len(nt2d)
        assert_close(out_right_empty_2d, nt2d)


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

    def test_stack_dim_zero_matches_per_sample(self):
        a = NT([torch.tensor([1, 2]), torch.tensor([3, 4])])
        b = NT([torch.tensor([5, 6]), torch.tensor([7, 8])])
        output = torch.stack([a, b], dim=0)
        reference = NT([torch.stack([a[i], b[i]], dim=0) for i in range(len(a))], **a._meta())
        assert isinstance(output, NestedTensor)
        assert_close(output, reference)


class TestFlattenUnflatten:

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
        with pytest.raises(ValueError):
            unflatten(nt, dim=0, sizes=(1, 2))

    def test_flatten_and_unflatten_round_trip(self):
        nt = NT(
            [
                torch.arange(8, dtype=torch.float32).reshape(2, 2, 2),
                torch.arange(8, 16, dtype=torch.float32).reshape(2, 2, 2),
            ]
        )
        flattened = torch.flatten(nt, start_dim=1)
        reference = torch.flatten(nt.tensor, start_dim=1)
        assert_close(flattened, reference)
        unflattened = torch.unflatten(flattened, dim=1, sizes=(2, 2, 2))
        assert_close(unflattened, nt)


class TestDimensionTransforms:

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

    def test_squeeze_unsqueeze_round_trip(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(1, 3, device=device, dtype=float_dtype),
                torch.randn(1, 3, device=device, dtype=float_dtype),
            ]
        )
        squeezed = torch.squeeze(nt, dim=1)
        output = torch.unsqueeze(squeezed, dim=2)
        reference = torch.unsqueeze(torch.squeeze(nt.tensor, dim=1), dim=2)
        assert_close(output, reference)

    def test_squeeze_default_matches_per_sample(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 1, 3, device=device, dtype=float_dtype),
                torch.randn(3, 1, 3, device=device, dtype=float_dtype),
            ]
        )
        output = torch.squeeze(nt)
        reference = NT([torch.squeeze(t) for t in nt], **nt._meta())
        assert_close(output, reference)

    def test_squeeze_default_with_all_ragged_ones(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(1, 2, device=device, dtype=float_dtype),
                torch.randn(1, 2, device=device, dtype=float_dtype),
            ]
        )
        output = torch.squeeze(nt)
        reference = NT([torch.squeeze(t) for t in nt], **nt._meta())
        assert_close(output, reference)

    def test_swapaxes_batch_dim_raises(self):
        nt = NestedTensor([torch.ones(2, 2)])
        with pytest.raises(ValueError):
            torch.swapaxes(nt, 0, 1)

    def test_moveaxis_matches_tensor(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
            ]
        )
        output = torch.moveaxis(nt, 1, 2)
        reference = torch.moveaxis(nt.tensor, 1, 2)
        assert_close(output, reference)

    def test_swapaxes_matches_tensor(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
            ]
        )
        output = torch.swapaxes(nt, 1, 2)
        reference = torch.swapaxes(nt.tensor, 1, 2)
        assert_close(output, reference)

    def test_torch_permute_accepts_dims_sequence(self, device, float_dtype):
        nt = NestedTensor([torch.randn(2, 3, 4, device=device, dtype=float_dtype)])
        dims = (0, 3, 2, 1)
        reference = nt.permute(*dims)
        output = torch.permute(nt, dims)
        assert_close(output, reference)
        assert output.shape == reference.shape


class TestGatherScatter:

    def test_gather_on_batch_dim_not_supported(self):
        nt = NestedTensor([torch.tensor([[1, 2]])])
        idx = torch.zeros_like(nt.tensor, dtype=torch.long)
        with pytest.raises(ValueError):
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
        reference = NT([torch.gather(t, 0, torch.zeros_like(t, dtype=torch.long)) for t in nt], **nt._meta())
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
        reference = NT([torch.scatter(t, 0, idx, src_slice) for t, idx, src_slice in zip(nt, index, src)], **nt._meta())
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
        reference = NT([torch.gather(t, 0, torch.zeros_like(t, dtype=torch.long)) for t in nt], **nt._meta())
        assert_close(output, reference)

    def test_gather_with_nested_index_ragged_dim(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], device=device, dtype=float_dtype),
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device=device, dtype=float_dtype),
            ]
        )
        index = NT(
            [
                torch.tensor([[1, 0, 1]], device=device, dtype=torch.long),
                torch.tensor([[2, 1, 0], [0, 2, 1]], device=device, dtype=torch.long),
            ]
        )
        output = gather(nt, 1, index)
        reference = NT([torch.gather(t, 0, idx) for t, idx in zip(nt, index)], **nt._meta())
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
            **nt._meta(),
        )
        assert_close(output, reference)


class TestUnaryBinaryMath:

    def test_neg(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, -2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([-4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.neg(nt)
        reference = NT([torch.neg(t) for t in nt], **nt._meta())
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
        reference = NT([torch.maximum(t, scalar) for t in nt], **nt._meta())
        assert_close(output, reference)

        output = torch.minimum(nt, scalar)
        reference = NT([torch.minimum(t, scalar) for t in nt], **nt._meta())
        assert_close(output, reference)

        output = torch.maximum(nt, nt.tensor)
        assert_close(output, nt)

    def test_addcmul(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.addcmul(nt, nt, nt, value=2)
        reference = NT([torch.addcmul(t, t, t, value=2) for t in nt], **nt._meta())
        assert_close(output, reference)

    def test_addcmul_tensor_input(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        reference = NT([torch.addcmul(t, t, t, value=2) for t in nt], **nt._meta())
        output = torch.addcmul(nt.tensor, nt, nt, value=2)
        assert_close(output, reference)

    def test_addcmul_tensor_input_requires_per_element_broadcast(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.ones(2, 3, device=device, dtype=float_dtype),
                torch.ones(4, 3, device=device, dtype=float_dtype),
            ]
        )
        bad = torch.arange(18, device=device, dtype=float_dtype).reshape(6, 3)

        with pytest.raises(RuntimeError, match="size of tensor"):
            torch.addcmul(nt, bad, bad, value=2)

    def test_addcdiv(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.addcdiv(nt, nt, nt + 1, value=2)
        reference = NT([torch.addcdiv(t, t, t + 1, value=2) for t in nt], **nt._meta())
        assert_close(output, reference)

    def test_addcdiv_tensor_input(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        reference = NT([torch.addcdiv(t, t, t + 1, value=2) for t in nt], **nt._meta())
        output = torch.addcdiv(nt.tensor, nt, nt + 1, value=2)
        assert_close(output, reference)

    @pytest.mark.parametrize("seed", [2, 17, 43])
    def test_addcmul_and_addcdiv_randomized_dense_parity(self, device, seed):
        dtype = torch.float32
        shapes = ragged_shapes(seed, batch_size=3, min_len=2, max_len=5, trailing_shape=(4,))
        nt = nested_rand(shapes, device, dtype)
        torch.manual_seed(seed)
        bias = torch.randn(4, device=device, dtype=dtype)
        scale = torch.randn(4, device=device, dtype=dtype)
        denom = torch.randn(4, device=device, dtype=dtype).abs() + 0.5

        addcmul_output = torch.addcmul(nt, bias, scale, value=0.25)
        addcmul_reference = NT([torch.addcmul(t, bias, scale, value=0.25) for t in nt], **nt._meta())
        assert_close(addcmul_output, addcmul_reference)

        addcdiv_output = torch.addcdiv(nt, scale, denom, value=-0.5)
        addcdiv_reference = NT([torch.addcdiv(t, scale, denom, value=-0.5) for t in nt], **nt._meta())
        assert_close(addcdiv_output, addcdiv_reference)

    def test_clamp_min_and_clamp_max(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, -2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([-4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.clamp_min(nt, 0.0)
        reference = NT([torch.clamp_min(t, 0.0) for t in nt], **nt._meta())
        assert_close(output, reference)

        output = torch.clamp_max(nt, 0.0)
        reference = NT([torch.clamp_max(t, 0.0) for t in nt], **nt._meta())
        assert_close(output, reference)

    def test_clamp_matches_tensor(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([[-1.0, 0.5], [2.0, 5.0]], device=device, dtype=float_dtype),
                torch.tensor([[10.0, -5.0], [1.0, 3.0]], device=device, dtype=float_dtype),
            ]
        )
        output = torch.clamp(nt, min=0.0, max=3.0)
        reference = torch.clamp(nt.tensor, min=0.0, max=3.0)
        assert_close(output, reference)

    def test_isfinite_isnan_nan_to_num(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([float("nan"), float("inf"), -float("inf")], device=device, dtype=float_dtype),
                torch.tensor([0.0, float("nan")], device=device, dtype=float_dtype),
            ]
        )
        output = torch.isnan(nt)
        reference = NT([torch.isnan(t) for t in nt], **nt._meta())
        assert_close(output, reference)

        output = torch.isfinite(nt)
        reference = NT([torch.isfinite(t) for t in nt], **nt._meta())
        assert_close(output, reference)

        output = torch.nan_to_num(nt, nan=0.0, posinf=1.0, neginf=-1.0)
        reference = NT([torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0) for t in nt], **nt._meta())
        assert_close(output, reference)


class TestWhere:

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
            **nt._meta(),
        )
        assert_close(output, reference)

    def test_where_dense_condition_matches_per_element(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 2, 3], device=device, dtype=float_dtype),
                torch.tensor([4, 5], device=device, dtype=float_dtype),
            ]
        )
        output = torch.where(nt.tensor > 2, nt, 0.0)
        reference = NT([torch.where(t > 2, t, 0.0) for t in nt], **nt._meta())
        assert_close(output, reference)

    def test_where_scalar_operands_match_per_element(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 2, 3], device=device, dtype=float_dtype),
                torch.tensor([4, 5], device=device, dtype=float_dtype),
            ]
        )
        scalar_self = torch.where(nt > 2, 1.0, nt)
        scalar_self_ref = NT([torch.where(t > 2, 1.0, t) for t in nt], **nt._meta())
        assert_close(scalar_self, scalar_self_ref)

        scalar_both = torch.where(nt > 2, 1.0, 0.0)
        scalar_both_ref = NT([torch.where(t > 2, 1.0, 0.0) for t in nt], **nt._meta())
        assert_close(scalar_both, scalar_both_ref)

    def test_where_length_mismatch_raises(self):
        cond = NestedTensor([torch.tensor([True, False])])
        input_nt = NestedTensor([torch.tensor([1, 2]), torch.tensor([3])])
        with pytest.raises(ValueError, match="batch length mismatch"):
            _ = torch.where(cond, input_nt, 0)

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_where_randomized_dense_parity(self, device, seed):
        dtype = torch.float32
        shapes = ragged_shapes(seed, batch_size=3, min_len=2, max_len=5, trailing_shape=(4,))
        nt = nested_rand(shapes, device, dtype)
        torch.manual_seed(seed)
        condition = torch.randn(4, device=device) > 0
        other = torch.randn(4, device=device, dtype=dtype)
        output = torch.where(condition, nt, other)
        reference = NT([torch.where(condition, t, other) for t in nt], **nt._meta())
        assert_close(output, reference)


class TestSelectionOps:

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

    def test_max_and_min_dim(self, device, float_dtype):
        a = torch.tensor([[1, 5, 3], [2, 0, 4]], device=device, dtype=float_dtype)
        b = torch.tensor([[9, 1, 2], [3, 7, 6], [4, 0, 8]], device=device, dtype=float_dtype)
        nt = NestedTensor([a, b])

        max_last = torch.max(nt, dim=2)
        max_last_ref_vals = NestedTensor([torch.max(a, dim=1).values, torch.max(b, dim=1).values], **nt._meta())
        max_last_ref_idxs = NestedTensor([torch.max(a, dim=1).indices, torch.max(b, dim=1).indices], **nt._meta())
        assert_close(max_last.values, max_last_ref_vals)
        assert_close(max_last.indices, max_last_ref_idxs)

        min_ragged = torch.min(nt, dim=1)
        min_ragged_ref_vals = torch.stack([torch.min(a, dim=0).values, torch.min(b, dim=0).values])
        min_ragged_ref_idxs = torch.stack([torch.min(a, dim=0).indices, torch.min(b, dim=0).indices])
        assert_close(min_ragged.values, min_ragged_ref_vals)
        assert_close(min_ragged.indices, min_ragged_ref_idxs)

    def test_argmax_and_argmin_dim(self, device, float_dtype):
        a = torch.tensor([[1, 5, 3], [2, 0, 4]], device=device, dtype=float_dtype)
        b = torch.tensor([[9, 1, 2], [3, 7, 6], [4, 0, 8]], device=device, dtype=float_dtype)
        nt = NestedTensor([a, b])

        argmax_last = torch.argmax(nt, dim=2)
        argmax_last_ref = NestedTensor([torch.argmax(a, dim=1), torch.argmax(b, dim=1)], **nt._meta())
        assert_close(argmax_last, argmax_last_ref)

        argmin_ragged = torch.argmin(nt, dim=1)
        argmin_ragged_ref = torch.stack([torch.argmin(a, dim=0), torch.argmin(b, dim=0)])
        assert_close(argmin_ragged, argmin_ragged_ref)

    def test_topk(self, device, float_dtype):
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
                **nt._meta(),
            ),
            NestedTensor([torch.tensor([1, 2], device=device), torch.tensor([0, 1], device=device)], **nt._meta()),
        )
        assert_close(output[0], reference[0])
        assert_close(output[1], reference[1])

    def test_sort(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 3, 2], device=device, dtype=float_dtype),
                torch.tensor([4, 0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.sort(nt, dim=1, descending=False)
        reference = (
            torch.tensor([[1, 2, 3], [0, 4, 0]], device=device, dtype=float_dtype),
            output[1],
        )
        assert_close(output[0], reference[0])

    def test_argsort(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 3, 2], device=device, dtype=float_dtype),
                torch.tensor([4, 0], device=device, dtype=float_dtype),
            ]
        )
        sorted_output = torch.sort(nt, dim=1, descending=False)
        reference_indices = sorted_output[1]
        assert_close(torch.argsort(nt), reference_indices)


class TestCumulativeOps:

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

    def test_cumulative_batch_dim_not_supported(self, device, float_dtype):
        nt = NestedTensor([torch.tensor([1.0, 2.0], device=device, dtype=float_dtype)])
        with pytest.raises(ValueError):
            torch.cumsum(nt, dim=0)
        with pytest.raises(ValueError):
            torch.cumprod(nt, dim=0)

        nt_bf_false = NestedTensor([torch.tensor([1.0, 2.0], device=device, dtype=float_dtype)], batch_first=False)
        with pytest.raises(ValueError):
            torch.cumsum(nt_bf_false, dim=1)
        with pytest.raises(ValueError):
            torch.cumprod(nt_bf_false, dim=1)

    def test_logcumsumexp_matches_per_tensor(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(3, device=device, dtype=float_dtype),
                torch.randn(2, device=device, dtype=float_dtype),
            ]
        )
        output = torch.logcumsumexp(nt, dim=1)
        reference = NT([torch.logcumsumexp(t, dim=0) for t in nt], **nt._meta())
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

        with pytest.raises(ValueError):
            torch.logcumsumexp(nt, dim=0)

    def test_cummax_returns_values_and_indices(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, 0.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0, 1.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.cummax(nt, dim=1)
        reference = tuple(NT([torch.cummax(t, dim=0)[idx] for t in nt], **nt._meta()) for idx in range(2))
        assert isinstance(output, tuple)
        assert_close(output[0], reference[0])
        assert_close(output[1], reference[1])

    def test_cummin_returns_values_and_indices(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, 0.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0, 1.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.cummin(nt, dim=1)
        reference = tuple(NT([torch.cummin(t, dim=0)[idx] for t in nt], **nt._meta()) for idx in range(2))
        assert_close(output[0], reference[0])
        assert_close(output[1], reference[1])

    def test_cummax_batch_dim_not_supported(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, 0.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0, 1.0], device=device, dtype=float_dtype),
            ]
        )
        with pytest.raises(ValueError):
            torch.cummax(nt, dim=0)


class TestLogicOps:

    def test_logical_ops_accept_python_scalars(self, device):
        nt = NestedTensor([torch.tensor([True, False], device=device), torch.tensor([True], device=device)])
        true_tensor = torch.tensor(True, device=device)
        false_tensor = torch.tensor(False, device=device)
        output = torch.logical_and(nt, true_tensor)
        assert_close(output, nt)

        output = torch.logical_or(nt, false_tensor)
        assert_close(output, nt)

        output = torch.logical_xor(nt, true_tensor)
        reference = NT([torch.logical_xor(t, true_tensor) for t in nt], **nt._meta())
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
        reference = NT([torch.bitwise_and(t, 1) for t in nt], **nt._meta())
        assert_close(output, reference)

        output = torch.bitwise_or(nt, 1)
        reference = NT([torch.bitwise_or(t, 1) for t in nt], **nt._meta())
        assert_close(output, reference)

        output = torch.bitwise_xor(nt, 1)
        reference = NT([torch.bitwise_xor(t, 1) for t in nt], **nt._meta())
        assert_close(output, reference)


class TestScatterOps:

    def test_scatter(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.zeros(2, 3, device=device, dtype=float_dtype),
                torch.zeros(3, 3, device=device, dtype=float_dtype),
            ]
        )
        index = NestedTensor(
            [
                torch.tensor([[0, 2, 1], [1, 0, 2]], device=device, dtype=torch.long),
                torch.tensor([[2, 1, 0], [0, 2, 1], [1, 1, 1]], device=device, dtype=torch.long),
            ]
        )
        src = NestedTensor(
            [
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device, dtype=float_dtype),
                torch.tensor(
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
                    device=device,
                    dtype=float_dtype,
                ),
            ]
        )

        output = torch.scatter(nt, 2, index, src)
        reference = NT([torch.scatter(t, 1, idx, s) for t, idx, s in zip(nt, index, src)], **nt._meta())
        assert_close(output, reference)

        scalar_output = torch.scatter(nt, 2, index, -1.0)
        scalar_reference = NT([torch.scatter(t, 1, idx, -1.0) for t, idx in zip(nt, index)], **nt._meta())
        assert_close(scalar_output, scalar_reference)

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
            **nt._meta(),
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
            **nt._meta(),
        )
        assert_close(output, reference)


class TestIndexingReadOps:

    def test_index_select(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device, dtype=float_dtype),
                torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], device=device, dtype=float_dtype),
            ]
        )
        output = torch.index_select(nt, 0, torch.tensor([1, 0, 1], device=device))
        reference = NT([nt[1], nt[0], nt[1]], **nt._meta())
        assert_close(output, reference)

        output = torch.index_select(nt, 2, torch.tensor([2, 0], device=device))
        reference = NT([torch.index_select(t, 1, torch.tensor([2, 0], device=device)) for t in nt], **nt._meta())
        assert_close(output, reference)

        output = torch.index_select(nt, 1, torch.tensor([1, 0], device=device))
        reference = NT([torch.index_select(t, 0, torch.tensor([1, 0], device=device)) for t in nt], **nt._meta())
        assert_close(output, reference)

    def test_index_select_batch_first_false(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=float_dtype),
                torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device, dtype=float_dtype),
            ],
            batch_first=False,
        )
        output = torch.index_select(nt, 1, torch.tensor([1, 0, 1], device=device))
        reference = NT([nt[1], nt[0], nt[1]], **nt._meta())
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
        reference = NT([torch.masked_select(t, m) for t, m in zip(nt, mask)], **nt._meta())
        assert_close(output, reference)

        output = torch.masked_select(nt, mask.tensor)
        assert_close(output, reference)

        output = torch.masked_select(nt, torch.tensor(True, device=device))
        reference = NT([t.reshape(-1) for t in nt], **nt._meta())
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
        reference = NT([torch.take_along_dim(t, i, dim=1) for t, i in zip(nt, indices_nt)], **nt._meta())
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "take_along_dim"), reason="requires torch.take_along_dim")
    def test_take_along_dim_with_nested_indices_ragged_dim(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], device=device, dtype=float_dtype),
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device=device, dtype=float_dtype),
            ]
        )
        indices = NT(
            [
                torch.tensor([[1, 1, 0]], device=device, dtype=torch.long),
                torch.tensor([[2, 2, 2], [0, 1, 0]], device=device, dtype=torch.long),
            ]
        )
        output = torch.take_along_dim(nt, indices, dim=1)
        reference = NT([torch.take_along_dim(t, idx, dim=0) for t, idx in zip(nt, indices)], **nt._meta())
        assert_close(output, reference)


class TestConcatAliases:

    @pytest.mark.parametrize("alias", [torch.concat, torch.concatenate])
    def test_concat_aliases_dim0(self, alias, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([4.0], device=device, dtype=float_dtype),
            ]
        )
        extra = torch.tensor([5.0, 6.0], device=device, dtype=float_dtype)
        output = alias((nt, extra), dim=0)
        reference = NT([*nt, extra], **nt._meta())
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
            **nt1._meta(),
        )
        output = torch.concat((nt1, nt2), dim=2)
        reference = NT([torch.cat([a, b], dim=1) for a, b in zip(nt1, nt2)], **nt1._meta())
        assert_close(output, reference)


class TestSplitChunkUnbind:

    def test_split_with_int_sections(self, device, float_dtype):
        nt = NestedTensor([torch.tensor([i], device=device, dtype=float_dtype) for i in range(5)])
        output = torch.split(nt, 2, dim=0)
        assert isinstance(output, tuple)
        assert [len(x) for x in output] == [2, 2, 1]
        storage = tuple(nt)
        reference = (
            NT(storage[:2], **nt._meta()),
            NT(storage[2:4], **nt._meta()),
            NT(storage[4:], **nt._meta()),
        )
        assert_close(output[0], reference[0])
        assert_close(output[1], reference[1])
        assert_close(output[2], reference[2])

    def test_split_with_list_sections(self, device, float_dtype):
        nt = NestedTensor([torch.tensor([i], device=device, dtype=float_dtype) for i in range(5)])
        output = torch.split(nt, [1, 3, 1], dim=0)
        assert [len(x) for x in output] == [1, 3, 1]

    def test_split_invalid_sections_raises(self, device, float_dtype):
        nt = NestedTensor([torch.tensor([i], device=device, dtype=float_dtype) for i in range(5)])
        with pytest.raises(ValueError):
            torch.split(nt, [1, 1], dim=0)

    def test_split_non_batch_dim(self, device, float_dtype):
        a = torch.randn(3, 6, device=device, dtype=float_dtype)
        b = torch.randn(5, 6, device=device, dtype=float_dtype)
        nt = NestedTensor(a, b)
        parts = torch.split(nt, 2, dim=-1)
        assert len(parts) == 3
        assert torch.equal(parts[0][0], a[:, :2])
        assert torch.equal(parts[0][1], b[:, :2])
        assert torch.equal(parts[1][0], a[:, 2:4])
        assert torch.equal(parts[2][0], a[:, 4:6])

    def test_split_non_batch_dim_mismatched_counts_raises(self, device, float_dtype):
        a = torch.randn(2, 2, device=device, dtype=float_dtype)
        b = torch.randn(2, 5, device=device, dtype=float_dtype)
        nt = NestedTensor(a, b)
        with pytest.raises(ValueError, match="uniform per-element split counts"):
            torch.split(nt, 2, dim=-1)

    def test_chunk_batch_dim(self, device, float_dtype):
        nt = NestedTensor([torch.tensor([i], device=device, dtype=float_dtype) for i in range(5)])
        output = torch.chunk(nt, 2, dim=0)
        assert [len(x) for x in output] == [3, 2]

    def test_chunk_excess_chunks(self, device, float_dtype):
        nt = NestedTensor([torch.tensor([i], device=device, dtype=float_dtype) for i in range(5)])
        output = torch.chunk(nt, 10, dim=0)
        assert [len(x) for x in output] == [1, 1, 1, 1, 1]

    def test_chunk_non_batch_dim(self, device, float_dtype):
        a = torch.randn(3, 6, device=device, dtype=float_dtype)
        b = torch.randn(5, 6, device=device, dtype=float_dtype)
        nt = NestedTensor(a, b)
        parts = torch.chunk(nt, 3, dim=-1)
        assert len(parts) == 3
        assert parts[0][0].shape == torch.Size([3, 2])
        assert torch.equal(parts[0][0], a[:, :2])
        assert torch.equal(parts[2][1], b[:, 4:6])

    def test_chunk_non_batch_dim_mismatched_counts_raises(self, device, float_dtype):
        a = torch.randn(2, 2, device=device, dtype=float_dtype)
        b = torch.randn(2, 5, device=device, dtype=float_dtype)
        nt = NestedTensor(a, b)
        with pytest.raises(ValueError, match="uniform per-element chunk counts"):
            torch.chunk(nt, 3, dim=-1)

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

    def test_all_ignores_padding(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.ones(3, device=device, dtype=float_dtype),
                torch.ones(1, device=device, dtype=float_dtype),
            ]
        )
        output = torch.all(nt)
        assert_close(output, torch.tensor(True, device=device))

    def test_any_ignores_padding(self, device, float_dtype):
        nt = NestedTensor(
            [torch.zeros(3, device=device, dtype=float_dtype), torch.zeros(1, device=device, dtype=float_dtype)],
            padding_value=1.0,
        )
        output = torch.any(nt)
        assert_close(output, torch.tensor(False, device=device))

    def test_prod_ignores_padding(self, device, float_dtype):
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

    def test_amax_amin_and_aminmax_ignore_padding(self, device, float_dtype):
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

    def test_logsumexp_matches_per_sample(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0], device=device, dtype=float_dtype),
            ]
        )
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

    def test_mean_multi_dim_integer_matches_dense_error(self, device):
        nt = NestedTensor(
            [
                torch.arange(24, device=device, dtype=torch.long).reshape(2, 3, 4),
                torch.arange(24, 48, device=device, dtype=torch.long).reshape(2, 3, 4),
            ]
        )
        with pytest.raises(RuntimeError, match="could not infer output dtype"):
            torch.mean(nt, dim=(1, 2))


class TestIndexingWriteOps:

    def test_masked_fill(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, -2.0, 3.0], device=device, dtype=float_dtype),
                torch.tensor([-4.0, 5.0], device=device, dtype=float_dtype),
            ]
        )
        mask = nt > 0
        output = torch.masked_fill(nt, mask, 0.0)
        reference = NT([torch.masked_fill(t, m, 0.0) for t, m in zip(nt, mask)], **nt._meta())
        assert_close(output, reference)

        output = torch.masked_fill(nt, mask.tensor, 0.0)
        assert_close(output, reference)

        tensor_value = torch.tensor(0.0, device=device, dtype=float_dtype)
        output = torch.masked_fill(nt, mask, tensor_value)
        assert_close(output, reference)

    def test_masked_scatter(self, device, float_dtype):
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
        reference = NT([torch.masked_scatter(t, m, s) for t, m, s in zip(base, mask, src)], **base._meta())
        assert_close(output, reference)

    def test_masked_scatter_same_shape_dense_source_preserves_per_element_semantics(self, device, float_dtype):
        base = NestedTensor(
            [torch.zeros(3, device=device, dtype=float_dtype), torch.zeros(2, device=device, dtype=float_dtype)]
        )
        mask = NestedTensor(
            [torch.tensor([True, False, True], device=device), torch.tensor([False, True], device=device)]
        )
        dense_source = torch.tensor(
            [[1.0, 2.0, 9.0], [3.0, 8.0, 7.0]],
            device=device,
            dtype=float_dtype,
        )

        output = torch.masked_scatter(base, mask, dense_source)
        nested_source = base.nested_like(dense_source, strict=False)
        reference = NT(
            [torch.masked_scatter(t, m, s) for t, m, s in zip(base, mask, nested_source)],
            **base._meta(),
        )
        assert_close(output, reference)

    def test_index_add(self, device, float_dtype):
        base = NestedTensor(
            [torch.zeros(5, device=device, dtype=float_dtype), torch.zeros(4, device=device, dtype=float_dtype)]
        )
        index = torch.tensor([0, 2], device=device)
        src = NestedTensor(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0, 4.0], device=device, dtype=float_dtype),
            ],
            **base._meta(),
        )
        output = torch.index_add(base, 1, index, src)
        reference = NT([torch.index_add(t, 0, index, s) for t, s in zip(base, src)], **base._meta())
        assert_close(output, reference)

    def test_index_add_static_dim(self, device, float_dtype):
        base = NestedTensor(
            [
                torch.zeros(2, 4, device=device, dtype=float_dtype),
                torch.zeros(3, 4, device=device, dtype=float_dtype),
            ]
        )
        index = torch.tensor([0, 2], device=device)
        src = NestedTensor(
            [
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=float_dtype),
                torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], device=device, dtype=float_dtype),
            ],
            **base._meta(),
        )
        output = torch.index_add(base, 2, index, src)
        reference = NT([torch.index_add(t, 1, index, s) for t, s in zip(base, src)], **base._meta())
        assert_close(output, reference)

    def test_index_copy(self, device, float_dtype):
        base = NestedTensor(
            [torch.zeros(5, device=device, dtype=float_dtype), torch.zeros(4, device=device, dtype=float_dtype)]
        )
        index = torch.tensor([0, 2], device=device)
        src = NestedTensor(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0, 4.0], device=device, dtype=float_dtype),
            ],
            **base._meta(),
        )
        output = torch.index_copy(base, 1, index, src)
        reference = NT([torch.index_copy(t, 0, index, s) for t, s in zip(base, src)], **base._meta())
        assert_close(output, reference)

    def test_index_copy_static_dim(self, device, float_dtype):
        base = NestedTensor(
            [
                torch.zeros(2, 4, device=device, dtype=float_dtype),
                torch.zeros(3, 4, device=device, dtype=float_dtype),
            ]
        )
        index = torch.tensor([0, 2], device=device)
        src = NestedTensor(
            [
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=float_dtype),
                torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], device=device, dtype=float_dtype),
            ],
            **base._meta(),
        )
        output = torch.index_copy(base, 2, index, src)
        reference = NT([torch.index_copy(t, 1, index, s) for t, s in zip(base, src)], **base._meta())
        assert_close(output, reference)

    def test_index_put(self, device, float_dtype):
        base = NestedTensor(
            [torch.zeros(5, device=device, dtype=float_dtype), torch.zeros(4, device=device, dtype=float_dtype)]
        )
        index = torch.tensor([0, 2], device=device)
        values = NestedTensor(
            [
                torch.tensor([10.0, 20.0], device=device, dtype=float_dtype),
                torch.tensor([30.0, 40.0], device=device, dtype=float_dtype),
            ],
            **base._meta(),
        )
        output = torch.index_put(base, (index,), values, accumulate=False)
        reference = NT([torch.index_put(t, (index,), v) for t, v in zip(base, values)], **base._meta())
        assert_close(output, reference)

    def test_index_put_row_write(self, device, float_dtype):
        base = NestedTensor(
            [
                torch.zeros(4, 3, device=device, dtype=float_dtype),
                torch.zeros(5, 3, device=device, dtype=float_dtype),
            ]
        )
        index = torch.tensor([0, 2], device=device)
        values = NestedTensor(
            [
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device, dtype=float_dtype),
                torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], device=device, dtype=float_dtype),
            ],
            **base._meta(),
        )
        output = torch.index_put(base, (index,), values, accumulate=False)
        reference = NT([torch.index_put(t, (index,), v) for t, v in zip(base, values)], **base._meta())
        assert_close(output, reference)

    def test_index_put_row_write_dense_values(self, device, float_dtype):
        base = NestedTensor(
            [
                torch.zeros(4, 3, device=device, dtype=float_dtype),
                torch.zeros(5, 3, device=device, dtype=float_dtype),
            ]
        )
        index = torch.tensor([0, 2], device=device)
        values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device, dtype=float_dtype)
        output = torch.index_put(base, (index,), values, accumulate=False)
        reference = NT([torch.index_put(t, (index,), values) for t in base], **base._meta())
        assert_close(output, reference)

    def test_index_put_row_write_scalar_tensor(self, device, float_dtype):
        base = NestedTensor(
            [
                torch.zeros(4, 3, device=device, dtype=float_dtype),
                torch.zeros(5, 3, device=device, dtype=float_dtype),
            ]
        )
        index = torch.tensor([0, 2], device=device)
        value = torch.tensor(-3.0, device=device, dtype=float_dtype)
        output = torch.index_put(base, (index,), value, accumulate=False)
        reference = NT([torch.index_put(t, (index,), value) for t in base], **base._meta())
        assert_close(output, reference)

    def test_index_put_row_write_accumulate_duplicate_indices(self, device, float_dtype):
        base = NestedTensor(
            [
                torch.zeros(4, 3, device=device, dtype=float_dtype),
                torch.zeros(5, 3, device=device, dtype=float_dtype),
            ]
        )
        index = torch.tensor([0, 0, 2], device=device)
        values = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [10.0, 20.0, 30.0],
                [4.0, 5.0, 6.0],
            ],
            device=device,
            dtype=float_dtype,
        )
        output = torch.index_put(base, (index,), values, accumulate=True)
        reference = NT([torch.index_put(t, (index,), values, accumulate=True) for t in base], **base._meta())
        assert_close(output, reference)

    def test_index_put_multi_index(self, device, float_dtype):
        base = NestedTensor(
            [
                torch.zeros(4, 3, device=device, dtype=float_dtype),
                torch.zeros(5, 3, device=device, dtype=float_dtype),
            ]
        )
        rows = torch.tensor([[0, -1], [1, 2]], device=device)
        cols = torch.tensor([[1, 0], [2, 1]], device=device)
        values = NestedTensor(
            [
                torch.tensor([[13.0, 17.0], [19.0, 23.0]], device=device, dtype=float_dtype),
                torch.tensor([[29.0, 31.0], [37.0, 41.0]], device=device, dtype=float_dtype),
            ],
            **base._meta(),
        )
        output = torch.index_put(base, (rows, cols), values, accumulate=False)
        reference = NT([torch.index_put(t, (rows, cols), v) for t, v in zip(base, values)], **base._meta())
        assert_close(output, reference)

    @pytest.mark.parametrize("seed", [13, 23, 41])
    def test_index_put_randomized_dense_parity(self, device, seed):
        dtype = torch.float32
        shapes = ragged_shapes(seed, batch_size=3, min_len=3, max_len=6, trailing_shape=(4,))
        base = nested_rand(shapes, device, dtype)
        min_rows = min(shape[0] for shape in shapes)
        generator = torch.Generator()
        generator.manual_seed(seed)

        rows = torch.randint(0, min_rows, (2,), generator=generator)
        rows[-1] = rows[-1] - min_rows
        rows = rows.to(device=device, dtype=torch.long)

        shared_values = torch.randn(2, 4, device=device, dtype=dtype)
        row_output = torch.index_put(base, (rows,), shared_values, accumulate=False)
        row_reference = NT([torch.index_put(t, (rows,), shared_values) for t in base], **base._meta())
        assert_close(row_output, row_reference)

        dup_rows = rows.clone()
        dup_rows[0] = dup_rows[-1]
        dup_values = torch.randn(2, 4, device=device, dtype=dtype)
        dup_output = torch.index_put(base, (dup_rows,), dup_values, accumulate=True)
        dup_reference = NT([torch.index_put(t, (dup_rows,), dup_values, accumulate=True) for t in base], **base._meta())
        assert_close(dup_output, dup_reference)

        point_rows = torch.randint(0, min_rows, (2, 2), generator=generator)
        point_rows[0, 0] = point_rows[0, 0] - min_rows
        point_rows = point_rows.to(device=device, dtype=torch.long)
        point_cols = torch.randint(0, 4, (2, 2), generator=generator).to(device=device, dtype=torch.long)
        point_values = NestedTensor(
            [torch.randn(2, 2, device=device, dtype=dtype) for _ in range(len(base))],
            **base._meta(),
        )
        point_output = torch.index_put(base, (point_rows, point_cols), point_values, accumulate=False)
        point_reference = NT(
            [torch.index_put(t, (point_rows, point_cols), v) for t, v in zip(base, point_values)],
            **base._meta(),
        )
        assert_close(point_output, point_reference)


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
        reference = NT([torch.rand_like(t, dtype=torch.float64) for t in nt], **nt._meta())
        assert_close(output, reference)

        torch.manual_seed(456)
        output = torch.randn_like(nt)
        torch.manual_seed(456)
        reference = NT([torch.randn_like(t) for t in nt], **nt._meta())
        assert_close(output, reference)

    def test_dropout_and_bernoulli_match_packed_values(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.ones(4, device=device, dtype=float_dtype),
                torch.ones(2, device=device, dtype=float_dtype),
            ]
        )
        torch.manual_seed(0)
        output = torch.dropout(nt, p=0.5, train=True)
        torch.manual_seed(0)
        reference = _packed_result(nt, torch.dropout(nt._values, p=0.5, train=True))
        assert_close(output, reference)

        torch.manual_seed(0)
        output = torch.bernoulli(nt)
        torch.manual_seed(0)
        reference = _packed_result(nt, torch.bernoulli(nt._values))
        assert_close(output, reference)


class TestLikeCreators:

    def test_zeros_like(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(1, 3, device=device, dtype=float_dtype),
            ]
        )

        output = torch.zeros_like(nt)
        reference = NT([torch.zeros_like(t) for t in nt], **nt._meta())
        assert_close(output, reference)

    def test_ones_like_dtype(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(1, 3, device=device, dtype=float_dtype),
            ]
        )
        output = torch.ones_like(nt, dtype=torch.float64)
        reference = NT([torch.ones_like(t, dtype=torch.float64) for t in nt], **nt._meta())
        assert_close(output, reference)
        assert output.dtype == torch.float64

    def test_full_like_dtype(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(1, 3, device=device, dtype=float_dtype),
            ]
        )
        output = torch.full_like(nt, 3.0, dtype=torch.float32)
        reference = NT([torch.full_like(t, 3.0, dtype=torch.float32) for t in nt], **nt._meta())
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
        reference = NT([torch.randint_like(t, 10) for t in nt], **nt._meta())
        assert_close(output, reference)


class TestFlipAndRoll:

    def test_flip_translates_dims_and_rejects_batch(self, device):
        nt = NT([torch.tensor([[1, 2], [3, 4]], device=device), torch.tensor([[5, 6]], device=device)])
        output = torch.flip(nt, dims=(-1,))
        reference = NT([torch.flip(t, dims=(-1,)) for t in nt], **nt._meta())
        assert_close(output, reference)

        with pytest.raises(ValueError):
            torch.flip(nt, dims=(0,))
        with pytest.raises(ValueError):
            torch.flip(nt, dims=(-3,))

    def test_roll_supports_dims_none(self, device):
        nt = NT([torch.tensor([1, 2, 3], device=device), torch.tensor([4, 5], device=device)])
        output = torch.roll(nt, shifts=1)
        reference = NT([torch.roll(t, shifts=1) for t in nt], **nt._meta())
        assert_close(output, reference)

    def test_roll_translates_dims_and_rejects_batch(self, device):
        nt = NT([torch.tensor([[1, 2], [3, 4]], device=device), torch.tensor([[5, 6]], device=device)])
        output = torch.roll(nt, shifts=1, dims=-1)
        reference = NT([torch.roll(t, shifts=1, dims=-1) for t in nt], **nt._meta())
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
        reference = NT([torch.softmax(t, dim=1) for t in nt], **nt._meta())
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

        output = torch.log_softmax(nt, dim=2)
        reference = NT([torch.log_softmax(t, dim=1) for t in nt], **nt._meta())
        assert_close(output, reference, atol=1e-6, rtol=1e-6)

        with pytest.raises(ValueError):
            torch.softmax(nt, dim=0)


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
        reference = NT([torch.matmul(t, weight) for t in nt], **nt._meta())
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
        reference = NT([torch.matmul(weight, t) for t in nt], **nt._meta())
        assert_close(output, reference)

    def test_mm_match_per_sample(self, device, float_dtype):
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
            **left._meta(),
        )
        output = torch.mm(left, right)
        reference = NT([torch.mm(a, b) for a, b in zip(left, right)], **left._meta())
        assert_close(output, reference)

    def test_mm_uses_packed_aten_path(self, device, float_dtype):
        left = NT(
            [
                torch.randn(3, 4, device=device, dtype=float_dtype),
                torch.randn(2, 4, device=device, dtype=float_dtype),
            ]
        )
        right = torch.randn(4, 5, device=device, dtype=float_dtype)
        original_iter = NestedTensor.__iter__

        def _fail_iter(self):
            raise AssertionError("torch.mm should not iterate per element for packed NT x dense inputs")

        NestedTensor.__iter__ = _fail_iter
        try:
            output = torch.mm(left, right)
        finally:
            NestedTensor.__iter__ = original_iter
        reference = NT([torch.mm(a, right) for a in left], **left._meta())
        assert_close(output, reference)

    def test_bmm_match_per_sample(self, device, float_dtype):
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
            **left._meta(),
        )
        output = torch.bmm(left, right)
        reference = NT([torch.bmm(a, b) for a, b in zip(left, right)], **left._meta())
        assert_close(output, reference)

    def test_bmm_uses_packed_aten_path(self, device, float_dtype):
        left = NT(
            [
                torch.randn(2, 1, 3, device=device, dtype=float_dtype),
                torch.randn(1, 1, 3, device=device, dtype=float_dtype),
            ]
        )
        right = NT(
            [
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
                torch.randn(1, 3, 4, device=device, dtype=float_dtype),
            ],
            **left._meta(),
        )
        original_iter = NestedTensor.__iter__

        def _fail_iter(self):
            raise AssertionError("torch.bmm should not iterate per element for matching-offset packed inputs")

        NestedTensor.__iter__ = _fail_iter
        try:
            output = torch.bmm(left, right)
        finally:
            NestedTensor.__iter__ = original_iter
        reference = NT([torch.bmm(a, b) for a, b in zip(left, right)], **left._meta())
        assert_close(output, reference)


class TestCountNonzero:

    def test_count_nonzero(self, device):
        nt = NT([torch.tensor([1, 0, 2], device=device), torch.tensor([0], device=device)])
        output = torch.count_nonzero(nt)
        assert_close(output, torch.tensor(2, device=device))

        output = torch.count_nonzero(nt, dim=1)
        assert_close(output, torch.tensor([2, 0], device=device))

        with pytest.raises(ValueError):
            torch.count_nonzero(nt, dim=0)


class TestDist:

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
            **a._meta(),
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
        reference = NT([torch.rot90(t, k=1, dims=(0, 1)) for t in nt], **nt._meta())
        assert_close(output, reference)

        with pytest.raises(ValueError):
            torch.rot90(nt)


class TestNonzeroAndTake:

    def test_nonzero_ignores_padding_value(self, device):
        nt = NT([torch.tensor([0, 1, 0], device=device), torch.tensor([2], device=device)], padding_value=9)
        output = torch.nonzero(nt, as_tuple=False)
        reference = NT([torch.nonzero(t, as_tuple=False) for t in nt], **nt._meta())
        assert_close(output, reference)

        output = torch.nonzero(nt, as_tuple=True)
        reference = (NT([torch.nonzero(t, as_tuple=True)[0] for t in nt], **nt._meta()),)
        assert isinstance(output, tuple)
        assert len(output) == 1
        assert_close(output[0], reference[0])

    def test_nonzero_matches_dense_empty_result_dtype(self, device):
        nt = NT([torch.zeros(3, device=device), torch.zeros(1, device=device)], padding_value=7)
        output = torch.nonzero(nt, as_tuple=False)
        reference = NT([torch.nonzero(t, as_tuple=False) for t in nt], **nt._meta())

        assert_close(output, reference)
        assert output.dtype == torch.long
        assert output[0].shape == torch.Size([0, 1])
        assert output[1].shape == torch.Size([0, 1])

    def test_nonzero_flattened_fallback_matches_per_element(self, device):
        nt = NT(
            [
                torch.tensor([[1.0], [0.0]], device=device),
                torch.tensor([[0.0, 2.0]], device=device),
            ]
        )

        output = torch.nonzero(nt, as_tuple=False)
        reference = NT([torch.nonzero(t, as_tuple=False) for t in nt], **nt._meta())
        assert_close(output, reference)

    def test_nonzero_as_tuple_requires_matching_ndim(self, device):
        with pytest.raises(ValueError, match="same number of dimensions"):
            NT([torch.tensor([1, 0], device=device), torch.tensor([[1], [0]], device=device)])

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
        index = NT([torch.tensor([1], device=device), torch.tensor([0, 2], device=device)], **nt._meta())
        output = torch.take(nt, index)
        reference = NT([torch.take(t.reshape(-1), i) for t, i in zip(nt, index)], **nt._meta())
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

    def test_order_statistics_static_dim(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([[3.0, 1.0, 2.0], [5.0, 4.0, 6.0]], device=device, dtype=float_dtype),
                torch.tensor([[9.0, 7.0, 8.0], [0.0, 2.0, 1.0], [6.0, 5.0, 4.0]], device=device, dtype=float_dtype),
            ]
        )

        kthvalue_out = torch.kthvalue(nt, k=2, dim=2)
        kthvalue_ref = tuple(NT([torch.kthvalue(t, k=2, dim=1)[i] for t in nt], **nt._meta()) for i in range(2))
        assert_close(kthvalue_out[0], kthvalue_ref[0])
        assert_close(kthvalue_out[1], kthvalue_ref[1])

        median_out = torch.median(nt, dim=2, keepdim=True)
        median_ref = tuple(NT([torch.median(t, dim=1, keepdim=True)[i] for t in nt], **nt._meta()) for i in range(2))
        assert_close(median_out[0], median_ref[0])
        assert_close(median_out[1], median_ref[1])

        mode_out = torch.mode(nt, dim=2)
        mode_ref = tuple(NT([torch.mode(t, dim=1)[i] for t in nt], **nt._meta()) for i in range(2))
        assert_close(mode_out[0], mode_ref[0])
        assert_close(mode_out[1], mode_ref[1])

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

    def test_nanmedian(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, float("nan"), 3.0], device=device, dtype=float_dtype),
                torch.tensor([float("nan"), 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.nanmedian(nt)
        reference = torch.nanmedian(torch.cat([t.reshape(-1) for t in nt]))
        assert_close(output, reference, equal_nan=True)

        nt_static = NT(
            [
                torch.tensor([[1.0, float("nan"), 3.0], [2.0, 4.0, float("nan")]], device=device, dtype=float_dtype),
                torch.tensor(
                    [[float("nan"), 7.0, 8.0], [0.0, float("nan"), 1.0], [6.0, 5.0, 4.0]],
                    device=device,
                    dtype=float_dtype,
                ),
            ]
        )
        output = torch.nanmedian(nt_static, dim=2)
        reference = tuple(NT([torch.nanmedian(t, dim=1)[i] for t in nt_static], **nt_static._meta()) for i in range(2))
        assert_close(output[0], reference[0], equal_nan=True)
        assert_close(output[1], reference[1])

    def test_mode(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, float("nan"), 3.0], device=device, dtype=float_dtype),
                torch.tensor([float("nan"), 5.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.mode(nt, dim=1)
        reference = tuple(torch.stack([torch.mode(t, dim=0)[idx] for t in nt]) for idx in range(2))
        assert_close(output[0], reference[0], equal_nan=True)
        assert_close(output[1], reference[1])

    def test_quantile(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, float("nan"), 3.0], device=device, dtype=float_dtype),
                torch.tensor([float("nan"), 5.0], device=device, dtype=float_dtype),
            ]
        )
        pair = _run_or_expect_unsupported(
            lambda: torch.quantile(nt, 0.5),
            lambda: torch.quantile(torch.cat([t.reshape(-1) for t in nt]), 0.5),
        )
        if pair is None:
            return
        output, reference = pair
        assert_close(output, reference, equal_nan=True)

        output = torch.quantile(nt, 0.5, keepdim=True)
        assert output.shape == (1, 1)

    def test_nanquantile(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, float("nan"), 3.0], device=device, dtype=float_dtype),
                torch.tensor([float("nan"), 5.0], device=device, dtype=float_dtype),
            ]
        )
        pair = _run_or_expect_unsupported(
            lambda: torch.nanquantile(nt, 0.5),
            lambda: torch.nanquantile(torch.cat([t.reshape(-1) for t in nt]), 0.5),
        )
        if pair is None:
            return
        output, reference = pair
        assert_close(output, reference, equal_nan=True)


class TestMatrixOps:
    """Tests for matrix/linalg ops registered in torch_functions.py."""

    def test_triu(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.triu(nt)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.triu(a))
        assert_close(result[1], torch.triu(b))

    def test_triu_diagonal(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.triu(nt, diagonal=1)
        assert_close(result[0], torch.triu(a, diagonal=1))
        assert_close(result[1], torch.triu(b, diagonal=1))

    def test_tril(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.tril(nt)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.tril(a))
        assert_close(result[1], torch.tril(b))

    def test_matrix_exp(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(2, 2, device=device, dtype=float_dtype)
        nt = NT([a, b])
        pair = _run_or_expect_unsupported(
            lambda: torch.matrix_exp(nt),
            lambda: NT([torch.matrix_exp(a), torch.matrix_exp(b)], **nt._meta()),
        )
        if pair is None:
            return
        result, reference = pair
        assert isinstance(result, NestedTensor)
        assert_close(result, reference, equal_nan=True)

    def test_diag_2d_to_1d(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.diag(nt)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.diag(a))
        assert_close(result[1], torch.diag(b))

    def test_diag_1d_to_2d(self, device, float_dtype):
        a = torch.randn(3, device=device, dtype=float_dtype)
        b = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.diag(nt)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.diag(a))
        assert_close(result[1], torch.diag(b))

    def test_diagflat(self, device, float_dtype):
        a = torch.randn(3, device=device, dtype=float_dtype)
        b = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.diagflat(nt)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.diagflat(a))
        assert_close(result[1], torch.diagflat(b))

    def test_diagonal(self, device, float_dtype):
        a = torch.randn(3, 4, 5, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, 5, device=device, dtype=float_dtype)
        nt = NT([a, b])
        # dim1=1, dim2=2 in NestedTensor → dim1=0, dim2=1 in element
        result = torch.diagonal(nt, dim1=1, dim2=2)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.diagonal(a, dim1=0, dim2=1))
        assert_close(result[1], torch.diagonal(b, dim1=0, dim2=1))

    def test_diagonal_static_dims(self, device, float_dtype):
        a = torch.randn(2, 3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 3, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.diagonal(nt, dim1=2, dim2=3)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.diagonal(a, dim1=1, dim2=2))
        assert_close(result[1], torch.diagonal(b, dim1=1, dim2=2))

    def test_trace(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.trace(nt)
        assert isinstance(result, NestedTensor)
        # Scalar results get packed as 1-D elements in NestedTensor
        assert_close(result[0].squeeze(), torch.trace(a))
        assert_close(result[1].squeeze(), torch.trace(b))

    def test_det(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        pair = _run_or_expect_unsupported(lambda: torch.det(nt), lambda: (torch.det(a), torch.det(b)))
        if pair is None:
            return
        result, reference = pair
        ref_a, ref_b = reference
        assert isinstance(result, NestedTensor)
        assert_close(result[0].squeeze(), ref_a)
        assert_close(result[1].squeeze(), ref_b)

    def test_inverse(self, device, float_dtype):
        # Use well-conditioned matrices
        a = torch.eye(3, device=device, dtype=float_dtype) + 0.1 * torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.eye(4, device=device, dtype=float_dtype) + 0.1 * torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        pair = _run_or_expect_unsupported(
            lambda: torch.inverse(nt),
            lambda: NT([torch.inverse(a), torch.inverse(b)], **nt._meta()),
        )
        if pair is None:
            return
        result, reference = pair
        assert isinstance(result, NestedTensor)
        assert_close(result, reference)

    def test_matrix_power(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.matrix_power(nt, 3)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.matrix_power(a, 3))
        assert_close(result[1], torch.matrix_power(b, 3))

    def test_repeat_interleave_no_dim(self, device, float_dtype):
        a = torch.randn(2, 3, device=device, dtype=float_dtype)
        b = torch.randn(3, 2, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.repeat_interleave(nt, 2)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.repeat_interleave(a, 2))
        assert_close(result[1], torch.repeat_interleave(b, 2))

    def test_repeat_interleave_with_dim(self, device, float_dtype):
        a = torch.randn(2, 3, device=device, dtype=float_dtype)
        b = torch.randn(2, 3, device=device, dtype=float_dtype)
        nt = NT([a, b])
        # dim=1 in NestedTensor → dim=0 in element
        result = torch.repeat_interleave(nt, 2, dim=1)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.repeat_interleave(a, 2, dim=0))
        assert_close(result[1], torch.repeat_interleave(b, 2, dim=0))


class TestLinalgOps:
    """Tests for torch.linalg ops registered in torch_functions.py."""

    def test_linalg_inv(self, device, float_dtype):
        a = torch.eye(3, device=device, dtype=float_dtype) + 0.1 * torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.eye(4, device=device, dtype=float_dtype) + 0.1 * torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        pair = _run_or_expect_unsupported(
            lambda: torch.linalg.inv(nt),
            lambda: NT([torch.linalg.inv(a), torch.linalg.inv(b)], **nt._meta()),
        )
        if pair is None:
            return
        result, reference = pair
        assert isinstance(result, NestedTensor)
        assert_close(result, reference)

    def test_linalg_det(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        pair = _run_or_expect_unsupported(
            lambda: torch.linalg.det(nt), lambda: (torch.linalg.det(a), torch.linalg.det(b))
        )
        if pair is None:
            return
        result, reference = pair
        ref_a, ref_b = reference
        assert isinstance(result, NestedTensor)
        assert_close(result[0].squeeze(), ref_a)
        assert_close(result[1].squeeze(), ref_b)

    def test_linalg_norm_no_dim(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 2, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.linalg.norm(nt)
        assert isinstance(result, NestedTensor)
        assert_close(result[0].squeeze(), torch.linalg.norm(a))
        assert_close(result[1].squeeze(), torch.linalg.norm(b))

    def test_linalg_norm_with_dim(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        # dim=1 in NestedTensor → dim=0 in element
        result = torch.linalg.norm(nt, dim=1)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.linalg.norm(a, dim=0))
        assert_close(result[1], torch.linalg.norm(b, dim=0))

    def test_linalg_norm_negative_ord_ragged_dim_falls_back(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])

        result = torch.linalg.norm(nt, ord=-1, dim=1)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.linalg.norm(a, ord=-1, dim=0))
        assert_close(result[1], torch.linalg.norm(b, ord=-1, dim=0))

    def test_linalg_svd(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 3, device=device, dtype=float_dtype)
        nt = NT([a, b])
        try:
            _, S_a, _ = torch.linalg.svd(a)
            _, S_b, _ = torch.linalg.svd(b)
        except (RuntimeError, NotImplementedError) as error:
            with pytest.raises(type(error)):
                torch.linalg.svd(nt)
            return
        U, S, Vh = torch.linalg.svd(nt)
        assert isinstance(U, NestedTensor)
        assert isinstance(S, NestedTensor)
        assert isinstance(Vh, NestedTensor)
        # SVD is unique up to sign, so compare singular values
        assert_close(S[0], S_a)
        assert_close(S[1], S_b)

    def test_linalg_qr(self, device, float_dtype):
        a = torch.randn(4, 3, device=device, dtype=float_dtype)
        b = torch.randn(5, 3, device=device, dtype=float_dtype)
        nt = NT([a, b])
        try:
            torch.linalg.qr(a)
            torch.linalg.qr(b)
        except (RuntimeError, NotImplementedError) as error:
            with pytest.raises(type(error)):
                torch.linalg.qr(nt)
            return
        Q, R = torch.linalg.qr(nt)
        assert isinstance(Q, NestedTensor)
        assert isinstance(R, NestedTensor)
        # Verify Q @ R ≈ original
        atol, rtol = low_precision_cuda_tolerances(
            device,
            float_dtype,
            default=(1e-8, 1e-5),
            fp16=(5e-4, 5e-3),
            bf16=(5e-3, 3e-2),
        )
        assert_close(Q[0] @ R[0], a, atol=atol, rtol=rtol)
        assert_close(Q[1] @ R[1], b, atol=atol, rtol=rtol)

    def test_linalg_cholesky(self, device, float_dtype):
        # Create positive-definite matrices
        a_raw = torch.randn(3, 3, device=device, dtype=float_dtype)
        a = a_raw @ a_raw.T + 3 * torch.eye(3, device=device, dtype=float_dtype)
        b_raw = torch.randn(4, 4, device=device, dtype=float_dtype)
        b = b_raw @ b_raw.T + 3 * torch.eye(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        pair = _run_or_expect_unsupported(
            lambda: torch.linalg.cholesky(nt),
            lambda: NT([torch.linalg.cholesky(a), torch.linalg.cholesky(b)], **nt._meta()),
        )
        if pair is None:
            return
        result, reference = pair
        assert isinstance(result, NestedTensor)
        assert_close(result, reference)

    def test_linalg_eigh(self, device, float_dtype):
        # Create symmetric matrices
        a_raw = torch.randn(3, 3, device=device, dtype=float_dtype)
        a = a_raw + a_raw.T
        b_raw = torch.randn(4, 4, device=device, dtype=float_dtype)
        b = b_raw + b_raw.T
        nt = NT([a, b])
        try:
            w_a, _ = torch.linalg.eigh(a)
            w_b, _ = torch.linalg.eigh(b)
        except (RuntimeError, NotImplementedError) as error:
            with pytest.raises(type(error)):
                torch.linalg.eigh(nt)
            return
        eigenvalues, eigenvectors = torch.linalg.eigh(nt)
        assert isinstance(eigenvalues, NestedTensor)
        assert isinstance(eigenvectors, NestedTensor)
        assert_close(eigenvalues[0], w_a)
        assert_close(eigenvalues[1], w_b)

    def test_linalg_solve(self, device, float_dtype):
        a = torch.eye(3, device=device, dtype=float_dtype) + 0.1 * torch.randn(3, 3, device=device, dtype=float_dtype)
        b_vec = torch.randn(3, 1, device=device, dtype=float_dtype)
        a2 = torch.eye(4, device=device, dtype=float_dtype) + 0.1 * torch.randn(4, 4, device=device, dtype=float_dtype)
        b2_vec = torch.randn(4, 1, device=device, dtype=float_dtype)
        nt_a = NT([a, a2])
        nt_b = NT([b_vec, b2_vec])
        pair = _run_or_expect_unsupported(
            lambda: torch.linalg.solve(nt_a, nt_b),
            lambda: NT([torch.linalg.solve(a, b_vec), torch.linalg.solve(a2, b2_vec)], **nt_a._meta()),
        )
        if pair is None:
            return
        result, reference = pair
        assert isinstance(result, NestedTensor)
        assert_close(result, reference)

    def test_linalg_solve_mismatched_batch_lengths_raises(self, device, float_dtype):
        a = torch.eye(3, device=device, dtype=float_dtype)
        b = torch.randn(3, 1, device=device, dtype=float_dtype)
        nt_a = NT([a, a])
        nt_b = NT([b])
        with pytest.raises(ValueError, match="linalg.solve: NestedTensor batch length mismatch"):
            torch.linalg.solve(nt_a, nt_b)


class TestMissingOpsAudit:
    """Tests for ops found missing during the torch.* audit."""

    def test_hypot(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, device=device, dtype=float_dtype)
        nt_a = NT([a, a])
        nt_b = NT([b, b])
        result = torch.hypot(nt_a, nt_b)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.hypot(a, b))

    def test_logaddexp(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, device=device, dtype=float_dtype)
        nt_a = NT([a, a])
        nt_b = NT([b, b])
        result = torch.logaddexp(nt_a, nt_b)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.logaddexp(a, b))

    def test_logaddexp2(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, device=device, dtype=float_dtype)
        nt_a = NT([a, a])
        nt_b = NT([b, b])
        result = torch.logaddexp2(nt_a, nt_b)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.logaddexp2(a, b))

    def test_nextafter(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, device=device, dtype=float_dtype)
        nt_a = NT([a, a])
        nt_b = NT([b, b])
        result = torch.nextafter(nt_a, nt_b)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.nextafter(a, b))

    def test_movedim(self, device, float_dtype):
        a = torch.randn(3, 4, 5, device=device, dtype=float_dtype)
        b = torch.randn(2, 4, 5, device=device, dtype=float_dtype)
        nt = NT([a, b])
        # dim -1 -> dim 1 in NestedTensor means last -> second
        result = torch.movedim(nt, -1, 1)
        assert isinstance(result, NestedTensor)
        ref = torch.moveaxis(nt, -1, 1)
        assert_close(result[0], ref[0])
        assert_close(result[1], ref[1])

    def test_searchsorted_both_nested(self, device, float_dtype):
        sorted_a = torch.tensor([1.0, 3.0, 5.0], device=device, dtype=float_dtype)
        sorted_b = torch.tensor([2.0, 4.0, 6.0, 8.0], device=device, dtype=float_dtype)
        vals_a = torch.tensor([2.0, 4.0], device=device, dtype=float_dtype)
        vals_b = torch.tensor([3.0, 7.0], device=device, dtype=float_dtype)
        nt_sorted = NT([sorted_a, sorted_b])
        nt_vals = NT([vals_a, vals_b])
        result = torch.searchsorted(nt_sorted, nt_vals)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.searchsorted(sorted_a, vals_a))
        assert_close(result[1], torch.searchsorted(sorted_b, vals_b))

    def test_searchsorted_mismatched_batch_lengths_raises(self, device, float_dtype):
        sorted_a = torch.tensor([1.0, 3.0, 5.0], device=device, dtype=float_dtype)
        vals_a = torch.tensor([2.0, 4.0], device=device, dtype=float_dtype)
        nt_sorted = NT([sorted_a, sorted_a])
        nt_vals = NT([vals_a])
        with pytest.raises(ValueError, match="searchsorted: NestedTensor batch length mismatch"):
            torch.searchsorted(nt_sorted, nt_vals)

    def test_searchsorted_shared_boundaries(self, device, float_dtype):
        boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0], device=device, dtype=float_dtype)
        vals_a = torch.tensor([2.0, 4.0, 6.0], device=device, dtype=float_dtype)
        vals_b = torch.tensor([0.5, 8.0], device=device, dtype=float_dtype)
        nt_vals = NT([vals_a, vals_b])
        result = torch.searchsorted(boundaries, nt_vals)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.searchsorted(boundaries, vals_a))
        assert_close(result[1], torch.searchsorted(boundaries, vals_b))

    def test_searchsorted_nested_sorter_requires_nested_sorted_sequence(self, device, float_dtype):
        boundaries = torch.tensor([1.0, 3.0, 5.0], device=device, dtype=float_dtype)
        vals_a = torch.tensor([2.0, 4.0], device=device, dtype=float_dtype)
        nt_vals = NT([vals_a])
        nt_sorter = NT([torch.tensor([0, 1, 2], device=device, dtype=torch.long)])
        with pytest.raises(TypeError, match="NestedTensor sorter requires sorted_sequence"):
            torch.searchsorted(boundaries, nt_vals, sorter=nt_sorter)

    def test_searchsorted_nested_sorter_with_nested_sorted_sequence(self, device, float_dtype):
        sorted_a = torch.tensor([1.0, 3.0, 5.0], device=device, dtype=float_dtype)
        sorted_b = torch.tensor([2.0, 4.0, 6.0], device=device, dtype=float_dtype)
        sorter_a = torch.tensor([0, 1, 2], device=device, dtype=torch.long)
        sorter_b = torch.tensor([0, 1, 2], device=device, dtype=torch.long)
        nt_sorted = NT([sorted_a, sorted_b])
        nt_sorter = NT([sorter_a, sorter_b])
        result = torch.searchsorted(nt_sorted, float(3.5), sorter=nt_sorter)
        reference = NT(
            [
                torch.searchsorted(sorted_a, float(3.5), sorter=sorter_a),
                torch.searchsorted(sorted_b, float(3.5), sorter=sorter_b),
            ],
            **nt_sorted._meta(),
        )
        assert_close(result, reference)

    def test_bucketize(self, device, float_dtype):
        boundaries = torch.tensor([1.0, 3.0, 5.0], device=device, dtype=float_dtype)
        a = torch.tensor([0.5, 2.0, 4.0, 6.0], device=device, dtype=float_dtype)
        b = torch.tensor([1.5, 3.5], device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.bucketize(nt, boundaries)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.bucketize(a, boundaries))
        assert_close(result[1], torch.bucketize(b, boundaries))

    def test_einsum_nt_times_tensor(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, 2, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.einsum("ij,jk->ik", nt, w)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.einsum("ij,jk->ik", a, w))
        assert_close(result[1], torch.einsum("ij,jk->ik", b, w))

    def test_einsum_both_nested(self, device, float_dtype):
        a1 = torch.randn(3, 4, device=device, dtype=float_dtype)
        a2 = torch.randn(3, 4, device=device, dtype=float_dtype)
        b1 = torch.randn(5, 4, device=device, dtype=float_dtype)
        b2 = torch.randn(5, 4, device=device, dtype=float_dtype)
        nt1 = NT([a1, b1])
        nt2 = NT([a2, b2])
        # Element-wise multiplication and sum over last dim
        result = torch.einsum("ij,ij->i", nt1, nt2)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.einsum("ij,ij->i", a1, a2))
        assert_close(result[1], torch.einsum("ij,ij->i", b1, b2))

    def test_einsum_mismatched_batch_lengths_raises(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, device=device, dtype=float_dtype)
        nt1 = NT([a, a])
        nt2 = NT([b])
        with pytest.raises(ValueError, match="einsum: NestedTensor batch length mismatch"):
            torch.einsum("ij,ij->i", nt1, nt2)
