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
from tests.tensors.utils import assert_close, low_precision_cuda_tolerances, nested_rand, packed_result, ragged_shapes

NT = NestedTensor


def _run_or_expect_unsupported(nested_call, tensor_call):
    try:
        reference = tensor_call()
    except RuntimeError as error:
        with pytest.raises(type(error)):
            nested_call()
        return None
    return nested_call(), reference


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

    def test_add_supports_multi_ragged_layouts(self, device, float_dtype):
        nt = NT(
            [
                torch.arange(6, device=device, dtype=float_dtype).reshape(2, 3),
                torch.arange(4, device=device, dtype=float_dtype).reshape(1, 4),
            ]
        )
        other = NT([torch.ones_like(t) for t in nt], **nt._meta())
        output = torch.add(nt, other)
        reference = NT([torch.add(x, y) for x, y in zip(nt, other)], **nt._meta())
        assert_close(output, reference)

    def test_add_dense_with_variable_length_dim(self, device, float_dtype):
        """NT[B, var_seq, D] + dense[1, max_seq, D] — the positional embedding pattern."""
        nt = NT(
            [
                torch.randn(5, 8, device=device, dtype=float_dtype),
                torch.randn(3, 8, device=device, dtype=float_dtype),
            ]
        )
        dense = torch.randn(1, 5, 8, device=device, dtype=float_dtype)
        output = nt + dense
        for i, elem in enumerate(nt):
            expected = elem + dense[0, : elem.shape[0]]
            assert_close(output._storage[i], expected)
        # Reverse
        output_rev = dense + nt
        for i, elem in enumerate(nt):
            expected = dense[0, : elem.shape[0]] + elem
            assert_close(output_rev._storage[i], expected)

    def test_add_dense_broadcastable_with_values(self, device, float_dtype):
        """NT[B, var_seq, D] + dense[D] — broadcast across packed dim."""
        nt = NT(
            [
                torch.randn(5, 8, device=device, dtype=float_dtype),
                torch.randn(3, 8, device=device, dtype=float_dtype),
            ]
        )
        bias = torch.randn(8, device=device, dtype=float_dtype)
        output = nt + bias
        for i, elem in enumerate(nt):
            assert_close(output._storage[i], elem + bias)

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

    def test_cat_dim0_incompatible_layouts(self):
        nt1 = NT([torch.arange(6.0).reshape(2, 3), torch.arange(3.0).reshape(1, 3)])
        nt2 = NT([torch.arange(8.0).reshape(2, 4), torch.arange(4.0).reshape(1, 4)])
        output = torch.cat((nt1, nt2), dim=0)
        reference = NT([*nt1, *nt2], **nt1._meta())
        assert_close(output, reference)

    def test_cat_dim0_preserves_state(self, device, float_dtype):
        nt = NestedTensor([torch.ones(2, device=device, dtype=float_dtype)])
        mixed = torch.zeros(2, device=device, dtype=float_dtype)
        output = cat((nt, mixed), dim=0)
        assert isinstance(output, NestedTensor)
        assert output.device == nt.device
        reference = NT([nt[0], mixed], **nt._meta())
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

    def test_cat_non_zero_dim_length_mismatch(self):
        nt1 = NestedTensor([torch.ones(2)])
        nt2 = NestedTensor([torch.ones(2), torch.ones(3)])
        with pytest.raises(ValueError):
            cat((nt1, nt2), dim=1)

    def test_cat_non_zero_dim_requires_nested_tensors(self):
        nt = NestedTensor([torch.ones(2)])
        with pytest.raises(NotImplementedError):
            cat((nt, torch.ones(2)), dim=1)


class TestCompile:

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize("op", [torch.det, torch.linalg.det], ids=["det", "linalg_det"])
    @pytest.mark.parametrize(
        "shapes",
        [[(2, 2), (2, 2)], [(3, 3), (4, 4)]],
        ids=["same_shape", "ragged"],
    )
    def test_compile_det(self, op, shapes):
        torch.manual_seed(1016)
        nt = NT([torch.randn(*s, dtype=torch.float32) for s in shapes])
        compiled = torch.compile(lambda x, operator=op: operator(x), backend="inductor", fullgraph=True)
        output = compiled(nt)
        reference = NT([op(t) for t in nt], **nt._meta())
        assert isinstance(output, NestedTensor)
        assert output._has_same_layout(reference)
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize(
        "view_fn",
        [
            pytest.param(lambda x: x.view(-1, 3), id="packed_fastpath"),
            pytest.param(lambda x: x.view(2, -1), id="explicit_batch_reduced_rank"),
        ],
    )
    def test_compile_view_method(self, view_fn, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(4, 3, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(view_fn, backend="inductor", fullgraph=True)
        output = compiled(nt)
        reference = view_fn(nt)
        assert isinstance(output, NestedTensor)
        assert output._has_same_layout(reference)
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize(
        ("op", "ref_fn"),
        [
            pytest.param(lambda x: x.unsqueeze(2).squeeze(2), lambda t: t.unsqueeze(1).squeeze(1), id="squeeze"),
            pytest.param(
                lambda x: x.view(-1, 3).unflatten(2, (1, 3)),
                lambda t: t.view(-1, 3).unflatten(1, (1, 3)),
                id="unflatten",
            ),
        ],
    )
    def test_compile_shape_rebuild_ops(self, op, ref_fn, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(4, 3, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(op, backend="inductor", fullgraph=True)
        output = compiled(nt)
        reference = NT([ref_fn(t) for t in nt], **nt._meta())
        assert isinstance(output, NestedTensor)
        assert output._has_same_layout(reference)
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize(
        "torch_fn,ref_fn",
        [
            pytest.param(
                lambda x: torch.topk(x, 2, dim=2, largest=True, sorted=True),
                lambda t: torch.topk(t, 2, dim=1).values,
                id="topk",
            ),
            pytest.param(lambda x: torch.cumsum(x, dim=2), lambda t: torch.cumsum(t, dim=1), id="cumsum"),
            pytest.param(lambda x: torch.cumprod(x, dim=2), lambda t: torch.cumprod(t, dim=1), id="cumprod"),
            pytest.param(
                lambda x: torch.logcumsumexp(x, dim=2), lambda t: torch.logcumsumexp(t, dim=1), id="logcumsumexp"
            ),
            pytest.param(lambda x: torch.cummax(x, dim=2), lambda t: torch.cummax(t, dim=1).values, id="cummax"),
            pytest.param(lambda x: torch.cummin(x, dim=2), lambda t: torch.cummin(t, dim=1).values, id="cummin"),
            pytest.param(lambda x: torch.flip(x, dims=[2]), lambda t: torch.flip(t, dims=[1]), id="flip"),
            pytest.param(
                lambda x: torch.sort(x, dim=2, descending=False),
                lambda t: torch.sort(t, dim=1, descending=False).values,
                id="sort",
            ),
            pytest.param(
                lambda x: torch.argsort(x, dim=2, descending=False),
                lambda t: torch.argsort(t, dim=1, descending=False),
                id="argsort",
            ),
            pytest.param(lambda x: torch.softmax(x, dim=1), lambda t: torch.softmax(t, dim=0), id="softmax"),
            pytest.param(
                lambda x: torch.log_softmax(x, dim=1), lambda t: torch.log_softmax(t, dim=0), id="log_softmax"
            ),
            pytest.param(lambda x: torch.layer_norm(x, (2,)), lambda t: torch.layer_norm(t, (2,)), id="layer_norm"),
        ],
    )
    def test_compile_ragged_op(self, torch_fn, ref_fn):
        nt = NT(
            [
                torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
                torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
            ]
        )
        compiled = torch.compile(torch_fn, backend="inductor", fullgraph=True)
        result = compiled(nt)
        if isinstance(result, tuple):
            result = result[0]
        reference = NT([ref_fn(t) for t in nt], **nt._meta())
        assert isinstance(result, NestedTensor)
        assert result._has_same_layout(reference)
        assert_close(result, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compile_dropout_and_bernoulli(self):
        nt = NT(
            [
                torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
                torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
            ]
        )
        nt_prob = NT(
            [
                torch.tensor([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]),
                torch.tensor([[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.35, 0.65], [0.6, 0.4]]),
            ]
        )

        def _compile(fn):
            return torch.compile(fn, backend="inductor", fullgraph=True)

        dropout_eval_fn = _compile(lambda x: torch.dropout(x, p=0.2, train=False))
        assert_close(dropout_eval_fn(nt), nt)

        dropout_train_fn = _compile(lambda x: torch.dropout(x, p=0.2, train=True))
        dropout_values_fn = _compile(lambda x: torch.dropout(x, p=0.2, train=True))
        torch.manual_seed(1016)
        dropout_comp = dropout_train_fn(nt)
        torch.manual_seed(1016)
        dropout_ref = dropout_values_fn(nt._values)
        assert isinstance(dropout_comp, NestedTensor)
        assert dropout_comp._has_same_layout(nt)
        assert_close(dropout_comp._values, dropout_ref)

        bernoulli_fn = _compile(lambda x: torch.bernoulli(x))
        bernoulli_values_fn = _compile(lambda x: torch.bernoulli(x))
        torch.manual_seed(1016)
        bernoulli_comp = bernoulli_fn(nt_prob)
        torch.manual_seed(1016)
        bernoulli_ref = bernoulli_values_fn(nt_prob._values)
        assert isinstance(bernoulli_comp, NestedTensor)
        assert bernoulli_comp._has_same_layout(nt_prob)
        assert_close(bernoulli_comp._values, bernoulli_ref)

    @pytest.mark.skipif(
        not hasattr(torch, "compile") or not hasattr(torch, "rms_norm"),
        reason="torch.compile or torch.rms_norm not available",
    )
    def test_compile_rms_norm(self):
        nt = NT(
            [
                torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
                torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
            ]
        )
        compiled = torch.compile(lambda x: torch.rms_norm(x, (2,)), backend="inductor", fullgraph=True)
        result = compiled(nt)
        reference = NT([torch.rms_norm(t, (2,)) for t in nt], **nt._meta())
        assert isinstance(result, NestedTensor)
        assert result._has_same_layout(reference)
        assert_close(result, reference)


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


class TestCumulativeOps:

    def test_cummax_batch_dim_not_supported(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, 0.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0, 1.0], device=device, dtype=float_dtype),
            ]
        )
        with pytest.raises(ValueError):
            torch.cummax(nt, dim=0)

    @pytest.mark.parametrize("op", [torch.cummax, torch.cummin])
    def test_cummax_cummin_returns_values_and_indices(self, op, device, float_dtype):
        nt = NT(
            [
                torch.tensor([1.0, 0.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0, 1.0], device=device, dtype=float_dtype),
            ]
        )
        output = op(nt, dim=1)
        reference = tuple(NT([op(t, dim=0)[idx] for t in nt], **nt._meta()) for idx in range(2))
        assert isinstance(output, tuple)
        assert_close(output[0], reference[0])
        assert_close(output[1], reference[1])

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

    def test_logcumsumexp(self, device, float_dtype):
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


class TestDimensionTransforms:

    def test_moveaxis(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
            ]
        )
        output = torch.moveaxis(nt, 1, 2)
        reference = torch.moveaxis(nt.tensor, 1, 2)
        assert_close(output, reference)

    def test_squeeze_default(self, device, float_dtype):
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

    def test_transpose_swaps_last_two_element_dims(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(4, 5, 7, device=device, dtype=float_dtype),
                torch.randn(4, 4, 6, device=device, dtype=float_dtype),
            ]
        )

        output = nt.transpose(-1, -2)
        alias = nt.mT
        reference = NT([tensor.transpose(-1, -2) for tensor in nt], **nt._meta())

        assert_close(output, reference)
        assert_close(alias, reference)
        assert output.shape == torch.Size([2, 4, 7, 5])

    def test_view_like_ops_preserve_autograd(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 1, 3, device=device, dtype=float_dtype, requires_grad=True),
                torch.randn(3, 1, 3, device=device, dtype=float_dtype, requires_grad=True),
            ]
        )

        outputs = [
            nt.squeeze(2),
            torch.squeeze(nt, dim=2),
            nt.unsqueeze(2),
            torch.unsqueeze(nt, dim=2),
            torch.transpose(nt, 1, 3),
            torch.permute(nt, (0, 3, 2, 1)),
            torch.reshape(nt, (-1, 3)),
            nt.flatten(1, 3),
            nt.view(-1, 3),
            nt.view(-1, 3).unflatten(2, (1, 3)),
        ]

        for output in outputs:
            assert isinstance(output, NestedTensor)
            assert output.requires_grad
            assert output._values.requires_grad
            assert output._values.grad_fn is not None
            grad = torch.autograd.grad(output._values.sum(), nt._values, retain_graph=True)[0]
            assert grad is not None
            assert grad.shape == nt._values.shape

    def test_axis_alias_methods_preserve_autograd(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 1, 3, device=device, dtype=float_dtype, requires_grad=True),
                torch.randn(3, 1, 3, device=device, dtype=float_dtype, requires_grad=True),
            ]
        )
        outputs = [
            nt.moveaxis(1, 3),
            nt.movedim(1, 3),
            nt.swapaxes(1, 3),
            nt.swapdims(1, 3),
            torch.moveaxis(nt, 1, 3),
            torch.movedim(nt, 1, 3),
            torch.swapaxes(nt, 1, 3),
            torch.swapdims(nt, 1, 3),
        ]

        for output in outputs:
            assert isinstance(output, NestedTensor)
            assert output.requires_grad
            assert output._values.requires_grad
            assert output._values.grad_fn is not None
            grad = torch.autograd.grad(output._values.sum(), nt._values, retain_graph=True)[0]
            assert grad is not None
            assert grad.shape == nt._values.shape

    def test_swapaxes_batch_dim_raises(self):
        nt = NestedTensor([torch.ones(2, 2)])
        with pytest.raises(ValueError):
            torch.swapaxes(nt, 0, 1)

    def test_swapaxes(self, device, float_dtype):
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

    def test_pairwise_unsqueeze_multiply_builds_square_map(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(26, 8, device=device, dtype=float_dtype),
                torch.randn(14, 8, device=device, dtype=float_dtype),
                torch.randn(9, 8, device=device, dtype=float_dtype),
            ]
        )
        output = nt.unsqueeze(1) * nt.unsqueeze(2)
        reference = NT(
            [torch.unsqueeze(t, 0) * torch.unsqueeze(t, 1) for t in nt],
            batch_first=nt.batch_first,
            padding_value=nt.padding_value,
            mask_value=nt.mask_value,
        )
        assert_close(output, reference)
        assert output.shape == torch.Size([3, 26, 26, 8])
        assert tuple(output._values.shape) == (26 * 26 + 14 * 14 + 9 * 9, 8)
        assert torch.equal(output._physical_shape[:, :2], torch.tensor([[26, 26], [14, 14], [9, 9]]))

    def test_pairwise_unsqueeze_multiply_from_tensor_mask_preserves_square_metadata(self, device, float_dtype):
        dense = torch.randn(1, 5, 3, device=device, dtype=float_dtype)
        mask = torch.ones(1, 5, device=device, dtype=torch.bool)
        nt = NestedTensor.from_tensor_mask(dense, mask)

        output = nt.unsqueeze(1) * nt.unsqueeze(2)
        reference = torch.unsqueeze(dense, 1) * torch.unsqueeze(dense, 2)

        assert output.shape == torch.Size([1, 5, 5, 3])
        expected_shape = torch.tensor([[5, 5, 3]], dtype=output._physical_shape.dtype)
        assert torch.equal(output._physical_shape, expected_shape)
        assert_close(output, reference)

        channels_first = torch.transpose(output, 1, 3)
        normalized = torch.nn.functional.group_norm(channels_first, 1)
        assert normalized.shape == torch.Size([1, 3, 5, 5])
        assert_close(normalized, torch.nn.functional.group_norm(torch.transpose(reference, 1, 3), 1))


class TestDist:

    def test_dist(self, device, float_dtype):
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


class TestEinsum:

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

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_einsum_matmul_equation_compile_fullgraph(self, device, float_dtype):
        a = torch.randn(2, 3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 3, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, 2, device=device, dtype=float_dtype)
        nt = NT([a, b])
        compiled = torch.compile(lambda x, y: torch.einsum("...ij,jk->...ik", x, y), backend="inductor", fullgraph=True)
        result = compiled(nt, w)
        reference = NT(
            [
                torch.einsum("...ij,jk->...ik", a, w),
                torch.einsum("...ij,jk->...ik", b, w),
            ],
            **nt._meta(),
        )
        assert_close(result, reference)

    def test_einsum_matmul_equation_with_ellipsis(self, device, float_dtype):
        a = torch.randn(2, 3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 3, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, 2, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.einsum("...ij,jk->...ik", nt, w)
        reference = NT(
            [
                torch.einsum("...ij,jk->...ik", a, w),
                torch.einsum("...ij,jk->...ik", b, w),
            ],
            **nt._meta(),
        )
        assert_close(result, reference)

    def test_einsum_matrix_vector_equation(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.einsum("ij,j->i", nt, w)
        reference = NT([torch.einsum("ij,j->i", a, w), torch.einsum("ij,j->i", b, w)], **nt._meta())
        assert_close(result, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_einsum_matrix_vector_equation_compile_fullgraph(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        compiled = torch.compile(lambda x, y: torch.einsum("ij,j->i", x, y), backend="inductor", fullgraph=True)
        result = compiled(nt, w)
        reference = NT([torch.einsum("ij,j->i", a, w), torch.einsum("ij,j->i", b, w)], **nt._meta())
        assert_close(result, reference)

    def test_einsum_mismatched_batch_lengths_raises(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(3, 4, device=device, dtype=float_dtype)
        nt1 = NT([a, a])
        nt2 = NT([b])
        with pytest.raises(ValueError, match="einsum: NestedTensor batch length mismatch"):
            torch.einsum("ij,ij->i", nt1, nt2)

    def test_einsum_nt_times_tensor(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, 2, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.einsum("ij,jk->ik", nt, w)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.einsum("ij,jk->ik", a, w))
        assert_close(result[1], torch.einsum("ij,jk->ik", b, w))

    def test_einsum_vector_matrix_equation(self, device, float_dtype):
        v = torch.randn(4, device=device, dtype=float_dtype)
        a = torch.randn(4, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 5, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.einsum("i,ij->j", v, nt)
        reference = NT([torch.einsum("i,ij->j", v, a), torch.einsum("i,ij->j", v, b)], **nt._meta())
        assert_close(result, reference)


class TestFlattenUnflatten:

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

    def test_flip_roll_rot90_preserve_autograd(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 1, 3, device=device, dtype=float_dtype, requires_grad=True),
                torch.randn(3, 1, 3, device=device, dtype=float_dtype, requires_grad=True),
            ]
        )
        outputs = [
            nt.flip((3,)),
            torch.flip(nt, dims=(3,)),
            nt.roll((1,), (3,)),
            torch.roll(nt, shifts=(1,), dims=(3,)),
            nt.rot90(1, (2, 3)),
            torch.rot90(nt, 1, (2, 3)),
        ]

        for output in outputs:
            assert isinstance(output, NestedTensor)
            assert output.requires_grad
            assert output._values.requires_grad
            assert output._values.grad_fn is not None
            grad = torch.autograd.grad(output._values.sum(), nt._values, retain_graph=True)[0]
            assert grad is not None
            assert grad.shape == nt._values.shape


class TestGatherScatter:

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

    def test_gather_on_batch_dim_not_supported(self):
        nt = NestedTensor([torch.tensor([[1, 2]])])
        idx = torch.zeros_like(nt.tensor, dtype=torch.long)
        with pytest.raises(ValueError):
            gather(nt, 0, idx)

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

    def test_masked_select_multi_ragged_exact_layout(self, device, float_dtype):
        nt = NT(
            [
                torch.tensor([[1.0, 0.0, 3.0], [0.0, 5.0, 6.0]], device=device, dtype=float_dtype),
                torch.tensor([[7.0, 0.0, 9.0, 10.0]], device=device, dtype=float_dtype),
            ]
        )
        mask = nt > 0
        output = torch.masked_select(nt, mask)
        reference = NT([torch.masked_select(t, m) for t, m in zip(nt, mask)], **nt._meta())
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


class TestIndexingWriteOps:

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
    def test_index_put_matches_dense(self, device, seed):
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

    def test_masked_scatter_same_shape_dense_source(self, device, float_dtype):
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


class TestInnerProducts:

    def test_dot_both_nested_ragged(self, device, float_dtype):
        lhs = NT(
            [
                torch.randn(4, device=device, dtype=float_dtype),
                torch.randn(3, device=device, dtype=float_dtype),
            ]
        )
        rhs = NT(
            [
                torch.randn(4, device=device, dtype=float_dtype),
                torch.randn(3, device=device, dtype=float_dtype),
            ]
        )
        result = torch.dot(lhs, rhs)
        reference = NT([torch.dot(a, b) for a, b in zip(lhs, rhs)], **lhs._meta())
        assert_close(result, reference)

    def test_dot_both_nested(self, device, float_dtype):
        lhs = NT(
            [
                torch.randn(4, device=device, dtype=float_dtype),
                torch.randn(4, device=device, dtype=float_dtype),
            ]
        )
        rhs = NT(
            [
                torch.randn(4, device=device, dtype=float_dtype),
                torch.randn(4, device=device, dtype=float_dtype),
            ]
        )
        original_iter = NestedTensor.__iter__

        def _fail_iter(self):
            raise AssertionError("torch.dot should not iterate per element for matching packed vectors")

        NestedTensor.__iter__ = _fail_iter
        try:
            result = torch.dot(lhs, rhs)
        finally:
            NestedTensor.__iter__ = original_iter
        reference = NT([torch.dot(a, b) for a, b in zip(lhs, rhs)], **lhs._meta())
        assert_close(result, reference)

    def test_inner_both_nested(self, device, float_dtype):
        a1 = torch.randn(3, 4, device=device, dtype=float_dtype)
        a2 = torch.randn(5, 4, device=device, dtype=float_dtype)
        b1 = torch.randn(3, 4, device=device, dtype=float_dtype)
        b2 = torch.randn(5, 4, device=device, dtype=float_dtype)
        nt1 = NT([a1, a2])
        nt2 = NT([b1, b2])
        result = torch.inner(nt1, nt2)
        reference = NT([torch.inner(a1, b1), torch.inner(a2, b2)], **nt1._meta())
        assert_close(result, reference)

    def test_inner_nested_tensor_with_vector(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.inner(nt, w)
        reference = NT([torch.inner(a, w), torch.inner(b, w)], **nt._meta())
        assert_close(result, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_inner_nested_tensor_with_vector_compile_fullgraph(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        compiled = torch.compile(lambda x, y: torch.inner(x, y), backend="inductor", fullgraph=True)
        result = compiled(nt, w)
        reference = NT([torch.inner(a, w), torch.inner(b, w)], **nt._meta())
        assert_close(result, reference)

    def test_inner_vector_with_nested_tensor(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.inner(w, nt)
        reference = NT([torch.inner(w, a), torch.inner(w, b)], **nt._meta())
        assert_close(result, reference)

    @pytest.mark.skipif(not hasattr(torch, "vdot"), reason="torch.vdot not available")
    def test_vdot_both_nested(self, device, float_dtype):
        lhs = NT(
            [
                torch.randn(4, device=device, dtype=float_dtype),
                torch.randn(4, device=device, dtype=float_dtype),
            ]
        )
        rhs = NT(
            [
                torch.randn(4, device=device, dtype=float_dtype),
                torch.randn(4, device=device, dtype=float_dtype),
            ]
        )
        result = torch.vdot(lhs, rhs)
        reference = NT([torch.vdot(a, b) for a, b in zip(lhs, rhs)], **lhs._meta())
        assert_close(result, reference)

    @pytest.mark.skipif(not hasattr(torch, "vdot"), reason="torch.vdot not available")
    def test_vdot_both_nested_ragged(self, device, float_dtype):
        lhs = NT(
            [
                torch.randn(4, device=device, dtype=float_dtype),
                torch.randn(3, device=device, dtype=float_dtype),
            ]
        )
        rhs = NT(
            [
                torch.randn(4, device=device, dtype=float_dtype),
                torch.randn(3, device=device, dtype=float_dtype),
            ]
        )
        result = torch.vdot(lhs, rhs)
        reference = NT([torch.vdot(a, b) for a, b in zip(lhs, rhs)], **lhs._meta())
        assert_close(result, reference)


class TestLikeCreators:

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

    def test_randint_like(self, device):
        nt = NT([torch.zeros(3, device=device), torch.zeros(1, device=device)])
        torch.manual_seed(1016)
        output = torch.randint_like(nt, 10)
        torch.manual_seed(1016)
        reference = NT([torch.randint_like(t, 10) for t in nt], **nt._meta())
        assert_close(output, reference)

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


class TestLinalgOps:
    """Tests for torch.linalg ops registered in torch_functions.py."""

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
        except RuntimeError as error:
            with pytest.raises(type(error)):
                torch.linalg.eigh(nt)
            return
        eigenvalues, eigenvectors = torch.linalg.eigh(nt)
        assert isinstance(eigenvalues, NestedTensor)
        assert isinstance(eigenvectors, NestedTensor)
        assert_close(eigenvalues[0], w_a)
        assert_close(eigenvalues[1], w_b)

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

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_linalg_norm_matrix_dims_compile_fullgraph(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
                torch.randn(5, 3, 4, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(
            lambda x: torch.linalg.norm(x, ord="fro", dim=(2, 3)),
            backend="inductor",
            fullgraph=True,
        )
        result = compiled(nt)
        reference = NT([torch.linalg.norm(t, ord="fro", dim=(1, 2)) for t in nt], **nt._meta())
        assert_close(result, reference)

    def test_linalg_norm_negative_ord_ragged_dim(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])

        result = torch.linalg.norm(nt, ord=-1, dim=1)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.linalg.norm(a, ord=-1, dim=0))
        assert_close(result[1], torch.linalg.norm(b, ord=-1, dim=0))

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

    def test_linalg_qr(self, device, float_dtype):
        a = torch.randn(4, 3, device=device, dtype=float_dtype)
        b = torch.randn(5, 3, device=device, dtype=float_dtype)
        nt = NT([a, b])
        try:
            torch.linalg.qr(a)
            torch.linalg.qr(b)
        except RuntimeError as error:
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

    def test_linalg_svd(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 3, device=device, dtype=float_dtype)
        nt = NT([a, b])
        try:
            _, S_a, _ = torch.linalg.svd(a)
            _, S_b, _ = torch.linalg.svd(b)
        except RuntimeError as error:
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

    def test_matmul_and_linalg(self):
        a0 = torch.randn(2, 3, 3)
        a1 = torch.randn(3, 3, 3)
        b0 = torch.randn(2, 3, 4)
        b1 = torch.randn(3, 3, 4)
        nt_a = NT([a0, a1])
        nt_b = NT([b0, b1])

        matmul_out = torch.matmul(nt_a, nt_b)
        matmul_ref = NT([torch.matmul(x, y) for x, y in zip(nt_a, nt_b)], **nt_a._meta())
        assert_close(matmul_out, matmul_ref)

        sym_a0 = a0 + a0.transpose(-1, -2)
        sym_a1 = a1 + a1.transpose(-1, -2)
        nt_sym = NT([sym_a0, sym_a1])
        rhs0 = torch.randn(2, 3, 2)
        rhs1 = torch.randn(3, 3, 2)
        nt_rhs = NT([rhs0, rhs1])
        solve_out = torch.linalg.solve(nt_sym, nt_rhs)
        solve_ref = NT([torch.linalg.solve(x, y) for x, y in zip(nt_sym, nt_rhs)], **nt_sym._meta())
        assert_close(solve_out, solve_ref, rtol=1e-5, atol=1e-5)

        svd_u, svd_s, svd_vh = torch.linalg.svd(nt_a, full_matrices=True)
        for u_elem, s_elem, vh_elem, a_elem in zip(svd_u, svd_s, svd_vh, nt_a):
            ref_s = torch.linalg.svd(a_elem, full_matrices=True).S
            assert_close(s_elem, ref_s, rtol=1e-5, atol=1e-5)
            recon = u_elem @ torch.diag_embed(s_elem) @ vh_elem
            assert_close(recon, a_elem, rtol=1e-5, atol=1e-5)


class TestLogicOps:

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


class TestMatrixMultiplication:

    @staticmethod
    def _compile_model_matrix_smoke(
        device: torch.device, *, hidden: int, heads: int, head_dim: int, seqs: tuple[int, int]
    ) -> None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        dense_left = torch.randn(seqs[0], hidden, device=device, dtype=dtype)
        right_2d = NT(
            [
                torch.randn(hidden, head_dim, device=device, dtype=dtype),
                torch.randn(hidden, head_dim, device=device, dtype=dtype),
            ]
        )
        bias_2d = torch.randn(seqs[0], head_dim, device=device, dtype=dtype)

        attention_left = torch.randn(heads, 1, head_dim, device=device, dtype=dtype)
        attention_right = NT(
            [
                torch.randn(heads, head_dim, seqs[0], device=device, dtype=dtype),
                torch.randn(heads, head_dim, seqs[1], device=device, dtype=dtype),
            ]
        )
        bias_3d = torch.randn(heads, 1, 1, device=device, dtype=dtype)

        compiled_matmul = torch.compile(lambda x, y: torch.matmul(x, y), backend="inductor", fullgraph=True)
        compiled_mm = torch.compile(lambda x, y: torch.mm(x, y), backend="inductor", fullgraph=True)
        compiled_addmm = torch.compile(lambda a, x, y: torch.addmm(a, x, y), backend="inductor", fullgraph=True)
        compiled_bmm = torch.compile(lambda x, y: torch.bmm(x, y), backend="inductor", fullgraph=True)
        compiled_baddbmm = torch.compile(lambda a, x, y: torch.baddbmm(a, x, y), backend="inductor", fullgraph=True)

        assert_close(
            compiled_matmul(dense_left, right_2d),
            NT([torch.matmul(dense_left, t) for t in right_2d], **right_2d._meta()),
        )
        assert_close(
            compiled_mm(dense_left, right_2d), NT([torch.mm(dense_left, t) for t in right_2d], **right_2d._meta())
        )
        assert_close(
            compiled_addmm(bias_2d, dense_left, right_2d),
            NT([torch.addmm(bias_2d, dense_left, t) for t in right_2d], **right_2d._meta()),
        )
        assert_close(
            compiled_bmm(attention_left, attention_right),
            NT([torch.bmm(attention_left, t) for t in attention_right], **attention_right._meta()),
        )
        assert_close(
            compiled_baddbmm(bias_3d, attention_left, attention_right),
            NT([torch.baddbmm(bias_3d, attention_left, t) for t in attention_right], **attention_right._meta()),
        )

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_addmm_compile_fullgraph_tensor_lhs(self, device, float_dtype):
        bias = torch.randn(5, 1, device=device, dtype=float_dtype)
        left = torch.randn(5, 4, device=device, dtype=float_dtype)
        right = NT(
            [
                torch.randn(4, 3, device=device, dtype=float_dtype),
                torch.randn(4, 2, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(lambda x, y, z: torch.addmm(x, y, z), backend="inductor", fullgraph=True)
        output = compiled(bias, left, right)
        reference = NT([torch.addmm(bias, left, b) for b in right], **right._meta())
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_addmm_compile_fullgraph_tensor_lhs_uniform_matrices(self, device, float_dtype):
        bias = torch.randn(3, 7, device=device, dtype=float_dtype)
        left = torch.randn(3, 5, device=device, dtype=float_dtype)
        right = NT(
            [
                torch.randn(5, 7, device=device, dtype=float_dtype),
                torch.randn(5, 7, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(lambda x, y, z: torch.addmm(x, y, z), backend="inductor", fullgraph=True)
        output = compiled(bias, left, right)
        reference = NT([torch.addmm(bias, left, b) for b in right], **right._meta())
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_addr_compile_fullgraph_dense_vector_with_nested_vector(self, device, float_dtype):
        bias = torch.randn(4, 1, device=device, dtype=float_dtype)
        left = torch.randn(4, device=device, dtype=float_dtype)
        right = NT(
            [
                torch.randn(3, device=device, dtype=float_dtype),
                torch.randn(2, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(lambda x, y, z: torch.addr(x, y, z), backend="inductor", fullgraph=True)
        output = compiled(bias, left, right)
        reference = NT([torch.addr(bias, left, b) for b in right], **right._meta())
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_addr_compile_fullgraph_nested_vector(self, device, float_dtype):
        bias = torch.randn(1, 4, device=device, dtype=float_dtype)
        left = NT(
            [
                torch.randn(3, device=device, dtype=float_dtype),
                torch.randn(2, device=device, dtype=float_dtype),
            ]
        )
        right = torch.randn(4, device=device, dtype=float_dtype)
        compiled = torch.compile(lambda x, y, z: torch.addr(x, y, z), backend="inductor", fullgraph=True)
        output = compiled(bias, left, right)
        reference = NT([torch.addr(bias, a, right) for a in left], **left._meta())
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_baddbmm_compile_fullgraph_tensor_lhs(self, device, float_dtype):
        bias = torch.randn(2, 3, 1, device=device, dtype=float_dtype)
        left = torch.randn(2, 3, 4, device=device, dtype=float_dtype)
        right = NT(
            [
                torch.randn(2, 4, 5, device=device, dtype=float_dtype),
                torch.randn(2, 4, 3, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(lambda x, y, z: torch.baddbmm(x, y, z), backend="inductor", fullgraph=True)
        output = compiled(bias, left, right)
        reference = NT([torch.baddbmm(bias, left, b) for b in right], **right._meta())
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_bert_base_model_shapes_compile_fullgraph(self, device):
        self._compile_model_matrix_smoke(device, hidden=768, heads=12, head_dim=64, seqs=(384, 512))

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_bmm_compile_fullgraph_tensor_lhs(self, device, float_dtype):
        left = torch.randn(2, 3, 4, device=device, dtype=float_dtype)
        right = NT(
            [
                torch.randn(2, 4, 5, device=device, dtype=float_dtype),
                torch.randn(2, 4, 3, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(lambda x, y: torch.bmm(x, y), backend="inductor", fullgraph=True)
        output = compiled(left, right)
        reference = NT([torch.bmm(left, b) for b in right], **right._meta())
        assert_close(output, reference)

    def test_bmm(self, device, float_dtype):
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

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_llama_13b_model_shapes_compile_fullgraph(self, device):
        if device.type != "cuda":
            pytest.skip("llama-13b full-shape compile smoke is only exercised on CUDA")
        self._compile_model_matrix_smoke(device, hidden=5120, heads=40, head_dim=128, seqs=(1536, 2048))

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_matmul_compile_fullgraph_tensor_lhs(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 4, 5, device=device, dtype=float_dtype),
                torch.randn(1, 4, 5, device=device, dtype=float_dtype),
            ]
        )
        weight = torch.randn(3, 4, device=device, dtype=float_dtype)
        compiled = torch.compile(lambda x, y: torch.matmul(x, y), backend="inductor", fullgraph=True)
        output = compiled(weight, nt)
        reference = NT([torch.matmul(weight, t) for t in nt], **nt._meta())
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_matmul_compile_fullgraph_tensor_lhs_batched_uniform_matrices(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(5, 7, device=device, dtype=float_dtype),
                torch.randn(5, 7, device=device, dtype=float_dtype),
            ]
        )
        weight = torch.randn(3, 2, 5, device=device, dtype=float_dtype)
        compiled = torch.compile(lambda x, y: torch.matmul(x, y), backend="inductor", fullgraph=True)
        output = compiled(weight, nt)
        reference = NT([torch.matmul(weight, t) for t in nt], **nt._meta())
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_matmul_compile_fullgraph_tensor_lhs_matrix(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(4, 5, device=device, dtype=float_dtype),
                torch.randn(4, 3, device=device, dtype=float_dtype),
            ]
        )
        weight = torch.randn(2, 4, device=device, dtype=float_dtype)
        compiled = torch.compile(lambda x, y: torch.matmul(x, y), backend="inductor", fullgraph=True)
        output = compiled(weight, nt)
        reference = NT([torch.matmul(weight, t) for t in nt], **nt._meta())
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_matmul_compile_fullgraph_tensor_rhs(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, 4, device=device, dtype=float_dtype),
                torch.randn(1, 3, 4, device=device, dtype=float_dtype),
            ]
        )
        weight = torch.randn(4, 5, device=device, dtype=float_dtype)
        compiled = torch.compile(lambda x, y: torch.matmul(x, y), backend="inductor", fullgraph=True)
        output = compiled(nt, weight)
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

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_mm_compile_fullgraph_tensor_lhs(self, device, float_dtype):
        left = torch.randn(5, 4, device=device, dtype=float_dtype)
        right = NT(
            [
                torch.randn(4, 3, device=device, dtype=float_dtype),
                torch.randn(4, 2, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(lambda x, y: torch.mm(x, y), backend="inductor", fullgraph=True)
        output = compiled(left, right)
        reference = NT([torch.mm(left, b) for b in right], **right._meta())
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_mm_compile_fullgraph_tensor_lhs_uniform_matrices(self, device, float_dtype):
        left = torch.randn(3, 5, device=device, dtype=float_dtype)
        right = NT(
            [
                torch.randn(5, 7, device=device, dtype=float_dtype),
                torch.randn(5, 7, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(lambda x, y: torch.mm(x, y), backend="inductor", fullgraph=True)
        output = compiled(left, right)
        reference = NT([torch.mm(left, b) for b in right], **right._meta())
        assert_close(output, reference)

    def test_mm(self, device, float_dtype):
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

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_outer_compile_fullgraph_dense_vector_with_nested_vector(self, device, float_dtype):
        left = torch.randn(4, device=device, dtype=float_dtype)
        right = NT(
            [
                torch.randn(3, device=device, dtype=float_dtype),
                torch.randn(2, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(lambda x, y: torch.outer(x, y), backend="inductor", fullgraph=True)
        output = compiled(left, right)
        reference = NT([torch.outer(left, b) for b in right], **right._meta())
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_outer_compile_fullgraph_nested_vector(self, device, float_dtype):
        left = NT(
            [
                torch.randn(3, device=device, dtype=float_dtype),
                torch.randn(2, device=device, dtype=float_dtype),
            ]
        )
        right = torch.randn(4, device=device, dtype=float_dtype)
        compiled = torch.compile(lambda x, y: torch.outer(x, y), backend="inductor", fullgraph=True)
        output = compiled(left, right)
        reference = NT([torch.outer(a, right) for a in left], **left._meta())
        assert_close(output, reference)

    def test_mv_dense_matrix_with_nested_vector(self, device, float_dtype):
        lhs = torch.randn(6, 4, device=device, dtype=float_dtype)
        rhs = NT(
            [
                torch.randn(4, device=device, dtype=float_dtype),
                torch.randn(4, device=device, dtype=float_dtype),
            ]
        )
        result = torch.mv(lhs, rhs)
        reference = NT([torch.mv(lhs, vec) for vec in rhs], **rhs._meta())
        assert_close(result, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_mv_nested_tensor_with_vector_compile_fullgraph(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        compiled = torch.compile(lambda x, y: torch.mv(x, y), backend="inductor", fullgraph=True)
        result = compiled(nt, w)
        reference = NT([torch.mv(a, w), torch.mv(b, w)], **nt._meta())
        assert_close(result, reference)

    def test_mv_nested_tensor_with_vector(self, device, float_dtype):
        a = torch.randn(3, 4, device=device, dtype=float_dtype)
        b = torch.randn(5, 4, device=device, dtype=float_dtype)
        w = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        original_iter = NestedTensor.__iter__

        def _fail_iter(self):
            raise AssertionError("torch.mv should not iterate per element for NestedTensor x dense vector")

        NestedTensor.__iter__ = _fail_iter
        try:
            result = torch.mv(nt, w)
        finally:
            NestedTensor.__iter__ = original_iter
        reference = NT([torch.mv(a, w), torch.mv(b, w)], **nt._meta())
        assert_close(result, reference)


class TestMatrixOps:
    """Tests for matrix/linalg ops registered in torch_functions.py."""

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

    def test_diag_1d_to_2d(self, device, float_dtype):
        a = torch.randn(3, device=device, dtype=float_dtype)
        b = torch.randn(4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.diag(nt)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.diag(a))
        assert_close(result[1], torch.diag(b))

    def test_diag_2d_to_1d(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
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

    def test_matrix_power(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.matrix_power(nt, 3)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.matrix_power(a, 3))
        assert_close(result[1], torch.matrix_power(b, 3))

    def test_matrix_power_batched_ragged(self, device, float_dtype):
        a = torch.randn(2, 3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 3, 3, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.matrix_power(nt, 3)
        reference = NT([torch.matrix_power(a, 3), torch.matrix_power(b, 3)], **nt._meta())
        assert isinstance(result, NestedTensor)
        assert_close(result, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_matrix_power_compile_fullgraph_batched_ragged(self, device, float_dtype):
        a = torch.randn(2, 3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 3, 3, device=device, dtype=float_dtype)
        nt = NT([a, b])
        compiled = torch.compile(lambda x: torch.matrix_power(x, 3), backend="inductor", fullgraph=True)
        result = compiled(nt)
        reference = NT([torch.matrix_power(a, 3), torch.matrix_power(b, 3)], **nt._meta())
        assert isinstance(result, NestedTensor)
        assert_close(result, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_repeat_interleave_compile_fullgraph(self, device, float_dtype):
        nt = NT(
            [
                torch.randn(2, 3, device=device, dtype=float_dtype),
                torch.randn(4, 3, device=device, dtype=float_dtype),
            ]
        )
        compiled = torch.compile(lambda x: torch.repeat_interleave(x, 2, dim=2), backend="inductor", fullgraph=True)
        result = compiled(nt)
        reference = NT([torch.repeat_interleave(t, 2, dim=1) for t in nt], **nt._meta())
        assert_close(result, reference)

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

    def test_trace(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.trace(nt)
        assert isinstance(result, NestedTensor)
        # Scalar results get packed as 1-D elements in NestedTensor
        assert_close(result[0].squeeze(), torch.trace(a))
        assert_close(result[1].squeeze(), torch.trace(b))

    def test_tril(self, device, float_dtype):
        a = torch.randn(3, 3, device=device, dtype=float_dtype)
        b = torch.randn(4, 4, device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.tril(nt)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.tril(a))
        assert_close(result[1], torch.tril(b))

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


class TestNonzeroAndTake:

    def test_nonzero_as_tuple_requires_matching_ndim(self, device):
        with pytest.raises(ValueError, match="same number of dimensions"):
            NT([torch.tensor([1, 0], device=device), torch.tensor([[1], [0]], device=device)])

    def test_nonzero_flattened(self, device):
        nt = NT(
            [
                torch.tensor([[1.0], [0.0]], device=device),
                torch.tensor([[0.0, 2.0]], device=device),
            ]
        )

        output = torch.nonzero(nt, as_tuple=False)
        reference = NT([torch.nonzero(t, as_tuple=False) for t in nt], **nt._meta())
        assert_close(output, reference)

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

    def test_nonzero_multi_ragged(self, device):
        nt = NT(
            [
                torch.tensor([[1, 0, 0], [0, 1, 1]], device=device),
                torch.tensor([[0, 2, 0, 3]], device=device),
            ]
        )
        output = torch.nonzero(nt, as_tuple=False)
        reference = NT([torch.nonzero(t, as_tuple=False) for t in nt], **nt._meta())
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

    def test_take_tensor_index(self, device, float_dtype):
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

    def test_count_nonzero(self, device):
        nt = NT([torch.tensor([1, 0, 2], device=device), torch.tensor([0], device=device)])
        output = torch.count_nonzero(nt)
        assert_close(output, torch.tensor(2, device=device))

        output = torch.count_nonzero(nt, dim=1)
        assert_close(output, torch.tensor([2, 0], device=device))

        with pytest.raises(ValueError):
            torch.count_nonzero(nt, dim=0)


class TestOrderStatistics:

    def test_kthvalue(self, device, float_dtype):
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


class TestRandomOps:

    def test_dropout_and_bernoulli(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.ones(4, device=device, dtype=float_dtype),
                torch.ones(2, device=device, dtype=float_dtype),
            ]
        )
        torch.manual_seed(1016)
        output = torch.dropout(nt, p=0.5, train=True)
        torch.manual_seed(1016)
        reference = packed_result(nt, torch.dropout(nt._values, p=0.5, train=True))
        assert_close(output, reference)

        torch.manual_seed(1016)
        output = torch.bernoulli(nt)
        torch.manual_seed(1016)
        reference = packed_result(nt, torch.bernoulli(nt._values))
        assert_close(output, reference)

    def test_rand_like_and_randn_like_respect_kwargs(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.zeros(2, 3, device=device, dtype=float_dtype),
                torch.zeros(1, 3, device=device, dtype=float_dtype),
            ]
        )
        torch.manual_seed(1016)
        output = torch.rand_like(nt, dtype=torch.float64)
        assert output.dtype == torch.float64

        torch.manual_seed(1016)
        reference = NT([torch.rand_like(t, dtype=torch.float64) for t in nt], **nt._meta())
        assert_close(output, reference)

        torch.manual_seed(1016)
        output = torch.randn_like(nt)
        torch.manual_seed(1016)
        reference = NT([torch.randn_like(t) for t in nt], **nt._meta())
        assert_close(output, reference)


class TestReductionOps:

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

    def test_all_ignores_padding(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.ones(3, device=device, dtype=float_dtype),
                torch.ones(1, device=device, dtype=float_dtype),
            ]
        )
        output = torch.all(nt)
        assert_close(output, torch.tensor(True, device=device))

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

    def test_any_ignores_padding(self, device, float_dtype):
        nt = NestedTensor(
            [torch.zeros(3, device=device, dtype=float_dtype), torch.zeros(1, device=device, dtype=float_dtype)],
            padding_value=1.0,
        )
        output = torch.any(nt)
        assert_close(output, torch.tensor(False, device=device))

    def test_logsumexp(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([3.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.logsumexp(nt, dim=1)
        reference = torch.stack([torch.logsumexp(t, dim=0) for t in nt])
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

    def test_prod_ignores_padding(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([2.0, 2.0], device=device, dtype=float_dtype),
                torch.tensor([2.0], device=device, dtype=float_dtype),
            ]
        )
        output = torch.prod(nt)
        assert_close(output, torch.tensor(8.0, device=device, dtype=float_dtype))

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

    @pytest.mark.parametrize(
        ("k", "expected_shapes"),
        [
            pytest.param(1, ((2, 5, 3), (4, 5, 3)), id="k1_swaps_plane"),
            pytest.param(2, ((2, 3, 5), (4, 3, 5)), id="k2_preserves_plane"),
            pytest.param(3, ((2, 5, 3), (4, 5, 3)), id="k3_swaps_plane"),
            pytest.param(4, ((2, 3, 5), (4, 3, 5)), id="k4_preserves_plane"),
        ],
    )
    def test_rot90_static_plane_metadata_respects_k(self, device, float_dtype, k, expected_shapes):
        nt = NT(
            [
                torch.randn(2, 3, 5, device=device, dtype=float_dtype),
                torch.randn(4, 3, 5, device=device, dtype=float_dtype),
            ]
        )

        output = torch.rot90(nt, k=k, dims=(2, 3))
        reference = NT([torch.rot90(t, k=k, dims=(1, 2)) for t in nt], **nt._meta())
        assert_close(output, reference)
        assert output._element_shapes == expected_shapes


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


class TestSearchsorted:

    def test_bucketize(self, device, float_dtype):
        boundaries = torch.tensor([1.0, 3.0, 5.0], device=device, dtype=float_dtype)
        a = torch.tensor([0.5, 2.0, 4.0, 6.0], device=device, dtype=float_dtype)
        b = torch.tensor([1.5, 3.5], device=device, dtype=float_dtype)
        nt = NT([a, b])
        result = torch.bucketize(nt, boundaries)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.bucketize(a, boundaries))
        assert_close(result[1], torch.bucketize(b, boundaries))

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

    def test_searchsorted_shared_boundaries(self, device, float_dtype):
        boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0], device=device, dtype=float_dtype)
        vals_a = torch.tensor([2.0, 4.0, 6.0], device=device, dtype=float_dtype)
        vals_b = torch.tensor([0.5, 8.0], device=device, dtype=float_dtype)
        nt_vals = NT([vals_a, vals_b])
        result = torch.searchsorted(boundaries, nt_vals)
        assert isinstance(result, NestedTensor)
        assert_close(result[0], torch.searchsorted(boundaries, vals_a))
        assert_close(result[1], torch.searchsorted(boundaries, vals_b))


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


class TestSplitChunkUnbind:

    def test_chunk_batch_dim(self, device, float_dtype):
        nt = NestedTensor([torch.tensor([i], device=device, dtype=float_dtype) for i in range(5)])
        output = torch.chunk(nt, 2, dim=0)
        assert [len(x) for x in output] == [3, 2]

    def test_chunk_excess_chunks(self, device, float_dtype):
        nt = NestedTensor([torch.tensor([i], device=device, dtype=float_dtype) for i in range(5)])
        output = torch.chunk(nt, 10, dim=0)
        assert [len(x) for x in output] == [1, 1, 1, 1, 1]
        recombined = NestedTensor([t for chunk in output for t in chunk], **nt._meta())
        assert_close(recombined, nt)

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


class TestStackFunction:

    def test_stack_dim_zero(self):
        a = NT([torch.tensor([1, 2]), torch.tensor([3, 4])])
        b = NT([torch.tensor([5, 6]), torch.tensor([7, 8])])
        output = torch.stack([a, b], dim=0)
        reference = NT([torch.stack([a[i], b[i]], dim=0) for i in range(len(a))], **a._meta())
        assert isinstance(output, NestedTensor)
        assert_close(output, reference)

    def test_stack_empty_sequence(self):
        with pytest.raises(ValueError):
            stack([])

    def test_stack_requires_dim_zero(self):
        nt = NestedTensor([torch.ones(1)])
        with pytest.raises(NotImplementedError):
            stack([nt], dim=1)

    def test_stack_requires_nested_tensor_inputs(self):
        with pytest.raises(NotImplementedError):
            stack([torch.ones(1)])


class TestUnaryBinaryMath:

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

    @pytest.mark.parametrize("seed", [2, 17, 43])
    def test_addcmul_and_addcdiv_matches_dense(self, device, seed):
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

    def test_clamp(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([[-1.0, 0.5], [2.0, 5.0]], device=device, dtype=float_dtype),
                torch.tensor([[10.0, -5.0], [1.0, 3.0]], device=device, dtype=float_dtype),
            ]
        )
        output = torch.clamp(nt, min=0.0, max=3.0)
        reference = torch.clamp(nt.tensor, min=0.0, max=3.0)
        assert_close(output, reference)

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

    def test_where_dense_condition(self, device, float_dtype):
        nt = NestedTensor(
            [
                torch.tensor([1, 2, 3], device=device, dtype=float_dtype),
                torch.tensor([4, 5], device=device, dtype=float_dtype),
            ]
        )
        output = torch.where(nt.tensor > 2, nt, 0.0)
        reference = NT([torch.where(t > 2, t, 0.0) for t in nt], **nt._meta())
        assert_close(output, reference)

    def test_where_length_mismatch_raises(self):
        cond = NestedTensor([torch.tensor([True, False])])
        input_nt = NestedTensor([torch.tensor([1, 2]), torch.tensor([3])])
        with pytest.raises(ValueError, match="batch length mismatch"):
            _ = torch.where(cond, input_nt, 0)

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_where_matches_dense(self, device, seed):
        dtype = torch.float32
        shapes = ragged_shapes(seed, batch_size=3, min_len=2, max_len=5, trailing_shape=(4,))
        nt = nested_rand(shapes, device, dtype)
        torch.manual_seed(seed)
        condition = torch.randn(4, device=device) > 0
        other = torch.randn(4, device=device, dtype=dtype)
        output = torch.where(condition, nt, other)
        reference = NT([torch.where(condition, t, other) for t in nt], **nt._meta())
        assert_close(output, reference)

    def test_where_scalar_operands(self, device, float_dtype):
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
