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
from torch.nn import functional as F

from danling.tensors import NestedTensor, aten_functions, nn_functions, torch_functions
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


def test_aten_extended_handlers_registered():
    """Ensure compile-oriented aten handlers are registered."""
    aten = torch.ops.aten
    for op in [
        aten.topk.default,
        aten.cumsum.default,
        aten.cumprod.default,
        aten.logcumsumexp.default,
        aten.cummax.default,
        aten.cummin.default,
        aten.flip.default,
        aten.roll.default,
        aten.rot90.default,
        aten.flatten.using_ints,
        aten.view.default,
        aten.reshape.default,
        aten.permute.default,
        aten.transpose.int,
        aten.unsqueeze.default,
        aten.squeeze.default,
        aten.squeeze.dim,
        aten.unflatten.int,
        aten.sum.dim_IntList,
        aten.mean.dim,
        aten.amax.default,
        aten.amin.default,
        aten.logsumexp.default,
        aten.nansum.default,
        aten.nanmean.default,
        aten.std.correction,
        aten.var.correction,
        aten.var_mean.correction,
        aten.count_nonzero.dim_IntList,
        aten.triu.default,
        aten.tril.default,
        aten.matrix_exp.default,
        aten.det.default,
        aten.inverse.default,
        aten.matrix_power.default,
        aten.linalg_inv.default,
        aten.linalg_det.default,
        aten.linalg_cholesky.default,
        aten.searchsorted.Tensor,
    ]:
        assert op in NestedTensorAtenRegistry, f"Aten op {op} missing from NestedTensorAtenRegistry"


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


def test_aten_sort_argsort_topk_cumulative_flip_match_per_tensor():
    """Verify aten-level handlers match per-element tensor semantics."""
    nt = NT(
        [
            torch.tensor([[3.0, 1.0, 2.0], [4.0, 0.0, 5.0]]),
            torch.tensor([[2.0, 5.0, 1.0], [7.0, 6.0, 8.0], [9.0, 4.0, 3.0]]),
        ]
    )

    # Fast path: static per-element dim (NestedTensor dim=2 -> tensor dim=1).
    sorted_vals, sorted_idxs = torch.ops.aten.sort.default(nt, 2, False)
    argsorted = torch.ops.aten.argsort.default(nt, 2, False)
    topk_vals, topk_idxs = torch.ops.aten.topk.default(nt, 2, 2, True, True)
    cumsum_fast = torch.ops.aten.cumsum.default(nt, 2)
    cumprod_fast = torch.ops.aten.cumprod.default(nt, 2)
    logcumsumexp_fast = torch.ops.aten.logcumsumexp.default(nt, 2)
    cummax_vals, cummax_idxs = torch.ops.aten.cummax.default(nt, 2)
    cummin_vals, cummin_idxs = torch.ops.aten.cummin.default(nt, 2)
    flipped_fast = torch.ops.aten.flip.default(nt, [2])

    for (
        got_vals,
        got_idxs,
        got_argsort,
        got_topv,
        got_topi,
        got_csum,
        got_cprod,
        got_lcse,
        got_cmaxv,
        got_cmaxi,
        got_cminv,
        got_cmini,
        got_flip,
        t,
    ) in zip(
        sorted_vals,
        sorted_idxs,
        argsorted,
        topk_vals,
        topk_idxs,
        cumsum_fast,
        cumprod_fast,
        logcumsumexp_fast,
        cummax_vals,
        cummax_idxs,
        cummin_vals,
        cummin_idxs,
        flipped_fast,
        nt,
    ):
        ref_sort = torch.sort(t, dim=1, descending=False)
        ref_topk = torch.topk(t, 2, dim=1, largest=True, sorted=True)
        ref_cummax = torch.cummax(t, dim=1)
        ref_cummin = torch.cummin(t, dim=1)
        torch.testing.assert_close(got_vals, ref_sort.values)
        torch.testing.assert_close(got_idxs, ref_sort.indices)
        torch.testing.assert_close(got_argsort, torch.argsort(t, dim=1, descending=False))
        torch.testing.assert_close(got_topv, ref_topk.values)
        torch.testing.assert_close(got_topi, ref_topk.indices)
        torch.testing.assert_close(got_csum, torch.cumsum(t, dim=1))
        torch.testing.assert_close(got_cprod, torch.cumprod(t, dim=1))
        torch.testing.assert_close(got_lcse, torch.logcumsumexp(t, dim=1))
        torch.testing.assert_close(got_cmaxv, ref_cummax.values)
        torch.testing.assert_close(got_cmaxi, ref_cummax.indices)
        torch.testing.assert_close(got_cminv, ref_cummin.values)
        torch.testing.assert_close(got_cmini, ref_cummin.indices)
        torch.testing.assert_close(got_flip, torch.flip(t, dims=[1]))

    # Fallback path: ragged per-element dim (NestedTensor dim=1 -> tensor dim=0).
    cumsum_ragged = torch.ops.aten.cumsum.default(nt, 1)
    for got, t in zip(cumsum_ragged, nt):
        torch.testing.assert_close(got, torch.cumsum(t, dim=0))


def test_aten_ragged_topk_cumulative_pair_and_flip_no_fallback():
    nt = NT(
        [
            torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
            torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
        ]
    )

    original_fallback = aten_functions.per_element_fallback

    def _fail_fallback(*_args, **_kwargs):
        raise AssertionError("per_element_fallback must not be used for covered ragged aten fastpaths")

    aten_functions.per_element_fallback = _fail_fallback
    try:
        topk_vals, topk_idxs = torch.ops.aten.topk.default(nt, 2, 1, True, True)
        cumsum_ragged = torch.ops.aten.cumsum.default(nt, 1)
        cumprod_ragged = torch.ops.aten.cumprod.default(nt, 1)
        logcumsumexp_ragged = torch.ops.aten.logcumsumexp.default(nt, 1)
        cummax_vals, cummax_idxs = torch.ops.aten.cummax.default(nt, 1)
        cummin_vals, cummin_idxs = torch.ops.aten.cummin.default(nt, 1)
        flipped_ragged = torch.ops.aten.flip.default(nt, [1])
    finally:
        aten_functions.per_element_fallback = original_fallback

    ref_topk_vals = NT([torch.topk(t, 2, dim=0).values for t in nt], **nt._meta())
    ref_topk_idxs = NT([torch.topk(t, 2, dim=0).indices for t in nt], **nt._meta())
    ref_cumsum = NT([torch.cumsum(t, dim=0) for t in nt], **nt._meta())
    ref_cumprod = NT([torch.cumprod(t, dim=0) for t in nt], **nt._meta())
    ref_logcumsumexp = NT([torch.logcumsumexp(t, dim=0) for t in nt], **nt._meta())
    ref_cummax_vals = NT([torch.cummax(t, dim=0).values for t in nt], **nt._meta())
    ref_cummax_idxs = NT([torch.cummax(t, dim=0).indices for t in nt], **nt._meta())
    ref_cummin_vals = NT([torch.cummin(t, dim=0).values for t in nt], **nt._meta())
    ref_cummin_idxs = NT([torch.cummin(t, dim=0).indices for t in nt], **nt._meta())
    ref_flip = NT([torch.flip(t, dims=[0]) for t in nt], **nt._meta())
    torch.testing.assert_close(topk_vals.tensor, ref_topk_vals.tensor)
    torch.testing.assert_close(topk_idxs.tensor, ref_topk_idxs.tensor)
    torch.testing.assert_close(cumsum_ragged.tensor, ref_cumsum.tensor)
    torch.testing.assert_close(cumprod_ragged.tensor, ref_cumprod.tensor)
    torch.testing.assert_close(logcumsumexp_ragged.tensor, ref_logcumsumexp.tensor)
    torch.testing.assert_close(cummax_vals.tensor, ref_cummax_vals.tensor)
    torch.testing.assert_close(cummax_idxs.tensor, ref_cummax_idxs.tensor)
    torch.testing.assert_close(cummin_vals.tensor, ref_cummin_vals.tensor)
    torch.testing.assert_close(cummin_idxs.tensor, ref_cummin_idxs.tensor)
    torch.testing.assert_close(flipped_ragged.tensor, ref_flip.tensor)


def test_aten_ragged_topk_k_exceeds_min_length_raises():
    nt = NT(
        [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        ]
    )
    with pytest.raises(ValueError, match="k <= min segment length"):
        torch.ops.aten.topk.default(nt, 3, 1, True, True)


def test_aten_1d_ragged_fastpaths_no_fallback():
    nt = NT(
        [
            torch.tensor([3.0, 1.0]),
            torch.tensor([4.0, 2.0, 5.0]),
        ]
    )

    original_fallback = aten_functions.per_element_fallback

    def _fail_fallback(*_args, **_kwargs):
        raise AssertionError("per_element_fallback must not be used for 1D ragged aten fastpaths")

    aten_functions.per_element_fallback = _fail_fallback
    try:
        sort_vals, sort_idxs = torch.ops.aten.sort.default(nt, 1, False)
        argsort_out = torch.ops.aten.argsort.default(nt, 1, False)
        topk_vals, topk_idxs = torch.ops.aten.topk.default(nt, 2, 1, True, True)
        cumsum_out = torch.ops.aten.cumsum.default(nt, 1)
        cumprod_out = torch.ops.aten.cumprod.default(nt, 1)
        logcumsumexp_out = torch.ops.aten.logcumsumexp.default(nt, 1)
        cummax_vals, cummax_idxs = torch.ops.aten.cummax.default(nt, 1)
        cummin_vals, cummin_idxs = torch.ops.aten.cummin.default(nt, 1)
        flip_out = torch.ops.aten.flip.default(nt, [1])
        softmax_out = torch.ops.aten._softmax.default(nt, 1, False)
        log_softmax_out = torch.ops.aten._log_softmax.default(nt, 1, False)
    finally:
        aten_functions.per_element_fallback = original_fallback

    ref_sort_vals = NT([torch.sort(t, dim=0, descending=False).values for t in nt], **nt._meta())
    ref_sort_idxs = NT([torch.sort(t, dim=0, descending=False).indices for t in nt], **nt._meta())
    ref_argsort = NT([torch.argsort(t, dim=0, descending=False) for t in nt], **nt._meta())
    ref_topk_vals = NT([torch.topk(t, 2, dim=0, largest=True, sorted=True).values for t in nt], **nt._meta())
    ref_topk_idxs = NT([torch.topk(t, 2, dim=0, largest=True, sorted=True).indices for t in nt], **nt._meta())
    ref_cumsum = NT([torch.cumsum(t, dim=0) for t in nt], **nt._meta())
    ref_cumprod = NT([torch.cumprod(t, dim=0) for t in nt], **nt._meta())
    ref_logcumsumexp = NT([torch.logcumsumexp(t, dim=0) for t in nt], **nt._meta())
    ref_cummax_vals = NT([torch.cummax(t, dim=0).values for t in nt], **nt._meta())
    ref_cummax_idxs = NT([torch.cummax(t, dim=0).indices for t in nt], **nt._meta())
    ref_cummin_vals = NT([torch.cummin(t, dim=0).values for t in nt], **nt._meta())
    ref_cummin_idxs = NT([torch.cummin(t, dim=0).indices for t in nt], **nt._meta())
    ref_flip = NT([torch.flip(t, dims=[0]) for t in nt], **nt._meta())
    ref_softmax = NT([torch.softmax(t, dim=0) for t in nt], **nt._meta())
    ref_log_softmax = NT([torch.log_softmax(t, dim=0) for t in nt], **nt._meta())
    torch.testing.assert_close(sort_vals.tensor, ref_sort_vals.tensor)
    torch.testing.assert_close(sort_idxs.tensor, ref_sort_idxs.tensor)
    torch.testing.assert_close(argsort_out.tensor, ref_argsort.tensor)
    torch.testing.assert_close(topk_vals.tensor, ref_topk_vals.tensor)
    torch.testing.assert_close(topk_idxs.tensor, ref_topk_idxs.tensor)
    torch.testing.assert_close(cumsum_out.tensor, ref_cumsum.tensor)
    torch.testing.assert_close(cumprod_out.tensor, ref_cumprod.tensor)
    torch.testing.assert_close(logcumsumexp_out.tensor, ref_logcumsumexp.tensor)
    torch.testing.assert_close(cummax_vals.tensor, ref_cummax_vals.tensor)
    torch.testing.assert_close(cummax_idxs.tensor, ref_cummax_idxs.tensor)
    torch.testing.assert_close(cummin_vals.tensor, ref_cummin_vals.tensor)
    torch.testing.assert_close(cummin_idxs.tensor, ref_cummin_idxs.tensor)
    torch.testing.assert_close(flip_out.tensor, ref_flip.tensor)
    torch.testing.assert_close(softmax_out.tensor, ref_softmax.tensor)
    torch.testing.assert_close(log_softmax_out.tensor, ref_log_softmax.tensor)


def test_aten_unregistered_same_shape_fastpath_no_fallback():
    nt = NT(
        [
            torch.tensor([[1.1, 2.2], [3.3, 4.4]]),
            torch.tensor([[5.5, 6.6], [7.7, 8.8], [9.9, 10.1]]),
        ]
    )

    original_fallback = aten_functions.per_element_fallback

    def _fail_fallback(*_args, **_kwargs):
        raise AssertionError("per_element_fallback must not be used for covered generic aten fastpath")

    aten_functions.per_element_fallback = _fail_fallback
    try:
        square_out = torch.ops.aten.square.default(nt)
        digamma_out = torch.ops.aten.digamma.default(nt)
        lgamma_out = torch.ops.aten.lgamma.default(nt)
    finally:
        aten_functions.per_element_fallback = original_fallback

    ref_square = NT([torch.square(t) for t in nt], **nt._meta())
    ref_digamma = NT([torch.digamma(t) for t in nt], **nt._meta())
    ref_lgamma = NT([torch.lgamma(t) for t in nt], **nt._meta())
    torch.testing.assert_close(square_out.tensor, ref_square.tensor)
    torch.testing.assert_close(digamma_out.tensor, ref_digamma.tensor)
    torch.testing.assert_close(lgamma_out.tensor, ref_lgamma.tensor)


def test_aten_shape_and_reduction_fastpaths_no_fallback():
    nt = NT(
        [
            torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4),
            torch.arange(3 * 3 * 4, dtype=torch.float32).reshape(3, 3, 4),
        ]
    )

    original_fallback = aten_functions.per_element_fallback

    def _fail_fallback(*_args, **_kwargs):
        raise AssertionError("per_element_fallback must not be used for covered shape/reduction aten fastpaths")

    aten_functions.per_element_fallback = _fail_fallback
    try:
        flattened = torch.ops.aten.flatten.using_ints(nt, 2, 3)
        permuted = torch.ops.aten.permute.default(nt, [0, 1, 3, 2])
        transposed = torch.ops.aten.transpose.int(nt, 2, 3)
        unsqueezed = torch.ops.aten.unsqueeze.default(nt, 3)
        squeezed = torch.ops.aten.squeeze.dim(unsqueezed, 3)
        unflattened = torch.ops.aten.unflatten.int(flattened, 2, [3, 4])
        sum_static = torch.ops.aten.sum.dim_IntList(nt, [2], False)
        sum_ragged = torch.ops.aten.sum.dim_IntList(nt, [1], False)
        mean_static = torch.ops.aten.mean.dim(nt, [2], False)
        sum_multi_static = torch.ops.aten.sum.dim_IntList(nt, [2, 3], False)
        mean_multi_static = torch.ops.aten.mean.dim(nt, [2, 3], False)
        amax_static = torch.ops.aten.amax.default(nt, [2], False)
        amax_multi_static = torch.ops.aten.amax.default(nt, [2, 3], False)
        amin_multi_static = torch.ops.aten.amin.default(nt, [2, 3], False)
        amin_ragged = torch.ops.aten.amin.default(nt, [1], False)
        rolled = torch.ops.aten.roll.default(nt, [1], [3])
        rotated = torch.ops.aten.rot90.default(nt, 1, [2, 3])
    finally:
        aten_functions.per_element_fallback = original_fallback

    ref_flattened = NT([torch.flatten(t, start_dim=1, end_dim=2) for t in nt], **nt._meta())
    ref_permuted = NT([torch.permute(t, (0, 2, 1)) for t in nt], **nt._meta())
    ref_transposed = NT([torch.transpose(t, 1, 2) for t in nt], **nt._meta())
    ref_unsqueezed = NT([torch.unsqueeze(t, dim=2) for t in nt], **nt._meta())
    ref_unflattened = NT([torch.unflatten(t, 1, [3, 4]) for t in ref_flattened], **nt._meta())
    ref_sum_static = NT([torch.sum(t, dim=1) for t in nt], **nt._meta())
    ref_sum_ragged = torch.stack([torch.sum(t, dim=0) for t in nt])
    ref_mean_static = NT([torch.mean(t, dim=1) for t in nt], **nt._meta())
    ref_sum_multi_static = NT([torch.sum(t, dim=(1, 2)) for t in nt], **nt._meta())
    ref_mean_multi_static = NT([torch.mean(t, dim=(1, 2)) for t in nt], **nt._meta())
    ref_amax_static = NT([torch.amax(t, dim=1) for t in nt], **nt._meta())
    ref_amax_multi_static = NT([torch.amax(t, dim=(1, 2)) for t in nt], **nt._meta())
    ref_amin_multi_static = NT([torch.amin(t, dim=(1, 2)) for t in nt], **nt._meta())
    ref_amin_ragged = torch.stack([torch.amin(t, dim=0) for t in nt])
    ref_rolled = NT([torch.roll(t, [1], [2]) for t in nt], **nt._meta())
    ref_rotated = NT([torch.rot90(t, 1, (1, 2)) for t in nt], **nt._meta())

    torch.testing.assert_close(flattened.tensor, ref_flattened.tensor)
    torch.testing.assert_close(permuted.tensor, ref_permuted.tensor)
    torch.testing.assert_close(transposed.tensor, ref_transposed.tensor)
    torch.testing.assert_close(unsqueezed.tensor, ref_unsqueezed.tensor)
    torch.testing.assert_close(squeezed.tensor, nt.tensor)
    torch.testing.assert_close(unflattened.tensor, ref_unflattened.tensor)
    torch.testing.assert_close(sum_static.tensor, ref_sum_static.tensor)
    torch.testing.assert_close(sum_ragged, ref_sum_ragged)
    torch.testing.assert_close(mean_static.tensor, ref_mean_static.tensor)
    torch.testing.assert_close(sum_multi_static.tensor, ref_sum_multi_static.tensor)
    torch.testing.assert_close(mean_multi_static.tensor, ref_mean_multi_static.tensor)
    torch.testing.assert_close(amax_static.tensor, ref_amax_static.tensor)
    torch.testing.assert_close(amax_multi_static.tensor, ref_amax_multi_static.tensor)
    torch.testing.assert_close(amin_multi_static.tensor, ref_amin_multi_static.tensor)
    torch.testing.assert_close(amin_ragged, ref_amin_ragged)
    torch.testing.assert_close(rolled.tensor, ref_rolled.tensor)
    torch.testing.assert_close(rotated.tensor, ref_rotated.tensor)


def test_aten_searchsorted_fastpaths_no_fallback():
    sorted_nt = NT(
        [
            torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]),
            torch.tensor([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0], [2.0, 6.0, 8.0]]),
        ]
    )
    values_nt = NT(
        [
            torch.tensor([[0.5, 3.0, 6.5], [1.0, 5.0, 6.0]]),
            torch.tensor([[1.0, 2.0, 4.0], [0.0, 3.0, 6.0], [1.5, 6.5, 9.0]]),
        ]
    )
    boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0])

    original_fallback = aten_functions.per_element_fallback

    def _fail_fallback(*_args, **_kwargs):
        raise AssertionError("per_element_fallback must not be used for covered searchsorted fastpaths")

    aten_functions.per_element_fallback = _fail_fallback
    try:
        nt_nt = torch.ops.aten.searchsorted.Tensor(sorted_nt, values_nt)
        tensor_nt = torch.ops.aten.searchsorted.Tensor(boundaries, values_nt)
    finally:
        aten_functions.per_element_fallback = original_fallback

    ref_nt_nt = NT([torch.searchsorted(s, v) for s, v in zip(sorted_nt, values_nt)], **values_nt._meta())
    ref_tensor_nt = NT([torch.searchsorted(boundaries, v) for v in values_nt], **values_nt._meta())

    torch.testing.assert_close(nt_nt.tensor, ref_nt_nt.tensor)
    torch.testing.assert_close(tensor_nt.tensor, ref_tensor_nt.tensor)


def test_aten_searchsorted_rejects_nested_sorter_without_nested_sorted_sequence():
    boundaries = torch.tensor([1.0, 3.0, 5.0])
    values = torch.tensor([2.0, 4.0])
    sorter = NT([torch.tensor([0, 1, 2], dtype=torch.long)])
    with pytest.raises(TypeError, match="NestedTensor sorter requires sorted_sequence"):
        torch.ops.aten.searchsorted.Tensor(boundaries, values, sorter=sorter)


def test_aten_reduce_multi_dim_empty_batch_drops_static_cols():
    values = torch.empty((0, 3, 4), dtype=torch.float32)
    offsets = torch.tensor([0], dtype=torch.long)
    shape_tensor = torch.empty((0, 3), dtype=torch.long)
    nt = NT._from_packed(
        values,
        offsets,
        shape_tensor,
        batch_first=True,
        padding_value=0.0,
        mask_value=False,
        pin_memory=False,
        outer_size=torch.Size([0, 0, 3, 4]),
    )
    out = torch.ops.aten.sum.dim_IntList(nt, [2, 3], False, dtype=None)
    assert isinstance(out, NT)
    assert out.shape == torch.Size([0, 0])
    assert out._shape_tensor.shape == torch.Size([0, 1])


def test_offsets_match_fake_tensors_not_false_positive():
    fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
    FakeTensorMode = fake_tensor_mod.FakeTensorMode

    with FakeTensorMode():
        offsets_a = torch.tensor([0, 2, 5], dtype=torch.long)
        offsets_b = torch.tensor([0, 2, 5], dtype=torch.long)
        assert aten_functions._offsets_match(offsets_a, offsets_a)
        assert not aten_functions._offsets_match(offsets_a, offsets_b)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_aten_1d_ragged_fastpaths_compile_smoke():
    nt = NT(
        [
            torch.tensor([3.0, 1.0]),
            torch.tensor([4.0, 2.0, 5.0]),
        ]
    )

    def _compile(fn):
        return torch.compile(fn, backend="eager", fullgraph=True)

    original_fallback = aten_functions.per_element_fallback

    def _fail_fallback(*_args, **_kwargs):
        raise AssertionError("per_element_fallback must not be used for 1D ragged aten fastpaths")

    aten_functions.per_element_fallback = _fail_fallback
    try:
        sort_fn = _compile(lambda x: torch.ops.aten.sort.default(x, 1, False))
        argsort_fn = _compile(lambda x: torch.ops.aten.argsort.default(x, 1, False))
        topk_fn = _compile(lambda x: torch.ops.aten.topk.default(x, 2, 1, True, True))
        cumsum_fn = _compile(lambda x: torch.ops.aten.cumsum.default(x, 1))
        cumprod_fn = _compile(lambda x: torch.ops.aten.cumprod.default(x, 1))
        logcumsumexp_fn = _compile(lambda x: torch.ops.aten.logcumsumexp.default(x, 1))
        cummax_fn = _compile(lambda x: torch.ops.aten.cummax.default(x, 1))
        cummin_fn = _compile(lambda x: torch.ops.aten.cummin.default(x, 1))
        flip_fn = _compile(lambda x: torch.ops.aten.flip.default(x, [1]))
        softmax_fn = _compile(lambda x: torch.ops.aten._softmax.default(x, 1, False))
        log_softmax_fn = _compile(lambda x: torch.ops.aten._log_softmax.default(x, 1, False))
        sort_vals, sort_idxs = sort_fn(nt)
        argsort_out = argsort_fn(nt)
        topk_vals, topk_idxs = topk_fn(nt)
        cumsum_out = cumsum_fn(nt)
        cumprod_out = cumprod_fn(nt)
        logcumsumexp_out = logcumsumexp_fn(nt)
        cummax_vals, cummax_idxs = cummax_fn(nt)
        cummin_vals, cummin_idxs = cummin_fn(nt)
        flip_out = flip_fn(nt)
        softmax_out = softmax_fn(nt)
        log_softmax_out = log_softmax_fn(nt)
    finally:
        aten_functions.per_element_fallback = original_fallback

    ref_sort_vals = NT([torch.sort(t, dim=0, descending=False).values for t in nt], **nt._meta())
    ref_sort_idxs = NT([torch.sort(t, dim=0, descending=False).indices for t in nt], **nt._meta())
    ref_argsort = NT([torch.argsort(t, dim=0, descending=False) for t in nt], **nt._meta())
    ref_topk_vals = NT([torch.topk(t, 2, dim=0, largest=True, sorted=True).values for t in nt], **nt._meta())
    ref_topk_idxs = NT([torch.topk(t, 2, dim=0, largest=True, sorted=True).indices for t in nt], **nt._meta())
    ref_cumsum = NT([torch.cumsum(t, dim=0) for t in nt], **nt._meta())
    ref_cumprod = NT([torch.cumprod(t, dim=0) for t in nt], **nt._meta())
    ref_logcumsumexp = NT([torch.logcumsumexp(t, dim=0) for t in nt], **nt._meta())
    ref_cummax_vals = NT([torch.cummax(t, dim=0).values for t in nt], **nt._meta())
    ref_cummax_idxs = NT([torch.cummax(t, dim=0).indices for t in nt], **nt._meta())
    ref_cummin_vals = NT([torch.cummin(t, dim=0).values for t in nt], **nt._meta())
    ref_cummin_idxs = NT([torch.cummin(t, dim=0).indices for t in nt], **nt._meta())
    ref_flip = NT([torch.flip(t, dims=[0]) for t in nt], **nt._meta())
    ref_softmax = NT([torch.softmax(t, dim=0) for t in nt], **nt._meta())
    ref_log_softmax = NT([torch.log_softmax(t, dim=0) for t in nt], **nt._meta())
    torch.testing.assert_close(sort_vals.tensor, ref_sort_vals.tensor)
    torch.testing.assert_close(sort_idxs.tensor, ref_sort_idxs.tensor)
    torch.testing.assert_close(argsort_out.tensor, ref_argsort.tensor)
    torch.testing.assert_close(topk_vals.tensor, ref_topk_vals.tensor)
    torch.testing.assert_close(topk_idxs.tensor, ref_topk_idxs.tensor)
    torch.testing.assert_close(cumsum_out.tensor, ref_cumsum.tensor)
    torch.testing.assert_close(cumprod_out.tensor, ref_cumprod.tensor)
    torch.testing.assert_close(logcumsumexp_out.tensor, ref_logcumsumexp.tensor)
    torch.testing.assert_close(cummax_vals.tensor, ref_cummax_vals.tensor)
    torch.testing.assert_close(cummax_idxs.tensor, ref_cummax_idxs.tensor)
    torch.testing.assert_close(cummin_vals.tensor, ref_cummin_vals.tensor)
    torch.testing.assert_close(cummin_idxs.tensor, ref_cummin_idxs.tensor)
    torch.testing.assert_close(flip_out.tensor, ref_flip.tensor)
    torch.testing.assert_close(softmax_out.tensor, ref_softmax.tensor)
    torch.testing.assert_close(log_softmax_out.tensor, ref_log_softmax.tensor)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_aten_ragged_fastpaths_compile_smoke():
    nt = NT(
        [
            torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
            torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
        ]
    )

    def _compile(fn):
        return torch.compile(fn, backend="eager", fullgraph=True)

    original_fallback = aten_functions.per_element_fallback

    def _fail_fallback(*_args, **_kwargs):
        raise AssertionError("per_element_fallback must not be used for covered ragged aten fastpaths")

    aten_functions.per_element_fallback = _fail_fallback
    try:
        topk_fn = _compile(lambda x: torch.ops.aten.topk.default(x, 2, 1, True, True))
        cumsum_fn = _compile(lambda x: torch.ops.aten.cumsum.default(x, 1))
        cumprod_fn = _compile(lambda x: torch.ops.aten.cumprod.default(x, 1))
        logcumsumexp_fn = _compile(lambda x: torch.ops.aten.logcumsumexp.default(x, 1))
        cummax_fn = _compile(lambda x: torch.ops.aten.cummax.default(x, 1))
        cummin_fn = _compile(lambda x: torch.ops.aten.cummin.default(x, 1))
        flip_fn = _compile(lambda x: torch.ops.aten.flip.default(x, [1]))
        sort_fn = _compile(lambda x: torch.ops.aten.sort.default(x, 1, False))
        argsort_fn = _compile(lambda x: torch.ops.aten.argsort.default(x, 1, False))
        softmax_fn = _compile(lambda x: torch.ops.aten._softmax.default(x, 1, False))
        log_softmax_fn = _compile(lambda x: torch.ops.aten._log_softmax.default(x, 1, False))
        topk_comp_vals, topk_comp_idxs = topk_fn(nt)
        cumsum_comp = cumsum_fn(nt)
        cumprod_comp = cumprod_fn(nt)
        logcumsumexp_comp = logcumsumexp_fn(nt)
        cummax_comp_vals, cummax_comp_idxs = cummax_fn(nt)
        cummin_comp_vals, cummin_comp_idxs = cummin_fn(nt)
        flip_comp = flip_fn(nt)
        sort_comp_vals, sort_comp_idxs = sort_fn(nt)
        argsort_comp = argsort_fn(nt)
        softmax_comp = softmax_fn(nt)
        log_softmax_comp = log_softmax_fn(nt)
    finally:
        aten_functions.per_element_fallback = original_fallback

    ref_topk_vals = NT([torch.topk(t, 2, dim=0).values for t in nt], **nt._meta())
    ref_topk_idxs = NT([torch.topk(t, 2, dim=0).indices for t in nt], **nt._meta())
    ref_cumsum = NT([torch.cumsum(t, dim=0) for t in nt], **nt._meta())
    ref_cumprod = NT([torch.cumprod(t, dim=0) for t in nt], **nt._meta())
    ref_logcumsumexp = NT([torch.logcumsumexp(t, dim=0) for t in nt], **nt._meta())
    ref_cummax_vals = NT([torch.cummax(t, dim=0).values for t in nt], **nt._meta())
    ref_cummax_idxs = NT([torch.cummax(t, dim=0).indices for t in nt], **nt._meta())
    ref_cummin_vals = NT([torch.cummin(t, dim=0).values for t in nt], **nt._meta())
    ref_cummin_idxs = NT([torch.cummin(t, dim=0).indices for t in nt], **nt._meta())
    ref_flip = NT([torch.flip(t, dims=[0]) for t in nt], **nt._meta())
    ref_sort_vals = NT([torch.sort(t, dim=0, descending=False).values for t in nt], **nt._meta())
    ref_sort_idxs = NT([torch.sort(t, dim=0, descending=False).indices for t in nt], **nt._meta())
    ref_argsort = NT([torch.argsort(t, dim=0, descending=False) for t in nt], **nt._meta())
    ref_softmax = NT([torch.softmax(t, dim=0) for t in nt], **nt._meta())
    ref_log_softmax = NT([torch.log_softmax(t, dim=0) for t in nt], **nt._meta())
    torch.testing.assert_close(topk_comp_vals.tensor, ref_topk_vals.tensor)
    torch.testing.assert_close(topk_comp_idxs.tensor, ref_topk_idxs.tensor)
    torch.testing.assert_close(cumsum_comp.tensor, ref_cumsum.tensor)
    torch.testing.assert_close(cumprod_comp.tensor, ref_cumprod.tensor)
    torch.testing.assert_close(logcumsumexp_comp.tensor, ref_logcumsumexp.tensor)
    torch.testing.assert_close(cummax_comp_vals.tensor, ref_cummax_vals.tensor)
    torch.testing.assert_close(cummax_comp_idxs.tensor, ref_cummax_idxs.tensor)
    torch.testing.assert_close(cummin_comp_vals.tensor, ref_cummin_vals.tensor)
    torch.testing.assert_close(cummin_comp_idxs.tensor, ref_cummin_idxs.tensor)
    torch.testing.assert_close(flip_comp.tensor, ref_flip.tensor)
    torch.testing.assert_close(sort_comp_vals.tensor, ref_sort_vals.tensor)
    torch.testing.assert_close(sort_comp_idxs.tensor, ref_sort_idxs.tensor)
    torch.testing.assert_close(argsort_comp.tensor, ref_argsort.tensor)
    torch.testing.assert_close(softmax_comp.tensor, ref_softmax.tensor)
    torch.testing.assert_close(log_softmax_comp.tensor, ref_log_softmax.tensor)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_aten_unregistered_same_shape_compile_smoke():
    nt = NT(
        [
            torch.tensor([[1.1, 2.2], [3.3, 4.4]]),
            torch.tensor([[5.5, 6.6], [7.7, 8.8], [9.9, 10.1]]),
        ]
    )

    def _compile(fn):
        return torch.compile(fn, backend="eager", fullgraph=True)

    original_fallback = aten_functions.per_element_fallback

    def _fail_fallback(*_args, **_kwargs):
        raise AssertionError("per_element_fallback must not be used for covered generic aten fastpath")

    aten_functions.per_element_fallback = _fail_fallback
    try:
        square_fn = _compile(lambda x: torch.ops.aten.square.default(x))
        digamma_fn = _compile(lambda x: torch.ops.aten.digamma.default(x))
        lgamma_fn = _compile(lambda x: torch.ops.aten.lgamma.default(x))
        square_comp = square_fn(nt)
        digamma_comp = digamma_fn(nt)
        lgamma_comp = lgamma_fn(nt)
    finally:
        aten_functions.per_element_fallback = original_fallback

    ref_square = NT([torch.square(t) for t in nt], **nt._meta())
    ref_digamma = NT([torch.digamma(t) for t in nt], **nt._meta())
    ref_lgamma = NT([torch.lgamma(t) for t in nt], **nt._meta())
    torch.testing.assert_close(square_comp.tensor, ref_square.tensor)
    torch.testing.assert_close(digamma_comp.tensor, ref_digamma.tensor)
    torch.testing.assert_close(lgamma_comp.tensor, ref_lgamma.tensor)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_torch_ragged_fastpaths_compile_smoke():
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
        return torch.compile(fn, backend="eager", fullgraph=True)

    original_fallback = aten_functions.per_element_fallback

    def _fail_fallback(*_args, **_kwargs):
        raise AssertionError("per_element_fallback must not be used for covered ragged torch fastpaths")

    aten_functions.per_element_fallback = _fail_fallback
    try:
        topk_fn = _compile(lambda x: torch.topk(x, 2, dim=1, largest=True, sorted=True))
        cumsum_fn = _compile(lambda x: torch.cumsum(x, dim=1))
        cumprod_fn = _compile(lambda x: torch.cumprod(x, dim=1))
        logcumsumexp_fn = _compile(lambda x: torch.logcumsumexp(x, dim=1))
        cummax_fn = _compile(lambda x: torch.cummax(x, dim=1))
        cummin_fn = _compile(lambda x: torch.cummin(x, dim=1))
        flip_fn = _compile(lambda x: torch.flip(x, dims=[1]))
        sort_fn = _compile(lambda x: torch.sort(x, dim=1, descending=False))
        argsort_fn = _compile(lambda x: torch.argsort(x, dim=1, descending=False))
        softmax_fn = _compile(lambda x: torch.softmax(x, dim=1))
        log_softmax_fn = _compile(lambda x: torch.log_softmax(x, dim=1))
        dropout_eval_fn = _compile(lambda x: torch.dropout(x, p=0.2, train=False))
        dropout_train_fn = _compile(lambda x: torch.dropout(x, p=0.2, train=True))
        bernoulli_fn = _compile(lambda x: torch.bernoulli(x))
        layer_norm_fn = _compile(lambda x: torch.layer_norm(x, (2,)))
        if hasattr(torch, "rms_norm"):
            rms_norm_fn = _compile(lambda x: torch.rms_norm(x, (2,)))
        topk_vals, topk_idxs = topk_fn(nt)
        cumsum_comp = cumsum_fn(nt)
        cumprod_comp = cumprod_fn(nt)
        logcumsumexp_comp = logcumsumexp_fn(nt)
        cummax_comp_vals, cummax_comp_idxs = cummax_fn(nt)
        cummin_comp_vals, cummin_comp_idxs = cummin_fn(nt)
        flip_comp = flip_fn(nt)
        sort_vals, sort_idxs = sort_fn(nt)
        argsort_comp = argsort_fn(nt)
        softmax_comp = softmax_fn(nt)
        log_softmax_comp = log_softmax_fn(nt)
        dropout_eval_comp = dropout_eval_fn(nt)
        torch.manual_seed(1234)
        dropout_train_comp = dropout_train_fn(nt)
        torch.manual_seed(5678)
        bernoulli_comp = bernoulli_fn(nt_prob)
        layer_norm_comp = layer_norm_fn(nt)
        if hasattr(torch, "rms_norm"):
            rms_norm_comp = rms_norm_fn(nt)
    finally:
        aten_functions.per_element_fallback = original_fallback

    ref_topk_vals = NT([torch.topk(t, 2, dim=0).values for t in nt], **nt._meta())
    ref_topk_idxs = NT([torch.topk(t, 2, dim=0).indices for t in nt], **nt._meta())
    ref_cumsum = NT([torch.cumsum(t, dim=0) for t in nt], **nt._meta())
    ref_cumprod = NT([torch.cumprod(t, dim=0) for t in nt], **nt._meta())
    ref_logcumsumexp = NT([torch.logcumsumexp(t, dim=0) for t in nt], **nt._meta())
    ref_cummax_vals = NT([torch.cummax(t, dim=0).values for t in nt], **nt._meta())
    ref_cummax_idxs = NT([torch.cummax(t, dim=0).indices for t in nt], **nt._meta())
    ref_cummin_vals = NT([torch.cummin(t, dim=0).values for t in nt], **nt._meta())
    ref_cummin_idxs = NT([torch.cummin(t, dim=0).indices for t in nt], **nt._meta())
    ref_flip = NT([torch.flip(t, dims=[0]) for t in nt], **nt._meta())
    ref_sort_vals = NT([torch.sort(t, dim=0, descending=False).values for t in nt], **nt._meta())
    ref_sort_idxs = NT([torch.sort(t, dim=0, descending=False).indices for t in nt], **nt._meta())
    ref_argsort = NT([torch.argsort(t, dim=0, descending=False) for t in nt], **nt._meta())
    ref_softmax = NT([torch.softmax(t, dim=0) for t in nt], **nt._meta())
    ref_log_softmax = NT([torch.log_softmax(t, dim=0) for t in nt], **nt._meta())
    torch.manual_seed(1234)
    ref_dropout_train = NT([torch.dropout(t, p=0.2, train=True) for t in nt], **nt._meta())
    torch.manual_seed(5678)
    ref_bernoulli = NT([torch.bernoulli(t) for t in nt_prob], **nt_prob._meta())
    ref_layer_norm = NT([torch.layer_norm(t, (2,)) for t in nt], **nt._meta())
    if hasattr(torch, "rms_norm"):
        ref_rms_norm = NT([torch.rms_norm(t, (2,)) for t in nt], **nt._meta())
    torch.testing.assert_close(topk_vals.tensor, ref_topk_vals.tensor)
    torch.testing.assert_close(topk_idxs.tensor, ref_topk_idxs.tensor)
    torch.testing.assert_close(cumsum_comp.tensor, ref_cumsum.tensor)
    torch.testing.assert_close(cumprod_comp.tensor, ref_cumprod.tensor)
    torch.testing.assert_close(logcumsumexp_comp.tensor, ref_logcumsumexp.tensor)
    torch.testing.assert_close(cummax_comp_vals.tensor, ref_cummax_vals.tensor)
    torch.testing.assert_close(cummax_comp_idxs.tensor, ref_cummax_idxs.tensor)
    torch.testing.assert_close(cummin_comp_vals.tensor, ref_cummin_vals.tensor)
    torch.testing.assert_close(cummin_comp_idxs.tensor, ref_cummin_idxs.tensor)
    torch.testing.assert_close(flip_comp.tensor, ref_flip.tensor)
    torch.testing.assert_close(sort_vals.tensor, ref_sort_vals.tensor)
    torch.testing.assert_close(sort_idxs.tensor, ref_sort_idxs.tensor)
    torch.testing.assert_close(argsort_comp.tensor, ref_argsort.tensor)
    torch.testing.assert_close(softmax_comp.tensor, ref_softmax.tensor)
    torch.testing.assert_close(log_softmax_comp.tensor, ref_log_softmax.tensor)
    torch.testing.assert_close(dropout_eval_comp.tensor, nt.tensor)
    torch.testing.assert_close(dropout_train_comp.tensor, ref_dropout_train.tensor)
    torch.testing.assert_close(bernoulli_comp.tensor, ref_bernoulli.tensor)
    torch.testing.assert_close(layer_norm_comp.tensor, ref_layer_norm.tensor)
    if hasattr(torch, "rms_norm"):
        torch.testing.assert_close(rms_norm_comp.tensor, ref_rms_norm.tensor)


def test_torch_cumulative_and_topk_wrappers_no_storage_mapping():
    nt = NT(
        [
            torch.tensor([3.0, 1.0, 2.0]),
            torch.tensor([4.0, 0.0]),
        ]
    )

    original_map = torch_functions._map_storage
    original_map_pair = torch_functions._map_storage_pair

    def _fail_map(*_args, **_kwargs):
        raise AssertionError("_map_storage must not be used for migrated torch cumulative wrappers")

    def _fail_map_pair(*_args, **_kwargs):
        raise AssertionError("_map_storage_pair must not be used for migrated torch cumulative wrappers")

    torch_functions._map_storage = _fail_map
    torch_functions._map_storage_pair = _fail_map_pair
    try:
        topk_vals, topk_idxs = torch.topk(nt, 2, dim=1, largest=True, sorted=True)
        cumsum_out = torch.cumsum(nt, dim=1)
        cumprod_out = torch.cumprod(nt, dim=1)
        logcumsumexp_out = torch.logcumsumexp(nt, dim=1)
        cummax_vals, cummax_idxs = torch.cummax(nt, dim=1)
        cummin_vals, cummin_idxs = torch.cummin(nt, dim=1)
    finally:
        torch_functions._map_storage = original_map
        torch_functions._map_storage_pair = original_map_pair

    ref_topk_vals = NT([torch.topk(t, 2, dim=0).values for t in nt], **nt._meta())
    ref_topk_idxs = NT([torch.topk(t, 2, dim=0).indices for t in nt], **nt._meta())
    ref_cumsum = NT([torch.cumsum(t, dim=0) for t in nt], **nt._meta())
    ref_cumprod = NT([torch.cumprod(t, dim=0) for t in nt], **nt._meta())
    ref_logcumsumexp = NT([torch.logcumsumexp(t, dim=0) for t in nt], **nt._meta())
    ref_cummax_vals = NT([torch.cummax(t, dim=0).values for t in nt], **nt._meta())
    ref_cummax_idxs = NT([torch.cummax(t, dim=0).indices for t in nt], **nt._meta())
    ref_cummin_vals = NT([torch.cummin(t, dim=0).values for t in nt], **nt._meta())
    ref_cummin_idxs = NT([torch.cummin(t, dim=0).indices for t in nt], **nt._meta())
    torch.testing.assert_close(topk_vals.tensor, ref_topk_vals.tensor)
    torch.testing.assert_close(topk_idxs.tensor, ref_topk_idxs.tensor)
    torch.testing.assert_close(cumsum_out.tensor, ref_cumsum.tensor)
    torch.testing.assert_close(cumprod_out.tensor, ref_cumprod.tensor)
    torch.testing.assert_close(logcumsumexp_out.tensor, ref_logcumsumexp.tensor)
    torch.testing.assert_close(cummax_vals.tensor, ref_cummax_vals.tensor)
    torch.testing.assert_close(cummax_idxs.tensor, ref_cummax_idxs.tensor)
    torch.testing.assert_close(cummin_vals.tensor, ref_cummin_vals.tensor)
    torch.testing.assert_close(cummin_idxs.tensor, ref_cummin_idxs.tensor)


def test_torch_shape_wrappers_no_storage_mapping():
    nt = NT(
        [
            torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4),
            torch.arange(3 * 3 * 4, dtype=torch.float32).reshape(3, 3, 4),
        ]
    )

    original_map = torch_functions._map_storage
    original_map_pair = torch_functions._map_storage_pair

    def _fail_map(*_args, **_kwargs):
        raise AssertionError("_map_storage must not be used for migrated torch shape wrappers")

    def _fail_map_pair(*_args, **_kwargs):
        raise AssertionError("_map_storage_pair must not be used for migrated torch shape wrappers")

    torch_functions._map_storage = _fail_map
    torch_functions._map_storage_pair = _fail_map_pair
    try:
        flattened = torch.flatten(nt, start_dim=2, end_dim=3)
        permuted = torch.permute(nt, (0, 1, 3, 2))
        transposed = torch.transpose(nt, 2, 3)
        unsqueezed = torch.unsqueeze(nt, 3)
        squeezed = torch.squeeze(unsqueezed, 3)
        unflattened = torch.unflatten(flattened, dim=2, sizes=(3, 4))
    finally:
        torch_functions._map_storage = original_map
        torch_functions._map_storage_pair = original_map_pair

    ref_flattened = NT([torch.flatten(t, start_dim=1, end_dim=2) for t in nt], **nt._meta())
    ref_permuted = NT([torch.permute(t, (0, 2, 1)) for t in nt], **nt._meta())
    ref_transposed = NT([torch.transpose(t, 1, 2) for t in nt], **nt._meta())
    ref_unsqueezed = NT([torch.unsqueeze(t, dim=2) for t in nt], **nt._meta())
    ref_unflattened = NT([torch.unflatten(t, 1, (3, 4)) for t in ref_flattened], **nt._meta())

    torch.testing.assert_close(flattened.tensor, ref_flattened.tensor)
    torch.testing.assert_close(permuted.tensor, ref_permuted.tensor)
    torch.testing.assert_close(transposed.tensor, ref_transposed.tensor)
    torch.testing.assert_close(unsqueezed.tensor, ref_unsqueezed.tensor)
    torch.testing.assert_close(squeezed.tensor, nt.tensor)
    torch.testing.assert_close(unflattened.tensor, ref_unflattened.tensor)


def test_torch_dropout_and_bernoulli_wrappers_no_concat_apply():
    nt = NT(
        [
            torch.tensor([1.0, 1.0, 1.0, 1.0]),
            torch.tensor([1.0, 1.0]),
        ]
    )
    nt_prob = NT(
        [
            torch.tensor([0.2, 0.8, 0.4, 0.6]),
            torch.tensor([0.1, 0.9]),
        ]
    )

    original_concat = torch_functions._concat_apply_same_shape

    def _fail_concat(*_args, **_kwargs):
        raise AssertionError("_concat_apply_same_shape must not be used for migrated torch dropout/bernoulli wrappers")

    torch_functions._concat_apply_same_shape = _fail_concat
    try:
        torch.manual_seed(1234)
        dropout_out = torch.dropout(nt, p=0.5, train=True)
        torch.manual_seed(5678)
        bernoulli_out = torch.bernoulli(nt_prob)
    finally:
        torch_functions._concat_apply_same_shape = original_concat

    torch.manual_seed(1234)
    ref_dropout = NT([torch.dropout(t, p=0.5, train=True) for t in nt], **nt._meta())
    torch.manual_seed(5678)
    ref_bernoulli = NT([torch.bernoulli(t) for t in nt_prob], **nt_prob._meta())
    torch.testing.assert_close(dropout_out.tensor, ref_dropout.tensor)
    torch.testing.assert_close(bernoulli_out.tensor, ref_bernoulli.tensor)


def test_dropout_probability_error_types_match_upstream():
    nt = NT(
        [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0]),
        ]
    )

    with pytest.raises(RuntimeError, match="between 0 and 1"):
        torch.dropout(nt, p=-0.1, train=False)
    with pytest.raises(RuntimeError, match="between 0 and 1"):
        torch.alpha_dropout(nt, p=-0.1, train=False)
    with pytest.raises(RuntimeError, match="between 0 and 1"):
        torch.feature_alpha_dropout(nt, p=-0.1, train=False)

    with pytest.raises(ValueError, match="between 0 and 1"):
        F.dropout(nt, p=-0.1, training=False)
    with pytest.raises(ValueError, match="between 0 and 1"):
        F.alpha_dropout(nt, p=-0.1, training=False)
    with pytest.raises(ValueError, match="between 0 and 1"):
        F.feature_alpha_dropout(nt, p=-0.1, training=False)


def test_torch_norm_wrappers_no_storage_mapping():
    nt = NT(
        [
            torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
            torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
        ]
    )

    original_map = torch_functions._map_storage
    original_map_pair = torch_functions._map_storage_pair

    def _fail_map(*_args, **_kwargs):
        raise AssertionError("_map_storage must not be used for migrated torch normalization wrappers")

    def _fail_map_pair(*_args, **_kwargs):
        raise AssertionError("_map_storage_pair must not be used for migrated torch normalization wrappers")

    torch_functions._map_storage = _fail_map
    torch_functions._map_storage_pair = _fail_map_pair
    try:
        layer_norm_out = torch.layer_norm(nt, (2,))
        if hasattr(torch, "rms_norm"):
            rms_norm_out = torch.rms_norm(nt, (2,))
    finally:
        torch_functions._map_storage = original_map
        torch_functions._map_storage_pair = original_map_pair

    ref_layer_norm = NT([torch.layer_norm(t, (2,)) for t in nt], **nt._meta())
    torch.testing.assert_close(layer_norm_out.tensor, ref_layer_norm.tensor)
    if hasattr(torch, "rms_norm"):
        ref_rms_norm = NT([torch.rms_norm(t, (2,)) for t in nt], **nt._meta())
        torch.testing.assert_close(rms_norm_out.tensor, ref_rms_norm.tensor)


def test_nn_functional_fastpaths_no_apply_per_element():
    conv_nt = NT(
        [
            torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4) / 10.0,
            torch.arange(3 * 3 * 4 * 4, dtype=torch.float32).reshape(3, 3, 4, 4) / 10.0,
        ]
    )
    conv_weight = torch.arange(5 * 3 * 3 * 3, dtype=torch.float32).reshape(5, 3, 3, 3) / 100.0
    conv_bias = torch.arange(5, dtype=torch.float32) / 50.0
    one_hot_nt = NT([torch.tensor([0, 2, 1], dtype=torch.long), torch.tensor([1, 0, 3, 2], dtype=torch.long)])
    grid_input = NT(
        [
            torch.arange(4.0, dtype=torch.float32).view(1, 1, 2, 2),
            torch.arange(4.0, 8.0, dtype=torch.float32).view(1, 1, 2, 2),
        ]
    )
    grid = NT(
        [
            torch.zeros(1, 2, 2, 2, dtype=torch.float32),
            torch.zeros(1, 2, 2, 2, dtype=torch.float32),
        ]
    )
    frac2_nt = NT([torch.randn(2, 1, 4, 4), torch.randn(3, 1, 4, 4)])
    frac3_nt = NT([torch.randn(2, 1, 4, 4, 4), torch.randn(3, 1, 4, 4, 4)])
    frac2_random = torch.rand(frac2_nt._values.size(0), frac2_nt._values.size(1), 2, dtype=frac2_nt.dtype)
    frac3_random = torch.rand(frac3_nt._values.size(0), frac3_nt._values.size(1), 3, dtype=frac3_nt.dtype)
    nt_logits = NT(
        [
            torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
            torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
        ]
    )
    nt_1d = NT(
        [
            torch.tensor([3.0, 1.0, 2.0]),
            torch.tensor([4.0, 0.0]),
        ]
    )
    bilinear_x1 = NT([torch.randn(2, 3), torch.randn(3, 3)])
    bilinear_x2 = NT([torch.randn(2, 4), torch.randn(3, 4)])
    bilinear_weight = torch.randn(5, 3, 4)
    bilinear_bias = torch.randn(5)

    original_apply = nn_functions._apply_per_element
    original_apply_pair = nn_functions._apply_pair

    def _fail_apply(*_args, **_kwargs):
        raise AssertionError("_apply_per_element must not be used for covered nn.functional fastpaths")

    def _fail_apply_pair(*_args, **_kwargs):
        raise AssertionError("_apply_pair must not be used for covered nn.functional fastpaths")

    nn_functions._apply_per_element = _fail_apply
    nn_functions._apply_pair = _fail_apply_pair
    try:
        conv2d_out = F.conv2d(conv_nt, conv_weight, conv_bias, stride=1, padding=1)
        max_pool2d_out = F.max_pool2d(conv_nt, kernel_size=2, stride=2)
        avg_pool2d_out = F.avg_pool2d(conv_nt, kernel_size=2, stride=2)
        interpolate_out = F.interpolate(conv_nt, scale_factor=2, mode="nearest")
        pad_out = F.pad(conv_nt, (1, 1, 1, 1), value=0.5)
        one_hot_out = F.one_hot(one_hot_nt, num_classes=4)
        grid_out = F.grid_sample(grid_input, grid, align_corners=False)
        gumbel_out = F.gumbel_softmax(nt_logits, dim=1, tau=1.0, hard=False)
        gumbel_1d_out = F.gumbel_softmax(nt_1d, dim=1, tau=1.0, hard=False)
        dropout_train_out = F.dropout(nt_logits, p=0.2, training=True)
        alpha_dropout_train_out = F.alpha_dropout(nt_logits, p=0.2, training=True)
        layer_norm_out = F.layer_norm(nt_logits, (2,))
        rms_norm_out = F.rms_norm(nt_logits, (2,))
        normalize_1d_out = F.normalize(nt_1d, dim=1)
        dropout1d_train_out = F.dropout1d(nt_logits, p=0.2, training=True)
        dropout2d_train_out = F.dropout2d(conv_nt, p=0.2, training=True)
        dropout3d_train_out = F.dropout3d(frac3_nt, p=0.2, training=True)
        feature_alpha_dropout_train_out = F.feature_alpha_dropout(conv_nt, p=0.2, training=True)
        frac2_out = F.fractional_max_pool2d(frac2_nt, kernel_size=2, output_size=2, _random_samples=frac2_random)
        frac3_out = F.fractional_max_pool3d(frac3_nt, kernel_size=2, output_size=2, _random_samples=frac3_random)
        bilinear_out = F.bilinear(bilinear_x1, bilinear_x2, bilinear_weight, bilinear_bias)
    finally:
        nn_functions._apply_per_element = original_apply
        nn_functions._apply_pair = original_apply_pair

    assert isinstance(conv2d_out, NT)
    assert isinstance(max_pool2d_out, NT)
    assert isinstance(avg_pool2d_out, NT)
    assert isinstance(interpolate_out, NT)
    assert isinstance(pad_out, NT)
    assert isinstance(one_hot_out, NT)
    assert isinstance(grid_out, NT)
    assert isinstance(gumbel_out, NT)
    assert isinstance(gumbel_1d_out, NT)
    assert isinstance(dropout_train_out, NT)
    assert isinstance(alpha_dropout_train_out, NT)
    assert isinstance(layer_norm_out, NT)
    assert isinstance(rms_norm_out, NT)
    assert isinstance(normalize_1d_out, NT)
    assert isinstance(dropout1d_train_out, NT)
    assert isinstance(dropout2d_train_out, NT)
    assert isinstance(dropout3d_train_out, NT)
    assert isinstance(feature_alpha_dropout_train_out, NT)
    assert isinstance(frac2_out, NT)
    assert isinstance(frac3_out, NT)
    assert isinstance(bilinear_out, NT)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_nn_functional_compile_smoke():
    nt = NT(
        [
            torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
            torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
        ]
    )
    nt_1d = NT(
        [
            torch.tensor([3.0, 1.0, 2.0]),
            torch.tensor([4.0, 0.0]),
        ]
    )
    nt_pair = NT(
        [
            torch.tensor([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]]),
            torch.tensor([[0.5, 0.1], [0.7, 0.3], [0.9, 0.2], [1.1, 0.4], [1.3, 0.5]]),
        ]
    )
    bilinear_x1 = NT([torch.randn(2, 3), torch.randn(3, 3)])
    bilinear_x2 = NT([torch.randn(2, 4), torch.randn(3, 4)])
    bilinear_weight = torch.randn(5, 3, 4)
    bilinear_bias = torch.randn(5)
    weight = torch.tensor([[0.2, -0.5], [1.1, 0.3]])
    bias = torch.tensor([0.4, -0.2])
    conv_nt = NT(
        [
            torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4) / 10.0,
            torch.arange(3 * 3 * 4 * 4, dtype=torch.float32).reshape(3, 3, 4, 4) / 10.0,
        ]
    )
    conv_weight = torch.arange(5 * 3 * 3 * 3, dtype=torch.float32).reshape(5, 3, 3, 3) / 100.0
    conv_bias = torch.arange(5, dtype=torch.float32) / 50.0
    one_hot_nt = NT([torch.tensor([0, 2, 1], dtype=torch.long), torch.tensor([1, 0, 3, 2], dtype=torch.long)])
    grid_input = NT(
        [
            torch.arange(4.0, dtype=torch.float32).view(1, 1, 2, 2),
            torch.arange(4.0, 8.0, dtype=torch.float32).view(1, 1, 2, 2),
        ]
    )
    grid = NT(
        [
            torch.zeros(1, 2, 2, 2, dtype=torch.float32),
            torch.zeros(1, 2, 2, 2, dtype=torch.float32),
        ]
    )
    frac2_nt = NT([torch.randn(2, 1, 4, 4), torch.randn(3, 1, 4, 4)])
    frac3_nt = NT([torch.randn(2, 1, 4, 4, 4), torch.randn(3, 1, 4, 4, 4)])
    frac2_random = torch.rand(
        frac2_nt._values.size(0), frac2_nt._values.size(1), 2, dtype=frac2_nt.dtype, device=frac2_nt.device
    )
    frac3_random = torch.rand(
        frac3_nt._values.size(0), frac3_nt._values.size(1), 3, dtype=frac3_nt.dtype, device=frac3_nt.device
    )

    def _compile(fn):
        return torch.compile(fn, backend="eager", fullgraph=True)

    linear_fn = _compile(lambda x: F.linear(x, weight, bias))
    softmax_fn = _compile(lambda x: F.softmax(x, dim=1))
    log_softmax_fn = _compile(lambda x: F.log_softmax(x, dim=1))
    normalize_fn = _compile(lambda x: F.normalize(x, dim=1))
    dropout_eval_fn = _compile(lambda x: F.dropout(x, p=0.2, training=False))
    dropout_train_fn = _compile(lambda x: F.dropout(x, p=0.2, training=True))
    alpha_dropout_eval_fn = _compile(lambda x: F.alpha_dropout(x, p=0.2, training=False))
    alpha_dropout_train_fn = _compile(lambda x: F.alpha_dropout(x, p=0.2, training=True))
    layer_norm_fn = _compile(lambda x: F.layer_norm(x, (2,)))
    rms_norm_fn = _compile(lambda x: F.rms_norm(x, (2,)))
    normalize_1d_fn = _compile(lambda x: F.normalize(x, dim=1))
    pairwise_fn = _compile(lambda x, y: F.pairwise_distance(x, y, p=2.0, eps=1e-6, keepdim=False))
    dropout1d_eval_fn = _compile(lambda x: F.dropout1d(x, p=0.2, training=False))
    dropout2d_eval_fn = _compile(lambda x: F.dropout2d(x, p=0.2, training=False))
    feature_alpha_dropout_eval_fn = _compile(lambda x: F.feature_alpha_dropout(x, p=0.2, training=False))
    dropout1d_train_fn = _compile(lambda x: F.dropout1d(x, p=0.2, training=True))
    dropout2d_train_fn = _compile(lambda x: F.dropout2d(x, p=0.2, training=True))
    dropout3d_train_fn = _compile(lambda x: F.dropout3d(x, p=0.2, training=True))
    feature_alpha_dropout_train_fn = _compile(lambda x: F.feature_alpha_dropout(x, p=0.2, training=True))
    bilinear_fn = _compile(lambda x, y: F.bilinear(x, y, bilinear_weight, bilinear_bias))
    conv2d_fn = _compile(lambda x: F.conv2d(x, conv_weight, conv_bias, stride=1, padding=1))
    max_pool2d_fn = _compile(lambda x: F.max_pool2d(x, kernel_size=2, stride=2))
    avg_pool2d_fn = _compile(lambda x: F.avg_pool2d(x, kernel_size=2, stride=2))
    interpolate_fn = _compile(lambda x: F.interpolate(x, scale_factor=2, mode="nearest"))
    pad_fn = _compile(lambda x: F.pad(x, (1, 1, 1, 1), value=0.5))
    one_hot_fn = _compile(lambda x: F.one_hot(x, num_classes=4))
    grid_sample_fn = _compile(lambda x, g: F.grid_sample(x, g, align_corners=False))
    gumbel_fn = _compile(lambda x: F.gumbel_softmax(x, dim=1, tau=1.0, hard=False))
    gumbel_1d_fn = _compile(lambda x: F.gumbel_softmax(x, dim=1, tau=1.0, hard=False))
    frac2_fn = _compile(lambda x, rs: F.fractional_max_pool2d(x, kernel_size=2, output_size=2, _random_samples=rs))
    frac3_fn = _compile(lambda x, rs: F.fractional_max_pool3d(x, kernel_size=2, output_size=2, _random_samples=rs))
    linear_comp = linear_fn(nt)
    softmax_comp = softmax_fn(nt)
    log_softmax_comp = log_softmax_fn(nt)
    normalize_comp = normalize_fn(nt)
    dropout_eval_comp = dropout_eval_fn(nt)
    dropout_train_comp = dropout_train_fn(nt)
    alpha_dropout_eval_comp = alpha_dropout_eval_fn(nt)
    alpha_dropout_train_comp = alpha_dropout_train_fn(nt)
    layer_norm_comp = layer_norm_fn(nt)
    rms_norm_comp = rms_norm_fn(nt)
    normalize_1d_comp = normalize_1d_fn(nt_1d)
    pairwise_comp = pairwise_fn(nt, nt_pair)
    dropout1d_eval_comp = dropout1d_eval_fn(nt)
    dropout2d_eval_comp = dropout2d_eval_fn(nt)
    feature_alpha_dropout_eval_comp = feature_alpha_dropout_eval_fn(nt)
    dropout1d_train_comp = dropout1d_train_fn(nt)
    dropout2d_train_comp = dropout2d_train_fn(conv_nt)
    dropout3d_train_comp = dropout3d_train_fn(frac3_nt)
    feature_alpha_dropout_train_comp = feature_alpha_dropout_train_fn(conv_nt)
    bilinear_comp = bilinear_fn(bilinear_x1, bilinear_x2)
    conv2d_comp = conv2d_fn(conv_nt)
    max_pool2d_comp = max_pool2d_fn(conv_nt)
    avg_pool2d_comp = avg_pool2d_fn(conv_nt)
    interpolate_comp = interpolate_fn(conv_nt)
    pad_comp = pad_fn(conv_nt)
    one_hot_comp = one_hot_fn(one_hot_nt)
    grid_sample_comp = grid_sample_fn(grid_input, grid)
    gumbel_comp = gumbel_fn(nt)
    gumbel_1d_comp = gumbel_1d_fn(nt_1d)
    frac2_comp = frac2_fn(frac2_nt, frac2_random)
    frac3_comp = frac3_fn(frac3_nt, frac3_random)

    ref_linear = NT([F.linear(t, weight, bias) for t in nt], **nt._meta())
    ref_softmax = NT([F.softmax(t, dim=0) for t in nt], **nt._meta())
    ref_log_softmax = NT([F.log_softmax(t, dim=0) for t in nt], **nt._meta())
    ref_normalize = NT([F.normalize(t, dim=0) for t in nt], **nt._meta())
    ref_layer_norm = NT([F.layer_norm(t, (2,)) for t in nt], **nt._meta())
    ref_rms_norm = NT([F.rms_norm(t, (2,)) for t in nt], **nt._meta())
    ref_normalize_1d = NT([F.normalize(t, dim=0) for t in nt_1d], **nt_1d._meta())
    ref_pairwise = NT([F.pairwise_distance(a, b) for a, b in zip(nt, nt_pair)], **nt._meta())
    ref_bilinear = NT(
        [F.bilinear(a, b, bilinear_weight, bilinear_bias) for a, b in zip(bilinear_x1, bilinear_x2)],
        **bilinear_x1._meta(),
    )
    ref_conv2d = NT([F.conv2d(t, conv_weight, conv_bias, stride=1, padding=1) for t in conv_nt], **conv_nt._meta())
    ref_max_pool2d = NT([F.max_pool2d(t, kernel_size=2, stride=2) for t in conv_nt], **conv_nt._meta())
    ref_avg_pool2d = NT([F.avg_pool2d(t, kernel_size=2, stride=2) for t in conv_nt], **conv_nt._meta())
    ref_interpolate = NT([F.interpolate(t, scale_factor=2, mode="nearest") for t in conv_nt], **conv_nt._meta())
    ref_pad = NT([F.pad(t, (1, 1, 1, 1), value=0.5) for t in conv_nt], **conv_nt._meta())
    ref_one_hot = NT([F.one_hot(t, num_classes=4) for t in one_hot_nt], **one_hot_nt._meta())
    ref_grid_sample = NT(
        [F.grid_sample(a, b, align_corners=False) for a, b in zip(grid_input, grid)], **grid_input._meta()
    )
    offsets2 = frac2_nt._offsets.tolist()
    ref_frac2 = NT(
        [
            F.fractional_max_pool2d(
                t,
                kernel_size=2,
                output_size=2,
                _random_samples=frac2_random[offsets2[i] : offsets2[i + 1]],
            )
            for i, t in enumerate(frac2_nt)
        ],
        **frac2_nt._meta(),
    )
    offsets3 = frac3_nt._offsets.tolist()
    ref_frac3 = NT(
        [
            F.fractional_max_pool3d(
                t,
                kernel_size=2,
                output_size=2,
                _random_samples=frac3_random[offsets3[i] : offsets3[i + 1]],
            )
            for i, t in enumerate(frac3_nt)
        ],
        **frac3_nt._meta(),
    )
    torch.testing.assert_close(linear_comp.tensor, ref_linear.tensor)
    torch.testing.assert_close(softmax_comp.tensor, ref_softmax.tensor)
    torch.testing.assert_close(log_softmax_comp.tensor, ref_log_softmax.tensor)
    torch.testing.assert_close(normalize_comp.tensor, ref_normalize.tensor)
    torch.testing.assert_close(dropout_eval_comp.tensor, nt.tensor)
    torch.testing.assert_close(alpha_dropout_eval_comp.tensor, nt.tensor)
    assert isinstance(dropout_train_comp, NT)
    assert isinstance(alpha_dropout_train_comp, NT)
    assert dropout_train_comp.shape == nt.shape
    assert alpha_dropout_train_comp.shape == nt.shape
    torch.testing.assert_close(layer_norm_comp.tensor, ref_layer_norm.tensor)
    torch.testing.assert_close(rms_norm_comp.tensor, ref_rms_norm.tensor)
    torch.testing.assert_close(normalize_1d_comp.tensor, ref_normalize_1d.tensor)
    torch.testing.assert_close(pairwise_comp.tensor, ref_pairwise.tensor)
    torch.testing.assert_close(dropout1d_eval_comp.tensor, nt.tensor)
    torch.testing.assert_close(dropout2d_eval_comp.tensor, nt.tensor)
    torch.testing.assert_close(feature_alpha_dropout_eval_comp.tensor, nt.tensor)
    assert isinstance(dropout1d_train_comp, NT)
    assert isinstance(dropout2d_train_comp, NT)
    assert isinstance(dropout3d_train_comp, NT)
    assert isinstance(feature_alpha_dropout_train_comp, NT)
    assert dropout1d_train_comp.shape == nt.shape
    assert dropout2d_train_comp.shape == conv_nt.shape
    assert dropout3d_train_comp.shape == frac3_nt.shape
    assert feature_alpha_dropout_train_comp.shape == conv_nt.shape
    torch.testing.assert_close(bilinear_comp.tensor, ref_bilinear.tensor)
    torch.testing.assert_close(conv2d_comp.tensor, ref_conv2d.tensor)
    torch.testing.assert_close(max_pool2d_comp.tensor, ref_max_pool2d.tensor)
    torch.testing.assert_close(avg_pool2d_comp.tensor, ref_avg_pool2d.tensor)
    torch.testing.assert_close(interpolate_comp.tensor, ref_interpolate.tensor)
    torch.testing.assert_close(pad_comp.tensor, ref_pad.tensor)
    torch.testing.assert_close(one_hot_comp.tensor, ref_one_hot.tensor)
    torch.testing.assert_close(grid_sample_comp.tensor, ref_grid_sample.tensor)
    torch.testing.assert_close(frac2_comp.tensor, ref_frac2.tensor)
    torch.testing.assert_close(frac3_comp.tensor, ref_frac3.tensor)
    assert isinstance(gumbel_comp, NT)
    for t in gumbel_comp:
        colsum = t.sum(dim=0)
        torch.testing.assert_close(colsum, torch.ones_like(colsum), atol=1e-5, rtol=1e-5)
    assert isinstance(gumbel_1d_comp, NT)
    for t in gumbel_1d_comp:
        rowsum = t.sum(dim=0)
        torch.testing.assert_close(rowsum, torch.ones_like(rowsum), atol=1e-5, rtol=1e-5)
