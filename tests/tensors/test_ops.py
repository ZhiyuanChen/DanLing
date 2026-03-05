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

from danling.tensors import NestedTensor, aten_functions
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
    for op in [aten.topk.default, aten.cumsum.default, aten.cumprod.default, aten.flip.default]:
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
    flipped_fast = torch.ops.aten.flip.default(nt, [2])

    for got_vals, got_idxs, got_argsort, got_topv, got_topi, got_csum, got_cprod, got_flip, t in zip(
        sorted_vals,
        sorted_idxs,
        argsorted,
        topk_vals,
        topk_idxs,
        cumsum_fast,
        cumprod_fast,
        flipped_fast,
        nt,
    ):
        ref_sort = torch.sort(t, dim=1, descending=False)
        ref_topk = torch.topk(t, 2, dim=1, largest=True, sorted=True)
        torch.testing.assert_close(got_vals, ref_sort.values)
        torch.testing.assert_close(got_idxs, ref_sort.indices)
        torch.testing.assert_close(got_argsort, torch.argsort(t, dim=1, descending=False))
        torch.testing.assert_close(got_topv, ref_topk.values)
        torch.testing.assert_close(got_topi, ref_topk.indices)
        torch.testing.assert_close(got_csum, torch.cumsum(t, dim=1))
        torch.testing.assert_close(got_cprod, torch.cumprod(t, dim=1))
        torch.testing.assert_close(got_flip, torch.flip(t, dims=[1]))

    # Fallback path: ragged per-element dim (NestedTensor dim=1 -> tensor dim=0).
    cumsum_ragged = torch.ops.aten.cumsum.default(nt, 1)
    for got, t in zip(cumsum_ragged, nt):
        torch.testing.assert_close(got, torch.cumsum(t, dim=0))


def test_aten_ragged_topk_cumsum_cumprod_flip_no_fallback():
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
        flipped_ragged = torch.ops.aten.flip.default(nt, [1])
    finally:
        aten_functions.per_element_fallback = original_fallback

    ref_topk_vals = NT([torch.topk(t, 2, dim=0).values for t in nt], **nt._meta())
    ref_topk_idxs = NT([torch.topk(t, 2, dim=0).indices for t in nt], **nt._meta())
    ref_cumsum = NT([torch.cumsum(t, dim=0) for t in nt], **nt._meta())
    ref_cumprod = NT([torch.cumprod(t, dim=0) for t in nt], **nt._meta())
    ref_flip = NT([torch.flip(t, dims=[0]) for t in nt], **nt._meta())
    torch.testing.assert_close(topk_vals.tensor, ref_topk_vals.tensor)
    torch.testing.assert_close(topk_idxs.tensor, ref_topk_idxs.tensor)
    torch.testing.assert_close(cumsum_ragged.tensor, ref_cumsum.tensor)
    torch.testing.assert_close(cumprod_ragged.tensor, ref_cumprod.tensor)
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
        flip_fn = _compile(lambda x: torch.ops.aten.flip.default(x, [1]))
        sort_fn = _compile(lambda x: torch.ops.aten.sort.default(x, 1, False))
        argsort_fn = _compile(lambda x: torch.ops.aten.argsort.default(x, 1, False))
        softmax_fn = _compile(lambda x: torch.ops.aten._softmax.default(x, 1, False))
        log_softmax_fn = _compile(lambda x: torch.ops.aten._log_softmax.default(x, 1, False))
        topk_comp_vals, topk_comp_idxs = topk_fn(nt)
        cumsum_comp = cumsum_fn(nt)
        cumprod_comp = cumprod_fn(nt)
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
    torch.testing.assert_close(flip_comp.tensor, ref_flip.tensor)
    torch.testing.assert_close(sort_comp_vals.tensor, ref_sort_vals.tensor)
    torch.testing.assert_close(sort_comp_idxs.tensor, ref_sort_idxs.tensor)
    torch.testing.assert_close(argsort_comp.tensor, ref_argsort.tensor)
    torch.testing.assert_close(softmax_comp.tensor, ref_softmax.tensor)
    torch.testing.assert_close(log_softmax_comp.tensor, ref_log_softmax.tensor)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_torch_ragged_fastpaths_compile_smoke():
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
        raise AssertionError("per_element_fallback must not be used for covered ragged torch fastpaths")

    aten_functions.per_element_fallback = _fail_fallback
    try:
        topk_fn = _compile(lambda x: torch.topk(x, 2, dim=1, largest=True, sorted=True))
        flip_fn = _compile(lambda x: torch.flip(x, dims=[1]))
        sort_fn = _compile(lambda x: torch.sort(x, dim=1, descending=False))
        argsort_fn = _compile(lambda x: torch.argsort(x, dim=1, descending=False))
        softmax_fn = _compile(lambda x: torch.softmax(x, dim=1))
        log_softmax_fn = _compile(lambda x: torch.log_softmax(x, dim=1))
        topk_vals, topk_idxs = topk_fn(nt)
        flip_comp = flip_fn(nt)
        sort_vals, sort_idxs = sort_fn(nt)
        argsort_comp = argsort_fn(nt)
        softmax_comp = softmax_fn(nt)
        log_softmax_comp = log_softmax_fn(nt)
    finally:
        aten_functions.per_element_fallback = original_fallback

    ref_topk_vals = NT([torch.topk(t, 2, dim=0).values for t in nt], **nt._meta())
    ref_topk_idxs = NT([torch.topk(t, 2, dim=0).indices for t in nt], **nt._meta())
    ref_flip = NT([torch.flip(t, dims=[0]) for t in nt], **nt._meta())
    ref_sort_vals = NT([torch.sort(t, dim=0, descending=False).values for t in nt], **nt._meta())
    ref_sort_idxs = NT([torch.sort(t, dim=0, descending=False).indices for t in nt], **nt._meta())
    ref_argsort = NT([torch.argsort(t, dim=0, descending=False) for t in nt], **nt._meta())
    ref_softmax = NT([torch.softmax(t, dim=0) for t in nt], **nt._meta())
    ref_log_softmax = NT([torch.log_softmax(t, dim=0) for t in nt], **nt._meta())
    torch.testing.assert_close(topk_vals.tensor, ref_topk_vals.tensor)
    torch.testing.assert_close(topk_idxs.tensor, ref_topk_idxs.tensor)
    torch.testing.assert_close(flip_comp.tensor, ref_flip.tensor)
    torch.testing.assert_close(sort_vals.tensor, ref_sort_vals.tensor)
    torch.testing.assert_close(sort_idxs.tensor, ref_sort_idxs.tensor)
    torch.testing.assert_close(argsort_comp.tensor, ref_argsort.tensor)
    torch.testing.assert_close(softmax_comp.tensor, ref_softmax.tensor)
    torch.testing.assert_close(log_softmax_comp.tensor, ref_log_softmax.tensor)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_nn_functional_compile_smoke():
    nt = NT(
        [
            torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
            torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
        ]
    )
    weight = torch.tensor([[0.2, -0.5], [1.1, 0.3]])
    bias = torch.tensor([0.4, -0.2])

    def _compile(fn):
        return torch.compile(fn, backend="eager", fullgraph=True)

    linear_fn = _compile(lambda x: F.linear(x, weight, bias))
    softmax_fn = _compile(lambda x: F.softmax(x, dim=1))
    log_softmax_fn = _compile(lambda x: F.log_softmax(x, dim=1))
    linear_comp = linear_fn(nt)
    softmax_comp = softmax_fn(nt)
    log_softmax_comp = log_softmax_fn(nt)

    ref_linear = NT([F.linear(t, weight, bias) for t in nt], **nt._meta())
    ref_softmax = NT([F.softmax(t, dim=0) for t in nt], **nt._meta())
    ref_log_softmax = NT([F.log_softmax(t, dim=0) for t in nt], **nt._meta())
    torch.testing.assert_close(linear_comp.tensor, ref_linear.tensor)
    torch.testing.assert_close(softmax_comp.tensor, ref_softmax.tensor)
    torch.testing.assert_close(log_softmax_comp.tensor, ref_log_softmax.tensor)
