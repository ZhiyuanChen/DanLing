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

from danling.tensors import NestedTensor, aten_functions, nn_functions
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
    nt_logits = NT(
        [
            torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
            torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
        ]
    )

    original_apply = nn_functions._apply_per_element

    def _fail_apply(*_args, **_kwargs):
        raise AssertionError("_apply_per_element must not be used for covered nn.functional fastpaths")

    nn_functions._apply_per_element = _fail_apply
    try:
        conv2d_out = F.conv2d(conv_nt, conv_weight, conv_bias, stride=1, padding=1)
        max_pool2d_out = F.max_pool2d(conv_nt, kernel_size=2, stride=2)
        avg_pool2d_out = F.avg_pool2d(conv_nt, kernel_size=2, stride=2)
        interpolate_out = F.interpolate(conv_nt, scale_factor=2, mode="nearest")
        one_hot_out = F.one_hot(one_hot_nt, num_classes=4)
        grid_out = F.grid_sample(grid_input, grid, align_corners=False)
        gumbel_out = F.gumbel_softmax(nt_logits, dim=1, tau=1.0, hard=False)
        torch.manual_seed(1234)
        frac2_out = F.fractional_max_pool2d(frac2_nt, kernel_size=2, output_size=2)
        torch.manual_seed(1234)
        frac3_out = F.fractional_max_pool3d(frac3_nt, kernel_size=2, output_size=2)
    finally:
        nn_functions._apply_per_element = original_apply

    assert isinstance(conv2d_out, NT)
    assert isinstance(max_pool2d_out, NT)
    assert isinstance(avg_pool2d_out, NT)
    assert isinstance(interpolate_out, NT)
    assert isinstance(one_hot_out, NT)
    assert isinstance(grid_out, NT)
    assert isinstance(gumbel_out, NT)
    assert isinstance(frac2_out, NT)
    assert isinstance(frac3_out, NT)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_nn_functional_compile_smoke():
    nt = NT(
        [
            torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
            torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
        ]
    )
    nt_pair = NT(
        [
            torch.tensor([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]]),
            torch.tensor([[0.5, 0.1], [0.7, 0.3], [0.9, 0.2], [1.1, 0.4], [1.3, 0.5]]),
        ]
    )
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

    def _compile(fn):
        return torch.compile(fn, backend="eager", fullgraph=True)

    linear_fn = _compile(lambda x: F.linear(x, weight, bias))
    softmax_fn = _compile(lambda x: F.softmax(x, dim=1))
    log_softmax_fn = _compile(lambda x: F.log_softmax(x, dim=1))
    normalize_fn = _compile(lambda x: F.normalize(x, dim=1))
    pairwise_fn = _compile(lambda x, y: F.pairwise_distance(x, y, p=2.0, eps=1e-6, keepdim=False))
    dropout1d_eval_fn = _compile(lambda x: F.dropout1d(x, p=0.2, training=False))
    dropout2d_eval_fn = _compile(lambda x: F.dropout2d(x, p=0.2, training=False))
    feature_alpha_dropout_eval_fn = _compile(lambda x: F.feature_alpha_dropout(x, p=0.2, training=False))
    conv2d_fn = _compile(lambda x: F.conv2d(x, conv_weight, conv_bias, stride=1, padding=1))
    max_pool2d_fn = _compile(lambda x: F.max_pool2d(x, kernel_size=2, stride=2))
    avg_pool2d_fn = _compile(lambda x: F.avg_pool2d(x, kernel_size=2, stride=2))
    interpolate_fn = _compile(lambda x: F.interpolate(x, scale_factor=2, mode="nearest"))
    one_hot_fn = _compile(lambda x: F.one_hot(x, num_classes=4))
    grid_sample_fn = _compile(lambda x, g: F.grid_sample(x, g, align_corners=False))
    gumbel_fn = _compile(lambda x: F.gumbel_softmax(x, dim=1, tau=1.0, hard=False))
    frac2_fn = _compile(lambda x: F.fractional_max_pool2d(x, kernel_size=2, output_size=2))
    frac3_fn = _compile(lambda x: F.fractional_max_pool3d(x, kernel_size=2, output_size=2))
    linear_comp = linear_fn(nt)
    softmax_comp = softmax_fn(nt)
    log_softmax_comp = log_softmax_fn(nt)
    normalize_comp = normalize_fn(nt)
    pairwise_comp = pairwise_fn(nt, nt_pair)
    dropout1d_eval_comp = dropout1d_eval_fn(nt)
    dropout2d_eval_comp = dropout2d_eval_fn(nt)
    feature_alpha_dropout_eval_comp = feature_alpha_dropout_eval_fn(nt)
    conv2d_comp = conv2d_fn(conv_nt)
    max_pool2d_comp = max_pool2d_fn(conv_nt)
    avg_pool2d_comp = avg_pool2d_fn(conv_nt)
    interpolate_comp = interpolate_fn(conv_nt)
    one_hot_comp = one_hot_fn(one_hot_nt)
    grid_sample_comp = grid_sample_fn(grid_input, grid)
    gumbel_comp = gumbel_fn(nt)
    torch.manual_seed(1234)
    frac2_comp = frac2_fn(frac2_nt)
    torch.manual_seed(1234)
    frac3_comp = frac3_fn(frac3_nt)

    ref_linear = NT([F.linear(t, weight, bias) for t in nt], **nt._meta())
    ref_softmax = NT([F.softmax(t, dim=0) for t in nt], **nt._meta())
    ref_log_softmax = NT([F.log_softmax(t, dim=0) for t in nt], **nt._meta())
    ref_normalize = NT([F.normalize(t, dim=0) for t in nt], **nt._meta())
    ref_pairwise = NT([F.pairwise_distance(a, b) for a, b in zip(nt, nt_pair)], **nt._meta())
    ref_conv2d = NT([F.conv2d(t, conv_weight, conv_bias, stride=1, padding=1) for t in conv_nt], **conv_nt._meta())
    ref_max_pool2d = NT([F.max_pool2d(t, kernel_size=2, stride=2) for t in conv_nt], **conv_nt._meta())
    ref_avg_pool2d = NT([F.avg_pool2d(t, kernel_size=2, stride=2) for t in conv_nt], **conv_nt._meta())
    ref_interpolate = NT([F.interpolate(t, scale_factor=2, mode="nearest") for t in conv_nt], **conv_nt._meta())
    ref_one_hot = NT([F.one_hot(t, num_classes=4) for t in one_hot_nt], **one_hot_nt._meta())
    ref_grid_sample = NT(
        [F.grid_sample(a, b, align_corners=False) for a, b in zip(grid_input, grid)], **grid_input._meta()
    )
    torch.manual_seed(1234)
    ref_frac2 = NT([F.fractional_max_pool2d(t, kernel_size=2, output_size=2) for t in frac2_nt], **frac2_nt._meta())
    torch.manual_seed(1234)
    ref_frac3 = NT([F.fractional_max_pool3d(t, kernel_size=2, output_size=2) for t in frac3_nt], **frac3_nt._meta())
    torch.testing.assert_close(linear_comp.tensor, ref_linear.tensor)
    torch.testing.assert_close(softmax_comp.tensor, ref_softmax.tensor)
    torch.testing.assert_close(log_softmax_comp.tensor, ref_log_softmax.tensor)
    torch.testing.assert_close(normalize_comp.tensor, ref_normalize.tensor)
    torch.testing.assert_close(pairwise_comp.tensor, ref_pairwise.tensor)
    torch.testing.assert_close(dropout1d_eval_comp.tensor, nt.tensor)
    torch.testing.assert_close(dropout2d_eval_comp.tensor, nt.tensor)
    torch.testing.assert_close(feature_alpha_dropout_eval_comp.tensor, nt.tensor)
    torch.testing.assert_close(conv2d_comp.tensor, ref_conv2d.tensor)
    torch.testing.assert_close(max_pool2d_comp.tensor, ref_max_pool2d.tensor)
    torch.testing.assert_close(avg_pool2d_comp.tensor, ref_avg_pool2d.tensor)
    torch.testing.assert_close(interpolate_comp.tensor, ref_interpolate.tensor)
    torch.testing.assert_close(one_hot_comp.tensor, ref_one_hot.tensor)
    torch.testing.assert_close(grid_sample_comp.tensor, ref_grid_sample.tensor)
    torch.testing.assert_close(frac2_comp.tensor, ref_frac2.tensor)
    torch.testing.assert_close(frac3_comp.tensor, ref_frac3.tensor)
    assert isinstance(gumbel_comp, NT)
    for t in gumbel_comp:
        colsum = t.sum(dim=0)
        torch.testing.assert_close(colsum, torch.ones_like(colsum), atol=1e-5, rtol=1e-5)
