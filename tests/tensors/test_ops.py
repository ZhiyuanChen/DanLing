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
from tests.tensors.utils import nested_rand, ragged_shapes

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
        aten.max.default,
        aten.max.dim,
        aten.min.default,
        aten.min.dim,
        aten.argmax.default,
        aten.argmin.default,
        aten.kthvalue.default,
        aten.median.default,
        aten.median.dim,
        aten.mode.default,
        aten.nanmedian.default,
        aten.nanmedian.dim,
        aten.gather.default,
        aten.index_select.default,
        aten.index_add.default,
        aten.index_copy.default,
        aten.index_put.default,
        aten.masked_fill.Scalar,
        aten.masked_fill.Tensor,
        aten.masked_select.default,
        aten.masked_scatter.default,
        aten.nonzero.default,
        aten.take.default,
        aten.scatter.src,
        aten.scatter.value,
        aten.scatter_add.default,
        aten.where.self,
        aten.where.ScalarOther,
        aten.where.ScalarSelf,
        aten.where.Scalar,
        aten.addcmul.default,
        aten.addcdiv.default,
        aten.alpha_dropout.default,
        aten.feature_alpha_dropout.default,
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
        aten.diagonal.default,
        aten.det.default,
        aten.inverse.default,
        aten.matrix_power.default,
        aten.trace.default,
        aten.linalg_inv.default,
        aten.linalg_det.default,
        aten.linalg_vector_norm.default,
        aten.linalg_cholesky.default,
        aten.matmul.default,
        aten.linalg_qr.default,
        aten.linalg_eigh.default,
        aten.linalg_svd.default,
        aten.linalg_solve.default,
        aten.searchsorted.Tensor,
    ]:
        assert op in NestedTensorAtenRegistry, f"Aten op {op} missing from NestedTensorAtenRegistry"
    if hasattr(aten, "scatter_reduce"):
        assert aten.scatter_reduce.two in NestedTensorAtenRegistry


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


def test_aten_binary_dense_tensor_same_shape_matches_per_element():
    nt = NT([torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])])
    dense = nt.tensor + 0.5
    nested_dense = nt.nested_like(dense, strict=False)

    pow_out = torch.ops.aten.pow.Tensor_Tensor(nt, dense)
    pow_rev_out = torch.ops.aten.pow.Tensor_Tensor(dense, nt)
    and_out = torch.ops.aten.bitwise_and.Tensor(nt.int(), dense.int())

    pow_ref = NT([torch.pow(a, b) for a, b in zip(nt, nested_dense)], **nt._meta())
    pow_rev_ref = NT([torch.pow(b, a) for a, b in zip(nt, nested_dense)], **nt._meta())
    and_ref = NT([torch.bitwise_and(a.int(), b.int()) for a, b in zip(nt, nested_dense)], **nt._meta())
    torch.testing.assert_close(pow_out.tensor, pow_ref.tensor)
    torch.testing.assert_close(pow_rev_out.tensor, pow_rev_ref.tensor)
    torch.testing.assert_close(and_out.tensor, and_ref.tensor)


def test_aten_masked_scatter_nested_source_matches_per_element():
    base = NT([torch.zeros(3), torch.zeros(2)])
    mask = NT([torch.tensor([True, False, True]), torch.tensor([False, True])])
    source = NT([torch.tensor([1.0, 2.0]), torch.tensor([3.0])])

    output = torch.ops.aten.masked_scatter.default(base, mask, source)
    reference = NT([torch.masked_scatter(t, m, s) for t, m, s in zip(base, mask, source)], **base._meta())

    torch.testing.assert_close(output.tensor, reference.tensor)


def test_aten_masked_fill_handler_matches_per_element():
    base = NT([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
    mask = NT([torch.tensor([True, False, True]), torch.tensor([False, True])])

    scalar_out = torch.ops.aten.masked_fill.Scalar(base, mask, 0.0)
    scalar_ref = NT([torch.masked_fill(t, m, 0.0) for t, m in zip(base, mask)], **base._meta())
    torch.testing.assert_close(scalar_out.tensor, scalar_ref.tensor)

    value = torch.tensor(-1.5, dtype=base.dtype, device=base.device)
    tensor_out = torch.ops.aten.masked_fill.Tensor(base, mask, value)
    tensor_ref = NT([torch.masked_fill(t, m, value) for t, m in zip(base, mask)], **base._meta())
    torch.testing.assert_close(tensor_out.tensor, tensor_ref.tensor)


def test_aten_masked_select_handler_matches_per_element():
    base = NT([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
    mask = NT([torch.tensor([True, False, True]), torch.tensor([False, True])])

    output = torch.ops.aten.masked_select.default(base, mask)
    reference = NT([torch.masked_select(t, m) for t, m in zip(base, mask)], **base._meta())
    torch.testing.assert_close(output.tensor, reference.tensor)


def test_aten_nonzero_handler_matches_per_element():
    base = NT([torch.tensor([[0.0, 1.0], [2.0, 0.0]]), torch.tensor([[3.0, 0.0]])])

    output = torch.ops.aten.nonzero.default(base)
    reference = NT([torch.nonzero(t, as_tuple=False) for t in base], **base._meta())

    torch.testing.assert_close(output.tensor, reference.tensor)
    assert output.dtype == torch.long


def test_aten_linalg_vector_norm_handler_matches_per_element():
    base = NT(
        [
            torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
            torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
        ]
    )

    global_out = torch.ops.aten.linalg_vector_norm.default(base, 2, None, False)
    global_ref = NT([torch.linalg.vector_norm(t, ord=2) for t in base], **base._meta())
    torch.testing.assert_close(global_out.tensor, global_ref.tensor)

    dim_out = torch.ops.aten.linalg_vector_norm.default(base, 2, [1], False)
    dim_ref = NT([torch.linalg.vector_norm(t, ord=2, dim=0) for t in base], **base._meta())
    torch.testing.assert_close(dim_out.tensor, dim_ref.tensor)


def test_aten_ternary_and_take_handlers_match_per_element():
    nt = NT(
        [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0]),
        ]
    )

    dense = nt.tensor
    where_out = torch.ops.aten.where.ScalarOther(dense > 2, nt, 0.0)
    where_ref = NT([torch.where(t > 2, t, 0.0) for t in nt], **nt._meta())
    torch.testing.assert_close(where_out.tensor, where_ref.tensor)

    where_scalar_out = torch.ops.aten.where.Scalar(nt > 2, 1.0, 0.0)
    where_scalar_ref = NT([torch.where(t > 2, 1.0, 0.0) for t in nt], **nt._meta())
    torch.testing.assert_close(where_scalar_out.tensor, where_scalar_ref.tensor)

    addcmul_out = torch.ops.aten.addcmul.default(dense, nt, nt, value=2)
    addcmul_ref = NT([torch.addcmul(t, t, t, value=2) for t in nt], **nt._meta())
    torch.testing.assert_close(addcmul_out.tensor, addcmul_ref.tensor)

    addcdiv_out = torch.ops.aten.addcdiv.default(dense, nt, nt + 1, value=2)
    addcdiv_ref = NT([torch.addcdiv(t, t, t + 1, value=2) for t in nt], **nt._meta())
    torch.testing.assert_close(addcdiv_out.tensor, addcdiv_ref.tensor)

    take_index = torch.tensor([0, 2, 4], dtype=torch.long)
    take_out = torch.ops.aten.take.default(nt, take_index)
    take_ref = torch.take(torch.cat([t.reshape(-1) for t in nt]), take_index)
    torch.testing.assert_close(take_out, take_ref)


def test_aten_ternary_handler_accepts_per_element_broadcast():
    nt = NT(
        [
            torch.ones(2, 3),
            torch.ones(4, 3),
        ]
    )
    bias = torch.arange(3.0)

    output = torch.ops.aten.addcmul.default(nt, bias, bias, value=2)
    reference = NT([torch.addcmul(t, bias, bias, value=2) for t in nt], **nt._meta())

    torch.testing.assert_close(output.tensor, reference.tensor)


def test_aten_ternary_handler_rejects_packed_only_broadcast():
    nt = NT(
        [
            torch.ones(2, 3),
            torch.ones(4, 3),
        ]
    )
    bad = torch.arange(18.0).reshape(6, 3)

    with pytest.raises(RuntimeError, match="size of tensor"):
        torch.ops.aten.addcmul.default(nt, bad, bad, value=2)


def test_aten_backward_handlers_fall_back_on_mismatched_offsets():
    grad = NT([torch.randn(3, 4), torch.randn(3, 4)])
    output = NT([torch.randn(2, 4), torch.randn(4, 4)])

    original_fallback = aten_functions.per_element_fallback

    def _fail_fallback(*_args, **_kwargs):
        raise RuntimeError("forced fallback")

    aten_functions.per_element_fallback = _fail_fallback
    try:
        with pytest.raises(RuntimeError, match="forced fallback"):
            torch.ops.aten._softmax_backward_data.default(grad, output, 1, output.dtype)

        with pytest.raises(RuntimeError, match="forced fallback"):
            torch.ops.aten.native_layer_norm_backward.default(
                grad,
                output,
                (4,),
                torch.zeros(1),
                torch.ones(1),
                None,
                None,
                (True, True, True),
            )
    finally:
        aten_functions.per_element_fallback = original_fallback


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_aten_ternary_randomized_dense_parity(seed, device):
    dtype = torch.float32
    shapes = ragged_shapes(seed, batch_size=3, min_len=2, max_len=5, trailing_shape=(4,))
    nt = nested_rand(shapes, device, dtype)

    torch.manual_seed(seed)
    bias = torch.randn(4, device=device, dtype=dtype)
    scale = torch.randn(4, device=device, dtype=dtype)
    denom = torch.randn(4, device=device, dtype=dtype).abs() + 0.5
    condition = torch.randn(4, device=device) > 0

    addcmul_out = torch.ops.aten.addcmul.default(nt, bias, scale, value=0.25)
    addcmul_ref = NT([torch.addcmul(t, bias, scale, value=0.25) for t in nt], **nt._meta())
    torch.testing.assert_close(addcmul_out.tensor, addcmul_ref.tensor)

    addcdiv_out = torch.ops.aten.addcdiv.default(nt, scale, denom, value=-0.5)
    addcdiv_ref = NT([torch.addcdiv(t, scale, denom, value=-0.5) for t in nt], **nt._meta())
    torch.testing.assert_close(addcdiv_out.tensor, addcdiv_ref.tensor)

    where_out = torch.ops.aten.where.self(condition, nt, bias)
    where_ref = NT([torch.where(condition, t, bias) for t in nt], **nt._meta())
    torch.testing.assert_close(where_out.tensor, where_ref.tensor)


@pytest.mark.parametrize("seed", [5, 17, 31])
def test_aten_index_put_randomized_dense_parity(seed, device):
    dtype = torch.float32
    shapes = ragged_shapes(seed, batch_size=3, min_len=3, max_len=6, trailing_shape=(4,))
    base = nested_rand(shapes, device, dtype)
    min_rows = min(shape[0] for shape in shapes)
    generator = torch.Generator()
    generator.manual_seed(seed)

    row_count = 2
    rows = torch.randint(0, min_rows, (row_count,), generator=generator)
    if row_count > 1:
        rows[-1] = rows[-1] - min_rows
    rows = rows.to(device=device, dtype=torch.long)

    shared_values = torch.randn(row_count, 4, device=device, dtype=dtype)
    row_out = torch.ops.aten.index_put.default(base, [rows], shared_values, False)
    row_ref = NT([torch.index_put(t, (rows,), shared_values) for t in base], **base._meta())
    torch.testing.assert_close(row_out.tensor, row_ref.tensor)

    dup_rows = rows.clone()
    dup_rows[0] = dup_rows[-1]
    dup_values = torch.randn(row_count, 4, device=device, dtype=dtype)
    dup_out = torch.ops.aten.index_put.default(base, [dup_rows], dup_values, True)
    dup_ref = NT([torch.index_put(t, (dup_rows,), dup_values, accumulate=True) for t in base], **base._meta())
    torch.testing.assert_close(dup_out.tensor, dup_ref.tensor)

    point_rows = torch.randint(0, min_rows, (2, 2), generator=generator)
    point_rows[0, 0] = point_rows[0, 0] - min_rows
    point_rows = point_rows.to(device=device, dtype=torch.long)
    point_cols = torch.randint(0, 4, (2, 2), generator=generator).to(device=device, dtype=torch.long)
    point_values = NT(
        [torch.randn(2, 2, device=device, dtype=dtype) for _ in range(len(base))],
        **base._meta(),
    )
    point_out = torch.ops.aten.index_put.default(base, [point_rows, point_cols], point_values, False)
    point_ref = NT(
        [torch.index_put(t, (point_rows, point_cols), v) for t, v in zip(base, point_values)], **base._meta()
    )
    torch.testing.assert_close(point_out.tensor, point_ref.tensor)


def test_torch_matmul_and_linalg_handlers_match_per_element():
    a0 = torch.randn(2, 3, 3)
    a1 = torch.randn(3, 3, 3)
    b0 = torch.randn(2, 3, 4)
    b1 = torch.randn(3, 3, 4)
    nt_a = NT([a0, a1])
    nt_b = NT([b0, b1])

    matmul_out = torch.matmul(nt_a, nt_b)
    matmul_ref = NT([torch.matmul(x, y) for x, y in zip(nt_a, nt_b)], **nt_a._meta())
    torch.testing.assert_close(matmul_out.tensor, matmul_ref.tensor)

    qr_q, qr_r = torch.ops.aten.linalg_qr.default(nt_a, mode="reduced")
    for q_elem, r_elem, a_elem in zip(qr_q, qr_r, nt_a):
        torch.testing.assert_close(torch.matmul(q_elem, r_elem), a_elem, rtol=1e-5, atol=1e-5)

    sym_a0 = a0 + a0.transpose(-1, -2)
    sym_a1 = a1 + a1.transpose(-1, -2)
    nt_sym = NT([sym_a0, sym_a1])
    eigh_vals, eigh_vecs = torch.ops.aten.linalg_eigh.default(nt_sym, UPLO="L")
    for vals_elem, vecs_elem, sym_elem in zip(eigh_vals, eigh_vecs, nt_sym):
        ref_vals, _ = torch.linalg.eigh(sym_elem, UPLO="L")
        torch.testing.assert_close(vals_elem, ref_vals, rtol=1e-5, atol=1e-5)
        recon = vecs_elem @ torch.diag_embed(vals_elem) @ vecs_elem.transpose(-1, -2)
        torch.testing.assert_close(recon, sym_elem, rtol=1e-5, atol=1e-5)

    rhs0 = torch.randn(2, 3, 2)
    rhs1 = torch.randn(3, 3, 2)
    nt_rhs = NT([rhs0, rhs1])
    solve_out = torch.linalg.solve(nt_sym, nt_rhs)
    solve_ref = NT([torch.linalg.solve(x, y) for x, y in zip(nt_sym, nt_rhs)], **nt_sym._meta())
    torch.testing.assert_close(solve_out.tensor, solve_ref.tensor, rtol=1e-5, atol=1e-5)

    svd_u, svd_s, svd_vh = torch.linalg.svd(nt_a, full_matrices=True)
    for u_elem, s_elem, vh_elem, a_elem in zip(svd_u, svd_s, svd_vh, nt_a):
        ref_s = torch.linalg.svd(a_elem, full_matrices=True).S
        torch.testing.assert_close(s_elem, ref_s, rtol=1e-5, atol=1e-5)
        recon = u_elem @ torch.diag_embed(s_elem) @ vh_elem
        torch.testing.assert_close(recon, a_elem, rtol=1e-5, atol=1e-5)


def test_aten_linalg_handlers_low_dim_fallback_matches_per_element():
    # 2-D element tensors produce packed _values.dim()==2 and should take per-element fallback paths.
    qr_a0 = torch.randn(3, 3)
    qr_a1 = torch.randn(5, 3)
    nt_qr = NT([qr_a0, qr_a1])

    qr_q, qr_r = torch.ops.aten.linalg_qr.default(nt_qr, mode="reduced")
    for q_elem, r_elem, a_elem in zip(qr_q, qr_r, nt_qr):
        torch.testing.assert_close(q_elem @ r_elem, a_elem, rtol=1e-5, atol=1e-5)

    sym0 = torch.randn(3, 3)
    sym1 = torch.randn(4, 4)
    sym0 = sym0 + sym0.transpose(-1, -2)
    sym1 = sym1 + sym1.transpose(-1, -2)
    nt_sym = NT([sym0, sym1])
    eig_vals, eig_vecs = torch.ops.aten.linalg_eigh.default(nt_sym, UPLO="L")
    for vals_elem, vecs_elem, sym_elem in zip(eig_vals, eig_vecs, nt_sym):
        ref_vals, _ = torch.linalg.eigh(sym_elem, UPLO="L")
        torch.testing.assert_close(vals_elem, ref_vals, rtol=1e-5, atol=1e-5)
        recon = vecs_elem @ torch.diag_embed(vals_elem) @ vecs_elem.transpose(-1, -2)
        torch.testing.assert_close(recon, sym_elem, rtol=1e-5, atol=1e-5)

    svd_a0 = torch.randn(3, 2)
    svd_a1 = torch.randn(4, 2)
    nt_svd = NT([svd_a0, svd_a1])
    svd_u, svd_s, svd_vh = torch.ops.aten.linalg_svd.default(nt_svd, full_matrices=False)
    for u_elem, s_elem, vh_elem, a_elem in zip(svd_u, svd_s, svd_vh, nt_svd):
        ref_s = torch.linalg.svd(a_elem, full_matrices=False).S
        torch.testing.assert_close(s_elem, ref_s, rtol=1e-5, atol=1e-5)
        recon = u_elem @ torch.diag_embed(s_elem) @ vh_elem
        torch.testing.assert_close(recon, a_elem, rtol=1e-5, atol=1e-5)

    spd0 = torch.randn(3, 3)
    spd1 = torch.randn(4, 4)
    spd0 = spd0 @ spd0.transpose(-1, -2) + 1e-1 * torch.eye(3)
    spd1 = spd1 @ spd1.transpose(-1, -2) + 1e-1 * torch.eye(4)
    nt_spd = NT([spd0, spd1])
    rhs0 = torch.randn(3, 2)
    rhs1 = torch.randn(4, 2)
    nt_rhs = NT([rhs0, rhs1])

    solve_out = torch.ops.aten.linalg_solve.default(nt_spd, nt_rhs)
    solve_ref = NT([torch.linalg.solve(a, b) for a, b in zip(nt_spd, nt_rhs)], **nt_spd._meta())
    for out_elem, ref_elem in zip(solve_out, solve_ref):
        torch.testing.assert_close(out_elem, ref_elem, rtol=1e-5, atol=1e-5)


def test_aten_extrema_and_arg_handlers_match_per_element():
    nt = NT(
        [
            torch.tensor([[1.0, 5.0, 3.0], [2.0, 0.0, 4.0]]),
            torch.tensor([[9.0, 1.0, 2.0], [3.0, 7.0, 6.0], [4.0, 0.0, 8.0]]),
        ]
    )

    torch.testing.assert_close(torch.ops.aten.max.default(nt), torch.max(nt._values))
    torch.testing.assert_close(torch.ops.aten.min.default(nt), torch.min(nt._values))

    max_vals_last, max_idxs_last = torch.ops.aten.max.dim(nt, 2, False)
    min_vals_last, min_idxs_last = torch.ops.aten.min.dim(nt, 2, False)
    max_vals_last_ref = NT([torch.max(t, dim=1).values for t in nt], **nt._meta())
    max_idxs_last_ref = NT([torch.max(t, dim=1).indices for t in nt], **nt._meta())
    min_vals_last_ref = NT([torch.min(t, dim=1).values for t in nt], **nt._meta())
    min_idxs_last_ref = NT([torch.min(t, dim=1).indices for t in nt], **nt._meta())
    torch.testing.assert_close(max_vals_last.tensor, max_vals_last_ref.tensor)
    torch.testing.assert_close(max_idxs_last.tensor, max_idxs_last_ref.tensor)
    torch.testing.assert_close(min_vals_last.tensor, min_vals_last_ref.tensor)
    torch.testing.assert_close(min_idxs_last.tensor, min_idxs_last_ref.tensor)

    max_vals_ragged, max_idxs_ragged = torch.ops.aten.max.dim(nt, 1, False)
    min_vals_ragged, min_idxs_ragged = torch.ops.aten.min.dim(nt, 1, False)
    max_vals_ragged_ref = torch.stack([torch.max(t, dim=0).values for t in nt])
    max_idxs_ragged_ref = torch.stack([torch.max(t, dim=0).indices for t in nt])
    min_vals_ragged_ref = torch.stack([torch.min(t, dim=0).values for t in nt])
    min_idxs_ragged_ref = torch.stack([torch.min(t, dim=0).indices for t in nt])
    torch.testing.assert_close(max_vals_ragged, max_vals_ragged_ref)
    torch.testing.assert_close(max_idxs_ragged, max_idxs_ragged_ref)
    torch.testing.assert_close(min_vals_ragged, min_vals_ragged_ref)
    torch.testing.assert_close(min_idxs_ragged, min_idxs_ragged_ref)

    torch.testing.assert_close(torch.ops.aten.argmax.default(nt), torch.tensor([1, 0], dtype=torch.long))
    torch.testing.assert_close(torch.ops.aten.argmin.default(nt), torch.tensor([4, 7], dtype=torch.long))

    argmax_last = torch.ops.aten.argmax.default(nt, 2, False)
    argmin_last = torch.ops.aten.argmin.default(nt, 2, False)
    argmax_last_ref = NT([torch.argmax(t, dim=1) for t in nt], **nt._meta())
    argmin_last_ref = NT([torch.argmin(t, dim=1) for t in nt], **nt._meta())
    torch.testing.assert_close(argmax_last.tensor, argmax_last_ref.tensor)
    torch.testing.assert_close(argmin_last.tensor, argmin_last_ref.tensor)

    argmax_ragged = torch.ops.aten.argmax.default(nt, 1, False)
    argmin_ragged = torch.ops.aten.argmin.default(nt, 1, False)
    argmax_ragged_ref = torch.stack([torch.argmax(t, dim=0) for t in nt])
    argmin_ragged_ref = torch.stack([torch.argmin(t, dim=0) for t in nt])
    torch.testing.assert_close(argmax_ragged, argmax_ragged_ref)
    torch.testing.assert_close(argmin_ragged, argmin_ragged_ref)


def test_aten_order_stat_handlers_match_per_element():
    nt = NT(
        [
            torch.tensor([[3.0, 1.0, 2.0], [5.0, 4.0, 6.0]]),
            torch.tensor([[9.0, 7.0, 8.0], [0.0, 2.0, 1.0], [6.0, 5.0, 4.0]]),
        ]
    )

    kth_vals, kth_idxs = torch.ops.aten.kthvalue.default(nt, 2, 2, False)
    kth_ref = tuple(NT([torch.kthvalue(t, 2, dim=1)[i] for t in nt], **nt._meta()) for i in range(2))
    torch.testing.assert_close(kth_vals.tensor, kth_ref[0].tensor)
    torch.testing.assert_close(kth_idxs.tensor, kth_ref[1].tensor)

    median_global = torch.ops.aten.median.default(nt)
    torch.testing.assert_close(median_global, torch.median(nt._values))

    median_vals, median_idxs = torch.ops.aten.median.dim(nt, 2, True)
    median_ref = tuple(NT([torch.median(t, dim=1, keepdim=True)[i] for t in nt], **nt._meta()) for i in range(2))
    torch.testing.assert_close(median_vals.tensor, median_ref[0].tensor)
    torch.testing.assert_close(median_idxs.tensor, median_ref[1].tensor)

    mode_vals, mode_idxs = torch.ops.aten.mode.default(nt, 2, False)
    mode_ref = tuple(NT([torch.mode(t, dim=1)[i] for t in nt], **nt._meta()) for i in range(2))
    torch.testing.assert_close(mode_vals.tensor, mode_ref[0].tensor)
    torch.testing.assert_close(mode_idxs.tensor, mode_ref[1].tensor)

    nan_nt = NT(
        [
            torch.tensor([[1.0, float("nan"), 3.0], [2.0, 4.0, float("nan")]]),
            torch.tensor([[float("nan"), 7.0, 8.0], [0.0, float("nan"), 1.0], [6.0, 5.0, 4.0]]),
        ]
    )
    nanmedian_global = torch.ops.aten.nanmedian.default(nan_nt)
    torch.testing.assert_close(nanmedian_global, torch.nanmedian(nan_nt._values), equal_nan=True)

    nanmedian_vals, nanmedian_idxs = torch.ops.aten.nanmedian.dim(nan_nt, 2, False)
    nanmedian_ref = tuple(NT([torch.nanmedian(t, dim=1)[i] for t in nan_nt], **nan_nt._meta()) for i in range(2))
    torch.testing.assert_close(nanmedian_vals.tensor, nanmedian_ref[0].tensor, equal_nan=True)
    torch.testing.assert_close(nanmedian_idxs.tensor, nanmedian_ref[1].tensor)


def test_aten_indexing_handlers_match_per_element():
    nt = NT(
        [
            torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        ]
    )

    batch_index = torch.tensor([1, 0, 1], dtype=torch.long)
    batch_out = torch.ops.aten.index_select.default(nt, 0, batch_index)
    batch_ref = NT([nt[1], nt[0], nt[1]], **nt._meta())
    torch.testing.assert_close(batch_out.tensor, batch_ref.tensor)

    gather_dense = torch.tensor(
        [
            [[0, 2, 1], [1, 0, 2], [0, 0, 0]],
            [[2, 1, 0], [0, 2, 1], [1, 1, 1]],
        ],
        dtype=torch.long,
    )
    gather_out = torch.ops.aten.gather.default(nt, 2, gather_dense, sparse_grad=False)
    gather_nt = nt.nested_like(gather_dense, strict=False)
    gather_ref = NT([torch.gather(t, 1, idx) for t, idx in zip(nt, gather_nt)], **nt._meta())
    torch.testing.assert_close(gather_out.tensor, gather_ref.tensor)

    ragged_index = NT(
        [
            torch.tensor([[1, 0, 1]], dtype=torch.long),
            torch.tensor([[2, 1, 0], [0, 2, 1]], dtype=torch.long),
        ]
    )
    gather_ragged = torch.ops.aten.gather.default(nt, 1, ragged_index, sparse_grad=False)
    gather_ragged_ref = NT([torch.gather(t, 0, idx) for t, idx in zip(nt, ragged_index)], **nt._meta())
    torch.testing.assert_close(gather_ragged.tensor, gather_ragged_ref.tensor)

    index = torch.tensor([2, 0], dtype=torch.long)
    index_select_out = torch.ops.aten.index_select.default(nt, 2, index)
    index_select_ref = NT([torch.index_select(t, 1, index) for t in nt], **nt._meta())
    torch.testing.assert_close(index_select_out.tensor, index_select_ref.tensor)


def test_aten_scatter_handlers_match_per_element():
    nt = NT(
        [
            torch.zeros(2, 3),
            torch.zeros(3, 3),
        ]
    )
    index = NT(
        [
            torch.tensor([[0, 2, 1], [1, 0, 2]], dtype=torch.long),
            torch.tensor([[2, 1, 0], [0, 2, 1], [1, 1, 1]], dtype=torch.long),
        ]
    )
    src = NT(
        [
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]),
        ]
    )

    scatter_src_out = torch.ops.aten.scatter.src(nt, 2, index, src)
    scatter_src_ref = NT([torch.scatter(t, 1, idx, s) for t, idx, s in zip(nt, index, src)], **nt._meta())
    torch.testing.assert_close(scatter_src_out.tensor, scatter_src_ref.tensor)

    scatter_value_out = torch.ops.aten.scatter.value(nt, 2, index, -1.0)
    scatter_value_ref = NT([torch.scatter(t, 1, idx, -1.0) for t, idx in zip(nt, index)], **nt._meta())
    torch.testing.assert_close(scatter_value_out.tensor, scatter_value_ref.tensor)

    scatter_add_out = torch.ops.aten.scatter_add.default(nt, 2, index, src)
    scatter_add_ref = NT([torch.scatter_add(t, 1, idx, s) for t, idx, s in zip(nt, index, src)], **nt._meta())
    torch.testing.assert_close(scatter_add_out.tensor, scatter_add_ref.tensor)

    if hasattr(torch.ops.aten, "scatter_reduce"):
        scatter_reduce_out = torch.ops.aten.scatter_reduce.two(nt, 2, index, src, "sum", include_self=True)
        scatter_reduce_ref = NT(
            [torch.scatter_reduce(t, 1, idx, s, reduce="sum", include_self=True) for t, idx, s in zip(nt, index, src)],
            **nt._meta(),
        )
        torch.testing.assert_close(scatter_reduce_out.tensor, scatter_reduce_ref.tensor)


def test_aten_index_write_handlers_match_per_element():
    nt = NT(
        [
            torch.zeros(2, 4),
            torch.zeros(3, 4),
        ]
    )
    index = torch.tensor([0, 2], dtype=torch.long)
    src = NT(
        [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        ]
    )

    index_add_out = torch.ops.aten.index_add.default(nt, 2, index, src)
    index_add_ref = NT([torch.index_add(t, 1, index, s) for t, s in zip(nt, src)], **nt._meta())
    torch.testing.assert_close(index_add_out.tensor, index_add_ref.tensor)

    index_copy_out = torch.ops.aten.index_copy.default(nt, 2, index, src)
    index_copy_ref = NT([torch.index_copy(t, 1, index, s) for t, s in zip(nt, src)], **nt._meta())
    torch.testing.assert_close(index_copy_out.tensor, index_copy_ref.tensor)

    row_nt = NT(
        [
            torch.zeros(4, 3),
            torch.zeros(5, 3),
        ]
    )
    row_index = torch.tensor([0, 2], dtype=torch.long)
    row_values = NT(
        [
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
        ]
    )

    index_put_out = torch.ops.aten.index_put.default(row_nt, [row_index], row_values, False)
    index_put_ref = NT([torch.index_put(t, (row_index,), v) for t, v in zip(row_nt, row_values)], **row_nt._meta())
    torch.testing.assert_close(index_put_out.tensor, index_put_ref.tensor)

    shared_values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    shared_out = torch.ops.aten.index_put.default(row_nt, [row_index], shared_values, False)
    shared_ref = NT([torch.index_put(t, (row_index,), shared_values) for t in row_nt], **row_nt._meta())
    torch.testing.assert_close(shared_out.tensor, shared_ref.tensor)

    scalar_value = torch.tensor(-3.0)
    scalar_out = torch.ops.aten.index_put.default(row_nt, [row_index], scalar_value, False)
    scalar_ref = NT([torch.index_put(t, (row_index,), scalar_value) for t in row_nt], **row_nt._meta())
    torch.testing.assert_close(scalar_out.tensor, scalar_ref.tensor)

    duplicate_index = torch.tensor([0, 0, 2], dtype=torch.long)
    duplicate_values = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            [4.0, 5.0, 6.0],
        ]
    )
    duplicate_out = torch.ops.aten.index_put.default(row_nt, [duplicate_index], duplicate_values, True)
    duplicate_ref = NT(
        [torch.index_put(t, (duplicate_index,), duplicate_values, accumulate=True) for t in row_nt], **row_nt._meta()
    )
    torch.testing.assert_close(duplicate_out.tensor, duplicate_ref.tensor)

    point_rows = torch.tensor([[0, -1], [1, 2]], dtype=torch.long)
    point_cols = torch.tensor([[1, 0], [2, 1]], dtype=torch.long)
    point_values = NT(
        [
            torch.tensor([[13.0, 17.0], [19.0, 23.0]]),
            torch.tensor([[29.0, 31.0], [37.0, 41.0]]),
        ]
    )
    point_out = torch.ops.aten.index_put.default(row_nt, [point_rows, point_cols], point_values, False)
    point_ref = NT(
        [torch.index_put(t, (point_rows, point_cols), v) for t, v in zip(row_nt, point_values)],
        **row_nt._meta(),
    )
    torch.testing.assert_close(point_out.tensor, point_ref.tensor)


def test_aten_diagonal_and_trace_handlers_match_per_element():
    nt_diagonal = NT(
        [
            torch.randn(2, 3, 4),
            torch.randn(3, 3, 4),
        ]
    )
    diagonal_out = torch.ops.aten.diagonal.default(nt_diagonal, 0, 2, 3)
    diagonal_ref = NT([torch.diagonal(t, offset=0, dim1=1, dim2=2) for t in nt_diagonal], **nt_diagonal._meta())
    torch.testing.assert_close(diagonal_out.tensor, diagonal_ref.tensor)

    nt_trace = NT(
        [
            torch.randn(3, 3),
            torch.randn(4, 4),
        ]
    )
    trace_out = torch.ops.aten.trace.default(nt_trace)
    trace_ref = NT([torch.trace(ti) for ti in nt_trace], **nt_trace._meta())
    torch.testing.assert_close(trace_out.tensor, trace_ref.tensor)


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
        raise AssertionError("per_element_fallback must not be used for covered unary aten fastpaths")

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


def test_aten_shape_and_linalg_fastpaths_work_without_python_meta():
    tensors = [
        torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4),
        torch.arange(3 * 3 * 4, dtype=torch.float32).reshape(3, 3, 4),
    ]
    nt = NT(tensors)
    nt._packed_sizes = None
    nt._element_shapes = None

    flattened = torch.ops.aten.flatten.using_ints(nt, 2, 3)
    permuted = torch.ops.aten.permute.default(nt, [0, 1, 3, 2])
    transposed = torch.ops.aten.transpose.int(nt, 2, 3)
    unsqueezed = torch.ops.aten.unsqueeze.default(nt, 3)
    squeezed = torch.ops.aten.squeeze.dim(unsqueezed, 3)
    unflattened = torch.ops.aten.unflatten.int(flattened, 2, [3, 4])

    ref_flattened = NT([torch.flatten(t, start_dim=1, end_dim=2) for t in tensors], **nt._meta())
    ref_permuted = NT([torch.permute(t, (0, 2, 1)) for t in tensors], **nt._meta())
    ref_transposed = NT([torch.transpose(t, 1, 2) for t in tensors], **nt._meta())
    ref_unsqueezed = NT([torch.unsqueeze(t, dim=2) for t in tensors], **nt._meta())
    ref_unflattened = NT([torch.unflatten(t, 1, [3, 4]) for t in ref_flattened], **nt._meta())

    torch.testing.assert_close(flattened.tensor, ref_flattened.tensor)
    torch.testing.assert_close(permuted.tensor, ref_permuted.tensor)
    torch.testing.assert_close(transposed.tensor, ref_transposed.tensor)
    torch.testing.assert_close(unsqueezed.tensor, ref_unsqueezed.tensor)
    torch.testing.assert_close(squeezed.tensor, nt.tensor)
    torch.testing.assert_close(unflattened.tensor, ref_unflattened.tensor)

    mat_tensors = [torch.randn(2, 3, 3), torch.randn(3, 3, 3)]
    mat_a = NT(mat_tensors)
    mat_a._packed_sizes = None
    mat_a._element_shapes = None

    qr_q, qr_r = torch.ops.aten.linalg_qr.default(mat_a, mode="reduced")
    for q_elem, r_elem, a_elem in zip(qr_q, qr_r, mat_tensors):
        torch.testing.assert_close(torch.matmul(q_elem, r_elem), a_elem, rtol=1e-5, atol=1e-5)

    sym0 = torch.randn(2, 3, 3)
    sym1 = torch.randn(3, 3, 3)
    sym0 = sym0 + sym0.transpose(-1, -2)
    sym1 = sym1 + sym1.transpose(-1, -2)
    sym_tensors = [sym0, sym1]
    nt_sym = NT(sym_tensors)
    nt_sym._packed_sizes = None
    nt_sym._element_shapes = None

    eig_vals, eig_vecs = torch.ops.aten.linalg_eigh.default(nt_sym, UPLO="L")
    det_vals = torch.linalg.det(nt_sym)
    for vals_elem, vecs_elem, det_elem, sym_elem in zip(eig_vals, eig_vecs, det_vals, sym_tensors):
        ref_vals, _ = torch.linalg.eigh(sym_elem, UPLO="L")
        torch.testing.assert_close(vals_elem, ref_vals, rtol=1e-5, atol=1e-5)
        recon = vecs_elem @ torch.diag_embed(vals_elem) @ vecs_elem.transpose(-1, -2)
        torch.testing.assert_close(recon, sym_elem, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(det_elem.squeeze(), torch.det(sym_elem), rtol=1e-5, atol=1e-5)

    svd_u, svd_s, svd_vh = torch.ops.aten.linalg_svd.default(mat_a, full_matrices=True)
    for u_elem, s_elem, vh_elem, a_elem in zip(svd_u, svd_s, svd_vh, mat_tensors):
        ref_s = torch.linalg.svd(a_elem, full_matrices=True).S
        torch.testing.assert_close(s_elem, ref_s, rtol=1e-5, atol=1e-5)
        recon = u_elem @ torch.diag_embed(s_elem) @ vh_elem
        torch.testing.assert_close(recon, a_elem, rtol=1e-5, atol=1e-5)

    low_dim_tensors = [torch.randn(3, 3), torch.randn(5, 3)]
    low_dim_nt = NT(low_dim_tensors)
    low_dim_nt._packed_sizes = None
    low_dim_nt._element_shapes = None

    qr_q, qr_r = torch.ops.aten.linalg_qr.default(low_dim_nt, mode="reduced")
    for q_elem, r_elem, a_elem in zip(qr_q, qr_r, low_dim_tensors):
        torch.testing.assert_close(torch.matmul(q_elem, r_elem), a_elem, rtol=1e-5, atol=1e-5)

    svd_u, svd_s, svd_vh = torch.ops.aten.linalg_svd.default(low_dim_nt, full_matrices=False)
    for u_elem, s_elem, vh_elem, a_elem in zip(svd_u, svd_s, svd_vh, low_dim_tensors):
        ref = torch.linalg.svd(a_elem, full_matrices=False)
        torch.testing.assert_close(s_elem, ref.S, rtol=1e-5, atol=1e-5)
        recon = u_elem @ torch.diag_embed(s_elem) @ vh_elem
        torch.testing.assert_close(recon, a_elem, rtol=1e-5, atol=1e-5)


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
    assert out._physical_shape.shape == torch.Size([0, 1])


def test_offsets_match_fake_tensors_not_false_positive():
    fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
    FakeTensorMode = fake_tensor_mod.FakeTensorMode

    with FakeTensorMode():
        offsets_a = torch.tensor([0, 2, 5], dtype=torch.long)
        offsets_b = torch.tensor([0, 2, 5], dtype=torch.long)
        offsets_c = torch.tensor([0, 2], dtype=torch.long)
        assert aten_functions._offsets_match(offsets_a, offsets_a)
        assert aten_functions._offsets_match(offsets_a, offsets_b)
        assert not aten_functions._offsets_match(offsets_a, offsets_c)


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
        raise AssertionError("per_element_fallback must not be used for covered unary aten fastpaths")

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
    ref_dropout_train = _packed_result(nt, torch.dropout(nt._values, p=0.2, train=True))
    torch.manual_seed(5678)
    ref_bernoulli = _packed_result(nt_prob, torch.bernoulli(nt_prob._values))
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

    original_map = torch_functions._map_storage_serial
    original_map_pair = torch_functions._map_storage_pair

    def _fail_map(*_args, **_kwargs):
        raise AssertionError("_map_storage_serial must not be used for migrated torch cumulative wrappers")

    def _fail_map_pair(*_args, **_kwargs):
        raise AssertionError("_map_storage_pair must not be used for migrated torch cumulative wrappers")

    torch_functions._map_storage_serial = _fail_map
    torch_functions._map_storage_pair = _fail_map_pair
    try:
        topk_vals, topk_idxs = torch.topk(nt, 2, dim=1, largest=True, sorted=True)
        cumsum_out = torch.cumsum(nt, dim=1)
        cumprod_out = torch.cumprod(nt, dim=1)
        logcumsumexp_out = torch.logcumsumexp(nt, dim=1)
        cummax_vals, cummax_idxs = torch.cummax(nt, dim=1)
        cummin_vals, cummin_idxs = torch.cummin(nt, dim=1)
    finally:
        torch_functions._map_storage_serial = original_map
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

    original_map = torch_functions._map_storage_serial
    original_map_pair = torch_functions._map_storage_pair

    def _fail_map(*_args, **_kwargs):
        raise AssertionError("_map_storage_serial must not be used for migrated torch shape wrappers")

    def _fail_map_pair(*_args, **_kwargs):
        raise AssertionError("_map_storage_pair must not be used for migrated torch shape wrappers")

    torch_functions._map_storage_serial = _fail_map
    torch_functions._map_storage_pair = _fail_map_pair
    try:
        flattened = torch.flatten(nt, start_dim=2, end_dim=3)
        permuted = torch.permute(nt, (0, 1, 3, 2))
        transposed = torch.transpose(nt, 2, 3)
        unsqueezed = torch.unsqueeze(nt, 3)
        squeezed = torch.squeeze(unsqueezed, 3)
        unflattened = torch.unflatten(flattened, dim=2, sizes=(3, 4))
    finally:
        torch_functions._map_storage_serial = original_map
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
    ref_dropout = _packed_result(nt, torch.dropout(nt._values, p=0.5, train=True))
    torch.manual_seed(5678)
    ref_bernoulli = _packed_result(nt_prob, torch.bernoulli(nt_prob._values))
    torch.testing.assert_close(dropout_out.tensor, ref_dropout.tensor)
    torch.testing.assert_close(bernoulli_out.tensor, ref_bernoulli.tensor)


def test_torch_masked_fill_and_alpha_dropout_wrappers_no_storage_mapping():
    nt = NT(
        [
            torch.tensor([[1.0, -2.0], [3.0, -4.0]]),
            torch.tensor([[5.0, -6.0], [7.0, -8.0], [9.0, -10.0]]),
        ]
    )
    mask = nt > 0

    original_map = torch_functions._map_storage_serial
    original_map_pair = torch_functions._map_storage_pair

    def _fail_map(*_args, **_kwargs):
        raise AssertionError(
            "_map_storage_serial must not be used for migrated torch masked_fill/alpha_dropout wrappers"
        )

    def _fail_map_pair(*_args, **_kwargs):
        raise AssertionError("_map_storage_pair must not be used for migrated torch masked_fill/alpha_dropout wrappers")

    torch_functions._map_storage_serial = _fail_map
    torch_functions._map_storage_pair = _fail_map_pair
    try:
        masked_fill_out = torch.masked_fill(nt, mask, 0.0)
        masked_fill_dense_mask_out = torch.masked_fill(nt, mask.tensor, 0.0)
        alpha_dropout_out = torch.alpha_dropout(nt, p=0.2, train=True)
        feature_alpha_dropout_out = torch.feature_alpha_dropout(nt, p=0.2, train=True)
    finally:
        torch_functions._map_storage_serial = original_map
        torch_functions._map_storage_pair = original_map_pair

    masked_fill_ref = NT([torch.masked_fill(t, m, 0.0) for t, m in zip(nt, mask)], **nt._meta())
    torch.testing.assert_close(masked_fill_out.tensor, masked_fill_ref.tensor)
    torch.testing.assert_close(masked_fill_dense_mask_out.tensor, masked_fill_ref.tensor)
    assert isinstance(alpha_dropout_out, NT)
    assert alpha_dropout_out.shape == nt.shape
    assert isinstance(feature_alpha_dropout_out, NT)
    assert feature_alpha_dropout_out.shape == nt.shape


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

    original_map = torch_functions._map_storage_serial
    original_map_pair = torch_functions._map_storage_pair

    def _fail_map(*_args, **_kwargs):
        raise AssertionError("_map_storage_serial must not be used for migrated torch normalization wrappers")

    def _fail_map_pair(*_args, **_kwargs):
        raise AssertionError("_map_storage_pair must not be used for migrated torch normalization wrappers")

    torch_functions._map_storage_serial = _fail_map
    torch_functions._map_storage_pair = _fail_map_pair
    try:
        layer_norm_out = torch.layer_norm(nt, (2,))
        linalg_norm_out = torch.linalg.norm(nt)
        linalg_norm_dim_out = torch.linalg.norm(nt, dim=1)
        if hasattr(torch, "rms_norm"):
            rms_norm_out = torch.rms_norm(nt, (2,))
    finally:
        torch_functions._map_storage_serial = original_map
        torch_functions._map_storage_pair = original_map_pair

    ref_layer_norm = NT([torch.layer_norm(t, (2,)) for t in nt], **nt._meta())
    torch.testing.assert_close(layer_norm_out.tensor, ref_layer_norm.tensor)
    ref_linalg_norm = NT([torch.linalg.norm(t) for t in nt], **nt._meta())
    ref_linalg_norm_dim = NT([torch.linalg.norm(t, dim=0) for t in nt], **nt._meta())
    torch.testing.assert_close(linalg_norm_out.tensor, ref_linalg_norm.tensor)
    torch.testing.assert_close(linalg_norm_dim_out.tensor, ref_linalg_norm_dim.tensor)
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
