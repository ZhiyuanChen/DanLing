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

r"""Tests for ``danling.tensors.aten_functions`` — aten-level dispatch handlers."""

import pytest
import torch

from danling.tensors import NestedTensor
from danling.tensors import ops as nt_ops
from tests.tensors.utils import assert_close, nested_rand, ragged_shapes

NT = NestedTensor


def _compile_inductor_fullgraph(fn):
    return torch.compile(fn, backend="inductor", fullgraph=True)


_1D_RAGGED = pytest.param(
    lambda: NT([torch.tensor([3.0, 1.0]), torch.tensor([4.0, 2.0, 5.0])]),
    id="1d_ragged",
)
_2D_RAGGED = pytest.param(
    lambda: NT(
        [
            torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
            torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
        ]
    ),
    id="2d_ragged",
)


class TestCompile:

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize("nt_factory", [_2D_RAGGED])
    @pytest.mark.parametrize(
        "aten_fn,ref_fn",
        [
            pytest.param(
                lambda x: torch.ops.aten.sort.default(x, 2, False),
                lambda t: torch.sort(t, dim=1, descending=False).values,
                id="sort",
            ),
            pytest.param(
                lambda x: torch.ops.aten.argsort.default(x, 2, False),
                lambda t: torch.argsort(t, dim=1, descending=False),
                id="argsort",
            ),
            pytest.param(
                lambda x: torch.ops.aten.topk.default(x, 2, 2, True, True),
                lambda t: torch.topk(t, 2, dim=1, largest=True, sorted=True).values,
                id="topk",
            ),
            pytest.param(
                lambda x: torch.ops.aten.cumsum.default(x, 2),
                lambda t: torch.cumsum(t, dim=1),
                id="cumsum",
            ),
            pytest.param(
                lambda x: torch.ops.aten.cumprod.default(x, 2),
                lambda t: torch.cumprod(t, dim=1),
                id="cumprod",
            ),
            pytest.param(
                lambda x: torch.ops.aten.logcumsumexp.default(x, 2),
                lambda t: torch.logcumsumexp(t, dim=1),
                id="logcumsumexp",
            ),
            pytest.param(
                lambda x: torch.ops.aten.cummax.default(x, 2),
                lambda t: torch.cummax(t, dim=1).values,
                id="cummax",
            ),
            pytest.param(
                lambda x: torch.ops.aten.cummin.default(x, 2),
                lambda t: torch.cummin(t, dim=1).values,
                id="cummin",
            ),
            pytest.param(
                lambda x: torch.ops.aten.flip.default(x, [2]),
                lambda t: torch.flip(t, dims=[1]),
                id="flip",
            ),
            pytest.param(
                lambda x: torch.ops.aten._softmax.default(x, 2, False),
                lambda t: torch.softmax(t, dim=1),
                id="softmax",
            ),
            pytest.param(
                lambda x: torch.ops.aten._log_softmax.default(x, 2, False),
                lambda t: torch.log_softmax(t, dim=1),
                id="log_softmax",
            ),
        ],
    )
    def test_compile_ragged_op(self, nt_factory, aten_fn, ref_fn):
        nt = nt_factory()
        compiled = _compile_inductor_fullgraph(aten_fn)
        result = compiled(nt)
        if isinstance(result, tuple):
            result = result[0]
        reference = NT([ref_fn(t) for t in nt], **nt._meta())
        assert isinstance(result, NestedTensor)
        assert result._has_same_layout(reference)
        assert_close(result, reference)

    @pytest.mark.parametrize(
        "aten_fn",
        [
            pytest.param(lambda x: torch.ops.aten.sort.default(x, 1, False), id="sort"),
            pytest.param(lambda x: torch.ops.aten.argsort.default(x, 1, False), id="argsort"),
            pytest.param(lambda x: torch.ops.aten.topk.default(x, 2, 1, True, True), id="topk"),
            pytest.param(lambda x: torch.ops.aten.cumsum.default(x, 1), id="cumsum"),
            pytest.param(lambda x: torch.ops.aten.cummax.default(x, 1), id="cummax"),
            pytest.param(lambda x: torch.ops.aten.flip.default(x, [1]), id="flip"),
        ],
    )
    def test_ragged_dim_is_rejected_when_compile_state_is_active(self, monkeypatch, aten_fn):
        nt = NT(
            [
                torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
                torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
            ]
        )
        monkeypatch.setattr(nt_ops, "_is_compiling", lambda: True)
        with pytest.raises(NotImplementedError, match="aten handler is marked eager-only"):
            aten_fn(nt)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize(
        "aten_fn,ref_fn",
        [
            pytest.param(
                lambda x: torch.ops.aten.square.default(x),
                torch.square,
                id="square",
            ),
            pytest.param(
                lambda x: torch.ops.aten.digamma.default(x),
                torch.digamma,
                id="digamma",
            ),
            pytest.param(
                lambda x: torch.ops.aten.lgamma.default(x),
                torch.lgamma,
                id="lgamma",
            ),
        ],
    )
    def test_compile_unary_op(self, aten_fn, ref_fn):
        nt = NT(
            [
                torch.tensor([[1.1, 2.2], [3.3, 4.4]]),
                torch.tensor([[5.5, 6.6], [7.7, 8.8], [9.9, 10.1]]),
            ]
        )
        compiled = _compile_inductor_fullgraph(aten_fn)
        result = compiled(nt)
        reference = NT([ref_fn(t) for t in nt], **nt._meta())
        assert isinstance(result, NestedTensor)
        assert result._has_same_layout(reference)
        assert_close(result, reference)


class TestElementwiseOps:

    def test_binary_dense_same_shape(self):
        nt = NT([torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])])
        dense = nt.tensor + 0.5
        nested_dense = nt.nested_like(dense, strict=False)

        pow_out = torch.ops.aten.pow.Tensor_Tensor(nt, dense)
        pow_rev_out = torch.ops.aten.pow.Tensor_Tensor(dense, nt)
        and_out = torch.ops.aten.bitwise_and.Tensor(nt.int(), dense.int())

        pow_ref = NT([torch.pow(a, b) for a, b in zip(nt, nested_dense)], **nt._meta())
        pow_rev_ref = NT([torch.pow(b, a) for a, b in zip(nt, nested_dense)], **nt._meta())
        and_ref = NT([torch.bitwise_and(a.int(), b.int()) for a, b in zip(nt, nested_dense)], **nt._meta())
        assert_close(pow_out, pow_ref)
        assert_close(pow_rev_out, pow_rev_ref)
        assert_close(and_out, and_ref)

    @pytest.mark.parametrize("op", [torch.mul, torch.add])
    def test_binary_op_roundtrip(self, op):
        nt = NT([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])])
        scalar = 2.0
        result = op(nt, scalar)
        assert isinstance(result, NestedTensor)
        for r, t in zip(result, nt):
            assert_close(r, op(t, scalar))

    @pytest.mark.parametrize("op", [torch.abs, torch.neg, torch.exp, torch.sign])
    def test_unary_op_roundtrip(self, op):
        nt = NT([torch.tensor([1.0, -2.0, 3.0]), torch.tensor([-4.0, 5.0])])
        result = op(nt)
        assert isinstance(result, NestedTensor)
        for r, t in zip(result, nt):
            assert_close(r, op(t))


class TestIndexingOps:

    @pytest.mark.parametrize("seed", [5, 17, 31])
    def test_index_put_matches_dense(self, seed, device):
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
        assert_close(row_out, row_ref)

        dup_rows = rows.clone()
        dup_rows[0] = dup_rows[-1]
        dup_values = torch.randn(row_count, 4, device=device, dtype=dtype)
        dup_out = torch.ops.aten.index_put.default(base, [dup_rows], dup_values, True)
        dup_ref = NT([torch.index_put(t, (dup_rows,), dup_values, accumulate=True) for t in base], **base._meta())
        assert_close(dup_out, dup_ref)

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
        assert_close(point_out, point_ref)

    def test_index_write(self):
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
        assert_close(index_add_out, index_add_ref)

        index_copy_out = torch.ops.aten.index_copy.default(nt, 2, index, src)
        index_copy_ref = NT([torch.index_copy(t, 1, index, s) for t, s in zip(nt, src)], **nt._meta())
        assert_close(index_copy_out, index_copy_ref)

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
        assert_close(index_put_out, index_put_ref)

        shared_values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        shared_out = torch.ops.aten.index_put.default(row_nt, [row_index], shared_values, False)
        shared_ref = NT([torch.index_put(t, (row_index,), shared_values) for t in row_nt], **row_nt._meta())
        assert_close(shared_out, shared_ref)

        scalar_value = torch.tensor(-3.0)
        scalar_out = torch.ops.aten.index_put.default(row_nt, [row_index], scalar_value, False)
        scalar_ref = NT([torch.index_put(t, (row_index,), scalar_value) for t in row_nt], **row_nt._meta())
        assert_close(scalar_out, scalar_ref)

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
            [torch.index_put(t, (duplicate_index,), duplicate_values, accumulate=True) for t in row_nt],
            **row_nt._meta(),
        )
        assert_close(duplicate_out, duplicate_ref)

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
        assert_close(point_out, point_ref)

    def test_indexing(self):
        nt = NT(
            [
                torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            ]
        )

        batch_index = torch.tensor([1, 0, 1], dtype=torch.long)
        batch_out = torch.ops.aten.index_select.default(nt, 0, batch_index)
        batch_ref = NT([nt[1], nt[0], nt[1]], **nt._meta())
        assert_close(batch_out, batch_ref)

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
        assert_close(gather_out, gather_ref)

        ragged_index = NT(
            [
                torch.tensor([[1, 0, 1]], dtype=torch.long),
                torch.tensor([[2, 1, 0], [0, 2, 1]], dtype=torch.long),
            ]
        )
        gather_ragged = torch.ops.aten.gather.default(nt, 1, ragged_index, sparse_grad=False)
        gather_ragged_ref = NT([torch.gather(t, 0, idx) for t, idx in zip(nt, ragged_index)], **nt._meta())
        assert_close(gather_ragged, gather_ragged_ref)

        index = torch.tensor([2, 0], dtype=torch.long)
        index_select_out = torch.ops.aten.index_select.default(nt, 2, index)
        index_select_ref = NT([torch.index_select(t, 1, index) for t in nt], **nt._meta())
        assert_close(index_select_out, index_select_ref)

    def test_scatter(self):
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
        assert_close(scatter_src_out, scatter_src_ref)

        scatter_value_out = torch.ops.aten.scatter.value(nt, 2, index, -1.0)
        scatter_value_ref = NT([torch.scatter(t, 1, idx, -1.0) for t, idx in zip(nt, index)], **nt._meta())
        assert_close(scatter_value_out, scatter_value_ref)

        scatter_add_out = torch.ops.aten.scatter_add.default(nt, 2, index, src)
        scatter_add_ref = NT([torch.scatter_add(t, 1, idx, s) for t, idx, s in zip(nt, index, src)], **nt._meta())
        assert_close(scatter_add_out, scatter_add_ref)

        if hasattr(torch.ops.aten, "scatter_reduce"):
            scatter_reduce_out = torch.ops.aten.scatter_reduce.two(nt, 2, index, src, "sum", include_self=True)
            scatter_reduce_ref = NT(
                [
                    torch.scatter_reduce(t, 1, idx, s, reduce="sum", include_self=True)
                    for t, idx, s in zip(nt, index, src)
                ],
                **nt._meta(),
            )
            assert_close(scatter_reduce_out, scatter_reduce_ref)

    def test_searchsorted(self):
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

        nt_nt = torch.ops.aten.searchsorted.Tensor(sorted_nt, values_nt)
        tensor_nt = torch.ops.aten.searchsorted.Tensor(boundaries, values_nt)
        ref_nt_nt = NT([torch.searchsorted(s, v) for s, v in zip(sorted_nt, values_nt)], **values_nt._meta())
        ref_tensor_nt = NT([torch.searchsorted(boundaries, v) for v in values_nt], **values_nt._meta())

        assert_close(nt_nt, ref_nt_nt)
        assert_close(tensor_nt, ref_tensor_nt)

    def test_searchsorted_rejects_nested_sorter_without_sorted_sequence(self):
        boundaries = torch.tensor([1.0, 3.0, 5.0])
        values = torch.tensor([2.0, 4.0])
        sorter = NT([torch.tensor([0, 1, 2], dtype=torch.long)])
        with pytest.raises(TypeError, match="NestedTensor sorter requires sorted_sequence"):
            torch.ops.aten.searchsorted.Tensor(boundaries, values, sorter=sorter)


class TestLinearAlgebraOps:

    def test_diagonal_and_trace(self):
        nt_diagonal = NT(
            [
                torch.randn(2, 3, 4),
                torch.randn(3, 3, 4),
            ]
        )
        diagonal_out = torch.ops.aten.diagonal.default(nt_diagonal, 0, 2, 3)
        diagonal_ref = NT([torch.diagonal(t, offset=0, dim1=1, dim2=2) for t in nt_diagonal], **nt_diagonal._meta())
        assert_close(diagonal_out, diagonal_ref)

        nt_trace = NT(
            [
                torch.randn(3, 3),
                torch.randn(4, 4),
            ]
        )
        trace_out = torch.ops.aten.trace.default(nt_trace)
        trace_ref = NT([torch.trace(ti) for ti in nt_trace], **nt_trace._meta())
        assert_close(trace_out, trace_ref)

    def test_linalg_qr_eigh_svd_solve(self):
        qr_a0 = torch.randn(3, 3)
        qr_a1 = torch.randn(5, 3)
        nt_qr = NT([qr_a0, qr_a1])

        qr_q, qr_r = torch.ops.aten.linalg_qr.default(nt_qr, mode="reduced")
        for q_elem, r_elem, a_elem in zip(qr_q, qr_r, nt_qr):
            assert_close(q_elem @ r_elem, a_elem, rtol=1e-5, atol=1e-5)

        sym0 = torch.randn(3, 3)
        sym1 = torch.randn(4, 4)
        sym0 = sym0 + sym0.transpose(-1, -2)
        sym1 = sym1 + sym1.transpose(-1, -2)
        nt_sym = NT([sym0, sym1])
        eig_vals, eig_vecs = torch.ops.aten.linalg_eigh.default(nt_sym, UPLO="L")
        for vals_elem, vecs_elem, sym_elem in zip(eig_vals, eig_vecs, nt_sym):
            ref_vals, _ = torch.linalg.eigh(sym_elem, UPLO="L")
            assert_close(vals_elem, ref_vals, rtol=1e-5, atol=1e-5)
            recon = vecs_elem @ torch.diag_embed(vals_elem) @ vecs_elem.transpose(-1, -2)
            assert_close(recon, sym_elem, rtol=1e-5, atol=1e-5)

        svd_a0 = torch.randn(3, 2)
        svd_a1 = torch.randn(4, 2)
        nt_svd = NT([svd_a0, svd_a1])
        svd_u, svd_s, svd_vh = torch.ops.aten.linalg_svd.default(nt_svd, full_matrices=False)
        for u_elem, s_elem, vh_elem, a_elem in zip(svd_u, svd_s, svd_vh, nt_svd):
            ref_s = torch.linalg.svd(a_elem, full_matrices=False).S
            assert_close(s_elem, ref_s, rtol=1e-5, atol=1e-5)
            recon = u_elem @ torch.diag_embed(s_elem) @ vh_elem
            assert_close(recon, a_elem, rtol=1e-5, atol=1e-5)

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
            assert_close(out_elem, ref_elem, rtol=1e-5, atol=1e-5)

    def test_linalg_qr_and_eigh_batched_matrices(self):
        """QR and eigendecomposition on nested tensors whose elements are batches of matrices."""
        a0 = torch.randn(2, 3, 3)
        a1 = torch.randn(3, 3, 3)
        nt_a = NT([a0, a1])

        qr_q, qr_r = torch.ops.aten.linalg_qr.default(nt_a, mode="reduced")
        for q_elem, r_elem, a_elem in zip(qr_q, qr_r, nt_a):
            assert_close(q_elem @ r_elem, a_elem, rtol=1e-5, atol=1e-5)

        sym_a0 = a0 + a0.transpose(-1, -2)
        sym_a1 = a1 + a1.transpose(-1, -2)
        nt_sym = NT([sym_a0, sym_a1])
        eigh_vals, eigh_vecs = torch.ops.aten.linalg_eigh.default(nt_sym, UPLO="L")
        for vals_elem, vecs_elem, sym_elem in zip(eigh_vals, eigh_vecs, nt_sym):
            ref_vals, _ = torch.linalg.eigh(sym_elem, UPLO="L")
            assert_close(vals_elem, ref_vals, rtol=1e-5, atol=1e-5)
            recon = vecs_elem @ torch.diag_embed(vals_elem) @ vecs_elem.transpose(-1, -2)
            assert_close(recon, sym_elem, rtol=1e-5, atol=1e-5)

    def test_linalg_vector_norm(self):
        base = NT(
            [
                torch.tensor([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]]),
                torch.tensor([[7.0, 8.0], [1.0, 0.0], [9.0, 6.0], [2.0, 3.0], [5.0, 4.0]]),
            ]
        )

        global_out = torch.ops.aten.linalg_vector_norm.default(base, 2, None, False)
        global_ref = NT([torch.linalg.vector_norm(t, ord=2) for t in base], **base._meta())
        assert_close(global_out, global_ref)

        dim_out = torch.ops.aten.linalg_vector_norm.default(base, 2, [1], False)
        dim_ref = NT([torch.linalg.vector_norm(t, ord=2, dim=0) for t in base], **base._meta())
        assert_close(dim_out, dim_ref)


class TestMaskingOps:

    def test_masked_fill(self):
        base = NT([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
        mask = NT([torch.tensor([True, False, True]), torch.tensor([False, True])])

        scalar_out = torch.ops.aten.masked_fill.Scalar(base, mask, 0.0)
        scalar_ref = NT([torch.masked_fill(t, m, 0.0) for t, m in zip(base, mask)], **base._meta())
        assert_close(scalar_out, scalar_ref)

        value = torch.tensor(-1.5, dtype=base.dtype, device=base.device)
        tensor_out = torch.ops.aten.masked_fill.Tensor(base, mask, value)
        tensor_ref = NT([torch.masked_fill(t, m, value) for t, m in zip(base, mask)], **base._meta())
        assert_close(tensor_out, tensor_ref)

    def test_masked_scatter_nested_source(self):
        base = NT([torch.zeros(3), torch.zeros(2)])
        mask = NT([torch.tensor([True, False, True]), torch.tensor([False, True])])
        source = NT([torch.tensor([1.0, 2.0]), torch.tensor([3.0])])

        output = torch.ops.aten.masked_scatter.default(base, mask, source)
        reference = NT([torch.masked_scatter(t, m, s) for t, m, s in zip(base, mask, source)], **base._meta())

        assert_close(output, reference)

    def test_masked_select(self):
        base = NT([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
        mask = NT([torch.tensor([True, False, True]), torch.tensor([False, True])])

        output = torch.ops.aten.masked_select.default(base, mask)
        reference = NT([torch.masked_select(t, m) for t, m in zip(base, mask)], **base._meta())
        assert_close(output, reference)

    def test_nonzero(self):
        base = NT([torch.tensor([[0.0, 1.0], [2.0, 0.0]]), torch.tensor([[3.0, 0.0]])])

        output = torch.ops.aten.nonzero.default(base)
        reference = NT([torch.nonzero(t, as_tuple=False) for t in base], **base._meta())

        assert_close(output, reference)
        assert output.dtype == torch.long


class TestReductionOps:

    @pytest.mark.parametrize(
        "aten_op,torch_op",
        [
            pytest.param(torch.ops.aten.max, torch.max, id="max"),
            pytest.param(torch.ops.aten.min, torch.min, id="min"),
        ],
    )
    def test_max_min(self, aten_op, torch_op):
        nt = NT(
            [
                torch.tensor([[1.0, 5.0, 3.0], [2.0, 0.0, 4.0]]),
                torch.tensor([[9.0, 1.0, 2.0], [3.0, 7.0, 6.0], [4.0, 0.0, 8.0]]),
            ]
        )
        # Global reduction
        assert_close(aten_op.default(nt), torch_op(nt._values))

        # Static dim
        vals_last, idxs_last = aten_op.dim(nt, 2, False)
        vals_last_ref = NT([torch_op(t, dim=1).values for t in nt], **nt._meta())
        idxs_last_ref = NT([torch_op(t, dim=1).indices for t in nt], **nt._meta())
        assert_close(vals_last, vals_last_ref)
        assert_close(idxs_last, idxs_last_ref)

        # Ragged dim
        vals_ragged, idxs_ragged = aten_op.dim(nt, 1, False)
        vals_ragged_ref = torch.stack([torch_op(t, dim=0).values for t in nt])
        idxs_ragged_ref = torch.stack([torch_op(t, dim=0).indices for t in nt])
        assert_close(vals_ragged, vals_ragged_ref)
        assert_close(idxs_ragged, idxs_ragged_ref)

    @pytest.mark.parametrize(
        "aten_op,torch_op,global_ref",
        [
            pytest.param(torch.ops.aten.argmax, torch.argmax, torch.tensor([1, 0], dtype=torch.long), id="argmax"),
            pytest.param(torch.ops.aten.argmin, torch.argmin, torch.tensor([4, 7], dtype=torch.long), id="argmin"),
        ],
    )
    def test_argmax_argmin(self, aten_op, torch_op, global_ref):
        nt = NT(
            [
                torch.tensor([[1.0, 5.0, 3.0], [2.0, 0.0, 4.0]]),
                torch.tensor([[9.0, 1.0, 2.0], [3.0, 7.0, 6.0], [4.0, 0.0, 8.0]]),
            ]
        )
        # Global
        assert_close(aten_op.default(nt), global_ref)

        # Static dim
        out_last = aten_op.default(nt, 2, False)
        ref_last = NT([torch_op(t, dim=1) for t in nt], **nt._meta())
        assert_close(out_last, ref_last)

        # Ragged dim
        out_ragged = aten_op.default(nt, 1, False)
        ref_ragged = torch.stack([torch_op(t, dim=0) for t in nt])
        assert_close(out_ragged, ref_ragged)

    def test_order_stat(self):
        nt = NT(
            [
                torch.tensor([[3.0, 1.0, 2.0], [5.0, 4.0, 6.0]]),
                torch.tensor([[9.0, 7.0, 8.0], [0.0, 2.0, 1.0], [6.0, 5.0, 4.0]]),
            ]
        )

        kth_vals, kth_idxs = torch.ops.aten.kthvalue.default(nt, 2, 2, False)
        kth_ref = tuple(NT([torch.kthvalue(t, 2, dim=1)[i] for t in nt], **nt._meta()) for i in range(2))
        assert_close(kth_vals, kth_ref[0])
        assert_close(kth_idxs, kth_ref[1])

        median_global = torch.ops.aten.median.default(nt)
        assert_close(median_global, torch.median(nt._values))

        median_vals, median_idxs = torch.ops.aten.median.dim(nt, 2, True)
        median_ref = tuple(NT([torch.median(t, dim=1, keepdim=True)[i] for t in nt], **nt._meta()) for i in range(2))
        assert_close(median_vals, median_ref[0])
        assert_close(median_idxs, median_ref[1])

        mode_vals, mode_idxs = torch.ops.aten.mode.default(nt, 2, False)
        mode_ref = tuple(NT([torch.mode(t, dim=1)[i] for t in nt], **nt._meta()) for i in range(2))
        assert_close(mode_vals, mode_ref[0])
        assert_close(mode_idxs, mode_ref[1])

        nan_nt = NT(
            [
                torch.tensor([[1.0, float("nan"), 3.0], [2.0, 4.0, float("nan")]]),
                torch.tensor([[float("nan"), 7.0, 8.0], [0.0, float("nan"), 1.0], [6.0, 5.0, 4.0]]),
            ]
        )
        nanmedian_global = torch.ops.aten.nanmedian.default(nan_nt)
        assert_close(nanmedian_global, torch.nanmedian(nan_nt._values), equal_nan=True)

        nanmedian_vals, nanmedian_idxs = torch.ops.aten.nanmedian.dim(nan_nt, 2, False)
        nanmedian_ref = tuple(NT([torch.nanmedian(t, dim=1)[i] for t in nan_nt], **nan_nt._meta()) for i in range(2))
        assert_close(nanmedian_vals, nanmedian_ref[0], equal_nan=True)
        assert_close(nanmedian_idxs, nanmedian_ref[1])

    def test_reduce_multi_dim_empty_batch(self):
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


class TestSortingOps:

    def test_ragged_topk_k_exceeds_min_length(self):
        nt = NT(
            [
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
            ]
        )
        with pytest.raises(ValueError, match="k <= min segment length"):
            torch.ops.aten.topk.default(nt, 3, 1, True, True)

    @pytest.mark.parametrize(
        "aten_fn,ref_fn",
        [
            pytest.param(
                lambda x: torch.ops.aten.sort.default(x, 2, False),
                lambda t: torch.sort(t, dim=1, descending=False).values,
                id="sort",
            ),
            pytest.param(
                lambda x: torch.ops.aten.argsort.default(x, 2, False),
                lambda t: torch.argsort(t, dim=1, descending=False),
                id="argsort",
            ),
            pytest.param(
                lambda x: torch.ops.aten.topk.default(x, 2, 2, True, True),
                lambda t: torch.topk(t, 2, dim=1, largest=True, sorted=True).values,
                id="topk",
            ),
            pytest.param(
                lambda x: torch.ops.aten.cumsum.default(x, 2),
                lambda t: torch.cumsum(t, dim=1),
                id="cumsum",
            ),
            pytest.param(
                lambda x: torch.ops.aten.cumprod.default(x, 2),
                lambda t: torch.cumprod(t, dim=1),
                id="cumprod",
            ),
            pytest.param(
                lambda x: torch.ops.aten.logcumsumexp.default(x, 2),
                lambda t: torch.logcumsumexp(t, dim=1),
                id="logcumsumexp",
            ),
            pytest.param(
                lambda x: torch.ops.aten.cummax.default(x, 2),
                lambda t: torch.cummax(t, dim=1).values,
                id="cummax",
            ),
            pytest.param(
                lambda x: torch.ops.aten.cummin.default(x, 2),
                lambda t: torch.cummin(t, dim=1).values,
                id="cummin",
            ),
            pytest.param(
                lambda x: torch.ops.aten.flip.default(x, [2]),
                lambda t: torch.flip(t, dims=[1]),
                id="flip",
            ),
        ],
    )
    def test_sorting_and_cumulative_op(self, aten_fn, ref_fn):
        nt = NT(
            [
                torch.tensor([[3.0, 1.0, 2.0], [4.0, 0.0, 5.0]]),
                torch.tensor([[2.0, 5.0, 1.0], [7.0, 6.0, 8.0], [9.0, 4.0, 3.0]]),
            ]
        )
        result = aten_fn(nt)
        if isinstance(result, tuple):
            result = result[0]
        reference = NT([ref_fn(t) for t in nt], **nt._meta())
        assert_close(result, reference)

    def test_cumsum_ragged_dim(self):
        nt = NT(
            [
                torch.tensor([[3.0, 1.0, 2.0], [4.0, 0.0, 5.0]]),
                torch.tensor([[2.0, 5.0, 1.0], [7.0, 6.0, 8.0], [9.0, 4.0, 3.0]]),
            ]
        )
        cumsum_ragged = torch.ops.aten.cumsum.default(nt, 1)
        for got, t in zip(cumsum_ragged, nt):
            assert_close(got, torch.cumsum(t, dim=0))


class TestTernaryOps:

    def test_ternary_and_take(self):
        nt = NT(
            [
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([4.0, 5.0]),
            ]
        )

        dense = nt.tensor
        where_out = torch.ops.aten.where.ScalarOther(dense > 2, nt, 0.0)
        where_ref = NT([torch.where(t > 2, t, 0.0) for t in nt], **nt._meta())
        assert_close(where_out, where_ref)

        where_scalar_out = torch.ops.aten.where.Scalar(nt > 2, 1.0, 0.0)
        where_scalar_ref = NT([torch.where(t > 2, 1.0, 0.0) for t in nt], **nt._meta())
        assert_close(where_scalar_out, where_scalar_ref)

        addcmul_out = torch.ops.aten.addcmul.default(dense, nt, nt, value=2)
        addcmul_ref = NT([torch.addcmul(t, t, t, value=2) for t in nt], **nt._meta())
        assert_close(addcmul_out, addcmul_ref)

        addcdiv_out = torch.ops.aten.addcdiv.default(dense, nt, nt + 1, value=2)
        addcdiv_ref = NT([torch.addcdiv(t, t, t + 1, value=2) for t in nt], **nt._meta())
        assert_close(addcdiv_out, addcdiv_ref)

        take_index = torch.tensor([0, 2, 4], dtype=torch.long)
        take_out = torch.ops.aten.take.default(nt, take_index)
        take_ref = torch.take(torch.cat([t.reshape(-1) for t in nt]), take_index)
        assert_close(take_out, take_ref)

    def test_ternary_accepts_broadcast(self):
        nt = NT(
            [
                torch.ones(2, 3),
                torch.ones(4, 3),
            ]
        )
        bias = torch.arange(3.0)

        output = torch.ops.aten.addcmul.default(nt, bias, bias, value=2)
        reference = NT([torch.addcmul(t, bias, bias, value=2) for t in nt], **nt._meta())

        assert_close(output, reference)

    def test_ternary_rejects_packed_only_broadcast(self):
        nt = NT(
            [
                torch.ones(2, 3),
                torch.ones(4, 3),
            ]
        )
        bad = torch.arange(18.0).reshape(6, 3)

        with pytest.raises(RuntimeError, match="size of tensor"):
            torch.ops.aten.addcmul.default(nt, bad, bad, value=2)

    @pytest.mark.parametrize("seed", [3, 11, 29])
    def test_ternary_matches_dense(self, seed, device):
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
        assert_close(addcmul_out, addcmul_ref)

        addcdiv_out = torch.ops.aten.addcdiv.default(nt, scale, denom, value=-0.5)
        addcdiv_ref = NT([torch.addcdiv(t, scale, denom, value=-0.5) for t in nt], **nt._meta())
        assert_close(addcdiv_out, addcdiv_ref)

        where_out = torch.ops.aten.where.self(condition, nt, bias)
        where_ref = NT([torch.where(condition, t, bias) for t in nt], **nt._meta())
        assert_close(where_out, where_ref)
