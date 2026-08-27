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

import copy
import io
import json
import os
import pickle
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from packaging.version import Version

from danling.tensors import NestedTensor, nested_execution_guard
from tests.tensors.utils import assert_close

NT = NestedTensor
TORCH_VERSION = Version(torch.__version__.split("+")[0])
random.seed(1016)


# ---------------------------------------------------------------------------
# Construction & Validation
# ---------------------------------------------------------------------------


class TestArithmetic:

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
        a = NestedTensor([[2, 3, 4], [5, 6]]).float()
        b = a.clone()
        assert_close((a + i), (i + a))
        assert_close((a - i), -(i - a))
        a += i
        b -= -i
        assert_close(a, b)

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
        a = NestedTensor([[2, 3, 4], [5, 6]]).float()
        b = a.clone()
        assert_close(a * i, i / (1 / a))
        assert_close(a / (1 / i), i * a)
        assert_close(a * i, i * a)
        a *= i
        b /= 1 / i
        assert_close(a, b)

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
        a = NestedTensor([[2, 3, 4], [5, 6]]).float()
        b = a.clone()
        assert_close(torch.log(a**i), torch.log(a) * i)
        a **= i
        assert_close(torch.log(a), torch.log(b) * i)

    @pytest.mark.parametrize(
        "i",
        [
            NestedTensor([[6, 5, 4], [3, 2]]),
            pytest.param(
                torch.tensor([[6, 5, 4], [3, 2, 1]]),
                marks=pytest.mark.xfail(reason="lshift.Tensor non-scalar not implemented"),
            ),
            1,
        ],
    )
    def test_shift(self, i):
        a = NestedTensor([[2, 3, 4], [5, 6]])
        assert_close(a << i >> i, a)
        b = a.clone()
        b <<= i
        assert_close(a << i, b)
        b >>= i
        assert_close(a, b)

    @pytest.mark.parametrize(
        "i",
        [
            pytest.param(
                NestedTensor([[6, 5, 4], [3, 2]]),
                marks=pytest.mark.xfail(reason="bitwise_and.Tensor non-scalar not implemented"),
            ),
            pytest.param(
                torch.tensor([[6, 5, 4], [3, 2, 1]]),
                marks=pytest.mark.xfail(reason="bitwise_and.Tensor non-scalar not implemented"),
            ),
            pytest.param(
                torch.randint(0, 9, (2, 3)),
                marks=pytest.mark.xfail(reason="bitwise_and.Tensor non-scalar not implemented"),
            ),
            1,
            0,
            -1,
            random.randint(0, 9),
        ],
    )
    def test_logic(self, i):
        a = NestedTensor([[2, 3, 4], [5, 6]]).int()
        assert_close(a & i, i & a)
        assert_close((a | i), (i | a))
        assert_close(a ^ i, i ^ a)
        assert_close(~a & ~i, ~(+a | +i))
        assert_close(~(+i | +a), ~i & ~a)
        b = a.clone() + 1
        assert_close(((a & i) | (i & b)), (i & (a | b)))
        assert_close(((i | a) & (b | i)), (i | (a & b)))
        assert_close(((a ^ i) ^ b), (a ^ (i ^ b)))
        b = a.clone()
        b &= i
        assert_close(a & i, b)
        b = a.clone()
        b |= i
        assert_close(a | i, b)
        b = a.clone()
        b ^= i
        assert_close(a ^ i, b)

    def test_to_torch_nested_prefers_jagged_for_ranked_storage(self):
        nt = NestedTensor([torch.randn(2, 3), torch.randn(1, 3)])
        output = nt.to_torch_nested()
        assert output.layout == torch.jagged

    def test_to_torch_nested_falls_back_for_scalar_storage(self):
        nt = NestedTensor([torch.tensor(1.0), torch.tensor(2.0)])
        output = nt.to_torch_nested()
        assert output.layout == torch.strided

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
        a = NestedTensor([[2, 3, 4], [5, 6]])
        a.padding_value = -1
        assert_close(a // i, a.tensor // i)
        assert_close(i // a, i // a.tensor)

    def test_ifloordiv(self):
        a = NestedTensor([[2, 3, 4], [5, 6]], dtype=torch.float32)
        b = a.clone()
        a.padding_value = -1
        a //= 1
        assert_close(a, b)
        a //= b
        assert_close(a, torch.ones(2, 3))
        a //= torch.ones(2, 3)
        assert_close(a, torch.ones(2, 3))

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
        a = NestedTensor([[2, 3, 4], [5, 6]])
        a.padding_value = -1
        assert_close(a % i, a.tensor % i)
        assert_close(i % a, i % a.tensor)

    def test_imod(self):
        a = NestedTensor([[2, 3, 4], [5, 6]])
        a %= NestedTensor([[6, 5, 4], [3, 2]])
        assert_close(a, NestedTensor([[2, 3, 0], [2, 0]]))
        a = NestedTensor([[2, 3, 4], [5, 6]])
        a %= 2
        assert_close(a, NestedTensor([[0, 1, 0], [1, 0]]))
        a %= torch.ones_like(a.tensor)
        assert_close(a, torch.zeros_like(a.tensor))


# ---------------------------------------------------------------------------
# Reduction Operations
# ---------------------------------------------------------------------------


class TestCat:

    def test_cat_extends_1d_sequence_max_length(self):
        lengths = [2, 3, 5, 7]
        additional_length = 11
        channels = 8
        nested_tensor = NestedTensor(torch.randn(length, channels) for length in lengths)
        lengths.append(additional_length)
        nested_tensor = torch.cat([nested_tensor, torch.randn(additional_length, channels)])
        tensor, mask = nested_tensor.tensor_mask
        assert nested_tensor.tensor.shape == torch.Size((len(lengths), max(lengths), channels))
        assert nested_tensor.mask.shape == torch.Size((len(lengths), max(lengths)))
        assert_close(nested_tensor.tensor @ nested_tensor.T, tensor @ nested_tensor.T)


# ---------------------------------------------------------------------------
# _pack Optimization
# ---------------------------------------------------------------------------


class TestComparison:

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

    def test_length_mismatch_equality_and_ops(self):
        shorter = NestedTensor([[1, 2]])
        longer = NestedTensor([[1, 2], [3]])

        assert torch.equal(shorter, longer) is False
        assert torch.equal(longer, shorter) is False
        assert torch.allclose(shorter, longer) is False
        with pytest.raises(ValueError):
            _ = shorter == longer
        with pytest.raises(ValueError):
            _ = torch.eq(shorter, longer)
        with pytest.raises(ValueError):
            _ = shorter + longer
        with pytest.raises(ValueError):
            _ = torch.add(shorter, longer)

    def test_equal_dense_shape_mismatch_returns_false(self):
        nested_tensor = NestedTensor([[1, 2], [3]])
        assert torch.equal(nested_tensor, torch.zeros(2, 2)) is False
        assert torch.equal(torch.zeros(2, 2), nested_tensor) is False

    def test_allclose_dense_shape_mismatch_matches_dense_error(self):
        nested_tensor = NestedTensor([[1.0, 2.0], [3.0]])
        dense = torch.zeros(2, 4)

        with pytest.raises(RuntimeError, match="must match"):
            torch.allclose(nested_tensor, dense)
        with pytest.raises(RuntimeError, match="must match"):
            torch.allclose(dense, nested_tensor)

    def test_allclose_dense_broadcast_matches_dense(self):
        nested_tensor = NestedTensor([[1.0, 2.0], [1.0, 2.0]])
        dense = torch.tensor([[1.0, 2.0]])
        assert torch.allclose(nested_tensor, dense) is torch.allclose(nested_tensor.tensor, dense)
        assert torch.allclose(dense, nested_tensor) is torch.allclose(dense, nested_tensor.tensor)


# ---------------------------------------------------------------------------
# Arithmetic Operators
# ---------------------------------------------------------------------------


class TestConstruction:

    def test_invalid_inputs_raise(self):
        a = NestedTensor([[2, 3, 4], [5, 6]])
        with pytest.raises(ValueError):
            _ = a[""]
        with pytest.raises(ValueError):
            _ = NestedTensor(False)

    def test_single_tensor_not_unbound(self):
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        nested_tensor = NestedTensor(tensor)

        assert len(nested_tensor) == 1
        assert_close(nested_tensor[0], tensor)
        assert nested_tensor.shape == torch.Size([1, 2, 3])
        assert_close(nested_tensor.tensor, tensor.unsqueeze(0))

    def test_mixed_dtype_inputs_promote_to_common_dtype(self):
        nested_tensor = NestedTensor([torch.tensor([1], dtype=torch.int64), torch.tensor([1.5], dtype=torch.float32)])
        assert nested_tensor.dtype == torch.float32
        assert all(t.dtype == torch.float32 for t in nested_tensor)
        assert nested_tensor.tensor.dtype == torch.float32
        assert_close(nested_tensor.tensor, torch.tensor([[1.0], [1.5]], dtype=torch.float32))

    def test_empty_nested_tensor_accessors(self):
        nested_tensor = NestedTensor([], dtype=torch.float32)
        assert nested_tensor.size() == torch.Size([0])
        assert nested_tensor.dim() == 1
        tensor, mask = nested_tensor.tensor_mask
        assert tensor.shape == torch.Size([0])
        assert mask.shape == torch.Size([0])
        assert nested_tensor.tensor.shape == torch.Size([0])
        assert nested_tensor.mask.shape == torch.Size([0])
        assert nested_tensor.occupancy == 0.0

    def test_empty_nested_tensor_honors_requested_device(self):
        nested_tensor = NestedTensor([], dtype=torch.float32, device=torch.device("meta"))
        assert nested_tensor.device.type == "meta"
        assert nested_tensor._values.device.type == "meta"
        assert nested_tensor._offsets.device.type == "cpu"
        assert nested_tensor._physical_shape.device.type == "cpu"

    def test_bool_nested_tensor_raises(self):
        with pytest.raises(RuntimeError, match="Boolean value of NestedTensor is ambiguous"):
            bool(NestedTensor([]))

    def test_requires_grad_tracks_packed_values(self):
        nt = NestedTensor([torch.tensor([1.0], requires_grad=True), torch.tensor([2.0])])
        assert nt.requires_grad is True

    def test_empty_requires_grad_is_preserved(self):
        nt = NestedTensor([], dtype=torch.float32, requires_grad=True)
        assert nt.requires_grad is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
    def test_pin_memory_pins_packed_storage_when_requested(self):
        nt = NestedTensor([torch.tensor([1.0, 2.0]), torch.tensor([3.0])], pin_memory=True)
        assert nt._pin_memory is nt._values.is_pinned()
        assert nt._pin_memory is True

    def test_dense_to_packed_values_uses_packed_dense_index_without_python_meta(self):
        nt = NestedTensor(
            [
                torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3),
                torch.arange(3 * 3, dtype=torch.float32).reshape(3, 3),
            ]
        )
        nt._packed_sizes = None
        nt._element_shapes = None

        packed = nt._dense_to_packed_values(nt.tensor)

        assert packed is not None
        assert_close(packed, nt.concat)

    def test_from_packed_rejects_non_monotonic_offsets(self):
        nt = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        bad_offsets = nt._offsets.clone()
        bad_offsets[1] = bad_offsets[2] + 1
        with pytest.raises(ValueError, match="offsets must be monotonically non-decreasing"):
            NestedTensor._from_packed(
                nt._values,
                bad_offsets,
                nt._physical_shape,
                permutation=nt._permutation,
                batch_first=nt.batch_first,
                padding_value=nt.padding_value,
                mask_value=nt.mask_value,
                pin_memory=nt._pin_memory,
                outer_size=nt._logical_shape,
                packed_sizes=nt._packed_sizes,
                element_shapes=nt._element_shapes,
            )

    def test_from_packed_rejects_packed_sizes_total_mismatch(self):
        nt = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        bad_packed_sizes = (1, 1)
        with pytest.raises(ValueError, match="packed_sizes must sum to the packed values length"):
            NestedTensor._from_packed(
                nt._values,
                nt._offsets,
                nt._physical_shape,
                permutation=nt._permutation,
                batch_first=nt.batch_first,
                padding_value=nt.padding_value,
                mask_value=nt.mask_value,
                pin_memory=nt._pin_memory,
                outer_size=nt._logical_shape,
                packed_sizes=bad_packed_sizes,
                element_shapes=nt._element_shapes,
            )

    def test_from_packed_rejects_inconsistent_element_shapes(self):
        nt = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        bad_shapes = ((2, 3), (4, 4))
        with pytest.raises(ValueError, match="element_shapes must match shape_tensor exactly"):
            NestedTensor._from_packed(
                nt._values,
                nt._offsets,
                nt._physical_shape,
                permutation=nt._permutation,
                batch_first=nt.batch_first,
                padding_value=nt.padding_value,
                mask_value=nt.mask_value,
                pin_memory=nt._pin_memory,
                outer_size=nt._logical_shape,
                packed_sizes=nt._packed_sizes,
                element_shapes=bad_shapes,
            )


# ---------------------------------------------------------------------------
# Packed Reconstruction
# ---------------------------------------------------------------------------


class TestDeclaredRaggedDims:

    @pytest.mark.parametrize("sizes", [(4,), (4, 4), (2, 4)])
    def test_declared_pair_topology_has_stable_flat_packing(self, sizes):
        elements = [torch.randn(size, size, 3) for size in sizes]

        nested = NestedTensor(elements, ragged_dims=(0, 1))

        assert nested.ragged_dims == (0, 1)
        assert nested.packed_dim_order == (0, 1, 2)
        assert nested.concat.shape == (sum(size * size for size in sizes), 3)
        assert nested._packed_sizes == tuple(size * size for size in sizes)
        assert [tuple(element.shape) for element in nested] == [tuple(element.shape) for element in elements]
        for actual, expected in zip(nested, elements):
            assert_close(actual, expected)

    def test_declared_multiragged_topology_has_persistent_level_offsets(self):
        nested = NestedTensor(
            [torch.empty(2, 3, 5), torch.empty(1, 4, 5)],
            ragged_dims=(0, 1),
        )

        level_offsets = nested._persistent_ragged_offsets()
        assert level_offsets is not None
        assert tuple(offset.tolist() for offset in level_offsets) == (
            [0, 2, 3],
            [0, 3, 6, 10],
        )
        assert nested._offsets.tolist() == [0, 6, 10]
        assert nested.packed_local_indices(0).tolist() == [0, 1, 0]
        assert nested.packed_local_indices(1).tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2, 3]

        attrs, ctx = nested.__tensor_flatten__()
        assert "_ragged_offsets_0" in attrs
        assert "_ragged_offsets_1" in attrs
        assert ctx["packed_sizes"] is None
        assert ctx["element_shapes"] is None

    def test_only_explicit_packed_prefix_multiragged_layouts_use_persistent_offsets(self):
        inferred = NestedTensor([torch.empty(2, 3, 5), torch.empty(1, 4, 5)])
        explicit_permuted = NestedTensor(
            [torch.empty(2, 3, 5), torch.empty(2, 3, 5)],
            ragged_dims=(1, 0),
        )
        inferred_permuted = inferred.permute(0, 3, 1, 2)

        assert inferred._persistent_ragged_offsets() is None
        assert inferred_permuted._persistent_ragged_offsets() is None
        assert explicit_permuted._persistent_ragged_offsets() is not None
        assert inferred.__tensor_flatten__()[1]["element_shapes"] is not None
        assert inferred_permuted.__tensor_flatten__()[1]["element_shapes"] is not None
        assert explicit_permuted.__tensor_flatten__()[1]["element_shapes"] is None

    def test_explicit_nonleading_single_ragged_layout_is_tensor_backed(self):
        sampled = NestedTensor(
            [torch.empty(2, 3, 5), torch.empty(2, 4, 5)],
            ragged_dims=(1,),
        )
        canonical = NestedTensor(
            [torch.empty(3, 5), torch.empty(4, 5)],
            ragged_dims=(0,),
        )

        assert sampled.ragged_dims == (1,)
        assert sampled.packed_dim_order == (1, 0, 2)
        sampled_offsets = sampled._persistent_ragged_offsets()
        assert sampled_offsets is not None
        assert sampled_offsets[0] is sampled._offsets
        assert sampled._dynamo_propagated_dynamic_indices == {2}
        assert sampled.concat._dynamo_propagated_dynamic_indices == {0}
        sampled_context = sampled.__tensor_flatten__()[1]
        assert sampled_context["packed_sizes"] is None
        assert sampled_context["element_shapes"] is None

        # Preserve the established list-construction contract for ordinary
        # leading single-ragged layouts. Tensor-backed ``packed_with_lengths``
        # outputs continue to use their separate metadata-free route.
        assert canonical._persistent_ragged_offsets() is None
        canonical_context = canonical.__tensor_flatten__()[1]
        assert canonical_context["packed_sizes"] == (3, 4)
        assert canonical_context["element_shapes"] == ((3, 5), (4, 5))

    @pytest.mark.parametrize(
        ("kind", "ragged_dims", "packed_order", "packed_shape", "dynamic_dims"),
        [
            ("shared", (2, 3), (2, 3, 0, 1), (13, 1, 4), {3, 4}),
            ("full", (0, 2, 3), (0, 2, 3, 1), (35, 4), {1, 3, 4}),
        ],
    )
    def test_permuted_triangle_bias_layouts_are_tensor_backed(
        self,
        kind,
        ragged_dims,
        packed_order,
        packed_shape,
        dynamic_dims,
    ):
        lengths = (2, 3)
        heads = 4
        if kind == "shared":
            elements = [torch.randn(1, heads, length, length) for length in lengths]
        else:
            elements = [torch.randn(length, heads, length, length) for length in lengths]

        nested = NestedTensor(elements, ragged_dims=ragged_dims)
        offsets = nested._persistent_ragged_offsets()
        attrs, context = nested.__tensor_flatten__()

        assert nested.ragged_dims == ragged_dims
        assert nested.packed_dim_order == packed_order
        assert nested.concat.shape == packed_shape
        assert offsets is not None
        assert offsets[0].tolist() == [0, 2, 5]
        assert offsets[1].tolist() == [0, 2, 4, 7, 10, 13]
        assert offsets[-1][-1].item() == packed_shape[0]
        assert len(offsets) == len(ragged_dims)
        assert all(f"_ragged_offsets_{level}" in attrs for level in range(len(ragged_dims)))
        assert context["packed_sizes"] is None
        assert context["element_shapes"] is None
        assert nested._dynamo_propagated_dynamic_indices == dynamic_dims
        assert nested.concat._dynamo_propagated_dynamic_indices == {0}
        assert all(offset._dynamo_propagated_dynamic_indices == {0} for offset in offsets)
        for actual, expected in zip(nested, elements):
            assert_close(actual, expected)

    def test_permuted_triangle_bias_dynamic_dims_respect_nonleading_batch(self):
        lengths = (2, 3)
        shared = NestedTensor(
            [torch.empty(1, 4, length, length) for length in lengths],
            ragged_dims=(2, 3),
            batch_first=False,
        )
        full = NestedTensor(
            [torch.empty(length, 4, length, length) for length in lengths],
            ragged_dims=(0, 2, 3),
            batch_first=False,
        )

        assert shared._dynamo_propagated_dynamic_indices == {3, 4}
        assert full._dynamo_propagated_dynamic_indices == {0, 3, 4}

    @pytest.mark.parametrize("kind", ["shared", "full"])
    def test_permuted_triangle_bias_fake_copy_pickle_and_rebuild(self, kind):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        lengths = (2, 3)
        heads = 4
        if kind == "shared":
            reference = NestedTensor(
                [torch.randn(1, heads, length, length) for length in lengths],
                ragged_dims=(2, 3),
            )
        else:
            reference = NestedTensor(
                [torch.randn(length, heads, length, length) for length in lengths],
                ragged_dims=(0, 2, 3),
            )
        expected_offsets = tuple(offset.tolist() for offset in reference._hierarchical_offsets)

        rebuilt = reference.packed_like(torch.randn_like(reference.concat))
        outputs = (
            rebuilt,
            copy.copy(reference),
            copy.deepcopy(reference),
            reference.clone(),
            reference.detach(),
            pickle.loads(pickle.dumps(reference)),
        )
        for output in outputs:
            assert output.ragged_dims == reference.ragged_dims
            assert output.packed_dim_order == reference.packed_dim_order
            assert tuple(offset.tolist() for offset in output._hierarchical_offsets) == expected_offsets
            assert output.__tensor_flatten__()[1]["element_shapes"] is None

        with fake_tensor_mod.FakeTensorMode() as mode:
            fake_reference = mode.from_tensor(reference)
            fake_output = fake_reference.packed_like(torch.empty_like(fake_reference.concat))

        assert fake_tensor_mod.is_fake(fake_output.concat)
        assert all(fake_tensor_mod.is_fake(offset) for offset in fake_output._hierarchical_offsets)
        assert fake_output.ragged_dims == reference.ragged_dims
        assert fake_output.packed_dim_order == reference.packed_dim_order
        assert fake_output.__tensor_flatten__()[1]["element_shapes"] is None

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize("kind", ["shared", "full"])
    def test_permuted_triangle_bias_reuses_dynamic_layout_and_static_tail_backward(self, kind):
        from torch._dynamo.testing import CompileCounter

        counter = CompileCounter()

        def consume(nested):
            reduced = nested.sum(2, keepdim=True)
            rebuilt = reduced.packed_like(reduced.concat.square())
            return (rebuilt.concat.sum(),) + tuple(
                rebuilt.ragged_level_offsets(level) for level in range(len(rebuilt.ragged_dims))
            )

        compiled = torch.compile(consume, backend=counter, fullgraph=True, dynamic=True)
        for lengths in ((2, 3), (3, 5)):
            heads = 4
            if kind == "shared":
                template = NestedTensor(
                    [torch.empty(1, heads, length, length) for length in lengths],
                    ragged_dims=(2, 3),
                )
            else:
                template = NestedTensor(
                    [torch.empty(length, heads, length, length) for length in lengths],
                    ragged_dims=(0, 2, 3),
                )
            values = torch.randn_like(template.concat, requires_grad=True)
            nested = template.packed_like(values)

            loss, *offsets = compiled(nested)
            loss.backward()

            expected_sum = values.sum(-1, keepdim=True)
            assert_close(values.grad, 2 * expected_sum.expand_as(values))
            assert len(offsets) == len(template.ragged_dims)
            assert all(
                torch.equal(actual, expected) for actual, expected in zip(offsets, template._hierarchical_offsets)
            )

        assert counter.frame_count == 1

    def test_default_pair_topology_keeps_existing_inference(self):
        nested = NestedTensor([torch.randn(4, 4, 3), torch.randn(4, 4, 3)])

        assert nested.ragged_dims == (0,)
        assert nested.concat.shape == (8, 4, 3)

    def test_declared_ragged_dims_are_read_only_and_validate_static_dims(self):
        nested = NestedTensor([torch.randn(2, 2, 3), torch.randn(4, 4, 3)], ragged_dims=(0, 1))

        with pytest.raises(AttributeError):
            nested.ragged_dims = (0,)  # type: ignore[misc]
        with pytest.raises(ValueError, match="not listed in ragged_dims"):
            NestedTensor([torch.randn(2, 2, 3), torch.randn(4, 4, 5)], ragged_dims=(0, 1))

    def test_declared_ragged_dims_survive_fake_tensor_rebuild(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        FakeTensorMode = fake_tensor_mod.FakeTensorMode
        is_fake = fake_tensor_mod.is_fake
        reference = NestedTensor([torch.empty(4, 4, 3), torch.empty(4, 4, 3)], ragged_dims=(0, 1))

        with FakeTensorMode() as mode:
            fake_reference = mode.from_tensor(reference)
            output = fake_reference.packed_like(torch.empty_like(fake_reference.concat))

        assert output.ragged_dims == (0, 1)
        assert output.concat.shape == (32, 3)
        assert is_fake(output.concat)
        assert is_fake(output._offsets)
        assert is_fake(output._physical_shape)
        assert output._packed_sizes is None
        assert output._element_shapes is None
        assert all(is_fake(offset) for offset in output._hierarchical_offsets)
        assert tuple(offset.shape for offset in output._hierarchical_offsets) == ((3,), (9,))

    def test_declared_ragged_dims_construct_directly_from_fake_elements(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        FakeTensorMode = fake_tensor_mod.FakeTensorMode
        is_fake = fake_tensor_mod.is_fake

        with FakeTensorMode():
            output = NestedTensor(
                [torch.empty(2, 2, 3), torch.empty(4, 4, 3)],
                ragged_dims=(0, 1),
            )

        assert is_fake(output.concat)
        assert all(is_fake(offset) for offset in output._hierarchical_offsets)
        assert tuple(offset.shape for offset in output._hierarchical_offsets) == ((3,), (7,))

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_declared_ragged_dims_survive_fullgraph_output(self):
        reference = NestedTensor([torch.randn(4, 4, 3), torch.randn(4, 4, 3)], ragged_dims=(0, 1))
        values = torch.randn_like(reference.concat, requires_grad=True)
        compiled = torch.compile(lambda ref, packed: ref.packed_like(packed), backend="aot_eager", fullgraph=True)

        output = compiled(reference, values)
        loss = output.concat.square().sum()
        loss.backward()

        assert output.ragged_dims == (0, 1)
        assert output.concat.shape == (32, 3)
        assert_close(values.grad, 2 * values)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_declared_pair_fullgraph_reuses_dynamic_layout_and_backward(self):
        from torch._dynamo.testing import CompileCounter

        counter = CompileCounter()

        def consume(nested):
            output = nested.packed_like(nested.concat.square())
            return (
                output.concat.sum(),
                output.ragged_level_offsets(0),
                output.ragged_level_offsets(1),
                output.packed_local_indices(0),
                output.packed_local_indices(1),
            )

        compiled = torch.compile(consume, backend=counter, fullgraph=True, dynamic=True)
        for lengths in ((2, 3), (3, 5)):
            reference = NestedTensor(
                [torch.empty(length, length, 3) for length in lengths],
                ragged_dims=(0, 1),
            )
            values = torch.randn_like(reference.concat, requires_grad=True)
            nested = reference.packed_like(values)

            loss, rows, cells, row_local, cell_local = compiled(nested)
            loss.backward()

            expected_rows = torch.nn.functional.pad(torch.tensor(lengths).cumsum(0), (1, 0))
            expected_cell_widths = torch.repeat_interleave(torch.tensor(lengths), torch.tensor(lengths))
            expected_cells = torch.nn.functional.pad(expected_cell_widths.cumsum(0), (1, 0))
            assert_close(rows, expected_rows)
            assert_close(cells, expected_cells)
            assert_close(row_local, torch.cat([torch.arange(length) for length in lengths]))
            assert_close(
                cell_local,
                torch.cat([torch.arange(length).repeat(length) for length in lengths]),
            )
            assert_close(values.grad, 2 * values)

        assert counter.frame_count == 1

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_declared_pair_static_tail_reuses_dynamic_layout_and_backward(self):
        from torch._dynamo.testing import CompileCounter

        counter = CompileCounter()

        def consume(nested, values):
            output = nested.packed_with_static_tail(values.square())
            return output.concat.sum(), output.ragged_level_offsets(0), output.ragged_level_offsets(1)

        compiled = torch.compile(consume, backend=counter, fullgraph=True, dynamic=True)
        for lengths in ((2, 3), (3, 5)):
            reference = NestedTensor(
                [torch.empty(length, length, 3) for length in lengths],
                ragged_dims=(0, 1),
            )
            values = torch.randn(reference.concat.shape[0], 7, requires_grad=True)

            loss, rows, cells = compiled(reference, values)
            loss.backward()

            assert_close(rows, reference.ragged_level_offsets(0))
            assert_close(cells, reference.ragged_level_offsets(1))
            assert_close(values.grad, 2 * values)

        assert counter.frame_count == 1

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_declared_pair_static_tail_movedim_reuses_dynamic_layout_and_backward(self):
        from torch._dynamo.testing import CompileCounter

        counter = CompileCounter()

        def consume(reference, values):
            projected = reference.packed_with_static_tail(values.square())
            head_major = torch.movedim(projected, -2, 1)
            return (
                head_major.concat.sum(),
                head_major.ragged_level_offsets(0),
                head_major.ragged_level_offsets(1),
            )

        compiled = torch.compile(consume, backend=counter, fullgraph=True, dynamic=True)
        for lengths in ((2, 3), (3, 5)):
            reference = NestedTensor(
                [torch.empty(length, length, 3) for length in lengths],
                ragged_dims=(0, 1),
            )
            values = torch.randn(sum(length * length for length in lengths), 2, 7, requires_grad=True)

            loss, rows, cells = compiled(reference, values)
            loss.backward()

            eager = torch.movedim(reference.packed_with_static_tail(values.detach().square()), -2, 1)
            assert eager.ragged_dims == (1, 2)
            assert eager.packed_dim_order == (1, 2, 0, 3)
            assert_close(rows, reference.ragged_level_offsets(0))
            assert_close(cells, reference.ragged_level_offsets(1))
            assert_close(values.grad, 2 * values)

        assert counter.frame_count == 1

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize(
        ("name", "operation"),
        [
            ("linear", lambda nested: torch.nn.functional.linear(nested, torch.ones(5, 3))),
            ("batch_transpose", lambda nested: nested.transpose(0, 1)),
            ("unsqueeze_tail", lambda nested: nested.unsqueeze(-1)),
            ("sum_static_tail", lambda nested: nested.sum(-1, keepdim=True)),
        ],
    )
    def test_declared_pair_shape_preserving_ops_reuse_dynamic_layout(self, name, operation):
        del name
        from torch._dynamo.testing import CompileCounter

        counter = CompileCounter()
        compiled = torch.compile(operation, backend=counter, fullgraph=True, dynamic=True)
        for lengths in ((2, 3), (3, 5)):
            nested = NestedTensor(
                [torch.randn(length, length, 3) for length in lengths],
                ragged_dims=(0, 1),
            )
            expected = operation(nested)
            output = compiled(nested)

            assert output.ragged_dims == expected.ragged_dims
            assert output.shape == expected.shape
            assert_close(output.concat, expected.concat)
            assert all(
                torch.equal(actual, reference)
                for actual, reference in zip(output._hierarchical_offsets, nested._hierarchical_offsets)
            )

        assert counter.frame_count == 1

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_cacheless_compiled_pair_rebuilds_do_not_materialize_python_metadata(self):
        reference = NestedTensor([torch.empty(2, 2, 3), torch.empty(3, 3, 3)], ragged_dims=(0, 1))
        producer = torch.compile(
            lambda nested, values: nested.packed_like(values),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        output = producer(reference, torch.randn_like(reference.concat))
        rebuilds = (
            output.packed_like(torch.randn_like(output.concat)),
            output.packed_with_static_tail(torch.randn(output.concat.shape[0], 7)),
            copy.copy(output),
            copy.deepcopy(output),
            output.clone(),
            output.detach(),
            pickle.loads(pickle.dumps(output)),
        )

        assert output._packed_sizes is None
        assert output._element_shapes is None
        assert all(rebuilt._packed_sizes is None for rebuilt in rebuilds)
        assert all(rebuilt._element_shapes is None for rebuilt in rebuilds)
        assert all(len(rebuilt._hierarchical_offsets) == 2 for rebuilt in rebuilds)

    def test_declared_ragged_dims_survive_meta_copy_and_pickle(self):
        reference = NestedTensor([torch.randn(2, 2, 3), torch.randn(4, 4, 3)], ragged_dims=(0, 1))

        outputs = (
            NestedTensor(reference._storage, **reference._meta()),
            copy.copy(reference),
            copy.deepcopy(reference),
            pickle.loads(pickle.dumps(reference)),
        )

        assert all(output.ragged_dims == (0, 1) for output in outputs)
        assert all(output.concat.shape == (20, 3) for output in outputs)
        expected_offsets = tuple(offset.tolist() for offset in reference._hierarchical_offsets)
        assert all(
            tuple(offset.tolist() for offset in output._hierarchical_offsets) == expected_offsets for output in outputs
        )

    def test_declared_pair_shape_preserving_rebuilds_propagate_offsets(self):
        reference = NestedTensor([torch.randn(2, 2, 3), torch.randn(4, 4, 3)], ragged_dims=(0, 1))
        packed = reference.packed_like(torch.randn_like(reference.concat))
        static_tail = reference.packed_with_static_tail(torch.randn(reference.concat.shape[0], 7))
        shallow = copy.copy(reference)
        detached = reference.detach()
        cloned = reference.clone()
        deep = copy.deepcopy(reference)
        restored = pickle.loads(pickle.dumps(reference))

        for output in (packed, static_tail, shallow):
            assert all(
                actual is expected
                for actual, expected in zip(output._hierarchical_offsets, reference._hierarchical_offsets)
            )
        assert tuple(offset.tolist() for offset in detached._hierarchical_offsets) == tuple(
            offset.tolist() for offset in reference._hierarchical_offsets
        )
        assert all(
            actual.data_ptr() == expected.data_ptr()
            for actual, expected in zip(detached._hierarchical_offsets, reference._hierarchical_offsets)
        )
        for output in (cloned, deep, restored):
            assert tuple(offset.tolist() for offset in output._hierarchical_offsets) == tuple(
                offset.tolist() for offset in reference._hierarchical_offsets
            )
            assert all(
                actual is not expected
                for actual, expected in zip(output._hierarchical_offsets, reference._hierarchical_offsets)
            )

    def test_declared_pair_dense_matmul_preserves_topology(self):
        elements = [torch.randn(2, 2, 3), torch.randn(4, 4, 3)]
        nested = NestedTensor(elements, ragged_dims=(0, 1))
        weight = torch.randn(3, 5)

        output = nested @ weight

        assert output.ragged_dims == (0, 1)
        assert output.packed_dim_order == (0, 1, 2)
        assert output.concat.shape == (20, 5)
        assert output._packed_sizes == (4, 16)
        for actual, expected in zip(output, elements):
            assert_close(actual, expected @ weight)

    def test_source_derived_empty_batch_preserves_declared_topology(self):
        nested = NestedTensor([torch.randn(2, 2, 3), torch.randn(4, 4, 3)], ragged_dims=(0, 1))

        sliced = nested[:0]
        split_empty, split_full = torch.split(nested, [0, 2], dim=0)

        for output in (sliced, split_empty):
            assert output.shape == (0, 4, 4, 3)
            assert output.ragged_dims == (0, 1)
            assert output.packed_dim_order == (0, 1, 2)
            assert output.concat.shape == (0, 3)
            assert output._physical_shape.shape == (0, 3)
        assert split_full.ragged_dims == (0, 1)

        with pytest.raises(ValueError, match="outside element rank 0"):
            NestedTensor([], ragged_dims=(0, 1))

    def test_shape_changing_ops_remap_declared_ragged_dims(self):
        elements = [torch.randn(2, 2, 3), torch.randn(2, 2, 3)]
        nested = NestedTensor(elements, ragged_dims=(0, 1))

        permuted = nested.permute(0, 3, 1, 2)
        indexed = nested[:, :, 0, :]

        assert permuted.ragged_dims == (1, 2)
        assert permuted.packed_dim_order == (1, 2, 0)
        assert permuted.concat.shape == (8, 3)
        assert indexed.ragged_dims == (0,)
        assert indexed.concat.shape == (4, 3)
        for actual, expected in zip(permuted, elements):
            assert_close(actual, expected.permute(2, 0, 1))
        for actual, expected in zip(indexed, elements):
            assert_close(actual, expected[:, 0, :])

        nonleading = NestedTensor([torch.randn(2, 3, 4), torch.randn(2, 3, 4)], ragged_dims=(2,))
        reduced = torch.sum(nonleading, dim=1)
        assert reduced.ragged_dims == (1,)
        assert reduced.packed_dim_order == (1, 0)
        for actual, expected in zip(reduced, nonleading):
            assert_close(actual, expected.sum(dim=0))

    def test_from_concatenated_honors_declared_ragged_order(self):
        elements = [torch.arange(12.0).reshape(2, 2, 3), torch.arange(12.0, 24.0).reshape(2, 2, 3)]
        reference = NestedTensor(elements, ragged_dims=(1, 0))

        values, shapes = reference.concatenate()
        output = NestedTensor.from_concatenated(values, shapes, ragged_dims=(1, 0))

        assert output.ragged_dims == (1, 0)
        assert output.packed_dim_order == (1, 0, 2)
        for actual, expected in zip(output, elements):
            assert_close(actual, expected)

    def test_serialized_state_rejects_topology_payload_mismatch(self):
        reference = NestedTensor([torch.randn(2, 2, 3), torch.randn(2, 2, 3)], ragged_dims=(0, 1))
        state = reference.__getstate__()
        state["_ragged_dims"] = (0,)

        with pytest.raises(ValueError, match="Expected one ragged offset tensor"):
            NestedTensor._from_state(state)

    def test_serialized_state_rejects_ragged_offset_payload_mismatch(self):
        reference = NestedTensor([torch.randn(2, 2, 3), torch.randn(3, 3, 3)], ragged_dims=(0, 1))
        state = reference.__getstate__()
        ragged_offsets = state["_ragged_offsets"]
        assert ragged_offsets is not None
        bad_inner = ragged_offsets[1].clone()
        bad_inner[2] += 1
        state["_ragged_offsets"] = (ragged_offsets[0], bad_inner)

        with pytest.raises(ValueError, match=r"ragged_offsets\[1\] does not match physical_shape"):
            NestedTensor._from_state(state)

    def test_serialized_state_rejects_single_level_offset_payload_mismatch(self):
        source = NestedTensor([torch.empty(1), torch.empty(1)])
        reference = source.packed_with_lengths(torch.randn(5, 3), torch.tensor([2, 3]))
        state = reference.__getstate__()
        state["_ragged_offsets"] = (torch.tensor([0, 1, 5]),)

        with pytest.raises(ValueError, match="single-level ragged offsets must match offsets"):
            NestedTensor._from_state(state)


class TestPackedLike:
    def test_nonleading_single_ragged_packed_like_accepts_external_fake_values(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        reference = NestedTensor(
            [torch.empty(2, 3, 5), torch.empty(2, 4, 5)],
            ragged_dims=(1,),
        )

        with fake_tensor_mod.FakeTensorMode():
            fake_values = torch.empty(tuple(reference.concat.shape), dtype=reference.dtype)
            output = reference.packed_like(fake_values)

        assert output.concat is fake_values
        assert fake_tensor_mod.is_fake(output.concat)
        assert fake_tensor_mod.is_fake(output.packed_offsets())
        assert output.ragged_level_offsets() is output.packed_offsets()
        assert output.packed_offsets().shape == reference.packed_offsets().shape
        assert output.ragged_dims == (1,)
        assert output.packed_dim_order == (1, 0, 2)

    def test_packed_dim_order_tracks_logical_permutation(self):
        reference = NestedTensor([torch.randn(2, 2, 3), torch.randn(4, 4, 3)])

        assert reference.packed_dim_order == (0, 1, 2)

        permuted = reference.permute(0, 3, 1, 2)
        assert permuted.packed_dim_order == (1, 2, 0)
        rebuilt = permuted.packed_like(torch.randn_like(permuted.concat))
        assert rebuilt.packed_dim_order == permuted.packed_dim_order

    def test_packed_like_preserves_structure_and_runtime_config(self):
        reference = NestedTensor(
            [torch.randn(2, 3), torch.randn(4, 3)],
            batch_first=False,
            padding_value=-1.5,
            mask_value=True,
        )
        packed_values = torch.arange(reference.concat.numel(), dtype=torch.float32).reshape(reference.concat.shape)

        output = reference.packed_like(packed_values)

        assert output._has_same_layout(reference)
        assert output.shape == reference.shape
        assert output.batch_first is False
        assert output.padding_value == -1.5
        assert output.mask_value is True
        assert output._offsets is reference._offsets
        assert output._physical_shape is reference._physical_shape
        assert output._permutation == reference._permutation
        assert output._packed_sizes is reference._packed_sizes
        assert output._element_shapes is reference._element_shapes

    def test_packed_like_preserves_multi_ragged_permuted_layout(self):
        reference = NestedTensor(
            [
                torch.randn(1, 2, 3, 5),
                torch.randn(1, 4, 2, 5),
            ]
        )
        assert reference._ragged_rank == 2
        assert reference._permutation != tuple(range(4))
        packed_values = torch.randn_like(reference.concat)

        output = reference.packed_like(packed_values)

        assert output._has_same_layout(reference)
        assert output._ragged_rank == 2
        assert output._permutation == reference._permutation
        assert_close(output.concat, packed_values)

    def test_packed_like_empty(self):
        reference = NestedTensor([], dtype=torch.float32, batch_first=False, padding_value=-2, mask_value=True)
        packed_values = torch.empty_like(reference.concat, dtype=torch.float64)

        output = reference.packed_like(packed_values)

        assert output.shape == reference.shape
        assert output.dtype == torch.float64
        assert output.concat is packed_values
        assert output.batch_first is False
        assert output.padding_value == -2
        assert output.mask_value is True

    def test_packed_like_uses_noncontiguous_values_without_copy(self):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        packed_values = torch.randn(3, reference.concat.size(0)).transpose(0, 1)
        assert not packed_values.is_contiguous()

        output = reference.packed_like(packed_values)

        assert output.concat is packed_values
        assert output.concat.data_ptr() == packed_values.data_ptr()
        assert output.concat.stride() == packed_values.stride()
        assert output.concat.storage_offset() == packed_values.storage_offset()

    def test_packed_like_rejects_shape_mismatch(self):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])

        with pytest.raises(ValueError, match="exactly the same shape"):
            reference.packed_like(reference.concat.reshape(-1))
        with pytest.raises(ValueError, match="exactly the same shape"):
            reference.packed_like(torch.empty(reference.concat.size(0) + 1, 3))

    @pytest.mark.parametrize("packed_values", [object(), 1, [1.0, 2.0]])
    def test_packed_like_rejects_non_tensor(self, packed_values):
        reference = NestedTensor([torch.randn(2), torch.randn(1)])

        with pytest.raises(TypeError, match="dense Tensor"):
            reference.packed_like(packed_values)  # type: ignore[arg-type]

    def test_packed_like_rejects_nested_tensor_values(self):
        reference = NestedTensor([torch.randn(2), torch.randn(1)])

        with pytest.raises(TypeError, match="dense Tensor"):
            reference.packed_like(reference)

    def test_packed_like_rejects_native_nested_tensor_values(self):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        packed_values = torch.nested.nested_tensor([torch.randn(2, 3), torch.randn(4, 3)])

        with pytest.raises(TypeError, match="dense Tensor with torch.strided layout"):
            reference.packed_like(packed_values)

    @pytest.mark.parametrize("layout", [torch.sparse_coo, torch.sparse_csr])
    def test_packed_like_rejects_sparse_values(self, layout):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        dense = torch.zeros_like(reference.concat)
        packed_values = dense.to_sparse() if layout == torch.sparse_coo else dense.to_sparse_csr()

        with pytest.raises(TypeError, match="dense Tensor with torch.strided layout"):
            reference.packed_like(packed_values)

    def test_packed_like_follows_dtype_and_autograd_history(self):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        leaf = torch.randn(reference.concat.shape, dtype=torch.float64, requires_grad=True)
        packed_values = leaf.square()

        output = reference.packed_like(packed_values)

        assert output.dtype == torch.float64
        assert output.requires_grad
        assert output.concat is packed_values
        assert output.concat.grad_fn is packed_values.grad_fn
        first_order = torch.autograd.grad(output.concat.sum(), leaf, create_graph=True)[0]
        second_order = torch.autograd.grad(first_order.sum(), leaf)[0]
        assert_close(first_order, 2 * leaf)
        assert_close(second_order, torch.full_like(leaf, 2))

    def test_packed_like_follows_values_device(self):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        packed_values = torch.empty(reference.concat.shape, device="meta")

        output = reference.packed_like(packed_values)

        assert output.device.type == "meta"
        assert output.concat is packed_values
        assert output._offsets.device.type == "cpu"
        assert output._physical_shape.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
    def test_packed_like_follows_values_pinning_without_implicit_pin(self):
        pinned_reference = NestedTensor(
            [torch.randn(2, 3), torch.randn(4, 3)],
            pin_memory=True,
        )
        unpinned_values = torch.randn(pinned_reference.concat.shape)

        unpinned_output = pinned_reference.packed_like(unpinned_values)

        assert unpinned_output.concat is unpinned_values
        assert not unpinned_output.concat.is_pinned()
        assert unpinned_output._pin_memory is False

        pinned_values = unpinned_values.pin_memory()
        pinned_output = pinned_reference.packed_like(pinned_values)

        assert pinned_output.concat is pinned_values
        assert pinned_output.concat.is_pinned()
        assert pinned_output._pin_memory is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
    def test_internal_packed_rebuilds_preserve_pinned_source(self):
        reference = NestedTensor(
            [torch.randn(2, 3), torch.randn(4, 3)],
            pin_memory=True,
        )

        outputs = (torch.sin(reference), torch.ops.aten.alias.default(reference))

        for output in outputs:
            assert output._pin_memory is True
            assert output.concat.is_pinned()

    def test_packed_like_reuses_only_hierarchical_shape_cache(self):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        _ = reference._hierarchical_offsets
        _ = reference.packed_batch_indices()
        _ = reference.packed_local_indices()
        _ = reference.packed_offsets(dtype=torch.int32)
        _ = reference.ragged_level_offsets(dtype=torch.int32)
        _ = reference.mask
        _ = reference.tensor
        packed_values = torch.randn_like(reference.concat)

        output = reference.packed_like(packed_values)

        assert output._cached_hierarchical_offsets is reference._cached_hierarchical_offsets
        assert output._cached_packed_batch_indices is None
        assert output._cached_packed_local_indices is None
        assert output._cached_packed_offsets is None
        assert output._cached_ragged_level_offsets is None
        assert output._cached_mask_view is None
        assert output._cached_tensor_view is None
        assert output._cached_storage is None

    @pytest.mark.parametrize(
        ("elements", "ragged_dims", "expected"),
        [
            ([torch.empty(2, 4), torch.empty(3, 4)], (0,), [0, 2, 5]),
            ([torch.empty(2, 2, 4), torch.empty(3, 3, 4)], (0, 1), [0, 4, 13]),
            ([torch.empty(2, 2, 4), torch.empty(2, 3, 4)], (1,), [0, 2, 5]),
        ],
        ids=["single", "multi", "nonleading"],
    )
    def test_packed_offsets_are_logical_batch_boundaries(self, elements, ragged_dims, expected):
        nested = NestedTensor(elements, ragged_dims=ragged_dims)

        offsets = nested.packed_offsets()

        assert offsets is nested._offsets
        assert offsets.tolist() == expected
        assert offsets[-1].item() == nested.concat.shape[0]

    @pytest.mark.parametrize(
        ("elements", "ragged_dims", "batch_first", "expected"),
        [
            (
                [torch.empty(2, 4), torch.empty(3, 4)],
                (0,),
                True,
                [[2, 4], [3, 4]],
            ),
            (
                [torch.empty(0, 3), torch.empty(0, 7)],
                (0, 1),
                True,
                [[0, 3], [0, 7]],
            ),
            (
                [torch.empty(2, 3, 4), torch.empty(2, 5, 4)],
                (1,),
                True,
                [[2, 3, 4], [2, 5, 4]],
            ),
            (
                [torch.empty(2, 3, 4), torch.empty(2, 5, 4)],
                (1,),
                False,
                [[2, 3, 4], [2, 5, 4]],
            ),
        ],
        ids=["single", "multi-zero-volume", "nonleading", "batch-first-false"],
    )
    def test_element_sizes_follow_logical_element_dimensions(self, elements, ragged_dims, batch_first, expected):
        nested = NestedTensor(elements, ragged_dims=ragged_dims, batch_first=batch_first)

        sizes = nested.element_sizes()

        assert sizes is nested._physical_shape
        assert sizes.device.type == "cpu"
        assert sizes.dtype == torch.int64
        assert sizes.tolist() == expected

    def test_element_sizes_empty_and_scalar_element_rank(self):
        empty = NestedTensor([], dtype=torch.float32)
        scalars = NestedTensor([torch.tensor(1.0), torch.tensor(2.0)])

        assert empty.element_sizes().shape == (0, 0)
        assert scalars.element_sizes().shape == (2, 0)
        assert empty.element_sizes() is empty._physical_shape
        assert scalars.element_sizes() is scalars._physical_shape

    def test_element_sizes_survive_copy_and_pickle(self):
        nested = NestedTensor(
            [torch.empty(0, 3), torch.empty(0, 7)],
            ragged_dims=(0, 1),
        )

        outputs = (copy.copy(nested), copy.deepcopy(nested), pickle.loads(pickle.dumps(nested)))
        for output in outputs:
            assert output.element_sizes() is output._physical_shape
            assert output.element_sizes().tolist() == [[0, 3], [0, 7]]

    def test_element_sizes_support_fake_tensor(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        nested = NestedTensor(
            [torch.empty(0, 3), torch.empty(0, 7)],
            ragged_dims=(0, 1),
        )

        with fake_tensor_mod.FakeTensorMode() as mode:
            fake = mode.from_tensor(nested)
            sizes = fake.element_sizes()

        assert sizes is fake._physical_shape
        assert fake_tensor_mod.is_fake(sizes)
        assert sizes.device.type == "cpu"
        assert sizes.dtype == torch.int64
        assert sizes.shape == (2, 2)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_element_sizes_reuse_one_dynamic_multiragged_graph(self):
        from torch._dynamo.testing import CompileCounter

        counter = CompileCounter()
        compiled = torch.compile(
            lambda nested: nested.element_sizes(),
            backend=counter,
            fullgraph=True,
            dynamic=True,
        )
        layouts = tuple(
            NestedTensor(
                [torch.empty(*shape) for shape in shapes],
                ragged_dims=(0, 1),
            )
            for shapes in (((0, 3), (2, 5)), ((0, 7), (4, 9)))
        )

        outputs = tuple(compiled(nested) for nested in layouts)

        assert_close(outputs[0], torch.tensor([[0, 3], [2, 5]]))
        assert_close(outputs[1], torch.tensor([[0, 7], [4, 9]]))
        assert layouts[0].packed_offsets()[1].item() == layouts[1].packed_offsets()[1].item() == 0
        assert counter.frame_count == 1

    def test_packed_offsets_cache_dtype_conversion_per_instance(self):
        nested = NestedTensor(
            [torch.empty(2, 2, 4), torch.empty(3, 3, 4)],
            ragged_dims=(0, 1),
        )

        converted = nested.packed_offsets(device="cpu", dtype=torch.int32)

        assert converted.dtype == torch.int32
        assert converted.tolist() == [0, 4, 13]
        assert nested.packed_offsets(device=torch.device("cpu"), dtype=torch.int32) is converted
        assert nested.packed_offsets(device="cpu", dtype=torch.long) is nested._offsets

    def test_packed_offsets_survive_copy_and_pickle(self):
        nested = NestedTensor(
            [torch.empty(2, 2, 4), torch.empty(3, 3, 4)],
            ragged_dims=(0, 1),
        )
        converted = nested.packed_offsets(dtype=torch.int32)
        shallow = copy.copy(nested)
        deep = copy.deepcopy(nested)
        restored = pickle.loads(pickle.dumps(nested))

        assert converted is nested.packed_offsets(dtype=torch.int32)
        for output in (shallow, deep, restored):
            assert output._cached_packed_offsets is None
            assert output.packed_offsets() is output._offsets
            assert output.packed_offsets().tolist() == [0, 4, 13]

    def test_packed_offsets_support_fake_tensor(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        nested = NestedTensor(
            [torch.empty(2, 2, 4), torch.empty(3, 3, 4)],
            ragged_dims=(0, 1),
        )

        with fake_tensor_mod.FakeTensorMode() as mode:
            fake = mode.from_tensor(nested)
            fake_offsets = fake.packed_offsets()
            fake_int32_offsets = fake.packed_offsets(dtype=torch.int32)

        assert fake_offsets is fake._offsets
        assert fake_tensor_mod.is_fake(fake_offsets)
        assert fake_tensor_mod.is_fake(fake_int32_offsets)
        assert fake_int32_offsets.dtype == torch.int32

    def test_indexless_non_cpu_offset_conversions_are_not_cached(self):
        nested = NestedTensor(
            [torch.empty(2, 2, 4), torch.empty(3, 3, 4)],
            ragged_dims=(0, 1),
        )

        first_packed = nested.packed_offsets(device="meta", dtype=torch.int32)
        second_packed = nested.packed_offsets(device="meta", dtype=torch.int32)
        first_cells = nested.ragged_level_offsets(1, device="meta", dtype=torch.int32)
        second_cells = nested.ragged_level_offsets(1, device="meta", dtype=torch.int32)

        assert first_packed.device.type == "meta"
        assert first_packed is not second_packed
        assert first_cells is not second_cells
        assert nested._cached_packed_offsets is None
        assert nested._cached_ragged_level_offsets is None
        assert NestedTensor._offset_conversion_device_key(torch.device("cuda")) is None
        assert NestedTensor._offset_conversion_device_key(torch.device("cuda:0")) == "cuda:0"

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_packed_offsets_reuse_one_dynamic_multiragged_graph(self):
        from torch._dynamo.testing import CompileCounter

        counter = CompileCounter()
        compiled = torch.compile(
            lambda nested: (
                nested.concat.sum(),
                nested.packed_offsets(dtype=torch.int32),
                nested.ragged_level_offsets(1, dtype=torch.int32),
            ),
            backend=counter,
            fullgraph=True,
            dynamic=True,
        )

        layouts = tuple(
            NestedTensor(
                [torch.ones(length, length, 4) for length in lengths],
                ragged_dims=(0, 1),
            )
            for lengths in ((2, 3), (3, 5))
        )
        first_packed = layouts[0].packed_offsets(dtype=torch.int32)
        first_cells = layouts[0].ragged_level_offsets(1, dtype=torch.int32)
        outputs = []

        for nested, lengths in zip(layouts, ((2, 3), (3, 5))):
            total, offsets, cell_offsets = compiled(nested)
            outputs.append(offsets)

            expected = torch.tensor([0, lengths[0] ** 2, lengths[0] ** 2 + lengths[1] ** 2], dtype=torch.int32)
            cell_widths = torch.repeat_interleave(torch.tensor(lengths), torch.tensor(lengths))
            expected_cells = torch.nn.functional.pad(cell_widths.cumsum(0), (1, 0)).to(torch.int32)
            assert_close(total, nested.concat.sum())
            assert_close(offsets, expected)
            assert_close(cell_offsets, expected_cells)

        assert layouts[0].packed_offsets(dtype=torch.int32) is first_packed
        assert layouts[0].ragged_level_offsets(1, dtype=torch.int32) is first_cells
        assert layouts[0]._cached_packed_offsets is not layouts[1]._cached_packed_offsets
        assert layouts[0]._cached_ragged_level_offsets is not layouts[1]._cached_ragged_level_offsets
        assert not torch.equal(outputs[0], outputs[1])
        assert counter.frame_count == 1

    def test_packed_local_indices_respect_each_ragged_level(self):
        reference = NestedTensor(
            [torch.empty(2, 2), torch.empty(3, 3)],
            ragged_dims=(0, 1),
        )

        assert_close(reference.packed_local_indices(level=0), torch.tensor([0, 1, 0, 1, 2]))
        assert_close(
            reference.packed_local_indices(level=1),
            torch.tensor([0, 1, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
        )

    def test_packed_like_supports_fake_tensor(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        FakeTensorMode = fake_tensor_mod.FakeTensorMode
        is_fake = fake_tensor_mod.is_fake
        reference = NestedTensor([torch.empty(2, 3), torch.empty(4, 3)])
        _ = reference._hierarchical_offsets
        packed_values = FakeTensorMode().from_tensor(torch.empty_like(reference.concat))

        output = reference.packed_like(packed_values)

        assert output.concat is packed_values
        assert output._cached_hierarchical_offsets is None
        assert output._has_same_layout(reference)
        assert is_fake(output._offsets)
        assert is_fake(output._physical_shape)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_packed_like_fullgraph_forward_backward(self):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        packed_values = torch.randn_like(reference.concat, requires_grad=True)

        compiled = torch.compile(
            lambda ref, values: ref.packed_like(values).concat.square().sum(),
            backend="aot_eager",
            fullgraph=True,
        )
        loss = compiled(reference, packed_values)
        assert type(loss.grad_fn).__name__ == "CompiledFunctionBackward"
        loss.backward()

        assert_close(loss, packed_values.square().sum())
        assert_close(packed_values.grad, 2 * packed_values)

    def test_packed_like_pickle_roundtrip(self):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        output = reference.packed_like(torch.randn_like(reference.concat, dtype=torch.float64))

        restored = pickle.loads(pickle.dumps(output))

        assert restored._has_same_layout(reference)
        assert restored.dtype == torch.float64
        assert_close(restored, output)

    def test_packed_like_preserves_nested_tensor_subclass(self):

        class DerivedNestedTensor(NestedTensor):
            pass

        reference = DerivedNestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        packed_values = torch.randn_like(reference.concat)

        output = reference.packed_like(packed_values)

        assert type(output) is DerivedNestedTensor
        assert output.concat is packed_values
        assert output._has_same_layout(reference)


class TestPackedWithStaticTail:

    def test_adds_static_tail_without_copy_and_preserves_autograd(self):
        reference = NestedTensor(
            [torch.empty(2), torch.empty(4)],
            batch_first=False,
            padding_value=-2.5,
            mask_value=True,
        )
        leaf = torch.randn(6, 2, 3, dtype=torch.float64, requires_grad=True)
        packed_values = leaf.square()

        output = reference.packed_with_static_tail(packed_values)

        assert output.concat is packed_values
        assert output.concat.stride() == packed_values.stride()
        assert output.ragged_dims == (0,)
        assert output.packed_dim_order == (0, 1, 2)
        assert output.shape == (4, 2, 2, 3)
        assert output.batch_first is False
        assert output.padding_value == -2.5
        assert output.mask_value is True
        assert output._offsets is reference._offsets
        assert [tuple(element.shape) for element in output] == [(2, 2, 3), (4, 2, 3)]
        gradient = torch.autograd.grad(output.concat.sum(), leaf)[0]
        assert_close(gradient, 2 * leaf)

    def test_replaces_static_tail_and_preserves_multi_ragged_lengths(self):
        reference = NestedTensor(
            [torch.empty(2, 2, 7), torch.empty(3, 3, 7)],
            ragged_dims=(0, 1),
        )
        packed_values = torch.randn(5, 13).transpose(0, 1)
        assert not packed_values.is_contiguous()

        output = reference.packed_with_static_tail(packed_values)

        assert output.concat is packed_values
        assert output.concat.stride() == packed_values.stride()
        assert output.ragged_dims == (0, 1)
        assert output.packed_dim_order == (0, 1, 2)
        assert output._packed_sizes == (4, 9)
        assert [tuple(element.shape) for element in output] == [(2, 2, 5), (3, 3, 5)]

    def test_replaces_nonleading_packed_static_dims_without_copy(self):
        reference = NestedTensor(
            [torch.empty(2, 2, 3), torch.empty(2, 4, 3)],
            ragged_dims=(1,),
            padding_value=-2.5,
            mask_value=True,
        )
        leaf = torch.randn(6, 2, 7, dtype=torch.float64, requires_grad=True)
        packed_values = leaf.square()

        with nested_execution_guard(
            forbid_iteration=True,
            forbid_storage_map=True,
            forbid_eager_fallback=True,
            forbid_padded_materialization=True,
            forbid_dense_repack=True,
        ):
            output = reference.packed_with_static_tail(packed_values)

        assert output.concat is packed_values
        assert output.concat.stride() == packed_values.stride()
        assert output.ragged_dims == (1,)
        assert output.packed_dim_order == (1, 0, 2)
        assert output.shape == (2, 2, 4, 7)
        assert output.padding_value == -2.5
        assert output.mask_value is True
        assert output._offsets is reference._offsets
        assert output.ragged_level_offsets() is reference.ragged_level_offsets()
        assert output._packed_sizes is None
        assert output._element_shapes is None
        assert output.__tensor_flatten__()[1]["packed_sizes"] is None
        assert output.__tensor_flatten__()[1]["element_shapes"] is None
        expected = (
            packed_values[:2].permute(1, 0, 2),
            packed_values[2:].permute(1, 0, 2),
        )
        assert [tuple(element.shape) for element in output] == [(2, 2, 7), (2, 4, 7)]
        for actual, wanted in zip(output, expected):
            assert_close(actual, wanted)
        gradient = torch.autograd.grad(output.concat.sum(), leaf)[0]
        assert_close(gradient, 2 * leaf)

    def test_nonleading_static_tail_supports_fake_and_copy(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        reference = NestedTensor(
            [torch.empty(2, 2, 3), torch.empty(2, 4, 3)],
            ragged_dims=(1,),
        )
        output = reference.packed_with_static_tail(torch.randn(6, 2, 7))

        copies = (
            copy.copy(output),
            copy.deepcopy(output),
            output.clone(),
            output.detach(),
            pickle.loads(pickle.dumps(output)),
        )
        for copied in copies:
            assert copied.ragged_dims == (1,)
            assert copied.packed_dim_order == (1, 0, 2)
            assert copied.shape == output.shape
            assert copied.__tensor_flatten__()[1]["packed_sizes"] is None
            assert copied.__tensor_flatten__()[1]["element_shapes"] is None
            assert_close(copied.concat, output.concat)

        with fake_tensor_mod.FakeTensorMode() as mode:
            fake_reference = mode.from_tensor(reference)
            fake_values = mode.from_tensor(torch.empty(6, 2, 7))
            fake_output = fake_reference.packed_with_static_tail(fake_values)

        assert fake_output.concat is fake_values
        assert fake_tensor_mod.is_fake(fake_output.concat)
        assert fake_tensor_mod.is_fake(fake_output._offsets)
        assert fake_tensor_mod.is_fake(fake_output._physical_shape)
        assert fake_output.ragged_dims == (1,)
        assert fake_output.packed_dim_order == (1, 0, 2)
        assert fake_output.shape == (2, 2, 4, 7)
        assert fake_output.__tensor_flatten__()[1]["packed_sizes"] is None
        assert fake_output.__tensor_flatten__()[1]["element_shapes"] is None

    def test_rejects_noncanonical_or_missing_ragged_topology(self):
        reference = NestedTensor([torch.empty(2, 3), torch.empty(4, 3)]).permute(0, 2, 1)
        permuted_explicit = NestedTensor(
            [torch.empty(1, 4, 2, 2), torch.empty(1, 4, 3, 3)],
            ragged_dims=(2, 3),
        )
        scalar_reference = NestedTensor([torch.tensor(1.0), torch.tensor(2.0)])

        with pytest.raises(ValueError, match="canonical packed order"):
            reference.packed_with_static_tail(torch.empty(reference.concat.shape[0], 5))
        with pytest.raises(ValueError, match="canonical packed order"):
            permuted_explicit.packed_with_static_tail(torch.empty(permuted_explicit.concat.shape[0], 5))
        with pytest.raises(ValueError, match="at least one ragged dimension"):
            scalar_reference.packed_with_static_tail(torch.empty(2, 5))

    @pytest.mark.parametrize("packed_shape", [(6, 7), (6, 2, 7, 1)])
    def test_nonleading_layout_rejects_changed_static_rank(self, packed_shape):
        reference = NestedTensor(
            [torch.empty(2, 2, 3), torch.empty(2, 4, 3)],
            ragged_dims=(1,),
        )

        with pytest.raises(ValueError, match="same number of packed static dimensions"):
            reference.packed_with_static_tail(torch.empty(packed_shape))

    def test_rejects_invalid_values(self):
        reference = NestedTensor([torch.empty(2), torch.empty(4)])

        with pytest.raises(TypeError, match="dense Tensor"):
            reference.packed_with_static_tail(object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="leading packed dimension"):
            reference.packed_with_static_tail(torch.tensor(1.0))
        with pytest.raises(ValueError, match="reference packed length"):
            reference.packed_with_static_tail(torch.empty(5, 3))
        with pytest.raises(TypeError, match="torch.strided"):
            reference.packed_with_static_tail(torch.empty(6, 3).to_sparse())

    def test_supports_fake_values(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        mode = fake_tensor_mod.FakeTensorMode()
        reference = NestedTensor([torch.empty(2), torch.empty(4)])
        packed_values = mode.from_tensor(torch.empty(6, 3, 5))

        output = reference.packed_with_static_tail(packed_values)

        assert output.concat is packed_values
        assert output.shape == (2, 4, 3, 5)
        assert fake_tensor_mod.is_fake(output._offsets)
        assert fake_tensor_mod.is_fake(output._physical_shape)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_fullgraph_forward_backward(self):
        reference = NestedTensor([torch.empty(2), torch.empty(4)])
        packed_values = torch.randn(6, 3, requires_grad=True)
        compiled = torch.compile(
            lambda ref, values: ref.packed_with_static_tail(values).concat.square().sum(),
            backend="aot_eager",
            fullgraph=True,
        )

        loss = compiled(reference, packed_values)
        loss.backward()

        assert type(loss.grad_fn).__name__ == "CompiledFunctionBackward"
        assert_close(loss, packed_values.detach().square().sum())
        assert_close(packed_values.grad, 2 * packed_values)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize("ragged_dim", [1, 2], ids=["dim1", "dim2"])
    def test_nonleading_fullgraph_derived_values_avoid_duck_shape_collision(self, ragged_dim):
        from torch._dynamo.testing import CompileCounter

        counter = CompileCounter()

        def consume(reference, weight):
            projected = reference.concat @ weight
            output = reference.packed_with_static_tail(projected)
            return output.concat.square().sum(), output.ragged_level_offsets()

        compiled = torch.compile(consume, backend=counter, fullgraph=True, dynamic=True)
        for index, lengths in enumerate(((2, 3), (3, 5))):
            elements = (
                [torch.empty(2, length, 5) for length in lengths]
                if ragged_dim == 1
                else [torch.empty(2, 4, length, 5) for length in lengths]
            )
            template = NestedTensor(elements, ragged_dims=(ragged_dim,))
            values = torch.randn_like(template.concat, requires_grad=True)
            reference = template.packed_like(values)
            weight = torch.randn(5, 7, requires_grad=True)
            if index == 0:
                # The first trace has ``sum(lengths) == source channels == 5``.
                # The in-graph projection replaces that source tail with 7 channels;
                # packed dim 0 must remain independently dynamic on the next layout.
                assert values.shape[0] == values.shape[-1]

            loss, offsets = compiled(reference, weight)
            actual_gradients = torch.autograd.grad(loss, (values, weight))
            expected_values = values.detach().requires_grad_()
            expected_weight = weight.detach().requires_grad_()
            expected_loss = (expected_values @ expected_weight).square().sum()
            expected_gradients = torch.autograd.grad(expected_loss, (expected_values, expected_weight))

            assert_close(loss, expected_loss)
            assert_close(offsets, reference.ragged_level_offsets())
            for actual, expected in zip(actual_gradients, expected_gradients):
                assert_close(actual, expected)

        assert counter.frame_count == 1


class TestPackedWithLengths:

    def test_replaces_lengths_without_copy_and_preserves_runtime_config(self):
        reference = NestedTensor(
            [torch.empty(1, 2), torch.empty(1, 2)],
            batch_first=False,
            padding_value=-3,
            mask_value=True,
        )
        lengths = torch.tensor([0, 3], dtype=torch.int32)
        leaf = torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True)
        packed_values = leaf.square()

        output = reference.packed_with_lengths(packed_values, lengths)

        assert output.concat is packed_values
        assert output.ragged_dims == (0,)
        assert output.packed_dim_order == (0, 1, 2)
        assert output.shape == (3, 2, 2, 4)
        assert output.batch_first is False
        assert output.padding_value == -3
        assert output.mask_value is True
        assert output._packed_sizes is None
        assert output._element_shapes is None
        assert_close(output._offsets, torch.tensor([0, 0, 3]))
        assert [tuple(element.shape) for element in output] == [(0, 2, 4), (3, 2, 4)]
        assert output._packed_sizes is None
        assert output._element_shapes is None
        gradient = torch.autograd.grad(output.concat.sum(), leaf)[0]
        assert_close(gradient, 2 * leaf)

    def test_tensor_backed_metadata_supports_iteration_and_pickle(self):
        reference = NestedTensor([torch.empty(1), torch.empty(1), torch.empty(1)])
        output = reference.packed_with_lengths(torch.randn(5, 4), torch.tensor([0, 2, 3]))
        restored = pickle.loads(pickle.dumps(output))

        assert output._packed_sizes is None
        assert output._element_shapes is None
        assert [tuple(element.shape) for element in output] == [(0, 4), (2, 4), (3, 4)]
        assert restored._packed_sizes is None
        assert restored._element_shapes is None
        assert [tuple(element.shape) for element in restored] == [(0, 4), (2, 4), (3, 4)]
        assert_close(restored.concat, output.concat)

    def test_supports_empty_batch(self):
        reference = NestedTensor([torch.empty(2)])[:0]
        packed_values = torch.empty(0, 3)

        output = reference.packed_with_lengths(packed_values, torch.empty(0, dtype=torch.long))

        assert output.concat is packed_values
        assert output.shape == (0, 0, 3)
        assert output.ragged_dims == (0,)
        assert output._offsets.tolist() == [0]
        assert output._physical_shape.shape == (0, 2)
        assert output._packed_sizes is None
        assert output._element_shapes is None

    def test_large_eager_batch_does_not_create_compile_max_length_binding(self):
        batch_size = 65
        reference = NestedTensor([torch.empty(1) for _ in range(batch_size)])
        lengths = torch.ones(batch_size, dtype=torch.long)

        output = reference.packed_with_lengths(torch.empty(batch_size, 3), lengths)

        assert output.shape == (batch_size, 1, 3)
        assert "_compile_max_length_binding" not in vars(output)
        assert output._packed_sizes is None
        assert output._element_shapes is None

    @pytest.mark.parametrize(
        ("lengths", "error", "match"),
        [
            (object(), TypeError, "dense Tensor"),
            (torch.tensor([2.0, 3.0]), TypeError, "integer dtype"),
            (torch.tensor([True, False]), TypeError, "integer dtype"),
            (torch.tensor([[2, 3]]), ValueError, "one-dimensional"),
            (torch.tensor([5]), ValueError, "one value per batch"),
            (torch.tensor([2, -1]), ValueError, "non-negative"),
            (torch.tensor([2, 2]), ValueError, "must sum"),
            (torch.empty(2, dtype=torch.long, device="meta"), ValueError, "on CPU"),
        ],
    )
    def test_validates_lengths(self, lengths, error, match):
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        packed_values = torch.empty(5, 3)

        with pytest.raises(error, match=match):
            reference.packed_with_lengths(packed_values, lengths)  # type: ignore[arg-type]

    def test_rejects_invalid_values(self):
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        lengths = torch.tensor([2, 3])

        with pytest.raises(TypeError, match="dense Tensor"):
            reference.packed_with_lengths(object(), lengths)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="leading packed dimension"):
            reference.packed_with_lengths(torch.tensor(1.0), lengths)
        with pytest.raises(TypeError, match="torch.strided"):
            reference.packed_with_lengths(torch.empty(5, 3).to_sparse(), lengths)

    def test_supports_fake_values_with_concrete_cpu_lengths(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        mode = fake_tensor_mod.FakeTensorMode()
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        lengths = torch.tensor([2, 3])
        packed_values = mode.from_tensor(torch.empty(5, 4))

        output = reference.packed_with_lengths(packed_values, lengths)

        assert output.concat is packed_values
        assert output.shape == (2, 3, 4)
        assert fake_tensor_mod.is_fake(output._offsets)
        assert fake_tensor_mod.is_fake(output._physical_shape)
        assert output._packed_sizes is None
        assert output._element_shapes is None

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_eager_packed_with_lengths_outputs_are_dynamic_compile_inputs(self):
        from torch._dynamo.testing import CompileCounter

        reference = NestedTensor([torch.empty(1), torch.empty(1), torch.empty(1)])
        counter = CompileCounter()
        compiled = torch.compile(
            lambda nested: (
                nested.concat.sum(),
                nested.packed_batch_indices(),
                nested.packed_local_indices(),
                nested.ragged_level_offsets(),
            ),
            backend=counter,
            fullgraph=True,
            dynamic=True,
        )
        layouts = (
            reference.packed_with_lengths(torch.arange(5), torch.tensor([0, 2, 3])),
            reference.packed_with_lengths(torch.arange(7), torch.tensor([4, 1, 2])),
        )

        for nested in layouts:
            total, batch_indices, local_indices, offsets = compiled(nested)
            lengths = nested._physical_shape[:, 0]
            assert_close(total, nested.concat.sum())
            assert_close(batch_indices, torch.repeat_interleave(torch.arange(3), lengths))
            assert_close(local_indices, torch.cat([torch.arange(int(length)) for length in lengths]))
            assert_close(offsets, torch.nn.functional.pad(lengths.cumsum(0), (1, 0)))

        assert counter.frame_count == 1

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_leading_tensor_backed_layout_avoids_duck_shape_collision(self):
        from torch._dynamo.testing import CompileCounter

        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        counter = CompileCounter()

        def consume(nested):
            output = nested.packed_like(nested.concat.square())
            return output.concat.sum(), output.ragged_level_offsets()

        compiled = torch.compile(consume, backend=counter, fullgraph=True, dynamic=True)
        for index, lengths in enumerate((torch.tensor([2, 3]), torch.tensor([3, 4]))):
            values = torch.randn(int(lengths.sum()), 5, requires_grad=True)
            nested = reference.packed_with_lengths(values, lengths)
            if index == 0:
                assert values.shape[0] == values.shape[-1]

            loss, offsets = compiled(nested)
            loss.backward()

            assert_close(offsets, nested.ragged_level_offsets())
            assert_close(values.grad, 2 * values)

        assert counter.frame_count == 1

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize("tensor_backed", [False, True])
    def test_fullgraph_vector_tail_broadcast(self, tensor_backed):
        elements = [torch.randn(2, 4), torch.randn(3, 4)]
        if tensor_backed:
            reference = NestedTensor([torch.empty(1), torch.empty(1)])
            nested = reference.packed_with_lengths(torch.cat(elements), torch.tensor([2, 3]))
        else:
            nested = NestedTensor(elements, ragged_dims=(0,))
        vector = torch.randn(4)
        compiled = torch.compile(
            lambda value, tail: (value + tail).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        output = compiled(nested, vector)

        assert_close(output, torch.cat([element + vector for element in elements]))

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize("tensor_backed", [False, True])
    def test_fullgraph_layer_norm_static_tail(self, tensor_backed):
        elements = [torch.randn(2, 4), torch.randn(3, 4)]
        if tensor_backed:
            reference = NestedTensor([torch.empty(1), torch.empty(1)])
            nested = reference.packed_with_lengths(torch.cat(elements), torch.tensor([2, 3]))
        else:
            nested = NestedTensor(elements, ragged_dims=(0,))
        compiled = torch.compile(
            lambda value: torch.layer_norm(value, (4,)).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        output = compiled(nested)

        assert_close(output, torch.cat([torch.layer_norm(element, (4,)) for element in elements]))

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_fullgraph_binary_tensor_backed_layout_validation(self):
        from torch._dynamo.testing import CompileCounter

        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        counter = CompileCounter()
        compiled = torch.compile(
            lambda lhs, rhs: (lhs + rhs).concat,
            backend=counter,
            fullgraph=True,
            dynamic=True,
        )

        for lengths in (torch.tensor([2, 3]), torch.tensor([1, 2])):
            lhs_values = torch.randn(int(lengths.sum()), 4)
            rhs_values = torch.randn(int(lengths.sum()), 4)
            lhs = reference.packed_with_lengths(lhs_values, lengths)
            rhs = reference.packed_with_lengths(rhs_values, lengths.clone())
            assert_close(compiled(lhs, rhs), lhs_values + rhs_values)

        mismatched_lhs = reference.packed_with_lengths(torch.randn(5, 4), torch.tensor([2, 3]))
        mismatched_rhs = reference.packed_with_lengths(torch.randn(5, 4), torch.tensor([3, 2]))
        with pytest.raises(RuntimeError, match="NestedTensor ragged offsets must match"):
            compiled(mismatched_lhs, mismatched_rhs)

        assert counter.frame_count == 1

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_fullgraph_binary_explicit_list_inputs(self):
        lhs_elements = [torch.randn(2, 4), torch.randn(3, 4)]
        rhs_elements = [torch.randn(2, 4), torch.randn(3, 4)]
        lhs = NestedTensor(lhs_elements, ragged_dims=(0,))
        rhs = NestedTensor(rhs_elements, ragged_dims=(0,))
        compiled = torch.compile(
            lambda left, right: (left + right).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        output = compiled(lhs, rhs)

        assert_close(output, torch.cat([left + right for left, right in zip(lhs_elements, rhs_elements)]))

    @pytest.mark.parametrize("operation", [torch.dot, torch.vdot])
    def test_tensor_backed_dot_returns_scalar_elements(self, operation):
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        for lengths in (torch.tensor([3, 3]), torch.tensor([2, 3])):
            lhs_values = torch.randn(int(lengths.sum()))
            rhs_values = torch.randn(int(lengths.sum()))
            lhs = reference.packed_with_lengths(lhs_values, lengths)
            rhs = reference.packed_with_lengths(rhs_values, lengths.clone())

            output = operation(lhs, rhs)
            expected = torch.stack([operation(left, right) for left, right in zip(lhs._storage, rhs._storage)])

            assert output.ragged_dims == ()
            assert_close(output.concat, expected)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize("operation", [torch.dot, torch.vdot])
    def test_tensor_backed_uniform_dot_fullgraph_and_fake(self, operation):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        lengths = torch.tensor([3, 3])
        lhs_values = torch.randn(6)
        rhs_values = torch.randn(6)
        lhs = reference.packed_with_lengths(lhs_values, lengths)
        rhs = reference.packed_with_lengths(rhs_values, lengths.clone())
        compiled = torch.compile(
            lambda left, right: operation(left, right).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        output = compiled(lhs, rhs)
        expected = torch.stack([operation(left, right) for left, right in zip(lhs._storage, rhs._storage)])
        assert_close(output, expected)

        mode = fake_tensor_mod.FakeTensorMode()
        fake_lhs = reference.packed_with_lengths(mode.from_tensor(lhs_values), lengths)
        fake_rhs = reference.packed_with_lengths(mode.from_tensor(rhs_values), lengths)
        fake_output = operation(fake_lhs, fake_rhs)
        assert fake_tensor_mod.is_fake(fake_output.concat)
        assert fake_output.concat.shape == (2,)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize(
        "case",
        [
            "view",
            "select",
            "slice",
            "pad",
            "constant_pad",
            "broadcast_tensors",
            "einsum",
            "ragged_softmax",
            "squeeze",
            "embedding_bag",
        ],
    )
    def test_tensor_backed_staged_consumers_fail_compile_clearly(self, case):
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        lengths = torch.tensor([2, 3])
        nested = reference.packed_with_lengths(torch.randn(5, 4), lengths)
        if case == "view":

            def operation(value):
                return value.view(2, 3, 4)

        elif case == "select":

            def operation(value):
                return value[:, 0]

        elif case == "slice":

            def operation(value):
                return value[:, :2]

        elif case == "pad":
            nested = reference.packed_with_lengths(torch.randn(5), lengths)

            def operation(value):
                return torch.nn.functional.pad(value, (1, 1))

        elif case == "constant_pad":
            nested = reference.packed_with_lengths(torch.randn(5), lengths)

            def operation(value):
                return torch.ops.aten.constant_pad_nd.default(value, (1, 1), 0)

        elif case == "broadcast_tensors":

            def operation(value):
                return torch.broadcast_tensors(value, value)

        elif case == "einsum":
            weight = torch.randn(2, 4, 3)

            def operation(value):
                return torch.einsum("bls,hsk->bhlk", value, weight)

        elif case == "ragged_softmax":

            def operation(value):
                return torch.softmax(value, dim=1)

        elif case == "squeeze":
            nested = reference.packed_with_lengths(torch.randn(5, 1), lengths)

            def operation(value):
                return torch.squeeze(value)

        else:
            nested = reference.packed_with_lengths(torch.randint(0, 8, (5,)), lengths)
            weight = torch.randn(8, 4)

            def operation(value):
                return torch.nn.functional.embedding_bag(value, weight)

        compiled = torch.compile(operation, backend="aot_eager", fullgraph=True, dynamic=True)

        with pytest.raises(Exception, match="tensor-backed"):
            compiled(nested)

    def test_rejects_data_less_fake_lengths(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        mode = fake_tensor_mod.FakeTensorMode()
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        fake_lengths = mode.from_tensor(torch.tensor([2, 3]))

        with pytest.raises(ValueError, match="FakeTensor lengths require FakeTensor packed_values"):
            reference.packed_with_lengths(torch.empty(5, 4), fake_lengths)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_fullgraph_backward_reuses_fixed_batch_across_lengths(self):
        from torch._dynamo.testing import CompileCounter

        reference = NestedTensor([torch.empty(1, 2), torch.empty(1, 2)])
        counter = CompileCounter()
        compiled = torch.compile(
            lambda ref, values, lengths: ref.packed_with_lengths(values, lengths).concat.square().sum(),
            backend=counter,
            fullgraph=True,
            dynamic=True,
        )

        for lengths_tuple in ((2, 4), (3, 5)):
            lengths = torch.tensor(lengths_tuple)
            packed_values = torch.randn(sum(lengths_tuple), 7, requires_grad=True)
            loss = compiled(reference, packed_values, lengths)
            loss.backward()
            assert_close(packed_values.grad, 2 * packed_values)

        assert counter.frame_count == 1

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_fullgraph_batch_32_chains_tensor_topology_in_one_graph(self):
        from torch._dynamo.testing import CompileCounter

        batch_size = 32
        reference = NestedTensor([torch.empty(1) for _ in range(batch_size)])
        counter = CompileCounter()

        def consume(ref, values, lengths):
            output = ref.packed_with_lengths(values, lengths)
            loss = output.concat.square().sum()
            return (
                loss,
                output.packed_batch_indices(),
                output.packed_local_indices(),
                output.ragged_level_offsets(),
            )

        compiled = torch.compile(consume, backend=counter, fullgraph=True, dynamic=True)
        layouts = (
            torch.arange(batch_size, dtype=torch.long).remainder(4),
            torch.arange(batch_size - 1, -1, -1, dtype=torch.long).remainder(5),
            torch.arange(batch_size, dtype=torch.long).mul(3).remainder(7),
        )
        for lengths in layouts:
            packed_values = torch.randn(int(lengths.sum()), 3, requires_grad=True)
            loss, batch_indices, local_indices, offsets = compiled(reference, packed_values, lengths)
            loss.backward()

            expected_batch = torch.repeat_interleave(torch.arange(batch_size), lengths)
            expected_local = torch.cat([torch.arange(int(length)) for length in lengths])
            expected_offsets = torch.nn.functional.pad(lengths.cumsum(0), (1, 0))
            assert_close(batch_indices, expected_batch)
            assert_close(local_indices, expected_local)
            assert_close(offsets, expected_offsets)
            assert_close(packed_values.grad, 2 * packed_values)

        assert counter.frame_count == 1

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_fullgraph_output_supports_iteration_and_pickle(self):
        reference = NestedTensor([torch.empty(1), torch.empty(1), torch.empty(1)])
        compiled = torch.compile(
            lambda ref, values, lengths: ref.packed_with_lengths(values, lengths),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        lengths = torch.tensor([0, 2, 3])
        packed_values = torch.randn(5, 4, requires_grad=True)

        output = compiled(reference, packed_values, lengths)
        restored = pickle.loads(pickle.dumps(output))

        assert output.concat.data_ptr() == packed_values.data_ptr()
        assert output._packed_sizes is None
        assert output._element_shapes is None
        assert [tuple(element.shape) for element in output] == [(0, 4), (2, 4), (3, 4)]
        assert output._packed_sizes is None
        assert output._element_shapes is None
        assert [tuple(element.shape) for element in restored] == [(0, 4), (2, 4), (3, 4)]
        assert_close(restored.concat, packed_values)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compiled_output_reenters_fullgraph_without_python_topology(self):
        from torch._dynamo.testing import CompileCounter

        reference = NestedTensor([torch.empty(1), torch.empty(1), torch.empty(1)])
        producer = torch.compile(
            lambda ref, values, lengths: ref.packed_with_lengths(values, lengths),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        counter = CompileCounter()
        consumer = torch.compile(
            lambda nested: (
                nested.concat.square().sum(),
                nested.packed_batch_indices(),
                nested.packed_local_indices(),
                nested.ragged_level_offsets(),
            ),
            backend=counter,
            fullgraph=True,
            dynamic=True,
        )

        # Keep the wrapper and packed shapes fixed here so this test isolates
        # topology values from ordinary tensor-shape specialization.
        for lengths in (torch.tensor([0, 2, 3]), torch.tensor([1, 1, 3])):
            packed_values = torch.randn(int(lengths.sum()), 4)
            output = producer(reference, packed_values, lengths)
            loss, batch_indices, local_indices, offsets = consumer(output)

            assert output._packed_sizes is None
            assert output._element_shapes is None
            assert_close(loss, packed_values.square().sum())
            assert_close(batch_indices, torch.repeat_interleave(torch.arange(3), lengths))
            assert_close(local_indices, torch.cat([torch.arange(int(length)) for length in lengths]))
            assert_close(offsets, torch.nn.functional.pad(lengths.cumsum(0), (1, 0)))

        assert counter.frame_count == 1

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_fullgraph_runtime_validates_length_values(self):
        reference = NestedTensor([torch.empty(1, 2), torch.empty(1, 2)])
        compiled = torch.compile(
            lambda ref, values, lengths: ref.packed_with_lengths(values, lengths).concat.square().sum(),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        compiled(reference, torch.randn(6, 7), torch.tensor([2, 4]))

        with pytest.raises(RuntimeError, match="lengths must be non-negative"):
            compiled(reference, torch.randn(6, 7), torch.tensor([-1, 7]))
        with pytest.raises(RuntimeError, match="lengths must sum"):
            compiled(reference, torch.randn(6, 7), torch.tensor([2, 3]))


class TestPackedWithSquareLengths:

    def test_rebuilds_square_topology_without_copy_and_preserves_runtime_config(self):
        class DerivedNestedTensor(NestedTensor):
            pass

        reference = DerivedNestedTensor(
            [torch.empty(1), torch.empty(1)],
            batch_first=False,
            padding_value=-3,
            mask_value=True,
        )
        lengths = torch.tensor([0, 3], dtype=torch.int32)
        leaf = torch.randn(9, 2, 4, dtype=torch.float64, requires_grad=True)
        packed_values = leaf.square()

        output = reference.packed_with_square_lengths(packed_values, lengths)

        assert type(output) is DerivedNestedTensor
        assert output.concat is packed_values
        assert output.ragged_dims == (0, 1)
        assert output.packed_dim_order == (0, 1, 2, 3)
        assert output.shape == (3, 2, 3, 2, 4)
        assert output.batch_first is False
        assert output.padding_value == -3
        assert output.mask_value is True
        assert output._packed_sizes is None
        assert output._element_shapes is None
        assert_close(output.packed_offsets(), torch.tensor([0, 0, 9]))
        assert_close(output.ragged_level_offsets(0), torch.tensor([0, 0, 3]))
        assert_close(output.ragged_level_offsets(1), torch.tensor([0, 3, 6, 9]))
        assert [tuple(element.shape) for element in output] == [(0, 0, 2, 4), (3, 3, 2, 4)]
        assert_close(torch.autograd.grad(output.concat.sum(), leaf)[0], 2 * leaf)

    def test_supports_scalar_tail_and_all_zero_lengths(self):
        reference = NestedTensor([torch.empty(1), torch.empty(1), torch.empty(1)])

        scalar_tail = reference.packed_with_square_lengths(torch.arange(13), torch.tensor([2, 0, 3]))
        all_zero = reference.packed_with_square_lengths(torch.empty(0, 5), torch.zeros(3, dtype=torch.long))

        assert scalar_tail.packed_dim_order == (0, 1)
        assert [tuple(element.shape) for element in scalar_tail] == [(2, 2), (0, 0), (3, 3)]
        assert all_zero.shape == (3, 0, 0, 5)
        assert_close(all_zero.packed_offsets(), torch.zeros(4, dtype=torch.long))
        assert_close(all_zero.ragged_level_offsets(0), torch.zeros(4, dtype=torch.long))
        assert_close(all_zero.ragged_level_offsets(1), torch.zeros(1, dtype=torch.long))
        assert [tuple(element.shape) for element in all_zero] == [(0, 0, 5)] * 3

    def test_tensor_metadata_survives_copy_and_pickle(self):
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        output = reference.packed_with_square_lengths(torch.randn(13, 4), torch.tensor([2, 3]))

        for rebuilt in (copy.copy(output), copy.deepcopy(output), pickle.loads(pickle.dumps(output))):
            assert rebuilt._packed_sizes is None
            assert rebuilt._element_shapes is None
            assert [tuple(element.shape) for element in rebuilt] == [(2, 2, 4), (3, 3, 4)]
            assert_close(rebuilt.packed_offsets(), torch.tensor([0, 4, 13]))
            assert_close(rebuilt.ragged_level_offsets(0), torch.tensor([0, 2, 5]))
            assert_close(rebuilt.ragged_level_offsets(1), torch.tensor([0, 2, 4, 7, 10, 13]))

    def test_rejects_invalid_inputs(self):
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        values = torch.empty(13, 4)

        with pytest.raises(TypeError, match="lengths must be a dense Tensor"):
            reference.packed_with_square_lengths(values, [2, 3])  # type: ignore[arg-type]
        for dtype in (torch.float32, torch.complex64, torch.bool):
            with pytest.raises(TypeError, match="integer dtype"):
                reference.packed_with_square_lengths(values, torch.tensor([2, 3], dtype=dtype))
        with pytest.raises(ValueError, match="one-dimensional"):
            reference.packed_with_square_lengths(values, torch.tensor([[2, 3]]))
        with pytest.raises(ValueError, match="one value per batch element"):
            reference.packed_with_square_lengths(values, torch.tensor([2, 3, 0]))
        with pytest.raises(ValueError, match="non-negative"):
            reference.packed_with_square_lengths(values, torch.tensor([-2, 3]))
        with pytest.raises(ValueError, match="squared lengths must sum"):
            reference.packed_with_square_lengths(values, torch.tensor([1, 3]))
        with pytest.raises(ValueError, match="lengths must be on CPU"):
            reference.packed_with_square_lengths(values, torch.empty(2, dtype=torch.long, device="meta"))
        with pytest.raises(TypeError, match="packed_values must be a dense Tensor"):
            reference.packed_with_square_lengths(object(), torch.tensor([2, 3]))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="leading packed dimension"):
            reference.packed_with_square_lengths(torch.tensor(1.0), torch.tensor([2, 3]))
        with pytest.raises(TypeError, match="torch.strided"):
            reference.packed_with_square_lengths(values.to_sparse(), torch.tensor([2, 3]))

    def test_supports_fake_values_with_concrete_cpu_lengths(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        mode = fake_tensor_mod.FakeTensorMode()
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        fake_values = mode.from_tensor(torch.empty(13, 2, 4))

        output = reference.packed_with_square_lengths(fake_values, torch.tensor([2, 3]))

        assert output.concat is fake_values
        assert output.shape == (2, 3, 3, 2, 4)
        assert fake_tensor_mod.is_fake(output._offsets)
        assert fake_tensor_mod.is_fake(output._physical_shape)
        assert fake_tensor_mod.is_fake(output.ragged_level_offsets(0))
        assert fake_tensor_mod.is_fake(output.ragged_level_offsets(1))

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_fullgraph_reuses_one_graph_across_square_lengths_with_backward(self):
        from torch._dynamo.testing import CompileCounter

        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        counter = CompileCounter()

        def consume(ref, values, lengths):
            output = ref.packed_with_square_lengths(values, lengths)
            return (
                output.concat.square().sum(),
                output.packed_offsets(),
                output.ragged_level_offsets(0),
                output.ragged_level_offsets(1),
            )

        compiled = torch.compile(consume, backend=counter, fullgraph=True, dynamic=True)
        for lengths in (torch.tensor([2, 3]), torch.tensor([1, 4])):
            packed_values = torch.randn(int(lengths.square().sum()), 4, requires_grad=True)
            loss, packed_offsets, level_zero, level_one = compiled(reference, packed_values, lengths)
            loss.backward()

            expected_rows = torch.repeat_interleave(lengths, lengths)
            assert_close(packed_offsets, torch.nn.functional.pad(lengths.square().cumsum(0), (1, 0)))
            assert_close(level_zero, torch.nn.functional.pad(lengths.cumsum(0), (1, 0)))
            assert_close(level_one, torch.nn.functional.pad(expected_rows.cumsum(0), (1, 0)))
            assert_close(packed_values.grad, 2 * packed_values)

        assert counter.frame_count == 1

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_fullgraph_runtime_validates_length_values(self):
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        compiled = torch.compile(
            lambda ref, values, lengths: ref.packed_with_square_lengths(values, lengths).concat.sum(),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        compiled(reference, torch.randn(13, 4), torch.tensor([2, 3]))

        with pytest.raises(RuntimeError, match="lengths must be non-negative"):
            compiled(reference, torch.randn(13, 4), torch.tensor([-2, 3]))
        with pytest.raises(RuntimeError, match="squared lengths must sum"):
            compiled(reference, torch.randn(13, 4), torch.tensor([1, 3]))


# ---------------------------------------------------------------------------
# Stable AOTAutograd cache metadata
# ---------------------------------------------------------------------------


class TestStableAOTCacheHash:

    @staticmethod
    def _tensor_backed(
        lengths,
        *,
        channels=4,
        dtype=torch.float32,
        requires_grad=False,
        batch_first=True,
        padding_value=0.0,
        mask_value=False,
        storage_offset=False,
    ):
        reference = NestedTensor(
            [torch.empty(1, channels) for _ in lengths],
            ragged_dims=(0,),
            batch_first=batch_first,
            padding_value=padding_value,
            mask_value=mask_value,
        )
        packed_length = sum(lengths)
        if storage_offset:
            backing = torch.randn(packed_length + 1, channels, dtype=dtype)
            values = backing[1:]
            values.requires_grad_(requires_grad)
        else:
            values = torch.randn(packed_length, channels, dtype=dtype, requires_grad=requires_grad)
        return reference.packed_with_lengths(values, torch.tensor(lengths, dtype=torch.long))

    def test_hash_ignores_values_and_distinguishes_static_metadata(self):
        base = self._tensor_backed((1, 4, 3))
        same_metadata = self._tensor_backed((1, 4, 3))

        assert base._stable_hash_for_caching() == same_metadata._stable_hash_for_caching()

        variants = (
            self._tensor_backed((1, 4, 3), dtype=torch.bfloat16),
            self._tensor_backed((1, 4, 3), requires_grad=True),
            self._tensor_backed((1, 4, 3), channels=5),
            self._tensor_backed((1, 4, 3), batch_first=False),
            self._tensor_backed((1, 4, 3), padding_value=-1.0),
            self._tensor_backed((1, 4, 3), mask_value=True),
            self._tensor_backed((1, 4, 3), storage_offset=True),
            NestedTensor([torch.empty(1, 1, 4), torch.empty(3, 3, 4)], ragged_dims=(0, 1)),
        )
        hashes = {base._stable_hash_for_caching(), *(variant._stable_hash_for_caching() for variant in variants)}
        assert len(hashes) == len(variants) + 1

    def test_tensor_backed_topology_values_do_not_change_hash(self):
        first = self._tensor_backed((1, 4, 3))
        second = self._tensor_backed((2, 4, 2))

        assert not torch.equal(first._offsets, second._offsets)
        assert first.shape == second.shape
        assert first._stable_hash_for_caching() == second._stable_hash_for_caching()

        first_pair = NestedTensor([torch.empty(1, 1, 4), torch.empty(3, 3, 4)], ragged_dims=(0, 1))
        second_pair = NestedTensor([torch.empty(3, 3, 4), torch.empty(1, 1, 4)], ragged_dims=(0, 1))
        assert not torch.equal(first_pair.ragged_level_offsets(1), second_pair.ragged_level_offsets(1))
        assert first_pair.shape == second_pair.shape
        assert first_pair._stable_hash_for_caching() == second_pair._stable_hash_for_caching()

    def test_legacy_python_topology_remains_layout_specific(self):
        first = NestedTensor(
            [torch.empty(length, 4) for length in (1, 4, 3)],
            ragged_dims=(0,),
        )
        second = NestedTensor(
            [torch.empty(length, 4) for length in (2, 4, 2)],
            ragged_dims=(0,),
        )

        first_context = first.__tensor_flatten__()[1]
        second_context = second.__tensor_flatten__()[1]
        assert first_context["packed_sizes"] is not None
        assert second_context["packed_sizes"] is not None
        assert first.shape == second.shape
        assert first.concat.shape == second.concat.shape
        assert first._stable_hash_for_caching() != second._stable_hash_for_caching()

    def test_symbolic_fake_hash_does_not_read_topology_values(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        symbolic_shapes = pytest.importorskip("torch.fx.experimental.symbolic_shapes")
        nested = NestedTensor([torch.empty(2, 2, 3), torch.empty(4, 4, 3)], ragged_dims=(0, 1))
        mode = fake_tensor_mod.FakeTensorMode(shape_env=symbolic_shapes.ShapeEnv())

        with mode:
            fake = mode.from_tensor(nested, static_shapes=False)
            digest = fake._stable_hash_for_caching()

        assert len(digest) == 32
        assert int(digest, 16) >= 0

    def test_hash_is_stable_across_python_hash_seeds(self):
        repository = Path(__file__).resolve().parents[2]
        script = textwrap.dedent("""
            import torch
            from danling.tensors import NestedTensor

            reference = NestedTensor([torch.empty(1, 4) for _ in range(3)], ragged_dims=(0,))
            nested = reference.packed_with_lengths(torch.zeros(8, 4), torch.tensor([1, 4, 3]))
            print(nested._stable_hash_for_caching())
            """)
        digests = []
        for seed in ("0", "314159"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=repository,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            digests.append(completed.stdout.strip().splitlines()[-1])

        assert digests[0] == digests[1]

    @pytest.mark.skipif(
        TORCH_VERSION < Version("2.13"), reason="stable AOTAutograd subclass cache hook is PyTorch 2.13+"
    )
    def test_inductor_aot_cache_reuses_dynamic_layout_without_warning(self, tmp_path):
        repository = Path(__file__).resolve().parents[2]
        script = textwrap.dedent("""
            import json
            import warnings

            import torch
            from danling.tensors import NestedTensor
            from torch._dynamo.utils import counters
            from torch._functorch import config
            from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache

            counters.clear()
            AOTAutogradCache.clear()

            def operation(nested):
                return nested.packed_like(nested.concat.square())

            stable_hash_warnings = []
            with config.patch(enable_autograd_cache=True):
                for lengths in ((2, 3), (3, 5)):
                    torch._dynamo.reset()
                    reference = NestedTensor([torch.empty(1, 4), torch.empty(1, 4)], ragged_dims=(0,))
                    nested = reference.packed_with_lengths(
                        torch.randn(sum(lengths), 4),
                        torch.tensor(lengths),
                    )
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        output = torch.compile(operation, fullgraph=True, dynamic=True)(nested)
                    assert output.concat.shape == (sum(lengths), 4)
                    stable_hash_warnings.extend(
                        str(warning.message)
                        for warning in caught
                        if "_stable_hash_for_caching" in str(warning.message)
                    )

            cache = counters["aot_autograd"]
            print(
                json.dumps(
                    {
                        "warnings": stable_hash_warnings,
                        "miss": cache["autograd_cache_miss"],
                        "saved": cache["autograd_cache_saved"],
                        "hit": cache["autograd_cache_hit"],
                    }
                )
            )
            """)
        environment = os.environ.copy()
        environment["TORCHINDUCTOR_CACHE_DIR"] = os.fspath(tmp_path / "inductor-cache")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        result = json.loads(completed.stdout.strip().splitlines()[-1])

        assert result == {"warnings": [], "miss": 1, "saved": 1, "hit": 1}


# ---------------------------------------------------------------------------
# Copy Semantics
# ---------------------------------------------------------------------------


class TestCopySemantics:

    def test_shallow_copy_shares_data(self):
        nt = NestedTensor(
            [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])],
            batch_first=False,
            padding_value=-1,
            mask_value=True,
        )
        shallow = copy.copy(nt)

        # Data is shared
        assert shallow._values.data_ptr() == nt._values.data_ptr()
        assert shallow._offsets.data_ptr() == nt._offsets.data_ptr()
        # Values are equal
        assert_close(shallow, nt)
        # State is preserved
        assert shallow.batch_first is False
        assert shallow.padding_value == -1
        assert shallow.mask_value is True

    def test_deep_copy_clones_data(self):
        nt = NestedTensor(
            [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])],
            batch_first=False,
            padding_value=-1,
            mask_value=True,
        )
        deep = copy.deepcopy(nt)

        # Data is NOT shared
        assert deep._values.data_ptr() != nt._values.data_ptr()
        assert deep._offsets.data_ptr() != nt._offsets.data_ptr()
        # Values are equal
        assert_close(deep, nt)
        # State is preserved
        assert deep.batch_first is False
        assert deep.padding_value == -1
        assert deep.mask_value is True
        # Mutation is independent
        deep._values.fill_(0)
        assert not torch.equal(deep._values, nt._values)

    def test_deep_copy_memo_reuse(self):
        nt = NestedTensor([torch.tensor([1.0, 2.0])])
        container = [nt, nt]
        cloned = copy.deepcopy(container)
        assert cloned[0] is cloned[1]  # memo ensures same object

    def test_pickle_roundtrip_restores_storage_cache_state(self):
        nt = NestedTensor([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
        _ = nt.tensor
        _ = nt.mask
        restored = pickle.loads(pickle.dumps(nt))
        assert isinstance(restored, NestedTensor)
        assert_close(restored, nt)
        assert restored._cached_tensor_view is None
        assert restored._cached_mask_view is None
        # Accessors should work after deserialization (regression for missing _cached_storage)
        assert_close(restored.tensor, nt.tensor)
        assert_close(restored.mask, nt.mask)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA map_location")
    def test_torch_load_map_location_keeps_metadata_on_cpu(self):
        nt = NestedTensor([torch.randn(2, 2, 3), torch.randn(4, 4, 3)], ragged_dims=(0, 1))
        buffer = io.BytesIO()
        torch.save(nt, buffer)
        buffer.seek(0)

        restored = torch.load(buffer, map_location="cuda:0", weights_only=False)

        assert restored.device.type == "cuda"
        assert restored._offsets.device.type == "cpu"
        assert restored._physical_shape.device.type == "cpu"
        assert all(offset.device.type == "cpu" for offset in restored._hierarchical_offsets)
        assert_close(restored.tensor.cpu(), nt.tensor)

    def test_pickle_roundtrip_preserves_noncanonical_permutation(self):
        nt = NestedTensor([torch.arange(6.0).reshape(2, 3)]).unsqueeze(1)

        restored = pickle.loads(pickle.dumps(nt))

        assert restored._permutation == nt._permutation
        assert restored._has_same_structure(nt)
        assert_close(restored.tensor, nt.tensor)

    def test_getstate_serializes_physical_shape(self):
        nt = NestedTensor([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
        state = nt.__getstate__()
        assert state["_state_version"] == NestedTensor._SERIALIZATION_VERSION
        assert state["_state_version"] == 3
        assert "_physical_shape" in state
        torch.testing.assert_close(state["_physical_shape"], nt._physical_shape)

    def test_setstate_rejects_unknown_state_version(self):
        nt = NestedTensor([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
        state = nt.__getstate__()
        state["_state_version"] = state["_state_version"] + 1
        with pytest.raises(ValueError, match="Unsupported NestedTensor state version"):
            restored = copy.copy(nt)
            restored.__setstate__(state)

    @pytest.mark.parametrize(
        "missing_key",
        ["_permutation", "_ragged_dims", "_packed_sizes", "_element_shapes", "_ragged_offsets"],
    )
    def test_setstate_rejects_missing_layout_metadata(self, missing_key):
        nt = NestedTensor([torch.arange(6.0).reshape(2, 3)]).unsqueeze(1)
        state = dict(nt.__getstate__())
        del state[missing_key]

        with pytest.raises(KeyError, match=missing_key):
            restored = copy.copy(nt)
            restored.__setstate__(state)


# ---------------------------------------------------------------------------
# From Factory Methods
# ---------------------------------------------------------------------------


class TestFromFactoryMethods:

    def test_from_tensor_mask_high_dimensional(self):
        padded = torch.tensor(
            [
                [[1, 2, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]],
                [[9, 8, 7, 0], [6, 5, 0, 0], [0, 0, 0, 0]],
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor(
            [
                [[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
                [[1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
            ],
            dtype=torch.bool,
        )
        nested = NestedTensor.from_tensor_mask(padded, mask)
        reference = NT(
            [
                torch.tensor([[1.0, 2.0], [3.0, 0.0]], dtype=torch.float32),
                torch.tensor([[9.0, 8.0, 7.0], [6.0, 5.0, 0.0]], dtype=torch.float32),
            ]
        )
        assert_close(nested, reference)

        padded_mask = ~mask
        nested_inverted = NestedTensor.from_tensor_mask(padded, padded_mask, mask_value=True)
        assert_close(nested_inverted, reference)

    def test_from_tensor_mask_1d_trims_with_mask(self):
        padded = torch.tensor([1, 2, 0, 0])
        mask = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
        nested = NestedTensor.from_tensor_mask(padded, mask)
        assert_close(nested, NT([torch.tensor([1, 2])]))
        padded_mask = ~mask
        nested_inverted = NestedTensor.from_tensor_mask(padded, padded_mask, mask_value=True)
        assert_close(nested_inverted, NT([torch.tensor([1, 2])]))

    def test_from_tensor_mask_1d_sparse_selects_masked_positions(self):
        padded = torch.tensor([10, 20, 30, 40])
        mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        nested = NestedTensor.from_tensor_mask(padded, mask)
        assert_close(nested, NT([torch.tensor([10, 30])]))

    def test_from_tensor_mask_batch_mismatch_raises(self):
        padded = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.float32)
        mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        with pytest.raises(ValueError, match="Tensor/mask batch dimension mismatch"):
            NestedTensor.from_tensor_mask(padded, mask)

    def test_from_tensor_mask_batched_scalar(self):
        padded = torch.tensor([1.0, 2.0, 0.0])
        mask = torch.tensor([True, True, False])
        output = NestedTensor.from_tensor_mask(padded, mask, batched=True)
        reference = NT([torch.tensor(1.0), torch.tensor(2.0)])
        assert_close(output, reference)

        padded_mask = ~mask
        output = NestedTensor.from_tensor_mask(padded, padded_mask, mask_value=True, batched=True)
        assert_close(output, reference)

    def test_from_tensor_mask_ndim3(self):
        padded = torch.tensor([[[1, 2, 0], [3, 0, 0]]])
        mask = torch.tensor([[[1, 1, 0], [1, 0, 0]]], dtype=torch.bool)
        output = NestedTensor.from_tensor_mask(padded, mask)
        assert output.tensor.shape == torch.Size([1, 2, 2])
        reference = NT([torch.tensor([[1, 2], [3, 0]])])
        assert_close(output, reference)

    def test_from_tensor_mask_ndim3_non_prefix_box_raises(self):
        padded = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor(
            [
                [[1, 0], [0, 1]],
                [[1, 1], [0, 0]],
            ],
            dtype=torch.bool,
        )
        with pytest.raises(ValueError, match="valid hierarchical ragged prefix"):
            NestedTensor.from_tensor_mask(padded, mask)

    def test_from_tensor_mask_2d_non_prefix_raises(self):
        padded = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32)
        mask = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
        with pytest.raises(ValueError, match="valid prefix mask"):
            NestedTensor.from_tensor_mask(padded, mask)

    def test_from_tensor_mask_channel_preserved(self):
        padded = torch.tensor([[[[1], [0]], [[2], [0]]]])
        mask = torch.tensor([[[1, 0], [1, 0]]], dtype=torch.bool)
        output = NestedTensor.from_tensor_mask(padded, mask)
        assert output.tensor.shape == torch.Size([1, 2, 1, 1])
        reference = NT([torch.tensor([[[1]], [[2]]])])
        assert_close(output, reference)

    def test_from_concatenated_extra_elements_raises(self):
        concat = torch.arange(4, dtype=torch.float32)
        shapes = (torch.Size([1, 1]), torch.Size([1, 1]))
        with pytest.raises(ValueError):
            NestedTensor.from_concatenated(concat, shapes)

    def test_from_concatenated_same_shapes(self):
        nested_tensor = NestedTensor([torch.randn(3, 5), torch.randn(3, 5)])
        concat, shapes = nested_tensor.concatenate()
        reconstructed = NestedTensor.from_concatenated(concat, shapes, **nested_tensor._meta())
        assert_close(reconstructed, nested_tensor)

    def test_from_concatenated_round_trip_multidim(self):
        nested_tensor = NestedTensor([torch.randn(2, 3, 4), torch.randn(2, 3, 4)])
        concat, shapes = nested_tensor.concatenate()
        output = NestedTensor.from_concatenated(concat, shapes)
        assert_close(output, nested_tensor)

    def test_from_concatenated_round_trip_mixed_shapes(self):
        nested_tensor = NestedTensor([torch.randn(2, 3, 4), torch.randn(1, 3, 4)])
        concat, shapes = nested_tensor.concatenate()
        output = NestedTensor.from_concatenated(concat, shapes)
        assert_close(output, nested_tensor)

    def test_from_concatenated_round_trip_non_leading_ragged_dim(self):
        nested_tensor = NestedTensor([torch.randn(2, 3, 4), torch.randn(2, 5, 4)])
        concat, shapes = nested_tensor.concatenate()
        output = NestedTensor.from_concatenated(concat, shapes)
        assert_close(output, nested_tensor)

    def test_concatenate_empty_preserves_dtype(self):
        nested_tensor = NestedTensor([], dtype=torch.float64)
        concat, shapes = nested_tensor.concatenate()
        assert concat.dtype == torch.float64
        assert concat.device == torch.device("cpu")
        assert shapes == ()


# ---------------------------------------------------------------------------
# Tensor / Mask Properties
# ---------------------------------------------------------------------------


class TestIndexing:

    def test_getitem_preserves_state(self):
        nested_tensor = NestedTensor(
            [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])],
            batch_first=False,
            padding_value=-1,
            mask_value=True,
        )

        tuple_indexed = nested_tensor[:, 0]
        assert isinstance(tuple_indexed, NestedTensor)
        assert tuple_indexed.batch_first is False
        assert tuple_indexed.padding_value == -1
        assert tuple_indexed.mask_value is True

        boolean_indexed = nested_tensor[nested_tensor.tensor > 0]
        assert boolean_indexed.batch_first is False

    def test_getitem_tuple_tensor_batch_index(self):
        nested_tensor = NestedTensor([torch.arange(3), torch.arange(5) + 10])
        batch_idx = torch.tensor([0, 1, 0])
        kv_idx = torch.tensor([0, 2, 1])
        assert torch.equal(nested_tensor[batch_idx, kv_idx], nested_tensor.tensor[batch_idx, kv_idx])

    def test_getitem_tensor_bool_batch_select(self):
        nested_tensor = NestedTensor([torch.arange(3), torch.arange(5) + 10, torch.arange(2) + 20])
        selected = nested_tensor[torch.tensor([True, False, True])]
        assert isinstance(selected, NestedTensor)
        assert_close(selected, NT([torch.arange(3), torch.arange(2) + 20]))

    def test_getitem_tensor_long_batch_select(self):
        nested_tensor = NestedTensor([torch.arange(3), torch.arange(5) + 10, torch.arange(2) + 20])
        selected = nested_tensor[torch.tensor([2, 0, 2])]
        assert isinstance(selected, NestedTensor)
        assert_close(selected, NT([torch.arange(2) + 20, torch.arange(3), torch.arange(2) + 20]))

    def test_getitem_tensor_bool_batch_length_mismatch_raises(self):
        nested_tensor = NestedTensor([torch.arange(3), torch.arange(5) + 10])
        with pytest.raises(IndexError, match="Boolean index has length 1 but batch size is 2"):
            _ = nested_tensor[torch.tensor([True])]

    def test_getitem_tuple_slice_rest(self):
        nested_tensor = NestedTensor([torch.arange(3), torch.arange(5)])
        sliced = nested_tensor[:, 1:]
        assert isinstance(sliced, NestedTensor)
        assert torch.equal(sliced[0], torch.tensor([1, 2]))
        assert torch.equal(sliced[1], torch.tensor([1, 2, 3, 4]))

    def test_tuple_getitem_with_leading_int(self):
        nested_tensor = NestedTensor([torch.tensor([1, 2]), torch.tensor([3])])
        assert nested_tensor[0, 0].item() == 1
        assert_close(nested_tensor[1, ...], torch.tensor([3]))

    def test_getitem_nested_index_length_mismatch(self):
        nested_tensor = NestedTensor([torch.tensor([1, 2]), torch.tensor([3])])
        nested_index = NestedTensor([torch.tensor([0])])
        with pytest.raises(ValueError, match="batch length mismatch"):
            _ = nested_tensor[nested_index]

    def test_setitem_same_shape(self):
        """Same-shape __setitem__ replaces the element correctly."""
        nt = NestedTensor([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
        _ = nt.tensor
        _ = nt.mask
        assert nt._cached_tensor_view is not None
        assert nt._cached_mask_view is not None
        nt[0] = torch.tensor([10.0, 20.0, 30.0])
        assert nt._cached_tensor_view is None
        assert nt._cached_mask_view is None
        assert_close(nt[0], torch.tensor([10.0, 20.0, 30.0]))
        assert_close(nt[1], torch.tensor([4.0, 5.0]))

    def test_getitem_second_dim_packed_is_view_of_values(self):
        nt = NestedTensor(
            [
                torch.arange(24.0).reshape(2, 3, 4),
                torch.arange(24.0, 56.0).reshape(2, 4, 4),
            ]
        )
        first = nt[0]
        first[0, 0, 0] = -1.0
        assert nt._values[0, 0, 0].item() == -1.0
        assert nt.tensor[0, 0, 0, 0].item() == -1.0
        assert nt.concat[0, 0, 0].item() == -1.0

    def test_tensor_cache_refreshes_after_getitem_alias_mutation(self):
        nt = NestedTensor([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
        _ = nt.tensor
        first = nt[0]
        first[0] = -7.0
        assert nt.tensor[0, 0].item() == -7.0

    def test_tensor_cache_refreshes_after_storage_alias_mutation(self):
        nt = NestedTensor([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
        _ = nt.tensor
        storage = nt._storage
        storage[1][0] = 11.0
        assert nt.tensor[1, 0].item() == 11.0

    def test_tensor_cache_refreshes_after_concat_alias_mutation(self):
        nt = NestedTensor([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
        _ = nt.tensor
        concat = nt.concat
        concat[0] = 13.0
        assert nt.tensor[0, 0].item() == 13.0

    def test_ellipsis_indexing_slices_last_dim(self):
        """``nt[..., :k]`` should slice the last dim, not the ragged dim."""
        nt = NestedTensor([torch.randn(3, 2, 4), torch.randn(5, 2, 4)])
        sliced = nt[..., :2]
        assert sliced[0].shape == torch.Size([3, 2, 2])
        assert sliced[1].shape == torch.Size([5, 2, 2])

        # Roundtrip: cat halves back together
        first_half = nt[..., :2]
        second_half = nt[..., 2:]
        recombined = torch.cat([first_half, second_half], dim=-1)
        for a, b in zip(recombined, nt):
            torch.testing.assert_close(a, b)

    def test_ellipsis_indexing_with_newaxis(self):
        nt = NestedTensor([torch.randn(3, 4), torch.randn(5, 4)])
        output = nt[..., None]
        reference = NT([tensor[..., None] for tensor in nt], **nt._meta())
        assert_close(output, reference)

    def test_ellipsis_setitem_targets_last_dim(self):
        nt = NestedTensor([torch.randn(3, 4), torch.randn(5, 4)])
        reference = [t.clone() for t in nt]
        nt[..., 0] = 0.0
        for tensor in reference:
            tensor[..., 0] = 0.0
        for output, expected in zip(nt, reference):
            assert_close(output, expected)

    def test_duplicate_ellipsis_raises(self):
        nt = NestedTensor([torch.randn(3, 4), torch.randn(5, 4)])
        with pytest.raises(IndexError):
            _ = nt[..., ..., 0]
        with pytest.raises(IndexError):
            nt[..., ..., 0] = -1.0

    def test_setitem_different_shape_slow_path(self):
        """Different-shape __setitem__ triggers full repack."""
        nt = NestedTensor([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
        original_ptr = nt._values.data_ptr()
        nt[1] = torch.tensor([9.0, 10.0, 11.0, 12.0])
        # Slow path: must repack, new buffer
        assert nt._values.data_ptr() != original_ptr
        assert_close(nt[0], torch.tensor([1.0, 2.0, 3.0]))
        assert_close(nt[1], torch.tensor([9.0, 10.0, 11.0, 12.0]))
        assert nt.shape == torch.Size([2, 4])

    def test_setitem_2d_same_trailing_shape(self):
        nt = NestedTensor(
            [
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[5.0, 6.0]]),
            ]
        )
        nt[0] = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        assert_close(nt[0], torch.tensor([[10.0, 20.0], [30.0, 40.0]]))
        assert_close(nt[1], torch.tensor([[5.0, 6.0]]))

    def test_setitem_2d_trailing_shape_change_repacks(self):
        nt = NestedTensor(
            [
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[5.0, 6.0]]),
            ]
        )
        nt[1] = torch.tensor([[7.0, 8.0, 9.0]])
        assert_close(nt[0], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert_close(nt[1], torch.tensor([[7.0, 8.0, 9.0]]))
        assert nt.shape == torch.Size([2, 2, 3])

    def test_setitem_multidim_same_shape(self):
        nt = NestedTensor(
            [
                torch.arange(6.0).reshape(2, 3),
                torch.arange(3.0).reshape(1, 3),
            ]
        )
        replacement = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        nt[0] = replacement
        assert_close(nt[0], replacement)
        assert_close(nt[1], torch.arange(3.0).reshape(1, 3))

    def test_setitem_multidim_different_shape(self):
        nt = NestedTensor(
            [
                torch.arange(6.0).reshape(2, 3),
                torch.arange(3.0).reshape(1, 3),
            ]
        )
        replacement = torch.arange(9.0).reshape(3, 3)
        nt[0] = replacement
        assert_close(nt[0], replacement)
        assert_close(nt[1], torch.arange(3.0).reshape(1, 3))
        assert nt.shape == torch.Size([2, 3, 3])

    def test_setitem_negative_index(self):
        nt = NestedTensor([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])])
        nt[-1] = torch.tensor([10.0, 20.0])
        assert_close(nt[1], torch.tensor([10.0, 20.0]))

    def test_setitem_out_of_range_raises(self):
        nt = NestedTensor([torch.tensor([1.0, 2.0])])
        with pytest.raises(IndexError):
            nt[5] = torch.tensor([1.0])

    def test_setitem_tuple_scalar_assignment_repacks_safely(self):
        nt = NestedTensor(
            [
                torch.arange(6.0).reshape(2, 3),
                torch.arange(3.0).reshape(1, 3),
            ]
        )
        nt[:, 0] = -1.0
        assert_close(nt[0], torch.tensor([[-1.0, -1.0, -1.0], [3.0, 4.0, 5.0]]))
        assert_close(nt[1], torch.tensor([[-1.0, -1.0, -1.0]]))
        nt._validate_metadata()

    def test_setitem_tuple_nested_values_assigns_per_element(self):
        nt = NestedTensor(
            [
                torch.arange(6.0).reshape(2, 3),
                torch.arange(3.0).reshape(1, 3),
            ]
        )
        replacement = NestedTensor(
            [
                torch.tensor([10.0, 20.0, 30.0]),
                torch.tensor([40.0, 50.0, 60.0]),
            ]
        )
        nt[:, 0] = replacement
        assert_close(nt[0], torch.tensor([[10.0, 20.0, 30.0], [3.0, 4.0, 5.0]]))
        assert_close(nt[1], torch.tensor([[40.0, 50.0, 60.0]]))
        nt._validate_metadata()

    def test_setitem_tuple_boolean_batch_index(self):
        nt = NestedTensor(
            [
                torch.arange(6.0).reshape(2, 3),
                torch.arange(3.0).reshape(1, 3),
            ]
        )
        nt[[True, False], 1] = 99.0
        assert_close(nt[0], torch.tensor([[0.0, 1.0, 2.0], [99.0, 99.0, 99.0]]))
        assert_close(nt[1], torch.arange(3.0).reshape(1, 3))
        nt._validate_metadata()


# ---------------------------------------------------------------------------
# Comparison Operators
# ---------------------------------------------------------------------------


class TestPackOptimization:

    def test_pack_physical_shape_values(self):
        """Verify _pack produces correct physical shape for mixed-shape inputs."""
        t1 = torch.tensor([[1, 2], [3, 4]])  # shape (2, 2)
        t2 = torch.tensor([[5, 6, 7]])  # shape (1, 3)
        nt = NestedTensor(t1, t2)
        expected = torch.tensor([[2, 2], [1, 3]], dtype=torch.long)
        torch.testing.assert_close(nt._physical_shape, expected)

    def test_pack_physical_shape_scalars(self):
        """Verify _pack handles scalar tensors (ndim=0)."""
        nt = NestedTensor(torch.tensor(1.0), torch.tensor(2.0))
        assert nt._physical_shape.shape == (2, 0) or nt._physical_shape.numel() == 0

    def test_pack_physical_shape_1d(self):
        """Verify _pack handles 1D tensors with different lengths."""
        nt = NestedTensor([1, 2, 3], [4, 5])
        expected = torch.tensor([[3], [2]], dtype=torch.long)
        torch.testing.assert_close(nt._physical_shape, expected)

    def test_pack_empty(self):
        """Verify _pack handles empty tensor list."""
        values, offsets, shape_tensor, packed_sizes, element_shapes = NestedTensor._pack(())
        assert values.numel() == 0
        assert offsets.shape == (1,)
        assert shape_tensor.numel() == 0
        assert packed_sizes == ()
        assert element_shapes == ()

    def test_pack_preserves_static_channel_suffix_for_multiragged_3d(self):
        t1 = torch.arange(3 * 5 * 7.0).reshape(3, 5, 7)
        t2 = torch.arange(3 * 4 * 6.0).reshape(3, 4, 6)
        nt = NestedTensor(t1, t2)

        assert nt._values.shape == torch.Size([59, 3])
        assert nt._packed_sizes == (35, 24)
        assert nt._permutation == (1, 2, 0)
        assert tuple(offset.tolist() for offset in nt._hierarchical_offsets) == (
            [0, 5, 9],
            [0, 7, 14, 21, 28, 35, 41, 47, 53, 59],
        )
        assert nt.concat.shape == torch.Size([59, 3])
        assert_close(nt[0], t1)
        assert_close(nt[1], t2)

    def test_pack_records_hierarchical_offsets_for_attention_layout(self):
        t1 = torch.arange(2 * 5 * 8.0).reshape(2, 5, 8)
        t2 = torch.arange(2 * 3 * 8.0).reshape(2, 3, 8)
        nt = NestedTensor(t1, t2)

        assert nt._permutation == (1, 0, 2)
        assert tuple(offset.tolist() for offset in nt._hierarchical_offsets) == ([0, 5, 8],)

    def test_packed_decode_helpers_roundtrip_multiragged_layout(self):
        t1 = torch.arange(3 * 5 * 7.0).reshape(3, 5, 7)
        t2 = torch.arange(3 * 4 * 6.0).reshape(3, 4, 6)
        nt = NestedTensor(t1, t2)

        batch_idx, local_idx = nt._packed_batch_local_indices()
        coords = nt._packed_varying_coords(batch_idx, local_idx)

        assert tuple(coord[:8].tolist() for coord in coords) == (
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 2, 3, 4, 5, 6, 0],
        )
        assert_close(nt.tensor[nt._packed_dense_index(device=nt.device)], nt._values)

    def test_layout_match_requires_same_permutation(self):
        nt = NestedTensor(torch.arange(2 * 5 * 8.0).reshape(2, 5, 8), torch.arange(2 * 3 * 8.0).reshape(2, 3, 8))
        mismatched = NestedTensor._from_packed(
            nt._values,
            nt._offsets,
            nt._physical_shape,
            permutation=(0, 1, 2),
            batch_first=nt.batch_first,
            padding_value=nt.padding_value,
            mask_value=nt.mask_value,
            pin_memory=nt._pin_memory,
            outer_size=nt._logical_shape,
            packed_sizes=nt._packed_sizes,
            element_shapes=nt._element_shapes,
            validate=False,
        )

        assert not nt._has_same_layout(mismatched)
        assert nt._has_same_layout(nt.clone())

    def test_repack_preserves_permutation(self):
        original = NestedTensor([torch.arange(6.0).reshape(2, 3)]).unsqueeze(1)
        rebuilt = NestedTensor([torch.arange(6.0).reshape(2, 3)]).unsqueeze(1)
        original_permutation = original._permutation
        original_values_shape = original._values.shape

        original._storage = original._storage

        assert original._permutation == original_permutation
        assert original._values.shape == original_values_shape
        assert original._has_same_structure(rebuilt)
        assert_close(original.tensor, rebuilt.tensor)

    def test_structure_match_allows_different_static_suffix(self):
        lhs = NestedTensor(torch.randn(2, 1, 3), torch.randn(1, 1, 3))
        rhs = NestedTensor(torch.randn(2, 3, 4), torch.randn(1, 3, 4))

        assert lhs._has_same_structure(rhs)
        assert not lhs._has_same_layout(rhs)

    def test_metadata_exposes_varying_and_static_dims(self):
        conv_like = NestedTensor(torch.randn(3, 5, 7), torch.randn(3, 4, 6))
        attention_like = NestedTensor(torch.randn(2, 5, 8), torch.randn(2, 3, 8))

        assert conv_like._varying_dims == (1, 2)
        assert conv_like._static_dims == (0,)
        assert attention_like._varying_dims == (1,)
        assert attention_like._static_dims == (0, 2)

    def test_batch_dense_shape_helpers_preserve_ragged_dims(self):
        conv_like = NestedTensor(torch.randn(3, 5, 7), torch.randn(3, 4, 6))
        attention_like = NestedTensor(torch.randn(2, 5, 8), torch.randn(2, 3, 8))

        conv_shape = conv_like._physical_shape_like_batch_dense((len(conv_like), 1, 5, 7))
        attention_shape = attention_like._physical_shape_like_batch_dense((len(attention_like), 2, 5, 4))

        assert tuple(tuple(int(size) for size in row) for row in conv_shape.tolist()) == ((1, 5, 7), (1, 4, 6))
        assert tuple(tuple(int(size) for size in row) for row in attention_shape.tolist()) == ((2, 5, 4), (2, 3, 4))
        assert conv_like._element_shapes_like_batch_dense((len(conv_like), 1, 5, 7)) == ((1, 5, 7), (1, 4, 6))
        assert attention_like._element_shapes_like_batch_dense((len(attention_like), 2, 5, 4)) == (
            (2, 5, 4),
            (2, 3, 4),
        )

    def test_shape_meta_from_components_supports_prefix_suffix_and_replacements(self):
        source = NestedTensor(torch.randn(5, 12), torch.randn(3, 12))

        projected_shape, projected_packed_sizes, projected_element_shapes = source._shape_meta_from_components(
            prefix=(4,),
            keep_dims=(0,),
            suffix=(3,),
        )
        restored_shape, restored_packed_sizes, restored_element_shapes = source._shape_meta_from_components(
            replace_dims={1: 7}
        )

        assert tuple(tuple(int(size) for size in row) for row in projected_shape.tolist()) == ((4, 5, 3), (4, 3, 3))
        assert projected_packed_sizes == (5, 3)
        assert projected_element_shapes == ((4, 5, 3), (4, 3, 3))
        assert tuple(tuple(int(size) for size in row) for row in restored_shape.tolist()) == ((5, 7), (3, 7))
        assert restored_packed_sizes == source._packed_sizes
        assert restored_element_shapes == ((5, 7), (3, 7))

    def test_mask_squeezes_channel_without_python_meta(self):
        nt = NestedTensor(torch.randn(2, 5, 8), torch.randn(2, 3, 8))
        nt._element_shapes = None

        assert nt._mask_squeezes_channel() is True

    def test_hierarchical_offsets_survive_without_python_meta(self):
        conv_like = NestedTensor(torch.randn(3, 5, 7), torch.randn(3, 4, 6))
        attention_like = NestedTensor(torch.randn(2, 5, 8), torch.randn(2, 3, 8))

        conv_like._packed_sizes = None
        conv_like._element_shapes = None
        conv_like._cached_hierarchical_offsets = None
        attention_like._packed_sizes = None
        attention_like._element_shapes = None
        attention_like._cached_hierarchical_offsets = None

        assert tuple(offset.tolist() for offset in conv_like._hierarchical_offsets) == (
            [0, 5, 9],
            [0, 7, 14, 21, 28, 35, 41, 47, 53, 59],
        )
        assert tuple(offset.tolist() for offset in attention_like._hierarchical_offsets) == ([0, 5, 8],)
        assert conv_like._ragged_rank == 2
        assert attention_like._ragged_rank == 1

    def test_trailing_physical_dim_helpers_work_without_python_meta(self):
        nt = NestedTensor(torch.randn(2, 5, 7), torch.randn(2, 3, 7))
        nt._packed_sizes = None
        nt._element_shapes = None

        dropped_shape, dropped_packed_sizes, dropped_element_shapes = nt._drop_trailing_physical_dims_meta(1)
        replaced_shape, replaced_packed_sizes, replaced_element_shapes = nt._replace_trailing_physical_dims_meta((4, 9))

        assert tuple(tuple(int(size) for size in row) for row in dropped_shape.tolist()) == ((2, 5), (2, 3))
        assert dropped_packed_sizes is None
        assert dropped_element_shapes is None
        assert tuple(tuple(int(size) for size in row) for row in replaced_shape.tolist()) == ((2, 4, 9), (2, 4, 9))
        assert replaced_packed_sizes is None
        assert replaced_element_shapes is None

    def test_concat_without_python_meta_preserves_attention_like_storage(self):
        nt = NestedTensor(torch.randn(4, 11, 32), torch.randn(4, 7, 32))
        nt._packed_sizes = None
        nt._element_shapes = None

        assert nt.concat.shape == torch.Size((18, 4, 32))


class TestPackedCacheInvalidation:

    def test_tensor_and_mask_views_are_cached_until_invalidated(self):
        nested_tensor = NestedTensor([torch.tensor([1.0, 2.0]), torch.tensor([3.0])], padding_value=-5.0)

        tensor1 = nested_tensor.tensor
        tensor2 = nested_tensor.tensor
        mask1 = nested_tensor.mask
        mask2 = nested_tensor.mask
        tensor3, mask3 = nested_tensor.tensor_mask

        assert tensor1 is tensor2
        assert tensor1 is tensor3
        assert mask1 is mask2
        assert mask1 is mask3

    def test_materialized_view_cache_respects_padding_and_mask_settings(self):
        nested_tensor = NestedTensor([torch.tensor([1.0, 2.0]), torch.tensor([3.0])], padding_value=-1.0)

        tensor_before = nested_tensor.tensor
        nested_tensor.padding_value = 7.0
        tensor_after = nested_tensor.tensor
        assert tensor_before is not tensor_after
        assert_close(tensor_after, torch.tensor([[1.0, 2.0], [3.0, 7.0]]))

        mask_before = nested_tensor.mask
        nested_tensor.mask_value = True
        mask_after = nested_tensor.mask
        assert mask_before is not mask_after
        assert_close(mask_after, ~mask_before)

    def test_inplace_unary_invalidates_storage_cache(self):
        nested_tensor = NestedTensor([torch.tensor([-1.0, 2.0]), torch.tensor([-3.0])])
        _ = nested_tensor._storage
        _ = nested_tensor.tensor
        _ = nested_tensor.mask
        assert nested_tensor._cached_storage is not None
        assert nested_tensor._cached_tensor_view is not None
        assert nested_tensor._cached_mask_view is not None
        nested_tensor.relu_()
        assert nested_tensor._cached_storage is None
        assert nested_tensor._cached_tensor_view is None
        assert nested_tensor._cached_mask_view is None
        assert_close(nested_tensor[0], torch.tensor([0.0, 2.0]))
        assert_close(nested_tensor[1], torch.tensor([0.0]))

    def test_inplace_binary_invalidates_storage_cache(self):
        nested_tensor = NestedTensor([torch.tensor([1.0, 2.0]), torch.tensor([3.0])])
        _ = nested_tensor._storage
        _ = nested_tensor.tensor
        assert nested_tensor._cached_storage is not None
        assert nested_tensor._cached_tensor_view is not None
        nested_tensor.add_(1.5)
        assert nested_tensor._cached_storage is None
        assert nested_tensor._cached_tensor_view is None
        assert_close(nested_tensor[0], torch.tensor([2.5, 3.5]))
        assert_close(nested_tensor[1], torch.tensor([4.5]))

    def test_copy_invalidates_storage_cache(self):
        dest = NestedTensor([torch.tensor([1.0, 2.0]), torch.tensor([3.0])])
        src = NestedTensor([torch.tensor([9.0, 8.0]), torch.tensor([7.0])], **dest._meta(include_dtype=False))
        _ = dest._storage
        _ = dest.tensor
        assert dest._cached_storage is not None
        assert dest._cached_tensor_view is not None
        dest.copy_(src)
        assert dest._cached_storage is None
        assert dest._cached_tensor_view is None
        assert_close(dest, src)

    def test_copy_requires_matching_layout_not_just_structure(self):
        dest = NestedTensor(
            [
                torch.randn(2, 5, 8),
                torch.randn(2, 3, 8),
            ]
        )
        src = NestedTensor._from_packed(
            dest._values.clone(),
            dest._offsets.clone(),
            dest._physical_shape.clone(),
            permutation=(0, 1, 2),
            batch_first=dest.batch_first,
            padding_value=dest.padding_value,
            mask_value=dest.mask_value,
            pin_memory=dest._pin_memory,
            outer_size=dest._logical_shape,
            packed_sizes=dest._packed_sizes,
            element_shapes=dest._element_shapes,
            validate=False,
        )

        with pytest.raises(NotImplementedError, match="matching packed layout"):
            dest.copy_(src)


# ---------------------------------------------------------------------------
# Indexing (__getitem__, __setitem__)
# ---------------------------------------------------------------------------


class TestReductions:

    def test_torch_all_consistency(self):
        nested_tensor = NestedTensor(torch.ones(2), torch.ones(3))
        assert nested_tensor.all()
        assert torch.all(nested_tensor)

    def test_torch_any_consistency(self):
        nested_tensor = NestedTensor(torch.zeros(2), torch.ones(3))
        assert nested_tensor.any()
        assert torch.any(nested_tensor)
        all_zero = NestedTensor(torch.zeros(2), torch.zeros(3))
        assert not all_zero.any()
        assert not torch.any(all_zero)

    def test_torch_isin_matches_tensor(self):
        a = NestedTensor([[2, 3, 4], [5, 6]])
        assert_close(torch.isin(a, a.tensor[0, 1]), torch.isin(a.tensor, a.tensor[0, 1]))

    def test_sum_with_list_dim_matches_int(self):
        nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        reference = nested_tensor.sum(dim=0)
        assert_close(nested_tensor.sum(dim=[0]), reference)

    def test_sum_multi_dim_batch_first_false(self):
        nested_tensor = NestedTensor(
            [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[7.0, 8.0, 9.0]])],
            batch_first=False,
            padding_value=5.0,
        )
        output = nested_tensor.sum(dim=[0, 2])
        reference = torch.tensor([21.0, 24.0])
        assert_close(output, reference)


# ---------------------------------------------------------------------------
# Torch Function Dispatch
# ---------------------------------------------------------------------------


class TestShapeManipulation:

    def test_permute(self):
        nested_tensor = NestedTensor([torch.randn(3, 4, 5), torch.randn(2, 4, 5)])
        original_shape = nested_tensor.shape
        assert original_shape == torch.Size([2, 3, 4, 5])

        permuted = nested_tensor.permute(0, 3, 1, 2)
        assert permuted.shape == torch.Size([2, 5, 3, 4])
        assert permuted is not nested_tensor

        assert permuted[0].shape == torch.Size([5, 3, 4])
        assert permuted[1].shape == torch.Size([5, 2, 4])
        assert nested_tensor.shape == torch.Size([2, 3, 4, 5])

        nested_tensor2 = NestedTensor([torch.randn(3, 4), torch.randn(2, 4)])
        permuted2 = nested_tensor2.permute(0, -1, -2)
        assert permuted2.shape == torch.Size([2, 4, 3])

        with pytest.raises(ValueError, match="Expected 3 dimensions"):
            nested_tensor2.permute(0, 1)

        nested_tensor3 = NestedTensor([torch.randn(3, 4), torch.randn(2, 4)])
        with pytest.raises(ValueError, match="Invalid permutation dims .* for shape with . dims"):
            nested_tensor3.permute(1, 2, -1)

        with pytest.raises(ValueError, match="batch dimension"):
            nested_tensor.permute(1, 0, 2, 3)

    def test_transpose(self):
        nested_tensor = NestedTensor([torch.randn(3, 4), torch.randn(2, 4)])
        original_shape = nested_tensor.shape
        assert original_shape == torch.Size([2, 3, 4])

        transposed = nested_tensor.transpose(1, 2)
        assert transposed.shape == torch.Size([2, 4, 3])
        assert transposed is not nested_tensor

        assert transposed[0].shape == torch.Size([4, 3])
        assert transposed[1].shape == torch.Size([4, 2])
        assert nested_tensor.shape == torch.Size([2, 3, 4])

        nested_tensor2 = NestedTensor([torch.randn(3, 4, 5), torch.randn(2, 4, 5)])
        transposed2 = nested_tensor2.transpose(-2, -1)
        assert transposed2.shape == torch.Size([2, 3, 5, 4])

        toggled = nested_tensor2.transpose(0, 1)
        assert toggled.batch_first is False
        assert toggled.shape == torch.Size([3, 2, 4, 5])
        assert_close(toggled.tensor, nested_tensor2.tensor.transpose(0, 1))

        toggled_back = toggled.transpose(0, 1)
        assert toggled_back.batch_first is True
        assert_close(toggled_back.tensor, nested_tensor2.tensor)

    def test_reshape(self):
        nested_tensor = NestedTensor([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])])
        original_shape = nested_tensor.shape
        assert original_shape == torch.Size([2, 2, 2])

        reshaped = nested_tensor.reshape(4)
        assert reshaped.shape == torch.Size([2, 4])
        assert reshaped is not nested_tensor

        assert reshaped[0].shape == torch.Size([4])
        assert reshaped[1].shape == torch.Size([4])
        assert nested_tensor.shape == torch.Size([2, 2, 2])

        nested_tensor2 = NestedTensor([torch.randn(2, 3, 4), torch.randn(2, 3, 4)])
        reshaped2 = nested_tensor2.reshape(len(nested_tensor2), -1, 4)
        assert reshaped2.shape == torch.Size([2, 6, 4])

        empty_nested = NestedTensor([])
        output = empty_nested.reshape(5)
        assert output is not empty_nested
        assert len(output) == 0

    def test_view(self):
        nested_tensor = NestedTensor([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])])
        original_shape = nested_tensor.shape
        assert original_shape == torch.Size([2, 2, 2])

        viewed = nested_tensor.view(4)
        assert viewed.shape == torch.Size([2, 4])
        assert viewed is not nested_tensor

        assert viewed[0].shape == torch.Size([4])
        assert viewed[1].shape == torch.Size([4])
        assert nested_tensor.shape == torch.Size([2, 2, 2])

        nested_tensor2 = NestedTensor([torch.randn(2, 6), torch.randn(2, 6)])
        viewed2 = nested_tensor2.view(len(nested_tensor2), -1, 3)
        assert viewed2.shape == torch.Size([2, 4, 3])

        nested_tensor3 = NestedTensor([torch.randn(4), torch.randn(4)])
        viewed3 = nested_tensor3.view(2, 2)
        assert viewed3[0].shape == torch.Size([2, 2])
        assert viewed3[1].shape == torch.Size([2, 2])

        empty_nested = NestedTensor([])
        output = empty_nested.view(5)
        assert output is not empty_nested
        assert len(output) == 0

    def test_view_with_different_shapes(self):
        nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])

        viewed = nested_tensor.view(len(nested_tensor), -1)
        assert viewed is not nested_tensor
        assert viewed[0].shape == torch.Size([3])
        assert viewed[1].shape == torch.Size([2])

    def test_view_with_batch_dim_and_dynamic_lengths(self):
        nested_tensor = NestedTensor([torch.randn(3, 640), torch.randn(5, 640)])
        target_shape = nested_tensor.size()[:-1] + (20, 32)
        viewed = nested_tensor.view(*target_shape)
        padded = nested_tensor.tensor
        assert viewed.shape == torch.Size([2, 5, 20, 32])
        assert viewed[0].shape == torch.Size([3, 20, 32])
        assert viewed[1].shape == torch.Size([5, 20, 32])
        assert torch.equal(viewed, padded.view(*target_shape))

    def test_view_with_explicit_batch_and_reduced_rank(self):
        nested_tensor = NestedTensor([torch.randn(3, 4), torch.randn(3, 4)])
        viewed = nested_tensor.view(len(nested_tensor), -1)
        assert viewed.shape == torch.Size([2, 12])
        assert viewed[0].shape == torch.Size([12])
        assert viewed[1].shape == torch.Size([12])

    def test_view_insert_dim_before_dynamic_length(self):
        nested_tensor = NestedTensor([torch.randn(3, 8), torch.randn(5, 8)])
        target_shape = (len(nested_tensor), 1, nested_tensor.size(1), nested_tensor.size(2))
        viewed = nested_tensor.view(*target_shape)
        assert viewed.shape == torch.Size([2, 1, 5, 8])
        assert viewed[0].shape == torch.Size([1, 3, 8])
        assert viewed[1].shape == torch.Size([1, 5, 8])

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_view_compile_matches_reference(self):
        nested_tensor = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        compiled = torch.compile(lambda x: x.view(-1, 3), backend="inductor", fullgraph=True)
        output = compiled(nested_tensor)
        reference = nested_tensor.view(-1, 3)
        assert isinstance(output, NestedTensor)
        assert output._has_same_layout(reference)
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_nested_like_compile_matches_reference(self):
        nested_tensor = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        dense = nested_tensor.tensor
        compiled = torch.compile(lambda x, y: x.nested_like(y), backend="inductor", fullgraph=True)
        output = compiled(nested_tensor, dense)
        reference = nested_tensor.nested_like(dense)
        assert isinstance(output, NestedTensor)
        assert output._has_same_layout(reference)
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_nested_like_larger_dense_compile_matches_reference(self):
        nested_tensor = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        dense = torch.randn(2, 6, 3)
        compiled = torch.compile(lambda x, y: x.nested_like(y, strict=False), backend="inductor", fullgraph=True)
        output = compiled(nested_tensor, dense)
        reference = nested_tensor.nested_like(dense, strict=False)
        assert isinstance(output, NestedTensor)
        assert output._has_same_layout(reference)
        assert_close(output, reference)

    def test_nested_like_smaller_dense(self):
        nested_tensor = NestedTensor([torch.arange(6.0).reshape(2, 3), torch.arange(12.0).reshape(4, 3)])
        dense = torch.arange(12.0).reshape(2, 2, 3)
        output = nested_tensor.nested_like(dense, strict=False)
        reference = NT([dense[0, :2, :3], dense[1, :2, :3]], **nested_tensor._meta())
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_reshape_compile_matches_reference(self):
        nested_tensor = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        compiled = torch.compile(lambda x: x.reshape(-1, 3), backend="inductor", fullgraph=True)
        output = compiled(nested_tensor)
        reference = nested_tensor.reshape(-1, 3)
        assert isinstance(output, NestedTensor)
        assert output._has_same_layout(reference)
        assert_close(output, reference)

    def test_view_irregular_tail_uses_linearized_packed_path(self):
        nested_tensor = NestedTensor([torch.arange(8.0).reshape(2, 4), torch.arange(8.0, 14.0).reshape(3, 2)])
        viewed = nested_tensor.view(len(nested_tensor), 2, -1)
        reference = NT([t.view(2, -1) for t in nested_tensor], **nested_tensor._meta())
        assert_close(viewed, reference)

    def test_method_chaining(self):
        nested_tensor = NestedTensor([torch.randn(2, 3, 4), torch.randn(2, 3, 4)])

        output = nested_tensor.transpose(1, 2).reshape(len(nested_tensor), -1, 6).view(24, 1)
        assert output is not nested_tensor
        assert output.shape == torch.Size([2, 24, 1])
        assert nested_tensor.shape == torch.Size([2, 2, 3, 4])


# ---------------------------------------------------------------------------
# Cat / Concatenation
# ---------------------------------------------------------------------------


class TestStatePreservation:

    def _assert_state(self, output, *, batch_first=False, padding_value=-1, mask_value=True, pin_memory=None):
        """Shared assertion for state fields."""
        assert isinstance(output, NestedTensor)
        assert output.batch_first is batch_first
        assert output.padding_value == padding_value
        assert output.mask_value is mask_value
        if pin_memory is not None:
            assert output._pin_memory is pin_memory

    def test_meta_with_dtype_preserves_empty_dtype(self):
        empty = NestedTensor([], dtype=torch.float64, batch_first=False, padding_value=-1, mask_value=True)
        rebuilt = NestedTensor([], **empty._meta(include_dtype=True))
        assert rebuilt.dtype == torch.float64
        self._assert_state(rebuilt, batch_first=False, padding_value=-1, mask_value=True)

    def test_meta_default_preserves_empty_dtype(self):
        empty = NestedTensor([], dtype=torch.float64, batch_first=False, padding_value=-1, mask_value=True)
        rebuilt = NestedTensor([], **empty._meta())
        assert rebuilt.dtype == torch.float64
        self._assert_state(rebuilt, batch_first=False, padding_value=-1, mask_value=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
    def test_comparison_preserves_state(self):
        state = {"batch_first": False, "padding_value": -1, "mask_value": True, "pin_memory": True}
        left = NestedTensor([torch.tensor([[2, 0], [1, 0]])], **state)
        right = NestedTensor([torch.tensor([[1, 0], [1, 0]])], **state)
        output = left > right
        self._assert_state(output, pin_memory=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
    def test_unary_ops_preserve_state(self):
        nested_tensor = NestedTensor(
            [torch.tensor([[1, -2], [3, -4]]), torch.tensor([[5]])],
            batch_first=False,
            padding_value=-1,
            mask_value=True,
            pin_memory=True,
        )
        for op in (lambda x: +x, lambda x: -x, lambda x: ~x):
            output = op(nested_tensor)
            self._assert_state(output, pin_memory=True)

    def test_dtype_change_preserves_state(self):
        nested_tensor = NestedTensor(
            [torch.tensor([1, 2], dtype=torch.int64), torch.tensor([3], dtype=torch.int64)],
            batch_first=False,
            padding_value=-1,
            mask_value=True,
        )
        floated = nested_tensor.float()
        assert floated.dtype == torch.float32
        assert all(t.dtype == torch.float32 for t in floated)
        self._assert_state(floated)

    def test_to_nested_tensor_preserves_state(self):
        nested_tensor = NestedTensor(
            [torch.tensor([1, 2], dtype=torch.float32), torch.tensor([3], dtype=torch.float32)],
            batch_first=False,
            padding_value=-1,
            mask_value=True,
        )
        other = NestedTensor([torch.tensor([1], dtype=torch.float64)])
        output = nested_tensor.to(other, non_blocking=True)

        assert output.dtype == other.dtype
        assert all(t.dtype == other.dtype for t in output)
        assert output.device == other.device
        self._assert_state(output)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to test device movement")
    def test_cuda_move_preserves_state(self):
        nested_tensor = NestedTensor(
            [torch.tensor([1, 2]), torch.tensor([3])],
            batch_first=False,
            padding_value=-1,
            mask_value=True,
        )
        moved = nested_tensor.cuda()
        assert moved.device.type == "cuda"
        assert all(t.device.type == "cuda" for t in moved)
        self._assert_state(moved)

    def test_dtype_and_device_properties_are_read_only(self):
        nested_tensor = NestedTensor([torch.tensor([1.0, 2.0]), torch.tensor([3.0])])
        with pytest.raises(AttributeError, match="read-only"):
            nested_tensor.dtype = torch.float64
        with pytest.raises(AttributeError, match="read-only"):
            nested_tensor.device = torch.device("cpu")

    def test_batch_first_mutation_reorients_logical_shape(self):
        nested_tensor = NestedTensor(
            [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])],
            batch_first=True,
            padding_value=-1,
        )

        batch_first_tensor = nested_tensor.tensor
        nested_tensor.batch_first = False

        assert nested_tensor.shape == torch.Size([2, 2, 2])
        assert_close(nested_tensor.tensor, batch_first_tensor.movedim(0, 1))
        assert_close(nested_tensor.mask, torch.tensor([[True, True], [True, False]]))

        nested_tensor.batch_first = True
        assert nested_tensor.shape == torch.Size([2, 2, 2])
        assert_close(nested_tensor.tensor, batch_first_tensor)

    def test_runtime_config_setters_validate_types(self):
        nested_tensor = NestedTensor([torch.tensor([1.0, 2.0]), torch.tensor([3.0])])

        with pytest.raises(TypeError, match="batch_first must be a bool"):
            nested_tensor.batch_first = 1  # type: ignore[assignment]
        with pytest.raises(TypeError, match="mask_value must be a bool"):
            nested_tensor.mask_value = 1  # type: ignore[assignment]
        with pytest.raises(TypeError, match="padding_value must be float-convertible"):
            nested_tensor.padding_value = object()  # type: ignore[assignment]


class TestTensorMaskProperties:

    def test_tensor_mask_does_not_squeeze_last_dim(self):
        nt = NestedTensor([torch.tensor([1]), torch.tensor([2])])
        tensor, mask = nt.tensor_mask
        assert tensor.shape == nt.tensor.shape
        assert mask.shape == nt.mask.shape == torch.Size([2, 1])

        nt_bf_false = NestedTensor([torch.tensor([1, 2, 3])], batch_first=False)
        tensor2, mask2 = nt_bf_false.tensor_mask
        assert tensor2.shape == nt_bf_false.tensor.shape
        assert mask2.shape == nt_bf_false.mask.shape == torch.Size([3, 1])

    def test_tensor_mask_shapes_for_1d_sequences(self):
        lengths = [2, 3, 5, 7]
        channels = 8
        nested_tensor = NestedTensor(torch.randn(length, channels) for length in lengths)
        tensor, mask = nested_tensor.tensor_mask
        assert tensor.shape == nested_tensor.tensor.shape == torch.Size((len(lengths), max(lengths), channels))
        assert mask.shape == nested_tensor.mask.shape == torch.Size((len(lengths), max(lengths)))
        assert_close(tensor @ nested_tensor.T, nested_tensor.tensor @ nested_tensor.T)

    def test_flat_equal_lengths(self):
        nested_tensor = NestedTensor([torch.tensor([1, 2]), torch.tensor([3, 4])])
        assert_close(nested_tensor.concat, torch.tensor([1, 2, 3, 4]))

    def test_flat_scalar_round_trip(self):
        nested_tensor = NestedTensor([torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)])
        assert_close(nested_tensor.concat, torch.tensor([1.0, 2.0, 3.0]))

        concat, shapes = nested_tensor.concatenate()
        output = NestedTensor.from_concatenated(concat, shapes, **nested_tensor._meta())
        assert_close(output, nested_tensor)

        nested_tensor = NestedTensor([torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)], batch_first=False)
        assert_close(nested_tensor.concat, torch.tensor([1.0, 2.0, 3.0]))

        concat, shapes = nested_tensor.concatenate()
        output = NestedTensor.from_concatenated(concat, shapes, **nested_tensor._meta())
        assert_close(output, nested_tensor)

    def test_flat_batch_first_false(self):
        nested_tensor = NestedTensor(
            [torch.arange(3).unsqueeze(1), torch.arange(3, 7).unsqueeze(1)],
            batch_first=False,
        )
        flat = nested_tensor.concat
        assert flat.shape == torch.Size([7, 1])
        assert_close(flat.squeeze(1), torch.arange(7))

    def test_concat_without_python_shape_meta_uses_physical_shape(self):
        nested_tensor = NestedTensor(
            [torch.arange(6, dtype=torch.float32).reshape(2, 3), torch.arange(4, dtype=torch.float32).reshape(1, 4)]
        )
        nested_tensor._element_shapes = None

        assert_close(nested_tensor.concat, nested_tensor._values)

    def test_size_and_nested_like_batch_first_false(self):
        tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8], [9, 10]])]
        nested_tensor = NestedTensor(tensors, batch_first=False)

        assert nested_tensor.size() == torch.Size([3, 2, 2])
        assert nested_tensor.size(0) == 3
        assert nested_tensor.size(1) == len(tensors)

        cloned = nested_tensor.nested_like(nested_tensor.tensor)
        assert_close(cloned, nested_tensor)


# ---------------------------------------------------------------------------
# State Preservation (one representative test per category)
# ---------------------------------------------------------------------------


class TestTorchFunctionDispatch:

    def test_torch_function_scalar_batch_fallback(self):
        nested_tensor = NestedTensor([torch.tensor(1.2), torch.tensor(2.3)])
        output = torch.ceil(nested_tensor)
        reference = NT([torch.tensor(2.0), torch.tensor(3.0)])
        assert_close(output, reference)


# ---------------------------------------------------------------------------
# Where
# ---------------------------------------------------------------------------


class TestWhere:

    @pytest.mark.skipif(TORCH_VERSION < Version("2.1"), reason="requires PyTorch 2.1 or higher")
    def test_where(self):
        a = NestedTensor([[2, 3, 4], [5, 6]])
        assert_close(a.where(a > 3, -1.0), NT([[-1.0, -1.0, 4.0], [5.0, 6.0]]))
        assert_close(a.where(a.tensor > 3, -1.0), NT([[-1.0, -1.0, 4.0], [5.0, 6.0]]))
        assert_close(a.where(torch.tensor(False), 1), NT([[1, 1, 1], [1, 1]]))


# ---------------------------------------------------------------------------
# Shape Manipulation
# ---------------------------------------------------------------------------
