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
import pickle

import pytest
import torch
from packaging.version import Version

from danling.tensors import NestedTensor
from danling.tensors.ops import nested_execution_guard
from tests.tensors.utils import assert_close

NT = NestedTensor


TORCH_VERSION = Version(torch.__version__.split("+")[0])


# ---------------------------------------------------------------------------
# Construction & Validation
# ---------------------------------------------------------------------------


class TestArithmetic:

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compiled_wrapper_outputs_chain_and_preserve_autograd(self):
        parts = (torch.randn(2, 3, requires_grad=True), torch.randn(4, 3, requires_grad=True))
        nested = NT(parts, ragged_dims=(0,))
        producer = torch.compile(lambda tensor: tensor * 2, backend="aot_eager", fullgraph=True)
        consumer = torch.compile(torch.sin, backend="aot_eager", fullgraph=True)

        output = consumer(producer(nested))
        gradients = torch.autograd.grad(output.concat.sum(), parts)

        assert_close(output.concat, torch.sin(nested.concat * 2))
        expected = 2 * torch.cos(nested.concat * 2)
        for actual, wanted in zip(gradients, expected.split([2, 4])):
            assert_close(actual, wanted)

    def test_concat_remains_read_only_for_users(self):
        nested = NT([torch.empty(2, 3), torch.empty(4, 3)], ragged_dims=(0,))

        with pytest.raises(AttributeError, match="read-only"):
            nested.concat = torch.empty_like(nested.concat)

    @pytest.mark.parametrize(
        "i",
        [
            NestedTensor([[6, 5, 4], [3, 2]]),
            torch.tensor([[6, 5, 4], [3, 2, 1]]),
            1,
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
            2,
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
            2,
        ],
    )
    def test_pow_log(self, i):
        a = NestedTensor([[2, 3, 4], [5, 6]]).float()
        b = a.clone()
        assert_close(torch.log(a**i), torch.log(a) * i)
        a **= i
        assert_close(torch.log(a), torch.log(b) * i)

    def test_shift(self):
        i = 1
        a = NestedTensor([[2, 3, 4], [5, 6]])
        assert_close(a << i >> i, a)
        b = a.clone()
        b <<= i
        assert_close(a << i, b)
        b >>= i
        assert_close(a, b)

    def test_logic(self):
        i = 1
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
            2,
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
            2,
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

        assert (big > small).all()
        assert (big >= small.tensor).all()
        assert (big == value).all()
        assert (small < torch.tensor(0)).all()
        assert (small <= big).all()
        with pytest.raises(TypeError):
            assert small < "small"
        assert small != "small"

    def test_length_mismatch_equality_and_ops(self):
        shorter = NestedTensor([[1, 2]])
        longer = NestedTensor([[1, 2], [3]])

        assert torch.equal(shorter, longer) is False
        assert torch.allclose(shorter, longer) is False
        with pytest.raises(ValueError):
            _ = shorter == longer
        with pytest.raises(ValueError):
            _ = shorter + longer

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
        nested = NestedTensor([[2, 3, 4], [5, 6]])
        with pytest.raises(ValueError):
            _ = nested[""]
        with pytest.raises(ValueError):
            NestedTensor(False)

    def test_single_tensor_not_unbound(self):
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        nested = NestedTensor(tensor)

        assert len(nested) == 1
        assert_close(nested[0], tensor)
        assert nested.shape == torch.Size([1, 2, 3])
        assert_close(nested.tensor, tensor.unsqueeze(0))

    def test_mixed_dtype_inputs_promote_to_common_dtype(self):
        nested = NestedTensor([torch.tensor([1], dtype=torch.int64), torch.tensor([1.5], dtype=torch.float32)])

        assert nested.dtype == torch.float32
        assert all(element.dtype == torch.float32 for element in nested)
        assert_close(nested.tensor, torch.tensor([[1.0], [1.5]], dtype=torch.float32))

    def test_empty_nested_tensor_accessors(self):
        nested = NestedTensor([], dtype=torch.float32)
        tensor, mask = nested.tensor_mask

        assert nested.size() == torch.Size([0])
        assert nested.dim() == 1
        assert tensor.shape == nested.tensor.shape == torch.Size([0])
        assert mask.shape == nested.mask.shape == torch.Size([0])
        assert nested.occupancy == 0.0

    def test_empty_nested_tensor_honors_requested_device(self):
        nested = NestedTensor([], dtype=torch.float32, device=torch.device("meta"))

        assert nested.device.type == "meta"
        assert nested.concat.device.type == "meta"

    def test_bool_nested_tensor_raises(self):
        with pytest.raises(RuntimeError, match="Boolean value of NestedTensor is ambiguous"):
            bool(NestedTensor([]))

    def test_requires_grad_tracks_values(self):
        nested = NestedTensor([torch.tensor([1.0], requires_grad=True), torch.tensor([2.0])])

        assert nested.requires_grad is True

    def test_empty_requires_grad_is_preserved(self):
        nested = NestedTensor([], dtype=torch.float32, requires_grad=True)

        assert nested.requires_grad is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
    def test_pin_memory_pins_concat_when_requested(self):
        nested = NestedTensor([torch.tensor([1.0, 2.0]), torch.tensor([3.0])], pin_memory=True)

        assert nested.concat.is_pinned()


# Packed Reconstruction
# ---------------------------------------------------------------------------


class TestDeclaredRaggedDims:

    def test_declared_ragged_dims_preserve_values_and_layout(self):
        elements = [torch.randn(2, 2, 3), torch.randn(4, 4, 3)]

        nested = NestedTensor(elements, ragged_dims=(0, 1))

        assert nested.ragged_dims == (0, 1)
        assert nested.packed_dim_order == (0, 1, 2)
        assert nested.concat.shape == (20, 3)
        for actual, expected in zip(nested, elements):
            assert_close(actual, expected)

    def test_declared_ragged_dims_validate_static_dimensions(self):
        inferred = NestedTensor([torch.randn(4, 4, 3), torch.randn(4, 4, 3)])
        declared = NestedTensor([torch.randn(2, 2, 3), torch.randn(4, 4, 3)], ragged_dims=(0, 1))

        assert inferred.ragged_dims == (0,)
        with pytest.raises(AttributeError):
            declared.ragged_dims = (0,)  # type: ignore[misc]
        with pytest.raises(ValueError, match="not listed in ragged_dims"):
            NestedTensor([torch.randn(2, 2, 3), torch.randn(4, 4, 5)], ragged_dims=(0, 1))

    def test_declared_ragged_dims_support_fake_tensor(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        reference = NestedTensor([torch.empty(2, 2, 3), torch.empty(4, 4, 3)], ragged_dims=(0, 1))

        with fake_tensor_mod.FakeTensorMode() as mode:
            fake_reference = mode.from_tensor(reference)
            output = fake_reference.packed_like(torch.empty_like(fake_reference.concat))

        assert fake_tensor_mod.is_fake(output.concat)
        assert output.ragged_dims == (0, 1)
        assert output.shape == reference.shape

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_declared_ragged_dims_survive_fullgraph_with_backward(self):
        reference = NestedTensor([torch.empty(2, 2, 3), torch.empty(4, 4, 3)], ragged_dims=(0, 1))
        values = torch.randn_like(reference.concat, requires_grad=True)
        compiled = torch.compile(
            lambda ref, packed: ref.packed_like(packed.square()),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        output = compiled(reference, values)
        output.concat.sum().backward()

        assert output.ragged_dims == (0, 1)
        assert output.shape == reference.shape
        assert_close(values.grad, 2 * values)

    def test_declared_pair_dense_matmul_matches_elementwise_reference(self):
        elements = [torch.randn(2, 2, 3), torch.randn(4, 4, 3)]
        nested = NestedTensor(elements, ragged_dims=(0, 1))
        weight = torch.randn(3, 5)

        output = nested @ weight

        assert output.ragged_dims == (0, 1)
        assert output.shape == (2, 4, 4, 5)
        for actual, expected in zip(output, elements):
            assert_close(actual, expected @ weight)

    def test_source_derived_empty_batch_preserves_shape_and_basic_indexing(self):
        source = NestedTensor(
            [torch.empty(4, 3, 4), torch.empty(4, 5, 4)],
            ragged_dims=(0, 1),
        )
        empty = source[:0]

        assert empty.shape == (0, 4, 5, 4)
        assert empty.ragged_dims == (0, 1)
        indices = (
            (slice(None), slice(1, 4, 2), slice(None), slice(1, None)),
            (slice(None), slice(None), 0, slice(None)),
        )
        for index in indices:
            actual = empty[index]
            expected = source[index][:0]
            assert actual.shape == expected.shape
            assert actual.ragged_dims == expected.ragged_dims

    def test_empty_batch_ragged_splits_match_nonempty_shapes(self):
        source = NestedTensor(
            [torch.empty(3, 3, 4), torch.empty(4, 5, 4)],
            ragged_dims=(0, 1),
        )

        expected = tuple(part[:0] for part in torch.split(source, 2, dim=1))
        actual = torch.split(source[:0], 2, dim=1)

        assert [part.shape for part in actual] == [part.shape for part in expected]
        assert [part.ragged_dims for part in actual] == [part.ragged_dims for part in expected]

    def test_indexing_empty_batch_keeps_autograd_connection(self):
        values = torch.randn(8, 2, requires_grad=True)
        nested = NestedTensor(
            [torch.empty(3, 2), torch.empty(5, 2)],
            ragged_dims=(0,),
        ).packed_like(values)

        nested[:0, 0].sum().backward()

        assert_close(values.grad, torch.zeros_like(values))

    def test_ragged_tensor_split_matches_dense_cut_semantics(self):
        cuts = [4, 1]
        nested = NestedTensor(
            [torch.arange(5.0), torch.arange(10.0, 15.0)],
            ragged_dims=(0,),
        )

        actual = torch.tensor_split(nested, cuts, dim=1)
        expected = [torch.tensor_split(element, cuts) for element in nested]

        for part_index, part in enumerate(actual):
            for element_index, element in enumerate(part):
                assert_close(element, expected[element_index][part_index])
        assert [part.shape for part in torch.tensor_split(nested[:0], cuts, dim=1)] == [
            part[:0].shape for part in actual
        ]

    def test_shape_changing_ops_remap_declared_ragged_dims(self):
        elements = [torch.randn(2, 2, 3), torch.randn(2, 2, 3)]
        nested = NestedTensor(elements, ragged_dims=(0, 1))

        permuted = nested.permute(0, 3, 1, 2)
        indexed = nested[:, :, 0, :]

        assert permuted.ragged_dims == (1, 2)
        assert indexed.ragged_dims == (0,)
        for actual, expected in zip(permuted, elements):
            assert_close(actual, expected.permute(2, 0, 1))
        for actual, expected in zip(indexed, elements):
            assert_close(actual, expected[:, 0, :])

    def test_from_concatenated_honors_declared_ragged_order(self):
        elements = [torch.arange(12.0).reshape(2, 2, 3), torch.arange(12.0, 24.0).reshape(2, 2, 3)]
        reference = NestedTensor(elements, ragged_dims=(1, 0))

        values, shapes = reference.concatenate()
        output = NestedTensor.from_concatenated(values, shapes, ragged_dims=(1, 0))

        assert output.ragged_dims == (1, 0)
        for actual, expected in zip(output, elements):
            assert_close(actual, expected)


class TestPackedLike:

    def test_packed_like_preserves_public_layout_and_runtime_config(self):
        reference = NestedTensor(
            [torch.randn(2, 3), torch.randn(4, 3)],
            batch_first=False,
            padding_value=-1.5,
            mask_value=True,
        )
        packed_values = torch.arange(reference.concat.numel(), dtype=torch.float32).reshape(reference.concat.shape)

        output = reference.packed_like(packed_values)

        assert output.concat is packed_values
        assert output.shape == reference.shape
        assert output.ragged_dims == reference.ragged_dims
        assert output.batch_first is False
        assert output.padding_value == -1.5
        assert output.mask_value is True

    def test_packed_dim_order_tracks_logical_permutation(self):
        reference = NestedTensor([torch.randn(2, 2, 3), torch.randn(4, 4, 3)])

        permuted = reference.permute(0, 3, 1, 2)
        rebuilt = permuted.packed_like(torch.randn_like(permuted.concat))

        assert reference.packed_dim_order == (0, 1, 2)
        assert permuted.packed_dim_order == (1, 2, 0)
        assert rebuilt.packed_dim_order == permuted.packed_dim_order

    def test_packed_like_accepts_noncontiguous_values_without_copy(self):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        packed_values = torch.randn(3, reference.concat.size(0)).transpose(0, 1)

        output = reference.packed_like(packed_values)

        assert output.concat is packed_values
        assert output.concat.stride() == packed_values.stride()

    def test_packed_like_follows_dtype_and_autograd_history(self):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        leaf = torch.randn(reference.concat.shape, dtype=torch.float64, requires_grad=True)

        output = reference.packed_like(leaf.square())
        first_order = torch.autograd.grad(output.concat.sum(), leaf, create_graph=True)[0]
        second_order = torch.autograd.grad(first_order.sum(), leaf)[0]

        assert output.dtype == torch.float64
        assert output.requires_grad
        assert_close(first_order, 2 * leaf)
        assert_close(second_order, torch.full_like(leaf, 2))

    def test_packed_like_follows_values_device(self):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        packed_values = torch.empty(reference.concat.shape, device="meta")

        output = reference.packed_like(packed_values)

        assert output.device.type == "meta"
        assert output.concat is packed_values

    def test_packed_like_rejects_invalid_values(self):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])

        with pytest.raises(ValueError, match="exactly the same shape"):
            reference.packed_like(reference.concat.reshape(-1))
        with pytest.raises(TypeError, match="dense Tensor"):
            reference.packed_like(object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="dense Tensor"):
            reference.packed_like(reference)
        with pytest.raises(TypeError, match="torch.strided"):
            reference.packed_like(torch.zeros_like(reference.concat).to_sparse())

    def test_packed_offsets_and_element_sizes_report_public_layout(self):
        nested = NestedTensor(
            [torch.empty(2, 2, 4), torch.empty(3, 3, 4)],
            ragged_dims=(0, 1),
        )

        assert_close(nested.packed_offsets(), torch.tensor([0, 4, 13]))
        assert_close(nested.element_sizes(), torch.tensor([[2, 2, 4], [3, 3, 4]]))
        assert_close(nested.packed_local_indices(level=0), torch.tensor([0, 1, 0, 1, 2]))

    def test_packed_layout_accessors_support_empty_and_fake_tensors(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        empty = NestedTensor([], dtype=torch.float32)
        reference = NestedTensor([torch.empty(2, 3), torch.empty(4, 3)])

        with fake_tensor_mod.FakeTensorMode() as mode:
            fake = mode.from_tensor(reference)
            output = fake.packed_like(torch.empty_like(fake.concat))

        assert empty.element_sizes().shape == (0, 0)
        assert fake_tensor_mod.is_fake(output.concat)
        assert fake_tensor_mod.is_fake(output.packed_offsets())
        assert output.shape == reference.shape

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
        loss.backward()

        assert_close(loss, packed_values.square().sum())
        assert_close(packed_values.grad, 2 * packed_values)

    def test_packed_like_preserves_nested_tensor_subclass(self):
        class DerivedNestedTensor(NestedTensor):
            pass

        reference = DerivedNestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        output = reference.packed_like(torch.randn_like(reference.concat))

        assert type(output) is DerivedNestedTensor
        assert output.shape == reference.shape


class TestPackedWithStaticTail:

    def test_packed_with_static_tail_preserves_values_config_and_autograd(self):
        reference = NestedTensor(
            [torch.empty(2), torch.empty(4)],
            batch_first=False,
            padding_value=-2.5,
            mask_value=True,
        )
        leaf = torch.randn(6, 2, 3, dtype=torch.float64, requires_grad=True)
        packed_values = leaf.square()

        output = reference.packed_with_static_tail(packed_values)
        gradient = torch.autograd.grad(output.concat.sum(), leaf)[0]

        assert output.concat is packed_values
        assert output.shape == (4, 2, 2, 3)
        assert output.batch_first is False
        assert output.padding_value == -2.5
        assert output.mask_value is True
        assert [tuple(element.shape) for element in output] == [(2, 2, 3), (4, 2, 3)]
        assert_close(gradient, 2 * leaf)

    def test_packed_with_static_tail_supports_nonleading_ragged_dimension(self):
        reference = NestedTensor(
            [torch.empty(2, 2, 3), torch.empty(2, 4, 3)],
            ragged_dims=(1,),
        )
        packed_values = torch.randn(6, 2, 7)

        output = reference.packed_with_static_tail(packed_values)

        assert output.concat is packed_values
        assert output.shape == (2, 2, 4, 7)
        assert output.ragged_dims == (1,)
        expected = (packed_values[:2].permute(1, 0, 2), packed_values[2:].permute(1, 0, 2))
        for actual, wanted in zip(output, expected):
            assert_close(actual, wanted)

    def test_packed_with_static_tail_rejects_invalid_layouts_and_values(self):
        permuted = NestedTensor([torch.empty(2, 3), torch.empty(4, 3)]).permute(0, 2, 1)
        scalar = NestedTensor([torch.tensor(1.0), torch.tensor(2.0)])
        reference = NestedTensor([torch.empty(2), torch.empty(4)])

        with pytest.raises(ValueError, match="canonical packed order"):
            permuted.packed_with_static_tail(torch.empty(permuted.concat.shape[0], 5))
        with pytest.raises(ValueError, match="at least one ragged dimension"):
            scalar.packed_with_static_tail(torch.empty(2, 5))
        with pytest.raises(TypeError, match="dense Tensor"):
            reference.packed_with_static_tail(object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="reference packed length"):
            reference.packed_with_static_tail(torch.empty(5, 3))

    def test_packed_with_static_tail_supports_fake_values(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        reference = NestedTensor(
            [torch.empty(2, 2, 3), torch.empty(2, 4, 3)],
            ragged_dims=(1,),
        )

        with fake_tensor_mod.FakeTensorMode() as mode:
            output = mode.from_tensor(reference).packed_with_static_tail(mode.from_tensor(torch.empty(6, 2, 7)))

        assert fake_tensor_mod.is_fake(output.concat)
        assert output.shape == (2, 2, 4, 7)
        assert output.ragged_dims == (1,)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_packed_with_static_tail_fullgraph_backward(self):
        reference = NestedTensor([torch.empty(2), torch.empty(4)])
        packed_values = torch.randn(6, 3, requires_grad=True)
        compiled = torch.compile(
            lambda ref, values: ref.packed_with_static_tail(values).concat.square().sum(),
            backend="aot_eager",
            fullgraph=True,
        )

        loss = compiled(reference, packed_values)
        loss.backward()

        assert_close(loss, packed_values.detach().square().sum())
        assert_close(packed_values.grad, 2 * packed_values)


class TestPackedWithLengths:

    def test_packed_with_lengths_preserves_values_config_and_autograd(self):
        reference = NestedTensor(
            [torch.empty(1, 2), torch.empty(1, 2)],
            batch_first=False,
            padding_value=-3,
            mask_value=True,
        )
        leaf = torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True)
        packed_values = leaf.square()

        output = reference.packed_with_lengths(packed_values, torch.tensor([0, 3]))
        gradient = torch.autograd.grad(output.concat.sum(), leaf)[0]

        assert output.concat is packed_values
        assert output.shape == (3, 2, 2, 4)
        assert output.batch_first is False
        assert output.padding_value == -3
        assert output.mask_value is True
        assert [tuple(element.shape) for element in output] == [(0, 2, 4), (3, 2, 4)]
        assert_close(gradient, 2 * leaf)

    def test_packed_with_lengths_supports_empty_batch(self):
        reference = NestedTensor([torch.empty(2)])[:0]

        output = reference.packed_with_lengths(torch.empty(0, 3), torch.empty(0, dtype=torch.long))

        assert output.shape == (0, 0, 3)
        assert output.ragged_dims == (0,)
        assert len(output) == 0

    @pytest.mark.parametrize(
        ("lengths", "error", "match"),
        [
            (object(), TypeError, "dense Tensor"),
            (torch.tensor([2.0, 3.0]), TypeError, "integer dtype"),
            (torch.tensor([5]), ValueError, "one value per batch"),
            (torch.tensor([2, -1]), ValueError, "non-negative"),
            (torch.tensor([2, 2]), ValueError, "must sum"),
        ],
    )
    def test_packed_with_lengths_validates_lengths(self, lengths, error, match):
        reference = NestedTensor([torch.empty(1), torch.empty(1)])

        with pytest.raises(error, match=match):
            reference.packed_with_lengths(torch.empty(5, 3), lengths)  # type: ignore[arg-type]

    def test_packed_with_lengths_rejects_invalid_values(self):
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        lengths = torch.tensor([2, 3])

        with pytest.raises(TypeError, match="dense Tensor"):
            reference.packed_with_lengths(object(), lengths)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="leading packed dimension"):
            reference.packed_with_lengths(torch.tensor(1.0), lengths)
        with pytest.raises(TypeError, match="torch.strided"):
            reference.packed_with_lengths(torch.empty(5, 3).to_sparse(), lengths)

    def test_packed_with_lengths_supports_fake_values(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        mode = fake_tensor_mod.FakeTensorMode()
        fake_values = mode.from_tensor(torch.empty(5, 4))

        output = reference.packed_with_lengths(fake_values, torch.tensor([2, 3]))

        assert fake_tensor_mod.is_fake(output.concat)
        assert output.shape == (2, 3, 4)

    def test_packed_with_lengths_outputs_support_standard_operations(self):
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        elements = [torch.randn(2, 4), torch.randn(3, 4)]
        nested = reference.packed_with_lengths(torch.cat(elements), torch.tensor([2, 3]))
        vector = torch.randn(4)

        output = nested + vector

        for actual, expected in zip(output, elements):
            assert_close(actual, expected + vector)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_packed_with_lengths_fullgraph_backward_across_layouts(self):
        reference = NestedTensor([torch.empty(1, 2), torch.empty(1, 2)])
        compiled = torch.compile(
            lambda ref, values, lengths: ref.packed_with_lengths(values, lengths).concat.square().sum(),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        for lengths_tuple in ((2, 4), (3, 5)):
            packed_values = torch.randn(sum(lengths_tuple), 7, requires_grad=True)
            loss = compiled(reference, packed_values, torch.tensor(lengths_tuple))
            loss.backward()

            assert_close(packed_values.grad, 2 * packed_values)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_packed_with_lengths_fullgraph_validates_runtime_lengths(self):
        reference = NestedTensor([torch.empty(1, 2), torch.empty(1, 2)])
        compiled = torch.compile(
            lambda ref, values, lengths: ref.packed_with_lengths(values, lengths).concat,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        assert compiled(reference, torch.randn(6, 7), torch.tensor([2, 4])).shape == (6, 7)
        with pytest.raises(RuntimeError, match="lengths must be non-negative"):
            compiled(reference, torch.randn(6, 7), torch.tensor([-1, 7]))
        with pytest.raises(RuntimeError, match="lengths must sum"):
            compiled(reference, torch.randn(6, 7), torch.tensor([2, 3]))


class TestPackedWithSquareLengths:

    def test_rebuilds_square_topology_and_preserves_public_behavior(self):
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
        assert output.ragged_dims == (0, 1)
        assert output.packed_dim_order == (0, 1, 2, 3)
        assert output.shape == (3, 2, 3, 2, 4)
        assert output.batch_first is False
        assert output.padding_value == -3
        assert output.mask_value is True
        assert [tuple(element.shape) for element in output] == [(0, 0, 2, 4), (3, 3, 2, 4)]
        assert_close(output.concat, packed_values)
        assert_close(output[1], packed_values.reshape(3, 3, 2, 4))
        assert_close(torch.autograd.grad(output.concat.sum(), leaf)[0], 2 * leaf)

    def test_supports_scalar_tail_and_all_zero_lengths(self):
        reference = NestedTensor([torch.empty(1), torch.empty(1), torch.empty(1)])

        scalar_tail = reference.packed_with_square_lengths(torch.arange(13), torch.tensor([2, 0, 3]))
        all_zero = reference.packed_with_square_lengths(torch.empty(0, 5), torch.zeros(3, dtype=torch.long))

        assert [tuple(element.shape) for element in scalar_tail] == [(2, 2), (0, 0), (3, 3)]
        assert all_zero.shape == (3, 0, 0, 5)
        assert [tuple(element.shape) for element in all_zero] == [(0, 0, 5)] * 3

    @pytest.mark.parametrize(
        ("lengths", "error", "match"),
        [
            ([2, 3], TypeError, "lengths must be a dense Tensor"),
            (torch.tensor([2.0, 3.0]), TypeError, "integer dtype"),
            (torch.tensor([-2, 3]), ValueError, "non-negative"),
            (torch.tensor([1, 3]), ValueError, "squared lengths must sum"),
        ],
    )
    def test_validates_square_lengths(self, lengths, error, match):
        reference = NestedTensor([torch.empty(1), torch.empty(1)])

        with pytest.raises(error, match=match):
            reference.packed_with_square_lengths(torch.empty(13, 4), lengths)

    @pytest.mark.parametrize(
        "lengths",
        [(2**32,), (2**31,) * 4],
        ids=("individual-square", "cumulative-square"),
    )
    def test_rejects_int64_overflow_in_square_lengths(self, lengths):
        reference = NestedTensor([torch.empty(1) for _ in lengths])

        with pytest.raises(ValueError, match="int64"):
            reference.packed_with_square_lengths(torch.empty(1, 4), torch.tensor(lengths))

    def test_supports_fake_values_with_concrete_cpu_lengths(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        mode = fake_tensor_mod.FakeTensorMode()
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        fake_values = mode.from_tensor(torch.empty(13, 2, 4))

        output = reference.packed_with_square_lengths(fake_values, torch.tensor([2, 3]))

        assert fake_tensor_mod.is_fake(output.concat)
        assert output.shape == (2, 3, 3, 2, 4)
        assert output.ragged_dims == (0, 1)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_fullgraph_backward_across_square_lengths(self):
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        compiled = torch.compile(
            lambda ref, values, lengths: ref.packed_with_square_lengths(values, lengths).concat.square().sum(),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        for lengths_tuple in ((2, 3), (1, 4)):
            packed_values = torch.randn(sum(length * length for length in lengths_tuple), 4, requires_grad=True)
            loss = compiled(reference, packed_values, torch.tensor(lengths_tuple))
            loss.backward()

            assert_close(packed_values.grad, 2 * packed_values)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_fullgraph_validates_square_lengths_at_runtime(self):
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
        with pytest.raises((ValueError, RuntimeError), match="int64"):
            compiled(reference, torch.randn(13, 4), torch.tensor([2**32, 0]))


class TestPackedWithRectangularLengths:

    def test_rebuilds_rectangular_topology_and_keeps_autograd(self):
        reference = NestedTensor(
            [torch.empty(1), torch.empty(1)],
            batch_first=False,
            padding_value=-3,
            mask_value=True,
        )
        rows = torch.tensor([0, 3], dtype=torch.int32)
        columns = torch.tensor([4, 2], dtype=torch.int64)
        leaf = torch.randn(6, 4, requires_grad=True)
        values = leaf.square()

        output = reference.packed_with_rectangular_lengths(values, rows, columns)

        assert output.ragged_dims == (0, 1)
        assert output.packed_dim_order == (0, 1, 2)
        assert output.shape == (3, 2, 4, 4)
        assert output.batch_first is False
        assert output.padding_value == -3
        assert output.mask_value is True
        assert output.element_sizes().tolist() == [[0, 4, 4], [3, 2, 4]]
        assert [tuple(element.shape) for element in output] == [(0, 4, 4), (3, 2, 4)]
        assert_close(output.concat, values)
        assert_close(torch.autograd.grad(output.concat.sum(), leaf)[0], 2 * leaf)

    @pytest.mark.parametrize(
        ("rows", "columns", "error", "match"),
        [
            ([2, 3], torch.tensor([4, 1]), TypeError, "row_lengths must be a dense Tensor"),
            (torch.tensor([2, 3]), torch.tensor([4.0, 1.0]), TypeError, "integer dtype"),
            (torch.tensor([-2, 3]), torch.tensor([4, 1]), ValueError, "non-negative"),
            (torch.tensor([1, 3]), torch.tensor([4, 1]), ValueError, "must sum to the packed values"),
        ],
    )
    def test_validates_rectangular_metadata(self, rows, columns, error, match):
        reference = NestedTensor([torch.empty(1), torch.empty(1)])

        with pytest.raises(error, match=match):
            reference.packed_with_rectangular_lengths(torch.empty(11, 3), rows, columns)

    def test_supports_fake_values_with_concrete_lengths(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        mode = fake_tensor_mod.FakeTensorMode()
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        fake_values = mode.from_tensor(torch.empty(11, 2))

        output = reference.packed_with_rectangular_lengths(
            fake_values,
            torch.tensor([2, 3]),
            torch.tensor([4, 1]),
        )

        assert fake_tensor_mod.is_fake(output.concat)
        assert output.shape == (2, 3, 4, 2)
        assert output.ragged_dims == (0, 1)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_fullgraph_rectangular_backward_across_layouts(self):
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        compiled = torch.compile(
            lambda ref, values, rows, columns: (
                ref.packed_with_rectangular_lengths(values, rows, columns).concat.square().sum()
            ),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        for row_lengths, column_lengths in (((2, 3), (4, 1)), ((1, 4), (2, 3))):
            values = torch.randn(
                sum(row * column for row, column in zip(row_lengths, column_lengths, strict=True)),
                3,
                requires_grad=True,
            )
            loss = compiled(
                reference,
                values,
                torch.tensor(row_lengths),
                torch.tensor(column_lengths),
            )
            loss.backward()

            assert_close(values.grad, 2 * values)


class TestToDtype:

    @pytest.mark.parametrize("target_dtype", [torch.float32, torch.float64])
    def test_to_dtype_preserves_packed_topology(self, target_dtype):
        elements = [torch.randn(length, 3) for length in (2, 4)]
        nested = NT(elements)

        with nested_execution_guard(
            forbid_iteration=True,
            forbid_storage_map=True,
            forbid_eager_fallback=True,
            forbid_padded_materialization=True,
            forbid_dense_repack=True,
        ):
            output = nested.to(target_dtype)

        assert output.dtype == target_dtype
        assert [element.shape for element in output] == [element.shape for element in nested]
        assert_close(output.concat, torch.cat(elements).to(target_dtype))

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize("target_dtype", [torch.float32, torch.float64])
    def test_to_dtype_dynamic_fullgraph_vjp(self, target_dtype):
        compiled = torch.compile(
            lambda nested: nested.to(target_dtype),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        for lengths in ((2, 4), (3, 1, 5)):
            elements = [torch.randn(length, 3, requires_grad=True) for length in lengths]
            reference_elements = [element.detach().clone().requires_grad_() for element in elements]
            nested = NT(elements, ragged_dims=(0,))
            reference = torch.cat(reference_elements).to(target_dtype)

            output = compiled(nested)
            cotangent = torch.randn_like(reference)
            gradients = torch.autograd.grad(output.concat, elements, cotangent)
            reference_gradients = torch.autograd.grad(reference, reference_elements, cotangent)

            assert output.dtype == target_dtype
            assert output.ragged_dims == nested.ragged_dims
            assert [element.shape for element in output] == [element.shape for element in nested]
            assert_close(output.concat, reference)
            for actual, expected in zip(gradients, reference_gradients, strict=True):
                assert_close(actual, expected)


# ---------------------------------------------------------------------------
# Copy Semantics
# ---------------------------------------------------------------------------


class TestCopySemantics:

    @staticmethod
    def configured():
        return NestedTensor(
            [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])],
            batch_first=False,
            padding_value=-1,
            mask_value=True,
        )

    def test_shallow_copy_shares_values_and_preserves_config(self):
        nested = self.configured()
        shallow = copy.copy(nested)

        assert_close(shallow, nested)
        assert shallow.batch_first is False
        assert shallow.padding_value == -1
        assert shallow.mask_value is True

        shallow.concat[0] = -7.0
        assert nested.concat[0].item() == -7.0

    def test_deep_copy_has_independent_values_and_preserves_config(self):
        nested = self.configured()
        original = nested.concat.clone()
        deep = copy.deepcopy(nested)

        assert_close(deep, nested)
        assert deep.batch_first is False
        assert deep.padding_value == -1
        assert deep.mask_value is True

        deep.concat.fill_(0)
        assert_close(nested.concat, original)
        assert not torch.equal(deep.concat, nested.concat)

    def test_deep_copy_reuses_memoized_objects(self):
        nested = NestedTensor([torch.tensor([1.0, 2.0])])

        first, second = copy.deepcopy([nested, nested])

        assert first is second

    def test_pickle_roundtrip_preserves_values_config_and_accessors(self):
        nested = self.configured()
        _ = nested.tensor_mask

        restored = pickle.loads(pickle.dumps(nested))

        assert type(restored) is NestedTensor
        assert_close(restored, nested)
        assert restored.batch_first is False
        assert restored.padding_value == -1
        assert restored.mask_value is True
        assert_close(restored.tensor, nested.tensor)
        assert_close(restored.mask, nested.mask)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA map_location")
    def test_torch_load_map_location_moves_values(self):
        nested = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        buffer = io.BytesIO()
        torch.save(nested, buffer)
        buffer.seek(0)

        restored = torch.load(buffer, map_location="cuda:0", weights_only=False)

        assert restored.device.type == "cuda"
        assert_close(restored.tensor.cpu(), nested.tensor)

    def test_pickle_roundtrip_preserves_permuted_layout_behavior(self):
        nested = NestedTensor([torch.arange(6.0).reshape(2, 3)]).unsqueeze(1)

        restored = pickle.loads(pickle.dumps(nested))

        assert restored.shape == nested.shape
        assert restored.ragged_dims == nested.ragged_dims
        assert restored.packed_dim_order == nested.packed_dim_order
        assert_close(restored, nested)


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

    @pytest.mark.parametrize("position", [0, -1])
    def test_static_tail_integer_index_preserves_values_and_vjp(self, position):
        template = NT([torch.empty(2, 3), torch.empty(5, 3)], ragged_dims=(0,))
        values = torch.randn_like(template.concat, requires_grad=True)
        coordinates = template.packed_like(values)
        expected = values[:, position]

        output = coordinates[:, :, position]

        cotangent = torch.randn_like(expected)
        actual_grad = torch.autograd.grad(output.concat, values, cotangent)[0]
        expected_grad = torch.autograd.grad(expected, values, cotangent)[0]
        assert type(output) is NestedTensor
        assert [tuple(element.shape) for element in output] == [(2,), (5,)]
        assert_close(output.concat, expected)
        assert_close(actual_grad, expected_grad)

    def test_boolean_scalar_index_keeps_dense_semantics(self):
        coordinates = NT(
            [torch.arange(6).reshape(2, 3), torch.arange(15).reshape(5, 3)],
            ragged_dims=(0,),
        )

        output = coordinates[:, :, True]
        expected = NT([element[:, True] for element in coordinates], ragged_dims=(0,))

        assert_close(output, expected)

    def test_basic_newaxis_index_preserves_values_and_vjp(self):
        template = NT(
            [torch.empty(2, 5, 3), torch.empty(4, 5, 3)],
            ragged_dims=(0,),
        )
        values = torch.randn_like(template.concat, requires_grad=True)
        atoms = template.packed_like(values)
        expected = values[:, None, :, None, :]

        output = atoms[:, :, None, :, None]

        cotangent = torch.randn_like(expected)
        actual_grad = torch.autograd.grad(output.concat, values, cotangent)[0]
        expected_grad = torch.autograd.grad(expected, values, cotangent)[0]
        assert output.shape == torch.Size((2, 4, 1, 5, 1, 3))
        assert output.ragged_dims == (0,)
        assert_close(output.concat, expected)
        assert_close(actual_grad, expected_grad)

    def test_static_and_newaxis_indices_support_fake_tensor(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        coordinates = NT(
            [torch.empty(2, 5, 3), torch.empty(4, 5, 3)],
            ragged_dims=(0,),
        )

        with fake_tensor_mod.FakeTensorMode() as mode:
            fake_coordinates = mode.from_tensor(coordinates)
            indexed = fake_coordinates[:, :, 0]
            expanded = fake_coordinates[:, :, None, :, None]

        assert fake_tensor_mod.is_fake(indexed.concat)
        assert indexed.concat.shape == (6, 3)
        assert indexed.ragged_dims == (0,)
        assert fake_tensor_mod.is_fake(expanded.concat)
        assert expanded.concat.shape == (6, 1, 5, 1, 3)
        assert expanded.ragged_dims == (0,)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_static_and_newaxis_indices_support_fullgraph_backward(self):
        compiled = torch.compile(
            lambda template, values: (
                template.packed_like(values)[:, :, 0].concat,
                template.packed_like(values)[:, :, None, :, None].concat,
            ),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        for lengths in ((2, 3), (3, 5)):
            template = NT([torch.empty(length, 5, 3) for length in lengths], ragged_dims=(0,))
            values = torch.randn_like(template.concat, requires_grad=True)
            expected_indexed = values[:, 0]
            expected_expanded = values[:, None, :, None, :]

            indexed, expanded = compiled(template, values)

            indexed_cotangent = torch.randn_like(expected_indexed)
            expanded_cotangent = torch.randn_like(expected_expanded)
            actual_grad = torch.autograd.grad(
                (indexed, expanded),
                values,
                (indexed_cotangent, expanded_cotangent),
            )[0]
            expected_grad = torch.autograd.grad(
                (expected_indexed, expected_expanded),
                values,
                (indexed_cotangent, expanded_cotangent),
            )[0]
            assert_close(indexed, expected_indexed)
            assert_close(expanded, expected_expanded)
            assert_close(actual_grad, expected_grad)

    @pytest.mark.parametrize("position", [3, -4])
    def test_static_tail_integer_index_preserves_bounds_errors(self, position):
        coordinates = NT([torch.empty(2, 3), torch.empty(5, 3)], ragged_dims=(0,))

        with pytest.raises(IndexError):
            _ = coordinates[:, :, position]

    def test_multi_ragged_static_channel_slice_fullgraph(self):
        template = NT(
            [torch.empty(2, 3, 4), torch.empty(3, 2, 4)],
            ragged_dims=(0, 1),
        )

        def slice_channels(reference, values):
            output = reference.packed_like(values)[:, :, :, 1:3]
            return output.concat, output.element_sizes(), output.ragged_dims

        compiled = torch.compile(slice_channels, backend="aot_eager", fullgraph=True, dynamic=True)
        values = torch.randn_like(template.concat, requires_grad=True)
        output, element_sizes, ragged_dims = compiled(template, values)
        expected = values[:, 1:3]

        cotangent = torch.randn_like(expected)
        actual_grad = torch.autograd.grad(output, values, cotangent)[0]
        expected_grad = torch.autograd.grad(expected, values, cotangent)[0]
        assert_close(output, expected)
        assert_close(element_sizes, torch.tensor([[2, 3, 2], [3, 2, 2]]))
        assert ragged_dims == (0, 1)
        assert_close(actual_grad, expected_grad)

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
        nt = NestedTensor([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
        _ = nt.tensor
        _ = nt.mask

        nt[0] = torch.tensor([10.0, 20.0, 30.0])

        assert_close(nt[0], torch.tensor([10.0, 20.0, 30.0]))
        assert_close(nt[1], torch.tensor([4.0, 5.0]))
        assert_close(nt.tensor, torch.tensor([[10.0, 20.0, 30.0], [4.0, 5.0, 0.0]]))

    def test_getitem_returns_a_mutable_view(self):
        nt = NestedTensor(
            [
                torch.arange(24.0).reshape(2, 3, 4),
                torch.arange(24.0, 56.0).reshape(2, 4, 4),
            ]
        )
        first = nt[0]
        first[0, 0, 0] = -1.0

        assert nt.tensor[0, 0, 0, 0].item() == -1.0
        assert nt.concat[0, 0, 0].item() == -1.0

    def test_materialized_tensor_reflects_getitem_alias_mutation(self):
        nt = NestedTensor([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
        _ = nt.tensor
        first = nt[0]
        first[0] = -7.0
        assert nt.tensor[0, 0].item() == -7.0

    def test_materialized_tensor_reflects_concat_alias_mutation(self):
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

        for actual, expected in zip(output, nt):
            assert_close(actual, expected[..., None])

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

    def test_setitem_accepts_a_different_shape(self):
        nt = NestedTensor([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])

        nt[1] = torch.tensor([9.0, 10.0, 11.0, 12.0])

        assert_close(nt[0], torch.tensor([1.0, 2.0, 3.0]))
        assert_close(nt[1], torch.tensor([9.0, 10.0, 11.0, 12.0]))
        assert nt.shape == torch.Size([2, 4])

    def test_setitem_accepts_trailing_shape_change(self):
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

    def test_setitem_tuple_scalar_assignment_updates_each_element(self):
        nt = NestedTensor(
            [
                torch.arange(6.0).reshape(2, 3),
                torch.arange(3.0).reshape(1, 3),
            ]
        )
        nt[:, 0] = -1.0
        assert_close(nt[0], torch.tensor([[-1.0, -1.0, -1.0], [3.0, 4.0, 5.0]]))
        assert_close(nt[1], torch.tensor([[-1.0, -1.0, -1.0]]))

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


# ---------------------------------------------------------------------------
# Comparison Operators
# ---------------------------------------------------------------------------


class TestMutationCoherence:

    def test_materialized_views_follow_runtime_settings(self):
        nested = NestedTensor([torch.tensor([1.0, 2.0]), torch.tensor([3.0])], padding_value=-1.0)
        _ = nested.tensor_mask

        nested.padding_value = 7.0
        assert_close(nested.tensor, torch.tensor([[1.0, 2.0], [3.0, 7.0]]))

        old_mask = nested.mask.clone()
        nested.mask_value = True
        assert_close(nested.mask, ~old_mask)

    def test_inplace_unary_updates_elements_and_materialized_tensor(self):
        nested = NestedTensor([torch.tensor([-1.0, 2.0]), torch.tensor([-3.0])])
        _ = list(nested)
        _ = nested.tensor_mask

        nested.relu_()

        assert_close(nested[0], torch.tensor([0.0, 2.0]))
        assert_close(nested[1], torch.tensor([0.0]))
        assert_close(nested.tensor, torch.tensor([[0.0, 2.0], [0.0, 0.0]]))

    def test_inplace_binary_updates_elements_and_materialized_tensor(self):
        nested = NestedTensor([torch.tensor([1.0, 2.0]), torch.tensor([3.0])])
        _ = list(nested)
        _ = nested.tensor

        nested.add_(1.5)

        assert_close(nested[0], torch.tensor([2.5, 3.5]))
        assert_close(nested[1], torch.tensor([4.5]))
        assert_close(nested.tensor, torch.tensor([[2.5, 3.5], [4.5, 0.0]]))

    def test_copy_updates_public_values_after_views_are_materialized(self):
        destination = NestedTensor([torch.tensor([1.0, 2.0]), torch.tensor([3.0])])
        source = NestedTensor([torch.tensor([9.0, 8.0]), torch.tensor([7.0])])
        _ = list(destination)
        _ = destination.tensor

        destination.copy_(source)

        assert_close(destination, source)
        assert_close(destination.tensor, source.tensor)

    def test_copy_requires_matching_public_layout(self):
        elements = [torch.randn(2, 5, 8), torch.randn(2, 3, 8)]
        destination = NestedTensor(elements)
        source = NestedTensor([element.clone() for element in elements], ragged_dims=(0, 1))

        assert destination.shape == source.shape
        assert destination.packed_dim_order != source.packed_dim_order
        with pytest.raises(NotImplementedError, match="matching packed layout"):
            destination.copy_(source)


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
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_nested_like_compile_matches_reference(self):
        nested_tensor = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        dense = nested_tensor.tensor
        compiled = torch.compile(lambda x, y: x.nested_like(y), backend="inductor", fullgraph=True)
        output = compiled(nested_tensor, dense)
        reference = nested_tensor.nested_like(dense)
        assert isinstance(output, NestedTensor)
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_nested_like_larger_dense_compile_matches_reference(self):
        nested_tensor = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        dense = torch.randn(2, 6, 3)
        compiled = torch.compile(lambda x, y: x.nested_like(y, strict=False), backend="inductor", fullgraph=True)
        output = compiled(nested_tensor, dense)
        reference = nested_tensor.nested_like(dense, strict=False)
        assert isinstance(output, NestedTensor)
        assert_close(output, reference)

    def test_nested_like_smaller_dense(self):
        nested_tensor = NestedTensor([torch.arange(6.0).reshape(2, 3), torch.arange(12.0).reshape(4, 3)])
        dense = torch.arange(12.0).reshape(2, 2, 3)
        output = nested_tensor.nested_like(dense, strict=False)

        assert_close(output[0], dense[0, :2, :3])
        assert_close(output[1], dense[1, :2, :3])

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_reshape_compile_matches_reference(self):
        nested_tensor = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        compiled = torch.compile(lambda x: x.reshape(-1, 3), backend="inductor", fullgraph=True)
        output = compiled(nested_tensor)
        reference = nested_tensor.reshape(-1, 3)
        assert isinstance(output, NestedTensor)
        assert_close(output, reference)

    def test_view_irregular_tail_matches_elementwise_reference(self):
        nested_tensor = NestedTensor([torch.arange(8.0).reshape(2, 4), torch.arange(8.0, 14.0).reshape(3, 2)])
        viewed = nested_tensor.view(len(nested_tensor), 2, -1)

        for actual, element in zip(viewed, nested_tensor):
            assert_close(actual, element.view(2, -1))

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

    def _assert_state(self, output, *, batch_first=False, padding_value=-1, mask_value=True):
        assert isinstance(output, NestedTensor)
        assert output.batch_first is batch_first
        assert output.padding_value == padding_value
        assert output.mask_value is mask_value

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
    def test_comparison_preserves_state(self):
        state = {"batch_first": False, "padding_value": -1, "mask_value": True, "pin_memory": True}
        left = NestedTensor([torch.tensor([[2, 0], [1, 0]])], **state)
        right = NestedTensor([torch.tensor([[1, 0], [1, 0]])], **state)
        output = left > right
        self._assert_state(output)
        assert output.concat.is_pinned()

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
            self._assert_state(output)
            assert output.concat.is_pinned()

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
        output = NestedTensor.from_concatenated(concat, shapes)
        assert_close(output, nested_tensor)

        nested_tensor = NestedTensor([torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)], batch_first=False)
        assert_close(nested_tensor.concat, torch.tensor([1.0, 2.0, 3.0]))

        concat, shapes = nested_tensor.concatenate()
        output = NestedTensor.from_concatenated(concat, shapes, batch_first=False)
        assert_close(output, nested_tensor)

    def test_flat_batch_first_false(self):
        nested_tensor = NestedTensor(
            [torch.arange(3).unsqueeze(1), torch.arange(3, 7).unsqueeze(1)],
            batch_first=False,
        )
        flat = nested_tensor.concat
        assert flat.shape == torch.Size([7, 1])
        assert_close(flat.squeeze(1), torch.arange(7))

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
