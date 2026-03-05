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
import random

import pytest
import torch
from packaging.version import Version

from danling.tensors import NestedTensor
from tests.tensors.utils import assert_close

NT = NestedTensor
TORCH_VERSION = Version(torch.__version__.split("+")[0])
random.seed(1016)


# ---------------------------------------------------------------------------
# Construction & Validation
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

    def test_bool_empty_nested_tensor(self):
        assert bool(NestedTensor([])) is False


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

    def test_from_tensor_mask_ndim3_sparse_selects_only_true_values(self):
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
        output = NestedTensor.from_tensor_mask(padded, mask)
        reference = NT(
            [
                torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
                torch.tensor([[5.0, 6.0]]),
            ]
        )
        assert_close(output, reference)

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

    def test_concatenate_empty_preserves_dtype(self):
        nested_tensor = NestedTensor([], dtype=torch.float64)
        concat, shapes = nested_tensor.concatenate()
        assert concat.dtype == torch.float64
        assert concat.device == torch.device("cpu")
        assert shapes == ()


# ---------------------------------------------------------------------------
# Tensor / Mask Properties
# ---------------------------------------------------------------------------


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

    def test_comparison_preserves_state(self):
        state = {"batch_first": False, "padding_value": -1, "mask_value": True, "pin_memory": True}
        left = NestedTensor([torch.tensor([[2, 0], [1, 0]])], **state)
        right = NestedTensor([torch.tensor([[1, 0], [1, 0]])], **state)
        output = left > right
        self._assert_state(output, pin_memory=True)

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


class TestPackedCacheInvalidation:

    def test_inplace_unary_invalidates_storage_cache(self):
        nested_tensor = NestedTensor([torch.tensor([-1.0, 2.0]), torch.tensor([-3.0])])
        _ = nested_tensor._storage
        assert nested_tensor._cached_storage is not None
        nested_tensor.relu_()
        assert nested_tensor._cached_storage is None
        assert_close(nested_tensor[0], torch.tensor([0.0, 2.0]))
        assert_close(nested_tensor[1], torch.tensor([0.0]))

    def test_inplace_binary_invalidates_storage_cache(self):
        nested_tensor = NestedTensor([torch.tensor([1.0, 2.0]), torch.tensor([3.0])])
        _ = nested_tensor._storage
        assert nested_tensor._cached_storage is not None
        nested_tensor.add_(1.5)
        assert nested_tensor._cached_storage is None
        assert_close(nested_tensor[0], torch.tensor([2.5, 3.5]))
        assert_close(nested_tensor[1], torch.tensor([4.5]))

    def test_copy_invalidates_storage_cache(self):
        dest = NestedTensor([torch.tensor([1.0, 2.0]), torch.tensor([3.0])])
        src = NestedTensor([torch.tensor([9.0, 8.0]), torch.tensor([7.0])], **dest._meta(include_dtype=False))
        _ = dest._storage
        assert dest._cached_storage is not None
        dest.copy_(src)
        assert dest._cached_storage is None
        assert_close(dest, src)


# ---------------------------------------------------------------------------
# Indexing (__getitem__, __setitem__)
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
        nt[0] = torch.tensor([10.0, 20.0, 30.0])
        assert_close(nt[0], torch.tensor([10.0, 20.0, 30.0]))
        assert_close(nt[1], torch.tensor([4.0, 5.0]))

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

    def test_setitem_negative_index(self):
        nt = NestedTensor([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])])
        nt[-1] = torch.tensor([10.0, 20.0])
        assert_close(nt[1], torch.tensor([10.0, 20.0]))

    def test_setitem_out_of_range_raises(self):
        nt = NestedTensor([torch.tensor([1.0, 2.0])])
        with pytest.raises(IndexError):
            nt[5] = torch.tensor([1.0])


# ---------------------------------------------------------------------------
# Comparison Operators
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


# ---------------------------------------------------------------------------
# Arithmetic Operators
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
            pytest.param(
                torch.tensor([[6, 5, 4], [3, 2, 1]]),
                marks=pytest.mark.xfail(reason="pow.Tensor_Tensor non-scalar not implemented"),
            ),
            pytest.param(
                torch.randn(2, 3),
                marks=pytest.mark.xfail(reason="pow.Tensor_Tensor non-scalar not implemented"),
            ),
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

        with pytest.raises(ValueError, match="Cannot transpose the batch dimension"):
            nested_tensor2.transpose(0, 1)

        with pytest.raises(ValueError, match="Cannot transpose the batch dimension"):
            nested_tensor2.transpose(1, 0)

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

    def test_method_chaining(self):
        nested_tensor = NestedTensor([torch.randn(2, 3, 4), torch.randn(2, 3, 4)])

        output = nested_tensor.transpose(1, 2).reshape(len(nested_tensor), -1, 6).view(24, 1)
        assert output is not nested_tensor
        assert output.shape == torch.Size([2, 24, 1])
        assert nested_tensor.shape == torch.Size([2, 2, 3, 4])


# ---------------------------------------------------------------------------
# Cat / Concatenation
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


class TestPackOptimization:

    def test_pack_shape_tensor_values(self):
        """Verify _pack produces correct shape_tensor for mixed-shape inputs."""
        t1 = torch.tensor([[1, 2], [3, 4]])  # shape (2, 2)
        t2 = torch.tensor([[5, 6, 7]])  # shape (1, 3)
        nt = NestedTensor(t1, t2)
        expected = torch.tensor([[2, 2], [1, 3]], dtype=torch.long)
        torch.testing.assert_close(nt._shape_tensor, expected)

    def test_pack_shape_tensor_scalars(self):
        """Verify _pack handles scalar tensors (ndim=0)."""
        nt = NestedTensor(torch.tensor(1.0), torch.tensor(2.0))
        assert nt._shape_tensor.shape == (2, 0) or nt._shape_tensor.numel() == 0

    def test_pack_shape_tensor_1d(self):
        """Verify _pack handles 1D tensors with different lengths."""
        nt = NestedTensor([1, 2, 3], [4, 5])
        expected = torch.tensor([[3], [2]], dtype=torch.long)
        torch.testing.assert_close(nt._shape_tensor, expected)

    def test_pack_empty(self):
        """Verify _pack handles empty tensor list."""
        values, offsets, shape_tensor = NestedTensor._pack(())
        assert values.numel() == 0
        assert offsets.shape == (1,)
        assert shape_tensor.numel() == 0
