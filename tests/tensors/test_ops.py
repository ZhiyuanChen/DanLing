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

r"""Tests for ``danling.tensors.ops`` — shared helpers and dimension translation."""

import pytest
import torch
from torch.nn import functional as F

from danling.tensors import NestedTensor, aten_functions, nn_functions
from danling.tensors.ops import (
    NestedTensorAtenRegistry,
    NestedTensorFuncRegistry,
    _concat_dim_for_tensor_dim,
    _map_storage_serial,
    _packed_layer_norm,
    _stack_or_nest,
    _translate_dim,
    _translate_dims,
    nested_execution_guard,
)

NT = NestedTensor


class TestCompilePolicy:

    def test_torch_binary_registry_rejects_dense_broadcast_fallback(self):
        nt = NT([torch.randn(2), torch.randn(1)])
        assert NestedTensorFuncRegistry.is_compile_safe(torch.add, (nt, 1.0), {})
        assert NestedTensorFuncRegistry.is_compile_safe(torch.add, (nt, torch.ones_like(nt.tensor)), {})
        assert not NestedTensorFuncRegistry.is_compile_safe(torch.add, (nt, torch.ones(2)), {})

    def test_aten_random_creation_registry_is_eager_only(self):
        nt = NT([torch.randn(2), torch.randn(1)])
        assert not NestedTensorAtenRegistry.is_compile_safe(torch.ops.aten.randn_like.default, (nt,), {})

    def test_aten_sort_and_cumulative_registry_accepts_static_dims(self):
        nt = NT([torch.randn(2, 3), torch.randn(1, 3)])
        assert NestedTensorAtenRegistry.is_compile_safe(torch.ops.aten.sort.default, (nt, 2, False), {})
        assert NestedTensorAtenRegistry.is_compile_safe(torch.ops.aten.topk.default, (nt, 2, 2, True, True), {})
        assert NestedTensorAtenRegistry.is_compile_safe(torch.ops.aten.cumsum.default, (nt, 2), {})
        assert NestedTensorAtenRegistry.is_compile_safe(torch.ops.aten.cummax.default, (nt, 2), {})
        assert NestedTensorAtenRegistry.is_compile_safe(torch.ops.aten.flip.default, (nt, [2]), {})

    def test_aten_sort_and_cumulative_registry_rejects_ragged_dims(self):
        nt = NT([torch.randn(2, 3), torch.randn(1, 3)])
        assert not NestedTensorAtenRegistry.is_compile_safe(torch.ops.aten.sort.default, (nt, 1, False), {})
        assert not NestedTensorAtenRegistry.is_compile_safe(torch.ops.aten.topk.default, (nt, 2, 1, True, True), {})
        assert not NestedTensorAtenRegistry.is_compile_safe(torch.ops.aten.cumsum.default, (nt, 1), {})
        assert not NestedTensorAtenRegistry.is_compile_safe(torch.ops.aten.cummax.default, (nt, 1), {})
        assert not NestedTensorAtenRegistry.is_compile_safe(torch.ops.aten.flip.default, (nt, [1]), {})


class TestDimTranslation:

    def test_concat_dim_for_tensor_dim(self):
        nt = NT([torch.randn(2, 3), torch.randn(1, 3)])
        assert _concat_dim_for_tensor_dim(nt, 0) is None
        assert _concat_dim_for_tensor_dim(nt, 1) == 1
        assert _concat_dim_for_tensor_dim(nt, -1) == 1

    def test_concat_dim_for_tensor_dim_out_of_range(self):
        nt = NT([torch.randn(2, 3), torch.randn(1, 3)])
        with pytest.raises(IndexError):
            _concat_dim_for_tensor_dim(nt, 2)

    def test_translate_dim_batch_first(self):
        nt = NT([torch.randn(2, 3), torch.randn(1, 3)])
        assert _translate_dim(nt, 1) == 0
        assert _translate_dim(nt, 2) == 1
        with pytest.raises(ValueError, match="batch dimension"):
            _translate_dim(nt, 0)

    def test_translate_dims_batch_first_false(self):
        nt = NT([torch.randn(2, 3), torch.randn(1, 3)], batch_first=False)
        assert _translate_dims(nt, (0, 2)) == (0, 1)
        with pytest.raises(ValueError, match="batch dimension"):
            _translate_dims(nt, (1,))


class TestDropoutValidation:

    def test_dropout_probability_error_types(self):
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


class TestExecutionGuards:

    def test_nested_execution_guard_blocks_iteration(self):
        nt = NT([torch.randn(2, 3), torch.randn(1, 3)])
        with nested_execution_guard(forbid_iteration=True), pytest.raises(RuntimeError, match="iterated storage"):
            tuple(nt)

    def test_nested_execution_guard_blocks_storage_map(self):
        nt = NT([torch.randn(2, 3), torch.randn(1, 3)])
        with nested_execution_guard(forbid_storage_map=True), pytest.raises(RuntimeError, match="storage mapping"):
            _map_storage_serial(nt, lambda t: t)

    def test_nested_execution_guard_blocks_padded_materialization(self):
        nt = NT([torch.randn(2, 3), torch.randn(1, 3)])
        with (
            nested_execution_guard(forbid_padded_materialization=True),
            pytest.raises(RuntimeError, match="materialized padded storage"),
        ):
            _ = nt.tensor

    def test_nested_execution_guard_blocks_dense_repack(self):
        nt = NT([torch.randn(2, 3), torch.randn(1, 3)])
        batch_dense = torch.randn(len(nt), 2, 3)
        with (
            nested_execution_guard(forbid_dense_repack=True),
            pytest.raises(RuntimeError, match="repacked from dense storage"),
        ):
            nn_functions._nested_from_batch_leading_tensor(nt, batch_dense)


class TestFakeTensors:

    def test_offsets_with_fake_tensors(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        FakeTensorMode = fake_tensor_mod.FakeTensorMode

        with FakeTensorMode():
            offsets_a = torch.tensor([0, 2, 5], dtype=torch.long)
            offsets_b = torch.tensor([0, 2, 5], dtype=torch.long)
            offsets_c = torch.tensor([0, 2], dtype=torch.long)
            assert aten_functions._offsets_match_identity_if_fake(offsets_a, offsets_a)
            # Under fake tensor mode, identity is required (conservative)
            assert not aten_functions._offsets_match_identity_if_fake(offsets_a, offsets_b)
            assert not aten_functions._offsets_match_identity_if_fake(offsets_a, offsets_c)

    def test_same_layout_allows_independent_fake_metadata(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        FakeTensorMode = fake_tensor_mod.FakeTensorMode

        with FakeTensorMode():
            lhs = NT([torch.empty(2, 3), torch.empty(1, 3)])
            rhs = NT([torch.empty(2, 3), torch.empty(1, 3)])
            mismatched = NT([torch.empty(3, 3), torch.empty(1, 3)])

            assert lhs._has_same_structure(rhs)
            assert lhs._has_same_layout(rhs)
            assert not lhs._has_same_structure(mismatched)

    def test_meta_tensor_equal_rejects_distinct_fake_metadata(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        FakeTensorMode = fake_tensor_mod.FakeTensorMode

        with FakeTensorMode():
            lhs = torch.tensor([[2, 3], [1, 3]], dtype=torch.long)
            rhs = torch.tensor([[9, 9], [9, 9]], dtype=torch.long)

            assert NT._meta_tensor_equal(lhs, lhs)
            assert not NT._meta_tensor_equal(lhs, rhs)

    def test_fake_clone_and_gather_preserve_layout_with_cached_python_meta(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        FakeTensorMode = fake_tensor_mod.FakeTensorMode

        with FakeTensorMode():
            source = NT([torch.empty(2, 3), torch.empty(1, 3)])
            cloned = torch.ops.aten.clone.default(source)
            index = NT(
                [
                    torch.zeros(2, 3, dtype=torch.long),
                    torch.zeros(1, 3, dtype=torch.long),
                ]
            )
            gathered = torch.ops.aten.gather.default(source, 1, index)

            assert source._has_same_layout(cloned)
            assert index._has_same_layout(gathered)

    def test_fake_batch_index_select_requires_concrete_index(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        FakeTensorMode = fake_tensor_mod.FakeTensorMode

        with FakeTensorMode():
            source = NT([torch.empty(2, 3), torch.empty(1, 3)])
            index = torch.tensor([1, 0, 1], dtype=torch.long)

            with pytest.raises(NotImplementedError, match="requires a concrete batch index"):
                torch.ops.aten.index_select.default(source, 0, index)


class TestNormalizationFastpaths:

    def test_packed_layer_norm_propagates_type_error(self, monkeypatch):
        nt = NT([torch.randn(2, 4), torch.randn(1, 4)])

        def _raise_type_error(*_args, **_kwargs):
            raise TypeError("bad fast-path call")

        monkeypatch.setattr(torch.ops.aten.native_layer_norm, "default", _raise_type_error)
        with pytest.raises(TypeError):
            _packed_layer_norm(nt, (4,), None, None, 1e-5)


class TestStackOrNest:

    def test_stack_or_nest_avoids_stack_for_mismatched_shapes(self, monkeypatch):
        nt = NT([torch.randn(2, 3), torch.randn(1, 3)])
        values = [torch.randn(2), torch.randn(3)]
        stack_called = False

        def _unexpected_stack(_values):
            nonlocal stack_called
            stack_called = True
            raise AssertionError("torch.stack should not be called for mismatched shapes")

        monkeypatch.setattr(torch, "stack", _unexpected_stack)
        output = _stack_or_nest(values, nt)

        assert not stack_called
        assert isinstance(output, NestedTensor)
        assert_close = torch.testing.assert_close
        assert_close(output[0], values[0])
        assert_close(output[1], values[1])
