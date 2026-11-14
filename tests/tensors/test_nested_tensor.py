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
import gc
import io
import pickle
import weakref

import pytest
import torch
from packaging.version import Version
from torch.nn import functional as F

from danling.tensors import NestedTensor
from tests.tensors.utils import assert_close

NT = NestedTensor


TORCH_VERSION = Version(torch.__version__.split("+")[0])


class TestFunctionTransforms:

    @pytest.mark.parametrize("lengths", [(1, 2), (2, 4)])
    @pytest.mark.parametrize("batch_first", [False, True])
    def test_compiled_transpose(self, device, lengths, batch_first):
        parts = [torch.randn(n, 3, device=device, dtype=torch.float64, requires_grad=True) for n in lengths]
        source = NT(parts, ragged_dims=(0,), batch_first=batch_first)
        operation = torch.compile(lambda value: value.transpose(0, 1).sin(), backend="aot_eager", fullgraph=True)
        output = operation(source)
        assert output.batch_first is not batch_first
        assert_close(output.unbind(1 if output.batch_first is False else 0), [part.sin() for part in parts])
        assert_close(torch.autograd.grad(output.square().sum(), parts), [2 * part.sin() * part.cos() for part in parts])

    @pytest.mark.parametrize("wrapper_leaf", [False, True])
    def test_reverse_transforms_preserve_ragged_geometry(self, device, wrapper_leaf):
        source = NT(
            [
                torch.randn(2, 3, 2, dtype=torch.float64, device=device),
                torch.randn(4, 2, 2, dtype=torch.float64, device=device),
            ],
            ragged_dims=(1, 0),
        )
        if wrapper_leaf:
            source.requires_grad_()
        values = source.values().detach().clone()

        def loss(value):
            return value.sin().square().sum()

        gradient = torch.func.grad(loss)(source)
        assert_close(gradient.element_sizes(), source.element_sizes())
        assert_close(gradient.values(), 2 * values.sin() * values.cos())
        second = torch.func.grad(lambda value: torch.func.grad(loss)(value).sum())(source)
        assert_close(second.values(), 2 * (2 * values).cos())
        output, pullback = torch.func.vjp(torch.sin, source)
        (gradient,) = pullback(torch.ones_like(output))
        assert_close(gradient.values(), values.cos())
        assert source.grad is None

    def test_native_forward_ad_preserves_packed_tangents(self, device):
        elements = [
            torch.randn(2, 3, 2, dtype=torch.float64, device=device),
            torch.randn(4, 2, 2, dtype=torch.float64, device=device),
        ]
        source = NT(elements, ragged_dims=(1, 0)).requires_grad_()
        direction = NT([torch.randn_like(element) for element in elements], ragged_dims=(0, 1))
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(source, direction)
            output = dual.pow(3)
            primal, tangent = torch.autograd.forward_ad.unpack_dual(output)
            for actual, value, vector in zip(tangent, elements, direction):
                assert_close(actual, 3 * value.square() * vector)
            packed_primal, packed_tangent = torch.autograd.forward_ad.unpack_dual(output.values())
            assert_close(packed_primal, primal.values())
            assert_close(packed_tangent, tangent.values())

    def test_jvp_and_factory_transforms(self, device):
        values = torch.randn(5, dtype=torch.float64, device=device)
        direction = torch.randn_like(values)

        def loss(packed):
            return NT([packed[:2], packed[2:]]).sin().sum()

        assert_close(torch.func.grad(loss)(values), values.cos())
        second = torch.func.grad(lambda packed: torch.func.grad(loss)(packed).sum())(values)
        assert_close(second, -values.sin())
        assert_close(torch.func.jacfwd(loss)(values), values.cos())
        primal, tangent = torch.func.jvp(loss, (values,), (direction,))
        assert_close(primal, values.sin().sum())
        assert_close(tangent, (values.cos() * direction).sum())
        source = NT([values[:2], values[2:]])
        vector = NT([direction[:2], direction[2:]])
        primal, tangent = torch.func.jvp(torch.sin, (source,), (vector,))
        assert_close(primal.values(), values.sin())
        assert_close(tangent.values(), values.cos() * direction)

    def test_vectorized_jacrev_and_static_vmap(self, device):
        elements = [
            torch.randn(2, 3, 2, dtype=torch.float64, device=device),
            torch.randn(4, 2, 2, dtype=torch.float64, device=device),
        ]
        source = NT(elements, ragged_dims=(1, 0))
        for chunk_size in (None, 1):
            gradient = torch.func.jacrev(lambda value: value.square().sum(), chunk_size=chunk_size)(source)
            assert_close(gradient.values(), 2 * source.values())
        jacobian = torch.func.jacrev(lambda value: torch.stack((value.sum(), value.square().sum())))(source)
        for actual, element in zip(jacobian, elements):
            assert_close(actual, torch.stack((torch.ones_like(element), 2 * element)))
        sums = torch.vmap(lambda value: value.sum(), in_dims=-1)(source)
        assert_close(sums, sum(element.sum(dim=(0, 1)) for element in elements))
        per_sample = torch.vmap(lambda value: value.square().sum())(source)
        assert_close(per_sample, torch.stack([element.square().sum() for element in elements]))

    def test_packed_value_jacobians(self, device):
        template = NT(
            [torch.zeros(2, dtype=torch.float64, device=device), torch.zeros(3, dtype=torch.float64, device=device)]
        )
        values = torch.linspace(0.1, 0.5, 5, dtype=torch.float64, device=device)

        def loss(packed):
            return template.packed_like(packed).sin().sum()

        assert_close(torch.func.jacfwd(loss)(values), values.cos())
        assert_close(torch.func.jacrev(loss)(values), values.cos())

    @pytest.mark.parametrize("shapes,ragged_dims", [(((2, 3), (4, 3)), (0,)), (((2, 3, 2), (4, 2, 2)), (1, 0))])
    def test_packed_like_vmap_values_and_derivatives(self, device, shapes, ragged_dims):
        template = NT(
            [torch.zeros(shape, device=device, dtype=torch.float64) for shape in shapes], ragged_dims=ragged_dims
        )
        values = torch.randn((3, *template.values().shape), device=device, dtype=torch.float64, requires_grad=True)

        def loss(packed):
            return template.packed_like(packed).sin().sum()

        output = torch.vmap(loss)(values)
        assert_close(output, values.sin().flatten(1).sum(1))
        assert_close(torch.autograd.grad(output.sum(), values)[0], values.cos())
        gradient = torch.vmap(torch.func.grad(loss))(values)
        assert_close(gradient, values.cos())
        second = torch.autograd.grad(gradient.sum(), values, create_graph=True)[0]
        assert_close(second, -values.sin())
        third = torch.autograd.grad(second.sum(), values)[0]
        assert_close(third, -values.cos())

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param([1.2, 2.3, 3.4, 4.5, 5.6], id="nonzero"),
            pytest.param([0.0, 2.3, 3.4, 4.5, 0.0], id="single-zero"),
            pytest.param([0.0, 0.0, 3.4, 0.0, 5.6], id="multiple-zeros"),
        ],
    )
    @pytest.mark.parametrize(
        "layout", ["scalar", "leading-channels", "trailing-channels", "leading-channels-batch-second"]
    )
    def test_cumprod_transform_derivatives(self, device, values, layout):
        values = torch.tensor(values, dtype=torch.float64, device=device)
        if layout != "scalar":
            values = values[:, None] * values.new_tensor([1.0, 0.5])
        direction = torch.randn_like(values)
        elements = values.split((2, 3))
        vectors = direction.split((2, 3))
        trailing = layout == "trailing-channels"
        if trailing:
            elements = [part.t() for part in elements]
            vectors = [part.t() for part in vectors]
        batch_first = not trailing and layout != "leading-channels-batch-second"
        source = NT(elements, ragged_dims=(1 if trailing else 0,), batch_first=batch_first)
        vector = NT(vectors, ragged_dims=source.ragged_dims, batch_first=source.batch_first)
        dim = 2 if trailing else 1 if batch_first else 0

        def reference(packed):
            return torch.cat([part.cumprod(0) for part in packed.split((2, 3))])

        expected, expected_tangent = torch.func.jvp(reference, (values,), (direction,))
        actual, tangent = torch.func.jvp(lambda value: torch.cumprod(value, dim=dim), (source,), (vector,))
        assert_close(actual.values(), expected)
        assert_close(tangent.values(), expected_tangent)

        def loss(packed):
            return source.packed_like(packed).cumprod(dim=dim).sum()

        expected_gradient = torch.func.grad(lambda packed: reference(packed).sum())(values)
        assert_close(torch.func.grad(loss)(values), expected_gradient)

    @pytest.mark.parametrize("index", [(slice(None), slice(None), slice(1, None)), (slice(None), slice(None), 1)])
    @pytest.mark.parametrize("batch_first", [False, True])
    def test_view_gradients(self, device, index, batch_first):
        template = NT(
            [torch.zeros(n, 3, device=device, dtype=torch.float64) for n in (2, 4)],
            ragged_dims=(0,),
            batch_first=batch_first,
        )
        values = torch.randn(6, 3, device=device, dtype=torch.float64)

        def loss(packed):
            return template.packed_like(packed)[index].sin().sum()

        def reference(packed):
            return packed[:, index[-1]].sin().sum()

        assert_close(torch.func.grad(loss)(values), torch.func.grad(reference)(values))
        gradient = torch.func.grad(lambda packed: torch.func.grad(loss)(packed).sum())(values)
        expected = torch.func.grad(lambda packed: torch.func.grad(reference)(packed).sum())(values)
        assert_close(gradient, expected)

    @pytest.mark.parametrize("direct", [False, True])
    @pytest.mark.parametrize(
        "operation,shapes",
        [
            ("slice", ((2, 3, 2), (4, 2, 2))),
            ("select", ((2, 3, 2), (4, 2, 2))),
            ("chain", ((2, 3, 2), (4, 2, 2))),
            ("slice", ((0, 3, 2), (1, 0, 2))),
        ],
    )
    def test_ragged_views(self, device, direct, operation, shapes):
        parts = tuple(torch.randn(shape, dtype=torch.double, device=device) for shape in shapes)
        source = NT(parts, ragged_dims=(1, 0))

        def transform(value, nested):
            if operation == "select":
                return value.select(int(nested), 0)
            if operation == "chain":
                value = F.relu((value + 1).select(-1, 1).sin(), inplace=True)
                return torch.ops.aten.slice.Tensor(value, int(nested), 1, None, 1)
            return torch.ops.aten.slice.Tensor(value, 1 + int(nested), -2, None, 2)

        def reference(examples):
            return sum(transform(value, False).sin().square().sum() for value in examples)

        def loss(value):
            if not direct:
                value = NT(value, ragged_dims=(1, 0))
                value = value.packed_like(value.values())
            assert value.requires_grad
            return transform(value, True).sin().square().sum()

        argument = source if direct else parts
        actual = torch.func.grad(loss)(argument)
        expected = torch.func.grad(reference)(parts)
        if direct:
            assert actual.ragged_dims == source.ragged_dims
            assert_close(actual.element_sizes(), source.element_sizes())
        assert_close(tuple(actual), expected)
        second = torch.func.grad(lambda value: sum(item.sum() for item in torch.func.grad(loss)(value)))(argument)
        expected_second = torch.func.grad(lambda value: sum(item.sum() for item in torch.func.grad(reference)(value)))(
            parts
        )
        assert_close(tuple(second), expected_second)

    def test_view_branches(self, device):
        left = NT([torch.zeros(n, 3, dtype=torch.double, device=device) for n in (2, 4)], ragged_dims=(0,))
        right = NT([torch.zeros(n, 3, dtype=torch.double, device=device) for n in (3, 4)], ragged_dims=(0,))
        a, b = torch.randn_like(left.values()), torch.randn_like(right.values())

        def transform(a, b):
            current = left.packed_like(a)
            current.element_sizes(dtype=torch.int32)
            return (current[:, :1] + right.packed_like(b)[:, :1]).sin()

        def reference(a, b):
            return (a[[0, 2]] + b[[0, 3]]).sin()

        output, pullback = torch.func.vjp(transform, a, b)
        gradients = pullback(torch.ones_like(output))
        expected = torch.func.grad(lambda a, b: reference(a, b).sum(), argnums=(0, 1))(a, b)
        assert_close(gradients, expected)
        directions = (torch.randn_like(a), torch.randn_like(b))
        primal, tangent = torch.func.jvp(transform, (a, b), directions)
        expected_primal, expected_tangent = torch.func.jvp(reference, (a, b), directions)
        assert_close(primal.values(), expected_primal)
        assert_close(tangent.values(), expected_tangent)
        assert_close(output.sin().values(), expected_primal.sin())
        assert left.element_sizes(dtype=torch.int32).tolist() == [[2, 3], [4, 3]]

    def test_leading_expand_preserves_ragged_samples(self, device):
        source = NT([torch.randn(2, 1, device=device), torch.randn(3, 1, device=device)])
        output = source.expand(4, 2, 3, 5)
        assert not output.batch_first
        for actual, element in zip(output, source):
            assert_close(actual, element.unsqueeze(0).expand(4, -1, 5))
        with pytest.raises(RuntimeError, match="non-negative"):
            source.expand(-1, 2, 3, 5)


class TestAutogradViewLifetime:

    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    @pytest.mark.parametrize("dim", [0, -1])
    def test_inference_views(self, device, backend, dim):
        torch.compiler.reset()
        parts = [torch.randn(n, 3, device=device) for n in (2, 4)]
        run = torch.compile(lambda value: value.unsqueeze(dim), backend=backend, fullgraph=True)
        with torch.inference_mode():
            source = NT(parts, ragged_dims=(0,))
            output = run(source)
            output.add_(1)
            assert output.is_inference()
            torch.testing.assert_close(source.unbind(), [part + 1 for part in parts])

    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    @pytest.mark.parametrize("layout", ["ragged", "batch-second", "existing-view", "chain", "alias", "detach", "clone"])
    def test_compiled_views(self, device, backend, layout):
        torch.compiler.reset()
        parts = [torch.randn(n, 3, device=device, dtype=torch.float64, requires_grad=True) for n in (2, 4)]
        references = [part.detach().clone().requires_grad_() for part in parts]
        source = NT(parts, ragged_dims=(0,))
        if layout == "batch-second":
            source = source.transpose(0, 1)
        elif layout == "existing-view":
            source = source.unsqueeze(-1)

        dim = -1 if layout == "batch-second" else 0

        def transform(value):
            result = value.unsqueeze(dim)
            if layout == "chain":
                return result.unsqueeze(-1)
            if layout == "alias":
                return torch.ops.aten.alias.default(result)
            if layout == "detach":
                return result.detach()
            return result.clone() if layout == "clone" else result

        expected = [part * 1 for part in references]
        expected_values = [part.unsqueeze(-1) for part in expected] if layout == "existing-view" else expected
        expected_outputs = []
        for part in expected_values:
            result = part.unsqueeze(dim)
            if layout == "chain":
                result = result.unsqueeze(-1)
            elif layout == "detach":
                result = result.detach()
            elif layout == "clone":
                result = result.clone()
            expected_outputs.append(result)
            result.mul_(2)

        output = torch.compile(transform, backend=backend, fullgraph=True)(source)
        output.mul_(2)
        assert not output.batch_first
        torch.testing.assert_close(output.unbind(1), expected_outputs)
        torch.testing.assert_close(source.unbind(0 if source.batch_first else 1), expected_values)
        torch.testing.assert_close(
            torch.autograd.grad(source.square().sum(), parts),
            torch.autograd.grad(sum(part.square().sum() for part in expected), references),
        )

    @pytest.mark.parametrize("length", [0, 1])
    def test_compiled_inplace_edges(self, device, length):
        torch.compiler.reset()
        parts = [torch.randn(length, 3, device=device, dtype=torch.float64, requires_grad=True) for _ in range(2)]
        others = [torch.randn_like(part, requires_grad=True) for part in parts]
        source = NT(parts, ragged_dims=(0,)).sin()
        other = NT(others, ragged_dims=(0,)).cos()
        operation = torch.compile(lambda left, right: left.add_(right), backend="aot_eager", fullgraph=True)
        output = operation(source, other)
        assert output is source
        torch.testing.assert_close(output.unbind(), [left.sin() + right.cos() for left, right in zip(parts, others)])
        torch.testing.assert_close(
            torch.autograd.grad(output.values().sum(), (*parts, *others)),
            tuple(part.cos() for part in parts) + tuple(-part.sin() for part in others),
        )

    def test_saved_hooks(self, device):
        parts = [
            torch.randn(a, b, 2, device=device, dtype=torch.float64, requires_grad=True) for a, b in ((2, 3), (4, 2))
        ]
        source = NT(parts, ragged_dims=(1, 0), batch_first=False)
        with torch.autograd.graph.saved_tensors_hooks(lambda value: value.detach().clone(), lambda value: value):
            output = source.sin()
            gradients = torch.autograd.grad(output.values().square().sum(), parts, create_graph=True)
        torch.testing.assert_close(gradients, tuple(2 * value.sin() * value.cos() for value in parts))
        second = torch.autograd.grad(sum(value.sum() for value in gradients), parts)
        torch.testing.assert_close(second, tuple(2 * (2 * value).cos() for value in parts))

    def test_eager_intermediate_wrapper_receives_gradients_and_hooks(self):
        leaf = torch.arange(1.0, 6.0, requires_grad=True)
        output = NT([leaf[:2], leaf[2:]]).square()
        output.retain_grad()
        observed = []
        output.register_hook(lambda gradient: observed.append(gradient))
        loss = output.values().sum()
        gradient = torch.autograd.grad(loss, output, retain_graph=True)[0]
        assert_close(gradient.values(), torch.ones(5))
        observed.clear()
        output.grad = None
        loss.backward()
        assert len(observed) == 1
        assert_close(observed[0].values(), torch.ones(5))
        assert_close(output.grad.values(), torch.ones(5))
        assert_close(leaf.grad, 2 * leaf)

    def test_backward_rejects_different_nested_tangent_boundaries(self):
        leaf = torch.arange(1.0, 6.0, requires_grad=True)
        output = NT([leaf[:2], leaf[2:]]).square()
        tangent = NT([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
        with pytest.raises(RuntimeError, match="tangent topology"):
            torch.autograd.grad(output, leaf, tangent)

    def test_inference_mutation_refreshes_materialized_values(self):
        with torch.inference_mode():
            source = NT([torch.tensor([1.0, 2.0]), torch.tensor([3.0])])
            assert_close(source.tensor, torch.tensor([[1.0, 2.0], [3.0, 0.0]]))
            source.values().add_(10)
            assert_close(source.tensor, torch.tensor([[11.0, 12.0], [13.0, 0.0]]))

    @pytest.mark.parametrize("backend", ["eager", "aot_eager"])
    def test_explicit_leaf_owns_gradients(self, device, backend):
        def loss(value):
            return torch.pow(value, 3).sum()

        run = loss if backend == "eager" else torch.compile(loss, backend=backend, fullgraph=True, dynamic=True)
        source = NT(
            [
                torch.randn(2, 3, 2, device=device, dtype=torch.float64),
                torch.randn(4, 2, 2, device=device, dtype=torch.float64),
            ],
            ragged_dims=(0, 1),
        ).requires_grad_()
        with torch.no_grad():
            values = source.values().clone()
        assert source.is_leaf
        for step in (1, 2):
            run(source).backward()
            assert_close(source.grad.element_sizes(), source.element_sizes())
            assert_close(source.grad.values(), step * 3 * values.square())
        source.grad = None
        gradient = torch.autograd.grad(run(source), source)[0]
        assert source.grad is None
        assert_close(gradient.values(), 3 * values.square())
        if backend == "eager":
            gradient = torch.autograd.grad(loss(source), source, create_graph=True)[0]
            second = torch.autograd.grad(gradient, source, grad_outputs=torch.ones_like(gradient))[0]
            assert_close(second.values(), 6 * values)
            try:
                loss(source).backward(create_graph=True)
                assert_close(source.grad.values(), 3 * values.square())
                second = torch.autograd.grad(source.grad, source, grad_outputs=torch.ones_like(source.grad))[0]
                assert_close(second.values(), 6 * values)
            finally:
                source.grad = None

    def test_explicit_leaf_materialized_views_release_owner(self, device):
        source = NT([torch.randn(2, 3, device=device), torch.randn(4, 3, device=device)]).requires_grad_()
        with torch.no_grad():
            values = source.values().clone()
        padded = source.tensor
        elements = source.unbind(0)
        loss = padded.square().sum() + sum(element.square().sum() for element in elements)
        loss.backward()
        assert_close(source.grad.values(), 4 * values)
        reference = weakref.ref(source)
        del source, padded, elements, loss
        gc.collect()
        assert reference() is None

    @staticmethod
    def view(output, name):
        if name == "concat":
            return output.values()
        if name == "unbind":
            return output.unbind(0)
        return output.tensor

    @pytest.mark.parametrize("name", ["concat", "unbind", "tensor"])
    @pytest.mark.parametrize("after_backward", [False, True])
    def test_unconsumed_compiled_views_release_wrapper_and_inputs(self, name, after_backward):
        torch.compiler.reset()
        compiled = torch.compile(torch.sin, backend="aot_eager", fullgraph=True, dynamic=True)
        try:
            for lengths in ((2, 4), (3, 5), (4, 2)):
                parts = [torch.randn(length, 3, requires_grad=True) for length in lengths]
                source = NT(parts, ragged_dims=(0,))
                output = compiled(source)
                if after_backward:
                    output.values().square().sum().backward()
                observed = self.view(output, name)
                output_ref = weakref.ref(output)
                input_refs = [weakref.ref(part) for part in parts]
                del observed, output, source, parts
                gc.collect()
                assert output_ref() is None
                assert all(reference() is None for reference in input_refs)
        finally:
            torch.compiler.reset()

    @pytest.mark.parametrize("name", ["unbind", "tensor"])
    @pytest.mark.parametrize("backend", ["eager", "aot_eager"])
    def test_no_grad_cached_views_preserve_later_weighted_gradients(self, name, backend):
        torch.compiler.reset()
        function = torch.sin if backend == "eager" else torch.compile(torch.sin, backend=backend, fullgraph=True)
        parts = [torch.randn(length, 3, dtype=torch.float64, requires_grad=True) for length in (2, 4)]
        output = function(NT(parts, ragged_dims=(0,)))
        try:
            with torch.no_grad():
                cached = self.view(output, name)
                repeated = self.view(output, name)
                assert repeated is cached
            weights = [torch.arange(1, part.numel() + 1, dtype=part.dtype).reshape(part.shape) for part in parts]
            tracked = self.view(output, name)
            if name == "unbind":
                loss = sum((value * weight).sum() for value, weight in zip(tracked, weights))
            else:
                dense_weights = torch.nn.utils.rnn.pad_sequence(weights, batch_first=True)
                loss = (tracked * dense_weights).sum()
            gradients = torch.autograd.grad(loss, parts)
            for actual, part, weight in zip(gradients, parts, weights):
                assert_close(actual, part.cos() * weight, atol=1e-12, rtol=1e-12)
        finally:
            torch.compiler.reset()

    @pytest.mark.parametrize("name", ["unbind", "tensor"])
    def test_grad_free_eager_views_still_reuse_their_cache(self, name):
        parts = [torch.randn(length, 3) for length in (2, 4)]
        output = torch.sin(NT(parts, ragged_dims=(0,)))
        first = self.view(output, name)
        second = self.view(output, name)
        assert first is second
        if name == "unbind":
            assert_close(first, tuple(part.sin() for part in parts))
        else:
            assert_close(first, torch.nn.utils.rnn.pad_sequence([part.sin() for part in parts], batch_first=True))

    def test_independent_compiled_view_consumers_support_repeated_backward(self):
        torch.compiler.reset()
        compiled = torch.compile(torch.sin, backend="aot_eager", fullgraph=True)
        parts = [torch.randn(length, 3, dtype=torch.float64, requires_grad=True) for length in (2, 4)]
        output = compiled(NT(parts, ragged_dims=(0,)))
        first = output.values()
        second = output.values()
        assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
        assert first.stride() == second.stride()
        elements = output.unbind(0)
        padded = output.tensor
        loss = first.sum() + 2 * second.sum() + 3 * sum(part.sum() for part in elements) + 4 * padded.sum()
        try:
            for retain_graph in (True, False):
                gradients = torch.autograd.grad(loss, parts, retain_graph=retain_graph)
                for actual, source in zip(gradients, parts):
                    assert_close(actual, 10 * source.cos(), atol=1e-12, rtol=1e-12)
        finally:
            torch.compiler.reset()

    def test_no_grad_concat_alias_updates_materialized_values(self):
        torch.compiler.reset()
        compiled = torch.compile(torch.sin, backend="aot_eager", fullgraph=True)
        parts = [torch.randn(length, 3, requires_grad=True) for length in (2, 4)]
        output = compiled(NT(parts, ragged_dims=(0,)))
        try:
            with torch.no_grad():
                before = output.tensor
                alias = output.values()
                alias.add_(1)
                after = output.tensor
                expected = torch.nn.utils.rnn.pad_sequence([part.sin() + 1 for part in parts], batch_first=True)
                assert_close(after, expected)
                assert before is not after
        finally:
            torch.compiler.reset()

    def test_concat_alias_mutation_keeps_saved_tensor_version_checks(self):
        torch.compiler.reset()
        compiled = torch.compile(torch.sin, backend="aot_eager", fullgraph=True)
        parts = [torch.randn(length, 3, requires_grad=True) for length in (2, 4)]
        output = compiled(NT(parts, ragged_dims=(0,)))
        loss = output.values().square().sum()
        try:
            with torch.no_grad():
                output.values().add_(1)
            with pytest.raises(RuntimeError, match="modified.*inplace|view.*modified|version"):
                loss.backward()
        finally:
            torch.compiler.reset()

    @pytest.mark.parametrize("backend", ["eager", "aot_eager"])
    def test_projection_after_inplace_rebase_keeps_current_gradient(self, backend):
        torch.compiler.reset()

        def scale(value):
            return value * 1.5

        function = scale if backend == "eager" else torch.compile(scale, backend=backend, fullgraph=True)
        parts = [torch.randn(length, 3, dtype=torch.float64, requires_grad=True) for length in (2, 4)]
        output = function(NT(parts, ragged_dims=(0,)))
        previous = output.values()
        output.mul_(2)
        try:
            current = output.values()
            assert_close(current, 3 * torch.cat(parts), atol=1e-12, rtol=1e-12)
            gradients = torch.autograd.grad(current.sum(), parts)
            for gradient, part in zip(gradients, parts):
                assert_close(gradient, torch.full_like(part, 3), atol=1e-12, rtol=1e-12)
            assert previous.untyped_storage().data_ptr() == current.untyped_storage().data_ptr()
        finally:
            torch.compiler.reset()

    @pytest.mark.parametrize("name", ["concat", "unbind", "tensor"])
    def test_compiled_chain_metadata_inspection_does_not_retain_real_wrappers(self, name):
        torch.compiler.reset()
        producer = torch.compile(lambda tensor: tensor * 2, backend="aot_eager", fullgraph=True)
        consumer = torch.compile(torch.sin, backend="aot_eager", fullgraph=True)
        try:
            parts = [torch.randn(length, 3, requires_grad=True) for length in (2, 4)]
            source = NT(parts, ragged_dims=(0,))
            intermediate = producer(source)
            output = consumer(intermediate)
            output.values().sum().backward()
            for part in parts:
                assert_close(part.grad, 2 * torch.cos(2 * part))
            del part
            observed = self.view(output, name)
            references = [weakref.ref(value) for value in (intermediate, output, *parts)]
            del observed, intermediate, output, source, parts
            gc.collect()
            assert all(reference() is None for reference in references)
        finally:
            torch.compiler.reset()

    @pytest.mark.parametrize("factory", ["constructor", "concatenated"])
    def test_factory_wrapper_survives_repeated_compiled_backward(self, device, factory):
        # AOTAutograd attaches its backward to the wrapper edge, so a wrapper built once
        # from a leaf must stay traversable across independent training steps.
        torch.compiler.reset()
        compiled = torch.compile(lambda value, scale: value * scale, backend="aot_eager", fullgraph=True, dynamic=True)
        try:
            leaf = torch.randn(6, 3, device=device, dtype=torch.float64, requires_grad=True)
            reference = leaf.detach().clone().requires_grad_()
            if factory == "constructor":
                nested = NT(list(leaf.split((2, 4))), ragged_dims=(0,))
            else:
                nested = NT.from_concatenated(leaf, (torch.Size((2, 3)), torch.Size((4, 3))), ragged_dims=(0,))
            scale = torch.tensor(1.5, device=device, dtype=torch.float64, requires_grad=True)
            reference_scale = scale.detach().clone().requires_grad_()
            for _ in range(3):
                output = compiled(nested, scale)
                projection_gradient = torch.autograd.grad(nested.values().sum(), leaf)[0]
                assert_close(projection_gradient, torch.ones_like(leaf), atol=1e-12, rtol=1e-12)
                output.values().square().sum().backward()
                (reference * reference_scale).square().sum().backward()
                assert_close(leaf.grad, reference.grad, atol=1e-12, rtol=1e-12)
                assert_close(scale.grad, reference_scale.grad, atol=1e-12, rtol=1e-12)
                with torch.no_grad():
                    scale -= 0.1 * scale.grad
                    reference_scale -= 0.1 * reference_scale.grad
                leaf.grad = reference.grad = None
                scale.grad = reference_scale.grad = None
        finally:
            torch.compiler.reset()


class TestCheckpointing:

    @staticmethod
    def operation(value, weight, bias, name):
        if name == "sin":
            return value.packed_like(value.values().sin()) if isinstance(value, NT) else value.sin()
        if name == "linear":
            return torch.nn.functional.linear(value, weight, bias)
        if name == "relu":
            return torch.nn.functional.relu(value)
        if name == "sin_mean":
            return torch.sin(value).mean(dim=-1)
        return value.mean(dim=-1)

    @pytest.mark.parametrize("operation", ["sin", "linear", "relu", "sin_mean", "mean"])
    @pytest.mark.parametrize("levels", [0, 1, 2, 3])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_nested_nonreentrant_checkpoint_preserves_values_and_gradients(self, device, operation, levels, dtype):
        from torch.utils.checkpoint import checkpoint

        leaves = [torch.randn(n, n, 4, device=device, dtype=dtype, requires_grad=True) for n in (2, 3)]
        weight = torch.randn(4, 4, device=device, dtype=dtype, requires_grad=True)
        bias = torch.randn(4, device=device, dtype=dtype, requires_grad=True)
        references = [leaf.detach().clone().requires_grad_() for leaf in leaves]
        reference_weight = weight.detach().clone().requires_grad_()
        reference_bias = bias.detach().clone().requires_grad_()
        nested = NT(leaves, ragged_dims=(0, 1))

        def inner(value):
            return self.operation(value, weight, bias, operation)

        def outer(value):
            updated = value.packed_like(value.values() * 1.5)
            return checkpoint(inner, updated, use_reentrant=False) if levels else inner(updated)

        run = outer
        for _ in range(max(0, levels - 1)):
            previous = run

            def run(value, function=previous):
                return checkpoint(function, value, use_reentrant=False)

        output = run(nested)
        expected = self.operation(
            torch.cat([leaf.reshape(-1, 4) for leaf in references]) * 1.5,
            reference_weight,
            reference_bias,
            operation,
        )
        output.values().square().sum().backward()
        expected.square().sum().backward()
        tolerance = 1e-12 if dtype == torch.float64 else 1e-5
        assert_close(output.values(), expected, atol=tolerance, rtol=tolerance)
        expected_shapes = [[n, n] if operation in ("sin_mean", "mean") else [n, n, 4] for n in (2, 3)]
        assert output.element_sizes().tolist() == expected_shapes
        assert output.packed_offsets(device="cpu").tolist() == [0, 4, 13]
        pairs = list(zip(leaves, references))
        if operation == "linear":
            pairs.extend(((weight, reference_weight), (bias, reference_bias)))
        for actual, reference in pairs:
            assert actual.grad is not None and reference.grad is not None
            assert_close(actual.grad, reference.grad, atol=tolerance, rtol=tolerance)

    def test_nested_checkpoint_releases_recomputable_packed_activation(self, device):
        from torch.multiprocessing.reductions import StorageWeakRef
        from torch.utils.checkpoint import checkpoint

        leaves = [torch.randn(n, n, 4, device=device, requires_grad=True) for n in (2, 3)]
        references = [leaf.detach().clone().requires_grad_() for leaf in leaves]
        nested = NT(leaves, ragged_dims=(0, 1))
        intermediate = {}

        def inner(value):
            return value.packed_like(value.values().sin())

        def outer(value):
            scaled = value.values() * 1.5
            intermediate["tensor"] = weakref.ref(scaled)
            intermediate["storage"] = StorageWeakRef(scaled.untyped_storage())
            updated = value.packed_like(scaled)
            return checkpoint(inner, updated, use_reentrant=False)

        output = checkpoint(outer, nested, use_reentrant=False)
        # Inspect before any backward/recomputation while input and final output
        # stay live. Weak storage observation must not own the intermediate.
        gc.collect()
        assert intermediate["tensor"]() is None
        assert intermediate["storage"].expired()
        output.values().square().sum().backward()
        expected = [(leaf * 1.5).sin() for leaf in references]
        sum(part.square().sum() for part in expected).backward()
        for actual, reference in zip(leaves, references):
            assert_close(actual.grad, reference.grad)

    def test_existing_projection_preserves_later_nested_checkpoint(self, device):
        from torch.utils.checkpoint import checkpoint

        leaves = [torch.randn(n, n, 4, device=device, dtype=torch.float64, requires_grad=True) for n in (2, 3)]
        references = [leaf.detach().clone().requires_grad_() for leaf in leaves]
        source = NT(leaves, ragged_dims=(0, 1))
        prior_projection = source.values()

        def inner(value):
            return value.packed_like(value.values().sin())

        def outer(value):
            updated = value.packed_like(value.values() * 1.5)
            return checkpoint(inner, updated, use_reentrant=False)

        output = checkpoint(outer, source, use_reentrant=False)
        output.values().square().sum().backward()
        expected = [(part * 1.5).sin() for part in references]
        sum(part.square().sum() for part in expected).backward()
        assert_close(prior_projection, torch.cat([part.reshape(-1, 4) for part in leaves]), atol=0, rtol=0)
        for actual, reference in zip(leaves, references):
            assert_close(actual.grad, reference.grad, atol=1e-12, rtol=1e-12)

    @pytest.mark.parametrize(
        "factory", ["lengths", "square_lengths", "rectangular_lengths", "static_tail", "concatenated"]
    )
    def test_reconstruction_factories_preserve_nested_checkpoint_edges(self, device, factory):
        from torch.multiprocessing.reductions import StorageWeakRef
        from torch.utils.checkpoint import checkpoint

        lengths = torch.tensor([2, 3])
        columns = torch.tensor([3, 2]) if factory == "rectangular_lengths" else lengths
        square = factory in ("square_lengths", "rectangular_lengths", "concatenated")
        shapes = [(n, m, 4) if square else (n, 4) for n, m in zip(lengths.tolist(), columns.tolist())]
        leaves = [torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
        references = [leaf.detach().clone().requires_grad_() for leaf in leaves]
        ragged_dims = (0, 1) if square else (0,)
        source = NT(leaves, ragged_dims=ragged_dims)
        intermediate = {}

        def inner(value):
            return value.packed_like(value.values().sin())

        def outer(value):
            scaled = value.values() * 1.5
            if factory == "lengths":
                rebuilt = value.packed_with_lengths(scaled, lengths)
            elif factory == "square_lengths":
                rebuilt = value.packed_with_square_lengths(scaled, lengths)
            elif factory == "rectangular_lengths":
                rebuilt = value.packed_with_rectangular_lengths(scaled, lengths, columns)
            elif factory == "static_tail":
                rebuilt = value.packed_with_static_tail(scaled)
            else:
                rebuilt = NT.from_concatenated(
                    scaled, tuple(torch.Size(shape) for shape in shapes), ragged_dims=ragged_dims
                )
            with torch.no_grad():
                intermediate["storage"] = StorageWeakRef(rebuilt.values().untyped_storage())
            return checkpoint(inner, rebuilt, use_reentrant=False)

        output = checkpoint(outer, source, use_reentrant=False)
        gc.collect()
        assert intermediate["storage"].expired()
        output.values().square().sum().backward()
        expected = (torch.cat([leaf.reshape(-1, 4) for leaf in references]) * 1.5).sin()
        expected.square().sum().backward()
        assert_close(output.values(), expected, atol=1e-12, rtol=1e-12)
        assert output.element_sizes().tolist() == [list(shape) for shape in shapes]
        for actual, reference in zip(leaves, references):
            assert actual.grad is not None
            assert_close(actual.grad, reference.grad, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Construction & Validation
# ---------------------------------------------------------------------------


class TestArithmetic:

    @pytest.mark.parametrize("name", ["flip_static", "flip_ragged", "squeeze_singleton", "squeeze_noop"])
    @pytest.mark.parametrize("interface", ["function", "method"])
    @pytest.mark.parametrize("backend", ["eager", "aot_eager"])
    def test_inherited_transform_gradients(self, device, name, interface, backend):
        torch.compiler.reset()
        namespace = torch if interface == "function" else torch.Tensor
        flip_dimension = 1 if name == "flip_ragged" else -1

        def transform(value):
            if name.startswith("flip"):
                return namespace.flip(value, (flip_dimension,))
            if name == "squeeze_singleton":
                return namespace.squeeze(value, -2)
            return namespace.squeeze(value, -1)

        run = (
            transform if backend == "eager" else torch.compile(transform, backend=backend, fullgraph=True, dynamic=True)
        )
        for lengths in ((2, 4), (3, 5)):
            parts = [
                torch.arange(n * 3, device=device, dtype=torch.float64).reshape(n, 1, 3).requires_grad_()
                for n in lengths
            ]
            references = [part.detach().clone().requires_grad_() for part in parts]
            nested = NT(parts, ragged_dims=(0,))
            assert nested.ndim == nested.dim() == 4
            if name == "flip_ragged":
                expected = torch.cat([reference.flip((0,)) for reference in references])
            else:
                expected = torch.cat([transform(reference) for reference in references])
            result = run(nested)
            actual = result.values()
            assert result.ndim == result.dim() == expected.ndim + 1
            if name.startswith("squeeze"):
                assert actual.untyped_storage().data_ptr() == nested.values().untyped_storage().data_ptr()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            weights = torch.arange(1, 2 * actual.numel() + 1, device=device, dtype=torch.float64)
            weights = weights.reshape(*actual.shape, 2)[..., 0]
            gradients = torch.autograd.grad(actual, parts, weights)
            expected_gradients = torch.autograd.grad(expected, references, weights)
            for observed, wanted in zip(gradients, expected_gradients):
                torch.testing.assert_close(observed, wanted, rtol=0, atol=0)

    @pytest.mark.parametrize("backend", ["eager", "aot_eager"])
    @pytest.mark.parametrize("center", [False, True])
    @pytest.mark.parametrize("packed_output", [False, True])
    def test_fullgraph_expand_gradients(self, device, backend, center, packed_output):
        torch.compiler.reset()

        def expand(tensor):
            if center:
                tensor = tensor - tensor.mean(-2, keepdim=True)
            output = tensor.unsqueeze(-3).expand(-1, 3, -1, -1)
            return output.values() if packed_output else output

        compiled = torch.compile(expand, backend=backend, fullgraph=True, dynamic=True)
        for lengths in ((2, 5), (4, 3)):
            parts = [
                torch.randn(length, 3, device=device, dtype=torch.float64, requires_grad=True) for length in lengths
            ]
            nested = NT(parts, ragged_dims=(0,))

            output = compiled(nested)
            values = output if packed_output else output.values()
            gradients = torch.autograd.grad(values.square().sum(), parts)

            expected = [part - part.mean(0) if center else part for part in parts]
            assert_close(values, torch.cat(expected).unsqueeze(1).expand(-1, 3, -1))
            if not center:
                assert values.untyped_storage().data_ptr() == nested.values().untyped_storage().data_ptr()
                assert values.stride(1) == 0
            for actual, wanted in zip(gradients, expected):
                assert_close(actual, 6 * wanted)

    @pytest.mark.parametrize("strided", [False, True])
    def test_compiled_weighted_vjp(self, device, strided):
        torch.compiler.reset()
        compiled = torch.compile(
            lambda tensor: (tensor * 2).unsqueeze(-3).expand(-1, 3, -1, -1),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        for lengths in ((2, 4), (3, 5)):
            parts = [
                torch.randn(length, 3, device=device, dtype=torch.float64, requires_grad=True) for length in lengths
            ]
            nested = NT(parts, ragged_dims=(0,))
            weights = (
                torch.arange(9 * sum(lengths), device=device, dtype=torch.float64)
                .reshape(3, sum(lengths), 3)
                .transpose(0, 1)
                .div(13)
            )
            if not strided:
                weights = weights.contiguous()
            assert weights.is_contiguous() is not strided

            output = compiled(nested)
            values = output.values()
            assert values.stride(1) == 0
            for scale in (1, -2):
                tangent = weights * scale
                assert tangent.is_contiguous() is not strided
                gradients = torch.autograd.grad(values, parts, tangent, retain_graph=scale == 1)
                expected = (2 * scale * weights.sum(1)).split(lengths)
                for actual, wanted in zip(gradients, expected):
                    torch.testing.assert_close(actual, wanted, rtol=1e-12, atol=1e-12)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compiled_wrapper_outputs_chain_and_preserve_autograd(self):
        parts = (torch.randn(2, 3, requires_grad=True), torch.randn(4, 3, requires_grad=True))
        nested = NT(parts, ragged_dims=(0,))
        producer = torch.compile(lambda tensor: tensor * 2, backend="aot_eager", fullgraph=True)
        consumer = torch.compile(torch.sin, backend="aot_eager", fullgraph=True)

        output = consumer(producer(nested))
        gradients = torch.autograd.grad(output.values().sum(), parts)

        assert_close(output.values(), torch.sin(nested.values() * 2))
        expected = 2 * torch.cos(nested.values() * 2)
        for actual, wanted in zip(gradients, expected.split([2, 4])):
            assert_close(actual, wanted)

    def test_concat_remains_read_only_for_users(self):
        nested = NT([torch.empty(2, 3), torch.empty(4, 3)], ragged_dims=(0,))

        with pytest.raises(AttributeError, match="read-only"):
            nested.concat = torch.empty_like(nested.values())

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

    @pytest.mark.parametrize("fullgraph", [False, True])
    @pytest.mark.parametrize(
        "ragged_dims,batch_first,patterns",
        [
            ((0, 1), True, (((1, 3), (1, 5)), ((2, 3), (4, 1)), ((0, 2), (1, 0)))),
            ((1, 0), False, (((2, 3, 2), (3, 4, 2)), ((1, 0, 2), (0, 1, 2)))),
            ((1,), True, (((2, 3, 2), (2, 5, 2)), ((2, 0, 2), (2, 1, 2)))),
        ],
        ids=["multi", "permuted", "trailing"],
    )
    def test_dynamic_rebuild(self, device, fullgraph, ragged_dims, batch_first, patterns):
        def rebuild(source):
            return NT(
                [part.sin() for part in source],
                ragged_dims=ragged_dims,
                batch_first=batch_first,
                padding_value=-2,
            )

        torch.compiler.reset()
        compiled = torch.compile(rebuild, backend="aot_eager", fullgraph=fullgraph, dynamic=True)
        try:
            for shapes in patterns:
                parts = [torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
                source = NT(parts, ragged_dims=ragged_dims)
                expected = tuple(part.sin() for part in parts)
                actual = compiled(source)
                torch.testing.assert_close(actual.unbind(0 if batch_first else 1), expected)
                assert actual.ragged_dims == ragged_dims
                assert actual.batch_first is batch_first
                assert actual.padding_value == -2
                assert actual.element_sizes().tolist() == [list(shape) for shape in shapes]
                gradients = torch.autograd.grad(actual.values().square().sum(), parts)
                torch.testing.assert_close(gradients, tuple(2 * part.sin() * part.cos() for part in parts))
                with torch.inference_mode():
                    actual = compiled(source)
                    torch.testing.assert_close(actual.unbind(0 if batch_first else 1), expected)
        finally:
            torch.compiler.reset()

    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    @pytest.mark.parametrize("fullgraph", [False, True])
    @pytest.mark.parametrize("nested", [False, True], ids=["dense", "nested"])
    def test_inferred_rebuild(self, device, backend, fullgraph, nested):
        def rebuild(source):
            return NT([part.sin() for part in source])

        torch.compiler.reset()
        compiled = torch.compile(rebuild, backend=backend, fullgraph=fullgraph, dynamic=True)
        patterns = (((1, 3), (1, 5)), ((2, 3), (4, 1)), ((2, 3), (2, 3)), ((0, 2), (1, 0)))
        try:
            for shapes in patterns:
                parts = [torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
                source = NT(parts, ragged_dims=(0, 1)) if nested else tuple(parts)
                if nested and fullgraph:
                    with pytest.raises(torch._dynamo.exc.Unsupported, match="ragged_dims"):
                        compiled(source)
                    return
                expected = tuple(part.sin() for part in parts)
                actual = compiled(source)
                torch.testing.assert_close(actual.unbind(0), expected)
                inferred = tuple(dim for dim in range(2) if shapes[0][dim] != shapes[1][dim]) or (0,)
                assert actual.ragged_dims == inferred
                assert actual.element_sizes().tolist() == [list(shape) for shape in shapes]
                gradients = torch.autograd.grad(actual.values().square().sum(), parts)
                torch.testing.assert_close(gradients, tuple(2 * part.sin() * part.cos() for part in parts))
                with torch.inference_mode():
                    torch.testing.assert_close(compiled(source).unbind(0), expected)
        finally:
            torch.compiler.reset()

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
        assert nested.values().device.type == "meta"

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

        assert nested.values().is_pinned()


# Packed Reconstruction
# ---------------------------------------------------------------------------


class TestDeclaredRaggedDims:

    def test_declared_ragged_dims_preserve_values_and_layout(self):
        elements = [torch.randn(2, 2, 3), torch.randn(4, 4, 3)]

        nested = NestedTensor(elements, ragged_dims=(0, 1))

        assert nested.ragged_dims == (0, 1)
        assert nested.packed_dim_order == (0, 1, 2)
        assert nested.values().shape == (20, 3)
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
            output = fake_reference.packed_like(torch.empty_like(fake_reference.values()))

        assert fake_tensor_mod.is_fake(output.values())
        assert output.ragged_dims == (0, 1)
        assert output.shape == reference.shape

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_declared_ragged_dims_survive_fullgraph_with_backward(self):
        reference = NestedTensor([torch.empty(2, 2, 3), torch.empty(4, 4, 3)], ragged_dims=(0, 1))
        values = torch.randn_like(reference.values(), requires_grad=True)
        compiled = torch.compile(
            lambda ref, packed: ref.packed_like(packed.square()),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        output = compiled(reference, values)
        output.values().sum().backward()

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

    def test_inferred_layout_is_declared_and_reuses_the_compiled_graph(self, device):
        # A layout inferred from the first batch is declared from then on, so it is tensor-backed
        # and Dynamo sees the same structural schema for every later length vector.
        torch.compiler.reset()
        captures = 0

        def backend(graph, inputs):
            nonlocal captures
            captures += 1
            from torch._dynamo.backends.registry import lookup_backend

            return lookup_backend("aot_eager")(graph, inputs)

        compiled = torch.compile(lambda tensor: tensor * 2, backend=backend, fullgraph=True, dynamic=True)
        try:
            for lengths in ((2, 4), (3, 5), (4, 2)):
                nested = NestedTensor([torch.randn(length, 3, device=device) for length in lengths])
                assert nested.ragged_dims == (0,)
                output = compiled(nested)
                assert_close(output.values(), 2 * nested.values())
            assert captures == 1
        finally:
            torch.compiler.reset()

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

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("ragged_dims,shapes", [((0,), ((2, 3), (4, 3))), ((0, 1), ((2, 3, 4), (4, 2, 4)))])
    def test_unflatten_preserves_packed_child(self, batch_first, ragged_dims, shapes):
        nested = NT([torch.randn(shape) for shape in shapes], ragged_dims=ragged_dims, batch_first=batch_first)
        names, metadata = nested.__tensor_flatten__()
        children = {name: getattr(nested, name) for name in names}
        replacement = torch.full_like(nested.values(), 7)
        children["concat"] = replacement
        strides = tuple(stride * 2 for stride in nested.stride())

        rebuilt = NT.__tensor_unflatten__(children, metadata, nested.shape, strides)

        assert rebuilt.values() is replacement
        assert rebuilt.shape == nested.shape
        assert rebuilt.stride() == strides
        assert rebuilt.batch_first is batch_first
        assert rebuilt.ragged_dims == nested.ragged_dims
        assert rebuilt.packed_dim_order == nested.packed_dim_order
        assert_close(rebuilt.packed_offsets(), nested.packed_offsets())
        assert_close(rebuilt.element_sizes(), nested.element_sizes())
        torch.testing.assert_close(
            rebuilt.unbind(0 if batch_first else 1), [torch.full(shape, 7.0) for shape in shapes]
        )

    def test_packed_like_preserves_public_layout_and_runtime_config(self):
        reference = NestedTensor(
            [torch.randn(2, 3), torch.randn(4, 3)],
            batch_first=False,
            padding_value=-1.5,
            mask_value=True,
        )
        packed_values = torch.arange(reference.values().numel(), dtype=torch.float32).reshape(reference.values().shape)

        output = reference.packed_like(packed_values)

        assert output.values() is packed_values
        assert output.shape == reference.shape
        assert output.ragged_dims == reference.ragged_dims
        assert output.batch_first is False
        assert output.padding_value == -1.5
        assert output.mask_value is True

    def test_packed_dim_order_tracks_logical_permutation(self):
        reference = NestedTensor([torch.randn(2, 2, 3), torch.randn(4, 4, 3)])

        permuted = reference.permute(0, 3, 1, 2)
        rebuilt = permuted.packed_like(torch.randn_like(permuted.values()))

        assert reference.packed_dim_order == (0, 1, 2)
        assert permuted.packed_dim_order == (1, 2, 0)
        assert rebuilt.packed_dim_order == permuted.packed_dim_order

    def test_packed_like_accepts_noncontiguous_values_without_copy(self):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        packed_values = torch.randn(3, reference.values().size(0)).transpose(0, 1)

        output = reference.packed_like(packed_values)

        assert output.values() is packed_values
        assert output.values().stride() == packed_values.stride()

    @pytest.mark.parametrize("pin_memory", [False, True])
    def test_packed_like_vmap_preserves_storage_and_pinning(self, pin_memory):
        if pin_memory and not torch.cuda.is_available():
            pytest.skip("pin_memory requires CUDA")
        template = NT([torch.zeros(2, 3), torch.zeros(4, 3)])
        values = torch.empty((2, 3, 6), pin_memory=pin_memory).normal_().transpose(1, 2)

        output = torch.vmap(lambda packed: template.packed_like(packed).values())(values)

        assert output.is_pinned() is pin_memory
        assert output.stride() == values.stride()
        assert output.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()
        assert_close(output, values)
        mapped = torch.vmap(lambda packed: template.packed_like(packed).sin().values())(values)
        assert mapped.is_pinned() is pin_memory
        assert_close(mapped, values.sin())

    def test_packed_like_follows_dtype_and_autograd_history(self):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        leaf = torch.randn(reference.values().shape, dtype=torch.float64, requires_grad=True)

        output = reference.packed_like(leaf.square())
        first_order = torch.autograd.grad(output.values().sum(), leaf, create_graph=True)[0]
        second_order = torch.autograd.grad(first_order.sum(), leaf)[0]

        assert output.dtype == torch.float64
        assert output.requires_grad
        assert_close(first_order, 2 * leaf)
        assert_close(second_order, torch.full_like(leaf, 2))

    def test_packed_like_follows_values_device(self):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        packed_values = torch.empty(reference.values().shape, device="meta")

        output = reference.packed_like(packed_values)

        assert output.device.type == "meta"
        assert output.values() is packed_values

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_packed_like_cuda_graph_reuses_warmed_structural_offsets(self, requires_grad):
        reference = NestedTensor([torch.empty(2, 2, 3), torch.empty(3, 3, 3)], ragged_dims=(0, 1))
        values = torch.randn(reference.values().shape, device="cuda", requires_grad=requires_grad)

        def run():
            wrapped = reference.packed_like(values)
            rows = wrapped.ragged_level_offsets(0, device=values.device)
            result = wrapped.values().square().sum() * rows[1]
            return (result, *torch.autograd.grad(result, values)) if requires_grad else (result,)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run()
        for factor in (0.5, 1.5):
            with torch.no_grad():
                values.mul_(factor)
            graph.replay()
            torch.cuda.synchronize()
            assert_close(captured[0], values.square().sum() * 2)
            if requires_grad:
                assert_close(captured[1], values * 4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    def test_packed_like_offset_conversion_observes_host_changes(self):
        reference = NestedTensor([torch.empty(2, 3), torch.empty(4, 3)], ragged_dims=(0,))
        values = torch.randn(reference.values().shape, device="cuda")
        first = reference.packed_like(values).packed_offsets(device=values.device)
        offsets = reference.packed_offsets()
        saved = offsets.clone()
        offsets[1] = 3
        changed = reference.packed_like(values).packed_offsets(device=values.device)
        assert_close(first.cpu(), saved)
        assert_close(changed.cpu(), offsets)
        offsets.copy_(saved)
        restored = reference.packed_like(values).packed_offsets(device=values.device)
        assert_close(restored.cpu(), saved)

    def test_packed_like_real_offset_conversions_survive_fake_mode(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        reference = NestedTensor([torch.empty(2, 2, 3), torch.empty(3, 3, 3)], ragged_dims=(0, 1))
        values = torch.randn_like(reference.values())
        # The first iteration starts cold; the second observes the eager conversions.
        for _ in range(2):
            with fake_tensor_mod.FakeTensorMode(allow_non_fake_inputs=True):
                output = reference.packed_like(values)
                assert fake_tensor_mod.is_fake(output.packed_offsets(dtype=torch.int32))
                assert fake_tensor_mod.is_fake(output.ragged_level_offsets(0, dtype=torch.int32))
            output = reference.packed_like(values)
            assert_close(output.packed_offsets(dtype=torch.int32), torch.tensor([0, 4, 13], dtype=torch.int32))
            assert_close(output.ragged_level_offsets(0, dtype=torch.int32), torch.tensor([0, 2, 5], dtype=torch.int32))

    def test_packed_like_inference_warmed_offsets_support_autograd(self):
        reference = NestedTensor([torch.empty(2, 3), torch.empty(4, 3)])
        values = torch.randn(reference.values().shape, requires_grad=True)
        with torch.inference_mode():
            reference.packed_offsets(dtype=torch.int32)
            reference.ragged_level_offsets(dtype=torch.int32)
        output = reference.packed_like(values)
        squared = output.values().square().sum()
        loss = squared * output.packed_offsets(dtype=torch.int32)[1]
        loss = loss + squared * output.ragged_level_offsets(dtype=torch.int32)[1]
        gradient = torch.autograd.grad(loss, values)[0]
        assert_close(loss, values.square().sum() * 4)
        assert_close(gradient, values * 8)

    def test_packed_like_rejects_invalid_values(self):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])

        with pytest.raises(ValueError, match="exactly the same shape"):
            reference.packed_like(reference.values().reshape(-1))
        with pytest.raises(TypeError, match="dense Tensor"):
            reference.packed_like(object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="dense Tensor"):
            reference.packed_like(reference)
        with pytest.raises(TypeError, match="torch.strided"):
            reference.packed_like(torch.zeros_like(reference.values()).to_sparse())

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_packed_offsets_and_element_sizes_report_public_layout(self, device, dtype):
        nested = NestedTensor(
            [torch.empty(2, 2, 4), torch.empty(3, 3, 4)],
            ragged_dims=(0, 1),
        )

        assert_close(nested.packed_offsets(), torch.tensor([0, 4, 13]))
        assert_close(nested.element_sizes(), torch.tensor([[2, 2, 4], [3, 3, 4]]))
        assert_close(nested.packed_local_indices(level=0), torch.tensor([0, 1, 0, 1, 2]))

        default_sizes = nested.element_sizes()
        assert default_sizes is nested.element_sizes()
        converted = nested.element_sizes(device=device, dtype=dtype)
        assert_close(converted, default_sizes.to(device=device, dtype=dtype))
        changed = nested.packed_with_static_tail(torch.empty(13, 7))
        assert_close(
            changed.element_sizes(device=device, dtype=dtype),
            torch.tensor([[2, 2, 7], [3, 3, 7]], device=device, dtype=dtype),
        )
        default_sizes[:, -1].fill_(5)
        assert_close(nested.element_sizes(device=device, dtype=dtype), default_sizes.to(device=device, dtype=dtype))

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices are required")
    @pytest.mark.parametrize(
        ("accessor", "coordinates"),
        [
            ("packed_local_indices", [0, 1, 0, 1, 2, 3]),
            ("packed_batch_indices", [0, 0, 1, 1, 1, 1]),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_packed_indices_follow_current_and_explicit_devices(self, accessor, coordinates, dtype):
        nested = NestedTensor([torch.empty(2, 3), torch.empty(4, 3)])
        indices = getattr(nested, accessor)
        for current in (0, 1, 0):
            with torch.cuda.device(current):
                assert_close(
                    indices(device="cuda", dtype=dtype),
                    torch.tensor(coordinates, device=torch.device("cuda", current), dtype=dtype),
                )
                for target in (0, 1):
                    device = torch.device("cuda", target)
                    assert_close(
                        indices(device=device, dtype=dtype), torch.tensor(coordinates, device=device, dtype=dtype)
                    )

    @pytest.mark.parametrize("accessor", ["packed_local_indices", "packed_batch_indices"])
    def test_packed_indices_do_not_reuse_eager_results_in_fake_mode(self, accessor):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        nested = NestedTensor([torch.empty(2, 3), torch.empty(4, 3)])
        indices = getattr(nested, accessor)
        expected = indices().clone()
        with fake_tensor_mod.FakeTensorMode(allow_non_fake_inputs=True):
            assert fake_tensor_mod.is_fake(indices())
        assert_close(indices(), expected)

    @pytest.mark.parametrize("warm_cuda", [False, True])
    def test_packed_layout_accessors_support_empty_and_fake_tensors(self, warm_cuda):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        empty = NestedTensor([], dtype=torch.float32)
        reference = NestedTensor([torch.empty(2, 3), torch.empty(4, 3)])
        if warm_cuda:
            if not torch.cuda.is_available():
                pytest.skip("CUDA is required to warm device metadata")
            device = torch.device("cuda", torch.cuda.current_device())
            reference.packed_offsets(device=device)
            reference.ragged_level_offsets(0, device=device)
            reference.element_sizes(device=device)

        with fake_tensor_mod.FakeTensorMode() as mode:
            fake = mode.from_tensor(reference)
            output = fake.packed_like(torch.empty_like(fake.values()))
            if warm_cuda:
                fake_offsets = output.ragged_level_offsets(0, device=device)
                assert fake_tensor_mod.is_fake(fake_offsets)
                assert fake_tensor_mod.is_fake(output.element_sizes(device=device))

        assert empty.element_sizes().shape == (0, 0)
        assert fake_tensor_mod.is_fake(output.values())
        assert fake_tensor_mod.is_fake(output.packed_offsets())
        assert output.shape == reference.shape

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_packed_like_fullgraph_forward_backward(self):
        reference = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        packed_values = torch.randn_like(reference.values(), requires_grad=True)
        compiled = torch.compile(
            lambda ref, values: ref.packed_like(values).values().square().sum(),
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
        output = reference.packed_like(torch.randn_like(reference.values()))

        assert type(output) is DerivedNestedTensor
        assert output.shape == reference.shape
        gradient = torch.func.grad(lambda value: value[:, :1].sin().sum())(reference)
        assert type(gradient) is DerivedNestedTensor
        assert type(torch.vmap(lambda value: value)(reference)) is DerivedNestedTensor


class TestPackedWithStaticTail:

    @pytest.mark.parametrize("same_shape", [False, True])
    @pytest.mark.parametrize("strided", [False, True])
    def test_values(self, same_shape, strided):
        reference = NestedTensor(
            [torch.empty(2, 2, 3), torch.empty(4, 2, 3)] if same_shape else [torch.empty(2), torch.empty(4)],
            batch_first=False,
            padding_value=-2.5,
            mask_value=True,
        )
        leaf = torch.randn((6, 3, 2) if strided else (6, 2, 3), dtype=torch.float64, requires_grad=True)
        packed_values = leaf.square()
        if strided:
            packed_values = packed_values.transpose(1, 2)

        output = reference.packed_with_static_tail(packed_values)
        gradient = torch.autograd.grad(output.values().sum(), leaf)[0]

        assert output.values().stride() == packed_values.stride()
        assert output.dtype == packed_values.dtype
        assert output.device == packed_values.device
        assert output.shape == (4, 2, 2, 3)
        assert output.batch_first is False
        assert output.padding_value == -2.5
        assert output.mask_value is True
        assert [tuple(element.shape) for element in output] == [(2, 2, 3), (4, 2, 3)]
        assert_close(gradient, 2 * leaf)
        expected = packed_values.detach().clone()
        with torch.no_grad():
            packed_values.add_(1)
        assert_close(output.values(), expected + 1)

    @pytest.mark.parametrize("width", [3, 7])
    def test_nonleading(self, width):
        reference = NestedTensor(
            [torch.empty(2, 2, 3), torch.empty(2, 4, 3)],
            ragged_dims=(1,),
        )
        packed_values = torch.randn(6, 2, width)

        output = reference.packed_with_static_tail(packed_values)

        assert output.values() is packed_values
        assert output.shape == (2, 2, 4, width)
        assert output.ragged_dims == (1,)
        expected = (packed_values[:2].permute(1, 0, 2), packed_values[2:].permute(1, 0, 2))
        for actual, wanted in zip(output, expected):
            assert_close(actual, wanted)

    def test_packed_with_static_tail_rejects_invalid_layouts_and_values(self):
        permuted = NestedTensor([torch.empty(2, 3), torch.empty(4, 3)]).permute(0, 2, 1)
        scalar = NestedTensor([torch.tensor(1.0), torch.tensor(2.0)])
        reference = NestedTensor([torch.empty(2), torch.empty(4)])

        # A permuted single-ragged layout is declared and tensor-backed, so it is accepted.
        retailed = permuted.packed_with_static_tail(torch.empty(permuted.values().shape[0], 5))
        assert retailed.ragged_dims == permuted.ragged_dims
        assert retailed.values().shape == (permuted.values().shape[0], 5)
        with pytest.raises(ValueError, match="at least one ragged dimension"):
            scalar.packed_with_static_tail(torch.empty_like(scalar.values()))
        unsupported = NT([torch.empty(2, 2, 3), torch.empty(4, 4, 3)], ragged_dims=(0, 1)).permute(0, 3, 1, 2)
        with pytest.raises(ValueError, match="canonical packed order"):
            unsupported.packed_with_static_tail(torch.empty_like(unsupported.values()))
        with pytest.raises(TypeError, match="dense Tensor"):
            reference.packed_with_static_tail(object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="reference packed length"):
            reference.packed_with_static_tail(torch.empty(5, 3))

    @pytest.mark.parametrize("width", [3, 7])
    def test_fake(self, width):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        reference = NestedTensor(
            [torch.empty(2, 2, 3), torch.empty(2, 4, 3)],
            ragged_dims=(1,),
        )

        with fake_tensor_mod.FakeTensorMode() as mode:
            output = mode.from_tensor(reference).packed_with_static_tail(mode.from_tensor(torch.empty(6, 2, width)))

        assert fake_tensor_mod.is_fake(output.values())
        assert output.shape == (2, 2, 4, width)
        assert output.ragged_dims == (1,)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize("same_shape", [False, True])
    def test_compiled(self, same_shape):
        reference = NT([torch.empty(2, 3), torch.empty(4, 3)] if same_shape else [torch.empty(2), torch.empty(4)])
        packed_values = torch.randn(6, 3, requires_grad=True)
        compiled = torch.compile(
            lambda ref, values: ref.packed_with_static_tail(values).values().square().sum(),
            backend="aot_eager",
            fullgraph=True,
        )

        loss = compiled(reference, packed_values)
        loss.backward()

        assert_close(loss, packed_values.detach().square().sum())
        assert_close(packed_values.grad, 2 * packed_values)

    @pytest.mark.parametrize("kind", ["subclass", "strided"])
    def test_reference(self, kind):
        class Derived(NT):
            pass

        factory = Derived if kind == "subclass" else NT
        reference = factory([torch.empty(2, 3), torch.empty(4, 3)], batch_first=False)
        if kind == "strided":
            names, metadata = reference.__tensor_flatten__()
            reference = NT.__tensor_unflatten__(
                {name: getattr(reference, name) for name in names}, metadata, reference.shape, (3, 12, 1)
            )
        values = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
        output = reference.packed_with_static_tail(values)
        assert type(output) is factory
        assert output.stride() == torch.empty(output.shape).stride()
        assert_close(output.unbind(1), (values[:2], values[2:]))
        assert_close(torch.autograd.grad(output.values().square().sum(), values)[0], 2 * values)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
    @pytest.mark.parametrize(
        "reference_pinned,values_pinned", [(False, False), (False, True), (True, False), (True, True)]
    )
    def test_pinning(self, reference_pinned, values_pinned):
        reference = NT([torch.empty(2, 3), torch.empty(4, 3)], pin_memory=reference_pinned)
        values = torch.empty(6, 3, pin_memory=values_pinned).normal_().requires_grad_()
        output = reference.packed_with_static_tail(values)
        assert output.values().is_pinned() is values_pinned
        assert_close(output.values(), values)
        assert_close(torch.autograd.grad(output.values().square().sum(), values)[0], 2 * values)
        with torch.no_grad():
            values.add_(1)
        assert_close(output.values(), values)

    def test_device(self):
        reference = NT([torch.empty(2, 3), torch.empty(4, 3)])
        values = torch.empty(6, 3, device="meta")
        output = reference.packed_with_static_tail(values)
        assert output.device.type == "meta"
        assert output.shape == reference.shape

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    @pytest.mark.parametrize("tail_width", [0, 3])
    def test_packed_with_static_tail_fullgraph_validates_dynamic_packed_length(self, device, backend, tail_width):
        torch.compiler.reset()
        reference = NestedTensor([torch.empty(2, device=device), torch.empty(4, device=device)])

        def select_and_rebuild(ref, values, mask):
            return ref.packed_with_static_tail(values[mask]).values().square().sum()

        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            compiled = torch.compile(select_and_rebuild, backend=backend, fullgraph=True, dynamic=True)
            for requires_grad in (True, False):
                values = torch.randn(8, tail_width, device=device, requires_grad=requires_grad)
                mask = torch.arange(8, device=device) < 6
                loss = compiled(reference, values, mask)
                expected = values[mask].square().sum()

                assert_close(loss, expected)
                if requires_grad:
                    gradient = torch.autograd.grad(loss, values)[0]
                    assert_close(gradient, torch.autograd.grad(expected, values)[0])
                for length in (5, 7):
                    with pytest.raises(RuntimeError, match="reference packed length"):
                        compiled(reference, values, torch.arange(8, device=device) < length)


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
        gradient = torch.autograd.grad(output.values().sum(), leaf)[0]

        assert output.values().untyped_storage().data_ptr() == packed_values.untyped_storage().data_ptr()
        assert output.values().stride() == packed_values.stride()
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

        assert fake_tensor_mod.is_fake(output.values())
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
            lambda ref, values, lengths: ref.packed_with_lengths(values, lengths).values().square().sum(),
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
            lambda ref, values, lengths: ref.packed_with_lengths(values, lengths).values(),
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
        assert_close(output.values(), packed_values)
        assert_close(output[1], packed_values.reshape(3, 3, 2, 4))
        assert_close(torch.autograd.grad(output.values().sum(), leaf)[0], 2 * leaf)

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

        assert fake_tensor_mod.is_fake(output.values())
        assert output.shape == (2, 3, 3, 2, 4)
        assert output.ragged_dims == (0, 1)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_fullgraph_backward_across_square_lengths(self):
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        compiled = torch.compile(
            lambda ref, values, lengths: ref.packed_with_square_lengths(values, lengths).values().square().sum(),
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
            lambda ref, values, lengths: ref.packed_with_square_lengths(values, lengths).values().sum(),
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
        assert_close(output.values(), values)
        assert_close(torch.autograd.grad(output.values().sum(), leaf)[0], 2 * leaf)

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

        assert fake_tensor_mod.is_fake(output.values())
        assert output.shape == (2, 3, 4, 2)
        assert output.ragged_dims == (0, 1)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_fullgraph_rectangular_backward_across_layouts(self):
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        compiled = torch.compile(
            lambda ref, values, rows, columns: (
                ref.packed_with_rectangular_lengths(values, rows, columns).values().square().sum()
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

        output = nested.to(target_dtype)

        assert output.dtype == target_dtype
        assert [element.shape for element in output] == [element.shape for element in nested]
        assert_close(output.values(), torch.cat(elements).to(target_dtype))

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
            gradients = torch.autograd.grad(output.values(), elements, cotangent)
            reference_gradients = torch.autograd.grad(reference, reference_elements, cotangent)

            assert output.dtype == target_dtype
            assert output.ragged_dims == nested.ragged_dims
            assert [element.shape for element in output] == [element.shape for element in nested]
            assert_close(output.values(), reference)
            for actual, expected in zip(gradients, reference_gradients, strict=True):
                assert_close(actual, expected)


# ---------------------------------------------------------------------------
# Copy Semantics
# ---------------------------------------------------------------------------


class TestCopySemantics:

    @pytest.mark.parametrize("batch_first", [False, True])
    def test_shape_checkpoint(self, batch_first, tmp_path, run_isolated):
        source = NT([torch.zeros(4, 3), torch.zeros(2, 3)], ragged_dims=(0,), batch_first=batch_first)
        path = tmp_path / "shape.pt"
        shape = source.shape
        torch.save(
            {
                "shape": shape,
                "tuple": tuple(shape),
                "list": list(shape),
                "axes": [source.size(dim) for dim in range(source.ndim)],
            },
            path,
        )
        expected = tuple(shape)
        run_isolated(f"""
            import torch
            values = torch.load({str(path)!r})
            assert type(values['shape']) is torch.Size
            assert type(values['tuple']) is tuple
            assert type(values['list']) is list
            for shape in values.values():
                assert tuple(shape) == {expected!r}
                assert all(type(dim) is int for dim in shape)
            """)

    @pytest.mark.parametrize("duplicate", [copy.copy, copy.deepcopy])
    def test_copy_preserves_independent_explicit_leaf_gradients(self, duplicate):
        source = NT([torch.arange(1.0, 4.0), torch.arange(1.0, 3.0)]).requires_grad_()
        copied = duplicate(source)
        copied.values().square().sum().backward()
        assert copied.is_leaf
        assert source.grad is None
        assert_close(copied.grad.values(), 2 * source.values())
        source.values().sum().backward()
        assert_close(source.grad.values(), torch.ones(5))

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

        shallow.values()[0] = -7.0
        assert nested.values()[0].item() == -7.0

    def test_deep_copy_has_independent_values_and_preserves_config(self):
        nested = self.configured()
        original = nested.values().clone()
        deep = copy.deepcopy(nested)

        assert_close(deep, nested)
        assert deep.batch_first is False
        assert deep.padding_value == -1
        assert deep.mask_value is True

        deep.values().fill_(0)
        assert_close(nested.values(), original)
        assert not torch.equal(deep.values(), nested.values())

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

    def test_mask_batch_selection_respects_layout_and_dtype(self):
        padded = torch.arange(24.0).reshape(3, 2, 4)
        output = NT.from_tensor_mask(
            padded, torch.tensor([True, False]), batched=True, batch_first=False, dtype=torch.float64
        )
        assert output.dtype == torch.float64
        assert_close(output.tensor, padded[:, :1].double())

    def test_nested_like_checks_nested_batch_and_topology(self):
        source = NT([torch.ones(2), torch.ones(4)])
        with pytest.raises(ValueError, match="batch size"):
            source.nested_like(NT([torch.ones(1)]))
        with pytest.raises(ValueError, match="shape"):
            source.nested_like(NT([torch.ones(4), torch.ones(2)]))
        assert_close(source.nested_like(source), source)

    def test_from_packed_rejects_inconsistent_static_tail(self):
        with pytest.raises(ValueError, match="static dimensions"):
            NT.from_packed(
                torch.arange(35.0).reshape(7, 5),
                torch.tensor([0, 3, 7]),
                torch.tensor([[3, 4], [4, 4]]),
                ragged_dims=(0,),
            )

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
        for supplied, mask_value in ((mask, False), (~mask, True)):
            with pytest.raises(ValueError, match="rectangular prefix"):
                NestedTensor.from_tensor_mask(padded, supplied, mask_value=mask_value)

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
        with pytest.raises(ValueError, match="rectangular prefix"):
            NestedTensor.from_tensor_mask(padded, mask)

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
        with pytest.raises(ValueError, match="rectangular prefix"):
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

    def test_from_concatenated_round_trip_preserves_permuted_layout(self):
        parts = [torch.arange(r * 6.0).reshape(r, 2, 3) for r in (2, 4)]
        source = NT(parts, ragged_dims=(0,)).permute(0, 3, 1, 2)
        values, shapes = source.concatenate()
        assert isinstance(shapes, tuple)
        restored = NT.from_concatenated(values, pickle.loads(pickle.dumps(shapes)))
        for actual, expected in zip(restored, parts):
            assert_close(actual, expected.permute(2, 0, 1))

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


class TestSlicing:

    @pytest.mark.parametrize("batch_first", [False, True])
    @pytest.mark.parametrize("packed", [False, True])
    def test_empty_transpose(self, device, batch_first, packed):
        leaf = torch.randn(2, 2, 1, device=device, dtype=torch.float64, requires_grad=True)
        source = NT([leaf], batch_first=batch_first)
        output = source[:0].transpose(0, 1)
        expected_shape = (2, 0, 2, 1) if batch_first else (0, 2, 2, 1)

        assert output.shape == expected_shape
        assert output.batch_first is not batch_first
        assert_close(output.tensor, torch.empty(expected_shape, device=device, dtype=leaf.dtype))
        loss = output.values().sum() if packed else output.sum()
        gradient = torch.autograd.grad(loss, leaf)[0]
        assert_close(gradient, torch.zeros_like(leaf))

    @pytest.mark.parametrize("batch_first", [False, True])
    @pytest.mark.parametrize("selector", [slice(None, 2), slice(1, None), slice(None, None, 2), slice(0, 0)])
    def test_empty_batch(self, device, batch_first, selector):
        source = NT(
            [torch.zeros(4, 3, device=device), torch.zeros(2, 3, device=device)],
            ragged_dims=(0,),
            batch_first=batch_first,
        )
        ragged_axis = 1 if batch_first else 0
        empty = source[:0]
        index = [slice(None)] * empty.dim()
        index[ragged_axis] = selector
        expected = torch.empty(tuple(empty.shape), device=device)[tuple(index)]
        output = empty[:, selector]
        assert output.shape == expected.shape
        assert output.element_sizes().shape == (0, 2)
        assert output.numel() == 0

    @pytest.mark.parametrize(
        ("selection", "dim", "transpose"),
        [
            (slice(1, None), 1, False),
            (slice(-3, 9, 2), 2, True),
            (slice(None, None, 2), 2, False),
            (slice(4, 4), 1, False),
        ],
    )
    def test_multi_ragged(self, device, selection, dim, transpose):
        parts = [
            torch.randn(*shape, 3, device=device, dtype=torch.float64, requires_grad=True) for shape in ((1, 2), (2, 4))
        ]
        references = [part.detach().clone().requires_grad_() for part in parts]
        source = NT(parts, ragged_dims=(0, 1))
        if transpose:
            source = source.transpose(1, 2)
        index = (slice(None),) * dim + (selection,)
        actual = source[index]
        expected = [(part.transpose(0, 1) if transpose else part)[index[1:]] for part in references]
        torch.testing.assert_close(actual.unbind(), expected)
        torch.testing.assert_close(actual.element_sizes(), torch.tensor([tuple(part.shape) for part in expected]))
        torch.testing.assert_close(
            torch.autograd.grad(actual.values().square().sum(), parts),
            torch.autograd.grad(sum(part.square().sum() for part in expected), references),
        )

    @pytest.mark.parametrize("backend", [None, "aot_eager"])
    @pytest.mark.parametrize(
        ("selection", "transpose"),
        [
            pytest.param(slice(None), False, id="all"),
            pytest.param(slice(None, -3), True, id="trim"),
            pytest.param(slice(1, -1, 2), False, id="step"),
            pytest.param(slice(-4, None), True, id="suffix"),
            pytest.param(slice(2, 2), False, id="empty"),
            pytest.param(slice(-99, 99), True, id="bounds"),
            pytest.param(slice(-3, 2), False, id="mixed"),
        ],
    )
    def test_ragged(self, device, backend, selection, transpose):
        torch._dynamo.reset()

        def select(template, values):
            source = template.packed_like(values)
            if transpose:
                return source.transpose(1, 2)[..., selection]
            return source[:, selection]

        operation = select if backend is None else torch.compile(select, backend=backend, fullgraph=True, dynamic=True)
        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            for lengths in ((0, 1, 3, 7), (2, 5, 1, 4)):
                template = NT([torch.empty(length, 3, device=device) for length in lengths], ragged_dims=(0,))
                values = torch.randn(sum(lengths), 3, device=device, dtype=torch.float64, requires_grad=True)
                reference_values = values.detach().clone().requires_grad_()
                parts = reference_values.split(lengths)
                expected = [part.T[..., selection] if transpose else part[selection] for part in parts]

                actual = operation(template, values)

                shape = (len(expected), *(max(part.shape[dim] for part in expected) for dim in range(2)))
                assert actual.shape == torch.Size(shape)
                torch.testing.assert_close(
                    actual.element_sizes(), torch.tensor([tuple(part.shape) for part in expected])
                )
                torch.testing.assert_close(actual.unbind(), expected)
                probes = [torch.randn_like(part) for part in expected]
                torch.testing.assert_close(
                    torch.autograd.grad(sum((part * probe).sum() for part, probe in zip(actual, probes)), values),
                    torch.autograd.grad(
                        sum((part * probe).sum() for part, probe in zip(expected, probes)), reference_values
                    ),
                )


class TestPackedMask:

    def test_values(self, device):
        parts = [torch.randn(n, device=device, dtype=torch.float64, requires_grad=True) for n in (3, 5)]
        references = [part.detach().clone().requires_grad_() for part in parts]
        masks = [
            torch.zeros(3, device=device, dtype=torch.bool),
            torch.tensor([True, False, True, True, False], device=device),
        ]
        output = NT(parts, ragged_dims=(0,))[NT(masks, ragged_dims=(0,))]
        expected = [part[mask] for part, mask in zip(references, masks)]
        for actual, wanted in zip(output, expected):
            torch.testing.assert_close(actual, wanted)
        probe = torch.randn_like(output.values())
        torch.testing.assert_close(
            torch.autograd.grad(output.values(), parts, probe),
            torch.autograd.grad(torch.cat(expected), references, probe),
        )


class TestPairedIndex:
    @pytest.mark.parametrize("grouped", [False, True])
    def test_values(self, device, grouped):
        lengths = (3, 5)
        template = NT([torch.empty(length, 3, device=device) for length in lengths], ragged_dims=(0,))
        if grouped:
            points = torch.tensor([[[0, -1], [1, 1]], [[0, 2], [-1, 2]]], device=device)
            tail = slice(1, None)
        else:
            points = torch.tensor([[0, -1, 1], [1, -1, 1]], device=device)
            tail = slice(None)
        batch = torch.tensor([0, -1], device=device).reshape(2, *([1] * (points.dim() - 1)))

        def select(reference, values, batch, points):
            return reference.packed_like(values)[batch, points, tail]

        for backend in (None, "aot_eager"):
            values = torch.arange(sum(lengths) * 3, device=device, dtype=torch.float64).reshape(-1, 3).requires_grad_()
            reference_values = values.detach().clone().requires_grad_()
            expected = torch.stack(
                [part[index][..., tail] for part, index in zip(reference_values.split(lengths), points)]
            )
            operation = select if backend is None else torch.compile(select, backend=backend, fullgraph=True)
            actual = operation(template, values, batch, points)
            torch.testing.assert_close(actual, expected)
            probe = torch.arange(actual.numel(), device=device, dtype=actual.dtype).reshape_as(actual) / 7
            torch.testing.assert_close(
                torch.autograd.grad(actual, values, probe),
                torch.autograd.grad(expected, reference_values, probe),
            )

    @pytest.mark.parametrize("point", [3, -4])
    def test_bounds(self, point):
        source = NT([torch.zeros(3, 2), torch.zeros(5, 2)], ragged_dims=(0,))
        with pytest.raises(IndexError):
            source[torch.tensor([0, 1]), torch.tensor([point, 0]), :]


class TestIndexing:

    @pytest.mark.parametrize("position", [0, 1, -1, -2])
    def test_compiled_batch_select_ragged_position(self, device, position):
        def select(source):
            return source[:, position, :]

        compiled = torch.compile(select, backend="aot_eager", fullgraph=True, dynamic=True)
        for lengths in ((3, 5), (4, 6)):
            parts = [torch.randn(length, 4, device=device, requires_grad=True) for length in lengths]
            references = [part.detach().clone().requires_grad_() for part in parts]
            actual = compiled(NT(parts, ragged_dims=(0,)))
            expected = torch.stack([part[position] for part in references])
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(
                torch.autograd.grad(actual.square().sum(), parts),
                torch.autograd.grad(expected.square().sum(), references),
            )

        invalid_lengths = (1, 5) if position in (1, -2) else (0, 5)
        invalid = NT([torch.randn(length, 4, device=device) for length in invalid_lengths], ragged_dims=(0,))
        with pytest.raises(RuntimeError, match="out of bounds"):
            compiled(invalid)

    @pytest.mark.parametrize(
        "entry,index,ragged,batch_first,patterns",
        [
            ("int", 0, (1,), True, (((1, 3), (1, 5)), ((1, 0), (1, 1)))),
            ("int", -1, (0,), False, (((2, 3), (4, 3)), ((0, 3), (1, 3)))),
            ("tuple-int", 0, (0, 1), True, (((2, 3, 2), (4, 1, 2)), ((0, 2, 2), (1, 0, 2)))),
            ("tuple-int", -1, (1, 0), False, (((3, 2, 2), (1, 4, 2)), ((2, 0, 2), (1, 1, 2)))),
            ("iterate", 0, (0,), True, (((2, 3), (5, 3)), ((0, 3), (1, 3)))),
            (
                "iterate",
                0,
                (0, 1),
                True,
                (((1, 3, 2), (2, 4, 2)), ((0, 2, 2), (1, 1, 2)), ((0, 2, 2), (1, 0, 2))),
            ),
        ],
        ids=["first-trailing", "last-leading", "first-multi", "last-permuted", "iterate-leading", "iterate-multi"],
    )
    @pytest.mark.parametrize(
        "backend,fullgraph,strided",
        [("aot_eager", False, False), ("aot_eager", True, False), ("inductor", False, True), ("inductor", True, True)],
    )
    def test_samples(self, device, entry, index, ragged, batch_first, patterns, backend, fullgraph, strided):
        fake_tensor = pytest.importorskip("torch._subclasses.fake_tensor")

        select = {
            ("int", 0): lambda source: source[0],
            ("int", -1): lambda source: source[-1],
            ("tuple-int", 0): lambda source: source[0, ...],
            ("tuple-int", -1): lambda source: source[-1, ...],
            ("iterate", 0): lambda source: tuple(source),
        }[entry, index]

        torch.compiler.reset()
        compiled = torch.compile(select, backend=backend, fullgraph=fullgraph, dynamic=True)
        try:
            for shapes in patterns:
                parts = [torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True) for shape in shapes]
                references = [part.detach().clone().requires_grad_() for part in parts]
                source = NT(parts, ragged_dims=ragged, batch_first=batch_first)
                if strided:
                    values = source.values().movedim(0, -1).clone(memory_format=torch.contiguous_format).movedim(-1, 0)
                    source = source.packed_like(values)
                    assert source.values().stride() == values.stride()
                actual = compiled(source)
                expected = tuple(references) if entry == "iterate" else references[index]
                torch.testing.assert_close(actual, expected)
                outputs = actual if entry == "iterate" else (actual,)
                expected_outputs = expected if entry == "iterate" else (expected,)
                assert all(isinstance(part, torch.Tensor) and not isinstance(part, NT) for part in outputs)
                gradients = torch.autograd.grad(
                    sum(part.square().sum() for part in outputs), parts, allow_unused=True, materialize_grads=True
                )
                reference_gradients = torch.autograd.grad(
                    sum(part.square().sum() for part in expected_outputs),
                    references,
                    allow_unused=True,
                    materialize_grads=True,
                )
                torch.testing.assert_close(gradients, reference_gradients)
                with torch.inference_mode():
                    torch.testing.assert_close(compiled(source), expected)
            with torch.no_grad(), fake_tensor.FakeTensorMode() as mode:
                actual = select(mode.from_tensor(source))
            outputs = actual if entry == "iterate" else (actual,)
            for part, reference in zip(outputs, expected_outputs):
                assert fake_tensor.is_fake(part) and not isinstance(part, NT)
                assert part.shape == reference.shape
        finally:
            torch.compiler.reset()

    @pytest.mark.parametrize("position", [0, -1])
    def test_static_tail_integer_index_preserves_values_and_vjp(self, position):
        template = NT([torch.empty(2, 3), torch.empty(5, 3)], ragged_dims=(0,))
        values = torch.randn_like(template.values(), requires_grad=True)
        coordinates = template.packed_like(values)
        expected = values[:, position]

        output = coordinates[:, :, position]

        cotangent = torch.randn_like(expected)
        actual_grad = torch.autograd.grad(output.values(), values, cotangent)[0]
        expected_grad = torch.autograd.grad(expected, values, cotangent)[0]
        assert type(output) is NestedTensor
        assert [tuple(element.shape) for element in output] == [(2,), (5,)]
        assert_close(output.values(), expected)
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
        values = torch.randn_like(template.values(), requires_grad=True)
        atoms = template.packed_like(values)
        expected = values[:, None, :, None, :]

        output = atoms[:, :, None, :, None]

        cotangent = torch.randn_like(expected)
        actual_grad = torch.autograd.grad(output.values(), values, cotangent)[0]
        expected_grad = torch.autograd.grad(expected, values, cotangent)[0]
        assert output.shape == torch.Size((2, 4, 1, 5, 1, 3))
        assert output.ragged_dims == (0,)
        assert_close(output.values(), expected)
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

        assert fake_tensor_mod.is_fake(indexed.values())
        assert indexed.values().shape == (6, 3)
        assert indexed.ragged_dims == (0,)
        assert fake_tensor_mod.is_fake(expanded.values())
        assert expanded.values().shape == (6, 1, 5, 1, 3)
        assert expanded.ragged_dims == (0,)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_static_and_newaxis_indices_support_fullgraph_backward(self):
        compiled = torch.compile(
            lambda template, values: (
                template.packed_like(values)[:, :, 0].values(),
                template.packed_like(values)[:, :, None, :, None].values(),
            ),
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )

        for lengths in ((2, 3), (3, 5)):
            template = NT([torch.empty(length, 5, 3) for length in lengths], ragged_dims=(0,))
            values = torch.randn_like(template.values(), requires_grad=True)
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
            return output.values(), output.element_sizes(), output.ragged_dims

        compiled = torch.compile(slice_channels, backend="aot_eager", fullgraph=True, dynamic=True)
        values = torch.randn_like(template.values(), requires_grad=True)
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
        assert_close(selected, NT([torch.arange(3), torch.arange(2) + 20]))

    def test_getitem_tensor_long_batch_select(self):
        nested_tensor = NestedTensor([torch.arange(3), torch.arange(5) + 10, torch.arange(2) + 20])
        selected = nested_tensor[torch.tensor([2, 0, 2])]
        assert_close(selected, NT([torch.arange(2) + 20, torch.arange(3), torch.arange(2) + 20]))

    def test_getitem_tensor_bool_batch_length_mismatch_raises(self):
        nested_tensor = NestedTensor([torch.arange(3), torch.arange(5) + 10])
        with pytest.raises(IndexError, match="Boolean index has length 1 but batch size is 2"):
            _ = nested_tensor[torch.tensor([True])]

    def test_getitem_tuple_slice_rest(self):
        nested_tensor = NestedTensor([torch.arange(3), torch.arange(5)])
        sliced = nested_tensor[:, 1:]
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
        assert nt.values()[0, 0, 0].item() == -1.0

    def test_materialized_tensor_reflects_getitem_alias_mutation(self):
        nt = NestedTensor([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
        _ = nt.tensor
        first = nt[0]
        first[0] = -7.0
        assert nt.tensor[0, 0].item() == -7.0

    def test_materialized_tensor_reflects_concat_alias_mutation(self):
        nt = NestedTensor([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
        _ = nt.tensor
        concat = nt.values()
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

    @pytest.mark.parametrize("inference", [False, True])
    def test_layout_mutation(self, device, inference):
        values = torch.arange(5.0, device=device)
        with torch.inference_mode(inference):
            offsets = torch.tensor([0, 2, 5])
            sizes = torch.tensor([[2], [3]])
            source = NT.from_packed(values, offsets, sizes, ragged_dims=(0,), outer_size=(2, 3), mark_dynamic=False)
            _ = source.tensor_mask, source.tolist(), source.element_sizes(dtype=torch.int32)

            offsets.copy_(torch.tensor([0, 3, 5]))
            sizes.copy_(torch.tensor([[3], [2]]))

            assert_close(source.tensor, torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 0.0]], device=device))
            assert_close(source.mask, torch.tensor([[True, True, True], [True, True, False]], device=device))
            assert source.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0]]
            assert_close(source.element_sizes(dtype=torch.int32), sizes.to(torch.int32))

    @pytest.mark.parametrize("explicit_leaf", [False, True])
    def test_assignment_gradient(self, explicit_leaf):
        parts = [
            torch.arange(12.0, dtype=torch.float64).reshape(2, 3, 2).requires_grad_(),
            torch.arange(16.0, dtype=torch.float64).reshape(4, 2, 2).requires_grad_(),
        ]
        source = NT(parts, ragged_dims=(0, 1))
        replacement = torch.arange(36.0, dtype=torch.float64).reshape(6, 3, 2).requires_grad_()
        if explicit_leaf:
            source = source.detach().requires_grad_()
            with torch.no_grad():
                source[0] = replacement
            source.values().sum().backward()
            assert source.grad.shape == source.shape
            assert_close(source.grad.element_sizes(), source.element_sizes())
        else:
            source = source.square()
            source[0] = replacement
            gradients = torch.autograd.grad(source.values().sum(), (*parts, replacement), allow_unused=True)
            assert_close(gradients[0], torch.zeros_like(parts[0]))
            assert_close(gradients[1], 2 * parts[1])
            assert_close(gradients[2], torch.ones_like(replacement))
        assert source.stride() == torch.empty(source.shape).stride()

    def test_shape_change_preserves_independent_alias_topology(self):
        source = NT([torch.arange(12.0).reshape(2, 3, 2), torch.arange(16.0).reshape(4, 2, 2)], ragged_dims=(0, 1))
        independent = source + 100
        before = independent.element_sizes().clone()
        replacement = torch.arange(12.0).reshape(6, 1, 2)
        source[0] = replacement
        assert_close(source[0], replacement)
        assert_close(independent.element_sizes(), before)
        assert independent.shape == (2, 4, 3, 2)
        assert_close(independent[0], torch.arange(12.0).reshape(2, 3, 2) + 100)
        before_values = source.values().clone()
        before_sizes = source.element_sizes().clone()
        with pytest.raises(ValueError, match="ndim"):
            source[0] = torch.ones(12)
        assert_close(source.values(), before_values)
        assert_close(source.element_sizes(), before_sizes)

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

    @pytest.mark.parametrize("batch_first", [True, False])
    def test_permuted_static_axes_preserve_dense_coordinates_and_gradients(self, batch_first):
        parts = [torch.arange(r * 6.0, dtype=torch.float64).reshape(r, 2, 3).requires_grad_() for r in (2, 4)]
        permutation = (0, 3, 1, 2) if batch_first else (3, 1, 0, 2)
        source = NT(parts, ragged_dims=(0,), batch_first=batch_first).permute(permutation)
        expected_parts = [part.permute(2, 0, 1) for part in parts]
        expected = torch.zeros(2, 3, 4, 2, dtype=torch.float64)
        for i, part in enumerate(expected_parts):
            expected[i, :, : part.size(1)] = part
        if not batch_first:
            expected = expected.movedim(0, 1)
        assert_close(source.tensor, expected)
        restored = source.nested_like(expected)
        for actual, reference in zip(restored, expected_parts):
            assert_close(actual, reference)
        weights = torch.arange(expected.numel(), dtype=expected.dtype).reshape(expected.shape)
        actual_grad = torch.autograd.grad((source.tensor * weights).sum(), parts, retain_graph=True)
        expected_grad = torch.autograd.grad((expected * weights).sum(), parts)
        assert_close(actual_grad, expected_grad)

    @pytest.mark.parametrize("explicit_leaf", [False, True])
    def test_changing_batch_axis_preserves_wrapper_gradients(self, explicit_leaf):
        parts = [torch.randn(r, 3, dtype=torch.float64, requires_grad=not explicit_leaf) for r in (2, 4)]
        source = NT(parts)
        if explicit_leaf:
            source.requires_grad_()
        else:
            source = source.square()
        source.batch_first = False
        weights = torch.arange(source.tensor.numel(), dtype=torch.float64).reshape(source.shape)
        loss = (source.tensor * weights).sum()
        if explicit_leaf:
            loss.backward()
            assert source.grad.shape == source.shape
            assert_close(source.grad.tensor, source.nested_like(weights).tensor)
        else:
            gradients = torch.autograd.grad(loss, parts)
            for index, (actual, part) in enumerate(zip(gradients, parts)):
                assert_close(actual, 2 * part * weights[: part.size(0), index])

    def test_zero_axes_and_scalar_batches_keep_their_rank(self):
        source = NT.from_packed(
            torch.empty(5, 0), torch.tensor([0, 2, 5]), torch.tensor([[2, 0], [3, 0]]), ragged_dims=(0,)
        )
        assert source.tensor.shape == (2, 3, 0)
        source.batch_first = False
        assert source.tensor.shape == (3, 2, 0)
        source[0] = torch.empty(2, 0)
        assert source.tensor.shape == (3, 2, 0)
        empty = NT.from_packed(
            torch.empty(0, 3), torch.tensor([0]), torch.empty(0, 2, dtype=torch.long), ragged_dims=(0,)
        )
        assert empty.shape == (0, 0, 3)
        assert_close(NT([torch.tensor(1.0), torch.tensor(2.0)], batch_first=False).tensor, torch.tensor([1.0, 2.0]))
        assert NT([], batch_first=False).tensor.shape == (0,)
        with pytest.raises(IndexError, match="Dimension out of range"):
            source.size(-4)

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

        batch_permuted = nested_tensor.permute(1, 0, 2, 3)
        assert batch_permuted.batch_first is False
        assert batch_permuted.shape == (3, 2, 4, 5)
        assert_close(batch_permuted.unbind(1), nested_tensor.unbind())

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
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_nested_like_compile_matches_reference(self):
        nested_tensor = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        dense = nested_tensor.tensor
        compiled = torch.compile(lambda x, y: x.nested_like(y), backend="inductor", fullgraph=True)
        output = compiled(nested_tensor, dense)
        reference = nested_tensor.nested_like(dense)
        assert_close(output, reference)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_nested_like_larger_dense_compile_matches_reference(self):
        nested_tensor = NestedTensor([torch.randn(2, 3), torch.randn(4, 3)])
        dense = torch.randn(2, 6, 3)
        compiled = torch.compile(lambda x, y: x.nested_like(y, strict=False), backend="inductor", fullgraph=True)
        output = compiled(nested_tensor, dense)
        reference = nested_tensor.nested_like(dense, strict=False)
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
        assert_close(output, reference)

    def test_irregular_view(self):
        parts = [torch.randn(shape, dtype=torch.float64, requires_grad=True) for shape in ((2, 4), (3, 2))]
        nested_tensor = NestedTensor(parts)
        viewed = nested_tensor.view(len(nested_tensor), 2, -1)

        for actual, element in zip(viewed, parts):
            assert_close(actual, element.view(2, -1))
        gradients = torch.autograd.grad(viewed.square().sum(), parts)
        assert_close(gradients, [2 * part for part in parts])
        with torch.no_grad():
            viewed.add_(1)
            assert_close(nested_tensor.unbind(), [part + 1 for part in parts])
            nested_tensor.mul_(2)
            assert_close(viewed.unbind(), [(part + 1).view(2, -1) * 2 for part in parts])

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
        assert output.values().is_pinned()

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
            assert output.values().is_pinned()

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

    def test_batch_layout(self):
        nested_tensor = NestedTensor(
            [torch.tensor([[1, 2], [3, 4], [7, 8]]), torch.tensor([[5, 6]])],
            batch_first=True,
            padding_value=-1,
        )

        batch_first_tensor = nested_tensor.tensor
        nested_tensor.batch_first = False

        assert nested_tensor.shape == torch.Size([3, 2, 2])
        assert nested_tensor.stride() == (4, 2, 1)
        assert_close(nested_tensor.tensor, batch_first_tensor.movedim(0, 1))
        assert_close(nested_tensor.mask, torch.tensor([[True, True], [True, False], [True, False]]))

        nested_tensor.batch_first = True
        assert nested_tensor.shape == torch.Size([2, 3, 2])
        assert nested_tensor.stride() == (6, 2, 1)
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

    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    def test_returned_padding(self, device, backend):
        torch.compiler.reset()
        run = torch.compile(lambda value: value.sin(), backend=backend, fullgraph=True, dynamic=True)
        for lengths in ((3, 5), (0, 4), (1, 0), (0, 0), ()):
            values = torch.randn(sum(lengths), 3, device=device, dtype=torch.float64, requires_grad=True)
            references = values.detach().clone().requires_grad_()
            source = NT(values.split(lengths or (0,)), ragged_dims=(0,), padding_value=-3)
            if not lengths:
                source = source[:0]
            width = max(lengths, default=0)
            expected = (
                torch.stack(
                    [
                        F.pad(part.sin(), (0, 0, 0, width - length), value=-3)
                        for part, length in zip(references.split(lengths), lengths)
                    ]
                )
                if lengths
                else references.reshape(0, 0, 3)
            )
            actual = run(source).tensor
            assert_close(actual, expected)
            tangent = torch.randn_like(actual)
            assert_close(
                torch.autograd.grad(actual, values, tangent)[0],
                torch.autograd.grad(expected, references, tangent)[0],
            )

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
        assert_close(nested_tensor.values(), torch.tensor([1, 2, 3, 4]))

    def test_flat_scalar_round_trip(self):
        nested_tensor = NestedTensor([torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)])
        assert_close(nested_tensor.values(), torch.tensor([1.0, 2.0, 3.0]))

        concat, shapes = nested_tensor.concatenate()
        output = NestedTensor.from_concatenated(concat, shapes)
        assert_close(output, nested_tensor)

        nested_tensor = NestedTensor([torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)], batch_first=False)
        assert_close(nested_tensor.values(), torch.tensor([1.0, 2.0, 3.0]))

        concat, shapes = nested_tensor.concatenate()
        output = NestedTensor.from_concatenated(concat, shapes, batch_first=False)
        assert_close(output, nested_tensor)

    def test_flat_batch_first_false(self):
        nested_tensor = NestedTensor(
            [torch.arange(3).unsqueeze(1), torch.arange(3, 7).unsqueeze(1)],
            batch_first=False,
        )
        flat = nested_tensor.values()
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


class TestFromPacked:

    @pytest.mark.parametrize("lengths", [(2, 3), (2, 2), ()])
    def test_values(self, device, lengths):
        counts = torch.tensor(lengths, dtype=torch.long)
        offsets = torch.nn.functional.pad(counts.cumsum(0), (1, 0))
        values = torch.arange(sum(lengths), dtype=torch.int32, device=device)
        shapes = counts[:, None] if lengths else torch.empty((0, 0), dtype=torch.long)
        output = NestedTensor.from_packed(
            values,
            offsets,
            shapes,
            permutation=(0,) if lengths else (),
            ragged_dims=(0,) if lengths else (),
            outer_size=(len(lengths), max(lengths)) if lengths else (0,),
            materialize_python_metadata=False,
            validate=False,
        )
        assert output.dtype == values.dtype
        assert output.device == values.device
        output.values().add_(1)
        assert_close(values, torch.arange(sum(lengths), dtype=torch.int32, device=device) + 1)
        expected, start = [], 0
        for length in lengths:
            expected.append(list(range(start + 1, start + length + 1)))
            start += length
        assert output.tolist() == expected

    @pytest.mark.parametrize("operation", [torch.sin, torch.Tensor.sin, torch.ops.aten.sin.default])
    def test_gradients(self, device, operation):
        values = torch.randn(3, 6, dtype=torch.double, device=device, requires_grad=True)
        packed = values.T
        output = NT.from_packed(
            packed,
            torch.tensor([0, 2, 6]),
            torch.tensor([[2, 3], [4, 3]]),
            ragged_dims=(0,),
        )
        result = operation(output)
        expected = packed.sin()
        assert_close(result.values(), expected)
        actual_grad = torch.autograd.grad(result.square().sum(), values)[0]
        expected_grad = 2 * values.sin() * values.cos()
        assert_close(actual_grad, expected_grad)

    @pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
    @pytest.mark.parametrize("fullgraph", [False, True])
    def test_compile(self, device, backend, fullgraph):
        def construct(values, offsets, sizes):
            return NT.from_packed(values, offsets, sizes, ragged_dims=(0,), outer_size=(2, 4, 3), validate=False).sin()

        values = torch.randn(6, 3, dtype=torch.double, device=device, requires_grad=True)
        compiled = torch.compile(construct, backend=backend, fullgraph=fullgraph, dynamic=True)
        output = compiled(values, torch.tensor([0, 2, 6]), torch.tensor([[2, 3], [4, 3]]))
        assert_close(output.values(), values.sin())
        assert_close(torch.autograd.grad(output.square().sum(), values)[0], 2 * values.sin() * values.cos())


def _metadata_pair(lengths: tuple[int, ...], channels: int = 2) -> NestedTensor:
    return NestedTensor(
        [torch.randn(length, length, channels) for length in lengths],
        ragged_dims=(0, 1),
    )


class TestMetadata:

    def test_shape_values(self):
        parts = [
            torch.arange(length * 4.0, dtype=torch.float64).reshape(length, 4).requires_grad_() for length in (17, 34)
        ]
        source = NT(parts, ragged_dims=(0,))
        shape = source.shape
        assert type(shape) is torch.Size
        assert all(type(dim) is int for dim in shape)
        restored = pickle.loads(pickle.dumps(shape))
        assert shape == torch.Size((2, 34, 4))
        assert restored == shape
        assert repr(restored) == "torch.Size([2, 34, 4])"
        assert torch.empty(restored).shape == restored
        assert torch.arange(2 * 34 * 4).reshape(restored).shape == restored
        output = source.reshape(-1).reshape_as(source)
        torch.testing.assert_close(output.unbind(0), tuple(parts))
        torch.testing.assert_close(torch.autograd.grad(output.square().sum(), parts), tuple(2 * part for part in parts))
        with pytest.raises(RuntimeError, match="invalid for input"):
            source.reshape(-1).reshape(restored)
        current = NT([torch.ones(2), torch.ones(4)], ragged_dims=(0,))
        old_shape = current.shape
        current[0] = torch.ones(3)
        assert old_shape == (2, 4)
        assert current.shape == (2, 4)
        torch.testing.assert_close(current.reshape(old_shape).unbind(0), current.unbind(0))
        transposed = NT([torch.arange(length * 2.0).reshape(length, 2) for length in (1, 2)], ragged_dims=(0,)).T
        identity = transposed.reshape(tuple(transposed.shape))
        assert identity.batch_first == transposed.batch_first
        torch.testing.assert_close(identity.unbind(0), transposed.unbind(0))
        empty = NT([])
        assert torch.empty(empty.shape).shape == (0,)
        torch.testing.assert_close(empty.reshape(empty.shape).values(), empty.values())
        assert empty.reshape((*empty.shape, 2)).shape == (0, 2)

    @pytest.mark.parametrize("split", [False, True])
    @pytest.mark.parametrize("operation", ["reshape", "view"])
    def test_captured_shape(self, split, operation):
        torch.compiler.reset()
        parts = [torch.randn(length, 4, requires_grad=True) for length in (2, 4)]
        source = NT(parts, ragged_dims=(0,))
        shape = (*source.shape[:-1], 2, 2) if split else source.shape

        def reshape(value):
            return getattr(value, operation)(shape)

        compiled = torch.compile(reshape, backend="aot_eager", fullgraph=True)
        output = compiled(source)
        expected = tuple(part.reshape(-1, 2, 2) if split else part for part in parts)
        torch.testing.assert_close(output.unbind(0), expected)
        torch.testing.assert_close(torch.autograd.grad(output.square().sum(), parts), tuple(2 * part for part in parts))
        mismatched = NT([torch.randn(length, 4) for length in (3, 4)], ragged_dims=(0,))
        expected = tuple(part.reshape(-1, 2, 2) if split else part for part in mismatched.unbind())
        torch.testing.assert_close(reshape(mismatched).unbind(), expected)
        torch.testing.assert_close(compiled(mismatched).unbind(), expected)
        wrong_batch = NT([torch.randn(2, 4)], ragged_dims=(0,))
        with pytest.raises(RuntimeError):
            reshape(wrong_batch)
        with pytest.raises(RuntimeError):
            compiled(wrong_batch)
        torch.compiler.reset()

    def test_mismatch(self):
        lhs = _metadata_pair((2, 4))
        rhs = _metadata_pair((4, 2))

        with pytest.raises(RuntimeError, match="broadcast-compatible"):
            lhs + rhs

    def test_fake(self):
        fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        pair = _metadata_pair((2, 3))
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        mode = fake_tensor_mod.FakeTensorMode(shape_env=ShapeEnv(allow_dynamic_output_shape_ops=True))
        with mode:
            fake_pair = mode.from_tensor(pair)
            fake_reference = mode.from_tensor(reference)
            rebuilt = fake_reference.packed_with_square_lengths(
                mode.from_tensor(torch.empty(13, 2)),
                mode.from_tensor(torch.tensor([2, 3])),
            )
            output = fake_pair + rebuilt

        assert fake_tensor_mod.is_fake(output.values())
        assert output.ndim == pair.ndim
        assert output.shape[0] == pair.shape[0]
        assert output.shape[-1] == pair.shape[-1]
        # The broadcast maxima depend on runtime lengths, but both square axes
        # retain the same symbolic extent.
        assert output.shape[1] == output.shape[2]
        assert output.ragged_dims == pair.ragged_dims

    def test_compiled(self):
        pair = _metadata_pair((2, 4))
        reference = NestedTensor([torch.empty(1), torch.empty(1)])
        values = torch.randn(20, 2)

        def consume(known, ref, packed_values, lengths):
            rebuilt = ref.packed_with_square_lengths(packed_values, lengths)
            return (known + rebuilt).values()

        compiled = torch.compile(consume, backend="aot_eager", fullgraph=True, dynamic=True)

        assert_close(compiled(pair, reference, values, torch.tensor([2, 4])), pair.values() + values)
        with pytest.raises(RuntimeError, match="broadcast-compatible"):
            compiled(pair, reference, values, torch.tensor([4, 2]))
