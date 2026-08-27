# DanLing
# Copyright (C) 2022-Present  DanLing

"""Public multi-ragged narrow contracts and inferred shared-mask composition."""

import math
from itertools import accumulate

import pytest
import torch

from danling.tensors import NestedTensor as NT
from danling.tensors.ops import nested_execution_guard

LAYOUTS = [
    pytest.param(((2, 2, 4), (3, 3, 4)), (0, 1), 0, id="square-outer"),
    pytest.param(((2, 2, 4), (3, 3, 4)), (0, 1), 1, id="square-inner"),
    pytest.param(((2, 4, 3), (3, 2, 3)), (0, 1), 0, id="rectangular-outer"),
    pytest.param(((2, 4, 3), (3, 2, 3)), (0, 1), 1, id="rectangular-inner"),
    pytest.param(((2, 2, 3, 4), (2, 3, 2, 4)), (1, 2), 1, id="nonleading-outer"),
    pytest.param(((2, 2, 3, 4), (2, 3, 2, 4)), (1, 2), 2, id="nonleading-inner"),
    pytest.param(((2, 3, 4), (3, 2, 4)), (1, 0), 0, id="reversed-inner"),
    pytest.param(((2, 3, 4), (3, 2, 4)), (1, 0), 1, id="reversed-outer"),
    pytest.param(((2, 3, 2, 4), (3, 2, 3, 4)), (0, 1, 2), 0, id="three-level-outer"),
    pytest.param(((2, 3, 2, 4), (3, 2, 3, 4)), (0, 1, 2), 1, id="three-level-middle"),
    pytest.param(((2, 3, 2, 4), (3, 2, 3, 4)), (0, 1, 2), 2, id="three-level-inner"),
]


def packed_guard():
    return nested_execution_guard(
        forbid_eager_fallback=True,
        forbid_storage_map=True,
        forbid_iteration=True,
        forbid_padded_materialization=True,
        forbid_dense_repack=True,
    )


def elements(shapes, device):
    return tuple(
        (
            torch.arange(math.prod(shape), device=device, dtype=torch.float64).reshape(shape) + 1000 * index
        ).requires_grad_()
        for index, shape in enumerate(shapes)
    )


def assert_structure(output, expected, ragged):
    shapes = [tuple(value.shape) for value in expected]
    assert output.element_sizes().tolist() == [list(shape) for shape in shapes]
    assert output.ragged_dims == ragged
    counts = [math.prod(shape[dim] for dim in ragged) for shape in shapes]
    assert output.packed_offsets().tolist() == [0, *accumulate(counts)]
    parents = [1] * len(shapes)
    for level, dim in enumerate(ragged):
        widths = [shape[dim] for shape, count in zip(shapes, parents) for _ in range(count)]
        assert output.ragged_level_offsets(level).tolist() == [0, *accumulate(widths)]
        parents = [count * shape[dim] for count, shape in zip(parents, shapes)]


def verify_narrow(shapes, ragged, element_dim, device, start, length, *, batch_first=True, method=False):
    inputs = elements(shapes, device)
    references = tuple(value.detach().clone().requires_grad_() for value in inputs)
    source = NT(inputs, ragged_dims=ragged, batch_first=batch_first)
    dim = element_dim + 1 if batch_first or element_dim > 0 else element_dim
    with packed_guard():
        output = source.narrow(dim, start, length) if method else torch.narrow(source, dim, start, length)
    expected = tuple(value.narrow(element_dim, start, length) for value in references)
    actual = output.unbind(0 if batch_first else 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert_structure(output, expected, ragged)
    assert output.packed_dim_order == source.packed_dim_order
    assert output.batch_first is batch_first
    weights = tuple(
        torch.arange(1, value.numel() + 1, device=device, dtype=value.dtype).reshape_as(value) for value in expected
    )
    loss = sum((value * weight).sum() for value, weight in zip(actual, weights))
    reference_loss = sum((value * weight).sum() for value, weight in zip(expected, weights))
    with packed_guard():
        observed = torch.autograd.grad(loss, inputs)
    wanted = torch.autograd.grad(reference_loss, references)
    torch.testing.assert_close(observed, wanted, rtol=0, atol=0)


@pytest.mark.parametrize(("shapes", "ragged", "element_dim"), LAYOUTS)
@pytest.mark.parametrize(("start", "length"), ((0, 1), (-1, 1), (-2, 2), (0, 0)))
def test_multi_ragged_narrow_values_partitions_and_vjp(device, shapes, ragged, element_dim, start, length):
    verify_narrow(shapes, ragged, element_dim, device, start, length)


@pytest.mark.parametrize("element_dim", (0, 1))
@pytest.mark.parametrize("batch_first", (True, False))
def test_method_narrow_preserves_batch_placement(device, element_dim, batch_first):
    verify_narrow(((2, 2, 4), (3, 3, 4)), (0, 1), element_dim, device, -1, 1, batch_first=batch_first, method=True)


@pytest.mark.parametrize(
    ("shapes", "element_dim", "length"),
    (
        (((2, 0, 4), (3, 2, 4)), 0, 1),
        (((0, 2, 4), (3, 2, 4)), 0, 0),
        (((2, 2, 0), (3, 3, 0)), 1, 1),
    ),
)
def test_narrow_retains_zero_volume_shapes(device, shapes, element_dim, length):
    verify_narrow(shapes, (0, 1), element_dim, device, 0, length)


@pytest.mark.parametrize("dim", (1, 2))
@pytest.mark.parametrize(("start", "length"), ((2, 1), (-3, 1), (3, 0)))
def test_multi_ragged_bounds_use_each_sample_extent(device, dim, start, length):
    source = NT(elements(((2, 2, 4), (3, 3, 4)), device), ragged_dims=(0, 1))
    with packed_guard(), pytest.raises(RuntimeError, match="exceeds dimension size.*element 0"):
        torch.narrow(source, dim, start, length)


@pytest.mark.parametrize("dim", (1, 2))
def test_empty_batch_narrow_keeps_rank_and_bounds(device, dim):
    inputs = elements(((2, 2, 4), (3, 3, 4)), device)
    source = torch.narrow(NT(inputs, ragged_dims=(0, 1)), 0, 0, 0)
    with packed_guard():
        output = torch.narrow(source, dim, 0, 1)
    shape = [0, 3, 3, 4]
    shape[dim] = 1
    assert tuple(output.shape) == tuple(shape)
    assert output.element_sizes().shape == (0, 3)
    assert output.ragged_dims == (0, 1)
    assert output.packed_offsets().tolist() == [0]
    assert all(output.ragged_level_offsets(level).tolist() == [0] for level in (0, 1))
    torch.testing.assert_close(
        torch.autograd.grad(output.concat.sum(), inputs), tuple(torch.zeros_like(x) for x in inputs)
    )
    with packed_guard(), pytest.raises(RuntimeError, match="exceeds dimension size"):
        torch.narrow(source, dim, 3, 1)


@pytest.mark.parametrize("dim", (0, 3))
def test_existing_batch_and_static_narrow_still_share_storage(device, dim):
    source = NT(elements(((2, 2, 4), (3, 3, 4)), device), ragged_dims=(0, 1))
    with packed_guard():
        output = torch.narrow(source, dim, 0, 1)
    assert output.concat.untyped_storage().data_ptr() == source.concat.untyped_storage().data_ptr()


def test_noncontiguous_packed_input_narrow(device):
    template = NT(elements(((2, 2, 4), (3, 3, 4)), device), ragged_dims=(0, 1))
    base = torch.arange(52, device=device, dtype=torch.float64).reshape(4, 13).requires_grad_()
    reference = base.detach().clone().requires_grad_()
    source = template.packed_like(base.t())
    with packed_guard():
        output = torch.narrow(source, 2, torch.tensor(-1, device=device), 1)
    parts = reference.t().split((4, 9))
    expected = tuple(part.reshape(n, n, 4).narrow(1, -1, 1) for part, n in zip(parts, (2, 3)))
    actual = output.unbind()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    observed = torch.autograd.grad(sum(value.square().sum() for value in actual), base)[0]
    wanted = torch.autograd.grad(sum(value.square().sum() for value in expected), reference)[0]
    torch.testing.assert_close(observed, wanted, rtol=0, atol=0)


def test_multi_ragged_compile_limit_is_explicit():
    source = NT(elements(((2, 2, 4), (3, 3, 4)), "cpu"), ragged_dims=(0, 1))
    compiled = torch.compile(
        lambda value: torch.narrow(value, 2, 0, 1), backend="aot_eager", fullgraph=True, dynamic=True
    )
    with pytest.raises(Exception, match="multi-ragged slicing requires concrete eager extents"):
        compiled(source)


@pytest.mark.parametrize("dim", (1, 2))
@pytest.mark.parametrize("factory", (torch.empty_like, torch.ones_like))
def test_inferred_shared_dropout_mask_composition(device, dim, factory):
    """A public-API composition, not a test of the unavailable external class."""
    inputs = elements(((2, 2, 4), (3, 3, 4)), device)
    references = tuple(value.detach().clone().requires_grad_() for value in inputs)
    source = NT(inputs, ragged_dims=(0, 1))
    with packed_guard():
        narrowed = torch.narrow(source, dim, 0, 1)
        mask = factory(narrowed)
        mask.bernoulli_(0.5, generator=torch.Generator(device=device).manual_seed(941)).div_(0.5)
        output = source * mask
    assert mask.requires_grad is False
    assert mask.ragged_dims == (0, 1)
    sampled_masks = tuple(value.detach() for value in mask.unbind())
    expected = tuple(value * sampled for value, sampled in zip(references, sampled_masks))
    actual = output.unbind()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert_structure(output, expected, (0, 1))
    weights = tuple(
        torch.arange(1, value.numel() + 1, device=device, dtype=value.dtype).reshape_as(value) for value in expected
    )
    observed = torch.autograd.grad(sum((x * w).sum() for x, w in zip(actual, weights)), inputs)
    wanted = torch.autograd.grad(sum((x * w).sum() for x, w in zip(expected, weights)), references)
    torch.testing.assert_close(observed, wanted, rtol=0, atol=0)
