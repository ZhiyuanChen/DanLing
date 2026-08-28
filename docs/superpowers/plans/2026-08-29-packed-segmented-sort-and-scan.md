# Packed Segmented Sort and Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the padded fallbacks in `sort`, `argsort`, `topk`, `cumprod`, `cummax`, `cummin` and `logcumsumexp` with packed segmented implementations, and add the sample-local `inverse_permutation` and `rank` operators that autoregressive decoding-rank generation needs.

**Architecture:** Two primitives in a new `danling/tensors/segmented.py` carry all the packed logic. `_segmented_sort_perm` composes two stable global sorts into a segmented one. `_segmented_scan` and `_segmented_arg_scan` run a log-step Hillis-Steele scan masked at segment boundaries. Every operator becomes a thin expression over one of them, so the packed reasoning is written and tested once.

**Tech Stack:** Python 3.10+, PyTorch (unpinned), pytest, black/isort/flake8 at line length 120.

**Spec:** `docs/superpowers/specs/2026-08-29-packed-segmented-ops-design.md`

## Global Constraints

- Line length 120 for black, isort (profile black) and flake8; `extend-ignore = ["E203"]`.
- Commit messages: subject style `NestedTensor: <lowercase description>`, imperative, <= 72 chars. Body wrapped at 72. End with `Signed-off-by: Zhiyuan Chen <this@zyc.ai>`. **Never** add a `Co-Authored-By` trailer or a "Generated with" line.
- Run tests with `/opt/conda/bin/python -m pytest`. That interpreter has torch 2.11 and the full test stack; the system `python3` has no torch.
- `tests/tensors/test_torch_functions.py::TestMatrixMultiplication::test_medium_matrix_compile` aborts on this machine from a duplicate OpenMP runtime in the conda env. It is unrelated to this work. Always deselect it:
  `--deselect tests/tensors/test_torch_functions.py::TestMatrixMultiplication::test_medium_matrix_compile`
- Every operator this plan touches must, at the end of its task, pass a test asserting `nested_execution_guard(forbid_padded_materialization=True)` does not raise.
- Every operator must also carry the applicable tests from the spec's "Failure modes to test for" section: dense parity on edge inputs, declared-topology projection, autograd after a `no_grad` warm-up, dtype rejection, device handling, and every argument spelling.

---

## File Structure

- **Create** `danling/tensors/segmented.py` — the two primitives and nothing else. Kept out of `torch_functions.py`, which is already about 5000 lines, so the packed segmented logic stays reviewable on its own.
- **Create** `tests/tensors/test_segmented.py` — primitive-level tests against per-element references.
- **Modify** `danling/tensors/aten_functions.py` — the ragged-dim branches of `sort`, `topk`, `cumulative` and `cumulative_pair`.
- **Modify** `danling/tensors/torch_functions.py` — the three new operators `inverse_permutation`, `rank` and `cumcount`.
- **Modify** `tests/tensors/test_torch_functions.py` — operator-level tests.

---

### Task 1: The segmented sort permutation primitive

**Files:**
- Create: `danling/tensors/segmented.py`
- Test: `tests/tensors/test_segmented.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `segmented_sort_perm(values: Tensor, offsets: Tensor, batch_idx: Tensor, *, descending: bool = False) -> tuple[Tensor, Tensor]`, returning `(perm, local)`. `perm` indexes `values` along packed dim 0 so that segments come out contiguous and internally sorted. `local` is the same permutation expressed as per-segment indices. For `values` with a static tail the sort is per column and both results have `values`' shape.

- [ ] **Step 1: Write the failing test**

Create `tests/tensors/test_segmented.py`:

```python
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

r"""Tests for ``danling.tensors.segmented`` — packed segmented primitives."""

import torch

from danling.tensors import NestedTensor, nested_execution_guard
from danling.tensors.segmented import segmented_sort_perm

NT = NestedTensor


def _lengths_case(shape_fn):
    # Includes a singleton segment, which is where an off-by-one in the offsets shows up.
    return NT([torch.randn(*shape_fn(n)) for n in (3, 5, 2, 1)])


class TestSegmentedSortPerm:

    def test_matches_per_element_argsort(self):
        nested = _lengths_case(lambda n: (n,))
        offsets = nested._offsets
        _, local = segmented_sort_perm(nested._values, offsets, nested.packed_batch_indices())

        for index, element in enumerate(nested):
            segment = local[offsets[index] : offsets[index + 1]]
            assert torch.equal(segment, torch.argsort(element, stable=True))

    def test_permutation_sorts_each_segment(self):
        nested = _lengths_case(lambda n: (n,))
        perm, _ = segmented_sort_perm(nested._values, nested._offsets, nested.packed_batch_indices())
        reference = torch.cat([torch.sort(element).values for element in nested])
        torch.testing.assert_close(nested._values[perm], reference)

    def test_static_tail_sorts_each_column(self):
        nested = _lengths_case(lambda n: (n, 4))
        offsets = nested._offsets
        _, local = segmented_sort_perm(nested._values, offsets, nested.packed_batch_indices())

        for index, element in enumerate(nested):
            segment = local[offsets[index] : offsets[index + 1]]
            assert torch.equal(segment, torch.argsort(element, dim=0, stable=True))

    def test_descending(self):
        nested = _lengths_case(lambda n: (n,))
        offsets = nested._offsets
        _, local = segmented_sort_perm(
            nested._values, offsets, nested.packed_batch_indices(), descending=True
        )

        for index, element in enumerate(nested):
            segment = local[offsets[index] : offsets[index + 1]]
            assert torch.equal(segment, torch.argsort(element, stable=True, descending=True))

    def test_is_stable_on_ties(self):
        # All-equal values: a stable sort must return the identity within each segment.
        nested = NT([torch.zeros(3), torch.zeros(4)])
        offsets = nested._offsets
        _, local = segmented_sort_perm(nested._values, offsets, nested.packed_batch_indices())

        for index in range(len(nested)):
            segment = local[offsets[index] : offsets[index + 1]]
            assert torch.equal(segment, torch.arange(segment.numel()))

    def test_does_not_materialize_padding(self):
        nested = _lengths_case(lambda n: (n,))
        with nested_execution_guard(forbid_padded_materialization=True):
            segmented_sort_perm(nested._values, nested._offsets, nested.packed_batch_indices())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_segmented.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'danling.tensors.segmented'`

- [ ] **Step 3: Write minimal implementation**

Create `danling/tensors/segmented.py` (copy the license header from `danling/tensors/ops.py`):

```python
r"""Packed segmented primitives.

Operators that act along a ragged dimension have historically materialized a dense padded
tensor, paying ``O(B x max_len)`` memory to run a dense kernel and mask the padding back
out. These primitives do the same work on the packed values directly.
"""

from __future__ import annotations

import torch
from torch import Tensor


def segmented_sort_perm(
    values: Tensor,
    offsets: Tensor,
    batch_idx: Tensor,
    *,
    descending: bool = False,
) -> tuple[Tensor, Tensor]:
    r"""
    Sort each segment of a packed tensor along packed dim 0.

    Two stable global sorts compose into a segmented one: sort by value, then stably sort by
    segment id. Because the second pass is stable it regroups by segment while preserving the
    ordering the first pass established, so segments come out contiguous and internally sorted.

    Args:
        values: Packed values, sorted along dim 0. A static tail is sorted per column.
        offsets: Segment offsets, length ``batch + 1``.
        batch_idx: Segment id of every packed row.
        descending: Sort each segment in descending order.

    Returns:
        tuple[Tensor, Tensor]: The permutation into ``values``, and the same permutation
        expressed as per-segment indices.
    """
    if values.dim() > 1:
        segments = batch_idx.view(-1, *([1] * (values.dim() - 1))).expand_as(values)
    else:
        segments = batch_idx
    by_value = torch.argsort(values, dim=0, stable=True, descending=descending)
    by_segment = torch.argsort(torch.gather(segments, 0, by_value), dim=0, stable=True)
    perm = torch.gather(by_value, 0, by_segment)
    return perm, perm - offsets.to(perm.device)[torch.gather(segments, 0, perm)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_segmented.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Format, lint and commit**

```bash
/opt/conda/bin/python -m black danling/tensors/segmented.py tests/tensors/test_segmented.py
/opt/conda/bin/python -m isort danling/tensors/segmented.py tests/tensors/test_segmented.py
/opt/conda/bin/python -m flake8 danling/tensors/segmented.py tests/tensors/test_segmented.py
git add danling/tensors/segmented.py tests/tensors/test_segmented.py
git commit -F - <<'EOF'
NestedTensor: add a packed segmented sort permutation

Sorting along a ragged dimension materialized a dense padded tensor to
run a dense kernel on. Two stable global sorts compose into a segmented
one instead: sort by value, then stably sort by segment id, which
regroups by segment while preserving the ordering the first pass
established.

This runs on the packed values, so it costs no padding, and it has no
data-dependent control flow, so unlike the padded path it is compile
safe.

Signed-off-by: Zhiyuan Chen <this@zyc.ai>
EOF
```

---

### Task 2: Route sort and argsort through the primitive

**Files:**
- Modify: `danling/tensors/aten_functions.py` — the `dim_adj == 0` branch of `sort`
- Test: `tests/tensors/test_torch_functions.py`

**Interfaces:**
- Consumes: `segmented_sort_perm` from Task 1.
- Produces: nothing new. `torch.sort` and `torch.argsort` on the ragged dim keep their existing signatures and return types.

- [ ] **Step 1: Write the failing test**

Add to `tests/tensors/test_torch_functions.py`, immediately before `class TestStackFunction:`:

```python
class TestSegmentedSort:
    r"""Sorting the ragged dimension runs on packed values."""

    @staticmethod
    def _nested(device, float_dtype):
        return NT([torch.randn(n, device=device, dtype=float_dtype) for n in (3, 5, 2, 1)])

    def test_sort_ragged_dim_matches_per_element(self, device, float_dtype):
        nested = self._nested(device, float_dtype)
        values, indices = torch.sort(nested, dim=1)
        assert_close(values, NT([torch.sort(e).values for e in nested], **nested._meta()))
        assert_close(indices, NT([torch.sort(e).indices for e in nested], **nested._meta()))

    def test_argsort_ragged_dim_matches_per_element(self, device, float_dtype):
        nested = self._nested(device, float_dtype)
        assert_close(
            torch.argsort(nested, dim=1),
            NT([torch.argsort(e, stable=True) for e in nested], **nested._meta()),
        )

    def test_descending(self, device, float_dtype):
        nested = self._nested(device, float_dtype)
        assert_close(
            torch.sort(nested, dim=1, descending=True).values,
            NT([torch.sort(e, descending=True).values for e in nested], **nested._meta()),
        )

    def test_does_not_materialize_padding(self, device, float_dtype):
        nested = self._nested(device, float_dtype)
        with nested_execution_guard(forbid_padded_materialization=True):
            torch.sort(nested, dim=1)
            torch.argsort(nested, dim=1)

    def test_compiles_once_across_length_distributions(self, device, float_dtype):
        from torch._dynamo.testing import CompileCounter

        counter = CompileCounter()
        compiled = torch.compile(
            lambda t: torch.sort(t, dim=1).values.concat, backend=counter, fullgraph=True, dynamic=True
        )
        for lengths in ((3, 5, 2), (1, 4, 6)):
            compiled(NT([torch.randn(n, device=device, dtype=float_dtype) for n in lengths]))
        assert counter.frame_count == 1
```

`nested_execution_guard` is already imported at the top of this file; confirm with
`grep -n "nested_execution_guard" tests/tensors/test_torch_functions.py` and add
`from danling.tensors.ops import nested_execution_guard` if missing.

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_torch_functions.py -k TestSegmentedSort -v`
Expected: `test_does_not_materialize_padding` FAILS with "NestedTensor hot path unexpectedly materialized padded storage", and `test_compiles_once_across_length_distributions` FAILS because the current path calls `_compile_unsupported`.

- [ ] **Step 3: Write minimal implementation**

In `danling/tensors/aten_functions.py`, replace the `dim_adj == 0` branch of `sort`:

```python
    if dim_adj == 0:
        if _is_compiling():
            _compile_unsupported("aten.sort.default", "ragged-dimension sort is eager-only under compile")
        fill_value = _topk_fill_value(source._values.dtype, largest=descending)
        padded, _, _, batch_idx, local_idx, _ = _packed_to_padded(source, fill_value=fill_value)
        vals, idxs = _call_sort(padded, 1)
        return source._packed_like_unchecked(vals[batch_idx, local_idx]), source._packed_like_unchecked(
            idxs[batch_idx, local_idx]
        )
```

with:

```python
    if dim_adj == 0:
        # Sort the packed values in place of a padded rectangle. The segmented permutation is
        # stable, which is what the ``stable=True`` overload asks for and is harmless otherwise.
        from .segmented import segmented_sort_perm

        perm, local = segmented_sort_perm(
            source._values,
            source._offsets,
            source.packed_batch_indices(),
            descending=descending,
        )
        return source._packed_like_unchecked(source._values[perm]), source._packed_like_unchecked(local)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_torch_functions.py -k "TestSegmentedSort or sort or argsort" -v`
Expected: PASS

Then run the whole suite to check for regressions:

Run: `/opt/conda/bin/python -m pytest tests/tensors -q --deselect tests/tensors/test_torch_functions.py::TestMatrixMultiplication::test_medium_matrix_compile`
Expected: no new failures against the baseline of 0 failures.

- [ ] **Step 5: Format, lint and commit**

```bash
/opt/conda/bin/python -m black danling/tensors/aten_functions.py tests/tensors/test_torch_functions.py
/opt/conda/bin/python -m flake8 danling/tensors/aten_functions.py tests/tensors/test_torch_functions.py
git add danling/tensors/aten_functions.py tests/tensors/test_torch_functions.py
git commit -F - <<'EOF'
NestedTensor: sort the ragged dimension without padding

Sorting along the ragged dim built a padded rectangle, filled it with
sentinels chosen to sort to the end, ran a dense kernel and gathered the
result back. That costs `O(B x max_len)` for `O(total)` of data, and the
path was marked compile-unsupported because of it.

Route it through the segmented sort permutation instead. The indices it
returns are already per-segment, which is what the operator reports, so
the padded round trip disappears entirely and the ragged sort becomes
compile safe.

Signed-off-by: Zhiyuan Chen <this@zyc.ai>
EOF
```

---

### Task 3: Route topk through the primitive

**Files:**
- Modify: `danling/tensors/aten_functions.py` — the ragged-dim branch of `topk`
- Test: `tests/tensors/test_torch_functions.py`

**Interfaces:**
- Consumes: `segmented_sort_perm` from Task 1.
- Produces: nothing new.

`torch.topk` on a ragged dim has a wrinkle the sort does not: `k` may exceed some segments' lengths. The dense operator errors when `k` exceeds the dimension. Per segment that is ambiguous, so this task keeps the existing behaviour of erroring when `k` exceeds the shortest segment, and the test pins that.

- [ ] **Step 1: Write the failing test**

Add to `tests/tensors/test_torch_functions.py` inside `class TestSegmentedSort`:

```python
    def test_topk_ragged_dim_matches_per_element(self, device, float_dtype):
        nested = NT([torch.randn(n, device=device, dtype=float_dtype) for n in (3, 5, 4)])
        values, indices = torch.topk(nested, 2, dim=1)
        assert_close(values, NT([torch.topk(e, 2).values for e in nested], **nested._meta()))
        assert_close(indices, NT([torch.topk(e, 2).indices for e in nested], **nested._meta()))

    def test_topk_smallest(self, device, float_dtype):
        nested = NT([torch.randn(n, device=device, dtype=float_dtype) for n in (3, 5, 4)])
        assert_close(
            torch.topk(nested, 2, dim=1, largest=False).values,
            NT([torch.topk(e, 2, largest=False).values for e in nested], **nested._meta()),
        )

    def test_topk_rejects_k_above_shortest_segment(self, device, float_dtype):
        nested = NT([torch.randn(n, device=device, dtype=float_dtype) for n in (3, 1)])
        with pytest.raises(RuntimeError):
            torch.topk(nested, 2, dim=1)

    def test_topk_does_not_materialize_padding(self, device, float_dtype):
        nested = NT([torch.randn(n, device=device, dtype=float_dtype) for n in (3, 5, 4)])
        with nested_execution_guard(forbid_padded_materialization=True):
            torch.topk(nested, 2, dim=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_torch_functions.py -k "topk" -v`
Expected: `test_topk_does_not_materialize_padding` FAILS with the padded-storage guard error.

- [ ] **Step 3: Write minimal implementation**

In `danling/tensors/aten_functions.py`, in `topk`, replace the `dim_adj == 0` branch (the
one calling `_packed_to_padded`) with:

```python
    if dim_adj == 0:
        # Take the first k of each sorted segment rather than padding to a rectangle. k is
        # checked against the shortest segment because a per-segment k has no dense meaning.
        from .segmented import segmented_sort_perm

        lengths = source._offsets[1:] - source._offsets[:-1]
        shortest = int(lengths.min())
        if k > shortest:
            raise RuntimeError(f"selected index k out of range: k={k} exceeds shortest segment {shortest}")
        batch_idx = source.packed_batch_indices()
        perm, local = segmented_sort_perm(source._values, source._offsets, batch_idx, descending=largest)
        positions = torch.arange(perm.shape[0], device=perm.device) - source._offsets.to(perm.device)[batch_idx]
        keep = positions < k
        # Every output segment now has the uniform width k, so the result is a plain dense
        # stack rather than anything ragged, and the ordinary constructor is enough.
        values = source._values[perm][keep].view(len(source), k)
        indices = local[keep].view(len(source), k)
        return (
            type(source)(list(values), **source._meta()),
            type(source)(list(indices), **source._meta()),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_torch_functions.py -k "topk" -v`
Expected: PASS

Run: `/opt/conda/bin/python -m pytest tests/tensors -q --deselect tests/tensors/test_torch_functions.py::TestMatrixMultiplication::test_medium_matrix_compile`
Expected: no new failures.

- [ ] **Step 5: Format, lint and commit**

```bash
/opt/conda/bin/python -m black danling/tensors/aten_functions.py tests/tensors/test_torch_functions.py
/opt/conda/bin/python -m flake8 danling/tensors/aten_functions.py tests/tensors/test_torch_functions.py
git add danling/tensors/aten_functions.py tests/tensors/test_torch_functions.py
git commit -F - <<'EOF'
NestedTensor: take topk of the ragged dimension without padding

The ragged-dim branch padded to a rectangle filled with sentinels, ran a
dense topk and gathered back. Sort the segments packed instead and keep
the first k of each.

k is validated against the shortest segment: a k larger than some
segments has no dense equivalent, so it is refused rather than silently
returning ragged output of varying width.

Signed-off-by: Zhiyuan Chen <this@zyc.ai>
EOF
```

---

### Task 4: The segmented scan primitives

**Files:**
- Modify: `danling/tensors/segmented.py`
- Test: `tests/tensors/test_segmented.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `segmented_scan(values: Tensor, batch_idx: Tensor, combine: Callable[[Tensor, Tensor], Tensor]) -> Tensor` — inclusive scan of `combine` within each segment.
  - `segmented_arg_scan(values: Tensor, batch_idx: Tensor, local_idx: Tensor, *, largest: bool) -> tuple[Tensor, Tensor]` — running extremum and the per-segment index that produced it.

- [ ] **Step 1: Write the failing test**

Add to `tests/tensors/test_segmented.py`:

```python
from danling.tensors.segmented import segmented_arg_scan, segmented_scan


class TestSegmentedScan:

    @staticmethod
    def _case():
        nested = _lengths_case(lambda n: (n,))
        batch = nested.packed_batch_indices()
        local = torch.arange(nested._values.numel()) - nested._offsets[batch]
        return nested, batch, local

    def test_cumprod(self):
        nested, batch, _ = self._case()
        output = segmented_scan(nested._values, batch, torch.mul)
        reference = torch.cat([torch.cumprod(element, 0) for element in nested])
        torch.testing.assert_close(output, reference)

    def test_logcumsumexp(self):
        nested, batch, _ = self._case()
        output = segmented_scan(nested._values, batch, torch.logaddexp)
        reference = torch.cat([torch.logcumsumexp(element, 0) for element in nested])
        torch.testing.assert_close(output, reference)

    def test_arg_scan_matches_cummax_values_and_indices(self):
        nested, batch, local = self._case()
        values, indices = segmented_arg_scan(nested._values, batch, local, largest=True)
        torch.testing.assert_close(values, torch.cat([torch.cummax(e, 0).values for e in nested]))
        assert torch.equal(indices, torch.cat([torch.cummax(e, 0).indices for e in nested]))

    def test_arg_scan_matches_cummin_values_and_indices(self):
        nested, batch, local = self._case()
        values, indices = segmented_arg_scan(nested._values, batch, local, largest=False)
        torch.testing.assert_close(values, torch.cat([torch.cummin(e, 0).values for e in nested]))
        assert torch.equal(indices, torch.cat([torch.cummin(e, 0).indices for e in nested]))

    def test_empty_batch(self):
        empty = torch.zeros(0)
        assert segmented_scan(empty, torch.zeros(0, dtype=torch.long), torch.mul).numel() == 0

    def test_does_not_materialize_padding(self):
        nested, batch, local = self._case()
        with nested_execution_guard(forbid_padded_materialization=True):
            segmented_scan(nested._values, batch, torch.mul)
            segmented_arg_scan(nested._values, batch, local, largest=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_segmented.py -k TestSegmentedScan -v`
Expected: FAIL — `ImportError: cannot import name 'segmented_scan'`

- [ ] **Step 3: Write minimal implementation**

Append to `danling/tensors/segmented.py`:

```python
def _shift_down(values: Tensor, step: int) -> Tensor:
    r"""Return ``values`` shifted ``step`` rows later, zero-filled at the front."""
    shifted = torch.zeros_like(values)
    shifted[step:] = values[:-step]
    return shifted


def _same_segment(batch_idx: Tensor, step: int, rank: int) -> Tensor:
    r"""Mask rows whose partner ``step`` back belongs to the same segment."""
    same = torch.zeros(batch_idx.shape[0], dtype=torch.bool, device=batch_idx.device)
    same[step:] = batch_idx[step:] == batch_idx[:-step]
    return same.view(-1, *([1] * (rank - 1)))


def segmented_scan(values: Tensor, batch_idx: Tensor, combine) -> Tensor:
    r"""
    Inclusive scan of ``combine`` within each segment.

    A log-step Hillis-Steele scan, masked so a row only combines with a partner in its own
    segment. The mask is what makes this a segmented scan rather than a global one, and it is
    needed because the trick ``cumsum`` uses — scan globally, then subtract the running total
    at each segment start — requires an inverse. ``cummax`` has none and ``cumprod``'s is
    division, which is unsafe across zeros.
    """
    total = values.shape[0]
    if total == 0:
        return values.clone()
    result = values.clone()
    step = 1
    while step < total:
        mask = _same_segment(batch_idx, step, result.dim())
        result = torch.where(mask, combine(result, _shift_down(result, step)), result)
        step *= 2
    return result


def segmented_arg_scan(
    values: Tensor,
    batch_idx: Tensor,
    local_idx: Tensor,
    *,
    largest: bool,
) -> tuple[Tensor, Tensor]:
    r"""
    Running extremum within each segment, with the per-segment index that produced it.

    The same masked log-step scan as :func:`segmented_scan`, carrying the index alongside the
    value so it follows whichever operand wins. A strict comparison keeps the earliest index on
    ties, which is what ``cummax`` and ``cummin`` report.
    """
    total = values.shape[0]
    if total == 0:
        return values.clone(), local_idx.clone()
    running, indices = values.clone(), local_idx.clone()
    step = 1
    while step < total:
        candidate, candidate_idx = _shift_down(running, step), _shift_down(indices, step)
        better = candidate > running if largest else candidate < running
        take = _same_segment(batch_idx, step, running.dim()) & better
        running = torch.where(take, candidate, running)
        indices = torch.where(take, candidate_idx, indices)
        step *= 2
    return running, indices
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_segmented.py -v`
Expected: PASS (all tests including Task 1's)

- [ ] **Step 5: Format, lint and commit**

```bash
/opt/conda/bin/python -m black danling/tensors/segmented.py tests/tensors/test_segmented.py
/opt/conda/bin/python -m flake8 danling/tensors/segmented.py tests/tensors/test_segmented.py
git add danling/tensors/segmented.py tests/tensors/test_segmented.py
git commit -F - <<'EOF'
NestedTensor: add a packed segmented scan

A masked log-step Hillis-Steele scan, so a row only ever combines with a
partner in its own segment.

A segmented scan is needed rather than the trick `cumsum` uses. `cumsum`
can scan globally and subtract the running total at each segment start
because addition has an inverse. `cummax` and `cummin` have none, and
`cumprod`'s is division, which is unsafe across zeros.

The arg variant carries the index alongside the value so it follows
whichever operand wins, with a strict comparison so ties keep the
earliest index, matching what `cummax` and `cummin` report.

Signed-off-by: Zhiyuan Chen <this@zyc.ai>
EOF
```

---

### Task 5: Route cumprod and logcumsumexp through the scan

**Files:**
- Modify: `danling/tensors/aten_functions.py:4437` — the `cumulative` handler
- Test: `tests/tensors/test_torch_functions.py`

**Interfaces:**
- Consumes: `segmented_scan` from Task 4.
- Produces: nothing new.

- [ ] **Step 1: Write the failing test**

Add to `tests/tensors/test_torch_functions.py`, before `class TestStackFunction:`:

```python
class TestSegmentedCumulative:

    @staticmethod
    def _nested(device, float_dtype):
        return NT([torch.randn(n, device=device, dtype=float_dtype) for n in (3, 5, 2, 1)])

    def test_cumprod_matches_per_element(self, device, float_dtype):
        nested = self._nested(device, float_dtype)
        assert_close(
            torch.cumprod(nested, dim=1),
            NT([torch.cumprod(e, 0) for e in nested], **nested._meta()),
        )

    def test_logcumsumexp_matches_per_element(self, device, float_dtype):
        nested = self._nested(device, float_dtype)
        assert_close(
            torch.logcumsumexp(nested, dim=1),
            NT([torch.logcumsumexp(e, 0) for e in nested], **nested._meta()),
        )

    def test_does_not_materialize_padding(self, device, float_dtype):
        nested = self._nested(device, float_dtype)
        with nested_execution_guard(forbid_padded_materialization=True):
            torch.cumprod(nested, dim=1)
            torch.logcumsumexp(nested, dim=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_torch_functions.py -k TestSegmentedCumulative -v`
Expected: `test_does_not_materialize_padding` FAILS with the padded-storage guard error.

- [ ] **Step 3: Write minimal implementation**

`cumprod` and `logcumsumexp` are handled by `cumulative` in `danling/tensors/aten_functions.py`,
registered on the aten ops rather than the torch functions. `cumsum` already has a packed path
there, `_segmented_cumsum_values`, which returns before this branch; only `cumprod` and
`logcumsumexp` reach it.

Replace the padded block inside `if dim_adj == 0:` — everything from the `_is_compiling()`
guard through the `return` that indexes `out_padded` — with:

```python
        # cumsum returned above through its own packed path; only cumprod and logcumsumexp
        # arrive here. Neither has a usable inverse, so they need a real segmented scan rather
        # than the global-scan-and-correct trick cumsum uses.
        from .segmented import segmented_scan

        combine = torch.mul if func is aten.cumprod.default else torch.logaddexp
        return source._packed_like_unchecked(
            segmented_scan(source._values, source.packed_batch_indices(), combine)
        )
```

Delete the now-unreachable `neutral` computation with it. Leave the `func is aten.cumsum.default`
early return above untouched.

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_torch_functions.py -k "TestSegmentedCumulative or cumprod or logcumsumexp" -v`
Expected: PASS

Run: `/opt/conda/bin/python -m pytest tests/tensors -q --deselect tests/tensors/test_torch_functions.py::TestMatrixMultiplication::test_medium_matrix_compile`
Expected: no new failures.

- [ ] **Step 5: Format, lint and commit**

```bash
/opt/conda/bin/python -m black danling/tensors/torch_functions.py tests/tensors/test_torch_functions.py
/opt/conda/bin/python -m flake8 danling/tensors/torch_functions.py tests/tensors/test_torch_functions.py
git add danling/tensors/torch_functions.py tests/tensors/test_torch_functions.py
git commit -F - <<'EOF'
NestedTensor: scan the ragged dimension without padding

`cumprod` and `logcumsumexp` built a padded rectangle along the ragged
dim, even though `cumsum` beside them already ran packed. Route them
through the segmented scan.

Signed-off-by: Zhiyuan Chen <this@zyc.ai>
EOF
```

---

### Task 6: Route cummax and cummin through the arg scan

**Files:**
- Modify: `danling/tensors/aten_functions.py:4495` — the `cumulative_pair` handler
- Test: `tests/tensors/test_torch_functions.py`

**Interfaces:**
- Consumes: `segmented_arg_scan` from Task 4.
- Produces: nothing new. `cumulative_pair` is an aten handler, so it returns a plain
  `(values, indices)` tuple, not a `torch.return_types` object.

- [ ] **Step 1: Write the failing test**

Add to `class TestSegmentedCumulative` in `tests/tensors/test_torch_functions.py`:

```python
    def test_cummax_matches_values_and_indices(self, device, float_dtype):
        nested = self._nested(device, float_dtype)
        output = torch.cummax(nested, dim=1)
        assert_close(output.values, NT([torch.cummax(e, 0).values for e in nested], **nested._meta()))
        assert_close(output.indices, NT([torch.cummax(e, 0).indices for e in nested], **nested._meta()))

    def test_cummin_matches_values_and_indices(self, device, float_dtype):
        nested = self._nested(device, float_dtype)
        output = torch.cummin(nested, dim=1)
        assert_close(output.values, NT([torch.cummin(e, 0).values for e in nested], **nested._meta()))
        assert_close(output.indices, NT([torch.cummin(e, 0).indices for e in nested], **nested._meta()))

    def test_cummax_keeps_earliest_index_on_ties(self, device, float_dtype):
        nested = NT([torch.zeros(4, device=device, dtype=float_dtype)])
        assert_close(
            torch.cummax(nested, dim=1).indices,
            NT([torch.cummax(e, 0).indices for e in nested], **nested._meta()),
        )

    def test_extrema_do_not_materialize_padding(self, device, float_dtype):
        nested = self._nested(device, float_dtype)
        with nested_execution_guard(forbid_padded_materialization=True):
            torch.cummax(nested, dim=1)
            torch.cummin(nested, dim=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_torch_functions.py -k "cummax or cummin" -v`
Expected: `test_extrema_do_not_materialize_padding` FAILS with the padded-storage guard error.

- [ ] **Step 3: Write minimal implementation**

In `danling/tensors/aten_functions.py`, in `cumulative_pair`, replace the padded block inside
`if dim_adj == 0:` — everything from the `_is_compiling()` guard through the `return` that
indexes `vals`/`idxs` — with:

```python
        from .segmented import segmented_arg_scan

        largest = func is aten.cummax.default
        batch_idx = source.packed_batch_indices()
        offsets = source._offsets.to(device=source._values.device, dtype=torch.long)
        local_idx = torch.arange(source._values.shape[0], device=source._values.device) - offsets[batch_idx]
        values, indices = segmented_arg_scan(source._values, batch_idx, local_idx, largest=largest)
        return source._packed_like_unchecked(values), source._packed_like_unchecked(indices)
```

Keep returning a bare tuple: this is an aten handler, and the surrounding code at the top of the
function already returns `source._packed_like_unchecked(vals), source._packed_like_unchecked(idxs)`
for the static-dim case.

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_torch_functions.py -k "cummax or cummin" -v`
Expected: PASS

Run: `/opt/conda/bin/python -m pytest tests/tensors -q --deselect tests/tensors/test_torch_functions.py::TestMatrixMultiplication::test_medium_matrix_compile`
Expected: no new failures.

- [ ] **Step 5: Format, lint and commit**

```bash
/opt/conda/bin/python -m black danling/tensors/torch_functions.py tests/tensors/test_torch_functions.py
/opt/conda/bin/python -m flake8 danling/tensors/torch_functions.py tests/tensors/test_torch_functions.py
git add danling/tensors/torch_functions.py tests/tensors/test_torch_functions.py
git commit -F - <<'EOF'
NestedTensor: accumulate ragged extrema without padding

`cummax` and `cummin` padded the ragged dim to a rectangle. Route them
through the segmented arg scan, which carries the index alongside the
running extremum so both come out of one pass.

Signed-off-by: Zhiyuan Chen <this@zyc.ai>
EOF
```

---

### Task 7: Sample-local inverse permutation

**Files:**
- Modify: `danling/tensors/torch_functions.py`
- Test: `tests/tensors/test_torch_functions.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `inverse_permutation(input: NestedTensor, dim: int = -1) -> NestedTensor`, exported from `danling.tensors`. Given a per-sample permutation it returns the permutation that undoes it.

- [ ] **Step 1: Write the failing test**

Add to `tests/tensors/test_torch_functions.py`, before `class TestStackFunction:`:

```python
class TestInversePermutation:

    def test_inverts_each_sample(self, device, float_dtype):
        from danling.tensors import inverse_permutation

        perm = NT([torch.randperm(n, device=device) for n in (3, 5, 2, 1)])
        inverse = inverse_permutation(perm, dim=1)
        assert_close(inverse, NT([torch.argsort(e) for e in perm], **perm._meta()))

    def test_round_trip_restores_order(self, device, float_dtype):
        from danling.tensors import inverse_permutation

        perm = NT([torch.randperm(n, device=device) for n in (3, 5, 2)])
        values = NT([torch.randn(n, device=device, dtype=float_dtype) for n in (3, 5, 2)])
        inverse = inverse_permutation(perm, dim=1)
        for element, permutation, undo in zip(values, perm, inverse):
            assert torch.allclose(element[permutation][undo], element)

    def test_does_not_materialize_padding(self, device, float_dtype):
        from danling.tensors import inverse_permutation

        perm = NT([torch.randperm(n, device=device) for n in (3, 5, 2)])
        with nested_execution_guard(forbid_padded_materialization=True):
            inverse_permutation(perm, dim=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_torch_functions.py -k TestInversePermutation -v`
Expected: FAIL — `ImportError: cannot import name 'inverse_permutation'`

- [ ] **Step 3: Write minimal implementation**

Add to `danling/tensors/torch_functions.py`, near `gather`:

```python
def inverse_permutation(input: NestedTensor, dim: int = -1) -> NestedTensor:
    r"""
    Invert a per-sample permutation.

    Given a NestedTensor whose every element is a permutation of ``range(n_i)``, return the
    permutation that undoes it, so ``values[perm][inverse]`` is ``values`` again.

    This is one scatter. Writing it as ``argsort(argsort(x))`` costs two sorts for the same
    result, which is why it is worth its own operator.

    Args:
        input: Per-sample permutations, one per element.
        dim: The ragged dimension. Only the ragged dimension is meaningful here.

    Returns:
        NestedTensor: The inverse permutations, with the input's structure.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor, inverse_permutation
        >>> perm = NestedTensor(torch.tensor([2, 0, 1]), torch.tensor([1, 0]))
        >>> [element.tolist() for element in inverse_permutation(perm, dim=1)]
        [[1, 2, 0], [1, 0]]
    """
    if _translate_dim(input, dim) != 0:
        raise ValueError("inverse_permutation applies to the ragged dimension")
    batch_idx = input.packed_batch_indices()
    base = input._offsets.to(batch_idx.device)[batch_idx]
    positions = torch.arange(input._values.shape[0], device=input._values.device) - base
    inverse = torch.empty_like(positions)
    inverse.scatter_(0, input._values.long() + base, positions)
    return input._packed_like_unchecked(inverse.to(input._values.dtype))
```

Export it from `danling/tensors/__init__.py` alongside `nested_execution_guard`, and add it to
`__all__`. Check the existing export block first:

```bash
grep -n "nested_execution_guard" danling/tensors/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_torch_functions.py -k TestInversePermutation -v`
Expected: PASS

Verify the doctest too:

Run: `/opt/conda/bin/python -m pytest --doctest-modules danling/tensors/torch_functions.py -k inverse_permutation`
Expected: PASS

- [ ] **Step 5: Format, lint and commit**

```bash
/opt/conda/bin/python -m black danling/tensors/torch_functions.py danling/tensors/__init__.py tests/tensors/test_torch_functions.py
/opt/conda/bin/python -m flake8 danling/tensors/torch_functions.py danling/tensors/__init__.py tests/tensors/test_torch_functions.py
git add danling/tensors/torch_functions.py danling/tensors/__init__.py tests/tensors/test_torch_functions.py
git commit -F - <<'EOF'
NestedTensor: add a sample-local inverse permutation

Inverting a per-sample permutation had no spelling of its own, so it was
written as `argsort(argsort(x))`: two sorts for what one scatter does.

`inverse_permutation` scatters the per-segment positions to the places
the permutation names, in one pass over the packed values.

Signed-off-by: Zhiyuan Chen <this@zyc.ai>
EOF
```

---

### Task 8: Sample-local rank

**Files:**
- Modify: `danling/tensors/torch_functions.py`
- Test: `tests/tensors/test_torch_functions.py`

**Interfaces:**
- Consumes: `segmented_sort_perm` from Task 1, `inverse_permutation` from Task 7.
- Produces: `rank(input: NestedTensor, dim: int = -1, *, descending: bool = False) -> NestedTensor`, exported from `danling.tensors`. Each element's position within its own sample.

- [ ] **Step 1: Write the failing test**

Add to `tests/tensors/test_torch_functions.py`:

```python
class TestRank:

    def test_rank_matches_argsort_of_argsort(self, device, float_dtype):
        from danling.tensors import rank

        nested = NT([torch.randn(n, device=device, dtype=float_dtype) for n in (3, 5, 2, 1)])
        assert_close(
            rank(nested, dim=1),
            NT([torch.argsort(torch.argsort(e, stable=True)) for e in nested], **nested._meta()),
        )

    def test_rank_descending(self, device, float_dtype):
        from danling.tensors import rank

        nested = NT([torch.randn(n, device=device, dtype=float_dtype) for n in (3, 5, 2)])
        assert_close(
            rank(nested, dim=1, descending=True),
            NT(
                [torch.argsort(torch.argsort(e, stable=True, descending=True)) for e in nested],
                **nested._meta(),
            ),
        )

    def test_rank_is_a_permutation_per_sample(self, device, float_dtype):
        from danling.tensors import rank

        nested = NT([torch.randn(n, device=device, dtype=float_dtype) for n in (3, 5, 2)])
        for element in rank(nested, dim=1):
            assert torch.equal(torch.sort(element).values, torch.arange(element.numel()))

    def test_does_not_materialize_padding(self, device, float_dtype):
        from danling.tensors import rank

        nested = NT([torch.randn(n, device=device, dtype=float_dtype) for n in (3, 5, 2)])
        with nested_execution_guard(forbid_padded_materialization=True):
            rank(nested, dim=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_torch_functions.py -k TestRank -v`
Expected: FAIL — `ImportError: cannot import name 'rank'`

- [ ] **Step 3: Write minimal implementation**

Add to `danling/tensors/torch_functions.py`, next to `inverse_permutation`:

```python
def rank(input: NestedTensor, dim: int = -1, *, descending: bool = False) -> NestedTensor:
    r"""
    Position of each element within its own sample.

    The inverse of the sample-local argsort: where ``argsort`` answers "which element belongs
    at position i", ``rank`` answers "at which position does element i belong". Rank validation
    asks the second question directly, which is why it gets its own operator.

    Args:
        input: The values to rank.
        dim: The ragged dimension.
        descending: Rank largest first.

    Returns:
        NestedTensor: Per-sample ranks, a permutation of ``range(n_i)`` for every element.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor, rank
        >>> nested = NestedTensor(torch.tensor([3.0, 1.0, 2.0]), torch.tensor([5.0, 4.0]))
        >>> [element.tolist() for element in rank(nested, dim=1)]
        [[2, 0, 1], [1, 0]]
    """
    from .segmented import segmented_sort_perm

    if _translate_dim(input, dim) != 0:
        raise ValueError("rank applies to the ragged dimension")
    _, local = segmented_sort_perm(
        input._values,
        input._offsets,
        input.packed_batch_indices(),
        descending=descending,
    )
    return inverse_permutation(input._packed_like_unchecked(local), dim=dim)
```

Export `rank` from `danling/tensors/__init__.py` and add it to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_torch_functions.py -k TestRank -v`
Expected: PASS

Run: `/opt/conda/bin/python -m pytest --doctest-modules danling/tensors/torch_functions.py -k rank`
Expected: PASS

Run the whole suite:

Run: `/opt/conda/bin/python -m pytest tests/tensors -q --deselect tests/tensors/test_torch_functions.py::TestMatrixMultiplication::test_medium_matrix_compile`
Expected: no failures.

- [ ] **Step 5: Format, lint and commit**

```bash
/opt/conda/bin/python -m black danling/tensors/torch_functions.py danling/tensors/__init__.py tests/tensors/test_torch_functions.py
/opt/conda/bin/python -m flake8 danling/tensors/torch_functions.py danling/tensors/__init__.py tests/tensors/test_torch_functions.py
git add danling/tensors/torch_functions.py danling/tensors/__init__.py tests/tensors/test_torch_functions.py
git commit -F - <<'EOF'
NestedTensor: add a sample-local rank

`argsort` answers which element belongs at position i; rank validation
asks the opposite, at which position element i belongs. Composing the
segmented sort permutation with its inverse answers that directly,
rather than making every caller write the composition themselves.

Signed-off-by: Zhiyuan Chen <this@zyc.ai>
EOF
```

---

### Task 9: Sample-local cumulative count

**Files:**
- Modify: `danling/tensors/torch_functions.py`
- Test: `tests/tensors/test_torch_functions.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `cumcount(input: NestedTensor, dim: int = -1) -> NestedTensor`, exported from
  `danling.tensors`. Each element's ordinal within its own sample, i.e. a per-sample
  `arange`.

There is no torch spelling for this, which is why it gets an operator rather than being
expressed in terms of one. It is the per-segment position that every packed handler already
computes internally; exposing it saves callers from re-deriving offsets by hand.

`argrank` from the spec is deliberately **not** added: it is what `argsort` already is, so a
second name for it would be redundant.

- [ ] **Step 1: Write the failing test**

Add to `tests/tensors/test_torch_functions.py`, before `class TestStackFunction:`:

```python
class TestCumcount:

    def test_counts_within_each_sample(self, device, float_dtype):
        from danling.tensors import cumcount

        nested = NT([torch.randn(n, device=device, dtype=float_dtype) for n in (3, 5, 2, 1)])
        assert_close(
            cumcount(nested, dim=1),
            NT([torch.arange(e.numel(), device=device) for e in nested], **nested._meta()),
        )

    def test_single_sample(self, device, float_dtype):
        from danling.tensors import cumcount

        nested = NT([torch.randn(4, device=device, dtype=float_dtype)])
        assert torch.equal(cumcount(nested, dim=1)[0], torch.arange(4, device=device))

    def test_does_not_materialize_padding(self, device, float_dtype):
        from danling.tensors import cumcount

        nested = NT([torch.randn(n, device=device, dtype=float_dtype) for n in (3, 5, 2)])
        with nested_execution_guard(forbid_padded_materialization=True):
            cumcount(nested, dim=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_torch_functions.py -k TestCumcount -v`
Expected: FAIL — `ImportError: cannot import name 'cumcount'`

- [ ] **Step 3: Write minimal implementation**

Add to `danling/tensors/torch_functions.py`, next to `inverse_permutation`:

```python
def cumcount(input: NestedTensor, dim: int = -1) -> NestedTensor:
    r"""
    Ordinal of each element within its own sample.

    A per-sample ``arange``: the position every packed handler derives internally from the
    offsets. There is no torch spelling for it, so callers would otherwise re-derive the
    offsets by hand.

    Args:
        input: Any NestedTensor; only its layout is read.
        dim: The ragged dimension.

    Returns:
        NestedTensor: Per-sample positions, with the input's structure and an integer dtype.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor, cumcount
        >>> nested = NestedTensor(torch.tensor([9.0, 8.0, 7.0]), torch.tensor([6.0, 5.0]))
        >>> [element.tolist() for element in cumcount(nested, dim=1)]
        [[0, 1, 2], [0, 1]]
    """
    if _translate_dim(input, dim) != 0:
        raise ValueError("cumcount applies to the ragged dimension")
    batch_idx = input.packed_batch_indices()
    offsets = input._offsets.to(device=batch_idx.device, dtype=torch.long)
    positions = torch.arange(input._values.shape[0], device=input._values.device) - offsets[batch_idx]
    return input._packed_like_unchecked(positions)
```

Export `cumcount` from `danling/tensors/__init__.py` and add it to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/conda/bin/python -m pytest tests/tensors/test_torch_functions.py -k TestCumcount -v`
Expected: PASS

Run: `/opt/conda/bin/python -m pytest --doctest-modules danling/tensors/torch_functions.py -k cumcount`
Expected: PASS

- [ ] **Step 5: Format, lint and commit**

```bash
/opt/conda/bin/python -m black danling/tensors/torch_functions.py danling/tensors/__init__.py tests/tensors/test_torch_functions.py
/opt/conda/bin/python -m flake8 danling/tensors/torch_functions.py danling/tensors/__init__.py tests/tensors/test_torch_functions.py
git add danling/tensors/torch_functions.py danling/tensors/__init__.py tests/tensors/test_torch_functions.py
git commit -F - <<'EOF'
NestedTensor: add a sample-local cumulative count

The ordinal of each element within its own sample is the position every
packed handler already derives from the offsets, but callers had no way
to ask for it without re-deriving the offsets themselves.

Signed-off-by: Zhiyuan Chen <this@zyc.ai>
EOF
```

---

## Follow-up plans

The spec covers two further groups that are independent of this one and get their own plans:

- **Masked reductions and NN operators** — the third primitive plus `logsumexp`, `nansum`, `nanmean`, `normalize`, `gumbel_softmax`, `local_response_norm`, `ctc_loss`, and the three operand-conditional cases (`masked_fill` with a dense mask, `allclose`/`equal` with a dense other, `batch_norm` on the eval branch).
- **Attention** — the packed CPU path for `scaled_dot_product_attention`, `multi_head_attention_forward`, `_native_multi_head_attention` and `_transformer_encoder_layer_fwd`. This one rests on an unverified assumption: the padding was measured on CPU only, and the flash and flex paths are gated on `is_cuda`. Re-run the audit on a GPU machine before starting it.
