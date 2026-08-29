# Packed segmented operations for NestedTensor

## Problem

`NestedTensor` stores elements packed, but 18 registered operators fall back to
materializing a dense padded tensor when they act on a ragged dimension. Each
such fallback costs `O(B x max_len)` memory instead of `O(total)`, and most are
additionally marked `_compile_unsupported`, so they break a `fullgraph` region.

Three downstream consumers generate and validate autoregressive decoding ranks.
That workload is a per-sample segmented sort followed by a sample-local
inverse-scatter and gather. Today the sort step alone pays the padded blowup and
cannot be compiled.

## Evidence

Every claim below was measured, not read off the source, by calling each
registered operator inside
`nested_execution_guard(forbid_padded_materialization=True)`.

Fixtures: elements of length `(3, 5, 2)`, so `total = 10` against a padded
`3 x 5 = 15`.

### This section understates the problem, in two known ways

Both were found by a later audit and are recorded here because the method,
not the conclusions, is what misled.

**One guard of five.** `nested_execution_guard` takes five independent flags:
`forbid_padded_materialization`, `forbid_iteration`, `forbid_storage_map`,
`forbid_eager_fallback`, `forbid_dense_repack`. Only the first was used. An
operator that loops over samples in Python trips `forbid_storage_map` and sails
past the padding guard, so it was reported clean. Across 54 gaps a later audit
reproduced, the guards actually tripped were `forbid_storage_map` 30 times and
`forbid_eager_fallback` 6, against `forbid_padded_materialization` 4. The one
flag measured here catches the rarest failure.

**One fixture per operator.** `torch.gather` on 1-D ragged elements is a Python
loop over `_storage`; on 2-D elements it calls `_packed_to_padded` twice. A
single fixture reached one branch and the table below called the operator
packed. The audit that found this used element rank 1-D through doubly-ragged,
inferred against explicit `ragged_dims`, non-leading ragged dims,
`batch_first=False`, single-sample, singleton and empty segments.

The table below is therefore a **lower bound on operators that materialize
padding**, and says nothing at all about operators that loop per sample. The
later audit found 54 further gaps in `gather`, `scatter`, `split`/`cat` and
dense-operand broadcast, 22 of which returned numerically wrong results rather
than merely slow ones.

Any future audit against this spec probes all five guards separately, on the
full fixture matrix, and additionally checks `fullgraph` on a cleared
`$TMPDIR/torchinductor_chenzhiyuan` — a stale Inductor cache was observed
serving silently wrong compiled results.

### Confirmed padding materialization

| Family | Operators |
| --- | --- |
| Sort | `sort`, `argsort`, `topk` |
| Cumulative | `cumprod`, `cummax`, `cummin`, `logcumsumexp` |
| Masked reduction | `logsumexp`, `nansum`, `nanmean`, `normalize` |
| NN | `gumbel_softmax`, `local_response_norm`, `ctc_loss` |
| Attention | `scaled_dot_product_attention`, `multi_head_attention_forward`, `_native_multi_head_attention`, `_transformer_encoder_layer_fwd` |

`cumsum` is already packed; the other four cumulative operators are not.

### Conditional on operand type

Packed with a `NestedTensor` operand, padded with a dense one:

- `masked_fill` with a dense mask
- `allclose` and `equal` with a dense other
- `batch_norm` on the eval branch

### Not audited

`scaled_mm`, `scaled_grouped_mm` and `flex_attention` could not be executed on
the audit machine (missing fp8 and CUDA kernels). They are unverified, not
cleared.

### Attention caveat

The four attention entry points select their flash and flex fast paths on
`query._values.is_cuda`. The measured padding is therefore the CPU fallback.
The CUDA path was not verified. This design fixes the CPU path and leaves the
CUDA path to a later GPU-run audit.

## Design

The 18 operators collapse onto three primitives. Each operator becomes a thin
expression over one of them, so the packed logic is written and tested once.

### Primitive 1: segmented sort permutation

```
_segmented_sort_perm(values, offsets, *, descending, stable) -> (perm, local)
```

Two stable global sorts compose into a segmented sort:

```python
p1 = torch.argsort(values, stable=True, descending=descending)
p2 = torch.argsort(batch_idx[p1], stable=True)
perm = p1[p2]
local = perm - offsets[batch_idx[perm]]
```

The second sort is stable, so it regroups by segment while preserving the value
ordering the first pass established. Segments come out contiguous and internally
sorted.

`O(N log N)` on packed values, no padding, and no data-dependent control flow,
so it is compile-safe. This replaces a path that is currently both padded and
`_compile_unsupported`.

Prototyped and verified against a per-element reference, including stability and
singleton segments.

Powers: `sort`, `argsort`, `topk`, group-major reorder, `rank`, `argrank`.

### Primitive 2: segmented scan

```
_segmented_scan(values, offsets, op) -> Tensor
```

A log-step Hillis-Steele scan with a segment-boundary mask: at step `k` a lane
combines with the lane `2^k` positions back only when that lane belongs to the
same segment.

A segmented scan is needed rather than the trick `cumsum` uses. `cumsum` can run
a global scan and subtract the running total at each segment start because
addition has an inverse. `cummax` and `cummin` have none, and `cumprod`'s
inverse is division, which is unsafe across zeros. The masked log-step scan is
`O(N log N)` and correct for any associative `op`.

Powers: `cumprod`, `cummax`, `cummin`, `logcumsumexp`, `cumcount`.

### Primitive 3: segmented masked reduction

```
_segmented_masked_reduce(values, offsets, reduce) -> Tensor
```

Built on `torch.segment_reduce` and `scatter_reduce`, which already accept
offsets and therefore never need the padded rectangle these operators construct
solely to mask it back out.

`logsumexp` decomposes into a segmented max followed by a segmented sum of
exponentials, both available here.

Powers: `logsumexp`, `nansum`, `nanmean`, `normalize`.

### New public operators

The guiding principle is to match torch's surface: no bespoke API where a torch
operator already expresses the semantics.

The four group semantics need no new API.

| Semantic | Spelling |
| --- | --- |
| Group-major reorder | `argsort` on group ids, via primitive 1 |
| Group to row broadcast | `gather` / `index_select`, already packed |
| Filter by group | `masked_select`, already packed |
| Group-local rank | inverse permutation, below |

Only two additions have no torch spelling:

- `inverse_permutation(nt, dim)` - the sample-local inverse of a permutation.
  A single `scatter_` of local positions, `O(N)`. danling currently forces
  `argsort(argsort(x))`, paying two sorts for what one scatter does. Verified by
  round-trip against a per-element reference.
- `rank` / `argrank(nt, dim)` - each element's position within its own sample.
  Expressible as `inverse_permutation(argsort(...))`, exposed directly because
  it is what rank validation actually asks for.

### Attention

The CPU fallback pads because the reference kernels want a rectangle. Replace it
with a segment-masked softmax over packed values, reusing primitive 3 for the
row-wise max and sum. The CUDA fast paths are untouched.

## Testing

Every operator gets a regression test asserting two things:

1. Correctness against a per-element reference built with plain tensors.
2. `nested_execution_guard(forbid_padded_materialization=True)` does not raise.

The second assertion is the one that prevents regressions, and it is why these
tests are worth writing even where behaviour is already correct.

Coverage must include the cases that break naive implementations: singleton
segments, zero-length segments, a single-sample batch, `batch_first=False`,
non-identity permutations, and multi-ragged layouts.

Compile coverage: each rewritten operator gets a `fullgraph=True` test asserting
one graph is reused across two different length distributions, matching the
existing `frame_count == 1` convention.

## Failure modes to test for

An external review of the operators already rewritten in this style found six
defects, none of which a happy-path test would have caught. Every operator added
under this design gets a test for each mode that applies to it.

**Divergence from the dense operator on edge inputs.** The rewritten
`cross_entropy` returned `0` where dense returns `NaN` for an entirely ignored
batch, indexed out of bounds on a sentinel above the class count, and scored an
invalid negative label as class `0` instead of raising. Compare against the
dense operator on empty, all-masked, and out-of-range inputs, not only on
well-formed ones.

**Declared topology not projected.** `unbind` forwarded `ragged_dims` unchanged
onto a reduced rank and raised `ValueError`; `cdist` dropped the declaration and
re-inferred. Any operator that changes rank or structure is tested with an
explicit `ragged_dims`, asserting both the projected value and
`_validate_metadata()`.

**Autograd severed through cached storage.** `_cached_storage` keeps whatever
the first access produced, so a cache filled under `no_grad` held detached
views. Any operator that reads elements rather than `_values` is tested after a
`no_grad` warm-up, asserting the gradient still reaches the packed values.

**Silent dtype coercion.** Batch `repeat_interleave` cast float counts to long,
repeating a truncated number of times without saying so. Operators taking index
or count tensors assert that invalid dtypes are refused, matching the dense
operator, rather than being coerced.

**Device assumptions.** The same cast fixed dtype but not device, so counts
computed on CUDA against CPU offsets mismatched. Operators taking an auxiliary
tensor move it onto the values' device.

**Incomplete argument spellings.** The `new_*` handlers accepted varargs and a
sequence but not the `size=` keyword, because the varargs parameter was itself
named `size`. Operators accept every spelling the dense method does, and the
tests exercise each.

## Out of scope

- The CUDA attention path, pending a GPU audit.
- `scaled_mm`, `scaled_grouped_mm`, `flex_attention`, which could not be audited.

## Open questions

None blocking. The audit script is kept at `scratchpad/audit/` and is
re-runnable on a GPU machine to close the unaudited set.
