# DanLing
# Copyright (C) 2022-Present DanLing

import pytest
import torch

from danling import NestedTensor


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("lengths", [[2, 3], [2, 2], []])
def test_public_from_packed_retains_shared_values_and_lazy_metadata(monkeypatch, device, lengths):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    counts = torch.tensor(lengths, dtype=torch.long)
    offsets = torch.nn.functional.pad(counts.cumsum(0), (1, 0))
    values = torch.arange(sum(lengths), dtype=torch.int32, device=device)
    shapes = counts[:, None] if lengths else torch.empty((0, 0), dtype=torch.long)

    def reject_python_metadata(*args, **kwargs):
        raise AssertionError("Packed construction must not create per-element Python metadata")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "tolist", reject_python_metadata)
        result = NestedTensor.from_packed(
            values,
            offsets,
            shapes,
            permutation=(0,) if lengths else (),
            ragged_dims=(0,) if lengths else (),
            outer_size=(len(lengths), max(lengths)) if lengths else (0,),
            materialize_python_metadata=False,
            validate=False,
        )

    assert result.concat is values
    assert result.dtype == values.dtype
    assert result.device == values.device
    assert result._packed_sizes is None
    assert result._element_shapes is None
    assert not hasattr(NestedTensor, "_from_packed")
    expected, start = [], 0
    for length in lengths:
        expected.append(list(range(start, start + length)))
        start += length
    assert result.tolist() == expected
