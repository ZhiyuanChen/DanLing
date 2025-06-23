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

from __future__ import annotations

import torch

from danling.tensors import NestedTensor


def available_float_dtypes():
    dtypes = [torch.float32, torch.float64]
    if torch.cuda.is_available():
        dtypes.append(torch.float16)
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
    else:
        try:
            torch.zeros(1, dtype=torch.bfloat16)
        except (TypeError, RuntimeError):
            pass
        else:
            dtypes.append(torch.bfloat16)
    return tuple(dict.fromkeys(dtypes))


FLOAT_DTYPES = available_float_dtypes()


def available_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def nested_rand(shapes, device, dtype):
    return NestedTensor([torch.randn(*shape, device=device, dtype=dtype) for shape in shapes])


def assert_nested_function_matches(fn, nested_tensor, *, atol=1e-6, rtol=1e-6, **kwargs):
    output = fn(nested_tensor, **kwargs)
    reference = fn(nested_tensor.tensor, **kwargs)
    assert_close(output, reference, atol=atol, rtol=rtol)


def assert_close(input, other, *, rtol: float | None = None, atol: float | None = None, equal_nan: bool = False):
    def _assert_tensors_close(a: torch.Tensor, b: torch.Tensor) -> None:
        if rtol is None and atol is None:
            torch.testing.assert_close(a, b, equal_nan=equal_nan)
        else:
            effective_rtol = 1e-05 if rtol is None else rtol
            effective_atol = 1e-08 if atol is None else atol
            torch.testing.assert_close(a, b, rtol=effective_rtol, atol=effective_atol, equal_nan=equal_nan)

    if not isinstance(input, NestedTensor) and not isinstance(other, NestedTensor):
        _assert_tensors_close(input, other)
        return
    if not isinstance(input, NestedTensor):
        input = other.nested_like(input)
    elif not isinstance(other, NestedTensor):
        other = input.nested_like(other)
    if len(input) != len(other):
        raise ValueError(f"NestedTensor batch length mismatch: {len(input)} vs {len(other)}")
    for x, y in zip(input._storage, other._storage):
        _assert_tensors_close(x, y)
