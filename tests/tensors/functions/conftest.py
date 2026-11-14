# DanLing
# Copyright (C) 2022-Present DanLing

import pytest
import torch

from danling.tensors import NestedTensor
from tests.tensors.utils import low_precision_cuda_tolerances


@pytest.fixture
def tolerances(device):
    def choose(dtype=torch.float32):
        if dtype == torch.float64:
            return 1e-8, 1e-8
        if device.type != "cuda":
            return 1e-5, 1e-5
        return low_precision_cuda_tolerances(device, dtype, default=(5e-3, 5e-3), fp16=(5e-3, 5e-3), bf16=(5e-2, 5e-2))

    return choose


@pytest.fixture
def check_pool():
    def check(inputs, function, **kwargs):
        references = [value.detach().clone().requires_grad_() for value in inputs]
        actual = function(NestedTensor(inputs), **kwargs)
        expected = [function(value, **kwargs) for value in references]
        if kwargs.get("return_indices", False):
            actual, indices = actual
            expected, expected_indices = zip(*expected)
            torch.testing.assert_close(indices.unbind(), expected_indices)
        torch.testing.assert_close(actual.unbind(), expected)
        actual.values().sum().backward()
        sum(value.sum() for value in expected).backward()
        torch.testing.assert_close([value.grad for value in inputs], [value.grad for value in references])

    return check
