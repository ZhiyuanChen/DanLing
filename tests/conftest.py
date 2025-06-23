# Makes `tests` importable for helper modules.


import pytest

from tests.tensors.utils import FLOAT_DTYPES, available_devices

collect_ignore = ["functional.py"]


@pytest.fixture(autouse=True)
def seed_all():
    import torch

    torch.manual_seed(1016)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1016)


@pytest.fixture(params=available_devices())
def device(request):
    return request.param


@pytest.fixture(params=FLOAT_DTYPES)
def float_dtype(request):
    import torch

    dtype = request.param
    device = request.getfixturevalue("device")
    if dtype in (torch.float16, torch.bfloat16) and device.type != "cuda":
        pytest.skip(f"{dtype} unsupported on CPU for these ops")
    return dtype
