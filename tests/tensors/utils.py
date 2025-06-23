import torch
from torch.testing import assert_close

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
    out = fn(nested_tensor, **kwargs)
    ref = NestedTensor([fn(t, **kwargs) for t in nested_tensor._storage], **nested_tensor._state)
    if isinstance(out, NestedTensor):
        assert_close(out.tensor, ref.tensor, atol=atol, rtol=rtol)
    else:
        assert_close(out, ref.tensor, atol=atol, rtol=rtol)


def make_range_nested(shapes, device, dtype):
    """Create a NestedTensor filled with range values for deterministic pooling/reshape checks."""
    tensors = []
    for shape in shapes:
        numel = torch.prod(torch.tensor(shape))
        tensors.append(torch.arange(numel, device=device, dtype=dtype).reshape(*shape))
    return NestedTensor(tensors)
