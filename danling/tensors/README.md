# Tensor

The `danling.tensors` module provides utilities for handling tensors with variable lengths in batched operations.
The core feature is the [`NestedTensor`][danling.tensors.NestedTensor] class which allows efficient representation of sequences of different lengths without excessive padding.

## Overview

In many deep learning tasks, especially those involving sequences (text, time series, etc.), each example in a batch may have a different length. Traditional approaches include:

1. **Padding**: Adding placeholder values to make all examples the same length (wastes computation)
2. **Bucketing**: Grouping similar-length examples (complicates training)
3. **Processing one sample at a time**: Slow and inefficient

The `NestedTensor` solves these problems by providing:

- A way to store variable-length tensors in a single object
- Automatic padding and mask generation for efficient computation
- Transparent access to the original tensors or padded representations
- Support for 400+ PyTorch operations via a multi-level dispatch system

## Key Components

- [`NestedTensor`][danling.tensors.NestedTensor]: Main class for handling variable-length tensors in a batch.
- [`PNTensor`][danling.tensors.PNTensor]: A tensor wrapper that can be converted to NestedTensor by PyTorch DataLoader.
- `tensor()`: Function to create a [`PNTensor`][danling.tensors.PNTensor] object (similar to `torch.tensor()`).
- [`TorchFuncRegistry`][danling.tensors.TorchFuncRegistry]: Registry for extending PyTorch functions to work with [`NestedTensor`][danling.tensors.NestedTensor].

## Quick Start

### Creating a NestedTensor

```python
import torch
from danling.tensors import NestedTensor

# Create from a list of tensors with different lengths
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5])
nested = NestedTensor(tensor1, tensor2)

# Access properties
print(nested.tensor)  # Padded tensor: [[1, 2, 3], [4, 5, 0]]
print(nested.mask)    # Mask: [[True, True, True], [True, True, False]]
print(nested.concat)  # Concatenated: [1, 2, 3, 4, 5]

# Index operations
print(nested[0])      # First tensor: [1, 2, 3]
print(nested[:, 1:])  # Slice: NestedTensor([[2, 3], [5]])
```

### Creating from Non-Tensor Data

```python
from danling.tensors import NestedTensor

# Create directly from lists
nested = NestedTensor([1, 2, 3], [4, 5])
print(nested)  # NestedTensor([[1, 2, 3], [4, 5]])
print(nested.tolist())  # [[1, 2, 3], [4, 5]]
```

### Converting to torch.nested_tensor

```python
import torch
from danling.tensors import NestedTensor

nested = NestedTensor([1.0, 2.0, 3.0], [4.0, 5.0])

# Convert to PyTorch's native nested tensor
native = nested.torch
print(type(native))  # <class 'torch.Tensor'> (nested layout)
```

## Working with NestedTensor

### Operations

NestedTensor supports many PyTorch operations:

```python
# Arithmetic operations
result = nested + 10
result = nested * 2

# Type conversion
float_nested = nested.float()
half_nested = nested.half()

# Device movement
gpu_nested = nested.cuda()
cpu_nested = gpu_nested.cpu()

# Shape operations
print(nested.shape)  # torch.Size([2, 3])
print(nested.size(0))  # 2
```

### Unpacking

You can easily convert back to original tensors:

```python
# Get as a list of lists
data = nested.tolist()

# Get as a tuple of (padded_tensor, mask)
tensor, mask = nested.tensor_mask

# Access individual items
first_item = nested[0]  # Returns the first tensor
```

## Architecture

NestedTensor uses a **packed representation** that stores all variable-length elements concatenated into a single contiguous tensor, tracked by offset metadata:

- `_values`: All element tensors concatenated along dim 0 (e.g., shape `[total_elements, *]`)
- `_offsets`: Cumulative element counts, shape `(B+1,)`, marking where each element starts/ends
- `_physical_shape`: Per-element shapes, shape `(B, ndim)`, recording each element's original dimensions

This avoids the waste of padding in the internal representation while allowing efficient batch operations.

### Dispatch System

Operations on NestedTensor are handled by a **three-tier dispatch system**, ordered from fastest to most flexible:

**Level 1 — Aten dispatch** (`aten_functions.py`, ~190 ops): Operates directly on the packed `_values` tensor via `__torch_dispatch__`. This is the fastest path — no Python loops, no unpacking. Used for elementwise ops (add, mul, sin, exp, ...), reductions, softmax, layer_norm, etc.

**Level 2 — Torch function dispatch** (`torch_functions.py`, ~90 explicit + bulk): Intercepts `torch.*` calls via `__torch_function__`. Handles ops that need dimension translation (e.g., `torch.flatten`, `torch.softmax` with non-default dim), multi-operand dispatch (e.g., `torch.einsum`), or per-element matrix ops (e.g., `torch.det`, `torch.linalg.svd`).

**Level 3 — NN function dispatch** (`nn_functions.py`, ~70 ops): Also via `__torch_function__`, handles `torch.nn.functional.*` ops like convolutions, pooling, normalization, attention, and embedding that require per-element spatial reasoning.

**Fallback**: Any aten op without an explicit handler falls back to `per_element_fallback`, which unpacks to individual tensors, applies the op element-by-element, and repacks.

```text
torch.some_op(nested_tensor)
    │
    ▼
__torch_function__  ──── handler in TorchFuncRegistry? ──── yes ──→ dispatch
    │ no
    ▼
aten decomposition  (PyTorch lowers to aten ops)
    │
    ▼
__torch_dispatch__  ──── handler in NestedTensorAtenRegistry? ──── yes ──→ dispatch
    │ no
    ▼
per_element_fallback  (unpack → apply per element → repack)
```

### Key Internal Helpers

- `_from_packed(values, offsets, shape_tensor, ...)`: Direct constructor from packed representation. Used by all aten handlers to build results without function call overhead.
- `_map_storage_serial(input, fn)`: Per-element slow path — applies `fn` to each element via `_unpack()`. Used when ops need individual element dimensionality.
- `_translate_non_batch_dim(nt, dim)`: Converts a NestedTensor dim index to the corresponding element-level dim (skipping the batch dimension).

## Integration with PyTorch DataLoader

The `PNTensor` class makes it easy to use `NestedTensor` with PyTorch's DataLoader:

```python
from torch.utils.data import Dataset, DataLoader
from danling.tensors import PNTensor

class VariableLengthDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a PNTensor; DataLoader can collate it into NestedTensor
        # after explicit registration.
        return PNTensor(self.data[idx])

# Example usage
from danling.tensors import register_pn_tensor_collate
register_pn_tensor_collate()
dataset = VariableLengthDataset([
    [1, 2, 3],
    [4, 5],
    [6, 7, 8, 9]
])
dataloader = DataLoader(dataset, batch_size=3)

# The batches are NestedTensor objects
for batch in dataloader:
    print(type(batch))  # <class 'danling.tensors.nested_tensor.NestedTensor'>
    print(batch.tensor)  # Padded tensor
    print(batch.mask)    # Mask
```

## Advanced Usage

### Custom Collation

If you need more control over collation:

```python
from torch.utils.data import DataLoader
from danling.tensors import NestedTensor

def custom_collate_fn(batch):
    return NestedTensor(*batch)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=custom_collate_fn
)
```

### Working with Masked Models

`NestedTensor` works well with models that support attention masks:

```python
# For transformer models
outputs = model(
    input_ids=nested_inputs.tensor,
    attention_mask=nested_inputs.mask
)
```

### Extending with New Operations

You can register new `torch.*` functions to work with NestedTensor:

```python
import torch
from danling.tensors.torch_functions import NestedTensorFuncRegistry
from danling.tensors.ops import _map_storage_serial

@NestedTensorFuncRegistry.implement(torch.my_custom_op)
def my_custom_op(input, *args, **kwargs):
    # For per-element ops, use _map_storage_serial:
    return _map_storage_serial(input, lambda t: torch.my_custom_op(t, *args, **kwargs))
```

For ops that are purely elementwise on the packed data, register at the aten level instead:

```python
from danling.tensors.aten_functions import NestedTensorAtenRegistry
aten = torch.ops.aten

def _my_handler(func, args, kwargs):
    source = args[0]
    return type(source)._from_packed(
        func(source._values, *args[1:], **kwargs),
        source._offsets, source._physical_shape,
        batch_first=source.batch_first, padding_value=source.padding_value,
        mask_value=source.mask_value, pin_memory=source._pin_memory,
        outer_size=source._logical_shape,
    )

NestedTensorAtenRegistry[aten.my_op.default] = _my_handler
```
