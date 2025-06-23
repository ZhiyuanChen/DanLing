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
- Support for 615+ PyTorch operations via a multi-level dispatch system

## Key Components

- [`NestedTensor`][danling.tensors.NestedTensor]: Main class for handling variable-length tensors in a batch.
- [`PNTensor`][danling.tensors.PNTensor]: A tensor wrapper that can be converted to NestedTensor by PyTorch DataLoader.
- `tensor()`: Function to create a [`PNTensor`][danling.tensors.PNTensor] object (similar to `torch.tensor()`).
- [`NestedTensorFuncRegistry`][danling.tensors.NestedTensorFuncRegistry]: Registry for `torch.*` and `F.*` dispatch handlers.
- [`NestedTensorAtenRegistry`][danling.tensors.NestedTensorAtenRegistry]: Registry for `aten` dispatch handlers.

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
native = nested.to_torch_nested()
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

**Level 1 — Aten dispatch** (`aten_functions.py`, 285 ops): Operates directly on the packed `_values` tensor via `__torch_dispatch__`. This is the fastest path — no Python loops, no unpacking. Used for elementwise ops (add, mul, sin, exp, ...), reductions, softmax, layer_norm, etc.

**Level 2 — Torch function dispatch** (`torch_functions.py`, 217 ops): Intercepts `torch.*` calls via `__torch_function__`. Handles ops that need dimension translation (e.g., `torch.flatten`, `torch.softmax` with non-default dim), multi-operand dispatch (e.g., `torch.einsum`), per-element matrix ops (e.g., `torch.det`, `torch.linalg.svd`), and fused attention (`torch._native_multi_head_attention`, `torch._transformer_encoder_layer_fwd`).

**Level 3 — NN function dispatch** (`nn_functions.py`, 113 ops): Also via `__torch_function__`, handles `torch.nn.functional.*` ops including convolutions, pooling, normalization, attention, embedding, activations (F.relu, F.gelu, F.silu, ...), and loss functions. Transformer-hot ops use packed fast paths; activation handlers strip inplace flags to preserve autograd on the wrapper subclass.

**Fallback**: Any aten op without an explicit handler falls back to `per_element_fallback`, which unpacks to individual tensors, applies the op element-by-element, and repacks. Under `torch.compile`, DanLing prefers explicit failure over silently entering those eager-only fallbacks.

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

### Working with PyTorch Models

`NestedTensor` works natively with PyTorch's built-in transformer and vision models — no padding or masks needed:

```python
import torch.nn as nn
from danling.tensors import NestedTensor

# Variable-length sequences
nt = NestedTensor([torch.randn(3, 512), torch.randn(7, 512)])

# Passes directly through nn.TransformerEncoder, nn.Transformer, etc.
encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
    num_layers=6,
)
output = encoder(nt)  # NestedTensor — only real tokens computed

# Also works with torchvision models for variable-size images
import torchvision.models as models
resnet = models.resnet18()
images = NestedTensor([torch.randn(3, 224, 224), torch.randn(3, 160, 160)])
features = resnet(images)
```

For models that require padded input (e.g., HuggingFace transformers), materialize with `.tensor` and `.mask`:

```python
outputs = model(
    input_ids=nested_inputs.tensor,
    attention_mask=nested_inputs.mask
)
```

### Extending with New Operations

You can register new `torch.*` functions to work with NestedTensor:

```python
import torch
from danling.tensors.ops import NestedTensorFuncRegistry

@NestedTensorFuncRegistry.implement(torch.my_custom_op)
def my_custom_op(input, *args, **kwargs):
    from danling.tensors.aten_functions import _packed_like
    # For elementwise ops, apply on packed _values:
    return _packed_like(input, torch.my_custom_op(input._values, *args, **kwargs))
```

For ops that are purely elementwise on the packed data, register at the aten level instead:

```python
from danling.tensors.ops import NestedTensorAtenRegistry
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

## Benchmarks

Benchmarked on a single NVIDIA H100 80GB GPU with PyTorch 2.10, bfloat16.

Run with: `python scripts/benchmark_nested_tensor.py`

### Models

| Model              | Mode  | Occ. | Padded (eager) | Padded (compiled) | DanLing (eager) | DanLing (compiled) | DL vs Padded | DL vs Compiled |
| :----------------- | :---- | :--- | :------------- | :---------------- | :-------------- | :----------------- | :----------- | :------------- |
| TransformerEncoder | Infer | 20%  | 3.59 ms        | 44.01 ms          | 8.25 ms         | 11.48 ms           | 0.43x        | 5.33x          |
| TransformerEncoder | Train | 20%  | 67.58 ms       | 57.92 ms          | 13.29 ms        | ERR ms             | 5.08x        | 4.36x          |
| TransformerEncoder | Infer | 35%  | 5.12 ms        | 43.70 ms          | 8.40 ms         | 3.13 ms            | 0.61x        | 5.20x          |
| TransformerEncoder | Train | 35%  | 67.38 ms       | 57.71 ms          | 17.27 ms        | ERR ms             | 3.90x        | 3.34x          |
| TransformerEncoder | Infer | 77%  | 9.95 ms        | 42.75 ms          | 8.72 ms         | 7.78 ms            | 1.14x        | 4.90x          |
| TransformerEncoder | Train | 77%  | 67.73 ms       | 58.04 ms          | 30.28 ms        | ERR ms             | 2.24x        | 1.92x          |
| TransformerDecoder | Infer | 20%  | 61.00 ms       | 38.25 ms          | 14.93 ms        | 2.90 ms            | 4.09x        | 2.56x          |
| TransformerDecoder | Train | 20%  | 120.40 ms      | 104.08 ms         | 23.01 ms        | ERR ms             | 5.23x        | 4.52x          |
| TransformerDecoder | Infer | 35%  | 61.37 ms       | 38.29 ms          | 14.57 ms        | 4.91 ms            | 4.21x        | 2.63x          |
| TransformerDecoder | Train | 35%  | 120.56 ms      | 104.20 ms         | 28.84 ms        | ERR ms             | 4.18x        | 3.61x          |
| TransformerDecoder | Infer | 77%  | 60.55 ms       | 38.56 ms          | 14.91 ms        | 12.44 ms           | 4.06x        | 2.59x          |
| TransformerDecoder | Train | 77%  | 121.02 ms      | 104.72 ms         | 49.87 ms        | ERR ms             | 2.43x        | 2.10x          |
| Transformer        | Infer | 21%  | 57.28 ms       | 71.10 ms          | 23.87 ms        | 4.77 ms            | 2.40x        | 2.98x          |
| Transformer        | Train | 21%  | 161.77 ms      | 137.56 ms         | 37.80 ms        | ERR ms             | 4.28x        | 3.64x          |
| Transformer        | Infer | 40%  | 58.26 ms       | 70.96 ms          | 24.07 ms        | 8.47 ms            | 2.42x        | 2.95x          |
| Transformer        | Train | 40%  | 161.93 ms      | 137.71 ms         | 47.95 ms        | ERR ms             | 3.38x        | 2.87x          |
| Transformer        | Infer | 84%  | 62.11 ms       | 70.01 ms          | 24.19 ms        | 20.92 ms           | 2.57x        | 2.89x          |
| Transformer        | Train | 84%  | 162.80 ms      | 139.16 ms         | 82.53 ms        | ERR ms             | 1.97x        | 1.69x          |
| ResNet-50          | Infer | 41%  | 61.21 ms       | ERR ms            | 298.47 ms       | ERR ms             | 0.21x        | N/A            |
| ResNet-50          | Train | 41%  | 256.04 ms      | ERR ms            | 241.48 ms       | ERR ms             | 1.06x        | N/A            |
| ResNet-50          | Infer | 52%  | 61.25 ms       | ERR ms            | 336.67 ms       | ERR ms             | 0.18x        | N/A            |
| ResNet-50          | Train | 52%  | 255.27 ms      | ERR ms            | 242.00 ms       | ERR ms             | 1.05x        | N/A            |
| ResNet-50          | Infer | 81%  | 61.29 ms       | ERR ms            | 435.88 ms       | ERR ms             | 0.14x        | N/A            |
| ResNet-50          | Train | 81%  | 255.17 ms      | ERR ms            | 245.97 ms       | ERR ms             | 1.04x        | N/A            |

> **Note:** ResNet-50 uses per-element dispatch (each image processed individually through conv/pool/BN layers). Inference is slower than padded due to per-element repacking overhead. BatchNorm statistics are computed correctly across all elements via concatenated storage.

### Operators

| Operator     | Occ. | Padded (eager) | Padded (compiled) | DanLing (eager) | DanLing (compiled) | torch.nested (eager) | torch.nested (compiled) | DL vs Padded | DL vs torch.nested |
| :----------- | :--- | :------------- | :---------------- | :-------------- | :----------------- | :------------------- | :---------------------- | :----------- | :----------------- |
| F.linear     | 35%  | 0.10 ms        | 0.10 ms           | 0.22 ms         | 0.27 ms            | 0.20 ms              | 0.61 ms                 | 0.37x        | 2.23x              |
| F.layer_norm | 35%  | 0.17 ms        | 0.10 ms           | 0.17 ms         | 0.26 ms            | 0.32 ms              | 0.58 ms                 | 0.37x        | 2.26x              |
| F.relu       | 35%  | 0.09 ms        | 0.09 ms           | 0.14 ms         | 0.25 ms            | 0.16 ms              | 0.57 ms                 | 0.37x        | 2.25x              |
| F.gelu       | 35%  | 0.10 ms        | 0.10 ms           | 0.14 ms         | 0.25 ms            | 0.16 ms              | 0.57 ms                 | 0.42x        | 2.29x              |
| F.softmax    | 35%  | 0.12 ms        | 0.09 ms           | 0.17 ms         | 0.25 ms            | 0.24 ms              | 0.56 ms                 | 0.38x        | 2.26x              |
| F.embedding  | 35%  | 0.05 ms        | 0.05 ms           | 0.25 ms         | 0.26 ms            | 0.21 ms              | 0.57 ms                 | 0.20x        | 2.20x              |
| torch.matmul | 35%  | 0.10 ms        | 0.10 ms           | 0.27 ms         | 0.27 ms            | 0.24 ms              | 0.60 ms                 | 0.37x        | 2.23x              |
| torch.add    | 35%  | 0.09 ms        | 0.09 ms           | 0.19 ms         | 0.26 ms            | 0.17 ms              | 0.56 ms                 | 0.37x        | 2.20x              |
