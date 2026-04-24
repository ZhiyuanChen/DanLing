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
        # Return a PNTensor; DataLoader collates it into NestedTensor.
        return PNTensor(self.data[idx])

# Example usage
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

Benchmarked on a single NVIDIA B200 180GB GPU with PyTorch 2.11, bfloat16.

Run with: `python scripts/benchmark_nested_tensor.py`

### IMDB Training

Real workload benchmark from [`examples/tensors/imdb.py`](../../examples/tensors/imdb.py), using a BERT-large-shaped `torch.nn.TransformerEncoder` on IMDB with long variable-length sequences.

Config: `bert-large-uncased`, 2 epochs, batch size `32`, max length `8192`, `d_model=1024`, `nhead=16`, `num_layers=24`

| Metric                                                 | NestedTensor  | Padded        | Result         |
| :----------------------------------------------------- | :------------ | :------------ | :------------- |
| Training step compute (forward + backward, all epochs) | `154819.4 ms` | `306926.7 ms` | `1.98x faster` |
| Peak extra CUDA memory per training step               | `12.68 GiB`   | `74.67 GiB`   | `83% lower`    |

This run measured nearly 2x faster model compute and an 83% reduction in peak extra CUDA memory for the NestedTensor path.

> **Note:** This benchmark compares native PyTorch `nn.TransformerEncoder` execution on NestedTensor vs padded input. The timing is model forward+backward compute, not full end-to-end wall clock including tokenization, data loading, or validation.

### Models

Synthetic model benchmarks covering `TransformerEncoder`, `TransformerDecoder`, `Transformer`, and `ResNet-50` across varying occupancy levels on a single NVIDIA B200 180GB GPU.

| Model              | Mode  | Occ. | Padded (eager) | Padded (compiled) | DanLing (eager) | DanLing (compiled) | DL vs Padded | DL vs Compiled |
| :----------------- | :---- | :--- | :------------- | :---------------- | :-------------- | :----------------- | :----------- | :------------- |
| TransformerEncoder | Infer | 20%  | 2.70 ms        | 39.98 ms          | 5.00 ms         | 1.10 ms            | 0.54x        | 7.99x          |
| TransformerEncoder | Train | 20%  | 26.95 ms       | 19.39 ms          | 8.60 ms         | ERR ms             | 3.13x        | 2.25x          |
| TransformerEncoder | Infer | 35%  | 3.68 ms        | 38.93 ms          | 5.20 ms         | 1.79 ms            | 0.71x        | 7.48x          |
| TransformerEncoder | Train | 35%  | 27.05 ms       | 19.46 ms          | 11.55 ms        | ERR ms             | 2.34x        | 1.68x          |
| TransformerEncoder | Infer | 77%  | 6.95 ms        | 36.39 ms          | 5.52 ms         | 4.45 ms            | 1.26x        | 6.59x          |
| TransformerEncoder | Train | 77%  | 27.15 ms       | 19.71 ms          | 21.37 ms        | ERR ms             | 1.27x        | 0.92x          |
| TransformerDecoder | Infer | 20%  | 47.38 ms       | 10.86 ms          | 9.27 ms         | 1.79 ms            | 5.11x        | 1.17x          |
| TransformerDecoder | Train | 20%  | 45.86 ms       | 33.16 ms          | 15.00 ms        | ERR ms             | 3.06x        | 2.21x          |
| TransformerDecoder | Infer | 35%  | 46.33 ms       | 10.91 ms          | 9.19 ms         | 2.91 ms            | 5.04x        | 1.19x          |
| TransformerDecoder | Train | 35%  | 45.98 ms       | 33.30 ms          | 19.67 ms        | ERR ms             | 2.34x        | 1.69x          |
| TransformerDecoder | Infer | 77%  | 43.79 ms       | 11.02 ms          | 9.41 ms         | 7.47 ms            | 4.65x        | 1.17x          |
| TransformerDecoder | Train | 77%  | 46.18 ms       | 33.50 ms          | 36.00 ms        | ERR ms             | 1.28x        | 0.93x          |
| Transformer        | Infer | 21%  | 47.99 ms       | 48.51 ms          | 14.81 ms        | 2.93 ms            | 3.24x        | 3.27x          |
| Transformer        | Train | 21%  | 68.79 ms       | 47.25 ms          | 24.53 ms        | ERR ms             | 2.80x        | 1.93x          |
| Transformer        | Infer | 40%  | 47.54 ms       | 47.52 ms          | 14.26 ms        | 5.39 ms            | 3.34x        | 3.33x          |
| Transformer        | Train | 40%  | 69.03 ms       | 47.49 ms          | 32.72 ms        | ERR ms             | 2.11x        | 1.45x          |
| Transformer        | Infer | 84%  | 48.37 ms       | 45.12 ms          | 15.76 ms        | 12.51 ms           | 3.07x        | 2.86x          |
| Transformer        | Train | 84%  | 69.52 ms       | 48.81 ms          | 59.88 ms        | ERR ms             | 1.16x        | 0.82x          |
| ResNet-50          | Infer | 41%  | 42.42 ms       | ERR ms            | 219.49 ms       | ERR ms             | 0.19x        | N/A            |
| ResNet-50          | Train | 41%  | 221.80 ms      | ERR ms            | 513.08 ms       | ERR ms             | 0.43x        | N/A            |
| ResNet-50          | Infer | 52%  | 42.56 ms       | ERR ms            | 246.27 ms       | ERR ms             | 0.17x        | N/A            |
| ResNet-50          | Train | 52%  | 221.37 ms      | ERR ms            | 551.29 ms       | ERR ms             | 0.40x        | N/A            |
| ResNet-50          | Infer | 81%  | 42.60 ms       | ERR ms            | 318.03 ms       | ERR ms             | 0.13x        | N/A            |
| ResNet-50          | Train | 81%  | 229.88 ms      | ERR ms            | 709.83 ms       | ERR ms             | 0.32x        | N/A            |

> **Note:** ResNet-50 uses per-element dispatch (each image processed individually through conv/pool/BN layers). Inference is slower than padded due to per-element repacking overhead. BatchNorm statistics are computed correctly across all elements via concatenated storage.

### Operators

Synthetic operator benchmarks covering common transformer-hot ops and tensor primitives across padded tensors, DanLing `NestedTensor`, and `torch.nested`.

| Operator     | Occ. | Padded (eager) | Padded (compiled) | DanLing (eager) | DanLing (compiled) | torch.nested (eager) | torch.nested (compiled) | DL vs Padded | DL vs torch.nested |
| :----------- | :--- | :------------- | :---------------- | :-------------- | :----------------- | :------------------- | :---------------------- | :----------- | :----------------- |
| F.linear     | 35%  | 0.05 ms        | 0.05 ms           | 0.13 ms         | 0.17 ms            | 0.13 ms              | 0.39 ms                 | 0.27x        | 2.29x              |
| F.layer_norm | 35%  | 0.17 ms        | 0.05 ms           | 0.10 ms         | 0.16 ms            | 0.20 ms              | 0.38 ms                 | 0.28x        | 2.36x              |
| F.relu       | 35%  | 0.05 ms        | 0.05 ms           | 0.09 ms         | 0.16 ms            | 0.10 ms              | 0.37 ms                 | 0.29x        | 2.38x              |
| F.gelu       | 35%  | 0.08 ms        | 0.08 ms           | 0.08 ms         | 0.16 ms            | 0.10 ms              | 0.36 ms                 | 0.51x        | 2.33x              |
| F.softmax    | 35%  | 0.12 ms        | 0.06 ms           | 0.11 ms         | 0.16 ms            | 0.15 ms              | 0.37 ms                 | 0.36x        | 2.31x              |
| F.embedding  | 35%  | 0.04 ms        | 0.04 ms           | 0.15 ms         | 0.16 ms            | 0.14 ms              | 0.36 ms                 | 0.25x        | 2.27x              |
| torch.matmul | 35%  | 0.04 ms        | 0.05 ms           | 0.17 ms         | 0.17 ms            | 0.15 ms              | 0.39 ms                 | 0.28x        | 2.31x              |
| torch.add    | 35%  | 0.05 ms        | 0.05 ms           | 0.12 ms         | 0.15 ms            | 0.11 ms              | 0.36 ms                 | 0.29x        | 2.36x              |
