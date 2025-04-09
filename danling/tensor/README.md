# Tensor

The `danling.tensor` module provides utilities for handling tensors with variable lengths in batched operations.
The core feature is the [`NestedTensor`][danling.tensor.NestedTensor] class which allows efficient representation of sequences of different lengths without excessive padding.

## Overview

In many deep learning tasks, especially those involving sequences (text, time series, etc.), each example in a batch may have a different length. Traditional approaches include:

1. **Padding**: Adding placeholder values to make all examples the same length (wastes computation)
2. **Bucketing**: Grouping similar-length examples (complicates training)
3. **Processing one sample at a time**: Slow and inefficient

The `NestedTensor` solves these problems by providing:

- A way to store variable-length tensors in a single object
- Automatic padding and mask generation for efficient computation
- Transparent access to the original tensors or padded representations
- PyTorch-like operations on nested structures

## Key Components

The module consists of several key components:

- [`NestedTensor`][danling.tensor.NestedTensor]: Main class for handling variable-length tensors in a batch.
- [`PNTensor`][danling.tensor.PNTensor]: A tensor wrapper that can be automatically converted to NestedTensor by PyTorch DataLoader.
- `tensor()`: Function to create a [`PNTensor`][danling.tensor.PNTensor] object (similar to `torch.tensor()`).
- [`TorchFuncRegistry`][danling.tensor.TorchFuncRegistry]: Registry for extending PyTorch functions to work with [`NestedTensor`][danling.tensor.NestedTensor].
- [`functional`][danling.tensor.functional]: Helper functions for padding, masking, and tensor manipulation.

## Quick Start

### Creating a NestedTensor

```python
import torch
from danling.tensor import NestedTensor

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
print(nested[:, 1:])  # Slice: NestedTensor([[2, 3], [5, 0]])
```

### Creating from Non-Tensor Data

```python
from danling.tensor import NestedTensor

# Create directly from lists
nested = NestedTensor([1, 2, 3], [4, 5])
print(nested)  # NestedTensor([[1, 2, 3], [4, 5, 0]])
print(nested.tolist())  # [[1, 2, 3], [4, 5]]
```

### Convert to torch.nested_tensor

```python
from danling.tensor import NestedTensor

# Create directly from lists
nested = NestedTensor([1, 2, 3], [4, 5])
print(nested)  # NestedTensor([[1, 2, 3], [4, 5, 0]])
print(nested.tolist())  # [[1, 2, 3], [4, 5]]
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
tensor, mask = nested[:]

# Access individual items
first_item = nested[0]  # Returns the first tensor
```

## Integration with PyTorch DataLoader

The `PNTensor` class makes it easy to use `NestedTensor` with PyTorch's DataLoader:

```python
from torch.utils.data import Dataset, DataLoader
from danling.tensor import PNTensor

class VariableLengthDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a PNTensor, which will be automatically
        # collated into a NestedTensor
        return PNTensor(self.data[idx])

# Example usage
dataset = VariableLengthDataset([
    [1, 2, 3],
    [4, 5],
    [6, 7, 8, 9]
])
dataloader = DataLoader(dataset, batch_size=3)

# The batches will be NestedTensor objects
for batch in dataloader:
    print(type(batch))  # <class 'danling.tensor.nested_tensor.NestedTensor'>
    print(batch.tensor)  # Padded tensor
    print(batch.mask)    # Mask
```

## Advanced Usage

### Custom Collation

If you need more control over collation:

```python
from torch.utils.data import DataLoader
from danling.tensor import NestedTensor

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

### Extending PyTorch Functions

You can extend PyTorch functions to work with NestedTensor:

```python
from danling.tensor.nested_tensor import NestedTensorFuncRegistry
import torch

@NestedTensorFuncRegistry.implement(torch.softmax)
def softmax(tensor, dim=-1):
    # Implement softmax for NestedTensor
    return tensor.nested_like(torch.softmax(tensor.tensor, dim=dim))
```
