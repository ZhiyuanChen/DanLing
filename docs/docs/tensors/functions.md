---
authors:
  - Zhiyuan Chen
date: 2022-05-04
---

# Functions

`NestedTensor` operation support is split by dispatch layer, not by a single
`utils.py` module.  The public documentation follows that structure:

## Torch Functions

`danling.tensors.torch_functions` registers `torch.*` handlers through
`__torch_function__`, such as `torch.cat`, `torch.stack`, reductions, indexing,
and matrix operations.

::: danling.tensors.torch_functions
    options:
      members: false

## NN Functions

`danling.tensors.nn_functions` registers `torch.nn.functional.*` handlers such
as attention, embedding, normalization, pooling, convolution, and loss
functions.

::: danling.tensors.nn_functions
    options:
      members: false

::: danling.tensors.nn_functions.create_flex_block_mask

## Aten Functions

`danling.tensors.aten_functions` registers packed-storage
`__torch_dispatch__` handlers and fallback behavior for aten ops.

::: danling.tensors.aten_functions
    options:
      members: false

## Dispatch Registries

`danling.tensors.ops` provides registry types, dispatch tables, and diagnostic
helpers used to extend or test `NestedTensor` operation support.

::: danling.tensors.ops
    options:
      members:
        - TorchFuncRegistry
        - NestedTensorFuncRegistry
        - NestedTensorAtenRegistry
        - nested_execution_guard

The files under `danling.tensors.functions` are specialized implementations
used by `nn_functions` for convolution, pooling, and channel operators. They are
kept out of the docs navigation because users normally call the corresponding
PyTorch or `torch.nn.functional` API directly.
