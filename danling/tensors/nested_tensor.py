from functools import lru_cache  # for backward compatibility with Python 3.6
from typing import Any, Callable, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class NestedTensor:
    r"""
    Wrap a sequence of tensors into a single tensor with a mask.

    In sequence to sequence tasks, elements of a batch are usually not of the same length.
    This made it tricky to use a single tensor to represent a batch of sequences.

    NestedTensor allows to store a sequence of tensors of different lengths in a single object.
    It also provides a mask that can be used to retrieve the original sequence of tensors.

    Attributes
    ----------
    storage: Sequence[torch.Tensor]
        The sequence of tensors.
    batch_first: bool = True
        Whether the first dimension of the tensors is the batch dimension.

        If `True`, the first dimension is the batch dimension, i.e., `B, N, *`.

        If `False`, the first dimension is the sequence dimension, i.e., `N, B, *`

    Parameters
    ----------
    tensors: Sequence[torch.Tensor]
    batch_first: bool = True

    Raises
    ------
    ValueError
        If `tensors` is not a sequence.

        If `tensors` is empty.

    Notes
    -----
    We have rewritten the `__getattr__` function to support as much native tensor operations as possible.
    However, not all operations are tested.

    Please file an issue if you find any bugs.

    Examples
    --------
    ```python
    >>> from danling.tensors import NestedTensor
    >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
    >>> nested_tensor.shape
    torch.Size([2, 3])
    >>> nested_tensor.device
    device(type='cpu')
    >>> nested_tensor.dtype
    torch.int64
    >>> nested_tensor.tensor
    tensor([[1, 2, 3],
            [4, 5, 0]])
    >>> nested_tensor.mask
    tensor([[ True,  True,  True],
            [ True,  True, False]])
    >>> nested_tensor.to(torch.float).tensor
    tensor([[1., 2., 3.],
            [4., 5., 0.]])
    >>> nested_tensor.half().tensor
    tensor([[1., 2., 3.],
            [4., 5., 0.]], dtype=torch.float16)

    ```
    """

    storage: Sequence[Tensor] = []
    batch_first: bool = True

    def __init__(self, tensors: Sequence[Tensor], batch_first: bool = True) -> None:
        if not isinstance(tensors, Sequence):
            raise ValueError(f"NestedTensor should be initialised with a Sequence, bug got {type(values)}.")
        if len(tensors) == 0:
            raise ValueError(f"NestedTensor should be initialised with a non-empty sequence.")
        if not isinstance(tensors[0], Tensor):
            tensors = tuple(torch.tensor(tensor) for tensor in tensors)
        self.storage = tensors
        self.batch_first = batch_first

    @property
    def tensor(self) -> Tensor:
        r"""
        Return a single tensor by padding all the tensors.

        Returns
        -------
        torch.Tensor

        Examples
        --------
        ```python
        >>> from danling.tensors import NestedTensor
        >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        >>> nested_tensor.tensor
        tensor([[1, 2, 3],
                [4, 5, 0]])

        ```
        """

        return self._tensor(tuple(self.storage), self.batch_first)

    @property
    def mask(self) -> Tensor:
        r"""
        Padding mask of `tensor`.

        Returns
        -------
        torch.Tensor

        Examples
        --------
        ```python
        >>> from danling.tensors import NestedTensor
        >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        >>> nested_tensor.mask
        tensor([[ True,  True,  True],
                [ True,  True, False]])

        ```
        """

        return self._mask(tuple(self.storage))

    @property
    def device(self) -> torch.device:
        r"""
        Device of the NestedTensor.

        Returns
        -------
        torch.Tensor

        Examples
        --------
        ```python
        >>> from danling.tensors import NestedTensor
        >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        >>> nested_tensor.device
        device(type='cpu')

        ```
        """

        return self._device(tuple(self.storage))

    @property
    def shape(self) -> torch.Size:
        r"""
        Alias for `size`.

        Returns
        -------
        torch.Size

        Examples
        --------
        ```python
        >>> from danling.tensors import NestedTensor
        >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        >>> nested_tensor.shape
        torch.Size([2, 3])
        >>> nested_tensor.storage.append(torch.tensor([6, 7, 8, 9]))
        >>> nested_tensor.shape
        torch.Size([3, 4])

        ```
        """

        return self.size()

    def size(self) -> torch.Size:
        r"""
        Shape of the NestedTensor.

        Returns
        -------
        torch.Size

        Examples
        --------
        ```python
        >>> from danling.tensors import NestedTensor
        >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        >>> nested_tensor.size()
        torch.Size([2, 3])
        >>> nested_tensor.storage[1] = torch.tensor([4, 5, 6, 7])
        >>> nested_tensor.size()
        torch.Size([2, 4])

        ```
        """

        return self._size(tuple(self.storage))

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        ret = self.storage[index]
        if isinstance(ret, Tensor):
            return ret, torch.ones_like(ret)  # pylint: disable=E1101
        return self.tensor, self.mask

    def __getattr__(self, name) -> Any:
        if not self.storage:
            raise ValueError(f"Unable to get {name} from an empty {self.__class__.__name__}")
        ret = [getattr(i, name) for i in self.storage]
        elem = ret[0]
        if isinstance(elem, Tensor):
            return NestedTensor(ret)
        if callable(elem):
            return _TensorFuncWrapper(ret)
        if len(set(ret)) == 1:
            return elem
        return ret

    def __len__(self) -> int:
        return len(self.storage)

    def __setstate__(self, storage) -> None:
        self.storage = storage

    def __getstate__(self) -> Sequence[Tensor]:
        return self.storage

    @staticmethod
    @lru_cache(maxsize=None)
    def _tensor(storage, batch_first) -> Tensor:
        return pad_sequence(storage, batch_first=batch_first)

    @staticmethod
    @lru_cache(maxsize=None)
    def _mask(storage) -> Tensor:
        lens = torch.tensor([len(t) for t in storage], device=storage[0].device)
        return torch.arange(max(lens), device=storage[0].device)[None, :] < lens[:, None]  # pylint: disable=E1101

    @staticmethod
    @lru_cache(maxsize=None)
    def _device(storage) -> torch.device:
        return storage[0].device

    @staticmethod
    @lru_cache(maxsize=None)
    def _size(storage) -> torch.Size:
        return torch.Size(  # pylint: disable=E1101
            [len(storage), max(t.shape[0] for t in storage), *storage[0].shape[1:]]
        )


class _TensorFuncWrapper:

    # pylint: disable=R0903

    storage: Sequence[Callable] = []

    def __init__(self, values) -> None:
        if not isinstance(values, Sequence):
            raise ValueError(
                f"_TensorFuncWrapper should be initialised with a list of Callables, bug got {type(values)}"
            )
        if not callable(values[0]):
            raise ValueError(f"Elements in _TensorFuncWrapper must be Callable, bug got {type(values[0])}")
        self.storage = values

    def __call__(self, *args, **kwargs) -> Sequence:
        ret = [call(*args, **kwargs) for call in self.storage]
        elem = ret[0]
        if isinstance(elem, Tensor):
            return NestedTensor(ret)
        if len(set(ret)) == 1:
            return elem
        return ret
