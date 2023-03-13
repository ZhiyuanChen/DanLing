# pylint: disable=C0116
from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .torch_func_registry import TorchFuncRegistry


class PNTensor(Tensor):
    r"""
    Wrapper for tensors to be converted to `NestedTensor`.

    `PNTensor` is a subclass of `torch.Tensor`.
    It implements two additional methods as `NestedTensor`: `tensor` and `mask`.

    Although it is possible to construct `NestedTensor` in dataset,
    the best practice is to do so in `collate_fn`.
    However, it is hard to tell if a batch of `Tensor` should be stacked or converted to `NestedTensor`.

    `PNTensor` is introduced overcome this limitation.

    Convert tensors that will be converted to `NestedTensor` to a `PNTensor`,
    and all you need to do is to convert `PNTensor` to `NestedTensor` in `collate_fn`.
    """

    @property
    def tensor(self) -> Tensor:
        r"""
        Identical to `self`.

        Returns:
            (torch.Tensor):

        Examples:
        ```python
        >>> tensor = torch.tensor([1, 2, 3])
        >>> pn_tensor = PNTensor(tensor)
        >>> (tensor == pn_tensor).all()
        PNTensor(True)
        >>> (tensor == pn_tensor.tensor).all()
        PNTensor(True)

        ```
        """

        return self

    @property
    def mask(self) -> Tensor:
        r"""
        Identical to `torch.ones_like(self)`.

        Returns:
            (torch.Tensor):

        Examples:
        ```python
        >>> tensor = torch.tensor([1, 2, 3])
        >>> pn_tensor = PNTensor(tensor)
        >>> (pn_tensor.mask == torch.ones_like(pn_tensor)).all()
        PNTensor(True)

        ```
        """

        return torch.ones_like(self)  # pylint: disable=E1101


class NestedTensor:
    r"""
    Wrap a sequence of tensors into a single tensor with a mask.

    In sequence to sequence tasks, elements of a batch are usually not of the same length.
    This made it tricky to use a single tensor to represent a batch of sequences.

    `NestedTensor` allows to store a sequence of tensors of different lengths in a single object.
    It also provides a mask that can be used to retrieve the original sequence of tensors.

    Attributes:

        storage: The sequence of tensors.
        batch_first:  Whether the first dimension of the tensors is the batch dimension.

            If `True`, the first dimension is the batch dimension, i.e., `B, N, *`.

            If `False`, the first dimension is the sequence dimension, i.e., `N, B, *`

    Args:
        tensors:
        batch_first:

    Raises:
        ValueError: If `tensors` is not a sequence.
        ValueError: If `tensors` is empty.

    Notes:
        We have rewritten the `__getattr__` function to support as much native tensor operations as possible.
        However, not all operations are tested.

        Please file an issue if you find any bugs.

    Examples:
    ```python
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

    # pylint: disable=C0103

    storage: Sequence[Tensor] = []
    batch_first: bool = True

    def __init__(self, tensors: Iterable[Tensor], batch_first: bool = True) -> None:
        if not isinstance(tensors, Iterable):
            raise ValueError(f"NestedTensor must be initialised with an Iterable, bug got {type(tensors)}.")
        tensors = list(tensors)
        if len(tensors) == 0:
            raise ValueError("NestedTensor must be initialised with a non-empty Iterable.")
        if not isinstance(tensors[0], Tensor):
            tensors = [torch.tensor(tensor) for tensor in tensors]  # pylint: disable=E1101
        self.storage = tensors
        self.batch_first = batch_first

    @property
    def tensor(self) -> Tensor:
        r"""
        Return a single tensor by padding all the tensors.

        Returns:
            (torch.Tensor):

        Examples:
        ```python
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

        Returns:
            (torch.Tensor):

        Examples:
        ```python
        >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        >>> nested_tensor.mask
        tensor([[ True,  True,  True],
                [ True,  True, False]])

        ```
        """

        return self._mask(tuple(self.storage))

    @property
    def device(self) -> torch.device:  # pylint: disable=E1101
        r"""
        Device of the NestedTensor.

        Returns:
            (torch.Tensor):

        Examples:
        ```python
        >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        >>> nested_tensor.device
        device(type='cpu')

        ```
        """

        return self._device(tuple(self.storage))

    @property
    def shape(self) -> torch.Size:  # pylint: disable=E1101
        r"""
        Alias for `size`.

        Returns:
            (torch.Size):

        Examples:
        ```python
        >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        >>> nested_tensor.shape
        torch.Size([2, 3])
        >>> nested_tensor.storage.append(torch.tensor([6, 7, 8, 9]))
        >>> nested_tensor.shape
        torch.Size([3, 4])

        ```
        """

        return self.size()

    def size(self) -> torch.Size:  # pylint: disable=E1101
        r"""
        Shape of the NestedTensor.

        Returns:
            (torch.Size):

        Examples:
        ```python
        >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        >>> nested_tensor.size()
        torch.Size([2, 3])
        >>> nested_tensor.storage[1] = torch.tensor([4, 5, 6, 7])
        >>> nested_tensor.size()
        torch.Size([2, 4])

        ```
        """

        return self._size(tuple(self.storage))

    def where(self, condition, other) -> NestedTensor:
        r"""
        Return a NestedTensor of elements selected from either self or other, depending on condition.

        Returns:
            (NestedTensor):

        Examples:
        ```python
        >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        >>> nested_tensor.size()
        torch.Size([2, 3])
        >>> nested_tensor.storage[1] = torch.tensor([4, 5, 6, 7])
        >>> nested_tensor.size()
        torch.Size([2, 4])

        ```
        """

        if isinstance(condition, NestedTensor) and isinstance(other, NestedTensor):
            return NestedTensor(x.where(c, y) for x, c, y in zip(self.storage, condition.storage, other.storage))
        if isinstance(condition, NestedTensor):
            return NestedTensor(x.where(c, other) for x, c in zip(self.storage, condition.storage))
        if isinstance(other, NestedTensor):
            return NestedTensor(x.where(condition, y) for x, y in zip(self.storage, other.storage))
        return NestedTensor(x.where(condition, other) for x in self.storage)

    def __abs__(self):
        return NestedTensor(abs(value) for value in self.storage)

    def __add__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(x + y for x, y in zip(self.storage, other.storage))
        return NestedTensor(value + other for value in self.storage)

    def __radd__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(y + x for x, y in zip(self.storage, other.storage))
        return NestedTensor(other + value for value in self.storage)

    def __iadd__(self, other):
        if isinstance(other, NestedTensor):
            for x, y in zip(self.storage, other.storage):
                x += y
        else:
            for value in self.storage:
                value += other
        return self

    def __and__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(x & y for x, y in zip(self.storage, other.storage))
        return NestedTensor(value & other for value in self.storage)

    def __rand__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(y & x for x, y in zip(self.storage, other.storage))
        return NestedTensor(other & value for value in self.storage)

    def __iand__(self, other):
        if isinstance(other, NestedTensor):
            for x, y in zip(self.storage, other.storage):
                x &= y
        else:
            for value in self.storage:
                value &= other
        return self

    def __floordiv__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(x // y for x, y in zip(self.storage, other.storage))
        return NestedTensor(value // other for value in self.storage)

    def __rfloordiv__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(y // x for x, y in zip(self.storage, other.storage))
        return NestedTensor(other // value for value in self.storage)

    def __ifloordiv__(self, other):
        if isinstance(other, NestedTensor):
            for x, y in zip(self.storage, other.storage):
                x //= y
        else:
            for value in self.storage:
                value //= other
        return self

    def __mod__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(x % y for x, y in zip(self.storage, other.storage))
        return NestedTensor(value % other for value in self.storage)

    def __rmod__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(y % x for x, y in zip(self.storage, other.storage))
        return NestedTensor(other % value for value in self.storage)

    def __imod__(self, other):
        if isinstance(other, NestedTensor):
            for x, y in zip(self.storage, other.storage):
                x %= y
        else:
            for value in self.storage:
                value %= other
        return self

    def __mul__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(x * y for x, y in zip(self.storage, other.storage))
        return NestedTensor(value * other for value in self.storage)

    def __rmul__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(y * x for x, y in zip(self.storage, other.storage))
        return NestedTensor(other * value for value in self.storage)

    def __imul__(self, other):
        if isinstance(other, NestedTensor):
            for x, y in zip(self.storage, other.storage):
                x *= y
        else:
            for value in self.storage:
                value *= other
        return self

    def __matmul__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(x @ y for x, y in zip(self.storage, other.storage))
        return NestedTensor(value @ other for value in self.storage)

    def __rmatmul__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(y @ x for x, y in zip(self.storage, other.storage))
        return NestedTensor(other @ value for value in self.storage)

    def __imatmul__(self, other):
        if isinstance(other, NestedTensor):
            for x, y in zip(self.storage, other.storage):
                x @= y
        else:
            for value in self.storage:
                value @= other
        return self

    def __pow__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(x**y for x, y in zip(self.storage, other.storage))
        return NestedTensor(value**other for value in self.storage)

    def __rpow__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(y**x for x, y in zip(self.storage, other.storage))
        return NestedTensor(other**value for value in self.storage)

    def __ipow__(self, other):
        if isinstance(other, NestedTensor):
            for x, y in zip(self.storage, other.storage):
                x *= y
        else:
            for value in self.storage:
                value *= other
        return self

    def __truediv__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(x / y for x, y in zip(self.storage, other.storage))
        return NestedTensor(value / other for value in self.storage)

    def __rtruediv__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(y / x for x, y in zip(self.storage, other.storage))
        return NestedTensor(other / value for value in self.storage)

    def __itruediv__(self, other):
        if isinstance(other, NestedTensor):
            for x, y in zip(self.storage, other.storage):
                x /= y
        else:
            for value in self.storage:
                value /= other
        return self

    def __sub__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(x - y for x, y in zip(self.storage, other.storage))
        return NestedTensor(value - other for value in self.storage)

    def __rsub__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor(y - x for x, y in zip(self.storage, other.storage))
        return NestedTensor(other - value for value in self.storage)

    def __isub__(self, other):
        if isinstance(other, NestedTensor):
            for x, y in zip(self.storage, other.storage):
                x -= y
        else:
            for value in self.storage:
                value -= other
        return self

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
            return NestedTensorFuncWrapper(ret)
        if elem.__hash__ is not None and len(set(ret)) == 1:
            return elem
        return ret

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in NestedTensorFunc or not all(issubclass(t, (torch.Tensor, NestedTensor)) for t in types):
            return NotImplemented
        return NestedTensorFunc[func](*args, **kwargs)

    def __len__(self) -> int:
        return len(self.storage)

    def __eq__(self, other) -> bool:
        return self.storage == other.storage

    def __getstate__(self) -> Mapping:
        return self.__dict__

    def __setstate__(self, states: Mapping) -> None:
        self.__dict__.update(states)

    @staticmethod
    @lru_cache(maxsize=None)
    def _tensor(storage, batch_first) -> Tensor:
        if storage[0].dim() == 0:
            return torch.stack(storage, dim=0)  # pylint: disable=E1101
        return pad_sequence(storage, batch_first=batch_first)

    @staticmethod
    @lru_cache(maxsize=None)
    def _mask(storage) -> Tensor:
        if storage[0].dim() == 0:
            return torch.ones(len(storage), dtype=torch.bool)  # pylint: disable=E1101
        lens = torch.tensor([len(t) for t in storage], device=storage[0].device)  # pylint: disable=E1101
        return torch.arange(max(lens), device=storage[0].device)[None, :] < lens[:, None]  # pylint: disable=E1101

    @staticmethod
    @lru_cache(maxsize=None)
    def _device(storage) -> torch.device:  # pylint: disable=E1101
        return storage[0].device

    @staticmethod
    @lru_cache(maxsize=None)
    def _size(storage) -> torch.Size:  # pylint: disable=E1101
        return torch.Size(  # pylint: disable=E1101
            [len(storage), max(t.shape[0] for t in storage), *storage[0].shape[1:]]
        )


NestedTensorFunc = TorchFuncRegistry()


@NestedTensorFunc.implement(torch.mean)  # pylint: disable=E1101
def mean(
    input,  # pylint: disable=W0622
    dim: Optional[int] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
):
    return input.mean(dim=dim, keepdim=keepdim, dtype=dtype)


@NestedTensorFunc.implement(torch.cat)  # pylint: disable=E1101
def cat(tensors, dim: int = 0):
    if dim != 0:
        raise NotImplementedError(f"NestedTensor only supports cat when dim=0, but got {dim}")
    return NestedTensor([t for tensor in tensors for t in tensor.storage])


@NestedTensorFunc.implement(torch.stack)  # pylint: disable=E1101
def stack(tensors, dim: int = 0):
    raise NotImplementedError("NestedTensor does not support stack as of now")


@NestedTensorFunc.implement(torch.isin)  # pylint: disable=E1101
def isin(elements, test_elements, *, assume_unique: bool = False, invert: bool = False):
    if isinstance(elements, NestedTensor):
        elements = elements.tensor
    if isinstance(test_elements, NestedTensor):
        test_elements = test_elements.tensor
    return torch.isin(elements, test_elements, assume_unique=assume_unique, invert=invert)  # pylint: disable=E1101


class NestedTensorFuncWrapper:
    r"""
    Wrapper for tensors to be converted to `NestedTensor`.

    `PNTensor` is a subclass of `torch.Tensor`.
    It implements two additional methods as `NestedTensor`: `tensor` and `mask`.

    Although it is possible to construct `NestedTensor` in dataset,
    the best practice is to do so in `collate_fn`.
    However, it is hard to tell if a batch of `Tensor` should be stacked or converted to `NestedTensor`.

    `PNTensor` is introduced overcome this limitation.

    Convert tensors that will be converted to `NestedTensor` to a `PNTensor`,
    and all you need to do is to convert `PNTensor` to `NestedTensor` in `collate_fn`.
    """

    # pylint: disable=R0903

    storage: Sequence[Callable] = []

    def __init__(self, callables) -> None:
        if not isinstance(callables, Sequence):
            raise ValueError(f"NestedTensorFuncWrapper must be initialised with a Sequence, bug got {type(callables)}")
        if len(callables) == 0:
            raise ValueError("NestedTensorFuncWrapper must be initialised with a non-empty Sequence.")
        if not callable(callables[0]):
            raise ValueError(
                f"NestedTensorFuncWrapper must be initialised with a Sequence of Callable, bug got {type(callables[0])}"
            )
        self.storage = callables

    def __call__(self, *args, **kwargs) -> Union[NestedTensor, Sequence[Tensor]]:
        ret = [call(*args, **kwargs) for call in self.storage]
        elem = ret[0]
        if isinstance(elem, Tensor):
            return NestedTensor(ret)
        if elem.__hash__ is not None and len(set(ret)) == 1:
            return elem
        return ret
