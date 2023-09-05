# pylint: disable=C0116
from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Sequence, SupportsFloat

import torch
from torch import Tensor

from .torch_func_registry import TorchFuncRegistry


def pad_tensor(
    tensors: tuple[Tensor], *, batch_first: bool = True, padding_value: float = 0.0, size: torch.Size | None = None
):
    tensor = tensors[0]
    if size is None:
        size = NestedTensor._size(tuple(tensors))  # pylint: disable=W0212
    ret = torch.zeros(size, dtype=tensor.dtype, device=tensor.device)  # pylint: disable=E1101
    if padding_value:
        ret.fill_(padding_value)
    for i, t in enumerate(tensors):
        ret[i][tuple(slice(0, t.shape[dim]) for dim in range(t.dim()))] = t
    if not batch_first:
        ret = ret.transpose(0, 1)
    return ret


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
            >>> tensor = torch.tensor([1, 2, 3])
            >>> pn_tensor = PNTensor(tensor)
            >>> (tensor == pn_tensor).all()
            PNTensor(True)
            >>> (tensor == pn_tensor.tensor).all()
            PNTensor(True)
        """

        return self

    @property
    def mask(self) -> Tensor:
        r"""
        Identical to `torch.ones_like(self)`.

        Returns:
            (torch.Tensor):

        Examples:
            >>> tensor = torch.tensor([1, 2, 3])
            >>> pn_tensor = PNTensor(tensor)
            >>> (pn_tensor.mask == torch.ones_like(pn_tensor)).all()
            PNTensor(True)
        """

        return torch.ones_like(self)  # pylint: disable=E1101


class NestedTensor:
    r"""
    Wrap a sequence of tensors into a single tensor with a mask.

    In sequence to sequence tasks, elements of a batch are usually not of the same length.
    This made it tricky to use a single tensor to represent a batch of sequences.

    `NestedTensor` allows to store a sequence of tensors of different lengths in a single object.
    It also provides a mask that can be used to retrieve the original sequence of tensors.

    When calling `__getitem__(arg)` on a `NestedTensor`, it has two return type:
    1. if arg is `int` or `slice`, returns a tuple of two `tensor`s, representing data and padding mask.
    2. if arg is a `tuple`, return a new `NestedTensor` with specified shape.

    Attributes:

        storage: The sequence of tensors.
        batch_first:  Whether the first dimension of the tensors is the batch dimension.

            If `True`, the first dimension is the batch dimension, i.e., `B, N, *`.

            If `False`, the first dimension is the sequence dimension, i.e., `N, B, *`
        padding_value: The value used to pad the tensors.

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
        >>> nested_tensor[:]
        (tensor([[1, 2, 3],
                [4, 5, 0]]), tensor([[ True,  True,  True],
                [ True,  True, False]]))
        >>> nested_tensor[1]
        (tensor([4, 5]), tensor([True, True]))
        >>> nested_tensor[:, 1:]
        NestedTensor([[2, 3],
                [5, 0]])
    """

    # pylint: disable=C0103

    storage: Sequence[Tensor] = []
    batch_first: bool = True
    padding_value: SupportsFloat = 0.0
    mask_value: bool = False

    def __init__(
        self,
        tensors: Iterable[Tensor],
        batch_first: bool = True,
        padding_value: SupportsFloat = 0.0,
        mask_value: bool = False,
    ) -> None:
        if not isinstance(tensors, Iterable):
            raise ValueError(f"NestedTensor must be initialised with an Iterable, bug got {type(tensors)}.")
        tensors = list(tensors)
        if len(tensors) == 0:
            raise ValueError("NestedTensor must be initialised with a non-empty Iterable.")
        if not isinstance(tensors[0], Tensor):
            tensors = [torch.tensor(tensor) for tensor in tensors]  # pylint: disable=E1101
        self.storage = tensors
        self.batch_first = batch_first
        self.padding_value = padding_value
        self.mask_value = mask_value

    @property
    def tensor(self) -> Tensor:
        r"""
        Return a single tensor by padding all the tensors.

        Returns:
            (torch.Tensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.tensor
            tensor([[1, 2, 3],
                    [4, 5, 0]])
        """

        return self._tensor(tuple(self.storage), self.batch_first, float(self.padding_value))

    @property
    def mask(self) -> Tensor:
        r"""
        Padding mask of `tensor`.

        Returns:
            (torch.Tensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.mask
            tensor([[ True,  True,  True],
                    [ True,  True, False]])
        """

        return self._mask(tuple(self.storage), self.mask_value)

    def nested_like(self, other: Tensor, unsafe: bool = False) -> NestedTensor:
        r"""
        Create a new `NestedTensor` from a `Tensor`.
        The newly created `NestedTensor` will have the same shape as current `NestedTensor`.

        Args:
            other: The `Tensor` to be nested.
            unsafe: Whether to check the shape of `other` and current `NestedTensor`.

        Returns:
            (NestedTensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> tensor = nested_tensor.tensor
            >>> new_tensor = nested_tensor.nested_like(tensor)
            >>> all([(x == y).all() for x, y in zip(nested_tensor.storage, new_tensor.storage)])
            True
            >>> f = nested_tensor.nested_like(torch.randn(2, 2))
            Traceback (most recent call last):
            ValueError: The shape of NestedTensor and input tensor does not match, torch.Size([2, 3]) != torch.Size([2, 2])
            >>> p = nested_tensor.nested_like(torch.randn(2, 2), True)
            >>> p = nested_tensor.nested_like(torch.randn(3, 3), True)
            Traceback (most recent call last):
            ValueError: The batch size of NestedTensor and input tensor does not match, 2 != 3
        """  # noqa: E501

        if not unsafe and self.shape != other.shape:
            raise ValueError(
                f"The shape of NestedTensor and input tensor does not match, {self.shape} != {other.shape}"
            )
        if self.size(0) != other.size(0):
            raise ValueError(
                f"The batch size of NestedTensor and input tensor does not match, {self.size(0)} != {other.size(0)}"
            )
        return NestedTensor([o[tuple(slice(0, dim) for dim in t.shape)] for t, o in zip(self.storage, other)])

    @property
    def device(self) -> torch.device:  # pylint: disable=E1101
        r"""
        Device of the NestedTensor.

        Returns:
            (torch.Tensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.device
            device(type='cpu')
        """

        return self._device(tuple(self.storage))

    @property
    def shape(self) -> torch.Size:  # pylint: disable=E1101
        r"""
        Alias for `size()`.
        """

        return self.size()

    @property
    def ndim(self) -> int:  # pylint: disable=E1101
        r"""
        Alias for `dim()`.
        """

        return self.dim()

    def tolist(self) -> list:
        return [t.tolist() for t in self.storage]

    def size(self, dim: int | None = None) -> torch.Size | int:  # pylint: disable=E1101
        r"""
        Returns the size of the self `NestedTensor`.

        Args:
            dim: If not specified, the returned value is a `torch.Size`, a subclass of `tuple`.
                If specified, returns an `int` holding the size of that dimension.
                Defaults to `None`.

        Returns:
            (torch.Size | int):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.size()
            torch.Size([2, 3])
            >>> nested_tensor.size(0)
            2
            >>> nested_tensor.storage[1] = torch.tensor([4, 5, 6, 7])
            >>> nested_tensor.shape
            torch.Size([2, 4])
            >>> nested_tensor.size(1)
            4
        """

        return self._size(tuple(self.storage), dim)

    def dim(self) -> int:  # pylint: disable=E1101
        r"""
        Number of dimension of the NestedTensor.

        Returns:
            (int):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.dim()
            2
            >>> nested_tensor.storage.append(torch.tensor([6, 7, 8, 9]))
            >>> nested_tensor.ndim
            2
        """

        return self._dim(tuple(self.storage))

    def where(self, condition, other) -> NestedTensor:
        r"""
        Return a NestedTensor of elements selected from either self or other, depending on condition.

        Returns:
            (NestedTensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.size()
            torch.Size([2, 3])
            >>> nested_tensor.storage[1] = torch.tensor([4, 5, 6, 7])
            >>> nested_tensor.size()
            torch.Size([2, 4])
        """

        if isinstance(condition, NestedTensor) and isinstance(other, NestedTensor):
            return NestedTensor(
                [x.where(c, y) for x, c, y in zip(self.storage, condition.storage, other.storage)], **self._state()
            )
        if isinstance(condition, NestedTensor):
            return NestedTensor([x.where(c, other) for x, c in zip(self.storage, condition.storage)], **self._state())
        if isinstance(other, NestedTensor):
            return NestedTensor([x.where(condition, y) for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor(x.where(condition, other) for x in self.storage)

    def __abs__(self):
        return NestedTensor([abs(value) for value in self.storage], **self._state())

    def __add__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor([x + y for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([value + other for value in self.storage], **self._state())

    def __radd__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor([y + x for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([other + value for value in self.storage], **self._state())

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
            return NestedTensor([x & y for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([value & other for value in self.storage], **self._state())

    def __rand__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor([y & x for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([other & value for value in self.storage], **self._state())

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
            return NestedTensor([x // y for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([value // other for value in self.storage], **self._state())

    def __rfloordiv__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor([y // x for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([other // value for value in self.storage], **self._state())

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
            return NestedTensor([x % y for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([value % other for value in self.storage], **self._state())

    def __rmod__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor([y % x for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([other % value for value in self.storage], **self._state())

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
            return NestedTensor([x * y for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([value * other for value in self.storage], **self._state())

    def __rmul__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor([y * x for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([other * value for value in self.storage], **self._state())

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
            return NestedTensor([x @ y for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([value @ other for value in self.storage], **self._state())

    def __rmatmul__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor([y @ x for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([other @ value for value in self.storage], **self._state())

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
            return NestedTensor([x**y for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([value**other for value in self.storage], **self._state())

    def __rpow__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor([y**x for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([other**value for value in self.storage], **self._state())

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
            return NestedTensor([x / y for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([value / other for value in self.storage], **self._state())

    def __rtruediv__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor([y / x for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([other / value for value in self.storage], **self._state())

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
            return NestedTensor([x - y for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([value - other for value in self.storage], **self._state())

    def __rsub__(self, other):
        if isinstance(other, NestedTensor):
            return NestedTensor([y - x for x, y in zip(self.storage, other.storage)], **self._state())
        return NestedTensor([other - value for value in self.storage], **self._state())

    def __isub__(self, other):
        if isinstance(other, NestedTensor):
            for x, y in zip(self.storage, other.storage):
                x -= y
        else:
            for value in self.storage:
                value -= other
        return self

    def __getitem__(self, index: int | slice | tuple) -> tuple[Tensor, Tensor] | NestedTensor:
        if isinstance(index, tuple):
            return NestedTensor([t[index[0]][index[1:]] for t in self.storage])
        if isinstance(index, (int, slice)):
            ret = self.storage[index]
            if isinstance(ret, Tensor):
                return ret, torch.ones_like(ret, dtype=bool)  # pylint: disable=E1101
            return self.tensor, self.mask
        raise ValueError(f"Unsupported index type {type(index)}")

    def __getattr__(self, name) -> Any:
        if not self.storage:
            raise ValueError(f"Unable to get {name} from an empty {self.__class__.__name__}")
        ret = [getattr(i, name) for i in self.storage]
        elem = ret[0]
        if isinstance(elem, Tensor):
            return NestedTensor(ret, **self._state())
        if callable(elem):
            return NestedTensorFuncWrapper(ret, state=self._state())
        if elem.__hash__ is not None and len(set(ret)) == 1:
            return elem
        return ret

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in NestedTensorFunc or not all(issubclass(t, (torch.Tensor, NestedTensor)) for t in types):
            args = [a.tensor if hasattr(a, "tensor") else a for a in args]
            return func(*args, **kwargs)
        return NestedTensorFunc[func](*args, **kwargs)

    def __len__(self) -> int:
        return len(self.storage)

    def __eq__(self, other) -> bool | Tensor | NestedTensor:  # type: ignore[override]
        if isinstance(other, NestedTensor):
            return self.storage == other.storage
        if isinstance(other, Tensor):
            return self.tensor == other
        if isinstance(other, SupportsFloat):
            return NestedTensor([x == other for x in self.storage], **self._state())
        raise NotImplementedError(f"Cannot compare {self.__class__.__name__} with {other.__class__.__name__}")

    def _state(self) -> Mapping:
        return {k: v for k, v in self.__dict__.items() if k != "storage"}

    def __state__(self) -> Mapping:
        return self.__dict__

    def __setstate__(self, state: Mapping) -> None:
        self.__dict__.update(state)

    def __repr__(self):
        return self.__class__.__name__ + repr(self.tensor)[len(self.tensor.__class__.__name__) :]  # noqa: E203

    @staticmethod
    @lru_cache(maxsize=None)
    def _tensor(storage, batch_first, padding_value: float = 0) -> Tensor:
        if storage[0].dim() == 0:
            return torch.stack(storage, dim=0)  # pylint: disable=E1101
        return pad_tensor(storage, batch_first=batch_first, padding_value=padding_value)

    @staticmethod
    @lru_cache(maxsize=None)
    def _mask(storage, mask_value: bool = False) -> Tensor:
        # pylint: disable=E1101
        if storage[0].dim() == 0:
            return torch.ones(len(storage), dtype=torch.bool)
        lens = torch.tensor([len(t) for t in storage], device=storage[0].device)
        arange = torch.arange(max(lens), device=storage[0].device)[None, :]
        return arange >= lens[:, None] if mask_value else arange < lens[:, None]

    @staticmethod
    @lru_cache(maxsize=None)
    def _device(storage) -> torch.device:  # pylint: disable=E1101
        return storage[0].device

    @staticmethod
    @lru_cache(maxsize=None)
    def _size(storage, dim: int | None = None) -> torch.Size | int:
        # pylint: disable=E1101
        if dim is not None:
            if dim == 0:
                return len(storage)
            return max(t.size(dim - 1) for t in storage)
        if max(t.dim() for t in storage) == 0:
            return torch.Size([len(storage)])
        ndim = max(t.dim() for t in storage)
        size = [max(t.shape[i] if i < len(t.shape) else 0 for t in storage) for i in range(ndim)]
        size.insert(0, len(storage))
        return torch.Size(size)

    @staticmethod
    @lru_cache(maxsize=None)
    def _dim(storage) -> torch.Size:
        # pylint: disable=E1101
        return max(t.dim() for t in storage) + 1


NestedTensorFunc = TorchFuncRegistry()


@NestedTensorFunc.implement(torch.mean)  # pylint: disable=E1101
def mean(
    input,  # pylint: disable=W0622
    dim: int | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
):
    return input.mean(dim=dim, keepdim=keepdim, dtype=dtype)


@NestedTensorFunc.implement(torch.cat)  # pylint: disable=E1101
def cat(tensors, dim: int = 0):
    if dim != 0:
        raise NotImplementedError(f"NestedTensor only supports cat when dim=0, but got {dim}")
    return NestedTensor([t for tensor in tensors for t in tensor.storage], tensors[0]._state())  # pylint: disable=W0212


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
    Function Wrapper to handle NestedTensor as input.
    """

    # pylint: disable=R0903

    storage: Sequence[Callable] = []
    state: Mapping = {}

    def __init__(self, callables, state: Mapping | None = None) -> None:
        if not isinstance(callables, Sequence):
            raise ValueError(f"NestedTensorFuncWrapper must be initialised with a Sequence, bug got {type(callables)}")
        if len(callables) == 0:
            raise ValueError("NestedTensorFuncWrapper must be initialised with a non-empty Sequence.")
        if not callable(callables[0]):
            raise ValueError(
                f"NestedTensorFuncWrapper must be initialised with a Sequence of Callable, bug got {type(callables[0])}"
            )
        self.storage = callables
        if state is None:
            state = {}
        self.state = state

    def __call__(self, *args, **kwargs) -> NestedTensor | Sequence[Tensor]:
        ret = [call(*args, **kwargs) for call in self.storage]
        elem = ret[0]
        if isinstance(elem, Tensor):
            return NestedTensor(ret, **self.state)
        if elem.__hash__ is not None and len(set(ret)) == 1:
            return elem
        return ret
