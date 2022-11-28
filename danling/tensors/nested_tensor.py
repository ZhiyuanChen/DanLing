from typing import Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class NestedTensor(object):

    storage: Sequence[Tensor] = []
    batch_first: bool = True

    def __init__(self, values, batch_first: bool = True):
        if not isinstance(values, Sequence):
            raise ValueError(f"NestedTensor should be initialised with a list of tensors, bug got {type(values)}")
        if not isinstance(values[0], Tensor):
            values = [Tensor(value) for value in values]
        self.storage = values
        self.batch_first = batch_first

    def __getitem__(self, index) -> Tensor:
        ret = self.storage[index]
        if isinstance(ret, Tensor):
            return ret, torch.ones_like(ret)
        else:
            tensor = pad_sequence(ret, batch_first=self.batch_first)
            lens = Tensor([len(t) for t in ret])
            mask = torch.arange(max(lens))[None, :] < lens[:, None]
            return tensor, mask

    def __len__(self):
        return len(self.storage)

    @property
    def tensor(self):
        return pad_sequence(self.storage, batch_first=self.batch_first)

    @property
    def mask(self):
        lens = Tensor([len(t) for t in self.storage])
        return (torch.arange(max(lens))[None, :] < lens[:, None]).to(self.storage[0].device)

    @property
    def device(self):
        return self.storage[0].device

    @property
    def shape(self):
        return self.size()

    def size(self):
        return torch.Size([len(self.storage), max(t.shape[0] for t in self.storage)])

    def cat(self):
        return torch.cat(self.storage)

    def sum(self):
        return torch.sum(self.cat())

    def __getattr__(self, name):
        if not self.storage:
            raise ValueError(f"Unable to get {name} from an empty {self.__class__.__name__}")
        ret = [getattr(i, name) for i in self.storage]
        elem = ret[0]
        if isinstance(elem, Tensor):
            return NestedTensor(ret)
        elif callable(elem):
            return TensorFuncWrapper(ret)
        elif len(set(ret)) == 1:
            return elem
        else:
            return ret

    def __setstate__(self, storage):
        self.storage = storage

    def __getstate__(self):
        return self.storage


class TensorFuncWrapper:
    def __init__(self, values):
        if not isinstance(values, Sequence):
            raise ValueError(f"TensorFuncWrapper should be initialised with a list of tensors, bug got {type(values)}")
        if not callable(values[0]):
            raise ValueError(f"Elements in TensorFuncWrapper must be Callable, bug got {type(values[0])}")
        self.storage = values

    def __call__(self, *args, **kwargs):
        ret = [call(*args, **kwargs) for call in self.storage]
        elem = ret[0]
        if isinstance(elem, Tensor):
            return NestedTensor(ret)
        elif len(set(ret)) == 1:
            return elem
        else:
            return ret
