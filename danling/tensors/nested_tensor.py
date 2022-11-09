from typing import Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class NestedTensor(object):

    values: Sequence[Tensor]
    batch_first: bool

    def __init__(self, values, batch_first: bool = True):
        if not isinstance(values, Sequence):
            raise ValueError("NestedTensor should be initialized with a list of tensors")
        if not isinstance(values[0], Tensor):
            values = [Tensor(value) for value in values]
        self.values = values
        self.batch_first = batch_first

    def to(self, placement):
        values = [value.to(placement) for value in self.values]
        return NestedTensor(values)

    def __getitem__(self, index) -> Tensor:
        ret = self.values[index]
        if isinstance(ret, Tensor):
            return ret, torch.ones_like(ret)
        else:
            tensor = pad_sequence(ret, batch_first=self.batch_first)
            lens = torch.tensor([len(t) for t in ret])
            mask = torch.arange(max(lens))[None, :] < lens[:, None]
            return tensor, mask

    def __len__(self):
        return len(self.values)

    @property
    def tensor(self):
        return pad_sequence(self.values, batch_first=self.batch_first)

    @property
    def mask(self):
        lens = torch.tensor([len(t) for t in self.values])
        return (torch.arange(max(lens))[None, :] < lens[:, None]).to(self.values[0].device)

    @property
    def shape(self):
        return torch.Size([len(self.values), max([t.shape[0] for t in self.values])])
