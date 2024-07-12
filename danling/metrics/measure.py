# DanLing
# Copyright (C) 2022-Present  DanLing

# This program is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

# pylint: disable=redefined-builtin
from __future__ import annotations

from collections.abc import Mapping
from math import nan
from typing import Any, Callable

import torch
from chanfig import FlatDict
from torch import Tensor

from danling.tensors import NestedTensor

from .utils import flist


class Measure(FlatDict):
    r"""
    A class to manage and compute multiple metrics for model evaluation.

    Inherited from [`FlatDict`][chanfig.FlatDict], this class provides a dictionary-like interface to manage multiple
        metric functions.
    It also provides a [`compute`][danling.metrics.Measure.compute] method to compute all metrics at once.
    """

    def set(self, name: str, metric: Callable[[Tensor, Tensor], Any]) -> None:
        if not callable(metric):
            raise TypeError(f"metric must be callable, but got {type(metric).__name__}")
        return super().set(name, metric)

    @torch.inference_mode()
    def compute(self, input: Tensor, target: Tensor) -> FlatDict[str, flist | float]:
        r"""
        Computes the metrics for the given input and target tensors.

        Args:
            input: The input tensor.
            target: The target tensor.

        Returns:
            A dictionary where the keys are metric names and the values are the computed metric values.
        """
        # if input and target are empty, return nan for all metrics
        if (
            isinstance(input, (Tensor, NestedTensor))
            and input.numel() == 0 == target.numel()
            or isinstance(input, (list, dict))
            and len(input) == 0 == len(target)
        ):
            return FlatDict({name: nan for name in self.keys()})
        ret = FlatDict()
        for name, metric in self.items():
            score = self._compute(metric, input, target)
            if isinstance(score, Mapping):
                ret.merge(score)
            else:
                ret[name] = score
        return ret

    @staticmethod
    @torch.inference_mode()
    def _compute(metric, input: Tensor, target: Tensor) -> flist | float:
        r"""
        Computes a single metric for the given input and target tensors.

        Args:
            metric (Callable[[Tensor, Tensor], Any]): The metric function to be computed.
            input (Tensor): The input tensor.
            target (Tensor): The target tensor.

        Returns:
            The computed metric value. If the metric returns a single value tensor, it is converted to a float.
                If the metric returns a tensor with multiple values, it is converted to a list of floats (flist).
        """
        score = metric(input, target)
        if isinstance(score, Tensor):
            return score.item() if score.numel() == 1 else flist(score.tolist())
        return score
