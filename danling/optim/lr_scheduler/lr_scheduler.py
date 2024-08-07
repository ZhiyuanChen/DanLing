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

from math import cos, pi
from typing import List, Optional
from warnings import warn

from torch.optim import Optimizer, lr_scheduler


class LRScheduler(lr_scheduler._LRScheduler):  # pylint: disable=protected-access
    r"""
    General learning rate scheduler.

    PyTorch LRScheduler is hard to extend.
    This class is a wrapper of PyTorch LRScheduler, which provides a more general interface.
    You only needs to add a new method which calculates a learning rate ratio (range from 0 to 1)
    with total progress (range from 0 to 1), and everything else will be done automatically.

    Moreover, this class has warmup and cooldown built-in.
    By default, the first 5% and last 20% of training steps will be warmup and cooldown respectively.
    You can alternate by passing `warmup_steps` and `cooldown_steps`, or disable them by setting them to 0.

    Args:
        optimizer: Wrapped optimizer.
        total_steps: Total number of trainable steps.
        final_lr_ratio: Final learning rate ratio to initial learning rate.
            Defaults to 1e-3.
        final_lr: Final learning rate.
        min_lr: Minimal learning rate.
            Defaults to 1e-9.
        strategy: Scaling strategy.
            Defaults to "cosine".
        warmup_steps: Number of warmup steps.
            Defaults to `steps // 20`.
        cooldown_steps: Number of cooldown steps.
            Defaults to `steps // 5`.
        last_epoch: The index of last epoch.
            Defaults to -1.
        method: Method to calculate learning rate given ratio, should be one of "percentile" or "numerical".
            Defaults to "percentile" if `final_lr_ratio` is set, otherwise "numerical".

    Examples:
        >>> from danling.optim import LRScheduler
        >>> import torch
        >>> from torch import optim
        >>> optimizer = optim.SGD([{'params': torch.tensor([0])}], lr=1, momentum=0.9)
        >>> scheduler = LRScheduler(optimizer, total_steps=5, final_lr_ratio=1e-5, strategy='linear')
        >>> lrs = []
        >>> for epoch in range(5):
        ...     lrs.append(scheduler.get_lr()[0])
        ...     scheduler.step()
        >>> [round(lr, 10) for lr in lrs]
        [0.1, 0.01, 0.001, 0.0001, 1e-09]
        >>> scheduler = LRScheduler(optimizer, total_steps=5, final_lr_ratio=1e-5, strategy='cosine')
        >>> lrs = []
        >>> for epoch in range(5):
        ...     lrs.append(scheduler.get_lr()[0])
        ...     scheduler.step()
        >>> [round(lr, 10) for lr in lrs]
        [0.3330753446, 0.0187302031, 0.000533897, 3.00232e-05, 1e-09]
        >>> scheduler = LRScheduler(optimizer, total_steps=5, final_lr_ratio=1e-5, strategy='linear', method='numerical')
        >>> lrs = []
        >>> for epoch in range(5):
        ...     lrs.append(scheduler.get_lr()[0])
        ...     scheduler.step()
        >>> [round(lr, 2) for lr in lrs]
        [0.8, 0.6, 0.4, 0.2, 0.0]
    """  # noqa: E501

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        final_lr_ratio: Optional[float] = None,
        final_lr: Optional[float] = None,
        min_lr: float = 1e-9,
        strategy: str = "cosine",
        warmup_steps: Optional[int] = None,
        cooldown_steps: Optional[int] = None,
        last_epoch: int = -1,
        method: Optional[str] = None,
    ):
        if total_steps <= 0:
            raise ValueError(f"Total steps must be positive, but got {total_steps}")
        if warmup_steps is None:
            warmup_steps = total_steps // 20
        elif warmup_steps > total_steps:
            raise ValueError(f"Warmup steps must be less than total steps, but got {warmup_steps} > {total_steps}")
        elif warmup_steps < 0:
            raise ValueError(f"Warmup steps must be positive, but got {warmup_steps}")
        if cooldown_steps is None:
            cooldown_steps = total_steps // 5
        elif cooldown_steps > total_steps:
            raise ValueError(f"Cooldown steps must be less than total steps, but got {cooldown_steps} > {total_steps}")
        elif cooldown_steps < 0:
            raise ValueError(f"Cooldown steps must be positive, but got {cooldown_steps}")
        if final_lr_ratio is not None:
            if final_lr is not None:
                raise ValueError("Only one of `final_lr_ratio` and `final_lr` should be set, but not both")
            if final_lr_ratio < 0:
                raise ValueError(f"`final_lr_ratio` must be positive, but got {final_lr_ratio}")
            if method is None:
                method = "percentile"
        if final_lr is not None and final_lr < 0:
            raise ValueError(f"`final_lr` must be positive, but got {final_lr}")
        if min_lr < 0:
            raise ValueError(f"`min_lr` must be positive, but got {min_lr}")
        self.strategies = {
            k: v for k, v in self.__class__.__dict__.items() if callable(v) and (not k.startswith("_") or k in "get_lr")
        }
        if strategy not in self.strategies:
            raise ValueError(f"Scaling strategy must be one of {self.strategies.keys()}, but got {strategy}")

        if final_lr_ratio is None and final_lr is None:
            final_lr_ratio = 1e-3
            if method is None:
                method = "percentile"
        if final_lr is not None and min_lr > final_lr:
            min_lr = final_lr
        if method is None:
            method = "numerical"

        self.final_lr_ratio = final_lr_ratio
        self.final_lr = final_lr
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.strategy = strategy
        self.method = method
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.cooldown_steps_begin = self.total_steps - self.cooldown_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step_count = self._step_count
        if step_count > self.total_steps + 1 or step_count < 1:
            warn(
                f"Step count {step_count} is out of range [1, {self.total_steps + 1}]",
                category=RuntimeWarning,
                stacklevel=2,
            )
        return [self._get_lr(lr, step_count) for lr in self.base_lrs]

    def _get_lr(
        self,
        lr: float,
        step_count: Optional[int] = None,
        progress: Optional[float] = None,
        warmup_ratio: Optional[float] = None,
        cooldown_ratio: Optional[float] = None,
        method: Optional[str] = None,
    ) -> float:
        method = method or self.method
        step_count = step_count or self._step_count
        progress = progress or min(max(step_count / self.total_steps, 0.0), 1.0)
        final_lr = self.final_lr if self.final_lr is not None else lr * self.final_lr_ratio  # type: ignore[operator]
        ratio = getattr(self, self.strategy)(progress)
        if method == "percentile":
            lr *= pow(final_lr / lr, ratio)
        elif method == "numerical":
            lr = (1 - ratio) * (lr - final_lr) + final_lr
        else:
            raise ValueError(f"Method must be one of ['percentile', 'numerical'], but got {method}")
        if self.warmup_steps > step_count > 0:
            warmup_ratio = warmup_ratio or step_count / self.warmup_steps
            lr = warmup_ratio * (lr - self.min_lr) + self.min_lr
        elif self.cooldown_steps > 0 and step_count > self.cooldown_steps_begin:
            cooldown_ratio = cooldown_ratio or 1 - (step_count - self.cooldown_steps_begin) / self.cooldown_steps
            lr = cooldown_ratio * (lr - self.min_lr) + self.min_lr
        return max(self.min_lr, lr)

    def linear(self, progress: float) -> float:
        return progress

    def cosine(self, progress: float) -> float:
        return 1 - ((1 + cos(pi * progress)) / 2)

    def constant(self, progress: float) -> float:  # pylint: disable=unused-argument
        return 0.0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.strategy}, method={self.method}, "
            f"final_lr_ratio={self.final_lr_ratio}, total_steps={self.total_steps}, "
            f"warmup_steps={self.warmup_steps}, cooldown_steps={self.cooldown_steps})"
        )
