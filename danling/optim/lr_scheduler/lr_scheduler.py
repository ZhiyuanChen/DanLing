from typing import Callable, List, Mapping, Optional
from warnings import warn

import numpy as np
from chanfig import Registry
from torch.optim import Optimizer, lr_scheduler

LR_SCHEDULER_STRATEGIES = Registry()


class LRScheduler(lr_scheduler._LRScheduler):  # pylint: disable=W0212
    r"""
    General learning rate scheduler.

    Args:
        optimizer: Wrapped optimizer.
        steps: Total number of steps.
        final_lr_ratio: Final learning rate ratio to initial learning rate.
        final_lr: Final learning rate. Deprecated, use `final_lr_ratio` instead.
        min_lr: Minimal learning rate.
        strategy: Scaling strategy.
        warmup_steps: Number of warmup steps.
        cooldown_steps: Number of cooldown steps.
        last_epoch: The index of last epoch.
        strategies: Custom scaling strategies.

    Examples:
        >>> from danling.optim import LRScheduler
        >>> import torch
        >>> from torch import optim
        >>> optimizer = optim.SGD([{'params': torch.tensor([0])}], lr=1, momentum=0.9)
        >>> scheduler = LRScheduler(optimizer, steps=5, final_lr_ratio=1e-5, strategy='linear')
        >>> lrs = []
        >>> for epoch in range(5):
        ...     lrs.append(scheduler.get_lr()[0])
        ...     scheduler.step()
        >>> [round(lr, 10) for lr in lrs]
        [0.1, 0.01, 0.001, 0.0001, 1e-09]
        >>> scheduler = LRScheduler(optimizer, steps=5, final_lr_ratio=1e-5, strategy='cosine')
        >>> lrs = []
        >>> for epoch in range(5):
        ...     lrs.append(scheduler.get_lr()[0])
        ...     scheduler.step()
        >>> [round(lr, 10) for lr in lrs]
        [0.3330753446, 0.0187302031, 0.000533897, 3.00232e-05, 1e-09]
    """

    def __init__(  # pylint: disable=R0913
        self,
        optimizer: Optimizer,
        steps: int,
        final_lr_ratio: float = 100,
        final_lr: Optional[float] = None,
        min_lr: float = 1e-9,
        strategy: str = "cosine",
        warmup_steps: Optional[int] = None,
        cooldown_steps: Optional[int] = None,
        last_epoch: int = -1,
        strategies: Optional[Mapping[str, Callable]] = None,
        method: str = "percentile",
    ):
        if warmup_steps is None:
            warmup_steps = steps // 20
        elif warmup_steps > steps:
            raise ValueError(f"Warmup steps must be less than total steps, but got {warmup_steps} > {steps}")
        elif warmup_steps < 0:
            raise ValueError(f"Warmup steps must be positive, but got {warmup_steps}")
        if cooldown_steps is None:
            cooldown_steps = steps // 5
        elif cooldown_steps > steps:
            raise ValueError(f"Cooldown steps must be less than total steps, but got {cooldown_steps} > {steps}")
        elif cooldown_steps < 0:
            raise ValueError(f"Cooldown steps must be positive, but got {cooldown_steps}")
        if final_lr_ratio < 0:
            raise ValueError(f"`final_lr_ratio` must be positive, but got {final_lr_ratio}")
        if min_lr < 0:
            raise ValueError(f"`min_lr` must be positive, but got {min_lr}")
        if strategies is None:
            strategies = {}
        elif not isinstance(strategies, Mapping):
            raise TypeError(f"`strategies` should be a mapping, but got {type(strategies)}")
        self.strategies = LR_SCHEDULER_STRATEGIES.clone()
        self.strategies.update(strategies)
        if strategy not in self.strategies:
            raise ValueError(f"Scaling strategy must be one of {self.strategies.keys()}, but got {strategy}")
        self.steps = steps
        if final_lr is not None:
            warn("Argument `final_lr` is deprecated, use `final_lr_ratio` instead", DeprecationWarning)
        self.final_lr = final_lr
        self.final_lr_ratio = final_lr_ratio if final_lr_ratio < 1 else 1 / final_lr_ratio
        self.min_lr = min_lr
        self.strategy = strategy
        self.method = method
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.cooldown_steps_begin = self.steps - self.cooldown_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step_count = self._step_count  # type: ignore
        if step_count > self.steps or step_count < 0:
            warn(f"Step count {step_count} is out of range [0, {self.steps}]", RuntimeWarning)
        progress = np.clip(step_count / self.steps, 0.0, 1.0)
        warmup_ratio = step_count / self.warmup_steps if self.warmup_steps > 0 else 1.0
        cooldown_ratio = (
            1 - (step_count - self.cooldown_steps_begin) / self.cooldown_steps if self.cooldown_steps > 0 else 1.0
        )
        return [self._get_lr(lr, step_count, progress, warmup_ratio, cooldown_ratio) for lr in self.base_lrs]

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
        step_count = step_count or self._step_count  # type: ignore
        progress = progress or np.clip(step_count / self.steps, 0.0, 1.0)
        final_lr = self.final_lr or lr * self.final_lr_ratio
        ratio = self.strategies[self.strategy](self, progress)
        if ratio > 1 or ratio < 0:
            unclipped_ratio, ratio = ratio, np.clip(ratio, 0.0, 1.0)
            warn(f"Ratio {unclipped_ratio} is out of range [0, 1], clipping to {ratio}", RuntimeWarning)
        if method == "percentile":
            lr *= pow(final_lr / lr, ratio)
        elif method == "numerical":
            lr = (1 - ratio) * (lr - final_lr) + final_lr
        else:
            raise ValueError(f"Method must be one of ['percentile', 'numerical'], but got {method}")
        if self.warmup_steps > step_count > 0:
            warmup_ratio = warmup_ratio or step_count / self.warmup_steps
            lr = warmup_ratio * (lr - self.min_lr) + self.min_lr
        if step_count > self.cooldown_steps_begin and self.cooldown_steps > 0:
            cooldown_ratio = cooldown_ratio or 1 - (step_count - self.cooldown_steps_begin) / self.cooldown_steps
            lr = cooldown_ratio * (lr - self.min_lr) + self.min_lr
        return max(self.min_lr, lr)

    @LR_SCHEDULER_STRATEGIES.register
    def linear(self, progress: float) -> float:  # pylint: disable=C0116
        return progress

    @LR_SCHEDULER_STRATEGIES.register
    def cosine(self, progress: float) -> float:  # pylint: disable=C0116
        return 1 - ((1 + np.cos(np.pi * progress)) / 2)

    @LR_SCHEDULER_STRATEGIES.register
    def constant(self, progress: float) -> float:  # pylint: disable=W0613, C0116
        return 0.0
