from math import ceil

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class Scheduler(_LRScheduler):
    """
    Schedule the learning rate according to the learning rate schedule policy
    Support `cosine` and `linear` only
    Note that the scheduler schedule learning rate by steps
    """
    def __init__(
        self,
        optimizer,
        steps,
        lr_final=1e-6,
        lr_min=1e-6,
        policy='cosine',
        warmup_steps=10_000,
        accum_steps=1,
        last_epoch=-1,
    ):
        if policy not in ('cosine', 'linear'):
            raise ValueError(
                f'Only "cosine" or "linear" schedule policy are supported, but got {policy}')
        self.steps = ceil(steps / accum_steps)
        self.lr_final = lr_final
        self.lr_min = lr_min
        self.policy = policy
        self.method = getattr(self, self.policy)
        self.warmup_steps = ceil(warmup_steps / accum_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = (self._step_count - self.warmup_steps) / float(self.steps - self.warmup_steps)
        progress = np.clip(progress, 0.0, 1.0)
        ratio = self.method(progress)
        if self.warmup_steps:
            ratio = ratio * np.minimum(1., self._step_count / self.warmup_steps)
        return [max(self.lr_min, lr * ratio) for lr in self.base_lrs]

    def linear(self, progress):
        return self.lr_final + (1 - self.lr_final) * (1.0 - progress)

    def cosine(self, progress):
        return 0.5 * (1. + np.cos(np.pi * progress))

    def __repr__(self):
        return f'{self.policy.capitalize()}Scheduler'
