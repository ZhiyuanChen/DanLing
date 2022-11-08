import math
from typing import Optional

import numpy as np
from torch.optim import Optimizer, lr_scheduler


class LRScheduler(lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        steps: int,
        final_lr: Optional[float] = 1e-6,
        min_lr: Optional[float] = 1e-6,
        strategy: Optional[str] = "cosine",
        warmup_steps: Optional[int] = 10_000,
        last_epoch: Optional[int] = -1,
    ):
        if strategy not in ("constant", "cosine", "linear"):
            raise ValueError("Only 'constant' or 'linear' warmup_method accepted" "got {}".format(strategy))
        self.steps = steps
        self.final_lr = final_lr
        self.min_lr = min_lr
        self.strategy = strategy
        self.warmup_steps = warmup_steps
        super(LRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = (self._step_count - self.warmup_steps) / float(self.steps - self.warmup_steps)
        progress = np.clip(progress, 0.0, 1.0)
        ratio = getattr(self, self.strategy)(progress)
        if self.warmup_steps:
            ratio = ratio * np.minimum(1.0, self._step_count / self.warmup_steps)
        return [max(self.min_lr, lr * ratio) for lr in self.base_lrs]

    def linear(self, progress):
        return self.final_lr + (1 - self.final_lr) * (1.0 - progress)

    def cosine(self, progress):
        return 0.5 * (1.0 + np.cos(np.pi * progress))
