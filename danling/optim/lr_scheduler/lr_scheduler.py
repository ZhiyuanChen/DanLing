from typing import List

import numpy as np
from torch.optim import Optimizer, lr_scheduler


class LRScheduler(lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        steps: int,
        final_lr: float = 1e-6,
        min_lr: float = 1e-6,
        strategy: str = "cosine",
        warmup_steps: int = 10_000,
        last_epoch: int = -1,
    ):
        if strategy not in ("constant", "cosine", "linear"):
            raise ValueError(f"Warmup strategy should be one of ('constant', 'linear', 'cosine'), but got {strategy}")
        self.steps = steps
        self.final_lr = final_lr
        self.min_lr = min_lr
        self.strategy = strategy
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        progress = (self._step_count - self.warmup_steps) / float(self.steps - self.warmup_steps)
        progress = np.clip(progress, 0.0, 1.0)
        ratio = getattr(self, self.strategy)(progress)
        if self.warmup_steps:
            ratio = ratio * np.minimum(1.0, self._step_count / self.warmup_steps)
        return [max(self.min_lr, lr * ratio) for lr in self.base_lrs]

    def linear(self, progress) -> float:
        return self.final_lr + (1 - self.final_lr) * (1.0 - progress)

    def cosine(self, progress) -> float:
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    def constant(self, progress) -> float:  # pylint: disable=W0613
        return 1.0
