from typing import List, Optional

import numpy as np
from torch.optim import Optimizer, lr_scheduler


class LRScheduler(lr_scheduler._LRScheduler):  # pylint: disable=W0212
    r"""
    General learning rate scheduler.
    """

    def __init__(  # pylint: disable=R0913
        self,
        optimizer: Optimizer,
        steps: int,
        final_lr: float = 1e-6,
        min_lr: float = 1e-6,
        strategy: Optional[str] = "cosine",
        warmup_steps: int = 10_000,
        last_epoch: int = -1,
    ):
        if strategy not in ("constant", "cosine", "linear", None):
            raise ValueError(f"Scaling strategy should be one of ('constant', 'linear', 'cosine'), but got {strategy}")
        self.steps = steps
        self.final_lr = final_lr
        self.min_lr = min_lr
        self.strategy = strategy
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore
        progress = (self._step_count - self.warmup_steps) / float(self.steps - self.warmup_steps)  # type: ignore
        progress = np.clip(progress, 0.0, 1.0)
        ratio = getattr(self, self.strategy)(progress) if self.strategy is not None else 1.0
        if self.warmup_steps:
            ratio = ratio * np.minimum(1.0, self._step_count / self.warmup_steps)  # type: ignore
        return [max(self.min_lr, lr * ratio) for lr in self.base_lrs]

    def linear(self, progress) -> float:  # pylint: disable=C0116
        return self.final_lr + (1 - self.final_lr) * (1.0 - progress)

    def cosine(self, progress) -> float:  # pylint: disable=C0116
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    def constant(self, progress) -> float:  # pylint: disable=W0613, C0116
        return 1.0
