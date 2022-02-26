from typing import Optional


class AverageMeter:
    """
    Compute and stores the average and current value
    """

    val: float
    avg: float
    sum: float
    count: float

    def __init__(self, batch_size: Optional[float] = 1):
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, batch_size: Optional[float] = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        self.avg = self.sum / self.count
