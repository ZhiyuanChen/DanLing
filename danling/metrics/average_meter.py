class AverageMeter:
    r"""
    Computes and stores the average and current value.
    """

    val: int = 0
    avg: float = 0
    sum: int = 0
    count: int = 0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        r"""
        Resets the meter.
        """

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        r"""
        Updates the average and current value in the meter.

        Parameters
        ----------
        val: int
            Value to be added to the average.
        n: int = 1
            Number of values to be added.
        """

        # pylint: disable=C0103

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
