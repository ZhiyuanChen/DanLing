class AverageMeter:
    r"""
    Computes and stores the average and current value.

    Attributes:
        val: Current value.
        avg: Average value.
        sum: Sum of values.
        count: Number of values.

    Examples:
    ```python
    >>> meter = AverageMeter()
    >>> meter.update(0.7)
    >>> meter.val
    0.7
    >>> meter.avg
    0.7
    >>> meter.update(0.9)
    >>> meter.val
    0.9
    >>> meter.avg
    0.8
    >>> meter.sum
    1.6
    >>> meter.count
    2
    >>> meter.reset()
    >>> meter.val
    0
    >>> meter.avg
    0

    ```
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

        Examples:
        ```python
        >>> meter = AverageMeter()
        >>> meter.update(0.7)
        >>> meter.val
        0.7
        >>> meter.avg
        0.7
        >>> meter.reset()
        >>> meter.val
        0
        >>> meter.avg
        0

        ```
        """

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        r"""
        Updates the average and current value in the meter.

        Args:
            val: Value to be added to the average.
            n: Number of values to be added.

        Examples:
        ```python
        >>> meter = AverageMeter()
        >>> meter.update(0.7)
        >>> meter.val
        0.7
        >>> meter.avg
        0.7
        >>> meter.update(0.9)
        >>> meter.val
        0.9
        >>> meter.avg
        0.8
        >>> meter.sum
        1.6
        >>> meter.count
        2

        ```
        """

        # pylint: disable=C0103

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
