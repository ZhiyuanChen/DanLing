from chanfig import DefaultDict

from .average_meter import AverageMeter


class AverageMeters(DefaultDict):
    r"""
    A `DefaultDict` for `AverageMeter`.


    Examples:
    ```python
    >>> meters = AverageMeters()
    >>> meters.loss.reset()
    >>> meters.loss.update(0.7)
    >>> meters.loss.val
    0.7
    >>> meters.loss.avg
    0.7
    >>> meters.update(0.9)
    >>> meters.loss.val
    0.9
    >>> meters.loss.avg
    0.8
    >>> meters.loss.sum
    1.6
    >>> meters.loss.count
    2
    >>> meters.reset()
    >>> meters.loss.val
    0
    >>> meters.loss.avg
    0

    ```
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(default_factory=AverageMeter, *args, **kwargs)

    def reset(self) -> None:
        r"""
        Resets all meters.

        Examples:
        ```python
        >>> meters = AverageMeters()
        >>> meters.loss.update(0.7)
        >>> meters.loss.val
        0.7
        >>> meters.loss.avg
        0.7
        >>> meters.reset()
        >>> meters.loss.val
        0
        >>> meters.loss.avg
        0

        ```
        """

        for meter in self.values():
            meter.reset()

    def update(self, val, n: int = 1) -> None:  # pylint: disable=W0237
        r"""
        Updates the average and current value in all meters.

        Args:
            val: Value to be added to the average.
            n: Number of values to be added.

        Note:
            This function is **NOT** recommended to use, as it alters all meters in the bank.

        Examples:
        ```python
        >>> meters = AverageMeters()
        >>> meters.loss.update(0.7)
        >>> meters.loss.val
        0.7
        >>> meters.loss.avg
        0.7
        >>> meters.update(0.9)
        >>> meters.loss.val
        0.9
        >>> meters.loss.avg
        0.8
        >>> meters.loss.sum
        1.6
        >>> meters.loss.count
        2

        ```
        """

        for meter in self.values():
            meter.update(val, n)
