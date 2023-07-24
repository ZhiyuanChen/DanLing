from chanfig import DefaultDict, FlatDict

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
        kwargs.setdefault("default_factory", AverageMeter)
        super().__init__(*args, **kwargs)

    @property
    def val(self) -> FlatDict[str, float]:
        return FlatDict({key: meter.val for key, meter in self.items()})

    @property
    def avg(self) -> FlatDict[str, float]:
        return FlatDict({key: meter.avg for key, meter in self.items()})

    @property
    def sum(self) -> FlatDict[str, float]:
        return FlatDict({key: meter.sum for key, meter in self.items()})

    @property
    def count(self) -> FlatDict[str, int]:
        return FlatDict({key: meter.count for key, meter in self.items()})

    def reset(self) -> None:
        r"""
        Resets all meters.

        Examples:
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
        """

        for meter in self.values():
            meter.reset()

    def update(self, val, n: int = 1) -> None:  # type: ignore # pylint: disable=W0237
        r"""
        Updates the average and current value in all meters.

        Args:
            val: Value to be added to the average.
            n: Number of values to be added.

        Note:
            This function is **NOT** recommended to use, as it alters all meters in the bank.

        Examples:
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
        """

        for meter in self.values():
            meter.update(val, n)

    def __format__(self, format_spec) -> str:
        return "\n".join(f"{key}: {meter.__format__(format_spec)}" for key, meter in self.items())
