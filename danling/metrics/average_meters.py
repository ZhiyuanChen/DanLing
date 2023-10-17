from __future__ import annotations

from chanfig import DefaultDict, NestedDict


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

    val: float = 0
    avg: float = 0
    sum: float = 0
    count: int = 0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        r"""
        Resets the meter.

        Examples:
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
        """

        # pylint: disable=C0103

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format_spec) -> str:
        return f"{self.val.__format__(format_spec)} ({self.avg.__format__(format_spec)})"


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
    def val(self) -> NestedDict[str, float]:
        return NestedDict({key: meter.val for key, meter in self.items()})

    @property
    def avg(self) -> NestedDict[str, float]:
        return NestedDict({key: meter.avg for key, meter in self.items()})

    @property
    def sum(self) -> NestedDict[str, float]:
        return NestedDict({key: meter.sum for key, meter in self.items()})

    @property
    def count(self) -> NestedDict[str, int]:
        return NestedDict({key: meter.count for key, meter in self.items()})

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
