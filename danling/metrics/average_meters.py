from __future__ import annotations

from typing import Dict

from chanfig import DefaultDict, NestedDict
from torch import distributed as dist

from .utils import get_world_size


class AverageMeter:
    r"""
    Computes and stores the average and current value.

    Attributes:
        val: Results of current batch on current device.
        bat: Results of current batch on all devices.
        avg: Results of all results on all devices.
        sum: Sum of values.
        count: Number of values.

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
        >>> meter.reset()
        >>> meter.val
        0
        >>> meter.avg
        nan
    """

    val: float = 0
    n: float = 1
    sum: float = 0
    count: float = 0

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
            nan
        """

        self.val = 0
        self.n = 1
        self.sum = 0
        self.count = 0

    def update(self, value, n: float = 1) -> None:
        r"""
        Updates the average and current value in the meter.

        Args:
            value: Value to be added to the average.
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

        self.val = value
        self.n = n
        self.sum += value * n
        self.count += n

    def batch(self):
        world_size = get_world_size()
        if world_size == 1:
            return self.val / self.n if self.n != 0 else float("nan")
        synced_tuple = [None for _ in range(world_size)]
        dist.all_gather_object(synced_tuple, (self.val * self.n, self.n))
        val, n = zip(*synced_tuple)
        count = sum(n)
        if count == 0:
            return float("nan")
        return sum(val) / count

    def average(self):
        world_size = get_world_size()
        if world_size == 1:
            return self.sum / self.count if self.count != 0 else float("nan")
        synced_tuple = [None for _ in range(world_size)]
        dist.all_gather_object(synced_tuple, (self.sum, self.count))
        val, n = zip(*synced_tuple)
        count = sum(n)
        if count == 0:
            return float("nan")
        return sum(val) / count

    @property
    def bat(self):
        return self.batch()

    @property
    def avg(self):
        return self.average()

    def __format__(self, format_spec) -> str:
        return f"{self.val.__format__(format_spec)} ({self.avg.__format__(format_spec)})"


class AverageMeters(DefaultDict):
    r"""
    A `DefaultDict` for `AverageMeter`.

    Examples:
        >>> meters = AverageMeters()
        >>> meters.loss.reset()
        >>> meters.update({"loss": 0.7})
        >>> meters.val
        NestedDict(
          ('loss'): 0.7
        )
        >>> meters.avg
        NestedDict(
          ('loss'): 0.7
        )
        >>> meters.update({"loss": {"value": 0.9, "n": 1}})
        >>> meters.val
        NestedDict(
          ('loss'): 0.9
        )
        >>> meters.avg
        NestedDict(
          ('loss'): 0.8
        )
        >>> meters.sum
        NestedDict(
          ('loss'): 1.6
        )
        >>> meters.count
        NestedDict(
          ('loss'): 2
        )
        >>> meters.reset()
        >>> meters.val
        NestedDict(
          ('loss'): 0
        )
        >>> meters.avg
        NestedDict(
          ('loss'): nan
        )
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("default_factory", AverageMeter)
        super().__init__(*args, **kwargs)

    def batch(self):
        return NestedDict({key: meter.batch() for key, meter in self.items()})

    def average(self):
        return NestedDict({key: meter.average() for key, meter in self.items()})

    @property
    def bat(self) -> NestedDict[str, float]:
        return NestedDict({key: meter.bat for key, meter in self.items()})

    @property
    def avg(self) -> NestedDict[str, float]:
        return NestedDict({key: meter.avg for key, meter in self.items()})

    @property
    def val(self) -> NestedDict[str, float]:
        return NestedDict({key: meter.val for key, meter in self.items()})

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
            nan
        """

        for meter in self.values():
            meter.reset()

    def update(self, values: Dict, *, n: int = 1) -> None:  # pylint: disable=W0237
        r"""
        Updates the average and current value in all meters.

        Args:
            values: Dict of values to be added to the average.
            n: Number of values to be added.

        Raises:
            ValueError: If the value is not an instance of (int, float, Mapping).

        Examples:
            >>> meters = AverageMeters()
            >>> meters.loss.update(0.7)
            >>> meters.loss.val
            0.7
            >>> meters.loss.avg
            0.7
            >>> meters.update(dict(loss=0.9))
            >>> meters.loss.val
            0.9
            >>> meters.loss.avg
            0.8
            >>> meters.loss.sum
            1.6
            >>> meters.loss.count
            2
            >>> meters.update(dict(loss=""))
            Traceback (most recent call last):
            ValueError: Value for AverageMeters should be of type inf, float or Mapping, buf got <class 'str'>
        """

        for meter, value in values.items():
            if isinstance(value, (int, float)):
                self[meter].update(value, n)
            elif isinstance(value, Dict):
                value.setdefault("n", n)
                self[meter].update(**value)
            else:
                raise ValueError(
                    f"Value for AverageMeters should be of type inf, float or Mapping, buf got {type(value)}"
                )

    def __format__(self, format_spec) -> str:
        return "\n".join(f"{key}: {meter.__format__(format_spec)}" for key, meter in self.items())
