from __future__ import annotations

from typing import Any, Dict

from chanfig import NestedDict
from torch import distributed as dist

from .multitask import MultiTaskDict
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

    def value(self):
        return self.val

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


class MultiTaskAverageMeter(MultiTaskDict):
    """
    Examples:
        >>> meters = MultiTaskAverageMeter()
        >>> meters.update({"loss": 0.6, "dataset1.cls.auroc": 0.7, "dataset1.reg.r2": 0.8, "dataset2.r2": 0.9})
        >>> print(f"{meters:.4f}")
        loss: 0.6000 (0.6000)
        dataset1.cls.auroc: 0.7000 (0.7000)
        dataset1.reg.r2: 0.8000 (0.8000)
        dataset2.r2: 0.9000 (0.9000)
        >>> meters.update({"loss": {"value": 0.9, "n": 1}})
        >>> print(f"{meters:.4f}")
        loss: 0.9000 (0.7500)
        dataset1.cls.auroc: 0.7000 (0.7000)
        dataset1.reg.r2: 0.8000 (0.8000)
        dataset2.r2: 0.9000 (0.9000)
        >>> meters.sum.dict()
        {'loss': 1.5, 'dataset1': {'cls': {'auroc': 0.7}, 'reg': {'r2': 0.8}}, 'dataset2': {'r2': 0.9}}
        >>> meters.count.dict()
        {'loss': 2, 'dataset1': {'cls': {'auroc': 1}, 'reg': {'r2': 1}}, 'dataset2': {'r2': 1}}
        >>> meters.reset()
        >>> print(f"{meters:.4f}")
        loss: 0.0000 (nan)
        dataset1.cls.auroc: 0.0000 (nan)
        dataset1.reg.r2: 0.0000 (nan)
        dataset2.r2: 0.0000 (nan)
        >>> meters = MultiTaskAverageMeter(return_average=True)
        >>> meters.update({"loss": 0.6, "dataset1.a.auroc": 0.7, "dataset1.b.auroc": 0.8, "dataset2.auroc": 0.9})
        >>> print(f"{meters:.4f}")
        loss: 0.6000 (0.6000)
        dataset1.a.auroc: 0.7000 (0.7000)
        dataset1.b.auroc: 0.8000 (0.8000)
        dataset2.auroc: 0.9000 (0.9000)
        >>> meters.update({"loss": 0.9, "dataset1.a.auroc": 0.8, "dataset1.b.auroc": 0.9, "dataset2.auroc": 1.0})
        >>> print(f"{meters:.4f}")
        loss: 0.9000 (0.7500)
        dataset1.a.auroc: 0.8000 (0.7500)
        dataset1.b.auroc: 0.9000 (0.8500)
        dataset2.auroc: 1.0000 (0.9500)
    """

    @property
    def sum(self) -> NestedDict[str, float]:
        return NestedDict({key: meter.sum for key, meter in self.all_items()})

    @property
    def count(self) -> NestedDict[str, int]:
        return NestedDict({key: meter.count for key, meter in self.all_items()})

    def update(self, values: Dict, *, n: int = 1) -> None:  # pylint: disable=W0237
        r"""
        Updates the average and current value in all meters.

        Args:
            values: Dict of values to be added to the average.
            n: Number of values to be added.

        Raises:
            ValueError: If the value is not an instance of (int, float, Mapping).

        Examples:
            >>> meters = MultiTaskAverageMeter()
            >>> meters.update({"loss": 0.6, "dataset1.cls.auroc": 0.7, "dataset1.reg.r2": 0.8, "dataset2.r2": 0.9})
            >>> meters.sum.dict()
            {'loss': 0.6, 'dataset1': {'cls': {'auroc': 0.7}, 'reg': {'r2': 0.8}}, 'dataset2': {'r2': 0.9}}
            >>> meters.count.dict()
            {'loss': 1, 'dataset1': {'cls': {'auroc': 1}, 'reg': {'r2': 1}}, 'dataset2': {'r2': 1}}
            >>> meters.update({"loss": {"value": 0.9, "n": 1}})
            >>> meters.sum.dict()
            {'loss': 1.5, 'dataset1': {'cls': {'auroc': 0.7}, 'reg': {'r2': 0.8}}, 'dataset2': {'r2': 0.9}}
            >>> meters.count.dict()
            {'loss': 2, 'dataset1': {'cls': {'auroc': 1}, 'reg': {'r2': 1}}, 'dataset2': {'r2': 1}}
            >>> meters.update({"loss": 0.8, "dataset1.cls.auroc": 0.9, "dataset1.reg.r2": 0.8, "dataset2.r2": 0.7})
            >>> meters.sum.dict()
            {'loss': 2.3, 'dataset1': {'cls': {'auroc': 1.6}, 'reg': {'r2': 1.6}}, 'dataset2': {'r2': 1.6}}
            >>> meters.count.dict()
            {'loss': 3, 'dataset1': {'cls': {'auroc': 2}, 'reg': {'r2': 2}}, 'dataset2': {'r2': 2}}
            >>> meters.update({"dataset1.cls.auroc": 0.7, "dataset1.reg.r2": 0.7, "dataset2.r2": 0.9})
            >>> meters.sum.dict()
            {'loss': 2.3, 'dataset1': {'cls': {'auroc': 2.3}, 'reg': {'r2': 2.3}}, 'dataset2': {'r2': 2.5}}
            >>> meters.count.dict()
            {'loss': 3, 'dataset1': {'cls': {'auroc': 3}, 'reg': {'r2': 3}}, 'dataset2': {'r2': 3}}
            >>> meters.update({"dataset1": {"cls.auroc": 0.9}, "dataset1.reg.r2": 0.8, "dataset2.r2": 0.9})
            Traceback (most recent call last):
            ValueError: Expected values to be int, float, or a flat dictionary, but got <class 'dict'>
            This is likely due to nested dictionary in the values.
            Nested dictionaries cannot be processed due to the method's design, which uses Mapping to pass both value and count ('n'). Ensure your input is a flat dictionary or a single value.
            >>> meters.update(dict(loss=""))
            Traceback (most recent call last):
            ValueError: Expected values to be int, float, or a flat dictionary, but got <class 'str'>
        """  # noqa: E501

        for meter, value in values.items():
            if isinstance(value, (int, float)):
                self[meter].update(value, n)
            elif isinstance(value, Dict):
                value.setdefault("n", n)
                try:
                    self[meter].update(**value)
                except TypeError:
                    raise ValueError(
                        f"Expected values to be int, float, or a flat dictionary, but got {type(value)}\n"
                        "This is likely due to nested dictionary in the values.\n"
                        "Nested dictionaries cannot be processed due to the method's design, which uses Mapping "
                        "to pass both value and count ('n'). Ensure your input is a flat dictionary or a single value."
                    ) from None
            else:
                raise ValueError(f"Expected values to be int, float, or a flat dictionary, but got {type(value)}")

    # eval hack, as the default_factory must not be set to make `NestedDict` happy
    # this have some side effects, it will break attribute style intermediate nested dict auto creation
    # but everything has a price
    def get(self, name: Any, default=None) -> Any:
        if not name.startswith("_") and not name.endswith("_"):
            return self.setdefault(name, AverageMeter())
        return super().get(name, default)
