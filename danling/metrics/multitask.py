from __future__ import annotations

from chanfig import DefaultDict, FlatDict, NestedDict


class MultiTaskDict(NestedDict):
    r"""
    A `MultiTaskDict` for better multi-task support of `AverageMeter` and `Metrics`.
    """

    return_average: bool

    def __init__(self, *args, return_average: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setattr("return_average", return_average)

    def value(self):
        output = NestedDict({key: metric.value() for key, metric in self.all_items()})
        if self.getattr("return_average", False):
            average = DefaultDict(default_factory=list)
            for key, metric in output.all_items():
                average[key.rsplit(".", 1)[-1]].append(metric)
            output["average"] = FlatDict({key: sum(values) / len(values) for key, values in average.items()})
        return output

    def batch(self):
        output = NestedDict({key: metric.batch() for key, metric in self.all_items()})
        if self.getattr("return_average", False):
            average = DefaultDict(default_factory=list)
            for key, metric in output.all_items():
                average[key.rsplit(".", 1)[-1]].append(metric)
            output["average"] = FlatDict({key: sum(values) / len(values) for key, values in average.items()})
        return output

    def average(self):
        output = NestedDict({key: metric.average() for key, metric in self.all_items()})
        if self.getattr("return_average", False):
            average = DefaultDict(default_factory=list)
            for key, metric in output.all_items():
                average[key.rsplit(".", 1)[-1]].append(metric)
            output["average"] = FlatDict({key: sum(values) / len(values) for key, values in average.items()})
        return output

    @property
    def val(self) -> NestedDict[str, float]:
        return self.value()

    @property
    def bat(self) -> NestedDict[str, float]:
        return self.batch()

    @property
    def avg(self) -> NestedDict[str, float]:
        return self.average()

    def reset(self) -> None:
        for metric in self.all_values():
            metric.reset()

    def __format__(self, format_spec) -> str:
        return "\n".join(f"{key}: {metric.__format__(format_spec)}" for key, metric in self.all_items())
