from __future__ import annotations

import logging
import logging.config
import os
from typing import Any, Callable, Mapping

try:
    from functools import cached_property  # type: ignore
except ImportError:
    from functools import lru_cache

    def cached_property(f):  # type: ignore
        return property(lru_cache()(f))


from chanfig import Config, FlatDict, NestedDict, Variable

from danling.metrics import AverageMeters
from danling.typing import File, PathStr
from danling.utils import catch, ensure_dir, load, save

from .runner_state import RunnerState


class RunnerBase:
    r"""
    Base class for all runners.

    `RunnerBase` is designed as a "dataclass".

    It defines all basic attributes and relevant properties such as `scores`, `progress`, etc.

    `RunnerBase` also defines basic IO operations such as `save`, `load`, `json`, `yaml`, etc.

    Attributes: Model:
        model (Callable):
        criterion (Callable):
        optimizer:
        scheduler:

    Attributes: Data:
        datasets (FlatDict): All datasets, should be in the form of ``{subset: dataset}``.
        datasamplers (FlatDict): All datasamplers, should be in the form of ``{subset: datasampler}``.
        dataloaders (FlatDict): All dataloaders, should be in the form of ``{subset: dataloader}``.
        batch_size (int, property): Number of samples per batch in train dataloader or the first dataloader.
        batch_size_equivalent (int, property): Total batch_size (`batch_size * world_size * accum_steps`).

    `datasets`, `datasamplers`, `dataloaders` should be a dict with the same keys.
    Their keys should be `split` (e.g. `train`, `val`, `test`).

    Attributes: Progress:
        progress (float, property): Running Progress, in `range(0, 1)`.

    Attributes: Results:
        latest_result (NestedDict, property): Most recent results,
            should be in the form of ``{subset: {index: score}}``.
        best_result (NestedDict, property): Best recent results, should be in the form of ``{subset: {index: score}}``.
        scores (List[float], property): All scores.
        latest_score (float, property): Most recent score.
        best_score (float, property): Best score.
        index_set (Optional[str]): The subset to calculate the core score.
            If is `None`, will use the last set of the result.
        index (str): The index to calculate the core score.
            Defaults to `"loss"`.
        is_best (bool, property): If `latest_score == best_score`.

    Attributes: IO:
        dir (str, property): Directory of the run.
            Defaults to `os.path.join(self.project_root, f"{self.name}-{self.id}")`.
        checkpoint_dir (str, property): Directory of checkpoints.
        log_path (str, property):  Path of log file.
        checkpoint_dir_name (str): The name of the directory under `runner.dir` to save checkpoints.
            Defaults to `"checkpoints"`.

    Attributes: Parallel Training:
        world_size (int, property): Number of processes.
        rank (int, property): Process index of all processes.
        local_rank (int, property): Process index of local processes.
        distributed (bool, property): If runner is running in distributed mode.
        is_main_process (bool, property): If current process is the main process of all processes.
        is_local_main_process (bool, property): If current process is the main process of local processes.

    Attributes: logging:
        logger:
        writer:

    Notes:
        The `RunnerBase` class is not intended to be used directly, nor to be directly inherit from.

        This is because `RunnerBase` is designed as a "dataclass",
        and is meant for demonstrating all attributes and properties only.

    See Also:
        [`RunnerState`][danling.runner.runner_state.RunnerState]: The runeer base that stores runtime information.
        [`BaseRunner`][danling.runner.BaseRunner]: The base runner class.
    """

    # pylint: disable=R0902, R0904
    # DO NOT set default value in class, as they won't be stored in `__dict__`.

    state: RunnerState

    model: Callable | None = None
    criterion: Callable | None = None
    optimizer: Any | None = None
    scheduler: Any | None = None

    datasets: FlatDict
    datasamplers: FlatDict
    dataloaders: FlatDict

    meters: AverageMeters | None = None
    logger: logging.Logger | None = None
    writer: Any | None = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.state = RunnerState(*args, **kwargs)
        self.meters = AverageMeters()
        self.datasets = FlatDict()
        self.datasamplers = FlatDict()
        self.dataloaders = FlatDict()

    @property
    def batch_size(self) -> int:
        r"""
        Batch size.

        Notes:
            If `train` is in `dataloaders`, then `batch_size` is the batch size of `train`.
            Otherwise, `batch_size` is the batch size of the first dataloader.

        Returns:
            (int):
        """

        if self.dataloaders:
            loader = self.dataloaders["train"] if "train" in self.dataloaders else next(iter(self.dataloaders.values()))
            return loader.batch_size
        raise AttributeError("batch_size could not be inferred, since no dataloaedr found.")

    @property
    def batch_size_equivalent(self) -> int:
        r"""
        Actual batch size.

        Returns:
            (int): `batch_size` * `world_size` * `accum_steps`
        """

        return self.batch_size * self.world_size * getattr(self, "accum_steps", 1)

    @property
    def accum_steps(self) -> int:
        r"""
        Accumulated steps.

        Returns:
            (int):
        """

        raise AttributeError("accum_steps is not defined.")

    @property
    def device(self) -> Any:
        r"""
        Device of runner.
        """

        raise NotImplementedError

    @property
    def world_size(self) -> int:
        r"""
        Number of processes.
        """

        return 1

    @property
    def rank(self) -> int:
        r"""
        Process index of all processes.
        """

        return 0

    @property
    def local_rank(self) -> int:
        r"""
        Process index of local processes.
        """

        return 0

    @property
    def distributed(self) -> bool:
        r"""
        If runner is running in distributed mode.
        """

        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        r"""
        If current process is the main process of all processes.
        """

        return self.rank == 0

    @property
    def is_local_main_process(self) -> bool:
        r"""
        If current process is the main process of local processes.
        """

        return self.local_rank == 0

    @catch
    def save(  # pylint: disable=W1113
        self, obj: Any, file: PathStr, main_process_only: bool = True, *args, **kwargs
    ) -> File:
        r"""
        Save any file with supported extensions.

        `Runner.save` internally calls `dl.save`,
        but with additional arguments to allow it save only on the main process.
        Moreover, any error raised by `Runner.save` will be caught and logged.
        """

        if main_process_only and self.is_main_process or not main_process_only:
            return save(obj, file, *args, **kwargs)
        return file

    @staticmethod
    def load(file: PathStr, *args, **kwargs) -> Any:  # pylint: disable=C0103
        r"""
        Load any file with supported extensions.

        `Runner.load` is identical to `dl.load`.
        """

        return load(file, *args, **kwargs)

    def dict(self, cls: Callable = dict) -> Mapping:
        r"""
        Convert state to Mapping.

        Args:
            cls: Target `clc to convert to.
        """

        # pylint: disable=C0103

        return self.state.dict(cls)

    @catch
    def json(self, file: File, main_process_only: bool = True, *args, **kwargs) -> None:  # pylint: disable=W1113
        r"""
        Dump Runner State to json file.
        """

        if main_process_only and self.is_main_process or not main_process_only:
            return self.state.json(file, *args, **kwargs)

    @classmethod
    def from_json(cls, file: File, *args, **kwargs) -> RunnerBase:
        r"""
        Construct Runner from json file.

        This function calls `self.from_jsons()` to construct object from json string.
        You may overwrite `from_jsons` in case something is not json serializable.
        """

        with FlatDict.open(file) as fp:  # pylint: disable=C0103
            return cls.from_jsons(fp.read(), *args, **kwargs)

    def jsons(self, *args, **kwargs) -> str:
        r"""
        Dump Runner State to json string.
        """

        return self.state.jsons(*args, **kwargs)

    @classmethod
    def from_jsons(cls, string: str, *args, **kwargs) -> RunnerBase:
        r"""
        Construct Runner from json string.
        """

        return cls(Config.from_jsons(string, *args, **kwargs))

    @catch
    def yaml(self, file: File, main_process_only: bool = True, *args, **kwargs) -> None:  # pylint: disable=W1113
        r"""
        Dump Runner State to yaml file.
        """

        if main_process_only and self.is_main_process or not main_process_only:
            return self.state.yaml(file, *args, **kwargs)

    @classmethod
    def from_yaml(cls, file: File, *args, **kwargs) -> RunnerBase:
        r"""
        Construct Runner from yaml file.

        This function calls `self.from_yamls()` to construct object from yaml string.
        You may overwrite `from_yamls` in case something is not yaml serializable.
        """

        with FlatDict.open(file) as fp:  # pylint: disable=C0103
            return cls.from_yamls(fp.read(), *args, **kwargs)

    def yamls(self, *args, **kwargs) -> str:
        r"""
        Dump Runner State to yaml string.
        """

        return self.state.yamls(*args, **kwargs)

    @classmethod
    def from_yamls(cls, string: str, *args, **kwargs) -> RunnerBase:
        r"""
        Construct Runner from yaml string.
        """

        return cls(Config.from_yamls(string, *args, **kwargs))

    @property
    def progress(self) -> float:
        r"""
        Training Progress.

        Returns:
            (float):

        Raises:
            RuntimeError: If no terminal is defined.
        """

        if hasattr(self.state, "iter_end"):
            return self.state.iters / self.state.iter_end
        if hasattr(self.state, "step_end"):
            return self.state.steps / self.state.step_end
        if hasattr(self.state, "epoch_end"):
            return self.state.epochs / self.state.epoch_end
        raise RuntimeError("DanLing cannot determine progress since no terminal is defined.")

    @property
    def best_fn(self) -> Callable:  # pylint: disable=C0103
        r"""
        Function to determine the best score from a list of scores.

        Subclass can override this method to accommodate needs, such as `min`.

        Returns:
            (callable): `max`
        """

        return max

    @property
    def latest_result(self) -> NestedDict | None:
        r"""
        Latest result.
        """

        return self.state.results[-1] if self.state.results else None

    @property
    def best_result(self) -> NestedDict | None:
        r"""
        Best result.
        """
        if not self.state.results:
            return None
        return self.state.results[-1 - self.scores[::-1].index(self.best_score)]  # type: ignore

    @property
    def scores(self) -> list[float]:
        r"""
        All scores.

        Scores are extracted from results by `index_set` and `runner.index`,
        following `[r[index_set][self.state.index] for r in self.state.results]`.

        By default, `index_set` points to `self.state.index_set` and is set to `val`,
        if `self.state.index_set` is not set, it will be the last key of the last result.

        Scores are considered as the index of the performance of the model.
        It is useful to determine the best model and the best hyper-parameters.
        """

        if not self.state.results:
            return []
        index_set = self.state.index_set or next(reversed(self.state.results[-1]))
        return [r[index_set][self.state.index] for r in self.state.results]

    @property
    def latest_score(self) -> float | None:
        r"""
        Latest score.
        """

        return self.scores[-1] if self.state.results else None

    @property
    def best_score(self) -> float | None:
        r"""
        Best score.
        """

        return self.best_fn(self.scores) if self.results else None

    @property
    def is_best(self) -> bool:
        r"""
        If current epoch is the best epoch.
        """

        try:
            return abs(self.latest_score - self.best_score) < 1e-7  # type: ignore
        except TypeError:
            return True

    @cached_property
    @ensure_dir
    def dir(self) -> str:
        r"""
        Directory of the run.
        """

        if "dir" in self.state:
            return self.state.dir
        return os.path.join(self.project_root, f"{self.name}-{self.id}")

    @cached_property
    def log_path(self) -> str:
        r"""
        Path of log file.
        """

        if "log_path" in self.state:
            return self.state.log_path
        return os.path.join(self.dir, "run.log")

    @cached_property
    @ensure_dir
    def checkpoint_dir(self) -> str:
        r"""
        Directory of checkpoints.
        """

        if "checkpoint_dir" in self.state:
            return self.state.checkpoint_dir
        return os.path.join(self.dir, self.checkpoint_dir_name)

    def __getattribute__(self, name) -> Any:
        if name in ("__class__", "__dict__"):
            return super().__getattribute__(name)
        if name in dir(self):
            return super().__getattribute__(name)
        if name in self.__dict__:
            return self.__dict__[name]
        if "state" in self and name in self.state:
            return self.state[name]
        return super().__getattribute__(name)

    def __getattr__(self, name) -> Any:
        if "state" not in self:
            raise RuntimeError("Runner is not initialised yet.")
        if name in dir(self.state):
            return getattr(self.state, name)
        raise super().__getattribute__(name)

    def __setattr__(self, name, value) -> None:
        if name in self.__dict__ and isinstance(self.__dict__[name], Variable):
            self.__dict__[name].set(value)
        else:
            self.__dict__[name] = value

    def __contains__(self, name) -> bool:
        return name in dir(self) or ("state" in self.__dict__ and name in dir(self.state))

    def __repr__(self):
        lines = []
        for key, value in self.__dict__.items():
            value_str = repr(value)
            value_str = self._add_indent(value_str)
            lines.append("(" + key + "): " + value_str)

        main_str = self.__class__.__name__ + "("
        if lines:
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def _add_indent(self, text):
        lines = text.split("\n")
        # don't do anything for single-line stuff
        if len(lines) == 1:
            return text
        first = lines.pop(0)
        # add 2 spaces to each line but the first
        lines = [(2 * " ") + line for line in lines]
        lines = "\n".join(lines)
        lines = first + "\n" + lines
        return lines
