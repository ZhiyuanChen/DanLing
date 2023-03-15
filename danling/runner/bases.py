from __future__ import annotations

import logging
import logging.config
import os
from datetime import datetime
from json import dumps as json_dumps
from random import randint
from typing import IO, Any, Callable, List, Mapping, Optional, Union
from uuid import UUID, uuid5

from chanfig import Config, FlatDict, NestedDict, Variable
from chanfig.utils import JsonEncoder, YamlDumper
from git.exc import InvalidGitRepositoryError
from git.repo import Repo
from yaml import dump as yaml_dump

from danling.utils import base62, catch, ensure_dir, is_json_serializable, load, save

NUMPY_AVAILABLE = True
try:
    import numpy as np
except ImportError:
    NUMPY_AVAILABLE = False

PathStr = Union[os.PathLike, str, bytes]
File = Union[PathStr, IO]

DEFAULT_EXPERIMENT_NAME = "DanLing"
DEFAULT_EXPERIMENT_ID = "xxxxxxxxxxxxxxxx"


class RunnerBase:
    r"""
    Base class for all runners.

    `RunnerBase` is designed as a "dataclass".

    It defines all basic attributes and relevant properties such as `scores`, `progress`, etc.

    `RunnerBase` also defines basic IO operations such as `save`, `load`, `json`, `yaml`, etc.

    Attributes: General:
        id: `f"{self.experiment_id:.4}{self.run_id:.4}{time_str}"`.
        uuid: `uuid5(self.run_id, self.id)`.
        name: `f"{self.experiment_name}-{self.run_name}"`.
        experiment_id: git hash of the current HEAD.
            Defaults to `"xxxxxxxxxxxxxxxx"` if Runner not under a git repo.
        experiment_uuid: UUID of `self.experiment_id`.
            Defaults to `UUID('78787878-7878-7878-7878-787878787878')` if Runner not under a git repo.
        experiment_name: Defaults to `"DanLing"`.
        run_id: hex of `self.run_uuid`.
        run_uuid: `uuid5(self.experiment_id, config.jsons())`.
        run_name: Defaults to `"DanLing"`.
        seed (int): Defaults to `randint(0, 2**32 - 1)`.
        deterministic (bool): Ensure [deterministic](https://pytorch.org/docs/stable/notes/randomness.html) operations.
            Defaults to `False`.

    Attributes: Progress:
        iters (int): The number of data samples processed.
            equals to `steps` when `batch_size = 1`.
        steps (int): The number of `step` calls.
        epochs (int): The number of complete passes over the datasets.
        iter_end (int): End running iters.
            Note that `step_end` not initialised since this variable may not apply to some Runners.
        step_end (int): End running steps.
            Note that `step_end` not initialised since this variable may not apply to some Runners.
        epoch_end (int): End running epochs.
            Note that `epoch_end` not initialised since this variable may not apply to some Runners.
        progress (float, property): Running Progress, in `range(0, 1)`.

    In general you should only use one of `iter_end`, `step_end`, `epoch_end` to indicate the length of running.

    Attributes: Model:
        model (Callable):
        criterion (Callable):
        optimizer:
        scheduler:

    Attributes: Data:
        datasets (FlatDict): All datasets, should be in the form of ``{subset: dataset}``.
        datasamplers (FlatDict): All datasamplers, should be in the form of ``{subset: datasampler}``.
        dataloaders (FlatDict): All dataloaders, should be in the form of ``{subset: dataloader}``.
        batch_size (int): Number of samples per batch.
        batch_size_equivalent (int, property): Total batch_size (`batch_size * world_size * accum_steps`).

    `datasets`, `datasamplers`, `dataloaders` should be a dict with the same keys.
    Their keys should be `split` (e.g. `train`, `val`, `test`).

    Attributes: Results:
        results (List[NestedDict]): All results, should be in the form of ``[{subset: {index: score}}]``.
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

    `results` should be a list of `result`.
    `result` should be a dict with the same `split` as keys, like `dataloaders`.
    A typical `result` might look like this:
    ```python
    {
        "train": {
            "loss": 0.1,
            "accuracy": 0.9,
        },
        "val": {
            "loss": 0.2,
            "accuracy": 0.8,
        },
        "test": {
            "loss": 0.3,
            "accuracy": 0.7,
        },
    }
    ```

    `scores` are usually a list of `float`, and are dynamically extracted from `results` by `index_set` and `index`.
    If `index_set = "val"`, `index = "accuracy"`, then `scores = 0.9`.

    Attributes: Parallel Training:
        world_size (int, property): Number of processes.
        rank (int, property): Process index of all processes.
        local_rank (int, property): Process index of local processes.
        distributed (bool, property): If runner is running in distributed mode.
        is_main_process (bool, property): If current process is the main process of all processes.
        is_local_main_process (bool, property): If current process is the main process of local processes.

    Attributes: IO:
        project_root (str): The root directory for all experiments.
            Defaults to `"experiments"`.
        dir (str, property): Directory of the run.
            Defaults to `os.path.join(self.project_root, f"{self.name}-{self.id}")`.
        checkpoint_dir (str, property): Directory of checkpoints.
        log_path (str, property):  Path of log file.
        checkpoint_dir_name (str): The name of the directory under `runner.dir` to save checkpoints.
            Defaults to `"checkpoints"`.

    `project_root` is the root directory of all **Experiments**, and should be consistent across the **Project**.

    `dir` is the directory of a certain **Run**.

    There is no attributes/properties for **Group** and **Experiment**.

    `checkpoint_dir_name` is relative to `dir`, and is passed to generate `checkpoint_dir`
    (`checkpoint_dir = os.path.join(dir, checkpoint_dir_name)`).
    In practice, `checkpoint_dir_name` is rarely called.

    Attributes: logging:
        log (bool): Whether to log the outputs.
            Defaults to `True`.
        logger:
        tensorboard (bool): Whether to use `tensorboard`.
            Defaults to `False`.
        writer:

    Notes:
        The `RunnerBase` class is not intended to be used directly, nor to be directly inherit from.

        This is because `RunnerBase` is designed as a "dataclass",
        and is meant for demonstrating all attributes and properties only.

    See Also:
        [`BaseRunner`][danling.runner.BaseRunner]: The base runner class.
    """

    # pylint: disable=R0902, R0904
    # DO NOT set default value in class, as they won't be stored in `__dict__`.

    run_id: str
    run_uuid: UUID
    run_name: str = "Run"
    experiment_id: str = DEFAULT_EXPERIMENT_ID
    experiment_uuid: UUID
    experiment_name: str = DEFAULT_EXPERIMENT_NAME

    seed: int
    deterministic: bool

    iters: int
    steps: int
    epochs: int
    # iter_begin: int  # Deprecated
    # step_begin: int  # Deprecated
    # epoch_begin: int  # Deprecated
    iter_end: int
    step_end: int
    epoch_end: int

    model: Optional[Callable] = None
    criterion: Optional[Callable] = None
    optimizer: Optional[Any] = None
    scheduler: Optional[Any] = None

    datasets: FlatDict
    datasamplers: FlatDict
    dataloaders: FlatDict

    batch_size: int

    results: List[NestedDict]
    index_set: Optional[str]
    index: str

    project_root: str = "experiments"
    checkpoint_dir_name: str = "checkpoints"
    log: bool = True
    logger: Optional[logging.Logger] = None
    tensorboard: bool = False
    writer: Optional[Any] = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        # Init attributes that should be kept in checkpoint inside `__init__`.
        # Note that attributes should be init before redefine `self.__dict__`.
        try:
            self.experiment_id = Repo(search_parent_directories=True).head.object.hexsha
        except InvalidGitRepositoryError:
            pass
        self.experiment_uuid = UUID(bytes=bytes(self.experiment_id.ljust(16, "x")[:16], encoding="ascii"))
        self.deterministic = False
        self.iters = 0
        self.steps = 0
        self.epochs = 0
        self.batch_size = 1
        self.seed = randint(0, 2**32 - 1)
        self.datasets = FlatDict()
        self.datasamplers = FlatDict()
        self.dataloaders = FlatDict()
        self.results = []
        self.index_set = None
        self.index = "loss"
        if len(args) == 1 and isinstance(args[0], FlatDict) and not kwargs:
            args, kwargs = (), args[0]
        self.__dict__.update(NestedDict(*args, **kwargs))
        self.run_uuid = uuid5(self.experiment_uuid, self.yamls())
        self.run_id = self.run_uuid.hex
        time = datetime.now()
        time_tuple = time.isocalendar()[1:] + (time.hour, time.minute, time.second, time.microsecond)
        time_str = "".join(base62.encode(i) for i in time_tuple)
        self.id = f"{self.experiment_id:.5}{self.run_id:.4}{time_str}"  # pylint: disable=C0103
        self.uuid = uuid5(self.run_uuid, self.id)
        self.name = f"{self.experiment_name}-{self.run_name}"
        # self.__dict__.update(NestedDict(**self.__dict__))

    @property
    def progress(self) -> float:
        r"""
        Training Progress.

        Returns:
            (float):

        Raises:
            RuntimeError: If no terminal is defined.
        """

        if hasattr(self, "iter_end"):
            return self.iters / self.iter_end
        if hasattr(self, "step_end"):
            return self.steps / self.step_end
        if hasattr(self, "epoch_end"):
            return self.epochs / self.epoch_end
        raise RuntimeError("DanLing cannot determine progress since no terminal is defined.")

    @property
    def batch_size_equivalent(self) -> int:
        r"""
        Actual batch size.

        Returns:
            (int): `batch_size` * `world_size` * `accum_steps`
        """

        return self.batch_size * self.world_size * getattr(self, "accum_steps", 1)

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
    def latest_result(self) -> Optional[NestedDict]:
        r"""
        Latest result.
        """

        return self.results[-1] if self.results else None

    @property
    def best_result(self) -> Optional[NestedDict]:
        r"""
        Best result.
        """

        return self.results[-1 - self.scores[::-1].index(self.best_score)] if self.results else None  # type: ignore

    @property
    def scores(self) -> List[float]:
        r"""
        All scores.

        Scores are extracted from results by `index_set` and `runner.index`,
        following `[r[index_set][self.index] for r in self.results]`.

        By default, `index_set` points to `self.index_set` and is set to `val`,
        if `self.index_set` is not set, it will be the last key of the last result.

        Scores are considered as the index of the performance of the model.
        It is useful to determine the best model and the best hyper-parameters.
        """

        if not self.results:
            return []
        index_set = self.index_set or next(reversed(self.results[-1]))
        return [r[index_set][self.index] for r in self.results]

    @property
    def latest_score(self) -> Optional[float]:
        r"""
        Latest score.
        """

        return self.scores[-1] if self.results else None

    @property
    def best_score(self) -> Optional[float]:
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

    @property  # type: ignore
    @ensure_dir
    def dir(self) -> str:
        r"""
        Directory of the run.
        """

        return os.path.join(self.project_root, f"{self.name}-{self.id}")

    @property
    def log_path(self) -> str:
        r"""
        Path of log file.
        """

        return os.path.join(self.dir, "run.log")

    @property  # type: ignore
    @ensure_dir
    def checkpoint_dir(self) -> str:
        r"""
        Directory of checkpoints.
        """

        return os.path.join(self.dir, self.checkpoint_dir_name)

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

        `Runner.load` is identical to `dl.save`.
        """

        return load(file, *args, **kwargs)

    def dict(self, cls: Callable = dict, only_json_serializable: bool = True) -> Mapping:
        r"""
        Convert config to Mapping.

        Note that all non-json-serializable objects will be removed.

        Args:
            cls: Target `clc to convert to.
            only_json_serializable: If only json serializable objects should be kept.
        """

        # pylint: disable=C0103

        ret = cls()
        for k, v in self.__dict__.items():
            if isinstance(v, FlatDict):
                v = v.dict(cls)
            if NUMPY_AVAILABLE:
                if isinstance(v, np.integer):
                    v = int(v)
                elif isinstance(v, np.floating):
                    v = float(v)
            if not only_json_serializable or is_json_serializable(v):
                ret[k] = v
        return ret

    @catch
    def json(self, file: File, main_process_only: bool = True, *args, **kwargs) -> None:  # pylint: disable=W1113
        r"""
        Dump Runner to json file.
        """

        if main_process_only and self.is_main_process or not main_process_only:
            with FlatDict.open(file, mode="w") as fp:  # pylint: disable=C0103
                fp.write(self.jsons(*args, **kwargs))

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
        Dump Runner to json string.
        """

        if "cls" not in kwargs:
            kwargs["cls"] = JsonEncoder
        return json_dumps(self.dict(), *args, **kwargs)

    @classmethod
    def from_jsons(cls, string: str, *args, **kwargs) -> RunnerBase:
        r"""
        Construct Runner from json string.
        """

        return cls(**Config.from_jsons(string, *args, **kwargs))

    @catch
    def yaml(self, file: File, main_process_only: bool = True, *args, **kwargs) -> None:  # pylint: disable=W1113
        r"""
        Dump Runner to yaml file.
        """

        if main_process_only and self.is_main_process or not main_process_only:
            with FlatDict.open(file, mode="w") as fp:  # pylint: disable=C0103
                self.yamls(fp, *args, **kwargs)

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
        Dump Runner to yaml string.
        """

        if "Dumper" not in kwargs:
            kwargs["Dumper"] = YamlDumper
        return yaml_dump(self.dict(), *args, **kwargs)  # type: ignore

    @classmethod
    def from_yamls(cls, string: str, *args, **kwargs) -> RunnerBase:
        r"""
        Construct Runner from yaml string.
        """

        return cls(**Config.from_yamls(string, *args, **kwargs))

    def __getattr__(self, name) -> Any:
        if "uuid" not in self:
            raise RuntimeError(f"{self.__class__.__name__} is not properly initialised")
        raise AttributeError(f"{self.__class__.__name__} does not contain {name}")

    def __setattr__(self, name, value) -> None:
        if name in self.__dict__ and isinstance(self.__dict__[name], Variable):
            self.__dict__[name].set(value)
        else:
            self.__dict__[name] = value

    def __contains__(self, name) -> bool:
        return name in self.__dict__

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
