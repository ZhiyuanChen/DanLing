from __future__ import annotations

import os
from datetime import datetime
from random import randint
from typing import IO, Callable, List, Optional, Union
from uuid import UUID, uuid5
from warnings import warn

from chanfig import NestedDict
from git.exc import InvalidGitRepositoryError
from git.repo import Repo

from danling.utils import base62, ensure_dir

PathStr = Union[os.PathLike, str, bytes]
File = Union[PathStr, IO]

DEFAULT_EXPERIMENT_NAME = "DanLing"
DEFAULT_EXPERIMENT_ID = "xxxxxxxxxxxxxxxx"
DEFAULT_IGNORED_KEYS_IN_HASH = {"iters", "steps", "epochs", "results", "index_set", "index"}


class RunnerState(NestedDict):
    r"""
    `RunnerState` is a `NestedDict` that contains all states of a `Runner`.

    `RunnerState` is designed to store all critical information of a Run so that you can resume a run
    from a state and corresponding weights or even restart a run from a state.

    `RunnerState` is also designed to be serialisable and hashable, so that you can save it to a file.
    `RunnerState` is saved in checkpoint together with weights by default.

    Since `RunnerState` is a `NestedDict`, you can access its attributes by `state["key"]` or `state.key`.

    Attributes: General:
        id (str): `f"{self.experiment_id:.4}{self.run_id:.4}{time_str}"`.
        uuid (UUID, property): `uuid5(self.run_id, self.id)`.
        name (str): `f"{self.experiment_name}-{self.run_name}"`.
        run_id (str): hex of `self.run_uuid`.
        run_uuid (UUID, property): `uuid5(self.experiment_id, config.jsons())`.
        run_name (str): Defaults to `"DanLing"`.
        experiment_id (str): git hash of the current HEAD.
            Defaults to `"xxxxxxxxxxxxxxxx"` if Runner not under a git repo or git/gitpython not installed.
        experiment_uuid (UUID, property): UUID of `self.experiment_id`.
            Defaults to `UUID('78787878-7878-7878-7878-787878787878')`
            if Runner not under a git repo or git/gitpython not installed.
        experiment_name (str): Defaults to `"DanLing"`.
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
        tensorboard (bool): Whether to use `tensorboard`.
            Defaults to `False`.

    Notes:
        `RunnerState` is a `NestedDict`, so you can access its attributes by `state["name"]` or `state.name`.

    See Also:
        [`RunnerBase`][danling.runner.runner_base.RunnerBase]: The runeer state that stores critical information.
        [`BaseRunner`][danling.runner.BaseRunner]: The base runner class.
    """

    # pylint: disable=R0902, R0904
    # DO NOT set default value in class, as they won't be stored in `__dict__`.

    id: str
    name: str
    run_id: str
    run_name: str = "Run"
    experiment_id: str = DEFAULT_EXPERIMENT_ID
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

    results: List[NestedDict]
    index_set: Optional[str]
    index: str

    project_root: str = "experiments"
    checkpoint_dir_name: str = "checkpoints"
    log: bool = True
    tensorboard: bool = False

    def __init__(self, *args, **kwargs):
        try:
            self.experiment_id = Repo(search_parent_directories=True).head.object.hexsha
        except ImportError:
            warn("GitPython is not installed, using default experiment id.")
        except InvalidGitRepositoryError:
            warn("Git reporitory is invalid, using default experiment id.")
        self.deterministic = False
        self.seed = randint(0, 2**32 - 1)
        self.iters = 0
        self.steps = 0
        self.epochs = 0
        self.results = []
        self.index_set = None
        self.index = "loss"
        super().__init__(*args, **kwargs)
        self.run_id = self.run_uuid.hex
        time = datetime.now()
        time_tuple = time.isocalendar()[1:] + (time.hour, time.minute, time.second, time.microsecond)
        time_str = "".join(base62.encode(i) for i in time_tuple)
        self.id = f"{self.experiment_id:.5}{self.run_id:.4}{time_str}"  # pylint: disable=C0103
        self.name = f"{self.experiment_name}-{self.run_name}"
        self.setattr("ignored_keys_in_hash", DEFAULT_IGNORED_KEYS_IN_HASH)

    @property
    def experiment_uuid(self) -> UUID:
        r"""
        UUID of the experiment.
        """

        return UUID(bytes=bytes(self.experiment_id.ljust(16, "x")[:16], encoding="ascii"))

    @property
    def run_uuid(self) -> UUID:
        r"""
        UUID of the run.
        """

        return uuid5(self.experiment_uuid, str(hash(self)))

    @property
    def uuid(self) -> UUID:
        r"""
        UUID of the state.
        """

        return uuid5(self.run_uuid, self.id)

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

    def __hash__(self) -> int:
        ignored_keys_in_hash = self.getattr("ignored_keys_in_hash", DEFAULT_IGNORED_KEYS_IN_HASH)
        state = NestedDict({k: v for k, v in self.dict().items() if k not in ignored_keys_in_hash})
        return hash(state.yamls())
