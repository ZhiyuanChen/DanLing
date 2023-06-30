from __future__ import annotations

import os
import sys
from datetime import datetime
from random import randint
from uuid import UUID, uuid5
from warnings import warn

from chanfig import NestedDict

try:
    from git.exc import InvalidGitRepositoryError
    from git.repo import Repo
except ImportError:
    warn("gitpython not installed, git hash will not be available", category=RuntimeWarning, stacklevel=2)
    Repo = None  # type: ignore

from danling.utils import base62

from . import defaults


class RunnerState(NestedDict):
    r"""
    `RunnerState` is a `NestedDict` that contains all states of a `Runner`.

    `RunnerState` is designed to store all critical information of a Run so that you can resume a run
    from a state and corresponding weights or even restart a run from a state.

    `RunnerState` is also designed to be serialisable and hashable, so that you can save it to a file.
    `RunnerState` is saved in checkpoint together with weights by default.

    Since `RunnerState` is a `NestedDict`, you can access its attributes by `state["key"]` or `state.key`.

    Attributes: General:
        id (str): `f"{time_str}{self.experiment_id:.5}{self.run_id:.4}"`.
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

    In general you should only use one of `iter_end`, `step_end`, `epoch_end` to indicate the length of running.

    Attributes: Results:
        results (List[NestedDict]): All results, should be in the form of ``[{subset: {index: score}}]``.

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
        print_interval (int): Interval of printing logs.
            Defaults to -1.
        save_interval (int): Interval of saving intermediate checkpoints.
            Defaults to -1, never save intermediate checkpoints.

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
    run_name: str
    experiment_id: str
    experiment_name: str

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

    results: list[NestedDict]
    index_set: str | None
    index: str

    project_root: str = "experiments"
    checkpoint_dir_name: str = "checkpoints"
    log: bool = True
    tensorboard: bool = False
    print_interval: int = -1
    save_interval: int = -1

    def __init__(self, *args, **kwargs):
        self.run_name = defaults.DEFAULT_RUN_NAME
        self.experiment_id = defaults.DEFAULT_EXPERIMENT_ID
        self.experiment_name = defaults.DEFAULT_EXPERIMENT_NAME
        if Repo is not None:
            try:
                self.experiment_id = Repo(search_parent_directories=True).head.object.hexsha
            except ImportError:
                warn(
                    "GitPython is not installed, fallback to `DEFAULT_EXPERIMENT_ID`.",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
            except (InvalidGitRepositoryError, ValueError):
                path = os.path.dirname(os.path.abspath(sys.argv[0]))
                warn(
                    "Unable to get git hash from CWD, fallback to top-level code environment.",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
                try:
                    self.experiment_id = Repo(path=path, search_parent_directories=True).head.object.hexsha
                except (InvalidGitRepositoryError, ValueError):
                    warn(
                        "Unable to get git hash from top-level code environment, fallback to `DEFAULT_EXPERIMENT_ID`.",
                        category=RuntimeWarning,
                        stacklevel=2,
                    )
        else:
            warn(
                "GitPython is not installed, fallback to `DEFAULT_EXPERIMENT_ID`.",
                category=RuntimeWarning,
                stacklevel=2,
            )
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
        time_tuple = time.isocalendar()[1:] + (
            time.hour,
            time.minute,
            time.second,
            time.microsecond,
        )
        time_str = "".join(base62.encode(i) for i in time_tuple)
        self.id = f"{time_str}{self.experiment_id:.5}{self.run_id:.4}"  # pylint: disable=C0103
        self.name = f"{self.experiment_name}-{self.run_name}"
        self.setattr("ignored_keys_in_hash", defaults.DEFAULT_IGNORED_KEYS_IN_HASH)

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

    def __hash__(self) -> int:  # type: ignore
        ignored_keys_in_hash = self.getattr("ignored_keys_in_hash", defaults.DEFAULT_IGNORED_KEYS_IN_HASH)
        state: NestedDict = NestedDict({k: v for k, v in self.dict().items() if k not in ignored_keys_in_hash})
        return hash(state.yamls())
