from __future__ import annotations

import logging
import logging.config
from typing import Any, Callable, List, Tuple

import accelerate
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from chanfig import OrderedDict, NestedDict


class AbstractRunner(NestedDict):

    id: str
    name: str
    seed: int
    deterministic: bool

    experiment_dir: str
    checkpoint_dir_name: str

    steps: int
    epochs: int

    accelerator: accelerate.Accelerator
    accelerate: NestedDict[str, Any]

    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler._LRScheduler

    datasets: OrderedDict[str, data.Dataset]
    datasamplers: OrderedDict[str, data.Sampler]
    dataloaders: OrderedDict[str, data.DataLoader]

    batch_size: int

    criterion: Tuple[nn.Module]

    metric: str
    results: List[NestedDict[str, any]]
    result_best: NestedDict[str, any]
    result_latest: NestedDict[str, any]
    score_best: float
    score_latest: float
    is_best: bool

    log: bool
    logger: logging.Logger
    tensorboard: bool
    writer: Callable

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = None
        self.name = "danling"
        self.seed = 1031
        self.deterministic = False

        self.experiment_dir = "experiments"
        self.checkpoint_dir_name = "checkpoints"

        self.steps = 0
        self.epochs = 0
        self.accelerate = {}

        self.model: nn.Module = None
        self.optimizer: optim.Optimizer = None
        self.scheduler: optim.lr_scheduler._LRScheduler = None

        self.datasets = OrderedDict()
        self.datasamplers = OrderedDict()
        self.dataloaders = OrderedDict()

        self.batch_size = 1

        self.criterion = None

        self.metric = "loss"
        self.results = []
        self.result_best = NestedDict()
        self.result_latest = NestedDict()
        self.score_best = 0
        self.score_latest = 0
        self.is_best = False

        self.log = True
        self.tensorboard = False
        self.writer = None

