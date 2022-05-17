from __future__ import annotations

import atexit
import json
import logging
import logging.config
import os
import random
import shutil
from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import accelerate
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from chanfig import Config

from danling.utils import catch, is_serializable

from .utils import ensure_dir, on_local_main_process, on_main_process


class BaseRunner(Config):
    """
    Set up everything for running a job
    """

    id: str
    name: str = 'danling'
    seed: int = 1031

    experiment_dir: str = 'experiments'
    checkpoint_dir_name: str = 'checkpoints'

    steps: int = 0
    epochs: int = 0

    accelerator: accelerate.Accelerator

    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler._LRScheduler

    datasets: OrderedDict[str, data.DataLoader] = OrderedDict()
    datasamplers: OrderedDict[str, data.DataLoader] = OrderedDict()
    dataloaders: OrderedDict[str, data.DataLoader] = OrderedDict()
    batch_size: int = 1

    criterions: Tuple[nn.Module]

    results: List[Dict[str, Any]] = []
    result_best: Dict[str, Any] = {}
    result_last: Dict[str, Any] = {}
    score_best: float = 0
    score_last: float = 0

    log: bool = True
    logger: logging.Logger
    tensorboard: bool = False
    writer: Callable

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # self.init_distributed()
        self.accelerator = accelerate.Accelerator(kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)])

        self.init_seed()

        if getattr(self, 'deterministic', False):
            self.init_deterministic()

        if not hasattr(self, 'id'):
            self.id = f'{self.name}-{self.seed}'

        if getattr(self, 'log', False):
            self.init_logger()
            self.init_print()

        if getattr(self, 'tensorboard', False):
            self.writer = self.init_tensorboard()

        if self.is_main_process:
            self.yaml(os.path.join(self.dir, 'runner.yaml'))

        print(self.dict())
        atexit.register(self.print_result)

    def init_distributed(self) -> None:
        """
        Set up distributed training
        """
        dist.init_process_group(backend='nccl')
        self.process_index = dist.get_rank()
        self.num_processes = dist.get_world_size()
        self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
        torch.cuda.set_device(self.local_process_index)
        self.is_main_process = self.process_index == 0
        self.is_local_main_process = self.local_process_index == 0

    def init_deterministic(self) -> None:
        """
        Set up deterministic
        """
        cudnn.benchmark = False
        cudnn.deterministic = True

    def init_seed(self) -> None:
        """
        Set up random seed
        """
        if self.seed is None:
            self.seed = random.randint(0, 100000)
            if self.distributed:
                self.seed = self.accelerator.gather(torch.tensor(self.seed).cuda()).unsqueeze(0).flatten()[0]
        torch.manual_seed(self.seed + self.process_index)
        torch.cuda.manual_seed(self.seed + self.process_index)
        np.random.seed(self.seed + self.process_index)
        random.seed(self.seed + self.process_index)

    @on_main_process
    def init_logger(self) -> None:
        """
        Set up logger
        """
        # TODO consider move stream output out of stderr
        # Why is setting up proper logging so !@?#! ugly?
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                },
            },
            'handlers': {
                'stderr': {
                    'level': 'INFO',
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                    'stream': 'ext://sys.stderr',
                },
                'logfile': {
                    'level': 'DEBUG',
                    'formatter': 'standard',
                    'class': 'logging.FileHandler',
                    'filename': os.path.join(self.dir, 'run.log'),
                    'mode': 'a',
                }
            },
            'loggers': {
                '': {
                    'handlers': ['stderr', 'logfile'],
                    'level': 'DEBUG',
                    'propagate': True
                },
            }
        })
        logging.captureWarnings(True)
        self.logger = logging.getLogger('runner')
        self.logger.flush = lambda: [h.flush() for h in self.logger.handlers]

    def init_print(self, process: Optional[int] = 0, precision: Optional[int] = 10) -> None:
        """
        Set up print
        Only print from a specific process or force indicated
        Replace default print function with logging.info
        """
        torch.set_printoptions(precision=precision)

        logger = logging.getLogger('print')
        logger.flush = lambda: [h.flush for h in logger.handlers]
        import builtins as __builtin__
        builtin_print = __builtin__.print

        @catch()
        def print(*args, force=False, end='\n', file=None, flush=False, **kwargs):
            if self.process_index == process or force:
                if self.log:
                    logger.info(*args, **kwargs)
                else:
                    builtin_print(*args, end=end, file=file, flush=flush, **kwargs)

        __builtin__.print = print

    @on_main_process
    def init_tensorboard(self, *args, **kwargs):
        from torch.utils.tensorboard.writer import SummaryWriter
        return SummaryWriter(*args, **kwargs)

    def init_lr(self, lr_scale_factor: Optional[float] = None, batch_size_base: Optional[int] = None) -> None:
        """
        Set up learning rate
        """
        if lr_scale_factor is None:
            if batch_size_base is None:
                batch_size_base = self.batch_size_base
            lr_scale_factor = self.batch_size_actual / batch_size_base
        self.lr_scale_factor = lr_scale_factor
        self.lr = self.lr * self.lr_scale_factor
        self.lr_final = self.lr_final * self.lr_scale_factor

    @catch()
    @on_main_process
    def save(self) -> None:
        """
        Save checkpoint to checkpoint_dir
        """
        last_path = os.path.join(self.checkpoint_dir, 'last.pth')
        self.accelerator.save(self.dict(), last_path)
        if (self.epochs + 1) % self.save_freq == 0:
            save_path = os.path.join(self.checkpoint_dir, f'epoch-{self.epochs}.pth')
            shutil.copy(last_path, save_path)
        if self.is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            shutil.copy(last_path, best_path)

    @catch()
    @on_main_process
    def save_result(self) -> None:
        """
        Save result
        """
        ret = {'id': self.id, 'name': self.name}
        ret.update(self.result_last)  # This is slower but ensure id in the first
        last_path = os.path.join(self.dir, 'last.json')
        with open(last_path, 'w') as f:
            json.dump(ret, f, indent=4)
        if self.is_best:
            best_path = os.path.join(self.dir, 'best.json')
            shutil.copy(last_path, best_path)

    def load(self, checkpoint: Union[MutableMapping, str]) -> None:
        """
        Load runner from checkpoint
        """
        print(f'=> loading checkpoint "{checkpoint}"')
        if isinstance(checkpoint, str):
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(f"checkpoint at {checkpoint} is not a file")
            checkpoint = torch.load(checkpoint)
        super().__init__(**checkpoint['runner'])
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        if self.optimizer is not None and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler is not None and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        print(f'=> loaded  checkpoint "{checkpoint}"')

    def dict(self, cls: Callable = dict) -> MutableMapping:
        dict = cls()
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                dict[k] = v.dict(cls)
            elif is_serializable(v):
                dict[k] = v
        return dict

    def state_dict(self, cls: Callable = OrderedDict) -> MutableMapping:
        """
        Return dict of all attributes for checkpoint
        """
        model = self.model
        if self.distributed:
            self.accelerator.wait_for_everyone()
            model = self.accelerator.unwrap_model(self.model)
        return cls(
            runner=self.dict(),
            model=model.state_dict(),
            optimizer=self.optimizer.state_dict() if self.optimizer else None,
            scheduler=self.scheduler.state_dict() if self.scheduler else None,
        )

    def step(self) -> None:
        """
        step optimizer and scheduler
        """
        if self.gradient_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()
        self.steps += 1

    def print_result(self) -> None:
        """
        Print last and best result
        """
        print(f'last result: {self.result_last}')
        print(f'best result: {self.result_best}')

    def __getattr__(self, name) -> Any:
        if (attr := getattr(self.accelerator, name, None)) is not None:
            return attr
        raise AttributeError(f"'Runner' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return self.id

    def __str__(self) -> str:
        return self.name

    @property
    def distributed(self):
        """
        If runner is in distributed mode
        """
        return self.num_processes > 1

    @property
    def iters(self) -> int:
        """
        Number of iterations
        """
        return self.steps * self.batch_size_actual

    @property
    def progress(self) -> float:
        """
        Training Progress
        """
        if hasattr(self, 'iter_end'):
            return self.iters / self.iter_end
        elif hasattr(self, 'step_end'):
            return self.steps / self.step_end
        elif hasattr(self, 'epoch_end'):
            return self.epochs / self.epoch_end
        return 0

    @property
    def batch_size_actual(self) -> int:
        """
        Actual batch size
        """
        return self.batch_size * self.num_processes * getattr(self, 'accum_steps', 1)

    @property
    def is_best(self) -> bool:
        """
        If current result is best
        """
        return self.score_last > self.score_best

    @property
    @ensure_dir
    def dir(self) -> str:
        return os.path.join(self.experiment_dir, self.id)

    @property
    @ensure_dir
    def checkpoint_dir(self) -> str:
        return os.path.join(self.dir, self.checkpoint_dir_name)
