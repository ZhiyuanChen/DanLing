import atexit
import json
import logging
import os
import random
import shutil
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import accelerate
# import click
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from danling.utils import Config, catch


class BaseRunner(object):
    """
    Set up everything for running a job
    """

    config: Any

    accelerator: accelerate.Accelerator = None

    model: nn.Module = None
    optimizer: optim.Optimizer = None
    scheduler: optim.lr_scheduler._LRScheduler = None

    datasets: OrderedDict[str, data.DataLoader] = OrderedDict()
    datasamplers: OrderedDict[str, data.DataLoader] = OrderedDict()
    dataloaders: OrderedDict[str, data.DataLoader] = OrderedDict()

    criterions: Tuple[nn.Module] = None

    results: List[Dict[str, Any]] = []
    result_best: Dict[str, Any] = {}
    result_last: Dict[str, Any] = {}
    score_best: float = 0
    score_last: float = 0

    epoch: int = 0
    epoch_is_best: bool = False

    logger: logging.Logger = None
    writer: SummaryWriter = None

    def __init__(self, config) -> None:
        self.config = config
        self.init_seed()
        if self.id is None:
            self.config.id = f'{self.name}-{self.seed}'
        self.dir = os.path.join(self.experiment_dir, self.id)
        self.checkpoint_dir = os.path.join(self.dir, self.checkpoint_dir_name)

        # self.init_distributed()
        self.accelerator = accelerate.Accelerator()

        if self.deterministic:
            self.init_deterministic()

        if self.is_main_process:
            os.makedirs(self.dir, exist_ok=True)
            if self.train:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            if self.log:
                self.init_logger()
            if self.tensorboard:
                self.writer = SummaryWriter(self.dir)

        if self.log:
            self.init_print()

        print(config)
        atexit.register(self.print_result)

    def init_distributed(self) -> None:
        """
        Set up distributed training
        """
        dist.init_process_group(backend='nccl')
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        torch.cuda.set_device(self.local_rank)
        self.is_main_process = self.rank == 0
        self.is_local_main_process = self.local_rank == 0

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
            self.config.seed = random.randint(0, 100000)
            if self.num_processes > 1:
                self.config.seed = self.accelerator.gather(torch.tensor(self.seed).cuda()).unsqueeze(0).flatten()[0]
        torch.manual_seed(self.rank + self.seed)
        torch.cuda.manual_seed(self.rank + self.seed)
        np.random.seed(self.rank + self.seed)
        random.seed(self.rank + self.seed)

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
        logger.flush = lambda: [h.flush() for h in logger.handlers]
        import builtins as __builtin__
        builtin_print = __builtin__.print

        @catch()
        def print(*args, force=False, end='\n', file=None, flush=False, **kwargs):
            if self.rank == process or force:
                if self.log:
                    logger.info(*args, **kwargs)
                else:
                    builtin_print(*args, end=end, file=file, flush=flush, **kwargs)

        __builtin__.print = print

    def init_lr(self, lr_scale_factor: Optional[float] = None, batch_size_base: Optional[int] = None) -> None:
        """
        Set up learning rate
        """
        if lr_scale_factor is None:
            if batch_size_base is None:
                batch_size_base = self.batch_size_base
            batch_size_actual = self.batch_size * self.world_size * self.accum_steps
            lr_scale_factor = batch_size_actual / batch_size_base
        self.config.lr_scale_factor = lr_scale_factor
        self.config.lr = self.lr * self.lr_scale_factor
        self.config.lr_final = self.lr_final * self.lr_scale_factor

    @catch()
    def save(self) -> None:
        """
        Save checkpoint to checkpoint_dir
        """
        last_path = os.path.join(self.checkpoint_dir, 'last.pth')
        self.accelerator.save(self.dict(), last_path)
        if (self.epoch + 1) % self.save_freq == 0:
            save_path = os.path.join(self.checkpoint_dir, f'epoch-{self.epoch}.pth')
            shutil.copy(last_path, save_path)
        if self.epoch_is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            shutil.copy(last_path, best_path)

    @catch()
    def save_result(self) -> None:
        """
        Save result
        """
        ret = {'id': self.id, 'name': self.name}
        ret.update(self.result_last)  # This is slower but ensure id in the first
        last_path = os.path.join(self.dir, 'last.json')
        with open(last_path, 'w') as f:
            json.dump(ret, f, indent=4)
        if self.epoch_is_best:
            best_path = os.path.join(self.dir, 'best.json')
            shutil.copy(last_path, best_path)

    def load(self, checkpoint: str) -> None:
        """
        Load checkpoint from checkpoint
        """
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError('checkpoint does not exist')
        print(f'=> loading checkpoint "{checkpoint}"')
        checkpoint = torch.load(checkpoint)
        if 'epoch' in checkpoint:
            self.epoch_start = checkpoint['epoch']
        if 'config' in checkpoint:
            self.config = Config(**checkpoint['config'])
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        if self.optimizer is not None and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler is not None and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if 'result' in checkpoint:
            self.result_best = checkpoint['result']
        print(f'=> loaded  checkpoint "{checkpoint}"')

    def dict(self) -> OrderedDict:
        """
        Return dict of all attributes for checkpoint
        """
        model = self.model
        if self.distributed:
            self.accelerator.wait_for_everyone()
            model = self.accelerator.unwrap_model(self.model)
        return OrderedDict(
            epoch=self.epoch,
            config=self.config.dict(),
            model=model.state_dict(),
            optimizer=self.optimizer.state_dict() if self.optimizer else None,
            scheduler=self.scheduler.state_dict() if self.scheduler else None,
            result=self.result_last
        )

    def print_result(self) -> None:
        """
        Print best and last result
        """
        print(f'best result: {self.result_best}')
        print(f'last result: {self.result_last}')

    def __getattr__(self, name) -> Any:
        if (attr := getattr(self.config, name, None)) is not None:
            return attr
        if (attr := getattr(self.accelerator, name, None)) is not None:
            return attr
        raise AttributeError(f"'Runner' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return self.id

    def __str__(self) -> str:
        return self.name
