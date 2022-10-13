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
from os import PathLike as _PathLike
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union

import accelerate
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from chanfig import NestedDict

from danling.utils import catch, is_json_serializable

from .base_runner import BaseRunner
from .utils import ensure_dir, on_local_main_process, on_main_process

PathLike = Union[str, _PathLike]
File = Union[PathLike, IO]


class StepRunner(BaseRunner):
    """
    Set up everything for running a job
    """
    
    step_begin: int
    step_end: int
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if 'step_begin' not in self:
            self.step_begin = 0
        if 'step_end' not in self:
            raise ValueError('"step_end" must be specified for StepRunner')
