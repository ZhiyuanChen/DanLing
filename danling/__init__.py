from lazy_imports import try_import

from danling import metrics, modules, optim, registry, runner, tensors, typing, utils

from .metrics import AverageMeter, AverageMeters, MultiTaskAverageMeters
from .registry import GlobalRegistry, Registry
from .runner import AccelerateRunner, BaseRunner, TorchRunner
from .tensors import NestedTensor, PNTensor, tensor
from .utils import (
    catch,
    debug,
    ensure_dir,
    flexible_decorator,
    is_json_serializable,
    load,
    load_pandas,
    method_cache,
    save,
)

with try_import():
    from .metrics import Metrics, MultiTaskMetrics

__all__ = [
    "metrics",
    "modules",
    "optim",
    "registry",
    "runner",
    "tensors",
    "utils",
    "typing",
    "BaseRunner",
    "AccelerateRunner",
    "TorchRunner",
    "Registry",
    "GlobalRegistry",
    "Metrics",
    "MultiTaskMetrics",
    "AverageMeter",
    "AverageMeters",
    "MultiTaskAverageMeters",
    "NestedTensor",
    "PNTensor",
    "tensor",
    "save",
    "load",
    "load_pandas",
    "catch",
    "debug",
    "flexible_decorator",
    "method_cache",
    "ensure_dir",
    "is_json_serializable",
]
