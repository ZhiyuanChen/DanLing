from functools import partial

from .average_meters import AverageMeter, AverageMeters
from .functional import accuracy, auprc, auroc, pearson, r2_score, rmse, spearman
from .metrics import Metrics

__all__ = [
    "Metrics",
    "AverageMeter",
    "AverageMeters",
    "regression_metrics",
    "binary_metrics",
    "multiclass_metrics",
    "multilabel_metrics",
]


def binary_metrics(**kwargs):
    return Metrics(auroc=auroc, auprc=auprc, acc=accuracy, **kwargs)


def multiclass_metrics(num_classes: int, **kwargs):
    p_auroc = partial(auroc, num_classes=num_classes)
    p_auprc = partial(auprc, num_classes=num_classes)
    p_acc = partial(accuracy, num_classes=num_classes)
    return Metrics(auroc=p_auroc, auprc=p_auprc, acc=p_acc, **kwargs)


def multilabel_metrics(num_labels: int, **kwargs):
    p_auroc = partial(auroc, num_labels=num_labels)
    p_auprc = partial(auprc, num_labels=num_labels)
    return Metrics(auroc=p_auroc, auprc=p_auprc, acc=accuracy, **kwargs)


def regression_metrics(**kwargs):
    return Metrics(
        pearson=pearson,
        spearman=spearman,
        r2=r2_score,
        rmse=rmse,
        **kwargs,
    )
