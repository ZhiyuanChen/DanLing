from functools import partial

from lazy_imports import try_import

from .average_meter import AverageMeter, MultiTaskAverageMeter

with try_import():
    from .functional import accuracy, auprc, auroc, matthews_corrcoef, pearson, r2_score, rmse, spearman
    from .metrics import Metrics, MultiTaskMetrics

__all__ = [
    "Metrics",
    "MultiTaskMetrics",
    "AverageMeter",
    "MultiTaskAverageMeter",
    "regression_metrics",
    "binary_metrics",
    "multiclass_metrics",
    "multilabel_metrics",
]


def binary_metrics(**kwargs):
    return Metrics(auroc=auroc, auprc=auprc, acc=accuracy, mcc=matthews_corrcoef, **kwargs)


def multiclass_metrics(num_classes: int, **kwargs):
    p_mcc = partial(matthews_corrcoef, num_classes=num_classes)
    p_auroc = partial(auroc, num_classes=num_classes)
    p_auprc = partial(auprc, num_classes=num_classes)
    p_acc = partial(accuracy, num_classes=num_classes)
    return Metrics(auroc=p_auroc, auprc=p_auprc, acc=p_acc, mcc=p_mcc, **kwargs)


def multilabel_metrics(num_labels: int, **kwargs):
    p_mcc = partial(matthews_corrcoef, num_labels=num_labels)
    p_auroc = partial(auroc, num_labels=num_labels)
    p_auprc = partial(auprc, num_labels=num_labels)
    p_acc = partial(accuracy, num_labels=num_labels)
    return Metrics(auroc=p_auroc, auprc=p_auprc, acc=p_acc, mcc=p_mcc, **kwargs)


def regression_metrics(**kwargs):
    return Metrics(
        pearson=pearson,
        spearman=spearman,
        r2=r2_score,
        rmse=rmse,
        **kwargs,
    )
