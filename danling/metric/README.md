# Metric

The `danling.metric` module provides a flexible and powerful system for computing, tracking, and aggregating metrics during model training and evaluation.
This module is designed to work seamlessly with PyTorch and supports both single-task and multi-task scenarios.

## Overview

Metrics are essential for measuring model performance during training and evaluation. The `danling.metric` module offers a comprehensive solution for:

- Computing various metrics (accuracy, AUROC, Pearson, etc.)
- Aggregating metrics across batches and devices
- Supporting complex scenarios like multi-task learning
- Integrating with distributed training environments

## Key Components

The module consists of three classes and several helpful functions:

- [`metrics`][danling.metric.metrics]: Keeps track of all predictions and labels to compute multiple metrics that require the entire dataset.
- [`average_meter`][danling.metric.average_meter]: Core component for averaging values over time.
- [`metric_meter`][danling.metric.metric_meter]: Computes and averages metrics on a per-batch basis.
- [`factory`][danling.metric.factory]: Convenient functions to create common metric for different task types.
- [`functional`][danling.metric.functional]: Implementation of common metric functions.

## Quick Start

### Binary Classification

```python
import danling as dl
import torch

metrics = dl.metric.binary_metrics()
pred = torch.randn(8, 32)
target = torch.randint(2, (8, 32))

metrics.update(pred, target)

print(metrics.val)
print(metrics.avg)
```

### Multiclass Classification

```python
import danling as dl
import torch

metrics = dl.metric.multiclass_metrics(num_classes=10)
pred = torch.randn(8, 10)
target = torch.randint(10, (8, ))

metrics.update(pred, target)

# Access specific metrics
print(f"Accuracy: {metrics.avg['acc']}")
print(f"F1 Score: {metrics.avg['f1']}")
```

### Regression

```python
import danling as dl
import torch

metrics = dl.metric.regression_metrics()
pred = dl.NestedTensor([torch.randn(2, ), torch.randn(3, ), torch.randn(5, )])
target = dl.NestedTensor([torch.randn(2, ), torch.randn(3, ), torch.randn(5, )])

metrics.update(pred, target)

print(f"{metrics:.4f}")
```

## Choosing the Right Metric Class

DanLing provides [`Metrics`][danling.metric.Metrics] and [`MetricMeters`][danling.metric.MetricMeters] for different use cases.
Understanding the differences will help you choose the right one for your specific needs.

!!! info "Use [`Metrics`][danling.metric.Metrics] when"

    - You need metrics that require the entire dataset (like AUROC, Spearman correlation)
    - You want to maintain the full history of predictions and labels
    - Memory is not a constraint for your dataset size
    - You need metrics that cannot be meaningfully averaged batch-by-batch
    - Precision is top priority

```python
import torch
from danling.metric import Metrics
from danling.metric.functional import auroc, auprc

metrics = Metrics(auroc=auroc, auprc=auprc)

pred = torch.randn(8, 32)
target = torch.randint(2, (8, 32))
metrics.update(pred, target)

print(f"Current batch AUROC: {metrics.val['auroc']}")
print(f"Overall AUROC: {metrics.avg['auroc']}")
```

**Best for**:

- Evaluation phases where you need high-quality metrics
- ROC curves and PR curves that require all predictions
- Correlation measures (Pearson, Spearman)
- Final model assessment

!!! info "Use [`MetricMeters`][danling.metric.MetricMeters] when"

    - You need to track metrics that can be averaged across batches (like accuracy, loss)
    - Memory efficiency is important (doesn't store all predictions)
    - Speed matters (syncing predictions across the entire process group takes time)
    - You want simple averaging of metrics across iterations
    - Approximation is good enough

```python
import torch
from danling.metric import MetricMeters
from danling.metric.functional import accuracy, f1_score

meters = MetricMeters(acc=accuracy, f1=f1_score)

pred = torch.randn(8, 32)
target = torch.randint(2, (8, 32))
meters.update(pred, target)

print(f"Current batch accuracy: {meters.val['acc']}")
print(f"Running average accuracy: {meters.avg['acc']}")
```

**Best for**:

- Training phases where speed and memory efficiency is critical
- Simple metrics like accuracy, precision, recall
- Loss tracking during training
- Large datasets where storing all predictions would be impractical

!!! note "[`Metrics`][danling.metric.Metrics] and [`MetricMeters`][danling.metric.MetricMeters] are mostly identical"

    [`Metrics`][danling.metric.Metrics] and [`MetricMeters`][danling.metric.MetricMeters] have a shared API, so that they are interchageable.

    You can easily converts a [`Metrics`][danling.metric.Metrics] to [`MetricMeters`][danling.metric.MetricMeters] by calling `meters = MetricMeters(metrics)` and vice versa.

### Key Differences

| Feature             | [`Metrics`][danling.metric.Metrics]    | [`MetricMeters`][danling.metric.MetricMeters] |
| ------------------- | -------------------------------------- | --------------------------------------------- |
| Storage             | Stores all predictions and labels      | Only stores running statistics                |
| Memory Usage        | Higher (scales with dataset size)      | Lower (constant)                              |
| Computation         | Computes metrics on full dataset       | Averages per-batch metrics                    |
| Multiple Metrics    | Stores multiple metrics with same data | Multiple metrics with same preprocessing      |
| Use Case            | For metrics requiring all data         | For multiple batch-averageable metrics        |
| Distributed Support | Yes                                    | Yes                                           |

## Factory Functions

The module provides convenient factory functions for common task types:

- [`binary_metrics()`][danling.metric.factory.binary_metrics]: For binary classification tasks
- [`multiclass_metrics(num_classes)`][danling.metric.factory.multiclass_metrics]: For multiclass classification tasks
- [`multilabel_metrics(num_labels)`][danling.metric.factory.multilabel_metrics]: For multi-label classification tasks
- [`regression_metrics(num_outputs)`][danling.metric.factory.regression_metrics]: For (multi-)regression tasks

Each factory creates a [`Metrics`][danling.metric.Metrics] instance pre-configured with appropriate metric functions and preprocessing.

!!! example "Using Factory Functions"

    ```python
    # Binary Classification
    metrics = dl.metric.binary_metrics()

    # Multiclass with specific ignore_index
    metrics = dl.metric.multiclass_metrics(num_classes=10, ignore_index=-1)

    # Regression
    metrics = dl.metric.regression_metrics(num_outputs=3)
    ```

## Advanced Usage

### Multi-Task Learning

For multi-task scenarios, use the multi-task variants:

```python
import danling as dl
import torch

metrics = dl.metric.MultiTaskMetrics()
metrics.classification = dl.metric.binary_metrics()
metrics.regression = dl.metric.regression_metrics(num_outputs=16)

metrics.update({
    'classification': (torch.randn(8, ), torch.randint(2, (8, ))),
    'regression': (torch.randn(8, 16), torch.randn(8, 16))
})

print(f"Classification AUROC: {metrics.avg.classification.auroc}")
print(f"Regression RMSE: {metrics.avg.regression.rmse}")
```

### Custom Preprocessing

Customize how inputs are preprocessed before metric calculation:

```python
import torch
from danling.metric import Metrics
from danling.metric.functional import accuracy
from functools import partial

def my_custom_preprocess(input, target, **kwargs):
    return input, target

metrics = Metrics(
    acc=partial(accuracy, num_classes=10),
    preprocess=my_custom_preprocess
)
metrics.update(torch.randn(8, 10), torch.randint(10, (8, )))
print(f"Accuracy: {metrics.avg.acc}")
```

## Custom Metrics

Create custom metric functions to use with the metrics system:

```python
import torch
from danling.metric import Metrics, MetricMeters
from danling.metric.functional import base_preprocess, with_preprocess

@with_preprocess(base_preprocess)
def my_custom_metric(input, target):
    return (input / target).mean()

# Use with Metrics
metrics = Metrics(custom=my_custom_metric)
metrics.update(torch.randn(8, 32), torch.randn(8, 32))
print(f"Custom: {metrics.avg.custom}")


# Or with MetricMeters
meters = MetricMeters(my_custom_metric)
metrics.update(torch.randn(8, 32), torch.randn(8, 32))
print(f"Custom: {metrics.avg.custom}")
```

Note that the `Metrics` and `MetricMeters` will apply a unified preprocess at once if is defined.
