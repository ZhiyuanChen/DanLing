# Metrics

`danling.metrics` provides metric containers and metric descriptors for large-scale training.
The design is exact-by-default and optimized to avoid redundant state computation and cross-rank sync.

## Design Summary

- Exact by default: factory functions return [`GlobalMetrics`][danling.metrics.GlobalMetrics] unless `mode="stream"` is set.
- Shared state: metric descriptors declare required artifacts (`preds/targets`, `confmat`) so containers build them once.
- Symmetric API: [`GlobalMetrics`][danling.metrics.GlobalMetrics] and [`StreamMetrics`][danling.metrics.StreamMetrics] share the same constructor signature.
- Extensible: users can provide custom [`MetricFunc`][danling.metrics.functional.MetricFunc] implementations (or plain callables for `StreamMetrics`).

## Core Components

- [`GlobalMetrics`][danling.metrics.GlobalMetrics]
  - Stores exact artifacts for global/global computation.
  - Computes values from shared [`MetricState`][danling.metrics.MetricState].
  - Performs distributed synchronization lazily in `average()` / `compute()`.
- [`StreamMetrics`][danling.metrics.StreamMetrics]
  - Computes per-sample/per-batch scores online and tracks running averages.
  - Uses the same metric descriptors and preprocess contract as `GlobalMetrics`.
  - Suitable for high-throughput training loops.
- [`MetricMeter`][danling.metrics.MetricMeter]
  - Single-metric streaming meter used internally by `StreamMetrics`.
- [`METRICS` registry][danling.metrics.METRICS]
  - Task factory registry with explicit `mode`.
- [`MultiTaskMetrics`][danling.metrics.MultiTaskMetrics]
  - Nested container for multi-head / multi-dataset evaluation.

## Quick Start

### Exact Metrics (Default)

```python
import torch
import danling as dl

metrics = dl.metrics.binary_metrics()  # mode="global" by default -> GlobalMetrics
metrics.update(torch.randn(32), torch.randint(2, (32,)))

print(metrics.val)  # last update
print(metrics.avg)  # exact average over all accumulated state
```

### Streaming Metrics

```python
import torch
import danling as dl

metrics = dl.metrics.multiclass_metrics(num_classes=10, mode="stream")  # -> StreamMetrics
metrics.update(torch.randn(64, 10), torch.randint(10, (64,)))

print(metrics.val)  # current batch metric
print(metrics.avg)  # running average
```

## Global vs Stream

| Aspect               | [`GlobalMetrics`][danling.metrics.GlobalMetrics] | [`StreamMetrics`][danling.metrics.StreamMetrics] |
| -------------------- | ------------------------------------------------ | ------------------------------------------------ |
| Default factory mode | `mode="global"`                                  | `mode="stream"`                                  |
| State                | Stores full required artifacts                   | Stores running meter stats                       |
| Sync pattern         | Sync once when computing average                 | Meter-level sync in running stats                |
| Typical use          | Exact eval, AUROC/AUPRC/correlation              | Fast training logs                               |
| Memory               | Higher                                           | Lower                                            |

## Shared Constructor Contract

`GlobalMetrics` and `StreamMetrics` intentionally share this signature:

```python
(*metric_funcs, preprocess=..., distributed=True, device=None, **metrics)
```

Rules:

- Positional `*metric_funcs` can be metric descriptors (or iterables of descriptors). `StreamMetrics` also accepts plain callables.
- Keyword `**metrics` are named metrics and override positional metrics with the same name.
- `preprocess` is applied once per `update`.
- `device` controls where internal artifacts/stat reductions live.

## Factory Functions

- [`binary_metrics`][danling.metrics.factory.binary_metrics]
- [`multiclass_metrics`][danling.metrics.factory.multiclass_metrics]
- [`multilabel_metrics`][danling.metrics.factory.multilabel_metrics]
- [`regression_metrics`][danling.metrics.factory.regression_metrics]

All factories accept:

- `mode="global" | "stream"` (`"global"` default)
- `*metrics_funcs`: if provided, defaults are replaced
- `**metrics`: named extra metrics (or overrides)
- task-specific arguments (`num_classes`, `num_labels`, `num_outputs`, `ignore_index`, etc.)

Example:

```python
import danling as dl
from danling.metrics.functional import binary_precision, binary_recall

# Keep defaults and add metrics
metrics = dl.metrics.binary_metrics(
    mode="global",
    precision=binary_precision(),
    recall=binary_recall(),
)

# Replace defaults completely
metrics_only_pr = dl.metrics.binary_metrics(
    binary_precision(),
    binary_recall(),
    mode="global",
)
```

## Default Metric Sets

Factories keep defaults minimal:

- Binary / Multiclass / Multilabel:
  - `auroc`, `auprc`, `acc`, `f1`, `mcc`
- Regression:
  - `pearson`, `spearman`, `r2`, `mse`, `rmse`

Additional built-ins (opt-in):

- Classification: `precision`, `recall`, `fbeta`, `specificity`, `balanced_accuracy`, `jaccard`, `iou`, `hamming_loss`
- Regression: `mae`

`multiclass_accuracy` also supports top-k via `k`.

## Custom Metric Descriptor (`MetricFunc`)

For consistent behavior across both containers, implement `MetricFunc` and read from `MetricState`.

```python
import torch
from danling.metrics import GlobalMetrics, StreamMetrics
from danling.metrics.functional import MetricFunc

class MeanBias(MetricFunc):
    def __init__(self, name: str = "mean_bias") -> None:
        super().__init__(name=name, preds_targets=True)

    def __call__(self, state):
        if state.preds.numel() == 0 or state.targets.numel() == 0:
            return torch.tensor(float("nan"))
        return (state.preds - state.targets).mean()

metric = MeanBias()

global_metrics = GlobalMetrics(metric, distributed=False)
stream_metrics = StreamMetrics(metric)

pred = torch.randn(16)
target = torch.randn(16)

global_metrics.update(pred, target)
stream_metrics.update(pred, target)
```

## Multi-Task Usage

```python
import torch
import danling as dl

metrics = dl.metrics.MultiTaskMetrics()
metrics.cls = dl.metrics.binary_metrics(mode="stream")
metrics.reg = dl.metrics.regression_metrics(num_outputs=4, mode="global", distributed=False)

metrics.update(
    {
        "cls": (torch.randn(32), torch.randint(2, (32,))),
        "reg": (torch.randn(32, 4), torch.randn(32, 4)),
    }
)

print(metrics.avg)
```

## Registry Usage

```python
from danling.metrics import METRICS

metrics = METRICS.build(type="multiclass", mode="stream", num_classes=10)
```
