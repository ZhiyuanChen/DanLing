---
authors:
  - Zhiyuan Chen
date: 2022-05-04
---

# DanLing

The following names are available directly from `danling`. Follow each link for
its signature, behavior and examples.

| Public name | API reference |
| --- | --- |
| `RunnerConfig` | [Runner configuration][danling.runners.RunnerConfig] |
| `Runner` | [Backend selection and runner entrypoint][danling.runners.Runner] |
| `BaseRunner` | [Shared runner lifecycle][danling.runners.BaseRunner] |
| `RunnerState` | [Checkpointed runner state][danling.RunnerState] |
| `OPTIMIZERS` | [Optimizer registry][danling.OPTIMIZERS] |
| `SCHEDULERS` | [Learning-rate scheduler registry][danling.SCHEDULERS] |
| `LRScheduler` | [Learning-rate schedules][danling.optim.LRScheduler] |
| `TorchRunner` | [PyTorch training, evaluation and inference][danling.runners.TorchRunner] |
| `DeepSpeedRunner` | [DeepSpeed backend][danling.DeepSpeedRunner] |
| `ParallelRunner` | [Parallel training backend][danling.ParallelRunner] |
| `METRICS` | [Metric factory registry][danling.metrics.METRICS] |
| `GlobalMetrics` | [Dataset-level metrics][danling.metrics.GlobalMetrics] |
| `MultiTaskMetrics` | [Metrics for multiple tasks][danling.metrics.MultiTaskMetrics] |
| `MetricMeter` | [A streaming metric][danling.metrics.MetricMeter] |
| `StreamMetrics` | [Streaming metric collection][danling.metrics.StreamMetrics] |
| `AverageMeter` | [A running average][danling.metrics.AverageMeter] |
| `AverageMeters` | [A collection of running averages][danling.metrics.AverageMeters] |
| `NestedTensor` | [Variable-length tensor batches][danling.tensors.NestedTensor] |
| `PNTensor` | [Tensor marker for collation][danling.tensors.PNTensor] |
| `tensor` | [PNTensor construction][danling.tensors.tensor] |
| `to_device` | [Moving nested data to a device][danling.to_device] |
| `save` | [Serialization][danling.utils.save] |
| `load` | [Deserialization][danling.utils.load] |
| `load_pandas` | [Loading tabular data][danling.utils.load_pandas] |
| `catch` | [Exception handling][danling.utils.catch] |
| `debug` | [Debug context manager][danling.utils.debug] |
| `flexible_decorator` | [Decorator invocation forms][danling.utils.flexible_decorator] |
| `method_cache` | [Method result caching][danling.utils.method_cache] |
| `ensure_dir` | [Directory creation on attribute access][danling.ensure_dir] |
| `is_json_serializable` | [JSON serialization check][danling.utils.is_json_serializable] |

`OPTIMIZERS` and `SCHEDULERS` are case-insensitive registries. Their registered
names select optimizer and scheduler constructors through `build`. Available
DeepSpeed optimizers depend on whether DeepSpeed is installed. `SCHEDULERS`
includes DanLing's linear, cosine and constant schedules as well as PyTorch
schedulers.

::: danling
    options:
      members:
        - RunnerState
        - OPTIMIZERS
        - SCHEDULERS
        - DeepSpeedRunner
        - ParallelRunner
        - to_device
        - ensure_dir
