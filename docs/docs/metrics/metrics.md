---
authors:
  - Zhiyuan Chen
date: 2022-05-04
---

# Metrics

Use [GlobalMetrics][danling.metrics.GlobalMetrics] to compute metrics from the
accumulated dataset, or [StreamMetrics][danling.metrics.StreamMetrics] to
aggregate batch-level metric values. [MultiTaskMetrics][danling.metrics.MultiTaskMetrics]
organizes metrics for multiple tasks.

Related public interfaces have their own reference pages:

- [MetricMeter][danling.metrics.MetricMeter] evaluates and tracks one streaming metric.
- [AverageMeter][danling.metrics.AverageMeter] and [AverageMeters][danling.metrics.AverageMeters]
  track running averages.
- [binary_metrics][danling.metrics.binary_metrics], [multiclass_metrics][danling.metrics.multiclass_metrics],
  [multilabel_metrics][danling.metrics.multilabel_metrics] and [regression_metrics][danling.metrics.regression_metrics]
  construct task-specific metric collections.

[MetricState][danling.metrics.MetricState] holds predictions, targets and an
optional confusion matrix for metric functions. It combines their requirements
and provides reusable multiclass and multilabel statistics.

[METRICS][danling.metrics.METRICS] is the case-insensitive factory registry for
`binary`, `multiclass`, `multilabel` and `regression` metrics. Its `build` method
accepts a registered name or a configuration with a `type` key; `mode` selects
global or streaming accumulation. Task dimensions must be positive, and
equivalent dimension arguments must agree when supplied together.

::: danling.metrics
    options:
      members:
        - GlobalMetrics
        - MultiTaskMetrics
        - StreamMetrics
        - MetricState
        - METRICS
