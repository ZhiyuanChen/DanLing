# DanLing Runner

DanLing runners are now **Torch-only** and focus on scalable distributed training.

The new design borrows from TorchTitan's Trainer philosophy:

- A single, explicit training lifecycle
- Backend-agnostic extension through `Runner`, with opt-in stack-specific subclasses when needed
- Backend-specific distributed logic in dedicated subclass runners
- Minimal shared contracts in `BaseRunner`

## Core APIs

- `RunnerConfig`: `chanfig`-based long-term configuration surface
- `Runner`: stack-selecting backend-agnostic entrypoint (`ddp`/`torch`, `deepspeed`/`ds`, `parallel`; default `ddp`)
- `TorchRunner`: shared Torch training core with explicit epoch-mode and step-mode loops
- `TorchRunner` / `ParallelRunner` / `DeepSpeedRunner`: class-based stack runners

## Common workflows

The default entrypoint is direct:

- Scratch: `runner = Runner(config)`
- Resume full training state: `runner = Runner.from_checkpoint(checkpoint)` or `runner.load_checkpoint(checkpoint)`
- Initialize model weights only: `runner = Runner.from_pretrained(config, checkpoint)` or `runner.load_pretrained(checkpoint)`

The config keys follow the same split:

| Key          | Meaning                                                                                                                             |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `checkpoint` | Explicit full-state checkpoint source. Loads runner state, model/EMA, optimizer, scheduler, and dataloaders.                        |
| `resume`     | Boolean auto-resume switch. When `True`, loads `latest` from the current run checkpoint directory.                                  |
| `pretrained` | Model-only initialization source. Loads EMA/model weights and leaves optimizer, scheduler, runner state, and dataloaders untouched. |
| `ckpt`       | Checkpoint write policy only: backend, directory, async mode, interval, retention, export dtype, and write enablement.              |

Load-source precedence is `checkpoint` > `resume` > `pretrained`.

Common config shapes:

```yaml
# Resume from a known full-state checkpoint.
checkpoint: experiments/lin-<code_id>/<id>/checkpoints/latest.pth

# Or auto-resume from this run's own latest checkpoint.
resume: true

# Finetune from model weights only.
pretrained: experiments/lin-<code_id>/<id>/checkpoints/model.pth

# Disable all checkpoint writes while still allowing checkpoint/pretrained loading.
ckpt:
  enabled: false
```

`ckpt.enabled=False` disables `latest`, `best`, history, and `model` writes. It does not disable reads from
`checkpoint`, `resume`, or `pretrained`.
CLI overrides use the same keys, for example `--checkpoint ...`, `--resume`, `--pretrained ...`, or
`--ckpt.enabled false`.

## BaseRunner contract

`BaseRunner` is inheritance-first and keeps only cross-backend functionality.
There is no plugin/spec registry layer.

Construction is intentionally direct: subclasses set concrete components in
`__init__`, call `super().__init__(config)`, then assign model/data/optimizer
objects. `MetaRunner` invokes `__post_init__` after subclass construction so
concrete runners can materialize models, optimizers, checkpoint state, and
metadata once user-owned attributes are present.

Execution API stubs in `BaseRunner` (all overridable):

- `train` / `train_epoch` / `train_steps` / `train_step`
- `evaluate` / `evaluate_epoch` / `evaluate_steps` / `evaluate_step`
- `infer`

Step-mode semantics:

- `train` dispatches to `train_steps` when `epochs` is unset (`is_step_mode=True`).
- `TorchRunner` keeps epoch-mode and step-mode loops separate on purpose to keep control flow local and
  override points explicit.
- `evaluate_steps(split, steps=None)` is always bounded-step evaluation.
  When `steps` is omitted, it defaults to `max(steps_budget // 20, 1)`.

## Validation during training

The default schedule is unchanged. Configure validation independently from
checkpoint frequency:

```yaml
evaluation:
  interval: 1000 # Successful optimizer updates; 0 disables this cadence.
  steps: null # Full split; a positive integer limits the number of batches.
score:
  metric: custom_error
  mode: min
```

`steps` also works without an interval. Epoch boundaries and the
end of training still validate; a completed validation at the same global step
is reused. Accumulation microbatches and rejected updates do not trigger the
cadence. Epoch-mode `score.patience` continues to count epochs.

Training-time validation automatically restores training meters, metric
references, module modes and RNG state, including after an exception.
Checkpoints saved during validation capture the entering training RNG and
original loader positions. Each native map-style validation loader starts from
its sample-index origin, including for bounded validation. Override
`build_evaluation_dataloader(split)` for custom samplers, stateful datasets or
streams, and `build_evaluation_metrics()` for custom collectors shared with
training. These factories provide independent validation resources; standalone
`evaluate`, `evaluate_epoch` and `evaluate_steps` keep their existing modes,
loader consumption, return values and logging semantics.

Only a successful validation of every selected split publishes a result row and
participates in native best selection. `ckpt.interval` retains its normal units
and history cadence; validation can publish an improving best between those
boundaries. An ordinary checkpoint does not republish an older validation score
as the current model's best. Native FT replicas each cover the validation set
within their local data-parallel group.

When these evaluation options are used, result rows use global-step indices in
both training modes. Otherwise epoch-mode rows keep epoch indices. Full resume
preserves the last completed validation step and avoids repeating it. A run with
existing epoch-indexed history must retain that indexing mode on full resume;
model-only loading can start a new run with the new schedule.

`score.mode` accepts `min` or `max`. Leaving it unset preserves the existing
rule: minimize `loss`, maximize other metric names.

## Loss reduction and clipping

Mean losses accumulate as numerators and are divided once by the global window's
denominator. Weighted CE/NLL uses the valid target class-weight sum. Batch
`loss_normalizer` metadata can supply a nonnegative real denominator/weight for
custom objectives. A criterion with `reduction="sum"` accumulates additively;
with explicit `loss_normalizer`, it declares a summed numerator to divide by that
denominator. Unknown criteria retain microbatch-mean accumulation by default.
Custom objectives can override `_get_loss_normalizer()` and `_get_loss_reduction()`.
Zero-weight microbatches contribute no gradient; an entirely zero-weight global
window skips the optimizer, step scheduler, EMA and update counter.

Pipeline `max_grad_norm` uses one model norm across stages and DTensor meshes.
Runner-owned fp16 scales the backward loss on the final pipeline stage, keeps
scales consistent on overflow, and includes scaler state in full checkpoints.

Checkpoint/score contracts (all overridable):

- `load_model`
- `read_checkpoint`
- `load_optimizer` / `load_scheduler`
- `load_state_dict`
- `scores`
- `best_index`

## Checkpoint semantics

The user-facing checkpoint contract is intentionally small:

- `latest`: the most recent persisted full-state checkpoint.
- `best`: the best persisted full-state checkpoint, updated only when `save_best=True` and the current score is best.
- `model`: a model-only checkpoint written after normal training completion for publishing or `pretrained` loading.
- History checkpoints: periodic snapshots created by the training loop.
  - epoch mode: `ckpt-e{completed_epoch:06d}`
  - step mode: `ckpt-s{global_step:012d}`
- `ckpt.keep_latest_k`: number of framework-generated historical checkpoints to retain.
  `0` disables automatic retention pruning. When enabled, current `latest` and `best` targets are protected.

Periodic checkpoint attempts are controlled by `ckpt.interval`.
When `ckpt.interval` is unset, runner defaults are used by mode (`epochs`: every epoch, `steps`: budget/20).
Forced checkpoints, including final and graceful-shutdown checkpoints, bypass the interval and update `latest`.
Normal training completion also writes `model` without changing `latest`, `best`, or history aliases.
All checkpoint writes are gated by `ckpt.enabled`; loading paths are independent.

Backend storage is an implementation detail:

- File backend stores `latest.pth`, `best.pth`, and `ckpt-*.pth`.
  `best` and history files are updated from `latest` via hardlink-first aliasing with copy fallback.
  It is intended for local, single-process, and small-run workflows. In async mode it snapshots live tensors before
  background writes; use DCP for distributed or large-model training.
- DCP backend exposes the same logical names (`latest`, `best`, `ckpt-*`) but stores immutable physical target directories plus pointer files.
  This avoids overwriting checkpoints while async writers or external readers may still hold references.
- DeepSpeed full-state saves use engine checkpoint tag directories plus pointer files for `latest`, `best`, and history names.
  DeepSpeed model-only export still writes a DanLing `model` payload for publication and `pretrained` loading.
- Retention policy is enforced by DanLing-managed file and DCP checkpoint managers.

Rank participation contract:

- `BaseRunner.save_checkpoint` and `save_model_checkpoint` delegate participation to the active checkpoint manager:
  file-style managers write on the main process; collective managers require all participating ranks to enter the call.
- `TorchRunner` uses the configured manager (`file`, `dcp`, or TorchFT-backed DCP) for full-state and model-only saves.
- `DeepSpeedRunner.save_checkpoint` uses DeepSpeed engine checkpointing for full-state saves; all ranks participate in those saves.
- `DeepSpeedRunner.save_model_checkpoint` exports the unwrapped engine module as a DanLing model-only checkpoint for publication or `pretrained`.
- When `config.ft.enabled=True` under `TorchRunner`/DDP or `ParallelRunner` FSDP, full DCP saves are gated to the TorchFT participating replica group.
  Each replica group also saves its own dataloader, RNG and Manager state.

Checkpoint persistence internals are implemented under `danling.runners.checkpoints`:

- `CheckpointManager`: base contract
- `FileCheckpointManager`: default async filesystem manager with reliable history/best queueing plus latest-wins coalescing for non-critical saves
- `TorchDistributedCheckpointManager`: Torch DCP backend used when `ckpt.backend="dcp"` in `TorchRunner`

`ParallelRunner` forces `ckpt.backend="dcp"` by default.

Full checkpoints expose native result rows as the top-level `results` mapping,
preserving their indices, metric keys, and best-score history on resume.
`load_pretrained` only imports weights and leaves that history untouched.
Legacy checkpoints without `results` retain the current runner's history.
DCP stores this growing mapping as one opaque item and decodes it before returning
the public checkpoint payload; `runner.results` remains readable and appendable.

## Dataloader state

Full-state checkpoints include split-keyed dataloader progress under the top-level
`dataloaders` payload.

- `TorchRunner.build_dataloaders()` uses `StatefulDataLoader` for datasets that have not already been materialized.
- User-provided dataloaders participate when they implement `state_dict()` and `load_state_dict()`.
- Non-stateful dataloaders are skipped rather than wrapped or special-cased.
- `BaseRunner.load_checkpoint()` restores dataloaders after model/EMA/optimizer/scheduler state.
- DCP stores RNG, dataloader and FT state under rank-specific keys, retaining the public payload shape when loaded.
- With TorchFT enabled, each replica's local checkpoint takes precedence for its dataloader, RNG and Manager state.
- `ParallelRunner` middle-stage proxy loaders save their placeholder cursor along with the underlying loader state.

Step-mode checkpoints retain the active split target and each split's sampler
epoch. A graceful SIGTERM waits until the current accumulation window finishes,
before fetching another batch, then saves and drains checkpoint I/O. A second
SIGTERM exits immediately. Custom loops can override `can_checkpoint_for_shutdown()`
to declare their own consistent data/update boundary.

## TorchFT

DanLing has an opt-in TorchFT integration for replicated-weight runners:

- enable with `config.ft.enabled=True`
- configure replica layout with:
  - `ft.replica_id`
  - `ft.group_size`
  - `ft.min_replica_size`
  - `ft.process_group`
  - `ft.process_group_timeout_seconds`
- requires the external `torchft` package and a TorchFT lighthouse setup

Current scope:

- supported: `TorchRunner`/DDP, `ParallelRunner` FSDP without tensor/pipeline axes
- not yet supported: `ParallelRunner` tensor/pipeline runs, `DeepSpeedRunner`

Current launch contract:

- launch one `torchrun` per replica group
- `WORLD_SIZE` is replica-local data-parallel size, while `ft.group_size` counts replica groups
- a single global `torchrun` spanning all replica groups is not a supported TorchFT launch mode

When TorchFT is enabled:

- automatic checkpoint backend selection uses DCP, including single-process replica groups
- dataloader sharding and seed bias are expanded across replica groups using
  `ft.replica_id` and `ft.group_size`
- FSDP2 replicated-dimension all-reduce hooks use TorchFT-managed process groups
- DDP gradients and loss/normalizer reductions span replica-local ranks and the TorchFT managed replica group
- per-replica local-state checkpoints are enabled automatically through the DCP checkpoint manager

Distributed state-dict invariants:

- Parallel distributed runs require DCP state-dict APIs for model/optimizer restore.
- Checkpoint restore order is model -> EMA -> optimizer -> scheduler -> dataloaders.
- File checkpoint fallback is only for non-distributed/basic flows.

Async policy is configured by `ckpt.async_mode`:

- `disabled`: synchronous checkpoint writes
- `async`: asynchronous uploads
- `async_with_pinned_mem`: process-based async uploads with staging hook support (`maybe_wait_for_staging`)

## Directory hierarchy

DanLing now uses a stable run layout with per-attempt timestamps:

`workspace.root/workspace.lineage[-code_id]/id`

- `workspace.lineage`: top-level lineage namespace
- `code_id`: git code identity (`<short_sha>` for clean trees, `<short_sha>-d<diff_sha10>` when dirty) appended to `workspace.lineage` when available
- `config_id`: low 48 bits of `config.fingerprint()` as 12 lowercase hexadecimal digits, cached on first access
- `id`: stable run identifier, defaults to `code_id-config_id` when git metadata is available and `config_id` otherwise
- `timestamp`: per-attempt runtime timestamp
- `workspace.experiment`: experiment label used in the default runner name and W&B group; it is not a path segment unless user code includes it in `workspace.dir`

`config.fingerprint()` recomputes an unsigned 64-bit integer from the first eight bytes
of SHA-1 over the UTF-8 canonical YAML (big-endian byte order). Configurations remain
mutable and unhashable. Runtime-only settings are omitted by `canonical()`.

By default:

- `dir` points to `workspace.root/workspace.lineage[-code_id]/id`
- checkpoints are stored at `ckpt.dir` (default `dir/checkpoints`)
- result snapshots are stored at `dir`
- logs are stored at `logging.file` (default `dir/logs/{timestamp}.log`)
- tensorboard logs are stored as `dir/tensorboard/{timestamp}`
- W&B local run files default to `dir/wandb/` when W&B logging is enabled
- MLflow tracking uses `mlflow.experiment_name` or defaults to `workspace.lineage`
- reproducibility artifacts are stored at `dir/metadata`:
  - `config.full.yaml`
  - `config.canonical.yaml`
  - `config.diff.yaml` (diff against baseline/default config)
  - `git.yaml`
  - `git.diff`
- logging is initialized on the global main process only

`id` is stable across retries for the same run. `timestamp` changes for each attempt.

## Runner state layout

`BaseRunner` uses a small checkpointed runtime boundary:

- `runner`: config snapshot used for semantic resume validation
- `state`: `RunnerState` training metadata (`train`, `elastic`, `rng`)
- `dataloaders`: split-keyed progress for stateful dataloaders

Concrete runners and checkpoint managers add model/EMA/optimizer/scheduler payloads
around that shared boundary. Runtime metadata (`timestamp`, `mode`) and live
trainable/data/metric objects (`model`, `optimizer`, `datasets`, `dataloaders`,
`results`, `meters`, etc.) stay as direct runner attributes for minimal
indirection in hot paths.

## Runner Construction

`TorchRunner` is explicit-first: define components in your runner `__init__`.

Example:

```python
import danling as dl
from torch import nn


class MyRunner(dl.TorchRunner):
    def __init__(self, config):
        super().__init__(config)
        self.model = MyModel(self.config.model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = dl.OPTIMIZERS.build(params=self.model.parameters(), **self.config.optim)

config = dl.RunnerConfig()
runner = MyRunner(config)
```

For class-based stack presets:

```python
import danling as dl

# Default entrypoint is DDP preset:
runner = dl.Runner(config)  # -> TorchRunner by default

# Explicit preset classes:
ddp_runner = dl.TorchRunner(config)
parallel_runner = dl.ParallelRunner(config)

# Stack selection through Runner:
cfg = dl.RunnerConfig(config)
cfg.stack = "parallel"
parallel_from_runner = dl.Runner(cfg)
```

## Parallel topology

Parallel topology is configured through semantic axes:

```yaml
parallel:
  axes:
    replicate: 1
    shard: 8
    context: 1
    pipeline: 4
    tensor: 8
    expert: 1
    expert_tensor: 1
```

Data loading and metric reduction are scoped to the logical `data` domain.
Middle pipeline stages automatically use step-only proxy loaders so pipeline schedules
preserve step count and dataloader state without consuming real input batches locally.
Optimizer control decisions, such as non-finite-gradient skip, are reduced over
the `optimizer` domain covering all configured parallel axes so stages/shards do
not diverge on step/skip boundaries.
`TorchRunner`-based stacks build `torchdata.stateful_dataloader.StatefulDataLoader`
by default so train dataloader progress is checkpointable across restart paths.
Explicit user-provided dataloaders are not rewrapped.

`ParallelRunner` exposes topology-, pipeline-, and FSDP-aware extension points:

For models that must be sharded before allocating their parameters, opt in with
`model_init.meta=True` and override `build_model() -> nn.Module`. The runner calls
that hook inside `torch.device("meta")`; already-bound meta models are also
accepted. `ParallelRunner` applies parallel transforms and FSDP before allocating
storage, then calls `initialize_model(model) -> None` before building the optimizer.
The default initialization calls `model.init_weights()`; override the hook to
initialize or load all parameters and buffers, including non-persistent buffers.
Normal checkpoint/pretrained restoration still follows initialization. Repeated
references to the same parameter or buffer remain tied during allocation, and
their dtypes are retained. Already-initialized models keep their values and the
ordinary device-move path. Mixed initialized/meta tensors require a custom
`materialize_model_values(model) -> nn.Module`; do not bind an optimizer to meta
parameters before materialization.
`model_init_device(model)` defaults to the runtime device; `ParallelRunner`
selects CPU parameter allocation for `CPUOffloadPolicy`, while buffers are
allocated on the runtime device before the model-owned initialization/load hook.
Buffers recreated by that hook are then moved to the runtime device, preserving
their values, dtypes, and current aliases.

`compile.regions=["blocks"]` selects named module groups instead of compiling the
whole model. Override `compile_regions(model) -> Mapping[str, Sequence[nn.Module]]`
or implement `model.compile_regions()` with the same mapping. Groups are resolved
after model transforms and activation-checkpoint wrapping, and use the existing
shared-cache `Compiler.compile_modules`; module identities, state-dict names, and
call hooks remain intact. Unselected modules stay eager. Compilation still
requires `compile.enabled=True`; `GraphRunner` retains its whole-step boundary.

For named activation-checkpoint boundaries, set
`activation_checkpoint.enabled=True` and `activation_checkpoint.regions=["blocks"]`
instead of `module_classes`. Override `activation_checkpoint_regions(model)` or
implement `model.activation_checkpoint_regions()` to return names mapped to
module FQNs (an empty FQN selects the root). The optional
`activation_checkpoint_policy(model, name)` runner hook, or
`model.activation_checkpoint_policy(name)`, returns a native selective-checkpoint
policy callable or list of `OpOverload` objects to save. Other operations are
recomputed. Selective policies require non-reentrant checkpointing;
`activation_checkpoint.preserve_rng_state` is forwarded to native checkpointing
and defaults to `True`.
These hooks are scoped to the model and introduce no global region registry.

- `build_topology`: validate/construct ordered parallel axes from config + world state
- `init_model_parallel_groups`: initialize model-parallel process groups/device mesh
- `parallelize_model`: apply model-specific TP/CP/EP transforms before compile/FSDP wrapping
- `build_pipeline_schedule`: build `torch.distributed.pipelining` schedule from config
- `bind_pipeline_modules`: bind local model parts to pipeline stages
- `apply_activation_checkpointing`: wrap local model parts before compile/FSDP wrapping
- `fsdp_kwargs` / `build_mixed_precision_policy` / `build_offload_policy`: customize FSDP wrapping policy
- `build_optimizer`: deduplicate parameters across local model parts before optimizer build

If any of `parallel.axes.tensor`, `parallel.axes.context`, `parallel.axes.expert`,
or `parallel.axes.expert_tensor` is greater than `1`, the model must either expose
`model.parallelize(parallel_context)` or the runner must override
`parallelize_model`. DanLing intentionally does not silently no-op model-parallel
axes, because TP/CP/EP transforms are model-architecture specific.

## Pipeline-aware hooks

`TorchRunner` exposes core training hooks that can be overridden:

- `train_context`

`ParallelRunner` wires pipeline execution through `build_pipeline_schedule` and
`bind_pipeline_modules`, and routes train/eval/infer paths based on whether a
pipeline schedule is active.

When `parallel.axes.pipeline > 1` and no explicit schedule is provided, `ParallelRunner` now
builds a `torch.distributed.pipelining` schedule internally from
`config.parallel.pipeline_schedule` (default: `"1F1B"`). The default is intentionally
non-interleaved for current pipeline+FSDP stability, with migration to
`"Interleaved1F1B"` planned after upstream issue
`pytorch/pytorch#164756` is resolved.

## Platform support

Supported runner stacks:

| Stack              | Full-state checkpoint path                                  | Model-only export         | Stateful dataloaders                         | TorchFT runtime     |
| ------------------ | ----------------------------------------------------------- | ------------------------- | -------------------------------------------- | ------------------- |
| `torch` / `ddp`    | file default, DCP opt-in                                    | DanLing payload           | default for built loaders                    | supported           |
| `parallel`         | DCP default                                                 | DCP/DanLing model payload | default plus pipeline proxy state delegation | FSDP-only supported |
| `deepspeed` / `ds` | DeepSpeed engine tag directories plus DanLing pointer files | DanLing payload           | default for built loaders via client state   | not supported       |

## Update commits and runtime diagnostics

TorchFT starts one quorum per optimizer accumulation window, before consuming its
first batch. An update must pass `Manager.should_commit()` before the optimizer,
step scheduler, EMA or global update counter advances. Nonfinite losses reject
the entire window, including when `skip_nonfinite_loss` replaces a microbatch
with zero. `ft.use_async_quorum` must remain `false` so healing restores model
and loader state before forward; `ft.manager_timeout_seconds` controls recovery
and `ft.max_consecutive_nonfinite_steps` bounds retries of divergent training.
Manager step metadata is saved with full checkpoints. Asynchronous healing is
rejected because TorchFT applies its recovered state after forward/backward.
Custom training loops should call `_begin_optimizer_window()` before reading
window data, use `train_context()` and finish through `optimizer_step()`.

`profiling.events: true` writes per-rank JSONL events under
`profiling.trace_dir/<timestamp>/rank-<rank>/events-<pid>.jsonl`. Events include
run, rank, optimizer/microstep, process and thread context. Host phase durations
cover microbatches, garbage collection, checkpoint submission, staging waits
and async staging/upload completion; they do not synchronize CUDA. Future
completion events retain the context of their submission and checkpoint target.
Event writes are serialized within each rank and diagnostics are disabled by default.

On CUDA, `profiling.memory_history: true` enables PyTorch allocator history with
at most `memory_history_max_entries` records. `memory_snapshot_interval` enables
periodic dumps; `memory_snapshot_on_oom` dumps on a training-context CUDA OOM.
Only the latest `memory_snapshot_keep` dumps from that runner are retained.
This optional facility uses PyTorch's experimental allocator snapshot API and
requires a build providing it. Events also include allocator retry/OOM counters.
CPU runs leave CUDA memory diagnostics inactive.
