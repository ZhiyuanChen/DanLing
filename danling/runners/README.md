# DanLing Runner

DanLing runners are now **Torch-only** and focus on scalable distributed training.

The new design borrows from TorchTitan's Trainer philosophy:

- A single, explicit training lifecycle
- Extension by subclassing `TorchRunner`
- Backend-specific distributed logic in dedicated subclass runners
- Minimal shared contracts in `BaseRunner`

## Core APIs

- `RunnerConfig`: `chanfig`-based long-term configuration surface
- `Runner`: stack-selecting entrypoint (`ddp`/`torch`, `fsdp`, `tppp`; default `ddp`)
- `TorchRunner`: shared Torch training core (single canonical loop)
- `TorchRunner` / `FsdpRunner` / `TpppRunner`: class-based stack runners

`TpppRunner` topology uses:

- `tp_degree`
- `pp_degree`
- inferred `dp_degree = WORLD_SIZE / (tp_degree * pp_degree)`

Data loading and metric reduction are data-parallel (`dp`) scoped in `TpppRunner`.

`TpppRunner` exposes topology- and pipeline-aware extension points:

- `build_topology`: validate/construct TP/PP/DP topology from config + world state
- `init_model_parallel_groups`: initialize model-parallel process groups/device mesh
- `build_pipeline_schedule`: build `torch.distributed.pipelining` schedule from config
- `bind_pipeline_modules`: bind local model parts to pipeline stages
- `build_optimizer`: deduplicate parameters across local model parts before optimizer build

## BaseRunner contract

`BaseRunner` is inheritance-first and keeps only cross-backend functionality.
There is no plugin/spec registry layer.

Core lifecycle methods (all overridable):

- `initialize_state`
- `initialize_runtime`
- `initialize_services`

Execution API stubs in `BaseRunner` (all overridable):

- `train` / `train_epoch` / `train_steps` / `train_step`
- `evaluate` / `evaluate_epoch` / `evaluate_steps` / `evaluate_step`
- `infer`

Step-mode semantics:

- `train` dispatches to `train_steps` when `epochs` is unset (`is_step_mode=True`).
- `evaluate_steps(split, steps=None)` is always bounded-step evaluation.
  When `steps` is omitted, it defaults to `max(steps_budget // 20, 1)`.

Checkpoint/score contracts (all overridable):

- `load_model`
- `read_checkpoint`
- `load_optimizer` / `load_scheduler`
- `load_state_dict`
- `adapt_checkpoint_payload_for_save` / `adapt_checkpoint_payload_for_load`
- `scores`
- `best_index`

## Launch modes

- Scratch: `runner = Runner(config)`
- Resume: `runner = Runner.from_checkpoint(checkpoint)` or `runner.load_checkpoint(checkpoint)`
- Finetune: `runner = Runner.from_pretrained(config, checkpoint)` or `runner.load_pretrained(checkpoint)`

Config source hints:

- `config.auto_resume`: when enabled, auto-resume from backend latest checkpoint source
- `config.resume`: source used for full-state resume workflows
- `config.pretrained`: source used for model-only initialization workflows
- `config.checkpoint`: checkpoint policy only (backend/async/interval/retention), never a source path

Source precedence:

- `resume` > `auto_resume` > `pretrained`

## Checkpoint files

- `latest.pth`: rolling latest checkpoint
- `best.pth`: rolling best checkpoint
- Archived checkpoints (when `checkpoint_interval` is set):
  - epoch mode: `ckpt-e{epoch:06d}.pth`
  - step mode: `ckpt-s{global_step:012d}.pth`

Periodic checkpoint attempts are controlled by `checkpoint_interval`.
When `checkpoint_interval` is unset, runner defaults are used by mode (`epochs`: every epoch, `steps`: budget/20).

`best` and archived files are updated from `latest` via hardlink-first aliasing (copy fallback), so training is not blocked by alias I/O failures.

Checkpoint persistence internals are implemented under `danling.runners.checkpoints`:

- `CheckpointManager`: base contract
- `FileCheckpointManager`: default async filesystem manager with reliable archive/best queueing plus
  latest-wins coalescing for non-critical saves
- `TorchDistributedCheckpointManager`: Torch DCP backend used when `checkpoint_backend="dcp"` in `TorchRunner`

`FsdpRunner` and `TpppRunner` force `checkpoint_backend="dcp"` by default.

Distributed state-dict invariants:

- FSDP/TPPP distributed runs require DCP state-dict APIs for model/optimizer restore.
- Checkpoint restore order is model -> optimizer -> scheduler.
- File checkpoint fallback is only for non-distributed/basic flows.

Async policy is configured by `checkpoint_async_mode`:

- `disabled`: synchronous checkpoint writes
- `async`: asynchronous uploads
- `async_with_pinned_mem`: process-based async uploads with staging hook support (`maybe_wait_for_staging`)

When `checkpoint_async_mode` is unset, DanLing falls back to legacy `checkpoint_async` (`True` -> `async`, `False` -> `disabled`).

## Directory hierarchy

DanLing now uses an experiment layout with internal runtime IDs:

`workspace_root/lineage[-code_id]/experiment-config_id`

- `lineage`: top-level lineage namespace
- `code_id`: git code identity (`<short_sha>` for clean trees, `<short_sha>-d<diff_sha10>` when dirty) appended to lineage when available
- `experiment`: experiment namespace
- `config_id`: deterministic hash of canonical config (`hash(config)` derived)
  By default:

- `dir` points to `workspace_root/lineage[-code_id]/experiment-config_id`
- checkpoints and result snapshots are stored at `dir`
- logs are stored as `dir/logs/{id}.log`
- tensorboard logs are stored as `dir/tensorboard/{id}`
- reproducibility artifacts are stored at `dir/metadata`:
  - `config.full.yaml`
  - `config.canonical.yaml`
  - `config.diff.yaml` (diff against baseline/default config)
  - `git.yaml`
  - `git.diff`
- logging is initialized on the global main process only

The runtime `id` is timestamp-based and not a filesystem directory level.

## Runner state layout

`BaseRunner` uses one checkpointed boundary:

- `RunnerState`: checkpointed training metadata (`train`, `elastic`, `rng`)

Non-checkpointed runtime metadata (`timestamp`, `mode`) and all
trainable/data/metric objects (`model`, `optimizer`, `datasets`, `dataloaders`, `results`, `meters`, etc.)
are direct runner attributes for minimal indirection in hot paths.

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
fsdp_runner = dl.FsdpRunner(config)
tppp_runner = dl.TpppRunner(config)

# Stack selection through Runner:
cfg = dl.RunnerConfig(config)
cfg.stack = "fsdp"
fsdp_from_runner = dl.Runner(cfg)
```

## Pipeline-aware hooks

`TorchRunner` exposes core training hooks that can be overridden:

- `train_context`

`TpppRunner` wires pipeline execution through `build_pipeline_schedule` and
`bind_pipeline_modules`, and routes train/eval/infer paths based on whether a
pipeline schedule is active.

When `pp_degree > 1` and no explicit schedule is provided, `TpppRunner` now
builds a `torch.distributed.pipelining` schedule internally from
`config.tppp.pipeline_schedule` (default: `"1F1B"`). The default is intentionally
non-interleaved for current PP+FSDP stability, with migration to
`"Interleaved1F1B"` planned after upstream issue
`pytorch/pytorch#164756` is resolved.

## Platform support

Supported runner stacks:

- `torch` / `ddp`
- `deepspeed` / `ds` (ZeRO-1/2 focused)
- `fsdp`
- `tppp`
- `dtpp`

## Migration note

This rewrite is intentionally breaking:

- No compatibility layer for legacy DeepSpeed/Accelerate runner APIs
- No compatibility promise for legacy backend-specific checkpoints
