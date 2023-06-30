# Runner

The Runner of DanLing sets up the basic environment for running neural networks.

## Components

For readability and maintainability, there are three levels of Runner + a RunnerState.

The RunnerState stores all the information that is critical for a run and is stored in the checkpoint (e.g. `epochs`, `run_id`, etc.).
With RunnerState and corresponding weights, you can resume a run from any point.

The Runner contains all runtime information that is irrelevant to the checkpoint (e.g. `world_size`, `rank`, etc.).

### [`RunnerState`][danling.runner.runner_state.RunnerState]

[`RunnerState`][danling.runner.runner_state.RunnerState] stores the state of a run.

All attributes stored in `RunnerState` will be saved in the checkpoint, and thus should be json serialisable.
Except for `@property` of json serialisable attributes.

### [`RunnerBase`][danling.runner.runner_base.RunnerBase]

[`RunnerBase`][danling.runner.abstract_base_runner.RunnerBase] gives you a basic instinct on what attributes and properties are provided by the Runner.

It works in an AbstractBaseClass manner and should neither be used directly nor be inherited from.

### [`BaseRunner`][danling.runner.BaseRunner]

[`BaseRunner`][danling.runner.BaseRunner] contains core methods of general basic functionality,
such as `init_logging`, `append_result`, `print_result`.

### [`Runner`][danling.runner.TorchRunner]

[`Runner`][danling.runner.TorchRunner] should only contain platform-specific features.
Currently, only [`TorchRunner`][danling.runner.TorchRunner] is supported.

## Experiments Management

DanLing Runner is designed for a 3.5-level experiments management system: **Project**, **Group**, **Experiment**, and, **Run**.

### Project

A project corresponds to your project.

Generally speaking, there should be only one project for each repository.

`project_root` is the root directory of all experiments of a certain project, and should be consistent across the project.

### Group

A group groups multiple experiments with similar characteristics.

For example, if you run multiple experiments on learning rate, you may want to group them into a group.

Note that Group is a virtual level (which is why it only counts 0.5) and does not correspond to anything.
There are no attributes/properties for groups.

### Experiment

An experiment is the basic unit of experiments.

Each experiment corresponds to a certain commit, which means the code should be consistent across the experiment.

DanLing will automatically generate `experiment_id` and `experiment_uuid` based on git revision.
They are unique for each commit.

You may also set a catchy custom `experiment_name` to identify each experiment.

### Run

A run is the basic unit of runnings.

Run corresponds to a certain run of an experiment, each run may have different hyperparameters.

DanLing will automatically generate `run_id` and `run_uuid` based on `experiment_uuid` and provided config.
They are unique for each commit and config.

You may also set a catchy custom `run_name` to identify each experiment.

### Identifiers

DanLing has two properties built-in to help you identify each run.

- `id` by default is the join of `experiment_id`, `run_id`, and `uuid`. It is automatically generated hex-strings and is unique for each run.
- `name` by default is `experiment_name-run_name`. It is manually specified and easy to read. Note that `name` is not guaranteed to be unique.

### Directories

To help you manage your experiments, DanLing will automatically generate directories for you.

`dir` is the directory of a certain run, defaults to `{dir/name-id}`.
All run files should be under this directory.

In particular, `checkpoint_dir`, which defaults to `dir/checkpoint_dir_name` contains all checkpoint files.

As a result, your `project_root` should looks like following:

```bash
- {project_root}
-     |- {name}-{id} (equivalents to {experiment_name}-{run_name}-{experiment_id}-{run_id}-{uuid})
-       |
-       |- {checkpoint_dir_name}
-       |    |
-       |    |- best.pth
-       |    |- latest.pth
-       |    |- epoch-10.pth
-       |
-       |- run.log
-       |- runner.yaml
-       |- results.json
-       |- latest.json
-       |- best.json
```
