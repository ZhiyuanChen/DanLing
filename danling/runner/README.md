# Runner

The Runner of DanLing sets up the basic environment for running neural networks.

## Components

For readability and maintainability, there are three levels of Runner:

### [`RunnerBase`][danling.runner.bases.RunnerBase]

[`RunnerBase`][danling.runner.bases.RunnerBase] gives you a basic instinct on what attributes and properties are provided by the Runner.

It works in an AbstractBaseClass manner and should neither be used directly nor be inherited from.

### [`BaseRunner`][danling.runner.BaseRunner]

[`BaseRunner`][danling.runner.BaseRunner] contains core methods of general basic functionality,
such as `init_logging`, `append_result`, `print_result`.

###  [`Runner`][danling.runner.TorchRunner].

`Runner` should only contain platform-specific features.
Currently, only [`TorchRunner`][danling.runner.TorchRunner] is supported.

## Philosophy

DanLing Runner is designed for a 3.5-level experiments management system: **Project**, **Group**, **Experiment**, and, **Run**.

### Project

Project corresponds to your project.

Generally speaking, there should be only one project for each repository.

### Group

A group groups multiple experiments with similar characteristics.

For example, if you run multiple experiments on learning rate, you may want to group them into a group.

Note that Group is a virtual level (which is why it only counts 0.5) and does not corresponds to anything.

### Experiment

Experiment is the basic unit of experiments.

Experiment usually corresponds to a commit, which means the code should be consistent across the experiment.

### Run

Run is the basic unit of running.

Run corresponds to a single run of an experiment, each run may have different hyperparameters.

## Attributes & Properties

### General

- id
- name
- seed
- deterministic

### Progress

- iters
- steps
- epochs
- iter_end
- step_end
- epoch_end
- progress

Note that generally you should only use one of `iter`, `step`, `epoch` to indicate the progress.

### Model

- model
- criterion
- optimizer
- scheduler

### Data

- datasets
- datasamplers
- dataloaders
- batch_size
- batch_size_equivalent

`datasets`, `datasamplers`, `dataloaders` should be a dict with the same keys.
Their keys should be `split` (e.g. `train`, `val`, `test`).

### Results

- results
- latest_result
- best_result
- scores
- latest_score
- best_score
- index_set
- index
- is_best

`results` should be a list of `result`.
`result` should be a dict with the same `split` as keys, like `dataloaders`.
A typical `result` might look like this:
```python
{
    "train": {
        "loss": 0.1,
        "accuracy": 0.9,
    },
    "val": {
        "loss": 0.2,
        "accuracy": 0.8,
    },
    "test": {
        "loss": 0.3,
        "accuracy": 0.7,
    },
}
```

`scores` are usually a list of `float`, and are dynamically extracted from `results` by `index_set` and `index`.
If `index_set = "val"`, `index = "accuracy"`, then `scores = 0.9`.

### DDP

- world_size
- rank
- local_rank
- distributed
- is_main_process
- is_local_main_process

### IO

- experiments_root
- dir
- checkpoint_dir
- log_path
- checkpoint_dir_name

`experiments_root` is the root directory of all experiments, and should be consistent across the **Project**.

`dir` is the directory of a certain **Run**.

There is no attributes/properties for **Group** and **Experiment**.

`checkpoint_dir_name` is relative to `dir`, and is passed to generate `checkpoint_dir` (`checkpoint_dir = os.path.join(dir, checkpoint_dir_name)`).
In practice, `checkpoint_dir_name` is rarely called.

### logging

- log
- logger
- tensorboard
- writer
