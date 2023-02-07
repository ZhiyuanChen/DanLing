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

###  [`Runner`][danling.runner.TorchRunner]

[`Runner`][danling.runner.TorchRunner] should only contain platform-specific features.
Currently, only [`TorchRunner`][danling.runner.TorchRunner] is supported.

## Experiments Management

DanLing Runner is designed for a 3.5-level experiments management system: **Project**, **Group**, **Experiment**, and, **Run**.

### Project

Project corresponds to your project.

Generally speaking, there should be only one project for each repository.

`Runner.experiments_root` is the root directory of all experiments of a project, and should be consistent across the project.

### Group

A group groups multiple experiments with similar characteristics.

For example, if you run multiple experiments on learning rate, you may want to group them into a group.

Note that Group is a virtual level (which is why it only counts 0.5) and does not correspond to anything.

### Experiment

Experiment is the basic unit of experiments.

Experiment usually corresponds to a commit, which means the code should be consistent across the experiment.

### Run

Run is the basic unit of running.

Run corresponds to a single run of an experiment, each run may have different hyperparameters.

Each run of an experiment must have a unique `Runner.id`.
Although the best practice is to ensure `Runner.id` is unique across the entire project.

`Runner.dir` is the directory of a certain run.
All run files should be under this directory.

In particular, `Runner.checkpoint_dir`, which is generated by `os.path.join(Runner.dir, Runner.checkpoint_dir_name)` contains all checkpoint files.

There is no attributes/properties for **Group** and **Experiment**.