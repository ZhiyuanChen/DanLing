# DanLing Runner

The Runner module provides a unified interface for the complete deep learning model lifecycle, supporting training, evaluation, inference, and experiment management across multiple distributed computing platforms.

## Core Concepts

DanLing uses a two-level architecture with `Runner` + `Config` to provide a flexible, extensible framework:

### Config

`Config` is a specialized dictionary that stores all serializable state, including:

- Hyperparameters (learning rate, batch size, etc.)
- Model configuration
- Dataset settings

Config extends the [`chanfig.Config`][chanfig.Config], it is hierarchical and can be accessed via attributes:

```python
config = dl.Config()
config.optim.lr = 1e-3
config.dataloader.batch_size = 32
config.precision = "fp16"
```

Configurations can be serialized to YAML/JSON and loaded from the command line:

```python
config = dl.Config()
config.parse()  # Loads from command line arguments
config.dump("config.yaml")  # Dump to yaml file
```

By default, the `Config` will load the config file specified in `--config config.yaml`.

### Runner

`Runner` is the central class that manages the entire model lifecycle. Key components include:

- **State Management**: Training/evaluation state and progress tracking
- **Model Handling**: Loading/saving model checkpoints
- **Operations**: Training, evaluation, and inference loops
- **Metrics**: Tracking and logging performance metrics
- **Distributed Execution**: Managing multi-device/multi-node execution

### Available Platforms

DanLing supports multiple distributed computing platforms:

1. **TorchRunner**: Native PyTorch DistributedDataParallel (DDP) implementation
2. **DeepSpeedRunner**: Microsoft DeepSpeed integration for large models
3. **AccelerateRunner**: HuggingFace Accelerate for simplified multi-platform execution

The base [`Runner`][danling.runners.Runner] class automatically selects the appropriate platform based on your configuration and available packages.

## Customizing Your Workflow

### Custom Configuration

Create a configuration class for better IDE support and documentation:

```python
class MyConfig(dl.Config):
    # Class-level defaults
    # The values will be copied into the config object once its initialized.
    # Note that you must specify a proper type hint, otherwise they will be considered as attributes for maintaining the object.
    # Use `typing.Any` if no type hints are appropriate.
    # If you wish to access the attributes for maintaining your customized Config class, use config.getattr("attr_name", default_value).

    epoch_end: int = 10
    log_interval: int = 100
    score_metric: str = "accuracy"

    def __init__(self):
        super().__init__()
        self.network.type = "resnet50"
        self.optim.type = "adamw"
        self.optim.lr = 5e-5

    # Called after parsing command line arguments
    # If no command line support needed, call `config.boot` for post-processing.
    def post(self):
        super().post()
        # Generate a descriptive experiment name
        self.experiment_name = f"{self.network.type}_{self.optim.lr}"
```

### Optimizers and Schedulers

DanLing supports all PyTorch optimizers plus DeepSpeed optimizers when available:

```python
from danling.optim import OPTIMIZERS, SCHEDULERS

# Using string identifiers (recommended for config-driven approach)
config.optim.type = "adamw"  # Options: sgd, adam, adamw, lamb, lion, etc.
config.optim.lr = 1e-3
config.optim.weight_decay = 0.01
optimizer = OPTIMIZERS.build(model.parameters(), **config.optim)

# Using the registry directly
optimizer = OPTIMIZERS.build("sgd", params=model.parameters(), lr=1e-3)

# Scheduler configuration
config.sched.type = "cosine"  # Options: step, cosine, linear, etc.
config.sched.total_steps = 100
scheduler = SCHEDULERS.build(optimizer, **config.sched)
```

### Mixed Precision and Performance Optimization

Enable mixed precision training for faster execution:

```python
config.precision = "fp16"  # Options: fp16, bf16
config.accum_steps = 4  # Gradient accumulation steps
```

### Custom Metrics

Register custom metrics or use built-in ones:

```python
# Built-in metrics
runner.metrics = dl.metrics.multiclass_metrics(num_classes=10)

# Custom metrics
from torchmetrics import Accuracy
runner.metrics = dl.MetricMeters({
    "accuracy": Accuracy(task="multiclass", num_classes=10),
    "my_custom_metric": MyCustomMetric()
})
```

## Extending DanLing

DanLing is designed to be extensible at multiple levels:

### Extension Pattern 1: Customize the Runner

Extend the `Runner` class (not TorchRunner directly) to preserve platform selection:

```python
class MyCustomRunner(dl.Runner):
    """Custom runner that works across all platforms."""

    def __init__(self, config):
        super().__init__(config)
        # Your initialization code

    def custom_method(self):
        """Add new functionality."""
        return "Custom result"

    # Override lifecycle methods when needed
    def build_dataloaders(self):
        """Customize dataloader creation."""
        super().build_dataloaders()
        # Your modifications
```

### Extension Pattern 2: Custom Distributed Framework

Extend `TorchRunner` only when implementing a new distributed training framework:

```python
class MyDistributedRunner(dl.TorchRunner):
    """Implement a custom distributed training framework."""

    def init_distributed(self):
        """Initialize custom distributed environment."""
        # Your distributed initialization code

    def backward(self, loss):
        """Custom backward pass."""
        # Your custom backward implementation
```

### Lifecycle Methods

Key methods you can override:

```python
# Data handling
def build_dataloaders(self)
def collate_fn(self, batch)

# Training/evaluation
def train_step(self, data)
def evaluate_step(self, data)
def advance(self, loss)

# Checkpointing
def save_checkpoint(self, name="latest")
def load_checkpoint(self, checkpoint)
```

## Platform Selection

DanLing's `Runner` automatically selects the appropriate platform using this logic:

1. Check the `platform` config value (`"auto"`, `"torch"`, `"deepspeed"`, or `"accelerate"`)
2. If `"auto"`, select DeepSpeed if available, otherwise use PyTorch
3. Dynamically transform the Runner into the selected platform implementation

You can explicitly select a platform:

```python
config = dl.Config({
    "platform": "deepspeed",  # Force DeepSpeed
    "deepspeed": {
        # DeepSpeed-specific configuration
        "zero_optimization": {"stage": 2}
    }
})
```

### Platform Comparison

| Platform         | Best For                                   | Key Features                           |
| ---------------- | ------------------------------------------ | -------------------------------------- |
| TorchRunner      | Flexibility, custom extensions             | Native PyTorch DDP, most customizable  |
| DeepSpeedRunner  | Very large models (billions of parameters) | ZeRO optimization, CPU/NVMe offloading |
| AccelerateRunner | Multi-platform compatibility               | Simple API, works on CPU/GPU/TPU       |

## Experiment Management

DanLing organizes experiments in a hierarchical system:

```
{project_root}/
    ├── {experiment_name}-{run_name}-{id}/  # Unique run directory
    │   ├── checkpoints/                    # Model checkpoints
    │   │   ├── best/                       # Best model
    │   │   ├── latest/                     # Most recent model
    │   │   └── epoch-{N}/                  # Periodic checkpoints
    │   ├── run.log                         # Execution logs
    │   ├── runner.yaml                     # Runner configuration
    │   ├── results.json                    # Complete results
    │   ├── latest.json                     # Latest results
    │   └── best.json                       # Best results
    └── ...
```

### Identifiers

DanLing provides both human-friendly and unique identifiers:

- `experiment_name`: Human-readable name for the experiment
- `run_name`: Human-readable name for the specific run
- `experiment_id` and `run_id`: Automatically generated unique IDs
- `id`: Combined unique identifier

### Checkpointing

Save and restore checkpoints:

```python
# Automatic checkpointing (controlled by config)
config.save_interval = 5  # Save every 5 epochs
config.checkpoint_dir = "checkpoints"  # Custom checkpoint directory

# Manual checkpointing
runner.save_checkpoint(name="custom_checkpoint")
runner.load_checkpoint("path/to/checkpoint")

# Resuming from previous run
config.checkpoint = "path/to/previous/checkpoint"
runner = MyRunner(config)  # Will resume from checkpoint
```

## Production and MLOps

### Reproducibility

Ensure reproducible results:

```python
config.seed = 42  # Set random seed
config.deterministic = True  # Force deterministic algorithms
```

### Logging and Visualization

Configure logging and TensorBoard:

```python
config.log = True  # Enable file logging
config.tensorboard = True  # Enable TensorBoard
config.log_interval = 100  # Log every 100 iterations
```

### Distributed Training

Configure multi-GPU and multi-node training:

```python
# Single-node, multi-GPU
config.platform = "torch"  # or "deepspeed" or "accelerate"
# Environment variables will be used: WORLD_SIZE, RANK, LOCAL_RANK

# Manual configuration
config.world_size = 4  # Number of processes
config.rank = 0  # Global rank
config.local_rank = 0  # Local rank
```

## Examples

DanLing includes several example implementations:

### MNIST with PyTorch

Complete image classification example using TorchRunner:

```python
--8<-- "demo/torch_mnist.py"
```

### IMDB Text Classification with Accelerate

Text classification using HuggingFace Transformers and Accelerate:

```python
--8<-- "demo/accelerate_imdb.py"
```

For more examples, see the [demo directory](../../demo/).
