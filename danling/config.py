from chanfig import Config as _Config
from torch.utils import tensorboard


class Config(_Config):
    """
    Basic Config for all experiments
    """

    id: str = None
    name: str = 'danling'

    experiment_dir: str = 'experiments'
    checkpoint_dir_name: str = 'checkpoints'

    log: bool = True
    tensorboard: bool = False
