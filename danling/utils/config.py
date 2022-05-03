from argparse import Namespace
from ast import literal_eval
from collections import OrderedDict


class Config(Namespace):
    """
    Basic Config for all experiments
    """

    id: str = None
    name: str = 'danling'

    experiment_dir: str = 'experiments'
    checkpoint_dir_name: str = 'checkpoints'

    log: bool = True
    tensorboard: bool = False

    def __setattr__(self, name, value) -> None:
        try:
            value = literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        if '.' in name:
            name = name.split('.')
            name, rest = name[0], '.'.join(name[1:])
            if not hasattr(self, name):
                setattr(self, name, type(self)())
            setattr(getattr(self, name), rest, value)
        else:
            super().__setattr__(name, value)

    def dict(self) -> OrderedDict:
        dict = OrderedDict()
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                dict[k] = v.dict()
            else:
                dict[k] = v
        return dict
