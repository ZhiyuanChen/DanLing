import sys
from argparse import ArgumentParser as _ArgumentParser
from argparse import Namespace
from ast import literal_eval


class ArgumentParser(_ArgumentParser):
    def parse_all_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]
        for arg in args:
            if arg.startswith('--') and arg not in self._option_string_actions:
                self.add_argument(arg)
        if namespace is None:
            namespace = Config()
        namespace, _ = self.parse_known_args(args, namespace)
        return namespace


class Config(Namespace):
    def __setattr__(self, name, value):
        try:
            value = literal_eval(value)
        except ValueError:
            pass
        if '.' in name:
            name = name.split('.')
            name, rest = name[0], '.'.join(name[1:])
            setattr(self, name, type(self)())
            setattr(getattr(self, name), rest, value)
        else:
            super().__setattr__(name, value)

    def dict(self):
        dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                dict[k] = v.dict()
            else:
                dict[k] = v
        return dict
