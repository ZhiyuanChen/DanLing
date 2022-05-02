import sys
from argparse import ArgumentParser as _ArgumentParser

from .config import Config


class ArgumentParser(_ArgumentParser):
    def parse_all_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]
        for arg in args:
            if arg.startswith('--') and args != '--' and arg not in self._option_string_actions:
                self.add_argument(arg)
        if namespace is None:
            namespace = Config()
        namespace, _ = self.parse_known_args(args, namespace)
        return namespace
