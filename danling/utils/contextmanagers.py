import sys
from contextlib import contextmanager
from typing import Optional

from danling.typing import Exceptions

try:
    import ipdb as pdb
except ImportError:
    import pdb


@contextmanager
def debug(
    enable: bool = True,
    error: Exceptions = Exception,
    exclude: Optional[Exceptions] = None,
):
    """
    Contextmanager to enter debug mode on `error` except for `exclude`.

    `debug` is intended to be used to catch the error and enter debug mode.
    Since it is mainly for development purpose, we intentionally do not catch `KeyboardInterrupt` and `SystemExit`.
    For example, `Runner` saves checkpoint regularly, however, this might break running if the space is full.
    Decorating `save` method with `catch` will allow you to catch these errors and continue your running.

    Args:
        error:
        exclude:
        print_args: Whether to print the arguments passed to the function.
    """

    if not enable:
        yield
    try:
        yield
    except error as exc:  # pylint: disable=W0703
        if exclude is not None and isinstance(exc, exclude):
            raise exc
        _, m, tb = sys.exc_info()  # pylint: disable=C0103
        print(m.__repr__(), file=sys.stderr)  # pylint: disable=C2801
        pdb.post_mortem(tb)
    finally:
        pass
