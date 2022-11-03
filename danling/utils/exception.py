import logging
import traceback
from functools import wraps

from .decorator import flexible_decorator


@flexible_decorator
def catch(error=Exception, exclude=None):
    """
    Catch error except for exclude
    """

    def decorator(func, error=Exception, exclude=None):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error as e:
                if exclude is not None and isinstance(e, exclude):
                    raise e
                trace = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
                logger = logging.getLogger("Exception")
                logger.error("".join(trace))
                logger.error(f"encountered when calling {func} with args {args} and kwargs {kwargs}")

        return wrapper

    return lambda func: decorator(func, error, exclude)
