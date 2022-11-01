import traceback
from functools import wraps

from .decorator import flexible_decorator


def flexible_decorator(maybe_decorator=None):
    def decorator(func):
        @wraps(decorator)
        def wrapper(*args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return func(**kwargs)(args[0])
            return func(*args, **kwargs)

        return wrapper

    if maybe_decorator is None:
        return decorator
    return decorator(maybe_decorator)


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
                print("".join(trace), force=True)
                print(
                    f"encoutered when calling {func} with args {args} and kwargs {kwargs}",
                    force=True,
                )

        return wrapper

    return lambda func: decorator(func, error, exclude)
