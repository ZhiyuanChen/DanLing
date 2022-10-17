import traceback
from functools import wraps


def catch(error=Exception, exclude=None):
    """
    Catch error except for exclude
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error as e:
                if exclude is not None and isinstance(e, exclude):
                    raise e
                trace = traceback.format_exception(
                    etype=type(e), value=e, tb=e.__traceback__
                )
                print("".join(trace), force=True)
                print(
                    f"encoutered when calling {func} with args {args} and kwargs {kwargs}",
                    force=True,
                )

        return wrapper

    return decorator