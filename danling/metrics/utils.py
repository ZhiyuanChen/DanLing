from torch import distributed as dist


def get_world_size() -> int:
    r"""Return the number of processes in the current process group."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


class flist(list):
    r"""Python `list` that support `__format__` and `to`."""

    def to(self, *args, **kwargs):
        return flist(i.to(*args, **kwargs) for i in self)

    def __format__(self, *args, **kwargs):
        return " ".join([x.__format__(*args, **kwargs) for x in self])
