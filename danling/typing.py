import os
from typing import IO, Tuple, Type, Union

PathStr = Union[os.PathLike, str, bytes]
File = Union[PathStr, IO]
Exceptions = Union[Type[Exception], Tuple[Type[Exception], ...]]
