import os
from typing import IO, Union

PathStr = Union[os.PathLike, str, bytes]
File = Union[PathStr, IO]
