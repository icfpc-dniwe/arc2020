import numpy as np
from functools import partial
import typing as t


class ImgMatrix(np.ndarray):

    def __new__(cls, *args):
        self = np.array(*args).view(cls)
        return self


# ImgMatrix = t.NewType('ImgMatrix', np.array)
ImgPair = t.Tuple[ImgMatrix, ImgMatrix]
TestImgPair = t.Tuple[ImgMatrix, t.Optional[ImgMatrix]]
Result = ImgMatrix
Operation = t.Union[t.Callable[[ImgMatrix], ImgMatrix],
                    # partial[ImgMatrix]
                    ]
