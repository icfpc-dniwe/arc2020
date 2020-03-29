import numpy as np
from functools import partial
import typing as t


ImgMatrix = t.NewType('ImgMatrix', np.ndarray)
ImgPair = t.Tuple[ImgMatrix, ImgMatrix]
TestImgPair = t.Tuple[ImgMatrix, t.Optional[ImgMatrix]]
Result = ImgMatrix
Operation = t.Union[t.Callable[[ImgMatrix], ImgMatrix],
                    partial[ImgMatrix]
                    ]
