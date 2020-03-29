import numpy as np
import typing as t


ImgMatrix = t.NewType('ImgMatrix', np.ndarray)
ImgPair = t.Tuple[ImgMatrix, ImgMatrix]
TestImgPair = t.Tuple[ImgMatrix, t.Optional[ImgMatrix]]
Result = ImgMatrix
