import numpy as np

from ...mytypes import ImgMatrix
from .operation import Operation
from ..classify import OutputSizeType


class Rotate(Operation):
    supported_outputs = [OutputSizeType.SAME]

    @staticmethod
    def _make_operation(num_rotations: int = 1):
        return lambda img: np.rot90(img, k=num_rotations)

class Transpose(Operation):
    supported_outputs = [OutputSizeType.SAME]

    @staticmethod
    def _make_operation():
        return lambda img: img.T

class Flip(Operation):
    supported_outputs = [OutputSizeType.SAME]

    @staticmethod
    def _make_operation(flip_axis: int = 1):
        return lambda img: np.flip(img, axis=flip_axis)
