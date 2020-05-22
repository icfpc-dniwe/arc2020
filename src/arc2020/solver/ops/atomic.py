import numpy as np

from ...mytypes import ImgMatrix
from .operation import Operation
from ..classify import OutputSizeType


class Rotate(Operation):
    @staticmethod
    def _make_operation(num_rotations: int = 1):
        return lambda img: np.rot90(img, k=num_rotations)

class Transpose(Operation):
    @staticmethod
    def _make_operation():
        return lambda img: img.T

class Flip(Operation):
    @staticmethod
    def _make_operation(flip_axis: int = 1):
        return lambda img: np.flip(img, axis=flip_axis)

class Inverse(Operation):
    @staticmethod
    def _make_operation(bg_color: int = 0):
        def inv(img: ImgMatrix) -> ImgMatrix:
            img = img.copy()
            bg_mask = img == bg_color
            other_color = img[1 - bg_mask][0]
            img[bg_mask] = other_color
            img[1 - bg_mask] = bg_color
            return img
        return inv
