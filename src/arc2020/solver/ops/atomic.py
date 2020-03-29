import numpy as np
from numba import njit

from ...mytypes import ImgMatrix


def rotate(img: ImgMatrix, num_rotations: int = 1) -> ImgMatrix:
    if num_rotations < 1 or num_rotations > 3:
        return img
    return np.rot90(img, k=num_rotations)


@njit
def transpose(img: ImgMatrix) -> ImgMatrix:
    return img.T


def flip(img: ImgMatrix, flip_axis: int = 0) -> ImgMatrix:
    if flip_axis < 0 or flip_axis > 1:
        return img
    return np.flip(img, axis=flip_axis)
