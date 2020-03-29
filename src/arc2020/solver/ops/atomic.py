import numpy as np

from ...mytypes import ImgMatrix


def rotate(img: ImgMatrix, num_rotations: int = 1) -> ImgMatrix:
    if num_rotations < 1 or num_rotations > 3:
        return img
    return np.rot90(img, k=num_rotations)
