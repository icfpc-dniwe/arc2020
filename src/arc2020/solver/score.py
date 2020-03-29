import numpy as np
from numba import njit
from ..mytypes import ImgMatrix


@njit
def proximity_metric(left_img: ImgMatrix, right_img: ImgMatrix) -> int:
    metric = 0
    if left_img.shape != right_img.shape:
        left_shape = np.array(left_img.shape)
        right_shape = np.array(right_img.shape)
        max_shape = np.maximum(left_shape, right_shape)
        metric += np.sum(2 * max_shape - left_shape - right_shape)
        tmp = np.zeros(max_shape, dtype=np.uint8)
        tmp[:left_img.shape[0], :left_img.shape[1]] = left_img
        left_img = tmp
        tmp = np.zeros(max_shape, dtype=np.uint8)
        tmp[:right_img.shape[0], :right_img.shape[1]] = right_img
        right_img = tmp
    metric += np.sum(left_img != right_img)
    return metric
