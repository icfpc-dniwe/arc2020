import numpy as np
from numba import njit
from numba.typed import List as NumbaList
from ....mytypes import ImgMatrix, ImgPair
from typing import List, Dict, Iterable, Any


ColorMapArray = np.ndarray
FullColorMap = np.ndarray


def from_list(python_list: List[Any]) -> NumbaList:
    # numba_list = NumbaList()
    # for x in python_list:
    #     numba_list.append(x)
    # return numba_list
    return python_list


@njit
def numba_pad(arr: np.ndarray, padding_size: int, padding_val: int = 11) -> np.ndarray:
    new_shape = (arr.shape[0] + 2 * padding_size, arr.shape[1] + 2 * padding_size)
    padded = np.zeros(new_shape, dtype=arr.dtype) + np.array([padding_val], dtype=arr.dtype)
    padded[padding_size:-padding_size, padding_size:-padding_size] = arr
    return padded


@njit
def recolor(img: ImgMatrix, color_map: ColorMapArray) -> ImgMatrix:
    new_img = img.copy()
    for row_idx in range(img.shape[0]):
        for col_idx in range(img.shape[1]):
            new_img[row_idx, col_idx] = color_map[img[row_idx, col_idx]]
    return new_img


@njit
def learn_map(source_img: ImgMatrix, target_img: ImgMatrix) -> FullColorMap:
    full_color_map = np.zeros((10, 10), dtype=np.int32)
    for row_idx in range(source_img.shape[0]):
        for col_idx in range(source_img.shape[1]):
            full_color_map[source_img[row_idx, col_idx]][target_img[row_idx, col_idx]] += 1
    return full_color_map


@njit
def merge_maps(img_pairs: List[ImgPair]) -> ColorMapArray:
    full_color_map = np.zeros((10, 10), dtype=np.int32)
    for cur_pair in img_pairs:
        cur_pair_map = learn_map(cur_pair[0], cur_pair[1])
        for idx in range(10):
            for inner_idx, val in enumerate(cur_pair_map[idx]):
                full_color_map[idx][inner_idx] += val
    color_map = np.empty((10,), dtype=np.uint8)
    for cur_idx in range(10):
        color_map[cur_idx] = np.argmax(full_color_map[cur_idx])
    return color_map
