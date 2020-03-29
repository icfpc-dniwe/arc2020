import numpy as np
from numba import njit
from ...task import Task
from ...mytypes import Operation, ImgMatrix, ImgPair
from typing import List, Dict, Iterable


ColorMap = np.ndarray
FullColorMap = np.ndarray


@njit
def recolor(img: ImgMatrix, color_map: ColorMap) -> ImgMatrix:
    new_img = img.copy()
    for row_idx in range(img.shape[0]):
        for col_idx in range(img.shape[1]):
            new_img[row_idx, col_idx] = color_map[img[row_idx, col_idx]]
    return new_img


@njit
def learn_map(source_img: ImgMatrix, target_img: ImgMatrix) -> FullColorMap:
    # full_color_map = {idx: [0 for _ in range(10)] for idx in range(10)}
    full_color_map = np.zeros((10, 10), dtype=np.int32)
    for row_idx in range(source_img.shape[0]):
        for col_idx in range(source_img.shape[1]):
            full_color_map[source_img[row_idx, col_idx]][target_img[row_idx, col_idx]] += 1
    # color_map = {idx: np.argmax(cur_targets) for idx, cur_targets in full_color_map.items()}
    return full_color_map


@njit
def merge_maps(img_pairs: List[ImgPair]) -> ColorMap:
    full_color_map = np.zeros((10, 10), dtype=np.int32)
    for cur_pair in img_pairs:
        cur_pair_map = learn_map(cur_pair[0], cur_pair[1])
        for idx in range(10):
            for inner_idx, val in enumerate(cur_pair_map[idx]):
                full_color_map[idx][inner_idx] += val
    # color_map = {idx: np.argmax(cur_targets) for idx, cur_targets in full_color_map.items()}
    color_map = np.empty((10,), dtype=np.uint8)
    for cur_idx in range(10):
        color_map[cur_idx] = np.argmax(full_color_map[cur_idx])
    return color_map


def learn_color_map(img_pairs: List[ImgPair]) -> Operation:
    color_map = merge_maps(img_pairs)

    def op(img: ImgMatrix) -> ImgMatrix:
        return recolor(img, color_map)

    return op
