import numpy as np
from numba import njit
from .common import merge_maps, recolor
from ...score import proximity_metric

from ....mytypes import ImgPair, ImgMatrix, Operation
from typing import List


def learn_color_map(img_pairs: List[ImgPair]) -> Operation:
    color_map = merge_maps(img_pairs)

    def op(img: ImgMatrix) -> ImgMatrix:
        return recolor(img, color_map)

    return op


def learn_fixed_output(img_pairs: List[ImgPair]) -> Operation:
    learned_output = img_pairs[0][1].copy()

    def op(img: ImgMatrix) -> ImgMatrix:
        return learned_output

    return op


# @njit
def apply_patches(img: ImgMatrix, source_patches: List[np.ndarray], target_patches: List[np.ndarray]) -> ImgMatrix:
    patch_size = source_patches[0].shape[0]
    padding = patch_size // 2
    padded_img = np.pad(img, padding)
    new_img = np.zeros_like(padded_img)
    for row_idx in range(padded_img.shape[0] - patch_size + 1):
        for col_idx in range(padded_img.shape[1] - patch_size + 1):
            input_patch = padded_img[row_idx:row_idx+patch_size, col_idx:col_idx+patch_size]
            metrics = [proximity_metric(input_patch, cur_patch) for cur_patch in source_patches]
            best_match = int(np.argmin(metrics))
            new_img[row_idx:row_idx+patch_size, col_idx:col_idx+patch_size] = target_patches[best_match]
    return new_img[padding:-padding, padding:-padding]


# @njit
def get_all_patches(img: ImgMatrix, patch_size: int) -> List[np.ndarray]:
    patches = []
    padding = patch_size // 2
    padded_img = np.pad(img, padding)
    for row_idx in range(padded_img.shape[0] - patch_size + 1):
        for col_idx in range(padded_img.shape[1] - patch_size + 1):
            pass


def learn_patches(img_pairs: List[ImgPair], patch_size: int = 2) -> Operation:
    patches = []

    def op(img: ImgMatrix) -> ImgMatrix:
        return apply_patches(img, patches, patches)

    return op
