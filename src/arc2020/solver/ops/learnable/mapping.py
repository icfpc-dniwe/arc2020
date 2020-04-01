import numpy as np
from numba import njit
from .common import merge_maps, recolor
from ...score import proximity_metric

from ....mytypes import ImgPair, ImgMatrix
from ..operation import LearnableOperation
from ...classify import OutputSizeType
from typing import List, Sequence


class ColorMap(LearnableOperation):
    supported_outputs = [OutputSizeType.SAME]

    @staticmethod
    def _make_learnable_operation():
        def learn(img_pairs):
            color_map = merge_maps(img_pairs)
            return lambda img: recolor(img, color_map)
        return learn


class FixedOutput(LearnableOperation):
    supported_outputs = [OutputSizeType.SAME]

    @staticmethod
    def _make_learnable_operation():
        def learn(img_pairs):
            learned_output = img_pairs[0][1]
            return lambda img: learned_output
        return learn


# @njit
def apply_patches(img: ImgMatrix,
                  source_patches: Sequence[np.ndarray],
                  target_patches: Sequence[np.ndarray]
                  ) -> ImgMatrix:
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
def get_all_patches(img: ImgMatrix, patch_size: int) -> np.ndarray:
    padding = patch_size // 2
    padded_img = np.pad(img, padding)
    num_patches = (padded_img.shape[0] - patch_size + 1) * (padded_img.shape[1] - patch_size + 1)
    patches = np.empty((num_patches, patch_size, patch_size), dtype=img.dtype)
    idx = 0
    for row_idx in range(padded_img.shape[0] - patch_size + 1):
        for col_idx in range(padded_img.shape[1] - patch_size + 1):
            patches[idx] = padded_img[row_idx:row_idx+patch_size, col_idx:col_idx+patch_size]
            idx += 1
    return patches


class Patches(LearnableOperation):
    supported_outputs = [OutputSizeType.SAME]

    @staticmethod
    def _make_learnable_operation(patch_size: int = 2):
        def learn(img_pairs):
            patches = []
            return lambda img: apply_patches(img, patches, patches)
        return learn
