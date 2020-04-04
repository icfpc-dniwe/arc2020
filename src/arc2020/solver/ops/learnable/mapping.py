import numpy as np
from numba import njit
from .common import merge_maps, recolor, from_list, numba_pad
from ...score import proximity_metric

from ....mytypes import ImgPair, ImgMatrix
from ..operation import LearnableOperation
from ...classify import OutputSizeType
from typing import List, Sequence


class ColorMap(LearnableOperation):
    supported_outputs = [OutputSizeType.SQUARE_SAME]

    @staticmethod
    def _make_learnable_operation(allow_trivial: bool = False):
        def learn(imgs, targets):
            color_map = merge_maps(imgs, targets, allow_trivial)
            return lambda img: recolor(img, color_map)
        return learn


class FixedOutput(LearnableOperation):
    supported_outputs = [OutputSizeType.SQUARE_SAME]

    @staticmethod
    def _make_learnable_operation():
        def learn(imgs, targets):
            learned_output = targets[0]
            return lambda img: learned_output
        return learn


@njit
def apply_patches(img: ImgMatrix,
                  source_patches: Sequence[np.ndarray],
                  target_patches: Sequence[np.ndarray],
                  ambiguous_patches: Sequence[np.ndarray],
                  patch_size: int,
                  ) -> ImgMatrix:
    if len(source_patches) < 2:
        return img
    source_patches = source_patches[1:]
    target_patches = target_patches[1:]
    ambiguous_patches = ambiguous_patches[1:]
    padding = patch_size // 2
    # padded_img = np.pad(img, padding, 11)
    padded_img = numba_pad(img, padding)
    new_img = np.empty_like(padded_img, dtype=img.dtype)
    for row_idx in range(padded_img.shape[0] - patch_size + 1):
        for col_idx in range(padded_img.shape[1] - patch_size + 1):
            input_patch = padded_img[row_idx:row_idx+patch_size, col_idx:col_idx+patch_size]
            if len(ambiguous_patches) > 0:
                ambiguity_metrics = [proximity_metric(input_patch, cur_patch) for cur_patch in ambiguous_patches]
                amb_metric = np.min(np.array(ambiguity_metrics))
            else:
                amb_metric = np.inf
            metrics = [proximity_metric(input_patch, cur_patch) for cur_patch in source_patches]
            best_match = int(np.argmin(np.array(metrics)))
            match_metric = metrics[best_match]
            if match_metric < amb_metric:
                new_img[row_idx:row_idx+patch_size, col_idx:col_idx+patch_size] = target_patches[best_match]
            else:
                new_img[row_idx:row_idx + patch_size, col_idx:col_idx + patch_size] = input_patch
    return new_img[padding:-padding, padding:-padding]


@njit
def get_all_patches(img: ImgMatrix, patch_size: int) -> np.ndarray:
    padding = patch_size // 2
    # padded_img = np.pad(img, padding, 11)
    padded_img = numba_pad(img, padding)
    num_patches = (padded_img.shape[0] - patch_size + 1) * (padded_img.shape[1] - patch_size + 1)
    patches = np.empty((num_patches, patch_size, patch_size), dtype=img.dtype)
    idx = 0
    for row_idx in range(padded_img.shape[0] - patch_size + 1):
        for col_idx in range(padded_img.shape[1] - patch_size + 1):
            patches[idx] = padded_img[row_idx:row_idx+patch_size, col_idx:col_idx+patch_size]
            idx += 1
    return patches


class Patches(LearnableOperation):
    supported_outputs = [OutputSizeType.SQUARE_SAME]

    @staticmethod
    def _make_learnable_operation(patch_size: int = 2):
        def learn(imgs: List[ImgMatrix], targets: List[ImgMatrix]):
            input_patches = np.concatenate([get_all_patches(img, patch_size) for img in imgs], axis=0)
            output_patches = np.concatenate([get_all_patches(img, patch_size) for img in targets], axis=0)
            source_patches = []
            target_patches = []
            ambiguous_patches = []
            unique_input, unique_indices, input_counts = np.unique(input_patches, axis=0,
                                                                   return_index=True, return_counts=True)
            for cur_idx, cur_count in zip(unique_indices, input_counts):
                if cur_count == 1:
                    source_patches.append(input_patches[cur_idx])
                    target_patches.append(output_patches[cur_idx])
                else:
                    ambiguous_patches.append(input_patches[cur_idx])

            first_patch = unique_input[0]
            return lambda img: apply_patches(img,
                                             from_list([first_patch] + source_patches),
                                             from_list([first_patch] + target_patches),
                                             from_list([first_patch] + ambiguous_patches),
                                             patch_size)
        return learn
