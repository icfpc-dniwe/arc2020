import numpy as np
from numba import njit
from .common import merge_maps, recolor, from_list, numba_pad
from ...score import proximity_metric

from ....mytypes import ImgPair, ImgMatrix
from ..operation import LearnableOperation
from ...classify import OutputSizeType
from typing import List, Sequence, Tuple


# @njit
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
    new_img = padded_img.copy()  # np.empty_like(padded_img, dtype=img.dtype)
    for row_idx in range(padded_img.shape[0] - patch_size + 1):
        for col_idx in range(padded_img.shape[1] - patch_size + 1):
            input_patch = padded_img[row_idx:row_idx+patch_size, col_idx:col_idx+patch_size]
            # if len(ambiguous_patches) > 0:
            #     ambiguity_metrics = [proximity_metric(input_patch, cur_patch) for cur_patch in ambiguous_patches]
            #     amb_metric = np.min(np.array(ambiguity_metrics))
            # else:
            #     amb_metric = np.inf
            metrics = [proximity_metric(input_patch, cur_patch) for cur_patch in source_patches]
            best_match = int(np.argmin(np.array(metrics)))
            match_metric = metrics[best_match]
            if match_metric == 0:
                new_img[row_idx:row_idx+patch_size, col_idx:col_idx+patch_size] = target_patches[best_match]
            # else:
            #     new_img[row_idx:row_idx+patch_size, col_idx:col_idx+patch_size] = input_patch
    return new_img[padding:-padding, padding:-padding]


@njit
def get_all_patches(img: ImgMatrix, patch_size: int) -> np.ndarray:
    padding = patch_size // 2
    # padded_img = np.pad(img, padding, 11)
    padded_img = numba_pad(img, padding, 11)
    num_patches = (padded_img.shape[0] - patch_size + 1) * (padded_img.shape[1] - patch_size + 1)
    patches = np.empty((num_patches, patch_size, patch_size), dtype=img.dtype)
    idx = 0
    for row_idx in range(padded_img.shape[0] - patch_size + 1):
        for col_idx in range(padded_img.shape[1] - patch_size + 1):
            patches[idx] = padded_img[row_idx:row_idx+patch_size, col_idx:col_idx+patch_size]
            idx += 1
    return patches


@njit
def is_adequate_patch(all_patches: Sequence[Tuple[np.ndarray, np.ndarray]],
                      test_patch: Tuple[np.ndarray, np.ndarray]
                      ) -> int:
    for idx, (cur_input, cur_target) in zip(*all_patches):
        if np.all(test_patch[0] == cur_input) and np.any(test_patch[1] != cur_target):
            return idx
    return -1


# @njit
def match_patches(input_patches: Sequence[np.ndarray],
                  output_patches: Sequence[np.ndarray]
                  ) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray]]:
    source_patches = []
    target_patches = []
    ambiguous_patches = []
    unique_patches, unique_indices = np.unique(input_patches, axis=0, return_index=True)
    patch_pairs = [(cur_patch, output_patches[cur_idx])
                   for cur_patch, cur_idx in zip(unique_patches, unique_indices)]
    for cur_input, cur_output in zip(input_patches, output_patches):
        # unique = True
        # pair_idx = 0
        # for idx, cur_pair in enumerate(patch_pairs):
        #     if np.all(cur_input == cur_pair[0]) and np.any(cur_output != cur_pair[1]):
        #         unique = False
        #         pair_idx = idx
        #         break
        pair_idx = is_adequate_patch(patch_pairs, (cur_input, cur_output))
        if pair_idx >= 0:
            ambiguous_patches.append(patch_pairs[pair_idx][0])
            del patch_pairs[pair_idx]
        # if unique:
        #     source_patches.append(cur_input)
        #     target_patches.append(cur_output)
        # else:
        #     ambiguous_patches.append(cur_input)
    for pair_input, pair_output in patch_pairs:
        if np.any(pair_input != pair_output):
            source_patches.append(pair_input)
            target_patches.append(pair_output)
        else:
            ambiguous_patches.append(pair_input)
    return source_patches, target_patches, ambiguous_patches


@njit
def common_patches(left_patches: Tuple[Sequence[np.ndarray], Sequence[np.ndarray]],
                   right_patches: Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]
                   ) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
    result_input = []
    result_target = []
    for cur_left in zip(*left_patches):
        for cur_right in zip(*right_patches):
            if np.all(cur_left[0] == cur_right[0]) and np.all(cur_left[1] == cur_right[1]):
                result_input.append(cur_left[0])
                result_target.append(cur_left[1])
    return result_input, result_target
