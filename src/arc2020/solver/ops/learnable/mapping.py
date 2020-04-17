import numpy as np
from .common import merge_maps, recolor, from_list
from .patches import apply_patches, get_all_patches, match_patches, common_patches
from ..patch import Patch, create_patch
from ....mytypes import ImgPair, ImgMatrix
from ..operation import LearnableOperation
from ...classify import OutputSizeType
from typing import List


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


class Patches(LearnableOperation):
    supported_outputs = [OutputSizeType.SQUARE_SAME]

    @staticmethod
    def _make_learnable_operation(patch_size: int = 2):
        def learn(imgs: List[ImgMatrix], targets: List[ImgMatrix]):
            # input_patches = np.concatenate([get_all_patches(img, patch_size) for img in imgs], axis=0)
            # output_patches = np.concatenate([get_all_patches(img, patch_size) for img in targets], axis=0)
            # source_patches, target_patches, ambiguous_patches = match_patches(input_patches, output_patches)
            patches = []
            for idx, (cur_input_img, cur_output_img) in enumerate(zip(imgs, targets)):
                input_patches = get_all_patches(cur_input_img, patch_size)
                output_patches = get_all_patches(cur_output_img, patch_size)
                cur_patches = match_patches(input_patches, output_patches)
                if idx < 1:
                    patches = cur_patches
                elif len(patches) < 1 or len(cur_patches) < 1:
                    patches = []
                    break
                else:
                    patches = common_patches(patches, cur_patches)
            zero_patch = np.zeros((patch_size, patch_size), dtype=imgs[0].dtype)
            first_patch = create_patch(zero_patch, zero_patch)
            return lambda img: apply_patches(img,
                                             from_list([first_patch] + patches),
                                             patch_size)
        return learn
