import numpy as np
from .common import merge_maps, recolor, from_list
from .patches import apply_patches, get_all_patches, match_patches, common_patches
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
            source_patches = []
            target_patches = []
            ambiguous_patches = []
            for idx, (cur_input_img, cur_output_img) in enumerate(zip(imgs, targets)):
                input_patches = get_all_patches(cur_input_img, patch_size)
                output_patches = get_all_patches(cur_output_img, patch_size)
                cur_source, cur_target, _ = match_patches(input_patches, output_patches)
                if idx < 1:
                    source_patches = cur_source
                    target_patches = cur_target
                elif len(source_patches) < 1 or len(cur_source) < 1:
                    source_patches = []
                    target_patches = []
                    break
                else:
                    source_patches, target_patches = common_patches((source_patches, target_patches),
                                                                    (cur_source, cur_target))
            first_patch = [np.zeros((patch_size, patch_size), dtype=imgs[0].dtype)]
            return lambda img: apply_patches(img,
                                             from_list([first_patch] + source_patches),
                                             from_list([first_patch] + target_patches),
                                             from_list([first_patch] + ambiguous_patches),
                                             patch_size)
        return learn
