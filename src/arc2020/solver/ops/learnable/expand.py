import numpy as np
from .common import merge_maps, recolor, from_list
from ..expand import ExpandPixel, EXPAND_PATCHES
from ....mytypes import ImgPair, ImgMatrix
from ..operation import LearnableOperation
from ..atomic import Transpose, Rotate, Flip, Inverse
from ...classify import OutputSizeType
from ...greedy import get_step_scores
from ...score import proximity_metric
from typing import List


available_atomic = \
    [Transpose.make_operation(), Inverse.make_operation()] + \
    [Rotate.make_operation(n) for n in range(1, 4)] + \
    [Flip.make_operation(n) for n in range(2)]


class LearnExpandPatches(LearnableOperation):
    @staticmethod
    def _make_learnable_operation(x_ratio: int, y_ratio: int):
        def learn(imgs, targets):
            pass
            return ExpandPixel.make_operation(x_ratio, y_ratio, EXPAND_PATCHES, expand_patches=expand_patches)
        return learn
