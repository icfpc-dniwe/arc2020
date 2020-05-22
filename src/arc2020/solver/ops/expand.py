import numpy as np
from ...mytypes import ImgMatrix
from .operation import Operation
from ..classify import OutputSizeType
from typing import Optional, Dict, Union, Any

EXPAND_BG = 0
EXPAND_COLOR = 1
EXPAND_IMG = 2
EXPAND_PATCHES = 3


class ExpandPixel(Operation):
    @staticmethod
    def _make_operation(x_ratio: int,
                        y_ratio: int,
                        expand_strategy: int = EXPAND_BG,
                        img_bg: int = 0,
                        expand_patches: Optional[Dict[int, Union[np.ndarray, str, Any]]] = None):

        def expand(img: ImgMatrix) -> ImgMatrix:
            new_img = np.zeros((img.shape[0] * x_ratio, img.shape[1] * y_ratio), dtype=img.dtype)
            for cur_row in img.shape[0]:
                sub_row_min = cur_row * x_ratio
                sub_row_max = sub_row_min + x_ratio
                for cur_col in img.shape[1]:
                    sub_col_min = cur_col * y_ratio
                    sub_col_max = sub_col_min + y_ratio
                    if expand_strategy == EXPAND_BG:
                        new_img[sub_row_min:sub_row_max, sub_col_min:sub_col_max] = img_bg
                    elif expand_strategy == EXPAND_COLOR:
                        new_img[sub_row_min:sub_row_max, sub_col_min:sub_col_max] = img[cur_row, cur_col]
                    elif expand_strategy == EXPAND_IMG:
                        if img[cur_row, cur_col] == img_bg:
                            new_img[sub_row_min:sub_row_max, sub_col_min:sub_col_max] = img_bg
                        else:
                            new_img[sub_row_min:sub_row_max, sub_col_min:sub_col_max] = img
                    elif expand_strategy == EXPAND_PATCHES:
                        if expand_patches is not None:
                            cur_patch = expand_patches[img[cur_row, cur_col]]
                            if isinstance(cur_patch, np.ndarray):
                                new_img[sub_row_min:sub_row_max, sub_col_min:sub_col_max] = cur_patch
                            elif isinstance(cur_patch, str):
                                if cur_patch == 'img':
                                    new_img[sub_row_min:sub_row_max, sub_col_min:sub_col_max] = img
                            else:
                                new_img[sub_row_min:sub_row_max, sub_col_min:sub_col_max] = cur_patch(img)
            return ImgMatrix(new_img)

        return expand


class ExpandCopy(Operation):
    @staticmethod
    def _make_operation(x_ratio: int, y_ratio: int):
        return lambda x: np.tile(x, (x_ratio, y_ratio))
