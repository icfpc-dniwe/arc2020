import numpy as np
from scipy import optimize
from ...classify import OutputSizeType
from ....mytypes import ImgMatrix
from ..operation import LearnableOperation
from typing import Sequence, Tuple, Dict


def calc_color_hist(img_matrix):
    unique_colors, counts = np.unique(img_matrix, return_counts=True)
    img_size = np.prod(img_matrix.shape)
    normalized_counts = counts / img_size
    return unique_colors, normalized_counts


def calc_color_map(img_matrix, target_hist, additional_weights: bool = False):
    unique_colors, normalized_counts = calc_color_hist(img_matrix)
    cost_matrix = np.zeros((len(unique_colors), 10), dtype=np.float32) + 10
    for cur_color, cur_norm_count in enumerate(normalized_counts):
        for target_color, target_norm_count in target_hist.items():
            metric = np.abs(target_norm_count - cur_norm_count)
            if additional_weights and (target_color == 0 or cur_color == 0) and target_color != cur_color:
                metric += 0.3
            cost_matrix[cur_color, target_color] = metric
    inp_color, tar_color = optimize.linear_sum_assignment(cost_matrix)
    residual = cost_matrix[inp_color, tar_color].sum()
    color_map = dict(zip(unique_colors[inp_color], tar_color))
    return color_map, residual


def inverse_color_map(color_map):
    return {v: k for k, v in color_map.items()}


def apply_color_map(img_matrix, color_map):
    new_matrix = np.array(img_matrix)
    for inp_color, tar_color in color_map.items():
        new_matrix[img_matrix == inp_color] = tar_color
    return new_matrix


def get_target_hist(imgs: Sequence[ImgMatrix]) -> Dict[int, float]:
    img_idx = 0
    num_colors = 0
    for cur_idx, cur_img in enumerate(imgs):
        unique_colors = np.unique(cur_img)
        if len(unique_colors) > num_colors:
            num_colors = len(unique_colors)
            img_idx = cur_idx
    return dict(zip(*calc_color_hist(imgs[img_idx])))


def has_same_colors(imgs: Sequence[ImgMatrix]) -> bool:
    pass


def max_num_colors(imgs: Sequence[ImgMatrix]) -> int:
    return np.max([len(np.unique(cur_img)) for cur_img in imgs])


class ColorMatching:
    supported_outputs = [e for e in OutputSizeType]

    def __init__(self, target_hist: Dict[int, float], additional_weights: bool = False):
        self.target_hist = target_hist
        self.inv_color_map = None
        self.additional_weights = additional_weights

    def transform(self, img: ImgMatrix, target: ImgMatrix) -> Tuple[ImgMatrix, ImgMatrix]:
        cur_color_map, residual = calc_color_map(img, self.target_hist, self.additional_weights)
        img = apply_color_map(img, cur_color_map)
        target = apply_color_map(target, cur_color_map)
        return img, target

    def forward(self, img: ImgMatrix) -> ImgMatrix:
        cur_color_map, residual = calc_color_map(img, self.target_hist, self.additional_weights)
        img = apply_color_map(img, cur_color_map)
        self.inv_color_map = inverse_color_map(cur_color_map)
        return img

    def backward(self, img: ImgMatrix) -> ImgMatrix:
        return apply_color_map(img, self.inv_color_map)


class LearnableCororMatching:
    @staticmethod
    def learn(imgs, targets, additional_weights: bool = False) -> ColorMatching:
        target_hist = get_target_hist(imgs)
        return ColorMatching(target_hist, additional_weights)
