import numpy as np
from functools import partial
from itertools import chain
from . import stub
from .score import proximity_metric
from ..mytypes import Result, ImgMatrix, Operation
from .solver import Solver
from typing import List, Tuple, Iterable


def get_step_scores(cur_img: ImgMatrix, gt_img: ImgMatrix, operations: Iterable[Operation]) -> List[int]:
    all_metrics = [proximity_metric(cur_op(cur_img), gt_img) for cur_op in operations]
    return all_metrics


class GreedySolver(Solver):
    def __init__(self, max_depth: int = 2):
        super().__init__()
        self.max_depth = max_depth

    def solve(self):
        cur_imgs = [img for img, gt in self.task.train]
        gt_imgs = [gt for img, gt in self.task.train]
        cur_metrics = np.array([proximity_metric(img, gt) for img, gt in self.task.train])

        operations = []
        max_depth = self.max_depth
        possible_operations = self.possible_operations
        learnable_operations = self.learnable_operations
        new_metrics = np.empty((len(cur_imgs), len(possible_operations) + len(learnable_operations)), dtype=np.int32)
        while len(operations) < max_depth and np.any(cur_metrics > 0):
            learned_ops = [learn(cur_imgs, gt_imgs) for learn in learnable_operations]
            all_operations = possible_operations + learned_ops
            for img_idx, (img, gt_img) in enumerate(zip(cur_imgs, gt_imgs)):
                # getting all metrics for current train pair
                new_metrics[img_idx, :] = get_step_scores(img, gt_img, all_operations)
            # we want the operation that minimize the distance between all training images
            operation_scores = np.sum(new_metrics, axis=0)
            best_op_idx = int(np.argmin(operation_scores))
            best_op = all_operations[best_op_idx]
            cur_imgs = [best_op(img) for img in cur_imgs]
            operations.append(best_op)
            cur_metrics[:] = new_metrics[:, best_op_idx]
        return operations
