import numpy as np
from functools import partial
from .classify import OutputSizeType, output_size_type
from . import stub
from .ops import suitable_operations, suitable_learnable_operations
from .score import proximity_metric
from .utils import apply_operations
from ..task import Task
from ..mytypes import Result, ImgMatrix, Operation
from typing import List, Tuple, Iterable


def get_step_scores(cur_img: ImgMatrix, gt_img: ImgMatrix, operations: Iterable[Operation]) -> List[int]:
    all_metrics = [proximity_metric(cur_op(cur_img), gt_img) for cur_op in operations]
    return all_metrics


def solve(task: Task, max_depth: int = 2) -> List[Result]:
    task_type = output_size_type(task)
    cur_imgs = [cur_pair[0] for cur_pair in task.train]
    cur_gt_imgs = [cur_pair[1] for cur_pair in task.train]
    cur_metrics = np.array([proximity_metric(cur_pair[0], cur_pair[1]) for cur_pair in task.train])
    possible_operations = suitable_operations(task_type)
    learnable_operations = suitable_learnable_operations(task_type)
    if len(possible_operations) + len(learnable_operations) == 0:
        return stub.solve(task)
    operations = []
    new_metrics = np.empty((len(cur_imgs), len(possible_operations) + len(learnable_operations)), dtype=np.int32)
    while len(operations) < max_depth and np.any(cur_metrics > 0):
        cur_pairs = list(zip(cur_imgs, cur_gt_imgs))
        learned_ops = [learn(cur_pairs) for learn in learnable_operations]
        all_operations = possible_operations + learned_ops
        for img_idx, (img, gt_img) in enumerate(zip(cur_imgs, cur_gt_imgs)):
            # getting all metrics for current train pair
            new_metrics[img_idx, :] = get_step_scores(img, gt_img, all_operations)
        # we want the operation that minimize the distance between all training images
        operation_scores = np.sum(new_metrics, axis=0)
        best_op_idx = int(np.argmin(operation_scores))
        cur_imgs = [all_operations[best_op_idx](img) for img in cur_imgs]
        operations.append(all_operations[best_op_idx])
        cur_metrics[:] = new_metrics[:, best_op_idx]
    return apply_operations(task, operations)
