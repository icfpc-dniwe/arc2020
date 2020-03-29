import numpy as np
from ..task import Task
from ..mytypes import Result, ImgMatrix, Operation
from typing import List, Tuple, Iterable


def apply_operations(task: Task, operations: Iterable[Operation]) -> List[Result]:
    results = []
    for cur_test_img, _ in task.test:
        for cur_op in operations:
            cur_test_img = cur_op(cur_test_img)
        results.append(cur_test_img)
    return results


def validate_result(task: Task, results: List[Result]) -> bool:
    for cur_pair, cur_result in zip(task.test, results):
        if np.any(cur_pair[1] != cur_result):
            return False
    return True
