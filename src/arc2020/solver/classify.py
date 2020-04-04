import numpy as np
from numba import njit
from enum import Enum
from ..task import Task


class OutputSizeType(Enum):
    OTHER = 0,
    SAME = 1,
    FIXED = 2


def output_size_type(task: Task) -> OutputSizeType:
    if np.all([cur_input.shape == cur_output.shape for cur_input, cur_output in task.train]):
        return OutputSizeType.SAME
    if np.all([cur_output.shape == task.train[0][1].shape for _, cur_output in task.train[1:]]):
        return OutputSizeType.FIXED
    return OutputSizeType.OTHER
