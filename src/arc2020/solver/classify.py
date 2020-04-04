import numpy as np
from numba import njit
from enum import Enum
import enum
from ..task import Task


class OutputSizeType(Enum):
    OTHER = enum.auto()
    FIXED = enum.auto()
    SAME = enum.auto()
    SQUARE_SAME = enum.auto()


def output_size_type(task: Task) -> OutputSizeType:
    if np.all([cur_input.shape == cur_output.shape for cur_input, cur_output in task.train]):
        if np.all([np.max(cur_input.shape) == np.min(cur_input.shape) for cur_input, cur_output in task.train]):
            return OutputSizeType.SQUARE_SAME
        else:
            return OutputSizeType.SAME
    if np.all([cur_output.shape == task.train[0][1].shape for _, cur_output in task.train[1:]]):
        return OutputSizeType.FIXED
    return OutputSizeType.OTHER
