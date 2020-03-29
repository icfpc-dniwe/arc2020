import numpy as np
from numba import njit
from enum import Enum
from ..task import Task


class OutputSizeType(Enum):
    OTHER = 0,
    SAME = 1,
    FIXED = 2


def output_size_type(task: Task) -> OutputSizeType:
    all_inputs = [elem[0] for elem in task.train]
    all_outputs = [elem[1] for elem in task.train]
    if np.all([cur_input.shape == cur_output.shape for cur_input, cur_output in zip(all_inputs, all_outputs)]):
        return OutputSizeType.SAME
    if np.all([cur_output.shape == all_outputs[0].shape for cur_output in all_outputs[1:]]):
        return OutputSizeType.FIXED
    return OutputSizeType.OTHER
