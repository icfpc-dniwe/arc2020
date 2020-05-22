#!/usr/bin/env python3

from .ops import operations, learnable_operations
from .classify import output_size_type
from ..task import Task

from ..mytypes import Operation
from typing import List


def filter_suitable(task_type, ops):
    return list(filter(lambda op: task_type in op.supported_outputs, ops))


class Solver:

    def __init__(self):
        self.operations = operations
        self.all_learnable_operations = learnable_operations

    def __call__(self, task: Task):
        self.task = task
        self.task_type = output_size_type(task)
        self.possible_operations = filter_suitable(self.task_type, self.operations)
        self.learnable_operations = filter_suitable(self.task_type, self.all_learnable_operations)

        if len(self.possible_operations) + len(self.learnable_operations) == 0:
            return []
        return self.solve()

    def solve(self) -> List[Operation]:
        raise NotImplemented
