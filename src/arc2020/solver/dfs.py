import numpy as np
from functools import partial
from itertools import chain
from . import stub
from .score import proximity_metric
from ..mytypes import Result, ImgMatrix, Operation
from .solver import Solver
from typing import List, Tuple, Iterable


class DFSSolver(Solver):
    def __init__(self, max_depth: int = 3):
        super().__init__()
        self.max_depth = max_depth

    def solve(self):
        gt_imgs = [gt for img, gt in self.task.train]
        possible_operations = self.possible_operations
        learnable_operations = self.learnable_operations

        def step(cur_imgs, depth):
            if depth == 0:
                return []
            else:
                learned_ops = [learn(cur_imgs, gt_imgs) for learn in learnable_operations]
                for op in chain(possible_operations, learned_ops):
                    new_cur_imgs = [op(img) for img in cur_imgs]
                    if all(map(np.array_equal, new_cur_imgs, gt_imgs)):
                        return [[op]]
                    else:
                        solutions = step(new_cur_imgs, depth - 1)
                        if len(solutions) > 0:
                            for ops in solutions:
                                ops.insert(0, op)
                            return solutions
                return []

        cur_imgs = [img for img, gt in self.task.train]
        solutions = step(cur_imgs, self.max_depth)
        if len(solutions) == 0:
            return []
        else:
            return solutions[0]
