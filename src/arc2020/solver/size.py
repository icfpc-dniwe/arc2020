import numpy as np
from .ops.learnable import gbt
from .solver import Solver


class SizeSolver(Solver):

    def __init__(self,
                 next_solver: Solver):
        super().__init__()
        self.next_solver = next_solver

    def solve(self):
        cur_imgs = [img for img, gt in self.task.train]
        gt_imgs = [gt for img, gt in self.task.train]
        ops = []
        return ops
