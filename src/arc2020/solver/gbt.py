import numpy as np
from .ops.learnable import gbt
from .solver import Solver


class GBTSolver(Solver):

    def __init__(self,
                 use_hist: bool = True,
                 use_aug: bool = False,
                 residual_learning: bool = True):
        super().__init__()
        self.learner = gbt.LearnGBT.make_learnable_operation(use_aug, residual_learning, use_hist)
        self.operations = []
        self.learnable_operations = [self.learner]

    def solve(self):
        cur_imgs = [img for img, gt in self.task.train]
        gt_imgs = [gt for img, gt in self.task.train]
        ops = [self.learner(cur_imgs, gt_imgs)]
        return ops
