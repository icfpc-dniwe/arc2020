import numpy as np
from .ops.learnable import gbt, histogram
from .solver import Solver


class GBTSolver(Solver):

    def __init__(self,
                 use_aug: bool = False,
                 residual_learning: bool = True):
        super().__init__()
        self.learner = gbt.LearnGBT.make_learnable_operation(use_aug, residual_learning)
        # self.use_hist = use_hist
        # self.additional_weights = additional_weights
        self.operations = []
        self.learnable_operations = [self.learner]

    def solve(self):
        cur_pairs = self.task.train
        cur_imgs = [img for img, gt in cur_pairs]
        gt_imgs = [gt for img, gt in cur_pairs]
        # if self.use_hist:
        #     transformer = histogram.LearnableCororMatching.learn(cur_imgs, gt_imgs, self.additional_weights)
        #     cur_pairs = [transformer.transform(img, gt) for img, gt in cur_pairs]
        #     cur_imgs = [img for img, gt in cur_pairs]
        #     gt_imgs = [gt for img, gt in cur_pairs]
        ops = [self.learner(cur_imgs, gt_imgs)]
        # if self.use_hist:
        #     ops = [transformer.forward] + ops + [transformer.backward]
        return ops
