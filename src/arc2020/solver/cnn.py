import numpy as np
from .ops.learnable import cnn
from .score import proximity_metric
from ..mytypes import Result, ImgMatrix, Operation
from .solver import Solver


class CNNSolver(Solver):

    def __init__(self, weights_learning: bool = True):
        super().__init__()
        self.weights_learning = weights_learning

    def solve(self):
        cur_imgs = [img for img, gt in self.task.train]
        gt_imgs = [gt for img, gt in self.task.train]
        ops = [cnn.LearnCNN.make_learnable_operation(self.weights_learning)(cur_imgs, gt_imgs)]
        return ops


class AutoEncoderTrain(Solver):

    def __init__(self):
        super().__init__()

    def solve(self):
        all_imgs = [img for img, gt in self.task.train] + [gt for img, gt in self.task.train] + \
                   [img for img, gt in self.task.test]
        ops = [cnn.LearnEncoder.make_learnable_operation()(all_imgs, None)]
        return ops

    @staticmethod
    def pretain(all_tasks):
        all_imgs = []
        for cur_task in all_tasks:
            all_imgs += [img for img, gt in cur_task.train] + [gt for img, gt in cur_task.train] + \
                        [img for img, gt in cur_task.test]
        ops = [cnn.LearnEncoder.make_learnable_operation()(all_imgs, None)]
        return ops
