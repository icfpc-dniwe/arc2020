import numpy as np
from .ops.learnable import histogram
from .solver import Solver


class ColorSolver(Solver):

    def __init__(self,
                 next_solver: Solver,
                 additional_weights: bool = True):
        super().__init__()
        self.next_solver = next_solver
        self.additional_weights = additional_weights

    def solve(self):
        cur_pairs = self.task.train
        cur_imgs = [img for img, gt in cur_pairs]
        gt_imgs = [gt for img, gt in cur_pairs]
        test_imgs = [img for img, _ in self.task.test]
        transform = False
        if histogram.max_num_colors(cur_imgs) >= histogram.max_num_colors(test_imgs):
            transform = True
            transformer = histogram.LearnableCororMatching.learn(cur_imgs, gt_imgs, self.additional_weights)
            cur_pairs = [transformer.transform(img, gt) for img, gt in cur_pairs]
            self.task.train = cur_pairs
        ops = self.next_solver(self.task)
        if transform:
            ops = [transformer.forward] + ops + [transformer.backward]
        return ops
