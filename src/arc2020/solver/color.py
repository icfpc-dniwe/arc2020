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

    def transform_task(self, task):
        cur_pairs = task.train
        cur_imgs = [img for img, gt in cur_pairs]
        gt_imgs = [gt for img, gt in cur_pairs]
        test_imgs = [img for img, _ in task.test]
        transformer = None
        if histogram.max_num_colors(cur_imgs) >= histogram.max_num_colors(test_imgs):
            transformer = histogram.LearnableCororMatching.learn(cur_imgs, gt_imgs, self.additional_weights)
            cur_pairs = [transformer.transform(img, gt) for img, gt in cur_pairs]
            task.train = cur_pairs
        return transformer

    def solve(self):
        ops = self.next_solver(self.task)
        transformer = self.transform_task(self.task)
        if transformer is not None:
            ops = [transformer.forward] + ops + [transformer.backward]
        return ops

    def pretrain(self, all_tasks):
        for cur_task in all_tasks.values():
            self.transform_task(cur_task)
        return self.next_solver.pretrain(all_tasks)
