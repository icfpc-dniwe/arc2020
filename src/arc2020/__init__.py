import os
from pathlib import Path
import torch.backends.cudnn as cudnn
from . import io, solve, solver

def main():
    cudnn.benchmark = True
    all_tasks = io.read_all_tasks(Path('/kaggle/input/abstraction-and-reasoning-challenge/test'))
    # all_tasks = io.read_all_tasks(Path.home() / 'python/arc2020/data/test')
    results1 = solve.solve(all_tasks, solver.color.ColorSolver,
                           next_solver=solver.cnn.CNNSolver(weights_learning=True), additional_weights=False)
    results2 = results1
    results3 = results1
    io.write_submission((results1, results2, results3), 'submission.csv')
