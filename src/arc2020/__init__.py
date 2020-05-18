import os
from pathlib import Path
import torch.backends.cudnn as cudnn
from . import io, solve, solver

def main():
    cudnn.benchmark = True
    all_tasks = io.read_all_tasks(Path('/kaggle/input/abstraction-and-reasoning-challenge/test'))
    # all_tasks = io.read_all_tasks(Path.home() / 'python/arc2020/data/test')
    results1 = solve.solve(all_tasks, solver.gbt.GBTSolver, use_hist=True, use_aug=False)
    results2 = results1  # solve.solve(all_tasks, solver.dfs.DFSSolver, max_depth=3)
    results3 = results2
    io.write_submission((results1, results2, results3), 'submission.csv')
