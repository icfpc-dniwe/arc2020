import os
from pathlib import Path
from . import io, solve, solver

def main():
    all_tasks = io.read_all_tasks(Path('/kaggle/input/abstraction-and-reasoning-challenge/test'))
    results2 = solve.solve(all_tasks, solver.SolverType.GREEDY, max_depth=2)
    results3 = solve.solve(all_tasks, solver.SolverType.GREEDY, max_depth=3)
    results4 = solve.solve(all_tasks, solver.SolverType.GREEDY, max_depth=4)
    io.write_submission((results2, results3, results4), 'submission.csv')
