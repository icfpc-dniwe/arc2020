import os
from pathlib import Path
from . import io, solve, solver

def main():
    all_tasks = io.read_all_tasks(Path('/kaggle/input/abstraction-and-reasoning-challenge/test'))
    results = solve.solve(all_tasks, solver.SolverType.STUB)
    io.write_submission((results, results, results), 'submission.csv')
