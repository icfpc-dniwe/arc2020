import os
from . import io, solve, solver

def main():
    all_tasks = io.read_all_tasks('/kaggle/input/test')
    results = solve.solve(all_tasks, solver.SolverType.STUB)
    io.write_submission((results, results, results), '/kaggle/working/submission.csv')
