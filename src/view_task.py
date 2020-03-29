from pathlib import Path
from matplotlib import pyplot as plt
from arc2020 import io, visualization, solve, solver


if __name__ == '__main__':
    data_path = Path('../data/training')
    task_name = '1e0a9b12.json'
    task = io.read_task(data_path / task_name)
    # visualization.plot_task(task)
    print(io.write_result(task.test[0][0]))
    all_tasks = io.read_all_tasks(data_path)
    results = solve.solve(all_tasks, solver.SolverType.STUB)
    io.write_submission((results, results, results), '../data/debug_submission.csv')
