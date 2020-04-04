#!/usr/bin/env python3

from pathlib import Path
from matplotlib import pyplot as plt
from arc2020 import io, visualization, solve, solver


if __name__ == '__main__':
    my_solver = solver.greedy.GreedySolver
    data_path = Path('../data/training')
    task_name = '0e206a2e.json'
    task = io.read_task(data_path / task_name)
    # visualization.plot_task(task)
    print(io.write_result(task.test[0][0]))
    all_tasks = io.read_all_tasks(data_path)
    results = solve.solve(all_tasks, my_solver)
    io.write_submission((results, results, results), '../data/debug_submission.csv')
    validation_num = len(results)
    positive_num = 0
    for task_name, task in all_tasks.items():
        res = solver.utils.validate_result(task, results[task_name])
        positive_num += int(res)
    print(positive_num, '/', validation_num)
    data_path = Path('../data/evaluation')
    all_tasks = io.read_all_tasks(data_path)
    results = solve.solve(all_tasks, my_solver)
    validation_num = len(results)
    positive_num = 0
    for task_name, task in all_tasks.items():
        res = solver.utils.validate_result(task, results[task_name])
        positive_num += int(res)
    print(positive_num, '/', validation_num)
