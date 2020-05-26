#!/usr/bin/env python3

from pathlib import Path
from matplotlib import pyplot as plt
from arc2020 import io, visualization, solve, solver, main, pretrain


if __name__ == '__main__':
    # main()
    my_solver = solver.size.SizeSolver
    # my_solver = solver.gbt.GBTSolver
    data_path = Path('../data/training')
    tasks = [
        # '54d9e175.json',
        # 'c0f76784.json',
        # '39e1d7f9.json',
        # 'e509e548.json',
        # '05269061.json',
        # '0ca9ddb6.json',
        'e73095fd.json',
        # 'f15e1fac.json'
    ]
    # tasks = [
    #     # '0e671a1a.json',
    #     '1da012fc.json'
    # ]
    cur_solver = my_solver(next_solver=solver.cnn.CNNSolver(), resize_max=True)
    for cur_task in tasks:
        print(cur_task)
        task = io.read_task(data_path / cur_task)
        visualization.plot_task(task)
        ops = cur_solver(task)
        print(ops)
        result_matricies = solver.utils.apply_operations(task, ops)
        num_test = len(result_matricies)
        fig, axs = plt.subplots(2, num_test, figsize=(3 * num_test, 3 * 2), squeeze=False)
        for i in range(num_test):
            visualization.plot_one(axs[0, i], task.test[i][0], True, False)
            visualization.plot_one(axs[1, i], result_matricies[i], False, False)
        plt.tight_layout()
        plt.show()
    quit()
    # task_name = '0e206a2e.json'
    # for cur_task in data_path.iterdir():
    #     print(cur_task.name)
    #     task = io.read_task(cur_task)
    #     visualization.plot_task(task)
    # print(io.write_result(task.test[0][0]))
    all_tasks = io.read_all_tasks(data_path)
    cur_tasks = all_tasks
    # first_keys = list(all_tasks.keys())[:300]
    # last_keys = list(all_tasks.keys())[300:]
    # cur_tasks = {key: all_tasks[key] for key in first_keys}
    # results = solve.solve(all_tasks, my_solver, use_hist=True, use_aug=True, additional_weights=False)
    results = solve.solve(all_tasks, solver.color.ColorSolver,
                          next_solver=my_solver(use_aug=True), additional_weights=True)
    # results, ops = pretrain.pretrain(cur_tasks, None)
    # io.write_submission((results, results, results), '../data/debug_submission.csv')
    validation_num = len(results)
    positive_num = 0
    for task_name, task in cur_tasks.items():
        res = solver.utils.validate_result(task, results[task_name])
        positive_num += int(res)
        if res:
            print(task_name)
    print(positive_num, '/', validation_num)
    data_path = Path('../data/evaluation')
    all_tasks = io.read_all_tasks(data_path)
    results = solve.solve(all_tasks, my_solver)
    validation_num = len(results)
    positive_num = 0
    for task_name, task in all_tasks.items():
        res = solver.utils.validate_result(task, results[task_name])
        positive_num += int(res)
        if res:
            print(task_name)
    print(positive_num, '/', validation_num)
    # task_name = '0e206a2e.json'
    # task_name = '39e1d7f9.json'
    # task = io.read_task(data_path / task_name)
    # task = all_tasks[last_keys[0]]
    # result_matricies = solver.utils.apply_operations(task, ops)
    # num_test = len(result_matricies)
    # fig, axs = plt.subplots(2, num_test, figsize=(3 * num_test, 3 * 2), squeeze=False)
    # for i in range(num_test):
    #     visualization.plot_one(axs[0, i], task.test[i][0], True, False)
    #     visualization.plot_one(axs[1, i], result_matricies[i], False, False)
    # plt.tight_layout()
    # plt.show()
