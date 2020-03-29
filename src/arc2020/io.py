from pathlib import Path
import numpy as np
from .task import Task

from .mytypes import Result
from typing import NoReturn, Dict, List, Tuple


AllResults = Dict[str, List[Result]]


def read_all_tasks(task_directory: Path) -> Dict[str, Task]:
    tasks = dict()
    for cur_path in task_directory.iterdir():
        if cur_path.is_file() and cur_path.suffix == '.json':
            task_name = str(cur_path.stem)
            tasks[task_name] = read_task(cur_path)
    return tasks


def read_task(task_path: Path) -> Task:
    return Task(task_path)


def flattener(pred: np.ndarray) -> str:
    rows = []
    for row_pred in pred:
        rows.append(''.join(str(elem) for elem in row_pred))
    str_pred = f"|{'|'.join(rows)}|"
    return str_pred


def write_result(result: Result) -> str:
    return flattener(result)


def write_submission(all_results: Tuple[AllResults, AllResults, AllResults], output_path: Path) -> NoReturn:
    with open(str(output_path), 'w') as f:
        print('output_id,output', file=f)
        for task_name, cur_results in all_results[0].items():
            # print(f"{task_name},{' '.join(write_result(result) for result in cur_results)}", file=f)
            for output_idx, first_result in enumerate(cur_results):
                output_name = f'{task_name}_{output_idx}'
                other_results = (res[task_name][output_idx] for res in all_results[1:])
                print(f"{output_name},{' '.join([write_result(elem) for elem in [first_result, *other_results]])}",
                      file=f)
