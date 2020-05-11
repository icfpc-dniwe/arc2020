from pathlib import Path
import numpy as np
import pandas as pd
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


def load_tags(tasks: Dict[str, Task], tags_csv: Path) -> None:
    with open(str(tags_csv), 'r') as f:
        header = f.readline().strip().split(',')
    #     for line in f:
    #         parts = line.strip().split(',')
    #         task_name = parts[1][:parts[1].find('.')]
    #         if task_name in tasks:
    #             tasks[task_name].tags = list(zip(header[3:], map(int, parts[3:])))
    tag_names = header[3:]
    frame = pd.read_csv(tags_csv)
    for idx, cur_row in frame.iterrows():
        task_name = cur_row['task_name']
        task_name = task_name[:task_name.find('.')]
        if task_name in tasks:
            tasks[task_name].tags = [(cur_tag, cur_row[cur_tag]) for cur_tag in tag_names]


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
