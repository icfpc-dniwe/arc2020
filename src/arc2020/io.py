from pathlib import Path
import numpy as np
from .task import Task

from .mytypes import Result
from typing import NoReturn, Dict


def read_task(task_path: Path) -> Task:
    return Task(task_path)


def flattener(pred: np.ndarray) -> str:
    rows = []
    for row_pred in pred:
        rows.append(f"|{'|'.join(str(elem) for elem in row_pred)}|")
    str_pred = '\n'.join(rows)
    return str_pred


def write_result(result: Result) -> str:
    return flattener(result)


def write_submission(results: Dict[str, Result], output_path: Path) -> NoReturn:
    pass
