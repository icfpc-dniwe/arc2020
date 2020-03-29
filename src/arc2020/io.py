from pathlib import Path
import numpy as np
from .task import Task

from .mytypes import Result
from typing import NoReturn


def read_task(task_path: Path) -> Task:
    return Task(task_path)


def write_result(result: Result) -> NoReturn:
    pass
