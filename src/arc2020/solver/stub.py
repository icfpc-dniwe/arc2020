from ..task import Task
from ..mytypes import Result
from typing import List


def solve(task: Task) -> List[Result]:
    return [elem[0] for elem in task.test]
