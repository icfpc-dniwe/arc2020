from . import solver
from .solver.solver import Solver
from .solver.utils import apply_operations
from .task import Task

from .mytypes import Result
from typing import Dict, List, Type


def solve(all_tasks: Dict[str, Task], solver_type: Type[Solver], **solver_kwargs) -> Dict[str, List[Result]]:
    solver_fn = solver_type(**solver_kwargs)
    results = dict()
    for task_name, cur_task in all_tasks.items():
        operations = solver_fn(cur_task)
        print(list(map(lambda x: x.name, operations)))
        results[task_name] = apply_operations(cur_task, operations)
    return results
