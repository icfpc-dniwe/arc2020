from . import solver
from .solver import SolverType
from .task import Task

from .mytypes import Result
from typing import Dict, List


def solve(all_tasks: Dict[str, Task], solver_type: SolverType, **solver_kwargs) -> Dict[str, List[Result]]:
    solver_fn = getattr(solver, str(solver_type)).solve
    results = dict()
    for task_name, cur_task in all_tasks.items():
        results[task_name] = solver_fn(cur_task, **solver_kwargs)
    return results
