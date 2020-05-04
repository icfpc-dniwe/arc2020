from src.arc2020.solver import solver
from tqdm import tqdm
from .solver.solver import Solver
from .solver.utils import apply_operations
from .task import Task
from .solver.cnn import AutoEncoderTrain

from .mytypes import Result
from typing import Dict, List, Type


def pretrain(all_tasks: Dict[str, Task], solver_type: Type[Solver], **solver_kwargs) -> Dict[str, List[Result]]:
    results = dict()
    tasks = list(all_tasks.values())
    operations = AutoEncoderTrain.pretain(tasks)
    for task_name, cur_task in tqdm(all_tasks.items()):
        # operations = solver_fn(cur_task)
        # print(list(map(lambda x: x.name, operations)))
        try:
            results[task_name] = apply_operations(cur_task, operations)
        except ValueError:
            print('Smth wrong with operations', operations)
            results[task_name] = apply_operations(cur_task, [])
    return results, operations
