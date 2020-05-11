from . import solver
from tqdm import tqdm
import pathos
from pathos.multiprocessing import Pool
from .solver.solver import Solver
from .solver.utils import apply_operations
from .task import Task

from .mytypes import Result, ImgMatrix
from typing import Dict, List, Type, Tuple


def solve(all_tasks: Dict[str, Task], solver_type: Type[Solver], **solver_kwargs) -> Dict[str, List[Result]]:
    solver_fn = solver_type(**solver_kwargs)
    results = dict()

    def worker(cur_item: Tuple[str, Task]) -> Tuple[str, List[ImgMatrix]]:
        operations = solver_fn(cur_item[1])
        return cur_item[0], apply_operations(cur_item[1], operations)

    with Pool(12) as p:
        results = dict(p.imap_unordered(worker, all_tasks.items(), chunksize=5))
    return results
