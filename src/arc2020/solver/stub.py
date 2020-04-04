from ..task import Task
from ..mytypes import Result
from typing import List
from .solver import Solver


class Stub(Solver):
    def solve(self):
        return []
