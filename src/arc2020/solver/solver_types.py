from enum import Enum


class SolverType(Enum):
    STUB = 'stub'
    GREEDY = 'greedy'

    def __str__(self) -> str:
        return self.value
