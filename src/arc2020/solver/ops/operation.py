from typing import Callable, List
from ...mytypes import ImgMatrix, ImgPair, Operation
from ..classify import OutputSizeType


class InvalidOperationError(Exception):
    def __init__(self):
        super().__init__("Invalid operation")


class Operation:
    supported_outputs = [e for e in OutputSizeType]

    @classmethod
    def make_operation(cls, *args, **kwargs) -> Operation:
        f = cls._make_operation(*args, **kwargs)
        f.supported_outputs = cls.supported_outputs
        return f

    @staticmethod
    def _make_operation():
        raise InvalidOperationError()


class LearnableOperation:
    supported_outputs = [e for e in OutputSizeType]

    @classmethod
    def make_learnable_operation(cls, *args, **kwargs) -> Callable[[List[ImgMatrix], List[ImgMatrix]], Operation]:
        f = cls._make_learnable_operation(*args, **kwargs)
        f.supported_outputs = cls.supported_outputs
        return f

    @staticmethod
    def _make_learnable_operation():
        raise InvalidOperationError()
