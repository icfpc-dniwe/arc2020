from itertools import chain
from typing import Callable, List
from ...mytypes import ImgMatrix, ImgPair, Operation
from ..classify import OutputSizeType
from typing import Tuple, Callable


class InvalidOperationError(Exception):
    def __init__(self):
        super().__init__("Invalid operation")


def operation_name(cls, args, kwargs):
    str_args = map(repr, args)
    str_kwargs = map(lambda kv: f"{kv[0]}={repr(kv[1])}", kwargs.items())
    all_str = ", ".join(chain(str_args, str_kwargs))
    return f"{cls.__name__}({all_str})"


class Operation:
    supported_outputs = [e for e in OutputSizeType]

    @classmethod
    def make_operation(cls, *args, **kwargs) -> Operation:
        name = operation_name(cls, args, kwargs)
        f = cls._make_operation(*args, **kwargs)
        f.supported_outputs = cls.supported_outputs
        f.name = name
        return f

    @staticmethod
    def _make_operation():
        raise InvalidOperationError()


class LearnableOperation:
    supported_outputs = [e for e in OutputSizeType]

    @classmethod
    def make_learnable_operation(cls, *args, **kwargs) -> Callable[[List[ImgMatrix], List[ImgMatrix]], Operation]:
        name = operation_name(cls, args, kwargs)
        learnt_name = operation_name(cls, args, kwargs) + "()"
        supported_outputs = cls.supported_outputs

        learn_f = cls._make_learnable_operation(*args, **kwargs)
        def learn(*args, **kwargs):
            f = learn_f(*args, **kwargs)
            f.supported_outputs = supported_outputs
            f.name = learnt_name
            return f
        learn.supported_outputs = supported_outputs
        learn.name = name
        return learn

    @staticmethod
    def _make_learnable_operation():
        raise InvalidOperationError()


# class Transform:
#     supported_outputs = [e for e in OutputSizeType]
#
#     @staticmethod
#     def forward(img: ImgMatrix) -> Tuple[Operation, ImgMatrix]:
#         raise NotImplemented


# class LearnableTransform:
#     supported_outputs = [e for e in OutputSizeType]
#
#     @classmethod
#     def make_learnable_transform(cls, *args, **kwargs) -> Callable[[List[ImgMatrix], List[ImgMatrix]], Transform]:
#         name = operation_name(cls, args, kwargs)
#         learnt_name = operation_name(cls, args, kwargs) + "()"
#         supported_outputs = cls.supported_outputs
#
#         learn_f = cls._make_learnable_transform(*args, **kwargs)
#
#         def learn(*args, **kwargs):
#             f = learn_f(*args, **kwargs)
#             f.supported_outputs = supported_outputs
#             f.name = learnt_name
#             return f
#
#         learn.supported_outputs = supported_outputs
#         learn.name = name
#         return learn
#
#     @staticmethod
#     def _make_learnable_transform():
#         raise InvalidOperationError()
