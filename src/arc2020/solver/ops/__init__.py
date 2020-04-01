from functools import partial
from ..classify import OutputSizeType
from . import atomic, learnable

operations = \
    [atomic.Rotate.make_operation(num_rotations=i) for i in range(1, 4)] + \
    [atomic.Transpose.make_operation()] + \
    [atomic.Flip.make_operation(flip_axis=i) for i in range(0, 2)]

learnable_operations = \
    [learnable.ColorMap.make_learnable_operation()]
     #learnable.Patches.make_learnable_operation()]

def filter_suitable(task_type, ops):
    return list(filter(lambda op: task_type in op.supported_outputs, ops))

def suitable_operations(task_type):
    return filter_suitable(task_type, operations)

def suitable_learnable_operations(task_type):
    return filter_suitable(task_type, learnable_operations)
