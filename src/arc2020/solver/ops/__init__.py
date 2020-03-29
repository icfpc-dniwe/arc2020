from functools import partial
from . import atomic

all_operations = [partial(atomic.rotate, num_rotations=i) for i in range(1, 4)] + \
    [atomic.transpose] + \
    [partial(atomic.flip, flip_axis=i) for i in range(0, 2)]
