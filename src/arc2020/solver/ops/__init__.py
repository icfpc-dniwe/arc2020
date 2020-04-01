from functools import partial
from . import atomic, learnable

atomic_operations = \
    [partial(atomic.rotate, num_rotations=i) for i in range(1, 4)] + \
    [atomic.transpose] + \
    [partial(atomic.flip, flip_axis=i) for i in range(0, 2)]


learnable_operations = \
    [learnable.learn_color_map]


output_learnable = \
    [learnable.learn_fixed_output, learnable.learn_patches]
