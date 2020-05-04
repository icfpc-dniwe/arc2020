from . import atomic, learnable

operations = \
    [atomic.Rotate.make_operation(num_rotations=i) for i in range(1, 4)] + \
    [atomic.Transpose.make_operation()] + \
    [atomic.Flip.make_operation(flip_axis=i) for i in range(0, 2)]

learnable_operations = \
    [learnable.ColorMap.make_learnable_operation(allow_trivial=i) for i in [False]] + \
    [learnable.Patches.make_learnable_operation(patch_size=i) for i in [3, 5, 7]]
