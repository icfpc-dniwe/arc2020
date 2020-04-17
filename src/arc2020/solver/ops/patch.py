import numpy as np
from typing import Optional, Tuple


# class Patch:
#
#     def __init__(self,
#                  inp_array: np.ndarray,
#                  out_array: np.ndarray,
#                  mask: Optional[np.ndarray] = None,
#                  background: int = 0):
#         self.inp_array = inp_array
#         if mask is None:
#             mask = out_array != background
#         self.mask = mask
#         self.out_array = out_array * mask


Patch = Tuple[np.ndarray, np.ndarray, np.ndarray]


def create_patch(inp_array: np.ndarray,
                 out_array: np.ndarray,
                 mask: Optional[np.ndarray] = None,
                 background: int = 0
                 ) -> Patch:
    if mask is None:
        mask = out_array != background
    return inp_array, out_array * mask, mask
