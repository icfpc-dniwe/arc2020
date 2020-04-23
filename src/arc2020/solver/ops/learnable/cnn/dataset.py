import torch
import torch.utils.data as data
import numpy as np
import random
from numba import njit
from itertools import permutations
from .....mytypes import ImgMatrix
from typing import List, Tuple, Sequence, Any, Optional


@njit
def convert_matrix(matrix: ImgMatrix, num_classes: int = 10) -> np.ndarray:
    new_m = np.zeros((*matrix.shape, num_classes), dtype=np.uint8)
    for cur_row in range(matrix.shape[0]):
        for cur_col in range(matrix.shape[1]):
            new_m[cur_row, cur_col, matrix[cur_row, cur_col]] = 1
    return new_m


class RandomPerm:

    def __init__(self, data: Sequence[Any], sample_size: int):
        self.data = data
        self.sample_size = sample_size

    def __len__(self) -> int:
        return self.sample_size

    def __getitem__(self, idx: int) -> Any:
        return random.choice(self.data)


# @njit
def add_palette(img: np.ndarray) -> np.ndarray:
    added_classes = 10
    height, width = img.shape[:2]
    # new_img = np.empty((height + 2 * added_classes, width + 2 * added_classes), dtype=img.dtype)
    # new_img[added_classes:height+added_classes, added_classes:width+added_classes] = img
    new_img = img
    for color_idx in range(added_classes):
        new_img = np.pad(new_img, 1, 'constant', constant_values=color_idx)
    return new_img


def prep_data(img: ImgMatrix,
              target: ImgMatrix,
              perm: Optional[Sequence[int]] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
    cur_img = convert_matrix(img)
    cur_target = convert_matrix(target)
    if perm is not None:
        cur_img = cur_img[:, :, perm]
        cur_target = cur_target[:, :, perm]
    cur_img = cur_img.transpose((2, 0, 1)).astype(np.float32)
    cur_target = cur_target.transpose((2, 0, 1)).astype(np.float32)
    return cur_img, cur_target


def aug(img: ImgMatrix, target: ImgMatrix) -> Tuple[ImgMatrix, ImgMatrix]:
    if random.random() > 0.5:
        axis = random.randrange(0, 2)
        img = np.flip(img, axis=axis)
        target = np.flip(target, axis=axis)
    if random.random() > 0.5:
        num_rot = random.randrange(0, 4)
        img = np.rot90(img, k=num_rot)
        target = np.rot90(target, k=num_rot)
    return img, target


class PermutableData(data.Dataset):

    def __init__(self, imgs: List[ImgMatrix], targets: List[ImgMatrix], sample: bool = False):
        super(PermutableData, self).__init__()
        self.data = []  # type: List[Tuple[np.ndarray, np.ndarray]]
        all_permutations = list(permutations(range(10)))
        self.data = list(zip(imgs, targets))
        if sample:
            self.permutations = RandomPerm(all_permutations, 3 * 10 ** 5)
        else:
            self.permutations = all_permutations
        # for cur_img, cur_target in zip(imgs, targets):
        #     diff = cur_target != cur_img
        #     cur_target = cur_target * diff + 10 * (1 - diff)
        #     cur_img = convert_matrix(cur_img)
        #     cur_target = convert_matrix(cur_target, 11)
        #     if sample:
        #         cur_permutations = random.sample(all_permutations, 3 * 10**5)
        #     else:
        #         cur_permutations = all_permutations
        #     self.data.extend((cur_img[:, :, cur_perm], np.argmax(cur_target[:, :, cur_perm], axis=-1))
        #                      for cur_perm in cur_permutations)

    def __len__(self) -> int:
        return len(self.permutations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # img, label = self.data[idx]
        # img = torch.from_numpy(img.transpose((2, 0, 1)).astype(np.float32))
        # label = torch.from_numpy(label)
        # cur_perm = self.permutations[idx]
        cur_perm = list(range(10))
        sample = random.choice(self.data)
        img, label = sample
        img, label = aug(img, label)
        img = add_palette(img)
        label = add_palette(img)
        img, label = prep_data(img, label, cur_perm)
        img = torch.from_numpy(img)
        label = torch.from_numpy(np.argmax(label, axis=0))
        return img, label


class TaskData(data.Dataset):

    def __init__(self, imgs: List[ImgMatrix], targets: List[ImgMatrix], sample: bool = False):
        super(TaskData, self).__init__()
        all_permutations = list(permutations(range(10)))
        # self.imgs = imgs
        # self.targets = targets
        self.data = [(add_palette(img), add_palette(target)) for img, target in zip(imgs, targets)]
        # self.data = list(zip(imgs, targets))
        if sample:
            self.permutations = RandomPerm(all_permutations, 3 * 10**4)
            # self.permutations = RandomPerm(all_permutations, 3000)
        else:
            self.permutations = all_permutations

    def __len__(self) -> int:
        return len(self.permutations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # cur_perm = self.permutations[idx]
        cur_perm = list(range(10))
        num_sample = random.randint(2, len(self.data))
        sample = random.sample(self.data, num_sample)
        sample = [aug(img, target) for img, target in sample]
        sample = [prep_data(img, target, cur_perm) for img, target in sample]
        pred_sample = [(torch.from_numpy(inp), torch.from_numpy(out)) for inp, out in sample[1:]]
        # return pred_sample, torch.from_numpy(sample[0][0]), torch.from_numpy(sample[0][1])
        return (pred_sample[0][0], pred_sample[0][1],
                torch.from_numpy(sample[0][0]), torch.from_numpy(np.argmax(sample[0][1], axis=0)))
