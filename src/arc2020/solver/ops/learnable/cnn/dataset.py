import torch
import torch.utils.data as data
import numpy as np
import random
from numba import njit
from itertools import permutations
from .....mytypes import ImgMatrix
from .....common import reverse_tuple
from typing import List, Tuple, Sequence, Any, Optional


@njit
def convert_matrix(matrix: ImgMatrix, num_classes: int = 10) -> np.ndarray:
    new_m = np.zeros((*matrix.shape, num_classes), dtype=np.uint8)
    for cur_row in range(matrix.shape[0]):
        for cur_col in range(matrix.shape[1]):
            if matrix[cur_row, cur_col] < num_classes:
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


def config_palette(img_size: int):

    def add_palette(img: np.ndarray) -> np.ndarray:
        added_classes = 10
        # height, width = img.shape[:2]
        # new_img = np.empty((height + 2 * added_classes, width + 2 * added_classes), dtype=img.dtype)
        # new_img[added_classes:height+added_classes, added_classes:width+added_classes] = img
        # for color_idx in range(added_classes):
        #     img = np.pad(img, 1, 'constant', constant_values=color_idx)
        new_img = np.zeros((img_size, img_size), dtype=img.dtype) + 11
        h, w = img.shape[:2]
        new_img[:h, :w] = img
        return new_img

    return add_palette


def prep_img(img: ImgMatrix,
             img_shape: Tuple[int, int],
             perm: Optional[Sequence[int]] = None
             ) -> np.ndarray:
    cur_img = convert_matrix(img)
    if perm is not None:
        cur_img = cur_img[:, :, perm]
    img_shape_tensor = np.zeros((*cur_img.shape[:2], 2), dtype=np.float32) + \
                       np.array(img_shape, dtype=np.float32)[np.newaxis, np.newaxis] / 30.
    cur_img = np.concatenate((img_shape_tensor, cur_img.astype(np.float32)), axis=-1)
    cur_img = cur_img.transpose((2, 0, 1))
    return cur_img


def prep_data(img: ImgMatrix,
              target: ImgMatrix,
              img_shape: Tuple[int, int],
              target_shape: Tuple[int, int],
              perm: Optional[Sequence[int]] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
    return prep_img(img, img_shape, perm), prep_img(target, target_shape, perm)


def aug(img: ImgMatrix,
        target: ImgMatrix,
        img_shape: Optional[Tuple[int, int]] = None,
        target_shape: Optional[Tuple[int, int]] = None,
        noise_colors: Optional[Sequence[int]] = None
        ) -> Tuple[ImgMatrix, ImgMatrix, Tuple[int, int], Tuple[int, int]]:
    if random.random() < 0.5:
        axis = random.randrange(0, 2)
        img = np.flip(img, axis=axis)
        target = np.flip(target, axis=axis)
    if random.random() < 0.5:
        num_rot = random.randrange(0, 4)
        img = np.rot90(img, k=num_rot)
        target = np.rot90(target, k=num_rot)
        if num_rot % 2 == 1:
            img_shape = reverse_tuple(img_shape)
            target_shape = reverse_tuple(target_shape)
    # if noise_colors is not None and len(noise_colors) > 0 and random.random() < 0.2:
    #     num_pixels = random.randint(0, img.shape[0] - 20)
    #     pos_x = np.random.randint(10, img.shape[0] - 10, size=(num_pixels,))
    #     pos_y = np.random.randint(10, img.shape[1] - 10, size=(num_pixels,))
    #     img[pos_x, pos_y] = np.random.choice(noise_colors, num_pixels, replace=True)
    #     pass
    return img, target, img_shape, target_shape


# class PermutableData(data.Dataset):
#
#     def __init__(self, imgs: List[ImgMatrix], targets: List[ImgMatrix], sample: bool = False):
#         super(PermutableData, self).__init__()
#         self.data = []  # type: List[Tuple[np.ndarray, np.ndarray]]
#         all_permutations = list(permutations(range(10)))
#         self.data = list(zip(imgs, targets))
#         if sample:
#             self.permutations = RandomPerm(all_permutations, 3 * 10 ** 5)
#         else:
#             self.permutations = all_permutations
#
#     def __len__(self) -> int:
#         return len(self.permutations)
#
#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         # img, label = self.data[idx]
#         # img = torch.from_numpy(img.transpose((2, 0, 1)).astype(np.float32))
#         # label = torch.from_numpy(label)
#         # cur_perm = self.permutations[idx]
#         cur_perm = list(range(10))
#         sample = random.choice(self.data)
#         img, label = sample[:2]
#         img, label = aug(img, label)[:2]
#         img = add_palette(img)
#         label = add_palette(img)
#         img, label = prep_data(img, label, cur_perm)
#         img = torch.from_numpy(img)
#         label = torch.from_numpy(np.argmax(label, axis=0))
#         return img, label


class TaskData(data.Dataset):

    def __init__(self, imgs: List[ImgMatrix], targets: List[ImgMatrix],
                 sample: bool = False, num_sample: int = 3 * 10**4,
                 max_size: int = -1):
        super(TaskData, self).__init__()
        all_permutations = list(permutations(range(10)))
        self.noise_palette = set(range(10))
        for cur_img in imgs:
            self.noise_palette -= set(np.unique(cur_img).tolist())
        if max_size < 0:
            max_size = np.max([img.shape for img in imgs] + [target.shape for target in targets])
        add_palette = config_palette(max_size)
        self.noise_palette = list(self.noise_palette)
        self.data = [(add_palette(img), add_palette(target), img.shape, target.shape)
                     for img, target in zip(imgs, targets)]
        if sample:
            self.permutations = RandomPerm(all_permutations, num_sample)
        else:
            self.permutations = all_permutations

    def __len__(self) -> int:
        return len(self.permutations)

    def __getitem__(self, idx: int
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # cur_perm = self.permutations[idx]
        cur_perm = None  # list(range(10))
        # num_sample = random.randint(2, len(self.data))
        num_sample = 2
        sample_indices = random.sample(range(len(self.data)), num_sample)
        # sample = random.sample(self.data, num_sample)
        sample = [self.data[i] for i in sample_indices]
        # sample = [aug(*data, noise_colors=self.noise_palette) for data in sample]
        sample = [prep_data(*data, cur_perm) for data in sample]
        train_sample = sample[0]
        train_labels = train_sample[1]
        train_pair = (torch.from_numpy(train_sample[0]), torch.from_numpy(np.argmax(train_labels[2:], axis=0)))
        train_shapes = torch.Tensor(train_labels[:2, 0, 0])
        pred_sample = [(torch.from_numpy(inp), torch.from_numpy(out)) for inp, out in sample[1:]]
        train_mask = np.zeros((1, *train_labels.shape[1:]), dtype=np.bool)
        h, w = (train_labels[:2, 0, 0] * 30).astype(np.int32)
        # train_mask[:, 10:10+h, 10:10+w] = 1
        train_mask[:, :h, :w] = 1
        train_mask = torch.from_numpy(train_mask)
        # return pred_sample, torch.from_numpy(sample[0][0]), torch.from_numpy(sample[0][1])
        return pred_sample[0][0], pred_sample[0][1], train_pair[0], train_pair[1], train_shapes, train_mask
