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
        new_img = np.zeros((img_size, img_size), dtype=img.dtype)
        h, w = img.shape[:2]
        new_img[:h, :w] = img
        return new_img

    return add_palette


def prep_img(img: ImgMatrix,
             perm: Optional[Sequence[int]] = None
             ) -> np.ndarray:
    cur_img = convert_matrix(img)
    if perm is not None:
        cur_img = cur_img[:, :, perm]
    cur_img = cur_img.transpose((2, 0, 1))
    return cur_img.astype(np.float32)


def prep_data(img: ImgMatrix,
              target: ImgMatrix,
              perm: Optional[Sequence[int]] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
    return prep_img(img, perm), prep_img(target, perm)


def aug(img: ImgMatrix,
        target: ImgMatrix,
        noise_colors: Optional[Sequence[int]] = None
        ) -> Tuple[ImgMatrix, ImgMatrix]:
    if random.random() < 0.5:
        axis = random.randrange(0, 2)
        img = np.flip(img, axis=axis)
        target = np.flip(target, axis=axis)
    if random.random() < 0.5:
        num_rot = random.randrange(0, 4)
        img = np.rot90(img, k=num_rot)
        target = np.rot90(target, k=num_rot)
    # if noise_colors is not None and len(noise_colors) > 0 and random.random() < 0.2:
    #     num_pixels = random.randint(0, img.shape[0] - 20)
    #     pos_x = np.random.randint(10, img.shape[0] - 10, size=(num_pixels,))
    #     pos_y = np.random.randint(10, img.shape[1] - 10, size=(num_pixels,))
    #     img[pos_x, pos_y] = np.random.choice(noise_colors, num_pixels, replace=True)
    #     pass
    return img, target


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
                 sample: bool = False, num_sample: int = 3 * 10**4):
        super(TaskData, self).__init__()
        all_permutations = list(permutations(range(10)))
        self.noise_palette = set(range(10))
        for cur_img in imgs:
            self.noise_palette -= set(np.unique(cur_img).tolist())
        self.noise_palette = list(self.noise_palette)
        self.data = list(zip(imgs, targets))
        if sample:
            self.permutations = RandomPerm(all_permutations, num_sample)
        else:
            self.permutations = all_permutations

    def __len__(self) -> int:
        return len(self.permutations)

    def __getitem__(self, idx: int
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # cur_perm = self.permutations[idx]
        cur_perm = None  # list(range(10))
        # num_sample = random.randint(2, len(self.data))
        num_sample = 2
        sample_indices = random.sample(range(len(self.data)), num_sample)
        # sample = random.sample(self.data, num_sample)
        sample = [self.data[i] for i in sample_indices]
        # sample = [aug(*data, noise_colors=self.noise_palette) for data in sample]
        # sample = [(self.add_palette(inp), self.add_palette(out), inp_shape, out_shape)
        #           for inp, out, inp_shape, out_shape in sample]
        train_data = sample[0][0]
        train_labels = sample[0][1]
        diff = train_labels != train_data
        train_labels = diff * train_labels + (1 - diff) * 10
        sample = [prep_data(*data, cur_perm) for data in sample]
        train_sample = sample[0]
        train_pair = (torch.from_numpy(train_sample[0]),
                      torch.from_numpy(train_labels))
        # train_shapes = torch.Tensor(train_labels[:2, 0, 0])
        pred_sample = [(torch.from_numpy(inp), torch.from_numpy(out))
                       for inp, out in sample[1:]]
        # train_mask = np.zeros((1, *train_labels.shape[1:]), dtype=np.bool)
        # h, w = (train_labels[:2, 0, 0] * 30).astype(np.int32)
        # train_mask[:, 10:10+h, 10:10+w] = 1
        # return pred_sample, torch.from_numpy(sample[0][0]), torch.from_numpy(sample[0][1])
        return pred_sample[0][0], pred_sample[0][1], train_pair[0], train_pair[1]


class TaggedDataset(data.Dataset):

    def __init__(self, img_pairs: Sequence[Tuple[ImgMatrix, ImgMatrix]], tags: Sequence[Sequence[Tuple[str, int]]],
                 max_size: int = -1, use_aug: bool = False):
        super().__init__()
        self.img_pairs = img_pairs
        # self.tags = self.sanitize_tags(tags)
        self.tags = tags
        if max_size < 0:
            max_size = np.max([np.maximum(left.shape, right.shape) for left, right in img_pairs])
        self.palette = config_palette(max_size)
        self.use_aug = use_aug

    @staticmethod
    def sanitize_tags(all_tags) -> Sequence[Sequence[Tuple[str, int]]]:
        raise NotImplemented
        all_tag_names = list(set([tag_name for cur_tags in all_tags for tag_name, tag_val in cur_tags]))
        sanitized_tags = []
        for cur_tags in all_tags:
            pass
        return sanitized_tags

    @property
    def num_tags(self) -> int:
        return len(self.tags[0])

    def __len__(self) -> int:
        return len(self.img_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left, right = self.img_pairs[idx]
        left_shape = left.shape
        right_shape = right.shape
        tags = self.tags[idx]
        if self.use_aug:
            left, right, left_shape, right_shape = aug(left, right, left_shape, right_shape)
        left = self.palette(left)
        right = self.palette(right)
        left, right = prep_data(left, right, left_shape, right_shape)
        # now for the tags
        labels = np.asarray([tag_val for tag_name, tag_val in tags]).astype(np.float32)
        return torch.from_numpy(left), torch.from_numpy(right), torch.from_numpy(labels)
