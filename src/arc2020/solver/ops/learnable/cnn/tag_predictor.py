import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from .net import SmallTagPredictor
from .dataset import convert_matrix, TaggedDataset, prep_data, config_palette, prep_img
from .metric import tag_accuracy
from .train import adjust_learning_rate
from ...operation import LearnableOperation
from ....classify import OutputSizeType
from .....task import Task
from .....mytypes import ImgMatrix, ImgPair
from functools import partial
import math
from typing import List, Tuple, Dict, Sequence


def train(train_img_pairs: List[ImgPair], train_tags: List[Sequence[Tuple[str, int]]],
          test_img_pairs: List[ImgPair], test_tags: List[Sequence[Tuple[str, int]]],
          max_size: int, use_gpu: bool = False
          ) -> Tuple[nn.Module, float]:

    # batch_size = 3 * 10**3
    batch_size = 256
    num_epochs = 50
    steps = (15, 25, 35, 50, 75, np.inf)
    initial_lr = 3 * 1e-3
    momentum = 0.9
    weight_decay = 3 * 1e-4
    warmup_epoch = 5
    gamma = 0.3

    device = torch.device("cuda" if use_gpu else "cpu")
    train_data = DataLoader(TaggedDataset(train_img_pairs, train_tags, max_size=max_size, use_aug=True),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=3,
                            pin_memory=True,
                            drop_last=True)
    val_data = DataLoader(TaggedDataset(test_img_pairs, test_tags, max_size=max_size, use_aug=False),
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=3,
                          pin_memory=True,
                          drop_last=False)
    data = {'train': train_data, 'val': val_data}
    net = SmallTagPredictor(train_data.dataset.num_tags)
    net.train()
    net.to(device)
    train_parameters = net.parameters()
    optimizer = optim.Adam(train_parameters,
                           lr=initial_lr,
                           weight_decay=weight_decay)
    criterion = F.binary_cross_entropy
    epoch_size = math.ceil(len(train_data.dataset) / batch_size)
    step_index = 0
    epoch_metric = 0
    early_stop = 0
    lr = initial_lr
    for epoch_idx in range(0, num_epochs):
        if (epoch_idx + 1) % steps[step_index] == 0:
            step_index += 1
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_metric = 0.0
            if phase == 'train':
                net.train()
            else:
                net.eval()
            for batch_idx, (left, right, tags) in enumerate(data[phase]):
                left = left.to(device)
                right = right.to(device)
                tags = tags.to(device)
                if phase == 'train':
                    iteration = epoch_idx * epoch_size + batch_idx
                    lr = adjust_learning_rate(optimizer, initial_lr, warmup_epoch,
                                              gamma, epoch_idx, step_index, iteration, epoch_size)
                    optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    preds = net(left, right)
                    loss = criterion(preds, tags)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                cur_loss = loss.item()
                cur_metric = tag_accuracy(preds, tags)
                running_loss += cur_loss * left.size(0)
                running_metric += cur_metric * left.size(0)
                print(f'Epoch[{epoch_idx:03d}][{phase}][{batch_idx:04d}] Loss: {cur_loss:.3f} | '
                      f'ACC: {cur_metric.mean():.5f} | LR: {lr:.6f}')
            epoch_loss = running_loss / len(data[phase].dataset)
            epoch_metric = running_metric / len(data[phase].dataset)
            print(f'Epoch[{epoch_idx:03d}][{phase}] Loss: {epoch_loss:.3f} | ACC: {epoch_metric:.5f}')
        if epoch_metric > 1.0 - 1e-7:
            print('1.0 accuracy, early stop')
            early_stop += 1
            if early_stop >= 2:
                break
        else:
            early_stop = 0
    net.eval()
    return net, float(epoch_metric)


def learn_tags(tasks: Dict[str, Task]):

    def run_cnn(img_pair: Tuple[ImgMatrix, ImgMatrix],
                predictor: nn.Module,
                img_size: int
                ) -> np.ndarray:
        add_palette = config_palette(img_size)
        prep_left = prep_img(add_palette(img_pair[0]), img_pair[0].shape).astype(np.float32)
        prep_right = prep_img(add_palette(img_pair[1]), img_pair[1].shape).astype(np.float32)
        prep_left = torch.from_numpy(prep_left)
        prep_right = torch.from_numpy(prep_right)
        with torch.set_grad_enabled(False):
            tag_preds = predictor(prep_left.unsqueeze(0), prep_right.unsqueeze(0))
        res = tag_preds.detach().squeeze(0).numpy()
        return res

    max_size = 32
    train_img_pairs = []
    test_img_pairs = []
    train_tags = []
    test_tags = []
    for cur_task in tasks.values():
        cur_pairs = cur_task.train
        train_img_pairs.extend(cur_pairs)
        train_tags += [cur_task.tags] * len(cur_pairs)
        cur_pairs = cur_task.test
        test_img_pairs.extend(cur_pairs)
        test_tags += [cur_task.tags] * len(cur_pairs)
    best_net, _ = train(train_img_pairs, train_tags, test_img_pairs, test_tags, max_size, True)
    best_net.cpu()
    torch.save({'state_dict': best_net.state_dict()}, 'tag_predictor.pth')

    return lambda img_pair: run_cnn(img_pair, best_net, max_size)
