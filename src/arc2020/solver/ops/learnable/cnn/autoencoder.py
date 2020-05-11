import cv2
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .net import AutoEncoder
from .dataset import convert_matrix, TaskData, prep_data, config_palette, prep_img
from .metric import multi_label_cross_entropy, masked_multi_label_accuracy, size_loss, multi_label_accuracy
from .train import adjust_learning_rate
from ...operation import LearnableOperation
from ....classify import OutputSizeType
from .....mytypes import ImgMatrix
from functools import partial
import math
from typing import List, Tuple, Dict


def train(imgs: List[ImgMatrix], targets: List[ImgMatrix], max_size: int, use_gpu: bool = False
          ) -> Tuple[nn.Module, float]:

    # batch_size = 3 * 10**3
    batch_size = 256
    num_epochs = 100
    steps = (15, 25, 35, 50, 75, np.inf)
    initial_lr = 1e-2
    momentum = 0.9
    weight_decay = 1 * 1e-4
    warmup_epoch = 5
    gamma = 0.3

    net = AutoEncoder()
    net.train()
    device = torch.device("cuda" if use_gpu else "cpu")
    net.to(device)
    train_parameters = net.parameters()
    data = DataLoader(TaskData(imgs, imgs, sample=True, num_sample=10 ** 4, max_size=max_size),
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=3,
                      pin_memory=True,
                      drop_last=True)
    optimizer = optim.SGD(train_parameters,
                          lr=initial_lr,
                          momentum=momentum,
                          weight_decay=weight_decay)
    weight = torch.from_numpy(np.array([1.0] * 10, dtype=np.float32))
    weight = weight.to(device)
    criterion = partial(multi_label_cross_entropy, weight=weight)
    epoch_size = math.ceil(len(data.dataset) / batch_size)
    step_index = 0
    epoch_metric = 0
    early_stop = 0
    for epoch_idx in range(0, num_epochs):
        if (epoch_idx + 1) % steps[step_index] == 0:
            step_index += 1
        running_loss = 0.0
        running_metric = 0.0
        for batch_idx, (weight_inputs, weight_labels, inputs, labels, sizes, masks) in enumerate(data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            sizes = sizes.to(device)
            masks = masks.to(device)
            iteration = epoch_idx * epoch_size + batch_idx
            lr = adjust_learning_rate(optimizer, initial_lr, warmup_epoch,
                                      gamma, epoch_idx, step_index, iteration, epoch_size)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                preds, size_preds = net(inputs)
                # zero_preds = torch.zeros_like(preds).to(device)
                # zero_preds[:, 0] = 0.5
                # preds = preds * masks + zero_preds * (~masks)
                labels = labels * masks.squeeze(1)
                s_loss = size_loss(size_preds, sizes)
                loss = criterion(preds, labels) + s_loss
                loss.backward()
                optimizer.step()
            cur_loss = loss.item()
            cur_metric = multi_label_accuracy(preds, labels)
            running_loss += cur_loss * inputs.size(0)
            running_metric += cur_metric * inputs.size(0)
            print(f'Epoch[{epoch_idx:03d}][{batch_idx:04d}] Loss: {cur_loss:.3f} {s_loss:.3f} | '
                  f'ACC: {cur_metric.mean():.5f} | LR: {lr:.6f}')
        epoch_loss = running_loss / len(data.dataset)
        epoch_metric = running_metric / len(data.dataset)
        print(f'Epoch[{epoch_idx:03d}] Loss: {epoch_loss:.3f} | ACC: {epoch_metric:.5f}')
        if epoch_metric > 1.0 - 1e-7:
            print('1.0 accuracy, early stop')
            early_stop += 1
            if early_stop >= 2:
                break
        else:
            early_stop = 0
    net.eval()
    return net, float(epoch_metric)


class LearnEncoder(LearnableOperation):
    supported_outputs = [e for e in OutputSizeType]

    @staticmethod
    def _make_learnable_operation():

        def run_cnn(img: ImgMatrix,
                    predictor: nn.Module,
                    img_size: int
                    ) -> ImgMatrix:
            add_palette = config_palette(img_size)
            prep = prep_img(add_palette(img), img.shape).astype(np.float32)
            prep = torch.from_numpy(prep)
            with torch.set_grad_enabled(False):
                res, size = predictor(prep.unsqueeze(0))
            res = res.detach().squeeze(0).numpy()
            size = (size.detach().squeeze(0).numpy() * 30).round().astype(np.int32)
            h, w = size
            labels = np.argmax(res, axis=0)
            return ImgMatrix(labels[:h, :w])

        def learn(imgs, targets):
            max_size = 32
            best_net, _ = train(imgs, targets, max_size, True)
            best_net.cpu()
            torch.save({'state_dict': best_net.state_dict()}, 'autoencoder.pth')
            return lambda img: run_cnn(img, best_net, max_size)

        return learn
