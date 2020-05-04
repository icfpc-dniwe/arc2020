import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .net import SmallRecolor, SmallWeightPredictor, SmallPredictor
from .dataset import convert_matrix, TaskData, prep_data, config_palette, prep_img
from .metric import multi_label_cross_entropy, masked_multi_label_accuracy, size_loss, weight_l2_norm
from ...operation import LearnableOperation
from ....classify import OutputSizeType
from .....mytypes import ImgMatrix
from functools import partial
import math
from typing import List, Tuple, Dict


def adjust_learning_rate(optimizer, initial_lr, warmup_epoch, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(imgs: List[ImgMatrix], targets: List[ImgMatrix], use_gpu: bool = False
          ) -> Tuple[Tuple[nn.Module, nn.Module], float]:

    # batch_size = 3 * 10**3
    batch_size = 256
    num_epochs = 15
    steps = (9, 15, np.inf)
    initial_lr = 3 * 1e-3
    momentum = 0.9
    weight_decay = 1 * 1e-4
    warmup_epoch = 1
    gamma = 0.1

    # net = SmallRecolor()
    predictor = SmallPredictor()
    weight_predictor = SmallWeightPredictor(predictor.params_shapes)
    # net.train()
    predictor.train()
    weight_predictor.train()
    device = torch.device("cuda" if use_gpu else "cpu")
    # net.to(device)
    predictor.to(device)
    weight_predictor.to(device)
    # train_parameters = net.parameters()
    train_parameters = list(predictor.parameters()) + list(weight_predictor.parameters())
    data = DataLoader(TaskData(imgs, targets, sample=True),
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=6,
                      pin_memory=True)
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
            # for batch_idx, (inputs, labels) in enumerate(data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            weight_inputs = weight_inputs.to(device)
            weight_labels = weight_labels.to(device)
            sizes = sizes.to(device)
            masks = masks.to(device)
            iteration = epoch_idx * epoch_size + batch_idx
            lr = adjust_learning_rate(optimizer, initial_lr, warmup_epoch,
                                      gamma, epoch_idx, step_index, iteration, epoch_size)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # preds = net(inputs)
                weight_preds = weight_predictor(weight_inputs, weight_labels)
                preds, size_preds = predictor(inputs, weight_preds)
                # preds = preds[:, :, 10:-10, 10:-10]
                # labels = labels[:, 10:-10, 10:-10]
                preds = preds * masks
                labels = labels * masks.squeeze(1)
                s_loss = size_loss(size_preds, sizes)
                loss = criterion(preds, labels) + weight_l2_norm(weight_preds, weight_decay) + s_loss
                loss.backward()
                optimizer.step()
            cur_loss = loss.item()
            cur_metric = masked_multi_label_accuracy(preds, labels, masks)
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
    predictor.eval()
    weight_predictor.eval()
    return (predictor, weight_predictor), float(epoch_metric)
    # net.eval()
    # return net, float(epoch_metric)


# def eval_net(net: nn.Module, imgs: List[ImgMatrix], targets: List[ImgMatrix], use_gpu: bool = False) -> float:
#     batch_size = 10**4
#     data = DataLoader(PermutableData(imgs, targets, sample=True),
#                       batch_size=batch_size,
#                       shuffle=True,
#                       num_workers=6,
#                       pin_memory=False)
#     running_metric = 0
#     for batch_idx, (inputs, labels) in enumerate(data):
#         if use_gpu:
#             inputs = inputs.cuda()
#             labels = labels.cuda()
#         with torch.set_grad_enabled(False):
#             preds = net(inputs)
#         cur_metric = multi_label_accuracy(preds, labels)
#         running_metric += cur_metric * inputs.size(0)
#     epoch_metric = running_metric / len(data.dataset)
#     return epoch_metric


def prepare_eval(imgs: List[ImgMatrix], targets: List[ImgMatrix], img_size: int, device
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    inp = []
    out = []
    add_palette = config_palette(img_size)
    for cur_img, cur_target in zip(imgs, targets):
        cur_inp, cur_out = prep_data(add_palette(cur_img), add_palette(cur_target),
                                     cur_img.shape, cur_target.shape)
        inp.append(cur_inp)
        out.append(cur_out)
    inp = torch.from_numpy(np.array(inp)).to(device)
    out = torch.from_numpy(np.array(out)).to(device)
    return inp, out


# def eval_weights(nets: Tuple[nn.Module, nn.Module],
#                  train_imgs: List[ImgMatrix],
#                  train_targets: List[ImgMatrix],
#                  test_imgs: List[ImgMatrix],
#                  test_targets: List[ImgMatrix],
#                  use_gpu: bool = False
#                  ) -> float:
#     device = torch.device("cuda" if use_gpu else "cpu")
#     predictor, weight_predictor = nets
#     train_inp, train_out = prepare_eval(train_imgs, train_targets, device)
#     test_inp, test_out = prepare_eval(test_imgs, test_targets, device)
#     with torch.set_grad_enabled(False):
#         weights = weight_predictor(train_inp, train_out)
#         preds = predictor(test_inp, weights)
#     return float(multi_label_accuracy(preds, test_out.argmax(dim=1)).mean())


class LearnCNN(LearnableOperation):
    supported_outputs = [e for e in OutputSizeType]

    @staticmethod
    def _make_learnable_operation():
        def run_cnn(img: ImgMatrix, net: nn.Module) -> ImgMatrix:
            prep = convert_matrix(img)
            prep = torch.from_numpy(prep.transpose((2, 0, 1)).astype(np.float32))
            with torch.set_grad_enabled(False):
                res = net(prep.unsqueeze(0))
            res = res.detach().squeeze(0).numpy()
            labels = np.argmax(res, axis=0)
            mask = labels == 10
            res = img * mask + labels * (1 - mask)
            res_matrix = ImgMatrix(res)
            return res_matrix

        def run_weight_cnn(img: ImgMatrix,
                           predictor: nn.Module,
                           weights,
                           img_size: int
                           ) -> ImgMatrix:
            add_palette = config_palette(img_size)
            prep = prep_img(add_palette(img), img.shape).astype(np.float32)
            prep = torch.from_numpy(prep)
            with torch.set_grad_enabled(False):
                res, size = predictor(prep.unsqueeze(0), weights)
            res = res.detach().squeeze(0).numpy()
            size = (size.detach().squeeze(0).numpy() * 30).round().astype(np.int32)
            h, w = size
            labels = np.argmax(res, axis=0)
            return ImgMatrix(labels[10:10+h, 10:10+w])

        def learn(imgs, targets):
            train_imgs = imgs
            train_targets = targets
            max_size = np.max([img.shape for img in imgs] + [target.shape for target in targets])
            # print('cur_max_size', max_size, '|', imgs[0].shape, '|', targets[0].shape)
            best_nets, _ = train(train_imgs, train_targets, True)
            # best_nets.cpu()
            best_nets = (best_nets[0].cpu(), best_nets[1].cpu())
            inps, outs = prepare_eval(imgs, targets, max_size, torch.device('cpu'))
            with torch.set_grad_enabled(False):
                weights = best_nets[1](inps, outs)
            # return lambda img: run_cnn(img, best_nets)
            return lambda img: run_weight_cnn(img, best_nets[0], weights, max_size)

        return learn
