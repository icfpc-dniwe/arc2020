import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .net import SmallRecolor, SmallWeightPredictor, SmallPredictor
from .dataset import PermutableData, convert_matrix, TaskData, prep_data, add_palette
from ...operation import LearnableOperation
from ....classify import OutputSizeType
from .....mytypes import ImgMatrix
from functools import partial
import math
from typing import List, Tuple, Dict


def multi_label_cross_entropy(preds: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor):
    preds = preds.reshape(preds.size(0), preds.size(1), -1)
    labels = labels.reshape(labels.size(0), -1)
    loss = F.cross_entropy(preds, labels, weight=weight)
    return loss


def multi_label_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    accuracy = np.mean(np.argmax(preds.detach().cpu().numpy(), axis=1) == labels.detach().cpu().numpy())
    return accuracy


def weight_l2_norm(weights: Dict[str, torch.Tensor], multiplier: float = 1e-4) -> torch.Tensor:
    norm = 0
    for cur_weight in weights.values():
        norm = torch.mean(torch.sum(cur_weight ** 2, dim=1)) + norm
    return norm * multiplier


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
          ) -> Tuple[nn.Module, float]:

    # batch_size = 3 * 10**3
    batch_size = 256
    num_epochs = 20
    steps = (9, 15, np.inf)
    initial_lr = 1e-2
    momentum = 0.9
    weight_decay = 4 * 1e-4
    warmup_epoch = 1
    gamma = 0.1

    net = SmallRecolor()
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
                      num_workers=4,
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
    for epoch_idx in range(0, num_epochs):
        if (epoch_idx + 1) % steps[step_index] == 0:
            step_index += 1
        running_loss = 0.0
        running_metric = 0.0
        for batch_idx, (weight_inputs, weight_labels, inputs, labels) in enumerate(data):
            # for batch_idx, (inputs, labels) in enumerate(data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            weight_inputs = weight_inputs.to(device)
            weight_labels = weight_labels.to(device)
            iteration = epoch_idx * epoch_size + batch_idx
            lr = adjust_learning_rate(optimizer, initial_lr, warmup_epoch,
                                      gamma, epoch_idx, step_index, iteration, epoch_size)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # preds = net(inputs)
                weight_preds = weight_predictor(weight_inputs, weight_labels)
                preds = predictor(inputs, weight_preds)
                preds = preds[:, :, 10:-10, 10:-10]
                labels = labels[:, 10:-10, 10:-10]
                loss = criterion(preds, labels) + weight_l2_norm(weight_preds, weight_decay)
                loss.backward()
                optimizer.step()
            cur_loss = loss.item()
            cur_metric = multi_label_accuracy(preds, labels)
            running_loss += cur_loss * inputs.size(0)
            running_metric += cur_metric * inputs.size(0)
            print(f'Epoch[{epoch_idx:03d}][{batch_idx:04d}] Loss: {cur_loss:.3f} | '
                  f'ACC: {cur_metric.mean():.5f} | LR: {lr:.6f}')
        epoch_loss = running_loss / len(data.dataset)
        epoch_metric = running_metric / len(data.dataset)
        print(f'Epoch[{epoch_idx:03d}] Loss: {epoch_loss:.3f} | ACC: {epoch_metric:.5f}')
        # if epoch_metric > 1.0 - 1e-7:
        #     print('1.0 accuracy, early stop')
        #     break
    predictor.eval()
    weight_predictor.eval()
    return (predictor, weight_predictor), float(epoch_metric)
    # net.eval()
    # return net, float(epoch_metric)


def eval_net(net: nn.Module, imgs: List[ImgMatrix], targets: List[ImgMatrix], use_gpu: bool = False) -> float:
    batch_size = 10**4
    data = DataLoader(PermutableData(imgs, targets, sample=True),
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=6,
                      pin_memory=False)
    running_metric = 0
    for batch_idx, (inputs, labels) in enumerate(data):
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        with torch.set_grad_enabled(False):
            preds = net(inputs)
        cur_metric = multi_label_accuracy(preds, labels)
        running_metric += cur_metric * inputs.size(0)
    epoch_metric = running_metric / len(data.dataset)
    return epoch_metric


def prepare_eval(imgs: List[ImgMatrix], targets: List[ImgMatrix], device) -> Tuple[torch.Tensor, torch.Tensor]:
    inp = []
    out = []
    for cur_img, cur_target in zip(imgs, targets):
        cur_inp, cur_out = prep_data(add_palette(cur_img), add_palette(cur_target))
        inp.append(cur_inp)
        out.append(cur_out)
    inp = torch.from_numpy(np.array(inp)).to(device)
    out = torch.from_numpy(np.array(out)).to(device)
    return inp, out


def eval_weights(nets: Tuple[nn.Module, nn.Module],
                 train_imgs: List[ImgMatrix],
                 train_targets: List[ImgMatrix],
                 test_imgs: List[ImgMatrix],
                 test_targets: List[ImgMatrix],
                 use_gpu: bool = False
                 ) -> float:
    device = torch.device("cuda" if use_gpu else "cpu")
    predictor, weight_predictor = nets
    train_inp, train_out = prepare_eval(train_imgs, train_targets, device)
    test_inp, test_out = prepare_eval(test_imgs, test_targets, device)
    with torch.set_grad_enabled(False):
        weights = weight_predictor(train_inp, train_out)
        preds = predictor(test_inp, weights)
    return float(multi_label_accuracy(preds, test_out.argmax(dim=1)).mean())


class LearnCNN(LearnableOperation):
    supported_outputs = [OutputSizeType.SQUARE_SAME]

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
                           weights
                           ) -> ImgMatrix:
            prep = convert_matrix(add_palette(img))
            prep = torch.from_numpy(prep.transpose((2, 0, 1)).astype(np.float32))
            with torch.set_grad_enabled(False):
                res = predictor(prep.unsqueeze(0), weights)
            res = res.detach().squeeze(0).numpy()
            labels = np.argmax(res, axis=0)
            return ImgMatrix(labels)

        def learn(imgs, targets):
            # best_nets = None
            # num_train = len(imgs)
            # best_metric = 0
            # for cur_leave in range(num_train):
            #     train_imgs = imgs[:cur_leave]
            #     train_targets = targets[:cur_leave]
            #     # if cur_leave < num_train - 1:
            #     #     train_imgs += imgs[cur_leave+1:]
            #     #     train_targets += targets[cur_leave+1:]
            #     nets, _ = train(train_imgs, train_targets, True)
            #     # metric = eval_net(nets, test_imgs, test_targets, True)
            #     # metric = eval_weights(nets, train_imgs, train_targets, test_imgs, test_targets, True)
            #     # if metric > best_metric:
            #     #     best_metric = metric
            #     #     best_nets = nets
            #     # break
            train_imgs = imgs
            train_targets = targets
            best_nets, _ = train(train_imgs, train_targets, True)
            # best_nets.cpu()
            best_nets = (best_nets[0].cpu(), best_nets[1].cpu())
            inps, outs = prepare_eval(imgs, targets, torch.device('cpu'))
            with torch.set_grad_enabled(False):
                weights = best_nets[1](inps, outs)
            # return lambda img: run_cnn(img, best_nets)
            return lambda img: run_weight_cnn(img, best_nets[0], weights)

        return learn
