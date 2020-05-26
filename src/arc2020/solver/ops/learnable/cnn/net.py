import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from pathos.multiprocessing import ThreadingPool
from typing import Dict, Iterable, Tuple


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def get_act(activation: str):
    if activation == 'relu':
        act = nn.ReLU
    elif activation == 'relu6':
        act = nn.ReLU6
    elif activation == 'prelu':
        act = nn.PReLU
    else:
        act = nn.ReLU
    return act


def conv_bn(inp, oup, stride = 1, kernel = 3, act=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        act(inplace=True)
    )


def conv_bn_no_relu(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride=1, act=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        act(inplace=True)
    )


def conv_dw(inp, oup, stride=1, act=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        act(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        act(inplace=True)
    )


class EncodeBlock(nn.Module):

    def __init__(self, input_features: int, num_features: int, activation: str = 'relu', reduction: int = 2):
        super(EncodeBlock, self).__init__()
        act = get_act(activation)
        self.reduce = nn.Sequential(
            nn.BatchNorm2d(input_features),
            act(),
            nn.Conv2d(input_features, num_features, reduction * 2, reduction, reduction // 2, bias=False),
            nn.BatchNorm2d(num_features),
            act(inplace=True)
        )

    def forward(self, input: torch.Tensor):
        return self.reduce(input)


class DecodeBlock(nn.Module):

    def __init__(self, input_features: int, num_features: int, activation: str = 'relu', upsample: int = 2):
        super(DecodeBlock, self).__init__()
        act = get_act(activation)
        self.upsample = nn.Sequential(
            nn.BatchNorm2d(input_features),
            act(),
            # nn.Conv2d(input_features, num_features, reduction * 2, reduction, reduction // 2, bias=False),
            nn.ConvTranspose2d(input_features, num_features, upsample * 2, upsample, upsample // 2),
            nn.BatchNorm2d(num_features),
            act()
        )

    def forward(self, input: torch.Tensor):
        return self.upsample(input)


class SmallRecolor(nn.Module):

    def __init__(self):
        super(SmallRecolor, self).__init__()
        self.prep1 = nn.Sequential(
            conv_bn(12, 12),
            conv_bn(12, 12),
            conv_bn(12, 12)
        )
        self.prep2 = nn.Sequential(
            conv_bn(24, 24),
            conv_bn(24, 24),
            conv_bn(24, 12)
        )
        self.res = nn.Sequential(
            conv_bn(24, 32),
            conv_bn(32, 32),
            conv_bn(32, 32)
        )
        self.mat_pred = nn.Conv2d(32, 10, 3, 1, 1)
        self.size_pred = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 2, 1, 1, 0),
            Flatten()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cur = self.prep1(x)
        cur = self.prep2(torch.cat((cur, x), dim=1))
        cur = self.res(torch.cat((cur, x), dim=1))
        return self.mat_pred(cur), self.size_pred(cur)


class SmallPredictor(nn.Module):

    def __init__(self):
        super(SmallPredictor, self).__init__()
        self.params_shapes = OrderedDict(
            stage1=OrderedDict(
                conv1_1=(12, 10, 3, 3),
                conv1_2=(12, 12, 3, 3),
                conv1_3=(12, 12, 3, 3),
            ),
            stage2=OrderedDict(
                conv2_1=(24, 22, 3, 3),
                conv2_2=(24, 24, 3, 3),
                conv2_3=(12, 24, 3, 3),
            ),
            stage3=OrderedDict(
                conv3_1=(24, 22, 3, 3),
                conv3_2=(24, 24, 3, 3),
                conv3_3=(12, 24, 3, 3),
            ),
            stage4=OrderedDict(
                conv_pred=(11, 22, 1, 1)
            )
        )
        self.stage_bns = {key: {ins_key: nn.BatchNorm2d(ins_val[0]) for ins_key, ins_val in val.items()}
                          for key, val in self.params_shapes.items() if key != 'stage4'}
        for key, ins_dict in self.stage_bns.items():
            for ins_key, ins_module in ins_dict.items():
                self.add_module(f'{key}_{ins_key}', ins_module)

    def prepare_weights(self, weights: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        res_weights = {}
        for key, cur_weights in weights.items():
            # if len(cur_weights.shape) > 1:
            #     cur_weights = cur_weights.mean(dim=0)
            res_weights[key] = {}
            cur_bg = 0
            for ins_key, ins_shape in self.params_shapes[key].items():
                cur_size = int(np.prod(ins_shape))
                res_weights[key][ins_key] = cur_weights[:, cur_bg:cur_bg+cur_size].reshape((-1, *ins_shape))
                cur_bg += cur_size
        return res_weights

    @staticmethod
    def splitted_conv2d(batch: torch.Tensor, weights: torch.Tensor, padding: int) -> torch.Tensor:
        with ThreadingPool(4) as p:
            results = p.map(lambda el: F.conv2d(el[0].unsqueeze(0), el[1], padding=padding), zip(batch, weights))
        return torch.cat(results, dim=0)

    def forward(self, x: torch.Tensor, weights: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        cur_stage = x
        weights = self.prepare_weights(weights)
        for stage_key, stage_vals in self.params_shapes.items():
            if stage_key == 'stage4':
                break
            for conv_key in stage_vals.keys():
                padding = self.params_shapes[stage_key][conv_key][2] // 2
                conv = self.splitted_conv2d(cur_stage, weights[stage_key][conv_key], padding=padding)
                bn = self.stage_bns[stage_key][conv_key](conv)
                cur_stage = F.relu(bn, inplace=True)
            cur_stage = torch.cat([cur_stage, x], dim=1)
        # cur_size = F.conv2d(cur_stage, weights['stage4']['size_pred1'], padding=0)
        # cur_size = F.adaptive_avg_pool2d(cur_size, (1, 1))
        # cur_size = F.conv2d(cur_size, weights['stage4']['size_pred2'], padding=0)
        # cur_size = cur_size.view(-1, 2)
        padding = self.params_shapes['stage4']['conv_pred'][2] // 2
        cur_stage = self.splitted_conv2d(cur_stage, weights['stage4']['conv_pred'], padding=padding)
        return cur_stage


class ExstraSmallPredictor(SmallPredictor):

    def __init__(self):
        super().__init__()
        self.params_shapes = OrderedDict(
            stage1=OrderedDict(
                conv1_1=(8, 10, 3, 3),
            ),
            stage2=OrderedDict(
                conv2_1=(8, 18, 3, 3),
            ),
            stage3=OrderedDict(
                conv3_1=(8, 18, 3, 3),
            ),
            stage4=OrderedDict(
                conv_pred=(11, 18, 3, 3)
            )
        )
        self.stage_bns = {key: {ins_key: nn.BatchNorm2d(ins_val[0]) for ins_key, ins_val in val.items()}
                          for key, val in self.params_shapes.items() if key != 'stage4'}
        for key, ins_dict in self.stage_bns.items():
            for ins_key, ins_module in ins_dict.items():
                self.add_module(f'{key}_{ins_key}', ins_module)


class SmallWeightPredictor(nn.Module):

    def __init__(self, params_shapes: Dict[str, Dict[str, Iterable[int]]]):
        super(SmallWeightPredictor, self).__init__()
        self.inp_s = self.head_conv()
        self.out_s = self.head_conv()
        self.stage_keys = params_shapes.keys()
        # self.stages = []
        self.w1_pred = self.weight_pred(24, self.calc_weight_size(params_shapes['stage1']))
        self.stage2 = self.stage_conv()
        self.w2_pred = self.weight_pred(24, self.calc_weight_size(params_shapes['stage2']))
        self.stage3 = self.stage_conv()
        self.w3_pred = self.weight_pred(24, self.calc_weight_size(params_shapes['stage3']))
        self.stage4 = self.stage_conv()
        self.w4_pred = self.weight_pred(24, self.calc_weight_size(params_shapes['stage4']))

    @staticmethod
    def calc_weight_size(params_shape: Dict[str, Iterable[int]]) -> int:
        return int(np.sum([np.prod(cur_shape) for cur_shape in params_shape.values()]))

    @staticmethod
    def weight_pred(inp_channels, weight_size) -> nn.Module:
        if weight_size % 9 == 0:
            pred = nn.Sequential(
                conv_bn(inp_channels, 24),
                nn.Conv2d(24, 48, 1),
                nn.AdaptiveAvgPool2d((3, 3)),
                nn.Conv2d(48, weight_size // 9, 3, padding=1),
                Flatten()
            )
        else:
            pred = nn.Sequential(
                conv_bn(inp_channels, 24),
                nn.Conv2d(24, 24, 1),
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Linear(24, weight_size)
            )
        return pred

    @staticmethod
    def head_conv() -> nn.Module:
        return nn.Sequential(
            conv_bn(10, 12),
            conv_bn(12, 12),
            conv_bn(12, 12)
        )

    @staticmethod
    def stage_conv() -> nn.Module:
        return nn.Sequential(
            conv_bn(24, 24),
            conv_bn(24, 24),
            conv_bn(24, 24)
        )

    def forward(self, inp: torch.Tensor, out: torch.Tensor) -> Dict[str, torch.Tensor]:
        preds = {}
        inp = self.inp_s(inp)
        out = self.out_s(out)
        stage1 = torch.cat([inp, out], dim=1)
        preds['stage1'] = self.w1_pred(stage1)
        stage2 = stage1 + self.stage2(stage1)
        preds['stage2'] = self.w2_pred(stage2)
        stage3 = stage2 + self.stage3(stage2)
        preds['stage3'] = self.w3_pred(stage3)
        stage4 = stage3 + self.stage4(stage3)
        preds['stage4'] = self.w4_pred(stage4)
        return preds


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(12, 12),
            conv_bn(12, 12),
            conv_bn(12, 12),
            conv_bn(12, 24, 2),
            conv_bn(24, 24),
            conv_bn(24, 24),
            conv_bn(24, 24),
            conv_bn(24, 24, 2),
            conv_bn(24, 24),
            conv_bn(24, 24),
            conv_bn(24, 24),
            conv_bn(24, 48, 2),
            conv_bn(48, 48),
            conv_bn(48, 48),
            conv_bn(48, 24),
            conv_bn(24, 48, 2),
            conv_bn(48, 48),
            conv_bn(48, 48)
        )
        self.upscale = nn.Sequential(
            conv_bn(48, 64),
            DecodeBlock(64, 128),
            conv_bn(128, 128),
            conv_bn(128, 128),
            conv_bn(128, 128),
            DecodeBlock(128, 64),
            conv_bn(64, 64),
            conv_bn(64, 64),
            conv_bn(64, 64),
            DecodeBlock(64, 64),
            conv_bn(64, 64),
            conv_bn(64, 64),
            conv_bn(64, 64),
            DecodeBlock(64, 32),
            conv_bn(32, 32),
            conv_bn(32, 32),
            conv_bn(32, 32),
        )
        self.mat_pred = nn.Conv2d(32, 10, 3, 1, 1)
        self.size_pred = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 2, 1, 1, 0),
            Flatten()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        upscale = self.upscale(features)
        return self.mat_pred(upscale), torch.sigmoid(self.size_pred(upscale))


class SmallTagPredictor(nn.Module):

    def __init__(self, num_tags: int):
        super().__init__()
        self.prep1_left = nn.Sequential(
            conv_dw(12, 12, stride=2),
            # conv_bn(12, 12),
            conv_dw(12, 12)
        )
        self.prep1_right = nn.Sequential(
            conv_dw(12, 12, stride=2),
            # conv_bn(12, 12),
            conv_dw(12, 12)
        )
        self.stage1 = nn.Sequential(
            conv_dw(24, 48, stride=2),
            # conv_bn(48, 48, stride=1),
            # conv_bn(48, 48, stride=1),
            conv_dw(48, 48, stride=1)
        )
        self.bottleneck1 = nn.Sequential(
            conv_dw(24, 48, stride=2)
        )
        self.stage2 = nn.Sequential(
            conv_dw(48, 48, stride=2),
            # conv_bn(48, 48, stride=1),
            # conv_bn(48, 48, stride=1),
            conv_dw(48, 48, stride=1),
            conv_bn1X1(48, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(128, num_tags)
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_p = self.prep1_left(left)
        right_p = self.prep1_left(right)
        prev = torch.cat((left_p, right_p), dim=1)
        cur = self.stage1(prev)
        bottle1 = self.bottleneck1(prev)
        cur = self.stage2(cur + bottle1)
        return torch.sigmoid(cur)
