# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:03:30 2021

@author: jcleng
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, DataParallel
from torch.nn.functional import pad

from fsdiffnet.MyClass import *
from fsdiffnet.utils import *


def padding_flex(x, size=0, mode="circular"):
    p = x.shape[-1]
    s = size
    if p >= s:
        x = pad(x, pad=(s, s, s, s), mode=mode)
    else:
        while s > p:
            x = pad(x, pad=(p, p, p, p), mode=mode)
            s -= p
            p += p
        x = pad(x, pad=(s, s, s, s), mode=mode)
    return x




class Baseline_double_500(nn.Module):
    def __init__(self):
        # 使用super()方法调用基类的构造器，即nn.Module.__init__(self)
        super(Baseline_double_500, self).__init__()
        self.diagconv11 = DiagConv2d(2, 50, (3, 3), dilation=1, diag_ratio=1)
        self.diagconv21 = DiagConv2d(50, 50, (3, 3), dilation=3, diag_ratio=1)
        self.diagconv31 = DiagConv2d(50, 50, (3, 3), dilation=9, diag_ratio=1)
        self.diagconv41 = DiagConv2d(50, 50, (3, 3), dilation=27, diag_ratio=1)
        self.diagconv51 = DiagConv2d(50, 50, (3, 3), dilation=81, diag_ratio=1)
        self.conv1 = Conv2d(50, 1, (1, 1), padding=0)

        self.batch1, self.batch2, self.batch3, self.batch4, self.batch5 = (
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
        )

    def forward(self, x):

        # (N,C,H,W) C is number of features
        x = pad(x, pad=(1, 1, 1, 1), mode="circular")
        x = self.diagconv11(x)
        x = self.batch1(x)
        x = F.relu(x)

        x = pad(x, pad=(3, 3, 3, 3), mode="circular")
        x = self.diagconv21(x)
        x = self.batch2(x)
        x = F.relu(x)

        x = pad(x, pad=(9, 9, 9, 9), mode="circular")  # (3-1)*2+3-7/2
        x = self.diagconv31(x)
        x = self.batch3(x)
        x = F.relu(x)

        x = pad(x, pad=(27, 27, 27, 27), mode="circular")  # (7-3)*2+7-15/2
        x = self.diagconv41(x)
        x = self.batch4(x)
        x = F.relu(x)

        x = pad(x, pad=(39, 39, 39, 39), mode="circular")  # (15-7)*2+15-31/2
        x = pad(x, pad=(42, 42, 42, 42), mode="circular")
        x = self.diagconv51(x)
        x = self.batch5(x)
        x = F.relu(x)

        # 1*1 conv layer
        x = self.conv1(x)
        x = torch.flatten(x, start_dim=1)

        # output layer
        x = torch.tanh(x)
        return x


class DeepNet_baseline(nn.Module):
    def __init__(self):
        # 使用super()方法调用基类的构造器，即nn.Module.__init__(self)
        super(DeepNet_baseline, self).__init__()
        self.diagconv11 = DiagConv2d(1, 50, (3, 3), dilation=1, diag_ratio=1)
        self.diagconv21 = DiagConv2d(50, 50, (3, 3), dilation=3, diag_ratio=1)
        self.diagconv31 = DiagConv2d(50, 50, (3, 3), dilation=9, diag_ratio=1)
        self.diagconv41 = DiagConv2d(50, 50, (3, 3), dilation=27, diag_ratio=1)
        self.diagconv51 = DiagConv2d(50, 50, (3, 3), dilation=81, diag_ratio=1)
        self.conv1 = Conv2d(50, 1, (1, 1), padding=0)

        self.batch1, self.batch2, self.batch3, self.batch4, self.batch5 = (
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
        )

    def forward(self, x):

        # (N,C,H,W) C is number of features
        x = padding_flex(x, 1, mode="circular")
        x = self.diagconv11(x)
        x = self.batch1(x)
        x = F.relu(x)

        x = padding_flex(x, 3, mode="circular")
        x = self.diagconv21(x)
        x = self.batch2(x)
        x = F.relu(x)

        x = padding_flex(x, 9, mode="circular")
        x = self.diagconv31(x)
        x = self.batch3(x)
        x = F.relu(x)

        x = padding_flex(x, 27, mode="circular")
        x = self.diagconv41(x)
        x = self.batch4(x)
        x = F.relu(x)

        x = padding_flex(x, 81, mode="circular")  # (15-7)*2+15-31/2
        x = self.diagconv51(x)
        x = self.batch5(x)
        x = F.relu(x)

        # 1*1 conv layer
        x = self.conv1(x)
        x = torch.flatten(x, start_dim=1)

        # output layer
        x = torch.tanh(x)
        return x


class Baseline_single_500(nn.Module):
    def __init__(self):
        super(Baseline_single_500, self).__init__()
        self.channel1 = DataParallel(DeepNet_baseline())
        self.channel2 = DataParallel(DeepNet_baseline())

    def forward(self, x):

        # (N,C,H,W) C is number of features
        x1 = x[:, 0:1, :, :]
        x2 = x[:, 1:2, :, :]
        x1 = self.channel1(x1)
        x2 = self.channel2(x2)
        # output layer
        x = x1 - x2
        return x


class FSDiffNet_500(nn.Module):
    def __init__(self):
        # 使用super()方法调用基类的构造器，即nn.Module.__init__(self)
        super(FSDiffNet_500, self).__init__()
        self.diagconv11 = DiagConv2d(1, 50, (3, 3), dilation=1, diag_ratio=1)
        self.diagconv12 = DiagConv2d(
            1, 50, (3, 3), dilation=1, diag_ratio=1, bias=False
        )
        self.diagconv21 = DiagConv2d(50, 50, (3, 3), dilation=3, diag_ratio=1)
        self.diagconv22 = DiagConv2d(
            100, 50, (3, 3), dilation=3, diag_ratio=1, bias=False
        )
        self.diagconv31 = DiagConv2d(50, 50, (3, 3), dilation=9, diag_ratio=1)
        self.diagconv32 = DiagConv2d(
            100, 50, (3, 3), dilation=9, diag_ratio=1, bias=False
        )
        self.diagconv41 = DiagConv2d(50, 50, (3, 3), dilation=27, diag_ratio=1)
        self.diagconv42 = DiagConv2d(
            100, 50, (3, 3), dilation=27, diag_ratio=1, bias=False
        )
        self.diagconv51 = DiagConv2d(50, 50, (3, 3), dilation=81, diag_ratio=1)
        self.diagconv52 = DiagConv2d(
            100, 50, (3, 3), dilation=81, diag_ratio=1, bias=False
        )

        # diff channel

        self.lambd6 = torch.nn.Parameter(torch.tensor([0.05]))
        self.lambd7 = torch.nn.Parameter(torch.tensor([0.05]))
        self.lambd8 = torch.nn.Parameter(torch.tensor([0.05]))
        self.lambd9 = torch.nn.Parameter(torch.tensor([0.05]))
        self.lambd10 = torch.nn.Parameter(torch.tensor([0.05]))

        self.conv1 = Conv2d(50, 1, (1, 1), padding=0)
        self.conv2 = Conv2d(100, 1, (1, 1), padding=0, bias=False)
        self.conv3 = Conv2d(2, 1, (1, 1), padding=0, bias=False)

        self.batch1, self.batch2, self.batch3, self.batch4, self.batch5 = (
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
        )
        self.batch6 = nn.BatchNorm2d(100, affine=False)
        self.batch7 = nn.BatchNorm2d(100, affine=False)
        self.batch8 = nn.BatchNorm2d(100, affine=False)
        self.batch9 = nn.BatchNorm2d(100, affine=False)
        self.batch10 = nn.BatchNorm2d(100, affine=False)
        self.batch11 = nn.BatchNorm2d(2, affine=False)

    def forward(self, x):

        # (N,C,H,W) C is number of features
        x_a = x[:, 0:1, :, :]
        x_b = x[:, 1:2, :, :]
        x_ab = [x_a, x_b]

        x3 = x_ab[0] - x_ab[1]
        x3 = pad(x3, pad=(1, 1, 1, 1), mode="circular")
        x3 = self.diagconv12(x3)

        x = x_ab[0]
        x = pad(x, pad=(1, 1, 1, 1), mode="circular")
        x = self.diagconv11(x)
        x_ab[0] = x

        x = x_ab[1]
        x = pad(x, pad=(1, 1, 1, 1), mode="circular")
        x = self.diagconv11(x)
        x_ab[1] = x

        x3 = torch.cat((x3, x_ab[0] - x_ab[1]), dim=1)
        x3 = self.batch6(x3)
        x3 = softsparse(x3, torch.abs(self.lambd6))

        x3 = pad(x3, pad=(3, 3, 3, 3), mode="circular")
        x3 = self.diagconv22(x3)

        x = x_ab[0]
        x = self.batch1(x)
        x = F.relu(x)
        x = pad(x, pad=(3, 3, 3, 3), mode="circular")
        x = self.diagconv21(x)
        x_ab[0] = x

        x = x_ab[1]
        x = self.batch1(x)
        x = F.relu(x)
        x = pad(x, pad=(3, 3, 3, 3), mode="circular")
        x = self.diagconv21(x)
        x_ab[1] = x

        x3 = torch.cat((x3, x_ab[0] - x_ab[1]), dim=1)
        x3 = self.batch7(x3)
        x3 = softsparse(x3, torch.abs(self.lambd7))

        x3 = pad(x3, pad=(9, 9, 9, 9), mode="circular")
        x3 = self.diagconv32(x3)

        x = x_ab[0]
        x = self.batch2(x)
        x = F.relu(x)
        x = pad(x, pad=(9, 9, 9, 9), mode="circular")
        x = self.diagconv31(x)
        x_ab[0] = x

        x = x_ab[1]
        x = self.batch2(x)
        x = F.relu(x)
        x = pad(x, pad=(9, 9, 9, 9), mode="circular")
        x = self.diagconv31(x)
        x_ab[1] = x

        x3 = torch.cat((x3, x_ab[0] - x_ab[1]), dim=1)
        x3 = self.batch8(x3)
        x3 = softsparse(x3, torch.abs(self.lambd8))

        x3 = pad(x3, pad=(27, 27, 27, 27), mode="circular")
        x3 = self.diagconv42(x3)

        x = x_ab[0]
        x = self.batch3(x)
        x = F.relu(x)
        x = pad(x, pad=(27, 27, 27, 27), mode="circular")
        x = self.diagconv41(x)
        x_ab[0] = x

        x = x_ab[1]
        x = self.batch3(x)
        x = F.relu(x)
        x = pad(x, pad=(27, 27, 27, 27), mode="circular")
        x = self.diagconv41(x)
        x_ab[1] = x

        x3 = torch.cat((x3, x_ab[0] - x_ab[1]), dim=1)
        x3 = self.batch9(x3)
        x3 = softsparse(x3, torch.abs(self.lambd9))
        x3 = pad(x3, pad=(81, 81, 81, 81), mode="circular")
        x3 = self.diagconv52(x3)

        x = x_ab[0]
        x = self.batch4(x)
        x = F.relu(x)
        x = pad(x, pad=(81, 81, 81, 81), mode="circular")
        x = self.diagconv51(x)
        x_ab[0] = x

        x = x_ab[1]
        x = self.batch4(x)
        x = F.relu(x)
        x = pad(x, pad=(81, 81, 81, 81), mode="circular")
        x = self.diagconv51(x)
        x_ab[1] = x

        x3 = torch.cat((x3, x_ab[0] - x_ab[1]), dim=1)
        x3 = self.batch10(x3)
        x3 = softsparse(x3, torch.abs(self.lambd10))
        x3 = self.conv2(x3)

        x = x_ab[0]
        x = self.batch5(x)
        x = F.relu(x)
        x = self.conv1(x)
        x_ab[0] = x

        x = x_ab[1]
        x = self.batch5(x)
        x = F.relu(x)
        x = self.conv1(x)
        x_ab[1] = x

        x3 = torch.cat((x3, x_ab[0] - x_ab[1]), dim=1)
        x3 = self.batch11(x3)
        x3 = self.conv3(x3)

        # 1*1 conv layer
        x = torch.flatten(x3, start_dim=1)

        # output layer
        x = torch.tanh(x)
        return x



class FSDiffNet(nn.Module):
    # FSDiffNet
    def __init__(self):
        # 使用super()方法调用基类的构造器，即nn.Module.__init__(self)
        super(FSDiffNet, self).__init__()
        self.diagconv11 = DiagConv2d(1, 50, (3, 3), dilation=1, diag_ratio=1)
        self.diagconv12 = DiagConv2d(
            1, 50, (3, 3), dilation=1, diag_ratio=1, bias=False
        )
        self.diagconv21 = DiagConv2d(50, 50, (3, 3), dilation=3, diag_ratio=1)
        self.diagconv22 = DiagConv2d(
            100, 50, (3, 3), dilation=3, diag_ratio=1, bias=False
        )
        self.diagconv31 = DiagConv2d(50, 50, (3, 3), dilation=9, diag_ratio=1)
        self.diagconv32 = DiagConv2d(
            100, 50, (3, 3), dilation=9, diag_ratio=1, bias=False
        )
        self.diagconv41 = DiagConv2d(50, 50, (3, 3), dilation=27, diag_ratio=1)
        self.diagconv42 = DiagConv2d(
            100, 50, (3, 3), dilation=27, diag_ratio=1, bias=False
        )
        self.diagconv51 = DiagConv2d(50, 50, (3, 3), dilation=81, diag_ratio=1)
        self.diagconv52 = DiagConv2d(
            100, 50, (3, 3), dilation=81, diag_ratio=1, bias=False
        )

        # diff channel

        self.lambd6 = torch.nn.Parameter(torch.tensor([0.05]))
        self.lambd7 = torch.nn.Parameter(torch.tensor([0.05]))
        self.lambd8 = torch.nn.Parameter(torch.tensor([0.05]))
        self.lambd9 = torch.nn.Parameter(torch.tensor([0.05]))
        self.lambd10 = torch.nn.Parameter(torch.tensor([0.05]))

        self.conv1 = Conv2d(50, 1, (1, 1), padding=0)
        self.conv2 = Conv2d(100, 1, (1, 1), padding=0, bias=False)
        self.conv3 = Conv2d(2, 1, (1, 1), padding=0, bias=False)

        self.batch1, self.batch2, self.batch3, self.batch4, self.batch5 = (
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
        )
        self.batch6 = nn.BatchNorm2d(100, affine=False)
        self.batch7 = nn.BatchNorm2d(100, affine=False)
        self.batch8 = nn.BatchNorm2d(100, affine=False)
        self.batch9 = nn.BatchNorm2d(100, affine=False)
        self.batch10 = nn.BatchNorm2d(100, affine=False)
        self.batch11 = nn.BatchNorm2d(2, affine=False)

    def forward(self, x):

        # (N,C,H,W) C is number of features
        x_a = x[:, 0:1, :, :]
        x_b = x[:, 1:2, :, :]
        x_ab = [x_a, x_b]

        x3 = x_ab[0] - x_ab[1]
        x3 = padding_flex(x3, 1, mode="circular")
        x3 = self.diagconv12(x3)

        x = x_ab[0]
        x = padding_flex(x, 1, mode="circular")
        x = self.diagconv11(x)
        x_ab[0] = x

        x = x_ab[1]
        x = padding_flex(x, 1, mode="circular")
        x = self.diagconv11(x)
        x_ab[1] = x

        x3 = torch.cat((x3, x_ab[0] - x_ab[1]), dim=1)
        x3 = self.batch6(x3)
        x3 = softsparse(x3, torch.abs(self.lambd6))

        x3 = padding_flex(x3, 3, mode="circular")
        x3 = self.diagconv22(x3)

        x = x_ab[0]
        x = self.batch1(x)
        x = F.relu(x)
        x = padding_flex(x, 3, mode="circular")
        x = self.diagconv21(x)
        x_ab[0] = x

        x = x_ab[1]
        x = self.batch1(x)
        x = F.relu(x)
        x = padding_flex(x, 3, mode="circular")
        x = self.diagconv21(x)
        x_ab[1] = x

        x3 = torch.cat((x3, x_ab[0] - x_ab[1]), dim=1)
        x3 = self.batch7(x3)
        x3 = softsparse(x3, torch.abs(self.lambd7))

        x3 = padding_flex(x3, 9, mode="circular")
        x3 = self.diagconv32(x3)

        x = x_ab[0]
        x = self.batch2(x)
        x = F.relu(x)
        x = padding_flex(x, 9, mode="circular")
        x = self.diagconv31(x)
        x_ab[0] = x

        x = x_ab[1]
        x = self.batch2(x)
        x = F.relu(x)
        x = padding_flex(x, 9, mode="circular")
        x = self.diagconv31(x)
        x_ab[1] = x

        x3 = torch.cat((x3, x_ab[0] - x_ab[1]), dim=1)
        x3 = self.batch8(x3)
        x3 = softsparse(x3, torch.abs(self.lambd8))

        x3 = padding_flex(x3, 27, mode="circular")
        x3 = self.diagconv42(x3)

        x = x_ab[0]
        x = self.batch3(x)
        x = F.relu(x)
        x = padding_flex(x, 27, mode="circular")
        x = self.diagconv41(x)
        x_ab[0] = x

        x = x_ab[1]
        x = self.batch3(x)
        x = F.relu(x)
        x = padding_flex(x, 27, mode="circular")
        x = self.diagconv41(x)
        x_ab[1] = x

        x3 = torch.cat((x3, x_ab[0] - x_ab[1]), dim=1)
        x3 = self.batch9(x3)
        x3 = softsparse(x3, torch.abs(self.lambd9))
        x3 = padding_flex(x3, 81, mode="circular")
        x3 = self.diagconv52(x3)

        x = x_ab[0]
        x = self.batch4(x)
        x = F.relu(x)
        x = padding_flex(x, 81, mode="circular")
        x = self.diagconv51(x)
        x_ab[0] = x

        x = x_ab[1]
        x = self.batch4(x)
        x = F.relu(x)
        x = padding_flex(x, 81, mode="circular")
        x = self.diagconv51(x)
        x_ab[1] = x

        x3 = torch.cat((x3, x_ab[0] - x_ab[1]), dim=1)
        x3 = self.batch10(x3)
        x3 = softsparse(x3, torch.abs(self.lambd10))
        x3 = self.conv2(x3)

        x = x_ab[0]
        x = self.batch5(x)
        x = F.relu(x)
        x = self.conv1(x)
        x_ab[0] = x

        x = x_ab[1]
        x = self.batch5(x)
        x = F.relu(x)
        x = self.conv1(x)
        x_ab[1] = x

        x3 = torch.cat((x3, x_ab[0] - x_ab[1]), dim=1)
        x3 = self.batch11(x3)
        x3 = self.conv3(x3)

        # 1*1 conv layer
        x = torch.flatten(x3, start_dim=1)

        # output layer
        x = torch.tanh(x)
        return x


class Baseline_single(nn.Module):
    def __init__(self):
        super(Baseline_single, self).__init__()
        self.channel1 = DataParallel(DeepNet_baseline())
        self.channel2 = DataParallel(DeepNet_baseline())

    def forward(self, x):

        # (N,C,H,W) C is number of features
        x1 = x[:, 0:1, :, :]
        x2 = x[:, 1:2, :, :]
        x1 = self.channel1(x1)
        x2 = self.channel2(x2)
        # output layer
        x = x1 - x2
        return x


class Baseline_double(nn.Module):
    def __init__(self):
        # 使用super()方法调用基类的构造器，即nn.Module.__init__(self)
        super(Baseline_double, self).__init__()
        self.diagconv11 = DiagConv2d(2, 50, (3, 3), dilation=1, diag_ratio=1)
        self.diagconv21 = DiagConv2d(50, 50, (3, 3), dilation=3, diag_ratio=1)
        self.diagconv31 = DiagConv2d(50, 50, (3, 3), dilation=9, diag_ratio=1)
        self.diagconv41 = DiagConv2d(50, 50, (3, 3), dilation=27, diag_ratio=1)
        self.diagconv51 = DiagConv2d(50, 50, (3, 3), dilation=81, diag_ratio=1)
        self.conv1 = Conv2d(50, 1, (1, 1), padding=0)

        self.batch1, self.batch2, self.batch3, self.batch4, self.batch5 = (
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
            nn.BatchNorm2d(50),
        )

    def forward(self, x):

        # (N,C,H,W) C is number of features
        x = pad(x, pad=(1, 1, 1, 1), mode="circular")
        x = self.diagconv11(x)
        x = self.batch1(x)
        x = F.relu(x)

        x = pad(x, pad=(3, 3, 3, 3), mode="circular")
        x = self.diagconv21(x)
        x = self.batch2(x)
        x = F.relu(x)

        x = pad(x, pad=(9, 9, 9, 9), mode="circular")  # (3-1)*2+3-7/2
        x = self.diagconv31(x)
        x = self.batch3(x)
        x = F.relu(x)

        x = pad(x, pad=(27, 27, 27, 27), mode="circular")  # (7-3)*2+7-15/2
        x = self.diagconv41(x)
        x = self.batch4(x)
        x = F.relu(x)

        x = pad(x, pad=(39, 39, 39, 39), mode="circular")  # (15-7)*2+15-31/2
        x = pad(x, pad=(42, 42, 42, 42), mode="circular")
        x = self.diagconv51(x)
        x = self.batch5(x)
        x = F.relu(x)

        # 1*1 conv layer
        x = self.conv1(x)
        x = torch.flatten(x, start_dim=1)

        # output layer
        x = torch.tanh(x)
        return x


# =============================================================================
# Loss Functions
# =============================================================================



class BCELoss_tanh(nn.modules.loss._Loss):
    def __init__(self, reduction="mean"):
        super(BCELoss_tanh, self).__init__(reduction)

    def forward(self, output, target):

        l = output.shape[1]
        EPS = 1e-12
        loss = -0.5 * (
            (1 + target) * torch.log((1 + output) / 2 + EPS)
            + (1 - target) * torch.log((1 - output) / 2 + EPS)
        ) + 0.5 * (
            (1 + target) * torch.log((1 + target) / 2 + EPS)
            + (1 - target) * torch.log((1 - target) / 2 + EPS)
        )

        loss = torch.sum(loss)
        loss = loss / l
        if self.reduction == "mean":
            return loss / len(output)
        elif self.reduction == "sum":
            return loss


class BCELoss_tanh_focal(nn.modules.loss._Loss):
    def __init__(self, alpha, reduction="mean"):
        super(BCELoss_tanh_focal, self).__init__(reduction)
        self.alpha = alpha

    def forward(self, output, target):

        l = output.shape[1]
        EPS = 1e-12
        # target=0 gradient decrease
        loss = (
            100
            * (torch.abs(target) + self.alpha)
            * (
                -0.5
                * (
                    (1 + target) * torch.log((1 + output) / 2 + EPS)
                    + (1 - target) * torch.log((1 - output) / 2 + EPS)
                )
                + 0.5
                * (
                    (1 + target) * torch.log((1 + target) / 2 + EPS)
                    + (1 - target) * torch.log((1 - target) / 2 + EPS)
                )
            )
        )
        loss = torch.sum(loss)
        loss = loss / l
        if self.reduction == "mean":
            return loss / len(output)
        elif self.reduction == "sum":
            return loss


class BCELoss_tanh_focal_l1(nn.modules.loss._Loss):
    def __init__(self, alpha, reduction="mean"):
        super(BCELoss_tanh_focal_l1, self).__init__(reduction)
        self.alpha = alpha

    def forward(self, output, target):

        l = output.shape[1]
        EPS = 1e-12
        # target=0 gradient decrease
        loss = (
            100
            * (torch.abs(target) + self.alpha)
            * (
                -0.5
                * (
                    (1 + target) * torch.log((1 + output) / 2 + EPS)
                    + (1 - target) * torch.log((1 - output) / 2 + EPS)
                )
                + 0.5
                * (
                    (1 + target) * torch.log((1 + target) / 2 + EPS)
                    + (1 - target) * torch.log((1 - target) / 2 + EPS)
                )
            )
        )
        lamb1 = 1e1

        loss = torch.sum(loss)
        loss += lamb1 * torch.norm(target, 1)
        loss = loss / l
        if self.reduction == "mean":
            return loss / len(output)
        elif self.reduction == "sum":
            return loss
