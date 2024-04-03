# -*- coding: utf-8 -*-
"""
Created on Sun May  9 21:06:50 2021

@author: jcleng
"""
import numpy as np
import torch
from rpy2.robjects import NULL, r
from sklearn.base import BaseEstimator
from torch import Tensor
from torch.nn import Conv2d
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

from fsdiffnet.utils import *


class lambd_pos(torch.nn.Module):
    def __init__(self, initial):
        super(lambd_pos, self).__init__()
        self.lambd = torch.nn.Parameter(
            torch.Tensor([initial]), requires_grad=True)

    def forward(self):  # no inputs
        x = torch.max(self.lambd, 0)  # get (0,1) value
        return x


class DiagConv2d(Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        diag_ratio: int = 1,
    ):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super(DiagConv2d, self).__init__(in_channels, out_channels,
                                         kernel_size_, stride_, padding_, dilation_, groups, bias, padding_mode)
        self.diag_ratio = diag_ratio
        self.conv1 = Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                            kernel_size=self.kernel_size, dilation=self.dilation, bias=bias, groups=self.groups)
        self.conv2 = Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                            kernel_size=self.kernel_size, dilation=self.dilation, bias=bias, groups=self.groups)
        assert diag_ratio <= 1 and diag_ratio >= 0, "diag_ratio must between 0 and 1!"

    def get_diag(self, x: Tensor):

        diag1, diag2 = torch.einsum(
            '...ii->...i', x), torch.einsum('...ii->...i', x)
        diag1 = torch.einsum(
            'i,...k->...ik', torch.ones(diag1.shape[-1]).to(x.device), diag1)
        diag2 = torch.einsum(
            'i,...k->...ki', torch.ones(diag2.shape[-1]).to(x.device), diag2)
        diag = diag1+diag2
        diag = diag*self.diag_ratio

        return diag

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.get_diag(x2)
        x = x1+x2
        return x



class HighDilationCirucularConv(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dilation: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 bias: bool = True):
        super(HighDilationCirucularConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias

        if self.bias:
            self.b = torch.nn.Parameter(torch.zeros(self.out_channels))

        self.W = torch.nn.Parameter(torch.zeros(
            (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))

        torch.nn.init.xavier_normal_(self.W)
        if self.bias:
            torch.nn.init.normal_(self.b)

    def forward(self, x):
        # x:(N,C_in,W,H)
        # y:(N,C_out,W,H)
        assert x.shape[1] == self.in_channels, "input X inchannels is not consistent with model!"
        y = torch.empty(
            (x.shape[0], self.out_channels, x.shape[2], x.shape[3])).to(x.device)
        for i1 in range(x.shape[0]):
            for i2 in range(self.out_channels):
                for i3 in range(x.shape[2]):
                    for i4 in range(x.shape[3]):
                        x_ = torch.empty(
                            (self.in_channels, self.kernel_size, self.kernel_size))
                        mid = self.kernel_size//2
                        for i in range(self.kernel_size):
                            for j in range(self.kernel_size):
                                for k in range(self.in_channels):

                                    x_[k, i, j] = x[i1, k, (i3+(i-mid)*self.dilation + x.shape[2]*self.dilation) % x.shape[2], (i4+(
                                        j-mid)*self.dilation + x.shape[3]*self.dilation) % x.shape[3]]
                        x_ = x_.to(x.device)
                        if self.bias:

                            y[i1, i2, i3, i4] = torch.sum(
                                self.W[i2, ] * x_) + self.b[i2]
                        else:
                            # print(y.device)
                            # print(x.device)
                            # print(x_.device)
                            # print(self.W.device)
                            y[i1, i2, i3, i4] = torch.sum(self.W[i2, ] * x_)

        return y


class DilationCirucularConv_test(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dilation: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 bias: bool = True):
        super(DilationCirucularConv_test, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias

        self.conv = Conv2d(in_channels, out_channels,
                           kernel_size, dilation=dilation, bias=bias)

    def forward(self, x):
        x = torch.nn.functional.pad(x, mode='circular', pad=(
            self.dilation, self.dilation, self.dilation, self.dilation))
        x = self.conv(x)
        return x


class Random(object):
    def __init__(self):
        pass

    def fit(self, X):
        p = X.shape[-1]
        self.pred = torch.rand(p, p)-0.5
        return self.pred


class Pinv(object):
    def __init__(self, mode='diff'):
        self.mode = mode

    def fit(self, X):
        if self.mode == 'diff':
            x0 = np.linalg.pinv(X[0], hermitian=True)
            x1 = np.linalg.pinv(X[1], hermitian=True)
            self.pred = -theta2partial(x0) + theta2partial(x1)
        elif self.model == 'single':
            self.pred = np.linalg.pinv(X)
        return self.pred


class NNGridSearch(BaseEstimator):
    def __init__(self, model: torch.nn.Module, alphas=10, n_refinement=4):
        self.model = model
        self.alphas = alphas
        self.n_refinement = n_refinement

    def fit(self, X: torch.Tensor):
        self.p = X.shape[-1]
        assert isinstance(
            X, torch.Tensor
        ), "X should be a torch.Tensor with shape (1, 2, p, p)."
        result = self.model(
            X.to(
                self.model.state_dict()[list(
                    self.model.state_dict().keys())[0]].device
            )
        ).detach()
        self.delta = result.reshape(self.p, self.p)
        return self

    def score(self, X, alpha):
        X1 = X[0, 0]
        X2 = X[0, 1]
        k = int((len(X1.flatten()) - self.p) * alpha)
        if k % 2 == 1:
            k += 1
        delta_ = keep_largest_k(self.delta.detach(), k)
        delta_ = delta_
        likelihood_ = 0.5 * torch.trace(delta_.mm(X2).mm(delta_).mm(X1)) - torch.trace(
            delta_.mm(X2 - X1)
        )
        return likelihood_

    def grid_search(self, X):
        grid_len = 1
        best_alpha = 0.5
        best_score = np.inf
        for i in range(self.n_refinement):
            self.alpha_grid = np.linspace(
                max(best_alpha - grid_len / 2, 0),
                min(best_alpha + grid_len / 2, 1),
                self.alphas,
            )
            for alpha_ in self.alpha_grid:
                score_ = self.score(X, alpha_)
                if score_ < best_score:
                    best_alpha = alpha_
                    best_score = score_

            grid_len /= 2
        self.best_alpha = best_alpha
        self.best_score = best_score
        self.best_k = int((len(X[0, 1].flatten()) - self.p) * self.best_alpha)
        if self.best_k % 2 == 1:
            self.best_k += 1
        self.best_delta = keep_largest_k(self.delta, self.best_k)
        return self


class Permut_NN(torch.nn.Module):
    def __init__(self, model, permut_time):
        super(Permut_NN, self).__init__()
        self.model = model()
        self.permut_time = permut_time

    def forward(self, x):
        assert len(x.shape) == 4, "x must with shape of (1, 2, p, p)"
        assert x.shape[0] == 1, "x must with shape of (1, 2, p, p)"
        permut_y = torch.zeros_like(x[0, 0])
        self.model = self.model.to(x.device)
        for i in range(self.permut_time):
            I = np.arange(x.shape[-1])
            np.random.shuffle(I)
            permut_x = torch.zeros_like(x)
            permut_x[0, 0] = x[0, 0][I, :]
            permut_x[0, 0] = permut_x[0, 0][:, I]
            permut_x[0, 1] = x[0, 1][I, :]
            permut_x[0, 1] = permut_x[0, 1][:, I]
            permut_x = permut_x.to(x.device)
            permut_y_ = self.model(permut_x).reshape(permut_y.shape)
            # depermut the order of the output.
            inv_I = [0] * x.shape[-1]
            for i in range(x.shape[-1]):
                # i=0,j=aa[i]=1 : j->i want i->j aa[j]=i
                inv_I[I[i]] = i
            permut_y_ = permut_y_[inv_I, :]
            permut_y_ = permut_y_[:, inv_I]

            permut_y += permut_y_

        permut_y /= self.permut_time
        permut_y = permut_y.view(1, -1)
        return permut_y


class Permut():
    def __init__(self, model, permut_time):
        self.model = model()
        self.permut_time = permut_time

    def fit(self, x):
        permut_y = torch.zeros_like(x[0])
        for i in range(self.permut_time):
            I = np.arange(x.shape[-1])
            np.random.shuffle(I)
            permut_x = torch.zeros_like(x)
            permut_x[0] = x[0][I, :]
            permut_x[0] = permut_x[0][:, I]
            permut_x[1] = x[1][I, :]
            permut_x[1] = permut_x[1][:, I]
            permut_x = permut_x
            pred_ = self.model.fit(permut_x)
            permut_y_ = pred_.reshape(permut_y.shape)
            # depermut the order of the output.
            inv_I = [0] * x.shape[-1]
            for i in range(x.shape[-1]):
                # i=0,j=aa[i]=1 : j->i want i->j aa[j]=i
                inv_I[I[i]] = i
            permut_y_ = permut_y_[inv_I, :]
            permut_y_ = permut_y_[:, inv_I]

            permut_y += permut_y_

        permut_y /= self.permut_time

        permut_y = permut_y.view(1, -1)
        return permut_y


class NetDiff(object):
    def __init__(self):
        import os

        import rpy2

        os.environ["R_HOME"] = "/usr/lib/R/"
        import rpy2.robjects.numpy2ri
        from rpy2.robjects.packages import importr

        rpy2.robjects.numpy2ri.activate()
        self.NetDiff = importr("NetDiff")
        pass

    def fit(self, X, n1=None):
        # X shape: (n1+n2, p)
        from sklearn import preprocessing
        X = preprocessing.scale(X)
        n = X.shape[0]
        if n1 is None:
            n1 = n // 2
        partition = ["state1"] * n1 + ["state2"] * (n-n1)
        partition = np.array(partition)
        results = self.NetDiff.netDiff(X, partition, parallel=True, Bayes_factor=1.05)
        self.theta1 = results[0][0]
        self.theta2 = results[0][1]
        self.delta = self.theta2 - self.theta1
        self.results = results
        return self.delta


class BDGraph(object):
    def __init__(self,cores=None,jump=None):
        import os

        import rpy2
        
        if cores is not None:
            self.cores = cores
        else:
            self.cores = 1
            
        if jump is not None:
            self.jump = jump
        else:
            self.jump = NULL
        os.environ["R_HOME"] = "/usr/lib/R/"
        import rpy2.robjects.numpy2ri
        from rpy2.robjects.packages import importr

        rpy2.robjects.numpy2ri.activate()
        self.BDgraph = importr("BDgraph")
        pass

    def fit(self, X):
        # X shape: (2, n, p)
        n = X[0].shape[0]
        p = X[0].shape[1]
        X1 = X[0]
        X2 = X[1]
        results1 = self.BDgraph.bdgraph(X1, method="gcgm", iter=1000, cores=self.cores, jump=self.jump)
        results2 = self.BDgraph.bdgraph(X2, method="gcgm", iter=1000, cores=self.cores, jump=self.jump)

        self.theta1 = results1[1]
        self.theta2 = results2[1] 
        self.delta = self.theta1 - self.theta2
        return self.delta
