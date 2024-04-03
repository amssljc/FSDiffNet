# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:14:41 2021

@author: jcleng
"""
import heapq
import itertools
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rpy2
import rpy2.robjects.numpy2ri
import scipy.sparse as sparse
import scipy.stats as stats
import torch
from joblib import Parallel, delayed
from numpy import cov, diag
from numpy.linalg import eig, eigvalsh, inv
from numpy.random import choice, rand, randint, random_sample, uniform
from progressbar import *
from rpy2.robjects.packages import importr
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from sklearn.datasets import make_sparse_spd_matrix
from torch.utils.data import DataLoader, Dataset

from fsdiffnet.utils import *

os.environ["R_HOME"] = "D:/Softwares/R-4.1.3/"

rpy2.robjects.numpy2ri.activate()

# =============================================================================
# data generator
# =============================================================================


def generate_theta1(p=50, net_rand_mode="ER", sparsity=0.1):
    if p > 2e3:
        print("input p is too large! please input p < 2000!")
        return
    if net_rand_mode == "ER":
        G = nx.erdos_renyi_graph(p, sparsity)
    elif net_rand_mode == "BA":
        G = nx.barabasi_albert_graph(p, int(np.ceil(p * sparsity * 0.5)))
    else:
        print("please input one the correct net_rand_mode: 'ER' or 'BA'!")
    theta1_A = nx.to_numpy_array(G)
    I = np.eye(p)
    np.random.shuffle(I)
    theta1_B = I.dot(theta1_A).dot(I.T)
    fill = np.random.uniform(-1, 1, theta1_B.shape)
    fill = (fill + fill.T) / 2
    theta1 = np.multiply(theta1_B, fill)

    return theta1


def generate_theta2(theta1, diff_mode="random", diff_ratio=0.5):
    # import R package BDgraph

    p = theta1.shape[0]
    diff_num = np.count_nonzero(theta1) * diff_ratio / 2
    sparsity = diff_num/(p**2-p)*2
    theta2 = np.triu(theta1.copy())
    if diff_mode == "random":
        ind_triu = np.triu_indices_from(theta1, 1)
        ind_change = choice(
            range(len(ind_triu[0])), size=int(diff_num), replace=False)
        ind_change_i = ind_triu[0][ind_change]
        ind_change_j = ind_triu[1][ind_change]
        for i in range(ind_change_i.size):
            theta2[ind_change_i[i], ind_change_j[i]] = uniform(-1, 1)
        theta2 = (theta2 + theta2.T) / 2

    elif diff_mode == "hub":
        hub_n = int(np.ceil(p / 10))
        degrees = np.sum(np.int32(theta1 != 0), axis=0) // 2
        hub_ind = heapq.nlargest(hub_n, range(len(degrees)), degrees.take)
        # hub_ind_tuple = [(i,i) for i in hub_ind]
        ind_change_i = []
        ind_change_j = []
        for i, hub in enumerate(hub_ind):
            _ind_change = choice(
                np.delete(np.arange(0, p), hub),
                size=int(np.ceil(diff_num / hub_n)),
                replace=False,
            )
            ind_change_i += [hub] * int(np.ceil(diff_num / hub_n))
            ind_change_j += list(_ind_change)
        for i in range(len(ind_change_i)):
            theta2[ind_change_i[i], ind_change_j[i]] = uniform(-1, 1)
        theta2 = (theta2 + theta2.T) / 2
        # TODO need to be completed
    elif diff_mode == "cluster":
        G = nx.random_partition_graph(
            [p // 4, p // 4, p // 4, p - 3 * (p // 4)],
            5*sparsity, 0
        )
        theta2 = nx.to_numpy_array(G)
        theta2 = theta2 * uniform(-1, 1, theta2.shape)
        theta2 = theta2 / 2 + theta2.T / 2
        I_ = np.arange(p)
        np.random.shuffle(I_)
        theta2_ = theta2.copy()
        theta2_ = theta2_[:, I_]
        theta2_ = theta2_[I_, :]
        theta2 = theta1 + theta2_
        theta2 = theta1 + theta2
        theta2 = np.maximum(theta2, -1)
        theta2 = np.minimum(theta2, 1)

    elif diff_mode == "scale-free":
        G = nx.barabasi_albert_graph(p, int(np.ceil(p * sparsity * 0.5)))
        theta2 = nx.to_numpy_array(G)
        theta2 = theta2 * uniform(-1, 1, theta2.shape)
        theta2 = theta2 / 2 + theta2.T / 2
        I_ = np.arange(p)
        np.random.shuffle(I_)
        theta2_ = theta2.copy()
        theta2_ = theta2_[:, I_]
        theta2_ = theta2_[I_, :]
        theta2 = theta1 + theta2_
        theta2 = np.maximum(theta2, -1)
        theta2 = np.minimum(theta2, 1)

    elif diff_mode == "None":
        theta2 = np.eye(p)
    return theta2


def generate_theta(theta1, theta2, eps=0.1):
    p = theta1.shape[0]
    # main time cost item
    theta1_ = sparse.csr_matrix(theta1)
    theta2_ = sparse.csr_matrix(theta2)
    eig_min = min(
        eigsh(theta1_, 1, which="SA", return_eigenvectors=(False), ncv=200).item(),
        eigsh(theta2_, 1, which="SA", return_eigenvectors=(False), ncv=200).item(),
    )
    I = (np.abs(eig_min) + eps) * np.eye(p, p)
    theta1, theta2 = theta1 + I, theta2 + I
    theta1, theta2 = theta2partial(theta1), theta2partial(theta2)
    return theta1, theta2


def generate_diff(theta1, theta2):

    delta = theta1 - theta2
    label = 1 * (delta != 0)
    label.dtype = int
    return delta, label


class ExpressionProfiles(Dataset):
    def __init__(self, p=[39], n=78, sample_n=10000, repeats=10, sparsity=[0.05, 0.1], diff_ratio=[0.3, 0.5], net_rand_mode='ER', diff_mode='none', target_type='abs', usage='training', flip=False, withdiag=False, sigma_diag=True):
        self.p = p
        self.n = n
        if diff_mode != 'none':
            self.diff_repeats = repeats
        else:
            self.diff_repeats = 1
        if diff_mode == 'none':
            ratio = repeats
        else:
            ratio = repeats**2
        self.sample_n = int(sample_n/ratio)
        self.sparsity = sparsity
        self.net_rand_mode = net_rand_mode
        self.diff_mode = diff_mode
        self.diff_ratio = diff_ratio
        self.repeats = repeats
        self.target_type = target_type
        self.usage = usage
        self.flip = flip
        self.withdiag = withdiag
        self.sigma_diag = sigma_diag
        self.data = self.generate_samples()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def generate_samples(self):
        data = []
        widgets = ['Generating datasets: ', Percentage(), ' ', Bar('#'), ' ',
                   ' ', ETA(), ' ']
        progress = ProgressBar(widgets=widgets)
        for i in progress(range(self.sample_n)):

            for i in range(self.repeats):
                p_tmp = np.random.choice(self.p, 1).item()
                sparsity_tmp = np.random.uniform(
                    min(self.sparsity), max(self.sparsity), 1).item()
                diff_ratio_tmp = np.random.uniform(
                    min(self.diff_ratio), max(self.diff_ratio), 1).item()

                theta1_ori = generate_theta1(
                    p_tmp, self.net_rand_mode, sparsity_tmp)
                for i in range(self.diff_repeats):
                    # 转为偏相关系数矩阵可能会导致特征值小于0
                    eig_min = 0
                    while eig_min <= 0:
                        theta1_ori = generate_theta1(
                            p_tmp, self.net_rand_mode, sparsity_tmp
                        )
                        theta2 = generate_theta2(
                            theta1_ori, self.diff_mode, diff_ratio_tmp
                        )
                        theta1, theta2 = generate_theta(theta1_ori, theta2)
                        theta1_ = sparse.csr_matrix(theta1)
                        theta2_ = sparse.csr_matrix(theta2)
                        eig_min = min(
                            eigsh(
                                theta1_,
                                1,
                                which="SA",
                                return_eigenvectors=(False),
                                ncv=200,
                            ).item(),
                            eigsh(
                                theta2_,
                                1,
                                which="SA",
                                return_eigenvectors=(False),
                                ncv=200,
                            ).item(),
                        )

                    delta, label = generate_diff(theta1, theta2)
                    cov1, cov2 = inv(theta1), inv(theta2)
                    # 是否去除target中的对角线
                    if not self.withdiag:
                        theta1, theta2 = remove_diag(
                            theta1), remove_diag(theta2)

                    X1, X2 = np.random.multivariate_normal(
                        np.zeros(p_tmp), cov1, size=self.n), np.random.multivariate_normal(np.zeros(p_tmp),  cov2, size=self.n)
                    sigma1, sigma2 = np.corrcoef(X1.T), np.corrcoef(X2.T)
                    if not self.sigma_diag:
                        sigma1, sigma2 = remove_diag(
                            sigma1), remove_diag(sigma2)

                    sigma1, sigma2, theta1, theta2, delta, label = torch.as_tensor(sigma1), torch.as_tensor(
                        sigma2), torch.as_tensor(theta1), torch.as_tensor(theta2), torch.as_tensor(delta), torch.as_tensor(label)
                    if self.diff_mode != 'none':

                        label = torch.flatten(label, start_dim=0)
                        label = label.float()
                        delta = torch.flatten(delta, start_dim=0)
                        delta = delta.float()

                        x = torch.stack((sigma1, sigma2), dim=0)
                        x = x.float()
                        if self.target_type == 'float':
                            target = delta
                        elif self.target_type == 'abs':
                            target = torch.abs(delta)
                        elif self.target_type == 'int':
                            target = label
                        else:
                            print('wrong target type!')

                        if self.usage == 'training':
                            data.append((x, target))
                        elif self.usage == 'comparison':
                            data.append((x, target, X1, X2))
                        else:
                            pass

                        if self.flip:
                            x = x.flip(0)
                            target = -target
                            if self.target_type == 'float':
                                pass
                            elif self.target_type == 'abs':
                                target = torch.abs(target)
                            elif self.target_type == 'int':
                                pass

                            if self.usage == 'training':
                                data.append((x, target))
                            elif self.usage == 'comparison':
                                data.append((x, target, X2, X1))
                            else:
                                pass
                    else:
                        theta1_label = torch.flatten(theta1, start_dim=0)
                        theta1_label = theta1_label.float()
                        sigma1 = sigma1.reshape(-1, p_tmp, p_tmp)
                        sigma1 = sigma1.float()
                        if self.target_type == 'float':
                            target = theta1_label
                        elif self.target_type == 'abs':
                            target = theta1_label
                            target = torch.abs(target)
                        elif self.target_type == 'int':
                            theta1_label = (theta1_label != 0).float()
                            target = theta1_label

                        if self.usage == 'training':
                            data.append((sigma1, target))
                        elif self.usage == 'comparison':
                            data.append((sigma1, target, X1))
                        else:
                            pass

        return data


class ExpressionProfilesParallel(Dataset):
    def __init__(
        self,
        p=39,
        n=78,
        sample_n=10000,
        repeats=10,
        parallel_loops=200,
        sparsity=[0.1, 0.1],
        diff_ratio=[0.3, 0.3],
        net_rand_mode="BA",
        diff_mode="mix",
        distribution="Gaussian",
        target_type="float",
        usage="training",
        flip=False,
        withdiag=False,
        sigma_diag=True,
        seed=1
    ):
        self.p = p
        self.n = n
        if diff_mode != "None":
            self.diff_repeats = repeats
        else:
            self.diff_repeats = 1
        if diff_mode == "None":
            ratio = repeats
        else:
            ratio = repeats ** 2
        if flip:
            ratio *= 2
        self.parallel_loops = parallel_loops

        self.sample_n = int(sample_n / ratio / self.parallel_loops)
        self.sparsity = sparsity
        self.net_rand_mode = net_rand_mode
        self.diff_mode = diff_mode
        self.diff_ratio = diff_ratio
        self.repeats = repeats
        self.distribution = distribution
        self.target_type = target_type
        self.usage = usage
        self.flip = flip
        self.withdiag = withdiag
        self.sigma_diag = sigma_diag
        self.bdg = importr("BDgraph")
        self.seed = seed

        self.results = Parallel(n_jobs=self.parallel_loops)(
            delayed(self.generate_samples)() for i in range(self.parallel_loops)
        )
        self.data = list(itertools.chain(*self.results))
        # self.data = self.generate_samples()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def generate_samples(self):
        data = []
        widgets = [
            "Generating datasets: ",
            Percentage(),
            " ",
            Bar("#"),
            " ",
            " ",
            ETA(),
            " ",
        ]
        progress = ProgressBar(widgets=widgets)
        diff_modes = ["random", "hub", "scale-free", "cluster"]
        for i in progress(range(self.sample_n)):

            for i in range(self.repeats):
                if self.diff_mode == 'mix':
                    diff_mode_i = np.random.randint(0, 3)
                    diff_mode = diff_modes[diff_mode_i]
                else:
                    diff_mode = self.diff_mode
                sparsity_tmp = np.random.uniform(
                    min(self.sparsity), max(self.sparsity), 1
                ).item()
                diff_ratio_tmp = np.random.uniform(
                    min(self.diff_ratio), max(self.diff_ratio), 1
                ).item()

                theta1_ori = generate_theta1(
                    self.p, self.net_rand_mode, sparsity_tmp)
                for i in range(self.diff_repeats):
                    # 转为偏相关系数矩阵可能会导致特征值小于0
                    eig_min = 0
                    while eig_min <= 0:
                        theta1_ori = generate_theta1(
                            self.p, self.net_rand_mode, sparsity_tmp
                        )
                        theta2 = generate_theta2(
                            theta1_ori, diff_mode, diff_ratio_tmp
                        )
                        theta1, theta2 = generate_theta(theta1_ori, theta2)
                        theta1_ = sparse.csr_matrix(theta1)
                        theta2_ = sparse.csr_matrix(theta2)
                        eig_min = min(
                            eigsh(
                                theta1_,
                                1,
                                which="SA",
                                return_eigenvectors=(False),
                                ncv=200,
                            ).item(),
                            eigsh(
                                theta2_,
                                1,
                                which="SA",
                                return_eigenvectors=(False),
                                ncv=200,
                            ).item(),
                        )
                    # theta2 = generate_theta2(
                    #     theta1_ori, self.diff_mode, diff_ratio_tmp)
                    # theta1, theta2 = generate_theta(theta1_ori, theta2)
                    delta, label = generate_diff(theta1, theta2)
                    cov1, cov2 = inv(theta1), inv(theta2)
                    # 是否去除target中的对角线
                    if not self.withdiag:
                        theta1, theta2 = remove_diag(
                            theta1), remove_diag(theta2)

                    if self.distribution == "Gaussian":
                        X1, X2 = (
                            np.random.multivariate_normal(
                                np.zeros(self.p), cov1, size=self.n
                            ),
                            np.random.multivariate_normal(
                                np.zeros(self.p), cov2, size=self.n
                            ),
                        )
                    elif self.distribution == "Exponential":
                        X1 = self.bdg.bdgraph_sim(
                            self.p, "fixed", self.n, "non-Gaussian", K=theta1
                        )
                        X2 = self.bdg.bdgraph_sim(
                            self.p, "fixed", self.n, "non-Gaussian", K=theta2
                        )
                        X1 = X1[2]
                        X2 = X2[2]
                    elif self.distribution == "categorical":
                        X1 = self.bdg.bdgraph_sim(
                            self.p, "fixed", self.n, "categorical", K=theta1
                        )
                        X2 = self.bdg.bdgraph_sim(
                            self.p, "fixed", self.n, "categorical", K=theta2
                        )
                        X1 = X1[2]
                        X2 = X2[2]
                    elif self.distribution == "binary":
                        X1 = self.bdg.bdgraph_sim(
                            self.p, "fixed", self.n, "binary", K=theta1
                        )
                        X2 = self.bdg.bdgraph_sim(
                            self.p, "fixed", self.n, "binary", K=theta2
                        )
                        X1 = X1[2]
                        X2 = X2[2]
                    elif self.distribution == 'mixed':
                        X1 = self.bdg.bdgraph_sim(
                            self.p, "fixed", self.n, "mixed", K=theta1
                        )
                        X2 = self.bdg.bdgraph_sim(
                            self.p, "fixed", self.n, "mixed", K=theta2
                        )
                        X1 = X1[2]
                        X2 = X2[2]
                    sigma1, sigma2 = np.corrcoef(X1.T), np.corrcoef(X2.T)
                    if not self.sigma_diag:
                        sigma1, sigma2 = remove_diag(
                            sigma1), remove_diag(sigma2)

                    sigma1, sigma2, theta1, theta2, delta, label = (
                        torch.as_tensor(sigma1),
                        torch.as_tensor(sigma2),
                        torch.as_tensor(theta1),
                        torch.as_tensor(theta2),
                        torch.as_tensor(delta),
                        torch.as_tensor(label),
                    )
                    if diff_mode != "None":

                        label = torch.flatten(label, start_dim=0)
                        label = label.float()
                        delta = torch.flatten(delta, start_dim=0)
                        delta = delta.float()

                        x = torch.stack((sigma1, sigma2), dim=0)
                        x = x.float()
                        if self.target_type == "float":
                            target = delta
                        elif self.target_type == "abs":
                            target = torch.abs(delta)
                        elif self.target_type == "int":
                            target = label
                        else:
                            print("wrong target type!")

                        if self.usage == "training":
                            data.append((x, target))
                        elif self.usage == "comparison":
                            data.append((x, target, X1, X2))
                        else:
                            pass

                        if self.flip:
                            x = x.flip(0)
                            target = -target
                            if self.target_type == "float":
                                pass
                            elif self.target_type == "abs":
                                target = torch.abs(target)
                            elif self.target_type == "int":
                                pass

                            if self.usage == "training":
                                data.append((x, target))
                            elif self.usage == "comparison":
                                data.append((x, target, X2, X1))
                            else:
                                pass
                    else:
                        theta1_label = torch.flatten(theta1, start_dim=0)
                        theta1_label = theta1_label.float()
                        sigma1 = sigma1.reshape(-1, self.p, self.p)
                        sigma1 = sigma1.float()
                        if self.target_type == "float":
                            target = theta1_label
                        elif self.target_type == "abs":
                            target = theta1_label
                            target = torch.abs(target)
                        elif self.target_type == "int":
                            theta1_label = (theta1_label != 0).float()
                            target = theta1_label

                        if self.usage == "training":
                            data.append((sigma1, target))
                        elif self.usage == "comparison":
                            data.append((sigma1, target, X1))
                        else:
                            pass

        return data
