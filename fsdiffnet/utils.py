# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:55:08 2021

@author: jcleng
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from numpy import diag
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_curve


def softshrink(x, lambd):
    mask1 = x > lambd
    mask2 = x < -lambd
    out = torch.zeros_like(x)
    out += mask1.float() * -lambd + mask1.float() * x
    out += mask2.float() * lambd + mask2.float() * x
    return out


def softsparse(x, lambd):
    mask1 = x > lambd
    mask2 = x < -lambd
    out = torch.zeros_like(x)
    out += mask1.float() * x
    out += mask2.float() * x
    return out


def softsparse_new(x, lambd):
    out_pos = torch.maximum(x - lambd, torch.tensor(0))
    out_neg = torch.maximum(-lambd - x, torch.tensor(0))
    out = out_pos + out_neg

    return out


def tanh(x):
    a = torch.exp(x) + torch.exp(-x)
    b = torch.exp(x) - torch.exp(-x)
    return b / a


# def triu_vector(M):
#     if type(M) == torch.Tensor:
#         result = M.cpu().detach().numpy()
#         result = triu_vector(result)
#         result = torch.as_tensor(result)
#     elif type(M) == np.ndarray:
#         if len(M.shape) ==2:
#             result = M[np.triu_indices(M.shape[0], k=1)]
#         elif len(M.shape) ==3:
#             result = [triu_vector(M[i]) for i in range(M.shape[0])]
#             result = np.stack(result)
#     return result


def triu_vec(M):
    """
    extract upper triangle elements to vectors.

    Parameters
    ----------
    M : TYPE
        DESCRIPTION.

    Returns
    -------
    vec : TYPE
        DESCRIPTION.

    """
    # not batch case, (p, p)
    if len(M.shape) == 2:
        if isinstance(M, np.ndarray):
            vec = M[np.triu(np.ones_like(M), k=1) == 1]
        elif isinstance(M, torch.Tensor):
            vec = M[torch.triu(torch.ones_like(M), diagonal=1) == 1]
        vec = vec[np.newaxis, :]
    # batch case, (B, p, p)
    elif len(M.shape) == 3:
        if isinstance(M, np.ndarray):
            vec = [m[np.triu(np.ones_like(m), k=1) == 1] for m in M]
            vec = np.stack(vec)
        elif isinstance(M, torch.Tensor):
            vec = [m[torch.triu(torch.ones_like(m), diagonal=1) == 1] for m in M]
            vec = torch.stack(vec)
    else:
        print("input M dim should <= 3.")
        return
    return vec


# def vec2mat(V):
#     if type(V) == torch.Tensor:
#         result = V.cpu().detach().numpy()
#         result = vec2mat(result)
#         result = torch.as_tensor(result)
#     elif type(V) == np.ndarray:
#         p=np.int(np.sqrt(V.shape[-1]))
#         if len(V.shape)  == 1:
#             result = np.reshape(V,(p,p))
#         elif len(V.shape)  == 2:
#             result = [vec2mat(V[i]) for i in range(V.shape[0])]
#             result = np.stack(result)
#     return result


def vec2mat(V):

    p = int(np.sqrt(V.shape[-1]))
    M = V.reshape(-1, p, p)
    return M


def remove_diag(theta):
    if isinstance(theta, np.ndarray):
        theta_ = theta.copy()
    elif isinstance(theta, torch.Tensor):
        theta_ = theta.clone()
        theta_ = np.array(theta_)
    np.fill_diagonal(theta_, 0)
    return theta_


def theta2partial(theta):
    sqrt_diag = np.sqrt(diag(1.0 / theta.diagonal()))
    partial = -np.dot(np.dot(sqrt_diag, theta), sqrt_diag)
    np.fill_diagonal(partial, 1)
    return partial


def split_pn(x):
    if isinstance(x, torch.Tensor):
        y_pos = x.clone().detach()
        y_neg = x.clone().detach()
    else:
        y_pos = x.copy()
        y_neg = x.copy()
    y_pos[y_pos < 0] = 0
    y_neg[y_neg > 0] = 0

    return y_pos, -y_neg


def evaluate(output, target, k=10):
    target = (target != 0).float()
    output = torch.abs(output)
    aupr = AUPR(output, target)
    auc = AUC(output, target)
    # precision, recall, f1 = topk_PR(output, target, k)
    return aupr, auc


def evaluate_new(output, target, k=10):
    target = target.sign()
    output = output.cpu().detach()
    # caluculate positive label and negtive label metrics respectively.
    target_pos, target_neg = split_pn(target)
    output_pos, output_neg = split_pn(output)
    aupr_pos = AUPR(output_pos, target_pos)
    auc_pos = AUC(output_pos, target_pos)
    aupr_neg = AUPR(output_neg, target_neg)
    auc_neg = AUC(output_neg, target_neg)
    precision_top1_pos = Topk_Precison(output_pos, target_pos, 1)
    precision_top1_neg = Topk_Precison(output_neg, target_neg, 1)
    precision_top5_pos = Topk_Precison(output_pos, target_pos, 5)
    precision_top5_neg = Topk_Precison(output_neg, target_neg, 5)
    aupr = (aupr_pos + aupr_neg) / 2
    auc = (auc_pos + auc_neg) / 2
    precision_top1 = (precision_top1_pos + precision_top1_neg) / 2
    precision_top5 = (precision_top5_pos + precision_top5_neg) / 2
    ase = ASE(output, target)
    return 100 * aupr, 100 * auc, 100 * ase, 100 * precision_top1, 100 * precision_top5


def AUPR(predict, label):
    AUPR = 0
    for i in range(predict.shape[0]):
        precision, recall, _ = precision_recall_curve(label[i], predict[i])
        if len(precision) == 2:
            AUPR += precision[0] / 2  # AUC of triangle PR curves
        else:
            AUPR += auc(recall, precision)
    return AUPR / predict.shape[0]


def AUC(predict, label):
    AUC = 0
    for i in range(predict.shape[0]):
        fpr, tpr, _ = roc_curve(label[i], predict[i])
        AUC += auc(fpr, tpr)
    return AUC / predict.shape[0]


def F1(predict, label):
    F1 = 0
    for i in range(predict.shape[0]):
        fpr, tpr, _ = f1_score(label[i], predict[i])
        F1 += auc(fpr, tpr)
    return F1 / predict.shape[0]


def ASE(predict, label):
    ASE = 0
    length = len(label[0])
    for i in range(predict.shape[0]):
        ase_ = torch.norm(label[i] - predict[i], 1) / length
        ASE += ase_.item()
    return ASE / predict.shape[0]


def Topk_Precison(predict, label, k):
    precision_topk = 0
    l = len(predict[0,])
    k = int(l * k / 100)
    for i in range(predict.shape[0]):
        predict_topk = set(np.where(keep_largest_k(predict[i,], k) != 0)[0])
        label_topk = set(np.where(keep_largest_k(label[i,], k) != 0)[0])
        real_positive = len(label_topk)
        true_positive = len(predict_topk.intersection(label_topk))
        precision_topk += true_positive / (real_positive + 1e-32)
    return precision_topk / predict.shape[0]


def show_matrix(matrix, ax=None, labels=None, title=None, figsize=(6, 5)):
    if isinstance(matrix, torch.Tensor):
        vmax = torch.max(torch.abs(matrix))
    else:
        vmax = np.max(np.abs(matrix))
    
    cmap = sns.diverging_palette(260, 10, as_cmap=True)
    if labels == None:
        labels = range(matrix.shape[-1])
    if ax == None:
        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            matrix,
            cmap=cmap,
            vmax=vmax,
            vmin=-vmax,
            square=True,
            yticklabels=labels,
            xticklabels=labels,
        )
    else:
        ax = sns.heatmap(
            matrix,
            cmap=cmap,
            vmax=vmax,
            vmin=-vmax,
            square=True,
            ax=ax,
            yticklabels=labels,
            xticklabels=labels,
        )

    if labels != None:
        ax.set_xticklabels(labels, rotation=80, fontsize=6)
        ax.set_yticklabels(labels, rotation=10, fontsize=6)
    if title != None:
        ax.set_title(title)
    return ax


class EarlyStopping:
    """
    Early stops the training if validation loss and
    validation metric doesn't improve after a given patience."""

    def __init__(
        self,
        patience=10,
        verbose=True,
        delta=0,
        experiment_name="test",
        trace_func=print,
        input_shape=None,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss_score = None
        self.best_metric_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_metric_max = -np.Inf
        self.delta = delta
        self.experiment_name = experiment_name
        self.model_path = "./models/model_" + experiment_name + ".pt"
        self.opt_path = "./optimizers/opt_" + experiment_name + ".pt"
        self.trace_func = trace_func
        self.input_shape = input_shape

    def __call__(self, val_loss, val_metric, model, optimizer):

        loss_score = -val_loss
        metric_score = val_metric
        if self.best_loss_score is None or self.best_metric_score is None:
            self.best_loss_score = loss_score
            self.best_metric_score = metric_score
            self.save_checkpoint(val_loss, val_metric, model, optimizer)
        elif (loss_score <= self.best_loss_score + self.delta) and (
            metric_score <= self.best_metric_score + self.delta
        ):
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        elif (
            loss_score > self.best_loss_score + self.delta
            and metric_score <= self.best_metric_score + self.delta
        ):
            self.best_loss_score = loss_score
            self.save_checkpoint(val_loss, val_metric, model, optimizer)
            self.counter = 0
        elif (
            loss_score <= self.best_loss_score + self.delta
            and metric_score > self.best_metric_score + self.delta
        ):
            self.best_metric_score = metric_score
            self.save_checkpoint(val_loss, val_metric, model, optimizer)
            self.counter = 0
        else:
            self.best_loss_score = loss_score
            self.best_metric_score = metric_score
            self.save_checkpoint(val_loss, val_metric, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_metric, model, optimizer):
        """Saves model when validation loss decrease."""
        if self.verbose:
            if val_loss < self.val_loss_min:
                self.trace_func(
                    f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).\
                        Saving model and optimizer ..."
                )
                self.val_loss_min = val_loss
            if val_metric > self.val_metric_max:
                self.trace_func(
                    f"Validation metric increased ({self.val_metric_max:.6f} --> {val_metric:.6f}).\
                        Saving model and optimizer ..."
                )
                self.val_metric_max = val_metric

        torch.save(model.state_dict(), self.model_path)
        # if isinstance(model, DataParallel):
        #     model_ = model.module
        # else:
        #     pass

        # torch.onnx.export(model_, torch.rand(self.input_shape).cuda(),
        #                   os.path.join(wandb.run.dir, f"{self.experiment_name}_model.onnx"),
        #                   opset_version=12)
        torch.save(model.state_dict(), self.model_path)
        # wandb.save(os.path.join(wandb.run.dir, f"{self.experiment_name}_model.pth"))

        torch.save(optimizer.state_dict(), self.opt_path)


def transfer_model(model_new, model_pretrain):
    # =============================================================================
    #     trainsfer pretrained model params to new model
    # =============================================================================
    model_dict = model_new.state_dict()
    model_pretrain_dict = model_pretrain.state_dict()
    for name, param in model_new.named_parameters():
        if (
            name in model_pretrain_dict.keys()
            and model_dict[name].size() == model_pretrain_dict[name].size()
        ):
            model_dict[name] = model_pretrain_dict[name]
            param.requires_grad = False
    model_new.load_state_dict(model_dict)
    return model_new


def validate_posdef(data):
    # =============================================================================
    #     validate if datasets are positive definate matrices
    # =============================================================================
    eig_min = np.Inf
    for i in range(len(data)):
        eig_min = np.min(np.linalg.eigvals(data[i][0]))
        if eig_min > 0:
            pass
        else:
            print("Data %d is not positive-definate!!!" % (i + 1))
    if eig_min > 0:
        print("Finish validation, all data is positive-definate~")


def isflipable(model: torch.nn.Module) -> bool:
    r"""
    Test if a model can get a opposite output of flip input with shape (N, C, H, W).
    e.g.

    x =

    tensor([[[[0.0352, 0.7290, 0.7055],

              [0.4338, 0.2678, 0.5579],

              [0.3464, 0.8159, 0.0508]],

             [[0.2807, 0.7550, 0.0613],

              [0.7510, 0.7461, 0.6666],

              [0.2304, 0.9302, 0.1280]]]])

    x.flip(1) =

    tensor([[[[0.2807, 0.7550, 0.0613],

              [0.7510, 0.7461, 0.6666],

              [0.2304, 0.9302, 0.1280]],

             [[0.0352, 0.7290, 0.7055],

              [0.4338, 0.2678, 0.5579],

              [0.3464, 0.8159, 0.0508]]]])

    If a model is flipable, it should have model(x)+model(x.flip(1))=0

    """
    x = torch.rand(10, 2, 39, 39)
    y = model(x) + model(x.flip(1))
    return (torch.sum(y) == 0).item()


def keep_largest_k(X, k):

    l = len(X.flatten())
    if isinstance(X, torch.Tensor):
        X_ = X.clone()
        X_ = X_.cpu()
    elif isinstance(X, np.ndarray):
        X_ = X.copy()
    if k == 0:
        X_[::] = 0
        return X_
    indices = np.argpartition(abs(X_), l - k - 1, axis=None)
    X_[tuple(np.array(np.unravel_index(indices, X_.shape, "C"))[:, :-k])] = 0
    return X_


def seed_everything(seed):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def calculate_flip_error(inferred_matrices):
    filp_error = 0 
    for i in range(len(inferred_matrices)):
        if i % 2 == 0:
            fe_ = np.linalg.norm(inferred_matrices[i].flatten()+inferred_matrices[i+1].flatten(), 1)
            filp_error += fe_.item()
        filp_error = filp_error/len(inferred_matrices)*2
        
    return filp_error