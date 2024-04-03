# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 10:27:21 2021

@author: jcleng
"""
import matplotlib.pyplot as plt
import torch

import wandb
from fsdiffnet.utils import *


def train_part(args, model, device, train_loader, optimizer, scheduler, loss_func, epoch, experiments_name='test', save_figs=False,):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()
        optimizer.step()

        output = output.cpu().detach()
        target = target.cpu().detach()
        output_, target_ = vec2mat(output), vec2mat(target)
        output_, target_ = triu_vec(output_), triu_vec(target_)
        aupr, auc = evaluate(
            output_, target_)
        log_name = 'train'
        wandb.log({
            log_name+"_Loss": loss.detach().item(),
            log_name+"_AUPR": aupr,
            log_name+"_AUC": auc,
            'epoch': epoch,
            'batch': batch_idx+1
        })
        if save_figs:
            if batch_idx % (len(train_loader)//5) == 0:
                output_mat = output[0].reshape_as(data[0, 0])[:39, :39]
                target_mat = target[0].reshape_as(data[0, 0])[:39, :39]
                output_mat = (output_mat + output_mat.t())/2
                # generate pictures and save
                fig, axs = plt.subplots(2, 2)
                show_matrix(data[0, 0][:39, :39].cpu().detach(), axs[0, 0])
                try:
                    show_matrix(data[0, 1][:39, :39].cpu().detach(), axs[0, 1])
                except:
                    show_matrix(torch.zeros_like(output_mat), axs[0, 1])
                show_matrix(target_mat, axs[1, 0])
                show_matrix(output_mat, axs[1, 1])
                axs[0, 0].set_title('Sigma1')
                axs[0, 1].set_title('Sigma2')
                axs[1, 0].set_title('Target')
                axs[1, 1].set_title('Output')
                fig.tight_layout()
                file_name = f'./pictures/{experiments_name}/train/train-{epoch}-{batch_idx}.png'
                plt.savefig(file_name, dpi=200)
                plt.close()

    wandb.log({
        # log_name+"_Examples": examples,
        'epoch': epoch
    })


def test_part(args, model, device, test_loader, loss_func, epoch,  experiments_name='test', save_figs=False,):
    model.eval()
    test_loss = 0
    examples = []
    auprs, aucs, precisions, recalls, f1s = 0, 0, 0, 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).detach().item()
            output = output.cpu().detach()
            target = target.cpu().detach()
            output_, target_ = vec2mat(output), vec2mat(
                target)  # vector to matrix
            output_, target_ = triu_vec(output_), triu_vec(
                target_)  # matrix to triu vector
            aupr, auc = evaluate(
                output_, target_)
            auprs += aupr
            aucs += auc

        l = len(test_loader)
        if save_figs:
            output_mat = output[0].reshape_as(data[0, 0])[:39, :39]
            target_mat = target[0].reshape_as(data[0, 0])[:39, :39]
            output_mat = (output_mat + output_mat.t())/2
            
            fig, axs = plt.subplots(2, 2)
            show_matrix(data[0, 0][:39, :39].cpu().detach(), axs[0, 0])
            try:
                show_matrix(data[0, 1][:39, :39].cpu().detach(), axs[0, 1])
            except:
                show_matrix(torch.zeros_like(output_mat), axs[0, 1])
            show_matrix(target_mat, axs[1, 0])
            show_matrix(output_mat, axs[1, 1])
            axs[0, 0].set_title('Sigma1')
            axs[0, 1].set_title('Sigma2')
            axs[1, 0].set_title('Target')
            axs[1, 1].set_title('Output')
            fig.tight_layout()
            file_name = f'./pictures/{experiments_name}/test/test-{epoch}.png'
            plt.savefig(file_name,dpi=200)
            plt.close()
        log_name = 'test'
        wandb.log({
            # log_name+"_Examples": examples,
            log_name+"_AUPR": auprs/l,
            log_name+"_AUC": aucs/l,
            log_name+"_Loss": test_loss/l,
            'epoch': epoch
        })
    return test_loss/l, auprs/l
