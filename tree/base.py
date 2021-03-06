#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by zhuoer.rex on 2020/12/22 9:10 下午
import os
import torch
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import r2_score, accuracy_score


class ScorerMixin:
    def score(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=self.device)
        y_hat = self(x).detach().cpu().numpy()
        if self.loss == "mse":
            print("mse: ", np.mean((y_hat - y) ** 2))
            return r2_score(y, y_hat, multioutput='variance_weighted')
        else:
            y_hat = np.argmax(y_hat, axis=-1)
            return accuracy_score(y, y_hat)


class LeafOptimizer(pl.callbacks.Callback):
    def __init__(self, train_set, val_set=None):
        super(LeafOptimizer, self).__init__()
        x, y = train_set
        if val_set is not None:
            x_val, y_val = val_set
            x = np.concatenate([x, x_val], axis=0)
            y = np.concatenate([y, y_val], axis=0)
        self.x = torch.tensor(x)
        self.y = y

    def on_train_batch_end(self, trainer, tree, outputs, batch, batch_idx, dataloader_idx):
        self.x = self.x.to(tree.device)
        tree.eval()
        tree.set_leaves(self.x, self.y)
        tree.train()


class ArchOptimizer(pl.callbacks.Callback):
    def __init__(self, dataloader, epsilon=0.0, l1=0.01):
        super(ArchOptimizer, self).__init__()

        # self.dataloader = dataloader
        self.dataloader = ArchOptimizer.infinite_dataloader(dataloader)
        self.epsilon = epsilon
        self.l1 = l1
        self.optimizer = None
        self.zero_optimizer = None
        self.params_store = []

    @classmethod
    def infinite_dataloader(cls, dataloader):
        i = 0
        while True:
            for batch_data in dataloader:
                yield i, batch_data

    def on_fit_start(self, trainer, tree):
        for param in tree.root.arch_parameters():
            self.params_store.append(param.data.detach().clone())

        self.optimizer = torch.optim.Adam(
            tree.root.arch_parameters(),
            lr=0.01
        )
        self.zero_optimizer = torch.optim.SGD(
            tree.root.tree_parameters(),
            lr=0.01
        )

    def on_fit_end(self, trainer, pl_module):
        self.binarization(pl_module.root, self.epsilon)
        print("Final arch:\n", list(pl_module.root.arch_parameters()))

    @torch.no_grad()
    def store(self, tree):
        for index, param in enumerate(tree.arch_parameters()):
            self.params_store[index].copy_(param.data)

    @torch.no_grad()
    def restore(self, tree):
        for index, param in enumerate(tree.arch_parameters()):
            param.data.copy_(self.params_store[index])

    @torch.no_grad()
    def binarization(self, tree, epsilon):
        self.store(tree)

        # TODO: 这个步骤需要推导一下, 是否一致
        for param in tree.arch_parameters():
            if len(param.size()) == 1 and param.size()[0] == 1:
                param.data.copy_((param >= 0.5).float())
            else:
                if len(param.size()) == 1:
                    m, n = 1, param.size()[0]
                else:
                    m, n = param.size()
                if np.random.rand() <= epsilon:
                    choice_index = np.random.choice(range(n), m)
                else:
                    choice_index = param.detach().cpu().numpy().argmax(axis=-1)
                self.proximal_step(param, choice_index, n)

    @torch.no_grad()
    def proximal_step(self, param, choice_index, n):
        bin_param = torch.eye(n, dtype=torch.float, device=param.device)[choice_index] * 1e5
        param.data.copy_(bin_param)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.optimizer.zero_grad()
        self.binarization(pl_module.root, self.epsilon)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.optimizer.zero_grad()
        _, (x, y) = next(self.dataloader)
        x, y = x.to(pl_module.device), y.to(pl_module.device)
        loss = pl_module.sharing_step((x, y), batch_idx)\
               + self.l1 * torch.cat(list(pl_module.root.arch_parameters()), dim=0).sum()
        loss.backward()

        self.restore(pl_module.root)
        self.optimizer.step()

    # def on_train_epoch_start(self, trainer, pl_module):
        # tensorboard = pl_module.logger.experiment
        # tensorboard.add_histogram("gamma_bin", torch.cat(list(pl_module.root.gamma_parameters()), dim=0))

    # def on_train_epoch_end(self, trainer, pl_module, outputs):
    #     for batch_idx, (x, y) in enumerate(self.dataloader):
    #         self.optimizer.zero_grad()
    #         self.binarization(pl_module.root, self.epsilon)
    #
    #         x, y = x.to(pl_module.device), y.to(pl_module.device)
    #         loss = pl_module.sharing_step((x, y), batch_idx)\
    #                + self.l1 * torch.cat(list(pl_module.root.arch_parameters()), dim=0).sum()
    #         loss.backward()
    #
    #         tensorboard = pl_module.logger.experiment
    #         tensorboard.add_histogram(
    #             "gamma_grad",
    #             torch.cat([param.grad for param in pl_module.root.gamma_parameters() if param.grad is not None], dim=0)
    #         )
    #
    #         self.restore(pl_module.root)
    #         self.optimizer.step()
