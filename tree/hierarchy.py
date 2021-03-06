#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by zhuoer.rex on 2020/12/22 12:42 下午
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from itertools import chain
from torch import nn
from torch.optim import Adam

from .base import ScorerMixin
from utils import gumbel_softmax, ohe_from_logits


LOSS_MAP = {
    "mse": F.mse_loss,
    "ce": F.cross_entropy
}


class MixNode(nn.Module):
    """

    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 depth,
                 loss,
                 max_depth=5):
        super(MixNode, self).__init__()

        # output of current node
        self.loss = loss
        self.leaf = nn.Parameter(torch.randn(1, output_dim), requires_grad=False)

        if depth < max_depth:
            self.is_leaf = False

            # whether pruned
            self.gamma = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=True)
            # choose which feature
            # self.omega = nn.Parameter(torch.rand(input_dim), requires_grad=True)
            # each threshold of feature
            # self.bias = nn.Parameter(torch.randn(input_dim), requires_grad=True)
            # route each instance
            self.linear = nn.Linear(input_dim, 2, bias=True)

            self.left_child = MixNode(input_dim, output_dim, depth + 1, loss, max_depth)
            self.right_child = MixNode(input_dim, output_dim, depth + 1, loss, max_depth)
        else:
            self.is_leaf = True
            self.left_child = None
            self.right_child = None

    def gamma_parameters(self):
        if self.is_leaf:
            yield from []
        else:
            yield from chain(
                [self.gamma],
                self.left_child.gamma_parameters(),
                self.right_child.gamma_parameters()
            )

    def omega_parameters(self):
        if self.is_leaf:
            yield from []
        else:
            yield from chain(
                [self.omega],
                self.left_child.omega_parameters(),
                self.right_child.omega_parameters()
            )
    
    def arch_parameters(self):
        yield from self.gamma_parameters()
        # yield from self.omega_parameters()

    def tree_parameters(self):
        yield from (
            param for name, param in self.named_parameters()
            if not (name.endswith('gamma'))
        )

    def route(self, x, hard=True):
        x = x.float()
        logits = self.linear(x)
        if hard and self.training:
            return gumbel_softmax(logits)
        elif hard and not self.training:
            return ohe_from_logits(logits)
        else:
            return logits.softmax(dim=-1)
        # return self.linear((x - self.bias) * self.omega).sigmoid()

    def forward(self, x):
        """
        \hat{y} = (1 - \gamma) * leaf + \gamma [I(x - b) * w * left + I(b - x) * w * right]
        :param x:
        :return: y FloatTensor (batch_size, output_dim)
        """
        batch_size = x.size()[0]
        y = self.leaf.repeat(batch_size, 1)
        if self.is_leaf:
            return y

        gamma = self.gamma.softmax(dim=-1)

        left = self.left_child(x)
        right = self.right_child(x)

        router = self.route(x)

        children = torch.stack([left, right], dim=-1)
        children = torch.bmm(children,  router.unsqueeze(-1)).squeeze(-1)
        y = torch.stack([children, y], dim=-1).matmul(gamma)

        return y

    def set_leaves(self, x, y, parent_leaf=None):
        if len(y) > 0:
            self.leaf.copy_(torch.tensor(np.mean(y, axis=0)))
        elif len(y) == 0 and parent_leaf is not None:
            self.leaf.copy_(parent_leaf)
        else:
            self.leaf.zero_()

        if self.is_leaf:
            return

        if len(y) > 0:
            router = np.argmax(self.route(x, hard=False).detach().cpu().numpy(), axis=-1)
            left_idxes = router == 0
            right_idxes = router == 1

            self.left_child.set_leaves(x[left_idxes], y[left_idxes], self.leaf)
            self.right_child.set_leaves(x[right_idxes], y[right_idxes], self.leaf)
        else:
            self.left_child.set_leaves(x, y, self.leaf)
            self.right_child.set_leaves(x, y, self.leaf)


class HierarchicalTree(pl.LightningModule, ScorerMixin):
    def __init__(self,
                 input_dim,
                 output_dim,
                 loss="mse",
                 max_depth=5,
                 lr=0.01
                 ):
        super(HierarchicalTree, self).__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.loss = loss
        self.lr = lr

        self.root = MixNode(input_dim, output_dim, 0,  loss, max_depth)
        if isinstance(loss, str):
            self.loss_func = LOSS_MAP[loss]
        else:
            self.loss_func = loss

    def configure_optimizers(
            self,
    ):
        adam = Adam(
            self.root.tree_parameters(),
            lr=self.lr
        )
        return adam

    def forward(self, x):
        return self.root(x)

    def set_leaves(self, x, y):
        if self.loss == "ce" and (len(y.shape) == 1 or y.shape[1] == 1):
            y = np.eye(self.output_dim)[y]
        return self.root.set_leaves(x, y)

    def sharing_step(self, batch, batch_idx):
        x, y = batch
        if self.output_dim > 1:
            y = y.squeeze()
        else:
            y = y.float()
        y_hat = self.root(x)
        loss = self.loss_func(y_hat, y).mean()
        # loss_item = loss.detach().cpu().item()
        # import math
        # if math.isnan(loss_item):
        #     print('here')
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.sharing_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.sharing_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--loss', type=str, default="mse")
        parser.add_argument('--max_depth', type=int, default=6)
        parser.add_argument('--patience', type=int, default=10)
        return parser
