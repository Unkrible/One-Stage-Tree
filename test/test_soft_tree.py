#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by zhuoer.rex on 2020/12/16 8:48 下午

import torch
import numpy as np
from unittest import TestCase
from sklearn.datasets import make_regression

from tree import SoftDecisionTree


class TestSoftDecisionTree(TestCase):

    def test_forward(self):
        tree = SoftDecisionTree(
            5, 3, loss="mse", max_depth=4
        )
        x = torch.randn(4, 5)
        out = tree(x)
        print(tree.leaves)
        print(out)

    def test_set_leaves(self):
        n_features = 30
        X, y = make_regression(n_features=n_features, n_samples=200, random_state=0)
        y = y.reshape(-1, 1)
        X_tensor = torch.tensor(X).float()
        tree = SoftDecisionTree(
            n_features, 1, loss="mse", max_depth=4
        )
        tree.eval()
        mse_before = np.sum(np.abs(y - tree(X_tensor).detach().cpu().numpy()) ** 2)
        tree.set_leaves(X_tensor, y)
        mse_after = np.sum(np.abs(y - tree(X_tensor).detach().cpu().numpy()) ** 2)
        print(f"MSE loss\n- before:\t{mse_before}\n- after:\t{mse_after}")
        assert mse_after < mse_before
