#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by zhuoer.rex on 2020/12/22 7:50 下午
from unittest import TestCase

import os
import torch
import pytorch_lightning as pl
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from tree.hierarchy import MixNode, HierarchicalTree
from tree.base import LeafOptimizer, ArchOptimizer
from utils.data import PlainDataset


class DummyDataset:

    def __init__(self):
        self.x, self.y = None, None
        self.x_train, self.y_train = None, None
        self.x_valid, self.y_valid = None, None
        self.x_test, self.y_test = None, None
        self.n_features = None

    def init_gpu(self, gpus="2"):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        torch.set_default_tensor_type(torch.FloatTensor)

    def init_dataset(self, x, y):
        self.x, self.y = x, y
        self.x_train, self.y_train = x, y
        self.x_valid, self.y_valid = x, y
        self.x_test, self.y_test = x, y

    def init_classification_env(self, n_features=20, n_samples=200, random_state=0):
        self.n_features = n_features
        x, y = make_classification(
            n_features=n_features, n_samples=n_samples, n_classes=4,
            n_informative=3,
            random_state=random_state
        )
        self.init_dataset(x, y)

    def init_regression_env(self, n_features=20, n_samples=20, random_state=0):
        self.n_features = n_features
        x, y = make_regression(
            n_features=n_features, n_samples=n_samples,
            random_state=random_state
        )
        y = y.reshape(-1, 1)
        self.init_dataset(x, y)

    def init_data_split(self, test_size=0.2, valid_size=.0, random_state=0):
        x,  y = self.x, self.y
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        if valid_size == 0.0:
            self.x_valid, self.y_valid = self.x_train, self.y_train
        else:
            x_train, x_valid, y_train, y_valid = train_test_split(
                self.x_train, self.y_train,
                test_size=valid_size, random_state=random_state
            )
            self.x_train, self.y_train = x_train, y_train
            self.x_valid, self.y_valid = x_valid, y_valid

    @property
    def train_set(self):
        return self.x_train, self.y_train

    @property
    def valid_set(self):
        return self.x_valid, self.y_valid

    @property
    def test_set(self):
        return self.x_test, self.y_test


class TestHierarchy(TestCase):

    def test_mix_node(self):
        dataset = DummyDataset()
        dataset.init_regression_env(n_features=20, n_samples=20)
        X, y = dataset.x, dataset.y
        X_tensor = torch.tensor(X).float()

        tree = MixNode(input_dim=dataset.n_features, output_dim=1, depth=0)
        y_hat = tree(X_tensor)

    def test_hierarchical_tree_overfit(self):
        dataset = DummyDataset()
        dataset.init_gpu()
        dataset.init_regression_env(n_features=20, n_samples=200)
        x, y = dataset.x, dataset.y

        tree = HierarchicalTree(
            input_dim=dataset.n_features,
            output_dim=1,
            loss='mse',
            max_depth=3
        )
        print("Random init score: ", tree.score(x, y))
        tree.set_leaves(torch.tensor(x, device=tree.device).float(), y)
        print("Set root score: ", tree.score(x, y))
        with torch.no_grad():
            tree.root.gamma.copy_(torch.ones(1, device=tree.device))
        print("Random init score 2 depth: ", tree.score(x, y))
        tree.set_leaves(torch.tensor(x, device=tree.device).float(), y)
        print("Set root score 2 depth: ", tree.score(x, y))

        trainer = pl.Trainer(
            gpus="0",
            max_epochs=500,
            callbacks=[
                LeafOptimizer((x, y))
            ]
        )
        trainer.fit(
            tree,
            train_dataloader=torch.utils.data.DataLoader(PlainDataset(x, y), batch_size=32)
        )

        print("After optim score: ", tree.score(x, y))

    def test_hierarchical_general(self):
        dummy = DummyDataset()
        dummy.init_gpu("2")
        dummy.init_regression_env(n_features=20, n_samples=200)
        dummy.init_data_split(test_size=0.2, valid_size=0.0)
        max_depth = 4

        x_train, y_train = dummy.train_set

        dataset = PlainDataset(x_train, y_train)

        sk_tree = DecisionTreeRegressor(max_depth=max_depth)
        sk_tree.fit(*dummy.train_set)
        print("Sklearn Train Score: ", sk_tree.score(*dummy.train_set))
        print("Sklearn Test Score: ", sk_tree.score(*dummy.test_set))

        tree = HierarchicalTree(
            input_dim=dummy.n_features,
            output_dim=1,
            loss="mse",
            max_depth=max_depth
        )

        with torch.no_grad():
            tree.root.gamma.copy_(torch.ones(1, device=tree.device))
        print("Random init score: ", tree.score(x_train, y_train))
        tree.set_leaves(torch.tensor(x_train, device=tree.device).float(), y_train)
        print("Set root score: ", tree.score(x_train, y_train))

        trainer = pl.Trainer(
            gpus="-1",
            max_epochs=100,
            callbacks=[
                LeafOptimizer(dummy.train_set),
                ArchOptimizer(torch.utils.data.DataLoader(PlainDataset(*dummy.valid_set), batch_size=32))
            ]
        )
        trainer.fit(
            tree,
            train_dataloader=torch.utils.data.DataLoader(dataset, batch_size=32)
        )

        print("Optimized Train Score: ", tree.score(*dummy.train_set))
        print("Optimized Test Score: ", tree.score(*dummy.test_set))

    def test_hierarchical_classification(self):
        dummy = DummyDataset()
        dummy.init_gpu("1")
        dummy.init_classification_env(n_features=20, n_samples=200)
        dummy.init_data_split(test_size=0.2, valid_size=0.2)
        n_classes = 4

        x_train, y_train = dummy.train_set
        x_valid, y_valid = dummy.valid_set
        x_test,  y_test = dummy.test_set

        dataset = PlainDataset(x_train, y_train, n_classes=n_classes)
        tree = HierarchicalTree(
            input_dim=dummy.n_features,
            output_dim=n_classes,
            loss="ce",
            max_depth=3
        )

        with torch.no_grad():
            tree.root.gamma.copy_(torch.ones(1, device=tree.device))
        print("Random init score: ", tree.score(x_test, y_test))
        tree.set_leaves(torch.tensor(x_train, device=tree.device).float(), y_train)
        print("Set root score: ", tree.score(x_test, y_test))

        trainer = pl.Trainer(
            gpus="0",
            max_epochs=500,
            callbacks=[
                LeafOptimizer((x_train, y_train))
            ]
        )
        trainer.fit(
            tree,
            train_dataloader=torch.utils.data.DataLoader(dataset, batch_size=32)
        )

        print("After optim score: ", tree.score(x_train, y_train))
        print("After optim score: ", tree.score(x_valid, y_valid))

