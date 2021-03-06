#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by zhuoer.rex on 2021/1/6 10:35 下午

import argparse
import os
import pytorch_lightning as pl
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from tree import SoftDecisionTree, HierarchicalTree, LeafOptimizer
from utils import TabularDataModule


def get_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SoftDecisionTree.add_model_specific_args(parser)
    parser = TabularDataModule.add_argparse_args(parser)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--devices', type=str, default="0")
    parser.add_argument('--tree', type=str, default='hierarchy')
    arguments = parser.parse_args()
    return arguments


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    args.gpus = -1
    pl.seed_everything(args.seed)
    dm = TabularDataModule.from_argparse_args(args)
    n_feature = dm.n_feature
    n_output = dm.n_output

    if args.tree == "hierarchy":
        tree = HierarchicalTree(
            input_dim=n_feature,
            output_dim=n_output,
            loss=args.loss,
            max_depth=args.max_depth
        )
    else:
        tree = SoftDecisionTree(
            n_feature, n_output,
            loss=args.loss, max_depth=args.max_depth, n_branch=args.n_branch
        )

    if dm.dataset.task == "regression":
        sk_tree = DecisionTreeRegressor(max_depth=args.max_depth)
    else:
        sk_tree = DecisionTreeClassifier(max_depth=args.max_depth)

    sk_tree.fit(*dm.origin_train_set)
    print(f"{args.dataset} SklearnTree Score:\nTrain set\t{sk_tree.score(*dm.train_set)}"
          f"\nValid set\t{sk_tree.score(*dm.val_set)}"
          f"\nTest set\t{sk_tree.score(*dm.test_set)}")

    tree.eval()
    print(f"{args.dataset} SoftTree(Before Training) Score:\nTrain set\t{tree.score(*dm.train_set)}"
          f"\nValid set\t{tree.score(*dm.val_set)}"
          f"\nTest set\t{tree.score(*dm.test_set)}")

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            LeafOptimizer(dm.origin_train_set)
        ]
    )
    trainer.fit(tree, datamodule=dm)

    tree.eval()
    print(f"{args.dataset} SoftTree(Trained) Score:\nTrain set\t{tree.score(*dm.train_set)}"
          f"\nValid set\t{tree.score(*dm.val_set)}"
          f"\nTest set\t{tree.score(*dm.test_set)}")


if __name__ == '__main__':
    _args = get_args()
    main(_args)

