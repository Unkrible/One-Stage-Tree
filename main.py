#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by zhuoer.rex on 2020/12/16 10:05 下午

import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from tree import HierarchicalTree, LeafOptimizer, ArchOptimizer
from utils import TabularDataModule


def get_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = HierarchicalTree.add_model_specific_args(parser)
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

    tree = HierarchicalTree(
        input_dim=n_feature,
        output_dim=n_output,
        loss=args.loss,
        max_depth=args.max_depth
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

    ckp_callback = ModelCheckpoint(
        monitor='train_loss',
        filename=f"{args.dataset}-" + "{train_loss:.3f}",
        save_top_k=3
    )

    if args.patience == -1:
        early_stopping = []
    else:
        early_stopping = [EarlyStopping(monitor="val_loss", patience=args.patience)]

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            LeafOptimizer(dm.train_set),
            ArchOptimizer(dm.val_dataloader()),
            ckp_callback,
            *early_stopping
        ]
    )
    trainer.fit(tree, datamodule=dm)

    tree.eval()
    print(f"{args.dataset} SoftTree(Trained) Score:\nTrain set\t{tree.score(*dm.train_set)}"
          f"\nValid set\t{tree.score(*dm.val_set)}"
          f"\nTest set\t{tree.score(*dm.test_set)}")
    print(f"Checkpoint Path: {ckp_callback.best_model_path}")


if __name__ == '__main__':
    _args = get_args()
    print(_args)
    main(_args)
