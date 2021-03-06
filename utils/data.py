#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by zhuoer.rex on 2020/12/16 10:58 下午

import json
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils import data


class Dataset:

    TASK_MAP = {
        'R': 'regression',
        'C': 'classification'
    }

    def __init__(self, name):
        project_path = os.path.abspath(os.path.dirname(__file__))
        data_path = f"{project_path[:project_path.find('TAS')]}/TAS/datasets"
        path = Path(data_path)
        self.dataset_name = name
        self.data = pd.read_csv(
            path / f"{name}.csv",
            header=None
        )
        with open(path / f"{name}.json", 'r') as f:
            self.meta = json.load(f)
        print(self.meta)
        self._x = None
        self._y = None
        self._label_encoder = LabelEncoder()

    @property
    def instances(self):
        if self._x is None:
            self._x = self.data.iloc[:, :-1].values.astype(np.float)
        return self._x

    @property
    def labels(self):
        if self._y is None:
            self._y = self.data.iloc[:, -1].values
            if self.task == Dataset.TASK_MAP['C']:
                self._y = self._label_encoder.fit_transform(self._y)
            else:
                self._y = self._y.astype(np.float)
        return self._y

    @property
    def task(self):
        return Dataset.TASK_MAP[self.meta['task']]

    @property
    def time_budget(self):
        return self.meta.get('time_budget', 24 * 3600)

    @property
    def n_feature(self):
        return self.instances.shape[1]

    @property
    def n_output(self):
        if self.task == "regression":
            return 1
        elif len(self.labels.shape) == 1:
            return max(self.labels) + 1
        else:
            return self.labels.shape[1]

    @property
    def features(self):
        return self.instances.columns


class PlainDataset(data.Dataset):
    def __init__(self, x, y, n_classes=1):
        if n_classes == 1 and len(y.shape) == 1:
            y = y.reshape(-1, 1).astype(np.float32)
        elif n_classes > 1 and (len(y.shape) < 2 or y.shape[1] == 1):
            y = y.astype(np.long)
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class TabularDataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset: str,
            batch_size: int = 32,
            val_size: float = 0.1,
            test_size: float = 0.2):
        super(TabularDataModule, self).__init__()
        self.dataset = Dataset(dataset)
        self.batch_size = batch_size
        self.origin_set = None
        self.train_set, self.val_set, self.test_set = None, None, None
        self.val_size, self.test_size = val_size, test_size
        self.is_first_prepare = True
        # TODO: use pipeline for category feature
        self.scaler = StandardScaler()
        self.prepare_data()

    def prepare_data(self):
        if not self.is_first_prepare:
            return

        val_size, test_size = self.val_size, self.test_size
        dataset = self.dataset
        x, y = dataset.instances, dataset.labels
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        self.origin_set = torch.tensor(x), y
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=0, shuffle=True
        )
        self.scaler.fit(x_train)
        x_train, x_test = self.scaler.transform(x_train), self.scaler.transform(x_test)

        if self.val_size > .0:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=val_size / (1 - test_size), random_state=0, shuffle=True
            )
        else:
            x_val,  y_val = x_train, y_train

        self.train_set = x_train, y_train
        self.val_set = x_val, y_val
        self.test_set = x_test, y_test
        self.is_first_prepare = False

    @property
    def origin_train_set(self):
        if self.val_size > 0.0:
            x_train, y_train = self.train_set
            x_valid, y_valid = self.val_set
            return np.concatenate([x_train, x_valid], axis=0), np.concatenate([y_train, y_valid], axis=0)
        else:
            return self.train_set

    def train_dataloader(self):
        return data.DataLoader(
            PlainDataset(*self.train_set, n_classes=self.n_output),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() - 1
        )

    def val_dataloader(self):
        val_set = PlainDataset(*self.val_set, n_classes=self.n_output)
        return data.DataLoader(
            val_set,
            batch_size=len(val_set)
        )

    def test_dataloader(self):
        return data.DataLoader(
            PlainDataset(*self.test_set, n_classes=self.n_output),
            batch_size=self.batch_size,
            num_workers=os.cpu_count() - 1
        )

    @property
    def n_feature(self):
        return self.dataset.n_feature

    @property
    def n_output(self):
        return self.dataset.n_output
