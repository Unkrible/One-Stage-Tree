#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by zhuoer.rex on 2020/12/16 10:57 下午

import torch
import numpy as np


def ohe_from_logits(logits, eps=0.0):
    logits_max, logits_max_idxes = logits.max(1, keepdim=True)
    argmax_acs = torch.eye(logits.size()[1], device=logits.device)[logits_max_idxes.squeeze()].float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = torch.eye(logits.shape[1], device=logits.device)[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]]

    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in enumerate(torch.rand(logits.shape[0]))])


def sample_gumbel(shape, eps=1e-20, dtype=torch.float,  device=None):
    """
    Sample from Gumbel(0, 1)
    """
    U = torch.rand(*shape, dtype=dtype, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, mask=None):
    gumbels = sample_gumbel(logits.shape, dtype=logits.dtype, device=logits.device)
    y = torch.log_softmax(logits, dim=1) + gumbels
    if mask is not None:
        y = y.masked_fill(mask, float('-inf'))
    return torch.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0, hard=True, mask=None):
    y = gumbel_softmax_sample(logits, temperature, mask=mask)
    if hard:
        y_hard = ohe_from_logits(y)
        # re-parameter trick
        y = (y_hard - y).detach() + y
    return y
