from torch import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def dice(input: Tensor, targs: Tensor, iou: bool = False, eps: float = 1e-8):
    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n, -1)
    targs = targs.view(n, -1)
    intersect = (input * targs).sum(dim=1).float()
    union = (input + targs).sum(dim=1).float()
    if not iou:
        l = 2. * intersect / union
    else:
        l = intersect / (union - intersect + eps)
    l[union == 0.] = 1.
    return l.mean()


def dice_multiply(logits, targets, cls_id=None):
    batch_size, class_cnt = logits.shape[0], logits.shape[1]

    dice_list = []
    for class_index in range(class_cnt):
        predict = logits.argmax(axis=1) == class_index
        target = (targets == class_index)

        predict = predict.view(batch_size, -1)
        target = target.view(batch_size, -1)

        # print('======', predict.shape,  target.shape)
        # print(predict.shape, logits.shape, target.shape)
        inter = torch.sum(predict * target, dim=1)
        union = torch.sum(predict, dim=1) + torch.sum(target, dim=1)
        dice = (2. * inter + 1) / (union + 1)
        # print(dice)
        dice = dice.mean()
        # print(dice, inter, union)
        dice_list.append(dice)
    # print(dice_list)
    if cls_id is None:
        return torch.Tensor(dice_list)
    else:
        return torch.Tensor(dice_list)[cls_id]
