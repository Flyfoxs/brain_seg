from torch import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from medpy.metric.binary import hd95


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


def hd95_multiply(logits, targets, cls_id=None):
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
        targets = targets.cpu().numpy()
    batch_size, class_cnt = logits.shape[0], logits.shape[1]

    #print('batch_size, class_cnt ', batch_size, class_cnt )
    hd95_score = []
    for class_index in range(class_cnt):
        #print(class_index, np.unique(logits.argmax(axis=1)))
        predict = logits.argmax(axis=-3) == class_index
        target = (targets == class_index)

        #print('predict, target', class_index, predict.shape, target.shape,
              #predict.sum(), target.sum(), np.unique(targets))
        score = [hd95(res, ref) for res, ref in zip(predict, target)
                 if np.count_nonzero(res) > 0 and np.count_nonzero(ref) > 0]
        #print('class_index=', class_index, score)
        hd95_score.append(np.mean(score))
    #print('hd95_score', hd95_score)

    if cls_id is None:
        return torch.Tensor(hd95_score)
    else:
        return torch.Tensor(hd95_score)[cls_id]
