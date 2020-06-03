from pytorch_lightning.callbacks import Callback
import logging as log
import numpy as np
import pandas as pd
import torch

from time import time
from easydict import EasyDict as edict


def format_time(t):
    t = int(t)
    h, m, s = t // 3600, (t // 60) % 60, t % 60
    if h != 0:
        return f'{h}:{m:02d}:{s:02d}'
    else:
        return f'{m:02d}:{s:02d}'


def convert_tensor(val):
    if isinstance(val, torch.Tensor):
        val = val.cpu().numpy()

    if isinstance(val, np.ndarray):
        val = val.squeeze()
        if val.ndim == 0:
            val = round(val.item(), 4)
        else:
            val = val.round(4)
    elif isinstance(val, (float, int)):
        val = round(val, 4)
    else:
        val = val
    return val


class MetricLogger(Callback):

    def __init__(self):
        super(MetricLogger, self).__init__()

        self.start_train = time()
        self.start_epoch = time()
        self.metrics_list = []

    def on_train_start(self, trainer, pl_module):
        print('checkpoint, v_num = ', trainer.logger.version)
        pass

    def on_epoch_start(self, trainer, pl_module):
        self.start_epoch = time()

    def on_epoch_end(self, trainer, pl_module):
        trn_dur = format_time(time() - self.start_train)
        ep_dur = format_time(time() - self.start_epoch)

        logs = trainer.callback_metrics.copy()
        for k, v in logs.items():
            logs[k] = convert_tensor(v)

        log_dict = edict(logs)
        log_dict.trn_dur = trn_dur
        log_dict.ep_dur = ep_dur
        log_dict.ct = str(pd.to_datetime('now'))[:19]
        self.metrics_list.append(log_dict)

        key_list = list(logs.keys()) + ['ep_dur', 'trn_dur', 'ct']
        key_list.remove('loss')
        if trainer.current_epoch == 0: print(key_list)
        print('\t  '.join(["%0.4f" % log_dict[key] if isinstance(log_dict[key], float) else str(log_dict[key]) for key in key_list]))

    def on_train_end(self, trainer, pl_module):
        df = pd.DataFrame(self.metrics_list)
        df = df.drop(columns=['loss', 'epoch', 'ct'], errors='ignore')
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('expand_frame_repr', False)
        print(df)
