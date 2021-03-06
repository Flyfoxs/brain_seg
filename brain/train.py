import os
import sys
import torch
import random
import numpy as np
# np.random.seed(2020)
# random.seed(2020)
# torch.manual_seed(2020)
# torch.cuda.manual_seed_all(2020)
# torch.backends.cudnn.benchmark = True

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pytorch_lightning.callbacks import EarlyStopping

from brain.module_pl import *
from dataset.ds_brain import *
from dataset.ds_brain_fastai import *
from dataset.img_prepare import *

from task_distribute.locker import task_locker
from easydict import EasyDict as edict

###### sacred begin ##############
from sacred import Experiment
from sacred.observers import MongoObserver

from sacred.arg_parser import get_config_updates

from callback.pl_callback import MetricLogger
from file_cache import *
db_url = 'mongodb://sample:password@10.10.20.107:27017/db?authSource=admin'


@lru_cache()
def get_ex():
    from sacred import SETTINGS
    SETTINGS.CAPTURE_MODE = 'fd'  # ‘no’, ‘sys’, ‘fd’
    ex = Experiment('brain_seg')
    ex.observers.append(MongoObserver(url=db_url, db_name='db'))
    return ex


ex = get_ex()

for file in glob('./**/*.py', recursive=True):
    ex.add_source_file(file)

@ex.command()
def main(_config):
    hparams = edict(_config)
    print(f'=====hparams:{hparams}')
    brain_model = BrainModel(hparams=hparams, ex=get_ex(), dl_type=hparams.dl_type, imgaug=hparams.imgaug)
    from pytorch_lightning.callbacks.lr_logger import LearningRateLogger
    #lr_logger = LearningRateLogger()
    metric_logger = MetricLogger()
    early_stop = EarlyStopping(
        monitor='dice',
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='max'
    )

    trainer = pl.Trainer(gpus=1,
                         max_epochs=hparams.epochs,
                         num_sanity_val_steps=0,
                         show_progress_bar=False,
                         progress_bar_refresh_rate=0,
                         weights_summary=None,
                         callbacks=[metric_logger, early_stop]
                         )

    trainer.fit(brain_model)


if __name__ == '__main__':

    """"
    export CUDA_VISIBLE_DEVICES=0 ; nohup python -u brain/train.py  main with epochs=10 lr=0.001 > 0.log  &
    """

    argv_conf, _ = get_config_updates(sys.argv[1:])
    logger.info(f'Input argsv:{argv_conf}')

    args = edict()

    args.epochs = 20
    args.lr = 1e-4
    args.n_classes = 5
    args.img_size = (224, 224)
    args.model_name = 'dynamic_unet'
    args.dl_type = 'normal'
    args.imgaug = True

    args.seed = np.random.randint(0, 1000)
    args.version = 'rand2'

    real_conf = args
    real_conf.update(argv_conf)
    real_conf = sorted_dict(real_conf)

    locker = task_locker(db_url, remove_failed=9, version=real_conf.version)
    task_id = f'{ex.path}_{real_conf}'
    with locker.lock_block(task_id=task_id) as lock_id:
        if lock_id is not None:
            ex.add_config({
                **real_conf,
                'lock_id': lock_id,
                'lock_name': task_id,
                'version': real_conf.version,
                'GPU': os.environ.get('CUDA_VISIBLE_DEVICES'),
            })
            try:
                res = ex.run_commandline()
            except Exception as e:
                logger.info(f'Process error:{args}')
                logger.exception(e)
