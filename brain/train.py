import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from brain.module_pl import *

###### sacred begin ##############
from task_distribute.locker import task_locker
from sacred import Experiment
from easydict import EasyDict as edict
from sacred.observers import MongoObserver
from sacred import SETTINGS

db_url = 'mongodb://sample:password@10.10.20.107:27017/db?authSource=admin'

SETTINGS.CAPTURE_MODE = 'fd'  # ‘no’, ‘sys’, ‘fd’


@lru_cache()
def get_ex():
    ex = Experiment('brain_seg')
    ex.observers.append(MongoObserver(url=db_url, db_name='db'))
    return ex


ex = get_ex()


@ex.command()
def main(_config):
    hparams = edict(_config)
    print('=====', hparams)

    print(f'hparams:{hparams}')
    brain_model = BrainModel(hparams=hparams)

    trainer = pl.Trainer(gpus=1,
                         max_epochs=hparams.epochs,
                         progress_bar_refresh_rate=0,
                         weights_summary=None)
    trainer.fit(brain_model)


if __name__ == '__main__':

    """"
    export CUDA_VISIBLE_DEVICES=0 ; nohup python -u brain/train.py  main with epochs=10 lr=0.001 > 0.log  &
    """

    from sacred.arg_parser import get_config_updates
    import sys

    argv_conf, _ = get_config_updates(sys.argv[1:])
    logger.info(f'Input argsv:{argv_conf}')

    args = edict()

    args.epochs = 10
    args.lr = 1e-4
    args.n_classes = 5
    args.img_size = (224, 224)

    args.seed = np.random.randint(0, 1000)
    args.version = 'rand'

    real_conf = args
    real_conf.update(argv_conf)
    real_conf = sorted_dict(real_conf)

    locker = task_locker(db_url, remove_failed=9, version=real_conf.version)
    task_id = f'brain_{real_conf}'
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
