import pytorch_lightning as pl
# from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import lr_scheduler

from brain.metrics import dice_multiply
# from brain.unet_model import *
from dataset.ds_brain import DataSet_brain
import torch
from brain.module_unet import *
from dataset.ds_brain_fastai import *


class BrainModel(pl.LightningModule):

    def __init__(self, hparams=None, ex=None):
        super(BrainModel, self).__init__()

        model_name = hparams.model_name
        model_fn = UNET_MODEL.get(model_name)
        self.unet = model_fn(n_classes=hparams.n_classes,
                             img_size=hparams.img_size)
        # self.loss_fn = DICELoss().cuda()
        self.loss_fn = partial(F.cross_entropy,
                               weight=torch.tensor([1, 1, 1, 1, 1.0]).cuda())
        # self.loss_fn = F.cross_entropy

        print(hparams)
        self.hparams = dict(hparams)


        self.ex = ex
        if self.ex is None: print('Ex is None')

    def forward(self, x):
        # called with self(x)
        # print(x.shape)
        return self.unet(x)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self(x)

        y = torch.squeeze(y, dim=1)
        # print('training_step', y_hat.shape, y.shape)
        loss = self.loss_fn(y_hat, y)

        # print(y_hat.shape, y.shape, loss)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y = torch.squeeze(y, dim=1)
        y_hat = self(x)
        # ipdb.set_trace()
        return {'val_loss': self.loss_fn(y_hat, y), 'dice': dice_multiply(y_hat, y)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # ipdb.set_trace()
        dice = torch.stack([x['dice'] for x in outputs])
        dice_cls = dice.mean(dim=0)
        dice_min = dice_cls.min()

        print('\n', {'epoch': self.current_epoch,
                     'val_loss': round(float(avg_loss), 4),
                     'dice': round(float(dice_min), 4),
                     'dice_all': list(dice_cls.numpy().round(4))
                     }),
        if self.ex:
            self.ex.log_scalar('ce', round(float(avg_loss), 5), self.current_epoch)
            self.ex.log_scalar('dice', round(float(dice_min), 5), self.current_epoch)

        tensorboard_logs = {'val_loss': avg_loss, 'dice': dice_min}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        print('test_step')
        x, y = batch
        y_hat = self(x)
        return {'test_loss': self.loss_fn(y_hat, y)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        # print('configure_optimizers', self.opt)
        # scheduler = lr_scheduler.OneCycleLR(self.opt, num_steps=406, lr_range=(0.1, 1.))

        # scheduler = optim.lr_scheduler.StepLR(self.opt, step_size=5)

        # scheduler = optim.lr_scheduler.OneCycleLR(self.opt, max_lr=0.01,
        #                                           steps_per_epoch=len(self.data_loader),
        #                                           epochs=10)

        # print('steps_per_epoch', len(self.train_dataloader()))

        opt = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'], betas=(0.9, 0.99))
        # scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=self.hparams.lr,
        #                                           steps_per_epoch=len(self.train_dataloader()),
        #                                           epochs=self.hparams.epochs)

        scheduler = optim.lr_scheduler.StepLR(opt, step_size=10)

        print('schedule', scheduler)
        return [opt], [scheduler]

    def train_dataloader(self):
        # REQUIRED
        return get_dl('train')
        # return DataLoader(get_ds('train'), batch_size=8, shuffle=True)
        # return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    def val_dataloader(self):
        # OPTIONAL
        return get_dl('valid')
        # return DataLoader(get_ds('valid'), batch_size=2, )
        ##return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    def test_dataloader(self):
        # OPTIONAL
        return get_dl('valid')
        # return DataLoader(get_ds('valid'), batch_size=2, )
        # return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32)
