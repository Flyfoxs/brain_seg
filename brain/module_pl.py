import pytorch_lightning as pl
# from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import lr_scheduler

from brain.metrics import dice_multiply, hd95_multiply
# from brain.unet_model import *
from dataset.ds_brain import DataSet_brain
import torch
from brain.module_unet import *
from dataset.ds_brain_fastai import *
from dataset.ds_brain import *
from fastai.basic_data import *
from hausdorff import hausdorff_distance


def denormalize(x: torch.Tensor, mean: torch.Tensor = torch.Tensor(imagenet_stats[0]),
                std: torch.Tensor = torch.Tensor(imagenet_stats[1]),
                do_x: bool = True) -> torch.Tensor:
    "Denormalize `x` with `mean` and `std`."
    return x.cpu().float() * std[..., None, None] + mean[..., None, None] if do_x else x.cpu()


class BrainModel(pl.LightningModule):

    def __init__(self, hparams=None, ex=None, dl_type='fastai', imgaug=True):
        super(BrainModel, self).__init__()
        print('BrainModel', locals())
        model_name = hparams.model_name
        model_fn = UNET_MODEL.get(model_name)
        self.unet = model_fn(n_classes=hparams.n_classes,
                             img_size=hparams.img_size)

        print(self.unet)
        # self.loss_fn = DICELoss().cuda()
        self.loss_fn = partial(F.cross_entropy,
                               weight=torch.tensor([1, 1, 1, 1, 1.0]).cuda())
        # self.loss_fn = F.cross_entropy

        print(hparams)
        self.hparams = dict(hparams)

        self.dl_type = dl_type
        self.imgaug = imgaug

        self.ex = ex
        if self.ex is None: print('Ex is None')

    def forward(self, x):
        # called with self(x)
        # print(x.shape)
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self(x)

        y = torch.squeeze(y, dim=1)
        # print('training_step', y_hat.shape, y.shape)
        loss = self.loss_fn(y_hat, y)

        if batch_idx == 0:
            import torchvision.utils as vutils
            input_x = (denormalize(x) * 255).long()
            # input_y = (torch.stack([y[0]] * 3).cpu()*40).long()
            # img = torch.cat([input_x, input_y], dim=1)

            img = vutils.make_grid(input_x, scale_each=False)
            self.logger.experiment.add_image('train_img', img, self.current_epoch)

        # print(y_hat.shape, y.shape, loss)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y = torch.squeeze(y, dim=1)
        y_hat = self(x)


        return {'val_loss': self.loss_fn(y_hat, y),
                'dice': dice_multiply(y_hat, y),
                'hd95': hd95_multiply(y_hat,y),
                }

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # ipdb.set_trace()
        dice = torch.stack([x['dice'] for x in outputs])
        dice_cls = dice.mean(dim=0)
        dice_min = dice_cls.min()

        hd95 = torch.stack([x['hd95'] for x in outputs])
        hd95_cls = torch.Tensor(np.nanmean(hd95.cpu().numpy(),axis=0))
        hd95_min = hd95_cls.min()

        if self.ex:
            self.ex.log_scalar('ce', round(float(avg_loss), 5), self.current_epoch)
            self.ex.log_scalar('dice', round(float(dice_min), 5), self.current_epoch)
            #self.ex.log_scalar('hd95', round(float(hd95_min), 5), self.current_epoch)

        tensorboard_logs = {'val_loss': avg_loss,
                            'dice': dice_min,
                            **dict([(f'dice_{i}', dice_cls[i]) for i in range(5)]),
                            #**dict([(f'hd95_{i}', hd95_cls[i]) for i in range(5)]),
                            }
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
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

        scheduler = optim.lr_scheduler.StepLR(opt, gamma=0.5, step_size=20)

        print('schedule', scheduler)
        return [opt], [scheduler]

    def train_dataloader(self):
        # REQUIRED
        if self.dl_type == 'fastai':
            return get_dl('train')
        else:
            return DataLoader(DataSet_brain('train', imgaug=self.imgaug), 8, shuffle=True, num_workers=10)


    def val_dataloader(self):
        # OPTIONAL
        if self.dl_type == 'fastai':
            return get_dl('valid')
        else:
            return DataLoader(DataSet_brain('valid', imgaug=self.imgaug), batch_size=8, )


    def test_dataloader(self):
        # OPTIONAL
        if self.dl_type == 'fastai':
            return get_dl('valid')
        else:
            return DataLoader(DataSet_brain('valid', imgaug=self.imgaug), imgaug=self.imgaug, batch_size=8, )
        # return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32)
