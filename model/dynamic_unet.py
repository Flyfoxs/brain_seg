import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from easydict import EasyDict as edict
import pandas as pd


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )


class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True):
        super().__init__()

        self.layer_index, self.channel_list = self.get_unet_config(encoder)

        self.encoder = encoder
        self.concat_input = concat_input

        self.up_conv1 = up_conv(self.channel_list[-1], 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)
        print('self.size[5], out_channels, kernel_size', self.size[5], out_channels)

        self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)


    def get_blocks_to_be_concat(self, model, x):
        shapes = set()
        blocks = OrderedDict()
        hooks = []
        count = 0

        def register_hook(module):

            def hook(module, input, output):
                try:
                    nonlocal count

                    if count in self.layer_index:
                        blocks[count] = output
                    count += 1
                except AttributeError:
                    pass

            hooks.append(module.register_forward_hook(hook))

        # register hook
        model.apply(register_hook)

        # make a forward pass to trigger the hooks
        model(x)

        # remove these hooks
        for h in hooks:
            h.remove()

        return blocks

    def get_unet_config(self, model, x=torch.rand(1, 3, 512, 512)):
        shapes = []
        blocks = OrderedDict()
        hooks = []
        count = 0
        channel = edict()
        res = []

        def register_hook(module):

            def hook(module, input, output):
                try:
                    nonlocal count

                    # print(output.shape)
                    if len(output.shape) == 4:
                        b, c, w, h = output.shape
                        res.append((count, type(module).__name__, c, w, h))
                    else:
                        pass
                        # print( count, output.shape,)

                    # channel[module.name]= output.size()[1]
                    count += 1
                except AttributeError:
                    print('Error:', type(module))
                    pass

            hooks.append(module.register_forward_hook(hook))

        # register hook
        model.apply(register_hook)

        # make a forward pass to trigger the hooks
        model(x)

        # remove these hooks
        for h in hooks:
            h.remove()

        tmp = pd.DataFrame(res, columns=['sn', 'layer', 'c', 'w', 'h'])

        img_size = [x.shape[-1] // (2 ** i) for i in range(6)]
        tmp = tmp.loc[(tmp.layer == 'BatchNorm2d') & (tmp.w.isin(img_size))].drop_duplicates(['w'], keep='last')

        print('layer_list, channel', list(tmp.sn), list(tmp.c))
        return list(tmp.sn), list(tmp.c)

    @property
    def n_channels(self):
        return self.channel_list[-1]

    @property
    def size(self):

        return [512 + self.channel_list[-2],
                256 + self.channel_list[-3],
                128 + self.channel_list[-4],
                64 + self.channel_list[-5],
                35,
                32]

    def forward(self, x):
        input_ = x

        blocks = self.get_blocks_to_be_concat(self.encoder, x)
        _, x = blocks.popitem()

        x = self.up_conv1(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv1(x)

        x = self.up_conv2(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv2(x)

        x = self.up_conv3(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv3(x)

        x = self.up_conv4(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv4(x)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)

        x = self.final_conv(x)

        return x


def get_efficientunet(name='b0', out_channels=5, concat_input=True):
    from efficientnet_pytorch import EfficientNet
    encoder = EfficientNet.from_pretrained(f'efficientnet-{name}')
    # encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model

if __name__ == '__main__':

    from torchvision import models
    from model.dynamic_unet import EfficientUnet
    encoder = models.resnet50(())
    unet = EfficientUnet(encoder, out_channels=5, concat_input=True)
    unet(torch.rand(1, 3, 224, 224))



    for i in range(7):
        tmp = get_efficientunet(f'b{i}')
        print(tmp(torch.rand(1, 3, 224, 224)).shape)

