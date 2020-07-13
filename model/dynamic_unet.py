import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from easydict import EasyDict as edict
import pandas as pd
from fastai.layers import *
from fastai.vision.models.unet import UnetBlock
from torch.nn import Module
from torch import Tensor
from enum import Enum


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


NormType = Enum('NormType', 'Batch BatchZero Weight Spectral')


def batchnorm_2d(nf: int, norm_type: NormType = NormType.Batch):
    "A batchnorm2d layer with `nf` features initialized depending on `norm_type`."
    bn = nn.BatchNorm2d(nf)
    with torch.no_grad():
        bn.bias.fill_(1e-3)
        bn.weight.fill_(0. if norm_type == NormType.BatchZero else 1.)
    return bn


class UnetBlock(Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."

    def __init__(self, up_in_c: int, x_in_c: int,
                 final_div: bool = True, blur: bool = False, leaky: float = None,
                 self_attention: bool = False, **kwargs):
        super(UnetBlock, self).__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, leaky=leaky, **kwargs)
        self.bn = batchnorm_2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = ni if final_div else ni // 2
        self.conv1 = conv_layer(ni, nf, leaky=leaky, **kwargs)
        self.conv2 = conv_layer(nf, nf, leaky=leaky, self_attention=self_attention, **kwargs)
        self.relu = relu(leaky=leaky)

    def forward(self, up_in: Tensor, encode_x: Tensor) -> Tensor:
        s = encode_x
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class EfficientUnet(nn.Module):
    def __init__(self, encoder, in_channels=3, out_channels=2, concat_input=True):
        super().__init__()

        self.layer_index, self.channel_list = self.get_unet_config(encoder)

        self.encoder = encoder
        self.concat_input = concat_input

        ni = self.channel_list[-1]
        self.middle_conv = nn.Sequential(
            batchnorm_2d(ni),
            nn.ReLU(),
            conv_layer(ni, ni * 2, norm_type=NormType),
            conv_layer(ni * 2, ni, norm_type=NormType),
        )

        dummy_x = torch.rand(1, 3, 224, 224)
        blocks = self.get_blocks_to_be_concat(self.encoder, dummy_x)

        up_x = self.middle_conv(blocks.popitem()[1])
        up_in_c = up_x.shape[1]
        x_in_c = self.channel_list[-2]
        print('conv1', up_in_c, x_in_c, up_x.shape)
        self.unet_block1 = UnetBlock(up_in_c, x_in_c, )

        up_x = self.unet_block1(up_x, blocks.popitem()[1])
        up_in_c = up_x.shape[1]
        x_in_c = self.channel_list[-3]
        print('conv2', up_in_c, x_in_c, up_x.shape)
        self.unet_block2 = UnetBlock(up_in_c, x_in_c )

        up_x = self.unet_block2(up_x, blocks.popitem()[1])
        up_in_c = up_x.shape[1]
        x_in_c = self.channel_list[-4]
        self.unet_block3 = UnetBlock(up_in_c, x_in_c )

        up_x = self.unet_block3(up_x, blocks.popitem()[1])
        up_in_c = up_x.shape[1]
        x_in_c = self.channel_list[-5]
        print('conv3', up_in_c, x_in_c, up_x.shape)
        self.unet_block4 = UnetBlock(up_in_c, x_in_c)

        up_x = self.unet_block4(up_x, blocks.popitem()[1])
        up_in_c = up_x.shape[1]
        print('conv4', up_in_c, None, up_x.shape)

        if self.concat_input:
            self.up_conv_input = up_conv(up_in_c, 32)
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
        select_layer = []

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
                    select_layer.append(module)
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
        return list(tmp.sn), list(tmp.c)  # , select_layer[list(tmp.sn)]

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

        x = self.middle_conv(x)

        x = self.unet_block1(x, blocks.popitem()[1])

        x = self.unet_block2(x, blocks.popitem()[1])

        x = self.unet_block3(x, blocks.popitem()[1])

        x = self.unet_block4(x, blocks.popitem()[1])

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
