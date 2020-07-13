import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from file_cache import *
from fastai.vision import *
from fastai.callbacks.hooks import *
from fvcore.common.registry import Registry

UNET_MODEL = Registry("UNET_MODEL")


@UNET_MODEL.register()
def unet_fsai_r34( pretrained: bool = True, blur_final: bool = True,
                 norm_type: Optional[NormType] = NormType, split_on: Optional[SplitFuncOrIdxList] = None,
                 blur: bool = False,
                 self_attention: bool = False, y_range: Optional[Tuple[float, float]] = None, last_cross: bool = True,
                 bottle: bool = False, cut: Union[int, Callable] = None,
                 n_classes=2, img_size=(224, 224), in_channels=1,
                 **learn_kwargs: Any) -> Learner:
    "Build Unet learner from `data` and `arch`."
    "blur: do maxpolling or not"
    from fastai.vision import models
    arch: Callable = models.resnet34
    body = create_body(arch, pretrained, cut)
    from fastai.vision import models
    model = to_device(
        models.unet.DynamicUnet(body, n_classes=n_classes, img_size=img_size, blur=blur, blur_final=blur_final,
                                self_attention=self_attention, y_range=y_range, norm_type=norm_type,
                                last_cross=last_cross,
                                bottle=bottle), 'cuda')
    return model


# @DeprecationWarning('Score is low than monai')
@UNET_MODEL.register()
def unet_normal(*args, **kwargs):
    from model.unet import UNet
    unet = UNet(in_channels=3,
                n_classes=5,
                padding=True,
                depth=5,
                up_mode='upsample',
                batch_norm=True,
                residual=False)
    # print(unet)
    return unet


@UNET_MODEL.register()
def unet_monai(*args, **kwargs):
    from monai.networks.nets import UNet
    model = UNet(
        dimensions=2,
        in_channels=3,
        out_channels=5,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    return model


@UNET_MODEL.register()
def unet3(*args, **kwargs):
    from model.attention_unet import U_Net
    return U_Net(img_ch=3, output_ch=5)


# Better
@UNET_MODEL.register()
def unet_att(*args, **kwargs):
    from model.attention_unet import AttU_Net
    return AttU_Net(img_ch=3, output_ch=5)


@UNET_MODEL.register()
def unet_effb0(*args, **kwargs):
    from efficientunet import get_efficientunet_b0
    unet = get_efficientunet_b0(out_channels=5, concat_input=True, pretrained=True)
    return unet


@UNET_MODEL.register()
def unet_effb3(*args, **kwargs):
    from efficientunet import get_efficientunet_b3
    unet = get_efficientunet_b3(out_channels=5, concat_input=True, pretrained=True)
    return unet


@UNET_MODEL.register()
def unet_effb4(*args, **kwargs):
    from efficientunet import get_efficientunet_b4
    unet = get_efficientunet_b4(out_channels=5, concat_input=True, pretrained=True)
    return unet


@UNET_MODEL.register()
def unet_effb5(*args, **kwargs):
    from efficientunet import get_efficientunet_b5
    unet = get_efficientunet_b5(out_channels=5, concat_input=True, pretrained=True)
    return unet


@UNET_MODEL.register()
def unet_effb6(*args, **kwargs):
    from efficientunet import get_efficientunet_b6
    unet = get_efficientunet_b6(out_channels=5, concat_input=True, pretrained=True)
    return unet


@UNET_MODEL.register()
def unet_effb7(*args, **kwargs):
    from efficientunet import get_efficientunet_b7
    unet = get_effientunet_b7(out_channels=5, concat_input=True, pretrained=True)
    return unet


@UNET_MODEL.register()
def unet_de5(*args, **kwargs):
    from model.dynamic_unet import get_efficientunet
    unet = get_efficientunet('b5')
    return unet


@UNET_MODEL.register()
def unet_de7(*args, **kwargs):
    from model.dynamic_unet import get_efficientunet
    unet = get_efficientunet('b7')
    return unet


@UNET_MODEL.register()
def unet_res50(*args, **kwargs):
    from model.dynamic_unet import EfficientUnet
    encoder = models.resnet50(())
    unet = EfficientUnet(encoder, out_channels=5, concat_input=True)
    return unet


from model.dynamic_unet_fastai import *


@UNET_MODEL.register()
def unet_fsai_b5(*args, **kwargs):
    encoder = eff(5)
    unet = to_device(
        DynamicUnet(encoder, n_classes=5, img_size=(224, 224), blur=False, blur_final=False,
                    self_attention=False, y_range=None, norm_type=NormType,
                    last_cross=True,
                    bottle=False), 'cuda')
    return unet


@UNET_MODEL.register()
def unet_fsai_b7(*args, **kwargs):
    encoder = eff(5)
    unet = to_device(
        DynamicUnet(encoder, n_classes=5, img_size=(224, 224), blur=False, blur_final=False,
                    self_attention=False, y_range=None, norm_type=NormType,
                    last_cross=True,
                    bottle=False), 'cuda')
    return unet


@UNET_MODEL.register()
def unet_fsai_b0_v2(arch: Callable = models.resnet34, pretrained: bool = True, blur_final: bool = True,
                    norm_type: Optional[NormType] = NormType, split_on: Optional[SplitFuncOrIdxList] = None,
                    blur: bool = False,
                    self_attention: bool = False, y_range: Optional[Tuple[float, float]] = None,
                    last_cross: bool = True,
                    bottle: bool = False, cut: Union[int, Callable] = None,
                    n_classes=2, in_channels=1,
                    **learn_kwargs: Any) -> Learner:
    "Build Unet learner from `data` and `arch`."
    "blur: do maxpolling or not"
    body = eff(0)
    from fastai.vision import models
    img_size = (224, 224)
    model = to_device(
        DynamicUnet(body, n_classes=n_classes, img_size=img_size, blur=blur, blur_final=blur_final,
                    self_attention=self_attention, y_range=y_range, norm_type=norm_type,
                    last_cross=last_cross,
                    bottle=bottle), 'cuda')
    return model


@UNET_MODEL.register()
def unet_fsai_b5_v2(arch: Callable = models.resnet34, pretrained: bool = True, blur_final: bool = True,
                    norm_type: Optional[NormType] = NormType, split_on: Optional[SplitFuncOrIdxList] = None,
                    blur: bool = False,
                    self_attention: bool = False, y_range: Optional[Tuple[float, float]] = None,
                    last_cross: bool = True,
                    bottle: bool = False, cut: Union[int, Callable] = None,
                    n_classes=2, in_channels=1,
                    **learn_kwargs: Any) -> Learner:
    "Build Unet learner from `data` and `arch`."
    "blur: do maxpolling or not"
    body = eff(5)
    from fastai.vision import models
    img_size = (224, 224)
    model = to_device(
        DynamicUnet(body, n_classes=n_classes, img_size=img_size, blur=blur, blur_final=blur_final,
                    self_attention=self_attention, y_range=y_range, norm_type=norm_type,
                    last_cross=last_cross,
                    bottle=bottle), 'cuda')
    return model


@UNET_MODEL.register()
def unet_fsai_b7_v2(arch: Callable = models.resnet34, pretrained: bool = True, blur_final: bool = True,
                    norm_type: Optional[NormType] = NormType, split_on: Optional[SplitFuncOrIdxList] = None,
                    blur: bool = False,
                    self_attention: bool = False, y_range: Optional[Tuple[float, float]] = None,
                    last_cross: bool = True,
                    bottle: bool = False, cut: Union[int, Callable] = None,
                    n_classes=2, in_channels=1,
                    **learn_kwargs: Any) -> Learner:
    "Build Unet learner from `data` and `arch`."
    "blur: do maxpolling or not"
    body = eff(7)
    from fastai.vision import models
    img_size = (224, 224)
    model = to_device(
        DynamicUnet(body, n_classes=n_classes, img_size=img_size, blur=blur, blur_final=blur_final,
                    self_attention=self_attention, y_range=y_range, norm_type=norm_type,
                    last_cross=last_cross,
                    bottle=bottle), 'cuda')
    return model

#
# @UNET_MODEL.register()
# def unet_fsai_r34(*args, **kwargs):
#     encoder = create_body(models.resnet34, pretrained=True, cut=None)
#     unet = to_device(
#         DynamicUnet(encoder, n_classes=5, img_size=(224, 224), blur=False, blur_final=False,
#                     self_attention=False, y_range=None, norm_type=NormType,
#                     last_cross=True,
#                     bottle=False), 'cuda')
#     return unet


@UNET_MODEL.register()
def unet_fsai_r34_v2(arch: Callable = models.resnet34, pretrained: bool = True, blur_final: bool = True,
                     norm_type: Optional[NormType] = NormType, split_on: Optional[SplitFuncOrIdxList] = None,
                     blur: bool = False,
                     self_attention: bool = False, y_range: Optional[Tuple[float, float]] = None,
                     last_cross: bool = True,
                     bottle: bool = False, cut: Union[int, Callable] = None,
                     n_classes=2, in_channels=1,
                     **learn_kwargs: Any) -> Learner:
    "Build Unet learner from `data` and `arch`."
    "blur: do maxpolling or not"
    body = create_body(arch, pretrained, cut)
    from fastai.vision import models
    img_size = (224, 224)
    model = to_device(
        DynamicUnet(body, n_classes=n_classes, img_size=img_size, blur=blur, blur_final=blur_final,
                    self_attention=self_attention, y_range=y_range, norm_type=norm_type,
                    last_cross=last_cross,
                    bottle=bottle), 'cuda')
    return model
