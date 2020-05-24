import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from file_cache import *
from fastai.vision import *
from fastai.callbacks.hooks import *

# gc.collect()
def dynamic_unet( arch:Callable, pretrained:bool=False, blur_final:bool=True,
                 norm_type:Optional[NormType]=NormType, split_on:Optional[SplitFuncOrIdxList]=None, blur:bool=False,
                 self_attention:bool=False, y_range:Optional[Tuple[float,float]]=None, last_cross:bool=True,
                 bottle:bool=False, cut:Union[int,Callable]=None,
                 n_classes=2, img_size = (224,224), in_channels = 1,
                 **learn_kwargs:Any)->Learner:
    "Build Unet learner from `data` and `arch`."
    "blur: do maxpolling or not"
    body = create_body(arch, pretrained, cut)
    model = to_device(models.unet.DynamicUnet(body, n_classes=n_classes, img_size=img_size, blur=blur, blur_final=blur_final,
          self_attention=self_attention, y_range=y_range, norm_type=norm_type, last_cross=last_cross,
          bottle=bottle), 'cuda')
    return model