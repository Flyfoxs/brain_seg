from fastai.vision import SegmentationItemList, get_transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from file_cache import *
from fastai.vision import imagenet_stats
from fastai.vision.data import ImageDataBunch, normalize_funcs

from dataset.ds_brain import DataSet_brain


@lru_cache()
def get_df():
    img_file_list = glob('/share/data2/body/brain/train/image/*MRI*/**/*.*', recursive=True)

    df = pd.DataFrame({'img_file': img_file_list})
    df['p_id_path'] = df.img_file.apply(lambda val: os.path.dirname(val))
    df['p_id'] = df['p_id_path'].rank(method='max').astype(int)
    df.sort_values('p_id')

    df['label_path'] = df.img_file.apply(lambda val: val.replace('image', 'label'))

    df['valid'] = df.p_id % 5 == 1

    df.valid.value_counts()
    return df


def get_data_bunch() -> ImageDataBunch:
    df = get_df()
    src = (SegmentationItemList.from_df(df, path='/', cols='img_file')
           .split_from_df(col='valid')
           # .label_from_func(get_y_fn, classes=codes)
           .label_from_df(cols='label_path', classes=range(5))
           )
    # .label_from_func(cols='label_path', classes=codes))

    print(len(src.train), len(src.valid))


    data = (src.transform(None, size=224, tfm_y=True)
            .databunch(bs=8)
            .normalize(imagenet_stats)
            )

    # data = (src.transform(None,  tfm_y=False)
    #         .databunch(bs=8)
    #         .normalize(imagenet_stats))

    return data


def get_dl(ds_type='train'):
    print('get_dl=', ds_type)
    data = get_data_bunch()

    # norm_stat = imagenet_stats
    # norm, denorm = normalize_funcs(*norm_stat, do_x=True, do_y=False)

    if ds_type == 'train':
        return data.train_dl #.create(data.train_ds, bs=8, shuffle=True, tfms=norm).dl
    elif ds_type == 'valid':
        #return data.valid_dl #.create(data.valid_ds, bs=8, shuffle=False, tfms=norm).dl
        return data.valid_dl#.create(DataSet_brain('valid'), bs=8, shuffle=False, tfms=None).dl
        #return data.valid_dl.dl
