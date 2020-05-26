from fastai.vision import SegmentationItemList
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from file_cache import *
from fastai.vision import imagenet_stats


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


def get_data_bunch():
    df = get_df()
    src = (SegmentationItemList.from_df(df, path='/', cols='img_file')
           .split_from_df(col='valid')
           # .label_from_func(get_y_fn, classes=codes)
           .label_from_df(cols='label_path', classes=range(5))
           )
    # .label_from_func(cols='label_path', classes=codes))

    print(len(src.train), len(src.valid))

    # get_transforms()
    data = (src.transform(None, size=224, tfm_y=True)
            .databunch(bs=8)
            .normalize(imagenet_stats))

    return data


def get_dl(ds_type='train'):
    data = get_data_bunch()

    if ds_type == 'train':
        return data.train_dl
    elif ds_type == 'valid':
        return data.valid_dl
