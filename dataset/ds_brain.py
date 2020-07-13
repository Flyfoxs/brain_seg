from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from file_cache import *
import torch
from .transform import *
import torch.nn.functional as F
from fastai.vision import imagenet_stats
import albumentations as A

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    Resize,
    Rotate,
    Normalize,
)

# https://debuggercafe.com/image-augmentation-using-pytorch-and-albumentations/

# from monai.transforms import \
#     Compose, LoadNiftid, AddChanneld, ScaleIntensityRanged, RandCropByPosNegLabeld, \
#     CropForegroundd, RandAffined, Spacingd, Orientationd, ToTensord, ToTensor,\
#     AsChannelFirstd, ScaleIntensityd, RandRotate90d,Resize

from PIL import Image


class DataSet_brain(Dataset):

    def __init__(self, ds_type='train', imgaug=True):
        print('DataSet_brain', locals())
        self.ds_type = ds_type
        self.imgaug = imgaug

        df = self.get_df()

        if ds_type == 'train':
            print('========', ds_type)
            print(df.valid.value_counts())
            self.df = df.loc[df.valid == False]
        elif ds_type == 'valid':
            self.df = df.loc[df.valid == True]

        # original_height, original_width = 224, 224
        size = 224
        crop_size = np.random.uniform(0.9, 1)
        self.aug_train = Compose([
            # OneOf([RandomSizedCrop(min_max_height=(50, 101), height=original_height, width=original_width, p=0.5),
            #        PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)], p=1),
            #CenterCrop(int(size*crop_size), int(size*crop_size)),
            VerticalFlip(p=0.5),
            Rotate(limit=(-20, 20)),
            Resize(height=size, width=size),
            Normalize()
        ])

        self.aug_val = Compose([
            Resize(height=size, width=size),
            Normalize()
        ])

    @lru_cache()
    def get_df(self):

        train_file_list = list(glob('/share/data2/body/brain/train_v2/image/iNPH*MRI*/**/*.*', recursive=True))
        val_file_list = list(glob('/share/data2/body/brain/valid_v2/image/*MRI*/**/*.*', recursive=True))
        img_file_list = train_file_list + val_file_list

        df = pd.DataFrame({'img_file': img_file_list})
        df['p_id_path'] = df.img_file.apply(lambda val: os.path.dirname(val))
        df['p_id'] = df['p_id_path'].rank(method='max').astype(int)
        df.sort_values('p_id')

        df['label_path'] = df.img_file.apply(lambda val: val.replace('image', 'label'))

        # df['valid'] = df.p_id % 5 == 1
        df['valid'] = df.img_file.str.contains('/valid')

        df.valid.value_counts()
        return df

    def __getitem__(self, index):
        image = self.df.img_file.iloc[index]
        image = np.array(Image.open(image).convert('RGB'))

        mask = self.df.label_path.iloc[index]
        mask = np.array(Image.open(mask))

        if self.ds_type=='train':
            augmented = self.aug_train(image=image, mask=mask)
        else:
            augmented = self.aug_val(image=image, mask=mask)

        image, mask = augmented['image'], augmented['mask']

        return torch.FloatTensor(image).permute([2, 0, 1]), torch.LongTensor(mask)

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':

    for sn, (a, b) in enumerate(DataSet_brain()):
        print('===', type(a), type(b), a.shape, b.shape, np.unique(b))
        if sn > 9: break
