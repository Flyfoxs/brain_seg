from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from file_cache import *
import torch
from .transform import *
import torch.nn.functional as F
from fastai.vision import imagenet_stats
import albumentations as A
from   albumentations.pytorch.transforms import ToTensorV2
#https://debuggercafe.com/image-augmentation-using-pytorch-and-albumentations/


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

        self.transforms_train = transforms.Compose([
                        transforms.ToTensor(),
                        Resize(mode='bilinear'),
                      #transforms.Normalize(*imagenet_stats),
                     ])

        self.transforms_valid = transforms.Compose([
                     Resize(mode='nearest'),
        ])


    @lru_cache()
    def get_df(self):

        img_file_list = glob('/share/data2/body/brain/train/image/*MRI*/**/*.*', recursive=True)

        df = pd.DataFrame({'img_file': img_file_list})
        df['p_id_path'] = df.img_file.apply(lambda val: os.path.dirname(val))
        df['p_id'] = df['p_id_path'].rank(method='max').astype(int)
        df.sort_values('p_id')

        df['label_path'] = df.img_file.apply(lambda val: val.replace('image', 'label'))

        df['valid'] = df.p_id % 5 == 1

        df.valid.value_counts()
        return df

    def __getitem__(self, index):

        image = self.df.img_file.iloc[index]
        image = Image.open(image).convert('RGB')
        image = self.transforms_train(image)

        label = self.df.label_path.iloc[index]
        label = cv2.imread(label, 0)
        label = torch.Tensor(np.stack([label]*3))
        label = self.transforms_valid(label)
        return torch.tensor(image, dtype=torch.float), torch.tensor(label)[0].round().long()


    def __len__(self):
        return len(self.df)

if __name__ == '__main__':

    for sn, (a, b) in enumerate(DataSet_brain()):
        print('===', type(a), type(b), a.shape, b.shape, np.unique(b))
        if sn > 9: break

