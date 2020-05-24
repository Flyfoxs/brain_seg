from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from file_cache import *


class DataSet_brain(Dataset):

    def __init__(self, ds_type='train'):
        self.image_size = 256
        self.ds_type = ds_type

        df = self.get_df()

        if ds_type == 'train':
            print('========', ds_type)
            print(df.valid.value_counts())
            self.df = df.loc[df.valid == False]
        elif ds_type == 'valid':
            self.df = df.loc[df.valid == True]

    #         self.transforms = transforms.Compose([transforms.ToTensor(),
    #                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                       transforms.Resize(224)
    #                      ])

    def transform(self, image, mask):
        # print(image.shape, mask.shape)
        # Resize
        resize_img = transforms.Compose([transforms.ToTensor(),
                                         # transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
                                         transforms.ToPILImage(),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         ])

        resize_label = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize((224, 224)),

                                           ])
        image = resize_img(image)
        mask = resize_label(mask)
        mask = np.array(mask).astype(int)

        return np.array(image), mask

    def get_p_cnt(self, file):
        label = nib.load(file).get_fdata()
        label = np.where(label > 0, 1, 0)
        label = label.sum(axis=0).sum(axis=0)
        return label

    @lru_cache()
    def get_df(self):

        img_file_list = glob('/share/data2/body/brain/train/image/**/*.*', recursive=True)

        df = pd.DataFrame({'img_file': img_file_list})
        df['p_id_path'] = df.img_file.apply(lambda val: os.path.dirname(val))
        df['p_id'] = df['p_id_path'].rank(method='max').astype(int)
        df.sort_values('p_id')

        df['label_path'] = df.img_file.apply(lambda val: val.replace('image', 'label'))

        df['valid'] = df.p_id % 5 == 1

        df.valid.value_counts()
        return df

    def __getitem__(self, index):
        img = self.df.img_file.iloc[index]
        # print(img)
        img = cv2.imread(img)
        # print(img)
        img = (img - img.min()) / (img.max() - img.min())

        # slice_sn = self.df.slice_sn.iloc[index]-1
        # print(slice_sn)

        label = self.df.label_path.iloc[index]
        # print(label)
        label = cv2.imread(label)[:, :, 0]

        # print(img.dtype, label.dtype)
        # return img, label
        return self.transform(img.astype(np.float32), label.astype(np.uint8))

    def __len__(self):
        return len(self.df)

if __name__ == '__main__':

    for sn, (a, b) in enumerate(DataSet_brain()):
        print('===', type(a), type(b), a.shape, b.shape, np.unique(b))
        if sn > 9: break

