from file_cache import *
from torchvision import transforms
from torchvision import transforms as TF

from torch import *
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class DataSet_brain(Dataset):

    def __init__(self):

        self.df = self.get_df()

        # if ds_type == 'train':
        #     print('========', ds_type)
        #     print(df.valid.value_counts())
        #     self.df = df.loc[df.valid == False]
        # elif ds_type == 'valid':
        #     self.df = df.loc[df.valid == True]
        #
        #

    def get_p_cnt(self, file):
        label = nib.load(file).get_fdata()
        label = np.where(label > 0, 1, 0)
        label = label.sum(axis=0).sum(axis=0)
        return label

    @lru_cache()
    def get_df(self):
        root = '/share/data2/body/brain/NPH_PROCESSED'
        file_list = glob(f'{root}/iNPH_*_PROCESSED/*/*.nii.gz')
        df_list = []
        file_list = [file for file in file_list if ' ' not in os.path.basename(file)]
        # print(len(file_list))
        for file in file_list:
            label_cnt = self.get_p_cnt(file)
            # print(os.path.dirname(file))
            fold = [fold for fold in glob(f'{os.path.dirname(file)}/*') if os.path.isdir(fold)][0]
            for dcm_file in glob(f'{fold}/*'):
                slice_sn = int(dcm_file.split('_')[-1].split('.')[0]) - 1
                df_list.append({'input_path': dcm_file,
                                'label_path': file,
                                'slice_sn': slice_sn,
                                'p_cnt': label_cnt[slice_sn],
                                })
            # print(glob(f'{file}/*_Processed.nii.gz')[0])
        df = pd.DataFrame(df_list)
        df = df.sample(frac=1, random_state=2020)
        df['p_id'] = df.label_path.rank()
        df['valid'] = df.input_path.rank() <= len(df) / 5
        # df = df.loc[df.p_cnt>0]
        return df

    def __getitem__(self, index):
        img_path = self.df.input_path.iloc[index]
        # print(img)
        img = pydicom.dcmread(img_path, force=True).pixel_array.astype(np.int32)
        img = (img - img.min()) / (img.max() - img.min())

        slice_sn = self.df.slice_sn.iloc[index]
        assert slice_sn >= 0, f'Slice_sn:{slice_sn}'
        p_cnt = self.df.p_cnt.iloc[index]
        # print(slice_sn)

        label = self.df.label_path.iloc[index]
        # print(label)
        label = nib.load(label).get_fdata()
        label = label[:, :, slice_sn].T

        label_path = img_path.replace('/NPH_PROCESSED/', '/train_v2/label/')
        img_path = img_path.replace('/NPH_PROCESSED/', '/train_v2/image/')

        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)

        # print(img_path)
        #img = cv2.resize(img, (224,224))
        cv2.imwrite(f'{img_path}_{slice_sn:02}_{p_cnt:05}.png', img * 255)

        #label = cv2.resize(label, (224, 224))
        cv2.imwrite(f'{label_path}_{slice_sn:02}_{p_cnt:05}.png', label)
        # print(img.dtype, label.dtype)
        # return img, label
        return img.astype(np.float32), label.astype(np.uint8)
        # return self.transform(img.astype(np.float32), label.astype(np.uint8))

    def __len__(self):
        return len(self.df)

if __name__ == '__main__':
    ds = DataSet_brain()
    for sn, (a, b) in tqdm(enumerate(ds), total=len(ds), desc='Prepare DS'):
        pass


# ds = DataSet_brain('valid')
# for sn, (a, b) in tqdm(enumerate(ds), total=len(ds)):
#     pass