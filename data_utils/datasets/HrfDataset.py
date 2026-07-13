import json
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from utils.common import replace_system_separator


class HrfDataset(Dataset):
    """
    数据集: HRF
    """
    def __init__(self, hypes, train: bool, transforms=None, fold=0):
        super(HrfDataset, self).__init__()
        self.transforms = transforms
        root = hypes['dataset']['root_dir']
        self.train_expand_rate = hypes['dataset']['train_expand_rate']
        self.train = train
        if fold == 0:
            data_root = root
            img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".jpg")]
            self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
            self.label = [os.path.join(data_root, "label", i.split(".")[0] + ".tif")
                           for i in img_names]
            # check files
            for i in self.label:
                if os.path.exists(i) is False:
                    raise FileNotFoundError(f"file {i} does not exists.")

            self.mask = [os.path.join(data_root, "mask", i.split(".")[0] + "_mask.tif") for i in img_names]
            # check files
            for i in self.mask:
                if os.path.exists(i) is False:
                    raise FileNotFoundError(f"file {i} does not exists.")
        else:
            # 使用fold
            json_path = os.path.join(root, 'fold.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                json_dict = json.load(f)
            flag = "train" if train else "test"
            self.img_list = [os.path.join(root, ab_path) for ab_path in json_dict[f'fold-{fold}'][f'{flag}_images']]
            self.label = [os.path.join(root, ab_path) for ab_path in json_dict[f'fold-{fold}'][f'{flag}_label']]
            self.mask = [os.path.join(root, ab_path) for ab_path in json_dict[f'fold-{fold}'][f'{flag}_mask']]

    def __getitem__(self, idx):
        if self.train:
            idx = idx % len(self.img_list)
        img = Image.open(replace_system_separator(self.img_list[idx])).convert('RGB')
        label = Image.open(replace_system_separator(self.label[idx])).convert('L') # 图像转换为灰度
        label = np.array(label) / 255  # 血管被标记成1
        mask = Image.open(replace_system_separator(self.mask[idx])).convert('L')  # 外面一圈的掩码, 黑的是0, 中间白的是255
        # mask作者制作时有点问题, 所以这里我们初步做一个二值化
        mask = np.array(mask)
        mask[mask < 128] = 0
        mask[mask >= 128] = 255
        mask = 255 - mask  # 反过来, 外面是255, 中间是0
        mask = np.clip(label + mask, a_min=0, a_max=255)  # 外面掩膜是255, 背景是0, 血管是1

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        if self.train:
            return len(self.img_list) * self.train_expand_rate
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

    @staticmethod
    def fold_dataset_split(hypes):
        """
        切分数据集, 默认使用五折交叉检验, 每个dataset都需要实现这个方法
        Args:
            hypes: 配置文件
        """
        fold_num = hypes['dataset']['fold_num']
        kf = KFold(n_splits=fold_num, shuffle=False)  # 初始化KFold
        dataset_path = hypes['dataset']['root_dir']

        image_size = len(os.listdir(os.path.join(dataset_path, 'images')))
        idx_list = [i for i in range(image_size)]
        # 得到相对位置
        images_list = np.array(
            [os.path.join('images', file) for file in os.listdir(os.path.join(dataset_path, 'images'))]
        )
        # mask_list = np.array(os.listdir(os.path.join(train_path, 'mask')) + os.listdir(os.path.join(test_path, 'mask')))
        mask_list = np.array(
            [os.path.join('mask', file.split(".")[0] + "_mask.tif") for file in os.listdir(os.path.join(dataset_path, 'images'))]
        )
        # manual_list = np.array(os.listdir(os.path.join(train_path, '1st_manual')) + os.listdir(
        #     os.path.join(test_path, '1st_manual')))

        label_list = np.array(
            [os.path.join('label', file.split(".")[0] + ".tif") for file in os.listdir(os.path.join(dataset_path, 'images'))]
        )

        json_dict = {}
        for index, (train_index, test_index) in enumerate(kf.split(idx_list)):  # 调用split方法切分数据
            json_dict[f'fold-{index + 1}'] = {}
            json_dict[f'fold-{index + 1}']['train_images'] = list(images_list[train_index])
            json_dict[f'fold-{index + 1}']['train_mask'] = list(mask_list[train_index])
            json_dict[f'fold-{index + 1}']['train_label'] = list(label_list[train_index])

            json_dict[f'fold-{index + 1}']['test_images'] = list(images_list[test_index])
            json_dict[f'fold-{index + 1}']['test_mask'] = list(mask_list[test_index])
            json_dict[f'fold-{index + 1}']['test_label'] = list(label_list[test_index])
            print('train_index:%s , test_index: %s ' % (train_index, test_index))
        file_txt = json.dumps(json_dict)
        with open(os.path.join(dataset_path, 'fold.json'), 'w', encoding='utf-8') as f:
            f.write(file_txt)


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
