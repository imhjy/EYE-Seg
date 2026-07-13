import json
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from utils.common import replace_system_separator


class OdcSegDataset(Dataset):
    """
    数据集: EYE-Seg
    """

    def __init__(self, hypes, train: bool, transforms=None, fold=0):
        super(OdcSegDataset, self).__init__()
        self.transforms = transforms
        root = hypes['dataset']['root_dir']
        self.train_expand_rate = hypes['dataset']['train_expand_rate']
        self.train = train
        if fold == 0:
            data_root = root
            img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".png")]
            self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
            self.label = [os.path.join(data_root, "label", i)
                          for i in img_names]
            # check files
            for i in self.label:
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

    def __getitem__(self, idx):
        if self.train:
            idx = idx % len(self.img_list)
        img = Image.open(replace_system_separator(self.img_list[idx])).convert('RGB')
        label = Image.open(replace_system_separator(self.label[idx])).convert('L')  # 图像转换为灰度
        # 创建全0的掩膜
        label = np.array(label)
        mask = np.zeros_like(label, dtype=np.uint8)

        # 假设经过测试得到的灰度值对应关系：
        # 白色 -> 255（对应原始RGB白色）
        # 红色 -> 76（由RGB红色[255,0,0]转换而来，0.299 * 255 ≈ 76）
        # 黑色 -> 0

        # 标记黑色区域（灰度值0-20）
        mask[(label >= 0) & (label <= 20)] = 0

        # 标记红色区域（灰度值21-150）
        mask[(label > 20) & (label <= 150)] = 1

        # 标记白色区域（灰度值151-255）
        mask[label > 150] = 2

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

        label_list = np.array(
            [os.path.join('label', file.split(".")[0] + ".png") for file in
             os.listdir(os.path.join(dataset_path, 'images'))]
        )

        json_dict = {}
        for index, (train_index, test_index) in enumerate(kf.split(idx_list)):  # 调用split方法切分数据
            json_dict[f'fold-{index + 1}'] = {}
            json_dict[f'fold-{index + 1}']['train_images'] = list(images_list[train_index])
            json_dict[f'fold-{index + 1}']['train_label'] = list(label_list[train_index])

            json_dict[f'fold-{index + 1}']['test_images'] = list(images_list[test_index])
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
