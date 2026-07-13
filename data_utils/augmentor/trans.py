import math

import cv2
import numpy as np
import random
import albumentations as A
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):

        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class RandomCrop2(object):
    def __init__(self, size):
        if isinstance(size, tuple) or isinstance(size, list):
            self.size = size  # 这个size是个元组
        else:
            self.size = (size, size)

    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image) # (C,H,W)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class ToNumpy(object):
    """
    将PIL的数据转换为Numpy,以便处理
    """
    def __call__(self, image, target):
        rgb_image = np.array(image)
        target = np.array(target)
        return rgb_image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ConvertToGrayscale(object):
    def __init__(self, weights=None):
        if weights is None:
            weights = [0.299, 0.587, 0.114]
        self.weights = weights

    def __call__(self, image, target):
        """
        自定义RGB权重进行灰度化
        Args:
            image: 三通道图像
            weights: RGB每个通道的权重, 相加必须为1

        Returns:

        """
        # 计算RGB通道的加权平均值，并将其赋值给灰度图像的每个像素
        image_float = image.astype(np.float32)
        grayscale_image = np.dot(image_float[..., :3], self.weights).astype(np.uint8)

        # cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 默认灰度, 权重为 [0.299, 0.587, 0.114]
        return grayscale_image, target


class Clahe(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        # self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image, target):
        """
        限制对比度自适应直方图均衡化
        """
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return clahe.apply(image), target


class GammaCorrection(object):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def __call__(self, image, target):
        """
        伽玛校正
        """
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table), target


class ToPIL:
    def __call__(self, image, target):
        """
        灰度的 ndarray 转为 PIL.Image
        """
        image = Image.fromarray(image)
        if isinstance(target, np.ndarray):
            target = Image.fromarray(target)
        return image, target


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=-1,prob=0.2):
        self.mean = mean
        self.variance = variance
        if amplitude == -1:
            amplitude = random.randint(1, 10) # 随机1到10之间一个数
        self.amplitude = amplitude
        self.prob = prob

    def __call__(self, image, target):
        """
        添加高斯噪音
        """
        if random.random() >= self.prob:
            return image, target

        img = np.array(image)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255  # 避免有值超过255而反转
        return img, target


class RandomReflectRotate(object):
    """
    带镜像填充的随机旋转, 避免引入黑色伪影
    """

    def __init__(self, angle=20, prob=0.5):
        self.angle = angle
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() >= self.prob:
            return image, mask

        angle = random.uniform(-self.angle, self.angle)
        # 转换为Tensor
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()  # (C,H,W) [0-255]
        mask_tensor = torch.from_numpy(np.array(mask)).float()  # (H,W) [0-255]

        # 计算动态填充尺寸
        angle_rad = math.radians(abs(angle))
        max_dim = max(img_tensor.shape[1], img_tensor.shape[2])
        pad_size = int(math.ceil(max_dim * math.tan(angle_rad)))  # 增加1像素冗余

        # 反射填充
        img_padded = torch.nn.functional.pad(img_tensor, (pad_size,) * 4, mode='reflect')
        mask_padded = torch.nn.functional.pad(mask_tensor.unsqueeze(0), (pad_size,) * 4, mode='reflect')[0]

        # 执行旋转
        rotated_img = F.rotate(img_padded, angle,
                               interpolation=F.InterpolationMode.BILINEAR)
        rotated_mask = F.rotate(mask_padded.unsqueeze(0), angle,
                                interpolation=F.InterpolationMode.NEAREST)[0].long()

        # 裁剪回原尺寸
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        rotated_img = rotated_img[:, pad_size:-pad_size, pad_size:-pad_size]
        rotated_mask = rotated_mask[pad_size:-pad_size, pad_size:-pad_size]

        # 转回Numpy
        return (
            rotated_img.permute(1, 2, 0).byte().numpy(),  # (H,W,C)
            Image.fromarray(rotated_mask.byte().numpy().astype(np.uint8))
        )

class nnUnetTransformer(object):
    def __init__(self):
        pass

    def __call__(self, image: np.ndarray, label: np.ndarray):
        """
        添加高斯噪音
        """
        trans = A.Compose([
            A.Affine(  # 大小变化, 变小边界填充0，变大裁剪中心，最终大小不变
                scale=(0.7, 1.4),
                translate_percent=(0, 0),
                rotate=(0, 0),
                shear=(0, 0),
                p=0.2
            ),
            A.VerticalFlip(p=0.2),
            A.HorizontalFlip(p=0.2),
            A.Affine(  # x,y轴随机移动
                scale=(1, 1),
                translate_percent=(-0.05, 0.05),
                rotate=(0, 0),
                shear=(0, 0),
                p=0.2
            ),
            A.Affine(
                scale=(1, 1),
                translate_percent=(0, 0),
                rotate=(-45, 45),
                shear=(0, 0),
                p=0.2
            ),
            A.GaussNoise(
                std_range=(0, 0.1),
                mean_range=(0, 0),
                per_channel=True,
                p=0.1
            ),
            A.GaussianBlur(
                blur_limit=0,
                sigma_limit=(0.5, 1.0),
                p=0.2
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.25),
                contrast_limit=(0, 0),
                brightness_by_max=True,
                ensure_safe_range=False,
                p=0.15
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(0, 0),
                contrast_limit=(-0.25, 0.25),
                brightness_by_max=True,
                ensure_safe_range=False,
                p=0.15
            ),
            A.Downscale(
                scale_range=(0.5, 1),
                interpolation_pair={"upscale": 0, "downscale": 0}
            ),
            A.RandomGamma(
                gamma_limit=(70, 150)
            )
        ])
        result = trans(image=image, mask=label)
        return result['image'], result['mask']