import random

import data_utils.augmentor.trans as T


def check_foreground_area(target, threshold=0.05):
    """
    我们保证前景区域大于 threshold 才认为这是一个合格的样本
    Args:
        threshold: 阈值
        target:
    Returns:
    """
    pixels_mask = ((target > 0) & (target < 255)).float()

    # 计算值为255的像素总数
    pixels_num = pixels_mask.sum()

    # 计算总像素数
    total_num = target.numel()

    # 计算前景区域的比例
    return (pixels_num / total_num).item() >= threshold


class CommonPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class CommonPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class EnhanceGrayPresetTrain:
    def __init__(self, crop_size, hflip_prob=0.5, vflip_prob=0.5, gauss_prob=0.2, rotate_prob=0.5, angle=20,
                 mean=[0.38145922], std=[0.07928617], threshold=0.05, weights=[0.299, 0.587, 0.114], **kwargs):
        self.threshold = threshold
        trans = [T.RandomCrop2(crop_size),
                 T.ToNumpy()]
        # 随机高斯模糊 可能会降低精度
        # trans.append(T.AddGaussianNoise(prob=gauss_prob))
        # 随机旋转
        trans.append(T.RandomReflectRotate(angle=angle, prob=rotate_prob))
        trans.extend([
            T.ConvertToGrayscale(weights=weights),
            T.Clahe(),
            T.GammaCorrection(),
            T.ToPIL()])
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        while True:
            trans_img, trans_target = self.transforms(img, target)
            if check_foreground_area(trans_target, self.threshold):
                break
        return trans_img, trans_target


class EnhanceGrayPresetEval:
    def __init__(self, mean=[0.38145922], std=[0.07928617], **kwargs):
        self.transforms = T.Compose([
            T.ToNumpy(),
            T.ConvertToGrayscale(),
            T.Clahe(),
            T.GammaCorrection(),
            T.ToPIL(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class nnUnetPresetTrain:
    def __init__(self,crop_size, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, threshold=0.05, **kwargs):
        self.threshold = threshold
        trans = [
            T.RandomCrop2(crop_size),
            T.ToNumpy(),
            T.nnUnetTransformer(),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        while True:
            trans_img, trans_target = self.transforms(img, target)
            if check_foreground_area(trans_target, self.threshold):
                break
        return trans_img, trans_target


class nnUnetPresetEval:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


def get_nnunet_transformer(train, hypes):
    """
    增强灰度数据集的数据增强
    Args:
        hypes: 配置文件
        train: 是否是训练集

    Returns:

    """
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    base_size = 565
    crop_size = 480

    if train:
        return nnUnetPresetTrain(**hypes['augmentor']['args'])
    else:
        return nnUnetPresetEval(**hypes['augmentor']['args'])

def get_common_transformer(train, hypes):
    """
    普通数据集的数据增强
    输入三通道图像
    输出三通道图像
    train:
        RandomResize
        RandomHorizontalFlip
        RandomVerticalFlip
        RandomCrop
        ToTensor
        Normalize
    Args:
        hypes: 配置文件
        train: 是否是训练集
    Returns:

    """
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    base_size = 565
    crop_size = 480

    if train:
        return CommonPresetTrain(**hypes['augmentor']['args'])
    else:
        return CommonPresetEval(**hypes['augmentor']['args'])


def get_enhance_gray_transformer(train, hypes):
    """
    增强灰度数据集的数据增强
    Args:
        hypes: 配置文件
        train: 是否是训练集

    Returns:

    """
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    base_size = 565
    crop_size = 480

    if train:
        return EnhanceGrayPresetTrain(**hypes['augmentor']['args'])
    else:
        return EnhanceGrayPresetEval(**hypes['augmentor']['args'])


class EnhanceGrayPresetTrain1:
    def __init__(self, crop_size, hflip_prob=0.5, vflip_prob=0.5, gauss_prob=0.2, rotate_prob=0.5, angle=20,
                 mean=[0.38145922], std=[0.07928617], threshold=0.05, weights=[0.299, 0.587, 0.114], **kwargs):
        self.threshold = threshold
        trans = [T.RandomCrop2(crop_size),
                 T.ToNumpy()]
        trans.extend([
            T.ConvertToGrayscale(weights=weights),
            T.Clahe(),
            T.GammaCorrection(),
            T.ToPIL()])
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        while True:
            trans_img, trans_target = self.transforms(img, target)
            if check_foreground_area(trans_target, self.threshold):
                break
        return trans_img, trans_target

class EnhanceGrayPresetTrain2:
    def __init__(self, crop_size, hflip_prob=0.5, vflip_prob=0.5, gauss_prob=0.2, rotate_prob=0.5, angle=20,
                 mean=[0.38145922], std=[0.07928617], threshold=0.05, weights=[0.299, 0.587, 0.114], **kwargs):
        self.threshold = threshold
        trans = [T.RandomCrop2(crop_size),
                 T.ToNumpy()]
        # 随机高斯模糊 可能会降低精度
        trans.append(T.AddGaussianNoise(prob=gauss_prob))
        trans.extend([
            T.ConvertToGrayscale(weights=weights),
            T.Clahe(),
            T.GammaCorrection(),
            T.ToPIL()])
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        while True:
            trans_img, trans_target = self.transforms(img, target)
            if check_foreground_area(trans_target, self.threshold):
                break
        return trans_img, trans_target

class EnhanceGrayPresetTrain3:
    def __init__(self, crop_size, hflip_prob=0.5, vflip_prob=0.5, gauss_prob=0.2, rotate_prob=0.5, angle=20,
                 mean=[0.38145922], std=[0.07928617], threshold=0.05, weights=[0.299, 0.587, 0.114], **kwargs):
        self.threshold = threshold
        trans = [T.RandomCrop2(crop_size),
                 T.ToNumpy()]
        # 随机旋转
        trans.append(T.RandomReflectRotate(angle=angle, prob=rotate_prob))
        trans.extend([
            T.ConvertToGrayscale(weights=weights),
            T.Clahe(),
            T.GammaCorrection(),
            T.ToPIL()])
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        while True:
            trans_img, trans_target = self.transforms(img, target)
            if check_foreground_area(trans_target, self.threshold):
                break
        return trans_img, trans_target

class EnhanceGrayPresetTrain4:
    def __init__(self, crop_size, hflip_prob=0.5, vflip_prob=0.5, gauss_prob=0.2, rotate_prob=0.5, angle=20,
                 mean=[0.38145922], std=[0.07928617], threshold=0.05, weights=[0.299, 0.587, 0.114], **kwargs):
        self.threshold = threshold
        trans = [T.RandomCrop2(crop_size),
                 T.ToNumpy()]
        # 随机高斯模糊 会降低0.4个点
        # trans.append(T.AddGaussianNoise(prob=gauss_prob))
        # 随机旋转 会增加0.1个点
        trans.append(T.RandomReflectRotate(angle=angle, prob=rotate_prob))
        trans.extend([
            T.ConvertToGrayscale(weights=weights),
            T.Clahe(),
            T.GammaCorrection(),
            T.ToPIL()])
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        while True:
            trans_img, trans_target = self.transforms(img, target)
            if check_foreground_area(trans_target, self.threshold):
                break
        return trans_img, trans_target

def get_enhance_gray_transformer1(train, hypes):
    """
    增强灰度数据集的数据增强
    Args:
        hypes: 配置文件
        train: 是否是训练集

    Returns:

    """
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    base_size = 565
    crop_size = 480

    if train:
        return EnhanceGrayPresetTrain1(**hypes['augmentor']['args'])
    else:
        return EnhanceGrayPresetEval(**hypes['augmentor']['args'])

def get_enhance_gray_transformer2(train, hypes):
    """
    增强灰度数据集的数据增强
    Args:
        hypes: 配置文件
        train: 是否是训练集

    Returns:

    """
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    base_size = 565
    crop_size = 480

    if train:
        return EnhanceGrayPresetTrain2(**hypes['augmentor']['args'])
    else:
        return EnhanceGrayPresetEval(**hypes['augmentor']['args'])

def get_enhance_gray_transformer3(train, hypes):
    """
    增强灰度数据集的数据增强
    Args:
        hypes: 配置文件
        train: 是否是训练集

    Returns:

    """
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    base_size = 565
    crop_size = 480

    if train:
        return EnhanceGrayPresetTrain3(**hypes['augmentor']['args'])
    else:
        return EnhanceGrayPresetEval(**hypes['augmentor']['args'])

def get_enhance_gray_transformer4(train, hypes):
    """
    增强灰度数据集的数据增强
    Args:
        hypes: 配置文件
        train: 是否是训练集

    Returns:

    """
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    base_size = 565
    crop_size = 480

    if train:
        return EnhanceGrayPresetTrain4(**hypes['augmentor']['args'])
    else:
        return EnhanceGrayPresetEval(**hypes['augmentor']['args'])