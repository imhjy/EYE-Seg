"""
计算数据集中的均值和标准差
"""

import os
from PIL import Image
import numpy as np

def convert_to_grayscale(image, weights=[0.299, 0.587, 0.114]):
    """
    自定义RGB权重进行灰度化
    Args:
        image: 三通道图像
        weights: RGB每个通道的权重, 相加必须为1

    Returns:

    """
    height, width = image.shape[:2]

    # 创建一个空白的灰度图像
    grayscale_image = np.zeros((height, width), dtype=np.uint8)

    # 计算RGB通道的加权平均值，并将其赋值给灰度图像的每个像素
    image_float = image.astype(np.float32)
    grayscale_image = np.dot(image_float[..., :3], weights).astype(np.uint8)

    # cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 默认灰度, 权重为 [0.299, 0.587, 0.114]
    return grayscale_image

def drive_origin():
    img_channels = 3
    img_dir = "D:/datasets/DRIVE/training/images"
    roi_dir = "D:/datasets/DRIVE/training/mask"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    assert os.path.exists(roi_dir), f"roi dir: '{roi_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir)]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        ori_path = os.path.join(roi_dir, img_name.replace(".tif", "_mask.gif"))
        img = np.array(Image.open(img_path)) / 255.
        roi_img = np.array(Image.open(ori_path).convert('L'))
        roi_img[roi_img < 128] = 0
        roi_img[roi_img >= 128] = 255
        # 只看掩膜部分
        img = img[roi_img == 255]
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")

def drive_transformer_gray():
    """
    灰度后的均值和标准差
    Returns:
    mean: [0.38145922]
    std: [0.07928617]
    """
    img_channels = 1
    img_dir = "D:/datasets/DRIVE/training/images"
    roi_dir = "D:/datasets/DRIVE/training/mask"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    assert os.path.exists(roi_dir), f"roi dir: '{roi_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir)]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        ori_path = os.path.join(roi_dir, img_name.replace(".tif", "_mask.gif"))
        origin_image = np.array(Image.open(img_path).convert('RGB'))
        gray_image = convert_to_grayscale(origin_image,weights=[0,1,0])
        img = np.array(gray_image) / 255.
        roi_img = np.array(Image.open(ori_path).convert('L'))
        roi_img[roi_img < 128] = 0
        roi_img[roi_img >= 128] = 255
        # 只看掩膜部分
        img = img[roi_img == 255]
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")


def hrf_transformer_gray():
    """
    灰度后的均值和标准差
    Returns:
    mean: [0.23687161]
    std: [0.06492036]
    """
    img_channels = 1
    img_dir = "D:/datasets/眼底图像分割/HRF/images"
    roi_dir = "D:/datasets/眼底图像分割/HRF/mask"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    assert os.path.exists(roi_dir), f"roi dir: '{roi_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir)]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        ori_path = os.path.join(roi_dir, img_name.replace(".jpg", "_mask.tif"))
        origin_image = np.array(Image.open(img_path).convert('RGB'))
        gray_image = convert_to_grayscale(origin_image,weights=[0,1,0]) # TODO weights 需要改动
        img = np.array(gray_image) / 255.
        roi_img = np.array(Image.open(ori_path).convert('L'))
        roi_img[roi_img < 128] = 0
        roi_img[roi_img >= 128] = 255
        # 只看掩膜部分
        img = img[roi_img == 255]
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")


def stare_transformer_gray():
    """
    灰度后的均值和标准差
    Returns:
    mean: [0.42517377]
    std: [0.09725328]
    """
    img_channels = 1
    img_dir = "D:/datasets/眼底图像分割/Stare/images"
    roi_dir = "D:/datasets/眼底图像分割/Stare/mask"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    assert os.path.exists(roi_dir), f"roi dir: '{roi_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir)]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        ori_path = os.path.join(roi_dir, img_name.replace(".ppm", ".jpg"))
        origin_image = np.array(Image.open(img_path).convert('RGB'))
        gray_image = convert_to_grayscale(origin_image,weights=[0,1,0]) # TODO weights 需要改动
        img = np.array(gray_image) / 255.
        roi_img = np.array(Image.open(ori_path).convert('L'))
        roi_img[roi_img < 128] = 0
        roi_img[roi_img >= 128] = 255
        # 只看掩膜部分
        img = img[roi_img == 255]
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")



def chase_transformer_gray():
    """
    灰度后的均值和标准差
    Returns:
    mean: [0.23612322]
    std: [0.10596604]
    """
    img_channels = 1
    img_dir = "D:/datasets/眼底图像分割/CHASEDB1/images"
    roi_dir = "D:/datasets/眼底图像分割/CHASEDB1/mask"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    assert os.path.exists(roi_dir), f"roi dir: '{roi_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir)]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        ori_path = os.path.join(roi_dir, img_name.replace(".jpg", ".jpg"))
        origin_image = np.array(Image.open(img_path).convert('RGB'))
        gray_image = convert_to_grayscale(origin_image,weights=[0,1,0]) # TODO weights 需要改动
        img = np.array(gray_image) / 255.
        roi_img = np.array(Image.open(ori_path).convert('L'))
        roi_img[roi_img < 128] = 0
        roi_img[roi_img >= 128] = 255
        # 只看掩膜部分
        img = img[roi_img == 255]
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")


def eyeseg_transformer_gray():
    """
    灰度后的均值和标准差
    Returns:
    mean: [0.28825587]
    std: [0.09410577]
    """
    img_channels = 1
    img_dir = "F:/眼底图像分割/EYE-Seg/images"
    roi_dir = "F:/眼底图像分割/EYE-Seg/mask"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    assert os.path.exists(roi_dir), f"roi dir: '{roi_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir)]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        ori_path = os.path.join(roi_dir, img_name.replace(".tif", ".png")) # mask图像
        origin_image = np.array(Image.open(img_path).convert('RGB'))
        gray_image = convert_to_grayscale(origin_image,weights=[0,1,0]) # TODO weights 需要改动
        img = np.array(gray_image) / 255.
        roi_img = np.array(Image.open(ori_path).convert('L'))
        roi_img[roi_img < 128] = 0
        roi_img[roi_img >= 128] = 255
        # 只看掩膜部分
        img = img[roi_img == 255]
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")




def test_transformer_gray():
    """
    灰度后的均值和标准差
    Returns:
    mean: [0.36263427]
    std: [0.1129151]
    """
    img_channels = 1
    img_dir = "C:\\Users\\24026\\Desktop\\test_db\\images\\Right"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir)]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        origin_image = np.array(Image.open(img_path).convert('RGB'))
        gray_image = convert_to_grayscale(origin_image,weights=[0,1,0]) # TODO weights 需要改动
        img = np.array(gray_image) / 255.
        cumulative_mean += img.flatten().mean(axis=0)
        cumulative_std += img.flatten().std(axis=0)

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == '__main__':
    eyeseg_transformer_gray()
