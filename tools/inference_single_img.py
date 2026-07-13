"""
用于推理单张图像
"""
import json
import os
import sys
import argparse

import numpy as np
import torch
from PIL import Image
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter

from data_utils.augmentor import build_dataset_transformer
from data_utils.datasets import build_dataset
from hypes_yaml import yaml_utils
from metric.calculator import Calculator
from utils import train_utils
import utils.distributed_utils as utils
from utils.common import str_dict_from_tensor, remove_small_areas_based_threshold, replace_system_separator
import matplotlib.pyplot as plt
import pandas as pd

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)

mask_path = '../paper_work/imgs/mask.png'


def parse_args():
    parser = argparse.ArgumentParser(description="推理参数")
    parser.add_argument('--model_dir', type=str,
                        default='../logs/EYE-Seg/EyeNet',
                        help='模型路径')
    parser.add_argument('--img_path', type=str,
                        default='C:/my/F/origin data3/image_051.tif',
                        help='分割图像的路径')
    parser.add_argument('--save_path', type=str,
                        default='../paper_work/imgs',
                        help='存储到哪个位置')
    opt = parser.parse_args()
    return opt


def modify_config(hypes):
    """
    保险
    Args:
        hypes: 配置文件对象

    Returns: 配置文件对象

    """
    hypes['amp'] = True
    hypes['train_params']['batch_size'] = 4
    hypes['optimizer']['lr'] = 0.001  # 统一学习率
    hypes['device'] = 'cuda'
    hypes['num-classes'] = 1
    hypes['num_workers'] = 4
    if hypes['dataset']['method'] == 'DriveDataset':
        hypes['dataset']['root_dir'] = "C:\\my\\F\\眼底图像分割\\DRIVE"  # F:\\眼底图像分割\\DRIVE
        hypes['dataset']['train_expand_rate'] = 8
        hypes['train_params']['epoches'] = 80
        hypes['train_params']['save_freq'] = 160
        hypes['train_params']['train_fold_list'] = [1, 2, 3, 4, 5]
        hypes['postprocess']['threshold'] = 5
        hypes['augmentor']['mean'] = [0.38145922]
        hypes['augmentor']['std'] = [0.07928617]
        hypes['augmentor']['threshold'] = 0.01
        hypes['augmentor']['weights'] = [0, 1, 0]
        hypes['early_stop']['use'] = True
        hypes['early_stop']['args']['patience'] = 20
    if hypes['dataset']['method'] == 'ChaseDataset':
        hypes['dataset']['root_dir'] = "C:\\my\\F\\眼底图像分割\\CHASEDB1"  # F:\\眼底图像分割\\CHASEDB1
        hypes['dataset']['train_expand_rate'] = 8
        hypes['train_params']['epoches'] = 80
        hypes['train_params']['save_freq'] = 160
        hypes['train_params']['train_fold_list'] = [1, 2, 3, 4, 5]
        hypes['postprocess']['threshold'] = 5
        hypes['augmentor']['mean'] = [0.23612322]
        hypes['augmentor']['std'] = [0.10596604]
        hypes['augmentor']['threshold'] = 0.01
        hypes['augmentor']['weights'] = [0, 1, 0]
        hypes['early_stop']['use'] = True
        hypes['early_stop']['args']['patience'] = 20
    if hypes['dataset']['method'] == 'StareDataset':
        hypes['dataset']['root_dir'] = "C:\\my\\F\\眼底图像分割\\Stare"  # F:\\眼底图像分割\\Stare
        hypes['dataset']['train_expand_rate'] = 8
        hypes['train_params']['epoches'] = 80
        hypes['train_params']['save_freq'] = 160
        hypes['train_params']['train_fold_list'] = [1, 2, 3, 4, 5]
        hypes['postprocess']['threshold'] = 5
        hypes['augmentor']['mean'] = [0.42517377]
        hypes['augmentor']['std'] = [0.09725328]
        hypes['augmentor']['threshold'] = 0.01
        hypes['augmentor']['weights'] = [0, 1, 0]
        hypes['early_stop']['use'] = True
        hypes['early_stop']['args']['patience'] = 20
    if hypes['dataset']['method'] == 'HrfDataset':
        hypes['dataset']['root_dir'] = "C:\\my\\F\\眼底图像分割\\HRF"  # F:\\眼底图像分割\\HRF
        hypes['dataset']['train_expand_rate'] = 8
        hypes['train_params']['epoches'] = 80
        hypes['train_params']['save_freq'] = 160
        hypes['train_params']['train_fold_list'] = [1, 2, 3, 4, 5]
        hypes['postprocess']['threshold'] = 20
        hypes['augmentor']['mean'] = [0.23687161]
        hypes['augmentor']['std'] = [0.06492036]
        hypes['augmentor']['threshold'] = 0.01
        hypes['augmentor']['weights'] = [0, 1, 0]
        hypes['early_stop']['use'] = True
        hypes['early_stop']['args']['patience'] = 20
        if hypes['name'] in ['R2UNet', 'DenseUNet', 'IterNet', 'MCDAUNet', 'U2Net', 'MGANet']:
            hypes['train_params']['batch_size'] = 4
            hypes['augmentor']['args']['crop_size'] = 384

        if hypes['name'] in ['TransUNet']:
            hypes['train_params']['batch_size'] = 4
            hypes['augmentor']['args']['crop_size'] = 384
            hypes['model']['args']['img_dim'] = 384
            hypes['model']['args']['patch_size'] = 8
    if hypes['dataset']['method'] == 'EyeSegDataset':
        hypes['dataset']['root_dir'] = "C:\\my\\F\\眼底图像分割\\EYE-Seg"  # F:\\眼底图像分割\\EYE-Seg
        hypes['dataset']['train_expand_rate'] = 8
        hypes['train_params']['epoches'] = 80
        hypes['train_params']['save_freq'] = 160
        hypes['train_params']['train_fold_list'] = [1, 2, 3, 4, 5]
        hypes['postprocess']['threshold'] = 20
        hypes['augmentor']['mean'] = [0.28825587]
        hypes['augmentor']['std'] = [0.09410577]
        hypes['augmentor']['threshold'] = 0.01
        hypes['augmentor']['weights'] = [0, 1, 0]
        hypes['early_stop']['use'] = True
        hypes['early_stop']['args']['patience'] = 20
        if hypes['name'] in ['R2UNet', 'DenseUNet', 'IterNet', 'MCDAUNet', 'U2Net', 'MGANet']:
            hypes['train_params']['batch_size'] = 4
            hypes['augmentor']['args']['crop_size'] = 384

        if hypes['name'] in ['TransUNet']:
            hypes['train_params']['batch_size'] = 4
            hypes['augmentor']['args']['crop_size'] = 384
            hypes['model']['args']['img_dim'] = 384
            hypes['model']['args']['patch_size'] = 8

    return hypes


def get_process_img(img_path, hypes):
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    transforms = build_dataset_transformer(hypes=hypes, train=False)
    img, mask = transforms(img, mask)
    return img


def main(args, hypes):
    model_name = hypes['name']
    device = torch.device(hypes['device'])
    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    print('-----------------Load Pretrained Model------------------')
    load_path = os.path.join(args.model_dir, f'fold-{1}')
    _, model, _, _, _, _ = train_utils.load_saved_model(load_path, model, None, None, None)
    model.to(device)
    print('-----------------Eval Step------------------')
    model.eval()
    torch.set_grad_enabled(False)
    image = get_process_img(args.img_path, hypes)
    image = image.unsqueeze(0).to(device)
    output = sliding_window_inference(inputs=image, roi_size=(
        hypes['augmentor']['args']['crop_size'], hypes['augmentor']['args']['crop_size']),
                                      sw_batch_size=1,
                                      predictor=model, overlap=0.25)  # 使用滑动窗口进行推理

    output = output['out']
    # ---- 保存推理图像 ----
    prediction = output.argmax(1).squeeze(0)
    prediction = prediction.to("cpu").numpy().astype(np.uint8)
    # 将前景对应的像素值改成255(白色)
    prediction[prediction == 1] = 255
    prediction[prediction == 0] = 0
    # 将不敢兴趣的区域像素设置成0(黑色)
    roi_img = Image.open(replace_system_separator(mask_path)).convert('L')
    roi_img = np.array(roi_img)
    prediction[roi_img == 0] = 0
    mask = Image.fromarray(prediction)
    file_name = os.path.basename(args.img_path)

    mask.save(f"{os.path.join(args.save_path, f'{model_name}_' + os.path.splitext(file_name)[0])}.png")


if __name__ == '__main__':
    list = [
        # ('../logs/EYE-Seg/EyeNet', 'C:/my/F/origin data3/image_052.tif', '../paper_work/imgs'),
        # ('../logs/EYE-Seg/CENet', 'C:/my/F/origin data3/image_052.tif', '../paper_work/imgs'),
        # ('../logs/EYE-Seg/SwinUNet', 'C:/my/F/origin data3/image_052.tif', '../paper_work/imgs'),
        # ('../logs/EYE-Seg/U2Net', 'C:/my/F/origin data3/image_052.tif', '../paper_work/imgs'),
        #
        # ('../logs/EYE-Seg/EyeNet', 'C:/my/F/origin data3/image_054.tif', '../paper_work/imgs'),
        # ('../logs/EYE-Seg/CENet', 'C:/my/F/origin data3/image_054.tif', '../paper_work/imgs'),
        # ('../logs/EYE-Seg/SwinUNet', 'C:/my/F/origin data3/image_054.tif', '../paper_work/imgs'),
        # ('../logs/EYE-Seg/U2Net', 'C:/my/F/origin data3/image_054.tif', '../paper_work/imgs'),
        #
        # ('../logs/EYE-Seg/EyeNet', 'C:/my/F/origin data3/image_056.tif', '../paper_work/imgs'),
        # ('../logs/EYE-Seg/CENet', 'C:/my/F/origin data3/image_056.tif', '../paper_work/imgs'),
        # ('../logs/EYE-Seg/SwinUNet', 'C:/my/F/origin data3/image_056.tif', '../paper_work/imgs'),
        # ('../logs/EYE-Seg/U2Net', 'C:/my/F/origin data3/image_056.tif', '../paper_work/imgs'),
        #
        # ('../logs/EYE-Seg/EyeNet', 'C:/my/F/origin data3/image_058.tif', '../paper_work/imgs'),
        # ('../logs/EYE-Seg/CENet', 'C:/my/F/origin data3/image_058.tif', '../paper_work/imgs'),
        # ('../logs/EYE-Seg/SwinUNet', 'C:/my/F/origin data3/image_058.tif', '../paper_work/imgs'),
        # ('../logs/EYE-Seg/U2Net', 'C:/my/F/origin data3/image_058.tif', '../paper_work/imgs'),

        ('../logs/EYE-Seg/EyeNet', 'C:/my/F/origin data3/image_068.tif', '../paper_work/imgs'),
        ('../logs/EYE-Seg/CENet', 'C:/my/F/origin data3/image_068.tif', '../paper_work/imgs'),
        ('../logs/EYE-Seg/SwinUNet', 'C:/my/F/origin data3/image_068.tif', '../paper_work/imgs'),
        ('../logs/EYE-Seg/U2Net', 'C:/my/F/origin data3/image_068.tif', '../paper_work/imgs'),
    ]
    if len(list) == 0:
        print('-----------------Analyze Config File------------------')
        args = parse_args()
        hypes = yaml_utils.load_yaml(None, args)
        hypes = modify_config(hypes)
        main(args, hypes)
    else:
        print('-----------------Analyze Config File------------------')
        for item in list:
            args = parse_args()
            args.model_dir, args.img_path, args.save_path = item
            hypes = yaml_utils.load_yaml(None, args)
            hypes = modify_config(hypes)
            main(args, hypes)
