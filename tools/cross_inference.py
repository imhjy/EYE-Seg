"""
交叉验证
"""
import csv
import json
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter

from data_utils.datasets import build_dataset
from hypes_yaml import yaml_utils
from metric.calculator import Calculator
from utils import train_utils
import utils.distributed_utils as utils
from utils.common import str_dict_from_tensor, remove_small_areas_based_threshold, replace_system_separator

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)


def parse_args():
    parser = argparse.ArgumentParser(description="推理参数")
    parser.add_argument('--use_model_dir', type=str,
                        default='../logs/compare/eyeseg/eyenet-80',
                        help='模型路径')
    parser.add_argument('--from_dataset', type=str,
                        default='DRIVE',
                        help='模型路径')
    parser.add_argument('--save_vis', type=bool, default=True,
                        help='保存语义分割后的图像')
    parser.add_argument('--eval_epoch', type=int, default=None,
                        help='加载哪个epoch的模型, 为None加载最好的模型')
    opt = parser.parse_args()
    return opt

def draw_and_save_roc(roc_dict, load_path):
    # 绘制roc曲线
    plt.figure()
    lw = 2
    plt.figure(figsize=(5, 5))
    plt.plot(roc_dict['fpr']['1'], roc_dict['tpr']['1'], color='darkorange', lw=lw,
             label='ROC curve (area = %0.2f)' % roc_dict['roc_auc'][1].item())
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    if not os.path.exists(os.path.join(load_path, 'roc')):
        os.makedirs(os.path.join(load_path, 'roc'), exist_ok=True)

    plt.savefig(f"{os.path.join(load_path, 'roc', 'roc.svg')}", dpi=600) # .png, .pdf, .ps, .eps, .svg

    df = pd.DataFrame({"fpr": roc_dict['fpr']['1'], "tpr": roc_dict['tpr']['1']})
    df.to_csv(f"{os.path.join(load_path, 'roc', 'roc_data.csv')}", index=False)  # index=False表示不保存行索引

def remove_all_csv(base_path, fold_num_list):
    """
    删除所有验证的csv文件
    """
    for fold in fold_num_list:
        csv_path = os.path.join(base_path, f'fold-{fold}', 'validate_results.csv')
        if os.path.exists(csv_path):
            os.remove(csv_path)
    csv_path = os.path.join(base_path, 'validate_results.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)

def write_csv(saved_path, calculator, info):
    # 定义CSV文件路径
    csv_path = os.path.join(saved_path, f'cross_{args.from_dataset}_validate_results.csv')

    # 字段名称需与用户需求完全匹配
    fieldnames = [
        'info', 'f1_score', 'dice', 'iou', 'sensitivity',
        'specificity', 'accuracy', 'roc_auc'
    ]
    # 检查文件是否存在以决定是否写入表头
    write_header = not os.path.exists(csv_path)
    dict = calculator.compute_dict
    # 写入CSV文件
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()  # 首次运行时写入表头[2,6](@ref)

        # 假设验证阶段已计算以下变量（需替换为实际变量名）
        row_data = {
            'info': info,
            'f1_score': f"{dict['f1_score'][1].item():.4f}",  # 示例：0.852
            'dice': f"{dict['mean_dice'][1].item():.4f}",  # 示例：0.791
            'iou': f"{dict['iou'][1].item():.4f}",
            'sensitivity': f"{dict['sensitivity'][1].item():.4f}",
            'specificity': f"{dict['specificity'][1].item():.4f}",
            'accuracy': f"{dict['accuracy'][1].item():.4f}",
            'roc_auc': f"{dict['roc_auc'][1].item():.4f}"
        }
        writer.writerow(row_data)  # 按列写入结构化数据[1,6](@ref)

def main(args, hypes):
    device = torch.device(hypes['device'])
    # 分割的分类数目 nun_classes + background
    num_classes = hypes['num-classes'] + 1
    # hypes['model']['args']['num_classes'] = num_classes
    # 第几折进行训练
    fold_num_list = hypes['train_params']['train_fold_list']
    calculator_list = []
    global_calculator = Calculator(num_classes=num_classes, ignore_index=255, compute_roc_auc=False)
    global_calculator_process = Calculator(num_classes=num_classes, ignore_index=255, compute_roc_auc=False)
    remove_all_csv(args.model_dir, fold_num_list)
    for fold in fold_num_list:
        print('-----------------Dataset Building------------------')
        val_dataset = build_dataset(hypes, train=False, fold=fold)
        num_workers = hypes['num_workers']
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 num_workers=num_workers,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 collate_fn=val_dataset.collate_fn)
        print('---------------Creating Model------------------')
        model = train_utils.create_model(hypes)
        print('-----------------Load Pretrained Model------------------')
        load_path = os.path.join(args.use_model_dir, f'fold-{fold}')
        _, model, _, _, _, _ = train_utils.load_saved_model(load_path, model, None, None, None)
        model.to(device)
        print('-----------------Eval Step------------------')
        model.eval()
        calculator = Calculator(num_classes=num_classes, ignore_index=255, compute_roc_auc=True)
        calculator_postprocess = Calculator(num_classes=num_classes, ignore_index=255, compute_roc_auc=True)
        dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f'Test Fold [{fold}/{len(fold_num_list)}]:'
        with torch.no_grad():
            idx = 0
            for image, target in metric_logger.log_every(val_loader, 100, header):
                image, target = image.to(device), target.to(device)
                # output = model(image)
                output = sliding_window_inference(inputs=image, roi_size=(
                    hypes['augmentor']['args']['crop_size'], hypes['augmentor']['args']['crop_size']),
                                                  sw_batch_size=1,
                                                  predictor=model, overlap=0.25)  # 使用滑动窗口进行推理
                del image
                target = target.to("cpu")
                output = output['out'].to("cpu")
                # ---- 保存推理图像 ----
                prediction = output.argmax(1).squeeze(0)
                prediction = prediction.to("cpu").numpy().astype(np.uint8)
                # 将前景对应的像素值改成255(白色)
                prediction[prediction == 1] = 255
                prediction[prediction == 0] = 0
                # 将不敢兴趣的区域像素设置成0(黑色)
                roi_img = Image.open(replace_system_separator(val_dataset.mask[idx])).convert('L')
                roi_img = np.array(roi_img)
                prediction[roi_img == 0] = 0
                mask = Image.fromarray(prediction)
                file_name = os.path.basename(val_dataset.img_list[idx])
                if not os.path.exists(os.path.join(load_path, 'save_images')):
                    os.makedirs(os.path.join(load_path, 'save_images'))
                mask.save(
                    f"{os.path.join(load_path, 'save_images', f'cross_{args.from_dataset}_' + os.path.splitext(file_name)[0])}.png")

                # --------------------
                # 后处理一下
                prediction_postprocess = remove_small_areas_based_threshold(output.argmax(1).squeeze(0),
                                                                            threshold=hypes['postprocess']['threshold'])
                calculator.update(target.flatten(), output.argmax(1).flatten(), target, output)
                global_calculator.update(target.flatten(), output.argmax(1).flatten(), target, output)

                calculator_postprocess.update(target.flatten(), prediction_postprocess.flatten(), target, output)
                global_calculator_process.update(target.flatten(), prediction_postprocess.flatten(), target, output)

                dice.update(output, target)

                # ---- 保存处理后的推理图像 ----
                prediction = prediction_postprocess.to("cpu").numpy().astype(np.uint8)
                # 将前景对应的像素值改成255(白色)
                prediction[prediction == 1] = 255
                prediction[prediction == 0] = 0
                # 将不敢兴趣的区域像素设置成0(黑色)
                roi_img = Image.open(replace_system_separator(val_dataset.mask[idx])).convert('L')
                roi_img = np.array(roi_img)
                prediction[roi_img == 0] = 0
                mask = Image.fromarray(prediction)
                file_name = os.path.basename(val_dataset.img_list[idx])
                if not os.path.exists(os.path.join(load_path, 'save_images')):
                    os.makedirs(os.path.join(load_path, 'save_images'))
                mask.save(
                    f"{os.path.join(load_path, 'save_images', f'cross_{args.from_dataset}_processed_' + os.path.splitext(file_name)[0])}.png")
                idx += 1
            calculator.reduce_from_all_processes()
            dice.reduce_from_all_processes()
        dice = dice.value.item()
        # 如果后处理更好, 就用后处理
        if calculator_postprocess.compute()['f1_score'][1] > calculator.compute()['f1_score'][1]:
            print(f"fold: {fold} 使用后处理!")
            calculator = calculator_postprocess
        val_info = str(calculator)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # 将本次验证的结果写入文件
        with open(os.path.join(load_path, f'cross_{args.from_dataset}_validate_results.txt'), "w") as f:
            # 验证集各指标
            f.write(val_info + "\n\n")
        write_csv(load_path, calculator, 'origin: ' + str(fold) + '/' + hypes['name'])
        calculator_list.append(calculator)
        del model
        del val_dataset
        del val_loader
        torch.cuda.empty_cache()
    # 计算总体结果, 并且写入文件
    if global_calculator_process.compute()['f1_score'][1] > global_calculator.compute()['f1_score'][1]:
        print(f"最终结果使用后处理!")
        global_calculator = global_calculator_process
    # with open(os.path.join(args.model_dir, f'cross_{args.from_dataset}_validate_results.json'), "w") as f:
    #     result = Calculator.mean_calculator_list(calculator_list)
    #     result = str_dict_from_tensor(result)
    #     print("最终结果: \n")
    #     print(result)
    #     f.write(json.dumps(result))
    write_csv(args.model_dir, global_calculator, 'origin')


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


def change_dataset(dataset_name, hypes):
    """

    Args:
        dataset_name: ['DRIVE', 'CHASE_DB1', 'STARE', 'HRF', 'EYE-Seg']

    Returns:

    """
    dataset_dict = {
        'method': 'EyeSegDataset',  # [DriveDataset ChaseDataset StareDataset HrfDataset EyeSegDataset]
        'root_dir': "C:\\my\\F\\眼底图像分割\\EYE-Seg",  # 数据集位置 C:\\my\\F\\眼底图像分割\\EYE-Seg
        'fold_num': 5,
        'ignore_index': 255,  # 忽略255掩码
        'train_expand_rate': 8,  # 训练集扩充倍数
    }
    if dataset_name == 'DRIVE':
        dataset_dict['method'] = 'DriveDataset'
        dataset_dict['root_dir'] = "D:\\F\\眼底图像分割\\DRIVE"
        hypes['augmentor']['mean'] = [0.38145922]
        hypes['augmentor']['std'] = [0.07928617]
    if dataset_name == 'CHASE_DB1':
        dataset_dict['method'] = 'ChaseDataset'
        dataset_dict['root_dir'] = "D:\\F\\眼底图像分割\\CHASEDB1"
        hypes['augmentor']['mean'] = [0.23612322]
        hypes['augmentor']['std'] = [0.10596604]
    if dataset_name == 'STARE':
        dataset_dict['method'] = 'StareDataset'
        dataset_dict['root_dir'] = "D:\\F\\眼底图像分割\\Stare"
        hypes['augmentor']['mean'] = [0.42517377]
        hypes['augmentor']['std'] = [0.09725328]
    if dataset_name == 'HRF':
        dataset_dict['method'] = 'HrfDataset'
        dataset_dict['root_dir'] = "D:\\F\\眼底图像分割\\HRF"
        hypes['augmentor']['mean'] = [0.23687161]
        hypes['augmentor']['std'] = [0.06492036]
    if dataset_name == 'EYE-Seg':
        dataset_dict['method'] = 'EyeSegDataset'
        dataset_dict['root_dir'] = "D:\\F\\眼底图像分割\\EYE-Seg"
        hypes['augmentor']['mean'] = [0.28825587]
        hypes['augmentor']['std'] = [0.09410577]
    hypes['dataset'] = dataset_dict
    return hypes


if __name__ == '__main__':
    use_queue_train = True  # 是否启用队列训练
    if use_queue_train:
        # (CENet, U2Net, SwinUnet, EyeNet)
        # (DRIVE => CHASE_DB1, CHASE_DB1 => STARE, EYE-Seg => HRF, HRF => EYE-Seg, EYE-Seg => DRIVE, DRIVE => HRF)
        model_dir_path = [
            ('../logs/DRIVE/CENet', 'CHASE_DB1'),
            ('../logs/DRIVE/U2Net', 'CHASE_DB1'),
            ('../logs/DRIVE/SwinUNet', 'CHASE_DB1'),
            ('../logs/DRIVE/EyeNet', 'CHASE_DB1'),


            ('../logs/CHASEDB1/CENet', 'STARE'),
            ('../logs/CHASEDB1/U2Net', 'STARE'),
            ('../logs/CHASEDB1/SwinUNet', 'STARE'),
            ('../logs/CHASEDB1/EyeNet', 'STARE'),



            ('../logs/EYE-Seg/CENet', 'HRF'),
            ('../logs/EYE-Seg/U2Net', 'HRF'),
            ('../logs/EYE-Seg/SwinUNet', 'HRF'),
            ('../logs/EYE-Seg/EyeNet', 'HRF'),


            ('../logs/HRF/CENet', 'EYE-Seg'),
            ('../logs/HRF/U2Net', 'EYE-Seg'),
            ('../logs/HRF/SwinUNet', 'EYE-Seg'),
            ('../logs/HRF/EyeNet', 'EYE-Seg'),


            ('../logs/EYE-Seg/CENet', 'DRIVE'),
            ('../logs/EYE-Seg/U2Net', 'DRIVE'),
            ('../logs/EYE-Seg/SwinUNet', 'DRIVE'),
            ('../logs/EYE-Seg/EyeNet', 'DRIVE'),


            ('../logs/DRIVE/CENet', 'HRF'),
            ('../logs/DRIVE/U2Net', 'HRF'),
            ('../logs/DRIVE/SwinUNet', 'HRF'),
            ('../logs/DRIVE/EyeNet', 'HRF'),
        ]
        for path, dataset_name in model_dir_path:
            print('-----------------Analyze Config File------------------')
            args = parse_args()
            args.use_model_dir = path
            args.model_dir = path
            args.from_dataset = dataset_name
            hypes = yaml_utils.load_yaml(None, args)
            modify_config(hypes)
            hypes = change_dataset(dataset_name, hypes)
            print(f'当前训练模型路径: {os.path.abspath(path)}')
            # hypes['train_params']['batch_size'] = 4
            # if hypes['name'] == 'R2UNet' or hypes['name'] == 'TransUNet':
            #     hypes['train_params']['batch_size'] = 2
            # if hypes['name'] == 'MCDAUNet':
            #     hypes['train_params']['batch_size'] = 3
            # hypes['optimizer']['lr'] = 0.002
            # if hypes['dataset']['method'] == 'DriveDataset':
            #     hypes['dataset']['root_dir'] = "/dataset/DRIVE"
            # if hypes['dataset']['method'] == 'ChaseDataset':
            #     hypes['dataset']['root_dir'] = "/dataset/CHASEDB1"
            # if hypes['dataset']['method'] == 'StareDataset':
            #     hypes['dataset']['root_dir'] = "/dataset/Stare"
            # if hypes['dataset']['method'] == 'HrfDataset':
            #     hypes['dataset']['root_dir'] = "/dataset/HRF"
            main(args, hypes)
    else:
        print('-----------------Analyze Config File------------------')
        args = parse_args()
        args.model_dir = args.use_model_dir
        dataset_name = args.from_dataset
        hypes = yaml_utils.load_yaml(None, args)
        hypes = change_dataset(dataset_name, hypes)
        main(args, hypes)
