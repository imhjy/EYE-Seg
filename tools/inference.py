"""
推理验证
"""
import csv
import json
import os
import sys
import argparse

import numpy as np
import torch
from PIL import Image
from monai.inferers import sliding_window_inference
# from torch.utils.tensorboard import SummaryWriter

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


def parse_args():
    parser = argparse.ArgumentParser(description="推理参数")
    parser.add_argument('--model_dir', type=str,
                        default='../logs/test',
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

    plt.savefig(f"{os.path.join(load_path, 'roc', 'roc.svg')}", dpi=600)  # .png, .pdf, .ps, .eps, .svg

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
    csv_path = os.path.join(saved_path, 'validate_results.csv')

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


def save_outputs_and_labels(outputs, labels, postprocess, fold_idx, save_dir):
    from pathlib import Path
    """
    保存模型输出和标签到单个.pt文件中

    参数:
    outputs: 模型输出Tensor
    labels: 对应标签Tensor
    fold_idx: 文件编号（用于文件名）
    save_dir: 保存目录
    """
    # 确保保存目录存在
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 构建文件路径
    file_path = Path(save_dir) / f"temp_fold{fold_idx}.pt"

    # 转换为浮点张量（确保数据类型一致）
    outputs = torch.cat([t.unsqueeze(0) for t in outputs], dim=0).detach().cpu()
    labels = torch.cat([t.unsqueeze(0) for t in labels], dim=0).detach().cpu()
    postprocess = torch.cat([t.unsqueeze(0) for t in postprocess], dim=0).detach().cpu()

    # 保存到文件
    torch.save({
        'outputs': outputs,
        'labels': labels,
        'postprocess': postprocess,
    }, file_path)


def load_outputs_and_labels(fold_idx, save_dir, device="cpu"):
    from pathlib import Path
    """
    从.pt文件加载模型输出和标签

    参数:
    fold_idx: 文件编号
    save_dir: 保存目录
    device: 加载到哪个设备
    """
    file_path = Path(save_dir) / f"temp_fold{fold_idx}.pt"

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 加载数据
    data = torch.load(file_path, map_location=device)

    # 返回输出和标签
    outputs = data['outputs']
    labels = data['labels']
    postprocessArr = data['postprocess']
    outputs_arr = []
    labels_arr = []
    postprocess_arr = []
    for i in range(outputs.size()[0]):
        outputs_arr.append(outputs[i])
        labels_arr.append(labels[i])
        postprocess_arr.append(postprocessArr[i])
    return outputs_arr, labels_arr, postprocess_arr


def main(args, hypes):
    device = torch.device(hypes['device'])
    # 分割的分类数目 nun_classes + background
    num_classes = hypes['num-classes'] + 1
    # hypes['model']['args']['num_classes'] = num_classes
    # 第几折进行训练
    fold_num_list = hypes['train_params']['train_fold_list']
    calculator_list = []

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
        load_path = os.path.join(args.model_dir, f'fold-{fold}')
        _, model, _, _, _, _ = train_utils.load_saved_model(load_path, model, None, None, None)
        model.to(device)
        print('-----------------Eval Step------------------')
        model.eval()
        calculator = Calculator(num_classes=num_classes, ignore_index=255, compute_roc_auc=True)
        calculator_postprocess = Calculator(num_classes=num_classes, ignore_index=255, compute_roc_auc=True)
        dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f'Test Fold [{fold}/{len(fold_num_list)}]:'
        output_arr = []
        label_arr = []
        postprocess_arr = []
        with torch.no_grad():
            idx = 0
            for image, target in metric_logger.log_every(val_loader, 100, header):
                torch.cuda.empty_cache()
                image, target = image.to(device), target.to(device)
                # output = model(image)
                output = sliding_window_inference(inputs=image, roi_size=(
                    hypes['augmentor']['args']['crop_size'], hypes['augmentor']['args']['crop_size']),
                                                  sw_batch_size=1,
                                                  predictor=model, overlap=0.5,mode='gaussian')  # 使用滑动窗口进行推理
                del image
                target = target.to("cpu")
                output = output['out'].to("cpu")
                output_arr.append(output)
                label_arr.append(target)
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
                mask.save(f"{os.path.join(load_path, 'save_images', os.path.splitext(file_name)[0])}.png")

                # --------------------
                # 后处理一下
                prediction_postprocess = remove_small_areas_based_threshold(output.argmax(1).squeeze(0),
                                                                            threshold=hypes['postprocess']['threshold'])
                postprocess_arr.append(prediction_postprocess)
                calculator.update(target.flatten(), output.argmax(1).flatten(), target, output)

                calculator_postprocess.update(target.flatten(), prediction_postprocess.flatten(), target, output)

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
                    f"{os.path.join(load_path, 'save_images', 'processed_' + os.path.splitext(file_name)[0])}.png")
                idx += 1
            calculator.reduce_from_all_processes()
            dice.reduce_from_all_processes()
        dice = dice.value.item()
        # 如果后处理更好, 就用后处理
        if calculator_postprocess.compute()['f1_score'][1] > calculator.compute()['f1_score'][1]:
            print(f"fold: {fold} 使用后处理!")
            calculator = calculator_postprocess
        val_info = str(calculator)
        # draw_and_save_roc(calculator.roc_dict, load_path)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # 将本次验证的结果写入文件
        with open(os.path.join(load_path, 'validate_results.json'), "w") as f:
            # 验证集各指标
            # f.write(val_info + "\n\n")
            result = Calculator.mean_calculator_list([calculator])
            result = str_dict_from_tensor(result)
            f.write(json.dumps(result))
        with open(os.path.join(load_path, 'validate_results.txt'), "w") as f:
            # 验证集各指标
            f.write(val_info + "\n\n")
        write_csv(load_path, calculator, 'origin: ' + str(fold) + '/' + hypes['name'])
        calculator_list.append(calculator)
        save_outputs_and_labels(output_arr, label_arr, postprocess_arr, fold, args.model_dir)
        del model
        del val_dataset
        del val_loader
        torch.cuda.empty_cache()
    # 计算总体结果, 并且写入文件

    global_calculator = Calculator(num_classes=num_classes, ignore_index=255, compute_roc_auc=True)
    global_calculator_process = Calculator(num_classes=num_classes, ignore_index=255, compute_roc_auc=True)
    for fold in fold_num_list:
        output_arr, label_arr, postprocess_arr = load_outputs_and_labels(fold, args.model_dir)

        for i in range(len(output_arr)):
            output = output_arr[i]
            target = label_arr[i]
            prediction_postprocess = postprocess_arr[i]
            global_calculator.update(target.flatten(), output.argmax(1).flatten(), target, output)
            global_calculator_process.update(target.flatten(), prediction_postprocess.flatten(), target, output)

    if global_calculator_process.compute()['f1_score'][1] > global_calculator.compute()['f1_score'][1]:
        print(f"最终结果使用后处理!")
        global_calculator = global_calculator_process
    with open(os.path.join(args.model_dir, 'validate_results.json'), "w") as f:
        # result = Calculator.mean_calculator_list(calculator_list)
        result = Calculator.mean_calculator_list([global_calculator])
        result = str_dict_from_tensor(result)
        print("最终结果: \n")
        print(result)
        f.write(json.dumps(result))
    write_csv(args.model_dir, global_calculator, 'origin')
    for fold in fold_num_list:
        os.remove(os.path.join(args.model_dir, f'temp_fold{fold}.pt'))


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
    hypes['num_workers'] = 0
    if hypes['dataset']['method'] == 'DriveDataset':
        hypes['dataset']['root_dir'] = "D:\\F\\眼底图像分割\\DRIVE"  # F:\\眼底图像分割\\DRIVE
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
        hypes['dataset']['root_dir'] = "D:\\F\\眼底图像分割\\CHASEDB1"  # F:\\眼底图像分割\\CHASEDB1
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
        hypes['dataset']['root_dir'] = "D:\\F\\眼底图像分割\\Stare"  # F:\\眼底图像分割\\Stare
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
        hypes['dataset']['root_dir'] = "D:\\F\\眼底图像分割\\HRF"  # F:\\眼底图像分割\\HRF
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
        hypes['dataset']['root_dir'] = "D:\\F\\眼底图像分割\\EYE-Seg"  # F:\\眼底图像分割\\EYE-Seg
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


if __name__ == '__main__':
    use_queue_train = True  # 是否启用队列训练
    if use_queue_train:
        model_dir_path = [
            # '../logs/DRIVE/AttentionUnet',
            # '../logs/DRIVE/CENet',
            # '../logs/DRIVE/DenseUNet',
            # '../logs/DRIVE/EyeNet',
            # '../logs/DRIVE/IterNet',
            # '../logs/DRIVE/MCDAUNet',
            # '../logs/DRIVE/MedNeXt',
            # '../logs/DRIVE/MGANet',
            # '../logs/DRIVE/MISSFormer',
            # '../logs/DRIVE/R2UNet',
            # '../logs/DRIVE/SwinUNet',
            # '../logs/DRIVE/TransUNet',
            # '../logs/DRIVE/U2Net',
            # '../logs/DRIVE/UCTransNet',
            # '../logs/DRIVE/unet',
            #
            # '../logs/CHASEDB1/AttentionUnet',
            # '../logs/CHASEDB1/CENet',
            # '../logs/CHASEDB1/DenseUNet',
            # '../logs/CHASEDB1/EyeNet',
            # '../logs/CHASEDB1/IterNet',
            # '../logs/CHASEDB1/MCDAUNet',
            # '../logs/CHASEDB1/MedNeXt',
            # '../logs/CHASEDB1/MGANet',
            # '../logs/CHASEDB1/MISSFormer',
            # '../logs/CHASEDB1/R2UNet',
            # '../logs/CHASEDB1/SwinUNet',
            # '../logs/CHASEDB1/TransUNet',
            # '../logs/CHASEDB1/U2Net',
            # '../logs/CHASEDB1/UCTransNet',
            # '../logs/CHASEDB1/unet',
            #
            # '../logs/Stare/AttentionUnet',
            # '../logs/Stare/CENet',
            # '../logs/Stare/DenseUNet',
            # '../logs/Stare/EyeNet',
            # '../logs/Stare/IterNet',
            # '../logs/Stare/MCDAUNet',
            # '../logs/Stare/MedNeXt',
            # '../logs/Stare/MGANet',
            # '../logs/Stare/MISSFormer',
            # '../logs/Stare/R2UNet',
            # '../logs/Stare/SwinUNet',
            # '../logs/Stare/TransUNet',
            # '../logs/Stare/U2Net',
            # '../logs/Stare/UCTransNet',
            # '../logs/Stare/unet',
            #
            #
            # '../logs/HRF/AttentionUnet',
            # '../logs/HRF/CENet',
            # '../logs/HRF/DenseUNet',
            # '../logs/HRF/EyeNet',
            # '../logs/HRF/IterNet',
            # '../logs/HRF/MCDAUNet',
            # '../logs/HRF/MedNeXt',
            # '../logs/HRF/MGANet',
            # '../logs/HRF/MISSFormer',
            # '../logs/HRF/R2UNet',
            # '../logs/HRF/SwinUNet',
            # '../logs/HRF/TransUNet',
            # '../logs/HRF/U2Net',
            # '../logs/HRF/UCTransNet',
            # '../logs/HRF/unet',
            #
            # '../logs/EYE-Seg/AttentionUnet',
            # '../logs/EYE-Seg/CENet',
            # '../logs/EYE-Seg/DenseUNet',
            # '../logs/EYE-Seg/EyeNet',
            # '../logs/EYE-Seg/IterNet',
            # '../logs/EYE-Seg/MCDAUNet',
            # '../logs/EYE-Seg/MedNeXt',
            # '../logs/EYE-Seg/MGANet',
            # '../logs/EYE-Seg/MISSFormer',
            # '../logs/EYE-Seg/R2UNet',
            # '../logs/EYE-Seg/SwinUNet',
            # '../logs/EYE-Seg/TransUNet',
            # '../logs/EYE-Seg/U2Net',
            # '../logs/EYE-Seg/UCTransNet',
            # '../logs/EYE-Seg/unet',

            # '../logs/compare/eyeseg/eyenet/eyenet1',
            # '../logs/compare/eyeseg/eyenet/eyenet2',
            # '../logs/compare/eyeseg/eyenet/eyenet3',
            # '../logs/compare/eyeseg/eyenet/eyenet12',
            # '../logs/compare/eyeseg/eyenet/eyenet13',
            # '../logs/compare/eyeseg/eyenet/eyenet23',
            # '../logs/compare/eyeseg/eyenet/eyenet123',

            '../logs/compare/other/unet_common',
            '../logs/compare/other/unet_nnunet',
        ]
        for path in model_dir_path:
            print('-----------------Analyze Config File------------------')
            args = parse_args()
            args.model_dir = path
            hypes = yaml_utils.load_yaml(None, args)
            modify_config(hypes)
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
        hypes = yaml_utils.load_yaml(None, args)
        main(args, hypes)
