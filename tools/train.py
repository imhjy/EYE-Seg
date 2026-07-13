import gc
import os
import sys

from utils.common import setup_seed, calculate_grad_norm
from utils.train_utils import initialize_layers

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)
sys.path.append('/projects/JY-MedSeg2D')
sys.path.append('/projects/JY-MedSeg2D/tools')
print(sys.executable)
import time
import datetime
import argparse
import torch
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter

from data_utils.datasets import build_dataset
from hypes_yaml import yaml_utils
from loss import build_loss
from metric.calculator import Calculator
from utils import train_utils
from lr_schedular import build_lr_schedular
import utils.distributed_utils as utils
from utils.early_stopping import EarlyStopping
import torch.nn.functional as F
from inference import main as inference_main


def parse_args():
    parser = argparse.ArgumentParser(description="训练参数")
    parser.add_argument("--hypes_yaml", type=str,
                        # default="../hypes_yaml/ODC-Seg/unet.yaml",
                        default="../logs/EYE-Seg/EyeNet/config.yaml",
                        help='配置文件路径')
    parser.add_argument('--model_dir', type=str,
                        # default='../logs/augmentor/gaussian',
                        help='训练路径,与hypes_yaml二选一')
    parser.add_argument('--save_vis', type=bool, default=True,
                        help='保存语义分割后的图像')
    parser.add_argument('--eval_epoch', type=int, default=None,
                        help='加载哪个epoch的模型, 为None加载最好的模型')
    parser.add_argument('--immediately_inference', type=bool, default=True,
                        help='是否训练好后立即推理')
    args = parser.parse_args()

    return args


def main(args, hypes):
    global early_stopping
    print(f"模型名称: {hypes['name']}")
    device = torch.device(hypes['device'])
    print(f"设备: {device}")
    print(f"数据增强方法: {hypes['augmentor']['core_method']}")
    print(f"patch size: {hypes['augmentor']['args']['crop_size']}")
    print(f"数据集: {hypes['dataset']['method']}")
    print(f"epoch数: {hypes['train_params']['epoches']}")
    batch_size = hypes['train_params']['batch_size']
    # 分割的分类数目 nun_classes + background
    num_classes = hypes['num-classes'] + 1
    # hypes['model']['args']['num_classes'] = num_classes
    # 第几折进行训练
    fold_num_list = hypes['train_params']['train_fold_list']
    # epoch数
    epoches = hypes['train_params']['epoches']

    # 是否使用梯度裁剪
    is_use_grad_norm = 'grad_norm' in hypes and hypes['grad_norm']['use'] is True

    # 随机数种子
    if 'seed' in hypes and hypes['seed'] != -1:
        setup_seed(hypes['seed'])
    origin_path = None  # 防止创建多个文件夹
    for fold in fold_num_list:
        print(f"模型名称: {hypes['name']}")
        print('-----------------Dataset Building------------------')
        print(f"模型名称: {hypes['name']}")
        train_dataset = build_dataset(hypes, train=True, fold=fold)
        val_dataset = build_dataset(hypes, train=False, fold=fold)
        num_workers = hypes['num_workers']
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   collate_fn=train_dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 collate_fn=val_dataset.collate_fn)

        print('---------------Creating Model------------------')
        model = train_utils.create_model(hypes)
        model.apply(initialize_layers)  # 进行模型初始化
        # 优化器
        # params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        # optimizer = torch.optim.SGD(
        #     params_to_optimize,
        #     lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        # )
        optimizer = train_utils.setup_optimizer(hypes, model)

        # 混合精度
        scaler = torch.cuda.amp.GradScaler() if hypes['amp'] else None

        # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
        num_steps = len(train_loader)
        # lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

        lr_scheduler = build_lr_schedular(optimizer, hypes, num_step=num_steps, epochs=epoches,
                                          **hypes['lr_scheduler']['args'])

        # 损失函数
        criterion = build_loss(hypes)

        lowest_val_epoch = -1

        # 如果我们想恢复训练
        if args.model_dir and hypes['train_params']['enable_resume']:
            print('-----------------Load Pretrained Model------------------')
            saved_path = os.path.join(args.model_dir, f'fold-{fold}')
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
            init_epoch, model, optimizer, lr_scheduler, scaler, f1score = train_utils.load_saved_model(saved_path,
                                                                                                       model,
                                                                                                       optimizer,
                                                                                                       lr_scheduler,
                                                                                                       scaler)
            lowest_val_epoch = init_epoch
        else:
            init_epoch = 0
            f1score = 0
            # 如果我们从头开始训练模型，我们需要创建一个文件夹去保存模型
            if origin_path is None:
                origin_path = train_utils.setup_train(hypes)
            saved_path = os.path.join(origin_path, f'fold-{fold}')
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)

        model.to(device)
        # 因为optimizer加载参数时,tensor默认在CPU上
        # 故需将所有的tensor都放到cuda,
        # 否则: 在optimizer.step()处报错：
        # RuntimeError: expected device cpu but got device cuda:0
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        # record training
        # writer = SummaryWriter(saved_path)

        if hypes['early_stop']['use']:
            early_stopping = EarlyStopping(**hypes['early_stop']['args'])

        # if args.resume:
        #     checkpoint = torch.load(args.resume, map_location='cpu')
        #     model.load_state_dict(checkpoint['model'])
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #     args.start_epoch = checkpoint['epoch'] + 1
        #     if args.amp:
        #         scaler.load_state_dict(checkpoint["scaler"])

        best_f1 = f1score
        print(f"初始化f1-score: {best_f1}")
        start_time = time.time()
        continue_train = True
        for epoch in range(init_epoch, max(epoches, init_epoch)):
            if not continue_train:
                break
            # ------------------训练------------------
            # mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
            #                                 lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
            model.train()
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            header = 'Epoch: [{}] Fold: [{}]'.format(epoch, fold)

            if num_classes == 2:
                # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
                loss_weight = torch.as_tensor([1.0, 2.0], device=device)
            elif num_classes == 3:
                # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
                loss_weight = torch.as_tensor([1.0, 2.0, 2.0], device=device)
            else:
                loss_weight = None

            for image, target in metric_logger.log_every(train_loader, 1, header):
                image, target = image.to(device), target.to(device)
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    output = model(image)
                    # torch.where(target == 2)[0].numel()
                    loss = criterion(output['out'], target, loss_weight, num_classes=num_classes, ignore_index=255)
                    if hypes['model']['core_method'] == 'MedNeXt' and hypes['model']['args'][
                        'deep_supervision'] == True:
                        # 下采样倍率列表
                        scale_factors = [1 / 2, 1 / 4, 1 / 8, 1 / 16]
                        aux_weight = [1 / 2, 1 / 4, 1 / 8, 1 / 16]
                        # 对每个倍率进行下采样
                        for idx, scale in enumerate(scale_factors):
                            # 计算目标尺寸
                            target_size = (int(target.shape[0] * scale), int(target.shape[1] * scale))

                            # 使用nearest进行下采样, 双线性插值会有无效浮点值
                            aux_target = F.interpolate(target.unsqueeze(1).float(), scale_factor=scale, mode='nearest')

                            loss += criterion(output[f'out{idx + 1}'], aux_target.squeeze(1).to(torch.long),
                                              loss_weight,
                                              num_classes=num_classes, ignore_index=255) * aux_weight[idx]

                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    # 梯度裁剪（按范数裁剪）, 只要开始两个batch会到6, 后面全是1~5之间, 所以选择5
                    # (如果使用模型初始化开始就是10多, 测试后发现加上梯度裁剪对模型收敛有坏处, 注释)
                    # 参数更新量 ≈ 学习率 × 梯度范数
                    if is_use_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_norm=hypes['grad_norm']['args']['max_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # 梯度裁剪（按范数裁剪）
                    if is_use_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_norm=hypes['grad_norm']['args']['max_norm'])
                    optimizer.step()
                # print(f"当前梯度: {calculate_grad_norm(model)}")
                if hypes['lr_scheduler']['step_per_batch']:
                    lr_scheduler.step()
                lr = optimizer.param_groups[0]["lr"]
                # 更新字段的值
                metric_logger.update(loss=loss.item(), lr=lr)
            if not hypes['lr_scheduler']['step_per_batch']:
                lr_scheduler.step()
            mean_loss = metric_logger.meters["loss"].global_avg
            # --------------------------------------

            # ----------------------------验证---------------------------------------
            # confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
            if epoch % hypes['train_params']['eval_freq'] == 0:
                model.eval()
                confmat = utils.ConfusionMatrix(num_classes)
                calculator = Calculator(num_classes=num_classes)
                dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
                metric_logger = utils.MetricLogger(delimiter="  ")
                header = 'Test:'
                val_loss = []
                with torch.no_grad():
                    for image, target in metric_logger.log_every(val_loader, 100, header):
                        image, target = image.to(device), target.to(device)
                        # output = model(image)
                        output = sliding_window_inference(inputs=image, roi_size=(
                            hypes['augmentor']['args']['crop_size'], hypes['augmentor']['args']['crop_size']),
                                                          sw_batch_size=1,
                                                          predictor=model, overlap=0.33)  # 使用滑动窗口进行推理
                        val_loss.append(
                            criterion(output['out'], target, loss_weight, num_classes=num_classes, ignore_index=255))
                        output = output['out']
                        confmat.update(target.flatten(), output.argmax(1).flatten())
                        calculator.update(target.flatten(), output.argmax(1).flatten())
                        dice.update(output, target)

                    confmat.reduce_from_all_processes()
                    calculator.reduce_from_all_processes()
                    dice.reduce_from_all_processes()

                dice_value = dice.value.item()
                val_info = str(confmat)
                print(val_info)
                val_info = str(calculator)
                print(val_info)
                print(f"dice coefficient: {dice_value:.5f}")
                print(f"validate loss: {torch.tensor(val_loss, dtype=torch.float32).mean().item():.3f}")
                if hypes['early_stop']['use']:
                    # 使用早停法
                    early_stopping(calculator.compute_dict['f1_score'][1].item())
                    if early_stopping.early_stop:
                        print("Early stopping")
                        continue_train = False  # 跳出迭代，结束训练
                # f1_value = calculator.compute_dict['f1_score'][1].item()
                f1_value = (sum(x.item() for x in calculator.compute_dict['f1_score'][1:]) /
                            (len(calculator.compute_dict['f1_score']) - 1))
                # 将本次验证的结果写入文件
                with open(os.path.join(saved_path, 'train_results.txt'), "a+") as f:
                    # 记录每个epoch对应的train_loss、lr以及验证集各指标
                    train_info = f"[epoch: {epoch}]\n" \
                                 f"train_loss: {mean_loss:.4f}\n" \
                                 f"lr: {lr:.6f}\n" \
                                 f"dice coefficient: {dice_value:.3f}\n"
                    f.write(train_info + val_info + "\n\n")
                del calculator
                del dice
                del confmat
            # ------------------------------------------
            # ----------------保存模型参数-------------------
            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args,
                         "best_f1": best_f1}
            if hypes['amp']:
                save_file["scaler"] = scaler.state_dict()
            if epoch % hypes['train_params']['save_freq'] == 0:
                torch.save(save_file, os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

            if best_f1 < f1_value:
                best_f1 = f1_value
                save_file['best_f1'] = best_f1
                torch.save(save_file, os.path.join(saved_path, 'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                                                          'net_epoch_bestval_at%d.pth' % (
                                                                                  lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                           'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1
            else:
                continue
            # ------------------------------------------------
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("training time {}".format(total_time_str))

        # 删除第一个模型并释放资源
        del model, optimizer  # 删除对象
        torch.cuda.empty_cache()  # 清空PyTorch的CUDA缓存
        gc.collect()  # 触发Python垃圾回收

    if args.immediately_inference:
        print(f'开始进行推理: {hypes["name"]} {hypes["dataset"]["method"]}')
        inference_main(args, hypes)


def get_model_dir_path(idx=0):
    paths = [
        [
            '../logs/DRIVE/AttentionUnet',
            '../logs/DRIVE/CENet',
            '../logs/DRIVE/DenseUNet',
            '../logs/DRIVE/EyeNet',
            '../logs/DRIVE/IterNet',
            '../logs/DRIVE/MCDAUNet',
            '../logs/DRIVE/MedNeXt',
            '../logs/DRIVE/MGANet',
            '../logs/DRIVE/MISSFormer',
            '../logs/DRIVE/R2UNet',
            '../logs/DRIVE/SwinUNet',
            '../logs/DRIVE/TransUNet',
            '../logs/DRIVE/U2Net',
            '../logs/DRIVE/UCTransNet',
            '../logs/DRIVE/unet',
        ],
        [
            '../logs/CHASEDB1/AttentionUnet',
            '../logs/CHASEDB1/CENet',
            '../logs/CHASEDB1/DenseUNet',
            '../logs/CHASEDB1/EyeNet',
            '../logs/CHASEDB1/IterNet',
            '../logs/CHASEDB1/MCDAUNet',
            '../logs/CHASEDB1/MedNeXt',
            '../logs/CHASEDB1/MGANet',
            '../logs/CHASEDB1/MISSFormer',
            '../logs/CHASEDB1/R2UNet',
            '../logs/CHASEDB1/SwinUNet',
            '../logs/CHASEDB1/TransUNet',
            '../logs/CHASEDB1/U2Net',
            '../logs/CHASEDB1/UCTransNet',
            '../logs/CHASEDB1/unet',
        ],
        [
            '../logs/Stare/AttentionUnet',
            '../logs/Stare/CENet',
            '../logs/Stare/DenseUNet',
            '../logs/Stare/EyeNet',
            '../logs/Stare/IterNet',
            '../logs/Stare/MCDAUNet',
            '../logs/Stare/MedNeXt',
            '../logs/Stare/MGANet',
            '../logs/Stare/MISSFormer',
            '../logs/Stare/R2UNet',
            '../logs/Stare/SwinUNet',
            '../logs/Stare/TransUNet',
            '../logs/Stare/U2Net',
            '../logs/Stare/UCTransNet',
            '../logs/Stare/unet',
        ],
        [
            '../logs/HRF/AttentionUnet',
            '../logs/HRF/CENet',
            '../logs/HRF/DenseUNet',
            '../logs/HRF/EyeNet',
            '../logs/HRF/IterNet',
            '../logs/HRF/MCDAUNet',
            '../logs/HRF/MedNeXt',
            '../logs/HRF/MGANet',
            '../logs/HRF/MISSFormer',
            '../logs/HRF/R2UNet',
            '../logs/HRF/SwinUNet',
            '../logs/HRF/TransUNet',
            '../logs/HRF/U2Net',
            '../logs/HRF/UCTransNet',
            '../logs/HRF/unet',
        ],
        [
            '../logs/EYE-Seg/AttentionUnet',
            '../logs/EYE-Seg/CENet',
            '../logs/EYE-Seg/DenseUNet',
            '../logs/EYE-Seg/EyeNet',
            '../logs/EYE-Seg/IterNet',
            '../logs/EYE-Seg/MCDAUNet',
            '../logs/EYE-Seg/MedNeXt',
            '../logs/EYE-Seg/MGANet',
            '../logs/EYE-Seg/MISSFormer',
            '../logs/EYE-Seg/R2UNet',
            '../logs/EYE-Seg/SwinUNet',
            '../logs/EYE-Seg/TransUNet',
            '../logs/EYE-Seg/U2Net',
            '../logs/EYE-Seg/UCTransNet',
            '../logs/EYE-Seg/unet',
        ],
        [
            '../logs/compare/other/unet_common',
            '../logs/compare/other/unet_nnunet',

        ],
    ]

    return paths[idx]


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
    setup_seed(3407)
    if not use_queue_train:
        print('-----------------Analyze Config File------------------')
        args = parse_args()
        hypes = yaml_utils.load_yaml(args.hypes_yaml, args)
        # hypes['device'] = 'cuda:6'
        hypes['amp'] = True
        # hypes['num_workers'] = 4
        hypes['train_params']['batch_size'] = 4
        # if hypes['name'] == 'R2UNet' or hypes['name'] == 'TransUNet':
        #     hypes['train_params']['batch_size'] = 2
        # if hypes['name'] == 'MCDAUNet':
        #     hypes['train_params']['batch_size'] = 3
        hypes['optimizer']['lr'] = 0.001
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
        device = 'cuda'  # 7 6 4 3
        # model_dir_path = [
        #     '../logs/DRIVE/MedNeXt-M',
        #     '../logs/DRIVE/MedNeXt-L',
        #     '../logs/DRIVE/EyeNet-M',
        #     '../logs/DRIVE/EyeNet-L',
        #     '../logs/DRIVE/MedNeXt-S',
        #     '../logs/DRIVE/EyeNet-S',
        #     '../logs/DRIVE/EyeNet-1',
        #     '../logs/DRIVE/EyeNet-2',
        #     '../logs/DRIVE/EyeNet-3',
        #     '../logs/DRIVE/EyeNet-1S',
        #     '../logs/DRIVE/EyeNet-2S',
        #     '../logs/DRIVE/EyeNet-3S',
        # ]
        model_dir_path = get_model_dir_path(5)
        for path in model_dir_path:
            print('-----------------Analyze Config File------------------')
            args = parse_args()
            args.model_dir = path
            hypes = yaml_utils.load_yaml(args.hypes_yaml, args)
            print(f'当前训练模型路径: {os.path.abspath(path)}')
            if device is not None:
                hypes['device'] = device
            # hypes['amp'] = True
            # hypes['num_workers'] = 4
            hypes = modify_config(hypes)
            # hypes['train_params']['epoches'] = 1  # 测试
            # hypes['dataset']['train_expand_rate'] = 1  # 测试
            # hypes['train_params']['train_fold_list'] = [1]  # 测试
            main(args, hypes)
