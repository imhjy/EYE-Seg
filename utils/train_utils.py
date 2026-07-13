# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import glob
import importlib
import math

import yaml
import sys
import os
import re
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init


def load_saved_model(saved_path, model, optimizer=None, lr_scheduler=None, scaler=None):
    """
    如果存在, 加载保存好的模型

    Parameters
    __________
    saved_path : str
       模型保存地址
    model : model object
        模型实例

    Returns
    -------
    model : model object
        加载好预训练参数的模型
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                if "bestval" in file_:
                    result = re.findall("net_epoch_bestval_at(.*).pth.*", file_)
                    initial_epoch_ = int(result[0])
                    return initial_epoch_, True
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_, False

    initial_epoch, flag = findLastCheckpoint(saved_path)
    best_f1 = 0
    if initial_epoch > 0:
        model_file = os.path.join(saved_path, 'net_epoch_bestval_at%d.pth' % initial_epoch) \
            if flag else os.path.join(saved_path, 'net_epoch%d.pth' % initial_epoch)
        print('resuming by loading epoch %d' % initial_epoch)
        checkpoint = torch.load(model_file, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        if optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print("optimizer loading error")
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        if "best_f1" in checkpoint:
            best_f1 = checkpoint["best_f1"]

        del checkpoint

    return initial_epoch, model, optimizer, lr_scheduler, scaler, best_f1


def setup_train(hypes):
    """
    根据当前时间步长和模型名称为保存的模型创建文件夹

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
            except FileExistsError:
                pass
        # save the yaml file
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

    return full_path


def create_model(hypes):
    """
    导入模块 “models.[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(**backbone_config)
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """
    创建优化器

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    print('优化器方法是: %s' % optimizer_method)

    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'args' in method_dict:
        return optimizer_method(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                lr=method_dict['lr'],
                                **method_dict['args'])
    else:
        return optimizer_method(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                lr=method_dict['lr'])


def setup_lr_schedular(hypes, optimizer, n_iter_per_epoch):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch. Optimizer
    """
    lr_schedule_config = hypes['lr_scheduler']

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)

    elif lr_schedule_config['core_method'] == 'exponential':
        print('ExponentialLR is chosen for lr scheduler')
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    elif lr_schedule_config['core_method'] == 'cosineannealwarm':
        print('cosine annealing is chosen for lr scheduler')
        from timm.scheduler.cosine_lr import CosineLRScheduler

        num_steps = lr_schedule_config['epoches'] * n_iter_per_epoch
        warmup_lr = lr_schedule_config['warmup_lr']
        warmup_steps = lr_schedule_config['warmup_epoches'] * n_iter_per_epoch
        lr_min = lr_schedule_config['lr_min']

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    else:
        sys.exit('not supported lr schedular')

    return scheduler


def to_device(inputs, device):
    """
    将这个inputs结构全部送进device设备
    Args:
        inputs: 字典, 列表, 或其他
        device: cuda:x/cpu

    Returns:

    """
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str):
            return inputs
        return inputs.to(device)


def kaiming_weight_init(net):
    """
    kaiming 初始化
    Args:
        net: 模型实例
    Returns:

    """
    # 进行权值初始化  如果不自己初始化，则使用的默认方法 init.kaiming_uniform_  0均值的正态分布
    for m in net.modules():  # 递归获得net的所有子代Module
        if isinstance(m, nn.Conv2d):  # 也可以使用torch.nn.init.  https://www.cnblogs.com/jfdwd/p/11269622.html
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))  # mean, std  从方差一致性出发
            # leakyrelu的初始化 0均值的正态分布改为  std = sqrt(2/(1+a^2)*fan_in)
        elif isinstance(m, nn.BatchNorm2d):  # 这里建议去学习一下BN的知识，有空我也会再写一篇
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return net


def initialize_layers(m):
    """
    对卷积层、归一化层和激活函数层进行统一初始化。
    说明：
        - 卷积层：使用He/Kaiming初始化（适应ReLU激活）
        - 归一化层（如BatchNorm）：gamma初始为1，beta初始为0
        - 激活函数层（如ReLU）：无需参数初始化（无操作）
    """
    if isinstance(m, nn.Conv2d):
        # He/Kaiming初始化（针对ReLU激活）
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # 偏置初始化为0
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        # 归一化层：缩放参数初始为1，偏移参数初始为0
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
