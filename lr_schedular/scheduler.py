import math

import torch


def unet_scheduler(optimizer,
                   num_step: int,
                   epochs: int,
                   warmup=True,
                   warmup_epochs=1,
                   warmup_factor=1e-3, **kwargs):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        x: 第几次调用step方法
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        多项式学习策略, 参考 deeplabv2 https://blog.csdn.net/qq_47233366/article/details/137103702
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def step(optimizer, step_size, gamma, **kwargs):
    from torch.optim.lr_scheduler import StepLR
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


def multistep(optimizer, step_size, gamma, **kwargs):
    from torch.optim.lr_scheduler import MultiStepLR
    return MultiStepLR(optimizer, milestones=step_size, gamma=gamma)


def exponential(optimizer, gamma, **kwargs):
    from torch.optim.lr_scheduler import ExponentialLR
    return ExponentialLR(optimizer, gamma)


def cosine_anneal_warm(optimizer,
                       num_step: int,
                       epochs: int,
                       step_size,
                       warmup_lr,
                       warmup_epoches,
                       lr_min,
                       gamma, **kwargs):
    from timm.scheduler.cosine_lr import CosineLRScheduler
    num_steps = epochs * num_step
    warmup_lr = warmup_lr
    warmup_steps = warmup_epoches * num_step
    lr_min = lr_min

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=lr_min,
        warmup_lr_init=warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    return scheduler


def cosine_annealing_lr_warm(optimizer, t_max=80):
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段：线性增加学习率
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            t_cur = self.last_epoch - self.warmup_epochs
            T_max = self.T_max - self.warmup_epochs
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t_cur / T_max)) / 2
                    for base_lr in self.base_lrs]
