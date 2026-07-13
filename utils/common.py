import platform
import random

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.backends import cudnn

def str_dict_from_tensor(obj):
    """
    将一个内容有Tensor的dict转换为普通形式
    Args:
        dict:

    Returns:

    """
    if isinstance(obj, dict):
        return {k: str_dict_from_tensor(v) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, list):
        return [str_dict_from_tensor(item) for item in obj]
    else:
        return obj


def replace_system_separator(path: str):
    sys = platform.system()
    if sys == "Linux":
        return path.replace('\\', '/')
    elif sys == "Windows":
        return path.replace('/', '\\')
    return path


def remove_small_areas_based_threshold(img: Tensor, threshold=5):
    """
    后处理, 去掉连通域小于阈值的预测前景区域
    Args:
        img: 预测的图片 Tensor((H,W))
        threshold: 阈值, 默认5, 根据不同数据集进行调整

    Returns:

    """
    device = img.device
    img = img.cpu().numpy().astype(np.uint8)
    # 使用cv2.connectedComponentsWithStats函数统计联通域信息
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # 遍历stats数组，将小于阈值的联通域在labels中标记为0
    for i in range(1, stats.shape[0]):
        area = stats[i, 4]  # stats数组中第5列是面积信息
        if area < threshold:
            labels[labels == i] = 0

    # 将labels标记图二值化处理, 大于0的换成255
    labels = labels.astype(np.uint8)
    ret, labels = cv2.threshold(labels, 0, 255, cv2.THRESH_BINARY)
    labels[labels > 0] = 1
    return torch.tensor(labels, device=device)


def setup_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 对比实验常规设置 严格可复现性
    cudnn.deterministic = True # cudnn随机数种子固定
    cudnn.benchmark = False # 每次启动训练时自动测试所有可用的卷积算法 并选择当前硬件下最快的算法 适用于输入张量的尺寸在训练过程中保持不变
    # torch.backends.cudnn.deterministic = True

def calculate_grad_norm(model):
    """
    计算当前模型的梯度, 使用前要把梯度裁剪的代码注释, 不然无效

    # 梯度裁剪（按范数裁剪）, 只要开始两个batch会到6, 后面全是1~5之间, 所以选择5
    # (如果使用模型初始化开始就是10多, 测试后发现加上梯度裁剪对模型收敛有坏处, 注释)
    # 参数更新量 ≈ 学习率 × 梯度范数
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    Args:
        model: nn.Module

    Returns:

    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

if __name__ == '__main__':
    tensor_dict = {
        'key1': torch.tensor(1),
        'key2': {
            'key3': torch.tensor(2),
            'key4': torch.tensor([3, 4, 5])
        },
        'key5': [torch.tensor(6), torch.tensor([7, 8])]
    }
    regular_dict = str_dict_from_tensor(tensor_dict)
    print(regular_dict)
