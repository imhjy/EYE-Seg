import torch
import numpy as np
from sklearn.metrics import roc_curve, auc

from utils.distributed_utils import DiceCoefficient


class SingleMeanDiceCalculator(object):
    """
    求平均Dice指数
    """

    def __init__(self, num_classes=2, ignore_index=255, ):
        super().__init__()
        self.num = 0  # 平均次数
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = 'cpu'
        self.dice = None

    def update(self, target, predict):
        self.num = self.num + 1
        n = self.num_classes
        mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        if self.device is None:
            self.device = target.device
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (target >= 0) & (target < n)  # 去除255掩码的影响
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * target[k].to(torch.int64) + predict[k]
            if not inds.dtype == torch.int64:
                inds = torch.tensor(inds,dtype=torch.int64)
            mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)
            matrix = mat.float()
            TP = torch.diag(matrix)
            FP = matrix.sum(axis=0) - TP  # 每列的和减去TP即为FP
            FN = matrix.sum(axis=1) - TP  # 每行的和减去TP即为FN
            # Dice系数
            dice = 2 * TP / (2 * TP + FP + FN + 1e-7)
            # 求和
            self.dice = dice if self.dice is None else self.dice + dice

    def compute(self):
        if self.num == 0:
            return 0
        else:
            return self.dice / self.num


class Calculator(object):
    """
    用于计算各种指标的组合类
    https://blog.csdn.net/wsljqian/article/details/99435808
    F1-score
    Dice-score
    accuracy
    Recall
    Precision
    Sensitivity 敏感度
    Specificity 特异度
    mAP
    https://blog.csdn.net/liujh845633242/article/details/102938143
    FROC
    ROC
    AUC
    先 update , 再 compute
    """

    def __init__(self, num_classes=2, ignore_index=255, compute_roc_auc=False):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # 需要计算ROC和AUC
        self.flag_roc_auc = compute_roc_auc
        self.mat = None
        self.roc_dict = None
        self.device = 'cpu'
        self.single_mean_dice_calculator = SingleMeanDiceCalculator(num_classes, ignore_index)

    def update(self, target, predict, target_roc_auc=None, predict_roc_auc=None):
        """
        更新混淆矩阵
        Args:
            target:
            predict:

        Returns:

        """
        assert isinstance(target, torch.Tensor) and isinstance(predict, torch.Tensor), '输入数据必须是Tensor类型!'
        if self.flag_roc_auc:
            # 计算ROC曲线和AUC值(ROC面积)
            self.update_roc_auc(target_roc_auc, predict_roc_auc)
        # target = target.flatten()
        # predict = predict.argmax(1).flatten()
        self.single_mean_dice_calculator.update(target, predict)
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
            self.device = target.device
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (target >= 0) & (target < n)  # 去除255掩码的影响
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * target[k].to(torch.int64) + predict[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def update_roc_auc(self, target, predict):
        self.device = target.device
        n = self.num_classes
        if self.roc_dict is None:
            self.roc_dict = {'fpr': {}, 'tpr': {}, 'thresholds': {}, 'roc_auc': [], 'predict': {}, 'target': {}}
        with torch.no_grad():
            k = (target >= 0) & (target < n)  # 去除255掩码的影响
            target = target[k]
            for i in range(self.num_classes):
                predict_t = predict[:, i][k]
                if f'{i}' not in self.roc_dict['target']:
                    self.roc_dict['target'][f'{i}'] = torch.tensor([], device=self.device)
                if f'{i}' not in self.roc_dict['predict']:
                    self.roc_dict['predict'][f'{i}'] = torch.tensor([], device=self.device)
                self.roc_dict['target'][f'{i}'] = torch.cat((self.roc_dict['target'][f'{i}'], target))
                self.roc_dict['predict'][f'{i}'] = torch.cat((self.roc_dict['predict'][f'{i}'], predict_t))

    def compute(self):
        # 计算每个分类的 TP, TN, FP, FN
        matrix = self.mat.float()
        TP = torch.diag(matrix)
        FP = matrix.sum(axis=0) - TP  # 每列的和减去TP即为FP
        FN = matrix.sum(axis=1) - TP  # 每行的和减去TP即为FN
        total_samples = matrix.sum()
        TN = torch.tensor([total_samples - (TP[i] + FP[i] + FN[i]) for i in range(len(TP))], device=self.device)
        # print("TP:", TP)
        # print("FP:", FP)
        # print("FN:", FN)
        # print("TN:", TN)
        # 计算Global Accuracy
        accuracy_global = torch.diag(matrix).sum() / (matrix.sum() + 1e-7)
        # 计算每个类别准确率
        accuracy = (TP + TN) / (TP + FN + FP + TN + 1e-7)

        # 特异性
        specificity = TN / (TN + FP + 1e-7)

        # 敏感度
        sensitivity = TP / (TP + FN + 1e-7)

        # 召回率 = 敏感度
        recall = TP / (TP + FN + 1e-7)

        # 精确率
        precision = TP / (TP + FP + 1e-7)

        # Dice系数
        dice = 2 * TP / (2 * TP + FP + FN + 1e-7)

        # Iou
        iou = TP / (TP + FP + FN + 1e-7)

        # F1-score
        f1_score = (2 * precision * recall) / (precision + recall)

        # 平均Dice指数
        mean_dice = self.single_mean_dice_calculator.compute()
        self.compute_dict = {
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'accuracy_global': accuracy_global,
            'accuracy': accuracy,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'recall': recall,
            'precision': precision,
            'dice': dice,
            'iou': iou,
            'f1_score': f1_score,
            'mean_dice': mean_dice,
            'roc_auc': torch.tensor([0.0,0.0])
        }
        if self.flag_roc_auc:
            # 计算ROC曲线和AUC值(ROC面积)
            roc_dict = self.compute_roc_auc()
            # 与compute_dict合并
            for key in roc_dict.keys():
                # 目前只保存 AUC 的值, 不然太大了
                if 'roc_auc' == key:
                    self.compute_dict[key] = roc_dict[key]
        return self.compute_dict

    def compute_roc_auc(self):
        with torch.no_grad():
            if len(self.roc_dict['roc_auc']) > 0:
                self.roc_dict['roc_auc'] = [] # 支持重复计算机
            for i in range(self.num_classes):
                # 计算ROC曲线需要的数据 TODO 需要绘制ROC曲线时数据从这里面取, fpr是横坐标, tpr是纵坐标
                self.roc_dict['fpr'][f'{i}'], self.roc_dict['tpr'][f'{i}'], self.roc_dict['thresholds'][f'{i}'] = \
                    roc_curve(self.roc_dict['target'][f'{i}'].cpu(), self.roc_dict['predict'][f'{i}'].cpu(),
                              pos_label=i)
                # 计算AUC
                self.roc_dict['roc_auc'].append(torch.tensor(
                    auc(self.roc_dict['fpr'][f'{i}'], self.roc_dict['tpr'][f'{i}']), device=self.device)
                )
            self.roc_dict['roc_auc'] = torch.tensor(self.roc_dict['roc_auc'])
        return self.roc_dict

    def reduce_from_all_processes(self):
        """
        进程同步
        Returns:

        """
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        self.compute_dict = self.compute()
        return (
            'Global Accuracy: {:.3f}\n'
            'Accuracy: {}\n'
            'Specificity: {}\n'
            'Sensitivity: {}\n'
            'Recall: {}\n'
            'Precision: {}\n'
            'Dice: {}\n'
            'Mean Dice: {}\n'
            'F1-Score: {}\n'
            'IoU: {}\n'
            'Mean IoU: {:.3f}'
        ).format(
            self.compute_dict['accuracy_global'].item() * 100,
            ['{:.1f}'.format(i) for i in (self.compute_dict['accuracy'] * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.compute_dict['specificity'] * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.compute_dict['sensitivity'] * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.compute_dict['recall'] * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.compute_dict['precision'] * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.compute_dict['dice'] * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.compute_dict['mean_dice'] * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.compute_dict['f1_score'] * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.compute_dict['iou'] * 100).tolist()],
            self.compute_dict['iou'].mean().item() * 100
        )

    @staticmethod
    def mean_calculator_list(calculator_list):
        """
        求多个calculator的n折交叉检验最后结果
        Args:
            calculator_list: calculator列表
        Returns:
            所有指标的Dict
        """
        assert len(calculator_list) > 0, 'mean_calculator_list方法入参列表长度必须大于0!'
        compute_num = len(calculator_list)
        keys = calculator_list[0].compute_dict.keys()
        result = {}
        for key in keys:
            value = None
            for calc in calculator_list:
                if value == None:
                    value = calc.compute_dict[key]
                else:
                    value = value + calc.compute_dict[key]
            result[key] = value / compute_num
        return result


if __name__ == '__main__':
    calculator = Calculator()

    t1 = torch.tensor([
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 0]
    ])
    t2 = torch.tensor([
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 1, 0]
    ])

    # [
    #         [5, 4],
    #         [2, 5],
    #     ]
    calculator.update(t1, t2)
    print(calculator)
    print('----------------------------')
    dice = DiceCoefficient(num_classes=2, ignore_index=255)
    dice.update(t2, t1)
    print(dice.value.item())


