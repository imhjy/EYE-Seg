from data_utils.augmentor import build_dataset_transformer
from data_utils.datasets.ChaseDataset import ChaseDataset
from data_utils.datasets.DriveDataset import DriveDataset
from data_utils.datasets.EyeSegDataset import EyeSegDataset
from data_utils.datasets.HrfDataset import HrfDataset
from data_utils.datasets.StareDataset import StareDataset
from data_utils.datasets.OdcSegDataset import OdcSegDataset


__all__ = {
    'DriveDataset': DriveDataset,
    'ChaseDataset': ChaseDataset,
    'HrfDataset': HrfDataset,
    'StareDataset': StareDataset,
    'EyeSegDataset': EyeSegDataset,
    'OdcSegDataset': OdcSegDataset,
}




def build_dataset(hypes, train=True, fold=0):
    """
    创建数据集
    Args:
        hypes: 配置文件字典
        train: 是否是训练模型
        fold: 使用哪一折, fold=0表示默认
    Returns:
        需要的数据集
    """

    dataset_name = hypes['dataset']['method']
    error_message = f"{dataset_name} 没有找到. " \
                    f"请将数据集添加到: " \
                    f"data_utils/datasets/init.py"
    assert dataset_name in ['DriveDataset', 'ChaseDataset', 'HrfDataset', 'StareDataset', 'EyeSegDataset', 'OdcSegDataset'], error_message

    dataset = __all__[dataset_name](
        hypes=hypes,
        train=train,
        transforms=build_dataset_transformer(hypes=hypes, train=train),
        fold=fold
    )

    return dataset
