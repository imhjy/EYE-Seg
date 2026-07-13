from data_utils.augmentor.augment import get_common_transformer, get_enhance_gray_transformer, \
    get_enhance_gray_transformer1, get_enhance_gray_transformer2, get_enhance_gray_transformer3, \
    get_enhance_gray_transformer4,get_nnunet_transformer

__all__ = {
    'Common': get_common_transformer,
    'EnhanceGray': get_enhance_gray_transformer,
    'EnhanceGray1': get_enhance_gray_transformer1,
    'EnhanceGray2': get_enhance_gray_transformer2,
    'EnhanceGray3': get_enhance_gray_transformer3,
    'EnhanceGray4': get_enhance_gray_transformer4,
    'nnunet': get_nnunet_transformer,
}


def build_dataset_transformer(hypes, train=True):
    """
    创建数据集对应的增强器
    Args:
        hypes: 配置文件字典
        train: 是否是训练模型

    Returns:
        需要的数据集
    """

    dataset_augmentor = __all__[hypes['augmentor']['core_method']](
        train=train,
        hypes=hypes
    )

    return dataset_augmentor
