import os
import sys
import argparse

from data_utils.datasets import build_dataset
from hypes_yaml import yaml_utils

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch unet training")
    parser.add_argument("--hypes_yaml", type=str,
                        default="../hypes_yaml/unet.yaml",
                        help='配置文件路径')
    parser.add_argument('--model_dir',
                        # default='../logs/Stare/AttentionUnet',
                        help='训练路径,与hypes_yaml二选一')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    hypes = yaml_utils.load_yaml(args.hypes_yaml, args)
    fold = hypes['dataset']['fold_num']
    if not fold or fold == 0:
        hypes['dataset']['fold_num'] = 5
    dataset = build_dataset(hypes)
    print(f'-------------开始分割数据集, fold为: {fold}-------------')
    dataset.fold_dataset_split(hypes)
