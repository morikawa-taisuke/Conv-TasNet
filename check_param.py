from __future__ import print_function

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from itertools import permutations
from tqdm.contrib import tenumerate
from tqdm import tqdm
import os
# 自作モジュール
from mymodule import const, my_func
import datasetClass
from models.MultiChannel_ConvTasNet_models import type_A, type_C, type_D, type_E, type_F
from make_dataset import split_data


def main(model_type):

    """ GPUの設定 """
    device = "cuda" if torch.cuda.is_available() else "cpu" # GPUが使えれば使う
    print(f'device:{device}')

    """ ネットワークの生成 """
    match model_type:
        case 'A':
            model = type_A().to(device) # ネットワークの生成
        case 'C':
            model = type_C().to(device) # ネットワークの生成
        case 'D':
            model = type_D().to(device) # ネットワークの生成
        case 'E':
            model = type_E().to(device) # ネットワークの生成
        case 'F':
            model = type_F().to(device)  # ネットワークの生成

    # print(f'\nmodel:{model}\n')                           # モデルのアーキテクチャの出力
    # optimizer = optim.Adam(model.parameters(), lr=0.001)    # optimizerを選択(Adam)

    def count_parameters(model):
        return sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f'{model_type} : {count_parameters(model)}')

if __name__ == '__main__':
    model_list = ['A', 'C', 'D', 'E']
    for model in model_list:
        main(model_type=model)