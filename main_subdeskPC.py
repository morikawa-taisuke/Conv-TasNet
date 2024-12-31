from __future__ import print_function

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib import tenumerate
from tqdm import tqdm
import os
# 自作モジュール
from mymodule import const, my_func
import datasetClass
from models.MultiChannel_ConvTasNet_models import type_A, type_C, type_D_2, type_E, type_F
import models.MultiChannel_ConvTasNet_models as MultiChannel_model
from Multi_Channel_ConvTasNet_train import main


import make_dataset
import Multi_Channel_ConvTasNet_test as test
import All_evaluation as eval



if __name__ == "__main__":
    print("start")
    #  "C:\Users\kataoka-lab\Desktop\sound_data\mix_data\subset_DEMAND_hoth_1010dB_4ch\subset_DEMAND_hoth_1010dB_05sec_4ch\"
    """ ファイル名等の指定 """
    base_name = "subset_DEMAND_hoth_1010dB_4ch\\subset_DEMAND_hoth_1010dB_05sec_4ch"
    mix_dir_name = "subset_DEMAND_hoth_1010dB_4ch\\subset_DEMAND_hoth_1010dB_05sec_4ch"
    wave_type_list = ["noise_reverbe", "reverbe_only", "noise_only"]  # "noise_reverbe", "reverbe_only", "noise_only"
    angle_list = ["Left"]  # "Right", "FrontRight", "Front", "FrontLeft", "Left"
    channel = 4
    """ datasetの作成 """
    print("---------- make_dataset ----------")
    dataset_dir = f"{const.DATASET_DIR}/{base_name}/"
    # for wave_type in wave_type_list:
    #     # for angel in angle_list:
    #     mix_dir = f"{const.MIX_DATA_DIR}/{mix_dir_name}/train"
    #     ## csvの場合
    #     # make_dataset.make_dataset_csv(mix_dir=os.path.join(mix_dir, wave_type),
    #     #                               target_dir=os.path.join(mix_dir, "clean"),
    #     #                               csv_path=os.path.join(dataset_dir, f"{base_name}_{wave_type}.csv"))
    #     ##NPZFファイルの場合
    #     make_dataset.multi_channel_dataset2(mix_dir=os.path.join(mix_dir, wave_type),
    #                                         target_dir=os.path.join(mix_dir, "clean"),
    #                                         out_dir=os.path.join(dataset_dir, wave_type),
    #                                         channel=4)
    """ train """
    print("---------- train ----------")
    pth_dir = f"{const.PTH_DIR}/{base_name}/"
    for wave_type in wave_type_list:
        for model_type in ["A", "B", "C", "D"]:
            main(dataset_path=os.path.join(dataset_dir, wave_type),
                 out_path=os.path.join(pth_dir, f"{base_name}_{wave_type}_{model_type}"),
                 train_count=100,
                 model_type=model_type,
                 channel=channel)

    """ test_evaluation """
    condition = {"speech_type": "subset_DEMAND",
                 "noise": "hoth",
                 "snr": 10,
                 "reverbe": 5}
    # for wave_type in wave_type_list:
    #     for model_type in ["A", "B", "C", "D"]:
    #         # for angel in angle_list:
    #         mix_dir = f"{const.MIX_DATA_DIR}/{base_name}\\test"
    #         out_wave_dir = f"{const.OUTPUT_WAV_DIR}/{base_name}/{wave_type}"
    #         print("---------- test ----------")
    #         test.test(mix_dir=os.path.join(mix_dir, wave_type),
    #                   out_dir=os.path.join(out_wave_dir, wave_type),
    #                   model_name=os.path.join(pth_dir, f"{base_name}_{wave_type}_{model_type}"),
    #                   channels=channel,
    #                   model_type="D")
    #
    #         evaluation_path = f"{const.EVALUATION_DIR}/{base_name}/{wave_type}.csv"
    #         print("---------- evaluation ----------")
    #         eval.main(target_dir=os.path.join(mix_dir, "clean"),
    #                   estimation_dir=os.path.join(out_wave_dir, wave_type),
    #                   out_path=evaluation_path,
    #                   condition=condition,
    #                   channel=channel)
