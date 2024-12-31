from __future__ import print_function

import time
from typing import Any

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib import tenumerate
from tqdm import tqdm
import os
from numpy import ndarray, dtype, floating, float_

# 自作モジュール
import datasetClass
from mymodule import my_func

def calc_power(wav_data):
    return 20*np.log10(np.abs(wav_data))

def main(wav_data, num_delay, c=340, SR=16000):
    # もとの音のpowerを計算
    origin_power = calc_power(wav_data)

    # 減衰後のpowerを計算
    decay_power = origin_power - 11 - 20*np.log10(c * num_delay/SR)

    # 減衰率を計算
    decay = 10 ** (decay_power/20)/wav_data
    print("max", max(decay))
    print("min", min(decay))

    # 出力音声 = もとの音 * 減衰率
    out_wav = wav_data * decay
    return out_wav



if __name__ == "__main__":
    print("start")
    # input_path = "C:\\Users\\kataoka-lab\\Desktop\\sound_data\\mix_data\\subset_DEMAND_hoth_1010dB_1ch\\subset_DEMAND_hoth_1010dB_05sec_1ch\\test\\noise_reverbe\\p232_068_16kHz_hoth_10db_05sec_None.wav"
    # input_data, prm = my_func.load_wav(input_path)
    # input_data[input_data==0] = 1
    # out_wav = []
    # win = 32
    # out_wav.append(input_data)
    # for ch in range(1, 3+1):
    #     decay = main(input_data, num_delay=win*ch)
    #     out_wav.append(decay)
    # out_wav = np.array(out_wav)
    # print("out_wav",out_wav.shape)
    # out_path = "C:\\Users\\kataoka-lab\\Desktop\\sound_data\\decay.wav"
    # my_func.save_wav(out_path=out_path,
    #                  wav_data=out_wav,
    #                  prm=prm)
    a = np.inf
    count = 0
    for i in range(1000):
        if a < 100 ** i:
            count+=1
    print("count:", count)

    # a = np.array([0, 10, 2, 3, 0, 0, 0])
    # print(a)
    # a[a==0] = 1
    # print(a)



