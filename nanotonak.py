from __future__ import print_function

import time
from typing import Any

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib import tenumerate
from tqdm import tqdm
import os
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit

# 自作モジュール
import datasetClass
from mymodule import my_func

def addition_data(input_data:ndarray, channel:int=0, delay:int=1)-> ndarray[Any, dtype[floating[_64Bit] | float_]]:
    """ 1chの信号を遅延・減衰 (減衰率はテキトー) させる

    Parameters
    ----------
    input_data:  1chの音声データ
    channel: 拡張したいch数
    delay: どれぐらい遅延させるか

    Returns
    -------

    """
    """ エラー処理 """
    if channel <= 0:  # channelsの数が0の場合or指定していない場合
        raise ValueError("channels must be greater than 0.")
    result = np.zeros((channel, len(input_data)))
    # print("result:", result.shape)
    # print(result)
    """ 遅延させるサンプル数を指定 """
    sampling_rate = 16000
    win = 2
    window_size = sampling_rate * win // 1000  # ConvTasNetの窓長と同じ
    delay_sample = window_size  # ConvTasNetの窓長と同じ
    # delay_sample = 1    # 1サンプルだけずらす

    """ 1ch目を基準に遅延させる """
    for i in range(channel):
        result[i, delay_sample*i:] = input_data[:len(input_data)-delay_sample*i]  # 1サンプルづつずらす 例は下のコメントアウトに記載
        result[i,:] = result[i, :] * (1/2**i)   # 音を減衰させる
        """
        例
        入力：[1,2,3,4,5]
        出力：
        [[1,2,3,4,5],
         [0,1,2,3,4],
         [0,0,1,2,3],
         [0,0,0,3,4],]
        """
    """ 線形アレイを模倣した遅延 """
    # result[0, delay_sample:] = input_data[:len(input_data) - delay_sample]
    # result[-1, delay_sample:] = input_data[:len(input_data) - delay_sample]

    return result

if __name__ == "__main__":

    path = "C:\\Users\\kataoka-lab\\Desktop\\sound_data\\mix_data\\subset_DEMAND_hoth_1010dB_1ch\\subset_DEMAND_hoth_1010dB_05sec_1ch\\train\\noise_reverbe\\p226_050_16kHz_hoth_10db_05sec_Left.wav"
    # C:\Users\kataoka - lab\Desktop\sound_data\mix_data\subset_DEMAND_hoth_1010dB_1ch\subset_DEMAND_hoth_1010dB_01sec_1ch\train\noise_reverbe

    wave_data, prm = my_func.load_wav(wave_path=path)
    print(wave_data.shape)

    gensui_data = addition_data(input_data=wave_data, channel=4)
    print(gensui_data.shape)
    # for i in range(0, 4):
    save_path = f"./RESULT/gensui_data.wav"
    my_func.save_wav(out_path=save_path, wav_data=gensui_data, prm=prm)

