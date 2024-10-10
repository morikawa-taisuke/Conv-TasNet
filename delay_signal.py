import os
import numpy as np
from tqdm import tqdm
import torch
import wave
from typing import Any
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit

from mymodule import my_func, const
from models.MultiChannel_ConvTasNet_models import type_A, type_C, type_D_2, type_E
from make_dataset import split_data


def delay_signal(input_data: ndarray, channel: int = 0, delay: int = 1):
    """ エラー処理 """
    if channel <= 0:  # channelsの数が0の場合or指定していない場合
        raise ValueError("channels must be greater than 0.")
    result = np.zeros((channel, len(input_data)))
    # print("result:", result.shape)
    # print(result)
    sampling_rate = 16000
    win = 2
    window_size = sampling_rate * win // 1000  # ConvTasNetの窓長と同じ
    delay_sample = window_size
    for i in range(channel):
        # result[i, i:] = input_data[:len(input_data) - i]  # 1サンプルづつずらす 例は下のコメントアウトに記載
        result[i, delay_sample*i:] = input_data[:len(input_data)-delay_sample*i]  # ConvTasNetの窓長づつずらす

        """
        例
        入力：[1,2,3,4,5]
        出力：
        [[1,2,3,4,5],
         [0,1,2,3,4],
         [0,0,1,2,3],
         [0,0,0,3,4],]
        """

    return result

def main(input_dir, output_dir, channel):
    print("input_dir:", input_dir)
    print("output_dir:", output_dir)

    """ ファイルリストの作成 """
    input_list = my_func.get_file_list(input_dir, ext=".wav")

    for input_file in tqdm(input_list):
        """ データの読み込み """
        input_data, prm = my_func.load_wav(input_file)  # waveでロード
        """ 信号を遅延させる """
        delay_data = delay_signal(input_data=input_data, channel=channel)   # 信号を遅延させる [[マイク1],[マイク2],,,]
        delay_data = np.hstack(delay_data)  # 2次元の信号を1次元に連結させる [マイク1, マイク2, ,,,]

        """ 保存 """
        out_file, _ = my_func.get_file_name(input_file)  # ファイル名の取得
        out_path = f"{output_dir}/{out_file}.wav"
        my_func.save_wav(out_path=out_path, wav_data=delay_data, prm=prm)

if __name__ == "__main__":
    print("dilay_signal")
    a = np.array([1, 2, 3, 4, 5])
    b = delay_signal(a, channel=4)
    print(b)
    print(np.hstack(b))
    input_dir_name = "subset_DEMAND_hoth_1010dB_1ch"
    output_dir_name = "subset_DEMAND_hoth_1010dB_1chto4ch_win"
    ch = 4
    for reverbe in range(1, 6):
        for wave_type in ["noise_only", "reverbe_only", "noise_reverbe", "clean"]:
            input_dir_path = f"{const.MIX_DATA_DIR}/{input_dir_name}/{reverbe:02}sec/test/{wave_type}"
            output_dir_path = f"{const.MIX_DATA_DIR}/{output_dir_name}/{reverbe:02}sec/test/{wave_type}"
            main(input_dir=input_dir_path, output_dir=output_dir_path, channel=ch)
