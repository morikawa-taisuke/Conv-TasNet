import os
import numpy as np
from tqdm import tqdm
import torch
import wave
from typing import Any
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit

from mymodule import my_func
from models.MultiChannel_ConvTasNet_models import type_A, type_C, type_D_2, type_E
from make_dataset import split_data


def delay_signal(input_data: ndarray, channel: int = 0, delay: int = 1) -> ndarray[Any, dtype[floating[_64Bit] | float_]]:
    """ エラー処理 """
    if channel <= 0:  # channelsの数が0の場合or指定していない場合
        raise ValueError("channels must be greater than 0.")
    result = np.zeros((channel, len(input_data)))
    print("result:", result.shape)
    # print(result)
    for i in range(channel):
        result[i, i:] = input_data[:len(input_data) - i]  # 1サンプルづつずらす 例は下のコメントアウトに記載
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
        delay_data = delay_signal(input_data=input_data, channel=channel)
        """ 保存 """
        out_file, _ = my_func.get_file_name(input_file)  # ファイル名の取得
        out_path = f"{output_dir}/{out_file}.wav"
        my_func.save_wav(out_path=out_path, wav_data=delay_data, prm=prm)

if __name__ == "__main__":
    print("signal")

