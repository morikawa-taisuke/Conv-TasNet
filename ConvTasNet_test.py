# coding:utf-8
from __future__ import print_function

import os.path

from tqdm import tqdm
import numpy as np
import torch
# 自作モジュール
from mymodule import my_func
from models import ConvTasNet_models as models

def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e

def test(mix_path:str, estimation_path:str, model_path:str, model_type:str='enhance')->None:
    """
    学習モデルの評価

    Parameters
    ----------
    model_type
    mix_path(str):入力データのパス
    estimation_path(str):出力データのパス
    model_path(str):学習モデル名
    model_type(str):Conv-TasNetのモデルの種類 enhance:音源強調 separate:音源分離

    Returns
    -------
    None
    """
    """ 入力データのリストアップ """
    mix_list = my_func.get_file_list(mix_path, ext='.wav')
    print(f'number of mixdown file:{len(mix_list)}')
    my_func.make_dir(estimation_path)

    """ GPUの設定 """
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPUが使えれば使う
    print(f'device:{device}')

    """ ネットワークの生成 """
    match model_type:
        case 'enhance': # 音源強調
            model = models.enhance_ConvTasNet().to(device)
        case 'separate':    # 音源分離
            model = models.separate_ConvTasNet().to(device)
        case _: # その他
            model = models.enhance_ConvTasNet().to(device)

    model.load_state_dict(torch.load(model_path))

    for mix_file in tqdm(mix_list):   # tqdm():
        """ データの読み込み """
        mix_data, prm = my_func.load_wav(mix_file)  # mix_data:振幅 prm:音源のパラメータ
        """ データ型の調整 """
        mix_data = mix_data.astype(np.float32)  # データ形式の変更
        mix_data_max = np.max(mix_data)     # 最大値の取得
        print(f'mix_data:{mix_data.shape}')

        mix_data = mix_data[np.newaxis, :]  # データ形状の変更 [音声長]->[1, 音声長]
        mix_data = torch.from_numpy(mix_data).to(device)    # データ型の変更 numpy->torch
        # print(f'mix_data:{mix_data.shape}')
        """ モデルの適用 """
        estimation_data = model(mix_data)   # モデルの適用
        print(f'estimation_data:{estimation_data.shape}')
        """ 推測データ型の調整 """
        for idx, estimation in enumerate(estimation_data[0, :, :]):
            print(f'estimation:{estimation.shape}')
            estimation = estimation * (mix_data_max / torch.max(estimation))  # データの正規化
            estimation = estimation.cpu()  # cpuに移動
            estimation = estimation.detach().numpy()  # データ型の変更 torch->numpy
            """ 保存 """
            out_name, _ = my_func.get_file_name(mix_file)  # ファイル名の取得
            out_path = f'{estimation_path}/speeker_{idx}/{out_name}.wav'
            my_func.save_wav(out_path, estimation, prm)  # 保存

        # estimation_data = estimation_data[0, 0, :]  # スライス
        # estimation_data = estimation_data * (mix_data_max / torch.max(estimation_data))    # データの正規化
        # estimation_data = estimation_data.cpu() # cpuに移動
        # estimation_data = estimation_data.detach().numpy()  # データ型の変更 torch->numpy
        """ 保存 """
        # out_name, _ = my_func.get_file_name(mix_file)   # ファイル名の取得
        # out_path = f'{estimation_path}/{out_name}.wav'
        # my_func.save_wav(out_path, estimation_data, prm)    # 保存


if __name__ == '__main__':

    #separate('', '', '')
    mix_dir = "C:\\Users\\kataoka-lab\\Desktop\\sound_data\\mix_data\\sebset_DEMAND_hoth_1010dB_05sec_1ch\\test\\"
    out_dir = "C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\RESULT\\output_wav\\subset_DEMAND_hoth_1010dB_05sec_1ch"
    # subdir_list = my_func.get_subdir_list(mix_dir)
    # subdir_list.remove("clean")
    # for subdir in subdir_list:
    #     test(mix_path=os.path.join(mix_dir, subdir),
    #          estimation_path=os.path.join(out_dir, subdir),
    #          model_path=f'C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\RESULT\\pth\\subset_DEMAND_hoth_1010dB_05sec_1ch\\{subdir}\\{subdir}_100.pth',
    #          model_type='enhance')
    # subdir = 'noise_reverbe'
    # test(mix_path=f'E:\\wav\\wav_data\\sample_data\\{subdir}.wav',
    #      estimation_path=f"E:\\wav\\wav_data\\Multi_channel\\multi\\{subdir}\\Conv-TasNet.wav",
    #      model_path=f'C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\RESULT\\pth\\subset_DEMAND_hoth_1010dB_05sec_1ch\\{subdir}\\{subdir}_100.pth',
    #      model_type='enhance')

    test(mix_path='C:\\Users\\kataoka-lab\\Desktop\\sound_data\\mix_data\\separate_sebset_DEMAND\\test\\mix',
         estimation_path='C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\RESULT\\output_wav\\separate_subset_DEMAND\\',
         model_path='C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\RESULT\\pth\\separation_subdir_DEMAND\\separation_subdir_DEMAND_100.pth',
         model_type='separate')