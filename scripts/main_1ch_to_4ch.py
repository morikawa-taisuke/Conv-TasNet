from __future__ import print_function

import torch
import numpy as np
from tqdm import tqdm
import os

# 自作モジュール
from src.utils import my_func, const
from src.models.MultiChannel_ConvTasNet_models import type_A, type_C, type_D_2, type_E
from src import models as Multichannel_model
from src.Multi_Channel_ConvTasNet_train import main
from data import make_dataset
from data.make_dataset import addition_data
from scripts import All_evaluation as eval


def test(mix_dir, out_dir, model_name, channels, model_type):
    print("mix_dir: ", mix_dir)
    print("out_dir: ", out_dir)
    print("model_name: ", model_name)

    filelist_mixdown = my_func.get_file_list(mix_dir)
    # print('number of mixdown file', len(filelist_mixdown))

    # ディレクトリを作成
    my_func.make_dir(out_dir)

    # model_name, _ = my_func.get_file_name(model_name)

    # モデルの読み込み
    match model_type:
        case 'A':
            TasNet_model = type_A().to("cuda")
        case 'C':
            TasNet_model = type_C().to("cuda")
        case 'D':
            TasNet_model = type_D_2(num_mic=channels).to("cuda")
        case 'E':
            TasNet_model = type_E().to("cuda")
        case '2stage':
            TasNet_model = Multichannel_model.type_D_2_2stage(num_mic=channels).to("cuda")
        case "single_to_multi":
            TasNet_model = Multichannel_model.single_to_multi(num_mic=channel).to("cuda")

    # TasNet_model.load_state_dict(torch.load('./pth/model/' + model_name + '.pth'))
    TasNet_model.load_state_dict(torch.load(model_name))
    # TCN_model.load_state_dict(torch.load('reverb_03_snr20_reverb1020_snr20-clean_DNN-WPE_TCN_100.pth'))

    for fmixdown in tqdm(filelist_mixdown):  # filelist_mixdownを全て確認して、それぞれをfmixdownに代入
        # y_mixdownは振幅、prmはパラメータ
        y_mixdown, prm = my_func.load_wav(fmixdown)  # waveでロード
        # print(f'y_mixdown.shape:{y_mixdown.shape}')
        y_mixdown = y_mixdown.astype(np.float32)  # 型を変形
        y_mixdown_max = np.max(y_mixdown)  # 最大値の取得
        # y_mixdown = my_func.load_audio(fmixdown)     # torchaoudioでロード
        # y_mixdown_max = torch.max(y_mixdown)

        y_mixdown = addition_data(y_mixdown, channel=channel)

        y_mixdown = y_mixdown[np.newaxis, :]
        # print(f"mix:{type(y_mixdown)}")

        # print(f'y_mixdown.shape:{y_mixdown.shape}')  # y_mixdown.shape=[1,チャンネル数×音声長]
        MIX = torch.tensor(y_mixdown, dtype=torch.float32)
        # MIX = split_data(y_mixdown, channel=channels)  # MIX=[チャンネル数,音声長]
        # print(f'MIX.shape:{MIX.shape}')
        # MIX = MIX[np.newaxis, :, :]  # MIX=[1,チャンネル数,音声長]
        # MIX = torch.from_numpy(MIX)
        # print("00MIX", MIX.shape)
        MIX = MIX.to("cuda")
        # print("11MIX", MIX.shape)
        separate = TasNet_model(MIX)  # モデルの適用
        # print("separate", separate.shape)
        separate = separate.cpu()
        separate = separate.detach().numpy()
        tas_y_m = separate[0, 0, :]
        # print(f'tas_y_m.shape:{tas_y_m.shape}')
        # y_mixdown_max=y_mixdown_max.detach().numpy()
        # tas_y_m_max=torch.max(tas_y_m)
        # tas_y_m_max=tas_y_m_max.detach().numpy()

        tas_y_m = tas_y_m * (y_mixdown_max / np.max(tas_y_m))

        # 分離した speechを出力ファイルとして保存する。
        # 拡張子を変更したパス文字列を作成
        foutname, _ = os.path.splitext(os.path.basename(fmixdown))
        # ファイル名とフォルダ名を結合してパス文字列を作成
        fname = os.path.join(out_dir, (foutname + '.wav'))
        # print('saving... ', fname)
        # 混合データを保存
        # mask = mask*y_mixdown
        my_func.save_wav(fname, tas_y_m, prm)
        torch.cuda.empty_cache()    # メモリの解放 1音声ごとに解放
        # torchaudio.save(
        #     fname,
        #     tas_y_m.detach().numpy(),
        #     const.SR,
        #     format='wav',
        #     encoding='PCM_S',
        #     bits_per_sample=16
        # )


if __name__ == "__main__":
    print("start")
    """ ファイル名等の指定 """
    # C:\Users\kataoka-lab\Desktop\sound_data\mix_data\1ch_to_4ch_decay_all_minus\train
    base_name = "1ch_to_4ch_decay_all_minus"
    wave_type_list = ["noise_reverbe", "reverbe_only", "noise_only"]     # "noise_reverbe", "reverbe_only", "noise_only"
    # angle_list = ["Right", "FrontRight", "Front", "FrontLeft", "Left"]  # "Right", "FrontRight", "Front", "FrontLeft", "Left"
    channel = 4
    """ wav_fileの作成 """
    # mix_dir = f"{const.MIX_DATA_DIR}/{base_name}/"
    # input_dir = f"{const.MIX_DATA_DIR}/{base_name}/"
    # for test_train in my_func.get_subdir_list(input_dir):
    #     # for wave_type in my_func.get_subdir_list(os.path.join(input_dir, test_train)):
    #     make_mixdown.decay_signal_all(signal_dir=os.path.join(input_dir, test_train),
    #                                   out_dir=os.path.join(mix_dir, test_train))

    """ datasetの作成 """
    print("make_dataset")
    dataset_dir = f"{const.DATASET_DIR}/{base_name}/"
    mix_dir = f"{const.MIX_DATA_DIR}/{base_name}/train"
    base_name = "1ch_to_4ch_decay_all_minus"
    for wave_type in wave_type_list:
        make_dataset.make_dataset_csv(mix_dir=os.path.join(mix_dir, wave_type),
                                      target_dir=os.path.join(mix_dir, "clean"),
                                      csv_path=os.path.join(dataset_dir, f"{wave_type}_{base_name}.csv"))

    """ train """
    print("train")
    pth_dir = f"{const.PTH_DIR}/{base_name}/"
    for wave_type in wave_type_list:
        main(dataset_path=os.path.join(dataset_dir, f"{wave_type}_{base_name}.csv"),
             out_path=os.path.join(pth_dir,wave_type),
             train_count=1000,
             model_type="D",
             channel=channel)

    """ test_evaluation """
    condition = {"speech_type": "subset_DEMAND",
                 "noise": "hoth",
                 "snr": 10,
                 "reverbe": 5}
    for wave_type in wave_type_list:
        name = f"{base_name}"
        mix_dir = f"{const.MIX_DATA_DIR}/{name}/test"
        out_wave_dir = f"{const.OUTPUT_WAV_DIR}/{base_name}/05sec/"
        print("test")
        test(mix_dir=os.path.join(mix_dir, wave_type),
             out_dir=os.path.join(out_wave_dir, wave_type),
             model_name=os.path.join(pth_dir, wave_type, f"{wave_type}_100.pth"),
             channels=channel,
             model_type="D")

        evaluation_path = f"{const.EVALUATION_DIR}/{base_name}/{wave_type}.csv"
        print("evaluation")
        eval.main(target_dir=os.path.join(mix_dir, "clean"),
                  estimation_dir=os.path.join(out_wave_dir, wave_type),
                  out_path=evaluation_path,
                  condition=condition,
                  channel=channel)
