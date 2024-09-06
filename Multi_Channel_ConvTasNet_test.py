# coding:utf-8

"""
    音源強調・音源分離(・残響抑圧)用の学習評価プログラム
    入力のチャンネルが2次元(多ch)の時に使うプログラム
"""
import os
import numpy as np
from tqdm import tqdm
import torch
import wave

from mymodule import my_func
from models.MultiChannel_ConvTasNet_models import type_A, type_C, type_D_2, type_E
from make_dataset import split_data


def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e


def test(mix_dir, out_dir, model_name, channels, model_type):
    filelist_mixdown = my_func.get_file_list(mix_dir)
    print('number of mixdown file', len(filelist_mixdown))

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

    # TasNet_model.load_state_dict(torch.load('./pth/model/' + model_name + '.pth'))
    TasNet_model.load_state_dict(torch.load(model_name))
    # TCN_model.load_state_dict(torch.load('reverb_03_snr20_reverb1020_snr20-clean_DNN-WPE_TCN_100.pth'))

    for fmixdown in tqdm(filelist_mixdown):  # filelist_mixdownを全て確認して、それぞれをfmixdownに代入
        # y_mixdownは振幅、prmはパラメータ
        y_mixdown, prm = my_func.load_wav(fmixdown)  # waveでロード
        # print(f'type(y_mixdown):{type(y_mixdown)}')  #
        # print(f'y_mixdown.shape:{y_mixdown.shape}')
        y_mixdown = y_mixdown.astype(np.float32)  # 型を変形
        y_mixdown_max = np.max(y_mixdown)  # 最大値の取得
        # y_mixdown = my_func.load_audio(fmixdown)     # torchaoudioでロード
        # y_mixdown_max = torch.max(y_mixdown)

        y_mixdown = torch.from_numpy(y_mixdown[np.newaxis, :])
        # print(f'type(y_mixdown):{type(y_mixdown)}')

        # print(f'y_mixdown.shape:{y_mixdown.shape}')  # y_mixdown.shape=[1,チャンネル数×音声長]
        # MIX = y_mixdown
        MIX = split_data(y_mixdown, channel=channels)  # MIX=[チャンネル数,音声長]
        # print(f'MIX.shape:{MIX.shape}')
        MIX = MIX[np.newaxis, :, :]  # MIX=[1,チャンネル数,音声長]
        # MIX = torch.from_numpy(MIX)
        # print('00type(MIX):', type(MIX))
        # print("00MIX", MIX.shape)
        MIX = try_gpu(MIX)
        # print('11type(MIX):', type(MIX))
        # print("11MIX", MIX.shape)
        separate = TasNet_model(MIX)  # モデルの適用
        # print("separate", separate.shape)
        separate = separate.cpu()
        # print(f'type(separate):{type(separate)}')
        separate = separate.detach().numpy()
        tas_y_m = separate[0, 0, :]
        # print(f'type(tas_y_m):{type(tas_y_m)}')
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
        # torchaudio.save(
        #     fname,
        #     tas_y_m.detach().numpy(),
        #     const.SR,
        #     format='wav',
        #     encoding='PCM_S',
        #     bits_per_sample=16
        # )


if __name__ == '__main__':
    print('enhance')
    # for idx in range(100,1000+1,100):
    #     print(f'idx:{idx}')
    # """  typeA  """
    # psd('../../sound_data/mic_array/mix_data/JA_hoth_10db_5sec_4ch/test/noise_reverberation',
    #     f'../../sound_data/mic_array/result/enhance/typeA/100_JA_hoth_10db_5sec_4ch',
    #     f'JA01_hoth_10db_5sec_4ch_clean_A_100',
    #     channels=4)
    # model_dir = 'sebset_DEMAND_hoth_1010dB_05sec_4ch_3cm_'
    # model_type = ['A', 'C']    #, 'A'
    angle_list = ['Right', 'FrontRight', 'Front', 'FrontLeft', 'Left']    # 'Right', 'FrontRight', 'Front', 'FrontLeft', 'Left'
    # loss_function = 'stft_MSE'
    for angle in angle_list:
        dir_name = f'sebset_DEMAND_hoth_1010dB_05sec_2ch_3cm_{angle}_Dtype' #sebset_DEMAND_hoth_1010dB_05sec_{ch}ch_3cm_{angle}_{model}type
        out_dir = f'./RESULT/output_wav/{dir_name}/'
        wave_path = f"C:\\Users\\kataoka-lab\\Desktop\\sound_data\\mix_data\\sebset_DEMAND_hoth_1010dB_05sec_2ch_3cm\\{angle}\\test\\"
        wave_type_list = ['noise_reverbe']    # 'noise_only', 'noise_reverbe', 'reverbe_only'
        for wave_type in wave_type_list:
            # for model in model_type:
                # print(f'mix_data: {wave_path}/{wave_type}_delay')
                # print(f'out_dir: {out_dir}/{wave_type}/type{model}/',)
                # print(f'model: ./pth/model/{dir_name}/{wave_type}_delay_{model}_100.pth')
                # print('\n')
            test(mix_dir=f'{wave_path}/{wave_type}',
                 out_dir=f'{out_dir}/{wave_type}',
                 model_name=f'C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\RESULT\\pth\\{dir_name}\\{dir_name}_100.pth',  #C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\RESULT\\pth\\sebset_DEMAND_hoth_1010dB_05sec_{ch}ch_3cm_{angle}_{model}type
                 channels=2,
                 model_type='D')

    """  type  """
    # for model in model_type:
    #     psd(mix_dir='../../sound_data/Experiment/mix_data/multi_ch/test/noise_reverberation',
    #         out_dir=f'../../sound_data/Experiment/result/multi_ch/noise_reverberation/type_{model}',
    #         model_name=f'multich_noise_reverberation_out1_{model}_100.pth',
    #         channels=4,
    #         model_type=model)
    # psd(mix_dir='../../sound_data/mic_array/mix_data/JA_hoth_10db_5sec_4ch/test/mix',
    #     out_dir='../../sound_data/mic_array/result/enhance/type_C/reverbe_only_delay/JA_hoth_10db_5sec_4ch_clean_C_stftMSE',
    #     model_name='JA_hoth_10db_5sec_4ch_clean_C_stftMSE_100',
    #     channels=4)
    """
    print('separate')
    psd('../../sound_data/mic_array/training/JA_M_F_03/noise_reverberation/JA01M001_JA01F049.wav',
        '../../sound_data/mic_array/result/JA_M_F_03_mix',
        'JA_M_F_03')
    """
    # separate('', '', '')
    # psd('../../data_sample/test/mix_05-k_2030/noise_reverberation', '../../data_sample/test/tasnet_mix_05-k_2030-clean_04-mse',
    #       'train1020_mix_05-k_2030-clean_04_k-noise_2030_stft', '0.5')

