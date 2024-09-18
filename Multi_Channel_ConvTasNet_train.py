# coding:utf-8

"""
多チャンネル 音源強調用モデル
入力：多次元，出力：1次元
self.sum_spekeer使用
ボトルネック層で4chから1chに変更
"""
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
from models.MultiChannel_ConvTasNet_models import type_A, type_C, type_D_2, type_E, type_F
from make_dataset import split_data


# GPU確認
def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e

def sisnr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

def si_snr_loss(ests, egs):
    # spks x n x S
    refs = egs
    num_speeker = len(refs)

    def sisnr_loss(permute):
        # for one permute
        return sum(
            [sisnr(ests[s], refs[t])
             for s, t in enumerate(permute)]) / len(permute)
        # average the value

    # P x N
    N = egs.size(0)
    #print("N", N)
    sisnr_mat = torch.stack(
        [sisnr_loss(p) for p in permutations(range(num_speeker))])
    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    # si-snr
    return -torch.sum(max_perutt) / N

def sisdr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisdr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-sdr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(x_zm * s_zm, dim=-1,keepdim=True) * s_zm / torch.sum(s_zm * s_zm, dim=-1,keepdim=True)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(t - x_zm) + eps))

def si_sdr_loss(ests, egs):
    # spks x n x S
    # ests: estimation
    # egs: target
    refs = egs
    num_speeker = len(refs)
    #print("spks", num_speeker)
    # print(f'ests:{ests.shape}')
    # print(f'egs:{egs.shape}')

    def sisdr_loss(permute):
        # for one permute
        #print("permute", permute)
        return sum([sisdr(ests[s], refs[t]) for s, t in enumerate(permute)]) / len(permute)
        # average the value

    # P x N
    N = egs.size(0)
    sisdr_mat = torch.stack([sisdr_loss(p) for p in permutations(range(num_speeker))])
    max_perutt, _ = torch.max(sisdr_mat, dim=0)
    # si-snr
    return -torch.sum(max_perutt) / N

def main(dataset_path, out_path, train_count, model_type, loss_func='SISDR', channel=1, checkpoint_path=None):
    """ 引数の処理 """
    parser = argparse.ArgumentParser(description='CNN Speech(Vocal) Separation')
    parser.add_argument('--dataset', '-t', default=dataset_path, help='Prefix Directory Name to input as dataset')
    parser.add_argument('--batchsize', '-b', type=int, default=const.BATCHSIZE, help='Number of track in each mini-batch')
    parser.add_argument('--patchlength', '-l', type=int, default=const.PATCHLEN, help='length of input frames in one track')
    parser.add_argument('--epoch', '-e', type=int, default=const.EPOCH, help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=1, help='Frequency of taking a snapshot')
    # default=-1 only last, default=1 every epoch, write out snapshot
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
    args = parser.parse_args()

    """ GPUの設定 """
    device = "cuda" if torch.cuda.is_available() else "cpu" # GPUが使えれば使う
    print(f'device:{device}')
    """ その他の設定 """
    out_name, _ = os.path.splitext(os.path.basename(out_path))  # 出力名の取得
    print("out_path: ", out_path)
    writer = SummaryWriter(log_dir=f'{const.LOG_DIR}\\{out_name}')  # logの保存先の指定('tensorboard --logdir ./logs'で確認できる)
    now = my_func.get_now_time()
    csv_path = f'{const.LOG_DIR}\\{out_name}\\{out_name}_{now}.csv'
    my_func.make_dir(csv_path)
    with open(csv_path, 'w') as csv_file:  # ファイルオープン
        csv_file.write(f'dataset,out_name,loss_func,model_type\n{dataset_path},{out_path},{loss_func},{model_type}')

    """ Load dataset データセットの読み込み """
    print(f"dataset:{args.dataset}")
    dataset = datasetClass.TasNet_dataset(args.dataset) # データセットの読み込み
    # print('\nmain_dataset')
    # print(f'type(dataset):{type(dataset)}')                                             # dataset2.TasNet_dataset
    # print(f'np.array(dataset.mix_list).shape:{np.array(dataset.mix_list).shape}')       # [データセットの個数,チャンネル数,音声長]
    # print(f'np.array(dataset.target_list).shape:{np.array(dataset.target_list).shape}') # [データセットの個数,1,音声長]
    # print('main_dataset\n')
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # print('\ndataset_loader')
    # print(f'type(dataset_loader):{type(dataset_loader.dataset)}')
    # print(f'dataset_loader.dataset:{dataset_loader.dataset}')
    # print('dataset_loader\n')


    """ ネットワークの生成 """
    match model_type:
        case 'A':
            model = type_A().to(device) # ネットワークの生成
        case 'C':
            model = type_C().to(device) # ネットワークの生成
        case 'D':
            model = type_D_2(num_mic=channel).to(device) # ネットワークの生成
        case 'E':
            model = type_E().to(device) # ネットワークの生成
        case 'F':
            model = type_F().to(device)  # ネットワークの生成

    print(f'\nmodel:{model}\n')                           # モデルのアーキテクチャの出力
    optimizer = optim.Adam(model.parameters(), lr=0.001)    # optimizerを選択(Adam)
    if loss_func != 'SISDR':
        loss_function = nn.MSELoss().to(device)                 # 損失関数に使用する式の指定(最小二乗誤差)

    """ チェックポイントの設定 """
    if checkpoint_path != None:
        print('restart_training')
        checkpoint = torch.load(checkpoint_path)  # checkpointの読み込み
        model.load_state_dict(checkpoint["model_state_dict"])  # 学習途中のモデルの読み込み
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # オプティマイザの読み込み
        # optimizerのstateを現在のdeviceに移す。これをしないと、保存前後でdeviceの不整合が起こる可能性がある。
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
    else:
        start_epoch = 1

    my_func.make_dir(out_path)
    model.train()                   # 学習モードに設定
    def train(epoch):
        model_loss_sum = 0              # 総損失の初期化
        print("Train Epoch:", epoch)    # 学習回数の表示
        for batch_idx, (mix_data, target_data) in tenumerate(dataset_loader):
            """ モデルの読み込み """
            mix_data, target_data = mix_data.to(device), target_data.to(device) # データをGPUに移動
            # print(mix_data.dtype)
            # print(target_data.dtype)
            # print('\nbefor_model')
            # print(f'type(mix_data):{type(mix_data)}')
            # print(f'mix_data.shape:{mix_data.shape}')
            # print(f'type(target_data):{type(target_data)}')
            # print(f'target_data.shape:{target_data.shape}')
            # print('mix_data.dtype:', mix_data.dtype)
            # print('target_data.dtype:', target_data.dtype)
            # print('befor_model\n')

            """ 勾配のリセット """
            optimizer.zero_grad()  # optimizerの初期化

            """ データの整形 """
            mix_data = mix_data.to(torch.float32)   # target_dataのタイプを変換 int16→float32
            target_data = target_data.to(torch.float32) # target_dataのタイプを変換 int16→float32
            # target_data = target_data[np.newaxis, :, :] # 次元を増やす[1,音声長]→[1,1,音声長]

            """ モデルに通す(予測値の計算) """
            estimate_data = model(mix_data)          # モデルに通す
            # print('\nafter_model')
            # print(f'type(estimate_data):{type(estimate_data)}') #[1,1,音声長*チャンネル数]
            # print(f'estimate_data.shape:{estimate_data.shape}')
            # print(f'type(target_data):{type(target_data)}')
            # print(f'target_data.shape:{target_data.shape}')
            # print('after_model\n')

            """ データの整形 """
            # split_estimate_data = split_data(estimate_data[0],channels)
            # split_estimate_data = split_estimate_data[np.newaxis, :, :]
            # print('\nsplit_data')
            # print(f'type(split_estimate_data):{type(split_estimate_data)}')
            # print(f'split_estimate_data.shape:{split_estimate_data.shape}') # [1,チャンネル数,音声長]
            # print(f'type(target_data):{type(target_data)}')
            # print(f'target_data.shape:{target_data.shape}')
            # print('split_data\n')

            """ 損失の計算 """
            match loss_func:
                case 'SISDR':
                    model_loss = si_sdr_loss(estimate_data, target_data)
                    # print(f'estimate:{estimate_data.shape}')
                    # print(f'target:{target_data.shape}')
                    # for estimate in estimate_data[0, :]:
                    #     # print_name_type_shape('estimate',estimate)
                    #     # print_name_type_shape('target_data[0,:]',target_data[0,:])
                    #     # print(f'estimate:{estimate.unsqueeze(0).shape}')
                    #     # print(f'estimate:{estimate.shape}')
                    #     # print(f'target:{target_data.shape}')
                    #     model_loss += si_sdr_loss(estimate.unsqueeze(0), target_data[0, :])  # si-sdrによる損失の計算
                    # model_loss = model_loss / channels
                case 'wave_MSE':
                    model_loss = loss_function(estimate_data, target_data)  # 時間波形上でMSEによる損失関数の計算
                case 'stft_MSE':
                    """ 周波数軸に変換 """
                    stft_estimate_data = torch.stft(estimate_data[0, :, :], n_fft=1024, return_complex=False)
                    stft_target_data = torch.stft(target_data[0, :, :], n_fft=1024, return_complex=False)
                    # print('\nstft')
                    # print(f'stft_estimate_data.shape:{stft_estimate_data.shape}')
                    # print(f'stft_target_data.shape:{stft_target_data.shape}')
                    # print('stft\n')
                    model_loss = loss_function(stft_estimate_data, stft_target_data)  # 時間周波数上MSEによる損失の計算
            # print(f'estimate_data.size(1):{estimate_data.size(1)}')

            model_loss_sum += model_loss  # 損失の加算

            """ 後処理 """
            model_loss.backward()           # 誤差逆伝搬
            optimizer.step()                # 勾配の更新
            """ チェックポイントの作成 """
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': model_loss_sum},
                   f'{out_path}/{out_name}_ckp.pth')

        writer.add_scalar(str(out_name[0]), model_loss_sum, epoch)
        #writer.add_scalar(str(str_name[0]) + "_" + str(a) + "_sisdr-sisnr", model_loss_sum, epoch)
        print(f'[{epoch}]model_loss_sum:{model_loss_sum}')  # 損失の出力
        # my_func.record_loss(file_name=f'./loss/{out_name}.csv', text=model_loss)
        with open(csv_path, 'a') as out_file:  # ファイルオープン
            out_file.write(f'{model_loss}\n')  # 書き込み

    start_time = time.time()    # 時間を測定
    for epoch in range(start_epoch, train_count+1):   # 学習回数
        train(epoch)
    """ 学習モデル(pthファイル)の出力 """
    print("model save")
    my_func.make_dir(out_path)
    torch.save(model.to(device).state_dict(), f'{out_path}/{out_name}_{epoch}.pth')         # 出力ファイルの保存

    writer.close()

    """ 学習時間の計算 """
    time_end = time.time()              # 現在時間の取得
    time_sec = time_end - start_time    # 経過時間の計算(sec)
    time_h = float(time_sec)/3600.0     # sec->hour
    print(f'time：{str(time_h)}h')      # 出力

def test(mix_dir, out_dir, model_name, channels, model_type):
    filelist_mixdown = my_func.get_wave_list(mix_dir)
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
            TasNet_model = type_D_2().to("cuda")
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

        tas_y_m = tas_y_m * (y_mixdown_max / np.max(tas_y_m))   # 正規化

        # 分離した speechを出力ファイルとして保存する。
        # 拡張子を変更したパス文字列を作成
        foutname, _ = os.path.splitext(os.path.basename(fmixdown))
        # ファイル名とフォルダ名を結合してパス文字列を作成
        file_name = os.path.join(out_dir, (foutname + '.wav'))
        # print('saving... ', file_name)
        # 混合データを保存
        # mask = mask*y_mixdown
        my_func.save_wav(file_name, tas_y_m, prm)
        # torchaudio.save(
        #     file_name,
        #     tas_y_m.detach().numpy(),
        #     const.SR,
        #     format='wav',
        #     encoding='PCM_S',
        #     bits_per_sample=16
        # )


if __name__ == '__main__':
    #print('ConvTasNet train start')

    # main('../../sound_data/mic_array/dataset/JA_hoth_10db_5sec_4ch_clean',
    #      f'JA_hoth_10db_5sec_4ch_clean_C_SISDR',
    #      train_count=100,
    #      channels=4,
    #      model_type='C',
    #      loss_func='SISDR')
    # main('../../sound_data/mic_array/dataset/JA_hoth_10db_5sec_4ch_clean',
    #      f'JA_hoth_10db_5sec_4ch_clean_C_stftMSE',
    #      train_count=100,
    #      channels=4,
    #      model_type='C',
    #      loss_func='stft_MSE')

    """ 本番 """
    loss_function = ['stft_MSE',]  # 'SISDR', 'stft_MSE', 'wave_MSE'
    # model_list = ['A', 'C', 'D', 'E']
    model = 'D'
    # model = 'C'
    # for loss in loss_function:
    wav_type_list = ['noise_only', 'noise_reverbe', 'reverbe_only']  #'noise_only', 'noise_reverbe', 'reverbe_only'
    # reverbe_list = ['03', '05', '07']
    angle_list = ['Right', 'FrontRight', 'Front', 'FrontLeft', 'Left']    # 'Right', 'FrontRight', 'Front', 'FrontLeft', 'Left'
    # reverbe = '05'
    # ch = [2, 4]
    ch = 4
    distance = 6
    # for ch in ch:
    for angle in angle_list:
        for wav_type in wav_type_list:
            main(dataset_path=f'{const.DATASET_DIR}\\subset_DEMAND_hoth_1010dB_05sec_{ch}ch_{distance}cm\\{angle}\\{wav_type}\\',
                 out_path=f'{const.PTH_DIR}\\subset_DEMAND_hoth_1010dB_05sec_{ch}ch_{distance}cm_{model}type\\{angle}\\{wav_type}',
                 train_count=100,
                 model_type=model,
                 channel=ch)


    """ サブセット """
    # main('../../sound_data/mic_array/dataset/JA01_hoth_10db_5sec_4ch_clean',
    #      'JA01_hoth_10db_5sec_4ch_clean_A',
    #      train_count=100,
    #      channels=4)
    """ 動作確認 """
    # wav_list = ['reverbe_only']   #,'noise_only','noise_reverbe'
    # model_list = ['D', 'E', 'A', 'C' ]   #'A',
    # main(dataset_path=f'C:\\Users\\kataoka-lab\\Desktop\\sound_file\\dataset\\subset_DEMAND_hoth_10dB_05sec_4ch_multi\\noise_reverbe',
    #      out_path=f'C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\RESULT\\pth\\subset_DEMAND_hoth_10dB_05sec_4ch_multi\\type_C\\noise_reverbe',
    #      train_count=100,
    #      model_type='C',
    #      loss_func='stft_MSE')#,
         # checkpoint_path='C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\RESULT\\pth\\subset_DEMAND_hoth_10dB_05sec_4ch_multi\\type_C\\noise_reverbe\\noise_reverbe_ckp.pth')

    # for wav_type in wav_list:
    #     for model_type in model_list:
    #         if (model_type != 'C' and wav_type == 'noise_reverbe') or (model_type == 'A' and wav_type == 'reverbe_only') or (model_type == 'C' and wav_type == 'reverbe_only'):
    #             print(f"model_type:{model_type}")
    #             print(f"wav_type:{wav_type}")
    #             print('skip')
    #             continue
    #
    #         # if wav_type == 'noise_only' and model_list == 'C':
    #         #     main(dataset_path=f'C:\\Users\\kataoka-lab\\Desktop\\sound_data\\dataset\\DEMAND_hoth_1010dB_05sec_4ch\\{wav_type}',
    #         #          out_path=f'C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\RESULT\\pth\\DEMAND_hoth_1010dB_05sec_4ch\\type_{model_type}\\{wav_type}',
    #         #          train_count=100,
    #         #          model_type=model_type,
    #         #          checkpoint_path='C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\RESULT\\pth\\DEMAND_hoth_1010dB_05sec_4ch\\type_C\\noise_only\\noise_only_ckp.pth')
    #
    #         # else:
    #         main(dataset_path=f'C:\\Users\\kataoka-lab\\Desktop\\sound_file\\dataset\\subset_DEMAND_hoth_10dB_05sec_4ch_multi\\{wav_type}',
    #              out_path=f'C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\RESULT\\pth\\subset_DEMAND_hoth_10dB_05sec_4ch_multi\\type_{model_type}\\{wav_type}',
    #              train_count=100,
    #              model_type=model_type,
    #              loss_func='stft_MSE')


