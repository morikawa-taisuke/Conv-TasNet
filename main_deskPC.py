from __future__ import print_function

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib import tenumerate
from tqdm import tqdm
import os
from itertools import permutations
# 自作モジュール
from mymodule import const, my_func
import datasetClass
from models.MultiChannel_ConvTasNet_models import type_A, type_C, type_D_2, type_E, type_F
import models.MultiChannel_ConvTasNet_models as Multichannel_model
import Multi_Channel_ConvTasNet_train
import make_dataset
from make_dataset import split_data, addition_data
import All_evaluation as eval

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
    # print(f"ests:{ests.shape}")
    # print(f"egs:{egs.shape}")

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

def main(dataset_path, out_path, train_count, model_type, loss_func="SISDR", channel=1, earlystopping_threshold=10, checkpoint_path=None):
    """ 引数の処理 """
    parser = argparse.ArgumentParser(description="CNN Speech(Vocal) Separation")
    parser.add_argument("--dataset", "-t", default=dataset_path, help="Prefix Directory Name to input as dataset")
    parser.add_argument("--batchsize", "-b", type=int, default=1, help="Number of track in each mini-batch")
    parser.add_argument("--patchlength", "-l", type=int, default=const.PATCHLEN, help="length of input frames in one track")
    parser.add_argument("--epoch", "-e", type=int, default=const.EPOCH, help="Number of sweeps over the dataset to train")
    parser.add_argument("--frequency", "-f", type=int, default=1, help="Frequency of taking a snapshot")
    # default=-1 only last, default=1 every epoch, write out snapshot
    parser.add_argument("--gpu", "-g", type=int, default=-1, help="GPU ID (negative value indicates CPU)")
    parser.add_argument("--resume", "-r", default="", help="Resume the training from snapshot")
    args = parser.parse_args()

    """ GPUの設定 """
    device = "cuda" if torch.cuda.is_available() else "cpu" # GPUが使えれば使う
    """ その他の設定 """
    out_name, _ = os.path.splitext(os.path.basename(out_path))  # 出力名の取得
    writer = SummaryWriter(log_dir=f"{const.LOG_DIR}\\{out_name}")  # logの保存先の指定("tensorboard --logdir ./logs"で確認できる)
    now = my_func.get_now_time()
    csv_path = f"{const.LOG_DIR}\\{out_name}\\{out_name}_{now}.csv"
    my_func.make_dir(csv_path)
    with open(csv_path, "w") as csv_file:  # ファイルオープン
        csv_file.write(f"dataset,out_name,loss_func,model_type\n{dataset_path},{out_path},{loss_func},{model_type}")

    """ Early_Stoppingの設定 """
    # earlystopping_threshold = 10
    best_loss = np.inf  # 損失関数の最小化が目的の場合，初めのbest_lossを無限大にする
    # best_loss = -1 * np.inf  # 損失関数の最大が目的の場合，初めのbest_lossを負の無限大にする
    earlystopping_count = 0

    """ Load dataset データセットの読み込み """
    # dataset = datasetClass.TasNet_dataset_csv(args.dataset, channel=channel, device=device) # データセットの読み込み
    dataset = datasetClass.TasNet_dataset(args.dataset) # データセットの読み込み
    # print("\nmain_dataset")
    # print(f"type(dataset):{type(dataset)}")                                             # dataset2.TasNet_dataset
    # print(f"np.array(dataset.mix_list).shape:{np.array(dataset.mix_list).shape}")       # [データセットの個数,チャンネル数,音声長]
    # print(f"np.array(dataset.target_list).shape:{np.array(dataset.target_list).shape}") # [データセットの個数,1,音声長]
    # print("main_dataset\n")
    # dataset_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)
    dataset_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, pin_memory=True)
    # print("\ndataset_loader")
    # print(f"type(dataset_loader):{type(dataset_loader.dataset)}")
    # print(f"dataset_loader.dataset:{dataset_loader.dataset}")
    # print("dataset_loader\n")


    """ ネットワークの生成 """
    match model_type:
        case "A":
            model = type_A().to(device)
        case "C":
            model = type_C().to(device)
        case "D":
            model = type_D_2(num_mic=channel).to(device)
        case "E":
            model = type_E().to(device)
        case "F":
            model = type_F().to(device)

    # print(f"\nmodel:{model}\n")                           # モデルのアーキテクチャの出力
    optimizer = optim.Adam(model.parameters(), lr=0.001)    # optimizerを選択(Adam)
    if loss_func != "SISDR":
        loss_function = nn.MSELoss().to(device)                 # 損失関数に使用する式の指定(最小二乗誤差)

    """ チェックポイントの設定 """
    if checkpoint_path != None:
        print("restart_training")
        checkpoint = torch.load(checkpoint_path)  # checkpointの読み込み
        model.load_state_dict(checkpoint["model_state_dict"])  # 学習途中のモデルの読み込み
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # オプティマイザの読み込み
        # optimizerのstateを現在のdeviceに移す。これをしないと、保存前後でdeviceの不整合が起こる可能性がある。
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint["epoch"] + 1
        loss = checkpoint["loss"]
    else:
        start_epoch = 1

    """ 学習の設定を出力 """
    print("====================")
    print("device: ", device)
    print("out_path: ", out_path)
    print("dataset: ", args.dataset)
    print("model: ", model_type)
    print("loss_func: ", loss_func)
    print("====================")


    my_func.make_dir(out_path)
    model.train()                   # 学習モードに設定

    start_time = time.time()    # 時間を測定
    epoch = 0
    for epoch in range(start_epoch, train_count+1):   # 学習回数
        model_loss_sum = 0              # 総損失の初期化
        print("Train Epoch:", epoch)    # 学習回数の表示
        for batch_idx, (mix_data, target_data) in tenumerate(dataset_loader):
            """ モデルの読み込み """
            mix_data, target_data = mix_data.to(device), target_data.to(device) # データをGPUに移動
            # print(mix_data.dtype)
            # print(target_data.dtype)
            # print("\nbefor_model")
            # print(f"type(mix_data):{type(mix_data)}")
            # print(f"mix_data.shape:{mix_data.shape}")
            # print(f"type(target_data):{type(target_data)}")
            # print(f"target_data.shape:{target_data.shape}")
            # print("mix_data.dtype:", mix_data.dtype)
            # print("target_data.dtype:", target_data.dtype)
            # print("befor_model\n")

            """ 勾配のリセット """
            optimizer.zero_grad()  # optimizerの初期化

            """ データの整形 """
            mix_data = mix_data.to(torch.float32)   # target_dataのタイプを変換 int16→float32
            target_data = target_data.to(torch.float32) # target_dataのタイプを変換 int16→float32
            # target_data = target_data[np.newaxis, :, :] # 次元を増やす[1,音声長]→[1,1,音声長]

            """ モデルに通す(予測値の計算) """
            estimate_data = model(mix_data)          # モデルに通す
            # print("\nafter_model")
            # print(f"type(estimate_data):{type(estimate_data)}") #[1,1,音声長*チャンネル数]
            # print(f"estimate_data.shape:{estimate_data.shape}")
            # print(f"type(target_data):{type(target_data)}")
            # print(f"target_data.shape:{target_data.shape}")
            # print("after_model\n")

            """ データの整形 """
            # split_estimate_data = split_data(estimate_data[0],channels)
            # split_estimate_data = split_estimate_data[np.newaxis, :, :]
            # print("\nsplit_data")
            # print(f"type(split_estimate_data):{type(split_estimate_data)}")
            # print(f"split_estimate_data.shape:{split_estimate_data.shape}") # [1,チャンネル数,音声長]
            # print(f"type(target_data):{type(target_data)}")
            # print(f"target_data.shape:{target_data.shape}")
            # print("split_data\n")

            """ 損失の計算 """
            match loss_func:
                case "SISDR":
                    model_loss = si_sdr_loss(estimate_data, target_data)
                    # print(f"estimate:{estimate_data.shape}")
                    # print(f"target:{target_data.shape}")
                    # for estimate in estimate_data[0, :]:
                    #     # print_name_type_shape("estimate",estimate)
                    #     # print_name_type_shape("target_data[0,:]",target_data[0,:])
                    #     # print(f"estimate:{estimate.unsqueeze(0).shape}")
                    #     # print(f"estimate:{estimate.shape}")
                    #     # print(f"target:{target_data.shape}")
                    #     model_loss += si_sdr_loss(estimate.unsqueeze(0), target_data[0, :])  # si-sdrによる損失の計算
                    # model_loss = model_loss / channels
                case "wave_MSE":
                    model_loss = loss_function(estimate_data, target_data)  # 時間波形上でMSEによる損失関数の計算
                case "stft_MSE":
                    """ 周波数軸に変換 """
                    stft_estimate_data = torch.stft(estimate_data[0, 0, :], n_fft=1024, return_complex=False)
                    stft_target_data = torch.stft(target_data[0, 0, :], n_fft=1024, return_complex=False)
                    # print("\nstft")
                    # print(f"stft_estimate_data.shape:{stft_estimate_data.shape}")
                    # print(f"stft_target_data.shape:{stft_target_data.shape}")
                    # print("stft\n")
                    model_loss = loss_function(stft_estimate_data, stft_target_data)  # 時間周波数上MSEによる損失の計算
            # print(f"estimate_data.size(1):{estimate_data.size(1)}")

            model_loss_sum += model_loss  # 損失の加算

            """ 後処理 """
            model_loss.backward()           # 誤差逆伝搬
            optimizer.step()                # 勾配の更新

            # del mix_data, target_data, estimate_data, model_loss    # 使用していない変数の削除
            # torch.cuda.empty_cache()    # メモリの解放 1iterationごとに解放

        """ チェックポイントの作成 """
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": model_loss_sum},
                   f"{out_path}/{out_name}_ckp.pth")

        writer.add_scalar(str(out_name[0]), model_loss_sum, epoch)
        #writer.add_scalar(str(str_name[0]) + "_" + str(a) + "_sisdr-sisnr", model_loss_sum, epoch)
        print(f"[{epoch}]model_loss_sum:{model_loss_sum}")  # 損失の出力

        # torch.cuda.empty_cache()    # メモリの解放 1iterationごとに解放
        with open(csv_path, "a") as out_file:  # ファイルオープン
            out_file.write(f"{model_loss_sum}\n")  # 書き込み
        # torch.cuda.empty_cache()    # メモリの解放 1epochごとに解放-

        """ Early_Stopping の判断 """
        # best_lossとmodel_loss_sumを比較
        if model_loss_sum < best_loss:  # model_lossのほうが小さい場合
            print(f"{epoch:3} [epoch] | {model_loss_sum:.6} <- {best_loss:.6}")
            my_func.make_dir(out_path)  # best_modelの保存
            torch.save(model.to(device).state_dict(), f"{out_path}/BEST_{out_name}.pth")  # 出力ファイルの保存
            best_loss = model_loss_sum  # best_lossの変更
            earlystopping_count = 0
        else:
            earlystopping_count += 1
            if (epoch > 100) and (earlystopping_count > earlystopping_threshold):
                break
        if epoch == 100:
            torch.save(model.to(device).state_dict(), f"{out_path}/{out_name}_{epoch}.pth")  # 出力ファイルの保存

    """ 学習モデル(pthファイル)の出力 """
    print("model save")
    my_func.make_dir(out_path)
    torch.save(model.to(device).state_dict(), f"{out_path}/{out_name}_{epoch}.pth")         # 出力ファイルの保存

    writer.close()

    """ 学習時間の計算 """
    time_end = time.time()              # 現在時間の取得
    time_sec = time_end - start_time    # 経過時間の計算(sec)
    time_h = float(time_sec)/3600.0     # sec->hour
    print(f"time：{str(time_h)}h")      # 出力

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
        case '2stage':
            TasNet_model = Multichannel_model.type_D_2_2stage(num_mic=channels).to("cuda")

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

        # y_mixdown = addition_data(y_mixdown, channel=channel)
        y_mixdown = split_data(y_mixdown, channel=channel)

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
    # C:\Users\kataoka-lab\Desktop\sound_data\dataset\subset_DEMAND_hoth_1010dB_05sec_4ch_10cm
    base_name = "subset_DEMAND_hoth_1010dB_05sec_4ch_10cm"
    wave_type_list = ["noise_reverbe", "reverbe_only", "noise_only"]     # "noise_reverbe", "reverbe_only", "noise_only"
    angle_list = ["Right"]  # "Right", "FrontRight", "Front", "FrontLeft", "Left"
    model_list = ["E"]
    angle = "Right"
    channel = 4
    """ datasetの作成 """
    print("\n---------- make_dataset ----------")
    dataset_dir = f"{const.DATASET_DIR}/{base_name}/{angle}"
    # for wave_type in wave_type_list:
    #     # for angle in angle_list:
    #     # C:\Users\kataoka - lab\Desktop\sound_data\mix_data\subset_DEMAND_hoth_1010dB_1ch\subset_DEMAND_hoth_1010dB_05sec_1ch\train
    #     mix_dir = f"{const.MIX_DATA_DIR}/{base_name}/{angle}/train/"
    #     make_dataset.multi_to_single_dataset(mix_dir=os.path.join(mix_dir, wave_type),
    #                                          target_dir=os.path.join(mix_dir, "clean"),
    #                                          out_dir=os.path.join(dataset_dir, wave_type),
    #                                          channel=channel)
    """ train """
    print("\n---------- train ----------")
    pth_dir = ""
    for wave_type in wave_type_list:
        for model in model_list:
            pth_dir = f"{const.PTH_DIR}/{base_name}/{model}_2/{angle}"
            main(dataset_path=os.path.join(dataset_dir, wave_type),
                 out_path=os.path.join(pth_dir, f"{wave_type}_angle"),
                 train_count=100,
                 model_type=model,
                 channel=channel,
                 loss_func="stft_MSE")

    """ test_evaluation """
    condition = {"speech_type": "subset_DEMAND",
                 "noise": "hoth",
                 "snr": 10,
                 "reverbe": 5}
    for wave_type in wave_type_list:
        for model in model_list:
            pth_dir = f"{const.PTH_DIR}/{base_name}/{model}_2/{angle}"
            # name = "subset_DEMAND_hoth_1010dB_05sec_4ch_3cm"
            mix_dir = f"{const.MIX_DATA_DIR}\\{base_name}\\{angle}\\test\\"
            # mix_dir = f"{const.MIX_DATA_DIR}/{name}/test"
            out_wave_dir = f"{const.OUTPUT_WAV_DIR}/{base_name}/{model}/{angle}"
            print("\n---------- test ----------")
            test(mix_dir=os.path.join(mix_dir, wave_type),
                 out_dir=os.path.join(out_wave_dir, wave_type),
                 model_name=os.path.join(pth_dir, f"{wave_type}_angle", f"BEST_{wave_type}_angle.pth"),
                 channels=channel,
                 model_type=model)

            evaluation_path = f"{const.EVALUATION_DIR}/{base_name}/{angle}/{wave_type}.csv"
            print("\n---------- evaluation ----------")
            eval.main(target_dir=os.path.join(mix_dir, "clean"),
                      estimation_dir=os.path.join(out_wave_dir, wave_type),
                      out_path=evaluation_path,
                      condition=condition,
                      channel=channel)
