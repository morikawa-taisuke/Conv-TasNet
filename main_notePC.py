from __future__ import print_function

import argparse
import time
# from imghdr import test_tiff

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


import make_dataset
import Multi_Channel_ConvTasNet_train
import Multi_Channel_ConvTasNet_test as test
import All_evaluation as eval


def main(dataset_path, out_path, train_count, model_type, channel=1, checkpoint_path=None):
    """ 引数の処理 """
    parser = argparse.ArgumentParser(description="CNN Speech(Vocal) Separation")
    parser.add_argument("--dataset", "-t", default=dataset_path, help="Prefix Directory Name to input as dataset")
    parser.add_argument("--batchsize", "-b", type=int, default=const.BATCHSIZE, help="Number of track in each mini-batch")
    parser.add_argument("--patchlength", "-l", type=int, default=const.PATCHLEN, help="length of input frames in one track")
    parser.add_argument("--epoch", "-e", type=int, default=const.EPOCH, help="Number of sweeps over the dataset to train")
    parser.add_argument("--frequency", "-f", type=int, default=1, help="Frequency of taking a snapshot")
    # default=-1 only last, default=1 every epoch, write out snapshot
    parser.add_argument("--gpu", "-g", type=int, default=-1, help="GPU ID (negative value indicates CPU)")
    parser.add_argument("--resume", "-r", default="", help="Resume the training from snapshot")
    args = parser.parse_args()

    """ GPUの設定 """
    device = "cuda" if torch.cuda.is_available() else "cpu" # GPUが使えれば使う
    print(f"device:{device}")
    """ その他の設定 """
    out_name, _ = os.path.splitext(os.path.basename(out_path))  # 出力名の取得
    print("out_path: ", out_path)
    writer = SummaryWriter(log_dir=f"{const.LOG_DIR}\\{out_name}")  # logの保存先の指定("tensorboard --logdir ./logs"で確認できる)
    now = my_func.get_now_time()
    csv_path = f"{const.LOG_DIR}\\{out_name}\\{out_name}_{now}.csv"
    my_func.make_dir(csv_path)
    with open(csv_path, "w") as csv_file:  # ファイルオープン
        csv_file.write(f"dataset,out_name,model_type\n{dataset_path},{out_path},{model_type}")

    """ Load dataset データセットの読み込み """
    print(f"dataset:{args.dataset}")
    dataset = datasetClass.TasNet_dataset(args.dataset) # データセットの読み込み
    # print("\nmain_dataset")
    # print(f"type(dataset):{type(dataset)}")                                             # dataset2.TasNet_dataset
    # print(f"np.array(dataset.mix_list).shape:{np.array(dataset.mix_list).shape}")       # [データセットの個数,チャンネル数,音声長]
    # print(f"np.array(dataset.target_list).shape:{np.array(dataset.target_list).shape}") # [データセットの個数,1,音声長]
    # print("main_dataset\n")
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True)
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
            # print("estimate_data:", estimate_data.shape)
            # print("target_data:", target_data.shape)
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

    start_time = time.time()    # 時間を測定
    for epoch in range(start_epoch, train_count+1):   # 学習回数
        train(epoch)
        # torch.cuda.empty_cache()    # メモリの解放 1epochごとに解放-
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



if __name__ == "__main__":
    print("start")

    """ ファイル名等の指定 """
    base_name = "subset_DEMAND_hoth_1010dB_4ch/subset_DEMAND_hoth_1010dB_05sec_4ch_10cm"
    wave_type_list = ["noise_reverbe", "reverbe_only", "noise_only"]     # "noise_reverbe", "reverbe_only", "noise_only"
    angle_list = ["Left"]  # "Right", "FrontRight", "Front", "FrontLeft", "Left"
    channel = 4
    """ datasetの作成 """
    print("make_dataset")
    dataset_dir = f"{const.DATASET_DIR}/{base_name}/"
    for wave_type in wave_type_list:
        for angel in angle_list:
            mix_dir = f"{const.MIX_DATA_DIR}/{base_name}/train/"

            make_dataset.multi_channle_dataset2(mix_dir=os.path.join(mix_dir, wave_type),
                                                target_dir=os.path.join(mix_dir, "clean"),
                                                out_dir=os.path.join(dataset_dir, wave_type),
                                                channel=channel)
    """ train """
    print("train")
    pth_dir = f"{const.PTH_DIR}/{base_name}/"
    for wave_type in wave_type_list:
        main(dataset_path=os.path.join(dataset_dir, wave_type),
             out_path=os.path.join(pth_dir,wave_type),
             train_count=100,
             model_type="D",
             channel=channel)

    """ test_evaluation """
    condition = {"speech_type": "subset_DEMAND",
                 "noise": "hoth",
                 "snr": 10,
                 "reverbe": 5}
    for wave_type in wave_type_list:
        for angel in angle_list:
            mix_dir = f"{const.MIX_DATA_DIR}/{base_name}/test"
            out_wave_dir = f"{const.OUTPUT_WAV_DIR}/{base_name}/"
            print("test")
            test.test(mix_dir=os.path.join(mix_dir, wave_type),
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
