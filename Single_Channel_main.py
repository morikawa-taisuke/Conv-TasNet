""" 1ch雑音付加，データセット作成，学習，評価を行う　"""
import os.path
import argparse
import time
import os
from distutils.command.clean import clean

from tqdm import tqdm
from tqdm.contrib import tenumerate
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from itertools import permutations

from datasetClass import dataset
# 自作モジュール
from mymodule import my_func, const
import datasetClass
from models import ConvTasNet_models as models

import make_mixdown
import make_dataset
import ConvTasNet_train
import ConvTasNet_test
from mymodule import my_func, const
import All_evaluation

""" 損失関数 """
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
        raise RuntimeError("Dimention mismatch when calculate si-snr, {} vs {}".format(x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)  # モデルの出力値 - 出力値の平均
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)  # 教師データ - 教師データの平均
    t = torch.sum(x_zm * s_zm, dim=-1, keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True) ** 2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))
# sisnr 損失関数?
def si_snr_loss(ests, egs):
    # spks x n x S
    refs = egs
    num_speekers = len(refs)

    def sisnr_loss(permute):  # snrの平均値
        # for one permute
        return sum([sisnr(ests[s], refs[t]) for s, t in enumerate(permute)]) / len(permute)
        # average the value

    # P x N
    N = egs.size(0)
    # print("N", N)
    sisnr_mat = torch.stack([sisnr_loss(p) for p in permutations(range(num_speekers))])
    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    # si-snr
    return -torch.sum(max_perutt) / N
# sisdr
def sisdr(x, s, eps=1e-8):
    """calculate training loss

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
            "Dimention mismatch when calculate si-sdr, {} vs {}".format(x.shape, s.shape)
        )
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(x_zm * s_zm, dim=-1, keepdim=True) * s_zm / torch.sum(s_zm * s_zm, dim=-1, keepdim=True)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(t - x_zm) + eps))
# sisdr 損失関数?
def si_sdr_loss(ests, egs):
    # spks x n x S
    # print("ests", ests.shape)
    # print("egs", egs.shape)
    refs = egs
    num_speekers = len(refs)

    # print("spks", num_speekers)

    def sisdr_loss(permute):
        # for one permute
        # print("permute", permute)
        return sum([sisdr(ests[s], refs[t]) for s, t in enumerate(permute)]) / len(permute)
        # average the value

    # P x N
    N = egs.size(0)
    sisdr_mat = torch.stack(
        [sisdr_loss(p) for p in permutations(range(num_speekers))]
    )
    max_perutt, _ = torch.max(sisdr_mat, dim=0)
    # si-snr
    return -torch.sum(max_perutt) / N
# 最小二乗誤差
def mse_loss(ests, egs):
    """
    2つのテンソル間の平均二乗誤差（Mean Squared Error、MSE）を計算します。

    Args:
        ests (Tensor): 推定信号、N x S のテンソル。
        egs (Tensor): グラウンドトゥルー（正解）信号、N x S のテンソル。

    Returns:
        mse_loss (Tensor): ests と egs 間の平均二乗誤差を表すスカラーテンソル。
    """
    if ests.shape != egs.shape:
        raise RuntimeError("MSE の計算時に次元が一致しません。{} vs {}".format(ests.shape, egs.shape))

    # 平均二乗誤差を計算
    mse = torch.mean((ests - egs) ** 2)

    return mse

def main(dataset_path:str, out_path:str, train_count:int=100, loss_func:str="SISDR", model_type:str="enhance", checkpoint_path:str=None, win:int=2) -> None:
    """
    ConvTasNetによる学習

    Parameters
    ----------
    dataset_path(str):使用するデータセットのパス
    out_name(str):出力ファイル(学習モデル)名
    train_count(int):学習回数
    loss_func(str):損失関数
    model_type(str):モデルのタイプ enhance->音源強調 separate->音源分離

    Returns
    -------
    None
    """
    """ 引数の処理 """
    # ArgumentParser→コマンドラインで引数を受け取る処理を簡単に実装できるライブラリ
    parser = argparse.ArgumentParser(description="CNN Speech(Vocal) Separation")  # バーサを作る
    parser.add_argument("--dataset_path", "-t", default=dataset_path, help="Prefix Directory Name to input as dataset")
    parser.add_argument("--batchsize", "-b", type=int, default=const.BATCHSIZE, help="Number of track in each mini-batch")
    parser.add_argument("--patchlength", "-l", type=int, default=const.PATCHLEN, help="length of input frames in one track")
    parser.add_argument("--frequency", "-f", type=int, default=1, help="Frequency of taking a snapshot")
    parser.add_argument("--resume", "-r", default="", help="Resume the training from snapshot")
    args = parser.parse_args()

    """ GPUの設定 """
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPUが使えれば使う
    print(f"device:{device}")

    """ その他の設定 """
    out_name, _ = os.path.splitext(os.path.basename(out_path))  # 出力名の取得
    print("out_path: ", out_path)
    writer = SummaryWriter(log_dir=f"{const.LOG_DIR}\\{out_name}")  # logの保存先の指定("tensorboard --logdir ./logs"で確認できる)
    now = my_func.get_now_time()
    csv_path = f"{const.LOG_DIR}\\{out_name}\\{out_name}_{now}.csv"
    my_func.make_dir(csv_path)
    with open(csv_path, "w") as csv_file:  # 学習曲線をcsvでも保存
        csv_file.write(f"dataset,out_name,loss_func,model_type\n{dataset_path},{out_path},{loss_func},{model_type}")
    my_func.make_dir(out_path)
    """ Load dataset データセットの読み込み """
    match model_type:
        case "enhance":  # 音源強調
            dataset = datasetClass.TasNet_dataset(args.dataset_path)  # データセットの読み込み
        case "separate":  # 音源分離
            # dataset = datasetClass.TasNet_dataset_csv_separate(args.dataset_path)  # データセットの読み込み
            dataset = datasetClass.TasNet_dataset(args.dataset_path)  # データセットの読み込み
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    """ ネットワークの生成 """
    match model_type:
        case "enhance":  # 音源強調
            model = models.enhance_ConvTasNet(win=win).to(device)
        case "separate":  # 音源分離
            model = models.separate_ConvTasNet().to(device)
    print(f"model_type:{model_type}")
    # print(f"\nmodel:{model}\n")                             # モデルのアーキテクチャの出力
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # optimizerを選択(Adam)
    if loss_func != "SISDR":
        loss_function = nn.MSELoss().to(device)  # 損失関数に使用する式の指定(最小二乗誤差)

    """ 学習を途中から始めるかどうか """
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
        start_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
    else:  # 初めから学習する場合
        start_epoch = 1

    start_time = time.time()  # 時間を測定

    """ 学習 """
    model.train()  # 学習モードに設定

    for epoch in range(start_epoch, train_count + 1):  # 学習回数
        model_loss_sum = 0  # 総損失の初期化
        # print(f"Train Epoch:{epoch}")   # 学習回数の表示
        for batch_idx, (mix_data, target_data) in tenumerate(dataset_loader):
            """ データの読み込み """
            mix_data, target_data = mix_data.to(device), target_data.to(device)  # 読み込んだデータをGPUに移動
            """ 勾配のリセット """
            optimizer.zero_grad()  # optimizerの初期化
            """ データの整形 """
            mix_data, target_data = mix_data.to(torch.float32), target_data.to(
                torch.float32)  # データのタイプを変更 int16 → float32
            # target_data = target_data[np.newaxis, :, :]     # 次元の拡張 [1,音声長]→[1,1,音声長]
            """ モデルに通す(予測値の計算) """
            estimate_data = model(mix_data)  # モデルの適用
            # print("estimation:", estimate_data.shape)
            # print("target:", target_data.shape)

            """ 損失の計算 """
            match loss_func:
                case "SISDR":
                    model_loss = si_sdr_loss(estimate_data[0], target_data)
                case "waveMSE":
                    model_loss = loss_function(estimate_data, target_data)
                case "stftMSE":
                    """ stft """
                    estimate_data = torch.stft(estimate_data[0, :], n_fft=1024, return_complex=False)  # 周波数軸に変換
                    target_data = torch.stft(target_data[0, :], n_fft=1024, return_complex=False)  # 周波数軸に変換
                    model_loss = loss_function(estimate_data, target_data)  # MSEによる損失の計算
            model_loss_sum += model_loss  # 損失の加算
            """ 誤差逆伝搬 """
            model_loss.backward()  # 誤差逆伝搬
            optimizer.step()  # パラメータの更新
        """ 記録 """
        writer.add_scalar(out_name, model_loss_sum, epoch)  # ログの記録
        print(f"[{epoch:3}] model_loss_sum: {model_loss_sum}\n")  # 損失の出力
        with open(csv_path, mode="a") as csv_file:
            csv_file.write(f"{epoch},{model_loss_sum}\n")
        """ 学習途中の出力 """
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": model_loss_sum},
                   f"{out_path}/{out_name}_cpk.pth")  # 途中経過の出力

    """ 学習モデル(pthファイル)の出力 """
    print("model save")
    torch.save(model.to(device).state_dict(), f"{out_path}/{out_name}_{train_count}.pth")  # 出力ファイルの保存

    writer.close()

    """ 学習時間の計算 """
    time_end = time.time()  # 現在時間の取得
    time_sec = time_end - start_time  # 経過時間の計算(sec)
    time_h = float(time_sec) / 3600.0  # sec->hour
    print(f"time：{str(time_h):.3}h")  # 出力


if __name__ == "__main__":
    print("start")
    """ 学習の条件 """
    # C:\\Users\\kataoka-lab\\Desktop\\sound_data\\mix_data/subset_DEMAND_hoth_1010dB_1ch\\subset_DEMAND_hoth_1010db_05sec_01ch/train/noise_reverbe'
    # C:\\Users\\kataoka-lab\\Desktop\\sound_data\\mix_data\subset_DEMAND_hoth_1010dB_1ch\\subset_DEMAND_hoth_1010dB_05sec_1ch\train\noise_reverbe
    snr_list = [10] # SNRの指定(list型)
    reverbe_sec = 5  # 残響時間
    ch = 1  # マイク数
    train_count = 100   # 学習回数
    win_list = [2]  # 100, 200, 300, 400, 500
    
    wave_type_list = ["noise_reverbe", "reverbe_only", "noise_only"]    # "noise_reverbe", "reverbe_only", "noise_only"

    """ パス関係 """
    out_dir_name = f"DEMAND_hoth_10dB_500msec"  # 出力するディレクトリ名
    print(f"out_dir_name:{out_dir_name}")
    dataset_dir = f"{const.DATASET_DIR}/{out_dir_name}/"  # データセットの出力先

    """ 雑音付加 """
    # print("\n---mixdown---")
    # for reverbe_sec in range(1, 6):
    #     mix_dir = f"{const.MIX_DATA_DIR}/{out_dir_name}/{reverbe_sec:02}sec/train/"  # 混合信号の出力先
    #     dataset_dir = f"{const.DATASET_DIR}/{out_dir_name}/{reverbe_sec:02}sec/"
    #     for wave_type in wave_type_list:
    #         make_mixdown.add_noise_all(target_dir = os.path.join(mix_dir, "clean"),
    #                                    noise_file = os.path.join(mix_dir, wave_type),
    #                                    out_dir=os.path.join(dataset_dir, wave_type),
    #                                    snr_list=snr_list)  # SNR
    """ データセット作成 """
    print("\n---make_dataset---")
    # for reverbe_sec in range(1, 6):
    #     # C:\Users\kataoka-lab\Desktop\sound_data\mix_data\subset_DEMAND_hoth_1010dB_1ch\subset_DEMAND_hoth_1010dB_01sec_1ch\train\noise_reverbe
    mix_dir = f"{const.MIX_DATA_DIR}/{out_dir_name}/train/"  # 混合信号の出力先
    dataset_dir = f"{const.DATASET_DIR}/{out_dir_name}/"
    for wave_type in wave_type_list:
        make_dataset.enhance_save_stft(mix_dir=os.path.join(mix_dir, wave_type),    # 入力データ(目的信号+雑音)
                                       target_dir=os.path.join(mix_dir, "clean"),  # 教師データ
                                       out_dir=os.path.join(dataset_dir, wave_type)) # 出力先
    """ 学習 """
    print("\n---train---")
    # for reverbe_sec in range(1, 6):
    for wave_type in wave_type_list:
        # {out_dir_name}/{reverbe_sec:02}sec
        dataset_dir = f"{const.DATASET_DIR}/{out_dir_name}/"
        main(dataset_path=os.path.join(dataset_dir, wave_type),
             out_path=f"{const.PTH_DIR}/{out_dir_name}/{out_dir_name}_{wave_type}",
             train_count=train_count)  # 学習回数
    """ モデルの適用(テスト) """
    print("\n---test---")
    condition = {"speech_type": "subset_DEMAND",
                 "noise": "hoth",
                 "snr": 10,
                 "reverbe": 5}
    # for reverbe_sec in range(1, 6):
    for wave_type in wave_type_list:
        mix_dir = f"{const.MIX_DATA_DIR}/{out_dir_name}/test/"  # 混合信号の出力先
        estimation_dir = f"{const.OUTPUT_WAV_DIR}/{out_dir_name}/ConvTasNet/"   # モデル適用後の出力先
        ConvTasNet_test.test(mix_path=os.path.join(mix_dir, wave_type),    # テスト用データ
                             estimation_path=os.path.join(estimation_dir, wave_type),    # 出力先
                             model_path=os.path.join(const.PTH_DIR, f"{out_dir_name}", f"{out_dir_name}_{wave_type}", f"{out_dir_name}_{wave_type}_100.pth"))   # 使用するモデルのパス
        """ 評価 """
        All_evaluation.main(target_dir=os.path.join(mix_dir, "clean"),    # 教師データ
                            estimation_dir=os.path.join(estimation_dir, wave_type),  # 評価するデータ
                            out_path=f"{const.EVALUATION_DIR}/{out_dir_name}/{out_dir_name}_{wave_type}.csv",
                            condition=condition)    # 出力先

