# coding:utf-8
from __future__ import print_function

import argparse
import time
import os

from tqdm.contrib import tenumerate
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from itertools import permutations
# 自作モジュール
from src.utils import my_func, const
from data import datasetClass
from src.models import ConvTasNet_models as models

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
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)  #モデルの出力値 - 出力値の平均
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)  #教師データ - 教師データの平均
    t = torch.sum(x_zm * s_zm, dim=-1, keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))
# sisnr 損失関数?
def si_snr_loss(ests, egs):
    # spks x n x S
    refs = egs
    num_speekers = len(refs)

    def sisnr_loss(permute):    # snrの平均値
        # for one permute
        return sum([sisnr(ests[s], refs[t]) for s, t in enumerate(permute)]) / len(permute)
        # average the value

    # P x N
    N = egs.size(0)
    #print("N", N)
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
    #print("ests", ests.shape)
    #print("egs", egs.shape)
    refs = egs
    num_speekers = len(refs)
    #print("spks", num_speekers)

    def sisdr_loss(permute):
        # for one permute
        #print("permute", permute)
        return sum([sisdr(ests[s], refs[t])for s, t in enumerate(permute)]) / len(permute)
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


def main(dataset_path:str, out_path:str, train_count:int, loss_func:str="SISDR", model_type:str="enhance", checkpoint_path:str=None, accumulation_steps:int=1)->None:
    """
    ConvTasNetによる学習

    Parameters
    ----------
    dataset_path(str):使用するデータセットのパス
    out_name(str):出力ファイル(学習モデル)名
    train_count(int):学習回数
    loss_func(str):損失関数
    model_type(str):モデルのタイプ enhance->音源強調 separate->音源分離
    accumulation_steps(int):勾配を蓄積するステップ数

    Returns
    -------
    None
    """
    """ 引数の処理 """
    # ArgumentParser→コマンドラインで引数を受け取る処理を簡単に実装できるライブラリ
    parser = argparse.ArgumentParser(description="CNN Speech(Vocal) Separation")  # バーサを作る
    parser.add_argument("--dataset_path", "-t", default=dataset_path,
                        help="Prefix Directory Name to input as dataset")
    parser.add_argument("--batchsize", "-b", type=int, default=const.BATCHSIZE,
                        help="Number of track in each mini-batch")
    parser.add_argument("--patchlength", "-l", type=int, default=const.PATCHLEN,
                        help="length of input frames in one track")
    parser.add_argument("--frequency", "-f", type=int, default=1,
                        help="Frequency of taking a snapshot")
    parser.add_argument("--resume", "-r", default="",
                        help="Resume the training from snapshot")
    parser.add_argument("--accumulation_steps", type=int, default=accumulation_steps,
                        help="Number of steps to accumulate gradients before performing an optimizer step.")
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
            # dataset = datasetClass.TasNet_dataset(args.dataset_path)  # データセットの読み込み
            dataset = datasetClass.TasNet_dataset_csv(args.dataset_path, channel=1, device=device)  # データセットの読み込み
        case "separate":  # 音源分離
            dataset = datasetClass.TasNet_dataset_csv_separate(args.dataset_path, channel=1, device=device)  # データセットの読み込み
            # dataset = datasetClass.TasNet_dataset(args.dataset_path)  # データセットの読み込み
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    """ ネットワークの生成 """
    match model_type:
        case "enhance": # 音源強調
            model = models.enhance_ConvTasNet().to(device)
        case "separate":    # 音源分離
            model = models.separate_ConvTasNet().to(device)
    print(f"model_type:{model_type}")
    # print(f"\nmodel:{model}\n")                             # モデルのアーキテクチャの出力
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # optimizerを選択(Adam)
    if loss_func != "SISDR":
        loss_function = nn.MSELoss().to(device)  # 損失関数に使用する式の指定(最小二乗誤差)

    """ 混合精度訓練のためのGradScalerの初期化 """
    scaler = GradScaler(enabled=torch.cuda.is_available())

    """ 学習を途中から始めるかどうか """
    if checkpoint_path != None:
        print("restart_training")
        checkpoint = torch.load(checkpoint_path)    # checkpointの読み込み
        model.load_state_dict(checkpoint["model_state_dict"])   # 学習途中のモデルの読み込み
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])       # オプティマイザの読み込み
        # optimizerのstateを現在のdeviceに移す。これをしないと、保存前後でdeviceの不整合が起こる可能性がある。
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
    else:   # 初めから学習する場合
        start_epoch = 1

    start_time = time.time()  # 時間を測定

    """ 学習 """
    model.train()   # 学習モードに設定

    for epoch in range(start_epoch, train_count+1):   # 学習回数
        model_loss_sum_epoch = 0  # エポックごとの総損失の初期化
        optimizer.zero_grad() # エポック開始時に勾配をリセット
        # print(f"Train Epoch:{epoch}")   # 学習回数の表示
        for batch_idx, (mix_data, target_data) in tenumerate(dataset_loader):
            """ データの読み込み """
            mix_data, target_data = mix_data.to(device), target_data.to(device)  # 読み込んだデータをGPUに移動

            with autocast(enabled=torch.cuda.is_available()):
                """ データの整形 """
                mix_data, target_data = mix_data.to(torch.float32), target_data.to(torch.float32)   # データのタイプを変更 int16 → float32
                
                """ モデルに通す(予測値の計算) """
                estimate_data = model(mix_data) # モデルの適用

                """ 損失の計算 """
                match loss_func:
                    case "SISDR":
                        model_loss = si_sdr_loss(estimate_data[0], target_data)
                    case "waveMSE":
                        model_loss = loss_function(estimate_data, target_data)
                    case "stftMSE":
                        """ stft """
                        # autocast内でstftを行う場合、入力がfloatであることを確認
                        estimate_data_stft = torch.stft(estimate_data[0, :].float(), n_fft=1024, return_complex=True)
                        target_data_stft = torch.stft(target_data[0, :].float(), n_fft=1024, return_complex=True)
                        model_loss = loss_function(estimate_data_stft, target_data_stft)    # MSEによる損失の計算
            
            # 勾配累積のための損失の正規化
            model_loss = model_loss / args.accumulation_steps
            
            """ 誤差逆伝搬 """
            scaler.scale(model_loss).backward()   # スケーリングされた損失で誤差逆伝搬
            
            model_loss_sum_epoch += model_loss.item() * args.accumulation_steps # 正規化前の損失を記録

            # accumulation_steps ごとにパラメータを更新
            if (batch_idx + 1) % args.accumulation_steps == 0:
                scaler.step(optimizer)  # オプティマイザのステップ
                scaler.update()         # スケーラーの更新
                optimizer.zero_grad()   # 勾配のリセット

            del mix_data, target_data, estimate_data, model_loss  # 使用していない変数の削除
            # イテレーションごとのキャッシュクリアはパフォーマンスに影響するためコメントアウトまたは削除を推奨
            # torch.cuda.empty_cache() 

        """ 記録 """
        avg_epoch_loss = model_loss_sum_epoch / len(dataset_loader)
        writer.add_scalar(out_name, avg_epoch_loss, epoch)  # ログの記録 (平均損失を記録)
        print(f"[{epoch:3}] avg_epoch_loss: {avg_epoch_loss:.4f}\n")    # 損失の出力
        with open(csv_path, mode="a") as csv_file:
            csv_file.write(f"{epoch},{avg_epoch_loss}\n")
        
        if device == "cuda":
            torch.cuda.empty_cache()    # メモリの解放 1epochごとに解放
            
        """ 学習途中の出力 """
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": model_loss_sum_epoch},
                    f"{out_path}/{out_name}_cpk.pth")    # 途中経過の出力

    """ 学習モデル(pthファイル)の出力 """
    print("model save")
    torch.save(model.to(device).state_dict(), f"{out_path}/{out_name}_{epoch}.pth")  # 出力ファイルの保存

    writer.close()

    """ 学習時間の計算 """
    time_end = time.time()  # 現在時間の取得
    time_sec = time_end - start_time  # 経過時間の計算(sec)
    time_h = float(time_sec) / 3600.0  # sec->hour
    print(f"time：{str(time_h):.3}h")  # 出力

def test(mix_path:str, estimation_path:str, model_path:str, model_type:str="enhance")->None:
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
    mix_list = my_func.get_wave_list(mix_path)
    print(f"number of mixdown file:{len(mix_list)}")
    my_func.make_dir(estimation_path)

    """ GPUの設定 """
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPUが使えれば使う
    print(f"device:{device}")

    """ ネットワークの生成 """
    match model_type:
        case "enhance": # 音源強調
            model = models.enhance_ConvTasNet().to(device)
        case "separate":    # 音源分離
            model = models.separate_ConvTasNet.to(device)
        case _: # その他
            model = models.enhance_ConvTasNet().to(device)

    model.load_state_dict(torch.load(model_path))

    for mix_file in mix_list:   # tqdm():
        """ データの読み込み """
        mix_data, prm = my_func.load_wav(mix_file)  # mix_data:振幅 prm:音源のパラメータ
        """ データ型の調整 """
        mix_data = mix_data.astype(np.float32)  # データ形式の変更
        mix_data_max = np.max(mix_data)     # 最大値の取得
        mix_data = mix_data[np.newaxis, :]  # データ形状の変更 [音声長]->[1, 音声長]
        mix_data = torch.from_numpy(mix_data).to(device)    # データ型の変更 numpy->torch
        # print(f"mix_data:{mix_data.shape}")
        """ モデルの適用 """
        estimation_data = model(mix_data)   # モデルの適用
        # print(f"estimation_data:{estimation_data.shape}")
        """ 推測データ型の調整 """
        estimation_data = estimation_data.cpu() # cpuに移動
        estimation_data = estimation_data.detach().numpy()    # データ型の変更 torch->numpy
        estimation_data = estimation_data[0, 0, :]  # スライス
        estimation_data = estimation_data * (mix_data_max / np.max(estimation_data))    # データの正規化
        """ 保存 """
        out_name, _ = my_func.get_file_name(mix_file) # ファイル名の取得
        out_path = f"{estimation_path}/{out_name}.wav"
        my_func.save_wav(out_path, estimation_data, prm)    # 保存


if __name__ == "__main__":

    # for wave_type in ["noise_reverbe", "reverbe_only"]:
    #     for reverbe in range(1, 6):
    #         dataset_dir = f"{const.DATASET_DIR}\\subset_DEMAND_hoth_1010dB_1ch\\subset_DEMAND_hoth_1010dB_{reverbe:02}sec_1ch\\{wave_type}"
    #         main(dataset_path=dataset_dir,
    #              out_path=f"{const.PTH_DIR}\\subset_DEMAND_hoth_1010dB_{reverbe:02}sec_1ch\\{wave_type}",
    #              train_count=100,
    #              loss_func="stftMSE")
    # "C:\Users\kataoka-lab\Desktop\sound_data\dataset\OC_ConvTasNet"
    main(dataset_path=f"{const.DATASET_DIR}/OC_ConvTasNet/OC_ConvTasNet.csv",
         out_path=f"{const.PTH_DIR}/OC_ConvTasNet",
         train_count=100,
         loss_func="stftMSE",
         accumulation_steps=4) # accumulation_stepsの値を調整してください
