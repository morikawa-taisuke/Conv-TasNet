# coding:utf-8
"""
評価（推論）のコアロジックを担うモジュール。
"""
import os
import torch
import numpy as np
from tqdm import tqdm

from src.utils import my_func
from data.make_dataset import split_data # Multi-channelで使用

def evaluate_and_save(model, test_loader, output_dir, device, model_channels):
    """
    モデルで推論を行い、結果をWAVファイルとして保存する。

    Parameters
    ----------
    model (torch.nn.Module): 評価対象のモデル
    test_loader (torch.utils.data.DataLoader): テスト用データローダー
    output_dir (str): 結果を保存するディレクトリのパス
    device (str): "cuda" or "cpu"
    model_channels (int): モデルの入力チャネル数
    """
    model.eval()  # 評価モード
    my_func.make_dir(output_dir)

    print(f"Starting evaluation... Output will be saved to: {output_dir}")

    with torch.no_grad():
        for mix_data, prm, file_path in tqdm(test_loader, desc="Evaluating files"):
            # データローダーからの出力はバッチになっているので、最初の要素を取得
            prm = {k: v[0] if isinstance(v, list) else v for k, v in prm.items()} # パラメータをデコード
            file_path = file_path[0]
            
            # データをfloat32に変換し、デバイスに送信
            mix_data = mix_data.to(torch.float32).to(device)
            
            # 元の振幅の最大値を保持（正規化のため）
            mix_data_max = torch.max(torch.abs(mix_data))

            # モデルの入力形式に合わせる
            # (B, C, T) or (B, T) -> (B, C, T)
            if mix_data.dim() == 2: # (B, T)
                mix_data = mix_data.unsqueeze(1)

            # 多チャネルモデルの場合の前処理
            # Note: この部分は元のtest関数にあったロジックを参考にしていますが、
            # データセットの作り方によっては調整が必要かもしれません。
            if model_channels > 1 and mix_data.shape[1] == 1:
                # (B, 1, C*T) -> (B, C, T) のような変換を想定
                mix_data = split_data(mix_data, channel=model_channels)
                mix_data = torch.from_numpy(mix_data).to(device)

            # モデルで推論
            estimate_data = model(mix_data)

            # 推論結果をCPUに移動し、numpy配列に変換
            estimate_data = estimate_data.cpu().detach().numpy()
            
            # (B, C, T) -> (T,) の形式に整形
            # ここでは最初のバッチ、最初のチャネルの出力を保存すると仮定
            if estimate_data.ndim == 3:
                estimate_data = estimate_data[0, 0, :]
            elif estimate_data.ndim == 2:
                estimate_data = estimate_data[0, :]

            # 元の振幅スケールに正規化
            if np.max(np.abs(estimate_data)) > 1e-8:
                estimate_data = estimate_data * (mix_data_max.cpu().numpy() / np.max(np.abs(estimate_data)))

            # 保存パスの生成
            out_name = os.path.basename(file_path)
            out_path = os.path.join(output_dir, out_name)

            # WAVファイルとして保存
            my_func.save_wav(out_path, estimate_data.astype(np.int16), prm)

    print("Evaluation finished.")
