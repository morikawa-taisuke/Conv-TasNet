# coding:utf-8
"""
評価（推論）のコアロジックを担うモジュール。
"""
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.utils import my_func
from src.data.make_dataset import split_data  # Multi-channelで使用
from src.evaluation.SI_SDR import sisdr_evaluation
from src.evaluation.PESQ import pesq_evaluation
from src.evaluation.STOI import stoi_evaluation


def evaluate_and_save(model, test_loader, target_loader, output_dir, device, config):
    """
    モデルで推論を行い、結果をWAVファイルとして保存し、客観評価を実行する。

    Parameters
    ----------
    model (torch.nn.Module): 評価対象のモデル
    test_loader (torch.utils.data.DataLoader): テスト用（混合音声）データローダー
    target_loader (torch.utils.data.DataLoader): 正解音声用データローダー
    output_dir (str): 結果を保存するディレクトリのパス
    device (str): "cuda" or "cpu"
    config (dict): 評価設定を含む辞書
    """
    model.eval()  # 評価モード
    my_func.make_dir(output_dir)

    model_channels = config["model"]["channels"]
    results = [] # 評価結果を格納するリスト

    print(f"Starting evaluation... Output will be saved to: {output_dir}")

    with torch.no_grad():
        for (mix_data, file_path), (target_data, _) in tqdm(zip(test_loader, target_loader), desc="Evaluating files", total=len(test_loader)):
            file_path = file_path[0]
            mix_data = mix_data.to(torch.float32).to(device)
            target_data = target_data.to(torch.float32)
            mix_data_max = torch.max(torch.abs(mix_data))

            if mix_data.dim() == 2:
                mix_data = mix_data.unsqueeze(1)

            if model_channels > 1 and mix_data.shape[1] == 1:
                mix_data = split_data(mix_data, channel=model_channels)
                mix_data = torch.from_numpy(mix_data).to(device)

            estimate_data = model(mix_data)
            estimate_data = estimate_data.cpu().detach().numpy()

            if estimate_data.ndim == 3:
                estimate_data = estimate_data[0, 0, :]
            elif estimate_data.ndim == 2:
                estimate_data = estimate_data[0, :]

            if np.max(np.abs(estimate_data)) > 1e-8:
                estimate_data = estimate_data * (mix_data_max.cpu().numpy() / np.max(np.abs(estimate_data)))

            out_name = os.path.basename(file_path)
            out_path = os.path.join(output_dir, out_name + ".wav")
            my_func.save_wav(Path(out_path), estimate_data)

            # --- 客観評価の実行 ---
            target_data_np = target_data.squeeze().cpu().numpy()
            min_len = min(len(target_data_np), len(estimate_data))
            target_data_np = target_data_np[:min_len]
            estimate_data = estimate_data[:min_len]

            sdr_score = sisdr_evaluation(target_data_np, estimate_data).item()
            pesq_score = pesq_evaluation(target_data_np, estimate_data)
            stoi_score = stoi_evaluation(target_data_np, estimate_data)

            results.append({
                "filename": out_name,
                "SI-SDR": sdr_score,
                "PESQ": pesq_score,
                "STOI": stoi_score
            })

    # --- 評価結果をCSVに保存 ---
    if results:
        df = pd.DataFrame(results)
        result_csv_path = os.path.join(output_dir, "_evaluation_scores.csv")
        
        # 平均と分散を計算
        means = df[["SI-SDR", "PESQ", "STOI"]].mean().rename("mean")
        variances = df[["SI-SDR", "PESQ", "STOI"]].var().rename("variance")

        # 結果を結合
        summary_df = pd.concat([means, variances], axis=1)
        
        # ファイルに書き込み
        df.to_csv(result_csv_path, index=False)
        with open(result_csv_path, 'a') as f:
            f.write('\n')
            summary_df.to_csv(f)

        print(f"Evaluation scores saved to {result_csv_path}")

    print("Evaluation finished.")
