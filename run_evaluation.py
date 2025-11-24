# coding:utf-8
"""
このスクリプトは、評価の実行を開始するためのエントリーポイントです。

評価用設定ファイル (JSON) を読み込み、その内容に基づいて
学習済みモデルやテストデータローダーを生成し、
`src/evaluate.py` の `evaluate_and_save` 関数に渡して評価を実行します。
"""

import argparse
import yaml
import json
import torch
from torch.utils.data import DataLoader

# 自作モジュール
from src.evaluate import evaluate_and_save
from src.datasetClass import EvaluationDataset
from src.models import ConvTasNet_models, MultiChannel_ConvTasNet_models

def main():
    """ メイン関数 """
    parser = argparse.ArgumentParser(description="Conv-TasNet Evaluation Launcher")
    parser.add_argument("--config", "-c", required=True, help="Path to the evaluation configuration file (JSON).")
    args = parser.parse_args()

    # --- 1. 設定ファイルの読み込み ---
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- 2. デバイスの決定 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 3. オブジェクトの生成 ---

    # パス設定
    model_path = config["path"]["model"]
    test_data_dir = config["path"]["test_data"]
    output_dir = config["path"]["output"]

    # モデル設定
    model_config = config["model"]

    # --- データローダーの生成 ---
    # 評価では、ファイルごとに処理するためバッチサイズは1に固定
    test_dataset = EvaluationDataset(data_dir=test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- モデルの生成と学習済み重みの読み込み ---
    if model_config["channels"] == 1:
        if model_config["type"] == "enhance":
            model = ConvTasNet_models.enhance_ConvTasNet().to(device)
        else:  # separate
            model = ConvTasNet_models.separate_ConvTasNet().to(device)
    else:  # multi-channel
        model_type = model_config["type"]
        num_mic = model_config["channels"]
        if model_type == "A":
            model = MultiChannel_ConvTasNet_models.type_A().to(device)
        elif model_type == "C":
            model = MultiChannel_ConvTasNet_models.type_C().to(device)
        elif model_type == "D":
            model = MultiChannel_ConvTasNet_models.type_D_2(num_mic=num_mic).to(device)
        elif model_type == "E":
            model = MultiChannel_ConvTasNet_models.type_E(num_mic=num_mic).to(device)
        elif model_type == "F":
            model = MultiChannel_ConvTasNet_models.type_F().to(device)
        else:
            raise ValueError(f"Unknown multi-channel model type: {model_type}")

    # 学習済みモデルの重みを読み込む
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")

    # --- 4. 評価の実行 ---
    evaluate_and_save(
        model=model,
        test_loader=test_loader,
        output_dir=output_dir,
        device=device,
        model_channels=model_config["channels"]
    )

if __name__ == "__main__":
    main()
