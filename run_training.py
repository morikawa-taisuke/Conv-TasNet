# coding:utf-8
"""
このスクリプトは、学習の実行を開始するためのエントリーポイントです。

設定ファイル (config.json) を読み込み、その内容に基づいて
モデル、データローダー、オプティマイザ等の必要なオブジェクトを生成し、
`src/train.py` の `train` 関数に渡して学習を実行します。
"""

import argparse
# import json
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 自作モジュール
from src.train import train
from src.datasetClass import TasNet_dataset, TasNet_dataset_csv, TasNet_dataset_csv_separate
from src.CsvDataset import CsvDataset
from src.models import ConvTasNet_models, MultiChannel_ConvTasNet_models
from src.losses import get_loss_function

def main():
    """ メイン関数 """
    parser = argparse.ArgumentParser(description="Conv-TasNet Training Launcher")
    parser.add_argument("--config", "-c", required=True, help="Path to the training configuration file (JSON).")
    args = parser.parse_args()

    # --- 1. 設定ファイルの読み込み ---
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- 2. デバイスの決定 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 3. オブジェクトの生成 ---
    # パス設定
    path_config = config["path"]
    train_dataset_path = path_config["dataset"]
    valid_dataset_path = path_config.get("validation") # 検証データパス（オプション）

    # モデルとバッチサイズ設定
    model_config = config["model"]
    batch_size = config["training"]["batch_size"]

    # --- データローダーの生成 ---
    # YAMLから列名を取得 (デフォルト値も設定)
    dataset_config = config.get("dataset", {})  # "dataset"セクションを取得
    mix_col = dataset_config.get("input", "noise_reverb")
    target_col = dataset_config.get("target", "clean")
    batch_size = dataset_config.get("batch_size", batch_size)

    # 学習用
    if model_config["channels"] == 1:
        if model_config["type"] == "enhance":
            train_dataset = CsvDataset(csv_path=train_dataset_path, input_column_header=mix_col, max_length_sec=5)
        else:  # separate
            train_dataset = TasNet_dataset_csv_separate(train_dataset_path, channel=1, device=device, mix_column=mix_col, target_column=target_col)
    else:  # multi-channel
        train_dataset = TasNet_dataset(train_dataset_path)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=CsvDataset.collate_fn)

    # 検証用 (パスが指定されている場合のみ)
    valid_loader = None
    if valid_dataset_path:
        if model_config["channels"] == 1:
            if model_config["type"] == "enhance":
                valid_dataset = CsvDataset(csv_path=valid_dataset_path, input_column_header=mix_col, max_length_sec=5)
            else: # separate
                valid_dataset = TasNet_dataset_csv_separate(valid_dataset_path, channel=1, device=device)
        else: # multi-channel
            valid_dataset = TasNet_dataset(valid_dataset_path)
        
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=CsvDataset.collate_fn)

    # --- モデルの生成 ---
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

    # --- オプティマイザと損失関数の生成 ---
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    loss_function = get_loss_function(config["training"]["loss_function"], device=device)
    
    # (オプション) チェックポイントからの再開
    if config.get("resume_checkpoint"):
        checkpoint_path = config["resume_checkpoint"]
        print(f"Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # --- 4. 学習の実行 ---
    train(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        train_loader=train_loader,
        valid_loader=valid_loader, # 検証用ローダーを渡す
        config=config,
        device=device
    )

if __name__ == "__main__":
    main()
