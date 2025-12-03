# coding:utf-8
"""
このスクリプトは、学習の実行を開始するためのエントリーポイントです。

設定ファイル (config.yml) を読み込み、その内容に基づいて
モデル、データローダー、オプティマイザ等の必要なオブジェクトを生成し、
`src/train.py` の `train` 関数に渡して学習を実行します。
"""

import argparse
import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import shutil  # shutilをインポート

# 自作モジュール
from src.train import train
from src.datasetClass import TasNet_dataset, TasNet_dataset_csv_separate
from src.CsvDataset import CsvDataset
from src.models import ConvTasNet_models, MultiChannel_ConvTasNet_models
from src.losses import get_loss_function
from src.utils import const


def main():
	""" メイン関数 """
	parser = argparse.ArgumentParser(description="Conv-TasNet Training Launcher")
	parser.add_argument("--config", "-c", required=True, help="Path to the configuration file (config.yml).")
	args = parser.parse_args()

	# --- 1. 設定ファイルの読み込み ---
	with open(args.config, encoding="utf-8") as f:
		config = yaml.safe_load(f)

	# --- 設定ファイルのコピー ---
	# チェックポイントが保存されるディレクトリに設定ファイルをコピーする
	output_dir = os.path.join(const.CHECKPOINT_DIR, config["common"]["out_dir_name"])
	os.makedirs(output_dir, exist_ok=True) # 出力ディレクトリを作成
	shutil.copy(args.config, os.path.join(output_dir, "config.yml"))
	print(f"Configuration file copied to {os.path.join(output_dir, 'config.yml')}")


	# --- 2. デバイスの決定 ---
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# --- 3. オブジェクトの生成 ---
	common_config = config["common"]  # パス設定

	train_dataset_path = os.path.join(const.MIX_DATA_DIR, common_config["dataset"], "train.csv")  # 学習用データセットのcsvのパス
	valid_dataset_path = os.path.join(const.MIX_DATA_DIR, common_config["dataset"], "val.csv")	# 評価用データセットのcsvパス

	# モデルと学習パラメータ設定を読み込む
	model_config = config["model"]
	train_config = config["train"]

	# --- データローダーの生成 ---
	batch_size = train_config["batch_size"] // train_config["accumulation_steps"]
	task_list = common_config["task_list"]
	max_length_sec = train_config.get("max_length_sec", None)

	for task in task_list:
		# --- データローダの生成 ---
		# 学習用
		if model_config["input_ch"] == 1:	# 1ch
			train_dataset = CsvDataset(csv_path=train_dataset_path, input_column_header=task, max_length_sec=max_length_sec)
		else:	# multi-ch
			train_dataset = TasNet_dataset(train_dataset_path)

		train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=CsvDataset.collate_fn)

		# 検証用
		if model_config["input_ch"] == 1:	# 1ch
			valid_dataset = CsvDataset(csv_path=valid_dataset_path, input_column_header=task, max_length_sec=max_length_sec)
		else:  # multi-channel separate
			valid_dataset = TasNet_dataset(valid_dataset_path)

		valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=CsvDataset.collate_fn)

		# --- モデルの生成 ---
		if model_config["input_ch"] == 1:
			model = ConvTasNet_models.enhance_ConvTasNet().to(device)
		else:  # multi-channel
			model_type = model_config.get("type", "D")  # デフォルト値を設定
			num_mic = model_config["input_ch"]
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
		optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])	# オプティマイザー
		loss_function = get_loss_function(train_config["loss_function"], device=device)	# 損失関数

		# (オプション) チェックポイントからの再開
		if train_config.get("checkpoint"):
			checkpoint_path = train_config["checkpoint"]	# 絶対パスを想定

			print(f"Resuming training from {checkpoint_path}")
			checkpoint = torch.load(checkpoint_path)
			model.load_state_dict(checkpoint["model_state_dict"])
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
			for state in optimizer.state.values():
				for k, v in state.items():
					if isinstance(v, torch.Tensor):
						state[k] = v.to(device)

		# --- 4. 学習の実行 ---
		# config全体を渡すように変更
		train(
			model=model,
			optimizer=optimizer,
			loss_function=loss_function,
			train_loader=train_loader,
			valid_loader=valid_loader,
			config=config,  # config全体を渡す
			device=device,
			task = task
		)

if __name__ == "__main__":
	main()
