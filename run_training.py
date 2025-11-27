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
	# 文字コードをUTF-8に指定してファイルを開く
	with open(args.config, encoding="utf-8") as f:
		config = yaml.safe_load(f)

	# --- 2. デバイスの決定 ---
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# --- 3. オブジェクトの生成 ---
	path_config = config["path"]  # パス設定
	train_dataset_path = os.path.join(const.DATASET_DIR, path_config["train_dataset"], "train.csv")  # 学習用データセットのcsvのパス

	valid_dataset_path = os.path.join(const.DATASET_DIR, path_config["train_dataset"], "val.csv")
	valid_dataset_path = valid_dataset_path if os.path.isfile(valid_dataset_path) else None	# 評価用のデータセットがあるかどうか確認

	# モデルと学習パラメータ設定
	model_config = config["model"]
	training_config = config["training"]
	train_params = training_config["params"]
	dataset_params = training_config["dataset"]

	# --- データローダーの生成 ---
	batch_size = train_params["batch_size"] // train_params["accumulation_steps"]
	mix_col = dataset_params["input_column"]
	max_length_sec = dataset_params.get("max_length_sec", None)

	# 学習用
	if model_config["type"] == "enhance":
		train_dataset = CsvDataset(csv_path=train_dataset_path, input_column_header=mix_col, max_length_sec=max_length_sec)
	elif model_config["channels"] == 1:  # separate
		train_dataset = TasNet_dataset_csv_separate(train_dataset_path, channel=1, device=device)
	else:  # multi-channel separate
		train_dataset = TasNet_dataset(train_dataset_path)

	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
							  collate_fn=CsvDataset.collate_fn if model_config["type"] == "enhance" else None)

	# 検証用 (パスが指定されている場合のみ)
	valid_loader = None
	if valid_dataset_path:
		if model_config["type"] == "enhance":
			valid_dataset = CsvDataset(csv_path=valid_dataset_path, input_column_header=mix_col, max_length_sec=max_length_sec)
		elif model_config["channels"] == 1:  # separate
			valid_dataset = TasNet_dataset_csv_separate(valid_dataset_path, channel=1, device=device)
		else:  # multi-channel separate
			valid_dataset = TasNet_dataset(valid_dataset_path)

		valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
								  collate_fn=CsvDataset.collate_fn if model_config["type"] == "enhance" else None)

	# --- モデルの生成 ---
	if model_config["channels"] == 1:
		if model_config["type"] == "enhance":
			model = ConvTasNet_models.enhance_ConvTasNet().to(device)
		else:  # separate
			model = ConvTasNet_models.separate_ConvTasNet().to(device)
	else:  # multi-channel
		model_type = model_config.get("type", "D")  # デフォルト値を設定
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
	optimizer = optim.Adam(model.parameters(), lr=train_params["learning_rate"])
	loss_function = get_loss_function(train_params["loss_function"], device=device)

	# (オプション) チェックポイントからの再開
	if training_config.get("resume_checkpoint"):
		checkpoint_path = training_config["resume_checkpoint"]
		# experiment_dirからの相対パスの場合、結合する
		if not os.path.isabs(checkpoint_path):
			checkpoint_path = os.path.join(path_config["experiment_dir"], checkpoint_path)

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
		task = mix_col
	)

if __name__ == "__main__":
	main()
