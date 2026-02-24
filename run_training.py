# coding:utf-8
"""
このスクリプトは、学習の実行を開始するためのエントリーポイントです。

設定ファイル (configs/config.yml) を読み込み、その内容に基づいて
モデル、データローダー、オプティマイザ等の必要なオブジェクトを生成し、
PyTorch LightningのTrainerにて学習を実行します。
"""

import argparse
import os
import yaml
import json
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import shutil

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# 自作モジュール
from src.lightning_modules.conv_tasnet import ConvTasNetLightning
from src.data.dataset_class import TasNet_dataset, TasNet_dataset_csv_separate
from src.data.csv_dataset import CsvDataset
from src.models import ConvTasNet_models, New_MultiChannel_ConvTasNet_models, MultiChannel_ConvTasNet_models
from src.losses import get_loss_function
from src.utils import const
from src.utils import my_func

def main():
	""" メイン関数 """
	parser = argparse.ArgumentParser(description="Conv-TasNet Training Launcher")
	parser.add_argument("--config", "-c", default="configs/config.yml", help="Path to the configuration file (default: configs/config.yml).")
	args = parser.parse_args()

	# --- 1. 設定ファイルの読み込み ---
	with open(args.config, encoding="utf-8") as f:
		config = yaml.safe_load(f)

	# --- 2. デバイスの決定 ---
	device = torch.device("cuda" if torch.cuda.is_available()  else "mps" if torch.backends.mps.is_available() else "cpu")

	# --- 3. オブジェクトの生成 ---
	common_config = config["common"]  # パス設定

	train_dataset_path = os.path.join(const.MIX_DATA_DIR, common_config["dataset"], "train.csv")  # 学習用データセットのcsvのパス
	valid_dataset_path = os.path.join(const.MIX_DATA_DIR, common_config["dataset"], "val.csv")	# 評価用データセットのcsvパス

	# モデルと学習パラメータ設定を読み込む
	model_config = config["model"]
	train_config = config["train"]

	# --- データローダーの生成 ---
	batch_size = train_config["batch_size"] // train_config.get("accumulation_steps", 1)
	task_list = common_config["task_list"]
	max_length_sec = train_config.get("max_length_sec", None)

	for task in task_list:
		# --- データローダの生成 ---
		# 学習用
		if model_config["input_ch"] == 1:	# 1ch
			train_dataset = CsvDataset(csv_path=train_dataset_path, input_column_header=task, max_length_sec=max_length_sec)
		else:	# multi-ch
			train_dataset = CsvDataset(csv_path=train_dataset_path, input_column_header=task, max_length_sec=max_length_sec)

		train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=CsvDataset.collate_fn)

		# 検証用
		if model_config["input_ch"] == 1:	# 1ch
			valid_dataset = CsvDataset(csv_path=valid_dataset_path, input_column_header=task, max_length_sec=max_length_sec)
		else:  # multi-channel separate
			valid_dataset = CsvDataset(csv_path=valid_dataset_path, input_column_header=task, max_length_sec=max_length_sec)

		valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=CsvDataset.collate_fn)

		# --- モデルの生成 ---
		if model_config["input_ch"] == 1:
			model = ConvTasNet_models.enhance_ConvTasNet()
		else:  # multi-channel
			model_type = model_config.get("type", "D")  # デフォルト値を設定
			num_mic = model_config["input_ch"]
			if model_type == "A":
				model = New_MultiChannel_ConvTasNet_models.type_A()
			elif model_type == "C":
				model = New_MultiChannel_ConvTasNet_models.type_C(channel=num_mic)
			elif model_type == "D":
				model = New_MultiChannel_ConvTasNet_models.type_D_2(num_mic=num_mic)
			elif model_type == "E":
				model = New_MultiChannel_ConvTasNet_models.type_E(num_mic=num_mic)
			elif model_type == "F":
				model = New_MultiChannel_ConvTasNet_models.type_F()
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

		# --- 4. 学習の実行 (PyTorch Lightning) ---
		print("==================================================")
		print(f"train (PyTorch Lightning with W&B) - Task: {task}")
		print("[学習の設定]:")
		print(json.dumps(config, indent=4))
		print("==================================================")

		out_dir_name = config["common"]["out_dir_name"]
		out_model_name = f"{out_dir_name}_{task}"
		out_dir = os.path.join(const.CHECKPOINT_DIR, out_dir_name)
		my_func.make_dir(os.path.join(out_dir_name, out_dir_name))

		# --- 設定ファイルのコピー ---
		# チェックポイントが保存されるディレクトリに設定ファイルをコピーする
		os.makedirs(out_dir, exist_ok=True) # 出力ディレクトリを作成
		shutil.copy(args.config, os.path.join(out_dir, "config.yml"))
		print(f"Configuration file copied to {os.path.join(out_dir, 'config.yml')}")

		# ロギング設定
		log_dir = "logs"
		os.makedirs(log_dir, exist_ok=True)
		
		# W&B Logger
		wandb_logger = WandbLogger(
			project="ConvTasNet",
			name=out_model_name,
			save_dir=log_dir,
			config=config
		)

		# Callbacks
		early_stopping_threshold = train_config.get("early_stopping_threshold", 10)
		early_stop_callback = EarlyStopping(
			monitor="Loss/validation",
			patience=early_stopping_threshold,
			mode="min",
			verbose=True
		)

		checkpoint_callback = ModelCheckpoint(
			dirpath=out_dir,
			filename=f"{out_model_name}_best",
			monitor="Loss/validation",
			mode="min",
			save_top_k=1,
			save_last=True
		)

		# Trainer Setup
		max_epoch = train_config["max_epoch"]
		accumulation_steps = train_config.get("accumulation_steps", 1)
		use_amp = train_config.get("amp", False)
		precision = "16-mixed" if use_amp else 32

		trainer = pl.Trainer(
			max_epochs=max_epoch,
			accelerator="auto",
			devices=1,
			precision=precision,
			accumulate_grad_batches=accumulation_steps,
			callbacks=[early_stop_callback, checkpoint_callback],
			logger=wandb_logger,
			enable_progress_bar=True
		)

		lightning_model = ConvTasNetLightning(
			model=model,
			optimizer=optimizer,
			loss_function=loss_function,
			config=config
		)

		start_time = time.time()
		trainer.fit(
			model=lightning_model,
			train_dataloaders=train_loader,
			val_dataloaders=valid_loader
		)

		time_end = time.time()
		time_sec = time_end - start_time
		print(f"Training finished. Time: {time_sec / 3600:.3f}h")

if __name__ == "__main__":
	main()
