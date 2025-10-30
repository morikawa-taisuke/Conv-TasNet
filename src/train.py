# coding:utf-8
import argparse
import yaml
import os
import time
import inspect  # 引数の有無を調べるために使用
import csv

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

# --- 自作モジュールのインポート ---
# (train.py がプロジェクトルートにある前提)
from src.utils import my_func, const
from src import datasetClass
from src import losses  # losses.py をインポート
from src.models import ConvTasNet_models, MultiChannel_ConvTasNet_models


def main(config_path: str):
	"""
	設定ファイルに基づいてConv-TasNetの学習を実行する統一スクリプト
	"""

	# --- 1. 設定ファイルの読み込み ---
	with open(config_path, 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)

	run_name = config['run_name']
	print(f"--- [ {run_name} ]: Starting Experiment ---")

	# --- 2. デバイス設定 ---
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"device: {device}")

	# --- 3. ログ・出力ディレクトリ設定 ---
	# const.py のパス定義 (e.g., 'C:\\Users\\...\\log') に基づいて結合
	log_dir_path = os.path.join(const.LOG_DIR, run_name)
	pth_dir_path = os.path.join(const.PTH_DIR, run_name)

	writer = SummaryWriter(log_dir=log_dir_path)
	now = my_func.get_now_time()

	# CSVログファイル
	csv_path = os.path.join(log_dir_path, f"{run_name}_{now}.csv")
	my_func.make_dir(csv_path)  # 親ディレクトリを作成

	# チェックポイント保存パス
	cpk_path = os.path.join(pth_dir_path, f"{run_name}_cpk.pth")
	my_func.make_dir(cpk_path)  # 親ディレクトリを作成

	print(f"log: {log_dir_path}")
	print(f"model: {pth_dir_path}")

	# CSVヘッダー書き込み
	with open(csv_path, "w", newline='', encoding='utf-8') as csv_file:
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow(["config_file", config_path])
		csv_writer.writerow(["run_name", run_name])
		csv_writer.writerow(["epoch", "avg_loss"])

	# --- 4. データローダーの動的構築 ---
	print("Initializing Dataloader...")
	dataset_config = config['dataset']
	loader_config = config['loader']

	DatasetClassName = getattr(datasetClass, dataset_config['name'])
	dataset_params = dataset_config['params'].copy()

	# datasetClassが 'device' 引数を取る場合 (TasNet_dataset_csvなど) のみ追加
	sig = inspect.signature(DatasetClassName.__init__)
	if 'device' in sig.parameters:
		dataset_params['device'] = device
		print(f"Passing 'device={device}' to {dataset_config['name']}")

	dataset = DatasetClassName(**dataset_params)
	dataset_loader = DataLoader(dataset,
	                            batch_size=loader_config['batch_size'],
	                            shuffle=loader_config['shuffle'],
	                            num_workers=loader_config.get('num_workers', 0),
	                            pin_memory=loader_config.get('pin_memory', True))
	print(f"Loaded Dataset: {dataset_config['name']} (Length: {len(dataset)})")

	# --- 5. モデルの動的構築 ---
	print("Initializing Model...")
	model_config = config['model']
	model_name = model_config['name']
	model_params = model_config['params']

	ModelClassName = None
	if hasattr(ConvTasNet_models, model_name):
		ModelClassName = getattr(ConvTasNet_models, model_name)
	elif hasattr(MultiChannel_ConvTasNet_models, model_name):
		ModelClassName = getattr(MultiChannel_ConvTasNet_models, model_name)
	else:
		raise ValueError(f"Model '{model_name}' not found in 'ConvTasNet_models.py' or 'MultiChannel_ConvTasNet_models.py'")

	model = ModelClassName(**model_params).to(device)
	print(f"Loaded Model: {model_name}")
	# print(model) # 必要ならモデル構造を表示

	# --- 6. 損失関数・オプティマイザ・スケーラーの構築 ---
	print("Initializing Loss & Optimizer...")
	train_config = config['training']

	# losses.py のファクトリ関数を呼び出す
	loss_function = losses.get_loss_function(train_config['loss_func'], device=device)
	print(f"Loss Function: {train_config['loss_func']}")

	if train_config['optimizer'].lower() == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
	else:
		# (必要なら他のオプティマイザを追加)
		raise ValueError(f"Optimizer {train_config['optimizer']} not supported.")

	# 混合精度（AMP）と勾配蓄積
	scaler = GradScaler(enabled=(device == "cuda"))
	accumulation_steps = train_config.get('accumulation_steps', 1)
	print(f"Gradient Accumulation Steps: {accumulation_steps}")

	# --- 7. チェックポイントからの再開 ---
	start_epoch = 1
	if train_config.get('checkpoint_path'):
		checkpoint_path = train_config['checkpoint_path']
		if os.path.exists(checkpoint_path):
			print(f"Loading checkpoint from: {checkpoint_path}")
			checkpoint = torch.load(checkpoint_path, map_location=device)
			model.load_state_dict(checkpoint["model_state_dict"])
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

			# オプティマイザのstateを現在のdeviceに移動
			for state in optimizer.state.values():
				for k, v in state.items():
					if isinstance(v, torch.Tensor):
						state[k] = v.to(device)

			start_epoch = checkpoint["epoch"] + 1
			print(f"Resuming training from epoch {start_epoch}")
		else:
			print(f"Checkpoint path not found (starting from scratch): {checkpoint_path}")

	# --- 8. 学習ループ ---
	print("--- Starting Training ---")
	start_time = time.time()
	model.train()  # 学習モード

	for epoch in range(start_epoch, train_config['epochs'] + 1):
		model_loss_sum_epoch = 0.0
		optimizer.zero_grad()  # 勾配蓄積のため、エポック開始時にリセット

		pbar = tqdm(dataset_loader, desc=f"Epoch {epoch}/{train_config['epochs']}")
		for batch_idx, (mix_data, target_data) in enumerate(pbar):

			mix_data, target_data = mix_data.to(device, non_blocking=True), target_data.to(device, non_blocking=True)

			# autocast: 混合精度計算
			with autocast(enabled=(device == "cuda")):
				mix_data = mix_data.to(torch.float32)
				target_data = target_data.to(torch.float32)

				estimate_data = model(mix_data)

				# 損失計算 (losses.py が (B, C, T) の波形入力を処理すると仮定)
				model_loss = loss_function(estimate_data, target_data)

			# 勾配蓄積のための損失正規化
			model_loss = model_loss / accumulation_steps

			# 誤差逆伝搬 (スケーラーを使用)
			scaler.scale(model_loss).backward()

			# 蓄積した損失（正規化前）を記録
			model_loss_sum_epoch += model_loss.item() * accumulation_steps

			# accumulation_steps ごと、または最終バッチでパラメータを更新
			if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataset_loader):
				scaler.step(optimizer)  # オプティマイザのステップ
				scaler.update()  # スケーラーの更新
				optimizer.zero_grad()  # 勾配のリセット

			pbar.set_postfix(loss=f"{model_loss.item() * accumulation_steps:.4f}")

		# --- 9. エポック終了時の記録 ---
		avg_epoch_loss = model_loss_sum_epoch / len(dataset_loader)

		# TensorBoard
		writer.add_scalar("Loss/train_avg", avg_epoch_loss, epoch)
		writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)

		print(f"Epoch {epoch:3} Summary: Avg. Loss = {avg_epoch_loss:.6f}")

		# CSV
		with open(csv_path, mode="a", newline='', encoding='utf-8') as csv_file:
			csv_writer = csv.writer(csv_file)
			csv_writer.writerow([epoch, f"{avg_epoch_loss:.6f}"])

		# チェックポイント保存
		torch.save({
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"loss": avg_epoch_loss,
		}, cpk_path)

	# --- 10. 学習終了 ---
	writer.close()
	print("--- Training Finished ---")

	# 最終モデルの保存
	final_pth_path = os.path.join(pth_dir_path, f"{run_name}_epoch{train_config['epochs']}.pth")
	torch.save(model.state_dict(), final_pth_path)
	print(f"Final model saved to: {final_pth_path}")

	time_end = time.time()
	time_sec = time_end - start_time
	time_h = float(time_sec) / 3600.0
	print(f"Total Training Time: {time_h:.3f} hours")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Unified Conv-TasNet Training Script")
	parser.add_argument("--config", "-c", type=str, required=True,
	                    help="Path to the experiment config file (.yaml)")
	args = parser.parse_args()

	main(args.config)