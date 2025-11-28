# coding:utf-8
"""
学習ループのコアロジックを担うモジュール。
この中の`train`関数は、特定のモデルやデータセットに依存せず、
外部から注入されたオブジェクトを使って学習を実行します。
"""
from __future__ import print_function

import time
import os
import json

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from src.utils import const

from src.utils import my_func


def validation(model, loss_function, valid_loader, device, use_amp, loss_func_name):
	"""モデルを検証データで評価する関数"""
	model.eval()  # 評価モード
	total_loss = 0.0
	with torch.no_grad():
		for mix_data, target_data in tqdm(valid_loader, desc="Validating", leave=False):
			mix_data, target_data = mix_data.to(device, non_blocking=True), target_data.to(device, non_blocking=True)
			mix_data, target_data = mix_data.to(torch.float32), target_data.to(torch.float32)

			with autocast(enabled=use_amp):
				estimate_data = model(mix_data)

				if loss_func_name in ["SISDR", "SISNR"] and mix_data.size(0) > 1:
					loss = 0
					for i in range(mix_data.size(0)):
						loss += loss_function(estimate_data[i].unsqueeze(0), target_data[i].unsqueeze(0))
					loss /= mix_data.size(0)
				else:
					loss = loss_function(estimate_data, target_data)

			total_loss += loss.item()

	return total_loss / len(valid_loader)


def train(model, optimizer, loss_function, train_loader, valid_loader, config, device, task):
	"""
	汎用的な学習実行関数 (Dependency Injection 版)

	Parameters
	----------
	valid_loader (torch.utils.data.DataLoader): 検証用データローダー
	...
	"""

	print("==================================================")
	print("train")
	print("[学習の設定]:")
	print(json.dumps(config, indent=4))
	print("==================================================")

	""" 設定の展開 """
	train_config = config["train"]
	out_dir_name = config["common"]["out_dir_name"]	# 出力ディレクトリの名前
	out_model_name = f"{out_dir_name}_{task}"
	out_dir = os.path.join(const.CHECKPOINT_DIR, out_dir_name)	# 出力ディレクトリの絶対パス
	my_func.make_dir(out_dir)

	max_epoch = train_config["max_epoch"]	# 最大学習回数
	loss_func_name = train_config["loss_function"]	# 損失関数名
	accumulation_steps = train_config.get("accumulation_steps", 1)	# 勾配の蓄積回数
	use_amp = train_config.get("amp", False) and torch.cuda.is_available()

	early_stopping_threshold = train_config.get("early_stopping_threshold", 10)

	""" ログ・チェックポイント設定 """
	log_dir = os.path.join(const.LOG_DIR, out_model_name)
	writer = SummaryWriter(log_dir=log_dir)
	now = my_func.get_now_time()
	csv_path = os.path.join(log_dir, f"{out_model_name}_{now}.csv")
	my_func.make_dir(csv_path)
	with open(csv_path, "w") as csv_file:
		json.dump(config, csv_file, indent=4)
		csv_file.write("\n\nepoch,train_loss,valid_loss\n")

	scaler = GradScaler(enabled=use_amp)
	best_valid_loss = np.inf
	early_stopping_counter = 0
	start_time = time.time()

	""" 学習ループ """
	epoch = 0
	for epoch in range(1, max_epoch + 1):
		model.train()  # 学習モード
		train_loss = 0.0
		optimizer.zero_grad()

		# --- 学習 ---
		for i, (mix_data, target_data) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{max_epoch}")):
			mix_data, target_data = mix_data.to(device, non_blocking=True), target_data.to(device, non_blocking=True)
			mix_data, target_data = mix_data.to(torch.float32), target_data.to(torch.float32)

			with autocast(enabled=use_amp):
				estimate_data = model(mix_data)
				if loss_func_name in ["SISDR", "SISNR"] and mix_data.size(0) > 1:
					loss = 0
					for j in range(mix_data.size(0)):
						loss += loss_function(estimate_data[j].unsqueeze(0), target_data[j].unsqueeze(0))
					loss /= mix_data.size(0)
				else:
					loss = loss_function(estimate_data, target_data)

			loss = loss / accumulation_steps
			scaler.scale(loss).backward()
			train_loss += loss.item() * accumulation_steps

			if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad()

		avg_train_loss = train_loss / len(train_loader)
		writer.add_scalar("Loss/train", avg_train_loss, epoch)

		# --- 検証ステップ ---
		avg_valid_loss = validation(model, loss_function, valid_loader, device, use_amp, loss_func_name)
		writer.add_scalar("Loss/validation", avg_valid_loss, epoch)
		print(f"[{epoch:3}] train_loss: {avg_train_loss:.4f}, valid_loss: {avg_valid_loss:.4f}")

		with open(csv_path, "a") as f:
			f.write(f"{epoch},{avg_train_loss},{avg_valid_loss}\n")

		if device == "cuda":
			torch.cuda.empty_cache()

		# --- チェックポイントと早期終了の判断 ---
		torch.save(
			{
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"loss": avg_train_loss,
			},
			f"{out_dir}/{out_model_name }_cpk.pth"
		)

		if avg_valid_loss < best_valid_loss:
			torch.save(model.state_dict(), f"{out_dir}/{out_model_name}_best.pth")
			best_valid_loss = avg_valid_loss
			early_stopping_counter = 0
			print(f"  -> Best model saved. Loss: {best_valid_loss:.4f}")
		else:
			early_stopping_counter += 1
			print(f"  -> Early stopping counter: {early_stopping_counter}/{early_stopping_threshold}")

		if early_stopping_counter >= early_stopping_threshold:
			print("Early stopping triggered.")
			break

	print("Training finished.")
	torch.save(model.state_dict(), f"{out_dir}/{out_model_name}_{epoch}.pth")
	writer.close()

	time_end = time.time()
	time_sec = time_end - start_time
	print(f"Training time: {time_sec / 3600:.3f}h")
