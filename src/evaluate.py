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
from src.utils import const


def evaluate_and_save(model, test_loader, device, config, out_dir_name):
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
	# パスの生成
	task = test_loader.dataset.input_column
	out_wav_dir = os.path.join(const.OUTPUT_WAV_DIR, out_dir_name, f"{out_dir_name}_{task}")
	out_csv_path = os.path.join(const.EVALUATION_DIR, out_dir_name, f"{out_dir_name}_{task}.csv")
	input_csv_dir = test_loader.dataset.csv_dir
	print("==================================================")
	print("test")
	print("入力するcsvのパス: ", input_csv_dir)
	print("推論データの出力先: ", out_wav_dir)
	print("客観評価のcsvファイルの出力先: ", out_csv_path)
	print("==================================================")
	model.eval()  # 評価モード
	my_func.make_dir(out_dir_name)
	my_func.make_dir(out_csv_path)

	results = []  # 評価結果を格納するリスト

	with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
		# メタ情報の書き込み
		f.write(f"input_csv,{input_csv_dir}\n")
		f.write(f"task,{task}\n")
		f.write(f"estimation_dir,{out_wav_dir}\n")
		# CSVヘッダーの書き込み
		f.write("target_name,estimation_name,pesq,stoi,sisdr\n")

	with torch.no_grad():
		for mix_data, target_data, mix_name, clean_name in tqdm(test_loader, desc="Evaluating files", total=len(test_loader)):
			mix_data = mix_data.to(torch.float32).to(device)
			target_data = target_data.to(torch.float32)
			mix_data_max = torch.max(torch.abs(mix_data))

			estimate_data = model(mix_data)
			# print(estimate_data.shape)

			# 推論したデータの調整
			# 形状の調整
			if estimate_data.ndim == 3:
				estimate_data = estimate_data[0, :]

			# 振幅の大きさの調整
			if torch.max(torch.abs(estimate_data)) > 1e-8:
				estimate_data = estimate_data * (mix_data_max / torch.max(torch.abs(estimate_data)))

			out_path = os.path.join(out_dir_name,  f"{mix_name}.wav")
			my_func.torch_save_wav(Path(out_path), estimate_data)

			# --- 客観評価の実行 ---
			# データの型をtorch型からnumpy型に変換
			estimate_data = estimate_data.squeeze().cpu().numpy()
			target_data_np = target_data.squeeze().cpu().numpy()
			# 長さの調節
			min_len = min(len(target_data_np), len(estimate_data))
			target_data_np = target_data_np[:min_len]
			estimate_data = estimate_data[:min_len]

			pesq_score = pesq_evaluation(target_data_np, estimate_data)
			sisdr_score = sisdr_evaluation(target_data_np, estimate_data).item()
			stoi_score = stoi_evaluation(target_data_np, estimate_data)

			results.append({
				"target": clean_name,
				"estimation": mix_name,
				"PESQ": pesq_score,
				"STOI": stoi_score,
				"SI-SDR": sisdr_score
			})

	# --- 評価結果をCSVに保存 ---
	df = pd.DataFrame(results)

	# 平均と分散を計算
	means = df[["PESQ", "STOI", "SI-SDR"]].mean().rename("mean")
	variances = df[["PESQ", "STOI", "SI-SDR"]].var().rename("variance")

	# 結果を結合
	summary_df = pd.concat([means, variances], axis=1)

	# ファイルに書き込み
	df.to_csv(out_csv_path, index=False)
	with open(out_csv_path, 'a') as f:
		f.write('\n')
		summary_df.to_csv(f)

	print("--------------------------------------------------")
	print(f"RESULT [{task}]")
	print(means)
	print(variances)
	print("Evaluation finished.")
