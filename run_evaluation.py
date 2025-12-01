# coding:utf-8
"""
このスクリプトは、評価の実行を開始するためのエントリーポイントです。

設定ファイル (config.yml) を読み込み、その内容に基づいて
学習済みモデルやテストデータローダーを生成し、
`src/evaluate.py` の `evaluate_and_save` 関数に渡して評価を実行します。
"""

import argparse
import glob
import os
import yaml
import torch
from torch.utils.data import DataLoader

# 自作モジュール
from src.evaluate import evaluate_and_save
from src.datasetClass import TasNet_dataset, TasNet_dataset_csv_separate
from src.CsvDataset import CsvInferenceDataset
from src.models import ConvTasNet_models, MultiChannel_ConvTasNet_models
from src.utils import const


def main():
	""" メイン関数 """
	parser = argparse.ArgumentParser(description="Conv-TasNet Evaluation Launcher")
	parser.add_argument("--config", "-c", required=True, help="Path to the configuration file (config.yml).")
	args = parser.parse_args()

	# --- 1. 設定ファイルの読み込み ---
	with open(args.config, encoding="utf-8") as f:
		config = yaml.safe_load(f)

	# --- 2. デバイスの決定 ---
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# --- 3. オブジェクトの生成 ---
	model_config = config["model"]
	common_config = config["common"]
	task_list = common_config["task_list"]

	test_dataset_path = os.path.join(const.MIX_DATA_DIR, common_config["dataset"], "test.csv")  # 学習用データセットのcsvのパス

	for task in task_list:
		# --- データローダの生成 ---
		if model_config["input_ch"] == 1:	# 1ch
			test_dataset = CsvInferenceDataset(csv_path=test_dataset_path, input_column_header=task)
		else:	# multi-ch
			test_dataset = TasNet_dataset(test_dataset_path)
		test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, pin_memory=True)

		# --- モデルの生成 ---
		# モデルの絶対パスを組み立てる
		model_path = glob.glob(os.path.join(const.CHECKPOINT_DIR, common_config["out_dir_name"], f"*_{task}_best.pth"))[0]
		print(f"model_path: {model_path}")

		# --- モデルの生成と学習済み重みの読み込み ---
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

		# 学習済みモデルの重みを読み込む
		model.load_state_dict(torch.load(model_path, map_location=device))

		# --- 4. 評価の実行 ---
		evaluate_and_save(model=model, test_loader=test_loader, device=device, config=config, out_dir_name=common_config["out_dir_name"])


if __name__ == "__main__":
	main()
