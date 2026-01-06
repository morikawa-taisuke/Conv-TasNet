import argparse
import os
import numpy as np

def pad_raw_audio(input_file, output_file, padding_samples, sample_width=2):
    """
    raw音声ファイルの先頭に0サンプルを追加します。

    Args:
        input_file (str): 入力raw音声ファイルのパス
        output_file (str): 出力raw音声ファイルのパス
        padding_samples (int): 先頭に追加する0サンプルの数
        sample_width (int): 1サンプルのバイト数 (例: 16-bitなら2)
    """
    try:
        # 0埋めするデータを作成
        zero_padding = b'\x00' * (padding_samples * sample_width)

        with open(output_file, 'wb') as f_out:
            # 最初に0データを書き込む
            f_out.write(zero_padding)

            # 元のファイルの内容を追記する
            with open(input_file, 'rb') as f_in:
                # 大きなファイルを効率的に扱うため、チャンクで読み書きする
                while True:
                    chunk = f_in.read(4096)  # 4KBずつ読み込む
                    if not chunk:
                        break
                    f_out.write(chunk)
        
        print(f"ファイル '{input_file}' の先頭に {padding_samples} サンプルの0を追加し、'{output_file}' に保存しました。")

    except FileNotFoundError:
        print(f"エラー: 入力ファイル '{input_file}' が見つかりません。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def mix_raw_audio_snr(signal_file, noise_file, output_file, snr_db, sample_width=2):
    """
    2つのraw音声ファイルを指定されたSNRでミキシングします。

    Args:
        signal_file (str): 信号となるraw音声ファイルのパス
        noise_file (str): ノイズとなるraw音声ファイルのパス
        output_file (str): 出力ファイルのパス
        snr_db (float): 混合後のSNR (dB)
        sample_width (int): 1サンプルのバイト数
    """
    try:
        # データ型を決定
        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            print(f"エラー: サポートされていないサンプル幅です: {sample_width} バイト")
            return

        # ファイルを読み込み、numpy配列に変換
        signal_bytes = np.fromfile(signal_file, dtype=dtype)
        noise_bytes = np.fromfile(noise_file, dtype=dtype)

        # ファイル長が短い方に合わせる
        min_len = min(len(signal_bytes), len(noise_bytes))
        signal_bytes = signal_bytes[:min_len]
        noise_bytes = noise_bytes[:min_len]

        # パワーを計算
        signal_power = np.mean(signal_bytes.astype(np.float64)**2)
        noise_power = np.mean(noise_bytes.astype(np.float64)**2)

        if signal_power == 0 or noise_power == 0:
            print("エラー: 信号またはノイズのパワーが0です。無音のファイルはミキシングできません。")
            return

        # ノイズのスケール係数を計算
        snr_linear = 10**(snr_db / 10)
        scale_factor = np.sqrt(signal_power / (noise_power * snr_linear))

        # ノイズをスケーリングして信号と混合
        scaled_noise = (noise_bytes.astype(np.float64) * scale_factor).astype(np.float64)
        mixed_signal = signal_bytes.astype(np.float64) + scaled_noise

        # クリッピング処理
        max_val = np.iinfo(dtype).max
        min_val = np.iinfo(dtype).min
        mixed_signal = np.clip(mixed_signal, min_val, max_val)

        # ファイルに書き出し
        mixed_signal.astype(dtype).tofile(output_file)

        print(f"'{signal_file}'と'{noise_file}'をSNR {snr_db}dBで混合し、'{output_file}'に保存しました。")

    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません: {e.filename}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="raw音声ファイルを処理するツール。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # subparsers = parser.add_subparsers(dest="command" "-c", required=True, help="実行するコマンド")

    # 'pad' コマンドのパーサー
    # parser_pad = subparsers.add_parser("pad", help="raw音声ファイルの先頭に0サンプルを追加します。")
    # parser_pad.add_argument("input_file", type=str, help="入力raw音声ファイルのパス")
    # parser_pad.add_argument("output_file", type=str, help="出力ファイルのパス")
    # parser_pad.add_argument("padding_samples", type=int, help="先頭に追加する0サンプルの数")
    # parser_pad.add_argument(
    #     "--sample_width", type=int, default=2,
    #     help="1サンプルのバイト数 (例: 16-bitなら2)。デフォルト: 2"
    # )
	#
    # # # 'mix' コマンドのパーサー
    # parser_mix = parser.add_parser("mix", help="2つのraw音声ファイルを指定のSNRで足し合わせます。")
    # parser_mix.add_argument("signal_file", type=str, help="信号(speech)となるraw音声ファイルのパス")
    # parser_mix.add_argument("noise_file", type=str, help="ノイズとなるraw音声ファイルのパス")
    # parser_mix.add_argument("output_file", type=str, help="出力ファイルのパス")
    # parser_mix.add_argument("snr", type=float, help="混合後のSNR (dB)")
    # parser_mix.add_argument(
    #     "--sample_width", type=int, default=2,
    #     help="1サンプルのバイト数 (例: 16-bitなら2)。デフォルト: 2"
    # )
	#
    # args = parser.parse_args()

    # if args.command == "pad":
    #     pad_raw_audio(args.input_file, args.output_file, args.padding_samples, args.sample_width)
    # elif args.command == "mix":
    #     mix_raw_audio_snr(args.signal_file, args.noise_file, args.output_file, args.snr, args.sample_width)
    mix_raw_audio_snr("/Users/a/Documents/C/10/sample/F03_01.raw",
                      "/Users/a/Documents/C/p341+N(raw)/hoth_100s.raw",
                      "/Users/a/Documents/C/10/sample/F03_noise.raw",
                      3)
