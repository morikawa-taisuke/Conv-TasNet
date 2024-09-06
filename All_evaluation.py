import os.path
import pandas as pd
from openpyxl import load_workbook
import glob
import numpy as np


from tqdm.contrib import tzip
# 自作モジュール
from evaluation.PESQ import pesq_evaluation
from evaluation.STOI import stoi_evaluation
from evaluation.SI_SDR import sisdr_evaluation
from mymodule import my_func

def split_data(input_data:list, channel:int=0)->list:
    """
    引数で受け取ったtensor型の配列の形状を変形させる[1,音声長×チャンネル数]->[チャンネル数,音声長]

    Parameters
    ----------
    input_data(list[int]):分割する前のデータ[1, 音声長*チャンネル数]
    channels(int):チャンネル数(分割する数)

    Returns
    -------
    split_data(list[float]):分割した後のデータ[チャンネル数, 音声長]
    """
    # print('\nsplit_data')    # 確認用
    """ エラー処理 """
    if channel <= 0:   # channelsの数が0の場合or指定していない場合
        raise ValueError("channels must be greater than 0.")

    # print(f'type(in_tensor):{type(in_tensor)}') # 確認用 # torch.Tensor
    # print(f'in_tensor.shape:{in_tensor.shape}') # 確認用 # [1,音声長×チャンネル数]

    """ 配列の要素数を取得 """
    n = input_data.shape[-1]  # 要素数の取得
    # print(f'n:{n}')         # 確認用 # n=音声長×チャンネル数
    if n % channel != 0:   # 配列の要素数をchannelsで割り切れないとき = チャンネル数が間違っているとき
        raise ValueError("Input array size must be divisible by the number of channels.")

    """ 配列の分割 """
    length = n // channel   # 分割後の1列当たりの要素数を求める
    # print(f'length:{length}')   # 確認用 # 音声長
    trun_input = input_data.T   # 転置
    # print_name_type_shape('trun_tensor', trun_tensor)
    split_input = trun_input.reshape(-1, length) # 分割
    # print_name_type_shape('split_tensor', split_input) # 確認用 # [チャンネル数, 音声長]
    # print('split_data\n')    # 確認用
    return split_input

def make_total_csv(condition:dict, original_path='./evaluation/total_score_original.xlsx', out_dir="./RESULT/evaluation/"):
    """　まとめファイルを作する
    動作確認していない
    
    :param condition: 書き込むファイルのpath
    :return: None
    """
    """ ディレクトリの作成 """
    out_name = f'{condition["speech_type"]}_{condition["noise"]}_{condition["snr"]}dB_{condition["reverbe"]}sec'   # ファイル名の作成
    out_path = f'{out_dir}/{out_name}.xlsx'   # 出力パスの作成
    my_func.make_dir(out_path)   # ディレクトリの作成
    print(f'out_path:{out_path}')
    
    if not os.path.isfile(out_path):    # ファイルが存在しない場合
        """ コピー元の読み込み """
        wb = load_workbook(original_path)   # コピー元の読み込み
        sheet = wb['Sheet1']    # シートの指定
        """ 実験条件の書き込み """
        for idx, item in enumerate(condition.values()):
            cell = sheet.cell(row=2, column=1+idx)
            cell.value = item
        """ 保存 """
        wb.save(out_path)   # ワークブックを別名(out_path)に保存
    return out_path
        
def main(target_dir, estimation_dir, out_path, condition, channel=1):
    """　客観評価を行う


    """

    """ 出力ファイルの作成"""
    my_func.make_dir(out_path)
    with open(out_path, 'w') as csv_file:
        csv_file.write(f'target_dir,{target_dir}\nestimation_dir,{estimation_dir}\n')
        csv_file.write(f'{out_path}\ntarget_name,estimation_name,pesq,stoi,sisdr\n')
        
    total_file = make_total_csv(condition=condition)
    

    """ ファイルリストの作成 """
    target_list = my_func.get_file_list(dir_path=target_dir, ext='.wav')
    estimation_list = my_func.get_file_list(dir_path=estimation_dir, ext='.wav')
    print(len(target_list))
    print(len(estimation_list))

    """ 初期化 """
    pesq_sum=0
    stoi_sum=0
    sisdr_sum=0

    for target_file, estimation_file in tzip(target_list, estimation_list):
        """ ファイル名の取得 """
        target_name, _ = my_func.get_file_name(target_file)
        estimation_name, _ = my_func.get_file_name(estimation_file)
        """ 音源の読み込み """
        # print(f'target_file:{target_file}')
        target_data, _ = my_func.load_wav(target_file)
        estimation_data, _ = my_func.load_wav(estimation_file)
        # print(f'target_data.shape:{target_data.shape}')
        # print(f'estimation_data:{estimation_data.shape}')
        if channel != 1:
            target_data = split_data(target_data, channel)[0]   # 0番目のマイクの音を取得 [音声長 * マイク数] → [音声長]

        max_length = max(len(target_data), len(estimation_data))
        target_data = np.pad(target_data, [0, max_length - len(target_data)], 'constant')
        estimation_data = np.pad(estimation_data, [0, max_length - len(estimation_data)], 'constant')
        #
        # if len(target_data)>len(estimation_data):
        #     target_data = target_data[:len(estimation_data)]
        # else:
        #     estimation_data = estimation_data[:len(target_data)]

        """ 客観評価の計算 """
        pesq_score = pesq_evaluation(target_data, estimation_data)
        stoi_score = stoi_evaluation(target_data, estimation_data)
        sisdr_score = sisdr_evaluation(target_data, estimation_data)
        pesq_sum += pesq_score
        stoi_sum += stoi_score
        sisdr_sum += sisdr_score

        """ 出力(ファイルへの書き込み) """
        with open(out_path, 'a') as csv_file:  # ファイルオープン
            text = f'{target_name},{estimation_name},{pesq_score},{stoi_score},{sisdr_score}\n'  # 書き込む内容の作成
            csv_file.write(text)  # 書き込み

    """ 平均の算出(ファイルへの書き込み) """
    pesq_ave=pesq_sum/len(estimation_list)
    stoi_ave=stoi_sum/len(estimation_list)
    sisdr_ave=sisdr_sum/len(estimation_list)
    with open(out_path, 'a') as csv_file:  # ファイルオープン
        text = f'average,,{pesq_ave},{stoi_ave},{sisdr_ave}\n'  # 書き込む内容の作成
        csv_file.write(text)  # 書き込み
    """ まとめファイルへの書き込み """
    work_book = load_workbook(total_file)
    sheet = work_book['Sheet1']
    max_row = my_func.get_max_row(sheet)
    # print(f'max_row:{max_row}')
    sheet.cell(row=max_row, column=1).value = out_path
    sheet.cell(row=max_row, column=2).value = float(pesq_ave)
    sheet.cell(row=max_row, column=3).value = float(stoi_ave)
    sheet.cell(row=max_row, column=4).value = float(sisdr_ave)
    sheet.cell(row=max_row, column=5).value = target_dir
    sheet.cell(row=max_row, column=6).value = estimation_dir
    work_book.save(total_file)

    print('')
    print(f'PESQ : {pesq_ave}')
    print(f'STOI : {stoi_ave}')
    print(f'SI-SDR : {sisdr_ave}')
    # print('pesq end')

if __name__ == '__main__':
    print('start evaluation')
    condition = {"speech_type": 'subset_DEMAND',
                 "noise": 'hoth',
                 "snr": 10,
                 "reverbe": 5}
    # make_total_csv(condition=condition, out_dir='./')

    model_list = ['type_A', 'type_C', 'type_D', 'type_E']
    wave_type_list = ['noise_only', 'reverbe_only', 'noise_reverbe']
    base_name = f'subset_DEMAND_hoth_1010dB_05sec'
    # for loss in loss_list:
    # for model in model_list:
    #     for wave_type in wave_type_list:
    #         main(target_dir=f'C:\\Users\\kataoka-lab\\Desktop\\RESULT\\sample data\\rec_4ch\\clean',
    #              estimation_dir=f"C:\\Users\\kataoka-lab\\Desktop\\RESULT\\Multi_channel_ConvTasNet\\{model}\\{wave_type}",
    #              out_path=f'C:\\Users\\kataoka-lab\\Desktop\\RESULT\\Multi_channel_ConvTasNet\\csv\\{model}_{wave_type}.csv',
    #              condition=condition,
    #              num_mic=4)
    # for wave_type in wave_type_list:
    main(target_dir=f'C:\\Users\\kataoka-lab\\Desktop\\RESULT\\sample data\\rec_4ch\\clean',
         estimation_dir=f"C:\\Users\\kataoka-lab\\Desktop\\RESULT\\sample data\\rec_4ch\\clean",
         out_path=f'C:\\Users\\kataoka-lab\\Desktop\\RESULT\\Multi_channel_ConvTasNet\\csv\\{base_name}_clean.csv',
         condition=condition,
         channel=4)
    # for wave_type in wave_type_list:
    #     main(target_dir=f'C:\\Users\\kataoka-lab\\Desktop\\sound_data\\mix_data\\sebset_DEMAND_hoth_1010dB_05sec_1ch\\test\\clean',
    #          estimation_dir=f'C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\RESULT\\output_wav\\subset_DEMAND_hoth_1010dB_05sec_1ch\\{wave_type}',
    #          out_path=f'C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\RESULT\\evaluation\\subset_DEMAND_hoth_1010dB_05sec_1ch\\{wave_type}_2.csv',
    #          condition=condition)
