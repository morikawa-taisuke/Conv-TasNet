# coding:utf-8

import os
from tqdm import tqdm
import const, my_func
import shutil
import wave
import random
import glob


def move_files(source_dir:str, destination_dir:str, search_str:str, is_remove:bool=False)->None:
    """
    ディレクトリから任意の文字列を含むファイル名を別のディレクトリにコピーする

    Parameters
    ----------
    source_dir(str):移動元のディレクトリ名
    destination_dir(str):移動先のディレクトリ名
    search_str(str):検索する文字列
    is_remove(bool):移動元から削除するかどうか (True:削除する, False:削除しない)

    Returns
    -------
    None
    """
    """ 出力先の作成 """
    my_func.make_dir(destination_dir)
    """ 移動元ディレクトリ内のファイルをリストアップ """
    file_list = os.listdir(source_dir)

    """ 条件に合致するファイルを移動 """
    for file in file_list:
        if search_str in file:
            """ パスの作成 """
            source_file_path = os.path.join(source_dir, file)   # 移動元
            destination_file_path = os.path.join(destination_dir, file) # 移動先
            """ ファイルのコピー """
            shutil.copy(source_file_path, destination_file_path)
            if is_remove:   # 移動元から削除する場合
                os.remove(source_file_path) # 削除

def split_wav_file(source_dir:str, destination_dir:str, num_splits:int=1)->None:
    """
    1つ音源ファイルを任意のファイルに分割する(pyroomacousticsで1chで録音した音源を分割するのに使用)

    Parameters
    ----------
    source_dir(str):分割する前のディレクトリ
    destination_dir(str):分割後のディレクトリ
    num_splits(int):分割数

    Returns
    -------
    None
    """
    """ 出力先の作成 """
    my_func.make_dir(destination_dir)
    # 移動元ディレクトリ内のwavファイルをリストアップ
    wav_file_list = [f for f in os.listdir(source_dir) if f.endswith(".wav")]

    for wav_file in wav_file_list:
        source_file_path = os.path.join(source_dir, wav_file)

        """読み込み"""
        with wave.open(source_file_path, 'rb') as original_wav:
            """分割後のサンプル数を算出"""
            num_samples = original_wav.getnframes() # 分割前のサンプル数
            samples_per_split = num_samples // num_splits   # 分割後のサンプル数

            for i in range(num_splits):
                """ 分割後のファイル名を生成 """
                split_file_name = f"{os.path.splitext(wav_file)[0]}_split_{i + 1}.wav"
                destination_file_path = os.path.join(destination_dir, split_file_name)
                """ 保存 """
                with wave.open(destination_file_path, 'wb') as split_wav:
                    split_wav.setparams(original_wav.getparams())
                    start_sample = i * samples_per_split
                    end_sample = (i + 1) * samples_per_split
                    original_wav.setpos(start_sample)
                    split_wav.writeframes(original_wav.readframes(end_sample - start_sample))

def rename_files_in_directory(directory, search_string, new_string):
    # ディレクトリ内のすべてのファイルを検索
    # directory=os.path.join(directory, '*')
    # print(directory)
    files = glob.glob(os.path.join(directory, '*'))
    print(files)

    for file in tqdm(files):
        # ファイル名に検索文字列が含まれているかをチェック
        # print(file)
        if search_string in os.path.basename(file):
            # 新しいファイル名を生成
            old_name, ext = my_func.get_file_name(file)
            print(ext)
            new_name = old_name.replace(search_string, new_string)
            new_file = os.path.join(directory, new_name + ext)
            # new_file = os.path.join(directory, new_name)
            # ファイル名を変更
            os.rename(file, new_file)
            tqdm.write(f'Renamed: {file} -> {new_file}')



"""
if __name__ == "__main__":
    # 移動元ディレクトリと移動先ディレクトリを指定
    source_directory = "../../sound_data/ConvTasNet/separate/result" #"移動元ディレクトリのパス"
    destination_directory = "../../sound_data/ConvTasNet/separate/split" #"移動先ディレクトリのパス"
    # 分割数を指定
    num_splits = 2
    # wavファイルを分割して保存
    split_wav_file(source_directory, destination_directory, num_splits)
"""

if __name__ == "__main__":
    # 移動元ディレクトリと移動先ディレクトリを指定
    # clean_mix_list = ['noise_reverberation','target','noisy']
    """ 条件に合致するファイルの検索文字列を指定 """
    # search_string = f"p257" #"検索文字列"
    # remove = True
    # """ ディレクトリ名の作成 """
    # source_directory = f"C:\\Users\\kataoka-lab\\Desktop\\CONV-TASNET\\sound_data\\sample_data\\speech\\VCTK-DEMAND_28spk_16kHz\\test\\"   # "移動元ディレクトリのパス"
    # destination_directory = f"C:\\Users\\kataoka-lab\\Desktop\\CONV-TASNET\\sound_data\\sample_data\\speech\\sub_set_VCTK-DEMAND_28spk_16kHz\\test\\"  # "移動先ディレクトリのパス"
    """ ファイルを移動 """
    # move_files(source_directory, destination_directory, search_string, is_remove=remove)

    # sub_dir_list = my_func.get_subdir_list(source_directory)
    # # print(sub_dir_list)
    # for sub_dir in sub_dir_list:
    #     All_wav_list = my_func.get_wave_list(f"{source_directory}/{sub_dir}")
    #     wav_path_list = random.sample(All_wav_list, 10)
    #     my_func.make_dir(destination_directory)
    #     for wav_path in wav_path_list:
    #         """ ファイルのコピー """
    #         shutil.copy(wav_path, destination_directory)

    """ 文字列の置換 """
    # 使用例
    directory = 'C:\\Users\\kataoka-lab\\Desktop\\sound_data\\mix_data\\subset_DEMAND_hoth_1010dB_05sec_1ch\\train'
    # subdir_list = my_func.get_subdir_list(directory).remove('noise_only', '')
    subdir_list = my_func.get_subdir_list(directory)
    subdir_list.remove('noise_only')
    print(subdir_list)
    search_string = '05sec'
    new_name = '05sec'
    for subdir in subdir_list:
        rename_files_in_directory(os.path.join(directory, subdir), search_string, new_name)

   