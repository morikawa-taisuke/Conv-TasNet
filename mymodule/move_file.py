# coding:utf-8

import os

# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tqdm import tqdm
import my_func
import shutil
import wave
import random
import glob
import const

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
    print("source_dir:", source_dir)
    print("destination_dir:", destination_dir)
    my_func.make_dir(destination_dir)
    """ 移動元ディレクトリ内のファイルをリストアップ """
    file_list = os.listdir(source_dir)

    """ 条件に合致するファイルを移動 """
    for file in tqdm(file_list):
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
        with wave.open(source_file_path, "rb") as original_wav:
            """分割後のサンプル数を算出"""
            num_samples = original_wav.getnframes() # 分割前のサンプル数
            samples_per_split = num_samples // num_splits   # 分割後のサンプル数

            for i in range(num_splits):
                """ 分割後のファイル名を生成 """
                split_file_name = f"{os.path.splitext(wav_file)[0]}_split_{i + 1}.wav"
                destination_file_path = os.path.join(destination_dir, split_file_name)
                """ 保存 """
                with wave.open(destination_file_path, "wb") as split_wav:
                    split_wav.setparams(original_wav.getparams())
                    start_sample = i * samples_per_split
                    end_sample = (i + 1) * samples_per_split
                    original_wav.setpos(start_sample)
                    split_wav.writeframes(original_wav.readframes(end_sample - start_sample))

def rename_files_in_directory(directory, search_string, new_string):
    # ディレクトリ内のすべてのファイルを検索
    # directory=os.path.join(directory, "*")
    # print(directory)
    files = glob.glob(os.path.join(directory, "*"))
    print(files)

    for file in tqdm(files):
        # ファイル名に検索文字列が含まれているかをチェック
        # print(file)
        if search_string in os.path.basename(file):
            # 新しいファイル名を生成
            old_name, ext = my_func.get_file_name(file)
            # print(ext)
            old_name = f"{old_name}{ext}"
            print(old_name)
            new_file = old_name.replace(search_string, new_string)
            new_file = os.path.join(directory, new_file)
            # ファイル名を変更
            os.rename(file, new_file)
            tqdm.write(f"Renamed: {file} -> {new_file}")


if __name__ == "__main__":
    # 移動元ディレクトリと移動先ディレクトリを指定
    # clean_mix_list = ["noise_reverbe", "target", "noisy"]
    # """ 条件に合致するファイルの検索文字列を指定 """
    # search_string = f"_Right" #"検索文字列"
    # remove = True
    # """ ディレクトリ名の作成 """
    # source_directory = f"C:\\Users\\kataoka-lab\\Desktop\\sound_data\\mix_data\\subset_DEMAND_hoth_1010dB_2ch\\subset_DEMAND_hoth_1010dB_05sec_2ch_3cm\\"    # "移動元ディレクトリのパス"
    # angle_list = ["00dig", "30dig", "45dig", "60dig", "90dig"]  # "Right", "FrontRight", "Front", "FrontLeft", "Left"
    # wave_type_list = ["clean"] # "noise_only", "noise_reverbe", "reverbe_only"
    # for angle in angle_list:
    #     for wave_type in wave_type_list:
    #         destination_directory = f"{source_directory}/{angle}/test/{wave_type}"  # "移動先ディレクトリのパス"
    #         """ ファイルを移動 """
    #         move_files(os.path.join(source_directory, "test", wave_type), destination_directory, angle, is_remove=remove)


    """ 文字列の置換 """
    # 使用例
    directory = f"{const.SAMPLE_DATA}/noise_data/16k"
    subdir_list = my_func.get_subdir_list(directory)
    print(subdir_list)

    for subdir in subdir_list:
        rename_files_in_directory(os.path.join(directory, subdir), "ch01", new_string=f"{subdir}_01ch")
    
    for subdir in subdir_list:
        move_files(os.path.join(directory, subdir), directory, "_01ch", is_remove=True)
