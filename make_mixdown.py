# coding:utf-8
""" 目的信号と雑音を足し合わせる """
import numpy as np
import random
from numpy import ndarray
from tqdm import tqdm
import os
from itertools import combinations


from mymodule import my_func

def calc_snr(signal_wave:list, noise_wave:list)->float:
    """
    引数のデータのsnrを求める


    Parameters
    ----------
    signal_power(list):目的信号のパワー
    noise_power(list):雑音のパワー

    Returns
    -------
    snr(float):snr
    """
    # snr = 10 * math.log10(calc_wave_power(signal_wave) / calc_wave_power(noise_wave))
    snr = 10 * np.log10(calc_wave_power(signal_wave) / calc_wave_power(noise_wave))
    return snr

def calc_wave_power(wave_data:list)-> ndarray:
    """
    引数のパワーを求める


    Parameters
    ----------
    wave_data(list):音源データ

    Returns
    -------
    power(float):パワー
    """
    power = np.mean(pow(np.abs(wave_data), 2))  # power = sigma(wave_data^2)
    # print(f'type(power):{type(power)}') # 確認用
    # print(f'power:{power.shape}')       # 確認用
    return power

def calculate_snr(signal, noise):
    signal_power = np.mean(pow(np.abs(signal), 2))  # 信号のパワーを計算
    noise_power = np.mean(pow(np.abs(noise), 2))    # 雑音のパワーを計算
    snr_dB = 10 * np.log10(signal_power / noise_power)  # SNRを計算 (dB単位)

    return snr_dB

def calc_ajast_noise(signal_data:list, noise_data:list, snr:int)->list:
    """
    任意のsnrになるように雑音を調整する

    Parameters
    ----------
    signal_data(list):目的信号
    noise_data(list):雑音信号
    snr(int):SNR

    Returns
    -------
    scale_noise_data(list):調整後のsnr
    """
    """ 任意のsnrの時のパワーの計算 """
    signal_power = calc_wave_power(signal_data) # 目的信号
    noise_power = signal_power/pow(10, snr/10)  # 雑音

    alpha = np.sqrt(noise_power / np.mean(pow(np.abs(noise_data), 2)))  # 雑音の大きさを調整するための係数
    scale_noise_data = alpha * noise_data  # α*雑音信号
    """ 調整後のsnrの判断 """
    # after_snr = calc_snr(signal_power, calc_wave_power(scale_noise_data))
    after_snr = calculate_snr(signal=signal_data, noise=scale_noise_data)   # 調整後のsnr
    after_snr = round(after_snr)  # 調整後のsnrを算出
    if after_snr != snr:    # 違った場合は出力
        print(f'{snr}->{after_snr}')
    return scale_noise_data


def add_noise(signal_path:str, noise_path:str, out_dir:str, snr:float)->None:
    """
    任意のsnrでsignalにnoiseを付加し，保存する

    Parameters
    ----------
    signal_path(str):目的信号のパス
    noise_path(str):雑音信号のパス
    out_path(str):出力先のパス
    snr(float):snr

    Returns
    -------
    None
    """
    """ ファイルの読み込み """
    signal_data, prm = my_func.load_wav(signal_path)
    noise_data, _ = my_func.load_wav(noise_path)
    """ noise_dataをsignal_dataと同じ長さで切り出す """
    random.seed(0)
    start = random.randint(0, len(noise_data) - len(signal_data))  # スタート位置をランダムに決定
    split_noise = noise_data[start: start + len(signal_data)]  # noise_dataのスライス
    # print(f'signal非負値? {np.any(signal_data > 0)}')
    # print(f'nosie 非負値? {np.any(split_noise > 0)}')
    """ 指定したSNRになるようにnoise_dataの大きさを調整 """
    # scale_noise_data = calc_scale_noise(signal_data=signal_data, noise_data=split_noise, snr=snr)
    scale_noise_data = calc_ajast_noise(signal_data=signal_data, noise_data=split_noise, snr=snr)
    """ mix_dataの作成 """
    mix_data = signal_data + scale_noise_data   # signal_dataとscale_noise_dataを足し合わせる
    # print(f'signal_data:{signal_data}')
    # print(f'noise_data:{noise_data}')
    # print(f'mix_data:{mix_data}')
    """ 保存 """
    signal_name, _ = my_func.get_file_name(signal_path)  # ファイル名の取得
    noise_name, _ = my_func.get_file_name(noise_path)  # ファイル名の取得
    out_name = f'{signal_name}_{noise_name}_{snr}dB.wav'    # 出力ファイル名
    out_path = f'{out_dir}/mix/{out_name}'  # 出力先
    target_path = f'{out_dir}/target/{out_name}'
    # print(f'out_path:{out_path}')
    # print(f'target_path:{target_path}')
    my_func.save_wav(out_path=out_path, wav_data=mix_data, prm=prm)
    my_func.save_wav(out_path=target_path, wav_data=signal_data, prm=prm)

def add_noise_all(target_dir:str, noise_file:str, out_dir:str, snr_list:list)->None:
    """
    指定したディレクトリ内のファイルすべてに対してadd_noiseを行う

    Parameters
    ----------
    signal_dir(str):目的信号のディレクトリ
    noise_dir(str):雑音信号のディレクトリ
    out_dir(str):出力先のディレクトリ
    snr(float):SNR

    Returns
    -------
    None
    """
    """wavファイルのリストを作成"""
    signal_list = my_func.get_file_list(target_dir, ext='.wav') # 目的信号
    noise_list = my_func.get_file_list(noise_file, ext='.wav')   # 雑音信号
    print(f'len(signal_list):{len(signal_list)}')
    print(f'len(noise_list):{len(noise_list)}')
    """ 出力先の作成 """
    my_func.make_dir(out_dir)
    for signal_path in tqdm(signal_list):   # tqdm()
        for noise_path in noise_list:
            for snr in snr_list:
                add_noise(signal_path=signal_path, noise_path=noise_path, out_dir=out_dir, snr=snr)

def add_speech(speaker_A_path:str, speaker_B_path:str, out_dir:str, out_txt:str)->None:
    """
    任意のsnrでsignalにnoiseを付加し，保存する

    Parameters
    ----------
    speaker_A(str):目的信号のパス
    speaker_B(str):雑音信号のパス
    out_path(str):出力先のパス
    out_txt(str):教師データと入力データのファイル名を記述したファイル

    Returns
    -------
    None
    """
    """ ファイルの読み込み """
    signal_data, prm = my_func.load_wav(speaker_A_path)
    noise_data, _ = my_func.load_wav(speaker_B_path)
    """ 音声長を合わせる　(長いほうに合わせる) """
    max_length = max(len(signal_data), len(noise_data))
    signal_data = np.pad(signal_data, (0, max_length-len(signal_data)))
    noise_data = np.pad(noise_data, (0, max_length-len(noise_data)))

    """ SNRが0になるようにnoise_dataの大きさを調整 """
    # scale_noise_data = calc_ajast_noise(signal_data=signal_data, noise_data=noise_data, snr=0)
    """ mix_dataの作成 """
    mix_data = signal_data + noise_data   # signal_dataとscale_noise_dataを足し合わせる
    # print(f'signal_data:{signal_data}')
    # print(f'noise_data:{noise_data}')
    # print(f'mix_data:{mix_data}')
    """ 保存 """
    speaker_A, _ = my_func.get_file_name(speaker_A_path)  # ファイル名の取得
    speaker_B, _ = my_func.get_file_name(speaker_B_path)  # ファイル名の取得
    out_name = f'{speaker_A}_{speaker_B}.wav'  # 出力ファイル名
    # ファイルの書き込み
    with open(out_txt, 'a') as csv_file:  # ファイルオープン
        text = f'{out_dir}/mix/{out_name},{out_dir}/speaker1/{out_name},{out_dir}/speaker2/{out_name}\n'  # 書き込む内容の作成
        csv_file.write(text)  # 書き込み0
    my_func.save_wav(out_path=os.path.join(out_dir, 'mix', out_name), wav_data=mix_data, prm=prm)
    my_func.save_wav(out_path=os.path.join(out_dir, 'speaker1', out_name), wav_data=signal_data, prm=prm)
    my_func.save_wav(out_path=os.path.join(out_dir, 'speaker2', out_name), wav_data=noise_data, prm=prm)
    # my_func.save_wav(out_path=target_path, wav_data=scale_noise_data, prm=prm)

def add_speech_all(speaker_dir:str, out_dir:str, out_txt_path:str)->None:
    """
    指定したディレクトリ内のファイルすべてに対してadd_noiseを行う

    Parameters
    ----------
    speaker_dir(str):話者信号のディレクトリ
    out_dir(str):出力先のディレクトリ
    out_txt_path(str):音声の組み合わせを記述したファイル(csvファイル)

    Returns
    -------
    None
    """
    speaker_list = my_func.get_subdir_list(dir_path=speaker_dir)  # 話者ディレクトリ内のサブディレクトリのリストを取得
    print(f'{len(speaker_list)}')
    # print(f'{speaker_list}')
    """ 出力先の作成 """
    my_func.make_dir(out_dir)
    with open(out_txt_path, 'w') as csv_file:  # ファイルオープン
        text = f'out_path,speaker_A_path,speaker_B_path\n'  # 書き込む内容の作成
        csv_file.write(text)  # 書き込み
    for speaker_A_dir, speaker_B_dir in tqdm(combinations(speaker_list, 2), total=len(list(combinations(speaker_list, 2)))):    # len(speaker_list) C 2 →組み合わせ
        # print(f'{speaker_A_dir} : {speaker_B_dir}')
        """wavファイルのリストを作成"""
        speaker_A_list = my_func.get_file_list(os.path.join(speaker_dir, speaker_A_dir), ext='.wav')    # 話者A
        speaker_B_list = my_func.get_file_list(os.path.join(speaker_dir, speaker_B_dir), ext='.wav')   # 話者B
        # print(f'len(speaker_A_list):{len(speaker_A_list)}')
        # print(f'len(speaker_B_list):{len(speaker_B_list)}')
        """ 各ファイルに対してadd_speechを行う """
        for speaker_A_path in speaker_A_list:   # tqdm()
            for speaker_B_path in speaker_B_list:
                add_speech(speaker_A_path=speaker_A_path, speaker_B_path=speaker_B_path, out_dir=out_dir, out_txt=out_txt_path)


def calc_power(wav_data):
    """ powerの計算 """
    # return 20*np.log10(np.abs(wav_data))
    return 10 * np.log10(np.mean(wav_data ** 2))

def decay_signal(signal_data, delay_sample, c=340, sr=16000):
    """
    signal_dataを減衰させる

    Parameters
    ----------
    signal_data: もとのデータ
    delay_sample: 遅延量
    c: 音速
    sr: サンプリングレート

    Returns
    -------

    """
    # もとの音のpowerを計算
    origin_power = calc_power(signal_data)

    # 減衰後のpowerを計算
    decay_power = origin_power - 11 - 20 * np.log10(c * delay_sample / sr)
    # print("decay_power: ", decay_power)

    # 減衰率を計算
    decay = 10 ** (decay_power / 10) / np.mean(signal_data ** 2)

    # 出力音声 = もとの音 * 減衰率
    return signal_data * decay

def decay_signal_all(signal_dir:str, out_dir:str, ch=4):
    win = 32
    sub_dir_list = my_func.get_subdir_list(dir_path=signal_dir)  # 話者ディレクトリ内のサブディレクトリのリストを取得
    for sub_dir in sub_dir_list:
        signal_list = my_func.get_file_list(os.path.join(signal_dir, sub_dir), ext='.wav')  # 話者A
        print(len(signal_list))

        for signal_path in tqdm(signal_list):
            out_wave = []
            signal_data, prm = my_func.load_wav(signal_path)
            for channel in range(0, ch):
                if channel==0:
                    out_wave.append(signal_data)
                else:
                    # print("win*channel: ", win, channel, win*channel)
                    out = np.zeros(signal_data.shape[-1])
                    # print("out: ", out[win*channel:].shape)
                    out[win*channel:] = decay_signal(signal_data, delay_sample=win*channel)[:len(signal_data)-(win*channel)]
                    # a = decay_signal(signal_data, delay_sample=win*channel)
                    # print("a: ", a[:len(single(signal_data))-(win*channel)])

                    out_wave.append(out)
            out_name, _ = my_func.get_file_name(signal_path)
            out_name = f"{out_name}.wav"
            my_func.save_wav(out_path=os.path.join(out_dir, out_name), wav_data=np.array(out_wave), prm=prm)


if __name__ == '__main__':
    print('signal_to_rate')
    """ 各自の環境・実験の条件によって書き換える """
    # target_dir = f'C:\\Users\\kataoka-lab\\Desktop\\sound_data\\speech\\subset_DEMAND_28spk_16kHz' # 目的信号のディレクトリ
    # noise_file = f'C:\\Users\\kataoka-lab\Desktop\\sound_data\\noise\\hoth.wav'    # 雑音のパス
    speaker_dir = 'C:\\Users\\kataoka-lab\\Desktop\\sound_data\\mix_data\\subset_DEMAND_hoth_1010dB_1ch\\subset_DEMAND_hoth_1010dB_05sec_1ch\\test'
    out_dir = 'C:\\Users\\kataoka-lab\\Desktop\\sound_data\\mix_data\\1ch_to_4ch_decay_all\\'   # 出力先
    snr_list = [10]  # SNR リスト形式で指定することで実験条件を変更可能

    # print(f'target:{target_dir}')
    # print(f'noise:{noise_file}')
    print(f'speaker_dir:{speaker_dir}')
    print(f'output:{out_dir}')

    decay_signal_all(signal_dir=speaker_dir,
                     out_dir=out_dir)

    # signale_subdir_list = my_func.get_subdir_list(speaker_dir)   # 子ディレクトリのリストを取得
    # print(signale_subdir_list)  # ディレクトリリストの出力
    # for signale_subdir in signale_subdir_list:
    #     # add_noise_all(target_dir=f'{target_dir}/{signale_subdir}',
    #     #               noise_file=noise_file,
    #     #               out_dir=f'{out_dir}/{signale_subdir}',
    #     #               snr_list=snr_list)
    #     add_speech_all(speaker_dir=os.path.join(speaker_dir, signale_subdir),
    #                    out_dir=os.path.join(out_dir, signale_subdir),
    #                    out_txt_path=os.path.join(out_dir, signale_subdir, 'list.csv'))


