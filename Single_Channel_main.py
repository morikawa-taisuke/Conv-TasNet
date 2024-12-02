""" 1ch雑音付加，データセット作成，学習，評価を行う　"""
import make_mixdown
import make_dataset
import ConvTasNet_train
import ConvTasNet_test
from mymodule import my_func, const
import All_evaluation

if __name__ == "__main__":
    print("start")
    """ 学習の条件 """
    speech_type = "sub_set_VCTK-DEMAND_28spk_16kHz"  # 話者のタイプ
    noise_type = "hoth" # 雑音の種類
    snr_list = [10] # SNRの指定(list型)
    reverbe = 0  # 残響時間
    ch = 1  # マイク数
    train_count = 1   # 学習回数
    
    """ 学習の条件を辞書型でまとめる """
    condition = {"speech_type":speech_type,  # 話者のタイプ JA:日本語 CMU:英語
                 "noise_type":noise_type, # 雑音の種類 sound_file/sample_data/noise内のファイル名を指定("hoth", "white")
                 "snr":f"{snr_list[0]:02}{snr_list[-1]:02}", # SNRの指定(list型)
                 "reverbe":f"{reverbe:02}", # 残響時間
                 "ch":f"{ch:02}",  # マイク数(チャンネル数)
                 }  # {speech_type:話者タイプ, noise_type:雑音の種類, snr_list:SNR, reverbe:残響時間, ch:マイク数}
    
    """ パス関係 """
    signal_dir = f"{const.SPEECH_DIR}/{condition['speech_type']}"  # 目的信号のパス
    signale_subdir_list = my_func.get_subdir_list(signal_dir)  # 子ディレクトリ(test, train)のリストを取得
    noise_dir = f"{const.NOISE_DIR}/{condition['noise_type']}.wav" # 雑音のパス
    out_dir_name = f"{condition['speech_type']}_{condition['noise_type']}_{condition['snr']}db_{condition['reverbe']}sec_{condition['ch']}ch"  # 出力するディレクトリ名
    # out_dir_name = f"JA_hoth_10dB_05sec_4ch"
    print(f"out_name:{out_dir_name}")
    dataset_dir = f"{const.DATASET_DIR}/{out_dir_name}/"  # データセットの出力先
    estimation_dir = f"{const.OUTPUT_WAV_DIR}/{out_dir_name}/"    # モデル適用後の出力先
    
    """ 雑音付加 """
    print("\n---mixdown---")
    mix_dir = f"{const.MIX_DATA_DIR}/{out_dir_name}/"  # 混合信号の出力先
    # for signale_subdir in signale_subdir_list:  # train, test
    #     make_mixdown.add_noise_all(target_dir=f"{signal_dir}/{signale_subdir}", noise_file=noise_dir,
    #                                out_dir=f"{mix_dir}/{signale_subdir}/", snr_list=snr_list)  # SNR
    """ データセット作成 """
    print("\n---make_dataset---")
    mix_path = f"{const.MIX_DATA_DIR}/{out_dir_name}/train/mix/"
    target_path = f"{const.MIX_DATA_DIR}/{out_dir_name}/train/target/"
    print(mix_path)
    print(target_path)
    make_dataset.enhance_save_stft(mix_dir=mix_path,    # 入力データ(目的信号+雑音)
                                   target_dir=target_path,  # 教師データ
                                   out_dir=dataset_dir) # 出力先
    """ 学習 """
    print("\n---train---")
    ConvTasNet_train.main(dataset_path=dataset_dir, out_path=out_dir_name, train_count=train_count)  # 学習回数
    """ モデルの適用(テスト) """
    print("\n---test---")
    ConvTasNet_test.test(mix_path=f"{mix_dir}/test/mix",    # テスト用データ
                         estimation_path=estimation_dir,    # 出力先
                         model_path=f"{const.PTH_DIR}/{out_dir_name}/{out_dir_name}_{train_count}.pth")   # 使用するモデルのパス
    """ 評価 """
    print("\n---evaluation---")
    All_evaluation.main(target_dir=f"{mix_dir}/test/target",    # 教師データ
                        estimation_dir=estimation_dir,  # 評価するデータ
                        out_path=f"{const.EVALUATION_DIR}/{out_dir_name}/{out_dir_name}.csv",
                        condition=condition)    # 出力先
    
    # wave_type_list = ["noise_only_delay", "noise_reverbe_delay", "reverbe_only_delay"]
    # for wave_type in wave_type_list:
    #     """ 学習 """
    #     print("train")
    #     pth_dir = f"./ConvTasNet00/pth/{out_dir_name}/"
    #     log_dir = f"./ConvTasNet00/logs/{out_dir_name}/"
    #     dataset_path = f"{dataset_dir}/{wave_type}"
    #     # out_dir_name = f"{wave_type}"
    #     ConvTasNet_train.main(dataset_path=dataset_path,  # データセットのディレクトリ
    #                          out_name=wave_type,  # 出力先
    #                          train_count=train_count,  # 学習回数
    #                          pth_dir=pth_dir,  # 学習済みモデルの出力先
    #                          log_dir=log_dir)  # logの出力先
    #     """ モデルの適用(テスト) """
    #     print("test")
    #     estimation_dir = f"{estimation_dir}/{wave_type}"
    #     ConvTasNet_test.test(mix_path=f"{mix_dir}/test/{wave_type}",  # テスト用データ
    #                          estimation_path=estimation_dir,  # 出力先
    #                          model_path=f"{pth_dir}/{wave_type}/{wave_type}_{train_count}.pth")  # 使用するモデルのパス
    #     """ 評価 """
    #     print("evaluation")
    #     All_evaluation.main(target_dir=f"{mix_dir}/test/clean_delay",  # 教師データ
    #                         estimation_dir=estimation_dir,  # 評価するデータ
    #                         out_path=f"../sound_data/result/{out_dir_name}/{wave_type}.csv")  # 出力先
    