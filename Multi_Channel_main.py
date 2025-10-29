""" 1ch雑音付加，データセット作成，学習，評価を行う　"""
from src import Multi_Channel_ConvTasNet_test, Multi_Channel_ConvTasNet_train
from src.utils import const
from scripts import All_evaluation

if __name__ == '__main__':
    print('start')
    """ 学習の条件 """
    speech_type = 'sub_set_VCTK-DEMAND_28spk_16kHz'  # 話者のタイプ
    noise_type = 'hoth'  # 雑音の種類
    snr_list = [10]  # SNRの指定(list型)
    reverbe = 0  # 残響時間
    ch = 1  # マイク数
    train_count = 1  # 学習回数

    """ 学習の条件を辞書型でまとめる """
    condition = {"speech_type": speech_type,  # 話者のタイプ JA:日本語 CMU:英語
                 "noise_type": noise_type,  # 雑音の種類 sound_file/sample_data/noise内のファイル名を指定('hoth', 'white')
                 "snr": f'{snr_list[0]:02}{snr_list[-1]:02}',  # SNRの指定(list型)
                 "reverbe": f'{reverbe:02}',  # 残響時間
                 "ch": f'{ch:02}',  # マイク数(チャンネル数)
                 }  # {speech_type:話者タイプ, noise_type:雑音の種類, snr_list:SNR, reverbe:残響時間, ch:マイク数}

    """ パス関係 """
    out_dir_name = f'{condition["speech_type"]}_{condition["noise_type"]}_{condition["snr"]}db_{condition["reverbe"]}sec_{condition["ch"]}ch'  # 出力するディレクトリ名
    mix_dir = f'{const.MIX_DATA_DIR}/{out_dir_name}/'   # 混合信号のディレクトリ
    print(f'out_name:{out_dir_name}')   # 出力先の確認

    """ データセット作成 """
    # print('\n---make_dataset---')
    mix_path = f'{mix_dir}/train/mix/'    # 入力データ
    target_path = f'{mix_dir}/train/target/'  # 教師データ
    dataset_dir = f'{const.DATASET_DIR}/{out_dir_name}/'  # データセットの出力先
    # print(mix_path)
    # print(target_path)
    # make_dataset.enhance_save_stft(mix_dir=mix_path,  # 入力データ(目的信号+雑音)
    #                                target_dir=target_path,  # 教師データ
    #                                out_dir=dataset_dir)  # 出力先
    """ 学習 """
    print('\n---train---')
    estimation_dir = f'{const.OUTPUT_WAV_DIR}/{out_dir_name}/'  # モデル適用後の出力先
    model_type_list = ['A', 'C', 'D', 'E']
    for model_type in model_type_list:
        Multi_Channel_ConvTasNet_train.main(dataset=dataset_dir, out_name=out_dir_name,
                                            train_count=condition["train_count"],
                                            model_type=model_type)  # 学習回数
    """ モデルの適用(テスト) """
    print('\n---test---')
    Multi_Channel_ConvTasNet_test.test(mix_dir=f'{mix_dir}/test/mix',  # テスト用データ
                                       out_dir=estimation_dir,  # 出力先
                                       model_name=f'{const.PTH_DIR}/{out_dir_name}/{out_dir_name}_{train_count}.pth')  # 使用するモデルのパス
    """ 評価 """
    print('\n---evaluation---')
    All_evaluation.main(target_dir=f'{mix_dir}/test/target',  # 教師データ
                        estimation_dir=estimation_dir,  # 評価するデータ
                        out_path=f'{const.EVALUATION_DIR}/{out_dir_name}/{out_dir_name}.csv',
                        condition=condition)  # 出力先
