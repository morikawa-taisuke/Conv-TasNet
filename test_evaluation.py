import All_evaluation as eval
import Multi_Channel_ConvTasNet_test as test
from mymodule import const


angle_list = ['Right', 'FrontRight', 'Front', 'FrontLeft', 'Left']  # 'Right', 'FrontRight', 'Front', 'FrontLeft', 'Left'

condition = {"speech_type": 'subset_DEMAND',
             "noise": 'hoth',
             "snr": 10,
             "reverbe": 5}
ch = 4
for angle in angle_list:
    dir_name = f'sebset_DEMAND_hoth_1010dB_05sec_{ch}ch_3cm_{angle}_Dtype'
    out_dir = f'{const.OUTPUT_WAV_DIR}/{dir_name}/'
    wave_path = f"{const.MIX_DATA_DIR}\\sebset_DEMAND_hoth_1010dB_05sec_{ch}ch_3cm\\{angle}\\test\\"
    wave_type_list = ['reverbe_only']    # 'noise_only', 'noise_reverbe', 'reverbe_only'
    for wave_type in wave_type_list:
        print('test')
        test.test(mix_dir=f'{wave_path}/{wave_type}',
                  out_dir=f'{out_dir}/{wave_type}',
                  model_name=f'{const.PTH_DIR}/{dir_name}/{dir_name}_100.pth',
                  channels=ch,
                  model_type='D')

        print('evaluation')
        eval.main(target_dir=f"{const.MIX_DATA_DIR}\\sebset_DEMAND_hoth_1010dB_05sec_4ch_3cm\\{angle}\\test\\clean",
                  estimation_dir=f'{out_dir}/{wave_type}',
                  out_path=f'{const.EVALUATION_DIR}/sebset_DEMAND_hoth_1010dB_05sec_4ch_3cm/{angle}/{wave_type}.csv',
                  condition=condition,
                  channel=ch)

