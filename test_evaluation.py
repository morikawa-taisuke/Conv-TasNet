import my_func
import All_evaluation as eval
import Multi_Channel_ConvTasNet_test as test


angle_list = ['Right', 'FrontRight', 'Front', 'FrontLeft', 'Left']  # 'Right', 'FrontRight', 'Front', 'FrontLeft', 'Left'

condition = {"speech_type": 'subset_DEMAND',
             "noise": 'hoth',
             "snr": 10,
             "reverbe": 5}
ch = 2
for angle in angle_list:
    dir_name = f'sebset_DEMAND_hoth_1010dB_05sec_{ch}ch_3cm_{angle}_Dtype'
    out_dir = f'./RESULT/output_wav/{dir_name}/'
    wave_path = f"C:\\Users\\kataoka-lab\\Desktop\\sound_data\\mix_data\\sebset_DEMAND_hoth_1010dB_05sec_{ch}ch_3cm\\{angle}\\test\\"
    wave_type_list = ['noise_reverbe']    # 'noise_only', 'noise_reverbe', 'reverbe_only'
    for wave_type in wave_type_list:
        print('test')
        test.test(mix_dir=f'{wave_path}/{wave_type}',
                  out_dir=f'{out_dir}/{wave_type}',
                  model_name=f'./RESULT/pth/{dir_name}/{dir_name}_100.pth',
                  channels=ch,
                  model_type='D')

        print('evaluation')
        eval.main(target_dir=f"C:\\Users\\kataoka-lab\\Desktop\\sound_data\\mix_data\\sebset_DEMAND_hoth_1010dB_05sec_4ch_3cm\\{angle}\\test\\clean",
                  estimation_dir=f'{out_dir}/{wave_type}',
                  out_path=f'./RESULT/evaluation/sebset_DEMAND_hoth_1010dB_05sec_4ch_3cm/{angle}/{wave_type}.csv',
                  condition=condition,
                  channel=ch)

