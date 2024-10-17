import All_evaluation as eval
import Multi_Channel_ConvTasNet_test as test
from mymodule import const


angle_list = ["Right", "FrontRight", "Front", "FrontLeft", "Left"]  # "Right", "FrontRight", "Front", "FrontLeft", "Left"

condition = {"speech_type": "subset_DEMAND",
             "noise": "hoth",
             "snr": 10,
             "reverbe": 5}
# ch = [2, 4]\
ch = 4
# distance = 6
# for ch in ch:
# for angle in angle_list:
for reverbe in range(1, 6):
    dir_name = "subset_DEMAND_hoth_1010dB_1ch_to_4ch_1sample_array"
    model_name = "subset_DEMAND_hoth_1010dB_1ch_to_4ch_1sample_array"
    out_dir = f"{const.OUTPUT_WAV_DIR}/{dir_name}/{reverbe:02}sec/"
    wave_path = f"{const.MIX_DATA_DIR}\\{dir_name}\\{reverbe:02}sec\\"
    wave_type_list = ["reverbe_only", "noise_reverbe"]    # "noise_only", "reverbe_only", "noise_reverbe"
    for wave_type in wave_type_list:
        print("test")
        test.test(mix_dir=f"{wave_path}/test/{wave_type}",
                  out_dir=f"{out_dir}/{wave_type}",
                  model_name=f"E:\sound_data\RESLUT\subset_DEMAND_hoth_1010dB_05sec_4ch_circular_6cm/{angle}/subset_DEMAND_hoth_1010dB_05sec_4ch_circular_6cm_noise_only/subset_DEMAND_hoth_1010dB_05sec_4ch_circular_6cm_noise_only_100.pth",
                  channels=ch,
                  model_type="D")


        print("evaluation")
        eval.main(target_dir=f"{wave_path}\\test\\clean",
                  estimation_dir=f"{out_dir}/{wave_type}",
                  out_path=f"{const.EVALUATION_DIR}\\{dir_name}_{reverbe:02}\\{wave_type}.csv",
                  condition=condition,
                  channel=ch)

