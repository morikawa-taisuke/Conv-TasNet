"""
construction
設定

"""
DATASET_KEY_TARGET = 'target'
DATASET_KEY_MIXDOWN = 'mix'

DIR_KEY_TARGET = 'target'
DIR_KEY_NOISE = 'noise'
DIR_KEY_MIX = 'mix'

SAMPLE_DATA_DIR = 'C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\sound_data\\sample_data\\'
SPEECH_DIR = f'{SAMPLE_DATA_DIR}\\speech\\'
NOISE_DIR = f'{SAMPLE_DATA_DIR}\\noise\\'
MIX_DATA_DIR = f'C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\sound_data\\mix_data\\'
DATASET_DIR = 'C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\DATASET\\'
RESULT_DIR = 'C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\RESULT\\'
OUTPUT_WAV_DIR = f'{RESULT_DIR}\\output_wav\\'
LOG_DIR = f'{RESULT_DIR}\\logs\\'
PTH_DIR = f'{RESULT_DIR}\\pth\\'
EVALUATION_DIR = f'{RESULT_DIR}\\evaluation\\'


SR = 16000  # サンプリング周波数
FFT_SIZE = 1024 # FFTのサイズ
H = 256 # 窓長

BATCHSIZE = 32  # バッチサイズ
PATCHLEN = 16   # パッチサイズ
EPOCH = 5
