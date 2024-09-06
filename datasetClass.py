# coding:utf-8
import numpy as np
from librosa.util import find_files
import torch
from torch.utils.data import DataLoader
import os

import csv

from mymodule import const, my_func

#　npzファイルの読み込み
def load_dataset(dataset_path:str):
    """
    npzファイルから入力データと教師データを読み込む

    Parameters
    ----------
    dataset_path(str):データセットのパス

    Returns
    -------
    mix_list:入力信号
    target_list:目的信号
    """
    # print('\nload_dataset')
    dataset_list = find_files(dataset_path, ext="npz", case_sensitive=True)
    # print('dataset_list:', len(dataset_list))
    mix_list = []
    target_list = []
    for dataset_file in dataset_list:
        dat = np.load(dataset_file)  # datファイルの読み込み
        # print(f'dat:{dat.files}')
        # print('dat:', dat['target'])
        # mix_list.append(dat[const.DATASET_KEY_MIXDOWN])  # 入力データの追加
        mix_list.append(dat['mix'])  # 入力データの追加
        # target_list.append(dat[const.DATASET_KEY_TARGET])  # 正解データの追加
        target_list.append(dat['target'])  # 正解データの追加
    # print('load:np.array(mix_list.shape):', np.array(mix_list).shape)
    # print('load:np.array(target_list.shape):', np.array(target_list).shape)
    # print('load_dataset\n')
    return mix_list, target_list

def load_dataset_csv(dataset_path:str):
    """
    npzファイルから入力データと教師データを読み込む

    Parameters
    ----------
    dataset_path(str):データセットのパス

    Returns
    -------
    mix_list:入力信号
    target_list:目的信号
    """
    # print('\nload_dataset')
    with open(dataset_path) as f:
        reader = csv.reader(f)
        path_list = [row for row in reader]
    path_list.remove(path_list[0])  # ヘッダーの削除
    print(f'path_list:{len(path_list)}')
    mix_list = []
    target_list = []
    for mix_file, target_file in path_list:
        # 音声の読み込み
        mix_data, prm = my_func.load_wav(mix_file)
        target_data, _ = my_func.load_wav(target_file)
        # タイプの変更
        mix_data = mix_data.astype(np.float32)
        target_data = target_data.astype(np.float32)
        target_data = np.pad(target_data, (0, len(mix_data)-len(target_data)))  # ゼロパティング
        mix_list.append(mix_data)  # 入力データの追加
        target_list.append(target_data)  # 教師データの追加

    return mix_list, target_list

def load_dataset_csv_separate(dataset_path:str):
    """
    npzファイルから入力データと教師データを読み込む

    Parameters
    ----------
    dataset_path(str):データセットのパス

    Returns
    -------
    mix_list:入力信号
    target_list:目的信号
    """
    print(f'{dataset_path}')
    with open(dataset_path) as f:
        # reader = csv.reader(f)
        path_list = [row for row in csv.reader(f)]
    path_list.remove(path_list[0])  # ヘッダーの削除
    print(f'path_list:{len(path_list)}')
    mix_list = []
    target_list = []
    for mix_file, target_A_file, target_B_file in path_list:
        # 音声の読み込み
        mix_data, prm = my_func.load_wav(mix_file)
        A_data, _ = my_func.load_wav(target_A_file)
        B_data, _ = my_func.load_wav(target_B_file)
        # タイプの変更
        mix_data = mix_data.astype(np.float32)
        A_data = A_data.astype(np.float32)
        B_data = B_data.astype(np.float32)
        # print('mix:', mix_data.shape)
        # print('A:', A_data.shape)
        # print('B:', B_data.shape)
        A_data = np.pad(A_data, (0, len(mix_data)-len(A_data)), 'constant')  # ゼロパティング
        B_data = np.pad(B_data, (0, len(mix_data)-len(B_data)), 'constant')  # ゼロパティング
        # print('mix:',mix_data.shape)
        # print('A:', A_data.shape)
        # print('B:', B_data.shape)
        target = np.stack([A_data, B_data])   # 教師データを1つにまとめる
        mix_list.append(mix_data)  # 入力データの追加
        target_list.append(target)  # 教師データの追加

    return mix_list, target_list

class dataset(torch.utils.data.Dataset):
    def __init__(self, batchsize, patchlength, dataset_path, subepoch_mag=1):
        """
        初期化

        Parameters
        ----------
        batchsize:1度に読み込むデータの個数
        patchlength:1つのデータを読み込むときの長さ(大きさ)
        dataset_path:データセットのパス
        subepoch_mag:
        """
        self.batchsize = batchsize  # バッチサイズ 32
        self.patchlength = patchlength  # 読み込む長さ 16
        self.subepoch_mag = subepoch_mag    # 1

        """ load train data """
        self.mix_list, self.target_list = load_dataset(dataset_path)    # datasetの読み込み
        self.len = len(self.mix_list)  # 音声ファイルの数
        # print(f'# len:{self.len}')
        # mix_list[0] = [513, 251] = [周波数の数, フレーム数]
        # print(f'# mix_list:{self.mix_list.shape}')

        # 251が2160個ある
        self.itemlength = [x.shape[1] for x in self.mix_list]
        # itemlen = 542160
        # print('# itemlen', sum(self.itemlength))
        self.subepoch = (sum(self.itemlength) // self.patchlength // self.batchsize) * self.subepoch_mag
        # subepoch = 1058
        # print('# subepoch ', self.subepoch)
        # number of patterns = 33856
        # print('# number of patterns', self.__len__())

    def __len__(self):
        # batchsize を　subepoch分　回す。 32*1058
        return self.batchsize * self.subepoch

    def __getitem__(self, i):
        #print('mix_list[0] len...', len(self.mix_list[0]))
        # X([1, 513 ,16]), Y([1, 513, 16])
        X = np.zeros((1, len(self.mix_list[0]), self.patchlength), dtype="float32")
        Y = np.zeros((1, len(self.mix_list[0]), self.patchlength), dtype="float32")
        # print("X",X.shape)
        # インデックスをデータセットの総数内に変換する。
        # def __len__(self)のreturnの値がランダムっぽい
        i0 = i % self.len
        # ランダムに　patch length 長の連続フレームをとりだす。
        randidx = np.random.randint(self.itemlength[i0] - self.patchlength - 1)
        # randidx = torch.from_numpy(randidx)
        #print('randidx', randidx)
        #print("mix_list", self.mix_list[i0].shape)
        X[0, :, :] = self.mix_list[i0][:, randidx:randidx + self.patchlength]
        Y[0, :, :] = self.target_list[i0][:, randidx:randidx + self.patchlength]
        # print("X", X.shape)

        #X = X[0, :, :]
        #Y = Y[0, :, :]

        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        return X, Y

    def get_all(self):
        X = []
        Y = []
        for f in range(len(self.mix_list)):
            x, y = self.__getitem__(f)
            X.append(x)
            Y.append(y)

        return X, Y

class LSTM_dataset(torch.utils.data.Dataset):
    """
    :param
        batchsize   : 1度に読み込むデータの個数
        patchlength : 1つのデータの長さ大きさ
        path_stft   : stftを保存したnpzファイルの場所
    :return
        なし
    """
    def __init__(self, path_stft):
        # load train data
        self.mix_list, self.target_list = load_dataset(path_stft)
        # print("mix_list.shape", self.mix_list[1].shape)
        # 音声ファイルの数
        self.len = len(self.mix_list)
        # print('# len', self.len)

        # print('# number of patterns', self.__len__())

    def __len__(self):
        # データセットのサンプル数が要求されたときに返す処理を実装
        return len(self.mix_list)

    def __getitem__(self, i):
        # i番目のサンプルが要求されたときに返す処理を実装
        # print("i", i)
        X = self.mix_list[i]
        Y = self.target_list[i]
        #print("X", X.dtype)
        #print("Y", Y.shape)

        #X.dtype = "float32"
        #print("X.dtype",X.dtype)
        #Y.dtype = "float32"
        #print("Y.dtype", Y.shape)

        #X = X[np.newaxis, :, :]
        #Y = Y[np.newaxis, :, :]

        #X = np.log(np.square(np.abs(X)))
        #print("X.dtype", X.dtype)
        #Y = np.log(np.square(np.abs(Y)))

        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        return X, Y

    def get_all(self):
        X = []
        Y = []
        for f in range(len(self.mix_list)):
            x, y = self.__getitem__(f)
            X.append(x)
            Y.append(y)
        return X, Y

class TasNet_dataset(torch.utils.data.Dataset):
    """
    :param
        batchsize   : 1度に読み込むデータの個数
        patchlength : 1つのデータの長さ大きさ
        path_stft   : stftを保存したnpzファイルの場所
    :return
        なし
    """
    def __init__(self, dataset_path:str):
        """
        初期化

        Parameters
        ----------
        dataset_path(str):データセットのパス
        """
        """ データの読み込み """
        self.mix_list, self.target_list = load_dataset(dataset_path)
        # print(f'mix_list:{np.array(self.mix_list).shape}')
        # print(f'target_list:{np.array(self.target_list).shape}')
        self.len = len(self.mix_list)  # 学習データの個数
        # print('# len', self.len)
        # print('# number of patterns', self.__len__())

    def __len__(self):
        """　データの個数を返す

        :return: データの個数
        """
        return len(self.mix_list)

    def __getitem__(self, i):
        """i番目の要素(データ)を返す

        :param i: 要素番号(インデックス)
        :return: i番目の要素(データ)
        """
        mix_data = self.mix_list[i]
        target_data = self.target_list[i]
        # print(f'mix_data.shape:{np.array(self.mix_list).shape}')  # 入力信号の形状
        # print(f'target_data.shape:{np.array(self.target_list).shape}')  # 目的信号の形状
        """変数の型の変更"""
        # mix_data.dtype = "float32"
        # target_data.dtype = "float32"
        # print("mix_data.dtype",mix_data.dtype)
        # print("target_data.dtype", target_data.shape)
        """変数の次元の変更　(2次元から3次元へ)"""
        # mix_data = mix_data[np.newaxis, :, :]
        # target_data = target_data[np.newaxis, :, :]

        # mix_data = np.log(np.square(np.abs(mix_data)))
        # target_data = np.log(np.square(np.abs(target_data)))
        # print("mix_data.dtype", mix_data.dtype)
        # print("target_data.dtype", target_data.dtype)
        """型の変更(numpy型からtorch型)"""
        mix_data = torch.from_numpy(mix_data)
        target_data = torch.from_numpy(target_data)

        return mix_data, target_data

    def get_all(self):
        mix_list = []
        target_list = []
        for idx in range(len(self.mix_list)):
            mix_data, target_data = self.__getitem__(idx)
            mix_list.append(mix_data)
            target_list.append(target_data)

        return mix_list, target_list

class TasNet_dataset_csv(torch.utils.data.Dataset):
    """
    :param
        batchsize   : 1度に読み込むデータの個数
        patchlength : 1つのデータの長さ大きさ
        path_stft   : stftを保存したnpzファイルの場所
    :return
        なし
    """
    def __init__(self, dataset_path:str):
        """
        初期化

        Parameters
        ----------
        dataset_path(str):データセットのパス
        """
        """ データの読み込み """
        self.mix_list, self.target_list = load_dataset_csv(dataset_path)
        # print(f'mix_list:{np.array(self.mix_list).shape}')
        # print(f'target_list:{np.array(self.target_list).shape}')
        self.len = len(self.mix_list)  # 学習データの個数
        # print('# len', self.len)
        # print('# number of patterns', self.__len__())

    def __len__(self):
        """　データの個数を返す

        :return: データの個数
        """
        return len(self.mix_list)

    def __getitem__(self, i):
        """i番目の要素(データ)を返す

        :param i: 要素番号(インデックス)
        :return: i番目の要素(データ)
        """
        mix_data = self.mix_list[i]
        target_data = self.target_list[i]
        # print(f'mix_data.shape:{np.array(self.mix_list).shape}')  # 入力信号の形状
        # print(f'target_data.shape:{np.array(self.target_list).shape}')  # 目的信号の形状
        """変数の型の変更"""
        # mix_data.dtype = "float32"
        # target_data.dtype = "float32"
        # print("mix_data.dtype",mix_data.dtype)
        # print("target_data.dtype", target_data.shape)
        """変数の次元の変更　(2次元から3次元へ)"""
        # mix_data = mix_data[np.newaxis, :, :]
        # target_data = target_data[np.newaxis, :, :]

        # mix_data = np.log(np.square(np.abs(mix_data)))
        # target_data = np.log(np.square(np.abs(target_data)))
        # print("mix_data.dtype", mix_data.dtype)
        # print("target_data.dtype", target_data.dtype)
        """型の変更(numpy型からtorch型)"""
        mix_data = torch.from_numpy(mix_data)
        target_data = torch.from_numpy(target_data)

        return mix_data, target_data

    def get_all(self):
        mix_list = []
        target_list = []
        for idx in range(len(self.mix_list)):
            mix_data, target_data = self.__getitem__(idx)
            mix_list.append(mix_data)
            target_list.append(target_data)

        return mix_list, target_list

class TasNet_dataset_csv_separate(TasNet_dataset_csv):
    """
    :param
        batchsize   : 1度に読み込むデータの個数
        patchlength : 1つのデータの長さ大きさ
        path_stft   : stftを保存したnpzファイルの場所
    :return
        なし
    """
    def __init__(self, dataset_path:str):
        """
        初期化

        Parameters
        ----------
        dataset_path(str):データセットのパス
        """
        """ データの読み込み """
        self.mix_list, self.target_list = load_dataset_csv_separate(dataset_path)
        # print(f'mix_list:{np.array(self.mix_list).shape}')
        # print(f'target_list:{np.array(self.target_list).shape}')
        self.len = len(self.mix_list)  # 学習データの個数
        # print('# len', self.len)
        # print('# number of patterns', self.__len__())

if __name__ == '__main__':
    data_path = '../../sound_data/ConvTasNet/dataset/JA_white_00dB2/'
    #x,y = load_dataset(data_path)
    train = TasNet_dataset(data_path)
    print('type(train):',type(train))
    print('np.array(self.mix_list).shape:',np.array(train.mix_list).shape)
    print('fin')

    path = os.getcwd()
