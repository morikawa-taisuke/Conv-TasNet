from __future__ import print_function

import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib import tenumerate
from tqdm import tqdm
import os
# 自作モジュール
import datasetClass




class model(nn.Module):
    def __init__(self, encoder_dim=512, feature_dim=128, sampling_rate=16000, win=2, layer=8, stack=3,
                 kernel=3, num_speeker=1, causal=False, num_mic=1):  # num_speeker=1もともとのやつ
        """
        decoderで2DConvによって多chから1chに畳み込む

        Parameters
        ----------
        encoder_dim
        feature_dim
        sampling_rate
        win
        layer
        stack
        kernel
        num_speeker
        causal
        num_mic
        """
        super(model, self).__init__()

        # hyper parameters
        self.num_speeker = num_speeker  # 話者数
        self.encoder_dim = encoder_dim  # エンコーダに入力する次元数
        self.feature_dim = feature_dim  # 特徴次元数
        self.win = int(sampling_rate * win / 1000)  # 窓長 (1回に処理するデータ量)
        self.stride = self.win // 2  # 畳み込み処理におけるフィルタが移動する幅
        self.layer = layer  # 層数
        self.stack = stack  #
        self.kernel = kernel  # カーネル
        self.causal = causal  #
        self.num_mic = num_mic  # チャンネル数

        self.patting = 0
        self.dilation = 1

        """
        # input encoder
        self.encoder = nn.Conv2d(in_channels=self.num_mic,      # 入力データの次元数 #=1もともとのやつ
                                 out_channels=self.encoder_dim, # 出力データの次元数
                                 kernel_size=self.win,          # 畳み込みのサイズ(波形領域なので窓長なの?)
                                 bias=False,                    # バイアスの有無(出力に学習可能なバイアスの追加)
                                 stride=self.stride)            # 畳み込み処理の移動幅
        """
        # input encoder
        self.encoder = nn.Conv1d(in_channels=1,  # 入力データの次元数 #=1もともとのやつ
                                 out_channels=self.encoder_dim,  # 出力データの次元数
                                 kernel_size=self.win,  # 畳み込みのサイズ(波形領域なので窓長のイメージ?)
                                 bias=False,  # バイアスの有無(出力に学習可能なバイアスの追加)
                                 stride=self.stride)  # 畳み込み処理の移動幅

        # output decoder
        self.decoder = nn.ConvTranspose1d(in_channels=encoder_dim,
                                          out_channels=1,
                                          kernel_size=self.win,
                                          bias=False,
                                          stride=self.stride)

        # self.decoder = nn.Conv2d(in_channels=num_mic,  # 入力次元数
        #                          out_channels=1,  # 出力次元数 1もともとのやつ
        #                          kernel_size=(self.encoder_dim, self.win),  # カーネルサイズ
        #                          bias=False,
        #                          stride=(1, self.stride))  # 畳み込み処理の移動幅
        # self.decoder = nn.ConvTranspose1d(in_channels=self.encoder_dim,  # 入力次元数
        #                                   out_channels=1,  # 出力次元数 1もともとのやつ
        #                                   kernel_size=self.win,  # カーネルサイズ
        #                                   bias=False,
        #                                   stride=self.stride)  # 畳み込み処理の移動幅

    def patting_signal(self, input):
        """入力データをパティング→畳み込み前の次元数と畳み込み後の次元数を同じにするために入力データを0で囲む操作

        :param input: 入力信号 tensor型[1,チャンネル数,音声長]
        :return:
        """
        # print('\npatting')
        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:  # inputの次元数が2or3出ないとき
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:  # inputの次元数が2の時
            # print('input.unsqueeze')
            input = input.unsqueeze(1)  # 形状のn番目が1になるように次元を追加(今回の場合n=1)

        # batch_size = input.size(0)    # バッチサイズ
        # channels = input.size(1)  # チャンネル数 (マイク数)
        # nsample = input.size(2)   # 音声長
        # print(f'input.dim:{input.dim()}')
        # print(f'input.size:{input.size()}')
        rest = self.win - (self.stride + input.size(2) % self.win) % self.win
        # print(f'rest:{rest}')

        if rest > 0:
            zero_tensr = torch.zeros(input.size(0), input.size(1), rest)  # tensor型の3次元配列を作成[batch_size, 1, rest]
            # print(f'zero_tensr.size:{zero_tensr.size()}')
            # pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            pad = Variable(zero_tensr).type(input.type())
            # print(f'pad.size():{pad.size()}')
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(input.size(0), self.num_mic, self.stride)).type(input.type())
        # print(f'pad_aux.size():{pad_aux.size()}')
        input = torch.cat([pad_aux, input, pad_aux], 2)

        # print('patting\n')
        return input, rest

    def get_dim_length(self, input_patting):
        """エンコード後の特徴量領域のデータ長を計算

        :param input_patting: パティングされた入力
        :return out_length: エンコード後のデータ長
        """
        in_length = input_patting.size(2)
        patting = 0
        dilation = 0
        # print(f'in_length:{in_length}')
        # print(f'{in_length},{self.patting}-{self.dilation},{self.win},{self.stride}')
        out_length = ((in_length + 2 * patting - dilation * (self.win - 1) - 1) / self.stride) + 1
        # print(f'out_length:{out_length}')
        return int(out_length)

    def forward(self, input):
        """学習の手順(フローチャート)"""
        print('\nstart forward')
        print(f'input.shape:{input.shape}') #input.shape[1,チャンネル数,音声長]
        wave_length = input.size(2)
        """ padding """
        input, rest = self.patting_signal(input)
        # print(f'type(input):{type(input)}')
        # print(f'input.shape:{input.shape}')
        batch_size = input.size(0)
        # print(f'batch_size:{batch_size}')

        """ encoder """
        # print('\nencoder')
        dim_length = self.get_dim_length(input) - 1
        encoder_output = torch.empty(self.num_mic, self.encoder_dim, dim_length).to("cuda")
        for idx, input in enumerate(input[0]):
            input = input.unsqueeze(0)  # 次元の追加 [128000]->[1,128000]
            input = self.encoder(input)  # エンコーダに通す
            # input=input.unsqueeze(0)
            encoder_output[idx] = input
        # encoder_output = self.encoder(input)  # B, N, L   # 元のやつ encoder_outputの形状が違う
        print(f'encoder_output.shape:{encoder_output.shape}')
        # print('encoder\n')

        # print(f"encoder_output:{encoder_output.shape}")
        """ decoder """
        # print('\ndecoder')
        # decoder_output = self.decoder(encoder_output.view(batch_size * self.num_speeker, self.encoder_dim, -1))  # B*C, 1, L   #元のやつ
        # encoder_output = encoder_output.squeeze()
        encoder_output = encoder_output.squeeze()
        # print_name_type_shape('encoder_output',encoder_output)
        decoder_output = torch.empty(self.num_mic, wave_length).to("cuda")
        for idx, output in enumerate(encoder_output):
            # print_name_type_shape('input',input)
            output = self.decoder(output)  # B*C, 1, L
            # print_name_type_shape(f'[{idx}]0:decoder_output',output)
            output = output[:, self.stride:-(rest + self.stride)].contiguous()  # B*C, 1, L
            # print_name_type_shape(f'[{idx}]1:decoder_output',output)
            output = output.view(batch_size, self.num_speeker, -1)  # B, C, T
            # print_name_type_shape(f'[{idx}]2:decoder_output',output)
            decoder_output[idx] = output
            # decoder_output = decoder_output.view(batch_size, self.num_speeker, -1)  # B, C, T
            # print_name_type_shape(f'2:decoder_output', decoder_output)
        # print_name_type_shape('decoder_output',decoder_output)
        decoder_output = decoder_output.unsqueeze(dim=0)

        decoder_output = self.decoder(encoder_output)
        print(f"decoder_output:{decoder_output.shape}")
        return decoder_output


if __name__ == "__main__":

    dataset_path = "C:\\Users\\kataoka-lab\\Desktop\\sound_data\\dataset\\DEMAND_hoth_1010dB_4ch_6cm\\DEMAND_hoth_1010dB_05sec_4ch_6cm\\pre_test"
    train_count = 1

    device = "cuda"
    dataset = datasetClass.TasNet_dataset(dataset_path) # データセットの読み込み
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True)    # データローダーの作成

    TasNet_model = model(num_mic=4).to(device)  # モデルの作成

    optimizer = optim.Adam(TasNet_model.parameters(), lr=0.001)    # optimizerを選択(Adam)
    loss_function = nn.MSELoss().to(device) # 損失関数に使用する式の指定(最小二乗誤差)
    writer = SummaryWriter(log_dir=f"./log_dir")  # logの保存先の指定("tensorboard --logdir ./logs"で確認できる)


    def train(epoch):
        model_loss_sum = 0              # 総損失の初期化
        print("Train Epoch:", epoch)    # 学習回数の表示
        for batch_idx, (mix_data, target_data) in tenumerate(dataset_loader):
            """ データの読み込み """
            mix_data, target_data = mix_data.to(device), target_data.to(device) # データをGPUに移動

            """ 勾配のリセット """
            optimizer.zero_grad()  # optimizerの初期化

            """ データの整形 """
            mix_data = mix_data.to(torch.float32)   # target_dataのタイプを変換 int16→float32
            target_data = target_data.to(torch.float32) # target_dataのタイプを変換 int16→float32
            # target_data = target_data[np.newaxis, :, :] # 次元を増やす[1,音声長]→[1,1,音声長]

            """ モデルに通す(予測値の計算) """
            estimate_data = TasNet_model(mix_data)          # モデルに通す

            """ データの整形 """
            # split_estimate_data = split_data(estimate_data[0],channels)
            # split_estimate_data = split_estimate_data[np.newaxis, :, :]

            """ 損失の計算 """
            """ 周波数軸に変換 """
            stft_estimate_data = torch.stft(estimate_data[0, 0, :], n_fft=1024, return_complex=False)
            stft_target_data = torch.stft(target_data[0, 0, :], n_fft=1024, return_complex=False)
            # print("\nstft")
            # print(f"stft_estimate_data.shape:{stft_estimate_data.shape}")
            # print(f"stft_target_data.shape:{stft_target_data.shape}")
            # print("stft\n")
            model_loss = loss_function(stft_estimate_data, stft_target_data)  # 時間周波数上MSEによる損失の計算
            # print(f"estimate_data.size(1):{estimate_data.size(1)}")

            model_loss_sum += model_loss  # 損失の加算

            """ 後処理 """
            model_loss.backward()           # 誤差逆伝搬
            optimizer.step()                # 勾配の更新

        writer.add_scalar("log_file", model_loss_sum, epoch)
        #writer.add_scalar(str(str_name[0]) + "_" + str(a) + "_sisdr-sisnr", model_loss_sum, epoch)
        print(f"[{epoch}]model_loss_sum:{model_loss_sum}")  # 損失の出力

        # torch.cuda.empty_cache()    # メモリの解放 1iterationごとに解放
        with open("./log_dir/log_file.csv", "a") as out_file:  # ファイルオープン
            out_file.write(f"{model_loss_sum}\n")  # 書き込み

    for epoch in range(1, train_count+1):   # 学習回数
        train(epoch)

    print("model save")
    torch.save(model.to(device).state_dict(), f"./log_dir/save_model.pth")         # 出力ファイルの保存
