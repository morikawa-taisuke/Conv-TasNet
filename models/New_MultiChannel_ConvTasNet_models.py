"""
ConvTasNetのモデルを定義したファイル
"""

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import sys

import sys

sys.path.append('/models\\')
from models import layer_models as models


class TasNet(nn.Module):
    """ 入力：多ch, 出力(マスク)：1ch 音源強調用ConvTasNet(ノーマル) """

    def __init__(self, encoder_dim=512, feature_dim=128, sampling_rate=16000, win=2, layer=8, stack=3,
                 kernel=3, causal=False):  # num_speeker=1もともとのやつ
        super(TasNet, self).__init__()

        """ ハイパーパラメータ """
        self.channel = 1  # 入力のチャンネル数(録音時に使用したマイクの数)
        self.encoder_dim = encoder_dim  # エンコーダの出力次元数
        self.feature_dim = feature_dim  # TCNのボトルネック層の出力次元数
        self.win = int(sampling_rate * win / 1000)  # エンコーダ・デコーダのカーネルサイズ
        self.stride = self.win // 2  # エンコーダ・デコーダのカーネルの移動幅
        self.layer = layer  # TCNの層数
        self.stack = stack  # TCNの繰り返し回数
        self.kernel = kernel  # TCNのカーネルサイズ
        self.causal = causal

        """ encoder エンコーダ """
        self.encoder = nn.Conv1d(in_channels=self.channel,  # 入力データの次元数 (チャンネル数)
                                 out_channels=self.encoder_dim,  # 出力データの次元数
                                 kernel_size=self.win,  # カーネルサイズ (波形領域なので窓長なの?)
                                 bias=False,  # バイアスの有無 (出力に学習可能なバイアスの追加)
                                 stride=self.stride)  # 畳み込み処理の移動幅

        """ TCN separator """
        self.TCN = models.TCN2(input_dim=self.encoder_dim,  # 入力データの次元数
                               output_dim=self.encoder_dim,  # 出力データの次元数
                               BN_dim=self.feature_dim,  # ボトルネック層の出力次元数
                               hidden_dim=self.feature_dim * 4,  # 隠れ層の出力次元数
                               layer=self.layer,  # 層数
                               stack=self.stack,  # スタック数
                               kernel=self.kernel,  # カーネルサイズ
                               causal=self.causal)
        self.receptive_field = self.TCN.receptive_field

        """ decoder """
        self.decoder = nn.ConvTranspose1d(in_channels=self.encoder_dim,  # 入力次元数
                                          out_channels=1,  # 出力次元数 1もともとのやつ
                                          kernel_size=self.win,  # カーネルサイズ
                                          bias=False,
                                          stride=self.stride)  # 畳み込み処理の移動幅

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

        batch_size = input.size(0)
        channels = input.size(1)
        nsample = input.size(2)
        # print(f'input.dim:{input.dim()}')
        # print(f'input.size:{input.size()}')
        rest = self.win - (self.stride + nsample % self.win) % self.win
        # print(f'rest:{rest}')

        if rest > 0:
            zero_tensr = torch.zeros(batch_size, channels, rest)  # tensor型の3次元配列を作成[batch_size, 1, rest]
            # print(f'zero_tensr.size:{zero_tensr.size()}')
            # pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            pad = Variable(zero_tensr).type(input.type())
            # print(f'pad.size():{pad.size()}')
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, self.channel, self.stride)).type(input.type())
        # print(f'pad_aux.size():{pad_aux.size()}')
        input = torch.cat([pad_aux, input, pad_aux], 2)

        # print('patting\n')
        return input, rest

    def forward(self, input):
        """学習の手順(フローチャート)"""
        print('\nstart forward')

        print(f'type(input):{type(input)}')
        print(f'input.shape:{input.shape}')  # input.shape[1,チャンネル数,音声長]
        # padding
        input_patting, rest = self.patting_signal(input)
        print(f'type(input_patting):{type(input_patting)}')
        print(f'input_patting.shape:{input_patting.shape}')
        batch_size = input_patting.size(0)
        print(f'batch_size:{batch_size}')

        # encoder
        print('\nencoder')
        encoder_output = self.encoder(input_patting)  # B, N, L
        print(f'type(encoder_output):{type(encoder_output)}')
        print(f'encoder_output.shape:{encoder_output.shape}')
        print('encoder\n')

        # generate masks (separation)
        print('\nmask')
        masks = torch.sigmoid(self.TCN(encoder_output)).view(batch_size, self.num_speeker, self.encoder_dim,
                                                             -1)  # B, C, N, L
        print(f'type(masks):{type(masks)}')
        print(f'masks.shape:{masks.shape}')
        masked_output = encoder_output.unsqueeze(1) * masks  # B, C, N, L
        print(f'type(masked_output):{type(masked_output)}')
        print(f'masked_output.shape:{masked_output.shape}')
        print('mask\n')

        # decoder
        print('\ndecoder')
        decoder_output = self.decoder(
            masked_output.view(batch_size * self.num_speeker, self.encoder_dim, -1))  # B*C, 1, L
        print(f'0:type(decoder_output):{type(decoder_output)}')
        print(f'0:decoder_output.shape:{decoder_output.shape}')
        decoder_output = decoder_output[:, :, self.stride:-(rest + self.stride)].contiguous()  # B*C, 1, L
        print(f'1:type(decoder_output):{type(decoder_output)}')
        print(f'1:decoder_output.shape:{decoder_output.shape}')
        decoder_output = decoder_output.view(batch_size, self.num_speeker, -1)  # B, C, T
        print(f'2:type(decoder_output):{type(decoder_output)}')
        print(f'2:decoder_output.shape:{decoder_output.shape}')
        print('decoder\n')

        print('end forward\n')
        return decoder_output
