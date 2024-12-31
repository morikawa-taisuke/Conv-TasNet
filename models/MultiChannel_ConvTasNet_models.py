"""
ConvTasNetのモデルを定義したファイル
"""

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import sys

import sys

sys.path.append('C:\\Users\\kataoka-lab\\Desktop\\Conv-TasNet\\models\\')
from models import layer_models as models


# from my_func import print_name_type_shape

class TasNet(nn.Module):
    """ 入力：多ch, 出力(マスク)：1ch 音源強調用ConvTasNet(ノーマル) """

    def __init__(self, encoder_dim=512, feature_dim=128, sampling_rate=16000, win=2, layer=8, stack=3,
                 kernel=3, causal=False):  # num_speeker=1もともとのやつ
        super(TasNet, self).__init__()

        """ ハイパーパラメータ """
        self.channel = 4  # 入力のチャンネル数(録音時に使用したマイクの数)
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


class type_A(nn.Module):
    def __init__(self, encoder_dim=512, feature_dim=128, sampling_rate=16000, win=2, layer=8, stack=3,
                 kernel=3, causal=False):  # num_speeker=1もともとのやつ
        super(type_A, self).__init__()

        """ ハイパーパラメータ """
        self.channel = 4  # 入力のチャンネル数(録音時に使用したマイクの数)
        self.encoder_dim = encoder_dim  # エンコーダの出力次元数
        self.feature_dim = feature_dim  # TCNのボトルネック層の出力次元数
        self.win = int(sampling_rate * win / 1000)  # エンコーダ・デコーダのカーネルサイズ
        self.stride = self.win // 2  # エンコーダ・デコーダのカーネルの移動幅
        self.layer = layer  # TCNの層数
        self.stack = stack  # TCNの繰り返し回数
        self.kernel = kernel  # 1-DConv(TCN)のカーネルサイズ
        self.causal = causal

        """ encoder エンコーダ """
        self.encoder = nn.Conv1d(in_channels=1,  # 入力データの次元数 #=1もともとのやつ
                                 out_channels=self.encoder_dim,  # 出力データの次元数
                                 kernel_size=self.win,  # カーネルサイズ(波形領域なので窓長のイメージ?)
                                 bias=False,  # バイアスの有無(出力に学習可能なバイアスの追加)
                                 stride=self.stride)  # 畳み込み処理の移動幅

        """ TCN separator """
        self.TCN = models.TCN_A(input_dim=self.encoder_dim,  # 入力データの次元数
                                output_dim=self.encoder_dim,  # 出力データの次元数
                                BN_dim=self.feature_dim,  # ボトルネック層の出力次元数
                                hidden_dim=self.feature_dim * 4,  # 隠れ層の出力次元数
                                layer=self.layer,  # TCNの層数
                                stack=self.stack,  # TCNの繰り返し回数
                                kernel=self.kernel,  # 1-DConvのカーネルサイズ
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
        return int(out_length) - 1

    def forward(self, input):
        """学習の手順(フローチャート)"""
        # print('\nstart forward')

        # print(f'type(input):{type(input)}')
        # print(f'input.shape:{input.shape}') #input.shape[1,チャンネル数,音声長]
        # print_name_type_shape('input',input)
        wave_length = input.size(2)
        """ padding """
        input_patting, rest = self.patting_signal(input)
        # print(f'type(input_patting):{type(input_patting)}')
        # print(f'input_patting.shape:{input_patting.shape}')
        # print_name_type_shape('input_patting',input_patting)
        batch_size = input_patting.size(0)
        # print(f'batch_size:{batch_size}')

        """ encoder """
        # print('\nencoder')
        dim_length = self.get_dim_length(input_patting)
        encoder_output = torch.empty(self.channel, self.encoder_dim, dim_length).to("cuda")
        for idx, input in enumerate(input_patting[0]):
            # input=input.unsqueeze(0)
            # print_name_type_shape('input:0',input)
            input = input.unsqueeze(0)  # 次元の追加 [128000]->[1,128000]
            # print_name_type_shape('input:1', input)
            input = self.encoder(input)  # エンコーダに通す
            # input=input.unsqueeze(0)
            # print_name_type_shape(f'for_input[{idx}]',input)
            encoder_output[idx] = input
        # print_name_type_shape('encoder_output',encoder_output)
        # encoder_output = self.encoder(input_patting)  # B, N, L   # 元のやつ encoder_outputの形状が違う
        # print(f'type(encoder_output):{type(encoder_output)}')
        # print(f'encoder_output.shape:{encoder_output.shape}')
        # print_name_type_shape('encoder_output',encoder_output)
        # print('encoder\n')

        """ generate masks (separation) """
        # print('\nmask')
        masks = torch.sigmoid(self.TCN(encoder_output)).view(batch_size, 1, self.encoder_dim,
                                                             -1)  # B, C, N, L
        # print(f'type(masks):{type(masks)}')
        # print(f'masks.shape:{masks.shape}')
        # print_name_type_shape('masks',masks)
        # print_name_type_shape('encoder_output.unsqueeze(1)',encoder_output.unsqueeze(1))
        masked_output = encoder_output.unsqueeze(1) * masks  # B, C, N, L
        # print(f'type(masked_output):{type(masked_output)}')
        # print(f'masked_output.shape:{masked_output.shape}')
        # print_name_type_shape('masked_output',masked_output)
        # print('mask\n')

        """ decoder """
        # print('\ndecoder')
        # decoder_output = self.decoder(masked_output.view(batch_size * self.num_speeker, self.encoder_dim, -1))  # B*C, 1, L   #元のやつ
        decoder_input = masked_output.squeeze()
        # print_name_type_shape('decoder_input',decoder_input)
        decoder_output = torch.empty(self.channel, wave_length).to("cuda")
        for idx, input in enumerate(decoder_input):
            # print_name_type_shape('input',input)
            output = self.decoder(input)  # B*C, 1, L
            # print_name_type_shape(f'[{idx}]0:decoder_output',output)
            output = output[:, self.stride:-(rest + self.stride)].contiguous()  # B*C, 1, L
            # print_name_type_shape(f'[{idx}]1:decoder_output',output)
            output = output.view(batch_size, 1, -1)  # B, C, T
            # print_name_type_shape(f'[{idx}]2:decoder_output',output)
            decoder_output[idx] = output
            # decoder_output = decoder_output.view(batch_size, self.num_speeker, -1)  # B, C, T
            # print_name_type_shape(f'2:decoder_output', decoder_output)
        # print_name_type_shape('decoder_output',decoder_output)
        decoder_output = decoder_output.unsqueeze(dim=0)
        # print_name_type_shape('decoder_output',decoder_output)

        # print('decoder\n')

        # print('end forward\n')
        return decoder_output


class type_C(nn.Module):
    def __init__(self, encoder_dim=512, feature_dim=128, sampling_rate=16000, win=2, layer=8, stack=3,
                 kernel=3, causal=False):  # num_speeker=1もともとのやつ
        super(type_C, self).__init__()

        """ ハイパーパラメータ """
        self.channel = 4  # 入力のチャンネル数(録音時に使用したマイクの数)
        self.encoder_dim = encoder_dim  # エンコーダの出力次元数
        self.feature_dim = feature_dim  # TCNのボトルネック層の出力次元数
        self.win = int(sampling_rate * win / 1000)  # エンコーダ・デコーダのカーネルサイズ
        self.stride = self.win // 2  # エンコーダ・デコーダのカーネルの移動幅
        self.layer = layer  # TCNの層数
        self.stack = stack  # TCNの繰り返し回数
        self.kernel = kernel  # 1-DConv(TCN)のカーネルサイズ
        self.causal = causal

        """ encoder """
        self.encoder = nn.Conv1d(in_channels=1,  # 入力データの次元数 #=1もともとのやつ
                                 out_channels=self.encoder_dim,  # 出力データの次元数
                                 kernel_size=self.win,  # 畳み込みのサイズ(波形領域なので窓長のイメージ?)
                                 bias=False,  # バイアスの有無(出力に学習可能なバイアスの追加)
                                 stride=self.stride)  # 畳み込み処理の移動幅

        """ TCN separator """
        self.TCN = models.TCN_C(input_dim=self.encoder_dim,  # 入力データの次元数
                                output_dim=self.encoder_dim,  # 出力データの次元数
                                BN_dim=self.feature_dim,  # ボトルネック層の出力次元数
                                hidden_dim=self.feature_dim * 4,  # 隠れ層の出力次元数
                                layer=self.layer,  # layer個の1-DConvブロックをつなげる
                                stack=self.stack,  # stack回繰り返す
                                kernel=self.kernel,  # 1-DConvのカーネルサイズ
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

    def get_dim_length(self, input_patting):
        """エンコード後の特徴量領域のデータ長を計算

        :param input_patting: パティングされた入力
        :return out_length: エンコード後のデータ長
        """
        in_length = input_patting.size(2)
        patting = 0
        dilation = 1
        # print(f'in_length:{in_length}')
        # print(f'{in_length},{self.patting}-{self.dilation},{self.win},{self.stride}')
        out_length = ((in_length + 2 * patting - dilation * (self.win - 1) - 1) / self.stride) + 1
        # print(f'out_length:{out_length}')
        return int(out_length)

    def forward(self, input):
        """学習の手順(フローチャート)"""
        # print('\nstart forward')

        # print_name_type_shape('input',input)  #input.shape[1,チャンネル数,音声長]
        wave_length = input.size(2)
        """ padding """
        input_patting, rest = self.patting_signal(input)
        # print_name_type_shape('input_patting',input_patting)
        batch_size = input_patting.size(0)
        # print(f'batch_size:{batch_size}')

        """ encoder """
        # print('\nencoder')
        dim_length = self.get_dim_length(input_patting)
        encoder_output = torch.empty(self.channel, self.encoder_dim, dim_length).to("cuda")
        for idx, input in enumerate(input_patting[0]):
            # input=input.unsqueeze(0)
            # print_name_type_shape('input:0',input)
            input = input.unsqueeze(0)  # 次元の追加 [128000]->[1,128000]
            # print_name_type_shape('input:1', input)
            input = self.encoder(input)  # エンコーダに通す
            # input=input.unsqueeze(0)
            # print_name_type_shape(f'for_input[{idx}]',input)
            encoder_output[idx] = input
        # encoder_output = self.encoder(input_patting)  # B, N, L   # 元のやつ encoder_outputの形状が違う
        # print_name_type_shape('encoder_output',encoder_output)
        # print('encoder\n')

        """ generate masks (separation) """
        # print('\nmask')
        masks = torch.sigmoid(self.TCN(encoder_output)).view(batch_size, 1, self.encoder_dim,
                                                             -1)  # B, C, N, L
        # print(f'type(masks):{type(masks)}')
        # print(f'masks.shape:{masks.shape}')
        # print_name_type_shape('masks',masks)
        # print_name_type_shape('encoder_output.unsqueeze(1)',encoder_output.unsqueeze(1))
        masked_output = encoder_output.unsqueeze(1) * masks  # B, C, N, L
        # print(f'type(masked_output):{type(masked_output)}')
        # print(f'masked_output.shape:{masked_output.shape}')
        # print_name_type_shape('masked_output',masked_output)
        # print('mask\n')

        """ decoder """
        # print('\ndecoder')
        # decoder_output = self.decoder(masked_output.view(batch_size * self.num_speeker, self.encoder_dim, -1))  # B*C, 1, L   #元のやつ
        decoder_input = masked_output.squeeze()
        # print_name_type_shape('decoder_input',decoder_input)
        decoder_output = torch.empty(self.channel, wave_length).to("cuda")
        for idx, input in enumerate(decoder_input):
            # print_name_type_shape('input',input)
            output = self.decoder(input)  # B*C, 1, L
            # print_name_type_shape(f'[{idx}]0:decoder_output',output)
            output = output[:, self.stride:-(rest + self.stride)].contiguous()  # B*C, 1, L
            # print_name_type_shape(f'[{idx}]1:decoder_output',output)
            output = output.view(batch_size, 1, -1)  # B, C, T
            # print_name_type_shape(f'[{idx}]2:decoder_output',output)
            decoder_output[idx] = output
            # decoder_output = decoder_output.view(batch_size, self.num_speeker, -1)  # B, C, T
            # print_name_type_shape(f'2:decoder_output', decoder_output)
        # print_name_type_shape('decoder_output',decoder_output)
        decoder_output = decoder_output.unsqueeze(dim=0)
        # print_name_type_shape('decoder_output',decoder_output)
        # print('decoder\n')

        # print('end forward\n')
        return decoder_output


class type_D(nn.Module):
    def __init__(self, encoder_dim=512, feature_dim=128, sampling_rate=16000, win=2, layer=8, stack=3,
                 kernel=3, num_speeker=1, causal=False):  # num_speeker=1もともとのやつ
        super(type_D, self).__init__()

        # hyper parameters
        self.num_speeker = num_speeker  # 話者数
        self.encoder_dim = encoder_dim  # エンコーダに入力する次元数
        self.feature_dim = feature_dim  # 特徴次元数
        self.win = int(sampling_rate * win / 1000)  # 窓長
        self.stride = self.win // 2  # 畳み込み処理におけるフィルタが移動する幅
        self.layer = layer  # 層数
        self.stack = stack  #
        self.kernel = kernel  # カーネル
        self.causal = causal  #
        self.channel = 4  # チャンネル数

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

        # TCN separator
        self.TCN = models.TCN_D(input_dim=self.encoder_dim,  # 入力データの次元数
                                output_dim=self.encoder_dim * self.num_speeker,  # 出力データの次元数
                                BN_dim=self.feature_dim,  # ボトルネック層の出力次元数
                                hidden_dim=self.feature_dim * 4,  # 隠れ層の出力次元数
                                layer=self.layer,  # layer個の1-DConvブロックをつなげる
                                stack=self.stack,  # stack回繰り返す
                                kernel=self.kernel,  # 1-DConvのカーネルサイズ
                                causal=self.causal)
        self.receptive_field = self.TCN.receptive_field

        # output decoder
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
        # print('\nstart forward')

        # print(f'type(input):{type(input)}')
        # print(f'input.shape:{input.shape}') #input.shape[1,チャンネル数,音声長]
        # print_name_type_shape('input',input)
        wave_length = input.size(2)
        """ padding """
        input_patting, rest = self.patting_signal(input)
        # print(f'type(input_patting):{type(input_patting)}')
        # print(f'input_patting.shape:{input_patting.shape}')
        # print_name_type_shape('input_patting',input_patting)
        batch_size = input_patting.size(0)
        # print(f'batch_size:{batch_size}')

        """ encoder """
        # print('\nencoder')
        dim_length = self.get_dim_length(input_patting) - 1
        encoder_output = torch.empty(self.channel, self.encoder_dim, dim_length).to("cuda")
        for idx, input in enumerate(input_patting[0]):
            input = input.unsqueeze(0)  # 次元の追加 [128000]->[1,128000]
            # print_name_type_shape('input:1', input)
            input = self.encoder(input)  # エンコーダに通す
            # input=input.unsqueeze(0)
            # print_name_type_shape(f'for_input[{idx}]',input)
            encoder_output[idx] = input
        # encoder_output = self.encoder(input_patting)  # B, N, L   # 元のやつ encoder_outputの形状が違う
        # print(f'type(encoder_output):{type(encoder_output)}')
        # print(f'encoder_output.shape:{encoder_output.shape}')
        # print_name_type_shape('encoder_output',encoder_output)
        # print('encoder\n')

        """ generate masks (separation) """
        # print('\nmask')
        # TVN_output=
        masks = torch.sigmoid(
            self.TCN(encoder_output))  # .view(batch_size, self.num_speeker, self.encoder_dim, -1)  # B, C, N, L
        # print(f'type(masks):{type(masks)}')
        # print(f'masks.shape:{masks.shape}')
        # print_name_type_shape('masks',masks)
        # print_name_type_shape('encoder_output.unsqueeze(1)',encoder_output.unsqueeze(1))
        masked_output = encoder_output * masks  # B, C, N, L
        # print(f'type(masked_output):{type(masked_output)}')
        # print(f'masked_output.shape:{masked_output.shape}')
        # print_name_type_shape('masked_output',masked_output)
        # print('mask\n')

        """ decoder """
        # print('\ndecoder')
        # decoder_output = self.decoder(masked_output.view(batch_size * self.num_speeker, self.encoder_dim, -1))  # B*C, 1, L   #元のやつ
        decoder_input = masked_output.squeeze()
        # print_name_type_shape('decoder_input',decoder_input)
        decoder_output = torch.empty(self.channel, wave_length).to("cuda")
        for idx, input in enumerate(decoder_input):
            # print_name_type_shape('input',input)
            output = self.decoder(input)  # B*C, 1, L
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
        # print_name_type_shape('decoder_output',decoder_output)

        # print('decoder\n')

        # print('end forward\n')
        return decoder_output


class type_E(nn.Module):

    def __init__(self, encoder_dim=512, feature_dim=128, sampling_rate=16000, win=2, layer=8, stack=3,
                 kernel=3, num_speeker=1, causal=False):  # num_speeker=1もともとのやつ
        super(type_E, self).__init__()

        # hyper parameters
        self.num_speeker = num_speeker  # 話者数
        self.encoder_dim = encoder_dim  # エンコーダに入力する次元数
        self.feature_dim = feature_dim  # 特徴次元数
        self.win = int(sampling_rate * win / 1000)  # 窓長
        self.stride = self.win // 2  # 畳み込み処理におけるフィルタが移動する幅
        self.layer = layer  # 層数
        self.stack = stack  #
        self.kernel = kernel  # カーネル
        self.causal = causal  #
        self.channel = 4  # チャンネル数

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

        # TCN separator
        self.TCN = models.TCN_E(input_dim=self.encoder_dim,  # 入力データの次元数
                                output_dim=self.encoder_dim * self.num_speeker,  # 出力データの次元数
                                BN_dim=self.feature_dim,  # ボトルネック層の出力次元数
                                hidden_dim=self.feature_dim * 4,  # 隠れ層の出力次元数
                                layer=self.layer,  # layer個の1-DConvブロックをつなげる
                                stack=self.stack,  # stack回繰り返す
                                kernel=self.kernel,  # 1-DConvのカーネルサイズ
                                causal=self.causal)
        self.receptive_field = self.TCN.receptive_field

        # output decoder
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
        # print(f'input.size(0):{input.size(0)}')
        # print(f'input.size(1):{input.size(1)}')
        # print(f'input.size(2):{input.size(2)}')
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
        # print('\nstart forward')

        # print(f'type(input):{type(input)}')
        # print(f'input.shape:{input.shape}') #input.shape[1,チャンネル数,音声長]
        # print_name_type_shape('input',input)
        wave_length = input.size(2)
        """ padding """
        input_patting, rest = self.patting_signal(input)
        # print(f'type(input_patting):{type(input_patting)}')
        # print(f'input_patting.shape:{input_patting.shape}')
        # print_name_type_shape('input_patting',input_patting)
        batch_size = input_patting.size(0)
        # print(f'batch_size:{batch_size}')

        """ encoder """
        # print('\nencoder')
        dim_length = self.get_dim_length(input_patting) - 1
        encoder_output = torch.empty(self.channel, self.encoder_dim, dim_length).to("cuda")
        for idx, input in enumerate(input_patting[0]):
            # input=input.unsqueeze(0)
            # print_name_type_shape('input:0',input)
            input = input.unsqueeze(0)  # 次元の追加 [128000]->[1,128000]
            # print_name_type_shape('input:1', input)
            input = self.encoder(input)  # エンコーダに通す
            # input=input.unsqueeze(0)
            # print_name_type_shape(f'for_input[{idx}]',input)
            encoder_output[idx] = input
        # encoder_output = self.encoder(input_patting)  # B, N, L   # 元のやつ encoder_outputの形状が違う
        # print(f'type(encoder_output):{type(encoder_output)}')
        # print(f'encoder_output.shape:{encoder_output.shape}')
        # print_name_type_shape('encoder_output',encoder_output)
        # print('encoder\n')

        """ generate masks (separation) """
        # print('\nmask')
        # TVN_output=
        masks = torch.sigmoid(
            self.TCN(encoder_output))  # .view(batch_size, self.num_speeker, self.encoder_dim, -1)  # B, C, N, L
        # print(f'type(masks):{type(masks)}')
        # print(f'masks.shape:{masks.shape}')
        # print_name_type_shape('masks',masks)
        # print_name_type_shape('encoder_output.unsqueeze(1)',encoder_output.unsqueeze(1))
        masked_output = encoder_output * masks  # B, C, N, L
        # print(f'type(masked_output):{type(masked_output)}')
        # print(f'masked_output.shape:{masked_output.shape}')
        # print_name_type_shape('masked_output',masked_output)
        # print('mask\n')

        """ decoder """
        # print('\ndecoder')
        # decoder_output = self.decoder(masked_output.view(batch_size * self.num_speeker, self.encoder_dim, -1))  # B*C, 1, L   #元のやつ
        decoder_input = masked_output.squeeze()
        # print_name_type_shape('decoder_input',decoder_input)
        decoder_output = torch.empty(self.channel, wave_length).to("cuda")
        for idx, input in enumerate(decoder_input):
            # print_name_type_shape('input',input)
            output = self.decoder(input)  # B*C, 1, L
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
        # print_name_type_shape('decoder_output',decoder_output)

        # print('decoder\n')

        # print('end forward\n')
        return decoder_output


class type_F(nn.Module):

    def __init__(self, encoder_dim=512, feature_dim=128, sampling_rate=16000, win=2, layer=8, stack=3,
                 kernel=3, num_speeker=1, causal=False):  # num_speeker=1もともとのやつ
        super(type_F, self).__init__()

        # hyper parameters
        self.num_speeker = num_speeker  # 話者数
        self.encoder_dim = encoder_dim  # エンコーダに入力する次元数
        self.feature_dim = feature_dim  # 特徴次元数
        self.win = int(sampling_rate * win / 1000)  # 窓長
        self.stride = self.win // 2  # 畳み込み処理におけるフィルタが移動する幅
        self.layer = layer  # 層数
        self.stack = stack  #
        self.kernel = kernel  # カーネル
        self.causal = causal  #
        self.channel = 4  # チャンネル数

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

        # TCN separator
        self.TCN = models.TCN_F(input_dim=self.encoder_dim,  # 入力データの次元数
                                output_dim=self.encoder_dim * self.num_speeker,  # 出力データの次元数
                                BN_dim=self.feature_dim,  # ボトルネック層の出力次元数
                                hidden_dim=self.feature_dim * 4,  # 隠れ層の出力次元数
                                layer=self.layer,  # layer個の1-DConvブロックをつなげる
                                stack=self.stack,  # stack回繰り返す
                                kernel=self.kernel,  # 1-DConvのカーネルサイズ
                                causal=self.causal)
        self.receptive_field = self.TCN.receptive_field

        # output decoder
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
        # print(f'input.size(0):{input.size(0)}')
        # print(f'input.size(1):{input.size(1)}')
        # print(f'input.size(2):{input.size(2)}')
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
        # print('\nstart forward')

        # print(f'type(input):{type(input)}')
        # print(f'input.shape:{input.shape}') #input.shape[1,チャンネル数,音声長]
        # print_name_type_shape('input',input)
        wave_length = input.size(2)
        """ padding """
        input_patting, rest = self.patting_signal(input)
        # print(f'type(input_patting):{type(input_patting)}')
        # print(f'input_patting.shape:{input_patting.shape}')
        # print_name_type_shape('input_patting',input_patting)
        batch_size = input_patting.size(0)
        # print(f'batch_size:{batch_size}')

        """ encoder """
        # print('\nencoder')
        dim_length = self.get_dim_length(input_patting)
        encoder_output = torch.empty(self.channel, self.encoder_dim, dim_length).to("cuda")
        for idx, input in enumerate(input_patting[0]):
            # input=input.unsqueeze(0)
            # print_name_type_shape('input:0',input)
            input = input.unsqueeze(0)  # 次元の追加 [128000]->[1,128000]
            # print_name_type_shape('input:1', input)
            input = self.encoder(input)  # エンコーダに通す
            # input=input.unsqueeze(0)
            # print_name_type_shape(f'for_input[{idx}]',input)
            encoder_output[idx] = input
        # encoder_output = self.encoder(input_patting)  # B, N, L   # 元のやつ encoder_outputの形状が違う
        # print(f'type(encoder_output):{type(encoder_output)}')
        # print(f'encoder_output.shape:{encoder_output.shape}')
        # print_name_type_shape('encoder_output',encoder_output)
        # print('encoder\n')

        """ generate masks (separation) """
        # print('\nmask')
        # TVN_output=
        masks = torch.sigmoid(
            self.TCN(encoder_output))  # .view(batch_size, self.num_speeker, self.encoder_dim, -1)  # B, C, N, L
        # print(f'type(masks):{type(masks)}')
        # print(f'masks.shape:{masks.shape}')
        # print_name_type_shape('masks',masks)
        # print_name_type_shape('encoder_output.unsqueeze(1)',encoder_output.unsqueeze(1))
        masked_output = encoder_output * masks  # B, C, N, L
        # print(f'type(masked_output):{type(masked_output)}')
        # print(f'masked_output.shape:{masked_output.shape}')
        # print_name_type_shape('masked_output',masked_output)
        # print('mask\n')

        """ decoder """
        # print('\ndecoder')
        # decoder_output = self.decoder(masked_output.view(batch_size * self.num_speeker, self.encoder_dim, -1))  # B*C, 1, L   #元のやつ
        decoder_input = masked_output.squeeze()
        # print_name_type_shape('decoder_input',decoder_input)
        decoder_output = torch.empty(self.channel, wave_length).to("cuda")
        for idx, input in enumerate(decoder_input):
            # print_name_type_shape('input',input)
            output = self.decoder(input)  # B*C, 1, L
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
        # print_name_type_shape('decoder_output',decoder_output)

        # print('decoder\n')

        # print('end forward\n')
        return decoder_output


class type_D_2(nn.Module):
    def __init__(self, encoder_dim=512, feature_dim=128, sampling_rate=16000, win=2, layer=8, stack=3,
                 kernel=3, num_speeker=1, causal=False, num_mic=1):  # num_speeker=1もともとのやつ
        super(type_D_2, self).__init__()

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

        # TCN separator
        self.TCN = models.TCN_D_2(input_dim=self.encoder_dim,  # 入力データの次元数
                                  output_dim=self.encoder_dim * self.num_speeker,  # 出力データの次元数
                                  BN_dim=self.feature_dim,  # ボトルネック層の出力次元数
                                  hidden_dim=self.feature_dim * 4,  # 隠れ層の出力次元数
                                  layer=self.layer,  # layer個の1-DConvブロックをつなげる
                                  stack=self.stack,  # stack回繰り返す
                                  kernel=self.kernel,  # 1-DConvのカーネルサイズ
                                  causal=self.causal,
                                  num_mic=num_mic)
        self.receptive_field = self.TCN.receptive_field

        # output decoder
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
        # print('\nstart forward')

        # print(f'type(input):{type(input)}')
        # print(f'input.shape:{input.shape}') #input.shape[1,チャンネル数,音声長]
        # print_name_type_shape('input',input)
        wave_length = input.size(2)
        """ padding """
        input, rest = self.patting_signal(input)
        # print(f'type(input):{type(input)}')
        # print(f'input.shape:{input.shape}')
        # print_name_type_shape('input',input)
        batch_size = input.size(0)
        # print(f'batch_size:{batch_size}')

        """ encoder """
        # print('\nencoder')
        dim_length = self.get_dim_length(input) - 1
        encoder_output = torch.empty(self.num_mic, self.encoder_dim, dim_length).to("cuda")
        for idx, input in enumerate(input[0]):
            input = input.unsqueeze(0)  # 次元の追加 [128000]->[1,128000]
            # print_name_type_shape('input:1', input)
            input = self.encoder(input)  # エンコーダに通す
            # input=input.unsqueeze(0)
            # print_name_type_shape(f'for_input[{idx}]',input)
            encoder_output[idx] = input
        # encoder_output = self.encoder(input)  # B, N, L   # 元のやつ encoder_outputの形状が違う
        # print(f'type(encoder_output):{type(encoder_output)}')
        # print(f'encoder_output.shape:{encoder_output.shape}')
        # print_name_type_shape('encoder_output',encoder_output)
        # print('encoder\n')

        """ generate masks (separation) """
        # print('\nmask')
        # TVN_output=
        masks = torch.sigmoid(
            self.TCN(encoder_output))  # .view(batch_size, self.num_speeker, self.encoder_dim, -1)  # B, C, N, L
        # print(f'type(masks):{type(masks)}')
        # print(f'masks.shape:{masks.shape}')
        # print_name_type_shape('masks',masks)
        # print_name_type_shape('encoder_output.unsqueeze(1)',encoder_output.unsqueeze(1))
        encoder_output = encoder_output * masks  # B, C, N, L
        # print(f'type(encoder_output):{type(encoder_output)}')
        # print(f'encoder_output.shape:{encoder_output.shape}')
        # print_name_type_shape('encoder_output',encoder_output)
        # print('mask\n')

        """ decoder """
        # print('\ndecoder')
        # decoder_output = self.decoder(encoder_output.view(batch_size * self.num_speeker, self.encoder_dim, -1))  # B*C, 1, L   #元のやつ
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
        # print_name_type_shape('decoder_output',decoder_output)

        # print('decoder\n')

        # print('end forward\n')
        return decoder_output

class type_D_2_single_out(nn.Module):
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
        super(type_D_2_single_out, self).__init__()

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

        # TCN separator
        self.TCN = models.TCN_D_2(input_dim=self.encoder_dim,  # 入力データの次元数
                                  output_dim=self.encoder_dim * self.num_speeker,  # 出力データの次元数
                                  BN_dim=self.feature_dim,  # ボトルネック層の出力次元数
                                  hidden_dim=self.feature_dim * 4,  # 隠れ層の出力次元数
                                  layer=self.layer,  # layer個の1-DConvブロックをつなげる
                                  stack=self.stack,  # stack回繰り返す
                                  kernel=self.kernel,  # 1-DConvのカーネルサイズ
                                  causal=self.causal,
                                  num_mic=num_mic)
        self.receptive_field = self.TCN.receptive_field

        # output decoder
        self.decoder = nn.ConvTranspose2d(in_channels=num_mic,  # 入力次元数
                                          out_channels=1,  # 出力次元数 1もともとのやつ
                                          kernel_size=(self.encoder_dim, self.win),  # カーネルサイズ
                                          bias=False,
                                          stride=(1, self.stride))  # 畳み込み処理の移動幅
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
        # print('\nstart forward')

        # print(f'type(input):{type(input)}')
        # print(f'input.shape:{input.shape}') #input.shape[1,チャンネル数,音声長]
        # print_name_type_shape('input',input)
        wave_length = input.size(2)
        """ padding """
        input, rest = self.patting_signal(input)
        # print(f'type(input):{type(input)}')
        # print(f'input.shape:{input.shape}')
        # print_name_type_shape('input',input)
        batch_size = input.size(0)
        # print(f'batch_size:{batch_size}')

        """ encoder """
        # print('\nencoder')
        dim_length = self.get_dim_length(input) - 1
        encoder_output = torch.empty(self.num_mic, self.encoder_dim, dim_length).to("cuda")
        for idx, input in enumerate(input[0]):
            input = input.unsqueeze(0)  # 次元の追加 [128000]->[1,128000]
            # print_name_type_shape('input:1', input)
            input = self.encoder(input)  # エンコーダに通す
            # input=input.unsqueeze(0)
            # print_name_type_shape(f'for_input[{idx}]',input)
            encoder_output[idx] = input
        # encoder_output = self.encoder(input)  # B, N, L   # 元のやつ encoder_outputの形状が違う
        # print(f'type(encoder_output):{type(encoder_output)}')
        # print(f'encoder_output.shape:{encoder_output.shape}')
        # print_name_type_shape('encoder_output',encoder_output)
        # print('encoder\n')

        """ generate masks (separation) """
        # print('\nmask')
        # TVN_output=
        masks = torch.sigmoid(self.TCN(encoder_output))  # .view(batch_size, self.num_speeker, self.encoder_dim, -1)  # B, C, N, L
        # print(f'type(masks):{type(masks)}')
        # print(f'masks.shape:{masks.shape}')
        # print_name_type_shape('masks',masks)
        # print_name_type_shape('encoder_output.unsqueeze(1)',encoder_output.unsqueeze(1))
        encoder_output = encoder_output * masks  # B, C, N, L
        # print(f'type(encoder_output):{type(encoder_output)}')
        # print(f'encoder_output.shape:{encoder_output.shape}')
        # print_name_type_shape('encoder_output',encoder_output)
        # print('mask\n')

        """ decoder """
        # print('\ndecoder')
        # decoder_output = self.decoder(encoder_output.view(batch_size * self.num_speeker, self.encoder_dim, -1))  # B*C, 1, L   #元のやつ
        # encoder_output = encoder_output.squeeze()
        print(f"encoder_output:{encoder_output.shape}")
        decoder_output = self.decoder(encoder_output)
        return decoder_output


class single_to_maulti_4(nn.Module):
    def __init__(self, encoder_dim=512, feature_dim=128, sampling_rate=16000, win=2, layer=8, stack=3,
                 kernel=3, num_speeker=1, causal=False, num_mic=1):  # num_speeker=1もともとのやつ
        super(single_to_maulti_4, self).__init__()

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
        self.encoder = nn.Conv2d(in_channels=1,
                                 out_channels=4,
                                 kernel_size=(1, self.win),
                                 stride=(1, self.stride),
                                 padding=(1, 0))
        # self.encoder = nn.Conv1d(in_channels=1,  # 入力データの次元数 #=1もともとのやつ
        #                          out_channels=self.encoder_dim,  # 出力データの次元数
        #                          kernel_size=self.win,  # 畳み込みのサイズ(波形領域なので窓長のイメージ?)
        #                          bias=False,  # バイアスの有無(出力に学習可能なバイアスの追加)
        #                          stride=self.stride)  # 畳み込み処理の移動幅

        # TCN separator
        self.TCN = models.TCN_D_2(input_dim=self.encoder_dim,  # 入力データの次元数
                                  output_dim=self.encoder_dim * self.num_speeker,  # 出力データの次元数
                                  BN_dim=self.feature_dim,  # ボトルネック層の出力次元数
                                  hidden_dim=self.feature_dim * 4,  # 隠れ層の出力次元数
                                  layer=self.layer,  # layer個の1-DConvブロックをつなげる
                                  stack=self.stack,  # stack回繰り返す
                                  kernel=self.kernel,  # 1-DConvのカーネルサイズ
                                  causal=self.causal,
                                  num_mic=num_mic)
        self.receptive_field = self.TCN.receptive_field

        # output decoder
        self.decoder = nn.ConvTranspose1d(in_channels=self.encoder_dim,  # 入力次元数
                                          out_channels=1,  # 出力次元数 1もともとのやつ
                                          kernel_size=self.win,  # カーネルサイズ
                                          bias=False,
                                          stride=self.stride)  # 畳み込み処理の移動幅

    def patting_signal(self, input):
        """入力データをパティング→畳み込み前の次元数と畳み込み後の次元数を同じにするために入力データを0で囲む操作

        :param input: 入力信号 tensor型[1,チャンネル数,音声長]
        :return:
        input: パティング後のデータ
        rest:
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

        pad_aux = Variable(torch.zeros(input.size(0), 1, self.stride)).type(input.type())   # torch.zero(input.size(0), 入力のマイク数, ストライド)
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
        # print('\nstart forward')

        # print(f'type(input):{type(input)}')
        # print(f'input.shape:{input.shape}') #input.shape[1,チャンネル数,音声長]
        wave_length = input.size(-1)
        if input.dim() < 3:
            input = torch.unsqueeze(input, dim=0)
        print(f'input:{input.shape}')
        """ padding """
        input, rest = self.patting_signal(input)
        # print(f'type(input):{type(input)}')
        # print(f'input.shape:{input.shape}')
        batch_size = input.size(0)
        # print(f'batch_size:{batch_size}')

        """ encoder """
        print('\nencoder')
        dim_length = self.get_dim_length(input) - 1
        print(f'input:{input.shape}')
        # encoder_output = torch.empty(self.num_mic, self.encoder_dim, dim_length).to("cuda")
        # for idx, input in enumerate(input[0]):
        #     input = input.unsqueeze(0)  # 次元の追加 [128000]->[1,128000]
        #     input = self.encoder(input)  # エンコーダに通す
        #     # input=input.unsqueeze(0)
        #     encoder_output[idx] = input
        encoder_output = self.encoder(input)
        print(f'encoder_output:{encoder_output.shape}')
        # print(f'type(encoder_output):{type(encoder_output)}')
        # print(f'encoder_output.shape:{encoder_output.shape}')
        print('encoder\n')

        """ generate masks (separation) """
        # print('\nmask')
        # TVN_output=
        masks = torch.sigmoid(
            self.TCN(encoder_output))  # .view(batch_size, self.num_speeker, self.encoder_dim, -1)  # B, C, N, L
        # print(f'type(masks):{type(masks)}')
        # print(f'masks.shape:{masks.shape}')
        # print_name_type_shape('masks',masks)
        # print_name_type_shape('encoder_output.unsqueeze(1)',encoder_output.unsqueeze(1))
        encoder_output = encoder_output * masks  # B, C, N, L
        # print(f'type(encoder_output):{type(encoder_output)}')
        # print(f'encoder_output.shape:{encoder_output.shape}')
        # print_name_type_shape('encoder_output',encoder_output)
        # print('mask\n')

        """ decoder """
        # print('\ndecoder')
        # decoder_output = self.decoder(encoder_output.view(batch_size * self.num_speeker, self.encoder_dim, -1))  # B*C, 1, L   #元のやつ
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
        # print_name_type_shape('decoder_output',decoder_output)

        # print('decoder\n')

        # print('end forward\n')
        return decoder_output

class type_D_2_2stage(nn.Module):
    def __init__(self, encoder_dim=512, feature_dim=128, sampling_rate=16000, win=2, layer=8, stack=3,
                 kernel=3, num_speeker=1, causal=False, num_mic=1):  # num_speeker=1もともとのやつ
        super(type_D_2_2stage, self).__init__()

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

        # TCN separator
        self.TCN = models.TCN_D_2(input_dim=self.encoder_dim,  # 入力データの次元数
                                  output_dim=self.encoder_dim * self.num_speeker,  # 出力データの次元数
                                  BN_dim=self.feature_dim,  # ボトルネック層の出力次元数
                                  hidden_dim=self.feature_dim * 4,  # 隠れ層の出力次元数
                                  layer=self.layer,  # layer個の1-DConvブロックをつなげる
                                  stack=self.stack,  # stack回繰り返す
                                  kernel=self.kernel,  # 1-DConvのカーネルサイズ
                                  causal=self.causal,
                                  num_mic=num_mic)
        self.receptive_field = self.TCN.receptive_field

        # output decoder
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
        # print('\nstart forward')

        # print(f'type(input):{type(input)}')
        # print(f'input.shape:{input.shape}') #input.shape[1,チャンネル数,音声長]
        # print_name_type_shape('input',input)
        wave_length = input.size(2)
        """ padding """
        input, rest = self.patting_signal(input)
        # print(f'type(input):{type(input)}')
        # print(f'input.shape:{input.shape}')
        # print_name_type_shape('input',input)
        batch_size = input.size(0)
        # print(f'batch_size:{batch_size}')

        """ encoder """
        # print('\nencoder')
        dim_length = self.get_dim_length(input) - 1
        encoder_output = torch.empty(self.num_mic, self.encoder_dim, dim_length).to("cuda")
        for idx, input in enumerate(input[0]):
            input = input.unsqueeze(0)  # 次元の追加 [128000]->[1,128000]
            # print_name_type_shape('input:1', input)
            input = self.encoder(input)  # エンコーダに通す
            # input=input.unsqueeze(0)
            # print_name_type_shape(f'for_input[{idx}]',input)
            encoder_output[idx] = input
        # encoder_output = self.encoder(input)  # B, N, L   # 元のやつ encoder_outputの形状が違う
        # print(f'type(encoder_output):{type(encoder_output)}')
        # print(f'encoder_output.shape:{encoder_output.shape}')
        # print_name_type_shape('encoder_output',encoder_output)
        # print('encoder\n')

        """ generate masks (separation) """
        # print('\nmask')
        # TVN_output=
        masks = torch.sigmoid(self.TCN(encoder_output))  # .view(batch_size, self.num_speeker, self.encoder_dim, -1)  # B, C, N, L
        # print(f'type(masks):{type(masks)}')
        # print(f'masks.shape:{masks.shape}')
        # print_name_type_shape('masks',masks)
        # print_name_type_shape('encoder_output.unsqueeze(1)',encoder_output.unsqueeze(1))
        denoise_signal = encoder_output * masks  # B, C, N, L
        # print(f'type(encoder_output):{type(encoder_output)}')
        # print(f'encoder_output.shape:{encoder_output.shape}')
        # print_name_type_shape('encoder_output',encoder_output)
        # print('mask\n')
        masks = torch.sigmoid(self.TCN(denoise_signal))
        estimation_signal = denoise_signal * masks  # B, C, N, L

        """ decoder """
        # print('\ndecoder')
        # decoder_output = self.decoder(encoder_output.view(batch_size * self.num_speeker, self.encoder_dim, -1))  # B*C, 1, L   #元のやつ
        denoise_signal = denoise_signal.squeeze()
        estimation_signal = estimation_signal.squeeze()
        # print_name_type_shape('encoder_output',encoder_output)
        denoise_output = torch.empty(self.num_mic, wave_length).to("cuda")
        estimation_output = torch.empty(self.num_mic, wave_length).to("cuda")
        for idx, (denoise, estimation) in enumerate(zip(denoise_signal, estimation_signal)):
            denoise = self.decoder(denoise)  # B*C, 1, L
            denoise = denoise[:, self.stride:-(rest + self.stride)].contiguous()  # B*C, 1, L
            denoise = denoise.view(batch_size, self.num_speeker, -1)  # B, C, T
            denoise_output[idx] = denoise
            estimation = self.decoder(estimation)  # B*C, 1, L
            estimation = estimation[:, self.stride:-(rest + self.stride)].contiguous()  # B*C, 1, L
            estimation = estimation.view(batch_size, self.num_speeker, -1)  # B, C, T
            estimation_output[idx] = estimation
            # decoder_output = decoder_output.view(batch_size, self.num_speeker, -1)  # B, C, T
            # print_name_type_shape(f'2:decoder_output', decoder_output)
        # print_name_type_shape('decoder_output',decoder_output)
        decoder_output = denoise_output.unsqueeze(dim=0)
        estimation_output = estimation_output.unsqueeze(dim=0)
        # print_name_type_shape('decoder_output',decoder_output)

        # print('decoder\n')

        # print('end forward\n')
        return decoder_output, estimation_output

class single_to_multi(nn.Module):
    def __init__(self, encoder_dim=512, feature_dim=128, sampling_rate=16000, win=2, layer=8, stack=3,
                 kernel=3, num_speeker=1, causal=False, num_mic=1):  # num_speeker=1もともとのやつ
        """ 1ch入力で異なるN個(任意のマイク数)のエンコーダーで多chの拡張を行う

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
        super(single_to_multi, self).__init__()

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
        self.encoder = nn.ModuleList([])
        for mic in range(self.num_mic):
            self.encoder.append(nn.Conv1d(in_channels=1,
                                          out_channels=self.encoder_dim,
                                          kernel_size=self.win,
                                          bias=False,
                                          stride=self.stride))
        # self.encoder = nn.Conv1d(in_channels=1,  # 入力データの次元数 #=1もともとのやつ
        #                          out_channels=self.encoder_dim,  # 出力データの次元数
        #                          kernel_size=self.win,  # 畳み込みのサイズ(波形領域なので窓長のイメージ?)
        #                          bias=False,  # バイアスの有無(出力に学習可能なバイアスの追加)
        #                          stride=self.stride)  # 畳み込み処理の移動幅

        # TCN separator
        self.TCN = models.TCN_D_2(input_dim=self.encoder_dim,  # 入力データの次元数
                                  output_dim=self.encoder_dim * self.num_speeker,  # 出力データの次元数
                                  BN_dim=self.feature_dim,  # ボトルネック層の出力次元数
                                  hidden_dim=self.feature_dim * 4,  # 隠れ層の出力次元数
                                  layer=self.layer,  # layer個の1-DConvブロックをつなげる
                                  stack=self.stack,  # stack回繰り返す
                                  kernel=self.kernel,  # 1-DConvのカーネルサイズ
                                  causal=self.causal,
                                  num_mic=num_mic)
        self.receptive_field = self.TCN.receptive_field

        # output decoder
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

        pad_aux = Variable(torch.zeros(input.size(0), 1, self.stride)).type(input.type())
        # print(f'pad_aux.size():{pad_aux.size()}')
        # print(f"input.shape:{input.shape}")
        # print("pad_aux shape:", pad_aux.shape)
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
        # print('\nstart forward')

        # print(f'input.shape:{input.shape}') #input.shape[1,チャンネル数,音声長]
        wave_length = input.size(-1)
        """ padding """
        input, rest = self.patting_signal(input)
        # print(f'type(input):{type(input)}')
        # print(f'input.shape:{input.shape}')
        # print_name_type_shape('input',input)
        batch_size = input.size(0)
        # print(f'batch_size:{batch_size}')

        """ encoder """
        # print('\nencoder')
        dim_length = self.get_dim_length(input) - 1
        encoder_output = torch.empty(self.num_mic, self.encoder_dim, dim_length).to("cuda")
        for mic in range(self.num_mic):
            # input = input.unsqueeze(0)  # 次元の追加 [128000]->[1,128000]
            # print_name_type_shape('input:1', input)
            encoder_output[mic] = self.encoder[mic](input)  # エンコーダに通す
            # input=input.unsqueeze(0)
            # print_name_type_shape(f'for_input[{idx}]',input)
            # encoder_output[mic] = input
        # encoder_output = self.encoder(input)  # B, N, L   # 元のやつ encoder_outputの形状が違う
        # print(f'type(encoder_output):{type(encoder_output)}')
        # print(f'encoder_output.shape:{encoder_output.shape}')
        # print_name_type_shape('encoder_output',encoder_output)
        # print('encoder\n')

        """ generate masks (separation) """
        # print('\nmask')
        # TVN_output=
        masks = torch.sigmoid(
            self.TCN(encoder_output))  # .view(batch_size, self.num_speeker, self.encoder_dim, -1)  # B, C, N, L
        # print(f'type(masks):{type(masks)}')
        # print(f'masks.shape:{masks.shape}')
        # print_name_type_shape('masks',masks)
        # print_name_type_shape('encoder_output.unsqueeze(1)',encoder_output.unsqueeze(1))
        encoder_output = encoder_output * masks  # B, C, N, L
        # print(f'type(encoder_output):{type(encoder_output)}')
        # print(f'encoder_output.shape:{encoder_output.shape}')
        # print_name_type_shape('encoder_output',encoder_output)
        # print('mask\n')

        """ decoder """
        # print('\ndecoder')
        # decoder_output = self.decoder(encoder_output.view(batch_size * self.num_speeker, self.encoder_dim, -1))  # B*C, 1, L   #元のやつ
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
        # print_name_type_shape('decoder_output',decoder_output)

        # print('decoder\n')

        # print('end forward\n')
        return decoder_output
