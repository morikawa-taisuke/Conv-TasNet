import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from my_func import print_name_type_shape

class cLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(cLN, self).__init__()
        
        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step
        
        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)
        
        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
        
        entry_cnt = np.arange(channel, channel*(time_step+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)
        
        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T
        
        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)
        
        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())
    
def repackage_hidden(h):
    """
    Wraps hidden states in new Variables, to detach them from their history.
    """

    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

class MultiRNN(nn.Module):
    """
    Container module for multiple stacked RNN layers.
    
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state. The corresponding output should 
                    have shape (batch, seq_len, hidden_size).
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, num_layers=1, bidirectional=False):
        super(MultiRNN, self).__init__()

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers, dropout=dropout, 
                                         batch_first=True, bidirectional=bidirectional)
        
        

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_direction = int(bidirectional) + 1

    def forward(self, input):
        hidden = self.init_hidden(input.size(0))
        self.rnn.flatten_parameters()
        return self.rnn(input, hidden)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_())

class FCLayer(nn.Module):
    """
    Container module for a fully-connected layer.
    
    args:
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, input_size).
        hidden_size: int, dimension of the output. The corresponding output should 
                    have shape (batch, hidden_size).
        nonlinearity: string, the nonlinearity applied to the transformation. Default is None.
    """
    
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=None):
        super(FCLayer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.FC = nn.Linear(self.input_size, self.hidden_size, bias=bias)
        if nonlinearity:
            self.nonlinearity = getattr(F, nonlinearity)
        else:
            self.nonlinearity = None
            
        self.init_hidden()
    
    def forward(self, input):
        if self.nonlinearity is not None:
            return self.nonlinearity(self.FC(input))
        else:
            return self.FC(input)
              
    def init_hidden(self):
        initrange = 1. / np.sqrt(self.input_size * self.hidden_size)
        self.FC.weight.data.uniform_(-initrange, initrange)
        if self.bias:
            self.FC.bias.data.fill_(0)

class DepthConv1d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False):
        """
        :param input_channel: 入力次元
        :param hidden_channel: 隠れ層の出力次元
        :param kernel: カーネルサイズ
        :param padding: パッティングに必要なサイズ
        :param dilation: 膨張係数
        :param skip: スキップ接続
        :param causal:
        """
        super(DepthConv1d, self).__init__()
        self.causal = causal
        self.skip = skip
        """ Convolution 畳み込み層 (ボトルネック層??) """
        self.conv1d = nn.Conv1d(in_channels=input_channel,      # 入力の次元数
                                out_channels=hidden_channel,    # 出力の次元数
                                kernel_size=1)                  # カーネルサイズ
        """ パティングのサイズを決める """
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:   # False
            self.padding = padding
        """ D-Conv (H,L) """
        self.dconv1d = nn.Conv1d(in_channels=hidden_channel,    # 入力の次元数
                                 out_channels=hidden_channel,   # 出力の次元数
                                 kernel_size=kernel,            # カーネルサイズ
                                 dilation=dilation,
                                 groups=hidden_channel,         # 各入力チャンネルが独自のフィルターのセット(サイズ)と畳み込まれる
                                 padding=self.padding)          # パッティングの量
        """ 残差接続用の畳み込み (B,L) """
        self.res_out = nn.Conv1d(in_channels=hidden_channel,    # 入力の次元数
                                 out_channels=input_channel,    # 出力の次元数
                                 kernel_size=1)                 # カーネルサイズ
        """ 活性化関数 (非線形関数) """
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        """ Normalization　正規化 """
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:   # False
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        """ スキップ接続用の畳み込み (Sc,L) """
        if self.skip:
            self.skip_out = nn.Conv1d(in_channels=hidden_channel,   # 入力の次元数
                                      out_channels=input_channel,   # 出力の次元数
                                      kernel_size=1)                # カーネルサイズ

    def forward(self, input):
        """ 1-D Conv blockの動作手順 (論文中図1(c)参照)

        :param input: 入力データ[]
        :return:
        """
        # print('\nstart 1-Dconv forward')
        # print_name_type_shape('input',input)
        """ 1×1-conv """
        conv1d_out=self.conv1d(input)
        # print_name_type_shape('conv1d_out', conv1d_out)
        """ 活性化関数(非線形関数) """
        nonlinearity1_out=self.nonlinearity1(conv1d_out)
        """ 正規化 """
        output = self.reg1(nonlinearity1_out)
        """ D-conv """
        D1conv_out=self.dconv1d(output)
        if self.causal:
            """ 活性化関数 (非線形関数) """
            nonlinearity2_out=self.nonlinearity2(D1conv_out[:, :, :-self.padding])
            """ 正規化 """
            output = self.reg2(nonlinearity2_out)
        else:
            """ 活性化関数 (非線形関数) """
            nonlinearity2_out=self.nonlinearity2(D1conv_out)
            """ 正規化 """
            output = self.reg2(nonlinearity2_out)
        """ 残差接続 """
        residual = self.res_out(output)
        """ スキップ接続 """
        if self.skip:
            skip = self.skip_out(output)
            # print('end 1-Dconv forward\n')
            return residual, skip
        else:
            print('end 1-Dconv forward\n')
            return residual

class DepthConv2d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False):
        """
        :param input_channel: 入力次元
        :param hidden_channel: 隠れ層の出力次元
        :param kernel: カーネルサイズ
        :param padding: パッティングに必要なサイズ
        :param dilation: 膨張係数
        :param skip: スキップ接続
        :param causal:
        """
        """
        N：オートエンコーダのフィルタ数
        L：フィルターの長さ（サンプル数）
        B：ボトルネックと残余パスの1×1-convブロックのチャンネル数
        Sc：スキップ接続パスの1×1-convブロックのチャンネル数
        H：畳み込みブロックのチャンネル数
        P：畳み込みブロックのカーネルサイズ
        X：各繰り返しにおける畳み込みブロックの数
        R：繰り返し回数
        """
        super(DepthConv2d, self).__init__()

        self.causal = causal
        self.skip = skip

        """ Convolution 畳み込み層 (ボトルネック層??) (H,L) """
        self.conv2d = nn.Conv2d(in_channels=input_channel,      # 入力の次元数
                                out_channels=hidden_channel,    # 出力の次元数
                                kernel_size=1)                  # カーネルサイズ
        """ パティングのサイズを決める """
        if self.causal:
            self.padding = (kernel-1) * dilation
        else:  # False
            self.padding = padding
        """ D-Conv (H,L) """
        self.dconv2d = nn.Conv2d(in_channels=hidden_channel,    # 入力の次元数
                                 out_channels=hidden_channel,   # 出力の次元数
                                 kernel_size=(2,kernel),        # カーネルサイズ
                                 dilation=dilation,             # カーネルの間隔
                                 groups=hidden_channel,         # 各入力チャンネルが独自のフィルターのセット(サイズ)と畳み込まれる
                                 padding=self.padding)          # パッティングの量
        """ 残差接続用の畳み込み (B,L) """
        self.res_out = nn.Conv2d(in_channels=hidden_channel,    # 入力の次元数
                                 out_channels=input_channel,    # 出力の次元数
                                 kernel_size=1)                 # カーネルサイズ
        """ スキップ接続用の畳み込み (Sc,L) """
        if self.skip:
            self.skip_out = nn.Conv2d(in_channels=hidden_channel,  # 入力の次元数
                                      out_channels=input_channel,  # 出力の次元数
                                      kernel_size=1)  # カーネルサイズ
        """ 活性化関数 (非線形関数) """
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        """ Normalization　正規化 """
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:  # False
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)


    def forward(self, input):
        """ 2-D Conv blockの動作手順 (論文中図1(c)参照)

        :param input: 入力データ[]
        :return:
        """
        # print('\nstart 2-Dconv forward')
        """ 1×1-conv """
        conv2d_out = self.conv2d(input)
        """ 活性化関数(非線形関数) """
        nonlinearity1_out = self.nonlinearity1(conv2d_out)
        """ 正規化 """
        output = self.reg1(nonlinearity1_out)
        """ D-conv """
        D2conv_out = self.dconv2d(output)
        if self.causal:
            """ 活性化関数 (非線形関数) """
            nonlinearity2_out = self.nonlinearity2(D2conv_out[:, :, :-self.padding])
            """ 正規化 """
            output = self.reg2(nonlinearity2_out)
        else:
            """ 活性化関数 (非線形関数) """
            nonlinearity2_out = self.nonlinearity2(D2conv_out)
            """ 正規化 """
            output = self.reg2(nonlinearity2_out)
        """ 残差接続 """
        residual = self.res_out(output)
        """ スキップ接続 """
        if self.skip:
            skip = self.skip_out(output)
            # print('end 2-Dconv forward\n')
            return residual, skip
        else:
            # print('end 2-Dconv forward\n')
            return residual

class DepthConv1d_D(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False):
        """
        :param input_channel: 入力次元
        :param hidden_channel: 隠れ層の出力次元
        :param kernel: カーネルサイズ
        :param padding: パッティングに必要なサイズ
        :param dilation: 膨張係数
        :param skip: スキップ接続
        :param causal:
        """
        super(DepthConv1d_D, self).__init__()

        self.causal = causal
        self.skip = skip


        """ Convolution 畳み込み層 (ボトルネック層??) """
        self.conv1d = nn.Conv1d(in_channels=input_channel,  # 入力の次元数
                                out_channels=hidden_channel,  # 出力の次元数
                                kernel_size=1)  # カーネルサイズ
        """ 2次元畳み込み 4ch→1ch """
        self.conv2d = nn.Conv2d(in_channels=4,
                                out_channels=1,
                                kernel_size=(3, 1),
                                stride=(1,1),
                                padding=(1, 0))

        """ パティングのサイズを決める """
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:  # False
            self.padding = padding
        """ D-Conv (H,L) """
        self.dconv1d = nn.Conv1d(in_channels=hidden_channel,  # 入力の次元数
                                 out_channels=hidden_channel,  # 出力の次元数
                                 kernel_size=kernel,  # カーネルサイズ
                                 dilation=dilation,
                                 groups=hidden_channel,  # 各入力チャンネルが独自のフィルターのセット(サイズ)と畳み込まれる
                                 padding=self.padding)  # パッティングの量
        """ 2次元畳み込み 1ch→4ch """
        self.inversion_Conv2d = nn.Conv2d(in_channels=1,
                                          out_channels=4,
                                          kernel_size=1)
        """ 残差接続用の畳み込み (B,L) """
        self.res_out = nn.Conv1d(in_channels=hidden_channel,  # 入力の次元数
                                 out_channels=input_channel,  # 出力の次元数
                                 kernel_size=1)  # カーネルサイズ
        """ 活性化関数 (非線形関数) """
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        """ Normalization　正規化 """
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:  # False
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        """ スキップ接続用の畳み込み (Sc,L) """
        if self.skip:
            self.skip_out = nn.Conv1d(in_channels=hidden_channel,  # 入力の次元数
                                      out_channels=input_channel,  # 出力の次元数
                                      kernel_size=1)  # カーネルサイズ

    def forward(self, input):
        """ 1-D Conv blockの動作手順 (論文中図1(c)参照)

        :param input: 入力データ[]
        :return:
        """
        # print('\nstart 1-Dconv forward')
        # print_name_type_shape('input', input)
        """ 1×1-conv """
        conv1d_out = self.conv1d(input)
        # print_name_type_shape('conv1d_out', conv1d_out)

        """ 活性化関数(非線形関数) """
        nonlinearity1_out = self.nonlinearity1(conv1d_out)
        """ 正規化 """
        output = self.reg1(nonlinearity1_out)

        """ ch数の削減 4ch→1ch """
        output = torch.unsqueeze(output, 0)
        # print_name_type_shape('output.unsqueeze', output)
        output = self.conv2d(output)
        output = torch.squeeze(output, 0)
        # print_name_type_shape('conv2d_out', output)

        """ 活性化関数(非線形関数) """
        nonlinearity1_out = self.nonlinearity1(output)
        """ 正規化 """
        output = self.reg1(nonlinearity1_out)

        """ D-conv """
        D1conv_out = self.dconv1d(output)
        if self.causal:
            """ 活性化関数 (非線形関数) """
            nonlinearity2_out = self.nonlinearity2(D1conv_out[:, :, :-self.padding])
            """ 正規化 """
            output = self.reg2(nonlinearity2_out)
        else:
            """ 活性化関数 (非線形関数) """
            nonlinearity2_out = self.nonlinearity2(D1conv_out)
            """ 正規化 """
            output = self.reg2(nonlinearity2_out)
        # print_name_type_shape('D1_Conv_out',D1conv_out)

        output = self.inversion_Conv2d(output)
        # print_name_type_shape('4ch_output',output)

        """ 活性化関数(非線形関数) """
        nonlinearity1_out = self.nonlinearity1(output)
        """ 正規化 """
        output = self.reg1(nonlinearity1_out)

        """ 残差接続 """
        residual = self.res_out(output)
        """ スキップ接続 """
        if self.skip:
            skip = self.skip_out(output)
            # print('end 1-Dconv forward\n')
            return residual, skip
        else:
            # print('end 1-Dconv forward\n')
            return residual

class DepthConv1d_D_2(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False, num_mic=4):
        """
        :param input_channel: 入力次元
        :param hidden_channel: 隠れ層の出力次元
        :param kernel: カーネルサイズ
        :param padding: パッティングに必要なサイズ
        :param dilation: 膨張係数
        :param skip: スキップ接続
        :param causal:
        """
        super(DepthConv1d_D_2, self).__init__()

        self.causal = causal
        self.skip = skip


        """ Convolution 畳み込み層 (ボトルネック層??) """
        self.conv1d = nn.Conv1d(in_channels=input_channel,  # 入力の次元数
                                out_channels=hidden_channel,  # 出力の次元数
                                kernel_size=1)  # カーネルサイズ
        """ 2次元畳み込み 4ch→1ch """
        self.conv2d = nn.Conv2d(in_channels=num_mic,
                                out_channels=1,
                                kernel_size=(3, 1),
                                stride=(1, 1),
                                padding=(1, 0))

        """ パティングのサイズを決める """
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:  # False
            self.padding = padding
        """ D-Conv (H,L) """
        self.dconv1d = nn.Conv1d(in_channels=hidden_channel,  # 入力の次元数
                                 out_channels=hidden_channel,  # 出力の次元数
                                 kernel_size=kernel,  # カーネルサイズ
                                 dilation=dilation,
                                 groups=hidden_channel,  # 各入力チャンネルが独自のフィルターのセット(サイズ)と畳み込まれる
                                 padding=self.padding)  # パッティングの量
        """ 2次元畳み込み 1ch→4ch """
        self.inversion_Conv2d = nn.Conv2d(in_channels=1,
                                          out_channels=num_mic,
                                          kernel_size=1)
        """ 残差接続用の畳み込み (B,L) """
        self.res_out = nn.Conv1d(in_channels=hidden_channel,  # 入力の次元数
                                 out_channels=input_channel,  # 出力の次元数
                                 kernel_size=1)  # カーネルサイズ
        """ 活性化関数 (非線形関数) """
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        """ Normalization　正規化 """
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:  # False
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        """ スキップ接続用の畳み込み (Sc,L) """
        if self.skip:
            self.skip_out = nn.Conv1d(in_channels=hidden_channel,  # 入力の次元数
                                      out_channels=input_channel,  # 出力の次元数
                                      kernel_size=1)  # カーネルサイズ

    def forward(self, input):
        """ 1-D Conv blockの動作手順 (論文中図1(c)参照)

        :param input: 入力データ[]
        :return:
        """
        # print('\nstart 1-Dconv forward')
        # print_name_type_shape('input', input)
        """ 1×1-conv """
        output = self.conv1d(input)
        # print_name_type_shape('conv1d_out', conv1d_out)

        """ 活性化関数(非線形関数) """
        output = self.nonlinearity1(output)
        """ 正規化 """
        output = self.reg1(output)

        """ ch数の削減 4ch→1ch """
        output = torch.unsqueeze(output, 0)
        output = self.conv2d(output)
        output = torch.squeeze(output, 0)

        """ 活性化関数(非線形関数) """
        output = self.nonlinearity1(output)
        """ 正規化 """
        output = self.reg1(output)

        """ D-conv """
        output = self.dconv1d(output)
        if self.causal:
            """ 活性化関数 (非線形関数) """
            output = self.nonlinearity2(output[:, :, :-self.padding])
            """ 正規化 """
            output = self.reg2(output)
        else:
            """ 活性化関数 (非線形関数) """
            output = self.nonlinearity2(output)
            """ 正規化 """
            output = self.reg2(output)

        output = self.inversion_Conv2d(output)

        """ 活性化関数(非線形関数) """
        output = self.nonlinearity1(output)
        """ 正規化 """
        output = self.reg1(output)

        """ 残差接続 """
        residual = self.res_out(output)
        """ スキップ接続 """
        if self.skip:
            skip = self.skip_out(output)
            # print('end 1-Dconv forward\n')
            return residual, skip
        else:
            # print('end 1-Dconv forward\n')
            return residual

class DepthConv2d_E(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False, num_mic=4):
        """
        :param input_channel: 入力次元
        :param hidden_channel: 隠れ層の出力次元
        :param kernel: カーネルサイズ
        :param padding: パッティングに必要なサイズ
        :param dilation: 膨張係数
        :param skip: スキップ接続
        :param causal:
        """
        super(DepthConv2d_E, self).__init__()

        self.causal = causal
        self.skip = skip


        """ Convolution 畳み込み層 (ボトルネック層??) (128 → 512に特徴量を拡張) """
        self.conv1d = nn.Conv1d(in_channels=input_channel,  # 入力の次元数
                                out_channels=hidden_channel,  # 出力の次元数
                                kernel_size=1)  # カーネルサイズ

        """ 2次元畳み込み 4ch→1ch """
        self.conv2d = nn.Conv2d(in_channels=num_mic,
                                out_channels=1,
                                kernel_size=(num_mic-1, 1),
                                stride=(1, 1),
                                padding=(0, 0))

        """ 2次元畳み込み 1ch→4ch """
        self.inversion_Conv2d = nn.Conv2d(in_channels=1,
                                          out_channels=num_mic,
                                          kernel_size=1)


        """ パティングのサイズを決める """
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:  # False
            self.padding = padding

        """ D-Conv (2次元に変更) (H,L) ([512,8002]→[512,8002]) """
        # self.dconv1d = nn.Conv2d(in_channels=hidden_channel,    # 入力の次元数
        #                          out_channels=hidden_channel,   # 出力の次元数
        #                          kernel_size=kernel,            # カーネルサイズ
        #                          dilation=dilation,
        #                          groups=hidden_channel,         # 各入力チャンネルが独自のフィルターのセット(サイズ)と畳み込まれる
        #                          padding=self.padding)          # パッティングの量
        # self.dconv2d_1 = nn.Conv2d(in_channels=4,     # 入力の次元数
        #                            out_channels=4,    # 出力の次元数
        #                            kernel_size=1,     # カーネルサイズ
        #                            dilation=1,
        #                            padding=0)         # パッティングの量
                                # groups = hidden_channel,  # 各入力チャンネルが独自のフィルターのセット(サイズ)と畳み込まれる
        """ D-Conv (H,L) """
        self.dconv1d = nn.Conv1d(in_channels=hidden_channel,  # 入力の次元数
                                 out_channels=hidden_channel,  # 出力の次元数
                                 kernel_size=kernel,  # カーネルサイズ
                                 dilation=dilation,
                                 groups=hidden_channel,  # 各入力チャンネルが独自のフィルターのセット(サイズ)と畳み込まれる
                                 padding=self.padding)  # パッティングの量

        """ 残差接続用の畳み込み (B,L) """
        # self.res_out = nn.Conv1d(in_channels=hidden_channel,    # 入力の次元数
        #                          out_channels=input_channel,    # 出力の次元数
        #                          kernel_size=1)                 # カーネルサイズ
        """ 残差接続用の畳み込み (B,L) """
        self.res_out = nn.Conv1d(in_channels=hidden_channel,  # 入力の次元数
                                 out_channels=input_channel,  # 出力の次元数
                                 kernel_size=1)  # カーネルサイズ

        """ 活性化関数 (非線形関数) """
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        """ Normalization　正規化 """
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:  # False
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        """ スキップ接続用の畳み込み (Sc,L) """
        if self.skip:
            self.skip_out = nn.Conv1d(in_channels=hidden_channel,   # 入力の次元数
                                      out_channels=input_channel,   # 出力の次元数
                                      kernel_size=1)                # カーネルサイズ
            # self.skip_out = nn.Conv2d(in_channels=4,
            #                           out_channels=1,
            #                           stride=(4, 1),
            #                           kernel_size=(5, 1),
            #                           padding=(1, 0))

    def forward(self, input):
        """ 1-D Conv blockの動作手順 (論文中図1(c)参照)

        :param input: 入力データ[]
        :return:
        """
        # print('\nstart 1-Dconv forward')
        # print_name_type_shape('input', input)
        """ 1×1-conv """
        # print("1-D 1*1_input:", input.shape)
        output = self.conv1d(input)
        # print('1-D 1*1_output: ', output.shape)
        """ 活性化関数(非線形関数) """
        output = self.nonlinearity1(output)
        """ 正規化 """
        output = self.reg1(output)
        """ D-conv """
        # print("1-D Dconv_input:", output.shape)
        output = self.dconv1d(output)
        # print("1-D Dconv_output:", output.shape)

        """ 活性化関数 正規化 """
        if self.causal:
            """ 活性化関数 (非線形関数) """
            nonlinearity2_out = self.nonlinearity2(output[:, :, :-self.padding])
            """ 正規化 """
            output = self.reg2(nonlinearity2_out)
        else:
            """ 活性化関数 (非線形関数) """
            nonlinearity2_out = self.nonlinearity2(output)
            """ 正規化 """
            output = self.reg2(nonlinearity2_out)

        """ 4ch_1ch """
        # print("1-D conv2D_input:", output.shape)
        output = self.conv2d(output)
        # print("1-D conv2D_output:", output.shape)

        """ 1ch_4ch """
        # print("1-D inversion_input:", output.shape)
        output = self.inversion_Conv2d(output)
        # print("1-D inversion_output:", output.shape)

        """ 残差接続 """
        # print("1-D res_input:", output.shape)
        residual = self.res_out(output)
        # print("1-D res_output:", output.shape)

        """ スキップ接続 """
        if self.skip:
            # print("1-D skip_input:", output.shape)
            skip = self.skip_out(output)
            # print("1-D skip_output:", output.shape)
            # print('end 1-Dconv forward\n')
            return residual, skip
        else:
            # print('end 1-Dconv forward\n')
            return residual

class DepthConv2d_F(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False):
        """
        :param input_channel: 入力次元
        :param hidden_channel: 隠れ層の出力次元
        :param kernel: カーネルサイズ
        :param padding: パッティングに必要なサイズ
        :param dilation: 膨張係数
        :param skip: スキップ接続
        :param causal:
        """
        super(DepthConv2d_F, self).__init__()
        self.causal = causal
        self.skip = skip
        """ Convolution 畳み込み層 (ポイントワイズ) """
        self.conv1d = nn.Conv1d(in_channels=input_channel,      # 入力の次元数
                                out_channels=hidden_channel,    # 出力の次元数
                                kernel_size=1)                  # カーネルサイズ
        """ パティングのサイズを決める """
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:   # False
            self.padding = padding
        """ D-Conv (H,L) """
        self.dconv2d = nn.Conv2d(in_channels=4,  # 入力の次元数
                                 out_channels=1,  # 出力の次元数
                                 kernel_size=(4, kernel),  # カーネルサイズ
                                 dilation=(1, dilation),
                                 groups=4,  # 各入力チャンネルが独自のフィルターのセット(サイズ)と畳み込まれる
                                 padding=self.padding)          # パッティングの量
        """ 残差接続用の畳み込み (B,L) """
        self.res_out = nn.Conv2d(in_channels=1,    # 入力の次元数
                                 out_channels=4,    # 出力の次元数
                                 kernel_size=1)                 # カーネルサイズ
        """ 活性化関数 (非線形関数) """
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        """ Normalization　正規化 """
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:   # False
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        """ スキップ接続用の畳み込み (Sc,L) """
        if self.skip:
            self.skip_out = nn.Conv1d(in_channels=hidden_channel,   # 入力の次元数
                                      out_channels=input_channel,   # 出力の次元数
                                      kernel_size=1)                # カーネルサイズ

    def forward(self, input):
        """ 1-D Conv blockの動作手順 (論文中図1(c)参照)

        :param input: 入力データ[]
        :return:
        """
        # print('\nstart 1-Dconv forward')
        # print_name_type_shape('input',input)
        """ 1×1-conv """
        conv1d_out=self.conv1d(input)
        # print_name_type_shape('conv1d_out', conv1d_out)
        """ 活性化関数(非線形関数) """
        nonlinearity1_out=self.nonlinearity1(conv1d_out)
        """ 正規化 """
        output = self.reg1(nonlinearity1_out)
        """ D-conv """
        D1conv_out=self.dconv1d(output)
        if self.causal:
            """ 活性化関数 (非線形関数) """
            nonlinearity2_out=self.nonlinearity2(D1conv_out[:, :, :-self.padding])
            """ 正規化 """
            output = self.reg2(nonlinearity2_out)
        else:
            """ 活性化関数 (非線形関数) """
            nonlinearity2_out=self.nonlinearity2(D1conv_out)
            """ 正規化 """
            output = self.reg2(nonlinearity2_out)
        """ 残差接続 """
        residual = self.res_out(output)
        """ スキップ接続 """
        if self.skip:
            skip = self.skip_out(output)
            # print('end 1-Dconv forward\n')
            return residual, skip
        else:
            print('end 1-Dconv forward\n')
            return residual


class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim, layer, stack,
                 kernel=3, skip=True, causal=False, dilated=True):
        """
        :param input_dim: TCNの入力次元数
        :param output_dim: TCNの出力次元数
        :param BN_dim: ボトルネック層の出力次元数
        :param hidden_dim: 隠れ層の出力次元数
        :param layer: layer個の1-DConvブロックをつなげる
        :param stack: stack回繰り返す
        :param kernel: 1-DConvブロックのカーネルサイズ
        :param skip: スキップ接続 True
        :param causal:
        :param dilated:
        """
        super(TCN, self).__init__()
        
        # input is a sequence of features of shape (B, N, L)
        
        """ normalization 正規化 """
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)
        """ ボトルネック層 """
        self.BN = nn.Conv1d(input_dim, BN_dim, 1)
        
        # TCN for feature extraction
        """ TCN(the Temporal Convolutional Network) """
        self.receptive_field = 0    # 受容野
        self.dilated = dilated      # 拡張係数? (True)
        
        self.TCN = nn.ModuleList([])
        for s in range(stack):  # 1-DConvブロックの塊をstack回繰り返す
            for i in range(layer):  # layer個の1-DConvブロック
                """ 1-DConvブロックの追加 """
                if self.dilated:    # (True)
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2 ** i, padding=2 ** i, skip=skip, causal=causal))
                else:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal))
                """ 受容野の更新 """
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += (kernel - 1)
                    
        #print("Receptive field: {:3d} frames.".format(self.receptive_field))
        
        """ output layer 出力層(残差接続) """
        self.output = nn.Sequential( nn.PReLU(), nn.Conv1d(BN_dim, output_dim, 1) )

        """ スキップ接続 """
        self.skip = skip
        
    def forward(self, input):
        
        # input shape: (B, N, L)=(ボトルネック後の特徴次元数, 行数, サイズ1の畳み込みカーネルサイズ)
        
        """ normalization → BN """
        output = self.BN(self.LN(input))    # ボトルネック層

        # print(f'len(self.TCN):{len(self.TCN)}')
        """ TCN """
        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual
            
        """ output layer """
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)
        
        return output

class TCN2(nn.Module):
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim, layer, stack, kernel=3, skip=True, causal=False,
                 dilated=True):
        """
        :param input_dim: TCNの入力次元数
        :param output_dim: TCNの出力次元数
        :param BN_dim: ボトルネック層の出力次元数
        :param hidden_dim: 隠れ層の出力次元数
        :param layer: layer個の1-DConvブロックをつなげる
        :param stack: stack回繰り返す
        :param kernel: 1-DConvブロックのカーネルサイズ
        :param skip: スキップ接続 True
        :param causal:
        :param dilated: 畳み込むカーネルの間隔
        """
        super(TCN2, self).__init__()

        # input is a sequence of features of shape (B, N, L)

        """ normalization 正規化 """
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)

        """ ボトルネック層 """
        self.BN = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        """ TCN(the Temporal Convolutional Network) """
        self.receptive_field = 0    # 受容野
        self.dilated = dilated      # 拡張係数? (True)

        self.TCN = nn.ModuleList([])
        for s in range(stack):  # 1-DConvブロックの塊をstack回繰り返す
            for i in range(layer):  # layer個の1-DConvブロック
                """ 1-DConvブロックの追加 """
                if self.dilated:  # (True)
                    self.TCN.append(DepthConv2d(BN_dim,
                                                hidden_dim,
                                                kernel,
                                                dilation=2 ** i,
                                                padding=2 ** i,
                                                skip=skip,
                                                causal=causal))
                else:
                    self.TCN.append(DepthConv2d(BN_dim,
                                                hidden_dim,
                                                kernel,
                                                dilation=1,
                                                padding=1,
                                                skip=skip,
                                                causal=causal))
                """ 受容野の更新 """
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2 ** i
                    else:
                        self.receptive_field += (kernel - 1)

        # print("Receptive field: {:3d} frames.".format(self.receptive_field))

        """ output layer 出力層(残差接続) """
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(BN_dim, output_dim, 1))

        """ スキップ接続 """
        self.skip = skip

    def forward(self, input):
        """学習の手順

        :param input: 入力データ [特徴次元数]
        :return output: 作成したマスク[]
        """

        # input shape: (B, N, L)=(ボトルネック後の特徴次元数, 行数, サイズ1の畳み込みカーネルサイズ)

        """ Layer Normalization レイヤーの正規化 """
        LN_input = self.LN(input)
        """ Bottleneck Layer ボトルネック層 """
        output = self.BN(LN_input)  # ボトルネック層

        """ TCN """
        # pass to TCN
        if self.skip:   # True
            skip_connection = 0.    # スキップ接続の初期化
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        """ output layer """
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)

        return output

class TCN_A(nn.Module):
    """ ボトルネック層で4chの特徴量を1chに減少させる """
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim, layer, stack, kernel=3, skip=True, causal=False,
                 dilated=True):
        """
        :param input_dim: TCNの入力次元数
        :param output_dim: TCNの出力次元数
        :param BN_dim: ボトルネック層の出力次元数
        :param hidden_dim: 隠れ層の出力次元数
        :param layer: layer個の1-DConvブロックをつなげる
        :param stack: stack回繰り返す
        :param kernel: 1-DConvブロックのカーネルサイズ
        :param skip: スキップ接続 True
        :param causal:
        :param dilated:
        """
        super(TCN_A, self).__init__()
        # input is a sequence of features of shape (B, N, L)
        """ normalization 正規化 """
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)
        """ ボトルネック層 """
        # print(f'Bottle Neck')
        # print(f'input_dim:{input_dim}')
        # print(f'BN_dim:{BN_dim}')
        self.BN2d = nn.Conv2d(4, 1, 1)  # 提案手法 4chを1chに畳み込み
        self.BN1d = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        """ TCN(the Temporal Convolutional Network) """
        self.receptive_field = 0  # 受容野
        self.dilated = dilated  # 拡張係数? (True)

        self.TCN = nn.ModuleList([])
        for s in range(stack):  # 1-DConvブロックをstack回繰り返す
            for i in range(layer):  # layer個の1-DConvブロック
                """ 1-DConvブロックの追加 """
                if self.dilated:  # (True)
                    self.TCN.append(
                        DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2 ** i, padding=2 ** i, skip=skip,
                                      causal=causal))
                else:
                    self.TCN.append(
                        DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal))
                """ 受容野の更新 """
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2 ** i
                    else:
                        self.receptive_field += (kernel - 1)

        # print("Receptive field: {:3d} frames.".format(self.receptive_field))

        """ output layer 出力層(残差接続) """
        self.output = nn.Sequential(nn.PReLU(), nn.Conv1d(BN_dim, output_dim, 1))

        """ スキップ接続 """
        self.skip = skip

    def forward(self, input):

        # input shape: (B, N, L)=(ボトルネック後の特徴次元数, 行数, サイズ1の畳み込みカーネルサイズ)
        """ normalization """
        input_normalization=self.LN(input)
        # print_name_type_shape('input_normalization',input_normalization)
        # print('normalization')
        """ ボトルネック層 """
        output = self.BN2d(input_normalization)  # ボトルネック層  4chを1chに畳み込み
        # print_name_type_shape('BN2d_output',output)
        output = self.BN1d(output)  # ボトルネック層
        # print_name_type_shape('BN1d_output', output)
        # print(f'len(self.TCN):{len(self.TCN)}')

        """ TCN """
        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        """ output layer """
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)

        return output

class TCN_B(nn.Module):
    """ ボトルネック層エンコーダ部分に移動させたモデル """
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim, layer, stack, kernel=3, skip=True, causal=False,
                 dilated=True):
        """
        :param input_dim: TCNの入力次元数
        :param output_dim: TCNの出力次元数
        :param BN_dim: ボトルネック層の出力次元数
        :param hidden_dim: 隠れ層の出力次元数
        :param layer: layer個の1-DConvブロックをつなげる
        :param stack: stack回繰り返す
        :param kernel: 1-DConvブロックのカーネルサイズ
        :param skip: スキップ接続 True
        :param causal:
        :param dilated:
        """
        super(TCN_B, self).__init__()

        # input is a sequence of features of shape (B, N, L)

        """ normalization 正規化 """
        if not causal:  # causal=False → not causal==True
            # print(f'if_causal:{causal}')
            # self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)    # 元のやつ
            self.LN = nn.GroupNorm(1, BN_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)

        # TCN for feature extraction
        """ TCN(the Temporal Convolutional Network) """
        self.receptive_field = 0  # 受容野
        self.dilated = dilated  # 拡張係数? (True)

        self.TCN = nn.ModuleList([])
        for s in range(stack):  # 1-DConvブロックの塊をstack回繰り返す
            for i in range(layer):  # layer個の1-DConvブロック
                """ 1-DConvブロックの追加 """
                if self.dilated:  # (True)
                    self.TCN.append(
                        DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2 ** i, padding=2 ** i, skip=skip,
                                      causal=causal))
                else:
                    self.TCN.append(
                        DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal))
                """ 受容野の更新 """
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2 ** i
                    else:
                        self.receptive_field += (kernel - 1)

        # print("Receptive field: {:3d} frames.".format(self.receptive_field))

        """ output layer 出力層(残差接続) """
        self.output = nn.Sequential(nn.PReLU(), nn.Conv1d(BN_dim, output_dim, 1))

        """ スキップ接続 """
        self.skip = skip

    def forward(self, input):
        # input shape: (B, N, L)=(ボトルネック後の特徴次元数, 行数, サイズ1の畳み込みカーネルサイズ)
        """ normalization """
        input_normalization = self.LN(input)
        # print_name_type_shape('input_normalization', input_normalization)
        # print('normalization')
        # """ ボトルネック層 """
        # output = self.BN2d(input_normalization)  # ボトルネック層
        # print_name_type_shape('BN2d_output', output)
        # output = self.BN1d(output)  # ボトルネック層
        # print_name_type_shape('BN1d_output', output)
        # print(f'len(self.TCN):{len(self.TCN)}')
        output= input_normalization

        """ TCN """
        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        """ output layer """
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)

        return output
class encoder_B(nn.Module):
    def __init__(self, encoder_dim=512, BN_dim=128, kernel_size=32):
        super(encoder_B, self).__init__()
        """ hyper parameters """
        self.encoder_dim = encoder_dim              # エンコーダの出力次元数
        self.BN_dim = BN_dim                        # ボトルネック層の出力次元数
        self.win = kernel_size
        self.stride = self.win // 2                 # 畳み込み処理におけるフィルタが移動する幅

        """ encoder """
        self.encoder = nn.Conv1d(in_channels=1,                 # 入力データの次元数 #=1もともとのやつ
                                 out_channels=self.encoder_dim, # 出力データの次元数
                                 kernel_size=self.win,          # 畳み込みのサイズ(波形領域なので窓長のイメージ?)
                                 bias=False,                    # バイアスの有無(出力に学習可能なバイアスの追加)
                                 stride=self.stride)            # 畳み込み処理の移動幅

        """ ボトルネック層 """
        # print(f'Bottle Neck')
        # print(f'input_dim:{input_dim}')
        # print(f'BN_dim:{BN_dim}')
        self.BN2d = nn.Conv2d(4, 1, 1)
        self.BN1d = nn.Conv1d(self.encoder_dim, self.BN_dim, 1)
    def get_dim_length(self,input_patting):
        """エンコード後の特徴量領域のデータ長を計算

        :param input_patting: パティングされた入力
        :return out_length: エンコード後のデータ長
        """
        patting=0
        dilation=1
        in_length = input_patting.size(2)
        # print(f'in_length:{in_length}')
        # print(f'{in_length},{self.patting}-{self.dilation},{self.win},{self.stride}')
        out_length=((in_length+2*patting-dilation*(self.win-1)-1)/self.stride) + 1
        # print(f'out_length:{out_length}')
        return int(out_length)
    def forward(self, input):
        """ encoder """
        print('\nencoder')
        # print_name_type_shape('input',input)
        dim_length = self.get_dim_length(input)
        encoder_output = torch.empty(4, self.encoder_dim, dim_length).to("cuda")
        # print_name_type_shape('encoder_output',encoder_output)
        for idx, input in enumerate(input[0]):
            # print_name_type_shape('input:0', input)
            input = input.unsqueeze(0)  # 次元の追加 [128000]->[1,128000]
            # print_name_type_shape('input:1', input)
            input = self.encoder(input)  # エンコーダに通す
            # print_name_type_shape(f'for[{idx}]_input', input)
            encoder_output[idx] = input
        # print_name_type_shape('encoder_output', encoder_output)
        print('encoder\n')

        """ ボトルネック層 """
        print('\nBN')
        output = self.BN2d(encoder_output)              # ボトルネック層 [4,512,8002]→[1,512,8002]
        # print_name_type_shape('BN2d_output', output)
        output = self.BN1d(output)                      # ボトルネック層 [1,512,8002]→[1,128,8002]
        # print_name_type_shape('BN1d_output', output)
        print('BN\n')

        return output

class TCN_C(nn.Module):
    """ ボトルネック層で4chの特徴量を1chに減少させる """
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim, layer, stack, kernel=3, skip=True, causal=False, dilated=True):
        """
        :param input_dim: TCNの入力次元数
        :param output_dim: TCNの出力次元数
        :param BN_dim: ボトルネック層の出力次元数
        :param hidden_dim: 隠れ層の出力次元数
        :param layer: layer個の1-DConvブロックをつなげる
        :param stack: stack回繰り返す
        :param kernel: 1-DConvブロックのカーネルサイズ
        :param skip: スキップ接続 True
        :param causal:
        :param dilated:
        """
        super(TCN_C, self).__init__()

        # input is a sequence of features of shape (B, N, L)

        """ normalization 正規化 """
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)
        """ ボトルネック層 """
        # self.BN2d = nn.Conv2d(4, 1, 1)
        self.BN1d = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        """ TCN(the Temporal Convolutional Network) """
        self.receptive_field = 0    # 受容野
        self.dilated = dilated      # 拡張係数? (True)

        self.TCN = nn.ModuleList([])
        for s in range(stack):  # 1-DConvブロックの塊をstack回繰り返す
            for i in range(layer):  # layer個の1-DConvブロック
                """ 1-DConvブロックの追加 """
                if self.dilated:  # (True)
                    self.TCN.append(
                        DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2 ** i, padding=2 ** i, skip=skip, causal=causal)
                    )
                else:
                    self.TCN.append(
                        DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal)
                    )
                """ 受容野の更新 """
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2 ** i
                    else:
                        self.receptive_field += (kernel - 1)

        # print("Receptive field: {:3d} frames.".format(self.receptive_field))

        """ output layer 出力層(残差接続) """
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(4, 1, 1), nn.Conv1d(BN_dim, output_dim, 1))

        """ スキップ接続 """
        self.skip = skip

    def forward(self, input):

        # input shape: (B, N, L)=(ボトルネック後の特徴次元数, 行数, サイズ1の畳み込みカーネルサイズ)
        """ normalization """
        input_normalization = self.LN(input)
        # print_name_type_shape('input_normalization',input_normalization)
        # print('normalization')
        """ ボトルネック層 """
        # output = self.BN2d(input_normalization)     # ボトルネック層
        # print_name_type_shape('BN2d_output',output)
        output = self.BN1d(input_normalization)                  # ボトルネック層
        # print_name_type_shape('BN1d_output', output)
        # print(f'len(self.TCN):{len(self.TCN)}')

        """ TCN """
        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        """ output layer """
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)

        return output

class TCN_C_2(nn.Module):
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim, layer, stack,
                 kernel=3, skip=True, causal=False, dilated=True):
        """
        :param input_dim: TCNの入力次元数
        :param output_dim: TCNの出力次元数
        :param BN_dim: ボトルネック層の出力次元数
        :param hidden_dim: 隠れ層の出力次元数
        :param layer: layer個の1-DConvブロックをつなげる
        :param stack: stack回繰り返す
        :param kernel: 1-DConvブロックのカーネルサイズ
        :param skip: スキップ接続 True
        :param causal:
        :param dilated:
        """
        super(TCN_C_2, self).__init__()

        # input is a sequence of features of shape (B, N, L)

        """ normalization 正規化 """
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)
        """ ボトルネック層 """
        self.BN = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        """ TCN(the Temporal Convolutional Network) """
        self.receptive_field = 0  # 受容野
        self.dilated = dilated  # 拡張係数? (True)

        self.TCN = nn.ModuleList([])
        for s in range(stack):  # 1-DConvブロックの塊をstack回繰り返す
            for i in range(layer):  # layer個の1-DConvブロック
                """ 1-DConvブロックの追加 """
                if self.dilated:  # (True)
                    self.TCN.append(
                        DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2 ** i, padding=2 ** i, skip=skip, causal=causal)
                    )
                else:
                    self.TCN.append(
                        DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal))
                """ 受容野の更新 """
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2 ** i
                    else:
                        self.receptive_field += (kernel - 1)

        # print("Receptive field: {:3d} frames.".format(self.receptive_field))

        """ output layer 出力層(残差接続) """
        self.output = nn.Sequential(nn.PReLU(), nn.Conv1d(BN_dim, output_dim, 1))

        """ スキップ接続 """
        self.skip = skip

    def forward(self, input):
        # input shape: (B, N, L)=(ボトルネック後の特徴次元数, 行数, サイズ1の畳み込みカーネルサイズ)
        # print_name_type_shape('input', input)
        """ normalization """
        output = self.LN(input)
        """ BN層 ボトルネック層 """
        output = self.BN(output)  # ボトルネック層
        # print_name_type_shape('BN',output)
        # print(f'len(self.TCN):{len(self.TCN)}')
        """ TCN """
        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        """ output layer """
        if self.skip:
            # print_name_type_shape('skip_connection',skip_connection)
            output = self.output(skip_connection)
        else:
            output = self.output(output)
        # print_name_type_shape('output',output)

        return output

class TCN_D(nn.Module):
    """ ボトルネック層で4chの特徴量を1chに減少させる """
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim, layer, stack, kernel=3, skip=True, causal=False,
                 dilated=True):
        """
        :param input_dim: TCNの入力次元数
        :param output_dim: TCNの出力次元数
        :param BN_dim: ボトルネック層の出力次元数
        :param hidden_dim: 隠れ層の出力次元数
        :param layer: layer個の1-DConvブロックをつなげる
        :param stack: stack回繰り返す
        :param kernel: 1-DConvブロックのカーネルサイズ
        :param skip: スキップ接続 True
        :param causal:
        :param dilated:
        """
        super(TCN_D, self).__init__()

        # input is a sequence of features of shape (B, N, L)

        """ normalization 正規化 """
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)
        """ ボトルネック層 """
        # self.BN2d = nn.Conv2d(4, 1, 1)
        self.BN1d = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        """ TCN(the Temporal Convolutional Network) """
        self.receptive_field = 0  # 受容野
        self.dilated = dilated  # 拡張係数? (True)

        self.TCN = nn.ModuleList([])
        for s in range(stack):      # 1-DConvブロックの塊をstack回繰り返す
            for i in range(layer):  # layer個の1-DConvブロック
                """ 1-DConvブロックの追加 """
                if self.dilated:    # (True)
                    self.TCN.append(
                        DepthConv1d_D(BN_dim, hidden_dim, kernel, dilation=2 ** i, padding=2 ** i, skip=skip, causal=causal)
                    )
                else:
                    self.TCN.append(
                        DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal)
                    )
                """ 受容野の更新 """
                if i == 0 and s == 0:   # 初回
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2 ** i
                    else:
                        self.receptive_field += (kernel - 1)

        # print("Receptive field: {:3d} frames.".format(self.receptive_field))

        """ output layer 出力層(残差接続) """
        self.output = nn.Sequential(nn.PReLU(), nn.Conv1d(BN_dim, output_dim, 1))

        """ スキップ接続 """
        self.skip = skip

    def forward(self, input):
        # print('\nTCN_D')
        # input shape: (B, N, L)=(ボトルネック後の特徴次元数, 行数, サイズ1の畳み込みカーネルサイズ)
        """ normalization """
        input_normalization=self.LN(input)
        # print_name_type_shape('input_normalization',input_normalization)
        # print('normalization')
        """ ボトルネック層 """
        # output = self.BN2d(input_normalization)  # ボトルネック層
        # print_name_type_shape('BN2d_output',output)
        output = self.BN1d(input_normalization)  # ボトルネック層
        # print_name_type_shape('BN1d_output', output)
        # print(f'len(self.TCN):{len(self.TCN)}')

        """ TCN """
        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        """ output layer """
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)
        # print_name_type_shape('output',output)
        # print('TCN_D\n')
        return output

class TCN_E(nn.Module):
    """ ボトルネック層で4chの特徴量を1chに減少させる """
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim, layer, stack, kernel=3, skip=True, causal=False,
                 dilated=True, num_mic=4):
        """
        :param input_dim: TCNの入力次元数
        :param output_dim: TCNの出力次元数
        :param BN_dim: ボトルネック層の出力次元数
        :param hidden_dim: 隠れ層の出力次元数
        :param layer: layer個の1-DConvブロックをつなげる
        :param stack: stack回繰り返す
        :param kernel: 1-DConvブロックのカーネルサイズ
        :param skip: スキップ接続 True
        :param causal:
        :param dilated:
        """
        super(TCN_E, self).__init__()

        # input is a sequence of features of shape (B, N, L)

        """ normalization 正規化 """
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)

        """ ボトルネック層 1D """
        # self.BN2d = nn.Conv2d(4, 1, 1)
        self.BN1d = nn.Conv1d(input_dim, BN_dim, 1)

        """ ボトルネック層 2D """
        self.BN2d = nn.Conv2d(in_channels=num_mic, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 0))
        # self.BN1d = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        """ TCN(the Temporal Convolutional Network) """
        self.receptive_field = 0  # 受容野
        self.dilated = dilated  # 拡張係数? (True)

        self.TCN = nn.ModuleList([])
        for s in range(stack):      # 1-DConvブロックの塊をstack回繰り返す
            for i in range(layer):  # layer個の1-DConvブロック
                """ 1-DConvブロックの追加 """
                if self.dilated:    # (True)
                    self.TCN.append(
                        DepthConv2d_E(BN_dim, hidden_dim, kernel, dilation=2 ** i, padding=2 ** i, skip=skip, causal=causal, num_mic=num_mic)
                    )
                else:
                    self.TCN.append(
                        DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal)
                    )
                """ 受容野の更新 """
                if i == 0 and s == 0:   # 初回
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2 ** i
                    else:
                        self.receptive_field += (kernel - 1)

        # print("Receptive field: {:3d} frames.".format(self.receptive_field))

        """ output layer 出力層(残差接続) """
        self.output = nn.Sequential(nn.PReLU(), nn.Conv1d(BN_dim, output_dim, 1))

        """ スキップ接続 """
        self.skip = skip

    def forward(self, input):
        # print('\nTCN_D')
        # input shape: (B, N, L)=(ボトルネック後の特徴次元数, 行数, サイズ1の畳み込みカーネルサイズ)
        """ normalization """
        input_normalization=self.LN(input)
        # print_name_type_shape('input_normalization',input_normalization)
        # print('normalization')
        """ ボトルネック層 """
        # output = self.BN2d(input_normalization)  # ボトルネック層
        # print('BN_input: ',input.shape)
        output = self.BN1d(input_normalization)  # ボトルネック層
        # print('BN_output: ',output.shape)

        """ TCN """
        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        """ output layer """
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)
        # print_name_type_shape('output',output)
        # print('TCN_D\n')
        return output

class TCN_F(nn.Module):
    """ ボトルネック層で4chの特徴量を1chに減少させる """
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim, layer, stack, kernel=3, skip=True, causal=False,
                 dilated=True):
        """
        :param input_dim: TCNの入力次元数
        :param output_dim: TCNの出力次元数
        :param BN_dim: ボトルネック層の出力次元数
        :param hidden_dim: 隠れ層の出力次元数
        :param layer: layer個の1-DConvブロックをつなげる
        :param stack: stack回繰り返す
        :param kernel: 1-DConvブロックのカーネルサイズ
        :param skip: スキップ接続 True
        :param causal:
        :param dilated:
        """
        super(TCN_F, self).__init__()

        # input is a sequence of features of shape (B, N, L)

        """ normalization 正規化 """
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)
        """ ボトルネック層 """
        self.BN2d = nn.Conv2d(in_channels=4, out_channels=4, stride=(1, 1), kernel_size=(1, 1), padding=(1, 0))
        # self.BN1d = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        """ TCN(the Temporal Convolutional Network) """
        self.receptive_field = 0  # 受容野
        self.dilated = dilated  # 拡張係数? (True)

        self.TCN = nn.ModuleList([])
        for s in range(stack):      # 1-DConvブロックの塊をstack回繰り返す
            for i in range(layer):  # layer個の1-DConvブロック
                """ 1-DConvブロックの追加 """
                if self.dilated:    # (True)
                    self.TCN.append(
                        DepthConv2d_F(BN_dim, hidden_dim, kernel, dilation=2 ** i, padding=2 ** i, skip=skip, causal=causal)
                    )
                else:
                    self.TCN.append(
                        DepthConv2d_F(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal)
                    )
                """ 受容野の更新 """
                if i == 0 and s == 0:   # 初回
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2 ** i
                    else:
                        self.receptive_field += (kernel - 1)

        # print("Receptive field: {:3d} frames.".format(self.receptive_field))

        self.ch_conv = nn.Conv2d(in_channels=2,out_channels=1,stride=(2,1),kernel_size=(5,1),padding=(1,0))

        """ output layer 出力層(残差接続) """
        self.output = nn.Sequential(nn.PReLU(), nn.Conv1d(BN_dim, output_dim, 1))

        """ スキップ接続 """
        self.skip = skip

    def forward(self, input):
        # print('\nTCN_D')
        # input shape: (B, N, L)=(ボトルネック後の特徴次元数, 行数, サイズ1の畳み込みカーネルサイズ)
        """ normalization """
        input_normalization=self.LN(input)
        # print_name_type_shape('input_normalization',input_normalization)
        # print('normalization')
        """ ボトルネック層 """
        # output = self.BN2d(input_normalization)  # ボトルネック層
        # print_name_type_shape('BN2d_output',output)
        output = self.BN2d(input_normalization)  # ボトルネック層
        # print_name_type_shape('BN1d_output', output)
        # print(f'len(self.TCN):{len(self.TCN)}')

        """ TCN """
        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        """ output layer """
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)
        # print_name_type_shape('output',output)
        # print('TCN_D\n')
        return output


class TCN_D_2(nn.Module):
    """ ボトルネック層で4chの特徴量を1chに減少させる """
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim, layer, stack, kernel=3, skip=True, causal=False, dilated=True, num_mic=4):
        """
        :param input_dim: TCNの入力次元数
        :param output_dim: TCNの出力次元数
        :param BN_dim: ボトルネック層の出力次元数
        :param hidden_dim: 隠れ層の出力次元数
        :param layer: layer個の1-DConvブロックをつなげる
        :param stack: stack回繰り返す
        :param kernel: 1-DConvブロックのカーネルサイズ
        :param skip: スキップ接続 True
        :param causal:
        :param dilated:
        """
        super(TCN_D_2, self).__init__()

        # input is a sequence of features of shape (B, N, L)

        """ normalization 正規化 """
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)
        """ ボトルネック層 """
        # self.BN2d = nn.Conv2d(4, 1, 1)
        self.BN1d = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        """ TCN(the Temporal Convolutional Network) """
        self.receptive_field = 0  # 受容野
        self.dilated = dilated  # 拡張係数? (True)

        self.TCN = nn.ModuleList([])
        for s in range(stack):      # 1-DConvブロックの塊をstack回繰り返す
            for i in range(layer):  # layer個の1-DConvブロック
                """ 1-DConvブロックの追加 """
                if self.dilated:    # (True)
                    self.TCN.append(
                        DepthConv1d_D_2(BN_dim, hidden_dim, kernel, dilation=2 ** i, padding=2 ** i, skip=skip, causal=causal, num_mic=num_mic)
                    )
                else:
                    self.TCN.append(
                        DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal)
                    )
                """ 受容野の更新 """
                if i == 0 and s == 0:   # 初回
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2 ** i
                    else:
                        self.receptive_field += (kernel - 1)

        # print("Receptive field: {:3d} frames.".format(self.receptive_field))

        """ output layer 出力層(残差接続) """
        self.output = nn.Sequential(nn.PReLU(), nn.Conv1d(BN_dim, output_dim, 1))

        """ スキップ接続 """
        self.skip = skip

    def forward(self, input):
        # print('\nTCN_D')
        # input shape: (B, N, L)=(ボトルネック後の特徴次元数, 行数, サイズ1の畳み込みカーネルサイズ)
        """ normalization """
        input=self.LN(input)
        # print_name_type_shape('input',input)
        # print('normalization')
        """ ボトルネック層 """
        # output = self.BN2d(input)  # ボトルネック層
        # print_name_type_shape('BN2d_output',output)
        output = self.BN1d(input)  # ボトルネック層
        # print_name_type_shape('BN1d_output', output)
        # print(f'len(self.TCN):{len(self.TCN)}')

        """ TCN """
        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        """ output layer """
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)
        # print_name_type_shape('output',output)
        # print('TCN_D\n')
        return output
