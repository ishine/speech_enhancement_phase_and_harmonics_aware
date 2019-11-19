import torch.nn as nn
import torch
from config import *


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # TODO 此处的in_channels应该为Ca，即2还需要修改
        # (N x C x H x W)
        self.conv_2d = nn.Conv2d(in_channels=2, out_channels=Cr, kernel_size=(1, 1))
        # (N x C x L)
        self.conv_1d = nn.Conv1d(in_channels=1005, out_channels=402, kernel_size=9, padding=4)
        # bn
        self.bn_2d = nn.BatchNorm2d(num_features=Cr)
        self.bn_1d = nn.BatchNorm1d(num_features=402)
        # activation function
        self.relu_2d = nn.ReLU()
        self.relu_1d = nn.ReLU()

    def forward(self, input):
        # 8, 2, 201, 1024
        # input:(N x C_A x F x T)
        conv_2d_output = self.relu_2d(self.bn_2d(self.conv_2d(input)))
        # conv_2d_output:(N, T, C_A, F)
        shape = conv_2d_output.shape
        temp = conv_2d_output.reshape(shape[0], -1, shape[3])
        conv_1d_output = self.relu_1d(self.bn_1d(self.conv_1d(temp)))
        output = conv_1d_output.reshape(input.shape[0], input.shape[1], input.shape[2], -1)
        return output


class FTB(nn.Module):
    def __init__(self):
        super(FTB, self).__init__()
        self.attention = Attention()
        self.linear = nn.Linear(bias=False, in_features=201, out_features=201)
        # in_channels:2C_a、out_channels:C_a
        self.conv_2d = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(1, 1))
        # 2C_a
        self.bn_2d = nn.BatchNorm2d(num_features=2)
        self.relu = nn.ReLU()

    def forward(self, input):
        attention_output = self.attention(input)    # input:(N x C_A x F x T)
        # 点对点相乘
        s_a = attention_output.mul(input)   # input:(N x C_A x F x T)
        # B和T捏起来，Freq-FC
        s_a = s_a.permute(0, 3, 1, 2)   # s_a:(N, C_A, T, F)
        s_a_shape = s_a.shape
        s_a = s_a.reshape(-1, s_a.shape[2], s_a.shape[3])
        s_tr = self.linear(s_a)
        # B和T分开
        s_tr = s_tr.reshape(s_a_shape[0], s_a_shape[1], s_a_shape[2], s_a_shape[3])
        s_tr = s_tr.permute(0, 2, 3, 1)
        concat_out = torch.cat((s_tr, input), 1)
        output = self.relu(self.bn_2d(self.conv_2d(concat_out)))
        return output


class TSB(nn.Module):
    def __init__(self):
        super(TSB, self).__init__()
        # A
        self.A_ftb_in = FTB()
        self.A_conv1 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(5, 5), padding=(2, 2))
        self.A_conv2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(25, 1))
        self.A_conv3 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(5, 5))
        self.A_ftb_out = FTB()
        self.A_bn1 = nn.BatchNorm2d(num_features=2)
        self.A_bn2 = nn.BatchNorm2d(num_features=2)
        self.A_bn3 = nn.BatchNorm2d(num_features=2)
        self.A_relu1 = nn.ReLU()
        self.A_relu2 = nn.ReLU()
        self.A_relu3 = nn.ReLU()
        # P no activation function is used
        # self.P_conv1 = nn.Conv2d(kernel_size=(5, 3))
        # self.P_conv2 = nn.Conv2d(kernel_size=(25, 1))
        # self.P_ln1 = nn.LayerNorm()
        # self.P_ln2 = nn.LayerNorm()

    def forward(self, input):
        input_shape = input.shape   # N,C,T,F
        ftb_in_out = self.A_ftb_in(input)
        # TODO 需要padding的
        A_conv1_out = self.A_relu1(self.A_bn1(self.A_conv1(ftb_in_out)))
        A_conv2_out = self.A_relu2(self.A_bn2(self.A_conv2(A_conv1_out)))
        A_conv3_out = self.A_relu3(self.A_bn3(self.A_conv3(A_conv2_out)))
        print(A_conv3_out.shape)






class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, input):
        pass