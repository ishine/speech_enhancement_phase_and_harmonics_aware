import torch.nn as nn
import torch
from config import *

# TODO 修改为2d的gLN
class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # (N x C x H x W)
        self.conv_2d = nn.Conv2d(in_channels=C_A, out_channels=Cr, kernel_size=(1, 1))
        # (N x C x L)
        self.conv_1d = nn.Conv1d(in_channels=1005, out_channels=201, kernel_size=9, padding=4)
        # bn
        self.bn_2d = nn.BatchNorm2d(num_features=Cr)
        self.bn_1d = nn.BatchNorm1d(num_features=201)
        # activation function
        self.relu_2d = nn.ReLU()
        self.relu_1d = nn.ReLU()

    def forward(self, input):
        # 8, 96, 201, 1024
        # input:(N x C_A x F x T)
        conv_2d_output = self.relu_2d(self.bn_2d(self.conv_2d(input)))
        # conv_2d_output:(N, T, C_A, F)
        shape = conv_2d_output.shape
        temp = conv_2d_output.reshape(shape[0], -1, shape[3])
        conv_1d_output = self.relu_1d(self.bn_1d(self.conv_1d(temp)))
        output = conv_1d_output.reshape(input.shape[0], input.shape[2], -1)
        return output


class FTB(nn.Module):
    def __init__(self):
        super(FTB, self).__init__()
        self.attention = Attention()
        self.linear = nn.Linear(bias=False, in_features=201, out_features=201)
        # in_channels:2C_a、out_channels:C_a
        self.conv_2d = nn.Conv2d(in_channels=C_A * 2, out_channels=C_A, kernel_size=(1, 1))
        # 2C_a
        self.bn_2d = nn.BatchNorm2d(num_features=C_A)
        self.relu = nn.ReLU()

    def forward(self, input):
        attention_output = self.attention(input)    # input:(N x C_A x F x T)
        # 点对点相乘
        s_a = attention_output.unsqueeze(1) * input   # input:(N x C_A x F x T)
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


def f(x1, x2):
    conv = nn.Conv2d(in_channels=x2.shape[1], out_channels=x1.shape[1], kernel_size=(1, 1))
    conv = conv.cuda(CUDA_ID[0])
    # conv = conv.cuda('cuda:1')
    conv_out = conv(x2)
    return x1 * torch.tanh(conv_out)


class TSB(nn.Module):
    def __init__(self):
        super(TSB, self).__init__()
        # A
        self.A_ftb_in = FTB()
        self.A_conv1 = nn.Conv2d(in_channels=C_A, out_channels=C_A, kernel_size=(5, 5), padding=(2, 2))
        self.A_conv2 = nn.Conv2d(in_channels=C_A, out_channels=C_A, kernel_size=(25, 1), padding=(12, 0))
        self.A_conv3 = nn.Conv2d(in_channels=C_A, out_channels=C_A, kernel_size=(5, 5), padding=(2, 2))
        self.A_ftb_out = FTB()
        self.A_bn1 = nn.BatchNorm2d(num_features=C_A)
        self.A_bn2 = nn.BatchNorm2d(num_features=C_A)
        self.A_bn3 = nn.BatchNorm2d(num_features=C_A)
        self.A_relu1 = nn.ReLU()
        self.A_relu2 = nn.ReLU()
        self.A_relu3 = nn.ReLU()
        # P no activation function is used
        self.P_conv1 = nn.Conv2d(in_channels=C_P, out_channels=C_P, kernel_size=(5, 3), padding=(2, 1))
        self.P_conv2 = nn.Conv2d(in_channels=C_P, out_channels=C_P, kernel_size=(25, 1), padding=(12, 0))
        self.P_gln1 = nn.BatchNorm2d(num_features=C_P)
        self.P_gln2 = nn.BatchNorm2d(num_features=C_P)
        # self.P_ln1 = nn.LayerNorm(normalized_shape=[2, 201, 1024])
        # self.P_ln2 = nn.LayerNorm(normalized_shape=[2, 201, 1024])

    def forward(self, input):
        A_input = input[0]
        P_input = input[1]
        # input_shape = input.shape   # N,C,F,T
        # A
        ftb_in_out = self.A_ftb_in(A_input)
        A_conv1_out = self.A_relu1(self.A_bn1(self.A_conv1(ftb_in_out)))
        A_conv2_out = self.A_relu2(self.A_bn2(self.A_conv2(A_conv1_out)))
        A_conv3_out = self.A_relu3(self.A_bn3(self.A_conv3(A_conv2_out)))
        ftb_out_out = self.A_ftb_out(A_conv3_out)

        # P
        P_conv1_out = self.P_gln1(self.P_conv1(P_input))
        P_conv2_out = self.P_gln2(self.P_conv2(P_conv1_out))
        # GLN
        A_out = f(ftb_out_out, P_conv2_out)
        P_out = f(P_conv2_out, ftb_out_out)
        return [A_out, P_out]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # A stream
        self.A_conv1 = nn.Conv2d(in_channels=2, out_channels=C_A_half, kernel_size=(1, 7), padding=(0, 3))
        self.A_conv2 = nn.Conv2d(in_channels=C_A_half, out_channels=C_A, kernel_size=(7, 1), padding=(3, 0))
        self.A_relu1 = nn.ReLU()
        self.A_relu2 = nn.ReLU()
        # P steam
        self.P_conv1 = nn.Conv2d(in_channels=2, out_channels=C_P_half, kernel_size=(5, 3), padding=(2, 1))
        self.P_conv2 = nn.Conv2d(in_channels=C_P_half, out_channels=C_P, kernel_size=(25, 1), padding=(12, 0))
        # GLN 用norm2d代替
        self.P_gln1 = nn.BatchNorm2d(num_features=C_P_half)
        self.P_gln2 = nn.BatchNorm2d(num_features=C_P)
        self.P_relu1 = nn.ReLU()
        self.P_relu2 = nn.ReLU()
        # TSB
        self.TSB1 = TSB()
        self.TSB2 = TSB()
        self.TSB3 = TSB()
        # TSB之后的卷积层
        # Note:文章中写，S^3出来之后被conv2d（1，1）变为8
        self.A_conv3 = nn.Conv2d(in_channels=C_A, out_channels=8, kernel_size=(1, 1))
        self.A_relu3 = nn.ReLU()
        self.A_blstm = nn.LSTM(bidirectional=True, batch_first=True, input_size=1608, hidden_size=600)
        self.A_fc1 = nn.Linear(in_features=1200, out_features=600)
        self.A_fc2 = nn.Linear(in_features=600, out_features=600)
        # TODO 先用201试试
        self.A_fc3 = nn.Linear(in_features=600, out_features=201)
        self.A_relu4 = nn.ReLU()
        self.A_relu5 = nn.ReLU()
        self.A_sigmoid = nn.Sigmoid()

        self.P_conv3 = nn.Conv1d(in_channels=C_P, out_channels=2, kernel_size=(1, 1))

    def forward(self, input):
        # input (B, C_A, T, F) -> input (B, C_A, F, T)
        input = input.permute(0, 1, 3, 2)
        A_conv1_out = self.A_relu1(self.A_conv1(input))
        A_conv2_out = self.A_relu2(self.A_conv2(A_conv1_out))

        P_conv1_out = self.P_relu1(self.P_conv1(input))
        P_conv2_out = self.P_relu2(self.P_conv2(P_conv1_out))

        tsb_input = [A_conv2_out, P_conv2_out]
        TSB1_out = self.TSB1(tsb_input)
        TSB2_out = self.TSB2(TSB1_out)
        TSB3_out = self.TSB2(TSB2_out)

        A3 = TSB3_out[0]
        P3 = TSB3_out[1]

        # A
        A_conv3_out = self.A_relu3(self.A_conv3(A3))    # A_conv3_out (B, C, F, T)
        # 把channels和F捏起来
        A_conv3_out = A_conv3_out.permute(0, 3, 1, 2)
        A_conv3_out = A_conv3_out.reshape(A_conv3_out.shape[0], A_conv3_out.shape[1], -1)
        # blstm输出是前向和后向cat到一起的
        A_blstm_out, _ = self.A_blstm(A_conv3_out)
        A_fc1_out = self.A_relu4(self.A_fc1(A_blstm_out))
        A_fc2_out = self.A_relu5(self.A_fc2(A_fc1_out))
        A_fc3_out = self.A_sigmoid(self.A_fc3(A_fc2_out))

        P_conv3_out = self.P_conv3(P3)
        P_out_real = P_conv3_out[:, 0, :, :] / (P_conv3_out[:, 0, :, :] ** 2 + P_conv3_out[:, 1, :, :] ** 2).sqrt()
        P_out_img = P_conv3_out[:, 1, :, :] / (P_conv3_out[:, 0, :, :] ** 2 + P_conv3_out[:, 1, :, :] ** 2).sqrt()
        P_out = torch.stack((P_out_real, P_out_img), 3)
        P_out = P_out.permute(0, 2, 1, 3)
        # A_fc_out (B, T, F)
        # P_out (B ,T, F, C)
        return [A_fc3_out, P_out]


