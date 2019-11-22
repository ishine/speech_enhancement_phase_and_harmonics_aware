import torch.nn as nn
import torch
from a_net.net import Attention, FTB, TSB, Net
import scipy.io as sio
from g_utils.stft_istft import STFT
import soundfile as sf
import numpy as np

if __name__ == '__main__':
    # (batch, channel, T, F)
    input = torch.randn([8, 2, 201, 1024])
    # attention = Attention()
    # output = attention(input)
    net = Net()
    output = net(input)
    print(output)
