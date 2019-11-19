import torch.nn as nn
import torch
from a_net.net import Attention, FTB, TSB

if __name__ == '__main__':
    # (batch, channel, T, F)
    input = torch.randn([8, 2, 201, 1024])
    # attention = Attention()
    # output = attention(input)
    tsb = TSB()
    output = tsb(input)
    print(output)
