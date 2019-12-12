import os
import torch
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
# from utils.util import get_alpha
from g_utils.stft_istft import STFT
from torch.utils.data import DataLoader, Dataset
from config import *
from g_utils.label_set import LabelHelper
import librosa


class SpeechDataset(Dataset):

    def __getitem__(self, index):
        """
        对于每个送入网络的数据进行处理
        PS：一般只对数据进行时域上的操作，其他操作如：STFT，送入CUDA之后进行
        :param index:送入网络数据的索引，一般是文件的索引
        :return:输入数据，相应的label
        """
        # 迭代输出需要的文件
        data = loadmat(self.root_dir + self.files[index])
        # 三者都是从文件里读进来的，一堆采样点组成的元组(xxx,1)
        speech = data['speech']
        noise = data['noise']
        mix = speech + noise
        # 不做能量归一化
        # c = get_alpha(mix)
        # speech *= c
        # mix *= c
        # 计算帧长
        nframe = (len(speech) - FILTER_LENGTH) // HOP_LENGTH + 1
        # len_speech = nframe * HOP_LENGTH
        # speech = speech[:len_speech, :]
        # noise = noise[:len_speech, :]
        # mix = mix[:len_speech, :]
        # # stft
        # speech_ = self.stft.transform(torch.Tensor(speech.T))
        # mix_ = self.stft.transform(torch.Tensor(mix.T))
        # noise_ = self.stft.transform(torch.Tensor(noise.T))
        # # (B,T,F)
        # mix_real = mix_[:, :, :, 0]
        # mix_imag = mix_[:, :, :, 1]
        #
        # noise_real = noise_[:, :, :, 0]
        # noise_imag = noise_[:, :, :, 1]
        #
        # # mix_mag(T,F)
        # mix_mag = torch.sqrt(mix_imag ** 2 + mix_real ** 2).squeeze()
        # noise_mag = torch.sqrt(noise_imag ** 2 + noise_real ** 2).squeeze()

        return torch.Tensor(mix), torch.Tensor(speech), torch.Tensor(noise), nframe

    def __len__(self):
        """
        返回总体数据的长度
        :return: 总体数据的长度
        """
        return len(self.files)

    def __init__(self, root_dir):
        """
        初始化dataset，读入文件的list
        :param root_dir: 文件的根目录
        :param type: 暂时未用
        :param transform: 暂时未用
        """
        # 初始化变量
        self.stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)


class SpeechDataLoader(object):
    def __init__(self, data_set, batch_size, is_shuffle=True, num_workers=0):
        """
        初始化一个系统的Dataloader，只重写他的collate_fn方法
        :param data_set: 送入网络的data,dataset对象
        :param batch_size: 每次送入网络的data的数量，即多少句话
        :param is_shuffle: 是否打乱送入网络
        :param num_workers: dataloader多线程工作数，一般我们取0
        """
        self.data_loader = DataLoader(dataset=data_set,
                                      batch_size=batch_size,
                                      shuffle=is_shuffle,
                                      num_workers=num_workers,
                                      collate_fn=self.collate_fn)

    # 静态方法，由类和对象调用
    # 该函数返回对数据的处理，返回target,load_data
    @staticmethod
    def collate_fn(batch):
        """
        将每个batch中的数据pad成一样长，采取补零操作
        切记为@staticmethod方法
        :param batch: input和label的list
        :return:input、label和真实帧长 的list
        """
        mix_list = []
        speech_list = []
        noise_list = []
        frame_size_list = []
        for item in batch:
            # (T,F)
            mix_list.append(item[0])
            speech_list.append(item[1])
            noise_list.append(item[2])
            # 储存每句话的真实帧长，时域信息，用于计算loss
            frame_size_list.append(item[3])
        mix_list = nn.utils.rnn.pad_sequence(mix_list)
        speech_list = nn.utils.rnn.pad_sequence(speech_list)
        noise_list = nn.utils.rnn.pad_sequence(noise_list)

        mix_list = mix_list.permute(1, 0, 2)
        speech_list = speech_list.permute(1, 0, 2)
        noise_list = noise_list.permute(1, 0, 2)

        return BatchInfo(mix_list, speech_list, noise_list, frame_size_list)

    def get_data_loader(self):
        """
        获取Dataloader
        :return: dataloader对象
        """
        return self.data_loader


class BatchInfo(object):

    def __init__(self, mix, speech, noise, nframe):
        self.mix = torch.Tensor(mix)
        self.speech = torch.Tensor(speech)
        self.noise = torch.Tensor(noise)
        self.nframe = nframe


class FeatureCreator(nn.Module):

    def __init__(self):
        super(FeatureCreator, self).__init__()
        self.stft = STFT(FILTER_LENGTH, HOP_LENGTH).cuda(CUDA_ID[0])
        self.label_helper = LabelHelper()

    def forward(self, batch_info):
        # librosa不能用tensor做
        # speech_spec = librosa.stft(y=batch_info.speech, win_length=400, hop_length=160, n_fft=512)
        batch_info.mix = batch_info.mix.cuda(CUDA_ID[0])
        batch_info.speech = batch_info.speech.cuda(CUDA_ID[0])
        batch_info.noise = batch_info.noise.cuda(CUDA_ID[0])

        mix_spec = self.stft.transform(batch_info.mix)
        speech_spec = self.stft.transform(batch_info.speech)
        noise_spec = self.stft.transform(batch_info.noise)
        mix_spec = mix_spec.permute(0, 3, 1, 2)
        # label[IRM, phase_angle]
        label = self.label_helper(speech_spec, noise_spec)
        return mix_spec, label, batch_info.nframe
