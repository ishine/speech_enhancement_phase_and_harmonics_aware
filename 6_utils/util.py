import os, struct
import time
import re
import torch
from config import *
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from utils.stft_istft import STFT
import numpy as np


def expandWindow(data, left, right):
    data = data.detach().cpu().numpy()
    sp = data.shape
    idx = 0
    exdata = np.zeros([sp[0], sp[1], sp[2] * (left + right + 1)])
    for i in range(-left, right+1):
        exdata[:, :, sp[2] * idx : sp[2] * (idx + 1)] = np.roll(data, shift=-i, axis=1)
        idx = idx + 1
    return torch.Tensor(exdata).cuda(CUDA_ID[0])


def context_window(data, left, right):
    """
    扩帧函数
    :param data:tensor类型的待扩帧数据，shape=(B,T,F)
    :param left: 左扩长度
    :param right: 右扩长度
    :return: 扩帧后的结果 shape = (B,T,F * (1 + left + right))
    """
    sp = data.size()
    # exdata = torch.zeros(sp[0], sp[1], sp[2] * (left + right + 1)).cuda(CUDA_ID[0])
    exdata = torch.zeros(sp[0], sp[1], sp[2] * (left + right + 1)).cuda(CUDA_ID[0])
    for i in range(1, left + 1):
        exdata[:, i:, sp[2] * (left - i) : sp[2] * (left - i + 1)] = data[:, :-i,:]
    for i in range(1, right+1):
        exdata[:, :-i, sp[2] * (left + i):sp[2]*(left+i+1)] = data[:, i:, :]
    exdata[:, :, sp[2] * left : sp[2] * (left + 1)] = data
    return exdata


def read_list(list_file):
    f = open(list_file, "r")
    lines = f.readlines()
    list_sig = []
    for x in lines:
        list_sig.append(x[:-1])
    f.close()
    return list_sig


def gen_list(wav_dir, append):
    """使用正则表达式获取相应文件的list
    wav_dir:路径
    append:文件类型，eg: .wav .mat
    """
    l = []
    lst = os.listdir(wav_dir)
    lst.sort()
    for f in lst:
        if re.search(append, f):
            l.append(f)
    return l


def write_log(file, name, train, validate):
    message = ''
    for m, val in enumerate(train):
        message += ' --TRerror%i=%.3f ' % (m, val.data.numpy())
    for m, val in enumerate(validate):
        message += ' --CVerror%i=%.3f ' % (m, val.data.numpy())
    file.write(name + ' ')
    file.write(message)
    file.write('/n')


def get_alpha(mix, constant=1):
    """
    求得进行能量归一化的alpha值
    :param mix: 带噪语音的采样点的array
    :param constant: 一般取值为1，即使噪声平均每个采样点的能量在1以内
    :return: 权值c
    """
    # c = np.sqrt(constant * mix.size / np.sum(mix**2)), s *= c, mix *= c
    return np.sqrt(constant * mix.size / np.sum(mix ** 2))


def wav_file_resample(src, dst, source_sample=44100, dest_sample=16000):
    """
    对WAV文件进行resample的操作
    :param file_path: 需要进行resample操作的wav文件的路径
    :param source_sample:原始采样率
    :param dest_sample:目标采样率
    :return:
    """
    sample_rate, sig = wavfile.read(src)
    result = int((sig.shape[0]) / source_sample * dest_sample)
    x_resampled = signal.resample(sig, result)
    x_resampled = x_resampled.astype(np.float64)
    return x_resampled, dest_sample
    # wavfile.write(dst, dest_sample, x_resampled)


def pixel_shuffle_1d(inp, upscale_factor=2):
    """
    shuflle 一维的实现
    :param inp: shape(F, d)
    :param upscale_factor:放缩的范围
    :return:shape(F/2, 2d)
    """
    batch_size, channels, in_width = inp.size()
    channels //= upscale_factor

    out_width = in_width * upscale_factor
    inp_view = inp.contiguous().view(batch_size, channels, upscale_factor, in_width)
    shuffle_out = inp_view.permute(0, 1, 3, 2).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_width)
    return shuffle_out


def write_bin_file(source_dir, dest_file):
    """
    写入bin文件
    :param source_dir:
    :param dest_file:
    :return:
    """
    work_dir = source_dir
    with open(dest_file, 'wb') as file:
        for parent, dirnames, filenames in os.walk(work_dir):
            for filename in filenames:
                file_path = os.path.join(parent, filename)
                if filename.lower().endswith('.wav'):
                    try:
                        start = time.time() * 1000
                        print('读取到WAV文件: {}'.format(file_path))
                        sample, signal = wavfile.read(file_path)
                        assert signal.dtype == np.int16
                        signal = signal / 32768.0
                        for i in signal:
                            f = struct.pack('f', i)
                            file.write(f)
                        end = time.time() * 1000
                        print("花费时间：{}".format(str(end - start)))
                    except:
                        pass


def cal_mean_var(speech, is_log=IS_LOG):
    """
    计算均值和方差
    :param speech:tensor,shape=(B,xxx)
    :param is_log: 是否为log谱
    :return:
    """
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
    speech_spec = stft.transform(speech)
    speech_mag = stft.spec_transform(speech_spec,is_log=is_log)
    mean = torch.mean(speech_mag, 0)
    var = torch.var(speech_mag, 0)
    return mean, var


def normalzation(abs_data):
    """
    归一化
    :param input_data:tensor,abs谱
    :return: 归一化后的谱
    """
    s = abs_data.clamp(-10000, 1e-4)
    s = 20 * torch.log(s + EPSILON) - 20
    s = (s + 100) / 100
    s = s.clamp(0, 1)
    return s


def re_normalization(normalization_data):
    """
    恢复normailzation的数据
    :param normalization_data:归一化的数据
    :return:愿数据，abs谱
    """
    s = normalization_data * 100 - 100
    s = torch.exp((s + 20) / 20)
    return s
