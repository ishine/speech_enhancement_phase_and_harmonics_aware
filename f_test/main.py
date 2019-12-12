import torch.nn as nn
import torch
from a_net.net import Attention, FTB, TSB, Net, GlobalLayerNorm
import scipy.io as sio
from g_utils.stft_istft import STFT
from g_utils.label_set import LabelHelper
import soundfile as sf
import numpy as np
from config import *
import progressbar
import shutil
import os


def test_gln():
    test_input = torch.randn([8, 96, 1024, 161])
    gln = GlobalLayerNorm(channel_size=2)
    out = gln(test_input)
    print(out.size())


def test_est():
    stft = STFT(filter_length=320, hop_length=160)
    label_helper = LabelHelper()
    mat = sio.loadmat('/data/yangyang/data/SPEECH_ENHANCE_DATA/cv/S_67_01_babble_0_0002874.mat')
    speech = mat['speech']
    noise = mat['noise']
    mix = speech + noise
    sr = 16000
    sf.write('speech.wav', speech, sr)
    sf.write('mix.wav', mix, sr)
    noise_spec = stft.transform(torch.Tensor(noise[np.newaxis, :]))
    label_spec = stft.transform(torch.Tensor(speech[np.newaxis, :]))
    input_spec = stft.transform(torch.Tensor(mix[np.newaxis, :]))
    input_real = input_spec[:, :, :, 0]
    input_img = input_spec[:, :, :, 1]

    input_mag = torch.sqrt(input_img ** 2 + input_real ** 2)
    phase_input = torch.atan2(input_img, input_real)

    label_real = label_spec[:, :, :, 0]
    label_imag = label_spec[:, :, :, 1]

    phase_label = torch.atan2(label_imag, label_real)
    IRM = label_helper(label_spec, noise_spec)

    out_mag = (IRM * input_mag)

    mix_phase = out_mag.numpy() * np.exp(1j * phase_input.numpy())
    out_speech = torch.Tensor(np.stack([mix_phase.real, mix_phase.imag], 3))
    out_sample = stft.inverse(out_speech)
    sf.write('est.wav', out_sample.squeeze().numpy(), sr)

    out_phase = out_mag.numpy() * np.exp(1j * phase_label.numpy())
    out_speech = torch.Tensor(np.stack([out_phase.real, out_phase.imag], 3))
    out_sample = stft.inverse(out_speech)
    sf.write('clean.wav', out_sample.squeeze().numpy(), sr)


def test_mat(list, path):
    bar = progressbar.ProgressBar(0, list.__len__())
    i = 0
    bar.start()
    for item in list:
        bar.update(i)
        mat = sio.loadmat(path + item)
        i = i + 1
    bar.finish()


def test_paper_recover():
    stft = STFT(filter_length=320, hop_length=160)
    label_helper = LabelHelper()
    mat = sio.loadmat('/data/yangyang/data/SPEECH_ENHANCE_DATA/cv/S_67_01_babble_0_0002874.mat')
    speech = mat['speech']
    noise = mat['noise']
    mix = speech + noise
    sr = 16000
    sf.write('speech.wav', speech, sr)
    sf.write('mix.wav', mix, sr)
    noise_spec = stft.transform(torch.Tensor(noise[np.newaxis, :]))
    label_spec = stft.transform(torch.Tensor(speech[np.newaxis, :]))
    input_spec = stft.transform(torch.Tensor(mix[np.newaxis, :]))
    input_real = input_spec[:, :, :, 0]
    input_img = input_spec[:, :, :, 1]

    input_mag = torch.sqrt(input_img ** 2 + input_real ** 2)
    phase_input = torch.atan2(input_img, input_real)

    label_real = label_spec[:, :, :, 0]
    label_imag = label_spec[:, :, :, 1]

    phase_label = torch.atan2(label_imag, label_real)
    phase = torch.atan(phase_label)
    IRM = label_helper(label_spec, noise_spec)


    paper_out = input_mag.unsqueeze(3) * IRM[0].unsqueeze(0).unsqueeze(3) * label_spec
    paper_out_sample = stft.inverse(paper_out)
    sf.write('paper_out.wav', paper_out_sample.squeeze().numpy(), sr)

    out_mag = (IRM[0] * input_mag.squeeze())

    mix_phase = out_mag.numpy() * np.exp(1j * phase_input.numpy())
    out_speech = torch.Tensor(np.stack([mix_phase.real, mix_phase.imag], 3))
    out_sample = stft.inverse(out_speech)
    sf.write('est.wav', out_sample.squeeze().numpy(), sr)
    # 使用永杰给的方法恢复
    out_phase = out_mag.numpy() * np.exp(1j * phase_label.numpy())
    out_speech = torch.Tensor(np.stack([out_phase.real, out_phase.imag], 3))
    out_sample = stft.inverse(out_speech)
    sf.write('clean.wav', out_sample.squeeze().numpy(), sr)


def test_snr():
    sig = sio.loadmat('/data/yangyang/data/SPEECH_ENHANCE_DATA/cv/S_67_01_babble_0_0002874.mat')
    speech = sig['speech']
    alpha = np.sqrt(np.sum(speech ** 2) / ((np.sum(speech ** 2)) * pow(10.0, 5)))
    speech_enhance = speech * alpha
    sf.write('speech.wav', speech, 16000)
    sf.write('speech_enhance.wav', speech_enhance, 16000)

def create_train_data():
    file = os.listdir(TRAIN_DATA_PATH)
    file.sort()
    for i in range(10000):
        # src, dist
        shutil.copy(TRAIN_DATA_PATH + file[i], '/data/yangyang/data/Data/train/')

def create_validation_data():
    file = os.listdir(VALIDATION_DATA_PATH)
    file.sort()
    for i in range(100):
        shutil.copy(VALIDATION_DATA_PATH + file[i], '/data/yangyang/data/Data/validation/')


def test_model():
    net = torch.load(MODEL_STORE + 'model_2000.pkl', 'cuda:1')
    net.eval()
    net.cuda('cuda:1')
    file = os.listdir(VALIDATION_DATA_PATH)
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
    bar = progressbar.ProgressBar(0, len(file))
    i = 0
    bar.start()
    for item in file:
        data = sio.loadmat(VALIDATION_DATA_PATH + item)
        name = item[:-4]
        speech = data['speech']
        mix = data['speech'] + data['noise']
        sf.write(name + '_speech.wav', speech.squeeze(), 16000)
        sf.write(name + '_mix.wav', mix.squeeze(), 16000)
        mix_spec = stft.transform(torch.Tensor(mix[np.newaxis, :])).cuda('cuda:1')
        # mix的相位
        mix_phase = torch.atan2(mix_spec[:, :, :, 1], mix_spec[:, :, :, 0])
        mix_mag = torch.sqrt(mix_spec[:, :, :, 0] ** 2 + mix_spec[:, :, :, 1] ** 2)

        mix_spec = mix_spec.permute(0, 3, 1, 2)
        mix_spec = mix_spec.cuda('cuda:1')
        output = net(mix_spec)
        mask = output[0]
        phase_spec = output[1]
        phase_real = phase_spec[:, :, :, 0]
        phase_imag = phase_spec[:, :, :, 1]
        phase = torch.atan2(phase_imag, phase_real)
        out_mag = (mask * mix_mag.squeeze())

        # 使用预测的相位
        est_spec = out_mag.detach().cpu().numpy() * np.exp(1j * phase.detach().cpu().numpy())
        est_speech = torch.Tensor(np.stack([est_spec.real, est_spec.imag], 3))
        est_sample = stft.inverse(est_speech)
        sf.write(name + '_est.wav', est_sample.squeeze().detach().cpu().numpy(), 16000)

        # 使用mix的相位
        est_spec_mix = out_mag.detach().cpu().numpy() * np.exp(1j * mix_phase.detach().cpu().numpy())
        est_speech_mix = torch.Tensor(np.stack([est_spec_mix.real, est_spec_mix.imag], 3))
        est_sample_mix = stft.inverse(est_speech_mix)
        sf.write(name + '_est_mix.wav', est_sample.squeeze().detach().cpu().numpy(), 16000)

        i += 1
        bar.update(i)
    bar.finish()


if __name__ == '__main__':
    test_model()
    # create_train_data()
    # create_validation_data()
    # test_paper_recover()
    # cv_list = os.listdir(VALIDATION_DATA_PATH)
    # test_list = os.listdir(TEST_DATA_PATH)
    # train_list = os.listdir(TRAIN_DATA_PATH)
    # test_mat(train_list, TRAIN_DATA_PATH)

    # (batch, channel, T, F)
    # input = torch.randn([8, 2, 201, 1024])
    # net = Net()
    # output = net(input)
    # print(output)
