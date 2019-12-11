import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import soundfile as sf
from scipy.io import loadmat
from g_utils.util import get_alpha
from g_utils.pesq import pesq
from g_utils.stft_istft import STFT
import torch
from config import *
from a_net.net import Net
from b_load_data.SpeechDataLoad import SpeechDataset, SpeechDataLoader, FeatureCreator
from g_utils.loss_set import LossHelper
import torch.optim as optim
from config import *
from tensorboardX import SummaryWriter
import progressbar


def validation(path, net):
    net.eval()
    files = os.listdir(path)
    pesq_unprocess = 0
    pesq_res = 0
    bar = progressbar.ProgressBar(0, len(files))
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
    for i in range(len(files)):
        bar.update(i)
        with torch.no_grad():
            speech = loadmat(path + files[i])['speech']
            noise = loadmat(path + files[i])['noise']
            mix = speech + noise

            sf.write('clean.wav', speech, 16000)
            sf.write('mix.wav', mix, 16000)

            c = get_alpha(mix)
            mix *= c
            speech *= c
            noise *= c

            speech = stft.transform(torch.Tensor(speech.T).cuda(CUDA_ID[0]))
            mix = stft.transform(torch.Tensor(mix.T).cuda(CUDA_ID[0]))
            noise = stft.transform(torch.Tensor(noise.T).cuda(CUDA_ID[0]))

            mix_real = mix[:, :, :, 0]
            mix_imag = mix[:, :, :, 1]
            mix_mag = torch.sqrt(mix_real ** 2 + mix_imag ** 2)


            # mix_(T,F)
            mix_mag = mix_mag.unsqueeze(0).cuda(CUDA_ID[0])
            # output(1, T, F)

            mapping_out = net(mix_mag)

            res_real = mapping_out * mix_real / mix_mag.squeeze(0)
            res_imag = mapping_out * mix_imag / mix_mag.squeeze(0)

            res = torch.stack([res_real, res_imag], 3)
            output = stft.inverse(res)

            output = output.permute(1, 0).detach().cpu().numpy()

            # 写入的必须是（F,T）istft之后的
            sf.write('est.wav', output / c, 16000)
            try:
                p1 = pesq('clean.wav', 'mix.wav', 16000)
                p2 = pesq('clean.wav', 'est.wav', 16000)
            except:
                print('wrong test item : ' + files[i])
                pass
            pesq_unprocess += p1[0]
            pesq_res += p2[0]

    bar.finish()
    net.train()
    return [pesq_unprocess / len(files), pesq_res / len(files)]


def validation_for_loss(net):
    net.eval()
    bar = progressbar.ProgressBar(0, validation_data_set.__len__() // VALIDATION_BATCH_SIZE)
    feature_creator = FeatureCreator()
    bar.start()
    sum_loss = 0
    mse = torch.nn.MSELoss()
    for batch_idx, batch_info in enumerate(validation_data_loader.get_data_loader()):
        bar.update(batch_idx)
        mix_spec, label, frame_list = feature_creator(batch_info)
        a_label = label[0]
        p_label = label[1]
        optimizer.zero_grad()
        output = net(mix_spec)
        a_out = output[0]
        p_out = output[1]
        # 构造s^out
        tmp_spec = mix_spec.permute(0, 3, 2, 1)
        mix_mag = torch.sqrt(tmp_spec[:, :, :, 0] ** 2 + tmp_spec[:, :, :, 1] ** 2)
        s_out_real = (mix_mag.permute(0, 2, 1) * a_out) * p_out[:, :, :, 0]
        s_out_imag = (mix_mag.permute(0, 2, 1) * a_out) * p_out[:, :, :, 1]
        s_out_mag = torch.sqrt(s_out_real ** 2 + s_out_imag ** 2)
        s_out = torch.stack([s_out_real, s_out_imag], 3)

        loss_A = loss_helper.mse_loss(s_out_mag, a_label, frame_list)
        loss_P = loss_helper.mse_loss_phase(s_out, p_label, frame_list)

        loss = torch.pow(loss_A, 0.3) * 0.5 + torch.pow(loss_P, 0.3) * 0.5
        sum_loss += loss.item()
    net.train()
    return sum_loss / (validation_data_set.__len__() // VALIDATION_BATCH_SIZE)


def train(net, optimizer, data_loader, loss_helper, epoch):
    global global_step
    writer = SummaryWriter(LOG_STORE)
    feature_creator = FeatureCreator()
    bar = progressbar.ProgressBar(0, train_data_set.__len__() // TRAIN_BATCH_SIZE)
    sum_loss = 0
    sum_loss_A = 0
    sum_loss_P = 0
    for i in range(epoch):
        bar.start()
        for batch_idx, batch_info in enumerate(data_loader.get_data_loader()):
            bar.update(batch_idx)
            mix_spec, label, frame_list = feature_creator(batch_info)
            a_label = label[0]
            p_label = label[1]
            optimizer.zero_grad()
            output = net(mix_spec)
            a_out = output[0]
            p_out = output[1]
            # 构造s^out
            tmp_spec = mix_spec.permute(0, 3, 2, 1)
            mix_mag = torch.sqrt(tmp_spec[:, :, :, 0] ** 2 + tmp_spec[:, :, :, 1] ** 2)
            s_out_real = (mix_mag.permute(0, 2, 1) * a_out) * p_out[:, :, :, 0]
            s_out_imag = (mix_mag.permute(0, 2, 1) * a_out) * p_out[:, :, :, 1]
            s_out_mag = torch.sqrt(s_out_real ** 2 + s_out_imag ** 2)
            s_out = torch.stack([s_out_real, s_out_imag], 3)

            loss_A = loss_helper.mse_loss(s_out_mag, a_label, frame_list)
            loss_P = loss_helper.mse_loss_phase(s_out, p_label, frame_list)

            loss = torch.pow(loss_A, 0.3) * 0.5 + torch.pow(loss_P, 0.3) * 0.5
            sum_loss += loss.item()
            sum_loss_A += loss_A.item()
            sum_loss_P += loss_P.item()
            loss.backward()
            optimizer.step()
            # 验证集
            # if global_step % 100 == 0 and global_step != 0:
            if global_step % 100 == 0:
                avg_loss = validation_for_loss(net)
                torch.save(net, MODEL_STORE + 'model_' + str(global_step) + '.pkl')
                writer.add_scalar('Validation/avg_loss', avg_loss, global_step)
            if global_step % 100 == 0 and global_step != 0:
                writer.add_scalar('Train/loss', sum_loss / 100, global_step)
                writer.add_scalar('Train/loss', sum_loss_A / 100, global_step)
                writer.add_scalar('Train/loss', sum_loss_P / 100, global_step)
                sum_loss = 0
                sum_loss_A = 0
                sum_loss_P = 0
            global_step += 1
        bar.finish()


if __name__ == '__main__':
    global_step = 0
    validation_data_set = SpeechDataset(root_dir=VALIDATION_DATA_PATH)
    validation_data_loader = SpeechDataLoader(data_set=validation_data_set,
                                              batch_size=VALIDATION_BATCH_SIZE,
                                              is_shuffle=True)
    train_data_set = SpeechDataset(root_dir=TRAIN_DATA_PATH)
    data_loader = SpeechDataLoader(data_set=train_data_set,
                                   batch_size=TRAIN_BATCH_SIZE,
                                   num_workers=NUM_WORKERS,
                                   is_shuffle=True)

    net = Net()
    net = net.cuda(CUDA_ID[0])
    optimizer = optim.Adam(net.parameters(), lr=LR)
    loss_helper = LossHelper()
    train(net, optimizer, data_loader, loss_helper, EPOCH)
