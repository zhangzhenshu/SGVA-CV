import os
import time
import numpy as np
import pandas as pd
from scipy import signal
import sys
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, "./../../../")

import argparse
import numpy as np
import torch
from torch.nn import functional as F
from paper_experiments.vae_semg.supervised.model_diva import DIVA

'''
SGVA-CV
data:self_data
'''



# --------------------------------------------- Data Defination -------------------------------------------------- #
def mav(emg_data):
    """
    :param emg_data:    <class 'np.ndarray'>    (8, 40)
    :return:            mav feature vector of the input emg matrix     (8, )
    """
    mav_result = np.mean(abs(emg_data), axis=1)

    return mav_result


def txt2array(txt_path):
    """
    :param txt_path:    specific path of a single txt file
    :return:            1-dimension preprocessed vector <class 'np.ndarray'>
                        of the input txt file
    """
    table_file = pd.read_table(txt_path, header=None)
    txt_file = table_file.iloc[:, :-1]
    txt_array = txt_file.values

    return txt_array


def preprocessing(data):
    """
    :param data:    8*400 emg data <class 'np.ndarray'>    400*8
    :return:        data instance after rectifying and filter  8*400
    """
    # scalar
    data = 2 * (data + 128) / 256 - 1

    # rectify
    data_processed = np.abs(data)

    # transpose (400, 8) -> (8, 400)
    data_processed = np.transpose(data_processed)

    # filter
    wn = 0.5  # (2 * fc) / fs
    order = 4
    b, a = signal.butter(order, wn, btype='low')
    data_processed = signal.filtfilt(b, a, data_processed)  # data_processed <class 'np.ndarray': 8*400>

    return data_processed  # <class 'np.ndarray'> 4*800


def detect_muscle_activity(emg_data):
    """
    :param      emg_date: 8 channels of emg data -> 8*400
    :return:
                index_start: star index of muscle activation region
                index_end:   end index of muscle activation region
    """


    fs = 200  # sampling frequency
    min_activation_length = 50
    num_frequency_of_spec = 50
    hamming_window_length = 28  # 25
    overlap_samples = 20  # 10
    threshold_along_frequency = 18

    sumEMG = emg_data.sum(axis=0)  # sum 8 channel data into one vector

    f, time, Sxx = signal.spectrogram(sumEMG, fs=fs,
                                      window='hamming',
                                      nperseg=hamming_window_length,
                                      noverlap=overlap_samples,
                                      nfft=num_frequency_of_spec,
                                      detrend=False,
                                      mode='complex')

    # 43.6893
    # test plot
    Sxx = Sxx * 43.6893

    spec_values = abs(Sxx)
    spec_vector = spec_values.sum(axis=0)

    # 使用np.diff 求差分
    # indicated_vector 标记序列中哪些位置的强度值高于阈值
    indicated_vector = np.zeros(shape=(spec_vector.shape[0] + 2), )

    for index, element in enumerate(spec_vector):
        if element > threshold_along_frequency:
            indicated_vector[index + 1] = 1

    index_greater_than_threshold = np.abs(np.diff(indicated_vector))

    if index_greater_than_threshold[-1] == 1:
        index_greater_than_threshold[-2] = 1

    # 删去最后一个元素
    index_greater_than_threshold = index_greater_than_threshold[:- 1]

    # 找出非零元素的序号
    index_non_zero = np.where(index_greater_than_threshold == 1)[0]

    index_of_samples = np.floor(fs * time - 1)
    num_of_index_non_zero = index_non_zero.shape[0]

    length_of_emg = sumEMG.shape[0]
    # print('length of emg : %f points' % length_of_emg)

    # find the start and end indexes
    if num_of_index_non_zero == 0:
        index_start = 1
        index_end = length_of_emg
    elif num_of_index_non_zero == 1:
        index_start = index_of_samples[index_non_zero]
        index_end = length_of_emg
    else:
        index_start = index_of_samples[index_non_zero[0]]
        index_end = index_of_samples[index_non_zero[-1] - 1]

    num_extra_samples = 25
    index_start = max(1, index_start - num_extra_samples)
    index_end = min(length_of_emg, index_end + num_extra_samples)

    if (index_end - index_start) < min_activation_length:
        index_start = 0
        index_end = length_of_emg - 1

    return index_start, index_end


def label_indicator(path):
    label = None
    if 'Fist' in path:
        label = 0
    elif 'WaveIn' in path:
        label = 1
    elif 'WaveOut' in path:
        label = 2
    elif 'Tap' in path:
        label = 3
    elif 'Open' in path:
        label = 4
    return label


def cal_spectrogram_vector(vector, fs=200, npserseg=57, noverlap=0):
    """
    @param vector:
    @param fs:  频率
    @param npserseg:    窗口长度
    @param noverlap:    重合采样点
    @return:    时频向量，时域信息，频域信息
    """
    frequencies_samples, time_segment_sample, spectrogram_of_vector = signal.spectrogram(x=vector, fs=fs,
                                                                                         nperseg=npserseg,
                                                                                         noverlap=noverlap,
                                                                                         window='hann',
                                                                                         scaling='spectrum')
    return spectrogram_of_vector, time_segment_sample, frequencies_samples


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='TwoTaskVae')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--num-supervised', default=1000, type=int,
                        help="number of supervised examples, /10 = samples per class")  # 没用到

    # Choose domains
    parser.add_argument('--list_train_domains', type=list, default=[0, 1, 2, 3, 4, 5, 6],
                        help='domains used during training')
    parser.add_argument('--list_test_domain', type=int, default=1,
                        help='domain used during testing')

    # Model
    parser.add_argument('--d-dim', type=int, default=6,
                        help='number of source domain?')
    parser.add_argument('--x-dim', type=int, default=416,
                        help='input size after flattening')
    parser.add_argument('--y-dim', type=int, default=6,
                        help='number of classes')
    parser.add_argument('--zd-dim', type=int, default=64,
                        help='size of latent space 1')
    parser.add_argument('--zx-dim', type=int, default=64,
                        help='size of latent space 2')
    parser.add_argument('--zy-dim', type=int, default=64,
                        help='size of latent space 3')

    # auxiliary multipliers:辅助分类器
    parser.add_argument('--aux_loss_multiplier_y', type=float, default=2000.,
                        help='multiplier for y classifier')
    parser.add_argument('--aux_loss_multiplier_d', type=float, default=4000.,
                        help='multiplier for d classifier')
    # Beta VAE part
    parser.add_argument('--beta_d', type=float, default=1.,
                        help='multiplier for KL d')
    parser.add_argument('--beta_x', type=float, default=1.,
                        help='multiplier for KL x')
    parser.add_argument('--beta_y', type=float, default=1.,
                        help='multiplier for KL y')

    parser.add_argument('-w', '--warmup', type=int, default=100, metavar='N',
                        help='number of epochs for warm-up. Set to 0 to turn warmup off.')
    parser.add_argument('--max_beta', type=float, default=5., metavar='MB',
                        help='max beta for warm-up')
    parser.add_argument('--min_beta', type=float, default=0.0, metavar='MB',
                        help='min beta for warm-up')

    parser.add_argument('--outpath', type=str, default='./',
                        help='where to save')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

    # Set seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # --------------------------------------------- Loading Data -------------------------------------------------- #
    path = '../../data/dataset'  # 父目录的父目录
    user_name = ['S%d'%x for x in range(1,8)]  # 效果比较好的7个人
    lr = ["RIGHT"]

    use_type = 'all'

    number_of_classes = 5
    window_size = 52
    threshold = 60 / 128

    # load training data -> 5 samples / gesture
    list_total_user_data = []
    list_total_user_labels = []

    for user in user_name:  # --------------for users---------------
        print("---------------------------------")
        print("user:", user)

        for arms in lr:  # ----------------for arms-------------------
            print("\narms:", arms)
            txt_count = 0

            user_path = path + '/' + str(user) + '/' + str(use_type) + '/' + str(arms) + '/'
            file_names = os.listdir(str(user_path))  # 读取所有txt路径
            file_paths = [user_path + i for i in file_names]  # 取出每个受试者的所有txt
            txt_total = len(file_names)

            user_data = []
            user_labels = []
            for txt_path in file_paths:  # --------for action------------
                txt_num = int(txt_path.split('-')[1].split('.')[0])  # 分割
                if txt_num > 5 and txt_num <= 30:  # 左开右闭
                    # if txt_num <= 5:  # 左开右闭
                    txt_count += 1
                    emg_array = txt2array(txt_path)  # <np.ndarray> (400, 8)
                    label = label_indicator(txt_path)
                    user_data.append(emg_array)
                    user_labels.append(label)
            list_total_user_data.append(user_data)
            list_total_user_labels.append(user_labels)

    total_user_data = np.array(list_total_user_data, dtype=np.float32)  # <np.ndarray> (11, 180, 400, 8)
    total_user_labels = np.array(list_total_user_labels, dtype=np.int64)  # <np.ndarray> (11, 180)

    for i in range(1):

        # Choose training domains
        args.list_test_domain = i
        all_training_domains = [0, 1, 2, 3, 4, 5, 6]
        all_training_domains.remove(args.list_test_domain)
        args.list_train_domains = all_training_domains
        print("*" * 30)
        print("Test_Subject:", args.list_test_domain)

        # Model name
        args.list_test_domain = [args.list_test_domain]
        model_name = args.outpath + 'test_domain_' + str(args.list_test_domain[0]) + '_diva_seed_' + str(
            args.seed)

        # Load model
        model = DIVA(args)
        model = torch.load(model_name + '.model').to(device='cpu')
        print(model_name)

        # get data
        original_train_data = total_user_data.tolist()[args.list_test_domain[0]]
        original_train_label = total_user_labels.tolist()[args.list_test_domain[0]]
        label_prediction = []
        label_truth = []

        for index in range(len(original_train_label)):  # 对每个txt

            train_data = np.array(original_train_data[index], dtype=np.float32)  # txt
            train_label = original_train_label[index]  # label of txt

            # pre-processing 滤波
            single_sample_preprocessed = preprocessing(train_data)  # <np.ndarray> (8, 400)

            # detect muscle activation region
            index_start, index_end = detect_muscle_activity(single_sample_preprocessed)
            activation_emg = single_sample_preprocessed[:, int(index_start): int(index_end)]  # (8, active_length)
            activation_length = index_end - index_start

            # slide
            total_silding_size = activation_emg.shape[1] - window_size
            segments_data = []

            for index in range(total_silding_size):  # 对每个切片序列
                emg_segment = activation_emg[:, index: index + window_size]  # (8, 52)

                segments_data.append(emg_segment)  # <list> (total_silding_size * 8 * window_length)

                segments_train_data = np.array(segments_data,
                                               dtype=np.float32)  # <np.ndarray> (total_silding_size * 8 * window_length)
            # train_label = np.array([train_label] * total_silding_size)

            train_datas = torch.tensor(segments_train_data).unsqueeze(1)
            # train_labels = torch.tensor(train_label).long()
            pred_d, pred_y = model.classifier(train_datas)

            pred_y_pos = torch.sum(pred_y, dim=0)  # gesture sum
            pred_label = torch.argmax(pred_y_pos)
            max_times = torch.max(pred_y_pos)
            label_prediction.append(pred_label.item())
            label_truth.append(train_label)

        count = 0
        for j in range(len(label_prediction)):
            a = label_prediction[j]
            b = label_truth[j]
            if a == b:
                count += 1
        acc = count / len(label_prediction)
        print(acc)


