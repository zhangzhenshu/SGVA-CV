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
import time
import xlwt
#import openpyxl as op


'''
Voting Threshold
Data : yangData
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
    table_file = pd.read_table(txt_path, sep=' ', header=None)
    txt_file = table_file.iloc[:, :]
    txt_array = txt_file.values

    return txt_array


# read relax txt
def txt2array0(txt_path):
    """
    :param txt_path:    specific path of a single txt file
    :return:            1-dimension preprocessed vector <class 'np.ndarray'>
                        of the input txt file
    """
    table_file = pd.read_table(txt_path, header=None)
    txt_file = table_file.iloc[:, :]
    txt_array = txt_file.values

    return txt_array


def preprocessing0(data):
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


def preprocessing(data):
    """
    :param data:    8*400 emg data <class 'np.ndarray'>    400*8
    :return:        data instance after rectifying and filter  8*400
    """
    # scalar
    # data = 2 * (data + 128) / 256 - 1

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
    if 'fist' in path:
        label = 0
    elif 'waveIn' in path:
        label = 1
    elif 'waveOut' in path:
        label = 2
    elif 'doubleTap' in path:
        label = 3
    elif 'fingersSpread' in path:
        label = 4
    elif 'relax' in path:
        label = 5
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
    parser.add_argument('--epochs', type=int, default=1,
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
    user_name = ['s%d'%x for x in range(1,9)]  # 8
    # user_name=['test']

    number_of_classes = 5
    window_size = 52

    # load training data -> 5 samples / gesture
    list_total_user_data = []
    list_total_user_labels = []

    for user in user_name:  # --------------for users---------------
        data = []
        labels = []
        user_data = None
        user_labels = []

        user_path = path + '/' + str(user) + '/'
        file_names = os.listdir(str(user_path))  # 读取所有txt路径
        file_paths = [user_path + i for i in file_names]  # 取出每个受试者的所有txt
        txt_total = len(file_names)

        user_data = []
        user_labels = []
        for txt_path in file_paths:  # --------for action------------
            txt_num = int(txt_path.split('_')[1].split('.')[0])  # 分割

            # if txt_num > 5 and txt_num <= 35:  # 左开右闭
            if txt_num <= 35:  # 左开右闭

                label = label_indicator(txt_path)
                if label == 5:  # relax
                    emg_array = txt2array0(txt_path)  # <np.ndarray> (400, 8)
                else:
                    emg_array = txt2array(txt_path)  # <np.ndarray> (400, 8)

                user_data.append(emg_array)
                user_labels.append(label)

        list_total_user_data.append(user_data)
        list_total_user_labels.append(user_labels)
    total_user_data = np.array(list_total_user_data, dtype=np.float32)  # <np.ndarray> (11, 180, 400, 8)
    total_user_labels = np.array(list_total_user_labels, dtype=np.int64)  # <np.ndarray> (11, 180)
    print('Load Data Finished...')

    # --------------------------------------------- Loading Data -------------------------------------------------- #
    excel=xlwt.Workbook(encoding='utf-8',style_compression=0)

    # 阈值投票
    threshold = 0.1
    # max_fit=90
    vote_fit = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

    # vote_fit = [10]
    acc_all_repeat = []
    for max_fit in vote_fit:

        for seed in range(10):
            accuracy = []
            args.seed = seed
            print("*" * 30)
            print("*" * 30)
            print('repeat{}'.format(args.seed))
            print("*" * 30)
            print("*" * 30)
            time_list = []
            acc_list = []
            acc_in_repeat = []
            for i in range(8):
                start_time = time.time()

                # Choose training domains
                sub = i
                args.list_test_domain = i
                all_training_domains = [0, 1, 2, 3, 4, 5, 6, 7]
                all_training_domains.remove(args.list_test_domain)
                args.list_train_domains = all_training_domains

                # Model name
                args.list_test_domain = [args.list_test_domain]
                model_name = '../saved_model/diva/' + 'test_domain_' + str(args.list_test_domain[0]) + '_diva_seed_' + str(
                    args.seed)

                # Load model
                model = DIVA(args)
                model = torch.load(model_name + '.model').to(device='cpu')

                # get data
                original_train_data = total_user_data.tolist()[args.list_test_domain[0]]
                original_train_label = total_user_labels.tolist()[args.list_test_domain[0]]
                label_prediction = []
                label_truth = []
                model.eval()


                ## vote
                for index in range(len(original_train_label)):  # 对每个txt

                    batch_correct = 0
                    sum_correct = 0
                    batch_len = 0
                    sum_len = 0
                    gesture_num_vector = [0] * 6

                    train_data = np.array(original_train_data[index], dtype=np.float32)  # txt
                    train_label = original_train_label[index]  # label of txt

                    # 由于输入txt之间存在差别引起，非对动作本身特别处理 (relax)
                    if train_label == 5:
                        # # pre-processing 滤波
                        single_sample_preprocessed = preprocessing0(train_data)  # <np.ndarray> (8, 400)
                    else:
                        single_sample_preprocessed = preprocessing(train_data)  # <np.ndarray> (8, 400)

                    # 处理不同的样本
                    if sum(mav(single_sample_preprocessed)) < threshold:  # relax'
                        activation_emg = single_sample_preprocessed
                        activation_length = single_sample_preprocessed.shape[1]
                        total_silding_size = activation_emg.shape[1] - window_size

                    else:
                        # detect muscle activation region
                        # index_start, index_end=0,400
                        index_start, index_end = detect_muscle_activity(single_sample_preprocessed)
                        activation_emg = single_sample_preprocessed[:,
                                         int(index_start): int(index_end)]  # (8, active_length)
                        activation_length = index_end - index_start

                        # slide
                        total_silding_size = activation_emg.shape[1] - window_size

                    segments_data = []
                    segments_label = []

                    for slice_num in range(total_silding_size):  # 对每个切片序列
                        emg_segment = activation_emg[:, slice_num: slice_num + window_size]  # (8, 52)

                        segments_train_data = np.array(emg_segment,
                                                       dtype=np.float32)  # <np.ndarray> (total_silding_size * 8 * window_length)

                        train_datas = torch.tensor(segments_train_data).unsqueeze(0).unsqueeze(0)
                        pred_d, pred_y = model.classifier(train_datas)
                        pred_pos = torch.argmax(pred_y, dim=1)
                        if pred_pos.item() < 6:
                            gesture_num_vector[pred_pos.item()] = gesture_num_vector[int(pred_pos.item())] + 1

                        max_num = max(gesture_num_vector)  # num of max gesture
                        pos_max = gesture_num_vector.index(max_num)  # max gesture(pos)

                        if max_num > max_fit:
                            break

                    final_prediction = pos_max
                    label_prediction.append(final_prediction)
                    label_truth.append(train_label)
                count = 0
                for i in range(len(label_prediction)):
                    if label_prediction[i] == label_truth[i]:
                        count += 1
                acc = count / len(label_prediction)
                print("Test_Subject:{}, acc:{:.4f},  ave_time:{:.3f}".format(sub, acc, (time.time() - start_time) / 35))
                acc_list.append(acc*100)
            acc_all_repeat.append(acc_list)
    acc_all_repeat = np.array(acc_all_repeat)
    pd.DataFrame(acc_all_repeat).to_excel('diva_vote.xlsx', sheet_name='0', index=False)


