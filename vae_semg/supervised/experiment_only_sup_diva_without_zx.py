import sys
import numpy

sys.path.insert(0, "./../../../")

import argparse
import pandas as pd
import numpy as np

import torch
from torch.nn import functional as F
import torch.optim as optim
from torchvision.utils import save_image
import torch.utils.data as data_utils

from paper_experiments.vae_semg.supervised.model_diva_without_zx import DIVA
from paper_experiments.vae_semg.dataset.semgdata_loader import semgdata_load


'''
Section Ablation Study
SGVA-Wx
'''


def train(train_loader, model, optimizer, epoch):
    model.train()
    train_loss = 0
    epoch_class_y_loss = 0
    # zd_list=[]
    # zx_list = []
    # zy_list=[]
    for batch_idx, (x, y, d) in enumerate(train_loader):
        # To device
        x, y, d = x.to(device), y.to(device), d.to(device)

        # if (epoch % 50 == 0) and (batch_idx == 1):
        #     save_reconstructions(model, d, x, y)

        optimizer.zero_grad()
        loss, class_y_loss, zd_q, zy_q,d_target, y_target = model.loss_function(d, x, y)
        loss.backward()
        optimizer.step()

        train_loss += loss
        epoch_class_y_loss += class_y_loss

    train_loss /= len(train_loader.dataset)
    epoch_class_y_loss /= len(train_loader.dataset)

    return train_loss, epoch_class_y_loss


def save_reconstructions(model, d, x, y):
    # Save reconstuction
    with torch.no_grad():
        x_recon, _, _, _, _, _, _, _, _, _, _, _ = model.forward(d, x, y)
        recon_batch = x_recon.view(-1, 1, 28, 28, 256)

        sample = torch.zeros(100, 1, 28, 28).cuda()

        for i in range(28):
            for j in range(28):

                # out[:, :, i, j]
                probs = F.softmax(recon_batch[:, :, i, j], dim=2).data

                # Sample single pixel (each channel independently)
                for k in range(1):
                    # 0 ~ 255 => 0 ~ 1
                    val, ind = torch.max(probs[:, k], dim=1)
                    sample[:, k, i, j] = ind.squeeze().float() / 255.

        n = min(x.size(0), 8)
        comparison = torch.cat([x.view(100, 1, 28, 28)[:n],
                                sample[:n]])
        save_image(comparison.cpu(),
                   'reconstruction_only_sup_' + str(epoch) + '.png', nrow=n)


def get_accuracy(data_loader, classifier_fn, batch_size):
    model.eval()
    """
    compute the accuracy over the supervised training set or the testing set
    """
    predictions_d, actuals_d, predictions_y, actuals_y = [], [], [], []

    with torch.no_grad():
        # use the right data loader
        for (xs, ys, ds) in data_loader:
            # To device
            xs, ys, ds = xs.to(device), ys.to(device), ds.to(device)

            # use classification function to compute all predictions for each batch
            pred_d, pred_y = classifier_fn(xs)
            predictions_d.append(pred_d)
            actuals_d.append(ds)
            predictions_y.append(pred_y)
            actuals_y.append(ys)

        # compute the number of accurate predictions
        accurate_preds_d = 0
        for pred, act in zip(predictions_d, actuals_d):
            for i in range(pred.size(0)):
                v = torch.sum(pred[i] == act[i])
                accurate_preds_d += (v.item() == 7)

        # calculate the accuracy between 0 and 1
        accuracy_d = (accurate_preds_d * 1.0) / (len(predictions_d) * batch_size)

        # compute the number of accurate predictions
        accurate_preds_y = 0
        for pred, act in zip(predictions_y, actuals_y):
            for i in range(pred.size(0)):
                v = torch.sum(pred[i] == act[i])
                accurate_preds_y += (v.item() == 6)

        # calculate the accuracy between 0 and 1
        accuracy_y = (accurate_preds_y * 1.0) / (len(predictions_y) * batch_size)

        return accuracy_d, accuracy_y


if __name__ == "__main__":
    # Training settings
    # 000 调参专用
    parser = argparse.ArgumentParser(description='TwoTaskVae')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--num-supervised', default=1000, type=int,
                        help="number of supervised examples, /10 = samples per class")  # 没用到

    # Choose domains
    # parser.add_argument('--list_train_domains', type=list, default=['s1', 's2', 's3', 's4', 's5', 's6', 's7'],
    #                     help='domains used during training')
    # parser.add_argument('--list_test_domain', type=str, default='s7',
    #                     help='domain used during testing')
    parser.add_argument('--list_train_domains', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7],
                        help='domains used during training')
    parser.add_argument('--list_test_domain', type=int, default=1,
                        help='domain used during testing')

    # Model
    parser.add_argument('--d-dim', type=int, default=7,
                        help='number of source domain?')
    parser.add_argument('--x-dim', type=int, default=416,
                        help='input size after flattening')
    parser.add_argument('--y-dim', type=int, default=6,
                        help='number of classes')
    parser.add_argument('--zd-dim', type=int, default=128,
                        help='size of latent space 1')
    parser.add_argument('--zx-dim', type=int, default=128,
                        help='size of latent space 2')
    parser.add_argument('--zy-dim', type=int, default=128,
                        help='size of latent space 3')

    # auxiliary multipliers:辅助分类器
    parser.add_argument('--aux_loss_multiplier_y', type=float, default=3500.,
                        help='multiplier for y classifier')
    parser.add_argument('--aux_loss_multiplier_d', type=float, default=2000.,
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
    parser.add_argument('--max_beta', type=float, default=1., metavar='MB',
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

    # ------------------------------------------------------------------------------------------------- #
    # train
    # ------------------------------------------------------------------------------------------------- #
    # set work method
    work_method = 'train'
    # work_method = 'test'

    print(work_method)

    if work_method == 'train':
        for seed in range(1):
            accuracy = []
            args.seed = seed
            print("*" * 30)
            print('repeat{}'.format(args.seed))
            print("*" * 30)

            for i in range(8):
                # i = 1
                # Choose training domains
                args.list_test_domain = i
                all_training_domains = [0, 1, 2, 3, 4, 5, 6, 7]
                all_training_domains.remove(args.list_test_domain)
                args.list_train_domains = all_training_domains
                print("Train_Subject:", args.list_test_domain, args.list_train_domains)

                # Model name
                args.list_test_domain = [args.list_test_domain]
                print(args.outpath)
                model_name = '../saved_model/' + 'test_domain_' + str(
                    args.list_test_domain[0]) + '_diva_without_x_seed_' + str(
                    args.seed)
                print("Test_Subject:", model_name)

                # Load supervised training
                train_loader = data_utils.DataLoader(
                    semgdata_load(args.list_train_domains, args.list_test_domain, args.num_supervised, args.seed,
                                 './../dataset/',
                                 train=True),
                    batch_size=args.batch_size,
                    shuffle=True, **kwargs)

                # setup the VAE
                model = DIVA(args).to(device)

                # setup the optimizer
                optimizer = optim.Adam(model.parameters(), lr=args.lr)

                best_loss = 10000.  # 1000
                best_y_acc = 0.

                early_stopping_counter = 1
                max_early_stopping = 10

                # training loop
                print('\nStart training:', args)
                for epoch in range(1, args.epochs + 1):
                    # model.beta_d = min([args.beta_d, args.beta_d * (epoch * 1.) / args.warmup])
                    # model.beta_y = min([args.beta_y, args.beta_y * (epoch * 1.) / args.warmup])
                    # model.beta_x = min([args.beta_x, args.beta_x * (epoch * 1.) / args.warmup])
                    model.beta_d =  3
                    model.beta_y = 3
                    model.beta_x = 3

                    # train
                    avg_epoch_losses_sup, avg_epoch_class_y_loss = train(train_loader, model, optimizer, epoch)

                    # store the loss and validation/testing accuracies in the logfile
                    str_loss_sup = avg_epoch_losses_sup
                    str_print = "{} epoch: avg loss {}".format(epoch, str_loss_sup)  # DIVA总损失
                    str_print += ", class y loss {}".format(avg_epoch_class_y_loss)  # 标签分类损失

                    # this test accuracy is only for logging, this is not used
                    # to make any decisions during training
                    train_accuracy_d, train_accuracy_y = get_accuracy(train_loader, model.classifier, args.batch_size)
                    str_print += "     train accuracy d {}".format(train_accuracy_d)  # 训练集域分类精度
                    str_print += ", y {}".format(train_accuracy_y)  # 训练集标签分类精度
                    str_print += ",beta_d{},y{},x{}".format(model.beta_d, model.beta_y, model.beta_x)
                    print(str_print)

                    if train_accuracy_y > best_y_acc:
                        early_stopping_counter = 1

                        best_y_acc = train_accuracy_y
                        best_loss = avg_epoch_class_y_loss

                        torch.save(model, model_name + '.model')

                    elif train_accuracy_y == best_y_acc:
                        if avg_epoch_class_y_loss < best_loss:
                            early_stopping_counter = 1

                            best_loss = avg_epoch_class_y_loss

                            torch.save(model, model_name + '.model')

                        else:
                            early_stopping_counter += 1
                            if early_stopping_counter == max_early_stopping:
                                break

                    else:
                        early_stopping_counter += 1
                        if early_stopping_counter == max_early_stopping:
                            break

                # Load supervised test
                test_loader = data_utils.DataLoader(
                    semgdata_load(args.list_train_domains, args.list_test_domain, args.num_supervised, args.seed,
                                 './../saved_model/diva',
                                 train=False),
                    batch_size=args.batch_size,
                    shuffle=True, **kwargs)

                # Load model
                model = DIVA(args).to(device)
                model = torch.load(model_name + '.model')

                # this test accuracy is only for logging, this is not used
                # to make any decisions during training
                test_accuracy_d, test_accuracy_y = get_accuracy(test_loader, model.classifier, args.batch_size)
                print("test accuracy y {}".format(test_accuracy_y))
                accuracy.append(test_accuracy_y)
            print(args.seed, accuracy)
    # ------------------------------------------------------------------------------------------------- #
    # test
    # ------------------------------------------------------------------------------------------------- #
    if work_method == 'test':
        acc_all_repeat = []
        for seed in range(1):
            args.seed = seed
            print("*" * 30)
            print('repeat{}'.format(args.seed))
            acc_in_repeat = []
            for i in range(8):
                # i=2
                # Choose training domains
                args.list_test_domain = i
                all_training_domains = [0, 1, 2, 3, 4, 5, 6, 7]
                all_training_domains.remove(args.list_test_domain)
                args.list_train_domains = all_training_domains

                print("Test_Subject:", args.list_test_domain)

                # Model name
                args.list_test_domain = [args.list_test_domain]
                model_name = '../saved_model/' + 'test_domain_' + str(
                    args.list_test_domain[0]) + '_diva_without_x_seed_' + str(
                    args.seed)
                # print("Test_Subject:", model_name)

                # Load supervised test
                test_loader = data_utils.DataLoader(
                    semgdata_load(args.list_train_domains, args.list_test_domain, args.num_supervised, args.seed,
                                 './../dataset/',
                                 train=False),
                    batch_size=args.batch_size,
                    shuffle=True, **kwargs)

                # Load model
                model = DIVA(args).to(device)
                model = torch.load(model_name + '.model')

                # this test accuracy is only for logging, this is not used
                # to make any decisions during training
                test_accuracy_d, test_accuracy_y = get_accuracy(test_loader, model.classifier, args.batch_size)
                print("test accuracy y {}".format(test_accuracy_y))
                acc_in_repeat.append(test_accuracy_y)
            acc_all_repeat.append(acc_in_repeat)
        acc_all_repeat = np.array(acc_all_repeat)
        pd.DataFrame(acc_all_repeat).to_excel('diva_without_d.xlsx', sheet_name='0', index=False)
