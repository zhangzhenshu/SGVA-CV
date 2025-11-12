import argparse
import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
import torch.optim as optim
from torchvision.utils import save_image
import torch.utils.data as data_utils

from model_diva import DIVA
from utils.semgdata_loader import semgdata_load


#------------------------------ Training  ------------------------------#
def train(train_loader, model, optimizer, epoch):
    # Inicializar el modo de entrenamiento
    model.train()
    train_loss = 0
    epoch_class_y_loss = 0
    # Bucle de datos
    for batch_idx, (x, y, d) in enumerate(train_loader):
        x, y, d = x.to(device), y.to(device), d.to(device)
        # if (epoch % 50 == 0) and (batch_idx == 1):
        #     save_reconstructions(model, d, x, y)
        optimizer.zero_grad()
        loss, class_y_loss, zd_q, zy_q, zx_q, d_target, y_target = model.loss_function(d, x, y)
        loss.backward()
        optimizer.step()
        train_loss += loss
        epoch_class_y_loss += class_y_loss
    train_loss /= len(train_loader.dataset)
    epoch_class_y_loss /= len(train_loader.dataset)
    #  retornar las perdidas promedio por epoca
    return train_loss, epoch_class_y_loss


# Funcion que calcula la precision de la clasificacion
def get_accuracy(data_loader, classifier_fn, batch_size):
    model.eval()
    predictions_d, actuals_d, predictions_y, actuals_y = [], [], [], []
    with torch.no_grad():
        # usar el clasificador para predecir las etiquetas de dominio y gesto
        for (xs, ys, ds) in data_loader:
            xs, ys, ds = xs.to(device), ys.to(device), ds.to(device)
            # usar el clasificador para predecir las etiquetas de dominio y gesto
            pred_d, pred_y = classifier_fn(xs)
            predictions_d.append(pred_d)
            actuals_d.append(ds)
            predictions_y.append(pred_y)
            actuals_y.append(ys)
        # recordar el numero de predicciones correctas
        accurate_preds_d = 0
        for pred, act in zip(predictions_d, actuals_d):
            for i in range(pred.size(0)):
                v = torch.sum(pred[i] == act[i])
                accurate_preds_d += (v.item() == 7)
        # calcular la precision entre 0 y 1
        accuracy_d = (accurate_preds_d * 1.0) / (len(predictions_d) * batch_size)
        # recordar el numero de predicciones correctas
        accurate_preds_y = 0
        for pred, act in zip(predictions_y, actuals_y):
            for i in range(pred.size(0)):
                v = torch.sum(pred[i] == act[i])
                accurate_preds_y += (v.item() == 6)
        # calcular la precision entre 0 y 1
        accuracy_y = (accurate_preds_y * 1.0) / (len(predictions_y) * batch_size)
        # retornar las precisiones
        return accuracy_d, accuracy_y


# ------------------------------ Main  ------------------------------#
if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------- #
    # Configuracion de Dispositivo
    # ------------------------------------------------------------------------------------------------- #

    # Ajuste de Entrenamiento
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser(description='TwoTaskVae')
    parser.add_argument('--no-cuda', action='store_true', default=False, 
                        help='disables CUDA training')  # false para usar cuda y true para no usar cuda
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')    # fijar la semilla aleatoria
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='input batch size for training (default: 64)') # tamaño del batch
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 10)') # numero de epocas
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')   # tasa de aprendizaje
    parser.add_argument('--num-supervised', default=1000, type=int,
                        help="number of supervised examples, /10 = samples per class")  # numero de ejemplos supervisados
    # parser.add_argument('--list_train_domains', type=list, default=['s1', 's2', 's3', 's4', 's5', 's6', 's7'],
    #                     help='domains used during training')  # dominios usados durante el entrenamiento
    # parser.add_argument('--list_test_domain', type=str, default='s7',
    #                     help='domain used during testing')    # dominio usado durante la prueba
    parser.add_argument('--list_train_domains', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7],
                        help='domains used during training')    # dominios usados durante el entrenamiento
    parser.add_argument('--list_test_domain', type=int, default=1,
                        help='domain used during testing')  # dominio usado durante la prueba
    
    # Hiperparámetros del modelo:
    parser.add_argument('--d-dim', type=int, default=7,
                        help='number of source domain?')    # numero de dominios de origen
    parser.add_argument('--x-dim', type=int, default=416,
                        help='input size after flattening') # tamaño de entrada despues del aplanamiento
    parser.add_argument('--y-dim', type=int, default=6,
                        help='number of classes')   # Tamaño del espacio latente: 6 clases de gestos
    parser.add_argument('--zd-dim', type=int, default=128,
                        help='size of latent space 1')  # tamaño del espacio latente 1
    parser.add_argument('--zx-dim', type=int, default=128,
                        help='size of latent space 2')  # tamaño del espacio latente 2
    parser.add_argument('--zy-dim', type=int, default=128,
                        help='size of latent space 3')  # tamaño del espacio latente 3

    # auxiliary multipliers:
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

    # Warm-up parameters
    parser.add_argument('-w', '--warmup', type=int, default=100, metavar='N',
                        help='number of epochs for warm-up. Set to 0 to turn warmup off.')
    parser.add_argument('--max_beta', type=float, default=1., metavar='MB',
                        help='max beta for warm-up')
    parser.add_argument('--min_beta', type=float, default=0.0, metavar='MB',
                        help='min beta for warm-up')
    # Output path
    parser.add_argument('--outpath', type=str, default='./',
                        help='where to save')

    # Parseo de los argumentos
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

    # Set seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # set work method
    work_method = 'train'
    # work_method = 'test'

    print(work_method)
    # ------------------------------------------------------------------------------------------------- #
    # train
    # ------------------------------------------------------------------------------------------------- #
    if work_method == 'train':
        # repetir el experimento con diferentes semillas
        for seed in range(10):
            accuracy = []
            args.seed = seed
            print("*" * 30)
            print('repeat{}'.format(args.seed))
            print("*" * 30)
            # entrenar y probar con cada sujeto como sujeto de prueba
            for i in range(8):
                # elegir los dominios de entrenamiento
                args.list_test_domain = i
                all_training_domains = [0, 1, 2, 3, 4, 5, 6, 7]
                all_training_domains.remove(args.list_test_domain)
                args.list_train_domains = all_training_domains
                print("Train_Subject:", args.list_test_domain, args.list_train_domains)
                # nombre del modelo
                args.list_test_domain = [args.list_test_domain]
                print(args.outpath)
                model_name = './saved_model/' + 'less0_test_domain_' + str(
                    args.list_test_domain[0]) + '_diva_seed_' + str(
                    args.seed)
                print("Test_Subject:", model_name)
                # cargar datos supervisados
                train_loader = data_utils.DataLoader(
                    semgdata_load(args.list_train_domains, args.list_test_domain, args.num_supervised, args.seed,
                                 './dataset/',
                                 train=True),
                    batch_size=args.batch_size,
                    shuffle=True, **kwargs)   
                # Inicializar el modelo DIVA
                model = DIVA(args).to(device)
                # Optimizer
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                # parametros de early stopping
                best_loss = 10000.  # 1000
                best_y_acc = 0.
                early_stopping_counter = 1
                max_early_stopping = 10

                # loop de entrenamiento
                print('\nStart training:', args)
                for epoch in range(1, args.epochs + 1):
                    # hiperparametros de beta
                    model.beta_d = 3
                    model.beta_y = 3
                    model.beta_x = 3
                    # train
                    avg_epoch_losses_sup, avg_epoch_class_y_loss = train(train_loader, model, optimizer, epoch)
                    # reguistrar las perdidas y la precision de la clasificacion
                    str_loss_sup = avg_epoch_losses_sup
                    str_print = "{} epoch: avg loss {}".format(epoch, str_loss_sup) 
                    str_print += ", class y loss {}".format(avg_epoch_class_y_loss) 
                    # estos test de precision son solo para registro, no se usan para tomar decisiones durante el entrenamiento
                    train_accuracy_d, train_accuracy_y = get_accuracy(train_loader, model.classifier, args.batch_size)
                    str_print += "     train accuracy d {}".format(train_accuracy_d)  # domain classification accuracy
                    str_print += ", y {}".format(train_accuracy_y)  # label classification accuracy
                    str_print += ",beta_d{},y{},x{}".format(model.beta_d, model.beta_y, model.beta_x)
                    print(str_print)
                    # esarly stopping basado en la precision de clasificacion y
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
                # cargar datos de prueba supervisados
                test_loader = data_utils.DataLoader(
                    semgdata_load(args.list_train_domains, args.list_test_domain, args.num_supervised, args.seed,
                                 './saved_model/diva',
                                 train=False),
                    batch_size=args.batch_size,
                    shuffle=True, **kwargs)
                # cargar el mejor modelo
                model = DIVA(args).to(device)
                model = torch.load(model_name + '.model')
                # estos test de precision son solo para registro, no se usan para tomar decisiones durante el entrenamiento
                test_accuracy_d, test_accuracy_y = get_accuracy(test_loader, model.classifier, args.batch_size)
                print("test accuracy y {}".format(test_accuracy_y))
                accuracy.append(test_accuracy_y)
            print(args.seed, accuracy)
    # ------------------------------------------------------------------------------------------------- #
    # test
    # ------------------------------------------------------------------------------------------------- #
    if work_method == 'test':
        acc_all_repeat = []
        for seed in range(7):
            args.seed = seed
            print("*" * 30)
            print('repeat{}'.format(args.seed))
            acc_in_repeat = []
            for i in range(8):
                # elegir los dominios de entrenamiento
                args.list_test_domain = i
                all_training_domains = [0, 1, 2, 3, 4, 5, 6, 7]
                all_training_domains.remove(args.list_test_domain)
                args.list_train_domains = all_training_domains
                print("Test_Subject:", args.list_test_domain)
                # nombre del modelo
                args.list_test_domain = [args.list_test_domain]
                model_name = './saved_model/' + 'less0_test_domain_' + str(
                    args.list_test_domain[0]) + '_diva_seed_' + str(
                    args.seed)
                # Cargar datos de prueba supervisados
                test_loader = data_utils.DataLoader(
                    semgdata_load(args.list_train_domains, args.list_test_domain, args.num_supervised, args.seed,
                                 './dataset/',
                                 train=False),
                    batch_size=args.batch_size,
                    shuffle=True, **kwargs)
                # Cargar el mejor modelo
                model = DIVA(args).to(device)
                model = torch.load(model_name + '.model')
                # estos test de precision son solo para registro, no se usan para tomar decisiones durante el entrenamiento
                test_accuracy_d, test_accuracy_y = get_accuracy(test_loader, model.classifier, args.batch_size)
                print("test accuracy y {}".format(test_accuracy_y))
                acc_in_repeat.append(test_accuracy_y)
            acc_all_repeat.append(acc_in_repeat)
        acc_all_repeat = np.array(acc_all_repeat)
        pd.DataFrame(acc_all_repeat).to_excel('diva_without_d.xlsx', sheet_name='0', index=False)