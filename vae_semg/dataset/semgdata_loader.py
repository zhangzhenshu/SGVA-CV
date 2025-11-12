import os
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


#--------------------- Cargar datos sEMG ---------------------#
class semgdata_load(data_utils.Dataset):
    # Contructor
    def __init__(self, list_train_domains, list_test_domain, num_supervised, mnist_subset, root, transform=None,
                 train=True, download=True):
        self.list_train_domains = list_train_domains
        self.list_test_domain = list_test_domain
        self.num_supervised = num_supervised
        self.mnist_subset = mnist_subset
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.download = download

        if self.train:
            self.train_data, self.train_labels, self.train_domain = self._get_data()
        else:
            self.test_data, self.test_labels, self.test_domain = self._get_data()

    # retorna los datos y las etiquetas
    def _get_data(self):
        """
        return
        train_data      n * 1*8*52
        train_label     n * num_class
        train_domains   n * num_domain
        """
        # cargar datos para entrenamiento o prueba
        if self.train:
            train_data_label_dict = np.load("./../dataset/8subs_6motions_5repeats_sEMG.npy",
                                        encoding="bytes", allow_pickle=True).item()
            original_train_data = train_data_label_dict["list_total_user_data"]
            original_train_label = train_data_label_dict['list_total_user_labels']
            new_train_data=[]
            new_train_label=[]
            # agregar datos de los sujetos de entrenamiento
            for i in self.list_train_domains:
                new_train_data.append(original_train_data[i])
                new_train_label.append(original_train_label[i])
            # combinar datos de los sujetos
            train_data = np.concatenate(new_train_data)
            train_label = np.concatenate(new_train_label)
            train_data = np.array(train_data, dtype=np.float32)
            # Crear etiquetas de dominio
            train_domains = torch.zeros(len(train_label))
            sum = 0
            for i in range(len(new_train_label)):
                length = len(new_train_label[i])
                train_domains[sum:sum + length] += i
                sum += length
            # Mezclar los datos
            inds = np.arange(len(train_label))
            np.random.shuffle(inds)
            train_data = torch.tensor(train_data[inds])
            train_label = torch.tensor(train_label[inds]).long()    # Etiquetas de clase
            train_domains = train_domains[inds].long()  # Etiquetas de dominio
            # Convertir a onehot
            d = torch.eye(7)  # Crear matriz diagonal
            train_domains = d[train_domains]  # Cada elemento en train_domains es reemplazado por la fila correspondiente en 
                                              # la matriz diagonal según su valor original (que representa el dominio)
            # Convertir a onehot
            y = torch.eye(6)
            train_label = y[train_label]

            return train_data.unsqueeze(1), train_label, train_domains  # (46475,1,8,52) (46475,7)  (46475,7)

        else:
            train_data_label_dict = np.load("./../dataset/8subs_6motions_35repeats_sEMG.npy",
                                       encoding="bytes", allow_pickle=True).item()
            original_train_data = train_data_label_dict["list_total_user_data"]
            original_train_label = train_data_label_dict['list_total_user_labels']
            original_train_data = original_train_data[self.list_test_domain[0]]
            original_train_label = original_train_label[self.list_test_domain[0]]
            # original_train_data = original_train_data.tolist()[self.list_test_domain[0]]
            # original_train_label = original_train_label.tolist()[self.list_test_domain[0]]
            train_data = torch.tensor(np.array(original_train_data, dtype=np.float32))
            train_label = torch.tensor(original_train_label).long()

            # Create domain labels
            train_domains = torch.zeros(len(train_label)).long()

            # Convert to onehot
            d = torch.eye(7)  # 创建对角矩阵
            train_domains = d[train_domains]  # train_domains中的每一个元素由该元素的原值（表示域）对应对角矩阵中的行向量代替

            # Convert to onehot
            y = torch.eye(6)
            train_label = y[train_label]

            return train_data.unsqueeze(1), train_label, train_domains

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)

    def __getitem__(self, index):
        if self.train:
            x = self.train_data[index]
            y = self.train_labels[index]
            d = self.train_domain[index]
        else:
            x = self.test_data[index]
            y = self.test_labels[index]
            d = self.test_domain[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y, d

