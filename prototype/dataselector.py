import random
from collections import defaultdict
import torch
import torchvision
import torchvision.transforms as transforms

# Cifar10データセットをdownlaod_pathにダウンロード
def load_cifar10(download_path):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train = torchvision.datasets.CIFAR10(root=download_path, train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root=download_path, train=False, download=True, transform=transform)
    return train, test

class DataSelector:

    def __init__(self, train, test):
        self.train = list(train)
        self.test = list(test)

    def get_dataset(self):
        dataset = (self.train, self.test)
        return dataset

    def print_len(self):
        print("\n------------------------------------------------------")
        print("訓練データ数:{0:>9}".format(len(self.train)))
        print("テストデータ数:{0:>9}".format(len(self.test)))

    def print_len_by_label(self):
        texts = ["訓練データ", "テストデータ"]
        for i, t in enumerate([self.train, self.test]):
            dic = defaultdict(int)
            for data in t:
                dic[data[1]] += 1
            print("\n{0}数  ----------------------------------------------".format(texts[i]))
            dic = sorted(dic.items())
            count = 0
            for key, value in dic:
                print("ラベル:  {0}    データ数:  {1}".format(key, value))
                count += value
            print("合計データ数:  {0}".format(count))

    def randomly_select_data_by_label(self, data_num, train=True):
        selected = []
        dataset = self.train if train == True else self.test
        dic = defaultdict(int)
        for data in dataset:
            if dic[data[1]] < data_num:
                selected.append(data)
                dic[data[1]] += 1
        if train == True:
            self.train = selected
        else:
            self.test = selected

    def __select_data_by_label(self, label):
        selected = [[], []]
        dataset = (self.train, self.test)
        for i, t in enumerate(dataset):
            for data in t:
                if data[1] == label:
                    selected[i].append(data)
        return selected

    def select_data_by_labels(self, labels):
        selected = [[], []]
        for label in labels:
            result = self.__select_data_by_label(label)
            selected[0] += result[0]
            selected[1] += result[1]
        self.train, self.test = selected

    def update_labels(self):
        updated = [[], []]
        label_mapping = defaultdict(lambda: -1)
        for i, t in enumerate([self.train, self.test]):
            t = sorted(t, key=lambda x:x[1])
            new_label = 0
            for data in t:
                if label_mapping[data[1]] == -1:
                    label_mapping[data[1]] = new_label
                    new_label += 1
                updated[i].append((data[0], label_mapping[data[1]]))
        self.train, self.test = updated
        print("\nラベルを変更しました.  ----------------------------------------------")
        for key, value in label_mapping.items():
            print("ラベル  {0} -> {1}".format(key, value))
