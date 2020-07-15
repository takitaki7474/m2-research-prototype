import random
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
        print("訓練データ数:{0:>9}".format(len(self.train)))
        print("テストデータ数:{0:>9}".format(len(self.test)))

    def print_len_by_label(self):
    texts = ["訓練データ", "テストデータ"]
    for i, t in enumerate([self.train, self.test]):
        dic = defaultdict(int)
        for data in t:
            dic[data[1]] += 1
        print("{0}数  ----------------------------------------------".format(texts[i]))
        count = 0
        for key in dic.keys():
            print("ラベル:  {0}    データ数:  {1}".format(key, dic[key]))
            count += dic[key]
        print("合計データ数:  {0}\n".format(count))

    def select_train_at_random(self, data_num, shuffle=False):
        if shuffle == True:
            self.train = random.sample(self.train, data_num)
        elif shuffle == False:
            self.train = self.train[:data_num]

    def select_test_at_random(self, data_num, shuffle=False):
        if shuffle == True:
            self.test = random.sample(self.test, data_num)
        elif shuffle == False:
            self.test = self.test[:data_num]

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
