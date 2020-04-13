import random
import torch
import torchvision
import torchvision.transforms as transforms

# Cifar10データセットをdata_pathにダウンロード
def get_cifar10(data_path):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    return train, test

# データセットからclass_labelのデータセットを取得
# ex. class_label=1, dataset=[(data, 2),(data, 3),(data, 0) ...]
# get_one_label_dataset で利用
def get_one_label_data(class_label, dataset):
    new_data = []
    for data in dataset:
        label = data[1]
        if label == class_label:
            new_data.append(data)
    return new_data

# train, testの両データセットからclass_labelのデータセットを取得
# ex. class_label=1, train=[(data, 2),(data, 3),(data, 0) ...], test=[(data, 2),(data, 3),(data, 0) ...]
# get_specific_label_dataset で利用
def get_one_label_dataset(class_label, train, test):
    train = get_one_label_data(class_label, train)
    test = get_one_label_data(class_label, test)
    return train, test

# データセットのラベルをnew_labelに変更
# ex. new_label=1, dataset=[(data, 2),(data, 3),(data, 0) ...]
# change_label_dataset で利用
def change_label_data(new_label, dataset):
    new_data = []
    for data in dataset:
        new_data.append((data[0], new_label))
    return new_data

# train, testの両データセットのラベルをnew_labelに変更
# ex. new_label=1, train=[(data, 2),(data, 3),(data, 0) ...], test=[(data, 2),(data, 3),(data, 0) ...]
# get_specific_label_dataset で利用
def change_label_dataset(new_label, train, test):
    train = change_label_data(new_label, train)
    test = change_label_data(new_label, test)
    return train, test

# data_n(1クラスのデータ数)分のデータをランダムに取得
# ex. data_n=50, dataset=[(data, 2),(data, 3),(data, 0) ...]
# get_specific_label_dataset で利用
def choice_data_at_random(data_n, dataset):
    dataset = random.sample(dataset, data_n)
    return dataset

# train, testの両データセットからclass_label_listのデータセットを取得
# ex. class_label_list=[3,4,5], train=[(data, 2),(data, 3),(data, 0) ...], test=[(data, 2),(data, 3),(data, 0) ...], data_n=50
def get_specific_label_dataset(class_label_list, train, test, data_n=None):
    new_train = []
    new_test = []
    for i, label in enumerate(class_label_list):
        got_train, got_test = get_one_label_dataset(label, train, test)
        got_train, got_test = change_label_dataset(i, got_train, got_test)
        if data_n != None:
            got_train = choice_data_at_random(data_n, got_train)
        new_train += got_train
        new_test += got_test
    return new_train, new_test
