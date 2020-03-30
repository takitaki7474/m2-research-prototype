import numpy as np
import chainer

# cifar10の読み込み
def load_cifar10():
    (train, test) = chainer.datasets.get_cifar10()
    return train, test

# 特定クラスラベル画像の取得
def get_specific_img(class_label, dataset):
    specific_img = []
    for data in dataset:
        label = data[1]
        if label == class_label:
            specific_img.append(data)
    return specific_img
