import chainer

# cifar10の読み込み
def load_cifar10():
    (train, test) = chainer.datasets.get_cifar10()
    return train, test

# 特定クラスラベル画像の取得
def get_one_label_data(class_label, dataset):
    new_data = []
    for data in dataset:
        label = data[1]
        if label == class_label:
            new_data.append(data)
    return new_data

# 特定クラスラベルのtrain, testデータセットを取得
def get_one_label_dataset(class_label, train, test):
    train = get_one_label_data(class_label, train)
    test = get_one_label_data(class_label, test)
    return train, test

# 特定クラスラベルリストのtrain, testデータセットを取得
def get_specific_label_dataset(class_label_list, train, test):
    new_train = []
    new_test = []
    for label in class_label_list:
        got_train, got_test = get_one_label_dataset(label, train, test)
        new_train += got_train
        new_test += got_test
    return new_train, new_test
