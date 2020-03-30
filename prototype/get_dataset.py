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
def get_one_label_dataset(class_label, train_dataset, test_dataset):
    train = get_one_label_data(class_label, train_dataset)
    test = get_one_label_data(class_label, test_dataset)
    return train, test

# 特定クラスラベルリストのtrain, testデータセットを取得
def get_specific_label_dataset(class_label_list, train_dataset, test_dataset):
    new_train = []
    new_test = []
    for label in class_label_list:
        train, test = get_one_label_dataset(label, train_dataset, test_dataset)
        new_train += train
        new_test += test
    return new_train, new_test
