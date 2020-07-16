import argparse
import os
import torch
from model import models
import dataselector
from dataselector import DataSelector
import training

DOWNLOAD_DIR = "./data/"
RESULT_TARGET_DIR = "./result/target_model/"
RESULT_PREMODEL_DIR = "./result/pre_model/"
LEARNED_TARGET_DIR = "./learned/target_model/"
LEARNED_PREMODEL_DIR = "./learned/pre_model/"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-mn', default='v1', type=str, help='Name of model to save')
    parser.add_argument('--train_num', '-trnum', default=None, type=int, help='Number of training data by class label')
    parser.add_argument('--test_num', '-tenum', default=None, type=int, help='Number of test data by class label')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of epoch')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', '-lr', type=float, default=0.001, help='Initial learning rate')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    classes = [1,2,8]
    net = models.LeNet(len(classes))
    # 保存ディレクトリの生成
    dir = os.path.join(RESULT_TARGET_DIR, args.model_name)
    if not os.path.isdir(dir): os.mkdir(dir)
    # データ選択
    train, test = dataselector.load_cifar10(DOWNLOAD_DIR)
    data_selector = DataSelector(train, test)
    data_selector.select_data_by_labels(classes)
    data_selector.randomly_select_data_by_label(args.train_num, train=True)
    data_selector.randomly_select_data_by_label(args.test_num, train=False)
    data_selector.update_labels()
    data_selector.print_len_by_label()
    train, test = data_selector.get_dataset()
    # 学習モデルの訓練
    net, record = training.train(train, test, net, args.epoch, args.batch_size, args.lr)
    # 学習モデルの保存
    torch.save(net.state_dict(), os.path.join(LEARNED_TARGET_DIR, args.model_name + ".pth"))
