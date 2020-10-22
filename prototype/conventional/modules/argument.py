import argparse
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-mn', type=str, default='v1', help='name of model to train')
    parser.add_argument('--base_result_ver', '-bv', type=str, default=None, help='version of the learning result to base on (default None)')
    parser.add_argument('--add_train', '-tr', type=int, default=None, help='number of train data by class label')
    parser.add_argument('--add_test', '-te', type=int, default=None, help='number of test data by class label')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', '-teb', type=int, default=128, help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', '-lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--seed', '-s', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()
    return args

def save_args(args, savepath="./args.json", **kwargs):
    dic = vars(args)
    for k, v in kwargs.items():
        dic[k] = v
    with open(savepath, "w") as f:
        json.dump(dic, f, indent=4)
