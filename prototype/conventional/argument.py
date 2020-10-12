import argparse
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', '-mn', type=str, default='v1', help='name of model to train')
    parser.add_argument('--train', '-tr', type=int, default=None, help='number of train data by class label')
    parser.add_argument('--test', '-te', type=int, default=None, help='number of test data by class label')
    parser.add_argument('--batch-size', '-b', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', '-teb', type=int, default=128, help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', '-lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--seed', '-s', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()
    return args

def save_args(args, savepath="./args.json"):
    args_dict = vars(args)
    with open(savepath, "w") as f:
        json.dump(args_dict, f, indent=4)
