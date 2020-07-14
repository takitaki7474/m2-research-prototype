import argparse

# コマンドライン引数をパース
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', default='v1', type=str, help='This is model name')
    parser.add_argument('--train_n', '-tr_n', default=None, type=int, help='This is number of train data in one class')
    parser.add_argument('--test_n', '-te_n', default=None, type=int, help='This is number of test data in one class')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='This is epoch')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='This is batch size')
    parser.add_argument('--alpha', '-a', type=float, default=0.001, help='This is learning rate')
    parser.add_argument('--gpu_id', '-gpu', type=int, default=-1, help='This is gpu id')
    args = parser.parse_args()
    return args
