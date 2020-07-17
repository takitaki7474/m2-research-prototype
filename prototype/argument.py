import argparse

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
