from model import chainer_cifar10
import args
import get_dataset
import train as tr
import log
import sys

# 学習対象のネットワークを指定
net = chainer_cifar10.Cifar10_Conv6()
# 使用するデータのクラスを指定
class_label_list = [1,2,8]

# モデル名の確認
def check_model_name(model_name):
    if model_name == "v1":
        while (1):
            val = input("The model name is 'v1', is it really OK? (y/n): ")
            if val == "y" or val == "yes":
                print("running")
                break
            elif val == "n" or val == "no":
                print("finished")
                sys.exit()


if __name__=="__main__":
    # コマンドライン引数を取得
    args = args.get_args()
    check_model_name(args.model_name)
    # データ生成
    train, test = get_dataset.load_cifar10()
    train, test = get_dataset.get_specific_label_dataset(class_label_list, train, test, args.data_n)
    # 訓練
    tr.train(args, net, train, test)
    # パラメータの記録
    log.save_param(args, net.__class__.__name__, class_label_list, (train,test))
