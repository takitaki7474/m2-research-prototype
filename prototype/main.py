from model import chainer_cifar10
import args
import get_dataset
import train as tr

# 学習対象のネットワークを指定
net = chainer_cifar10.Cifar10()
# 使用するデータのクラスを指定
class_label_list = [0,1,2]

if __name__=="__main__":
    # コマンドライン引数を取得
    args = args.get_args()
    # データ生成
    train, test = get_dataset.load_cifar10()
    train, test = get_dataset.get_specific_label_dataset(class_label_list, train, test)
    # 訓練
    tr.train(args, net, train, test)
