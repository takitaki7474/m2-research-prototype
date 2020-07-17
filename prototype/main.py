import os
import configparser
import sys
import torch
import argument
from utils import check
import dataselector
import lr_patterns
from model import target_models
import paramlogging
import plot
import training

if __name__=="__main__":
    inifile = configparser.SafeConfigParser()
    inifile.read('./settings.ini')
    result_target_dir = inifile.get("OutputDirectories", "result_targetmodel")
    args = argument.get_args()
    classes = [1,2,8]
    net = target_models.LeNet(len(classes))
    # 保存ディレクトリの生成
    if not check.should_overwrite_model(os.path.join(result_target_dir, args.model_name)):
        print("Execution interruption")
        sys.exit()
    dir = os.path.join(result_target_dir, args.model_name)
    if not os.path.isdir(dir): os.mkdir(dir)
    # モデルの初期設定のログ
    paramlogging.log_setting(args, net, classes, os.path.join(result_target_dir, args.model_name, "model_setting.json"))
    # データ選択
    train, test = dataselector.load_cifar10(inifile.get("OutputDirectories", "download"))
    data_selector = dataselector.DataSelector(train, test)
    data_selector.select_data_by_labels(classes)
    data_selector.randomly_select_data_by_label(args.train_num, train=True)
    data_selector.randomly_select_data_by_label(args.test_num, train=False)
    data_selector.update_labels()
    data_selector.print_len_by_label()
    train, test = data_selector.get_dataset()
    # 学習モデルの訓練
    logging = paramlogging.ParamLogging()
    net, record = training.train(train, test, net, args.epoch, args.batch_size, args.lr, lr_scheduling=lr_patterns.lr_v1, logging=logging)
    logging.save(os.path.join(result_target_dir, args.model_name, "log.json"))
    # 学習結果のplot
    plot.plot_loss(record["train_loss"], record["test_loss"], os.path.join(result_target_dir, args.model_name, "loss.png"))
    plot.plot_acc(record["train_acc"], record["test_acc"], os.path.join(result_target_dir, args.model_name, "accuracy.png"))
    # 学習モデルの保存
    torch.save(net.state_dict(), os.path.join(inifile.get("OutputDirectories", "learned_targetmodel"), args.model_name + ".pth"))
