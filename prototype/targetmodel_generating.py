# 本番学習モデルの生成
import os
import configparser
import sys
import torch
import argument
from utils import check
import data_selector
import lr_patterns
from model import target_models
import paramlogging
import plot
import training

if __name__=="__main__":
    inifile = configparser.SafeConfigParser()
    inifile.read('./settings.ini')
    result_save_dir = inifile.get("OutputDirectories", "result_targetmodel")
    learned_save_dir = inifile.get("OutputDirectories", "learned_targetmodel")
    args = argument.get_args()
    classes = [1,2,8]
    net = target_models.LeNet(len(classes))
    # 保存ディレクトリの生成
    if not check.should_overwrite_model(os.path.join(result_save_dir, args.model_name)):
        print("Execution interruption")
        sys.exit()
    dir = os.path.join(result_save_dir, args.model_name)
    if not os.path.isdir(dir): os.mkdir(dir)
    # モデルの初期設定のログ
    paramlogging.log_setting(args, net, classes, os.path.join(result_save_dir, args.model_name, "model_setting.json"))
    # データ選択
    train, test = data_selector.load_cifar10(inifile.get("OutputDirectories", "download"))
    selector = data_selector.DataSelector(train, test)
    selector.select_data_by_labels(classes)
    selector.randomly_select_data_by_label(args.train_num, train=True)
    selector.randomly_select_data_by_label(args.test_num, train=False)
    selector.update_labels()
    selector.print_len_by_label()
    train, test = selector.get_dataset()
    # 学習モデルの訓練
    logging = paramlogging.ParamLogging()
    net, record = training.train(train, test, net, args.epoch, args.batch_size, args.lr, lr_scheduling=lr_patterns.lr_v1, logging=logging)
    logging.save(os.path.join(result_save_dir, args.model_name, "log.json"))
    # 学習結果のplot
    plot.plot_loss(record["train_loss"], record["test_loss"], os.path.join(result_save_dir, args.model_name, "loss.png"))
    plot.plot_acc(record["train_acc"], record["test_acc"], os.path.join(result_save_dir, args.model_name, "accuracy.png"))
    # 学習モデルの保存
    torch.save(net.state_dict(), os.path.join(learned_save_dir, args.model_name + ".pth"))
