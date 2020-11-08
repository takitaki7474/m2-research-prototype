# 実行例
# python feature_extraction.py -mn v5 -tr 5000 -te 500 -e 100 -lr 0.01

import configparser
import os
import sys
import torch

from model import pre_models
from modules import argument, dataselector, training, plot
from modules.extractor import FeatureExtractor
from modules import preprocessor as pre
from utils import alert, tools, lr_patterns



args = argument.get_args()
inifile = configparser.SafeConfigParser()
inifile.read("./settings.ini")

# 学習モデルの設定
MODEL = pre_models.PreLeNet(4)

# データセットテーブルの読み込みディレクトリ設定
TRAIN_DATASET_TABLE_PATH = "./dataset_table/train_dt4.db"
TEST_DATASET_TABLE_PATH = "./dataset_table/test_dt4.db"

# フィーチャ抽出結果の保存ディレクトリ設定
FEATURE_TABLE_PATH = "./feature_table/ft4.db"

# 学習率の更新scheduleの設定
LR_SCHEDULING = lr_patterns.lr_v1
#LR_SCHEDULING = None

# 学習結果の保存ディレクトリ設定
RESULT_DIR = inifile.get("TrainResultDirs", "pre_result")
LEARNED_DIR = inifile.get("TrainResultDirs", "pre_learned")

# 実験データの保存ディレクトリ設定
DATA_DIR = inifile.get("InputDataDir", "data_dir")

# 本バージョンの学習結果の保存ディレクトリ設定
ISDIR_CHECK_DIR = os.path.join(RESULT_DIR, args.model_name)
LEARN_RESULT_DIR = os.path.join(RESULT_DIR, args.model_name)

# 学習設定値の保存ディレクトリ設定
LEARN_SETTINGS_PATH = os.path.join(RESULT_DIR, args.model_name, "settings.json")

# 学習結果の保存ディレクトリ設定
TRAIN_LOG_PATH = os.path.join(RESULT_DIR, args.model_name, "log.json")
MODEL_SAVE_PATH = os.path.join(LEARNED_DIR, args.model_name + ".pth")
LOSS_PLOT_RESULT_PATH = os.path.join(RESULT_DIR, args.model_name, "loss.png")
ACC_PLOT_RESULT_PATH = os.path.join(RESULT_DIR, args.model_name, "accuracy.png")



if __name__=="__main__":


    # 既に同名のモデルが存在する場合、上書きするかどうか確認
    if not alert.should_overwrite_model(ISDIR_CHECK_DIR): sys.exit()

    # 学習結果の保存ディレクトリを生成
    if not os.path.isdir(LEARN_RESULT_DIR): os.mkdir(LEARN_RESULT_DIR)


    # 訓練データの選択と生成
    train_dt_indexes = dataselector.init_dataset_table_indexes()
    train_dt = pre.load(dbpath=TRAIN_DATASET_TABLE_PATH)
    selector = dataselector.DataSelector(train_dt, train_dt_indexes)
    train_dt_indexes, _ = selector.randomly_add(dataN=args.add_train, seed=args.seed)
    train = selector.out_selected_dataset()
    print("Number of train data: {0}".format(len(train)))
    trainloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2)


    # テストデータの選択と生成
    test_dt_indexes = dataselector.init_dataset_table_indexes()
    test_dt = pre.load(dbpath=TEST_DATASET_TABLE_PATH)
    selector = dataselector.DataSelector(test_dt, test_dt_indexes)
    test_dt_indexes, _ = selector.randomly_add(dataN=args.add_test, seed=args.seed)
    test = selector.out_selected_dataset()
    print("Number of test data: {0}".format(len(test)))
    testloader = torch.utils.data.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, num_workers=2)


    # 学習と評価
    model = training.process(trainloader, testloader, MODEL, args.epochs, args.lr, lr_scheduling=LR_SCHEDULING, log_savepath=TRAIN_LOG_PATH)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # 学習結果の可視化
    train_losses, test_losses, train_accs, test_accs = tools.load_train_log(path=TRAIN_LOG_PATH)
    plot.plot_loss(train_losses, test_losses, savepath=LOSS_PLOT_RESULT_PATH)
    plot.plot_acc(train_accs, test_accs, savepath=ACC_PLOT_RESULT_PATH)

    # フィーチャ抽出と保存
    extractor = FeatureExtractor(model=model, dataloader=trainloader)
    extractor.make_feature_table()
    extractor.save_feature_table(savepath=FEATURE_TABLE_PATH)

    # 学習の設定値の記録
    net_name = MODEL.__class__.__name__
    argument.save_args(args, LEARN_SETTINGS_PATH, net=net_name, total_train=len(train), total_test=len(test))
