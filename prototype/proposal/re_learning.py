# 実行例
# python re_learning.py -mn v4 -bv v3 -tr 100 -te 10 -e 10 -eval 1

import configparser
import os
import sys
import torch

from model import domain_models
from modules import argument, dataselector, training, plot, eval_loss
from modules import preprocessor as pre
from utils import alert, tools, lr_patterns



# Avoid MacOS spec error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

args = argument.get_args()
inifile = configparser.SafeConfigParser()
inifile.read("./settings.ini")

# 学習モデルの設定
MODEL = domain_models.LeNet3(3)

# データセットテーブルの読み込みディレクトリ設定
FEATURE_TABLE_PATH = "./feature_table/ft.db"
TEST_DATASET_TABLE_PATH = "./dataset_table/test_dt.db"

# 学習率の更新scheduleの設定
LR_SCHEDULING = lr_patterns.lr_v1
#LR_SCHEDULING = None

# 学習結果の保存ディレクトリ設定
RESULT_DIR = inifile.get("TrainResultDirs", "domain_result")
LEARNED_DIR = inifile.get("TrainResultDirs", "domain_learned")

# 実験データの保存ディレクトリ設定
DATA_DIR = inifile.get("InputDataDir", "data_dir")

# 前回学習の訓練誤差収束速度の評価結果
ERR_SPEED_EVAL = args.err_speed_eval

# 本バージョンの学習結果の保存ディレクトリ設定
ISDIR_CHECK_DIR = os.path.join(RESULT_DIR, args.model_name)
LEARN_RESULT_DIR = os.path.join(RESULT_DIR, args.model_name)

# 学習設定値の保存ディレクトリ設定
LEARN_SETTINGS_PATH = os.path.join(RESULT_DIR, args.model_name, "settings.json")

# データセットテーブルインデックスの保存ディレクトリ設定
BASE_TRAIN_FT_INDEXES_PATH = os.path.join(RESULT_DIR, str(args.base_result_ver), "train_ft_indexes.json")
BASE_TEST_DT_INDEXES_PATH = os.path.join(RESULT_DIR, str(args.base_result_ver), "test_dt_indexes.json")
ADDED_TRAIN_FT_INDEXES_PATH = os.path.join(RESULT_DIR, args.model_name, "train_ft_indexes.json")
ADDED_TEST_DT_INDEXES_PATH = os.path.join(RESULT_DIR, args.model_name, "test_dt_indexes.json")

# 学習結果の保存ディレクトリ設定
TRAIN_LOG_PATH = os.path.join(RESULT_DIR, args.model_name, "log.json")
LOSS_PLOT_RESULT_PATH = os.path.join(RESULT_DIR, args.model_name, "loss.png")
ACC_PLOT_RESULT_PATH = os.path.join(RESULT_DIR, args.model_name, "accuracy.png")

# 訓練誤差収束速度の保存ディレクトリ設定
ERR_SPEED_LOAD_PATH = os.path.join(RESULT_DIR, str(args.base_result_ver), "err_speed.json")
ERR_SPEED_SAVE_PATH = os.path.join(RESULT_DIR, args.model_name, "err_speed.json")



if __name__=="__main__":

    # 既に同名のモデルが存在する場合、上書きするかどうか確認
    # if not alert.should_overwrite_model(ISDIR_CHECK_DIR): sys.exit()

    # 学習結果の保存ディレクトリを生成
    if not os.path.isdir(LEARN_RESULT_DIR): os.mkdir(LEARN_RESULT_DIR)

    """
    # 前回学習の訓練誤差収束速度の評価
    if args.base_result_ver is not None:
        err_speed = eval_loss.load(path=ERR_SPEED_LOAD_PATH)
        if err_speed >= args.err_speed_std: # 訓練誤差収束速度が基準値を満たした場合
            ERR_SPEED_EVAL = 1
    ERR_SPEED_EVAL = 1 # 実験用
    """


    # 訓練データの選択と生成
    train_ft = pre.load(dbpath=FEATURE_TABLE_PATH)
    if args.base_result_ver is None:
        train_ft_indexes = dataselector.init_feature_table_indexes(feature_table=train_ft)
    else:
        train_ft_indexes = dataselector.load_table_indexes(path=BASE_TRAIN_FT_INDEXES_PATH)
    selector = dataselector.FeatureSelector(train_ft, train_ft_indexes)
    selector.make_faiss_indexes()
    if ERR_SPEED_EVAL == 1: # 訓練誤差収束速度が基準を満たした場合
        _ = selector.update_FP_queries() # クエリを最遠傍点(FP)に更新
    train_ft_indexes, _ = selector.add_NN(dataN=args.add_train)
    train = selector.out_selected_dataset()
    dataselector.save_feature_table_indexes(train_ft_indexes, savepath=ADDED_TRAIN_FT_INDEXES_PATH)
    print("Number of train data: {0}".format(len(train)))
    trainloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2)


    # テストデータの選択と生成
    test_dt = pre.load(dbpath=TEST_DATASET_TABLE_PATH)
    if args.base_result_ver is None:
        test_dt_indexes = dataselector.init_dataset_table_indexes()
    else:
        test_dt_indexes = dataselector.load_table_indexes(path=BASE_TEST_DT_INDEXES_PATH)
    selector = dataselector.DataSelector(test_dt, test_dt_indexes)
    test_dt_indexes, _ = selector.randomly_add(dataN=args.add_test, seed=args.seed)
    test = selector.out_selected_dataset()
    dataselector.save_dataset_table_indexes(test_dt_indexes, savepath=ADDED_TEST_DT_INDEXES_PATH)
    print("Number of test data: {0}".format(len(test)))
    testloader = torch.utils.data.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, num_workers=2)


    # 学習と評価
    model = training.process(trainloader, testloader, MODEL, args.epochs, args.lr, lr_scheduling=LR_SCHEDULING, log_savepath=TRAIN_LOG_PATH)

    # 学習結果の可視化
    train_losses, test_losses, train_accs, test_accs = tools.load_train_log(path=TRAIN_LOG_PATH)
    plot.plot_loss(train_losses, test_losses, savepath=LOSS_PLOT_RESULT_PATH)
    plot.plot_acc(train_accs, test_accs, savepath=ACC_PLOT_RESULT_PATH)

    # 訓練誤差収束速度の評価
    train_err_speed = eval_loss.eval_err_speed(train_losses)
    test_err_speed = eval_loss.eval_err_speed(test_losses)
    eval_loss.save(savepath=ERR_SPEED_SAVE_PATH, train_err_speed=train_err_speed, test_err_speed=test_err_speed)
    print("train loss error speed:  {0}".format(train_err_speed))
    print("test loss error speed:  {0}".format(test_err_speed))

    # 学習の設定値の記録
    net_name = MODEL.__class__.__name__
    argument.save_args(args, LEARN_SETTINGS_PATH, net=net_name, total_train=len(train), total_test=len(test), train_err_speed=train_err_speed, test_err_speed=test_err_speed)
