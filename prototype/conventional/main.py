import configparser
import os
import sys

from utils import alert
from modules import argument
from modules import dataselector
from modules import preprocessor as pre
from model import models

TRAIN_CLASSES = [1,2,8]
MODEL = models.LeNet(len(TRAIN_CLASSES))

inifile = configparser.SafeConfigParser()
inifile.read("./settings.ini")
RESULT_DIR = inifile.get("TrainResultDirs", "result_dir")
LEARNED_DIR = inifile.get("TrainResultDirs", "learned_dir")
DATA_DIR = inifile.get("InputDataDir", "data_dir")

if __name__=="__main__":

    args = argument.get_args()
    # 既に同名のモデルが存在する場合、上書きするかどうか確認
    check_path = os.path.join(RESULT_DIR, args.model_name)
    if not alert.should_overwrite_model(check_path): sys.exit()
    # 学習結果の保存ディレクトリを生成
    result_dir = os.path.join(RESULT_DIR, args.model_name)
    if not os.path.isdir(result_dir): os.mkdir(result_dir)
    # 学習の設定値を記録
    settings_path = os.path.join(result_dir, "settings.json")
    net = MODEL.__class__.__name__
    argument.save_args(args, settings_path, net=net)

    # dataset_table_indexesの読み込み
    if args.dt_indexes_ver is None:
        train_dt_indexes = dataselector.init_dataset_table_indexes()
        test_dt_indexes = dataselector.init_dataset_table_indexes()
    else:
        loadpath = os.path.join(RESULT_DIR, args.dt_indexes_ver, "train_dt_indexes.json")
        train_dt_indexes = dataselector.load_dataset_table_indexes(path=loadpath)
        loadpath = os.path.join(RESULT_DIR, args.dt_indexes_ver, "test_dt_indexes.json")
        test_dt_indexes = dataselector.load_dataset_table_indexes(path=loadpath)

    # 訓練データの選択
    train_dt = pre.load(dbpath="./dataset_table/train_dt.db")
    selector = dataselector.DataSelector(train_dt, train_dt_indexes)
    train_dt_indexes = selector.randomly_add(dataN=args.train, seed=args.seed)
    train = selector.out_selected_dataset()
    savepath = os.path.join(RESULT_DIR, args.model_name, "train_dt_indexes.json")
    dataselector.save_dataset_table_indexes(train_dt_indexes, savepath=savepath)

    # テストデータの選択
    test_dt = pre.load(dbpath="./dataset_table/test_dt.db")
    selector = dataselector.DataSelector(test_dt, test_dt_indexes)
    test_dt_indexes = selector.randomly_add(dataN=args.test, seed=args.seed)
    test = selector.out_selected_dataset()
    savepath = os.path.join(RESULT_DIR, args.model_name, "test_dt_indexes.json")
    dataselector.save_dataset_table_indexes(test_dt_indexes, savepath=savepath)
