import argument
import configparser
import os
import sys

from utils import alert
from model import models

TRAIN_CLASSES = [1,2,8]
MODEL = models.LeNet(len(TRAIN_CLASSES))

inifile = configparser.SafeConfigParser().read("./settings.ini")
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
