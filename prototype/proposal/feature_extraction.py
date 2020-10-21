import configparser
import os
import sys
import torch

from model import pre_models
from modules import argument, dataselector, training, plot
from modules.extractor import FeatureExtractor
from modules import preprocessor as pre
from utils import alert, tools, lr_patterns

TRAIN_CLASSES = [1,2,8]
MODEL = pre_models.PreLeNet(len(TRAIN_CLASSES))

inifile = configparser.SafeConfigParser()
inifile.read("./settings.ini")
RESULT_DIR = inifile.get("TrainResultDirs", "pre_result")
LEARNED_DIR = inifile.get("TrainResultDirs", "pre_learned")
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

    # dataset_table_indexesの初期化
    train_dt_indexes = dataselector.init_dataset_table_indexes()
    test_dt_indexes = dataselector.init_dataset_table_indexes()

    # 訓練データの選択
    train_dt = pre.load(dbpath="./dataset_table/train_dt.db")
    selector = dataselector.DataSelector(train_dt, train_dt_indexes)
    train_dt_indexes = selector.randomly_add(dataN=args.train, seed=args.seed)
    train = selector.out_selected_dataset()
    print("Number of train data: {0}".format(len(train)))

    # テストデータの選択
    test_dt = pre.load(dbpath="./dataset_table/test_dt.db")
    selector = dataselector.DataSelector(test_dt, test_dt_indexes)
    test_dt_indexes = selector.randomly_add(dataN=args.test, seed=args.seed)
    test = selector.out_selected_dataset()
    print("Number of test data: {0}".format(len(test)))

    trainloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    # 学習と評価
    log_path = os.path.join(RESULT_DIR, args.model_name, "log.json")
    model = training.process(trainloader, testloader, MODEL, args.epochs, args.lr, log_savepath=log_path)
    modelpath = os.path.join(LEARNED_DIR, args.model_name + ".pth")
    torch.save(model.state_dict(), modelpath)

    # 学習結果の可視化
    train_losses, test_losses, train_accs, test_accs = tools.load_train_log(path=log_path)
    savepath = os.path.join(RESULT_DIR, args.model_name, "loss.png")
    plot.plot_loss(train_losses, test_losses, savepath=savepath)
    savepath = os.path.join(RESULT_DIR, args.model_name, "accuracy.png")
    plot.plot_acc(train_accs, test_accs, savepath=savepath)

    # フィーチャ抽出と保存
    extractor = FeatureExtractor(model=model, dataloader=trainloader)
    extractor.make_feature_table()
    extractor.save_feature_table(savepath="./feature_tables/ft.db")
