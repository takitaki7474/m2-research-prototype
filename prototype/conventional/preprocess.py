import configparser
from modules import preprocessor as pre



inifile = configparser.SafeConfigParser()
inifile.read("./settings.ini")

TRAIN_CLASSES = [0,1,2,3]

# 実験データの保存ディレクトリ設定
DATA_DIR = inifile.get("InputDataDir", "data_dir")

# データセットテーブルの保存ディレクトリ設定
TRAIN_DATASET_TABLE_PATH = "./dataset_table/train_dt4.db"
TEST_DATASET_TABLE_PATH = "./dataset_table/test_dt4.db"



if __name__=="__main__":

    # CIFAR10データセットの読み込み
    train, test = pre.download_cifar10(DATA_DIR)

    # データの前処理 (学習対象ラベルの選択, ラベルを連番に更新, データセットをDataFrame形式に変換)
    preprocessor = pre.DatasetPreprocessor(train, test)
    preprocessor.select_by_label(TRAIN_CLASSES)
    preprocessor.update_labels()
    preprocessor.show_labels()
    train_df, test_df = preprocessor.out_datasets(dataframe=True)

    # DataFrame形式のデータセットの保存
    pre.save(train_df, savepath=TRAIN_DATASET_TABLE_PATH)
    pre.save(test_df, savepath=TEST_DATASET_TABLE_PATH)
