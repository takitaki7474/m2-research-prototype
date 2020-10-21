import configparser
from modules import preprocessor as pre

TRAIN_CLASSES = [1,2,8]

inifile = configparser.SafeConfigParser()
inifile.read("./settings.ini")
DATA_DIR = inifile.get("InputDataDir", "data_dir")

if __name__=="__main__":

    train, test = pre.download_cifar10(DATA_DIR)

    preprocessor = pre.DatasetPreprocessor(train, test)
    preprocessor.select_by_label(TRAIN_CLASSES)
    preprocessor.update_labels()
    preprocessor.show_labels()
    train_df, test_df = preprocessor.out_datasets(dataframe=True)

    pre.save(train_df, savepath="./dataset_table/train_dt.db")
    pre.save(test_df, savepath="./dataset_table/test_dt.db")
