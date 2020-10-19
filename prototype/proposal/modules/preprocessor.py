from collections import defaultdict
import json
import pandas as pd
import sqlite3
from typing import List, Tuple, TypeVar
import torchvision
import torchvision.transforms as transforms

Torchvision = TypeVar("torchvision.datasets.cifar.CIFAR10")
Dataframe = TypeVar("pandas.core.frame.DataFrame")

def download_cifar10(savepath: str) -> Tuple[Torchvision, Torchvision]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train = torchvision.datasets.CIFAR10(root=savepath, train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root=savepath, train=False, download=True, transform=transform)
    return train, test

# DataFrame形式のdatasetをsqliteに保存
def save(dataset: Dataframe, savepath="./ft.db", tablename="feature_table"):
    conn = sqlite3.connect(savepath)
    c = conn.cursor()
    dataset.to_sql(tablename, conn, if_exists='replace')
    conn.close()

# datasetをsqliteからDataFrame形式で読み込み
def load(dbpath="./ft.db", tablename="feature_table") -> Dataframe:
    conn=sqlite3.connect(dbpath)
    c = conn.cursor()
    dataset = pd.read_sql('SELECT * FROM ' + tablename, conn)
    return dataset


class DatasetPreprocessor:

    def __init__(self, train: List[Tuple], test: List[Tuple]):
        self.train = train
        self.test = test

    def out_datasets(self, dataframe=False) -> Tuple[List[Tuple], List[Tuple]] or Tuple[Dataframe, Dataframe]:
        if dataframe == False:
            return self.train, self.test
        elif dataframe == True:
            train_df = self.__convert_dataframe(self.train)
            test_df = self.__convert_dataframe(self.test)
            return train_df, test_df

    def show_length(self):
        print("\n------------------------------------------------------------")
        print("Number of train data:{0:>9}".format(len(self.train)))
        print("Number of test data:{0:>9}".format(len(self.test)))

    def show_labels(self):
        texts = ["Number of train data", "Number of test data"]
        for i, dataset in enumerate([self.train, self.test]):
            label_count = defaultdict(int)
            for data in dataset:
                label_count[data[1]] += 1
            print("\n{0}  ----------------------------------------------".format(texts[i]))
            label_count = sorted(label_count.items())
            sum = 0
            for label, count in label_count:
                print("label:  {0}    count:  {1}".format(label, count))
                sum += count
            print("total:  {0}".format(sum))

    # labelsに含まれるラベルのデータを選択
    def select_by_label(self, labels: List[int]):
        self.train = [data for data in self.train if data[1] in labels]
        self.test = [data for data in self.test if data[1] in labels]

    # 正解ラベルを0からの連番に更新
    def update_labels(self):
        updated = [[], []]
        label_mapping = defaultdict(lambda: -1)
        for i, dataset in enumerate([self.train, self.test]):
            dataset = sorted(dataset, key=lambda x:x[1])
            new_label = 0
            for data in dataset:
                if label_mapping[data[1]] == -1:
                    label_mapping[data[1]] = new_label
                    new_label += 1
                updated[i].append((data[0], label_mapping[data[1]]))
        self.train, self.test = updated
        print("\nChanged the label.  ----------------------------------------------")
        for old, new in label_mapping.items():
            print("label:  {0} -> {1}".format(old, new))

    # dataset(train, test) をDataFrame形式に変換
    def __convert_dataframe(self, dataset: List[Tuple]) -> Dataframe:
        dic = defaultdict(list)
        for image, label in dataset:
            dic["image"].append(json.dumps(image.cpu().numpy().tolist()))
            dic["label"].append(int(label))
        df = pd.DataFrame(dic)
        return df
