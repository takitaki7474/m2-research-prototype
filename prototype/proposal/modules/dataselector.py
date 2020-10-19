import faiss
import json
import numpy as np
import pandas as pd
import random
import sqlite3
import sys
import torch
from typing import List, Tuple, TypeVar, Dict

Dataframe = TypeVar("pandas.core.frame.DataFrame")




# feature_table_indexes(ft_indexes) をランダムに初期化
def init_feature_table_indexes(feature_table: Dataframe, seed=1) -> Dict[str, List[int]]:
    random.seed(seed)
    ft_indexes = {} # feature_table_indexes
    ft_indexes["selected"], ft_indexes["query"], ft_indexes["selected_query"] = [], [], []
    labels = sorted(feature_table["label"].unique())
    ft_labelby = feature_table.groupby("label")
    for label in labels:
        df = ft_labelby.get_group(label)
        queries = df["index"].values.tolist()
        query = random.sample(queries, 1)[0]
        ft_indexes["query"].append(query)
    return ft_indexes

# feature_table_indexes(ft_indexes) をjson形式で保存
def save_feature_table_indexes(ft_indexes: Dict[str, List[int]], savepath="./ft_indexes_v1.json"):
    ft_indexes["selected"] = [int(index) for index in ft_indexes["selected"]]
    ft_indexes["query"] = [int(index) for index in ft_indexes["query"]]
    ft_indexes["selected_query"] = [int(index) for index in ft_indexes["selected_query"]]
    with open(savepath, "w") as f:
        json.dump(ft_indexes, f, indent=4)

# feature_table_indexes(ft_indexes) を辞書形式で読み込み
def load_feature_table_indexes(path: str) -> Dict[str, List[int]]:
    with open(path, "r") as f:
        ft_indexes = json.load(f)
    return ft_indexes




class DataSelector:
    def __init__(self, dataset_table: Dataframe, dataset_table_indexes: Dict[str, List[int]]):
        self.dt = dataset_table
        self.dt_indexes = dataset_table_indexes
        self.labels = sorted(dataset_table["label"].unique())
        # 選択済みのデータを削除済みのdataset_table
        self.dropped_dt = self.__drop_selected_data(dataset_table, dataset_table_indexes)

    def __drop_selected_data(self, dataset_table: Dataframe, dataset_table_indexes: Dict[str, List[int]]) -> Dataframe:
        dt = dataset_table.drop(index=dataset_table_indexes["selected"])
        dt = dt.reset_index(drop=True)
        return dt

    def randomly_add(self, dataN: int, seed=None) -> Dict[str, List[int]]:
        dt_labelby = self.dropped_dt.groupby("label")
        for label in self.labels:
            df = dt_labelby.get_group(label)
            df = df.sample(n=dataN, random_state=seed)
            self.dt_indexes["selected"] += list(df["index"].values)
        return self.dt_indexes

    def out_dataset_table_indexes(self) -> Dict[str, List[int]]:
        return self.dt_indexes

    def out_selected_dataset(self) -> List[Tuple]:
        selected_dataset = []
        for index in self.dt_indexes["selected"]:
            irow = self.dt[self.dt["index"]==index]
            image = json.loads(irow["image"].iloc[0])
            image = np.array(image)
            image = torch.from_numpy(image.astype(np.float32)).clone()
            label = irow["label"].iloc[0]
            selected_dataset.append((image, label))
        return selected_dataset




class FeatureSelector(DataSelector):
    def __init__(self, feature_table: Dataframe, feature_table_indexes: Dict[str, List[int]]):
        super().__init__(feature_table, feature_table_indexes)
        self.faiss_indexes = {} # ラベルごとのフィーチャ全体のfaissインデックス

    # フィーチャを検索するための,フィーチャ全体のfaissインデックスを作成
    def make_faiss_indexes(self):
        dt_labelby = self.dropped_dt.groupby("label")
        for label in self.labels:
            features = []
            df = dt_labelby.get_group(label)
            for feature in df["feature"]:
                features.append(json.loads(feature))
            features = np.array(features).astype("float32")
            dim = len(features[0])
            index = faiss.IndexFlatL2(dim)
            index.add(features)
            self.faiss_indexes[label] = index

    # ラベルごとにクエリと最近傍(NN)のフィーチャをdataN分選択し、選択したフィーチャのdt_indexesをdt_indexes["selected"]に追加
    def add_NN(self, dataN: int) -> Dict[str, List[int]]:
        if len(self.faiss_indexes) is 0:
            print("\nPlease run the process to make faiss indexes in advance.")
            sys.exit()
        dt_labelby = self.dropped_dt.groupby("label")
        for label in self.labels:
            index = self.faiss_indexes[label]
            k = index.ntotal # 検索対象データ数
            query = self.dropped_dt[self.dropped_dt["index"]==self.dt_indexes["query"][label]]["feature"].iat[0] # queryに指定されたフィーチャを取得
            query = json.loads(query)
            query = np.array([query]).astype("float32")
            D, I = index.search(query, k)
            self.dt_indexes["selected_query"].append(self.dt_indexes["query"][label])
            for i in I[0][:dataN + 1]:
                train_index = dt_labelby.get_group(label).iloc[i]["index"]
                if train_index == self.dt_indexes["query"][label]: continue # クエリはselectedに含めない
                self.dt_indexes["selected"].append(train_index)
        return self.dt_indexes

    # クエリを最遠傍点(FP)に更新
    def update_FP_queries(self) -> Dict[str, List[int]]:
        if len(self.faiss_indexes) is 0:
            print("\nPlease run the process to make faiss indexes in advance.")
            sys.exit()
        dt_labelby = self.dropped_dt.groupby("label")
        for label in self.labels:
            index = self.faiss_indexes[label]
            k = index.ntotal # 検索対象データ数
            query = self.dropped_dt[self.dropped_dt["index"]==self.dt_indexes["query"][label]]["feature"].iat[0] # queryに指定されたフィーチャを取得
            query = json.loads(query)
            query = np.array([query]).astype("float32")
            D, I = index.search(query, k)
            for i in reversed(I[0]):
                FP_query_index = dt_labelby.get_group(label).iloc[i]["index"]
                if FP_query_index not in self.dt_indexes["selected_query"]:
                    break
            self.dt_indexes["query"][label] = FP_query_index
        return self.dt_indexes
