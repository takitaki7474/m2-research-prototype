import copy
import faiss
import json
import numpy as np
import pandas as pd
import random
import sqlite3
import sys
import torch
from typing import TypeVar, List, Tuple, Dict

Dataframe = TypeVar("pandas.core.frame.DataFrame")
Tensor = TypeVar("torch.Tensor")
NpInt64 = TypeVar("numpy.int64")




class DataSelector:
    def __init__(self, dt: Dataframe, dt_indexes: Dict[str, Dict[int, List]]):
        self.default_dt = dt
        self.dt_indexes = copy.deepcopy(dt_indexes)
        self.labels = sorted(dt["label"].unique())
        # 学習済みのデータを削除したdataset_table
        self.dt = self.__init_dt(dt, dt_indexes)

    def __init_dt(self, dt: Dataframe, dt_indexes: Dict[str, Dict[int, List]]) -> Dataframe:
        drop_indexes = []
        for indexes in dt_indexes["selected_data"].values():
            drop_indexes += indexes
        dt = dt.drop(index=drop_indexes)
        return dt

    def __convert_to_tensor_image(self, json_image) -> Tensor:
        image = json.loads(json_image)
        image = np.array(image)
        image = torch.from_numpy(image.astype(np.float32)).clone()
        return image

    def drop_data(self, indexes: List):
        self.dt = self.dt.drop(index=indexes)

    def get_dt_indexes(self) -> Dict[str, Dict[int, List]]:
        return self.dt_indexes

    def get_dataset(self, indexes_labelby: Dict[int, List]) -> List[Tuple[Tensor, NpInt64]]:
        dataset = []
        dt_labelby = self.default_dt.groupby("label")
        for label in self.labels:
            indexes = indexes_labelby[label]
            dt = dt_labelby.get_group(label)
            rows = dt[dt["index"].isin(indexes)]
            images = rows["image"].values
            labels = rows["label"].values
            for image, label in zip(images, labels):
                image = self.__convert_to_tensor_image(image)
                dataset.append((image, label))
        return dataset

    def randomly_select_dt_indexes(self, dataN: int, seed=0) -> Dict[int, List]:
        indexes_labelby = {}
        dt_labelby = self.dt.groupby("label")
        for label in self.labels:
            dt = dt_labelby.get_group(label)
            dt = dt.sample(n=dataN, random_state=seed)
            selected_indexes = list(dt["index"].values)
            indexes_labelby[label] = selected_indexes
            self.dt_indexes["selected_data"][label] += selected_indexes
            self.drop_data(selected_indexes)
        return indexes_labelby




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
    def add_NN(self, dataN: int) -> Tuple[Dict[str, List[int]], List[int]]:
        if len(self.faiss_indexes) is 0:
            print("\nPlease run the process to make faiss indexes in advance.")
            sys.exit()
        selected_indexes = []
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
                selected_indexes.append(train_index)
                self.dt_indexes["selected"].append(train_index)
        return self.dt_indexes, selected_indexes

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
        print("Updated query to Farthest point.")
        return self.dt_indexes
