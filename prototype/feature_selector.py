import random
import sys
import faiss
import json
import numpy as np
import pandas as pd
import sqlite3

def load_feature_table(dbpath, tablename):
    conn=sqlite3.connect(dbpath)
    c = conn.cursor()
    df = pd.read_sql('SELECT * FROM ' + tablename, conn)
    return df

def init_selected_indexes_randomly(feature_table):
    selected_indexes = [[], [], []]
    labels = feature_table["label"].unique()
    df = feature_table.groupby("label")
    for label in labels:
        df_labelby = df.get_group(label)
        indexes = df_labelby["index"].values.tolist()
        random_index = random.sample(indexes, 1)[0]
        selected_indexes[1].append(random_index)
    return selected_indexes

def load_selected_data(datapath):
    d = np.load(datapath)
    d = list(d)
    return d

# 次に学習するべきフィーチャを選択
class FeatureSelector:
    def __init__(self, feature_table, selected_indexes):
        self.table = self.__drop_selected_from_table(feature_table, selected_indexes) # feature_table
        self.tables_labelby = self.__get_tables_labelby(self.table) # ラベルごとに分割したfeature_table
        self.selected = selected_indexes # 前の学習で選択済みのデータ [trained_indexes, query_indexes, selected_query_indexes]
        self.feature_indexes = [] # ラベルごとのフィーチャを格納したfaiss-index

     # 前の学習で選択したデータをfeature_tableから削除
    def __drop_selected_from_table(self, feature_table, selected_indexes):
        trained_indexes = selected_indexes[0]
        df = feature_table.drop(index=trained_indexes)
        df = df.reset_index(drop=True)
        return df

    # feature_tableをラベルごとに分割
    def __get_tables_labelby(self, feature_table):
        tables_labelby = []
        labels = feature_table["label"].unique()
        df = feature_table.groupby("label")
        for label in labels:
            df_labelby = df.get_group(label)
            df_labelby = df_labelby.reset_index(drop=True)
            tables_labelby.append(df_labelby)
        return tables_labelby

    def save_selected_indexes(self, savepath):
        np.save(savepath, np.array(self.selected))

    # 検索のための、フィーチャのインデックスを作成
    def add_feature_indexes(self):
        for table in self.tables_labelby:
            features = []
            for feature in table["feature"]:
                feature = json.loads(feature)
                features.append(feature)
            features = np.array(features).astype("float32")
            dim = len(features[0])
            index = faiss.IndexFlatL2(dim)
            index.add(features)
            self.feature_indexes.append(index)

    # ラベルごとに、クエリとの最近傍のデータをdata_num数選択
    def select_NN_train(self, data_num, savepath):
        if self.feature_indexes == []:
            print("Please execute add_feature_indexes method in advance.")
            sys.exit()
        train_indexes = []
        selected_query_indexes = []
        query_indexes = self.selected[1]
        for i, index in enumerate(self.feature_indexes):
            k = index.ntotal
            query = self.table[self.table["index"] == query_indexes[i]]["feature"].iat[0]
            query = json.loads(query)
            query = np.array([query]).astype("float32")
            D, I = index.search(query, k)
            selected_query_indexes.append(query_indexes[i])
            for j in I[0][:data_num+1]:
                train_index = self.tables_labelby[i].iloc[j]["index"]
                if train_index == query_indexes[i]:
                    continue
                train_indexes.append(train_index)
        self.selected[0] = self.selected[0] + train_indexes
        self.selected[2] = list(set(self.selected[2] + selected_query_indexes))
        np.save(savepath, np.array(train_indexes)) #次に学習するtrainのインデックスを保存

    # selected_dataのクエリを最遠傍点に更新
    def update_FP_queries(self):
        if self.feature_indexes == []:
            print("Please execute add_feature_indexes method in advance.")
            sys.exit()
        updated = []
        query_indexes = self.selected[1]
        for i, index in enumerate(self.feature_indexes):
            k = index.ntotal
            query = self.table[self.table["index"] == query_indexes[i]]["feature"].iat[0]
            query = json.loads(query)
            query = np.array([query]).astype("float32")
            D, I = index.search(query, k)
            for j in range(1,k):
                FP_query_index = self.tables_labelby[i].iloc[I[0][-j]]["index"]
                if FP_query_index not in self.selected[2]:
                    break
            updated.append(FP_query_index)
        self.selected[1] = updated
