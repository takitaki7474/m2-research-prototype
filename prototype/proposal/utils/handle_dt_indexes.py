import json
import pandas as pd
import random
import sqlite3
from typing import TypeVar, Dict

Dataframe = TypeVar("pandas.core.frame.DataFrame")

def blank_dt_indexes(dt: Dataframe) -> Dict[str, Dict]:
    labels = sorted(dt["label"].unique())
    dt_indexes = {}
    dt_indexes["selected_data"] = {}
    for label in labels:
        dt_indexes["selected_data"][label] = []
    return dt_indexes

# feature_table_indexesの初期化 (queryはランダムに選択)
def blank_ft_indexes(ft: Dataframe) -> Dict[str, Dict]:
    labels = sorted(ft["label"].unique())
    ft_indexes = {}
    ft_indexes["queries"], ft_indexes["used_queries"], ft_indexes["selected_data"],  = {}, {}, {}
    for label in labels:
        ft_indexes["used_queries"][label] = []
        ft_indexes["selected_data"][label] = []
        ft_indexes["queries"][label] = []
    return ft_indexes

def save_dt_indexes(dt_indexes: Dict[int, Dict], savepath="./dt_indexes.json"):
    dic = {}
    for k1, v1 in dt_indexes.items():
        dic[k1] = {}
        for k2, v2 in dt_indexes[k1].items():
            dic[k1][str(k2)] = [int(i) for i in v2]
    with open(savepath, "w") as f:
        json.dump(dic, f, indent=4)

def load_dt_indexes(path="./dt_indexes.json") -> Dict[int, Dict]:
    dt_indexes = {}
    with open(path, "r") as f:
        dic = json.load(f)
    for k1, v1 in dic.items():
        dt_indexes[k1] = {}
        for k2, v2 in dic[k1].items():
            dt_indexes[k1][int(k2)] = v2
    return dt_indexes
