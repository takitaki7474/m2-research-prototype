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
