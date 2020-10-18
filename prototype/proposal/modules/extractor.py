from collections import defaultdict
import json
import pandas as pd
import sqlite3
import time
from typing import TypeVar

Dataframe = TypeVar("pandas.core.frame.DataFrame")

class FeatureExtractor:

    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.feature_table: Dataframe = None

    def make_feature_table(self):
        start = time.time()
        table = defaultdict(list)
        for (inputs, labels) in self.dataloader:
            outputs, features = self.model(inputs)
            for label, image, feature in zip(labels, inputs, features):
                table["label"].append(int(label))
                table["image"].append(json.dumps(image.cpu().numpy().tolist()))
                table["feature"].append(json.dumps(feature.data.cpu().numpy().tolist()))
        for k, v in table.items():
            table[k] = pd.Series(v)
        self.feature_table = pd.DataFrame(table)
        elapsed_time = time.time() - start
        print("elapsed time of feature extraction:  {0}m {1}s".format(elapsed_time//60, elapsed_time%60))

    def save_feature_table(self, savepath="./ft.db", tablename="feature_table"):
        if self.feature_table is None:
             print("feature table does not exist.")
        else:
            conn = sqlite3.connect(savepath)
            c = conn.cursor()
            self.feature_table.to_sql(tablename, conn, if_exists='replace')
            conn.close()
