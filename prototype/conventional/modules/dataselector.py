import json
import numpy as np
import torch
from typing import List, Tuple, TypeVar, Dict

Dataframe = TypeVar("pandas.core.frame.DataFrame")

def init_dataset_table_indexes() -> Dict[str, List]:
    dt_indexes = {}
    dt_indexes["selected"] = []
    return dt_indexes

# dataset_table_indexes(dt_indexes) をjson形式で保存
def save_dataset_table_indexes(dt_indexes: Dict[str, List[int]], savepath="./dt_indexes_v1.json"):
    dt_indexes["selected"] = [int(index) for index in dt_indexes["selected"]]
    with open(savepath, "w") as f:
        json.dump(dt_indexes, f, indent=4)

# dataset_table_indexes(dt_indexes) を辞書形式で読み込み
def load_dataset_table_indexes(path: str) -> Dict[str, List[int]]:
    with open(path, "r") as f:
        dt_indexes = json.load(f)
    return dt_indexes


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
