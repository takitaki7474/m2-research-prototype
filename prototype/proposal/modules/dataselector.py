import copy
import faiss
import json
import numpy as np
import sys
import torch
from typing import TypeVar, List, Tuple, Dict

Dataframe = TypeVar("pandas.core.frame.DataFrame")
Tensor = TypeVar("torch.Tensor")
NpInt64 = TypeVar("numpy.int64")
NpArrayFloat32 = TypeVar("numpy.ndarray.float32")
FaissIndexFlatL2 = TypeVar("faiss.swigfaiss.IndexFlatL2")




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
    def __init__(self, ft: Dataframe, ft_indexes: Dict[str, Dict[int, List]]):
        super().__init__(ft, ft_indexes)

    def __ft_to_features(self, ft: Dataframe) -> NpArrayFloat32:
        features = [json.loads(f) for f in ft["feature"]]
        features = np.array(features).astype("float32")
        return features

    def __indexes_to_features(self, ft: Dataframe, indexes: List[int]) -> NpArrayFloat32:
        features = []
        for index in indexes:
            feature = ft[ft["index"] == index]["feature"].iloc[0]
            feature = json.loads(feature)
            features.append(feature)
        features = np.array(features).astype("float32")
        if len(features) != len(indexes): print("There is a feature that cannot be obtained")
        return features

    def __generate_faiss_index(self, vectors: NpArrayFloat32) -> FaissIndexFlatL2:
        dim = len(vectors[0])
        faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(vectors)
        return faiss_index

    def __search_NN_ft_indexes(self, ft: Dataframe, query_ft_indexes: List[int], dataN: int) -> List[int]:
        queries = self.__indexes_to_features(self.default_dt, query_ft_indexes)
        features = self.__ft_to_features(ft)
        faiss_index = self.__generate_faiss_index(features)
        k = faiss_index.ntotal # 検索対象データ数
        D, I = faiss_index.search(queries, k) # 近傍探索

        NN_ft_indexes = []
        all_query_indexes = [index for indexes in self.dt_indexes["queries"].values() for index in indexes]
        for indexes in I:
            cnt, i = 0, 0
            while cnt < dataN:
                ft_index = ft.iloc[indexes[i]]["index"]
                i += 1
                if ft_index in NN_ft_indexes: continue # 既に選択済みのインデックスは検索対象外
                if ft_index in all_query_indexes: continue # クエリは検索対象外
                NN_ft_indexes.append(ft_index)
                cnt += 1

        return NN_ft_indexes

    def __search_FP_ft_indexes(self, ft: Dataframe, query_ft_indexes: List[int]) -> List[int]:
        queries = self.__indexes_to_features(self.default_dt, query_ft_indexes)
        features = self.__ft_to_features(ft)
        faiss_index = self.__generate_faiss_index(features)
        k = faiss_index.ntotal # 検索対象データ数
        D, I = faiss_index.search(queries, k) # 近傍探索

        FP_ft_indexes = []
        all_used_query_indexes = [index for indexes in self.dt_indexes["used_queries"].values() for index in indexes]
        for indexes in I:
            for index in reversed(indexes):
                ft_index = ft.iloc[index]["index"]
                if ft_index in FP_ft_indexes: continue # 既に選択済みのインデックスは検索対象外
                if ft_index not in all_used_query_indexes: break # 一度でも使用されたクエリは検索対象外
            FP_ft_indexes.append(ft_index)

        return FP_ft_indexes

    def __search_MP_ft_indexes(self, ft: Dataframe, query_ft_indexes: List[int]) -> List[int]:
        queries = self.__indexes_to_features(self.default_dt, query_ft_indexes)
        features = self.__ft_to_features(ft)
        faiss_index = self.__generate_faiss_index(features)
        k = faiss_index.ntotal # 検索対象データ数
        D, I = faiss_index.search(queries, k) # 近傍探索

        MP_ft_indexes=[]
        all_used_query_indexes = [index for indexes in self.dt_indexes["used_queries"].values() for index in indexes]
        for indexes in I:
            cnt, i = 0, k//2
            while (1):
                cnt += 1
                index = indexes[i]
                ft_index = ft.iloc[index]["index"]
                i += 1
                if ft_index in MP_ft_indexes: continue # 既に選択済みのインデックスは検索対象外
                if ft_index not in all_used_query_indexes: break # 一度でも使用されたクエリは検索対象外
                if cnt > k:
                    print("The maximum number of data has been exceeded")
                    sys.exit()
            MP_ft_indexes.append(ft_index)

        return MP_ft_indexes

    def init_ft_indexes(self, queryN=1, seed=0) -> Dict[str, Dict[int, List]]:
        ft_indexes = {}
        ft_indexes["queries"], ft_indexes["used_queries"], ft_indexes["selected_data"],  = {}, {}, {}
        ft_labelby = self.default_dt.groupby("label")

        for label in self.labels:
            ft_indexes["selected_data"][label] = []
            ft_indexes["used_queries"][label] = []
            ft = ft_labelby.get_group(label)
            query = json.loads(ft.sample(n=1, random_state=seed)["feature"].iloc[0])
            query = np.array([query]).astype("float32")
            features = self.__ft_to_features(ft)
            faiss_index = self.__generate_faiss_index(features)
            k = faiss_index.ntotal
            linspace = k//queryN
            D, I = faiss_index.search(query, k)

            indexes = I[0][::linspace][:queryN]
            tmp_ft_indexes = []
            for index in indexes:
                ft_index = ft.iloc[index]["index"]
                tmp_ft_indexes.append(ft_index)
            ft_indexes["queries"][label] = tmp_ft_indexes

        return ft_indexes

    def select_NN_ft_indexes(self, dataN: int) -> Dict[int, List]:
        indexes_labelby = {}
        ft_labelby = self.dt.groupby("label")

        for label in self.labels:
            ft = ft_labelby.get_group(label)
            query_ft_indexes = self.dt_indexes["queries"][label]
            NN_ft_indexes = self.__search_NN_ft_indexes(ft, query_ft_indexes, dataN)

            indexes_labelby[label] = NN_ft_indexes
            self.dt_indexes["selected_data"][label] += NN_ft_indexes
            self.dt_indexes["used_queries"][label] += query_ft_indexes
            self.drop_data(NN_ft_indexes)

        return indexes_labelby

    # クエリをFP(最遠傍点)へ更新
    def update_to_FP_queries(self) -> Dict[int, List]:
        indexes_labelby = {}
        ft_labelby = self.dt.groupby("label")

        for label in self.labels:
            ft = ft_labelby.get_group(label)
            query_ft_indexes = self.dt_indexes["queries"][label]
            FP_ft_indexes = self.__search_FP_ft_indexes(ft, query_ft_indexes)

            indexes_labelby[label] = FP_ft_indexes
            self.dt_indexes["queries"][label] = FP_ft_indexes

        return indexes_labelby

    # クエリをMP(中間点)へ更新
    def update_to_MP_queries(self) -> Dict[int, List]:
        indexes_labelby = {}
        ft_labelby = self.dt.groupby("label")

        for label in self.labels:
            ft = ft_labelby.get_group(label)
            query_ft_indexes = self.dt_indexes["queries"][label]
            MP_ft_indexes = self.__search_MP_ft_indexes(ft, query_ft_indexes)

            indexes_labelby[label] = MP_ft_indexes
            self.dt_indexes["queries"][label] = MP_ft_indexes

        return indexes_labelby
