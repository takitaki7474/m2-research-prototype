import argparse
import configparser
import json
import os




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ver1", "-mv1", type=str, default="v1", help="model name of the comparison1 to compare loss")
    #parser.add_argument("--model_ver2", "-mv2", type=str, default="v2", help="model name of the comparison2 to compare loss")
    args = parser.parse_args()
    return args

def load_train_err_speed(jsonpath: str):
    with open(jsonpath) as f:
        dic = json.load(f)
    train_err_speed = dic["train_err_speed"]
    return train_err_speed

def load_test_err_speed(jsonpath: str):
    with open(jsonpath) as f:
        dic = json.load(f)
    test_err_speed = dic["test_err_speed"]
    return test_err_speed

def eval1(err_speed1: float, err_speed2: float) -> int:
    result = 0
    std = err_speed1 + err_speed1 * 0.05
    if err_speed2 <= std:
        result = 1
    return result

def eval2(err_speed: float) -> int:
    result = 0
    std = 40.0
    if err_speed >= std:
        result = 1
    return result




inifile = configparser.SafeConfigParser()
inifile.read("./settings.ini")

# 学習結果が保存されているディレクトリ
RESULT_DIR = inifile.get("TrainResultDirs", "domain_result")




if __name__=="__main__":
    args = get_args()
    #path2 = os.path.join(RESULT_DIR, args.model_ver2, "err_speed.json")
    #err_speed2 = load_train_err_speed(jsonpath=path2)
    path1 = os.path.join(RESULT_DIR, args.model_ver1, "err_speed.json")
    err_speed1 = load_train_err_speed(jsonpath=path1)
    result = eval2(err_speed1)
    print(result)
