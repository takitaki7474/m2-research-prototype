from collections import defaultdict
import sqlite3
import pandas as pd
import json
import torch

def save_feature_table(df, dbname):
    pass

def make_feature_table(net, model_path, data, batch_size):
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)
    net.load_state_dict(torch.load(model_path))
    d = defaultdict(list)
    for i, (inputs, labels) in enumerate(dataloader):
        outputs, features = net(inputs)
        for label, image, feature in zip(labels, inputs, features):
            d["label"].append(int(label))
            d["image"].append(json.dumps(image.cpu().numpy().tolist()))
            d["feature"].append(json.dumps(feature.data.cpu().numpy().tolist()))
    for k, v in d.items():
        d[k] = pd.Series(v)
    df = pd.DataFrame(d)
    return df
