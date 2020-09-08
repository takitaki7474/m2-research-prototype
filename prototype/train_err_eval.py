import json

def load_json(json_path):
    f = open(json_path, "r")
    loaded_json = json.load(f)
    return loaded_json

def get_train_losses(train_log):
    train_losses = []
    for i in train_log:
        train_losses.append(i["train loss"])
    return train_losses

def get_standard_losses(epoch=80):
    x = list(range(1, epoch+1))
    standard_losses = []
    for i in x:
        standard_losses.append(0.8**i + 0.25)
    return standard_losses
