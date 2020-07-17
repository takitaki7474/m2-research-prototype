import os
import json

def log_setting(args, net, classes, save_path):
    setting = {
    "model": args.model_name,
    "net": net.__class__.__name__,
    "epoch": args.epoch,
    "batch_size": args.batch_size,
    "initial_lr": args.lr,
    "classes": classes,
    "train_num_by_label": args.train_num,
    "test_num_by_label": args.test_num,
    }
    with open(save_path, "w") as f:
        json.dump(setting, f, indent=4)

class ParamLogging:

    def __init__(self):
        self.log_list = []

    def emit(self, log):
        self.log_list.append(log)

    def save(self, save_path):
        with open(save_path, "w") as f:
            json.dump(self.log_list, f, indent=4)
