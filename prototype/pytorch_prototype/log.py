import json
import os

def save_param(args, net_name, class_label_list, datasets):
    path = os.path.join("./result", args.model_name, "parameter.json")
    train, test = datasets

    parameters = {
    "model_name": args.model_name,
    "net_name": net_name,
    "epoch": args.epoch,
    "batch_size": args.batch_size,
    "learning rate": args.alpha,
    "class_label_list": class_label_list,
    "class_num": len(class_label_list),
    "train_num": len(train),
    "test_num": len(test),
    "gpu_id": args.gpu_id
    }

    with open(path, "w") as f:
        json.dump(parameters, f, indent=4)


def save_log_func(args):
    path = os.path.join("./result", args.model_name, "log.json")
    log_list = []

    def save_log(log=None, save_flag=0):
        if save_flag == 0:
            log_list.append(log)
        elif save_flag == 1:
            with open(path, "w") as f:
                json.dump(log_list, f, indent=4)

    return save_log
