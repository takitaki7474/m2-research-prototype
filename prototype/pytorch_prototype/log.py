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
