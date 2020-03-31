import json

def save_param(args, class_num, datasets):
    path = os.path.join("./result", args.model_name, "parameter.json")
    train, test = datasets

    parameters = {
    "model_name": args.model_name,
    "epoch": args.epoch,
    "batch_size": args.batch_size,
    "learning rate": args.alpha,
    "class_num": class_num,
    "train_num": len(train),
    "test_num": len(test),
    "gpu_id": args.gpu_id
    }

    with open(path, "w") as f:
        json.dump(parameters, f)
