import json
from typing import List, Tuple

def load_train_log(path: str):

    def load_losses(path: str) -> Tuple[List[float], List[float]]:
        with open(path) as f:
            logs = json.load(f)
        train_losses, test_losses = [], []
        for log in logs:
            train_losses.append(log["train_loss"])
            test_losses.append(log["test_loss"])
        return train_losses, test_losses

    def load_accs(path: str) -> Tuple[List[float], List[float]]:
        with open(path) as f:
            logs = json.load(f)
        train_accs, test_accs = [], []
        for log in logs:
            train_accs.append(log["train_acc"])
            test_accs.append(log["test_acc"])
        return train_accs, test_accs

    train_losses, test_losses = load_losses(path)
    train_accs, test_accs = load_accs(path)

    return train_losses, test_losses, train_accs, test_accs
