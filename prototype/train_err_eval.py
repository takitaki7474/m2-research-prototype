import json

def load_json(json_path):
    f = open(json_path, "r")
    loaded_json = json.load(f)
    return loaded_json

# logファイルから訓練誤差の配列を取得
def get_train_losses(train_log):
    train_losses = []
    for i in train_log:
        train_losses.append(i["train loss"])
    return train_losses

# 訓練誤差の推移の基準関数としてy = 0.8^x + 0.25を配列で取得
def get_standard_losses(epoch=80):
    x = list(range(1, epoch+1))
    standard_losses = []
    for i in x:
        standard_losses.append(0.8**i + 0.25)
    return standard_losses

# 訓練誤差収束速度の評価
# a: 係数
# n: 誤差の傾きの計測範囲
# Ep: pエポックにおける訓練誤差
# 訓練誤差収束速度: S = Σ-a/k(Enk - Enk-n+1)
def eval_train_err_speed(train_losses, a=50, n=10):
    train_err_speed = 0.0
    E = train_losses
    maxepoch = len(train_losses)
    ks = maxepoch//n
    for k in range(1, ks+1):
        train_err_speed += (-a / k) * (E[n*k-1] - E[n*k - n])
    return train_err_speed
