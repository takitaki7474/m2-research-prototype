# 訓練誤差収束速度の評価
# a: 係数
# n: 誤差の傾きの計測範囲
# Ep: pエポックにおける訓練誤差
# 訓練誤差収束速度: S = Σ-a/k(Enk - Enk-n+1)
def eval_err_speed(losses, a=50, n=10):
    err_speed = 0.0
    E = losses
    maxepoch = len(losses)
    ks = maxepoch//n
    for k in range(1, ks+1):
        err_speed += (-a / k) * (E[n*k-1] - E[n*k - n])
    return err_speed
