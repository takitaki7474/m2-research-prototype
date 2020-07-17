
def lr_v1(epoch):
    if epoch < 20:
        return 1
    elif epoch < 40:
        return 0.1**1
    elif epoch < 60:
        return 0.1**2
    else:
        return 0.1**3
