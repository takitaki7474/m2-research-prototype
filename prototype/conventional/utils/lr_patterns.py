

def lr_v1(epoch):
    if epoch < 50:
        return 1
    elif epoch < 70:
        return 0.1**1
    elif epoch < 90:
        return 0.1**2
    else:
        return 0.1**3
