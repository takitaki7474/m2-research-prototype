import matplotlib.pyplot as plt

# lossのplot
def plot_loss(train_loss_list, test_loss_list, save_path):
    max_axisX = len(train_loss_list)
    plt.figure(figsize=(7,5))
    plt.plot(range(max_axisX), train_loss_list)
    plt.plot(range(max_axisX), test_loss_list, c='#ed7700')
    plt.ylim(bottom=0)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train/loss', 'test/loss'])
    plt.grid()
    plt.savefig(save_path)
    plt.clf()

# accuracyのplot
def plot_acc(train_acc_list, test_acc_list, save_path):
    max_axisX = len(train_acc_list)
    plt.figure(figsize=(7,5))
    plt.plot(range(max_axisX), train_acc_list)
    plt.plot(range(max_axisX), test_acc_list, c='#ed7700')
    plt.ylim(bottom=0)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train/acc', 'test/acc'])
    plt.grid()
    plt.savefig(save_path)
    plt.clf()
