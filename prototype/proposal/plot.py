import matplotlib.pyplot as plt

# lossのplot
def plot_loss(train_losses, test_losses, save_path):
    max_axisX = len(train_losses)
    plt.figure(figsize=(7,5))
    plt.plot(range(max_axisX), train_losses)
    plt.plot(range(max_axisX), test_losses, c='#ed7700')
    plt.ylim(bottom=0)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train/loss', 'test/loss'])
    plt.grid()
    plt.savefig(save_path)
    plt.clf()

# accuracyのplot
def plot_acc(train_accs, test_accs, save_path):
    max_axisX = len(train_accs)
    plt.figure(figsize=(7,5))
    plt.plot(range(max_axisX), train_accs)
    plt.plot(range(max_axisX), test_accs, c='#ed7700')
    plt.ylim(bottom=0)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train/acc', 'test/acc'])
    plt.grid()
    plt.savefig(save_path)
    plt.clf()
