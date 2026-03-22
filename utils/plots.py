import matplotlib.pyplot as plt

def plot_loss(history, title):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(title)
    plt.legend(["Train", "Val"])
    plt.show()


def plot_prediction(y_true, y_pred, title):
    plt.plot(y_true[:200])
    plt.plot(y_pred[:200])
    plt.title(title)
    plt.legend(["Actual", "Predicted"])
    plt.show()