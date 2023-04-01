import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch


def generate_data(N=1_000, n_loops=2.5, noise=0.6, seed=501, debug=False):

    np.random.seed(seed)
    r = np.sqrt(np.random.rand(N)) * n_loops * (2 * np.pi)

    class0 = np.column_stack(
        [
            -np.cos(r) * r + np.random.rand(N) * noise,
            np.sin(r) * r + np.random.rand(N) * noise,
        ]
    )
    class1 = -class0
    X = np.vstack([class0, class1]).astype(np.float32)
    y = np.hstack([np.zeros(N), np.ones(N)])

    index = np.arange(len(y))
    np.random.shuffle(index)
    X = X[index, :]
    y = y[index]

    if debug:
        return X, y, r
    else:
        return X, y


def plot_spiral(X, y, xlim=None, ylim=None, title=None):
    mask_1 = y == 1
    plt.scatter(
        X[~mask_1, 0], X[~mask_1, 1], s=70, c="darkblue", label="Class 0", alpha=0.3
    )
    plt.scatter(
        X[mask_1, 0], X[mask_1, 1], s=70, c="darkred", label="Class 1", alpha=0.3
    )
    plt.legend(loc="upper left")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if title is not None:
        plt.title(title)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.tight_layout()


class Net(torch.nn.Module):
    def __init__(self, n_features, layers):
        super().__init__()
        self.hidden = torch.nn.ModuleList()
        layers_in = [n_features] + layers
        layers_out = layers + [1]
        for i in range(len(layers) + 1):
            self.hidden.append(torch.nn.Linear(layers_in[i], layers_out[i]))

    def forward(self, xb):
        for i, h in enumerate(self.hidden):
            xb = h(xb)
            if i < (len(self.hidden) - 1):
                xb = torch.nn.functional.relu(xb)
        return xb.flatten()


def score(x, y, trainer, xlim=(-20, 20), ylim=(-20, 20)):

    p = trainer.predict(x).data.numpy()
    yhat = (p > 0.5).astype(int)

    accuracy = np.mean(y == yhat)
    average_precision = sklearn.metrics.average_precision_score(y, p)
    rocauc = sklearn.metrics.roc_auc_score(y, p)
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y, p)
    prauc = sklearn.metrics.auc(recall, precision)

    xx = np.linspace(xlim[0], xlim[1], 500)
    yy = np.linspace(ylim[0], ylim[1], 500)
    gx, gy = np.meshgrid(xx, yy)
    Z = (
        trainer.predict(torch.tensor(np.c_[gx.ravel(), gy.ravel()].astype(np.float32)))
        .detach()
        .numpy()
        .reshape(gx.shape)
    )

    plt.contourf(gx, gy, Z, cmap=plt.cm.coolwarm, alpha=0.7)

    mask_11 = (y == 1) & (yhat == 1)
    mask_10 = (y == 1) & (yhat == 0)
    mask_00 = (y == 0) & (yhat == 0)
    mask_01 = (y == 0) & (yhat == 1)

    plt.scatter(
        x[mask_11, 0],
        x[mask_11, 1],
        s=100,
        c="darkred",
        marker="o",
        label="True Positive",
    )
    plt.scatter(
        x[mask_10, 0],
        x[mask_10, 1],
        s=100,
        c="darkred",
        marker="^",
        label="False Negative",
    )
    plt.scatter(
        x[mask_00, 0],
        x[mask_00, 1],
        s=100,
        c="darkblue",
        marker="o",
        label="True Negative",
    )
    plt.scatter(
        x[mask_01, 0],
        x[mask_01, 1],
        s=100,
        c="darkblue",
        marker="^",
        label="False Positive",
    )

    plt.title(
        f"acc={accuracy:.2f} | avg(prec)={average_precision:.2f} | rocauc={rocauc:.2f} | prauc={prauc:.2f}"
    )
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.legend(loc="upper left")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
