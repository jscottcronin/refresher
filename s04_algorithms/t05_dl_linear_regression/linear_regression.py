import torch
import matplotlib.pyplot as plt


def get_data():
    n = 10000
    m = -1.5
    b = -80.0

    xs = torch.linspace(1, 100, n)
    ys = m * xs + b + torch.randn_like(xs)
    return xs, ys


def split(x, y, train_size=0.8):
    train_inds = torch.randperm(len(x))[: int(train_size * len(x))]
    return x[train_inds], x[~train_inds], y[train_inds], y[~train_inds]


def train_by_hand(x_train, x_test, y_train, y_test):
    epochs = 3
    lr = 1e-1
    bs = 128

    # Model
    coeffs = torch.randn(2, requires_grad=True)

    # Normalize data by hand
    u, s = x_train.mean(), x_train.std()

    losses = []
    for _ in range(epochs):
        inds = torch.randperm(len(x_train))
        batches = [
            (x_train[inds][i : i + bs], y_train[inds][i : i + bs])
            for i in range(0, len(x_train), bs)
        ]
        for batch in batches:
            x, y = batch
            y_preds = coeffs[0] * ((x - u) / s) + coeffs[1]
            loss = (y_preds - y).pow(2).mean()
            losses.append(loss.item())
            loss.backward()
            with torch.no_grad():
                coeffs -= lr * coeffs.grad
                coeffs.grad.zero_()
    plt.plot(losses, marker="o")
    plt.yscale("log")
    plt.show()

    y_preds = coeffs[0] * ((x_test - u) / s) + coeffs[1]
    plt.scatter(x_test, y_test)
    plt.plot(x_test, y_preds.detach(), color="k")
    plt.show()


def train(x_train, x_test, y_train, y_test):
    epochs = 50
    lr = 1e-3
    bs = 10000

    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(1),
        torch.nn.Linear(1, 1),
    )
    losses = []
    for _ in range(epochs):
        inds = torch.randperm(len(x_train))
        batches = [
            (x_train[inds][i : i + bs], y_train[inds][i : i + bs])
            for i in range(0, len(x_train), bs)
        ]
        for batch in batches:
            x, y = batch
            y_preds = model(x.unsqueeze(1)).squeeze()
            loss = (y_preds - y).pow(2).mean()
            losses.append(loss.item())
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param -= lr * param.grad
                    param.grad.zero_()

    plt.plot(losses, marker="o")
    plt.yscale("log")
    plt.show()

    y_preds = model(x_test.unsqueeze(1)).squeeze().detach()
    plt.scatter(x_test, y_test)
    plt.plot(x_test, y_preds, color="k")
    plt.show()


if __name__ == "__main__":
    x, y = get_data()
    x_train, x_test, y_train, y_test = split(x, y)
    # train_by_hand(x_train, x_test, y_train, y_test)
    train(x_train, x_test, y_train, y_test)
