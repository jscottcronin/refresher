import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch


def load_data():
    df = sns.load_dataset("titanic")

    # Fill missing values
    modes = df.mode().iloc[0]
    df = df.fillna(modes)

    # Log transform fare into more normal distribution
    df["log_fare"] = np.log(df["fare"] + 1)

    # Drop columns
    drop_cols = ["fare", "embark_town", "alive", "who", "class"]
    df = df.drop(drop_cols, axis=1)

    # Convert bools to ints
    bool_cols = ["adult_male", "alone"]
    df[bool_cols] = df[bool_cols].astype(int)

    # dummify categorical columns
    dummy_cols = ["sex", "embarked", "pclass", "deck"]
    df = pd.get_dummies(df, columns=dummy_cols)

    # Normalize columns
    df = df / df.max(axis=0)

    # Add interceopt
    df["intercept"] = 1
    return df


def split(data):
    y = torch.tensor(df["survived"].values, dtype=torch.float32)
    X = torch.tensor(df.drop("survived", axis=1).values, dtype=torch.float32)
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    num_features = x_train.shape[1]
    epochs = 50
    lr = 10

    coeffs = torch.rand(num_features, requires_grad=True)

    train_loss = []
    val_loss = []
    val_acc = []
    for epoch in range(epochs):
        inds = torch.randperm(len(x_train))

        y_preds = torch.nn.functional.sigmoid((x_train * coeffs).sum(dim=1))
        loss = torch.abs(y_train - y_preds).mean()
        train_loss.append(loss.item())
        loss.backward()

        with torch.no_grad():
            y_preds = torch.nn.functional.sigmoid((x_test * coeffs).sum(dim=1))

            val_loss.append(torch.abs(y_test - y_preds).mean().item())
            val_acc.append(((y_preds > 0.5).float() == y_test).float().mean().item())

            # update coefficients
            coeffs -= coeffs.grad * lr
            coeffs.grad.zero_()

    print(f"Accuracy: {val_acc[-1]:0.2%}")

    plt.plot(train_loss, marker="o", label="train_loss")
    plt.plot(val_loss, marker="o", label="val_loss")
    plt.legend()
    plt.show()

    plt.plot(val_acc, marker="o", label="val_acc")
    plt.show()
