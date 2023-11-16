import numpy as np


class LinearRegression:
    def __init__(self, intercept=True, scale=True, regularization=None, reg=0.01):
        self.intercept = intercept
        self.scale = scale
        self.regularization = regularization
        self.reg = reg
        self.x_mu = None
        self.x_sigma = None
        self.b = None
        self.training_error = []

    def fit(self, X, y, epochs=30, lr=0.01, tol=1e-6, bs=20):
        if self.scale:
            self.x_mu = X.mean(axis=0)
            self.x_sigma = X.std(axis=0)
        X = self._preprocess(X)
        self.b = np.ones(X.shape[1])

        last_epoch = self._cost(X, y)
        for epoch in range(epochs):
            inds = np.argsort(np.random.rand(X.shape[0]))
            for i in range(0, X.shape[0], bs):
                X_batch = X[inds[i : i + bs], :]
                y_batch = y[inds[i : i + bs]]
                c = self._cost(X_batch, y_batch)
                self.training_error.append(c)
                grad = self._grad(X_batch, y_batch)
                self.b = self.b - lr * grad
            this_epoch = self._cost(X, y)
            delta = abs((this_epoch - last_epoch) / last_epoch)
            if delta <= tol:
                break
            last_epoch = this_epoch
        print(f"Completed {epoch} / {epochs} epochs w tolerance: {delta:.2e}")

    def predict(self, X):
        X = self._preprocess(X)
        return np.matmul(X, self.b)

    def evaluate(self, y, y_hat):
        rss = sum((y - y_hat) ** 2)
        tss = sum((y - y.mean()) ** 2)
        r2 = (tss - rss) / tss
        print(rss, tss, r2)
        print(f"R2: {r2:0.1%}")

    def _preprocess(self, X):
        if self.scale:
            X = (X - self.x_mu) / self.x_sigma
        if self.intercept:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        return X

    def _cost(self, X, y):
        c = sum((y - np.matmul(X, self.b)) ** 2) / X.shape[0]
        if self.regularization == "ridge":
            c += self.reg * sum(self.b**2)
        if self.regularization == "lasso":
            c += self.reg * sum(abs(self.b))
        return c

    def _grad(self, X, y):
        y_delta = np.dot(X, self.b) - y
        grad = np.dot(y_delta, X)
        if self.regularization == "ridge":
            grad += 2 * self.reg * self.b
        if self.regularization == "lasso":
            grad += self.reg * np.sign(self.b)
        return grad


class LinearRegressionClosedFormModel:
    def __init__(self, intercept=True):
        self.intercept = intercept
        self.b = None

    def _preprocess_X(self, X):
        if X.ndim == 1:
            X = X.reshape(X.shape[0], -1)
        if self.intercept:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        return X

    def fit(self, X, y):
        X = self._preprocess_X(X)
        self.b = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)

    def predict(self, X):
        X = self._preprocess_X(X)
        return np.matmul(X, self.b)

    def evaluate(self, y, y_hat):
        rss = sum((y - y_hat) ** 2)
        tss = sum((y - y.mean()) ** 2)
        r2 = (tss - rss) / tss
        print(f"R2: {r2:0.1%}")


if __name__ == "__main__":
    n = 100
    np.random.seed(15)
    b0, b1 = 8, 3.2
    e = np.random.normal(0, 30, n)
    X = np.random.uniform(0, 100, n).reshape(-1, 1)
    y = b0 + b1 * X[:, 0] + e

    m = LinearRegression(intercept=True, scale=True, regularization="ridge")
    m.fit(X, y, epochs=1000, lr=0.001)
