import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone


class AdaBoostBinaryClassifier(object):
    """
    INPUT:
    - n_estimator (int)
      * The number of estimators to use in boosting
      * Default: 50

    - learning_rate (float)
      * Determines how fast the error would shrink
      * Lower learning rate means more accurate decision boundary,
        but slower to converge
      * Default: 1
    """

    def __init__(self, n_estimators=50, learning_rate=1):
        self.base_estimator = DecisionTreeClassifier(max_depth=1)
        self.n_estimator = n_estimators
        self.learning_rate = learning_rate
        self.estimators_ = []
        self.estimator_weight_ = np.zeros(self.n_estimator, dtype=float)

    def fit(self, x, y):
        """
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels

        Build the estimators for the AdaBoost estimator.
        """
        sample_weight = np.ones(x.shape[0]) / x.shape[0]
        for i in range(self.n_estimator):
            est, sample_weight, alpha = self._boost(x, y, sample_weight)
            self.estimators_.append(est)
            self.estimator_weight_[i] = alpha

    def _boost(self, x, y, sample_weight):
        """
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels
        - sample_weight: numpy array

        OUTPUT:
        - estimator: DecisionTreeClassifier
        - sample_weight: numpy array (updated weights)
        - estimator_weight: float (weight of estimator)

        Go through one iteration of the AdaBoost algorithm. Build one estimator.
        """

        estimator = clone(self.base_estimator)
        estimator.fit(x, y, sample_weight=sample_weight)
        incorrect = estimator.predict(x) != y
        err = np.sum(sample_weight * incorrect) / sample_weight.sum()
        alpha = self.learning_rate * np.log((1.0 - err) / err)
        sample_weight *= np.exp(alpha * incorrect)
        return estimator, sample_weight, alpha

    def predict(self, x):
        """
        INPUT:
        - x: 2d numpy array, feature matrix

        OUTPUT:
        - labels: numpy array of predictions (0 or 1)
        """
        y_pred = np.zeros(x.shape[0], dtype=float)
        for i, est in enumerate(self.estimators_):
            preds = est.predict(x)
            preds[preds == 0] = -1
            y_pred += self.estimator_weight_[i] * preds
        return (y_pred > 0).astype(float)

    def score(self, x, y):
        """
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels

        OUTPUT:
        - score: float (accuracy score between 0 and 1)
        """
        return (self.predict(x) == y).mean()
