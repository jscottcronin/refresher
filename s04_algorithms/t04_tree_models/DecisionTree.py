import pandas as pd
import numpy as np
import math
from collections import Counter
from TreeNode import TreeNode


class DecisionTree:
    """
    A decision tree class.
    """

    def __init__(
        self,
        impurity_criterion="entropy",
        max_depth=None,
        min_samples_leaf=1,
        leaf_fraction_to_stop=0.9,
        min_impurity_decrease=0.01,
    ):
        """
        Initialize an empty DecisionTree.
        """

        self.root = None  # root Node
        self.max_depth = max_depth  # max depth for tree
        self.min_samples_leaf = min_samples_leaf  # min samples per leaf
        self.leaf_fraction_to_stop = leaf_fraction_to_stop  # stop splitting if leaf mostly the same
        self.min_impurity_decrease = min_impurity_decrease  # stop splitting if gain is too small
        self.feature_names = None  # string names of features (for interpreting the tree)
        self.categorical = None  # Boolean array of whether variable is categorical (or continuous)
        self.impurity_criterion = self._entropy if impurity_criterion == "entropy" else self._gini

    def fit(self, X, y, feature_names=None):
        """
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - feature_names: numpy array of strings
        OUTPUT: None

        Build the decision tree.
        X is a 2 dimensional array with each column being a feature and each row a data point.
        y is a 1 dimensional array with each value being the corresponding label.
        feature_names is an optional list containing the names of each of the features.
        """

        if feature_names is None or len(feature_names) != X.shape[1]:
            self.feature_names = np.arange(X.shape[1])
        else:
            self.feature_names = feature_names

        self.categorical = np.array([isinstance(i, (str, bool)) for i in X[0, :]])
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=1):
        """
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
        OUTPUT:
            - TreeNode

        Recursively build the decision tree. Return the root node.
        """
        node = TreeNode()
        node.depth = depth
        index, value, splits = self._choose_split_index(X, y)

        # Stopping Conditions
        at_max_depth = self.max_depth is not None and depth >= self.max_depth
        y_unique = len(np.unique(y)) == 1
        index_is_none = index is None
        mostly_the_same = abs(y.mean() - 0.5) > self.leaf_fraction_to_stop

        if index_is_none or y_unique or at_max_depth or mostly_the_same:
            node.leaf = True
            node.classes = Counter(y)
            node.name = node.classes.most_common(1)[0][0]
        else:
            X1, y1, X2, y2 = splits
            node.column = index
            node.name = self.feature_names[index]
            node.value = value
            node.categorical = self.categorical[index]
            node.left = self._build_tree(X1, y1, depth + 1)
            node.right = self._build_tree(X2, y2, depth + 1)
        return node

    def _entropy(self, y):
        """
        INPUT:
            - y: 1d numpy array
        OUTPUT:
            - float

        Return the entropy of the array y.
        """
        return -1 * sum(
            [v / len(y) * np.log(v / len(y)) for k, v in Counter(y).items()]
        )

    def _gini(self, y):
        """
        INPUT:
            - y: 1d numpy array
        OUTPUT:
            - float

        Return the gini impurity of the array y.
        """
        return 1 - sum([(v / len(y)) ** 2 for k, v in Counter(y).items()])

    def _make_split(self, X, y, split_index, split_value):
        """
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - split_index: int (index of feature)
            - split_value: int/float/bool/str (value of feature)
        OUTPUT:
            - X1: 2d numpy array (feature matrix for subset 1)
            - y1: 1d numpy array (labels for subset 1)
            - X2: 2d numpy array (feature matrix for subset 2)
            - y2: 1d numpy array (labels for subset 2)

        Return the two subsets of the dataset achieved by the given feature and
        value to split on.

        Call the method like this:
        >>> X1, y1, X2, y2 = self._make_split(X, y, split_index, split_value)

        X1, y1 is a subset of the data.
        X2, y2 is the other subset of the data.
        """
        if self.categorical[split_index]:
            inds = X[:, split_index] == split_value
        else:
            inds = X[:, split_index] >= split_value
        return X[inds, :], y[inds], X[~inds, :], y[~inds]

    def _information_gain(self, y, y1, y2):
        """
        INPUT:
            - y: 1d numpy array
            - y1: 1d numpy array (labels for subset 1)
            - y2: 1d numpy array (labels for subset 2)
        OUTPUT:
            - float

        Return the information gain of making the given split.

        Use self.impurity_criterion(y) rather than calling _entropy or _gini
        directly.
        """
        return self.impurity_criterion(y) - sum(
            [len(s) / len(y) * self.impurity_criterion(s) for s in [y1, y2]]
        )

    def _choose_split_index(self, X, y):
        """
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
        OUTPUT:
            - index: int (index of feature)
            - value: int/float/bool/str (value of feature)
            - splits: (2d array, 1d array, 2d array, 1d array)

        Determine which feature and value to split on. Return the index and
        value of the optimal split along with the split of the dataset.

        Return None, None, None if there is no split which improves information
        gain.

        Call the method like this:
        >>> index, value, splits = self._choose_split_index(X, y)
        >>> X1, y1, X2, y2 = splits
        """
        index = None
        value = None
        splits = None

        if len(np.unique(y)) == 1:
            return index, value, splits

        max_gain = 0
        for ind in range(X.shape[1]):
            values = np.unique(X[:, ind])
            if len(values) == 1:
                continue
            for v in values:
                X1, y1, X2, y2 = self._make_split(X, y, ind, v)

                # Skip split if either subset has less than min_samples_leaf
                if len(y1) < self.min_samples_leaf or len(y2) < self.min_samples_leaf:
                    continue

                gain = self._information_gain(y, y1, y2)
                gain_improvement = abs(gain) / self.impurity_criterion(y)

                if gain > max_gain and gain_improvement > self.min_impurity_decrease:
                    index, value, splits = ind, v, (X1, y1, X2, y2)
                    max_gain = gain
        return index, value, splits

    def predict(self, X):
        """
        INPUT:
            - X: 2d numpy array
        OUTPUT:
            - y: 1d numpy array

        Return an array of predictions for the feature matrix X.
        """

        return np.apply_along_axis(self.root.predict_one, axis=1, arr=X)

    def __str__(self):
        """
        Return string representation of the Decision Tree.
        """
        return str(self.root)
    
    def prune(self):
        """
        Prune tree using reduced error pruning.
        """
        self.root.prune()



if __name__ == "__main__":
    X = np.array([["1", "bat"], ["2", "cat"], ["2", "rat"], ["3", "bat"], ["3", "bat"]])
    y = np.array([1, 0, 1, 0, 1])
    dt = DecisionTree()
    dt.fit(X, y)
