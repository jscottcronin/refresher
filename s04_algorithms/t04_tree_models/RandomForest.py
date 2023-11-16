from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from DecisionTree import DecisionTree

class RandomForest(object):
    '''A Random Forest class'''

    def __init__(self, num_trees, num_features):
        '''
           num_trees:  number of trees to create in the forest:
        num_features:  the number of features to consider when choosing the
                           best split for each node of the decision trees
        '''
        self.num_trees = num_trees
        self.num_features = num_features
        self.forest = None

    def fit(self, X, y):
        '''
        X:  two dimensional numpy array representing feature matrix
                for test data
        y:  numpy array representing labels for test data
        '''
        self.forest = self.build_forest(X, y, self.num_trees, X.shape[0], \
                                        self.num_features)

    def build_forest(self, X, y, num_trees, num_samples, num_features):
        '''
        Return a list of num_trees DecisionTrees.
        '''
        forest = []
        for _ in range(num_trees):
            inds = np.random.choice(len(X), size=len(X), replace=True)
            X_sample, y_sample = X[inds, :], y[inds]
            tree = DecisionTree(num_features=num_features)
            tree.fit(X_sample, y_sample)
            forest.append(tree)
        return forest

    def predict(self, X):
        '''
        Return a numpy array of the labels predicted for the given test data.
        '''        
        
        def get_most_common(y):
            c = Counter(y)
            most_common = [k for k, v in c.items() if v == max(c.values())]
            return np.random.choice(most_common)
        
        preds = []
        for tree in self.forest:
            preds.append(tree.predict(X))
        preds = np.array(preds).T
        return np.apply_along_axis(get_most_common, axis=1, arr=preds)

    def score(self, X, y):
        '''
        Return the accuracy of the Random Forest for the given test data and
        labels.
        '''
        y_hat = self.predict(X)
        return np.mean(y == y_hat)
    

if __name__ == '__main__':
    df = pd.read_csv('data/playgolf.csv')
    df.columns = df.columns.str.lower()
    result = df.pop('result').map({"Don't Play": 0, 'Play': 1})
    df = df.join(result)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = RandomForest(10, 3)
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)