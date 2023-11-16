import numpy as np


class GradientAscent(object):

    def __init__(self, cost, gradient, predict_func, fit_intercept=True, scale=True, l=0.0, step_size=None):
        '''
        INPUT: GradientAscent, function, function
        OUTPUT: None

        Initialize class variables. Takes two functions:
        cost: the cost function to be minimized
        gradient: function to calculate the gradient of the cost function
        '''
        # Initialize coefficients in run method once you know how many features
        # you have.
        self.coeffs = None
        self.cost = cost
        self.gradient = gradient
        self.predict_func = predict_func
        self.fit_intercept = fit_intercept
        self.scale = scale
        self.scale_mean = None
        self.scale_std = None
        self.l = l
        self.step_size = step_size
        self.training = np.array([])

    def run(self, X, y, alpha=0.01, num_iterations=10000):
        '''
        INPUT: GradientAscent, 2 dimensional numpy array, numpy array
               float, int
        OUTPUT: None

        Run the gradient ascent algorithm for num_iterations repititions. Use
        the gradient method and the learning rate alpha to update the
        coefficients at each iteration.
        '''
        if self.scale:
            self.scale_mean = X.mean(axis=0)
            self.scale_std = X.std(axis=0)
            X = self.scale_data(X)
        if self.fit_intercept:
            X = self.add_intercept(X)
        self.coeffs = np.zeros(X.shape[1])
        
        if self.step_size:
            old = self.cost(X, y, self.coeffs)
            delta = self.step_size + 1
            while delta > self.step_size:
                self.coeffs += alpha * self.gradient(X, y, self.coeffs, self.l)
                new = self.cost(X, y, self.coeffs)
                delta = abs(new - old) / old
                self.training = np.append(self.training, new)
                old = new
        else:
            for _ in range(num_iterations):
                self.coeffs += alpha * self.gradient(X, y, self.coeffs, self.l)
                self.training = np.append(self.training, self.cost(X, y, self.coeffs))


    def run_sgd(self, X, y, alpha=0.01, batch_size=1, num_iterations=10000):
        if self.scale:
            self.scale_mean = X.mean(axis=0)
            self.scale_std = X.std(axis=0)
            X = self.scale_data(X)
        if self.fit_intercept:
            X = self.add_intercept(X)
        self.coeffs = np.zeros(X.shape[1])

        if self.step_size:
            old = self.cost(X, y, self.coeffs)
            delta = self.step_size + 1
            while delta > self.step_size:
                shuffled_indices = np.argsort(np.random.rand(X.shape[0]))
                for ind in shuffled_indices:
                    self.coeffs += alpha * self.gradient(X[[ind], :], y[ind], self.coeffs, self.l)
                
                new = self.cost(X, y, self.coeffs)
                delta = abs(new - old) / old
                self.training = np.append(self.training, new)
                old = new
        else:
            for _ in range(num_iterations):
                shuffled_indices = np.argsort(np.random.rand(X.shape[0]))
                for ind in shuffled_indices:
                    self.coeffs += alpha * self.gradient(X[[ind], :], y[ind], self.coeffs, self.l)
                self.training = np.append(self.training, self.cost(X, y, self.coeffs))


    def predict(self, X):
        '''
        INPUT: GradientAscent, 2 dimensional numpy array
        OUTPUT: numpy array (ints)

        Use the coeffs to compute the prediction for X. Return an array of 0's
        and 1's. Call self.predict_func.
        '''
        if self.scale:
            X = self.scale_data(X)
        if self.fit_intercept:
            X = self.add_intercept(X)
        return self.predict_func(X, self.coeffs)
    
    def add_intercept(self, X):
        return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    
    def scale_data(self, X):
        return (X - self.scale_mean) / self.scale_std
