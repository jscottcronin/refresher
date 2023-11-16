import numpy as np

def hypothesis(X, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array
    OUTPUT: numpy array

    Calculate the predicted percentages (floats between 0 and 1) for the given
    data with the given coefficients.
    '''
    return 1. / (1 + np.exp(-np.dot(X, coeffs)))

def predict(X, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array
    OUTPUT: numpy array

    Calculate the predicted values (0 or 1) for the given data with the given
    coefficients.
    '''
    return (hypothesis(X, coeffs) > 0.5).astype(int)

def log_likelihood(X, y, coeffs, l=0.0):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array
    OUTPUT: float

    Calculate the log likelihood of the data with the given coefficients.
    '''
    y_pred = hypothesis(X, coeffs)
    likelihood = y.dot(np.log(y_pred)) + (1 - y).dot(np.log(1 - y_pred)) - l * np.sum(coeffs ** 2)
    return likelihood

def log_likelihood_gradient(X, y, coeffs, l=0.0):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array
    OUTPUT: numpy array

    Calculate the gradient of the log likelihood at the given value for the
    coeffs. Return an array of the same size as the coeffs array.
    '''
    return (y - hypothesis(X, coeffs)).dot(X) - 2 * l * coeffs

def accuracy(y_true, y_pred):
    '''
    INPUT: numpy array, numpy array
    OUPUT: float

    Calculate the percent of predictions which equal the true values.
    '''
    return (y_true == y_pred).mean()

def precision(y_true, y_pred):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: float

    Calculate the percent of positive predictions which were correct.
    '''
    return y_true[y_pred == 1].mean()

def recall(y_true, y_pred):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: float

    Calculate the percent of positive cases which were correctly predicted.
    '''
    return y_pred[y_true == 1].mean()
