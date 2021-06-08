import numpy as np
import math

# The kernel function for the Kernel Smoother
# nonparametric regression.
def kernel_function(x):
    term1 = 1 /(math.sqrt(2 * math.pi))
    term2 = np.exp(np.multiply(-1/2, np.multiply(x, x)))
    return np.multiply(term1, term2)

# kernel_smoother: A nonparametric regression function.
# X: The data points to fit into the regression model.
# Y: The labels of the given data points.
# h: A hyperparameter that indicates the window size of the input
#   of the kernel function.
def kernel_smoother(X, X_t, Y, h):

    # Checking for bad inputs------------------------------------------------

    # If the first parameter is not an array.
    if not (isinstance(X, list) or isinstance(X, np.ndarray)):
        raise Exception("Invalid input type. X is of type {}".format(type(X)))

    # If the second parameter is not an array.
    if not (isinstance(Y, list) or isinstance(Y, np.ndarray)):
        raise Exception("Invalid input type. Y is of type {}".format(type(Y)))

    # If the two inputs are of different shapes.
    if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        if X.shape != Y.shape:
            raise Exception("Unmatched input shapes. X.shape = {}, Y.shape = {}".format(X.shape, Y.shape))

    elif isinstance(X, list) and isinstance(Y, list):
        if len(X) != len(Y):
            raise Exception("Unmatched input lengths for X and Y. Length of X = {}, length of Y = {}".format(len(X), len(Y)))

    # If the value of h is less than or equal to zero.
    if h <= 0:
        raise Exception("Invalid h value. h value should be greater than 0. given h = {}".format(h))

    # ---------------------------------------------------------------------------
    
    # Nonparametric Regression Algorithm.

    Y_pred = []

    for x in X_t:
        U = np.divide(np.subtract(x, X), h)
        U = kernel_function(U)
        K = np.multiply(U, Y)

        y_hat = np.divide(np.sum(K), np.sum(U))
        Y_pred.append(y_hat)

    return np.array(Y_pred)
