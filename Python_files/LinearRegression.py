import numpy as np
import RegressionUtils as ru


def LinearRegression(X, Y):


    if not (isinstance(X, list) or isinstance(X, np.ndarray)):
            raise Exception("Invalid input type. X is of type {}".format(type(X)))
       

    if not (isinstance(Y, list) or isinstance(Y, np.ndarray)):
            raise Exception("Invalid input type. Y is of type {}".format(type(Y)))

        # If the two inputs are of different shapes.
    if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        if X.shape != Y.shape:
            raise Exception("Unmatched input shapes. X.shape = {}, Y.shape = {}".format(X.shape, Y.shape))

    if isinstance(X, list) and isinstance(Y, list):
        if len(X) != len(Y):
            raise Exception(
                    "Unmatched input lengths for X and Y. Length of X = {}, length of Y = {}".format(len(X), len(Y)))


   # paramters estimation: w0, w1
    N = X.shape[0]
    A = np.vstack(([N, np.sum(X)], [np.sum(X), np.dot(X,X)]))
    Y = np.array([np.sum(Y), np.dot(Y,X)])
    Y = np.reshape(Y, (Y.shape[0], 1))
    W = np.matmul(np.linalg.inv(A), Y)
    W = np.reshape(W,(len(W), ))
      
    # return parameters
    w0 = W[0]
    w1 = W[1]
    return  w0, w1    


# for predixtion
def predict(X, w0, w1):
    return  np.array([(w0 + w1 * x) for x in X])








