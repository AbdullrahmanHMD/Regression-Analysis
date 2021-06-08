import numpy as np
from numpy.linalg.linalg import transpose
import RegressionUtils as ru




def PolynomialRegression(X, Y, K):

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

    N = len(X)

    #feature matrix D
    D = []
    for x in X:
        row = []
        for i in range(K + 1):
            row.append(x ** i)
        D.append(row)
    D = np.array(D)

   #parameter estimation
    W = np.linalg.inv(np.matmul(D.transpose(), D))
    W = np.matmul(W, D.transpose())
    W = np.matmul(W, np.reshape(Y, (len(Y), 1)))
    return np.reshape(W, ((K + 1), ))



def predict(X, W, K):
    y_pred = []
    for x in X:
        sum = W[0]
        for i in range(K):
            sum += np.multiply(W[i+1], np.power(x, (i+1)))
        y_pred.append(sum)
    return np.array(y_pred)
            

            


