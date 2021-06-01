import numpy as np
import RegressionUtils as ru


class LinearRegression():

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

   
    def isValidData(self, X, Y):
        # If the first parameter is not an array.
        if not (isinstance(X, list) or isinstance(X, np.ndarray)):
            raise Exception("Invalid input type. X is of type {}".format(type(X)))
            return False

        # If the second parameter is not an array.
        if not (isinstance(Y, list) or isinstance(Y, np.ndarray)):
            raise Exception("Invalid input type. Y is of type {}".format(type(Y)))
            return False

        # If the two inputs are of different shapes.
        if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
            if X.shape != Y.shape:
                raise Exception("Unmatched input shapes. X.shape = {}, Y.shape = {}".format(X.shape, Y.shape))
                return False

        elif isinstance(X, list) and isinstance(Y, list):
            if len(X) != len(Y):
                raise Exception(
                    "Unmatched input lengths for X and Y. Length of X = {}, length of Y = {}".format(len(X), len(Y)))
                return False

        return True


    def estimation_parameters(self):

        X = self.X
        Y = self.Y
        if self.isValidData(X, Y):
            N = X.shape[0]
            A = np.vstack(([N, np.sum(X)], [np.sum(X), np.dot(X,X)]))
            print(A.shape)
            Y = np.array([np.sum(Y), np.dot(Y,X)])
            Y = np.reshape(Y, (Y.shape[0], 1))
            w = np.matmul(np.linalg.inv(A), Y)
            return w
        else:
            raise Exception("Invalid inputs!.")
            return -1

    def predict(self, X_test, W):
        w0 = W[0]
        w1 = W[1]
        y_predicted = np.array([(w0 + w1 * x) for x in X_test])
        return y_predicted







