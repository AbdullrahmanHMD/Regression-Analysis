import numpy as np
import RegressionUtils as ru


class LinearRegression(Object):

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


    def train_with_linear_regression(self, X, Y):

        if self.isValidData(X, Y):
            X_train, X-test, y_train, y_test = ru.split_data(X, Y)


