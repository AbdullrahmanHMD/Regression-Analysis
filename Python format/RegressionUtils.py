import numpy as np
import math


# RMSE: Root Mean Squared Error.
# Y:        The true labels of a data set.
# Y_pred:   The predicted labels of a data set.
# N:        The number of points in the data set
#           whose labels are Y.
def RMSE(Y, Y_pred, N):
    delta_Y = np.subtract(Y, Y_pred)
    delta_Y = np.square(delta_Y)
    delta_Y = np.sum(delta_Y)

    rmse = np.divide(delta_Y, N)
    rmse = math.sqrt(rmse)

    return rmse
