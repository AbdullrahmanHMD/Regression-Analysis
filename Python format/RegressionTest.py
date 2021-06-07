from operator import index
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import pandas as pd
import os
import math

# Data retrieval:
file_name = "Real_estate.csv"
folder_name = "Datasets"
path = os.path.abspath(os.getcwd()) + '\\' + folder_name + '\\' + file_name
real_estate_data = pd.read_csv(path).to_numpy()

X = np.array(real_estate_data[:,1: -1])
N = len(X)
Y_truth = np.array(real_estate_data[:,-1])

# -----------------------------------------------------
# Data Manipulation:


# Feartue selection:
from RegressionUtils import select_n_features
# Selecting top 3 features.
n = 3
X = select_n_features(real_estate_data, n)

# Dimensionality reduction:
from RegressionUtils import dimensinality_reduction
# The number of dimensions after performing dimensionality
# reduction
n = 1
X = dimensinality_reduction(X, n)
X = np.reshape(X, (N, ))

# # Plotting:
#
# plt.plot(X, Y_truth, "b.", markersize = 10)
#
# plt.plot(x_t, Y_pred, "r-")
#
# plt.xlabel("x")
# plt.ylabel("y")
#
# plt.show()
#
# #
# -----------------------------------------------------

from LinearRegression import LinearRegression
import RegressionUtils as ru

X_train, X_test, y_train, y_test = ru.split_data(X, Y_truth)
print(np.dot(Y_truth, X))
model = LinearRegression(X_train, y_train)
parameters = LinearRegression.estimation_parameters(model)
print(parameters)
y_pred = LinearRegression.predict(model, X_test, parameters)
print((y_pred - y_test))

plt.plot(X, Y_truth, "b.", markersize = 10)

plt.plot(X_test, y_pred, "r-")

plt.xlabel("x")
plt.ylabel("y")

plt.show()

