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

#
# -----------------------------------------------------
# Regression:

from NonparametricRegression import kernel_smoother
x_t = linspace(min(X), max(X), len(X))
h = 500
Y_pred = kernel_smoother(X, x_t, Y_truth.reshape(-1 , 1), h)

#
# -----------------------------------------------------
# Plotting:

plt.plot(X, Y_truth, "b.", markersize = 10)

plt.plot(x_t, Y_pred, "r-")

plt.xlabel("x")
plt.ylabel("y")

plt.show()

#
# -----------------------------------------------------