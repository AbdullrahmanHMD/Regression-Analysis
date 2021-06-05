import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import pandas as pd
import os

from pandas.core.dtypes.missing import isnull

# Real_estate Data retrieval:
file_name = "Real_estate.csv"
folder_name = "Datasets"
path = os.path.abspath(os.getcwd()) + '\\' + folder_name + '\\' + file_name
real_estate_data = pd.read_csv(path).to_numpy()

X = np.array(real_estate_data[:,1: -1])
Y_truth = np.array(real_estate_data[:,-1]).reshape(-1 , 1)

# -----------------------------------------------------
# Real_estate Data Manipulation:

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

# Constructing train and test sets:
from sklearn.model_selection import train_test_split

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_truth, test_size = test_size, random_state=1)

#
# -----------------------------------------------------
# Regression:

from NonparametricRegression import kernel_smoother
x_t = linspace(min(X_train), max(X_train), len(X_train))
h = 150
Y_pred = kernel_smoother(X_train, x_t, Y_train, h)

#
# -----------------------------------------------------
# Plotting Real_estate:

plt.plot(X_train, Y_train, "b.", markersize = 7)
plt.plot(X_test, Y_test, "g.", markersize = 5)

plt.plot(x_t, Y_pred, "r-")

plt.xlabel("x")
plt.ylabel("y")

plt.show()

#
# -----------------------------------------------------

# insurance Data retrieval:
file_name = "insurance.csv"
folder_name = "Datasets"
path = os.path.abspath(os.getcwd()) + '\\' + folder_name + '\\' + file_name
insurance_data = pd.read_csv(path).to_numpy()

X = np.array(insurance_data[:-1])
Y_truth = np.array(insurance_data[:,-1]).reshape(-1 , 1)

# -----------------------------------------------------
# insurance Data Manipulation:

# Feartue selection:
from RegressionUtils import select_n_features
# Selecting top 3 features.
n = 5
X = select_n_features(insurance_data, n)

# Dimensionality reduction:
from RegressionUtils import dimensinality_reduction
# The number of dimensions after performing dimensionality
# reduction
n = 1
X = dimensinality_reduction(X, n)

# Constructing train and test sets:
from sklearn.model_selection import train_test_split

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_truth, test_size = test_size, random_state=1)

#
# -----------------------------------------------------
# Regression:

from NonparametricRegression import kernel_smoother
x_t = linspace(min(X_train), max(X_train), len(X_train))
h = 1
Y_pred = kernel_smoother(X_train, x_t, Y_train, h)

#
# -----------------------------------------------------
# Plotting insurance:

plt.plot(X_train, Y_train, "b.", markersize = 7)
plt.plot(X_test, Y_test, "g.", markersize = 5)

plt.plot(x_t, Y_pred, "r-")

plt.xlabel("x")
plt.ylabel("y")

plt.show()

#--------------------------------------------------------------------------------

# CarPrice Data retrieval:
file_name = "CarPrice_Assignment.csv"
folder_name = "Datasets"
path = os.path.abspath(os.getcwd()) + '\\' + folder_name + '\\' + file_name
car_price = pd.read_csv(path).to_numpy()

X = np.array(car_price[:: -1])
Y_truth = np.array(car_price[:,-1]).reshape(-1 , 1)

# Replacing NaN values with average column values.
from RegressionUtils import replace_nan_values
X = replace_nan_values(X)

# # -----------------------------------------------------
# # CarPrice Data Manipulation:

# Feartue selection:
from RegressionUtils import select_n_features
# Selecting top 3 features.
n = 3
car_price = replace_nan_values(car_price)
X = select_n_features(car_price, n)

# # Dimensionality reduction:
from RegressionUtils import dimensinality_reduction
# The number of dimensions after performing dimensionality
# reduction
n = 1
X = dimensinality_reduction(X, n)

# # Constructing train and test sets:
from sklearn.model_selection import train_test_split
test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_truth, test_size = test_size, random_state=1)

#
# -----------------------------------------------------
# Regression:

from NonparametricRegression import kernel_smoother
x_t = linspace(min(X_train), max(X_train), len(X_train))

h = 30
Y_pred = kernel_smoother(X_train, x_t, Y_train, h)

#
# -----------------------------------------------------
# Plotting CarPrice:

plt.plot(X_train, Y_train, "g.", markersize = 5)
plt.plot(X_test, Y_test, "b.", markersize = 5)

plt.plot(x_t, Y_pred, "r-")

plt.xlabel("x")
plt.ylabel("y")

plt.show()