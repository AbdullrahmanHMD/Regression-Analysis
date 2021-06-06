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
features = pd.read_csv(path).columns.to_numpy()

X = np.array(real_estate_data[:: -1])
Y_truth = np.array(real_estate_data[:,-1]).reshape(-1 , 1)

# Replacing NaN values with average column values.
from RegressionUtils import replace_nan_values

X = replace_nan_values(X)

# -----------------------------------------------------
# Real_estate Data Manipulation:

# Feartue selection:
from RegressionUtils import select_n_features
# Selecting top 3 features.
n = 5
X, feature_indecies = select_n_features(real_estate_data, n)

# Retrieving the remaining features.
features = features[feature_indecies]

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
h = 250
Y_pred = kernel_smoother(X_train, x_t, Y_train, h)

# Calculating RMSE:

from RegressionUtils import RMSE
x_t_test = linspace(min(X_test), max(X_test), len(X_test))
Y_pred_test = kernel_smoother(X_test, x_t_test, Y_test, h)
rmse = RMSE(Y_test, Y_pred_test)

print("The RMSE for the kernel smoother with the Real_estate dataset is {}".format(rmse))

# Calculating Squared Errors

from RegressionUtils import mean_squared_errors

x_t_test = linspace(min(X_test), max(X_test), len(X_test))
Y_pred_test = kernel_smoother(X_test, x_t_test, Y_test, h)
mse = mean_squared_errors(Y_test, Y_pred_test)

print("The squared errors for the kernel smoother with the Real_estate dataset is {}".format(mse))

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

X = replace_nan_values(X)

features = pd.read_csv(path).columns.to_numpy()

# -----------------------------------------------------
# insurance Data Manipulation:

# Feartue selection:
from RegressionUtils import select_n_features
# Selecting top 3 features.
n = 5
X, feature_indecies = select_n_features(insurance_data, n)

# Retrieving the remaining features.
features = features[feature_indecies]

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

features = pd.read_csv(path).columns.to_numpy()

X = replace_nan_values(X)

# # -----------------------------------------------------
# # CarPrice Data Manipulation:

# Feartue selection:
from RegressionUtils import select_n_features
# Selecting top 3 features.
n = 3
car_price = replace_nan_values(car_price)

X, feature_indecies = select_n_features(car_price, n)

# Retrieving the remaining features.
features = features[feature_indecies]

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

# -----------------------------------------------------